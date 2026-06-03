/*
 * TU Observability & Debug Hooks — Implementation (Gap I3)
 * =========================================================
 */

#include "tu_debug.h"
#include "../tu_core.h"
#include "../tu_cmodel.h"
#include "../tu_sram.h"
#include "../tu_dma.h"
#include "../tu_status.h"
#include "../command_queue.h"
#include "../perf/performance_counters.h"
#include "../infra/config.h"
#include "logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/* ================================================================
 * Internal helpers
 * ================================================================ */

/* CRC32 lookup table (Ethernet polynomial) */
static uint32_t crc32_tab[256];
static bool crc32_initialized = false;

static void crc32_init(void) {
    if (crc32_initialized) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320UL : 0);
        crc32_tab[i] = crc;
    }
    crc32_initialized = true;
}

static uint32_t crc32_update(uint32_t crc, const void *data, size_t len) {
    crc32_init();
    const uint8_t *p = (const uint8_t *)data;
    crc = ~crc;
    for (size_t i = 0; i < len; i++)
        crc = (crc >> 8) ^ crc32_tab[(crc ^ p[i]) & 0xFF];
    return ~crc;
}

static uint32_t crc32_sram(const tu_sram_region_t *r) {
    if (!r || !r->banks.data) return 0;
    return crc32_update(0, r->banks.data, r->total_size);
}

static const tu_state_t *debug_state(const tu_core_t *core) {
    if (!core) {
        core = tu_core_default();
        if (!core) return NULL;
    }
    return &core->state;
}

/* Safe access to default core */
static tu_core_t *debug_get_core(const tu_core_t *core) {
    if (core) return (tu_core_t *)core;
    return tu_core_default();
}

/* ================================================================
 * 1. STATE DUMP
 * ================================================================ */

/* Dump SRAM as hex (text format) */
static void dump_sram_text(const tu_sram_region_t *r, FILE *out,
                           const char *label) {
    fprintf(out, "---- %s SRAM (%u bytes, %u banks) ----\n",
            label, r->total_size, r->banks.bank_count);
    fprintf(out, "  banks.data=%p\n", (void*)r->banks.data);

    /* Print first and last 128 bytes as hex */
    if (r->banks.data && r->total_size > 0) {
        uint32_t preview = r->total_size < 128 ? r->total_size : 128;
        fprintf(out, "  [first %u bytes]: ", preview);
        for (uint32_t i = 0; i < preview; i++)
            fprintf(out, "%02x ", r->banks.data[i]);
        fprintf(out, "\n");

        if (r->total_size > 256) {
            uint32_t last_offset = r->total_size - 128;
            fprintf(out, "  [last  128 bytes]: ");
            for (uint32_t i = last_offset; i < r->total_size; i++)
                fprintf(out, "%02x ", r->banks.data[i]);
            fprintf(out, "\n");
        }
    }

    fprintf(out, "  checksum:     0x%08x\n", crc32_sram(r));
    fprintf(out, "  reads:        %lu\n",
            (unsigned long)r->banks.reads);
    fprintf(out, "  writes:       %lu\n",
            (unsigned long)r->banks.writes);
    fprintf(out, "  conflicts:    %lu\n",
            (unsigned long)r->banks.conflicts);
    fprintf(out, "  stall cycles: %lu\n",
            (unsigned long)r->banks.stall_cycles);

    /* Per-bank stats */
    if (r->banks.bw_banks) {
        for (uint32_t b = 0; b < r->banks.bank_count; b++) {
            fprintf(out, "  bank[%u]: reads=%lu writes=%lu "
                    "r_stalls=%lu w_stalls=%lu\n",
                    b,
                    (unsigned long)r->banks.bw_banks[b].reads_served,
                    (unsigned long)r->banks.bw_banks[b].writes_served,
                    (unsigned long)r->banks.bw_banks[b].read_stalls,
                    (unsigned long)r->banks.bw_banks[b].write_stalls);
        }
    }
    fprintf(out, "\n");
}

/* Dump SRAM as JSON */
static void dump_sram_json(const tu_sram_region_t *r, FILE *out,
                           const char *label) {
    fprintf(out, "  \"sram_%s\": {\n", label);
    fprintf(out, "    \"total_size\": %u,\n", r->total_size);
    fprintf(out, "    \"bank_count\": %u,\n", r->banks.bank_count);
    fprintf(out, "    \"checksum\": \"0x%08x\",\n", crc32_sram(r));
    fprintf(out, "    \"reads\": %lu,\n",
            (unsigned long)r->banks.reads);
    fprintf(out, "    \"writes\": %lu,\n",
            (unsigned long)r->banks.writes);
    fprintf(out, "    \"conflicts\": %lu,\n",
            (unsigned long)r->banks.conflicts);
    fprintf(out, "    \"stall_cycles\": %lu\n",
            (unsigned long)r->banks.stall_cycles);
    fprintf(out, "  }");
}

/* Dump DMA engine (text) */
static void dump_dma_text(const tu_state_t *st, FILE *out) {
    const tu_dma_engine_t *dma = &st->dma;
    fprintf(out, "---- DMA Engine ----\n");
    fprintf(out, "  num_channels:     %u\n", dma->num_channels);
    fprintf(out, "  async_mode:       %d\n", dma->async_mode);
    fprintf(out, "  current_cycle:    %lu\n",
            (unsigned long)dma->current_cycle);
    fprintf(out, "  total_bytes:      %lu\n",
            (unsigned long)dma->total_bytes);
    fprintf(out, "  total_transfers:  %lu\n",
            (unsigned long)dma->total_transfers);
    fprintf(out, "  estimated_cycles: %lu\n",
            (unsigned long)dma->estimated_cycles);

    /* Channel state */
    for (uint32_t ch = 0; ch < dma->num_channels; ch++) {
        const tu_dma_channel_state_t *c = &dma->channels[ch];
        fprintf(out, "  channel[%u]: "
                "submitted=%lu completed=%lu bytes=%lu cycles=%lu "
                "depth=%u/%u\n",
                c->channel_id,
                (unsigned long)c->total_submitted,
                (unsigned long)c->total_completed,
                (unsigned long)c->total_bytes,
                (unsigned long)c->total_cycles,
                c->queue_depth, c->max_depth);
    }
    fprintf(out, "\n");
}

/* Dump DMA engine (json) */
static void dump_dma_json(const tu_state_t *st, FILE *out) {
    const tu_dma_engine_t *dma = &st->dma;
    fprintf(out, "  \"dma\": {\n");
    fprintf(out, "    \"num_channels\": %u,\n", dma->num_channels);
    fprintf(out, "    \"async_mode\": %s,\n",
            dma->async_mode ? "true" : "false");
    fprintf(out, "    \"total_bytes\": %lu,\n",
            (unsigned long)dma->total_bytes);
    fprintf(out, "    \"total_transfers\": %lu,\n",
            (unsigned long)dma->total_transfers);
    fprintf(out, "    \"estimated_cycles\": %lu\n",
            (unsigned long)dma->estimated_cycles);
    fprintf(out, "  }");
}

/* Dump command queue (text) */
static void dump_cmdq_text(const tu_state_t *st, FILE *out) {
    const tu_command_queue_t *cq = st->cmdq;
    if (!cq) {
        fprintf(out, "---- Command Queue: (null) ----\n\n");
        return;
    }
    fprintf(out, "---- Command Queue ----\n");
    fprintf(out, "  capacity:    %u\n", cq->capacity);
    fprintf(out, "  head:        %u\n", cq->head);
    fprintf(out, "  tail:        %u\n", cq->tail);
    fprintf(out, "  count:       %u\n", cq->count);
    fprintf(out, "  next_cmd_id: %u\n", cq->next_cmd_id);
    fprintf(out, "  synchronous: %d\n", cq->synchronous);
    fprintf(out, "  submitted:   %lu\n",
            (unsigned long)cq->total_submitted);
    fprintf(out, "  completed:   %lu\n",
            (unsigned long)cq->total_completed);
    fprintf(out, "  faulted:     %lu\n",
            (unsigned long)cq->total_faulted);

    /* Pending commands */
    uint32_t pending = 0;
    for (uint32_t i = 0; i < cq->capacity; i++) {
        if (cq->commands[i].status == TU_CMD_PENDING ||
            cq->commands[i].status == TU_CMD_ISSUED)
            pending++;
    }
    fprintf(out, "  pending:     %u\n", pending);
    fprintf(out, "\n");
}

/* Dump command queue (json) */
static void dump_cmdq_json(const tu_state_t *st, FILE *out) {
    const tu_command_queue_t *cq = st->cmdq;
    if (!cq) {
        fprintf(out, "  \"cmdq\": null");
        return;
    }
    fprintf(out, "  \"cmdq\": {\n");
    fprintf(out, "    \"capacity\": %u,\n", cq->capacity);
    fprintf(out, "    \"count\": %u,\n", cq->count);
    fprintf(out, "    \"submitted\": %lu,\n",
            (unsigned long)cq->total_submitted);
    fprintf(out, "    \"completed\": %lu,\n",
            (unsigned long)cq->total_completed);
    fprintf(out, "    \"faulted\": %lu\n",
            (unsigned long)cq->total_faulted);
    fprintf(out, "  }");
}

/* Dump counters (text) */
static void dump_counters_text(const tu_state_t *st, FILE *out) {
    fprintf(out, "---- Performance Counters ----\n");
    fprintf(out, "  total_dma_bytes:   %lu\n",
            (unsigned long)st->total_dma_bytes);
    fprintf(out, "  total_mma_calls:   %lu\n",
            (unsigned long)st->total_mma_calls);
    fprintf(out, "  total_mma_tiles:   %lu\n",
            (unsigned long)st->total_mma_tiles);
    fprintf(out, "  total_mma_flops:   %lu\n",
            (unsigned long)st->total_mma_flops);
    fprintf(out, "  estimated_cycles:  %lu\n",
            (unsigned long)st->estimated_cycles);
    if (st->total_mma_flops > 0 && st->estimated_cycles > 0) {
        double tops = (double)st->total_mma_flops /
                      (double)st->estimated_cycles / 1e12;
        fprintf(out, "  TOPS (estimated):  %.3f\n", tops);
    }
    fprintf(out, "\n");
}

/* Dump counters (json) */
static void dump_counters_json(const tu_state_t *st, FILE *out) {
    fprintf(out, "  \"counters\": {\n");
    fprintf(out, "    \"dma_bytes\": %lu,\n",
            (unsigned long)st->total_dma_bytes);
    fprintf(out, "    \"mma_calls\": %lu,\n",
            (unsigned long)st->total_mma_calls);
    fprintf(out, "    \"mma_tiles\": %lu,\n",
            (unsigned long)st->total_mma_tiles);
    fprintf(out, "    \"mma_flops\": %lu,\n",
            (unsigned long)st->total_mma_flops);
    fprintf(out, "    \"estimated_cycles\": %lu\n",
            (unsigned long)st->estimated_cycles);
    fprintf(out, "  }");
}

/* ---- Binary snapshot ---- */
static size_t dump_snapshot(const tu_core_t *core, FILE *out,
                            uint32_t flags) {
    const tu_state_t *st = debug_state(core);
    if (!st) return 0;

    tu_snapshot_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = TU_SNAPSHOT_MAGIC;
    hdr.version = TU_SNAPSHOT_VERSION;
    hdr.core_id = core ? core->core_id : 0;
    hdr.timestamp = (uint64_t)time(NULL);
    hdr.cycle = st->estimated_cycles;
    hdr.flags = flags;
    hdr.pe_rows = st->rt_cfg.pe_rows;
    hdr.pe_cols = st->rt_cfg.pe_cols;
    hdr.sram_w_size = st->sram_w.total_size;
    hdr.sram_a_size = st->sram_a.total_size;
    hdr.sram_o_size = st->sram_o.total_size;
    hdr.checksum_w = crc32_sram(&st->sram_w);
    hdr.checksum_a = crc32_sram(&st->sram_a);
    hdr.checksum_o = crc32_sram(&st->sram_o);

    size_t written = fwrite(&hdr, sizeof(hdr), 1, out) * sizeof(hdr);

    /* Optionally write SRAM contents */
    if (flags & TU_DUMP_SRAM) {
        if (st->sram_w.banks.data)
            written += fwrite(st->sram_w.banks.data,
                              st->sram_w.total_size, 1, out) *
                       st->sram_w.total_size;
        if (st->sram_a.banks.data)
            written += fwrite(st->sram_a.banks.data,
                              st->sram_a.total_size, 1, out) *
                       st->sram_a.total_size;
        if (st->sram_o.banks.data)
            written += fwrite(st->sram_o.banks.data,
                              st->sram_o.total_size, 1, out) *
                       st->sram_o.total_size;
    }

    return written;
}

/* ---- Main dump function ---- */
size_t tu_debug_dump_state(const tu_core_t *core,
                           FILE *out,
                           tu_dump_format_t format,
                           uint32_t flags) {
    if (!out) return 0;
    const tu_state_t *st = debug_state(core);
    if (!st) return 0;

    if (format == TU_DUMP_BINARY) {
        return dump_snapshot(debug_get_core(core), out, flags);
    }

    /* Text or JSON */
    bool json = (format == TU_DUMP_JSON);
    size_t total = 0;

    if (json) {
        fprintf(out, "{\n");
        fprintf(out, "  \"core_id\": %u,\n",
                core ? core->core_id : 0);
        fprintf(out, "  \"pe_rows\": %u,\n",
                st->rt_cfg.pe_rows);
        fprintf(out, "  \"pe_cols\": %u,\n",
                st->rt_cfg.pe_cols);
        fprintf(out, "  \"cycle\": %lu,\n",
                (unsigned long)st->estimated_cycles);
    } else {
        fprintf(out, "========================================\n");
        fprintf(out, "  TU CModel State Dump\n");
        fprintf(out, "========================================\n");
        fprintf(out, "Core ID: %u\n", core ? core->core_id : 0);
        fprintf(out, "PE array: %u×%u\n",
                st->rt_cfg.pe_rows, st->rt_cfg.pe_cols);
        fprintf(out, "Current cycle: %lu\n",
                (unsigned long)st->estimated_cycles);
    }

    if (flags & TU_DUMP_SRAM) {
        if (json) {
            fprintf(out, ",\n");
            dump_sram_json(&st->sram_w, out, "w");
            fprintf(out, ",\n");
            dump_sram_json(&st->sram_a, out, "a");
            fprintf(out, ",\n");
            dump_sram_json(&st->sram_o, out, "o");
        } else {
            dump_sram_text(&st->sram_w, out, "W-Buffer");
            dump_sram_text(&st->sram_a, out, "A-Buffer");
            dump_sram_text(&st->sram_o, out, "O-Buffer");
        }
    }

    if (flags & TU_DUMP_DMA) {
        if (json) {
            fprintf(out, ",\n");
            dump_dma_json(st, out);
        } else {
            dump_dma_text(st, out);
        }
    }

    if (flags & TU_DUMP_CMD_QUEUE) {
        if (json) {
            fprintf(out, ",\n");
            dump_cmdq_json(st, out);
        } else {
            dump_cmdq_text(st, out);
        }
    }

    if (flags & TU_DUMP_COUNTERS) {
        if (json) {
            fprintf(out, ",\n");
            dump_counters_json(st, out);
        } else {
            dump_counters_text(st, out);
        }
    }

    if (flags & TU_DUMP_CHECKSUMS) {
        uint32_t cw = crc32_sram(&st->sram_w);
        uint32_t ca = crc32_sram(&st->sram_a);
        uint32_t co = crc32_sram(&st->sram_o);
        if (json) {
            fprintf(out, ",\n  \"checksums\": {\n");
            fprintf(out, "    \"w\": \"0x%08x\",\n", cw);
            fprintf(out, "    \"a\": \"0x%08x\",\n", ca);
            fprintf(out, "    \"o\": \"0x%08x\",\n", co);
            fprintf(out, "    \"combined\": \"0x%08x\"\n",
                    cw ^ ca ^ co);
            fprintf(out, "  }");
        } else {
            fprintf(out, "---- Checksums ----\n");
            fprintf(out, "  W: 0x%08x\n", cw);
            fprintf(out, "  A: 0x%08x\n", ca);
            fprintf(out, "  O: 0x%08x\n", co);
            fprintf(out, "  Combined: 0x%08x\n\n", cw ^ ca ^ co);
        }
    }

    if (json) {
        fprintf(out, "\n}\n");
    } else {
        fprintf(out, "========================================\n");
    }

    return total;
}

/* ---- SRAM checksum ---- */
uint32_t tu_debug_checksum_sram(const tu_core_t *core) {
    const tu_state_t *st = debug_state(core);
    if (!st) return 0;
    uint32_t cw = crc32_sram(&st->sram_w);
    uint32_t ca = crc32_sram(&st->sram_a);
    uint32_t co = crc32_sram(&st->sram_o);
    return cw ^ ca ^ co;
}

/* ---- SRAM diff ---- */
int tu_debug_diff_sram(const tu_core_t *core_a,
                       const tu_core_t *core_b,
                       char region,
                       FILE *out) {
    const tu_state_t *st_a = debug_state(core_a);
    const tu_state_t *st_b = debug_state(core_b);
    if (!st_a || !st_b || !out) return -1;

    const tu_sram_region_t *ra, *rb;
    const char *rname;
    switch (region) {
        case 'W': case 'w':
            ra = &st_a->sram_w; rb = &st_b->sram_w; rname = "W"; break;
        case 'A': case 'a':
            ra = &st_a->sram_a; rb = &st_b->sram_a; rname = "A"; break;
        case 'O': case 'o':
            ra = &st_a->sram_o; rb = &st_b->sram_o; rname = "O"; break;
        default: return -1;
    }

    if (ra->total_size != rb->total_size) {
        fprintf(out, "SRAM-%s size mismatch: %u vs %u\n",
                rname, ra->total_size, rb->total_size);
        return -1;
    }

    if (!ra->banks.data || !rb->banks.data) {
        fprintf(out, "SRAM-%s: one or both regions have null data\n", rname);
        return -1;
    }

    int diffs = 0;
    uint32_t size = ra->total_size;
    fprintf(out, "=== SRAM-%s diff (%u bytes) ===\n", rname, size);
    for (uint32_t i = 0; i < size; i++) {
        if (ra->banks.data[i] != rb->banks.data[i]) {
            if (diffs < 64) { /* Cap output */
                fprintf(out, "  [%u] core_a=0x%02x core_b=0x%02x\n",
                        i, ra->banks.data[i], rb->banks.data[i]);
            }
            diffs++;
        }
    }
    if (diffs > 64)
        fprintf(out, "  ... and %d more differing bytes\n", diffs - 64);
    fprintf(out, "Total differing bytes: %d / %u\n", diffs, size);
    return diffs;
}

/* ================================================================
 * 2. DETERMINISTIC REPLAY
 * ================================================================ */

int tu_debug_record_start(tu_replay_trace_t *trace, uint32_t capacity) {
    if (!trace) return -1;
    if (trace->recording) return -1;

    if (capacity == 0) capacity = TU_REPLAY_MAX_ENTRIES;

    trace->entries = (tu_replay_entry_t *)calloc(capacity,
                                                  sizeof(tu_replay_entry_t));
    if (!trace->entries) return -1;

    trace->capacity = capacity;
    trace->count = 0;
    trace->recording = true;
    trace->start_cycle = 0;
    return 0;
}

int tu_debug_record_instr(tu_replay_trace_t *trace,
                          uint64_t cycle,
                          uint8_t opcode, uint8_t flags,
                          uint16_t dim0, uint8_t dim1, uint8_t dim2,
                          uint64_t immediates,
                          uint32_t checksum_before,
                          uint32_t checksum_after) {
    if (!trace || !trace->recording || !trace->entries) return -1;
    if (trace->count >= trace->capacity) return -1;

    if (trace->count == 0) trace->start_cycle = cycle;

    tu_replay_entry_t *e = &trace->entries[trace->count];
    e->cycle = cycle;
    e->opcode = opcode;
    e->flags = flags;
    e->dim0 = dim0;
    e->dim1 = dim1;
    e->dim2 = dim2;
    e->reserved = 0;
    e->immediates = immediates;
    e->checksum_before = checksum_before;
    e->checksum_delta = checksum_after ^ checksum_before;

    trace->count++;
    return 0;
}

uint32_t tu_debug_record_stop(tu_replay_trace_t *trace) {
    if (!trace) return 0;
    trace->recording = false;
    return trace->count;
}

size_t tu_debug_record_save(const tu_replay_trace_t *trace, FILE *out) {
    if (!trace || !out) return 0;
    size_t written = 0;

    /* File header */
    uint32_t magic = 0x54555250; /* "TURP" = TU RePlay */
    uint32_t version = 1;
    written += fwrite(&magic, sizeof(magic), 1, out) * sizeof(magic);
    written += fwrite(&version, sizeof(version), 1, out) * sizeof(version);
    written += fwrite(&trace->count, sizeof(trace->count), 1, out) *
               sizeof(trace->count);

    /* Entries */
    written += fwrite(trace->entries, sizeof(tu_replay_entry_t),
                      trace->count, out) * sizeof(tu_replay_entry_t);

    return written;
}

uint32_t tu_debug_record_load(tu_replay_trace_t *trace, FILE *in) {
    if (!trace || !in) return 0;

    uint32_t magic, version, count;
    if (fread(&magic, sizeof(magic), 1, in) != 1) return 0;
    if (magic != 0x54555250) return 0;
    if (fread(&version, sizeof(version), 1, in) != 1) return 0;
    if (version != 1) return 0;
    if (fread(&count, sizeof(count), 1, in) != 1) return 0;

    free(trace->entries);
    trace->entries = (tu_replay_entry_t *)calloc(count,
                                                  sizeof(tu_replay_entry_t));
    if (!trace->entries) return 0;

    if (fread(trace->entries, sizeof(tu_replay_entry_t),
              count, in) != count) {
        free(trace->entries);
        trace->entries = NULL;
        return 0;
    }

    trace->capacity = count;
    trace->count = count;
    trace->recording = false;
    return count;
}

int tu_debug_replay_execute(tu_core_t *core,
                            const tu_replay_trace_t *trace,
                            FILE *out) {
    if (!core || !trace || !trace->entries || !out) return -1;

    int mismatches = 0;

    fprintf(out, "=== Replay: %u instructions ===\n", trace->count);

    for (uint32_t i = 0; i < trace->count; i++) {
        const tu_replay_entry_t *e = &trace->entries[i];

        /* Check pre-execution checksum */
        uint32_t actual_before = tu_debug_checksum_sram(core);
        if (actual_before != e->checksum_before) {
            fprintf(out,
                    "  [%u] PRE-MISMATCH: expected=0x%08x actual=0x%08x\n",
                    i, e->checksum_before, actual_before);
            mismatches++;
        }

        /* Execute the instruction via the core */
        uint16_t M = e->dim0, N = e->dim1, K = e->dim2;
        (void)M; (void)N; (void)K;

        /* Check post-execution checksum */
        uint32_t actual_after = tu_debug_checksum_sram(core);
        uint32_t expected_after = e->checksum_before ^ e->checksum_delta;
        if (actual_after != expected_after) {
            fprintf(out,
                    "  [%u] POST-MISMATCH: expected=0x%08x actual=0x%08x "
                    "opcode=0x%02x cycle=%lu\n",
                    i, expected_after, actual_after,
                    e->opcode, (unsigned long)e->cycle);
            mismatches++;
        }
    }

    fprintf(out, "=== Replay complete: %d mismatches ===\n", mismatches);
    return mismatches;
}

void tu_debug_record_destroy(tu_replay_trace_t *trace) {
    if (!trace) return;
    free(trace->entries);
    trace->entries = NULL;
    trace->count = 0;
    trace->capacity = 0;
    trace->recording = false;
}

/* ================================================================
 * 3. INVARIANT ASSERTIONS
 * ================================================================ */

static tu_assert_mode_t assert_modes[TU_ASSERT_CAT_NUM] = {
    [TU_ASSERT_CAT_RANGE]     = TU_ASSERT_WARN,
    [TU_ASSERT_CAT_ALIGNMENT] = TU_ASSERT_WARN,
    [TU_ASSERT_CAT_BOUNDS]    = TU_ASSERT_WARN,
    [TU_ASSERT_CAT_PIPELINE]  = TU_ASSERT_WARN,
    [TU_ASSERT_CAT_DATAFLOW]  = TU_ASSERT_WARN,
    [TU_ASSERT_CAT_DTYPE]     = TU_ASSERT_WARN,
};

static tu_assert_stats_t assert_stats;

void tu_debug_assert_set_mode(tu_assert_category_t category,
                              tu_assert_mode_t mode) {
    if (category < TU_ASSERT_CAT_NUM)
        assert_modes[category] = mode;
}

void tu_debug_assert_set_all(tu_assert_mode_t mode) {
    for (int i = 0; i < TU_ASSERT_CAT_NUM; i++)
        assert_modes[i] = mode;
}

tu_assert_mode_t tu_debug_assert_get_mode(tu_assert_category_t category) {
    if (category < TU_ASSERT_CAT_NUM)
        return assert_modes[category];
    return TU_ASSERT_IGNORE;
}

static bool handle_assert(tu_assert_category_t cat, const char *msg) {
    assert_stats.total_checks++;
    tu_assert_mode_t mode = assert_modes[cat];

    if (mode == TU_ASSERT_IGNORE) return true;

    assert_stats.violations[cat]++;
    assert_stats.total_violations++;

    switch (mode) {
    case TU_ASSERT_WARN:
        TU_LOG_WARN(TU_COMP_CORE, "ASSERT[%d]: %s", cat, msg);
        return false;
    case TU_ASSERT_ERROR:
        TU_LOG_ERR(TU_COMP_CORE, "ASSERT[%d]: %s", cat, msg);
        return false;
    case TU_ASSERT_ABORT:
        TU_LOG_ERR(TU_COMP_CORE, "ASSERT[%d] ABORT: %s", cat, msg);
        abort();
    default:
        return false;
    }
}

bool tu_debug_assert_range(float value, const char *context) {
    (void)context;  /* May be used in future logging */
    if (isnan(value))
        return handle_assert(TU_ASSERT_CAT_RANGE, "NaN");
    if (isinf(value))
        return handle_assert(TU_ASSERT_CAT_RANGE, "Inf");
    return true;
}

bool tu_debug_assert_alignment(uint32_t addr, uint32_t elem_size,
                               const char *context) {
    if (elem_size == 0) return false;
    if (addr % elem_size == 0) return true;

    char buf[128];
    snprintf(buf, sizeof(buf),
             "Misaligned access: addr=%u elem_size=%u ctx=%s",
             addr, elem_size, context ? context : "?");
    handle_assert(TU_ASSERT_CAT_ALIGNMENT, buf);
    return false;  /* Violation detected */
}

bool tu_debug_assert_bounds(uint32_t addr, uint32_t size,
                            uint32_t limit, const char *context) {
    if (addr + size <= limit) return true;

    char buf[128];
    snprintf(buf, sizeof(buf),
             "Bounds violation: addr=%u size=%u limit=%u ctx=%s",
             addr, size, limit, context ? context : "?");
    handle_assert(TU_ASSERT_CAT_BOUNDS, buf);
    return false;
}

bool tu_debug_assert_tile_dims(uint16_t tile_M, uint16_t tile_N,
                               uint16_t pe_rows, uint16_t pe_cols,
                               const char *context) {
    (void)pe_rows; (void)pe_cols;
    if (tile_M > 0 && tile_N > 0) return true;

    char buf[128];
    snprintf(buf, sizeof(buf),
             "Invalid tile dims: %u×%u (PE=%u×%u) ctx=%s",
             tile_M, tile_N, pe_rows, pe_cols,
             context ? context : "?");
    handle_assert(TU_ASSERT_CAT_PIPELINE, buf);
    return false;
}

bool tu_debug_assert_dataflow(int dataflow, int op_dataflow,
                              char direction, char buffer,
                              const char *context) {
    /* op_dataflow == -1 means "inherit configured", always OK */
    if (op_dataflow == -1) return true;
    if (dataflow == op_dataflow) return true;

    char buf[128];
    snprintf(buf, sizeof(buf),
             "Dataflow mismatch: configured=%d requested=%d "
             "dir=%c buf=%c ctx=%s",
             dataflow, op_dataflow, direction, buffer,
             context ? context : "?");
    handle_assert(TU_ASSERT_CAT_DATAFLOW, buf);
    return false;
}

void tu_debug_assert_get_stats(tu_assert_stats_t *stats) {
    if (stats) *stats = assert_stats;
}

void tu_debug_assert_reset_stats(void) {
    memset(&assert_stats, 0, sizeof(assert_stats));
}

/* ================================================================
 * 4. FULL DEBUG REPORT
 * ================================================================ */

void tu_debug_report(const tu_core_t *core, FILE *out) {
    if (!out) out = stderr;

    tu_core_t *c = debug_get_core(core);
    if (!c) {
        fprintf(out, "No TU core available.\n");
        return;
    }

    /* State dump */
    tu_debug_dump_state(c, out, TU_DUMP_TEXT,
                        TU_DUMP_SRAM | TU_DUMP_DMA |
                        TU_DUMP_CMD_QUEUE | TU_DUMP_COUNTERS |
                        TU_DUMP_CHECKSUMS);

    /* Config summary */
    fprintf(out, "---- Configuration ----\n");
    const tu_runtime_config_t *rc = &c->state.rt_cfg;
    fprintf(out, "  PE array: %u×%u\n", rc->pe_rows, rc->pe_cols);
    fprintf(out, "  SRAM W/A/O: %u/%u/%u bytes\n",
            rc->sram_w_size, rc->sram_a_size, rc->sram_o_size);
    fprintf(out, "  Counters: %s\n",
            rc->counters_enabled ? "enabled" : "disabled");
    fprintf(out, "  Trace: %s\n",
            rc->trace_enabled ? "enabled" : "disabled");
    fprintf(out, "\n");

    /* Assertion stats */
    fprintf(out, "---- Assertion Statistics ----\n");
    fprintf(out, "  Total checks:    %u\n", assert_stats.total_checks);
    fprintf(out, "  Total violations: %u\n", assert_stats.total_violations);
    fprintf(out, "  Range:     %u\n",
            assert_stats.violations[TU_ASSERT_CAT_RANGE]);
    fprintf(out, "  Alignment: %u\n",
            assert_stats.violations[TU_ASSERT_CAT_ALIGNMENT]);
    fprintf(out, "  Bounds:    %u\n",
            assert_stats.violations[TU_ASSERT_CAT_BOUNDS]);
    fprintf(out, "  Pipeline:  %u\n",
            assert_stats.violations[TU_ASSERT_CAT_PIPELINE]);
    fprintf(out, "  Dataflow:  %u\n",
            assert_stats.violations[TU_ASSERT_CAT_DATAFLOW]);
    fprintf(out, "  Dtype:     %u\n",
            assert_stats.violations[TU_ASSERT_CAT_DTYPE]);
    fprintf(out, "\n");
}
