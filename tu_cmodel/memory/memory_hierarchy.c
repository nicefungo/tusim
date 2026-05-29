/*
 * TU CModel — Multi-Level Memory Hierarchy Implementation
 * =========================================================
 * Gap A3: Production-grade memory hierarchy with RegFile,
 * Local SPAD, Global Buffer, and DRAM levels.
 */

#include "memory_hierarchy.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Default level configurations ---- */

static const tu_mem_level_config_t defaults[TU_MEM_NUM_LEVELS] = {
    /* Level 0: RegFile */
    { TU_MEM_REGFILE,    "RegFile",    256,    1,  4,  1,  1,  1,  1,  0, false },
    /* Level 1: Local SPAD */
    { TU_MEM_LOCAL_SPAD, "LocalSPAD",  65536,  8,  4,  2,  2,  2,  4,  2, true  },
    /* Level 2: Global Buffer */
    { TU_MEM_GLOBAL_BUF, "GlobalBuf",  1048576, 16, 8,  4,  4,  1,  4,  2, false },
    /* Level 3: DRAM */
    { TU_MEM_DRAM,       "DRAM",       0,       1, 64, 50, 50,  1,  1,  0, false },
};

/* ---- Helpers ---- */

static void apply_level_config(tu_memory_hierarchy_t *h, tu_mem_level_t level) {
    h->level_configs[level] = defaults[level];
}

/* ---- Lifecycle ---- */

void tu_mem_hierarchy_init(tu_memory_hierarchy_t *h) {
    memset(h, 0, sizeof(*h));

    /* Set defaults, then override from compile-time config */
    for (int i = 0; i < TU_MEM_NUM_LEVELS; i++) {
        apply_level_config(h, (tu_mem_level_t)i);
    }

    /* Build RegFile model */
    tu_regfile_init(&h->regfile,
                    TU_MEM_REGFILE_PER_PE,
                    TU_PE_ROWS * TU_PE_COLS);

    /* Build Global Buffer */
    tu_mem_level_config_t gcfg = defaults[TU_MEM_GLOBAL_BUF];
    gcfg.size_bytes = TU_MEM_GBUF_SIZE;
    gcfg.bank_count = TU_MEM_GBUF_BANKS;
    gcfg.bank_width  = TU_MEM_GBUF_BANK_WIDTH;
    h->level_configs[TU_MEM_GLOBAL_BUF] = gcfg;
    tu_gbuf_init(&h->gbuf, &gcfg);

    /* Build DRAM model */
    h->dram = tu_dram_create(TU_DRAM_HBM2);
    if (!h->dram) {
        h->dram = tu_dram_create(TU_DRAM_IDEAL);
    }

    /* Override local SPAD config */
    tu_mem_level_config_t lcfg = defaults[TU_MEM_LOCAL_SPAD];
    lcfg.size_bytes       = TU_SRAM_TOTAL;
    lcfg.bank_count       = TU_SRAM_BANKS;
    lcfg.bank_width        = TU_SRAM_BANK_WIDTH;
    lcfg.words_per_cycle   = TU_SRAM_WORDS_PER_CYCLE;
    lcfg.bw_window_cycles  = TU_SRAM_BW_WINDOW_CYCLES;
    lcfg.stall_penalty     = TU_SRAM_BW_STALL_PENALTY;
    h->level_configs[TU_MEM_LOCAL_SPAD] = lcfg;

    h->initialized = true;
}

void tu_mem_hierarchy_set_level_config(tu_memory_hierarchy_t *h,
                                        tu_mem_level_t level,
                                        const tu_mem_level_config_t *config) {
    if (level < TU_MEM_NUM_LEVELS) {
        h->level_configs[level] = *config;
    }
}

void tu_mem_hierarchy_destroy(tu_memory_hierarchy_t *h) {
    tu_gbuf_destroy(&h->gbuf);
    if (h->dram) {
        tu_dram_destroy(h->dram);
        h->dram = NULL;
    }
    memset(h, 0, sizeof(*h));
}

void tu_mem_hierarchy_reset(tu_memory_hierarchy_t *h) {
    /* Keep config but zero stats */
    memset(h->level_reads,  0, sizeof(h->level_reads));
    memset(h->level_writes, 0, sizeof(h->level_writes));
    memset(h->level_bytes_read,  0, sizeof(h->level_bytes_read));
    memset(h->level_bytes_written, 0, sizeof(h->level_bytes_written));
    memset(h->level_stall_cycles, 0, sizeof(h->level_stall_cycles));
    h->current_cycle = 0;

    h->regfile.total_reads  = 0;
    h->regfile.total_writes = 0;

    h->gbuf.total_hits   = 0;
    h->gbuf.total_misses = 0;
    tu_sram_region_t *gs = tu_gbuf_get_sram(&h->gbuf);
    if (gs) {
        gs->banks.reads  = 0;
        gs->banks.writes = 0;
        gs->banks.stall_cycles = 0;
    }

    if (h->dram) {
        tu_dram_reset(h->dram);
    }
}

/* ---- Access Implementation ---- */

/*
 * Resolve level → SRAM region.
 * For LOCAL_SPAD and GLOBAL_BUF, the caller provides the region.
 * For REGFILE, no SRAM needed (modeled as zero-latency).
 * For DRAM, delegates to tu_dram_model.
 */
static int mem_access_impl(tu_memory_hierarchy_t *h,
                            tu_mem_level_t level,
                            tu_sram_region_t *region,
                            uint32_t addr, void *buf, uint32_t bytes,
                            bool is_write,
                            uint64_t *stall_out) {
    if (!h->initialized) return -1;

    uint64_t stall = 0;

    switch (level) {
    case TU_MEM_REGFILE: {
        /* RegFile: zero-latency access (no actual backing store in functional mode).
         * In cycle-accurate mode, model port contention per PE. */
        if (is_write) {
            h->regfile.total_writes++;
        } else {
            h->regfile.total_reads++;
        }
        /* No actual data movement in functional mode — the register file is
         * an abstraction. Reads return zero-initialized data, writes are no-ops.
         * This is correct for a functional cmodel where register allocation
         * is managed by the compiler. */
        if (!is_write && buf) memset(buf, 0, bytes);
        stall = 0;
        break;
    }

    case TU_MEM_LOCAL_SPAD: {
        if (!region) return -1;
        uint32_t word_bytes = region->banks.bank_width;
        uint32_t words = (bytes + word_bytes - 1) / word_bytes;
        for (uint32_t i = 0; i < words; i++) {
            uint32_t off = addr + i * word_bytes;
            if (is_write) {
                const uint8_t *src = (const uint8_t *)buf + i * word_bytes;
                uint32_t copy = (i == words - 1 && bytes % word_bytes != 0)
                                ? bytes % word_bytes : word_bytes;
                /* Pad partial last word with zeros from destination */
                uint8_t tmp[64];
                if (copy < word_bytes) {
                    memcpy(tmp, region->banks.data + off, word_bytes);
                    memcpy(tmp, src, copy);
                    stall += tu_sram_write(region, off, tmp);
                } else {
                    stall += tu_sram_write(region, off, src);
                }
            } else {
                uint8_t tmp[64];
                stall += tu_sram_read(region, off, tmp);
                uint32_t copy = (i == words - 1 && bytes % word_bytes != 0)
                                ? bytes % word_bytes : word_bytes;
                memcpy((uint8_t *)buf + i * word_bytes, tmp, copy);
            }
        }
        break;
    }

    case TU_MEM_GLOBAL_BUF: {
        tu_sram_region_t *gr = tu_gbuf_get_sram(&h->gbuf);
        if (!gr) return -1;
        uint32_t word_bytes = gr->banks.bank_width;
        uint32_t words = (bytes + word_bytes - 1) / word_bytes;
        for (uint32_t i = 0; i < words; i++) {
            uint32_t off = addr + i * word_bytes;
            if (is_write) {
                const uint8_t *src = (const uint8_t *)buf + i * word_bytes;
                stall += tu_sram_write(gr, off, src);
            } else {
                stall += tu_sram_read(gr, off, (uint8_t *)buf + i * word_bytes);
            }
        }
        if (addr + bytes <= gr->total_size) {
            h->gbuf.total_hits++;
        } else {
            h->gbuf.total_misses++;
        }
        break;
    }

    case TU_MEM_DRAM: {
        if (!h->dram) return -1;
        if (is_write) {
            uint64_t cyc = 0, st = 0;
            tu_dram_write(h->dram, addr, bytes, &cyc, &st);
            stall = st;
        } else {
            uint64_t cyc = 0, st = 0;
            tu_dram_read(h->dram, addr, bytes, &cyc, &st);
            stall = st;
        }
        break;
    }

    default:
        return -1;
    }

    /* Update aggregate counters */
    if (is_write) {
        h->level_writes[level]++;
        h->level_bytes_written[level] += bytes;
    } else {
        h->level_reads[level]++;
        h->level_bytes_read[level] += bytes;
    }
    h->level_stall_cycles[level] += stall;

    if (stall_out) *stall_out = stall;
    return 0;
}

int tu_mem_hierarchy_read(tu_memory_hierarchy_t *h,
                           tu_mem_level_t level,
                           tu_sram_region_t *region,
                           uint32_t addr,
                           void *out, uint32_t bytes,
                           uint64_t *stall_out) {
    return mem_access_impl(h, level, region, addr, out, bytes, false, stall_out);
}

int tu_mem_hierarchy_write(tu_memory_hierarchy_t *h,
                            tu_mem_level_t level,
                            tu_sram_region_t *region,
                            uint32_t addr,
                            const void *data, uint32_t bytes,
                            uint64_t *stall_out) {
    return mem_access_impl(h, level, region, addr, (void *)data, bytes, true, stall_out);
}

int tu_mem_hierarchy_read_word(tu_memory_hierarchy_t *h,
                                tu_mem_level_t level,
                                tu_sram_region_t *region,
                                uint32_t addr,
                                void *out, uint32_t word_bytes,
                                uint64_t *stall_out) {
    return tu_mem_hierarchy_read(h, level, region, addr, out, word_bytes, stall_out);
}

int tu_mem_hierarchy_write_word(tu_memory_hierarchy_t *h,
                                 tu_mem_level_t level,
                                 tu_sram_region_t *region,
                                 uint32_t addr,
                                 const void *data, uint32_t word_bytes,
                                 uint64_t *stall_out) {
    return tu_mem_hierarchy_write(h, level, region, addr, data, word_bytes, stall_out);
}

/* ---- Global Buffer ---- */

void tu_gbuf_init(tu_global_buffer_t *gbuf, const tu_mem_level_config_t *config) {
    memset(gbuf, 0, sizeof(*gbuf));
    gbuf->config = *config;
    tu_sram_init_bw(&gbuf->sram, config->size_bytes, config->name,
                    (uint8_t)config->words_per_cycle,
                    TU_SRAM_ARB_ROUND_ROBIN,
                    (uint8_t)config->stall_penalty,
                    config->bw_window_cycles);
    /* Override bank config */
    gbuf->sram.banks.bank_count = config->bank_count;
    gbuf->sram.banks.bank_width = config->bank_width;
}

void tu_gbuf_destroy(tu_global_buffer_t *gbuf) {
    tu_sram_destroy(&gbuf->sram);
    memset(gbuf, 0, sizeof(*gbuf));
}

tu_sram_region_t *tu_gbuf_get_sram(tu_global_buffer_t *gbuf) {
    return &gbuf->sram;
}

bool tu_gbuf_contains(const tu_global_buffer_t *gbuf,
                      uint32_t addr, uint32_t bytes) {
    return addr + bytes <= gbuf->sram.total_size;
}

/* ---- RegFile ---- */

void tu_regfile_init(tu_regfile_model_t *rf,
                     uint32_t size_per_pe, uint32_t num_pes) {
    memset(rf, 0, sizeof(*rf));
    rf->size_per_pe = size_per_pe;
    rf->num_pes     = num_pes;
}

void tu_regfile_record_read(tu_regfile_model_t *rf, uint32_t count) {
    rf->total_reads += count;
}

void tu_regfile_record_write(tu_regfile_model_t *rf, uint32_t count) {
    rf->total_writes += count;
}

/* ---- Statistics ---- */

const char *tu_mem_level_name(tu_mem_level_t level) {
    switch (level) {
    case TU_MEM_REGFILE:     return "RegFile (L0)";
    case TU_MEM_LOCAL_SPAD:  return "LocalSPAD (L1)";
    case TU_MEM_GLOBAL_BUF:  return "GlobalBuf (L2)";
    case TU_MEM_DRAM:        return "DRAM (L3)";
    default:                 return "Unknown";
    }
}

void tu_mem_hierarchy_print_stats(const tu_memory_hierarchy_t *h, FILE *out) {
    fprintf(out, "=== TU Memory Hierarchy Statistics ===\n");
    fprintf(out, "%-16s  %10s %10s %12s %12s %10s\n",
            "Level", "Reads", "Writes", "BytesRead", "BytesWritten", "Stalls");
    fprintf(out, "---------------------------------------------------------------\n");

    for (int i = 0; i < TU_MEM_NUM_LEVELS; i++) {
        fprintf(out, "%-16s  %10lu %10lu %12lu %12lu %10lu\n",
                tu_mem_level_name((tu_mem_level_t)i),
                (unsigned long)h->level_reads[i],
                (unsigned long)h->level_writes[i],
                (unsigned long)h->level_bytes_read[i],
                (unsigned long)h->level_bytes_written[i],
                (unsigned long)h->level_stall_cycles[i]);
    }

    /* RegFile detail */
    fprintf(out, "\nRegFile: %u B/PE × %u PEs = %u B total  |  reads=%lu writes=%lu\n",
            h->regfile.size_per_pe, h->regfile.num_pes,
            h->regfile.size_per_pe * h->regfile.num_pes,
            (unsigned long)h->regfile.total_reads,
            (unsigned long)h->regfile.total_writes);

    /* Global Buffer detail */
    fprintf(out, "GlobalBuf: %u B, %u banks, %u B/bank  |  hits=%lu misses=%lu\n",
            h->gbuf.config.size_bytes,
            h->gbuf.config.bank_count,
            h->gbuf.config.bank_width,
            (unsigned long)h->gbuf.total_hits,
            (unsigned long)h->gbuf.total_misses);

    /* DRAM detail */
    if (h->dram) {
        fprintf(out, "DRAM: ");
        tu_dram_print_stats(h->dram, out);
    }

    fprintf(out, "On-chip total: %lu B  |  Cycle: %lu\n",
            (unsigned long)tu_mem_hierarchy_get_onchip_total(h),
            (unsigned long)h->current_cycle);

    /* GBuf bandwidth utilization */
    tu_sram_region_t *gs = tu_gbuf_get_sram((tu_global_buffer_t *)&h->gbuf);
    if (gs) {
        float u = tu_sram_get_bandwidth_utilization(gs);
        fprintf(out, "GlobalBuf BW util: %.1f%%\n", u * 100.0f);
    }
}

uint64_t tu_mem_hierarchy_get_bytes(const tu_memory_hierarchy_t *h,
                                     tu_mem_level_t level) {
    return h->level_bytes_read[level] + h->level_bytes_written[level];
}

uint64_t tu_mem_hierarchy_get_stalls(const tu_memory_hierarchy_t *h,
                                      tu_mem_level_t level) {
    return h->level_stall_cycles[level];
}

uint64_t tu_mem_hierarchy_get_onchip_total(const tu_memory_hierarchy_t *h) {
    uint64_t total = 0;
    total += (uint64_t)h->regfile.size_per_pe * h->regfile.num_pes;
    total += h->level_configs[TU_MEM_LOCAL_SPAD].size_bytes;
    total += h->level_configs[TU_MEM_GLOBAL_BUF].size_bytes;
    return total;
}

void tu_mem_hierarchy_tick(tu_memory_hierarchy_t *h, uint64_t cycles) {
    h->current_cycle += cycles;
    if (h->dram) {
        for (uint64_t c = 0; c < cycles; c++) {
            tu_dram_tick(h->dram);
        }
    }
}

uint64_t tu_mem_hierarchy_get_cycle(const tu_memory_hierarchy_t *h) {
    return h->current_cycle;
}
