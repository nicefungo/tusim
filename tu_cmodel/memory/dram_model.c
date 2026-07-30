/*
 * TU CModel — DRAM Model Implementation
 * =======================================
 * Supports: ideal, HBM2, HBM2e, HBM3, DDR4, DDR5, LPDDR5, custom.
 */

#include "dram_model.h"
#include "../infra/config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Built-in DRAM Type Parameters ---- */
/* Bandwidth values are aggregate across all channels */
static const tu_dram_params_t dram_presets[] = {
    /* TU_DRAM_TYPE_IDEAL */
    { .clock_ghz = 1.0, .bandwidth_gbps = 1e12, .read_latency_cycles = 0,
      .write_latency_cycles = 0, .bus_width_bytes = 256,
      .burst_length = 64, .channels = 1, .banks_per_channel = 1,
      .row_buffer_size = 0, .model_row_conflicts = false },

    /* TU_DRAM_TYPE_HBM2: ~256 GB/s, 2.0 GT/s, 1024-bit wide, 8 channels */
    { .clock_ghz = 1.0, .bandwidth_gbps = 256.0, .read_latency_cycles = 50,
      .write_latency_cycles = 50, .bus_width_bytes = 128,
      .burst_length = 64, .channels = 8, .banks_per_channel = 16,
      .row_buffer_size = 2048, .model_row_conflicts = false },

    /* TU_DRAM_TYPE_HBM2E: ~460 GB/s */
    { .clock_ghz = 1.0, .bandwidth_gbps = 460.0, .read_latency_cycles = 50,
      .write_latency_cycles = 50, .bus_width_bytes = 128,
      .burst_length = 64, .channels = 8, .banks_per_channel = 16,
      .row_buffer_size = 2048, .model_row_conflicts = false },

    /* TU_DRAM_TYPE_HBM3: ~819 GB/s */
    { .clock_ghz = 1.0, .bandwidth_gbps = 819.0, .read_latency_cycles = 40,
      .write_latency_cycles = 40, .bus_width_bytes = 128,
      .burst_length = 64, .channels = 16, .banks_per_channel = 32,
      .row_buffer_size = 4096, .model_row_conflicts = false },

    /* TU_DRAM_TYPE_DDR4: ~25.6 GB/s, DDR4-3200, 64-bit wide, 1 channel */
    { .clock_ghz = 1.6, .bandwidth_gbps = 25.6, .read_latency_cycles = 75,
      .write_latency_cycles = 75, .bus_width_bytes = 8,
      .burst_length = 64, .channels = 1, .banks_per_channel = 16,
      .row_buffer_size = 1024, .model_row_conflicts = false },

    /* TU_DRAM_TYPE_DDR5: ~51.2 GB/s, DDR5-6400, 64-bit wide */
    { .clock_ghz = 3.2, .bandwidth_gbps = 51.2, .read_latency_cycles = 65,
      .write_latency_cycles = 65, .bus_width_bytes = 8,
      .burst_length = 64, .channels = 1, .banks_per_channel = 32,
      .row_buffer_size = 2048, .model_row_conflicts = false },

    /* TU_DRAM_TYPE_LPDDR5: ~51.2 GB/s, lower latency */
    { .clock_ghz = 3.2, .bandwidth_gbps = 51.2, .read_latency_cycles = 60,
      .write_latency_cycles = 60, .bus_width_bytes = 8,
      .burst_length = 64, .channels = 1, .banks_per_channel = 16,
      .row_buffer_size = 2048, .model_row_conflicts = false },

    /* TU_DRAM_TYPE_CUSTOM (placeholder) */
    { .clock_ghz = 1.0, .bandwidth_gbps = 100.0, .read_latency_cycles = 50,
      .write_latency_cycles = 50, .bus_width_bytes = 32,
      .burst_length = 64, .channels = 4, .banks_per_channel = 16,
      .row_buffer_size = 2048, .model_row_conflicts = false },
};

/* Verify preset count matches enum */
_Static_assert(sizeof(dram_presets)/sizeof(dram_presets[0]) == TU_DRAM_TYPE_COUNT,
               "DRAM preset count must match tu_dram_type_t");

/* ---- Name mapping ---- */
static const char *dram_names[] = {
    "ideal", "HBM2", "HBM2e", "HBM3", "DDR4", "DDR5", "LPDDR5", "custom"
};

static bool allocate_runtime_state(tu_dram_model_t *dram) {
    size_t row_count = (size_t)dram->num_channels * dram->params.banks_per_channel;
    dram->channel_available_cycle = calloc(dram->num_channels, sizeof(uint64_t));
    dram->open_rows = malloc(row_count * sizeof(uint64_t));
    if (!dram->channel_available_cycle || !dram->open_rows) {
        free(dram->channel_available_cycle);
        free(dram->open_rows);
        dram->channel_available_cycle = NULL;
        dram->open_rows = NULL;
        return false;
    }
    for (size_t i = 0; i < row_count; ++i) dram->open_rows[i] = UINT64_MAX;
    return true;
}

const char *tu_dram_type_name(tu_dram_type_t type) {
    if (type >= TU_DRAM_TYPE_COUNT) return "unknown";
    return dram_names[type];
}

/* ---- Lifecycle ---- */

tu_dram_model_t *tu_dram_create(tu_dram_type_t type) {
    if (type >= TU_DRAM_TYPE_COUNT) return NULL;

    tu_dram_model_t *dram = calloc(1, sizeof(tu_dram_model_t));
    if (!dram) return NULL;

    dram->type = type;
    dram->name = tu_dram_type_name(type);
    dram->params = dram_presets[type];
    dram->num_channels = dram->params.channels;
    dram->row_policy = TU_DRAM_ROW_LEGACY;
    dram->address_mapping = TU_DRAM_ADDR_BURST_INTERLEAVED;
    dram->row_miss_penalty_cycles = 10;

    /* Allocate per-channel state */
    if (!allocate_runtime_state(dram)) {
        free(dram);
        return NULL;
    }

    return dram;
}

tu_dram_model_t *tu_dram_create_custom(const tu_dram_params_t *params,
                                        const char *name) {
    if (!params || params->channels == 0 || params->banks_per_channel == 0 ||
        params->burst_length == 0) return NULL;

    tu_dram_model_t *dram = calloc(1, sizeof(tu_dram_model_t));
    if (!dram) return NULL;

    dram->type = TU_DRAM_TYPE_CUSTOM;
    dram->name = name ? name : "custom";
    dram->params = *params;
    dram->num_channels = params->channels;
    dram->row_policy = TU_DRAM_ROW_LEGACY;
    dram->address_mapping = TU_DRAM_ADDR_BURST_INTERLEAVED;
    dram->row_miss_penalty_cycles = 10;

    if (!allocate_runtime_state(dram)) {
        free(dram);
        return NULL;
    }

    return dram;
}

tu_dram_model_t *tu_dram_create_from_config(const struct tu_config_t *cfg) {
    if (!cfg || cfg->dram_type < TU_DRAM_TYPE_IDEAL ||
        cfg->dram_type >= TU_DRAM_TYPE_COUNT) return NULL;
    tu_dram_model_t *dram = tu_dram_create((tu_dram_type_t)cfg->dram_type);
    if (!dram) return NULL;

    dram->params.bandwidth_gbps = cfg->dram_bandwidth_gbps;
    dram->params.read_latency_cycles = (uint32_t)cfg->dram_latency_read;
    dram->params.write_latency_cycles = (uint32_t)cfg->dram_latency_write;
    dram->params.model_row_conflicts = cfg->dram_model_row_conflicts;
    if (cfg->dram_channels > 0 && cfg->dram_channels != dram->num_channels) {
        free(dram->channel_available_cycle);
        free(dram->open_rows);
        dram->channel_available_cycle = NULL;
        dram->open_rows = NULL;
        dram->num_channels = cfg->dram_channels;
        dram->params.channels = cfg->dram_channels;
        if (!allocate_runtime_state(dram)) {
            tu_dram_destroy(dram);
            return NULL;
        }
    }
    if (!tu_dram_set_row_policy(dram,
            (tu_dram_row_policy_mode_t)cfg->dram_row_policy,
            cfg->dram_row_miss_penalty_cycles)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    if (!tu_dram_set_address_mapping(dram,
            (tu_dram_address_mapping_mode_t)cfg->dram_address_mapping)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    return dram;
}

void tu_dram_destroy(tu_dram_model_t *dram) {
    if (!dram) return;
    free(dram->channel_available_cycle);
    free(dram->open_rows);
    free(dram);
}

void tu_dram_reset(tu_dram_model_t *dram) {
    if (!dram) return;
    memset(&dram->stats, 0, sizeof(tu_dram_stats_t));
    dram->current_cycle = 0;
    dram->bandwidth_available = 0;
    dram->pending_read_bytes = 0;
    dram->pending_write_bytes = 0;
    memset(dram->channel_available_cycle, 0,
           dram->num_channels * sizeof(uint64_t));
    size_t row_count = (size_t)dram->num_channels * dram->params.banks_per_channel;
    for (size_t i = 0; i < row_count; ++i) dram->open_rows[i] = UINT64_MAX;
}

/* ---- Internal: bandwidth metering ---- */

static void ensure_bandwidth(tu_dram_model_t *dram) {
    /* Refill bandwidth budget every BW window.
     * BW window = 1e3 cycles by default. */
    if (dram->bw_window_size_cycles == 0) {
        dram->bw_window_size_cycles = 1000;
        dram->bw_window_start = dram->current_cycle;
    }

    if (dram->current_cycle - dram->bw_window_start >= dram->bw_window_size_cycles) {
        /* Replenish: bytes per window = BW_gbps * 1e9 / (core_cycles_per_sec) * window_cycles */
        double core_cycles_per_sec = 1.0e9;  /* 1 GHz assumed core clock */
        double bytes_per_cycle = (dram->params.bandwidth_gbps * 1e9) / core_cycles_per_sec;
        dram->bandwidth_available = (uint64_t)(bytes_per_cycle * dram->bw_window_size_cycles);
        dram->bw_window_start = dram->current_cycle;

        /* Reset pending counters at window boundary */
        dram->pending_read_bytes = 0;
        dram->pending_write_bytes = 0;
    }
}

/* ---- Internal: configurable channel/bank/row address mapping ---- */

bool tu_dram_decode_address(const tu_dram_model_t *dram, uint64_t addr,
                            uint32_t *channel_out, uint32_t *bank_out,
                            uint64_t *row_out) {
    if (!dram || dram->num_channels == 0 ||
        dram->params.banks_per_channel == 0 ||
        dram->params.burst_length == 0 ||
        dram->address_mapping < TU_DRAM_ADDR_BURST_INTERLEAVED ||
        dram->address_mapping > TU_DRAM_ADDR_ROW_INTERLEAVED) return false;

    uint64_t bursts_per_row = dram->params.row_buffer_size /
                              dram->params.burst_length;
    if (bursts_per_row == 0) bursts_per_row = 1;
    uint64_t burst = addr / dram->params.burst_length;
    uint32_t channel;
    uint64_t channel_group;

    if (dram->address_mapping == TU_DRAM_ADDR_ROW_INTERLEAVED) {
        uint64_t row_group = burst / bursts_per_row;
        channel = (uint32_t)(row_group % dram->num_channels);
        channel_group = row_group / dram->num_channels;
    } else {
        channel = (uint32_t)(burst % dram->num_channels);
        uint64_t channel_burst = burst / dram->num_channels;
        channel_group = channel_burst / bursts_per_row;
    }

    uint32_t bank = (uint32_t)(channel_group %
                               dram->params.banks_per_channel);
    uint64_t row = channel_group / dram->params.banks_per_channel;
    if (channel_out) *channel_out = channel;
    if (bank_out) *bank_out = bank;
    if (row_out) *row_out = row;
    return true;
}

static uint32_t addr_to_channel(const tu_dram_model_t *dram, uint64_t addr) {
    uint32_t channel = 0;
    (void)tu_dram_decode_address(dram, addr, &channel, NULL, NULL);
    return channel;
}

static uint64_t explicit_row_penalty(tu_dram_model_t *dram, uint64_t addr) {
    if (dram->row_policy == TU_DRAM_ROW_LEGACY) return 0;

    uint32_t channel = 0, bank = 0;
    uint64_t row = 0;
    if (!tu_dram_decode_address(dram, addr, &channel, &bank, &row)) return 0;

    if (dram->row_policy == TU_DRAM_ROW_CLOSED_PAGE) {
        dram->stats.total_row_conflicts++;
        return dram->row_miss_penalty_cycles;
    }

    size_t idx = (size_t)channel * dram->params.banks_per_channel + bank;
    if (dram->open_rows[idx] == row) {
        dram->stats.total_row_hits++;
        return 0;
    }
    dram->open_rows[idx] = row;
    dram->stats.total_row_conflicts++;
    return dram->row_miss_penalty_cycles;
}

/* ---- Cycle & Timing ---- */

void tu_dram_tick(tu_dram_model_t *dram) {
    if (!dram) return;
    dram->current_cycle++;
    ensure_bandwidth(dram);
}

void tu_dram_read(tu_dram_model_t *dram, uint64_t addr,
                  uint32_t num_bytes, uint64_t *cycles_out, uint64_t *stall_out) {
    if (!dram) {
        if (cycles_out) *cycles_out = 0;
        if (stall_out)  *stall_out = 0;
        return;
    }

    /* Ideal DRAM: zero-cycle, no tracking */
    if (dram->type == TU_DRAM_TYPE_IDEAL) {
        dram->stats.total_reads++;
        dram->stats.total_read_bytes += num_bytes;
        if (cycles_out) *cycles_out = 0;
        if (stall_out)  *stall_out = 0;
        return;
    }

    ensure_bandwidth(dram);

    uint32_t channel = addr_to_channel(dram, addr);
    uint64_t stall = 0;

    /* ---- Latency ---- */
    uint64_t base_latency = dram->params.read_latency_cycles;

    /* ---- Row buffer conflict modeling ---- */
    if (dram->row_policy != TU_DRAM_ROW_LEGACY) {
        base_latency += explicit_row_penalty(dram, addr);
    } else if (dram->params.model_row_conflicts) {
        /* Simple model: add penalty for potential row miss.
         * Full row-buffer state tracking deferred to future heartbeat. */
        base_latency += dram->row_miss_penalty_cycles;
        dram->stats.total_row_conflicts++;
    }

    /* ---- Channel contention ---- */
    if (dram->channel_available_cycle[channel] > dram->current_cycle) {
        stall = dram->channel_available_cycle[channel] - dram->current_cycle;
    }

    /* ---- Bandwidth contention ---- */
    uint64_t consumed_bw = dram->pending_read_bytes + dram->pending_write_bytes + num_bytes;
    if (consumed_bw > dram->bandwidth_available) {
        /* Stall until next BW window */
        stall += dram->bw_window_size_cycles -
                 (dram->current_cycle - dram->bw_window_start);
    }

    /* Update channel availability */
    uint64_t total_cycles = base_latency;
    dram->channel_available_cycle[channel] = dram->current_cycle + total_cycles;
    dram->pending_read_bytes += num_bytes;
    dram->bandwidth_available = (dram->bandwidth_available > num_bytes)
                                ? (dram->bandwidth_available - num_bytes) : 0;

    /* Update statistics */
    dram->stats.total_reads++;
    dram->stats.total_read_bytes += num_bytes;
    dram->stats.total_read_cycles += total_cycles;
    dram->stats.total_stall_cycles += stall;

    if (cycles_out) *cycles_out = total_cycles;
    if (stall_out)  *stall_out = stall;
}

void tu_dram_write(tu_dram_model_t *dram, uint64_t addr,
                   uint32_t num_bytes, uint64_t *cycles_out, uint64_t *stall_out) {
    if (!dram) {
        if (cycles_out) *cycles_out = 0;
        if (stall_out)  *stall_out = 0;
        return;
    }

    if (dram->type == TU_DRAM_TYPE_IDEAL) {
        dram->stats.total_writes++;
        dram->stats.total_write_bytes += num_bytes;
        if (cycles_out) *cycles_out = 0;
        if (stall_out)  *stall_out = 0;
        return;
    }

    ensure_bandwidth(dram);

    uint32_t channel = addr_to_channel(dram, addr);
    uint64_t stall = 0;
    uint64_t base_latency = dram->params.write_latency_cycles;

    if (dram->row_policy != TU_DRAM_ROW_LEGACY)
        base_latency += explicit_row_penalty(dram, addr);

    if (dram->channel_available_cycle[channel] > dram->current_cycle) {
        stall = dram->channel_available_cycle[channel] - dram->current_cycle;
    }

    uint64_t consumed_bw = dram->pending_read_bytes + dram->pending_write_bytes + num_bytes;
    if (consumed_bw > dram->bandwidth_available) {
        stall += dram->bw_window_size_cycles -
                 (dram->current_cycle - dram->bw_window_start);
    }

    uint64_t total_cycles = base_latency;
    dram->channel_available_cycle[channel] = dram->current_cycle + total_cycles;
    dram->pending_write_bytes += num_bytes;
    dram->bandwidth_available = (dram->bandwidth_available > num_bytes)
                                ? (dram->bandwidth_available - num_bytes) : 0;

    dram->stats.total_writes++;
    dram->stats.total_write_bytes += num_bytes;
    dram->stats.total_write_cycles += total_cycles;
    dram->stats.total_stall_cycles += stall;

    if (cycles_out) *cycles_out = total_cycles;
    if (stall_out)  *stall_out = stall;
}

uint64_t tu_dram_estimate_transfer(tu_dram_model_t *dram,
                                    uint32_t num_bytes, bool is_read) {
    if (!dram || dram->type == TU_DRAM_TYPE_IDEAL) return 0;

    /* Simple estimate: bytes / bandwidth * cycles + latency */
    double bw_bytes_per_cycle = (dram->params.bandwidth_gbps * 1e9) / 1.0e9;
    if (bw_bytes_per_cycle <= 0) return num_bytes;

    uint64_t bw_cycles = (uint64_t)ceil(num_bytes / bw_bytes_per_cycle);
    uint64_t latency_cycles = is_read ? dram->params.read_latency_cycles
                                      : dram->params.write_latency_cycles;
    return bw_cycles + latency_cycles;
}

/* ---- Information ---- */

const char *tu_dram_get_name(const tu_dram_model_t *dram) {
    return dram ? dram->name : "null";
}

void tu_dram_get_stats(const tu_dram_model_t *dram, tu_dram_stats_t *stats) {
    if (!dram || !stats) return;
    memcpy(stats, &dram->stats, sizeof(tu_dram_stats_t));

    /* Compute derived metrics */
    uint64_t total_cycles = dram->current_cycle > 0 ? dram->current_cycle : 1;
    double core_cycles_per_sec = 1.0e9;

    stats->effective_read_bandwidth =
        (double)stats->total_read_bytes / total_cycles * core_cycles_per_sec / 1e9;
    stats->effective_write_bandwidth =
        (double)stats->total_write_bytes / total_cycles * core_cycles_per_sec / 1e9;

    double peak_bw = dram->params.bandwidth_gbps;
    double total_bw = stats->effective_read_bandwidth + stats->effective_write_bandwidth;
    stats->utilization = (peak_bw > 0) ? (total_bw / peak_bw) : 0.0;
}

void tu_dram_print_stats(const tu_dram_model_t *dram, FILE *out) {
    if (!dram || !out) return;

    tu_dram_stats_t s;
    tu_dram_get_stats(dram, &s);

    fprintf(out,
        "\n── DRAM Statistics (%s) ──\n"
        "  Type:                  %s\n"
        "  Peak bandwidth:        %.1f GB/s\n"
        "  Channels:              %u\n"
        "  Read latency:          %u cycles\n"
        "  Write latency:         %u cycles\n"
        "  ─────────────────────────────────\n"
        "  Total reads:           %lu\n"
        "  Total writes:          %lu\n"
        "  Read bytes:            %lu\n"
        "  Write bytes:           %lu\n"
        "  Read cycles:           %lu\n"
        "  Write cycles:          %lu\n"
        "  Stall cycles:          %lu\n"
        "  Row conflicts:         %lu\n"
        "  Row hits:              %lu\n"
        "  ─────────────────────────────────\n"
        "  Eff. read BW:          %.2f GB/s\n"
        "  Eff. write BW:         %.2f GB/s\n"
        "  Utilization:           %.1f%%\n"
        "────────────────────────────────\n",
        dram->name, tu_dram_type_name(dram->type),
        dram->params.bandwidth_gbps, dram->num_channels,
        dram->params.read_latency_cycles, dram->params.write_latency_cycles,
        (unsigned long)s.total_reads,
        (unsigned long)s.total_writes,
        (unsigned long)s.total_read_bytes,
        (unsigned long)s.total_write_bytes,
        (unsigned long)s.total_read_cycles,
        (unsigned long)s.total_write_cycles,
        (unsigned long)s.total_stall_cycles,
        (unsigned long)s.total_row_conflicts,
        (unsigned long)s.total_row_hits,
        s.effective_read_bandwidth, s.effective_write_bandwidth,
        s.utilization * 100.0
    );
}

uint64_t tu_dram_peak_bw_per_cycle(const tu_dram_model_t *dram,
                                    double core_clock_ghz) {
    if (!dram || core_clock_ghz <= 0) return 0;
    return (uint64_t)(dram->params.bandwidth_gbps * 1e9 / core_clock_ghz / 1e9);
}

/* ---- Configuration ---- */

void tu_dram_set_core_clock(tu_dram_model_t *dram, double clock_ghz) {
    (void)dram;
    (void)clock_ghz;
    /* Per-instance core clock tracking deferred to future heartbeat. */
}

void tu_dram_set_row_modeling(tu_dram_model_t *dram, bool enabled) {
    if (dram) dram->params.model_row_conflicts = enabled;
}

bool tu_dram_set_row_policy(tu_dram_model_t *dram,
                            tu_dram_row_policy_mode_t policy,
                            uint32_t miss_penalty_cycles) {
    if (!dram || policy < TU_DRAM_ROW_LEGACY ||
        policy > TU_DRAM_ROW_CLOSED_PAGE) return false;
    dram->row_policy = policy;
    dram->row_miss_penalty_cycles = miss_penalty_cycles;
    size_t row_count = (size_t)dram->num_channels * dram->params.banks_per_channel;
    for (size_t i = 0; i < row_count; ++i) dram->open_rows[i] = UINT64_MAX;
    return true;
}

bool tu_dram_set_address_mapping(tu_dram_model_t *dram,
                                 tu_dram_address_mapping_mode_t mapping) {
    if (!dram || mapping < TU_DRAM_ADDR_BURST_INTERLEAVED ||
        mapping > TU_DRAM_ADDR_ROW_INTERLEAVED) return false;
    dram->address_mapping = mapping;
    size_t row_count = (size_t)dram->num_channels * dram->params.banks_per_channel;
    for (size_t i = 0; i < row_count; ++i) dram->open_rows[i] = UINT64_MAX;
    return true;
}
