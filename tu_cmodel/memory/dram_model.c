/*
 * TU CModel — DRAM Model Implementation
 * =======================================
 * Supports: ideal, HBM2, HBM2e, HBM3, DDR4, DDR5, LPDDR5, custom.
 */

#include "dram_model.h"
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

    /* Allocate per-channel state */
    dram->channel_available_cycle = calloc(dram->num_channels, sizeof(uint64_t));
    if (!dram->channel_available_cycle) {
        free(dram);
        return NULL;
    }

    return dram;
}

tu_dram_model_t *tu_dram_create_custom(const tu_dram_params_t *params,
                                        const char *name) {
    if (!params) return NULL;

    tu_dram_model_t *dram = calloc(1, sizeof(tu_dram_model_t));
    if (!dram) return NULL;

    dram->type = TU_DRAM_TYPE_CUSTOM;
    dram->name = name ? name : "custom";
    dram->params = *params;
    dram->num_channels = params->channels;

    dram->channel_available_cycle = calloc(dram->num_channels, sizeof(uint64_t));
    if (!dram->channel_available_cycle) {
        free(dram);
        return NULL;
    }

    return dram;
}

void tu_dram_destroy(tu_dram_model_t *dram) {
    if (!dram) return;
    free(dram->channel_available_cycle);
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

/* ---- Internal: pick channel for address (simple round-robin channel interleaving) ---- */

static uint32_t addr_to_channel(const tu_dram_model_t *dram, uint64_t addr) {
    if (dram->num_channels <= 1) return 0;
    /* Interleave at burst granularity */
    return (addr / dram->params.burst_length) % dram->num_channels;
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
    if (dram->params.model_row_conflicts) {
        /* Simple model: add penalty for potential row miss.
         * Full row-buffer state tracking deferred to future heartbeat. */
        base_latency += 10; /* tRP + tRCD penalty simplified */
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
