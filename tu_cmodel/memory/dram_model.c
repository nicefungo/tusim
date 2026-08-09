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
    dram->refresh_next = malloc(dram->params.banks_per_channel * sizeof(uint64_t));
    dram->refresh_until = malloc(dram->params.banks_per_channel * sizeof(uint64_t));
    if (!dram->channel_available_cycle || !dram->open_rows ||
        !dram->refresh_next || !dram->refresh_until) {
        free(dram->channel_available_cycle);
        free(dram->open_rows);
        free(dram->refresh_next);
        free(dram->refresh_until);
        dram->channel_available_cycle = NULL;
        dram->open_rows = NULL;
        dram->refresh_next = NULL;
        dram->refresh_until = NULL;
        return false;
    }
    for (size_t i = 0; i < row_count; ++i) dram->open_rows[i] = UINT64_MAX;
    for (uint32_t i = 0; i < dram->params.banks_per_channel; ++i) {
        dram->refresh_next[i] = UINT64_MAX;  /* NONE mode: never fires */
        dram->refresh_until[i] = 0;
    }
    return true;
}

static double effective_core_clock(const tu_dram_model_t *dram) {
    return (dram && dram->core_clock_ghz > 0.0) ? dram->core_clock_ghz : 1.0;
}

static bool latency_cycles_for(tu_dram_latency_domain_t domain, double source,
                               double core_clock_ghz, uint32_t *cycles_out) {
    if (!cycles_out || !isfinite(source) || source < 0.0 ||
        source > 100000000.0 || !isfinite(core_clock_ghz) ||
        core_clock_ghz <= 0.0) return false;
    double converted = (domain == TU_DRAM_LATENCY_PHYSICAL_NS)
                           ? ceil(source * core_clock_ghz) : source;
    if (domain < TU_DRAM_LATENCY_CORE_CYCLES ||
        domain > TU_DRAM_LATENCY_PHYSICAL_NS || converted > UINT32_MAX)
        return false;
    /* CORE_CYCLES intentionally preserves the historical truncating cast. */
    *cycles_out = (uint32_t)converted;
    return true;
}

const char *tu_dram_type_name(tu_dram_type_t type) {
    if (type >= TU_DRAM_TYPE_COUNT) return "unknown";
    return dram_names[type];
}

/* ---- Internal: refresh model (JEDEC tREFI/tRFC) ---- */
static void refresh_init_state(tu_dram_model_t *dram); /* used by tu_dram_reset */

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
    dram->row_conflict_penalty_cycles = 10;
    dram->core_clock_ghz = 1.0;
    dram->latency_domain = TU_DRAM_LATENCY_CORE_CYCLES;
    dram->read_latency_source = dram->params.read_latency_cycles;
    dram->write_latency_source = dram->params.write_latency_cycles;

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
    dram->row_conflict_penalty_cycles = 10;
    dram->core_clock_ghz = 1.0;
    dram->latency_domain = TU_DRAM_LATENCY_CORE_CYCLES;
    dram->read_latency_source = params->read_latency_cycles;
    dram->write_latency_source = params->write_latency_cycles;

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
    dram->params.model_row_conflicts = cfg->dram_model_row_conflicts;
    if (!tu_dram_configure_core_clock(dram,
            cfg->dram_core_clock_ghz > 0.0 ? cfg->dram_core_clock_ghz : 1.0)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    if (!tu_dram_set_latency_domain(dram,
            (tu_dram_latency_domain_t)cfg->dram_latency_domain,
            cfg->dram_latency_read, cfg->dram_latency_write)) {
        tu_dram_destroy(dram);
        return NULL;
    }
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
    if (!tu_dram_set_row_policy_timing(dram,
            (tu_dram_row_policy_mode_t)cfg->dram_row_policy,
            cfg->dram_row_miss_penalty_cycles,
            cfg->dram_row_conflict_penalty_cycles
                ? cfg->dram_row_conflict_penalty_cycles
                : cfg->dram_row_miss_penalty_cycles)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    if (!tu_dram_set_address_mapping(dram,
            (tu_dram_address_mapping_mode_t)cfg->dram_address_mapping)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    if (!tu_dram_set_refresh(dram,
            (tu_dram_refresh_mode_t)cfg->dram_refresh_mode,
            (tu_dram_refresh_scheduling_t)cfg->dram_refresh_scheduling,
            cfg->dram_refresh_rate,
            cfg->dram_trefi_ns, cfg->dram_trfc_ns,
            cfg->dram_trfc_pb_ns, cfg->dram_refresh_max_deferral_ns)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    return dram;
}

void tu_dram_destroy(tu_dram_model_t *dram) {
    if (!dram) return;
    free(dram->channel_available_cycle);
    free(dram->open_rows);
    free(dram->refresh_next);
    free(dram->refresh_until);
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
    refresh_init_state(dram);
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
        double core_cycles_per_sec = effective_core_clock(dram) * 1.0e9;
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
        dram->address_mapping > TU_DRAM_ADDR_XOR_INTERLEAVED) return false;

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
        uint32_t base_channel = (uint32_t)(burst % dram->num_channels);
        uint64_t channel_burst = burst / dram->num_channels;
        channel_group = channel_burst / bursts_per_row;
        channel = base_channel;
        if (dram->address_mapping == TU_DRAM_ADDR_XOR_INTERLEAVED) {
            if ((dram->num_channels & (dram->num_channels - 1)) != 0)
                return false;
            channel = base_channel ^
                      (uint32_t)(channel_group & (dram->num_channels - 1));
        }
    }

    uint32_t bank = (uint32_t)(channel_group %
                               dram->params.banks_per_channel);
    uint64_t row = channel_group / dram->params.banks_per_channel;
    if (channel_out) *channel_out = channel;
    if (bank_out) *bank_out = bank;
    if (row_out) *row_out = row;
    return true;
}

static uint64_t explicit_row_penalty(tu_dram_model_t *dram, uint64_t addr) {
    if (dram->row_policy == TU_DRAM_ROW_LEGACY) return 0;

    uint32_t channel = 0, bank = 0;
    uint64_t row = 0;
    if (!tu_dram_decode_address(dram, addr, &channel, &bank, &row)) return 0;

    if (dram->row_policy == TU_DRAM_ROW_CLOSED_PAGE) {
        dram->stats.total_row_conflicts++;
        dram->stats.total_row_empty_misses++;
        return dram->row_miss_penalty_cycles;
    }

    size_t idx = (size_t)channel * dram->params.banks_per_channel + bank;
    if (dram->open_rows[idx] == row) {
        dram->stats.total_row_hits++;
        return 0;
    }
    dram->stats.total_row_conflicts++;
    uint64_t penalty;
    if (dram->open_rows[idx] == UINT64_MAX) {
        dram->stats.total_row_empty_misses++;
        penalty = dram->row_miss_penalty_cycles;
    } else {
        dram->stats.total_row_replacements++;
        penalty = dram->row_conflict_penalty_cycles;
    }
    dram->open_rows[idx] = row;
    return penalty;
}

/* ---- Cycle & Timing ---- */

/* ---- Internal: refresh model (JEDEC tREFI/tRFC) ---- */

/* Convert physical ns into the configured TU/core cycle domain. DRAM command
 * timings remain user-supplied abstractions rather than a DRAM-clock model. */
static uint64_t refresh_ns_to_cycles(const tu_dram_model_t *dram, uint64_t ns) {
    return (uint64_t)ceil((double)ns * effective_core_clock(dram));
}

static uint64_t refresh_effective_interval(const tu_dram_model_t *dram) {
    uint64_t rate = (dram->refresh_rate == 0) ? 1 : dram->refresh_rate;
    uint64_t trefi = dram->refresh_trefi_cycles;
    return (trefi + rate - 1) / rate;  /* ceil division; 1x/2x/4x only */
}

/* (Re)derive per-bank refresh schedule from the active mode/timings.
 * ALL_BANK: a single global schedule (bank 0 slot); first refresh after one
 * full interval so the model does not stall access at cycle 0.
 * PER_BANK: stagger the first refresh of bank b at (b+1)*tREFI/B across the
 * first interval so per-bank commands do not collide (DDR5-style spread). */
static void refresh_init_state(tu_dram_model_t *dram) {
    if (!dram->refresh_next || !dram->refresh_until) return;
    uint32_t B = dram->params.banks_per_channel;
    uint64_t trefi = refresh_effective_interval(dram);
    for (uint32_t i = 0; i < B; ++i) {
        dram->refresh_until[i] = 0;
        dram->refresh_next[i] = UINT64_MAX;
    }
    if (dram->refresh_mode == TU_DRAM_REFRESH_ALL_BANK) {
        dram->refresh_next[0] = trefi;
    } else if (dram->refresh_mode == TU_DRAM_REFRESH_PER_BANK && B > 0) {
        for (uint32_t i = 0; i < B; ++i)
            dram->refresh_next[i] = ((uint64_t)(i + 1) * trefi) / B;
    }
    /* NONE: all slots stay UINT64_MAX — never fires. */
}

/*
 * Bring refresh state up to cycle T.
 * FIXED scheduling fires a due refresh exactly at its schedule (any access in
 * the window pays the remainder). DEFERRED scheduling fires at the earlier of
 * (a) an access to the bank after the schedule (opportunistic, `on_access`)
 * or (b) the hard deadline schedule + max_deferral (`on_access` == false, or
 * when the access arrives after the deadline). The deferred window is clamped
 * to the effective interval so the worst-case actual interval never exceeds
 * 2× the scheduled average.
 * Firing a refresh precharges the row buffer (open-row state is invalidated),
 * which is why row-policy accounting must run after this catch-up.
 */
static void refresh_catchup(tu_dram_model_t *dram, uint64_t T, bool on_access) {
    if (!dram || dram->refresh_mode == TU_DRAM_REFRESH_NONE ||
        dram->type == TU_DRAM_TYPE_IDEAL || !dram->refresh_next ||
        !dram->refresh_until)
        return;
    uint32_t B = dram->params.banks_per_channel;
    uint64_t trefi = refresh_effective_interval(dram);
    uint64_t max_def = dram->refresh_max_deferral_cycles;
    uint64_t dur = (dram->refresh_mode == TU_DRAM_REFRESH_ALL_BANK)
                       ? dram->refresh_trfc_cycles
                       : dram->refresh_trfc_pb_cycles;
    uint32_t slots = (dram->refresh_mode == TU_DRAM_REFRESH_ALL_BANK) ? 1 : B;
    for (uint32_t b = 0; b < slots; ++b) {
        while (dram->refresh_next[b] <= T) {
            bool fire = false;
            uint64_t fire_at = 0;
            if (dram->refresh_scheduling ==
                TU_DRAM_REFRESH_SCHEDULING_FIXED) {
                fire = true;
                fire_at = dram->refresh_next[b];
            } else {
                uint64_t deadline = dram->refresh_next[b] + max_def;
                if (on_access) {
                    fire = true;
                    fire_at = (deadline > T) ? T : deadline;
                } else if (deadline <= T) {
                    fire = true;
                    fire_at = deadline;
                }
            }
            if (!fire) break;  /* deferred: still within window, no access yet */
            dram->refresh_until[b] = fire_at + dur;
            dram->refresh_next[b] += trefi;
            dram->stats.total_refresh_events++;
            if (dram->row_policy != TU_DRAM_ROW_LEGACY) {
                /* Refresh precharges the addressed bank(s) on every channel. */
                if (dram->refresh_mode == TU_DRAM_REFRESH_ALL_BANK) {
                    size_t row_count = (size_t)dram->num_channels * B;
                    for (size_t i = 0; i < row_count; ++i)
                        dram->open_rows[i] = UINT64_MAX;
                } else {
                    for (uint32_t ch = 0; ch < dram->num_channels; ++ch)
                        dram->open_rows[(size_t)ch * B + b] = UINT64_MAX;
                }
            }
        }
    }
}

/* Cycles an access to `bank` must wait because its bank is refreshing. */
static uint64_t refresh_stall_for(const tu_dram_model_t *dram, uint32_t bank,
                                  uint64_t T) {
    if (!dram || dram->refresh_mode == TU_DRAM_REFRESH_NONE ||
        dram->type == TU_DRAM_TYPE_IDEAL || !dram->refresh_until)
        return 0;
    uint32_t slot = (dram->refresh_mode == TU_DRAM_REFRESH_ALL_BANK) ? 0 : bank;
    if (dram->refresh_until[slot] > T)
        return dram->refresh_until[slot] - T;
    return 0;
}

void tu_dram_tick(tu_dram_model_t *dram) {
    if (!dram) return;
    dram->current_cycle++;
    ensure_bandwidth(dram);
    /* Fire refreshes that reach their schedule/deadline even with no traffic
     * so event counters and forced-lockout state stay accurate. */
    refresh_catchup(dram, dram->current_cycle, false);
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

    uint32_t channel = 0, bank = 0;
    (void)tu_dram_decode_address(dram, addr, &channel, &bank, NULL);
    uint64_t stall = 0;

    /* ---- Refresh lockout: catch up schedules, then wait out any window.
     * Runs before row accounting because refresh precharges the row buffer. */
    refresh_catchup(dram, dram->current_cycle, true);
    uint64_t refresh_stall = refresh_stall_for(dram, bank, dram->current_cycle);

    /* ---- Latency ---- */
    uint64_t base_latency = dram->params.read_latency_cycles + refresh_stall;

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
    dram->stats.total_refresh_stall_cycles += refresh_stall;

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

    uint32_t channel = 0, bank = 0;
    (void)tu_dram_decode_address(dram, addr, &channel, &bank, NULL);
    uint64_t stall = 0;

    /* Refresh lockout before row accounting (refresh precharges rows). */
    refresh_catchup(dram, dram->current_cycle, true);
    uint64_t refresh_stall = refresh_stall_for(dram, bank, dram->current_cycle);

    uint64_t base_latency = dram->params.write_latency_cycles + refresh_stall;

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
    dram->stats.total_refresh_stall_cycles += refresh_stall;

    if (cycles_out) *cycles_out = total_cycles;
    if (stall_out)  *stall_out = stall;
}

uint64_t tu_dram_estimate_transfer(tu_dram_model_t *dram,
                                    uint32_t num_bytes, bool is_read) {
    if (!dram || dram->type == TU_DRAM_TYPE_IDEAL) return 0;

    /* Simple estimate: bytes / bandwidth * cycles + latency */
    double bw_bytes_per_cycle = dram->params.bandwidth_gbps /
                                effective_core_clock(dram);
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
    double core_cycles_per_sec = effective_core_clock(dram) * 1.0e9;

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
        "  Empty-row activates:   %lu\n"
        "  Open-row replacements: %lu\n"
        "  Refresh events:        %lu\n"
        "  Refresh stall cycles:  %lu\n"
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
        (unsigned long)s.total_row_empty_misses,
        (unsigned long)s.total_row_replacements,
        (unsigned long)s.total_refresh_events,
        (unsigned long)s.total_refresh_stall_cycles,
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
    (void)tu_dram_configure_core_clock(dram, clock_ghz);
}

bool tu_dram_configure_core_clock(tu_dram_model_t *dram, double clock_ghz) {
    if (!dram || !isfinite(clock_ghz) || clock_ghz <= 0.0 || clock_ghz > 10.0)
        return false;
    uint32_t read_cycles, write_cycles;
    if (!latency_cycles_for(dram->latency_domain, dram->read_latency_source,
                            clock_ghz, &read_cycles) ||
        !latency_cycles_for(dram->latency_domain, dram->write_latency_source,
                            clock_ghz, &write_cycles)) return false;
    dram->core_clock_ghz = clock_ghz;
    dram->params.read_latency_cycles = read_cycles;
    dram->params.write_latency_cycles = write_cycles;
    dram->bw_window_size_cycles = 0;
    dram->bw_window_start = dram->current_cycle;
    dram->bandwidth_available = 0;
    dram->pending_read_bytes = 0;
    dram->pending_write_bytes = 0;
    if (dram->refresh_trefi_ns != 0) {
        dram->refresh_trefi_cycles = refresh_ns_to_cycles(dram, dram->refresh_trefi_ns);
        dram->refresh_trfc_cycles = refresh_ns_to_cycles(dram, dram->refresh_trfc_ns);
        dram->refresh_trfc_pb_cycles = refresh_ns_to_cycles(dram, dram->refresh_trfc_pb_ns);
        dram->refresh_max_deferral_cycles =
            refresh_ns_to_cycles(dram, dram->refresh_max_deferral_ns);
        refresh_init_state(dram);
    }
    return true;
}

bool tu_dram_set_latency_domain(tu_dram_model_t *dram,
                                tu_dram_latency_domain_t domain,
                                double read_latency, double write_latency) {
    if (!dram) return false;
    uint32_t read_cycles, write_cycles;
    double clock = effective_core_clock(dram);
    if (!latency_cycles_for(domain, read_latency, clock, &read_cycles) ||
        !latency_cycles_for(domain, write_latency, clock, &write_cycles))
        return false;
    dram->latency_domain = domain;
    dram->read_latency_source = read_latency;
    dram->write_latency_source = write_latency;
    dram->params.read_latency_cycles = read_cycles;
    dram->params.write_latency_cycles = write_cycles;
    return true;
}

void tu_dram_set_row_modeling(tu_dram_model_t *dram, bool enabled) {
    if (dram) dram->params.model_row_conflicts = enabled;
}

bool tu_dram_set_row_policy(tu_dram_model_t *dram,
                            tu_dram_row_policy_mode_t policy,
                            uint32_t miss_penalty_cycles) {
    return tu_dram_set_row_policy_timing(dram, policy,
                                         miss_penalty_cycles,
                                         miss_penalty_cycles);
}

bool tu_dram_set_row_policy_timing(tu_dram_model_t *dram,
                                   tu_dram_row_policy_mode_t policy,
                                   uint32_t activate_penalty_cycles,
                                   uint32_t conflict_penalty_cycles) {
    if (!dram || policy < TU_DRAM_ROW_LEGACY ||
        policy > TU_DRAM_ROW_CLOSED_PAGE) return false;
    dram->row_policy = policy;
    dram->row_miss_penalty_cycles = activate_penalty_cycles;
    dram->row_conflict_penalty_cycles = conflict_penalty_cycles;
    size_t row_count = (size_t)dram->num_channels * dram->params.banks_per_channel;
    for (size_t i = 0; i < row_count; ++i) dram->open_rows[i] = UINT64_MAX;
    return true;
}

bool tu_dram_set_address_mapping(tu_dram_model_t *dram,
                                 tu_dram_address_mapping_mode_t mapping) {
    if (!dram || mapping < TU_DRAM_ADDR_BURST_INTERLEAVED ||
        mapping > TU_DRAM_ADDR_XOR_INTERLEAVED) return false;
    if (mapping == TU_DRAM_ADDR_XOR_INTERLEAVED &&
        (dram->num_channels == 0 ||
         (dram->num_channels & (dram->num_channels - 1)) != 0)) return false;
    dram->address_mapping = mapping;
    size_t row_count = (size_t)dram->num_channels * dram->params.banks_per_channel;
    for (size_t i = 0; i < row_count; ++i) dram->open_rows[i] = UINT64_MAX;
    return true;
}

bool tu_dram_set_refresh(tu_dram_model_t *dram,
                         tu_dram_refresh_mode_t mode,
                         tu_dram_refresh_scheduling_t scheduling,
                         uint32_t rate,
                         uint64_t trefi_ns, uint64_t trfc_ns,
                         uint64_t trfc_pb_ns, uint64_t max_deferral_ns) {
    if (!dram || !dram->refresh_next || !dram->refresh_until) return false;
    if (mode < TU_DRAM_REFRESH_NONE || mode > TU_DRAM_REFRESH_PER_BANK)
        return false;
    if (scheduling < TU_DRAM_REFRESH_SCHEDULING_FIXED ||
        scheduling > TU_DRAM_REFRESH_SCHEDULING_DEFERRED)
        return false;
    if (rate != 0 && rate != 1 && rate != 2 && rate != 4) return false;

    /* Zero fields mean "use defaults" (legacy zero-initialized callers). */
    uint64_t trefi = (trefi_ns == 0) ? 7800 : trefi_ns;
    uint64_t trfc  = (trfc_ns == 0) ? 350 : trfc_ns;
    uint64_t trfc_pb = (trfc_pb_ns == 0) ? 90 : trfc_pb_ns;
    uint64_t max_def = (max_deferral_ns == 0) ? trefi : max_deferral_ns;
    if (max_def > trefi) return false;  /* cannot defer past next schedule */
    if (mode != TU_DRAM_REFRESH_NONE &&
        (trefi == 0 || trfc == 0 || trfc_pb == 0)) return false;

    dram->refresh_mode = mode;
    dram->refresh_scheduling = scheduling;
    dram->refresh_rate = (rate == 0) ? 1 : rate;
    dram->refresh_trefi_ns = trefi;
    dram->refresh_trfc_ns = trfc;
    dram->refresh_trfc_pb_ns = trfc_pb;
    dram->refresh_max_deferral_ns = max_def;
    dram->refresh_trefi_cycles = refresh_ns_to_cycles(dram, trefi);
    dram->refresh_trfc_cycles = refresh_ns_to_cycles(dram, trfc);
    dram->refresh_trfc_pb_cycles = refresh_ns_to_cycles(dram, trfc_pb);
    dram->refresh_max_deferral_cycles = refresh_ns_to_cycles(dram, max_def);

    refresh_init_state(dram);
    return true;
}
