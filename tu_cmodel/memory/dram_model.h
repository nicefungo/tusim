/*
 * TU CModel — DRAM Model
 * =======================
 * Multi-type DRAM interface with configurable bandwidth, latency,
 * burst modeling, and contention detection.
 *
 * Gap M1: No DRAM model → Pluggable DRAM with BW/latency/contention.
 *
 * Architecture:
 *   The DRAM model provides a unified memory-level interface for
 *   off-chip memory. It supports multiple DRAM types (ideal, HBM2,
 *   HBM3, DDR5, LPDDR5) with configurable bandwidth and latency.
 *
 *   Bandwidth is modeled as bytes-per-cycle capacity. Contention
 *   arises when multiple concurrent accesses exceed the available
 *   bandwidth, causing stall cycles.
 *
 *   The model is pluggable: new DRAM types can be registered at
 *   runtime. The DMA engine queries the DRAM model for transfer
 *   timing estimates.
 */

#ifndef TU_DRAM_MODEL_H
#define TU_DRAM_MODEL_H

#include "../tu_config.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tu_config_t;

/* ---- DRAM Types ---- */
typedef enum {
    TU_DRAM_TYPE_IDEAL       = 0,  /* Zero-latency, infinite bandwidth */
    TU_DRAM_TYPE_HBM2        = 1,  /* 256 GB/s, ~100 ns latency */
    TU_DRAM_TYPE_HBM2E       = 2,  /* 460 GB/s, ~100 ns latency */
    TU_DRAM_TYPE_HBM3        = 3,  /* 819 GB/s, ~80 ns latency */
    TU_DRAM_TYPE_DDR4        = 4,  /* 25.6 GB/s, ~75 ns latency */
    TU_DRAM_TYPE_DDR5        = 5,  /* 51.2 GB/s, ~65 ns latency */
    TU_DRAM_TYPE_LPDDR5      = 6,  /* 51.2 GB/s, ~60 ns latency */
    TU_DRAM_TYPE_CUSTOM      = 7,  /* User-defined parameters */
    TU_DRAM_TYPE_COUNT
} tu_dram_type_t;

typedef enum {
    TU_DRAM_ROW_LEGACY = 0,
    TU_DRAM_ROW_OPEN_PAGE = 1,
    TU_DRAM_ROW_CLOSED_PAGE = 2,
    TU_DRAM_ROW_ADAPTIVE_TIMEOUT = 3
} tu_dram_row_policy_mode_t;

typedef enum {
    TU_DRAM_ADDR_BURST_INTERLEAVED = 0,
    TU_DRAM_ADDR_ROW_INTERLEAVED = 1,
    TU_DRAM_ADDR_XOR_INTERLEAVED = 2
} tu_dram_address_mapping_mode_t;

typedef enum {
    TU_DRAM_LATENCY_CORE_CYCLES = 0,
    TU_DRAM_LATENCY_PHYSICAL_NS = 1
} tu_dram_latency_domain_t;

typedef enum {
    TU_DRAM_ROW_TIMEOUT_CORE_CYCLES = 0,
    TU_DRAM_ROW_TIMEOUT_PHYSICAL_NS = 1
} tu_dram_row_timeout_domain_t;

typedef enum {
    TU_DRAM_TURNAROUND_NONE = 0,
    TU_DRAM_TURNAROUND_FIXED = 1,
    TU_DRAM_TURNAROUND_IDLE_CREDIT = 2,
    TU_DRAM_TURNAROUND_BURST_CREDIT = 3,
    TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT = 4
} tu_dram_turnaround_mode_t;

typedef enum {
    TU_DRAM_TURNAROUND_CORE_CYCLES = 0,
    TU_DRAM_TURNAROUND_PHYSICAL_NS = 1
} tu_dram_turnaround_domain_t;

/* Refresh model (JEDEC tREFI/tRFC). Subsystem enum family is disjoint from
 * the generated TU_DRAM_REFRESH_MODE_* / TU_DRAM_REFRESH_SCHED_* macros and
 * the canonical TU_DRAM_CONFIG_REFRESH_* enums (preprocessor namespace). */
typedef enum {
    TU_DRAM_REFRESH_NONE = 0,      /* Compatibility: no refresh overhead */
    TU_DRAM_REFRESH_ALL_BANK = 1,  /* Traditional REFAB: whole device locks */
    TU_DRAM_REFRESH_PER_BANK = 2   /* DDR5-style staggered per-bank refresh */
} tu_dram_refresh_mode_t;

typedef enum {
    TU_DRAM_REFRESH_SCHEDULING_FIXED = 0,    /* Refresh at exact tREFI schedule */
    TU_DRAM_REFRESH_SCHEDULING_DEFERRED = 1  /* Bounded postponement (≤ max deferral) */
} tu_dram_refresh_scheduling_t;

/* ---- DRAM Timing Parameters ---- */
typedef struct {
    double    clock_ghz;          /* DRAM clock frequency in GHz */
    double    bandwidth_gbps;     /* Peak bandwidth in GB/s */
    uint32_t  read_latency_cycles;  /* CAS latency + bus turnaround (in DRAM cycles) */
    uint32_t  write_latency_cycles; /* Write latency (in DRAM cycles) */
    uint32_t  bus_width_bytes;    /* DRAM bus width in bytes */
    uint32_t  burst_length;       /* Bytes per burst (typically 64 for DDR) */
    uint32_t  channels;           /* Number of independent channels */
    uint32_t  banks_per_channel;  /* Banks per channel */
    uint32_t  row_buffer_size;    /* Row buffer size in bytes */
    bool      model_row_conflicts; /* Model row-buffer hit/miss */
} tu_dram_params_t;

/* ---- DRAM Access Statistics ---- */
typedef struct {
    /* Access counters */
    uint64_t  total_reads;
    uint64_t  total_writes;
    uint64_t  total_read_bytes;
    uint64_t  total_write_bytes;

    /* Cycle accounting */
    uint64_t  total_read_cycles;      /* Aggregate read cycles */
    uint64_t  total_write_cycles;     /* Aggregate write cycles */
    uint64_t  total_stall_cycles;     /* Cycles stalled on contention */
    uint64_t  total_row_conflicts;    /* Row buffer misses */
    uint64_t  total_row_hits;         /* Reuse of an already-open bank row */
    uint64_t  total_row_empty_misses; /* Activate from precharged/closed state */
    uint64_t  total_row_replacements; /* Precharge an open row, then activate */
    uint64_t  total_row_timeout_precharges; /* Lazy adaptive idle closures */
    uint64_t  total_turnaround_events;  /* Read/write direction changes */
    uint64_t  total_turnaround_cycles;  /* Returned-service turnaround cost */

    /* Refresh accounting (JEDEC tREFI/tRFC) */
    uint64_t  total_refresh_events;       /* Refresh commands issued */
    uint64_t  total_refresh_stall_cycles; /* Access cycles lost to refresh lockout */

    /* Derived metrics */
    double    effective_read_bandwidth;   /* Actual achieved read BW in GB/s */
    double    effective_write_bandwidth;  /* Actual achieved write BW in GB/s */
    double    utilization;                /* Bandwidth utilization (0.0–1.0) */

} tu_dram_stats_t;

/* ---- DRAM Model Instance ---- */
typedef struct {
    tu_dram_type_t    type;
    const char       *name;
    tu_dram_params_t  params;
    tu_dram_stats_t   stats;

    /* Runtime state */
    uint64_t  current_cycle;            /* Simulator time reference */
    uint64_t  bandwidth_available;     /* Bytes of BW remaining in current window */
    uint64_t  bw_window_size_cycles;   /* Size of BW metering window */
    uint64_t  bw_window_start;         /* Start cycle of current BW window */
    double    core_clock_ghz;           /* TU/core clock defining one sim cycle */
    tu_dram_latency_domain_t latency_domain;
    double    read_latency_source;      /* cycles or ns according to domain */
    double    write_latency_source;     /* cycles or ns according to domain */

    /* Access queues for contention modeling */
    uint64_t  pending_read_bytes;
    uint64_t  pending_write_bytes;

    /* Per-channel access tracking (for channel parallelism) */
    uint64_t *channel_available_cycle; /* Per-channel next-available cycle */
    uint8_t  *channel_last_direction; /* 0=none, 1=read, 2=write */
    uint32_t  num_channels;
    tu_dram_row_policy_mode_t row_policy;
    tu_dram_address_mapping_mode_t address_mapping;
    uint32_t row_miss_penalty_cycles;     /* Activate from closed state */
    uint32_t row_conflict_penalty_cycles; /* Precharge + activate replacement */
    uint32_t row_open_timeout_cycles;     /* Adaptive idle threshold */
    tu_dram_row_timeout_domain_t row_timeout_domain;
    double row_open_timeout_source;       /* cycles or ns according to domain */
    uint64_t *open_rows;              /* channel×bank rows; UINT64_MAX=none */
    uint64_t *row_last_access_cycle;   /* channel×bank; UINT64_MAX=never */

    /* Per-channel bidirectional data-bus turnaround. */
    tu_dram_turnaround_mode_t turnaround_mode;
    tu_dram_turnaround_domain_t turnaround_domain;
    double read_to_write_turnaround_source; /* cycles or ns by domain */
    double write_to_read_turnaround_source; /* cycles or ns by domain */
    uint32_t read_to_write_turnaround_cycles;
    uint32_t write_to_read_turnaround_cycles;

    /* Refresh state (JEDEC tREFI/tRFC; ns converted at core_clock_ghz) */
    tu_dram_refresh_mode_t      refresh_mode;
    tu_dram_refresh_scheduling_t refresh_scheduling;
    uint32_t refresh_rate;                /* 1x, 2x, 4x; high-temp retention */
    uint64_t refresh_trefi_cycles;        /* scheduled per-bank interval (rate-adjusted) */
    uint64_t refresh_trfc_cycles;         /* all-bank lockout duration */
    uint64_t refresh_trfc_pb_cycles;      /* per-bank lockout duration */
    uint64_t refresh_max_deferral_cycles; /* deferred hard deadline (≤ trefi) */
    uint64_t refresh_trefi_ns;            /* source timings retained for clock changes */
    uint64_t refresh_trfc_ns;
    uint64_t refresh_trfc_pb_ns;
    uint64_t refresh_max_deferral_ns;
    uint64_t *refresh_next;               /* per-bank next scheduled refresh cycle */
    uint64_t *refresh_until;              /* per-bank busy-until; 0 = idle */

} tu_dram_model_t;

/* ---- Lifecycle ---- */

/* Create a DRAM model by type. Returns NULL on invalid type. */
tu_dram_model_t *tu_dram_create(tu_dram_type_t type);

/* Create a custom DRAM model with user-specified parameters. */
tu_dram_model_t *tu_dram_create_custom(const tu_dram_params_t *params,
                                        const char *name);

tu_dram_model_t *tu_dram_create_from_config(const struct tu_config_t *cfg);

/* Destroy and free a DRAM model. */
void tu_dram_destroy(tu_dram_model_t *dram);

/* Reset statistics and internal state. */
void tu_dram_reset(tu_dram_model_t *dram);

/* ---- Cycle & Timing ---- */

/* Advance the DRAM simulator clock. Must be called periodically. */
void tu_dram_tick(tu_dram_model_t *dram);

/*
 * Schedule a read access. Returns the number of cycles required
 * for this access, accounting for latency, bandwidth contention,
 * and row-buffer effects (if enabled).
 *
 *   dram:       DRAM model
 *   addr:       Byte address in DRAM space
 *   num_bytes:  Number of bytes to read
 *   cycles:     Output: total cycles for this access
 *   stall:      Output: stall cycles due to contention (0 = no stall)
 */
void tu_dram_read(tu_dram_model_t *dram, uint64_t addr,
                  uint32_t num_bytes, uint64_t *cycles, uint64_t *stall);

/*
 * Schedule a write access. Same semantics as tu_dram_read().
 */
void tu_dram_write(tu_dram_model_t *dram, uint64_t addr,
                   uint32_t num_bytes, uint64_t *cycles, uint64_t *stall);

/*
 * Estimate cycles for a bulk transfer of `num_bytes` bytes without
 * actually executing it. Returns total cycles required.
 * Useful for DMA engine planning.
 */
uint64_t tu_dram_estimate_transfer(tu_dram_model_t *dram,
                                    uint32_t num_bytes, bool is_read);

/* ---- Information ---- */

/* Get DRAM type name string. */
const char *tu_dram_type_name(tu_dram_type_t type);

/* Get human-readable DRAM model name. */
const char *tu_dram_get_name(const tu_dram_model_t *dram);

/* Get a copy of the current statistics. */
void tu_dram_get_stats(const tu_dram_model_t *dram, tu_dram_stats_t *stats);

/* Print a formatted DRAM statistics report. */
void tu_dram_print_stats(const tu_dram_model_t *dram, FILE *out);

/* Get peak bandwidth in bytes per cycle (at the core clock). */
uint64_t tu_dram_peak_bw_per_cycle(const tu_dram_model_t *dram,
                                    double core_clock_ghz);

/* ---- Configuration ---- */

/* Set the core clock frequency (GHz) for bandwidth calculations. */
void tu_dram_set_core_clock(tu_dram_model_t *dram, double core_clock_ghz);

/* Validated core-clock setter. Recomputes bandwidth and refresh cycle-domain
 * state; returns false without mutation for non-finite/out-of-range clocks. */
bool tu_dram_configure_core_clock(tu_dram_model_t *dram, double core_clock_ghz);

/* Select whether read/write base latency values are fixed TU/core cycles or
 * physical nanoseconds converted with ceil(ns * core_clock_ghz). */
bool tu_dram_set_latency_domain(tu_dram_model_t *dram,
                                tu_dram_latency_domain_t domain,
                                double read_latency, double write_latency);

/* Enable/disable row buffer conflict modeling. */
void tu_dram_set_row_modeling(tu_dram_model_t *dram, bool enabled);

bool tu_dram_set_row_policy(tu_dram_model_t *dram,
                            tu_dram_row_policy_mode_t policy,
                            uint32_t miss_penalty_cycles);

/* Configure separate closed-bank activation and open-row replacement costs.
 * The legacy setter above assigns both costs the same value. */
bool tu_dram_set_row_policy_timing(tu_dram_model_t *dram,
                                   tu_dram_row_policy_mode_t policy,
                                   uint32_t activate_penalty_cycles,
                                   uint32_t conflict_penalty_cycles);

bool tu_dram_set_row_policy_timeout(tu_dram_model_t *dram,
                                    tu_dram_row_policy_mode_t policy,
                                    uint32_t activate_penalty_cycles,
                                    uint32_t conflict_penalty_cycles,
                                    uint32_t timeout_cycles);

/* Configure the adaptive timeout in fixed core cycles or physical ns. The
 * legacy timeout setter above remains a CORE_CYCLES compatibility wrapper. */
bool tu_dram_set_row_policy_timeout_domain(
    tu_dram_model_t *dram, tu_dram_row_policy_mode_t policy,
    uint32_t activate_penalty_cycles, uint32_t conflict_penalty_cycles,
    tu_dram_row_timeout_domain_t domain, double timeout_value);

bool tu_dram_set_address_mapping(tu_dram_model_t *dram,
                                 tu_dram_address_mapping_mode_t mapping);

/* Model a shared bidirectional channel bus. NONE is the compatibility mode.
 * FIXED charges the full directional cost on every direction change.
 * IDLE_CREDIT subtracts channel-idle cycles since prior base service completed.
 * BURST_CREDIT delays that boundary by ceil(bytes / channel bus width), a
 * conservative serialized-data-burst alternative.
 * BURST_ROUND_CREDIT first rounds bytes up to the configured protocol burst,
 * representing fixed-granularity DRAM transfers for sub-burst/tail accesses.
 * Values are core cycles or physical ns. */
bool tu_dram_set_turnaround(tu_dram_model_t *dram,
                            tu_dram_turnaround_mode_t mode,
                            tu_dram_turnaround_domain_t domain,
                            double read_to_write, double write_to_read);

/* Configure the refresh model. ns timings are converted into the configured
 * TU/core-clock cycle domain. `rate` 0 is treated
 * as 1x; `max_deferral_ns` 0 defaults to tREFI and is clamped to the effective
 * interval internally. Returns false for unsupported mode/scheduling/rate or
 * for max_deferral > tREFI. */
bool tu_dram_set_refresh(tu_dram_model_t *dram,
                         tu_dram_refresh_mode_t mode,
                         tu_dram_refresh_scheduling_t scheduling,
                         uint32_t rate,
                         uint64_t trefi_ns, uint64_t trfc_ns,
                         uint64_t trfc_pb_ns, uint64_t max_deferral_ns);

/* Decode an address according to the active channel/bank/row mapping. */
bool tu_dram_decode_address(const tu_dram_model_t *dram, uint64_t addr,
                            uint32_t *channel, uint32_t *bank, uint64_t *row);

#ifdef __cplusplus
}
#endif

#endif /* TU_DRAM_MODEL_H */
