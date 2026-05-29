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
    uint64_t  total_stall_cycles;     /* Cycles stalled on bandwidth contention */
    uint64_t  total_row_conflicts;    /* Row buffer misses */

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

    /* Access queues for contention modeling */
    uint64_t  pending_read_bytes;
    uint64_t  pending_write_bytes;

    /* Per-channel access tracking (for channel parallelism) */
    uint64_t *channel_available_cycle; /* Per-channel next-available cycle */
    uint32_t  num_channels;

} tu_dram_model_t;

/* ---- Lifecycle ---- */

/* Create a DRAM model by type. Returns NULL on invalid type. */
tu_dram_model_t *tu_dram_create(tu_dram_type_t type);

/* Create a custom DRAM model with user-specified parameters. */
tu_dram_model_t *tu_dram_create_custom(const tu_dram_params_t *params,
                                        const char *name);

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

/* Enable/disable row buffer conflict modeling. */
void tu_dram_set_row_modeling(tu_dram_model_t *dram, bool enabled);

#ifdef __cplusplus
}
#endif

#endif /* TU_DRAM_MODEL_H */
