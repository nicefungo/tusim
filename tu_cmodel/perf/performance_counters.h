/*
 * TU CModel — Performance Counter Infrastructure
 * ===============================================
 *
 * Comprehensive performance monitoring for production-grade
 * accelerator modeling. Provides cycle-accurate counting,
 * utilization tracking, stall analysis, and energy estimation.
 *
 * Architecture:
 *   Per-component counters (DMA, compute, memory, power)
 *   Global aggregation with snapshot/merge capability
 *   Design principle: all counters are monotonically increasing;
 *   snapshots capture point-in-time values for differential analysis.
 *
 * Gap: E4 (Power/Energy Model), P2.5 (Cycle-Accurate Model) foundation
 */

#ifndef TU_PERFORMANCE_COUNTERS_H
#define TU_PERFORMANCE_COUNTERS_H

#include "../tu_config.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * DM3: Scatter/Gather DMA — decl for perf counter tracking
 * ================================================================ */

/* ---- Counter categories ---- */

/* DMA transfer counters */
typedef struct {
    uint64_t dma_read_bytes;            /* Bytes transferred DRAM→SRAM */
    uint64_t dma_write_bytes;           /* Bytes transferred SRAM→DRAM */
    uint64_t dma_internal_bytes;        /* SRAM→SRAM transfers */
    uint64_t dma_read_cycles;           /* Active cycles for reads */
    uint64_t dma_write_cycles;          /* Active cycles for writes */
    uint64_t dma_stall_cycles;          /* Cycles stalled waiting for memory */
    uint64_t dma_transfers_linear;      /* Count: linear transfers */
    uint64_t dma_transfers_strided_2d;  /* Count: 2D strided */
    uint64_t dma_transfers_strided_3d;  /* Count: 3D strided */
    uint64_t dma_transfers_scatter;     /* Count: scatter (DM3) */
    uint64_t dma_transfers_gather;      /* Count: gather (DM3) */
    uint64_t dma_channel_stalls[8];     /* Per-channel stall cycles */
    uint64_t dma_channel_bytes[8];      /* Per-channel bytes transferred */
} tu_dma_counters_t;

/* Compute engine counters */
typedef struct {
    uint64_t compute_total_cycles;      /* Total cycles compute engine was busy */
    uint64_t compute_active_cycles;     /* Cycles doing useful work (non-idle) */
    uint64_t compute_stall_cycles;      /* Cycles stalled waiting for input data */
    uint64_t compute_idle_cycles;       /* Cycles idle (no work available) */
    uint64_t compute_pipeline_bubbles;  /* Pipeline empty slots */
    uint64_t total_macs;                /* Total multiply-accumulate operations */
    uint64_t total_flops;                /* Total floating-point operations (2× MACs) */
    float    compute_utilization;       /* active / total ratio [0.0, 1.0] */

    /* Per-dataflow utilization */
    uint64_t df_ws_cycles;              /* Weight-stationary cycles */
    uint64_t df_os_cycles;              /* Output-stationary cycles */

    /* Per-operation counters */
    uint64_t op_mma_fp16;               /* FP16 MMA operations */
    uint64_t op_mma_bf16;               /* BF16 MMA operations */
    uint64_t op_mma_int8;               /* INT8 MMA operations */
    uint64_t op_mma_fp8;                /* FP8 MMA operations */
    uint64_t op_conv2d;                 /* 2D convolution operations */
    uint64_t op_attention;              /* Attention operations */
    uint64_t op_elementwise;            /* Elementwise operations */
    uint64_t op_softmax;                /* Softmax operations */
    uint64_t op_layernorm;              /* LayerNorm operations */
    uint64_t op_rmsnorm;                /* RMSNorm operations */
    uint64_t op_pool_max;               /* MaxPool operations (O6) */
    uint64_t op_pool_avg;               /* AvgPool operations (O6) */
    uint64_t op_other;                  /* Catch-all for other ops */

    /* Tile counters */
    uint64_t total_tiles;               /* Total MMA tiles processed */
    uint64_t edge_tiles;                /* Non-full tiles at edges */
    uint64_t full_tiles;                /* Full-sized tiles */
} tu_compute_counters_t;

/* Memory hierarchy counters */
typedef struct {
    /* Per-level access counts */
    uint64_t mem_reqfile_reads;         /* Register file reads */
    uint64_t mem_reqfile_writes;
    uint64_t mem_spad_reads;            /* Local scratchpad reads */
    uint64_t mem_spad_writes;
    uint64_t mem_spad_bank_conflicts;   /* Bank conflict count */
    uint64_t mem_spad_stall_cycles;     /* Stalls due to bank conflicts */
    uint64_t mem_gbuf_reads;            /* Global buffer reads */
    uint64_t mem_gbuf_writes;
    uint64_t mem_gbuf_bank_conflicts;
    uint64_t mem_dram_reads;            /* DRAM reads */
    uint64_t mem_dram_writes;
    uint64_t mem_dram_row_hits;         /* DRAM row buffer hits */
    uint64_t mem_dram_row_misses;       /* DRAM row buffer misses */
    uint64_t mem_dram_stall_cycles;     /* DRAM stall cycles */
    uint64_t mem_dram_bytes_read;       /* Total DRAM read bytes */
    uint64_t mem_dram_bytes_written;    /* Total DRAM write bytes */

    /* Bandwidth utilization */
    float    spad_bw_utilization;       /* Scratchpad bandwidth [0,1] */
    float    gbuf_bw_utilization;       /* Global buffer bandwidth [0,1] */
    float    dram_bw_utilization;       /* DRAM bandwidth [0,1] */
} tu_memory_counters_t;

/* Power/Energy counters (E4) */
typedef struct {
    double   energy_mac_pj;             /* MAC energy (picojoules) */
    double   energy_sram_read_pj;       /* SRAM read energy */
    double   energy_sram_write_pj;      /* SRAM write energy */
    double   energy_dram_pj;            /* DRAM access energy */
    double   energy_dma_pj;             /* DMA transfer energy */
    double   energy_leakage_pj;         /* Static leakage energy */
    double   energy_total_pj;           /* Total energy */

    /* Configurable energy parameters (per-technology-node) */
    double   pj_per_mac;                /* pJ per MAC operation */
    double   pj_per_sram_read;          /* pJ per SRAM read (per word) */
    double   pj_per_sram_write;         /* pJ per SRAM write (per word) */
    double   pj_per_dram_access;        /* pJ per DRAM access */
    double   pj_per_dma_byte;           /* pJ per DMA byte transferred */
    double   pj_leakage_per_cycle;      /* pJ leakage per cycle */

    bool     power_modeling_enabled;    /* Enable/disable power tracking */
} tu_power_counters_t;

/* Global performance counter aggregate */
typedef struct {
    tu_dma_counters_t      dma;
    tu_compute_counters_t  compute;
    tu_memory_counters_t   memory;
    tu_power_counters_t    power;

    /* Global state */
    uint64_t  total_cycles;             /* Global cycle counter */
    uint64_t  wall_clock_ns;            /* Simulated wall-clock time */
    double    clock_freq_mhz;           /* Clock frequency for wall-clock conversion */
    bool      enabled;                  /* Master enable/disable */
} tu_perf_counters_t;

/* ---- Lifecycle ---- */

/* Initialize all counters to zero with default energy params */
void tu_perf_init(tu_perf_counters_t *c, double clock_freq_mhz);

/* Reset all counters (but keep energy parameters) */
void tu_perf_reset(tu_perf_counters_t *c);

/* Enable/disable all counters */
void tu_perf_set_enabled(tu_perf_counters_t *c, bool enabled);

/* ---- Cycle Management ---- */

/* Advance the global cycle counter. All per-access counters
 * should call this to keep the clock consistent. */
void tu_perf_tick(tu_perf_counters_t *c, uint64_t cycles);

/* Get current cycle count */
uint64_t tu_perf_get_cycle(const tu_perf_counters_t *c);

/* ---- DMA Counter API ---- */

void tu_perf_dma_record_read(tu_perf_counters_t *c, uint32_t bytes,
                              uint64_t active_cycles, uint64_t stall_cycles,
                              uint8_t channel, uint8_t transfer_type);
void tu_perf_dma_record_write(tu_perf_counters_t *c, uint32_t bytes,
                               uint64_t active_cycles, uint64_t stall_cycles,
                               uint8_t channel);
void tu_perf_dma_record_internal(tu_perf_counters_t *c, uint32_t bytes,
                                  uint64_t active_cycles, uint8_t channel);

/* ---- Compute Counter API ---- */

void tu_perf_compute_record_mma(tu_perf_counters_t *c,
                                 uint64_t macs, uint32_t m, uint32_t n, uint32_t k,
                                 uint32_t tiles, uint32_t edge_tiles,
                                 uint64_t active_cycles, uint64_t stall_cycles,
                                 uint8_t precision_type, uint8_t dataflow_mode);
void tu_perf_compute_record_op(tu_perf_counters_t *c, uint8_t op_code,
                                uint64_t active_cycles, uint64_t stall_cycles,
                                uint64_t flops);
void tu_perf_compute_record_idle(tu_perf_counters_t *c, uint64_t cycles);
void tu_perf_compute_record_pipeline_bubble(tu_perf_counters_t *c, uint64_t count);

/* ---- Memory Counter API ---- */

void tu_perf_mem_record_spad_access(tu_perf_counters_t *c,
                                     bool is_write, uint32_t words,
                                     uint32_t bank_conflicts, uint64_t stall_cycles);
void tu_perf_mem_record_gbuf_access(tu_perf_counters_t *c,
                                     bool is_write, uint32_t words,
                                     uint32_t bank_conflicts);
void tu_perf_mem_record_dram_access(tu_perf_counters_t *c,
                                     bool is_write, uint32_t bytes,
                                     bool row_hit, uint64_t stall_cycles);
void tu_perf_mem_record_reqfile_access(tu_perf_counters_t *c,
                                        bool is_write, uint32_t words);

/* ---- Power Counter API ---- */

/* Configure energy parameters (per-technology node) */
void tu_perf_power_config(tu_perf_counters_t *c,
                           double pj_mac, double pj_sram_r, double pj_sram_w,
                           double pj_dram, double pj_dma_byte, double pj_leakage);

/* Enable/disable power modeling */
void tu_perf_power_set_enabled(tu_perf_counters_t *c, bool enabled);

/* ---- Snapshot & Merge ---- */

/* Take a snapshot of current counters */
typedef struct { tu_perf_counters_t counters; uint64_t snapshot_cycle; } tu_perf_snapshot_t;
tu_perf_snapshot_t tu_perf_snapshot(const tu_perf_counters_t *c);

/* Compute differential counters between two snapshots */
tu_perf_counters_t tu_perf_diff(const tu_perf_snapshot_t *before,
                                 const tu_perf_snapshot_t *after);

/* Merge src counters into dst (adds all values) */
void tu_perf_merge(tu_perf_counters_t *dst, const tu_perf_counters_t *src);

/* ---- Reporting ---- */

/* Print a formatted performance report to stdout */
void tu_perf_print_report(const tu_perf_counters_t *c);

/* Print a compact one-line summary */
void tu_perf_print_summary(const tu_perf_counters_t *c);

/* Compute derived metrics (utilization, throughput, etc.) */
typedef struct {
    float    compute_utilization;       /* [0,1] */
    float    dma_bandwidth_gbps;        /* Effective DMA bandwidth */
    float    dram_bandwidth_gbps;       /* Effective DRAM bandwidth */
    float    mac_throughput_tops;       /* TOPS (FP16-equivalent) */
    float    mac_efficiency;            /* Effective MACs / Peak MACs */
    float    spad_hit_rate;             /* Scratchpad hit rate */
    float    energy_per_mac_pj;         /* Average energy per MAC */
    float    power_mw;                  /* Average power draw */
} tu_perf_metrics_t;

tu_perf_metrics_t tu_perf_compute_metrics(const tu_perf_counters_t *c);

/* ---- Integration: Wire into existing tu_state_t ---- */

/* Update the global performance counters from a DMA descriptor execution.
 * Called by tu_dma_execute_desc() and friends. */
void tu_perf_from_dma_descriptor(tu_perf_counters_t *c,
                                  uint32_t bytes, uint8_t channel,
                                  uint8_t transfer_type, bool is_read,
                                  uint64_t active_cycles, uint64_t stall_cycles,
                                  uint64_t sram_stall_cycles);

#ifdef __cplusplus
}
#endif

#endif /* TU_PERFORMANCE_COUNTERS_H */
