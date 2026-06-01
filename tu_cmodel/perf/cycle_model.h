/*
 * TU CModel — Cycle-Accurate Timing Model (Gap P2.5)
 * =====================================================
 *
 * Production-grade cycle-level simulator that models execution with
 * realistic pipeline hazards, memory contention, and DRAM behavior.
 *
 * Abstraction levels (selectable):
 *   FUNCTIONAL (0):    Pure functional — no cycle accounting
 *   ESTIMATED (1):     Simplified model — pipeline fill/drain + DRAM latency
 *   CYCLE_ACCURATE (2): Full model — pipeline hazards, bank conflicts,
 *                       DRAM row buffer, bus contention
 *
 * Architecture:
 *   Pipeline stages per MAC: Fetch → Decode → Read Reg → MAC → Accumulate
 *   Hazard detection: RAW (read-after-write) on register file
 *   SRAM bank model: per-bank refill budget, conflict detection, stall accounting
 *   DRAM model: row buffer hit/miss, open-page policy, per-bank state
 *   DMA bus: shared bus arbitration between channels
 *
 * Dependencies: tu_config.h, perf/performance_counters.h
 */

#ifndef TU_CYCLE_MODEL_H
#define TU_CYCLE_MODEL_H

#include "../tu_config.h"
#include "performance_counters.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Pipeline Stage Model
 * ================================================================ */

/*
 * Pipeline stages for a MAC operation in the systolic array.
 *
 * Stage 0: IF  — Instruction Fetch (from command queue)
 * Stage 1: ID  — Instruction Decode (determine op, operands)
 * Stage 2: RR  — Register Read (read operands from RegFile or SPAD)
 * Stage 3: MAC — Multiply-Accumulate (FP/INT arithmetic)
 * Stage 4: WB  — Writeback (write partial sum to accumulator or SPAD)
 *
 * Pipeline depth = TU_PE_PIPELINE_DEPTH (configured in tu_config.h)
 */
typedef enum {
    TU_CYCLE_STAGE_IF  = 0,
    TU_CYCLE_STAGE_ID  = 1,
    TU_CYCLE_STAGE_RR  = 2,
    TU_CYCLE_STAGE_MAC = 3,
    TU_CYCLE_STAGE_WB  = 4,
    TU_CYCLE_NUM_STAGES = 5
} tu_cycle_pipeline_stage_t;

/*
 * Hazard types modeled:
 *   RAW (Read-After-Write): later instruction reads a register before
 *        earlier instruction writes it → stall until write completes
 *   WAW (Write-After-Write): two instructions write same register →
 *        stall to preserve program order (output-stationary)
 *   WAR (Write-After-Read): handled by register renaming in real HW,
 *        modeled as no-op here (assumed renamed)
 *   STRUCT: resource conflict (same SRAM bank, same DMA channel)
 */
typedef enum {
    TU_HAZARD_NONE   = 0,
    TU_HAZARD_RAW    = 1,
    TU_HAZARD_WAW    = 2,
    TU_HAZARD_STRUCT = 3,
} tu_hazard_type_t;

/*
 * Pipeline state for tracking in-flight operations.
 * Each entry tracks one tile in the systolic pipeline.
 */
typedef struct {
    uint64_t    issue_cycle;       /* Cycle when this tile was issued */
    uint64_t    complete_cycle;    /* Cycle when this tile completed */
    uint16_t    stage;             /* Current pipeline stage (0-4) */
    uint16_t    m_start, n_start, k_start;  /* Tile coordinates */
    uint16_t    m_count, n_count, k_count;  /* Tile dimensions */
    bool        active;            /* Is this entry in use */
    uint32_t    reg_deps[4];       /* Register file addresses this tile reads */
    uint32_t    reg_outputs[4];    /* Register file addresses this tile writes */
} tu_pipeline_entry_t;

/*
 * Pipeline tracker — records in-flight tiles and detects hazards.
 */
typedef struct {
    tu_pipeline_entry_t *entries;   /* Circular buffer of pipeline entries */
    uint32_t    num_entries;         /* Max in-flight entries (pipeline depth) */
    uint32_t    head;                /* Next entry to allocate */
    uint32_t    tail;                /* Next entry to complete */
    uint64_t    total_stall_cycles;  /* Accumulated hazard stalls */
    uint64_t    total_bubble_cycles; /* Pipeline empty slot cycles */
    uint64_t    total_issues;        /* Total tiles issued */
    uint64_t    total_completions;   /* Total tiles completed */
} tu_pipeline_tracker_t;

/*
 * Initialize the pipeline tracker.
 */
void tu_cycle_pipeline_init(tu_pipeline_tracker_t *pt, uint32_t max_in_flight);

/*
 * Issue a new tile into the pipeline. Checks for hazards and returns
 * stall cycles needed before issue is safe.
 *
 * Returns: 0 if no stall, >0 for cycles to stall before issue.
 */
uint64_t tu_cycle_pipeline_issue(tu_pipeline_tracker_t *pt,
                            uint16_t m_start, uint16_t m_count,
                            uint16_t n_start, uint16_t n_count,
                            uint16_t k_start, uint16_t k_count,
                            const uint32_t *src_regs, uint32_t num_src,
                            const uint32_t *dst_regs, uint32_t num_dst,
                            uint64_t current_cycle);

/*
 * Complete the oldest in-flight tile. Advances tail.
 * Returns: cycles this tile spent in pipeline, or 0 if none active.
 */
uint64_t tu_cycle_pipeline_complete(tu_pipeline_tracker_t *pt, uint64_t current_cycle);

/*
 * Get pipeline utilization: fraction of entries that are active.
 */
float tu_cycle_pipeline_utilization(const tu_pipeline_tracker_t *pt);

/*
 * Destroy pipeline tracker and free resources.
 */
void tu_cycle_pipeline_destroy(tu_pipeline_tracker_t *pt);

/* ================================================================
 * Bank Conflict Model
 * ================================================================ */

/*
 * SRAM bank state for conflict modeling.
 * Each bank has a refill-based bandwidth budget and tracks
 * concurrent accesses for conflict detection.
 */
typedef struct {
    uint32_t    bank_id;
    uint32_t    words_available;    /* Words available in current refill window */
    uint32_t    max_words_per_cycle; /* Bandwidth: max words per cycle per bank */
    uint32_t    refill_cycle;       /* Next cycle when budget refills */
    uint64_t    total_reads;
    uint64_t    total_writes;
    uint64_t    read_stalls;        /* Stalls due to exhausted budget */
    uint64_t    write_stalls;
    uint64_t    conflict_count;     /* Access conflicts (multiple requests) */
    double      utilization;        /* Fraction of capacity used */
} tu_bank_state_t;

/*
 * Bank conflict tracking for a multi-bank SRAM.
 */
typedef struct {
    tu_bank_state_t *banks;
    uint32_t    num_banks;
    uint32_t    bank_width_bytes;
    uint32_t    refill_window_cycles; /* Cycles between bandwidth budget refills */
    uint32_t    stall_penalty;         /* Cycles to stall when bandwidth exhausted */
    uint32_t    max_accesses_per_cycle; /* Max concurrent accesses per cycle */
    uint64_t    current_cycle;
} tu_bank_model_t;

/*
 * Initialize bank conflict model.
 */
void tu_bank_model_init(tu_bank_model_t *bm, uint32_t num_banks,
                         uint32_t bank_width_bytes, uint32_t refill_window,
                         uint32_t stall_penalty, uint32_t max_accesses);

/*
 * Attempt a read or write access to a specific bank.
 * Returns: 0 if access succeeds, >0 for cycles to stall.
 *
 * The caller should accumulate stall cycles and retry.
 */
uint32_t tu_bank_model_access(tu_bank_model_t *bm, uint32_t bank_id,
                               bool is_write, uint32_t word_count,
                               uint64_t cycle);

/*
 * Advance the global cycle counter and refill bandwidth budgets.
 */
void tu_bank_model_tick(tu_bank_model_t *bm, uint64_t cycle);

/*
 * Get bank utilization statistics.
 */
void tu_bank_model_get_stats(const tu_bank_model_t *bm,
                              uint64_t *total_reads, uint64_t *total_writes,
                              uint64_t *total_stalls, uint64_t *total_conflicts,
                              double *avg_utilization);

/*
 * Destroy bank model.
 */
void tu_bank_model_destroy(tu_bank_model_t *bm);

/* ================================================================
 * DRAM Row Buffer Model
 * ================================================================ */

/*
 * DRAM bank state with row buffer tracking.
 * Models open-page policy: a row stays open after access.
 *
 *   Row hit:    requested row = open row → low latency
 *   Row miss:   different row requested → precharge + activate + access
 *   Row empty:  no row open → activate + access
 */
typedef struct {
    uint32_t    open_row;            /* Currently open row (UINT32_MAX = none) */
    uint32_t    num_rows;            /* Total rows in this bank */
    uint64_t    access_count;
    uint64_t    row_hits;
    uint64_t    row_misses;
    uint64_t    row_conflicts;       /* Different bank, same rank conflict */
    uint64_t    total_cycles_active; /* Cycles with row open */
    uint64_t    total_cycles_idle;   /* Cycles in precharge state */
} tu_dram_bank_state_t;

/*
 * DRAM timing parameters (JEDEC-style).
 * All times in cycles at the configured clock frequency.
 */
typedef struct {
    uint32_t    tRCD;    /* RAS-to-CAS delay (activate to read/write) */
    uint32_t    tRP;     /* Row precharge time */
    uint32_t    tRAS;    /* Row active to precharge (minimum) */
    uint32_t    tRC;     /* Row cycle time (tRAS + tRP) */
    uint32_t    tCL;     /* CAS latency (read command to data) */
    uint32_t    tCWL;    /* CAS write latency */
    uint32_t    tWR;     /* Write recovery time */
    uint32_t    tRTP;    /* Read to precharge */
    uint32_t    tFAW;    /* Four-activate window */
    uint32_t    tRRD;    /* Row-to-row delay (same bank group) */
    uint32_t    tCCD;    /* Column-to-column delay */
    uint32_t    tBL;     /* Burst length (typically 8 for DDR, 4 for HBM) */
    uint32_t    bus_width_bytes;  /* Bytes per transfer */
    uint32_t    num_banks;        /* Number of banks */
    uint32_t    num_bank_groups;  /* Bank groups (DDR4+) */
    uint32_t    rows_per_bank;    /* Rows per bank (e.g., 65536 for 16Gb DDR4) */
    uint32_t    columns_per_row;  /* Columns per row (page size / bus width) */
    double      freq_mhz;         /* DRAM frequency in MHz */
} tu_dram_timing_t;

/*
 * DRAM channel model with per-bank row buffer tracking.
 */
typedef struct {
    tu_dram_timing_t    timing;
    tu_dram_bank_state_t *banks;
    uint64_t            current_cycle;
    uint64_t            total_accesses;
    uint64_t            total_row_hits;
    uint64_t            total_row_misses;
    uint64_t            total_conflicts;
    uint64_t            total_cycles_stalled;
    uint64_t            total_bytes_transferred;
    double              effective_bandwidth_gbps;  /* Computed dynamically */
} tu_dram_channel_t;

/*
 * Initialize DRAM channel with standard timing parameters.
 *
 * dram_type: TU_DRAM_HBM2, TU_DRAM_DDR4, etc. (from tu_config.h)
 */
void tu_dram_channel_init(tu_dram_channel_t *ch, uint32_t dram_type,
                           double freq_mhz, uint32_t bus_width_bytes);

/*
 * Get timing parameters for a standard DRAM type.
 */
void tu_dram_timing_preset(tu_dram_timing_t *t, uint32_t dram_type,
                            double freq_mhz, uint32_t bus_width_bytes);

/*
 * Access DRAM at a given byte address.
 * Returns: access latency in cycles (includes row hit/miss overhead).
 *
 * address: byte address in DRAM space
 * is_write: true for write, false for read
 * bytes: number of bytes to transfer
 */
uint64_t tu_dram_access(tu_dram_channel_t *ch, uint64_t address,
                         bool is_write, uint32_t bytes, uint64_t cycle);

/*
 * Compute the bank and row from a byte address.
 */
void tu_dram_decode_address(const tu_dram_channel_t *ch, uint64_t address,
                             uint32_t *bank, uint32_t *row, uint32_t *column);

/*
 * Advance DRAM cycle counter and update state.
 */
void tu_cycle_dram_tick(tu_dram_channel_t *ch, uint64_t cycle);

/*
 * Get DRAM performance statistics.
 */
void tu_cycle_dram_get_stats(const tu_dram_channel_t *ch,
                        uint64_t *accesses, uint64_t *row_hits,
                        uint64_t *row_misses, double *hit_rate,
                        double *effective_bw_gbps, uint64_t *stall_cycles);

/*
 * Destroy DRAM channel.
 */
void tu_cycle_dram_destroy(tu_dram_channel_t *ch);

/* ================================================================
 * Cycle Model Integration
 * ================================================================ */

/*
 * Top-level cycle model state.
 * Composes pipeline tracker, bank model, DRAM channel, and bus model.
 */
typedef struct {
    uint32_t    mode;             /* FUNCTIONAL / ESTIMATED / CYCLE_ACCURATE */
    uint64_t    current_cycle;    /* Global clock */

    /* Sub-models */
    tu_pipeline_tracker_t  *pipeline;
    tu_bank_model_t        *bank_model;
    tu_dram_channel_t      *dram_channel;

    /* Bus contention between DMA channels */
    uint32_t    dma_channels;
    uint64_t    dma_bus_cycles[8];      /* Bus cycles consumed per channel */
    uint64_t    dma_bus_stall_cycles;   /* Total DMA bus contention stalls */

    /* Integration with performance counters */
    tu_perf_counters_t  *perf;
} tu_cycle_model_t;

/*
 * Create a cycle model with sub-models driven by config.
 *
 * mode: TU_CYCLE_MODEL_FUNCTIONAL, _ESTIMATED, or _CYCLE_ACCURATE
 */
tu_cycle_model_t *tu_cycle_model_create(uint32_t mode, tu_perf_counters_t *perf);

/*
 * Reset the cycle model (zero all counters, clear pipeline).
 */
void tu_cycle_model_reset(tu_cycle_model_t *cm);

/*
 * Simulate a systolic array tile execution with full cycle accounting.
 *
 * For each tile:
 *   1. Decode instruction
 *   2. Issue into pipeline (detect hazards)
 *   3. Schedule SRAM reads (detect bank conflicts)
 *   4. Execute MAC operations
 *   5. Writeback results
 *   6. Complete pipeline entry
 *
 * Returns: total cycles consumed by this tile (including stalls).
 */
uint64_t tu_cycle_model_execute_tile(
    tu_cycle_model_t *cm,
    uint16_t m_start, uint16_t m_count,
    uint16_t n_start, uint16_t n_count,
    uint16_t k_start, uint16_t k_count,
    uint32_t w_sram_addr, uint32_t a_sram_addr, uint32_t o_sram_addr);

/*
 * Simulate DMA transfer with bus contention and DRAM modeling.
 *
 * Returns: total cycles including bus arbitration and DRAM latency.
 */
uint64_t tu_cycle_model_dma_transfer(
    tu_cycle_model_t *cm,
    uint8_t channel, uint32_t bytes,
    bool is_read,      /* true = DRAM→SRAM, false = SRAM→DRAM */
    uint64_t dram_addr, uint32_t sram_bank);

/*
 * Simulate bus arbitration: N channels competing for shared DMA bus.
 * Returns additional stall cycles for this channel.
 */
uint64_t tu_cycle_model_dma_arbitrate(
    tu_cycle_model_t *cm, uint8_t channel, uint64_t transfer_cycles);

/*
 * Advance the global cycle counter.
 */
void tu_cycle_model_advance(tu_cycle_model_t *cm, uint64_t cycles);

/*
 * Get a summary report of cycle model statistics.
 */
void tu_cycle_model_report(const tu_cycle_model_t *cm);

/*
 * Destroy cycle model.
 */
void tu_cycle_model_destroy(tu_cycle_model_t *cm);

/* ================================================================
 * Convenience: Standard DRAM Presets
 * ================================================================ */

/*
 * Initialize timing parameters for common DRAM types.
 *
 * HBM2:  2 Gbps/pin, 1024-bit bus, tRC ~45ns, 8 banks
 * HBM2E: 3.6 Gbps/pin, 1024-bit bus
 * HBM3:  6.4 Gbps/pin, 1024-bit bus, 16 banks
 * DDR4:  3.2 Gbps/pin, 64-bit bus, 16 banks, bg-mode
 * DDR5:  6.4 Gbps/pin, 64-bit bus, 32 banks
 * LPDDR5: 6.4 Gbps/pin, 64-bit bus
 */
void tu_dram_preset_hbm2(tu_dram_timing_t *t, uint32_t bus_width_bytes);
void tu_dram_preset_hbm2e(tu_dram_timing_t *t, uint32_t bus_width_bytes);
void tu_dram_preset_hbm3(tu_dram_timing_t *t, uint32_t bus_width_bytes);
void tu_dram_preset_ddr4(tu_dram_timing_t *t, uint32_t bus_width_bytes);
void tu_dram_preset_ddr5(tu_dram_timing_t *t, uint32_t bus_width_bytes);
void tu_dram_preset_lpddr5(tu_dram_timing_t *t, uint32_t bus_width_bytes);
void tu_dram_preset_ideal(tu_dram_timing_t *t, uint32_t bus_width_bytes);

#ifdef __cplusplus
}
#endif

#endif /* TU_CYCLE_MODEL_H */
