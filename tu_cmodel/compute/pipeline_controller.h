/*
 * TU CModel — Software Pipelining Controller (Gap E2)
 * ===================================================
 * Tile-level software pipelining: DMA tile N+1 while computing tile N.
 *
 * This is the key performance optimization in production systolic
 * accelerators (TPU, Gemmini, Eyeriss). With double buffering (A7)
 * and async DMA (DM1/DM2) already in place, the pipeline controller
 * orchestrates overlap between DMA and compute.
 *
 * Architecture:
 *   The controller manages a configurable-depth pipeline of tiles.
 *   Each tile goes through stages:
 *     STAGE_DMA_PRELOAD  — DMA writes tile N+1 data into shadow buffer
 *     STAGE_COMPUTE      — PE array computes tile N using active buffer
 *     STAGE_DMA_STORE    — DMA stores tile N results to DRAM
 *     STAGE_DONE         — Tile complete, buffers can be recycled
 *
 *   Overlap model:
 *     Without pipelining:  total = Σ (DMA_load + compute + DMA_store)
 *     With pipelining:     total ≈ max(DMA_load, compute, DMA_store) per tile
 *     The pipeline depth controls how many tiles can overlap.
 *
 *   Pipeline depth = 1:  Sequential (DMA→compute→store for each tile)
 *   Pipeline depth = 2:  DMA tile N+1 during compute tile N
 *   Pipeline depth = 3:  DMA N+2 during compute N+1 during store N
 *
 * Integration:
 *   - pipeline_controller coordinates: double_buffer, dma_engine, command_queue
 *   - tu_pipeline_submit_tile() stages a tile through the DMA queue
 *   - tu_pipeline_advance() checks for completed stages and advances the pipeline
 *   - tu_pipeline_get_overlap_stats() reports cycle savings
 *
 * Gap: E2 — Tile-level software pipelining (P1)
 * Dependencies: A7 (double buffering), DM1/DM2 (async DMA), E1 (command queue)
 */

#ifndef TU_PIPELINE_CONTROLLER_H
#define TU_PIPELINE_CONTROLLER_H

#include "../tu_config.h"
#include "../tu_sram.h"
#include "../dma_descriptor.h"
#include "../command_queue.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Pipeline stage enumeration ---- */
typedef enum {
    TU_PIPE_STAGE_IDLE           = 0,  /* Slot unused */
    TU_PIPE_STAGE_DMA_PRELOAD    = 1,  /* DMA loading tile data into shadow buffer */
    TU_PIPE_STAGE_COMPUTE        = 2,  /* PE array computing on active buffer */
    TU_PIPE_STAGE_DMA_STORE      = 3,  /* DMA storing results to DRAM */
    TU_PIPE_STAGE_DONE           = 4,  /* Tile complete */
} tu_pipeline_stage_t;

/* ---- Pipeline configuration ---- */
typedef struct {
    uint32_t    max_depth;          /* Max pipeline depth (1–8) */
    bool        enable_load_overlap;  /* Overlap DMA load with compute */
    bool        enable_store_overlap; /* Overlap DMA store with compute */
    bool        enable_triple_overlap;/* Load tile N+2 during compute N+1, store N */
    uint32_t    tile_timeout_cycles;  /* Max cycles before tile stall is flagged */
    bool        model_stalls;         /* Account for pipeline stalls in cycle model */
} tu_pipeline_config_t;

/* ---- Pipeline tile descriptor ---- */
typedef struct {
    uint32_t            tile_id;        /* Monotonically increasing tile ID */
    tu_pipeline_stage_t stage;          /* Current stage */
    uint64_t            cycle_entered;  /* When tile entered current stage */
    uint64_t            cycle_expected; /* Estimated completion cycle */

    /* DMA descriptors for this tile's preload */
    tu_dma_descriptor_t *load_desc;     /* DMA load descriptor (weight/activation) */
    uint32_t            load_signal_id; /* Completion signal for load */

    /* DMA descriptors for this tile's store */
    tu_dma_descriptor_t *store_desc;    /* DMA store descriptor (output) */
    uint32_t            store_signal_id;/* Completion signal for store */

    /* Compute info */
    uint64_t            compute_cycles; /* Estimated compute cycles for this tile */
    uint32_t            cmd_id;         /* Command queue ID for compute op */

    /* Buffer tracking (for double-buffered regions) */
    tu_sram_region_t   *buffer_region;  /* Which SRAM region is double-buffered */
    bool                swapped;        /* Has the double-buffer been swapped? */

    /* Statistics */
    uint64_t            cycles_saved;   /* Compute cycles overlapped with DMA */
    bool                stalled;        /* Tile was stalled waiting for DMA */
} tu_pipeline_tile_t;

/* ---- Pipeline controller ---- */
typedef struct {
    tu_pipeline_tile_t  *slots;         /* Array of pipeline tile slots */
    uint32_t            depth;          /* Configured pipeline depth */
    uint32_t            active_count;   /* Number of active tiles in pipeline */

    tu_pipeline_config_t config;        /* Configuration */

    uint32_t            next_tile_id;   /* Monotonically increasing tile counter */

    /* Overlap accounting */
    uint64_t            total_compute_cycles;    /* Total compute cycles across all tiles */
    uint64_t            total_load_cycles;       /* Total DMA load cycles */
    uint64_t            total_store_cycles;      /* Total DMA store cycles */
    uint64_t            overlapped_load_cycles;  /* Load cycles overlapped with compute */
    uint64_t            overlapped_store_cycles; /* Store cycles overlapped with compute */
    uint64_t            stall_cycles;            /* Cycles lost to pipeline stalls */

    /* Sequential baseline (for comparison) */
    uint64_t            sequential_total;        /* Σ(load + compute + store) for all tiles */

    /* Global cycle counter (shared with the core) */
    uint64_t            current_cycle;

    /* Status */
    bool                initialized;
    uint32_t            total_tiles_processed;
    uint32_t            total_stalls;

} tu_pipeline_controller_t;

/* ---- External reference (global singleton, matches existing architecture) ---- */
extern tu_pipeline_controller_t g_tu_pipeline;

/* ---- API ---- */

/*
 * Initialize the pipeline controller.
 *
 *   depth: pipeline depth (1–8)
 *     depth=1: sequential (no overlap), useful for verification baseline
 *     depth=2: DMA tile N+1 loads during tile N compute
 *     depth=3: DMA N+2 loads during tile N+1 compute during tile N store
 *
 * The `config` struct provides fine-grained control over which overlaps
 * are enabled. Call with tu_pipeline_config_default() for sensible defaults.
 */
void tu_pipeline_init(uint32_t depth, const tu_pipeline_config_t *config);

/*
 * Get sensible default configuration.
 */
tu_pipeline_config_t tu_pipeline_config_default(void);

/*
 * Destroy pipeline controller and free resources.
 */
void tu_pipeline_destroy(void);

/*
 * Reset pipeline state (between workloads).
 */
void tu_pipeline_reset(void);

/*
 * Submit a new tile into the pipeline.
 *
 * The controller will schedule DMA preloads and compute based on
 * current pipeline occupancy. If the pipeline is full, this is
 * a blocking call (models backpressure).
 *
 *   load_desc:  DMA descriptor for loading this tile's data
 *               (set to NULL for the first tile where data is already in place)
 *   store_desc: DMA descriptor for storing results
 *   compute_cycles: estimated compute time for this tile
 *   cmd_id: command queue ID for the compute operation
 *   buffer_region: which SRAM region is double-buffered
 *                  (NULL = no double buffering, pipelining disabled for this tile)
 *
 * Returns: assigned tile_id, or -1 on error (pipeline full).
 */
int tu_pipeline_submit_tile(tu_dma_descriptor_t *load_desc,
                             tu_dma_descriptor_t *store_desc,
                             uint64_t compute_cycles,
                             uint32_t cmd_id,
                             tu_sram_region_t *buffer_region);

/*
 * Advance the pipeline.
 *
 * Call this once per "cycle tick" to:
 *   1. Check for completed DMA preloads → advance tile to COMPUTE
 *   2. Check for completed compute → advance tile to DMA_STORE
 *   3. Check for completed stores → advance tile to DONE
 *   4. Update overlap statistics
 *
 * Returns: number of tiles that advanced this tick.
 */
int tu_pipeline_advance(void);

/*
 * Check if the pipeline has available capacity.
 * Returns: number of free slots.
 */
int tu_pipeline_free_slots(void);

/*
 * Wait for all tiles in the pipeline to complete.
 * In functional mode, this is a no-op (all tiles complete immediately).
 */
void tu_pipeline_sync(void);

/*
 * Check if the pipeline is idle (all tiles done).
 */
bool tu_pipeline_is_idle(void);

/* ---- Statistics ---- */

typedef struct {
    uint32_t    depth;
    uint32_t    total_tiles;
    uint64_t    total_compute_cycles;
    uint64_t    total_load_cycles;
    uint64_t    total_store_cycles;
    uint64_t    overlapped_load_cycles;
    uint64_t    overlapped_store_cycles;
    uint64_t    stall_cycles;
    uint64_t    sequential_total;
    uint64_t    pipelined_total;
    double      speedup;           /* sequential / pipelined */
    double      load_overlap_pct;  /* % of load cycles overlapped */
    double      store_overlap_pct; /* % of store cycles overlapped */
    uint32_t    total_stalls;
} tu_pipeline_stats_t;

/*
 * Get comprehensive pipeline statistics.
 */
void tu_pipeline_get_stats(tu_pipeline_stats_t *stats);

/*
 * Print pipeline statistics to stderr.
 */
void tu_pipeline_print_stats(void);

/*
 * Get the cycle savings from pipelining since last query.
 * Returns: overlapped cycles (these would have been sequential without pipelining).
 */
uint64_t tu_pipeline_get_saved_cycles(void);

#ifdef __cplusplus
}
#endif

#endif /* TU_PIPELINE_CONTROLLER_H */
