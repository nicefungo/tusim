/*
 * TU Pluggable Dataflow Interface
 * ================================
 * A4: Dataflow Flexibility — production-grade pluggable dataflow architecture.
 *
 * Every dataflow plugin implements this interface. The compute engine dispatches
 * MMA operations through the selected dataflow plugin. Each plugin encapsulates:
 *   - Tile-level execution strategy (how tiles map to the PE array)
 *   - Microarchitectural loop order (which dimension streams, which is stationary)
 *   - Cycle accounting (pipeline fill, compute cycles, drain cycles)
 *
 * Supported dataflows:
 *   WS — Weight-Stationary (systolic): W preloaded in PEs, A streams right, psum flows down
 *   OS — Output-Stationary (vector): O stays in PEs, W and A stream in
 *   RS — Row-Stationary: each PE stores 1 row of W/A/psum; maximizes 1D reuse
 *   NLR — No Local Reuse: feed-forward, minimal PE storage, all data streams through
 *
 * Each dataflow is compiled as a separate object file and registered at link time.
 * The registry maps dataflow enum values to plugin instances.
 */

#ifndef TU_DATAFLOW_INTERFACE_H
#define TU_DATAFLOW_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Dataflow Enumerations ---- */

typedef enum {
    TU_DATAFLOW_WEIGHT_STATIONARY = 0,
    TU_DATAFLOW_OUTPUT_STATIONARY = 1,
    TU_DATAFLOW_ROW_STATIONARY = 2,
    TU_DATAFLOW_NO_LOCAL_REUSE = 3,
    TU_DATAFLOW_COUNT
} tu_dataflow_id_t;

/* ---- Tensor Descriptor (forward-declared for dataflow use) ---- */

/* Each tensor is a 2D matrix row-major in SRAM */
typedef struct {
    const void  *data;        /* Pointer to raw bytes in SRAM */
    uint32_t     rows;        /* First dimension (e.g., M for W[M][K], K for A[K][N]) */
    uint32_t     cols;        /* Second dimension (e.g., K for W, N for A) */
    uint32_t     stride;      /* Row stride in bytes (cols * elem_size for dense) */
    uint32_t     elem_size;   /* Bytes per element (2 for FP16, 4 for FP32) */
} tu_dataflow_tensor_t;

/* ---- MMA Operation Descriptor ---- */

typedef struct {
    tu_dataflow_tensor_t  W;       /* Weight matrix: W[M][K], FP16, row-major */
    tu_dataflow_tensor_t  A;       /* Activation matrix: A[K][N], FP16, row-major */
    tu_dataflow_tensor_t  O;       /* Output/accumulator: O[M][N], FP32, row-major */
    bool                  has_bias; /* If true, O already contains FP32 bias values */

    /* Tile dimensions (derived from PE array size at runtime) */
    uint16_t  tile_m;               /* PE rows = tile height */
    uint16_t  tile_n;               /* PE cols = tile width */
    uint16_t  tile_k;               /* Inner tile depth (typically same as tile_n) */

    /* Pipeline depth (cycles to fill systolic pipeline) */
    uint16_t  pipeline_depth;
} tu_mma_op_t;

/* ---- Dataflow Plugin Interface ---- */

typedef struct tu_dataflow_plugin_t tu_dataflow_plugin_t;

struct tu_dataflow_plugin_t {
    /* Human-readable name (e.g., "weight_stationary", "output_stationary") */
    const char *name;

    /* Which dataflow this implements */
    tu_dataflow_id_t id;

    /* ---- Lifecycle ---- */

    /* Initialize plugin state. Called once when dataflow is selected. */
    void (*init)(tu_dataflow_plugin_t *plugin);

    /* ---- Core MMA Execution ---- */

    /*
     * Execute a tiled MMA operation using this dataflow.
     *
     * The caller has already decomposed the full matrix into tiles.
     * This function processes a single tile:
     *   O_tile[m_tile][n_tile] += W_tile[m_tile][k_tile] × A_tile[k_tile][n_tile]
     *
     * Parameters:
     *   plugin:  this dataflow instance
     *   op:      MMA operation descriptor with full tensor pointers
     *   m_start: row offset of this tile in the M dimension
     *   m_count: number of M rows in this tile (≤ tile_m)
     *   n_start: column offset of this tile in the N dimension
     *   n_count: number of N columns in this tile (≤ tile_n)
     *   k_start: inner offset of this tile in the K dimension
     *   k_count: inner elements in this tile (≤ tile_k)
     *
     * Returns: estimated cycle count for this tile execution.
     */
    uint64_t (*execute_tile)(tu_dataflow_plugin_t *plugin,
                             const tu_mma_op_t *op,
                             uint16_t m_start, uint16_t m_count,
                             uint16_t n_start, uint16_t n_count,
                             uint16_t k_start, uint16_t k_count);

    /* ---- Cycle Accounting ---- */

    /*
     * Get pipeline fill cycles (overhead before first MAC result emerges).
     * WS: pipeline_depth * tile_n (fill columns sequentially)
     * OS: 0 (vector engine has no systolic fill)
     */
    uint64_t (*get_fill_cycles)(const tu_dataflow_plugin_t *plugin,
                                uint16_t tile_n, uint16_t tile_k);

    /*
     * Get pipeline drain cycles (overhead after last MAC until result is complete).
     * WS: pipeline_depth * tile_m (drain rows)
     * OS: 0 or small constant
     */
    uint64_t (*get_drain_cycles)(const tu_dataflow_plugin_t *plugin,
                                 uint16_t tile_m);

    /*
     * Get compute cycles for a tile.
     * WS: k_count (1 MAC per cycle per PE after fill)
     * OS: k_count (same basic model, different microarchitecture)
     */
    uint64_t (*get_compute_cycles)(const tu_dataflow_plugin_t *plugin,
                                   uint16_t m_count, uint16_t n_count,
                                   uint16_t k_count);

    /* ---- Statistics ---- */

    /* Total FLOPS executed by this dataflow instance (MACs × 2) */
    uint64_t total_flops;

    /* Total tiles executed */
    uint64_t total_tiles;

    /* Total cycles consumed */
    uint64_t total_cycles;

    /* Private implementation data (opaque pointer) */
    void *impl_data;
};

/* ---- Type-erased execute call (for dispatch table) ---- */

/*
 * High-level MMA execution through the selected dataflow.
 *
 * This function handles full tiling: it decomposes the M×N×K operation into
 * tile_m × tile_n × tile_k tiles and calls execute_tile() for each.
 *
 * Parameters:
 *   plugin:   the selected dataflow plugin
 *   W:        weight matrix descriptor [M][K]
 *   A:        activation matrix descriptor [K][N]
 *   O:        output matrix descriptor [M][N] (FP32 accumulators)
 *   tile_m:   PE rows = M-dimension tile size
 *   tile_n:   PE cols = N-dimension tile size
 *   tile_k:   K-dimension tile size
 *   pipeline_depth: MAC pipeline stages
 */
uint64_t tu_dataflow_execute_mma(tu_dataflow_plugin_t *plugin,
                                  const tu_dataflow_tensor_t *W,
                                  const tu_dataflow_tensor_t *A,
                                  tu_dataflow_tensor_t *O,
                                  uint16_t tile_m, uint16_t tile_n,
                                  uint16_t tile_k, uint16_t pipeline_depth);

/*
 * Convenience: convert FP16 element to FP32 for MAC.
 * (Defined here so dataflow implementations don't need to include tu_precision.h
 *  directly — though they may include it for bulk conversion.)
 */
float tu_dataflow_fp16_to_fp32(uint16_t h);
uint16_t tu_dataflow_fp32_to_fp16(float f);

#ifdef __cplusplus
}
#endif

#endif /* TU_DATAFLOW_INTERFACE_H */
