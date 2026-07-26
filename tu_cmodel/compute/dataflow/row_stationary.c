/*
 * Row-Stationary (Eyeriss) Dataflow
 * ===================================
 * A4: Production-grade RS dataflow plugin, completing the WS/OS/RS trifecta.
 *
 * Dataflow semantics (Eyeriss v1, Chen et al., ISCA 2016):
 *   - Each PE stores one ROW of filter weights, one row of activations,
 *     and one row of partial sums (stationary across 3 dimensions simultaneously)
 *   - Maximizes data reuse across all dimensions: convolutional reuse within PE,
 *     spatial reuse across PE array
 *   - Pattern: For GEMM O[M][N] = W[M][K] × A[K][N]:
 *       PE(i, j) computes O[i][j] as the dot product W[i][:] · A[:][j]
 *       W row i preloaded into PE row i
 *       A column j streams through PE column j
 *       O element stays stationary during K-dimension accumulation
 *
 * Key characteristics (vs WS and OS):
 *   - RS: Each PE holds 1 row of W and 1 element of O (stationary)
 *         W rows are distributed across PE rows; A streams column-wise
 *   - WS: W[M][K] preloaded across all PEs (W stationary)
 *         A streams, psum flows down
 *   - OS: O stays, W and A both stream (highest bandwidth demand)
 *
 * RS is particularly efficient for:
 *   - Convolutional layers (1D filter reuse within PE, 2D map across array)
 *   - Deep networks where weight memory dominates
 *   - Workloads benefiting from maximal data reuse at all levels
 *
 * Cycle model:
 *   - Pipeline fill: pipeline_depth * tile_n (fill columns, similar to WS)
 *   - Compute: k_count cycles (1 MAC per cycle per PE after fill)
 *   - Pipeline drain: pipeline_depth * tile_m (drain rows)
 *
 * Reference: Chen et al., "Eyeriss: A Spatial Architecture for Energy-Efficient
 *   Dataflow for Convolutional Neural Networks," ISCA 2016
 * Reference: Chen et al., "Eyeriss v2: A Flexible Accelerator for Emerging Deep
 *   Neural Networks," JSSC 2019
 */

#include "dataflow_interface.h"
#include "tu_precision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Row-Stationary Plugin ---- */

typedef struct {
    uint16_t tile_m;
    uint16_t tile_n;
    uint16_t tile_k;
    uint16_t pipeline_depth;

    /*
     * RS-specific: track per-PE-local weight rows for reuse analysis.
     * In functional mode, we don't need per-PE storage; the row-stationary
     * property is encoded in the loop ordering. In cycle-accurate mode,
     * this would track which PE holds which row of W.
     */
    uint64_t w_reuse_hits;   /* Number of times a weight was reused from PE-local storage */
    uint64_t w_reuse_misses; /* Times a weight had to be fetched from SRAM */
} rs_impl_t;

static void rs_init(tu_dataflow_plugin_t *plugin) {
    rs_impl_t *rs = (rs_impl_t *)plugin->impl_data;
    if (!rs) {
        rs = (rs_impl_t *)calloc(1, sizeof(rs_impl_t));
        plugin->impl_data = rs;
    }
    rs->w_reuse_hits = 0;
    rs->w_reuse_misses = 0;
}

/*
 * RS execute_tile: row-stationary MAC execution.
 *
 * Dataflow pattern (per tile):
 *   For each output element O[m][n] within the tile:
 *     - The full dot product Σ_k W[m][k] * A[k][n] is accumulated
 *     - W[m][:] row is conceptually stationary (fetched once per tile, reused
 *       across all n positions in the tile)
 *
 * In the WS dataflow, the outer loops are (k, m, n) reflecting systolic
 * streaming. In RS, the outer loops are (m, n, k) reflecting row-stationary
 * computation — each output element is computed completely before moving on.
 *
 * Reuse accounting:
 *   - W row reuse: each W[m][k] is read once per m-row and reused for
 *     all n_count output columns → w_reuse_hits += n_count - 1 per element
 *   - A reuse: A[k][n] is read once per k-step and reused for all m_count
 *     output rows → implicit in row-stationary loop order
 */
static uint64_t rs_execute_tile(tu_dataflow_plugin_t *plugin,
                                 const tu_mma_op_t *op,
                                 uint16_t m_start, uint16_t m_count,
                                 uint16_t n_start, uint16_t n_count,
                                 uint16_t k_start, uint16_t k_count) {
    (void)plugin;
    rs_impl_t *rs = (rs_impl_t *)plugin->impl_data;

    const uint16_t *W_fp16 = (const uint16_t *)op->W.data;
    const uint16_t *A_fp16 = (const uint16_t *)op->A.data;
    float          *O_fp32 = (float *)op->O.data;

    uint32_t W_stride_el = op->W.stride / 2; /* stride in FP16 elements */
    uint32_t A_stride_el = op->A.stride / 2;
    uint32_t O_stride_el = op->O.stride / 4; /* stride in FP32 elements */

    /*
     * RS loop order: m (output rows, stationary W rows) →
     *                 n (output columns, streaming A columns) →
     *                 k (inner reduction, weight reuse across n)
     *
     * Each W[m_start+m][k_start+k] is reused for all n in 0..n_count,
     * modeling the row-stationary property where each PE holds a W row.
     */
    uint64_t w_reads = 0;

    for (uint16_t m = 0; m < m_count; m++) {
        for (uint16_t n = 0; n < n_count; n++) {
            float psum = 0.0f;
            for (uint16_t k = 0; k < k_count; k++) {
                float w_val = tu_fp16_to_fp32(
                    W_fp16[(m_start + m) * W_stride_el + (k_start + k)]);
                float a_val = tu_fp16_to_fp32(
                    A_fp16[(k_start + k) * A_stride_el + (n_start + n)]);
                psum += w_val * a_val;

                /* Count W reads: first n uses read W, subsequent reuse */
                if (n == 0) w_reads++;
            }
            O_fp32[(m_start + m) * O_stride_el + (n_start + n)] += psum;
        }
    }

    /* Update reuse statistics */
    if (rs) {
        /* W elements read once per m-row, reused for remaining n-1 outputs */
        uint64_t w_total_reads = (uint64_t)m_count * k_count;
        uint64_t w_total_uses  = (uint64_t)m_count * n_count * k_count;
        if (n_count > 1) {
            rs->w_reuse_hits  += w_total_uses - w_total_reads;
            rs->w_reuse_misses += w_total_reads;
        } else {
            rs->w_reuse_misses += w_total_reads;
        }
    }

    /*
     * RS cycle model:
     *   - Compute: k_count cycles (all PEs MAC in parallel)
     *   - RS has similar systolic characteristics to WS:
     *     W rows are preloaded, but A still needs to stream through columns.
     *     Add a small fill/drain overhead mirroring WS but slightly reduced
     *     because RS has better reuse of W rows.
     */
    return (uint64_t)k_count;
}

static uint64_t rs_get_fill_cycles(const tu_dataflow_plugin_t *plugin,
                                    uint16_t n_count, uint16_t k_count,
                                    uint16_t pipeline_depth) {
    (void)plugin; (void)k_count;
    /*
     * RS fill: similar to WS because A still streams through columns.
     * However, W rows are pre-distributed so fill is slightly less than WS.
     * WS uses pipeline_depth * tile_n; RS uses (pipeline_depth - 1) * tile_n + 1
     * reflecting the fact that W doesn't need to stream into place.
     */
    if (pipeline_depth <= 1) return 0;
    return (uint64_t)(pipeline_depth - 1) * n_count + 1;
}

static uint64_t rs_get_drain_cycles(const tu_dataflow_plugin_t *plugin,
                                     uint16_t m_count,
                                     uint16_t pipeline_depth) {
    (void)plugin;
    if (pipeline_depth <= 1) return 0;
    return (uint64_t)(pipeline_depth - 1) * m_count;
}

static uint64_t rs_get_compute_cycles(const tu_dataflow_plugin_t *plugin,
                                       uint16_t m_count, uint16_t n_count,
                                       uint16_t k_count) {
    (void)plugin; (void)m_count; (void)n_count;
    return (uint64_t)k_count;
}

/*
 * Create a row-stationary dataflow plugin.
 * The caller owns the returned plugin; call tu_dataflow_rs_destroy() to free.
 */
tu_dataflow_plugin_t *tu_dataflow_rs_create(void) {
    tu_dataflow_plugin_t *plugin = (tu_dataflow_plugin_t *)calloc(1, sizeof(tu_dataflow_plugin_t));
    if (!plugin) return NULL;

    plugin->name          = "row_stationary";
    plugin->id            = TU_DATAFLOW_ROW_STATIONARY;
    plugin->init          = rs_init;
    plugin->execute_tile  = rs_execute_tile;
    plugin->get_fill_cycles   = rs_get_fill_cycles;
    plugin->get_drain_cycles  = rs_get_drain_cycles;
    plugin->get_compute_cycles = rs_get_compute_cycles;

    rs_impl_t *rs = (rs_impl_t *)calloc(1, sizeof(rs_impl_t));
    plugin->impl_data = rs;

    return plugin;
}

void tu_dataflow_rs_destroy(tu_dataflow_plugin_t *plugin) {
    if (plugin) {
        free(plugin->impl_data);
        free(plugin);
    }
}
