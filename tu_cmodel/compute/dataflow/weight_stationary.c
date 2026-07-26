/*
 * Weight-Stationary (Systolic) Dataflow
 * ======================================
 * A4: Production-grade WS dataflow plugin.
 *
 * Dataflow semantics:
 *   - Weights are preloaded into PEs (stationary)
 *   - Activations stream right across the PE array columns
 *   - Partial sums flow down across the PE array rows
 *   - Pattern: W[i][k] stays in PE(i, k); A[k][j] enters PE(i, k) from left;
 *              psum[i][j] enters PE(i, k) from above, MAC result exits below
 *
 * Cycle model:
 *   - Pipeline fill: pipeline_depth * tile_n cycles (fill the columns)
 *   - Compute: k_count cycles (1 MAC/cycle per PE after fill)
 *   - Pipeline drain: pipeline_depth * tile_m cycles (drain the rows)
 *
 * This is the original TinyTU dataflow, refactored into a plugin.
 */

#include "dataflow_interface.h"
#include "tu_precision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Weight-Stationary Plugin ---- */

typedef struct {
    uint16_t tile_m;
    uint16_t tile_n;
    uint16_t tile_k;
    uint16_t pipeline_depth;
} ws_impl_t;

static void ws_init(tu_dataflow_plugin_t *plugin) {
    ws_impl_t *ws = (ws_impl_t *)plugin->impl_data;
    if (!ws) {
        ws = (ws_impl_t *)calloc(1, sizeof(ws_impl_t));
        plugin->impl_data = ws;
    }
    /* No per-invocation state to reset; tiles are independent. */
}

static uint64_t ws_execute_tile(tu_dataflow_plugin_t *plugin,
                                 const tu_mma_op_t *op,
                                 uint16_t m_start, uint16_t m_count,
                                 uint16_t n_start, uint16_t n_count,
                                 uint16_t k_start, uint16_t k_count) {
    (void)plugin;
    const uint16_t *W_fp16 = (const uint16_t *)op->W.data;
    const uint16_t *A_fp16 = (const uint16_t *)op->A.data;
    float          *O_fp32 = (float *)op->O.data;

    uint32_t W_stride_el = op->W.stride / 2; /* stride in FP16 elements */
    uint32_t A_stride_el = op->A.stride / 2;
    uint32_t O_stride_el = op->O.stride / 4; /* stride in FP32 elements */

    for (uint16_t m = 0; m < m_count; m++) {
        for (uint16_t n = 0; n < n_count; n++) {
            float psum = 0.0f;
            for (uint16_t k = 0; k < k_count; k++) {
                float w_val = tu_fp16_to_fp32(
                    W_fp16[(m_start + m) * W_stride_el + (k_start + k)]);
                float a_val = tu_fp16_to_fp32(
                    A_fp16[(k_start + k) * A_stride_el + (n_start + n)]);
                psum += w_val * a_val;
            }
            O_fp32[(m_start + m) * O_stride_el + (n_start + n)] += psum;
        }
    }

    return (uint64_t)k_count; /* compute cycles */
}

static uint64_t ws_get_fill_cycles(const tu_dataflow_plugin_t *plugin,
                                    uint16_t n_count, uint16_t k_count,
                                    uint16_t pipeline_depth) {
    (void)plugin; (void)k_count;
    /* WS: pipeline_depth * tile_n — must fill all columns */
    return (uint64_t)pipeline_depth * n_count;
}

static uint64_t ws_get_drain_cycles(const tu_dataflow_plugin_t *plugin,
                                     uint16_t m_count,
                                     uint16_t pipeline_depth) {
    (void)plugin;
    return (uint64_t)pipeline_depth * m_count;
}

static uint64_t ws_get_compute_cycles(const tu_dataflow_plugin_t *plugin,
                                       uint16_t m_count, uint16_t n_count,
                                       uint16_t k_count) {
    (void)plugin; (void)m_count; (void)n_count;
    return (uint64_t)k_count;
}

/*
 * Create a weight-stationary dataflow plugin.
 * The caller owns the returned plugin; call tu_dataflow_ws_destroy() to free.
 */
tu_dataflow_plugin_t *tu_dataflow_ws_create(void) {
    tu_dataflow_plugin_t *plugin = (tu_dataflow_plugin_t *)calloc(1, sizeof(tu_dataflow_plugin_t));
    plugin->name          = "weight_stationary";
    plugin->id            = TU_DATAFLOW_WEIGHT_STATIONARY;
    plugin->init          = ws_init;
    plugin->execute_tile  = ws_execute_tile;
    plugin->get_fill_cycles   = ws_get_fill_cycles;
    plugin->get_drain_cycles  = ws_get_drain_cycles;
    plugin->get_compute_cycles = ws_get_compute_cycles;

    ws_impl_t *ws = (ws_impl_t *)calloc(1, sizeof(ws_impl_t));
    plugin->impl_data = ws;

    return plugin;
}

void tu_dataflow_ws_destroy(tu_dataflow_plugin_t *plugin) {
    if (plugin) {
        free(plugin->impl_data);
        free(plugin);
    }
}
