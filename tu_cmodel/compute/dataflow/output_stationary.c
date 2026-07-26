/*
 * Output-Stationary (Vector) Dataflow
 * =====================================
 * A4: Production-grade OS dataflow plugin.
 *
 * Dataflow semantics (as used in TPUv2+, NVIDIA TensorCores):
 *   - Output/accumulators stay in PEs (stationary)
 *   - Weights and activations stream in
 *   - Each PE computes a full dot product for one output element
 *   - Pattern: O[i][j] stays in PE(i, j);
 *              W[i][k] is broadcast across row i;
 *              A[k][j] is broadcast down column j;
 *              PE(i, j) computes O[i][j] += Σ W[i][k] * A[k][j]
 *
 * Key difference from WS:
 *   - OS has NO systolic pipeline fill/drain overhead
 *   - Weights are fetched from SRAM per tile (not pre-stationary in PEs)
 *   - Higher bandwidth demand (reads W and A simultaneously each cycle)
 *   - Natural fit for vector/SIMD engines and TPU-style architectures
 *
 * Cycle model:
 *   - Pipeline fill: 0 (vector engine, no systolic latency)
 *   - Compute: k_count cycles (all PEs MAC in parallel)
 *   - Pipeline drain: 0
 *
 * The OS dataflow is bandwidth-limited, not compute-limited.
 * The cmodel accounts for additional SRAM read cycles for weight streaming.
 */

#include "dataflow_interface.h"
#include "tu_precision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Output-Stationary Plugin ---- */

typedef struct {
    uint16_t tile_m;
    uint16_t tile_n;
    uint16_t tile_k;
} os_impl_t;

static void os_init(tu_dataflow_plugin_t *plugin) {
    os_impl_t *os = (os_impl_t *)plugin->impl_data;
    if (!os) {
        os = (os_impl_t *)calloc(1, sizeof(os_impl_t));
        plugin->impl_data = os;
    }
}

/*
 * OS execute_tile: output-stationary MAC.
 *
 * For each output element (m, n), accumulate the dot product:
 *   O[m][n] += Σ_{k} W[m][k] * A[k][n]
 *
 * This is mathematically identical to the WS result, but the execution
 * pattern is reversed: the outer loops are over output positions, and
 * the inner loop is the reduction over k. In hardware, all PEs compute
 * in parallel; in our functional model, we iterate m/n/k.
 *
 * BW model: OS fetches W and A simultaneously per cycle.
 * WS has W stationary so no W fetch overhead after initial load.
 * OS adds ceil(k_count * tile_m * tile_n / PE_count) W-fetch cycles.
 */
static uint64_t os_execute_tile(tu_dataflow_plugin_t *plugin,
                                 const tu_mma_op_t *op,
                                 uint16_t m_start, uint16_t m_count,
                                 uint16_t n_start, uint16_t n_count,
                                 uint16_t k_start, uint16_t k_count) {
    (void)plugin;

    const uint16_t *W_fp16 = (const uint16_t *)op->W.data;
    const uint16_t *A_fp16 = (const uint16_t *)op->A.data;
    float          *O_fp32 = (float *)op->O.data;

    uint32_t W_stride_el = op->W.stride / 2;
    uint32_t A_stride_el = op->A.stride / 2;
    uint32_t O_stride_el = op->O.stride / 4;

    /* OS dataflow: output elements are stationary. Accumulate dot products. */
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

    /*
     * OS cycle model:
     *   - k_count compute cycles (all PEs MAC in parallel)
     *   - Weight fetch overhead: OS must re-fetch W each K-step
     *     because weights are NOT stationary in PEs.
     *     Overhead: (m_count * k_count * 2 bytes) / bus_width cycles
     *     In practice, W-fetch is hidden by double-buffering, so
     *     we add a small overhead: ceil(k_count / 4) extra cycles.
     */
    return (uint64_t)k_count + (k_count + 3) / 4;
}

static uint64_t os_get_fill_cycles(const tu_dataflow_plugin_t *plugin,
                                    uint16_t n_count, uint16_t k_count,
                                    uint16_t pipeline_depth) {
    (void)plugin; (void)n_count; (void)k_count; (void)pipeline_depth;
    /* OS has no systolic pipeline — fill overhead is 0 */
    return 0;
}

static uint64_t os_get_drain_cycles(const tu_dataflow_plugin_t *plugin,
                                     uint16_t m_count,
                                     uint16_t pipeline_depth) {
    (void)plugin; (void)m_count; (void)pipeline_depth;
    return 0;
}

static uint64_t os_get_compute_cycles(const tu_dataflow_plugin_t *plugin,
                                       uint16_t m_count, uint16_t n_count,
                                       uint16_t k_count) {
    (void)plugin; (void)m_count; (void)n_count;
    return (uint64_t)k_count;
}

/*
 * Create an output-stationary dataflow plugin.
 */
tu_dataflow_plugin_t *tu_dataflow_os_create(void) {
    tu_dataflow_plugin_t *plugin = (tu_dataflow_plugin_t *)calloc(1, sizeof(tu_dataflow_plugin_t));
    plugin->name          = "output_stationary";
    plugin->id            = TU_DATAFLOW_OUTPUT_STATIONARY;
    plugin->init          = os_init;
    plugin->execute_tile  = os_execute_tile;
    plugin->get_fill_cycles   = os_get_fill_cycles;
    plugin->get_drain_cycles  = os_get_drain_cycles;
    plugin->get_compute_cycles = os_get_compute_cycles;

    os_impl_t *os = (os_impl_t *)calloc(1, sizeof(os_impl_t));
    plugin->impl_data = os;

    return plugin;
}

void tu_dataflow_os_destroy(tu_dataflow_plugin_t *plugin) {
    if (plugin) {
        free(plugin->impl_data);
        free(plugin);
    }
}
