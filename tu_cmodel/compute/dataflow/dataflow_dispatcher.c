/*
 * TU Dataflow Dispatcher — Tiling + Plugin Dispatch
 * ==================================================
 * A4: This is the high-level MMA entry point. It:
 *   1. Decomposes the full M×N×K matrix into tile_m×tile_n×tile_k tiles
 *   2. Calls the selected dataflow plugin's execute_tile() for each tile
 *   3. Accumulates cycle counts and stats
 *
 * This is the function that replaces the hard-coded tiling loop in tu_cmodel.c.
 */

#include "dataflow_interface.h"
#include <stdio.h>

uint64_t tu_dataflow_execute_mma(tu_dataflow_plugin_t *plugin,
                                  const tu_dataflow_tensor_t *W,
                                  const tu_dataflow_tensor_t *A,
                                  tu_dataflow_tensor_t *O,
                                  uint16_t tile_m, uint16_t tile_n,
                                  uint16_t tile_k, uint16_t pipeline_depth) {
    if (!plugin || !W || !A || !O || !plugin->execute_tile) {
        fprintf(stderr, "TU DATAFLOW ERROR: invalid plugin or tensors\n");
        return 0;
    }

    uint16_t M = (uint16_t)W->rows;
    uint16_t N = (uint16_t)A->cols;
    uint16_t K = (uint16_t)W->cols; /* W[M][K], A[K][N] */

    /* Build the MMA op descriptor */
    tu_mma_op_t op = {
        .W = *W,
        .A = *A,
        .O = *O,
        .has_bias       = false, /* Bias handled by caller before dispatch */
        .tile_m         = tile_m,
        .tile_n         = tile_n,
        .tile_k         = tile_k,
        .pipeline_depth = pipeline_depth,
    };

    /* Initialize plugin if needed */
    if (plugin->init) plugin->init(plugin);

    /* Tile counts */
    uint16_t mt = (M + tile_m - 1) / tile_m;
    uint16_t nt = (N + tile_n - 1) / tile_n;
    uint16_t kt = (K + tile_k - 1) / tile_k;

    uint64_t total_cycles = 0;
    uint64_t total_flops  = 0;
    uint64_t tile_count   = 0;

    for (uint16_t mi = 0; mi < mt; mi++) {
        uint16_t m_start = mi * tile_m;
        uint16_t m_count = ((mi + 1) * tile_m <= M) ? tile_m : (M - m_start);

        for (uint16_t ni = 0; ni < nt; ni++) {
            uint16_t n_start = ni * tile_n;
            uint16_t n_count = ((ni + 1) * tile_n <= N) ? tile_n : (N - n_start);

            for (uint16_t ki = 0; ki < kt; ki++) {
                uint16_t k_start = ki * tile_k;
                uint16_t k_count = ((ki + 1) * tile_k <= K) ? tile_k : (K - k_start);

                tile_count++;

                /* Compute FLOPs: each MAC = 2 FLOPs (multiply + add) */
                total_flops += (uint64_t)m_count * n_count * k_count * 2;

                /* Pipeline fill cycles */
                if (plugin->get_fill_cycles) {
                    total_cycles += plugin->get_fill_cycles(plugin, tile_n, tile_k);
                }

                /* Execute the tile through the selected dataflow */
                uint64_t tile_cycles = plugin->execute_tile(
                    plugin, &op,
                    m_start, m_count, n_start, n_count, k_start, k_count);
                total_cycles += tile_cycles;

                /* Pipeline drain cycles */
                if (plugin->get_drain_cycles) {
                    total_cycles += plugin->get_drain_cycles(plugin, tile_m);
                }
            }
        }
    }

    /* Update plugin stats */
    plugin->total_flops  += total_flops;
    plugin->total_tiles  += tile_count;
    plugin->total_cycles += total_cycles;

    return total_cycles;
}
