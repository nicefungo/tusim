/*
 * Dataflow Sweep: WS vs OS vs RS for GEMM 128×128×256
 * =====================================================
 * Verifies functional correctness across all three dataflows and
 * computes analytical cycle model for each.
 *
 * The cmodel runs in FUNCTIONAL mode (TU_CYCLE_MODEL=0) which means
 * all dataflows produce identical cycle counts. The analytical model
 * captures fill/drain overhead that would appear in cycle-accurate mode.
 */

#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_core.h"
#include "tu_cmodel/tu_config.h"
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/dma_descriptor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define M_WORKLOAD 128
#define N_WORKLOAD 128
#define K_WORKLOAD 256

static float g_W_fp32[M_WORKLOAD * K_WORKLOAD];
static float g_A_fp32[K_WORKLOAD * N_WORKLOAD];
static float g_O_ws[M_WORKLOAD * N_WORKLOAD];
static float g_O_os[M_WORKLOAD * N_WORKLOAD];
static float g_O_rs[M_WORKLOAD * N_WORKLOAD];
static fp16_t  g_W_fp16[M_WORKLOAD * K_WORKLOAD];
static fp16_t  g_A_fp16[K_WORKLOAD * N_WORKLOAD];

static void fill_random(float *data, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static int run_mma_and_capture(tu_core_t *core, int df_id, float *out) {
    tu_set_dataflow(df_id);

    /* DMA load W, A */
    for (int i = 0; i < M_WORKLOAD * K_WORKLOAD; i++)
        g_W_fp16[i] = tu_fp32_to_fp16(g_W_fp32[i]);
    for (int i = 0; i < K_WORKLOAD * N_WORKLOAD; i++)
        g_A_fp16[i] = tu_fp32_to_fp16(g_A_fp32[i]);

    tu_core_dma_load_w(core, g_W_fp16, 0, M_WORKLOAD * K_WORKLOAD * sizeof(fp16_t));
    tu_core_dma_load_a(core, g_A_fp16, 0, K_WORKLOAD * N_WORKLOAD * sizeof(fp16_t));

    /* MMA */
    tu_core_mma(core, M_WORKLOAD, N_WORKLOAD, K_WORKLOAD, 0, 0, 0, false);
    tu_core_sync(core);

    /* DMA store O */
    tu_core_dma_store_o(core, out, 0, M_WORKLOAD * N_WORKLOAD * sizeof(float));

    return 0;
}

static int check_identical(const float *a, const float *b, int n, const char *label) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > 1e-5f) {
            printf("  MISMATCH at %s[%d]: %.6f vs %.6f (diff %.6e)\n",
                   label, i, (double)a[i], (double)b[i], (double)fabsf(a[i]-b[i]));
            return 0;
        }
    }
    return 1;
}

int main(void) {
    fill_random(g_W_fp32, M_WORKLOAD * K_WORKLOAD, 42);
    fill_random(g_A_fp32, K_WORKLOAD * N_WORKLOAD, 99);

    tu_runtime_config_t cfg = tu_runtime_config_default();

    printf("\n=== TU CModel Dataflow Sweep: GEMM %d×%d×%d ===\n\n",
           M_WORKLOAD, N_WORKLOAD, K_WORKLOAD);
    printf("PE array: 16×16 (256 MACs), clock: 1 GHz, peak: 0.51 TOPS\n");
    printf("Workload: %lu MFLOPs, DMA: %lu KB (W=%lu KB, A=%lu KB, O=%lu KB)\n\n",
           (unsigned long)((uint64_t)M_WORKLOAD * N_WORKLOAD * K_WORKLOAD * 2 / 1000000),
           (unsigned long)(M_WORKLOAD * K_WORKLOAD * 2 / 1024 +
                           K_WORKLOAD * N_WORKLOAD * 2 / 1024 +
                           M_WORKLOAD * N_WORKLOAD * 4 / 1024),
           (unsigned long)(M_WORKLOAD * K_WORKLOAD * 2 / 1024),
           (unsigned long)(K_WORKLOAD * N_WORKLOAD * 2 / 1024),
           (unsigned long)(M_WORKLOAD * N_WORKLOAD * 4 / 1024));

    /* Functional correctness: all three dataflows must agree */
    printf("--- Functional Correctness ---\n");

    tu_core_t *core;

    core = tu_core_create(&cfg); tu_core_init(core);
    run_mma_and_capture(core, 0, g_O_ws);
    tu_core_destroy(core);
    printf("  weight_stationary:  O[0]=%.6f, O[%d]=%.6f — OK\n",
           g_O_ws[0], M_WORKLOAD*N_WORKLOAD-1, g_O_ws[M_WORKLOAD*N_WORKLOAD-1]);

    core = tu_core_create(&cfg); tu_core_init(core);
    run_mma_and_capture(core, 1, g_O_os);
    tu_core_destroy(core);
    printf("  output_stationary:  O[0]=%.6f, O[%d]=%.6f — %s\n",
           g_O_os[0], M_WORKLOAD*N_WORKLOAD-1, g_O_os[M_WORKLOAD*N_WORKLOAD-1],
           check_identical(g_O_ws, g_O_os, M_WORKLOAD*N_WORKLOAD, "OS") ? "match WS ✓" : "MISMATCH ✗");

    core = tu_core_create(&cfg); tu_core_init(core);
    run_mma_and_capture(core, 2, g_O_rs);
    tu_core_destroy(core);
    printf("  row_stationary:     O[0]=%.6f, O[%d]=%.6f — %s\n\n",
           g_O_rs[0], M_WORKLOAD*N_WORKLOAD-1, g_O_rs[M_WORKLOAD*N_WORKLOAD-1],
           check_identical(g_O_ws, g_O_rs, M_WORKLOAD*N_WORKLOAD, "RS") ? "match WS ✓" : "MISMATCH ✗");

    /* Analytical cycle model */
    printf("--- Analytical Cycle Model (pipeline depth=2) ---\n\n");
    uint16_t pe_r = cfg.pe_rows, pe_c = cfg.pe_cols;
    uint16_t pd = 2;
    uint16_t tile_m = (M_WORKLOAD + pe_r - 1) / pe_r;
    uint16_t tile_n = (N_WORKLOAD + pe_c - 1) / pe_c;

    uint64_t total_flops = (uint64_t)M_WORKLOAD * N_WORKLOAD * K_WORKLOAD * 2;
    uint64_t total_dma_bytes = M_WORKLOAD * K_WORKLOAD * 2 +
                                K_WORKLOAD * N_WORKLOAD * 2 +
                                M_WORKLOAD * N_WORKLOAD * 4;

    /* Formula per tile:
     *   WS: pd*N + K + pd*M   (fill + compute + drain)
     *   OS: K                  (no fill/drain, vector-style)
     *   RS: (pd-1)*N+1 + K + (pd-1)*M   (reduced fill/drain)
     */
    uint64_t ws_cyc = 0, os_cyc = 0, rs_cyc = 0;
    for (uint16_t tm = 0; tm < tile_m; tm++) {
        uint16_t mc = (tm == tile_m - 1) ? M_WORKLOAD - tm * pe_r : pe_r;
        for (uint16_t tn = 0; tn < tile_n; tn++) {
            uint16_t nc = (tn == tile_n - 1) ? N_WORKLOAD - tn * pe_c : pe_c;
            ws_cyc += (pd * nc) + K_WORKLOAD + (pd * mc);
            os_cyc += K_WORKLOAD;
            rs_cyc += ((pd - 1) * nc + 1) + K_WORKLOAD + ((pd - 1) * mc);
        }
    }

    /* DMA cycles: bus_width=32 B/cycle, total bytes / bytes_per_cycle */
    uint64_t dma_cyc = total_dma_bytes / TU_DMA_BUS_WIDTH_BYTES;

    printf("  Tiles: %d M-rows × %d N-cols = %d tiles\n", tile_m, tile_n, tile_m * tile_n);
    printf("  Inner dim K=%d, pipeline depth=%d\n\n", K_WORKLOAD, pd);

    double peak_flops_per_cycle = 256.0 * 2.0;  /* 256 MACs × 2 flops/MAC */
    double total_macs = (double)total_flops / 2.0;
    printf("  ┌─────────────────────┬──────────┬──────────┬──────────┐\n");
    printf("  │ Dataflow            │  kCycles │   mTOPS  │  Util %%  │\n");
    printf("  ├─────────────────────┼──────────┼──────────┼──────────┤\n");
    printf("  │ weight_stationary   │ %8lu │ %8.1f │ %7.1f  │\n",
           (unsigned long)(ws_cyc / 1000),
           total_macs / (double)ws_cyc,
           100.0 * (double)total_flops / (double)(ws_cyc) / peak_flops_per_cycle);
    printf("  │ output_stationary   │ %8lu │ %8.1f │ %7.1f  │\n",
           (unsigned long)(os_cyc / 1000),
           total_macs / (double)os_cyc,
           100.0 * (double)total_flops / (double)(os_cyc) / peak_flops_per_cycle);
    printf("  │ row_stationary      │ %8lu │ %8.1f │ %7.1f  │\n",
           (unsigned long)(rs_cyc / 1000),
           total_macs / (double)rs_cyc,
           100.0 * (double)total_flops / (double)(rs_cyc) / peak_flops_per_cycle);
    printf("  └─────────────────────┴──────────┴──────────┴──────────┘\n\n");

    printf("  DMA overhead: %lu kCycles (%lu KB at %d B/cycle)\n",
           (unsigned long)(dma_cyc / 1000),
           (unsigned long)(total_dma_bytes / 1024),
           TU_DMA_BUS_WIDTH_BYTES);
    printf("  Total w/ DMA: WS=%lu kCyc, OS=%lu kCyc, RS=%lu kCyc\n\n",
           (unsigned long)((ws_cyc + dma_cyc) / 1000),
           (unsigned long)((os_cyc + dma_cyc) / 1000),
           (unsigned long)((rs_cyc + dma_cyc) / 1000));

    /* Finding */
    printf("--- Key Finding ---\n");
    printf("  OS is %.0f%% faster than WS (%.0f kCyc vs %.0f kCyc)\n",
           100.0 * (1.0 - (double)os_cyc / (double)ws_cyc),
           (double)os_cyc / 1000.0, (double)ws_cyc / 1000.0);
    printf("  RS is %.0f%% faster than WS (%.0f kCyc vs %.0f kCyc)\n",
           100.0 * (1.0 - (double)rs_cyc / (double)ws_cyc),
           (double)rs_cyc / 1000.0, (double)ws_cyc / 1000.0);
    printf("  OS is %.0f%% faster than RS (cycle model omits fill/drain entirely)\n",
           100.0 * (1.0 - (double)os_cyc / (double)rs_cyc));
    printf("  RS's fill/drain reduction (pd-1 vs pd) saves %.0f cycles per tile\n",
           (double)(ws_cyc - rs_cyc) / (double)(tile_m * tile_n));
    printf("\n");

    return 0;
}
