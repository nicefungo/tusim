/*
 * Rounding Mode Accuracy Sweep: RNE vs RTZ vs Stochastic for GEMM 128×128×256
 * ===========================================================================
 * Measures how rounding mode affects numerical accuracy for matrix multiply.
 * Computes FP64 golden reference. Compares FP16 GEMM under RNE/RTZ/Stochastic.
 */

#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_core.h"
#include "tu_cmodel/tu_config.h"
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/rounding.h"
#include "tu_cmodel/dma_descriptor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define M_WL 128
#define N_WL 128
#define K_WL 256
#define N_ELEMS (M_WL * K_WL + K_WL * N_WL + M_WL * N_WL)

static double  g_W_fp64[M_WL * K_WL];
static double  g_A_fp64[K_WL * N_WL];
static double  g_O_golden[M_WL * N_WL];
static fp16_t  g_W_fp16[M_WL * K_WL];
static fp16_t  g_A_fp16[K_WL * N_WL];
static float   g_O_result[M_WL * N_WL];

static void fill_random_double(double *data, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        data[i] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

/* FP64 golden GEMM */
static void gemm_fp64(double *W, double *A, double *O, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++)
                sum += W[m * K + k] * A[k * N + n];
            O[m * N + n] = sum;
        }
    }
}

static void run_gemm_with_rounding(tu_rounding_mode_t mode, const char *name) {
    tu_runtime_config_t cfg = tu_runtime_config_default();
    tu_core_t *core = tu_core_create(&cfg);
    tu_core_init(core);
    tu_set_rounding_mode(mode);
    tu_set_subnormal_mode(TU_SUBNORMAL_FLUSH);

    /* Convert to FP16 using current rounding mode */
    for (int i = 0; i < M_WL * K_WL; i++)
        g_W_fp16[i] = tu_fp32_to_fp16((float)g_W_fp64[i]);
    for (int i = 0; i < K_WL * N_WL; i++)
        g_A_fp16[i] = tu_fp32_to_fp16((float)g_A_fp64[i]);

    tu_core_dma_load_w(core, g_W_fp16, 0, M_WL * K_WL * sizeof(fp16_t));
    tu_core_dma_load_a(core, g_A_fp16, 0, K_WL * N_WL * sizeof(fp16_t));
    tu_core_mma(core, M_WL, N_WL, K_WL, 0, 0, 0, false);
    tu_core_sync(core);
    tu_core_dma_store_o(core, g_O_result, 0, M_WL * N_WL * sizeof(float));
    tu_core_destroy(core);

    /* Compare against golden */
    double max_err = 0.0, sum_err = 0.0;
    double max_rel = 0.0;
    int violations = 0;
    for (int i = 0; i < M_WL * N_WL; i++) {
        double err = fabs((double)g_O_result[i] - g_O_golden[i]);
        sum_err += err;
        if (err > max_err) max_err = err;
        double rel = (fabs(g_O_golden[i]) > 1e-10) ? err / fabs(g_O_golden[i]) : err;
        if (rel > max_rel) max_rel = rel;
        if (err > 0.5) violations++;  /* tolerance for FP16 accumulation */
    }
    double mean_err = sum_err / (M_WL * N_WL);
    printf("  %-22s  max_err=%.6e  mean_err=%.6e  max_rel=%.4f  viol>0.5=%d\n",
           name, max_err, mean_err, max_rel, violations);
}

int main(void) {
    fill_random_double(g_W_fp64, M_WL * K_WL, 42);
    fill_random_double(g_A_fp64, K_WL * N_WL, 99);
    gemm_fp64(g_W_fp64, g_A_fp64, g_O_golden, M_WL, N_WL, K_WL);

    printf("\n=== Rounding Mode Accuracy Sweep: GEMM %d×%d×%d ===\n\n", M_WL, N_WL, K_WL);
    printf("  Golden done (FP64). Range: W=[%.3f, %.3f], A=[%.3f, %.3f]\n",
           g_W_fp64[0], g_W_fp64[M_WL*K_WL-1],
           g_A_fp64[0], g_A_fp64[K_WL*N_WL-1]);
    printf("  Golden O[0]=%.6f\n\n", g_O_golden[0]);

    printf("  Rounding       Error\n");
    printf("  ─────────────────────────────────────────────────────\n");
    run_gemm_with_rounding(TU_ROUND_RNE, "RNE (default)");
    run_gemm_with_rounding(TU_ROUND_RTZ, "RTZ (truncate)");
    run_gemm_with_rounding(TU_ROUND_STOCHASTIC, "Stochastic");

    /* Repeat stochastic with different seed to check variance */
    tu_stochastic_set_seed(0xDEADBEEF);
    run_gemm_with_rounding(TU_ROUND_STOCHASTIC, "Stochastic (seed=DEAD)");

    printf("\n  --- Finding ---\n");
    printf("  RNE is the IEEE 754 default — unbiased, min error.\n");
    printf("  RTZ truncates toward zero, introducing systematic negative bias.\n");
    printf("  Stochastic dithers the LSB — unbiased in expectation but higher variance.\n");
    printf("  For GEMM with K=256 (many accumulations), rounding differences compound.\n\n");

    return 0;
}
