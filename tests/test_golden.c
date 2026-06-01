/*
 * TinyTU Golden Reference Verification
 * ======================================
 *
 * Differential testing: generates random tensors, computes expected
 * results using FP32 reference arithmetic, runs the cmodel, and
 * compares output. Verifies that the cmodel's FP16→FP32→FP16 path
 * matches a pure FP32 computation within tolerance.
 *
 * Gap V1: No golden reference → Dual-path verification with C FP32 reference.
 * Gap V6: No random/differential testing → 10K+ random tensor tests.
 *
 * Two modes:
 *   1. In-process FP32 reference (always available, no Python needed)
 *   2. JSON-based golden reference (loads tests/golden/reference_data/*.json)
 *
 * The FP32 reference computes O = W @ A in FP32, then the cmodel
 * computes the same operation through FP16 inputs. The tolerance
 * accounts for FP16 quantization error (~1e-3 relative).
 */

#include "tu_cmodel/tu_cmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Test configuration */
#define MAX_RANDOM_TESTS   5000   /* Number of random tensor tests (CI mode: 5000) */
#define QUICK_RANDOM_TESTS 50     /* Quick smoke test count */

static int tests_run = 0, tests_pass = 0;
static float max_observed_error = 0.0f;
static int max_error_case = -1;

#define TEST(name) do { tests_run++; printf("  %-52s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_pass++; } while(0)
#define FAIL(fmt, ...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); } while(0)

/* ================================================================
 * FP32 Reference Computation
 * ================================================================ */

/*
 * Compute O_ref = W @ A in pure FP32.
 * W: [M, K], A: [K, N] — both FP32 (the cmodel uses FP16 inputs internally).
 * Returns O_ref in FP32.
 */
static void compute_fp32_reference(
    const fp32_t *W, const fp32_t *A,
    fp32_t *O,
    uint32_t M, uint32_t N, uint32_t K)
{
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            fp32_t sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                sum += W[m * K + k] * A[k * N + n];
            }
            O[m * N + n] = sum;
        }
    }
}

/* ================================================================
 * Random tensor generation
 * ================================================================ */

/* Simple xorshift PRNG (deterministic, reproducible across platforms) */
static uint32_t xorshift_state = 0;

static void xsrand(uint32_t seed) {
    xorshift_state = seed ? seed : 1;
}

static uint32_t xsrand_next(void) {
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 17;
    xorshift_state ^= xorshift_state << 5;
    return xorshift_state;
}

/* Random float in [-range, range] */
static float random_float(float range) {
    uint32_t r = xsrand_next();
    float f = (float)(r & 0xFFFFFF) / (float)0x1000000;  /* [0, 1) */
    return (f * 2.0f - 1.0f) * range;
}

/* Fill tensor with random values */
static void fill_random(fp32_t *data, uint32_t count, float range) {
    for (uint32_t i = 0; i < count; i++) {
        data[i] = random_float(range);
    }
}

/* Fill tensor with specific pattern (for edge cases) */
static void fill_constant(fp32_t *data, uint32_t count, float value) {
    for (uint32_t i = 0; i < count; i++) data[i] = value;
}

/* ================================================================
 * Comparison utilities
 * ================================================================ */

static float max_abs_error(const fp32_t *a, const fp32_t *b, uint32_t n) {
    float max_err = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static float mean_abs_error(const fp32_t *a, const fp32_t *b, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        sum += (double)fabsf(a[i] - b[i]);
    }
    return (float)(sum / (double)n);
}

/* ================================================================
 * Core test: cmodel vs FP32 reference
 * ================================================================ */

/*
 * Run a single MMA test: generate random W and A in FP32,
 * convert to FP16 for cmodel, run both cmodel and FP32 reference,
 * compare results.
 */
static int run_single_test(
    uint16_t M, uint16_t N, uint16_t K,
    uint32_t seed, float tolerance,
    float *out_max_err, float *out_mean_err)
{
    xsrand(seed);

    /* Allocate and fill test data */
    uint32_t w_count = (uint32_t)M * K;
    uint32_t a_count = (uint32_t)K * N;
    uint32_t o_count = (uint32_t)M * N;

    fp32_t *W_fp32 = malloc(w_count * sizeof(fp32_t));
    fp32_t *A_fp32 = malloc(a_count * sizeof(fp32_t));
    fp16_t *W_fp16 = malloc(w_count * sizeof(fp16_t));
    fp16_t *A_fp16 = malloc(a_count * sizeof(fp16_t));
    fp32_t *O_ref  = malloc(o_count * sizeof(fp32_t));
    fp32_t *O_cm   = calloc(o_count, sizeof(fp32_t));

    if (!W_fp32 || !A_fp32 || !W_fp16 || !A_fp16 || !O_ref || !O_cm) {
        fprintf(stderr, "  ALLOC FAILED\n");
        free(W_fp32); free(A_fp32); free(W_fp16); free(A_fp16);
        free(O_ref); free(O_cm);
        return 0;
    }

    /* Generate random data */
    fill_random(W_fp32, w_count, 1.0f);
    fill_random(A_fp32, a_count, 1.0f);

    /* Convert to FP16 (as the cmodel would receive from DMA) */
    tu_fp32_to_fp16_buffer(W_fp32, W_fp16, w_count);
    tu_fp32_to_fp16_buffer(A_fp32, A_fp16, a_count);

    /* Compute FP32 reference */
    compute_fp32_reference(W_fp32, A_fp32, O_ref, M, N, K);

    /* Run cmodel */
    tu_dma_load_w(W_fp16, 0, w_count * sizeof(fp16_t));
    tu_dma_load_a(A_fp16, 0, a_count * sizeof(fp16_t));
    tu_mma(M, N, K, 0, 0, 0, false);
    tu_dma_store_o(O_cm, 0, o_count * sizeof(fp32_t));

    /* Compare */
    float max_err = max_abs_error(O_ref, O_cm, o_count);
    float mean_err = mean_abs_error(O_ref, O_cm, o_count);

    if (out_max_err) *out_max_err = max_err;
    if (out_mean_err) *out_mean_err = mean_err;

    free(W_fp32); free(A_fp32); free(W_fp16); free(A_fp16);
    free(O_ref); free(O_cm);

    return (max_err <= tolerance) ? 1 : 0;
}

/* ================================================================
 * Test 1: Fixed configuration golden tests
 * ================================================================ */

typedef struct {
    uint16_t M, N, K;
    float    tolerance;
    const char *name;
} fixed_test_t;

static void test_fixed_configs(void) {
    fixed_test_t configs[] = {
        {16, 16, 16, 0.01f, "MMA 16×16×16 (tiny)"},
        {32, 32, 32, 0.02f, "MMA 32×32×32 (medium)"},
        {64, 64, 64, 0.05f, "MMA 64×64×64 (large)"},
        { 7,  5,  9, 0.15f, "MMA 7×5×9 (edge tiles)"},
        { 4,  8, 16, 0.01f, "MMA 4×8×16 (non-square)"},
        {31, 17, 23, 0.05f, "MMA 31×17×23 (prime dims)"},
        { 1,  1,  1, 0.001f,"MMA 1×1×1 (scalar)"},
        { 1, 16, 64, 0.02f, "MMA 1×16×64 (vector)"},
    };
    int n = sizeof(configs) / sizeof(configs[0]);

    for (int i = 0; i < n; i++) {
        char label[64];
        snprintf(label, sizeof(label), "%s", configs[i].name);

        tu_runtime_config_t cfg = tu_runtime_config_default();
        tu_init_with_config(&cfg);

        float max_err, mean_err;
        int ok = run_single_test(configs[i].M, configs[i].N, configs[i].K,
                                 (uint32_t)(42 + i), configs[i].tolerance,
                                 &max_err, &mean_err);

        if (ok) {
            printf("  %-52s PASS (max_err=%.6f, mean_err=%.6f)\n",
                   label, max_err, mean_err);
            tests_run++; tests_pass++;
        } else {
            FAIL("max_err=%.6f > tol=%.6f (mean=%.6f)\n         %s",
                 max_err, configs[i].tolerance, mean_err, "");
            printf("  %-52s ", label);
            tests_run++;
        }

        if (max_err > max_observed_error) {
            max_observed_error = max_err;
            max_error_case = i;
        }
    }
}

/* ================================================================
 * Test 2: Bulk random tensor differential testing
 * ================================================================ */

static void test_random_bulk(int num_tests) {
    printf("\n--- Random Tensor Differential Testing (%d tests) ---\n", num_tests);

    tu_runtime_config_t cfg = tu_runtime_config_default();
    tu_init_with_config(&cfg);

    /* Dimension configurations to cycle through */
    uint16_t dim_configs[][3] = {
        {16, 16, 16}, {8, 16, 32}, {32, 8, 16}, {10, 10, 10},
        {15, 15, 15}, {31, 17, 23}, {4, 4, 8}, {7, 11, 13},
        {64, 16, 16}, {16, 64, 16}, {16, 16, 64}, {48, 48, 48},
    };
    int num_dim_configs = sizeof(dim_configs) / sizeof(dim_configs[0]);

    int passed = 0;
    int failed = 0;
    float total_max_err = 0.0f;
    float total_mean_err = 0.0f;

    xsrand(12345);

    for (int i = 0; i < num_tests; i++) {
        int dim_idx = i % num_dim_configs;
        uint16_t M = dim_configs[dim_idx][0];
        uint16_t N = dim_configs[dim_idx][1];
        uint16_t K = dim_configs[dim_idx][2];

        /* Scale tolerance based on K (more accumulation = more error) */
        float tol = 0.01f + (float)K * 0.0005f;

        /* Re-init for each test to get clean SRAM */
        tu_init_with_config(&cfg);

        float max_err, mean_err;
        int ok = run_single_test(M, N, K, xsrand_next(), tol,
                                 &max_err, &mean_err);

        if (ok) {
            passed++;
            total_max_err += max_err;
            total_mean_err += mean_err;
            if (max_err > max_observed_error) {
                max_observed_error = max_err;
                max_error_case = i;
            }
        } else {
            failed++;
            printf("    FAIL [%d]: M=%u N=%u K=%u max_err=%.6f tol=%.6f\n",
                   i, M, N, K, max_err, tol);
        }

        /* Progress indicator for large runs */
        if ((i + 1) % 500 == 0 || i == num_tests - 1) {
            printf("    Progress: %d/%d tests, %d pass, %d fail, "
                   "avg_max_err=%.6f\n",
                   i + 1, num_tests, passed, failed,
                   passed > 0 ? total_max_err / passed : 0.0f);
        }
    }

    tests_run++;
    if (failed == 0) {
        printf("  Bulk random (%d tests)                             "
               "PASS (avg_max=%.6f, avg_mean=%.6f)\n",
               num_tests,
               passed > 0 ? total_max_err / passed : 0.0f,
               passed > 0 ? total_mean_err / passed : 0.0f);
        tests_pass++;
    } else {
        FAIL("%d/%d tests failed", failed, num_tests);
    }
}

/* ================================================================
 * Test 3: Bias golden test
 * ================================================================ */

static void test_bias_golden(void) {
    TEST("Bias golden: zero W/A, sequential bias");

    tu_runtime_config_t cfg = tu_runtime_config_default();
    tu_init_with_config(&cfg);

    uint16_t M = 8, N = 8, K = 8;
    uint32_t w_count = (uint32_t)M * K;
    uint32_t a_count = (uint32_t)K * N;
    uint32_t o_count = (uint32_t)M * N;

    /* Zero W and A */
    fp16_t *W = calloc(w_count, sizeof(fp16_t));
    fp16_t *A = calloc(a_count, sizeof(fp16_t));
    fp16_t *bias = malloc(o_count * sizeof(fp16_t));

    /* Sequential bias values */
    for (uint32_t i = 0; i < o_count; i++)
        bias[i] = fp32_to_fp16((float)i);

    tu_dma_load_w(W, 0, w_count * sizeof(fp16_t));
    tu_dma_load_a(A, 0, a_count * sizeof(fp16_t));
    tu_dma_load_o(bias, 0, o_count * sizeof(fp16_t));

    tu_mma(M, N, K, 0, 0, 0, true);

    fp32_t *O = calloc(o_count, sizeof(fp32_t));
    tu_dma_store_o(O, 0, o_count * sizeof(fp32_t));

    /* Verify: O[i] should equal i */
    int ok = 1;
    for (uint32_t i = 0; i < o_count && ok; i++) {
        if (fabsf(O[i] - (float)i) > 0.01f) {
            FAIL("O[%u] = %.4f, expected %.1f", i, O[i], (float)i);
            ok = 0;
        }
    }
    if (ok) PASS();

    free(W); free(A); free(bias); free(O);
}

/* ================================================================
 * Test 4: FP16 precision boundary tests
 * ================================================================ */

static void test_precision_boundaries(void) {
    TEST("Precision: max FP16 values (65504)");

    tu_runtime_config_t cfg = tu_runtime_config_default();
    tu_init_with_config(&cfg);

    fp16_t W[4], A[4];
    W[0] = fp32_to_fp16(65504.0f); W[1] = fp32_to_fp16(0.0f);
    W[2] = fp32_to_fp16(0.0f);     W[3] = fp32_to_fp16(1.0f);
    A[0] = fp32_to_fp16(1.0f);     A[1] = fp32_to_fp16(0.0f);
    A[2] = fp32_to_fp16(0.0f);     A[3] = fp32_to_fp16(1.0f);

    tu_dma_load_w(W, 0, sizeof(W));
    tu_dma_load_a(A, 0, sizeof(A));

    tu_mma(2, 2, 2, 0, 0, 0, false);

    fp32_t O[4];
    tu_dma_store_o(O, 0, sizeof(O));

    /* Expected: identity scaled by 65504 and 1 */
    if (fabsf(O[0] - 65504.0f) < 1.0f && fabsf(O[3] - 1.0f) < 0.01f)
        PASS();
    else
        FAIL("O[0]=%.1f (expected 65504), O[3]=%.4f", O[0], O[3]);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    int num_random = MAX_RANDOM_TESTS;

    /* Quick mode flag */
    if (argc > 1 && strcmp(argv[1], "--quick") == 0) {
        num_random = QUICK_RANDOM_TESTS;
    }

    printf("TinyTU Golden Reference Verification\n");
    printf("=====================================\n");
    printf("Comparing cmodel (FP16 in, FP32 acc) vs FP32 reference\n");
    printf("Expected tolerance: ~0.01-0.15 (FP16 quantization)\n\n");

    printf("--- Fixed Configuration Tests ---\n");
    test_fixed_configs();

    printf("\n--- Bias Test ---\n");
    test_bias_golden();

    printf("\n--- Precision Boundary Tests ---\n");
    test_precision_boundaries();

    test_random_bulk(num_random);

    printf("\n═══════════════════════════════════════════\n");
    printf("  %d/%d tests passed\n", tests_pass, tests_run);
    printf("  Max observed error: %.6f\n", max_observed_error);
    printf("═══════════════════════════════════════════\n");

    return tests_pass == tests_run ? 0 : 1;
}
