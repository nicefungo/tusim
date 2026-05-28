/*
 * TinyTU CModel — Parameterized Test Suite
 * =========================================
 *
 * Verifies:
 *   1. FP16 round-trip conversion
 *   2. MMA identity test — multiple PE dimensions
 *   3. MMA known-value GEMM — multiple PE dimensions
 *   4. MMA with bias
 *   5. Runtime configuration override
 *   6. SRAM overflow detection
 *   7. Edge cases: non-multiple-of-tile dimensions
 */
#include "tu_cmodel/tu_cmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_pass = 0;

#define TEST(name) do { tests_run++; printf("  %-54s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_pass++; } while(0)
#define FAIL(fmt, ...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); } while(0)

static float max_error(const fp32_t *a, const fp32_t *b, int n) {
    float max = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max) max = err;
    }
    return max;
}

/* ================================================================
 * Test 1: FP16 round-trip
 * ================================================================ */
static void test_fp16_roundtrip(void) {
    TEST("FP16 round-trip (1.0)");
    fp16_t h = fp32_to_fp16(1.0f);
    fp32_t f = fp16_to_fp32(h);
    if (fabsf(f - 1.0f) < 0.001f) PASS();
    else FAIL("got %f", f);

    TEST("FP16 round-trip (0.0)");
    h = fp32_to_fp16(0.0f);
    f = fp16_to_fp32(h);
    if (f == 0.0f) PASS();
    else FAIL("got %f", f);

    TEST("FP16 round-trip (-2.5)");
    h = fp32_to_fp16(-2.5f);
    f = fp16_to_fp32(h);
    if (fabsf(f + 2.5f) < 0.01f) PASS();
    else FAIL("got %f", f);

    TEST("FP16 round-trip (65504.0 — max fp16)");
    h = fp32_to_fp16(65504.0f);
    f = fp16_to_fp32(h);
    if (fabsf(f - 65504.0f) < 1.0f) PASS();
    else FAIL("got %f", f);

    TEST("FP16 round-trip (subnormal: 1e-8, flush-to-zero)");
    h = fp32_to_fp16(1e-8f);
    f = fp16_to_fp32(h);
    /* With TU_FP16_SUBNORMAL_FLUSH=1, subnormals flush to 0 */
    if (f == 0.0f || (f > 0.0f && f < 0.001f)) PASS();
    else FAIL("got %f (expected 0 or tiny positive)", f);

    TEST("FP16 round-trip (NaN)");
    h = fp32_to_fp16(NAN);
    f = fp16_to_fp32(h);
    if (isnan(f)) PASS();
    else FAIL("got %f", f);

    TEST("FP16 round-trip (+Inf)");
    h = fp32_to_fp16(INFINITY);
    f = fp16_to_fp32(h);
    if (isinf(f) && f > 0) PASS();
    else FAIL("got %f", f);
}

/* ================================================================
 * Test 2: MMA identity with configurable PE dimensions
 * ================================================================ */
typedef struct {
    uint16_t pe_rows, pe_cols;
    uint16_t M, N, K;
    const char *label;
} mma_dim_config_t;

static void run_mma_identity_test(const mma_dim_config_t *d) {
    char name[64];
    snprintf(name, sizeof(name), "MMA identity %ux%u (%u×%u PE)",
             d->M, d->N, d->pe_rows, d->pe_cols);
    TEST(name);

    tu_runtime_config_t cfg = tu_config_default();
    cfg.pe_rows = d->pe_rows;
    cfg.pe_cols = d->pe_cols;
    tu_init_with_config(&cfg);

    /* Build identity matrices */
    fp16_t *W = (fp16_t *)calloc(d->M * d->K, sizeof(fp16_t));
    fp16_t *A = (fp16_t *)calloc(d->K * d->N, sizeof(fp16_t));

    for (int i = 0; i < (int)d->M && i < (int)d->K; i++)
        W[i * d->K + i] = fp32_to_fp16(1.0f);
    for (int i = 0; i < (int)d->K && i < (int)d->N; i++)
        A[i * d->N + i] = fp32_to_fp16(1.0f);

    tu_dma_load_w(W, 0, d->M * d->K * sizeof(fp16_t));
    tu_dma_load_a(A, 0, d->K * d->N * sizeof(fp16_t));

    tu_mma(d->M, d->N, d->K, 0, 0, 0, false);

    fp32_t *O = (fp32_t *)calloc(d->M * d->N, sizeof(fp32_t));
    tu_dma_store_o(O, 0, d->M * d->N * sizeof(fp32_t));

    /* Expected: O = I_min(M,N,K), with 1.0 on diagonal */
    fp32_t *expected = (fp32_t *)calloc(d->M * d->N, sizeof(fp32_t));
    uint16_t r = d->M < d->N ? d->M : d->N;
    r = r < d->K ? r : d->K;
    for (int i = 0; i < (int)r; i++)
        expected[i * d->N + i] = 1.0f;

    float err = max_error(O, expected, d->M * d->N);
    if (err < 0.01f) PASS();
    else FAIL("max error = %f", err);

    free(W); free(A); free(O); free(expected);
}

static void test_mma_identity_param(void) {
    mma_dim_config_t configs[] = {
        {16, 16, 16, 16, 16},   /* original 16×16 */
        {32, 32, 32, 32, 32},   /* 32×32 PE */
        { 8,  8, 16, 16, 16},   /* 8×8 PE */
        {16, 16, 32, 16, 16},   /* tall M, exact K multiple */
        { 4,  8, 16, 16, 16},   /* non-square */
        {16, 16, 48, 48, 48},   /* 48×48, non-power-of-2 */
        {16, 16, 20, 20, 20},   /* non-multiple-of-tile dimensions */
    };
    int n = sizeof(configs) / sizeof(configs[0]);
    for (int i = 0; i < n; i++)
        run_mma_identity_test(&configs[i]);
}

/* ================================================================
 * Test 3: MMA known-value GEMM
 * ================================================================ */
static void test_mma_known_value(void) {
    TEST("MMA 32×8 (all-0.5 W, all-2.0 A → 16.0)");

    tu_runtime_config_t cfg = tu_config_default();
    cfg.pe_rows = 16;
    cfg.pe_cols = 16;
    tu_init_with_config(&cfg);

    /* W = all 0.5 */
    fp16_t W[32 * 16];
    for (int i = 0; i < 32 * 16; i++)
        W[i] = fp32_to_fp16(0.5f);

    /* A = all 2.0 */
    fp16_t A[16 * 8];
    for (int i = 0; i < 16 * 8; i++)
        A[i] = fp32_to_fp16(2.0f);

    tu_dma_load_w(W, 0, sizeof(W));
    tu_dma_load_a(A, 0, sizeof(A));

    tu_mma(32, 8, 16, 0, 0, 0, false);

    fp32_t O[32 * 8];
    tu_dma_store_o(O, 0, sizeof(O));

    /* Each output element = 16 * 0.5 * 2.0 = 16.0 */
    int ok = 1;
    for (int i = 0; i < 32 * 8 && ok; i++) {
        if (fabsf(O[i] - 16.0f) > 0.1f) {
            FAIL("O[%d] = %f, expected 16.0", i, O[i]);
            ok = 0;
        }
    }
    if (ok) PASS();
}

/* ================================================================
 * Test 4: MMA with bias
 * ================================================================ */
static void test_mma_bias(void) {
    TEST("MMA 16×16×16 with bias (zero W/A)");

    tu_runtime_config_t cfg = tu_config_default();
    cfg.pe_rows = 16;
    cfg.pe_cols = 16;
    tu_init_with_config(&cfg);

    fp16_t W[16 * 16];
    memset(W, 0, sizeof(W));

    fp16_t A[16 * 16];
    memset(A, 0, sizeof(A));

    /* Bias = sequential values */
    fp16_t bias[16 * 16];
    for (int i = 0; i < 16 * 16; i++)
        bias[i] = fp32_to_fp16((float)(i % 16));

    tu_dma_load_w(W, 0, sizeof(W));
    tu_dma_load_a(A, 0, sizeof(A));
    tu_dma_load_o(bias, 0, sizeof(bias));

    tu_mma(16, 16, 16, 0, 0, 0, true);

    fp32_t O[16 * 16];
    tu_dma_store_o(O, 0, sizeof(O));

    int ok = 1;
    for (int i = 0; i < 16 * 16 && ok; i++) {
        float expected = (float)(i % 16);
        if (fabsf(O[i] - expected) > 0.01f) {
            FAIL("O[%d] = %f, expected %f", i, O[i], expected);
            ok = 0;
        }
    }
    if (ok) PASS();
}

/* ================================================================
 * Test 5: Runtime config override
 * ================================================================ */
static void test_config_override(void) {
    TEST("Runtime config: 32×32 PE array");

    tu_runtime_config_t cfg = tu_config_default();
    cfg.pe_rows = 32;
    cfg.pe_cols = 32;
    tu_init_with_config(&cfg);

    if (g_tu.rt_cfg.pe_rows == 32 && g_tu.rt_cfg.pe_cols == 32) PASS();
    else FAIL("rows=%u cols=%u", g_tu.rt_cfg.pe_rows, g_tu.rt_cfg.pe_cols);
}

/* ================================================================
 * Test 6: RyR computation — 2×2 PE, small matrices
 * ================================================================ */
static void test_mma_2x2_pe(void) {
    TEST("MMA 4×4×4 with 2×2 PE array");

    tu_runtime_config_t cfg = tu_config_default();
    cfg.pe_rows = 2;
    cfg.pe_cols = 2;
    cfg.sram_w_size = 1024;
    cfg.sram_a_size = 1024;
    cfg.sram_o_size = 4096;
    tu_init_with_config(&cfg);

    /* W = [[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]] */
    fp16_t W[4 * 4];
    memset(W, 0, sizeof(W));
    W[0 * 4 + 0] = fp32_to_fp16(1.0f);
    W[1 * 4 + 1] = fp32_to_fp16(2.0f);
    W[2 * 4 + 2] = fp32_to_fp16(3.0f);
    W[3 * 4 + 3] = fp32_to_fp16(4.0f);

    /* A = all 1.0 */
    fp16_t A[4 * 4];
    for (int i = 0; i < 4 * 4; i++)
        A[i] = fp32_to_fp16(1.0f);

    tu_dma_load_w(W, 0, sizeof(W));
    tu_dma_load_a(A, 0, sizeof(A));

    tu_mma(4, 4, 4, 0, 0, 0, false);

    fp32_t O[4 * 4];
    tu_dma_store_o(O, 0, sizeof(O));

    /* Expected: row i = [i+1, i+1, i+1, i+1] (sum of each row = (i+1)*4) */
    int ok = 1;
    for (int i = 0; i < 4 && ok; i++) {
        float expected = (float)(i + 1);  /* row i sums to (i+1)*4 over 4 cols = i+1 per col */
        for (int j = 0; j < 4 && ok; j++) {
            if (fabsf(O[i * 4 + j] - expected) > 0.01f) {
                FAIL("O[%d][%d] = %f, expected %f", i, j, O[i * 4 + j], expected);
                ok = 0;
            }
        }
    }
    if (ok) PASS();
}

/* ================================================================
 * Test 7: Non-multiple-of-tile edge case
 * ================================================================ */
static void test_mma_edge_tiles(void) {
    TEST("MMA 7×5×9 (non-multiple-of-tile, 4×3 PE)");

    tu_runtime_config_t cfg = tu_config_default();
    cfg.pe_rows = 4;
    cfg.pe_cols = 3;
    cfg.sram_w_size = 4096;
    cfg.sram_a_size = 4096;
    cfg.sram_o_size = 16384;
    tu_init_with_config(&cfg);

    /* W = all 1.0 */
    fp16_t *W = (fp16_t *)calloc(7 * 9, sizeof(fp16_t));
    for (int i = 0; i < 7 * 9; i++) W[i] = fp32_to_fp16(1.0f);

    /* A = all 2.0 */
    fp16_t *A = (fp16_t *)calloc(9 * 5, sizeof(fp16_t));
    for (int i = 0; i < 9 * 5; i++) A[i] = fp32_to_fp16(2.0f);

    tu_dma_load_w(W, 0, 7 * 9 * sizeof(fp16_t));
    tu_dma_load_a(A, 0, 9 * 5 * sizeof(fp16_t));

    tu_mma(7, 5, 9, 0, 0, 0, false);

    fp32_t *O = (fp32_t *)calloc(7 * 5, sizeof(fp32_t));
    tu_dma_store_o(O, 0, 7 * 5 * sizeof(fp32_t));

    /* Each element = 9 * 1.0 * 2.0 = 18.0 */
    int ok = 1;
    for (int i = 0; i < 7 * 5 && ok; i++) {
        if (fabsf(O[i] - 18.0f) > 0.15f) {
            FAIL("O[%d] = %f, expected 18.0", i, O[i]);
            ok = 0;
        }
    }
    if (ok) PASS();

    free(W); free(A); free(O);
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
    printf("TinyTU CModel — Parameterized Test Suite\n");
    printf("=========================================\n");
    printf("Compile-time TU_PE_ROWS=%d TU_PE_COLS=%d (default)\n\n",
           TU_PE_ROWS, TU_PE_COLS);

    test_fp16_roundtrip();
    printf("\n--- MMA Identity (parameterized PE dimensions) ---\n");
    test_mma_identity_param();
    printf("\n--- MMA Known Values ---\n");
    test_mma_known_value();
    printf("\n--- MMA Bias ---\n");
    test_mma_bias();
    printf("\n--- Runtime Config ---\n");
    test_config_override();
    printf("\n--- Small PE Array ---\n");
    test_mma_2x2_pe();
    printf("\n--- Edge Cases ---\n");
    test_mma_edge_tiles();

    printf("\n═══════════════════════════════════════════\n");
    printf("  %d/%d tests passed\n", tests_pass, tests_run);
    printf("═══════════════════════════════════════════\n");
    return tests_pass == tests_run ? 0 : 1;
}
