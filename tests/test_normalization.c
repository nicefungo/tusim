/*
 * Normalization Engine Unit Tests
 * =================================
 * Tests for: LayerNorm, RMSNorm, numerical stability,
 * per-row normalization, edge cases.
 */

#include "tu_cmodel/tu_config.h"
#include "tu_cmodel/tu_sram.h"
#include "tu_cmodel/compute/normalization_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int passed = 0, failed = 0;

#define TEST(name) do { \
    printf("  %-55s", name); \
    fflush(stdout); \
} while(0)

#define PASS() do { printf("PASS\n"); passed++; } while(0)
#define FAIL(msg) do { printf("FAIL — %s\n", msg); failed++; } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define CHECK_FLOAT(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        printf("FAIL — %s (got %e, expected %e, diff=%e)\n", msg, a, b, fabsf((a)-(b))); \
        failed++; return; \
    } \
} while(0)

/* Helper: read FP32 element from SRAM */
static float sram_get_f32(tu_sram_region_t *sram, uint32_t byte_offset) {
    float val;
    tu_sram_read(sram, byte_offset, &val);
    return val;
}

/* Helper: write FP32 element to SRAM */
static void sram_put_f32(tu_sram_region_t *sram, uint32_t byte_offset, float val) {
    tu_sram_write(sram, byte_offset, &val);
}

/* ================================================================
 * Test 1: LayerNorm — identity (already normalized data)
 * ================================================================ */
static void test_layernorm_identity(void) {
    TEST("LayerNorm — identity (μ≈0, σ≈1)");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    /* Pre-normalized data: should stay close to original */
    float input[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 1.5f, -1.5f, 0.0f};
    uint32_t n = sizeof(input) / sizeof(float);

    for (uint32_t i = 0; i < n; i++)
        sram_put_f32(&sram, i * sizeof(float), input[i]);

    uint64_t stall = tu_layernorm(&sram, 0, n, NULL, NULL, 1e-5f, true);
    (void)stall;

    /* After LayerNorm: mean ≈ 0, std ≈ 1 */
    double sum = 0, sum2 = 0;
    for (uint32_t i = 0; i < n; i++) {
        float v = sram_get_f32(&sram, i * sizeof(float));
        sum += v;
        sum2 += v * v;
    }
    double mean = sum / n;
    double var  = sum2 / n - mean * mean;

    CHECK(fabs(mean) < 1e-4, "mean not zero after LayerNorm");
    CHECK(fabs(sqrt(var) - 1.0) < 0.2, "std not near 1.0 after LayerNorm");

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Test 2: LayerNorm — constant input → all zeros
 * ================================================================ */
static void test_layernorm_constant(void) {
    TEST("LayerNorm — constant input (all 3.0)");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    float input[] = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    uint32_t n = sizeof(input) / sizeof(float);

    for (uint32_t i = 0; i < n; i++)
        sram_put_f32(&sram, i * sizeof(float), input[i]);

    uint64_t stall = tu_layernorm(&sram, 0, n, NULL, NULL, 1e-5f, true);
    (void)stall;

    /* All values should be near zero (variance is zero, so output = 0) */
    for (uint32_t i = 0; i < n; i++) {
        float v = sram_get_f32(&sram, i * sizeof(float));
        CHECK(fabs(v) < 1e-4, "constant input should produce zero output");
    }

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Test 3: RMSNorm — basic correctness
 * ================================================================ */
static void test_rmsnorm_basic(void) {
    TEST("RMSNorm — basic correctness");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    /* RMS = sqrt(mean of squares) = sqrt((4+9+16+25)/4) = sqrt(13.5) ≈ 3.674 */
    float input[] = {2.0f, 3.0f, 4.0f, 5.0f};
    uint32_t n = sizeof(input) / sizeof(float);

    for (uint32_t i = 0; i < n; i++)
        sram_put_f32(&sram, i * sizeof(float), input[i]);

    uint64_t stall = tu_rmsnorm(&sram, 0, n, NULL, 1e-5f, true);
    (void)stall;

    /* Normalized: each element divided by RMS */
    float expected_rms = sqrtf((4+9+16+25)/4.0f);
    for (uint32_t i = 0; i < n; i++) {
        float v = sram_get_f32(&sram, i * sizeof(float));
        float exp = input[i] / expected_rms;
        CHECK_FLOAT(v, exp, 1e-4f, "RMSNorm output mismatch");
    }

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Test 4: LayerNorm with gamma (scale)
 * ================================================================ */
static void test_layernorm_with_gamma(void) {
    TEST("LayerNorm with gamma=2.0");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    float input[] = {-1.0f, 0.0f, 1.0f};
    float gamma[] = {2.0f, 2.0f, 2.0f};
    uint32_t n = 3;

    for (uint32_t i = 0; i < n; i++)
        sram_put_f32(&sram, i * sizeof(float), input[i]);

    uint64_t stall = tu_layernorm(&sram, 0, n, gamma, NULL, 1e-5f, true);
    (void)stall;

    /* After LayerNorm with gamma=2: std should be ~2 */
    double sum2 = 0;
    for (uint32_t i = 0; i < n; i++) {
        float v = sram_get_f32(&sram, i * sizeof(float));
        sum2 += v * v;
    }
    double rms = sqrt(sum2 / n);
    CHECK(fabs(rms - 2.0) < 0.5, "std not scaled by gamma=2");

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Test 5: LayerNorm with beta (bias)
 * ================================================================ */
static void test_layernorm_with_beta(void) {
    TEST("LayerNorm with beta offset");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    float input[] = {-1.0f, 0.0f, 1.0f};
    float beta[]  = {10.0f, 20.0f, 30.0f};
    uint32_t n = 3;

    for (uint32_t i = 0; i < n; i++)
        sram_put_f32(&sram, i * sizeof(float), input[i]);

    uint64_t stall = tu_layernorm(&sram, 0, n, NULL, beta, 1e-5f, true);
    (void)stall;

    /* Values should include beta offset */
    float sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        float v = sram_get_f32(&sram, i * sizeof(float));
        sum += v;
    }
    /* Mean should be (10+20+30)/3 = 20 since normalized input has mean 0 */
    double mean = sum / n;
    CHECK(fabs(mean - 20.0) < 1.0, "mean not shifted by beta");

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Test 6: Per-row LayerNorm (2D)
 * ================================================================ */
static void test_layernorm_2d(void) {
    TEST("Per-row LayerNorm (3 rows × 4 cols)");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    /* 3 rows, 4 cols */
    float data[] = {
        1, 2, 3, 4,     /* row 0 */
        5, 6, 7, 8,     /* row 1 */
        1, 1, 1, 1,     /* row 2 — constant */
    };
    uint32_t rows = 3, cols = 4;

    for (uint32_t i = 0; i < rows * cols; i++)
        sram_put_f32(&sram, i * sizeof(float), data[i]);

    uint64_t stall = tu_layernorm_2d(&sram, 0, rows, cols, NULL, NULL, 1e-5f);
    (void)stall;

    /* Check row 0: normalized */
    double sum_r0 = 0, sum2_r0 = 0;
    for (uint32_t c = 0; c < cols; c++) {
        float v = sram_get_f32(&sram, c * sizeof(float));
        sum_r0 += v; sum2_r0 += v * v;
    }
    double mean_r0 = sum_r0 / cols;
    CHECK(fabs(mean_r0) < 1e-4, "row 0 mean not zero");

    /* Row 2 (constant): should be all zeros */
    for (uint32_t c = 0; c < cols; c++) {
        float v = sram_get_f32(&sram, (2 * cols + c) * sizeof(float));
        CHECK(fabs(v) < 1e-4, "constant row should be zero");
    }

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Test 7: Per-row RMSNorm (2D)
 * ================================================================ */
static void test_rmsnorm_2d(void) {
    TEST("Per-row RMSNorm (2 rows × 3 cols)");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    float data[] = {2, 2, 2,     /* row 0: RMS = 2, normalized = 1 */
                    1, 2, 3};    /* row 1: RMS = sqrt(14/3) ≈ 2.16 */
    uint32_t rows = 2, cols = 3;

    for (uint32_t i = 0; i < rows * cols; i++)
        sram_put_f32(&sram, i * sizeof(float), data[i]);

    uint64_t stall = tu_rmsnorm_2d(&sram, 0, rows, cols, NULL, 1e-5f);
    (void)stall;

    /* Row 0: all ≈ 1.0 */
    for (uint32_t c = 0; c < cols; c++) {
        float v = sram_get_f32(&sram, c * sizeof(float));
        CHECK_FLOAT(v, 1.0f, 1e-4f, "RMSNorm row 0 not 1.0");
    }

    /* Row 1: RMS = 1.0 (normalized) */
    double sum2_r1 = 0;
    for (uint32_t c = 0; c < cols; c++) {
        float v = sram_get_f32(&sram, (cols + c) * sizeof(float));
        sum2_r1 += v * v;
    }
    double rms_r1 = sqrt(sum2_r1 / cols);
    CHECK(fabs(rms_r1 - 1.0) < 1e-4, "row 1 RMS not 1.0 after normalization");

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Test 8: Single element (edge case)
 * ================================================================ */
static void test_single_element(void) {
    TEST("Single-element LayerNorm");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    sram_put_f32(&sram, 0, 42.0f);
    uint64_t stall = tu_layernorm(&sram, 0, 1, NULL, NULL, 1e-5f, true);
    (void)stall;

    /* Single element: (42 - 42) / sqrt(0 + ε) ≈ 0 */
    float v = sram_get_f32(&sram, 0);
    CHECK(fabs(v) < 1e-4, "single element not zero");

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Test 9: RMSNorm single element
 * ================================================================ */
static void test_rmsnorm_single(void) {
    TEST("Single-element RMSNorm");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    sram_put_f32(&sram, 0, 5.0f);
    uint64_t stall = tu_rmsnorm(&sram, 0, 1, NULL, 1e-5f, true);
    (void)stall;

    /* Single element: 5 / sqrt(25 + ε) ≈ 1.0 */
    float v = sram_get_f32(&sram, 0);
    CHECK_FLOAT(v, 1.0f, 1e-4f, "single element RMSNorm not 1.0");

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Test 10: Validation — null descriptor
 * ================================================================ */
static void test_validate_null(void) {
    TEST("Validate null descriptor");

    bool ok = tu_norm_validate_desc(NULL);
    CHECK(!ok, "should reject null descriptor");
    PASS();
}

/* ================================================================
 * Test 11: Validation — zero elements
 * ================================================================ */
static void test_validate_zero_elem(void) {
    TEST("Validate zero element count");

    tu_sram_region_t sram;
    tu_sram_init(&sram, 256, "test");

    tu_norm_desc_t desc = {
        .mode = TU_NORM_LAYER_NORM,
        .data_sram = &sram,
        .data_offset = 0,
        .elem_count = 0,
        .epsilon = 1e-5f,
    };

    bool ok = tu_norm_validate_desc(&desc);
    CHECK(!ok, "should reject zero elem_count");

    tu_sram_destroy(&sram);
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
    printf("TinyTU Normalization Engine — Unit Tests\n");
    printf("==========================================\n\n");

    test_layernorm_identity();
    test_layernorm_constant();
    test_rmsnorm_basic();
    test_layernorm_with_gamma();
    test_layernorm_with_beta();
    test_layernorm_2d();
    test_rmsnorm_2d();
    test_single_element();
    test_rmsnorm_single();
    test_validate_null();
    test_validate_zero_elem();

    printf("\n═══════════════════════════════════════════\n");
    printf("  %d/%d tests passed\n", passed, passed + failed);
    printf("═══════════════════════════════════════════\n");

    return failed > 0 ? 1 : 0;
}
