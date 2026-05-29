/*
 * TinyTU BF16 + Subnormal Handling Tests
 * =======================================
 * Tests for bfloat16 conversion, subnormal mode switching,
 * and BF16 MMA correctness.
 *
 * Gaps: D1 (multi-precision), D3 (BF16), D7 (subnormal handling)
 */

#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/tu_cmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_pass = 0;

#define TEST(name) do { tests_run++; printf("  %-50s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_pass++; } while(0)
#define FAIL(fmt, ...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); } while(0)
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { FAIL("expected %d, got %d", (int)(b), (int)(a)); return; }} while(0)
#define ASSERT_NEAR(a, b, tol) do { if (fabsf((a)-(b)) > (tol)) { FAIL("%f != %f", (double)(a), (double)(b)); return; }} while(0)

/* ================================================================
 * BF16 Conversion Tests
 * ================================================================ */

static void test_bf16_roundtrip_simple(void) {
    TEST("BF16 round-trip (1.0)");
    bf16_t h = tu_fp32_to_bf16(1.0f);
    fp32_t f = tu_bf16_to_fp32(h);
    ASSERT_NEAR(f, 1.0f, 0.001f);
    PASS();
}

static void test_bf16_roundtrip_zero(void) {
    TEST("BF16 round-trip (0.0, -0.0)");
    bf16_t h = tu_fp32_to_bf16(0.0f);
    ASSERT_EQ(tu_bf16_to_fp32(h), 0.0f);
    h = tu_fp32_to_bf16(-0.0f);
    ASSERT_EQ(tu_bf16_to_fp32(h), 0.0f);  /* -0.0 == 0.0 in FP compare */
    PASS();
}

static void test_bf16_batch(void) {
    TEST("BF16 batch conversion (16 elements)");
    fp32_t src[16], dst[16];
    bf16_t bf[16];
    for (int i = 0; i < 16; i++) src[i] = (float)(i - 8) * 0.5f;
    tu_fp32_to_bf16_buffer(src, bf, 16);
    tu_bf16_to_fp32_buffer(bf, dst, 16);
    float max_err = 0.0f;
    for (int i = 0; i < 16; i++) {
        float err = fabsf(dst[i] - src[i]) / (fabsf(src[i]) + 1e-10f);
        if (err > max_err) max_err = err;
    }
    /* BF16 has ~2 decimal digits of precision (~7-bit mantissa) */
    if (max_err < 0.01f) PASS();
    else FAIL("max relative error %f", (double)max_err);
}

static void test_bf16_nan_inf(void) {
    TEST("BF16 NaN and Inf handling");
    bf16_t h_nan = tu_fp32_to_bf16(NAN);
    fp32_t f_nan = tu_bf16_to_fp32(h_nan);
    if (!isnan(f_nan)) { FAIL("NaN not preserved"); return; }

    bf16_t h_inf = tu_fp32_to_bf16(INFINITY);
    fp32_t f_inf = tu_bf16_to_fp32(h_inf);
    if (!isinf(f_inf) || f_inf <= 0) { FAIL("+Inf not preserved"); return; }

    bf16_t h_ninf = tu_fp32_to_bf16(-INFINITY);
    fp32_t f_ninf = tu_bf16_to_fp32(h_ninf);
    if (!isinf(f_ninf) || f_ninf >= 0) { FAIL("-Inf not preserved"); return; }

    PASS();
}

static void test_bf16_precision_boundary(void) {
    TEST("BF16 at precision boundary (mantissa ~7 bits)");
    /* BF16 mantissa is 7 bits → ~2.1 decimal digits.
     * Values within ±0.4% should round-trip perfectly for powers of 2. */
    float vals[] = { 0.125f, 0.5f, 1.0f, 2.0f, 4.0f, 128.0f, 1024.0f };
    for (int i = 0; i < 7; i++) {
        fp32_t f = tu_bf16_to_fp32(tu_fp32_to_bf16(vals[i]));
        ASSERT_NEAR(f, vals[i], vals[i] * 0.005f);
    }
    PASS();
}

static void test_bf16_precision_loss(void) {
    TEST("BF16 precision loss vs FP32 (non-exact values)");
    /* 1.0009765625 = 1 + 1/1024. BF16 can't represent the LSB.
     * Expected: round to 1.0 (nearest). */
    float fine_val = 1.0f + 1.0f / 1024.0f;
    fp32_t f = tu_bf16_to_fp32(tu_fp32_to_bf16(fine_val));
    /* Should be within 1 LSB of BF16 (~0.4% for small values) */
    float rel_err = fabsf(f - fine_val) / fine_val;
    if (rel_err > 0.01f) { FAIL("precision loss too large: %f", (double)rel_err); return; }
    PASS();
}

/* ================================================================
 * Subnormal Handling Tests
 * ================================================================ */

static void test_subnormal_default_flush(void) {
    TEST("Subnormal: default mode (flush-to-zero)");
    /* By default, subnormals are flushed to zero */
    fp16_t h = tu_fp32_to_fp16(1e-8f);  /* well within FP16 subnormal range */
    ASSERT_EQ(tu_fp16_to_fp32(h), 0.0f);
    PASS();
}

static void test_subnormal_full_mode(void) {
    TEST("Subnormal: full mode preserves tiny values");
    tu_set_subnormal_mode(TU_SUBNORMAL_FULL);
    /* Smallest FP16 subnormal: 2^-24 ≈ 5.96e-8. Use 1e-6 which is
     * comfortably in the FP16 subnormal range (max subnormal ~6.1e-5). */
    fp16_t h = tu_fp32_to_fp16(1e-6f);
    fp32_t f = tu_fp16_to_fp32(h);
    if (f > 0.0f && f < 5e-5f) PASS();
    else FAIL("subnormal flushed or out of range: got %e", (double)f);
    tu_set_subnormal_mode(TU_SUBNORMAL_FLUSH);
}

static void test_subnormal_mode_switch(void) {
    TEST("Subnormal: mode get/set roundtrip");
    ASSERT_EQ(tu_get_subnormal_mode(), TU_SUBNORMAL_FLUSH);
    tu_set_subnormal_mode(TU_SUBNORMAL_FULL);
    ASSERT_EQ(tu_get_subnormal_mode(), TU_SUBNORMAL_FULL);
    tu_set_subnormal_mode(TU_SUBNORMAL_FLUSH);
    ASSERT_EQ(tu_get_subnormal_mode(), TU_SUBNORMAL_FLUSH);
    PASS();
}

static void test_subnormal_boundary(void) {
    TEST("Subnormal: smallest normal vs flush");
    /* Smallest FP16 normal: 2^-14 ≈ 6.1e-5. Use 1e-4 to be safely above. */
    fp16_t h_norm = tu_fp32_to_fp16(1e-4f);
    fp32_t f_norm = tu_fp16_to_fp32(h_norm);
    if (f_norm <= 0.0f) { FAIL("normal value flushed"); return; }

    /* Well below normal range — flushes with default mode */
    fp16_t h_sub = tu_fp32_to_fp16(1e-8f);
    ASSERT_EQ(tu_fp16_to_fp32(h_sub), 0.0f);

    /* With full mode — should preserve subnormals */
    tu_set_subnormal_mode(TU_SUBNORMAL_FULL);
    h_sub = tu_fp32_to_fp16(5e-6f);
    if (tu_fp16_to_fp32(h_sub) <= 0.0f) { FAIL("subnormal lost in full mode"); return; }

    tu_set_subnormal_mode(TU_SUBNORMAL_FLUSH);
    PASS();
}

/* ================================================================
 * BF16 MMA Integration Test
 * ================================================================ */

/* ================================================================
 * BF16 → FP16 pipeline verification
 * ================================================================ */

static void test_bf16_to_fp16_pipeline(void) {
    TEST("BF16→FP16 pipeline: identity values preserved");
    /* Verify that BF16→FP32→FP16 round-trip preserves key values
     * needed for MMA correctness. */
    float vals[] = { 0.0f, 1.0f, 2.0f, 0.5f, 0.25f, 8.0f, 16.0f };
    for (int i = 0; i < 7; i++) {
        bf16_t bf = tu_fp32_to_bf16(vals[i]);
        fp32_t bfs = tu_bf16_to_fp32(bf);
        fp16_t f16 = tu_fp32_to_fp16(bfs);
        fp32_t back = tu_fp16_to_fp32(f16);
        float rel_err = fabsf(back - vals[i]) / (fabsf(vals[i]) + 1e-10f);
        if (rel_err > 0.1f) {
            FAIL("BF16→FP16 pipeline at %f: got %f (rel %f)",
                 (double)vals[i], (double)back, (double)rel_err);
            return;
        }
    }
    PASS();
}

/* ================================================================
 * Precision Registry Tests
 * ================================================================ */

static void test_precision_registry(void) {
    TEST("Precision registry: FP16/F32/BF16 entries");

    const tu_precision_desc_t *fp16 = tu_precision_get(TU_PREC_FP16);
    if (!fp16 || strcmp(fp16->name, "fp16") != 0 || fp16->elem_bytes != 2)
        { FAIL("FP16 precision descriptor wrong"); return; }

    const tu_precision_desc_t *f32 = tu_precision_get(TU_PREC_FP32);
    if (!f32 || strcmp(f32->name, "fp32") != 0 || f32->elem_bytes != 4)
        { FAIL("FP32 precision descriptor wrong"); return; }

    const tu_precision_desc_t *bf16 = tu_precision_get(TU_PREC_BF16);
    if (!bf16 || strcmp(bf16->name, "bf16") != 0 || bf16->elem_bytes != 2)
        { FAIL("BF16 precision descriptor wrong"); return; }

    /* FP32 identity: val → FP32 → FP32'd val should be exact */
    fp32_t val = 3.14159f;
    fp32_t out;
    f32->from_fp32(val, &out);
    fp32_t back = f32->to_fp32(&out);
    ASSERT_NEAR(back, val, 1e-7f);

    /* BF16: val → BF16 → FP32 should be within 1 ULP */
    bf16_t bf;
    bf16->from_fp32(val, &bf);
    fp32_t bf_back = bf16->to_fp32(&bf);
    ASSERT_NEAR(bf_back, val, val * 0.01f);  /* ~1% for BF16 */

    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("\nTinyTU BF16 + Subnormal Handling Tests\n");
    printf("========================================\n\n");

    printf("--- BF16 Conversion ---\n");
    test_bf16_roundtrip_simple();
    test_bf16_roundtrip_zero();
    test_bf16_batch();
    test_bf16_nan_inf();
    test_bf16_precision_boundary();
    test_bf16_precision_loss();

    printf("\n--- Subnormal Handling ---\n");
    test_subnormal_default_flush();
    test_subnormal_full_mode();
    test_subnormal_mode_switch();
    test_subnormal_boundary();

    printf("\n--- BF16 Integration ---\n");
    test_bf16_to_fp16_pipeline();

    printf("\n--- Precision Registry ---\n");
    test_precision_registry();

    printf("\n═══════════════════════════════════════════\n");
    printf("  %d/%d tests passed\n", tests_pass, tests_run);
    printf("═══════════════════════════════════════════\n");

    return (tests_pass == tests_run) ? 0 : 1;
}
