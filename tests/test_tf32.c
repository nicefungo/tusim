/*
 * Test: TF32 TensorFloat-32 (D3)
 * ================================
 * Comprehensive tests for TF32 (1-8-10) per NVIDIA Ampere spec.
 *
 * TF32 is FP32 with 10-bit mantissa: same exponent range, reduced precision.
 * Key properties:
 *   - 1.0 roundtrips exactly (FP32 1.0 has no low mantissa bits)
 *   - Dynamic range matches FP32
 *   - Rounding modes affect the 13 dropped mantissa bits
 *   - Batch conversions are correct
 *   - Mixed precision (TF32↔FP16, TF32↔BF16) works
 *   - Precision registry integration works
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "tu_cmodel/tf32.h"
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/rounding.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tests_total = 0;
static int tests_passed = 0;

#define TEST(name) do { tests_total++; printf("  TEST %-54s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    float diff = fabsf((float)(a) - (float)(b)); \
    float denom = fabsf((float)(b)) > 1e-10f ? fabsf((float)(b)) : 1.0f; \
    if (diff / denom > (tol)) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "%s (%.6e vs %.6e, rel=%.2e)", msg, (double)(a), (double)(b), (double)(diff/denom)); \
        FAIL(buf); return; \
    } \
} while(0)

/* ================================================================
 * Basic Roundtrip Tests
 * ================================================================ */

static void test_tf32_zero(void) {
    TEST("TF32: 0.0 roundtrip");
    tf32_t enc = tu_fp32_to_tf32(0.0f);
    ASSERT_EQ(enc, 0x00000000u, "TF32 +0 encode");
    float back = tu_tf32_to_fp32(enc);
    ASSERT_NEAR(back, 0.0f, 1e-10f, "TF32 +0 decode");
    PASS();
}

static void test_tf32_neg_zero(void) {
    TEST("TF32: -0.0 roundtrip");
    tf32_t enc = tu_fp32_to_tf32(-0.0f);
    ASSERT_EQ(enc, 0x80000000u, "TF32 -0 encode");
    float back = tu_tf32_to_fp32(enc);
    ASSERT_NEAR(back, 0.0f, 1e-10f, "TF32 -0 decode");
    PASS();
}

static void test_tf32_one(void) {
    TEST("TF32: 1.0 roundtrip");
    tf32_t enc = tu_fp32_to_tf32(1.0f);
    float back = tu_tf32_to_fp32(enc);
    ASSERT_NEAR(back, 1.0f, 1e-10f, "TF32 1.0 roundtrip");
    PASS();
}

static void test_tf32_neg_one(void) {
    TEST("TF32: -1.0 roundtrip");
    tf32_t enc = tu_fp32_to_tf32(-1.0f);
    float back = tu_tf32_to_fp32(enc);
    ASSERT_NEAR(back, -1.0f, 1e-10f, "TF32 -1.0 roundtrip");
    PASS();
}

static void test_tf32_half(void) {
    TEST("TF32: 0.5 roundtrip");
    /* 0.5 = 2^-1, mantissa = 0 — should be exact */
    tf32_t enc = tu_fp32_to_tf32(0.5f);
    float back = tu_tf32_to_fp32(enc);
    ASSERT_NEAR(back, 0.5f, 1e-10f, "TF32 0.5 roundtrip");
    PASS();
}

static void test_tf32_two(void) {
    TEST("TF32: 2.0 roundtrip");
    tf32_t enc = tu_fp32_to_tf32(2.0f);
    float back = tu_tf32_to_fp32(enc);
    ASSERT_NEAR(back, 2.0f, 1e-10f, "TF32 2.0 roundtrip");
    PASS();
}

static void test_tf32_pi(void) {
    TEST("TF32: PI ≈ 3.14159 (10-bit precision)");
    /* PI = 3.1415927... 10-bit mantissa gives ~3 decimal digit precision */
    tf32_t enc = tu_fp32_to_tf32((float)M_PI);
    float back = tu_tf32_to_fp32(enc);
    float err = fabsf(back - (float)M_PI) / (float)M_PI;
    /* 10-bit mantissa: relative error should be < 1/1024 ≈ 0.001 */
    if (err > 0.002f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "rel_err=%.6f > 0.002 (%.6f vs %.6f)", (double)err, (double)back, M_PI);
        FAIL(buf); return;
    }
    PASS();
}

/* ================================================================
 * Dynamic Range Tests (TF32 has full FP32 range)
 * ================================================================ */

static void test_tf32_max_normal(void) {
    TEST("TF32: max normal (~3.4e38)");
    float v = 3.0e38f;
    tf32_t enc = tu_fp32_to_tf32(v);
    float back = tu_tf32_to_fp32(enc);
    float err = fabsf(back - v) / v;
    if (err > 0.002f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "rel_err=%.6f > 0.002", (double)err);
        FAIL(buf); return;
    }
    PASS();
}

static void test_tf32_min_normal(void) {
    TEST("TF32: min positive normal (~1.18e-38)");
    float v = 1.2e-38f;
    tf32_t enc = tu_fp32_to_tf32(v);
    float back = tu_tf32_to_fp32(enc);
    float err = fabsf(back - v) / v;
    if (err > 0.002f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "rel_err=%.6f > 0.002", (double)err);
        FAIL(buf); return;
    }
    PASS();
}

static void test_tf32_small_subnormal(void) {
    TEST("TF32: small value (1e-38)");
    float v = 1e-38f;
    tf32_t enc = tu_fp32_to_tf32(v);
    float back = tu_tf32_to_fp32(enc);
    /* Subnormals have reduced precision — use absolute tolerance */
    if (fabsf(back - v) > v * 0.1f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "abs_err=%.6e", (double)fabsf(back - v));
        FAIL(buf); return;
    }
    PASS();
}

/* ================================================================
 * Infinity and NaN Tests
 * ================================================================ */

static void test_tf32_infinity(void) {
    TEST("TF32: +Inf roundtrip");
    tf32_t enc = tu_fp32_to_tf32(INFINITY);
    float back = tu_tf32_to_fp32(enc);
    if (!isinf(back) || back < 0) { FAIL("not +Inf"); return; }
    PASS();
}

static void test_tf32_neg_infinity(void) {
    TEST("TF32: -Inf roundtrip");
    tf32_t enc = tu_fp32_to_tf32(-INFINITY);
    float back = tu_tf32_to_fp32(enc);
    if (!isinf(back) || back > 0) { FAIL("not -Inf"); return; }
    PASS();
}

static void test_tf32_nan(void) {
    TEST("TF32: NaN preserve");
    tf32_t enc = tu_fp32_to_tf32(NAN);
    float back = tu_tf32_to_fp32(enc);
    if (!isnan(back)) { FAIL("not NaN"); return; }
    PASS();
}

/* ================================================================
 * Rounding Mode Tests
 * ================================================================ */

static void test_tf32_rne_round_up(void) {
    TEST("TF32: RNE rounds up (3.3)");
    tu_set_rounding_mode(TU_ROUND_RNE);
    /* 3.3 = binary 11.0100110011... With 10 mantissa bits, RNE rounds */
    tf32_t enc = tu_fp32_to_tf32(3.3f);
    float back = tu_tf32_to_fp32(enc);
    /* Should be close to 3.3 */
    float err = fabsf(back - 3.3f) / 3.3f;
    if (err > 0.002f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "rel_err=%.6f (got %.6f)", (double)err, (double)back);
        FAIL(buf); return;
    }
    PASS();
}

static void test_tf32_rtz_truncate(void) {
    TEST("TF32: RTZ truncates toward zero");
    tu_set_rounding_mode(TU_ROUND_RTZ);
    tf32_t enc = tu_fp32_to_tf32(3.7f);
    float back = tu_tf32_to_fp32(enc);
    /* RTZ should give value ≤ 3.7 */
    if (back > 3.7f) { FAIL("RTZ gave value > 3.7"); return; }
    float err = fabsf(back - 3.7f) / 3.7f;
    if (err > 0.002f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "too far from 3.7: %.6f", (double)back);
        FAIL(buf); return;
    }
    PASS();
}

static void test_tf32_stochastic_average(void) {
    TEST("TF32: Stochastic rounding produces unbiased average");
    tu_set_rounding_mode(TU_ROUND_STOCHASTIC);

    /* 1.0001 in FP32: mantissa has low bits set, so stochastic
     * rounding will round up with probability ~0.5 */
    float v = 1.0f + (1.0f / 2048.0f);  /* ~1.000488 */
    int count_up = 0;
    int trials = 1000;
    for (int i = 0; i < trials; i++) {
        tf32_t enc = tu_fp32_to_tf32(v);
        float back = tu_tf32_to_fp32(enc);
        if (back > v) count_up++;
    }
    /* With ~50% probability of rounding up, expect 350-650 out of 1000 */
    float ratio = (float)count_up / trials;
    if (ratio < 0.30f || ratio > 0.70f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "ratio=%.2f (expected ~0.5, got %d/%d ups)",
                 (double)ratio, count_up, trials);
        FAIL(buf); return;
    }
    tu_set_rounding_mode(TU_ROUND_RNE);
    PASS();
}

/* ================================================================
 * Batch Conversion Tests
 * ================================================================ */

static void test_tf32_batch(void) {
    TEST("TF32: Batch 100 values roundtrip");
    float src[100];
    tf32_t mid[100];
    float dst[100];

    for (int i = 0; i < 100; i++)
        src[i] = (float)(i - 50) * 0.5f + 0.1f;

    tu_fp32_to_tf32_buffer(src, mid, 100);
    tu_tf32_to_fp32_buffer(mid, dst, 100);

    int ok = 1;
    for (int i = 0; i < 100 && ok; i++) {
        float err = fabsf(dst[i] - src[i]);
        float denom = fabsf(src[i]) > 1e-10f ? fabsf(src[i]) : 1.0f;
        if (err / denom > 0.002f) {
            char buf[128];
            snprintf(buf, sizeof(buf), "batch[%d]: %.6f vs %.6f", i, (double)dst[i], (double)src[i]);
            FAIL(buf); ok = 0;
        }
    }
    if (ok) PASS();
}

/* ================================================================
 * Mixed Precision Tests
 * ================================================================ */

static void test_tf32_to_fp16(void) {
    TEST("TF32 → FP16: 1.0 passes through");
    tf32_t tf32_val = tu_fp32_to_tf32(1.0f);
    uint16_t fp16_val = tu_tf32_to_fp16(tf32_val);
    float back = tu_fp16_to_fp32(fp16_val);
    ASSERT_NEAR(back, 1.0f, 1e-6f, "TF32→FP16 1.0");
    PASS();
}

static void test_tf32_from_fp16(void) {
    TEST("FP16 → TF32: 1.0 passes through");
    uint16_t fp16_val = tu_fp32_to_fp16(1.0f);
    tf32_t tf32_val = tu_fp16_to_tf32(fp16_val);
    float back = tu_tf32_to_fp32(tf32_val);
    ASSERT_NEAR(back, 1.0f, 1e-6f, "FP16→TF32 1.0");
    PASS();
}

static void test_tf32_to_bf16(void) {
    TEST("TF32 → BF16: 1.0 passes through");
    tf32_t tf32_val = tu_fp32_to_tf32(1.0f);
    uint16_t bf16_val = tu_tf32_to_bf16(tf32_val);
    float back = tu_bf16_to_fp32(bf16_val);
    ASSERT_NEAR(back, 1.0f, 1e-6f, "TF32→BF16 1.0");
    PASS();
}

static void test_tf32_from_bf16(void) {
    TEST("BF16 → TF32: 1.0 passes through");
    uint16_t bf16_val = tu_fp32_to_bf16(1.0f);
    tf32_t tf32_val = tu_bf16_to_tf32(bf16_val);
    float back = tu_tf32_to_fp32(tf32_val);
    ASSERT_NEAR(back, 1.0f, 1e-6f, "BF16→TF32 1.0");
    PASS();
}

/* ================================================================
 * Precision Registry Tests
 * ================================================================ */

static void test_tf32_registry_lookup(void) {
    TEST("TF32: Precision registry lookup");
    const tu_precision_desc_t *desc = tu_precision_get(TU_PREC_TF32);
    if (!desc) { FAIL("registry lookup returned NULL"); return; }
    if (desc->type != TU_PREC_TF32) { FAIL("wrong type"); return; }
    if (desc->elem_bytes != 4) { FAIL("wrong elem_bytes"); return; }
    if (strcmp(desc->name, "tf32") != 0) { FAIL("wrong name"); return; }
    PASS();
}

static void test_tf32_registry_convert(void) {
    TEST("TF32: Registry convert 1.0 roundtrip");
    const tu_precision_desc_t *desc = tu_precision_get(TU_PREC_TF32);
    if (!desc) { FAIL("registry lookup failed"); return; }

    tf32_t mid;
    desc->from_fp32(1.0f, &mid);
    float back = desc->to_fp32(&mid);
    ASSERT_NEAR(back, 1.0f, 1e-6f, "registry convert 1.0");
    PASS();
}

static void test_tf32_registry_convert_pi(void) {
    TEST("TF32: Registry convert PI (~10bit precision)");
    const tu_precision_desc_t *desc = tu_precision_get(TU_PREC_TF32);
    if (!desc) { FAIL("registry lookup failed"); return; }

    tf32_t mid;
    desc->from_fp32((float)M_PI, &mid);
    float back = desc->to_fp32(&mid);
    float err = fabsf(back - (float)M_PI) / (float)M_PI;
    if (err > 0.002f) {
        char buf[128];
        snprintf(buf, sizeof(buf), "rel_err=%.6f", (double)err);
        FAIL(buf); return;
    }
    PASS();
}

/* ================================================================
 * Range Query
 * ================================================================ */

static void test_tf32_range_query(void) {
    TEST("TF32: Range query matches FP32");
    float min_norm, max_norm;
    tu_tf32_get_range(&min_norm, &max_norm);
    if (max_norm < 1e38f) { FAIL("max normal too small"); return; }
    if (min_norm > 1e-37f) { FAIL("min normal too large"); return; }
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("\nTU TF32 (TensorFloat-32) Tests\n");
    printf("══════════════════════════════\n\n");

    /* Basic roundtrip */
    test_tf32_zero();
    test_tf32_neg_zero();
    test_tf32_one();
    test_tf32_neg_one();
    test_tf32_half();
    test_tf32_two();
    test_tf32_pi();

    /* Dynamic range */
    test_tf32_max_normal();
    test_tf32_min_normal();
    test_tf32_small_subnormal();

    /* Special values */
    test_tf32_infinity();
    test_tf32_neg_infinity();
    test_tf32_nan();

    /* Rounding modes */
    test_tf32_rne_round_up();
    test_tf32_rtz_truncate();
    test_tf32_stochastic_average();

    /* Batch */
    test_tf32_batch();

    /* Mixed precision */
    test_tf32_to_fp16();
    test_tf32_from_fp16();
    test_tf32_to_bf16();
    test_tf32_from_bf16();

    /* Registry */
    test_tf32_registry_lookup();
    test_tf32_registry_convert();
    test_tf32_registry_convert_pi();

    /* Range */
    test_tf32_range_query();

    printf("\n═══════════════════════════════\n");
    printf("  Results: %d/%d passed\n", tests_passed, tests_total);
    printf("═══════════════════════════════\n");

    return tests_passed == tests_total ? 0 : 1;
}
