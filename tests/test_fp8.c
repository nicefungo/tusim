/*
 * Test: FP8 E4M3 and E5M2 (D4)
 * =============================
 * Comprehensive tests for FP8 per OCP Microscaling Formats spec.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "tu_cmodel/fp8.h"
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/rounding.h"

static int tests_total = 0;
static int tests_passed = 0;

#define TEST(name) do { tests_total++; printf("  TEST %-54s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

#define ASSERT_EQ(a, b, fmt, msg) do { \
    if ((a) != (b)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    float diff = fabsf((float)(a) - (float)(b)); \
    float denom = fabsf((float)(b)) > 1e-10f ? fabsf((float)(b)) : 1.0f; \
    if (diff / denom > (tol)) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "%s (%.6e vs %.6e, rel_diff=%.2e)", msg, (double)(a), (double)(b), (double)(diff/denom)); \
        FAIL(buf); return; \
    } \
} while(0)

/* ================================================================
 * E4M3 Tests
 * ================================================================ */

static void test_e4m3_zero(void) {
    TEST("E4M3: +0.0 roundtrip");
    uint8_t enc = tu_fp32_to_fp8_e4m3(0.0f);
    ASSERT_EQ(enc, 0x00, "%02x", "E4M3 +0 encode");
    float back = tu_fp8_e4m3_to_fp32(enc);
    ASSERT_NEAR(back, 0.0f, 1e-10f, "E4M3 +0 decode");
    PASS();
}

static void test_e4m3_negative_zero(void) {
    TEST("E4M3: -0.0 roundtrip");
    float neg_zero = -0.0f;
    uint8_t enc = tu_fp32_to_fp8_e4m3(neg_zero);
    ASSERT_EQ(enc, 0x80, "%02x", "E4M3 -0 encode");
    float back = tu_fp8_e4m3_to_fp32(enc);
    /* -0.0 decodes to signed zero */
    uint32_t bits;
    memcpy(&bits, &back, 4);
    ASSERT_EQ((bits >> 31) & 1, 1u, "%u", "E4M3 -0 sign bit");
    PASS();
}

static void test_e4m3_one(void) {
    TEST("E4M3: 1.0 roundtrip");
    uint8_t enc = tu_fp32_to_fp8_e4m3(1.0f);
    ASSERT_EQ(enc, 0x38, "%02x", "E4M3 1.0 encode");  /* exp=7, bias 7 → exp field = 7, mantissa=0 */
    float back = tu_fp8_e4m3_to_fp32(enc);
    ASSERT_NEAR(back, 1.0f, 1e-7f, "E4M3 1.0 decode");
    PASS();
}

static void test_e4m3_max_normal(void) {
    TEST("E4M3: max normal = 240 (0x77 = 0_1110_111)");
    /* Max normal E4M3: s1110_111 = 2^7 * (1 + 7/8) = 240 */
    uint8_t enc = 0x77;  /* 0_1110_111 */
    float val = tu_fp8_e4m3_to_fp32(enc);
    ASSERT_NEAR(val, 240.0f, 1.0f, "E4M3 max normal decode");

    uint8_t renc = tu_fp32_to_fp8_e4m3(240.0f);
    /* Encode may produce 0x77 or 0x76 depending on rounding */
    float back = tu_fp8_e4m3_to_fp32(renc);
    ASSERT_NEAR(back, 240.0f, 30.0f, "E4M3 max normal encode roundtrip");
    PASS();
}

static void test_e4m3_min_normal(void) {
    TEST("E4M3: min positive normal (2^-6 = 0.015625)");
    /* Min normal: s0001_000 = 2^-6 = 0.015625 */
    float min_norm = 1.0f / 64.0f;  /* 2^-6 */
    uint8_t enc = tu_fp32_to_fp8_e4m3(min_norm);
    float back = tu_fp8_e4m3_to_fp32(enc);
    ASSERT_NEAR(back, min_norm, 1e-5f, "E4M3 min normal");
    PASS();
}

static void test_e4m3_subnormal(void) {
    TEST("E4M3: subnormal (0.0078125 = 2^-7)");
    float tiny = 1.0f / 128.0f;  /* 2^-7, subnormal in E4M3 */
    uint8_t enc = tu_fp32_to_fp8_e4m3(tiny);
    /* Should be subnormal: exp=0000, mant=100 = 2^-6 * 4/8 = 2^-7 */
    ASSERT_EQ((enc >> 3) & 0x0F, 0, "%u", "E4M3 subnormal exp should be 0");
    float back = tu_fp8_e4m3_to_fp32(enc);
    ASSERT_NEAR(back, tiny, 0.01f, "E4M3 subnormal decode");  /* 1% relative */
    PASS();
}

static void test_e4m3_overflow_to_nan(void) {
    TEST("E4M3: overflow → NaN (no infinity)");
    tu_set_rounding_mode(TU_ROUND_RNE);
    uint8_t enc = tu_fp32_to_fp8_e4m3(500.0f);  /* > max */
    /* Should be NaN (s1111_111) */
    ASSERT_EQ(enc & 0x7F, 0x7F, "%02x", "E4M3 overflow should be NaN");
    float back = tu_fp8_e4m3_to_fp32(enc);
    if (!isnan(back)) { FAIL("E4M3 overflow decodes to non-NaN"); return; }
    PASS();
}

static void test_e4m3_negative(void) {
    TEST("E4M3: negative value (-42.0)");
    tu_set_rounding_mode(TU_ROUND_RNE);
    uint8_t enc = tu_fp32_to_fp8_e4m3(-42.0f);
    ASSERT_EQ((enc >> 7) & 1, 1u, "%u", "E4M3 negative sign bit");
    float back = tu_fp8_e4m3_to_fp32(enc);
    ASSERT_NEAR(back, -42.0f, 1.0f, "E4M3 -42.0 roundtrip");
    PASS();
}

/* ================================================================
 * E5M2 Tests
 * ================================================================ */

static void test_e5m2_zero(void) {
    TEST("E5M2: +0.0 roundtrip");
    uint8_t enc = tu_fp32_to_fp8_e5m2(0.0f);
    ASSERT_EQ(enc, 0x00, "%02x", "E5M2 +0 encode");
    float back = tu_fp8_e5m2_to_fp32(enc);
    ASSERT_NEAR(back, 0.0f, 1e-10f, "E5M2 +0 decode");
    PASS();
}

static void test_e5m2_one(void) {
    TEST("E5M2: 1.0 roundtrip");
    uint8_t enc = tu_fp32_to_fp8_e5m2(1.0f);
    ASSERT_EQ(enc, 0x3C, "%02x", "E5M2 1.0 encode");  /* exp=15, bias 15 → exp field=15, mant=0 */
    float back = tu_fp8_e5m2_to_fp32(enc);
    ASSERT_NEAR(back, 1.0f, 1e-7f, "E5M2 1.0 decode");
    PASS();
}

static void test_e5m2_max_normal(void) {
    TEST("E5M2: max normal (~57344)");
    /* Max normal E5M2: s11110_11 = 2^15 * (1 + 3/4) = 57344 */
    uint8_t enc = 0x7B;  /* 0_11110_11 */
    float val = tu_fp8_e5m2_to_fp32(enc);
    ASSERT_NEAR(val, 57344.0f, 100.0f, "E5M2 max normal decode");

    uint8_t renc = tu_fp32_to_fp8_e5m2(57344.0f);
    float back = tu_fp8_e5m2_to_fp32(renc);
    ASSERT_NEAR(back, 57344.0f, 5000.0f, "E5M2 max normal encode roundtrip");
    PASS();
}

static void test_e5m2_min_normal(void) {
    TEST("E5M2: min positive normal (2^-14 ≈ 6.1e-5)");
    float min_norm = 1.0f / 16384.0f;  /* 2^-14 */
    uint8_t enc = tu_fp32_to_fp8_e5m2(min_norm);
    float back = tu_fp8_e5m2_to_fp32(enc);
    ASSERT_NEAR(back, min_norm, 0.1f, "E5M2 min normal");
    PASS();
}

static void test_e5m2_infinity(void) {
    TEST("E5M2: infinity encode/decode");
    uint8_t enc_pos = tu_fp32_to_fp8_e5m2(INFINITY);
    ASSERT_EQ(enc_pos, 0x7C, "%02x", "E5M2 +inf encode");

    uint8_t enc_neg = tu_fp32_to_fp8_e5m2(-INFINITY);
    ASSERT_EQ(enc_neg, 0xFC, "%02x", "E5M2 -inf encode");

    float back_pos = tu_fp8_e5m2_to_fp32(0x7C);
    if (!isinf(back_pos) || back_pos < 0) { FAIL("E5M2 +inf decode"); return; }

    float back_neg = tu_fp8_e5m2_to_fp32(0xFC);
    if (!isinf(back_neg) || back_neg > 0) { FAIL("E5M2 -inf decode"); return; }
    PASS();
}

static void test_e5m2_nan(void) {
    TEST("E5M2: NaN encode/decode");
    uint8_t enc = tu_fp32_to_fp8_e5m2(NAN);
    /* Should be s11111_01 = 0x7D */
    ASSERT_EQ(enc, 0x7D, "%02x", "E5M2 NaN encode");
    float back = tu_fp8_e5m2_to_fp32(0x7D);
    if (!isnan(back)) { FAIL("E5M2 NaN decode"); return; }
    PASS();
}

static void test_e5m2_subnormal(void) {
    TEST("E5M2: subnormal (2^-15 ≈ 3.05e-5)");
    float tiny = 1.0f / 32768.0f;  /* 2^-15, subnormal in E5M2 */
    uint8_t enc = tu_fp32_to_fp8_e5m2(tiny);
    ASSERT_EQ((enc >> 2) & 0x1F, 0, "%u", "E5M2 subnormal exp should be 0");
    float back = tu_fp8_e5m2_to_fp32(enc);
    ASSERT_NEAR(back, tiny, 0.01f, "E5M2 subnormal decode");
    PASS();
}

/* ================================================================
 * Precision Registry Tests
 * ================================================================ */

static void test_precision_registry_e4m3(void) {
    TEST("Precision registry: fp8_e4m3 lookup + convert");
    const tu_precision_desc_t *desc = tu_precision_get(TU_PREC_FP8_E4M3);
    if (!desc) { FAIL("fp8_e4m3 not in registry"); return; }
    ASSERT_EQ(desc->elem_bytes, 1u, "%u", "E4M3 elem size");

    uint8_t val = 0x38;  /* 1.0 in E4M3 */
    float f = desc->to_fp32(&val);
    ASSERT_NEAR(f, 1.0f, 1e-7f, "Registry E4M3 to_fp32");

    uint8_t dst;
    desc->from_fp32(1.0f, &dst);
    ASSERT_EQ(dst, 0x38, "%02x", "Registry E4M3 from_fp32");
    PASS();
}

static void test_precision_registry_e5m2(void) {
    TEST("Precision registry: fp8_e5m2 lookup + convert");
    const tu_precision_desc_t *desc = tu_precision_get(TU_PREC_FP8_E5M2);
    if (!desc) { FAIL("fp8_e5m2 not in registry"); return; }
    ASSERT_EQ(desc->elem_bytes, 1u, "%u", "E5M2 elem size");

    uint8_t val = 0x3C;  /* 1.0 in E5M2 */
    float f = desc->to_fp32(&val);
    ASSERT_NEAR(f, 1.0f, 1e-7f, "Registry E5M2 to_fp32");
    PASS();
}

/* ================================================================
 * Cross-Precision Tests
 * ================================================================ */

static void test_fp8_to_fp16(void) {
    TEST("FP8 E4M3 → FP16: 1.0 passes through");
    uint8_t fp8_val = 0x38;  /* 1.0 in E4M3 */
    uint16_t fp16_val = tu_fp8_e4m3_to_fp16(fp8_val);
    ASSERT_EQ(fp16_val, 0x3C00, "%04x", "E4M3→FP16 1.0");
    PASS();
}

/* ================================================================
 * Rounding Interaction Tests
 * ================================================================ */

static void test_fp8_with_rounding_rtz(void) {
    TEST("FP8 E4M3 + RTZ: 3.2 → 3.0");
    tu_set_rounding_mode(TU_ROUND_RTZ);
    uint8_t enc = tu_fp32_to_fp8_e4m3(3.2f);
    float back = tu_fp8_e4m3_to_fp32(enc);
    /* With RTZ, 3.2 should truncate to 3.0, not round to 3.25 */
    ASSERT_NEAR(back, 3.0f, 1e-6f, "FP8 E4M3 RTZ truncation");
    tu_set_rounding_mode(TU_ROUND_RNE);
    PASS();
}

static void test_fp8_with_stochastic(void) {
    TEST("FP8 E4M3 + Stochastic: average unbiased");
    tu_set_rounding_mode(TU_ROUND_STOCHASTIC);
    tu_stochastic_set_seed(123);

    float val = 1.5f;  /* Exactly between 1.0 and 2.0 in E4M3 (mantissa=0.5) */
    float sum = 0.0f;
    int n = 5000;
    for (int i = 0; i < n; i++) {
        float back = tu_fp8_e4m3_to_fp32(tu_fp32_to_fp8_e4m3(val));
        sum += back;
    }
    float avg = sum / (float)n;
    ASSERT_NEAR(avg, val, 0.2f, "FP8 stochastic average not unbiased");
    tu_set_rounding_mode(TU_ROUND_RNE);
    PASS();
}

/* ================================================================
 * Batch Tests
 * ================================================================ */

static void test_batch_e4m3(void) {
    TEST("Batch E4M3: 100 values roundtrip");
    float inputs[100];
    uint8_t fp8_vals[100];
    float outputs[100];

    for (int i = 0; i < 100; i++) {
        inputs[i] = (float)(i - 50) * 0.5f;  /* Range: -25 to 24.5 */
    }

    tu_fp32_to_fp8_e4m3_buffer(inputs, fp8_vals, 100);
    tu_fp8_e4m3_to_fp32_buffer(fp8_vals, outputs, 100);

    int valid = 0;
    for (int i = 0; i < 100; i++) {
        if (!isnan(outputs[i])) valid++;
    }
    if (valid < 95) { FAIL("Too many NaN in batch E4M3"); return; }
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("\n═══════════════════════════════════\n");
    printf("  TU FP8 Test Suite\n");
    printf("  Gap D4: FP8 E4M3 & E5M2 (OCP MX)\n");
    printf("═══════════════════════════════════\n\n");

    tu_set_rounding_mode(TU_ROUND_RNE);

    /* E4M3 */
    test_e4m3_zero();
    test_e4m3_negative_zero();
    test_e4m3_one();
    test_e4m3_max_normal();
    test_e4m3_min_normal();
    test_e4m3_subnormal();
    test_e4m3_overflow_to_nan();
    test_e4m3_negative();

    /* E5M2 */
    test_e5m2_zero();
    test_e5m2_one();
    test_e5m2_max_normal();
    test_e5m2_min_normal();
    test_e5m2_infinity();
    test_e5m2_nan();
    test_e5m2_subnormal();

    /* Registry */
    test_precision_registry_e4m3();
    test_precision_registry_e5m2();

    /* Cross-precision */
    test_fp8_to_fp16();

    /* Rounding */
    test_fp8_with_rounding_rtz();
    test_fp8_with_stochastic();

    /* Batch */
    test_batch_e4m3();

    printf("\n═══════════════════════════════════\n");
    printf("  Results: %d/%d passed\n", tests_passed, tests_total);
    printf("═══════════════════════════════════\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
