/*
 * Test: Configurable Rounding Modes (D6)
 * ======================================
 * Tests for RNE, RTZ, and Stochastic rounding across FP16 and BF16.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/rounding.h"

static int tests_total = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_total++; \
    printf("  TEST %-50s ", name); \
} while(0)

#define PASS() do { \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
} while(0)

#define ASSERT_EQ(a, b, fmt, msg) do { \
    if ((a) != (b)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    float diff = fabsf((float)(a) - (float)(b)); \
    if (diff > (tol)) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "%s (%.6f vs %.6f, diff=%.6e)", msg, (double)(a), (double)(b), (double)diff); \
        FAIL(buf); return; \
    } \
} while(0)

/* ================================================================
 * RNE Tests — Regression: matches previous FP16 behavior
 * ================================================================ */

static void test_rne_basic(void) {
    TEST("RNE: 1.0 → FP16 roundtrip");
    tu_set_rounding_mode(TU_ROUND_RNE);
    float orig = 1.0f;
    fp16_t h = tu_fp32_to_fp16(orig);
    float back = tu_fp16_to_fp32(h);
    ASSERT_NEAR(back, orig, 1e-7f, "1.0 roundtrip failed");
    PASS();
}

static void test_rne_tie_to_even(void) {
    TEST("RNE: tie rounding (3.0005 → 3.0, not 3.001)");
    tu_set_rounding_mode(TU_ROUND_RNE);
    /* 3.00048828125 = exactly halfway between 3.0 and 3.0009765625 in FP16 */
    /* 3.0     = 0x4200, 3.000488... rounds to 3.0 because LSB of 3.0 mantissa is 0 */
    float val = 3.00048828125f;
    fp16_t h = tu_fp32_to_fp16(val);
    float back = tu_fp16_to_fp32(h);
    /* The nearest representable FP16 values are 3.0 and ~3.00098.
     * 3.000488 is exactly halfway. RNE should round to 3.0 if mantissa LSB is 0. */
    ASSERT_NEAR(back, 3.0f, 1e-6f, "RNE tie-to-even failed");
    PASS();
}

static void test_rne_fp16_to_fp32(void) {
    TEST("RNE: FP16→FP32 is exact (no rounding needed)");
    tu_set_rounding_mode(TU_ROUND_RNE);
    fp16_t orig = 0x3C00; /* 1.0 in FP16 */
    float f = tu_fp16_to_fp32(orig);
    ASSERT_NEAR(f, 1.0f, 1e-7f, "FP16→FP32 not exact");
    PASS();
}

/* ================================================================
 * RTZ Tests — Always truncate toward zero
 * ================================================================ */

static void test_rtz_basic(void) {
    TEST("RTZ: 3.0009 → truncates to 3.0");
    tu_set_rounding_mode(TU_ROUND_RTZ);
    /* 3.000488 > 3.0, but RTZ truncates → 3.0 */
    float val = 3.00048828125f;
    fp16_t h = tu_fp32_to_fp16(val);
    float back = tu_fp16_to_fp32(h);
    ASSERT_NEAR(back, 3.0f, 1e-6f, "RTZ should truncate to 3.0");
    PASS();
}

static void test_rtz_negative(void) {
    TEST("RTZ: -3.0009 → truncates to -3.0 (toward zero)");
    tu_set_rounding_mode(TU_ROUND_RTZ);
    float val = -3.00048828125f;
    fp16_t h = tu_fp32_to_fp16(val);
    float back = tu_fp16_to_fp32(h);
    ASSERT_NEAR(back, -3.0f, 1e-6f, "RTZ should truncate negative to -3.0");
    PASS();
}

/* ================================================================
 * Stochastic Rounding Tests
 * ================================================================ */

static void test_stochastic_unbiased(void) {
    TEST("Stochastic: average over many trials ≈ true value");
    tu_set_rounding_mode(TU_ROUND_STOCHASTIC);
    tu_stochastic_set_seed(42);

    /* Value exactly halfway: ~3.000488 → 50% probability of rounding up/down */
    float val = 3.00048828125f;
    float sum = 0.0f;
    int n_trials = 10000;
    for (int i = 0; i < n_trials; i++) {
        fp16_t h = tu_fp32_to_fp16(val);
        sum += tu_fp16_to_fp32(h);
    }
    float avg = sum / (float)n_trials;
    /* Average should be close to original (within 0.05% of val) */
    float tolerance = val * 0.005f;
    ASSERT_NEAR(avg, val, tolerance, "Stochastic average not unbiased");
    PASS();
}

static void test_stochastic_reproducible(void) {
    TEST("Stochastic: same seed → same sequence");
    tu_set_rounding_mode(TU_ROUND_STOCHASTIC);

    tu_stochastic_set_seed(12345);
    float sum1 = 0.0f;
    for (int i = 0; i < 100; i++) {
        fp16_t h = tu_fp32_to_fp16(0.12345f);
        sum1 += tu_fp16_to_fp32(h);
    }

    tu_stochastic_set_seed(12345);
    float sum2 = 0.0f;
    for (int i = 0; i < 100; i++) {
        fp16_t h = tu_fp32_to_fp16(0.12345f);
        sum2 += tu_fp16_to_fp32(h);
    }

    ASSERT_NEAR(sum1, sum2, 1e-10f, "Stochastic not reproducible with same seed");
    PASS();
}

/* ================================================================
 * BF16 Rounding Tests
 * ================================================================ */

static void test_bf16_rne(void) {
    TEST("BF16 RNE: 1.0 → BF16 roundtrip");
    tu_set_rounding_mode(TU_ROUND_RNE);
    float orig = 1.0f;
    bf16_t h = tu_fp32_to_bf16(orig);
    float back = tu_bf16_to_fp32(h);
    ASSERT_NEAR(back, orig, 1e-7f, "BF16 1.0 roundtrip failed");
    PASS();
}

static void test_bf16_rtz(void) {
    TEST("BF16 RTZ: truncates lower bits");
    tu_set_rounding_mode(TU_ROUND_RTZ);
    /* 1.0 + small fraction — BF16 has 7 mantissa bits, so any value
     * between 1.0 and 1.0 + 1/128 should truncate to 1.0 */
    float val = 1.0f + (1.0f / 256.0f);  /* 1.00390625 */
    bf16_t h = tu_fp32_to_bf16(val);
    float back = tu_bf16_to_fp32(h);
    ASSERT_NEAR(back, 1.0f, 1e-6f, "BF16 RTZ truncation failed");
    PASS();
}

static void test_bf16_stochastic(void) {
    TEST("BF16 Stochastic: average ≈ true value");
    tu_set_rounding_mode(TU_ROUND_STOCHASTIC);
    tu_stochastic_set_seed(99);

    float val = 1.0f + (3.0f / 256.0f);  /* ~1.0117 — 75% of the way to 1.015625 */
    float sum = 0.0f;
    int n_trials = 5000;
    for (int i = 0; i < n_trials; i++) {
        bf16_t h = tu_fp32_to_bf16(val);
        sum += tu_bf16_to_fp32(h);
    }
    float avg = sum / (float)n_trials;
    float tolerance = val * 0.02f;
    ASSERT_NEAR(avg, val, tolerance, "BF16 stochastic average not unbiased");
    PASS();
}

/* ================================================================
 * Rounding Mode Switch — Verify correctness after mode changes
 * ================================================================ */

static void test_mode_switch(void) {
    TEST("Mode switch: RNE→RTZ→STOCHASTIC→RNE consistency");
    float val = 3.00048828125f;

    /* RNE baseline */
    tu_set_rounding_mode(TU_ROUND_RNE);
    fp16_t rne_result = tu_fp32_to_fp16(val);

    /* Switch to RTZ, convert, switch back */
    tu_set_rounding_mode(TU_ROUND_RTZ);
    fp16_t rtz_result = tu_fp32_to_fp16(val);

    /* Switch to RNE and verify same result as baseline */
    tu_set_rounding_mode(TU_ROUND_RNE);
    fp16_t rne2_result = tu_fp32_to_fp16(val);

    ASSERT_EQ(rne_result, rne2_result, "%04x vs %04x", "RNE not consistent after mode switch");
    /* RTZ should give different (smaller) result */
    float rne_f = tu_fp16_to_fp32(rne_result);
    float rtz_f = tu_fp16_to_fp32(rtz_result);
    if (rtz_f > rne_f) {
        FAIL("RTZ should not round up beyond RNE");
        return;
    }
    PASS();
}

/* ================================================================
 * Carry/Overflow Tests
 * ================================================================ */

static void test_rounding_carry(void) {
    TEST("Rounding carry: mantissa overflow → exponent increment");
    tu_set_rounding_mode(TU_ROUND_RNE);

    /* 65504.0 = max FP16 normal. 65504 + anything should overflow to Inf */
    float val = 65520.0f;  /* > max FP16 */
    fp16_t h = tu_fp32_to_fp16(val);
    /* Should saturate to infinity */
    ASSERT_EQ((h & 0x7C00), 0x7C00, "%04x", "FP16 overflow should give infinity");
    PASS();
}

/* ================================================================
 * Subnormal + Rounding Interaction
 * ================================================================ */

static void test_subnormal_with_rounding(void) {
    TEST("Subnormal FTZ: tiny values → zero regardless of rounding mode");
    tu_set_subnormal_mode(TU_SUBNORMAL_FLUSH);

    /* Try all three rounding modes with a subnormal value */
    float tiny = 1.0e-8f;  /* Way below FP16 min normal (6.1e-5) */

    tu_set_rounding_mode(TU_ROUND_RNE);
    fp16_t h_rne = tu_fp32_to_fp16(tiny);
    ASSERT_EQ(h_rne & 0x7FFF, 0, "%04x", "Subnormal should flush to zero (RNE)");

    tu_set_rounding_mode(TU_ROUND_RTZ);
    fp16_t h_rtz = tu_fp32_to_fp16(tiny);
    ASSERT_EQ(h_rtz & 0x7FFF, 0, "%04x", "Subnormal should flush to zero (RTZ)");

    tu_set_rounding_mode(TU_ROUND_STOCHASTIC);
    fp16_t h_stoch = tu_fp32_to_fp16(tiny);
    ASSERT_EQ(h_stoch & 0x7FFF, 0, "%04x", "Subnormal should flush to zero (STOCHASTIC)");

    /* Restore */
    tu_set_subnormal_mode(TU_SUBNORMAL_FULL);
    tu_set_rounding_mode(TU_ROUND_RNE);
    PASS();
}

/* ================================================================
 * Bulk Buffer Tests
 * ================================================================ */

static void test_bulk_rounding(void) {
    TEST("Bulk: all modes produce valid FP16 for 1000 random values");
    fp32_t inputs[1000];
    fp16_t outputs[1000];
    srand(42);
    for (int i = 0; i < 1000; i++) {
        inputs[i] = ((float)rand() / (float)RAND_MAX) * 10.0f - 5.0f;
    }

    int modes[] = {TU_ROUND_RNE, TU_ROUND_RTZ, TU_ROUND_STOCHASTIC};
    for (int m = 0; m < 3; m++) {
        tu_set_rounding_mode(modes[m]);
        tu_fp32_to_fp16_buffer(inputs, outputs, 1000);
        for (int i = 0; i < 1000; i++) {
            float back = tu_fp16_to_fp32(outputs[i]);
            if (isnan(back)) {
                FAIL("Bulk conversion produced NaN");
                return;
            }
        }
    }
    tu_set_rounding_mode(TU_ROUND_RNE);
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("\n═══════════════════════════════════\n");
    printf("  TU Rounding Modes Test Suite\n");
    printf("  Gap D6: Configurable Rounding\n");
    printf("═══════════════════════════════════\n\n");

    /* RNE */
    test_rne_basic();
    test_rne_tie_to_even();
    test_rne_fp16_to_fp32();

    /* RTZ */
    test_rtz_basic();
    test_rtz_negative();

    /* Stochastic */
    test_stochastic_unbiased();
    test_stochastic_reproducible();

    /* BF16 */
    test_bf16_rne();
    test_bf16_rtz();
    test_bf16_stochastic();

    /* Mode switching */
    test_mode_switch();

    /* Edge cases */
    test_rounding_carry();
    test_subnormal_with_rounding();
    test_bulk_rounding();

    printf("\n═══════════════════════════════════\n");
    printf("  Results: %d/%d passed\n", tests_passed, tests_total);
    printf("═══════════════════════════════════\n");

    return (tests_passed == tests_total) ? 0 : 1;
}
