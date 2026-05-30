/*
 * TinyTU Shared Test Framework — Reusable test utilities
 * ========================================================
 * Gap V3/V6: Unified test harness for all TU cmodel tests.
 *
 * Provides: TEST/PASS/FAIL macros, random tensor generation,
 *           golden comparison utilities, progress reporting.
 *
 * Usage:
 *   #include "tu_cmodel/infra/random_tensor.h"
 *   #include "tests/test_framework.h"
 */

#ifndef TINYTU_TEST_FRAMEWORK_H
#define TINYTU_TEST_FRAMEWORK_H

#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/tu_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Test Statistics ─────────────────────────────────────────── */

typedef struct {
    int tests_run;
    int tests_pass;
    int tests_fail;
    float max_observed_error;
    int   max_error_test_id;
    float total_max_err;      /* accumulated for averaging */
    float total_mean_err;
    int   num_error_samples;  /* count of tests with error data */
} tu_test_stats_t;

extern tu_test_stats_t g_test_stats;

/* Initialize test statistics */
static inline void test_stats_init(void) {
    memset(&g_test_stats, 0, sizeof(g_test_stats));
}

/* ── Test Macros ─────────────────────────────────────────────── */

/* Declare and start a test */
#define TEST(name) do { \
    g_test_stats.tests_run++; \
    printf("  %-54s ", name); \
} while(0)

/* Pass the current test */
#define PASS() do { \
    printf("PASS\n"); \
    g_test_stats.tests_pass++; \
} while(0)

/* Fail the current test with formatted message */
#define FAIL(fmt, ...) do { \
    printf("FAIL: " fmt "\n", ##__VA_ARGS__); \
    g_test_stats.tests_fail++; \
} while(0)

/* ── Comparison Utilities ───────────────────────────────────── */

/* Maximum absolute error between two FP32 arrays */
static inline float max_abs_error(const fp32_t *a, const fp32_t *b, uint32_t n) {
    float max_err = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/* Mean absolute error between two FP32 arrays */
static inline float mean_abs_error(const fp32_t *a, const fp32_t *b, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        sum += (double)fabsf(a[i] - b[i]);
    }
    return (float)(sum / (double)n);
}

/* Relative error (normalized by max absolute value) */
static inline float max_rel_error(const fp32_t *a, const fp32_t *b, uint32_t n) {
    float max_err = 0.0f;
    float max_val = 1e-10f;  /* avoid divide-by-zero */
    for (uint32_t i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        float val = fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (val > max_val) max_val = val;
        if (err > max_err) max_err = err;
    }
    return (max_val > 1e-10f) ? max_err / max_val : max_err;
}

/* Record error statistics */
static inline void record_error(int test_id, float max_err, float mean_err) {
    if (max_err > g_test_stats.max_observed_error) {
        g_test_stats.max_observed_error = max_err;
        g_test_stats.max_error_test_id = test_id;
    }
    g_test_stats.total_max_err += max_err;
    g_test_stats.total_mean_err += mean_err;
    g_test_stats.num_error_samples++;
}

/* Compare two tensors with tolerance, return 1 if pass, 0 if fail */
static inline int compare_tensors(const char *label,
                                   const fp32_t *expected,
                                   const fp32_t *actual,
                                   uint32_t count,
                                   float tolerance) {
    float max_err = max_abs_error(expected, actual, count);
    float mean_err = mean_abs_error(expected, actual, count);

    if (max_err <= tolerance) {
        printf("  %-52s PASS (max_err=%.6f, mean_err=%.6f)\n",
               label, max_err, mean_err);
        g_test_stats.tests_run++;
        g_test_stats.tests_pass++;
        record_error(-1, max_err, mean_err);
        return 1;
    } else {
        printf("  %-52s FAIL: max_err=%.6f > tol=%.6f (mean=%.6f)\n",
               label, max_err, tolerance, mean_err);
        g_test_stats.tests_run++;
        g_test_stats.tests_fail++;
        return 0;
    }
}

/* ── Test Summary ───────────────────────────────────────────── */

static inline void print_test_summary(const char *suite_name) {
    printf("\n");
    printf("═══════════════════════════════════════════\n");
    printf("  %s\n", suite_name);
    printf("  %d/%d tests passed\n",
           g_test_stats.tests_pass, g_test_stats.tests_run);
    if (g_test_stats.num_error_samples > 0) {
        printf("  Max observed error: %.6f\n", g_test_stats.max_observed_error);
        printf("  Avg max error: %.6f\n",
               g_test_stats.total_max_err / g_test_stats.num_error_samples);
        printf("  Avg mean error: %.6f\n",
               g_test_stats.total_mean_err / g_test_stats.num_error_samples);
    }
    printf("═══════════════════════════════════════════\n");
}

/* ── Test Exit ───────────────────────────────────────────────── */

static inline int test_exit(void) {
    print_test_summary("Test Suite Complete");
    return (g_test_stats.tests_fail == 0) ? 0 : 1;
}

#ifdef __cplusplus
}
#endif

#endif /* TINYTU_TEST_FRAMEWORK_H */
