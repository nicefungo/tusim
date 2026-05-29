/*
 * TU CModel — Online Softmax Engine Tests
 * =========================================
 * Comprehensive test suite covering all softmax modes, row-wise operation,
 * masking, scaling, and edge cases. Uses the host reference functions
 * for golden comparison.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/tu_sram.h"
#include "../tu_cmodel/compute/softmax_engine.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  TEST %s ... ", name); \
    fflush(stdout); \
} while(0)

#define PASS() do { \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
    tests_failed++; \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        printf("FAIL: %s (expected %d, got %d)\n", msg, (int)(b), (int)(a)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, tol, msg) do { \
    float diff = fabsf((a) - (b)); \
    float denom = fmaxf(fabsf(b), 1.0f); \
    if (diff / denom > (tol)) { \
        printf("FAIL: %s (expected %.6f, got %.6f, diff=%.6e)\n", \
               msg, (double)(b), (double)(a), (double)(diff)); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ---- Helper: create SRAM, write FP32 data, run softmax, read back ---- */

static void sram_write_floats(tu_sram_region_t *s, uint32_t off,
                               const float *data, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        tu_sram_write(s, off + i * sizeof(float), &data[i]);
}

static void sram_read_floats(tu_sram_region_t *s, uint32_t off,
                              float *out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        tu_sram_read(s, off + i * sizeof(float), &out[i]);
}

/* ---- Test 1: Trivial softmax (single element) ---- */
static void test_single_element(void) {
    TEST("single element");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    float input = 42.0f;
    sram_write_floats(&sram, 0, &input, 1);
    tu_softmax(&sram, 0, 1, 0.0f, true);

    float output;
    sram_read_floats(&sram, 0, &output, 1);

    ASSERT_FLOAT_EQ(output, 1.0f, 1e-6f, "single element softmax should be 1.0");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 2: Uniform vector (all same value) ---- */
static void test_uniform_vector(void) {
    TEST("uniform vector");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 8;
    float input[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float expected[8];
    memcpy(expected, input, sizeof(input));

    sram_write_floats(&sram, 0, input, n);

    /* Compute golden on host */
    tu_softmax_host(expected, n, 0.0f);

    /* Run cmodel */
    tu_softmax(&sram, 0, n, 0.0f, true);

    float output[8];
    sram_read_floats(&sram, 0, output, n);

    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], expected[i], 1e-6f, "uniform element");
    }
    /* Sum should be 1.0 */
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += output[i];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "softmax sum should be 1.0");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 3: Known values (compare with numpy) ---- */
static void test_known_values(void) {
    TEST("known values");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 4;
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float golden[4];
    memcpy(golden, input, sizeof(input));

    sram_write_floats(&sram, 0, input, n);

    /* Golden: softmax([1,2,3,4]) */
    tu_softmax_host(golden, n, 0.0f);

    tu_softmax(&sram, 0, n, 0.0f, true);

    float output[4];
    sram_read_floats(&sram, 0, output, n);

    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-6f, "known value element");
    }

    /* Verify monotonic: output[0] < output[1] < output[2] < output[3] */
    assert(output[0] < output[1] && output[1] < output[2] && output[2] < output[3]);

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += output[i];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "sum to 1");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 4: Large values (numerical stability) ---- */
static void test_large_values(void) {
    TEST("large values (stability)");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 4;
    float input[4] = {100.0f, 200.0f, 300.0f, 400.0f};
    float golden[4];
    memcpy(golden, input, sizeof(input));

    sram_write_floats(&sram, 0, input, n);

    tu_softmax_host(golden, n, 0.0f);
    tu_softmax(&sram, 0, n, 0.0f, true);

    float output[4];
    sram_read_floats(&sram, 0, output, n);

    /* The largest value should dominate */
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-6f, "large value element");
    }
    /* output[3] should be ~1.0, others ~0 */
    ASSERT_FLOAT_EQ(output[3], golden[3], 1e-4f, "max should dominate");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 5: Negative values ---- */
static void test_negative_values(void) {
    TEST("negative values");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 4;
    float input[4] = {-4.0f, -3.0f, -2.0f, -1.0f};
    float golden[4];
    memcpy(golden, input, sizeof(input));

    sram_write_floats(&sram, 0, input, n);

    tu_softmax_host(golden, n, 0.0f);
    tu_softmax(&sram, 0, n, 0.0f, true);

    float output[4];
    sram_read_floats(&sram, 0, output, n);

    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-6f, "negative value element");
    }
    /* output[3] (for -1) should be largest */
    assert(output[3] > output[2] && output[2] > output[1] && output[1] > output[0]);

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 6: Batched 2D softmax ---- */
static void test_batched_2d(void) {
    TEST("batched 2D");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t rows = 3, cols = 4;
    float input[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        4.0f, 3.0f, 2.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    float golden[12];
    memcpy(golden, input, sizeof(input));

    sram_write_floats(&sram, 0, input, rows * cols);

    /* Compute golden: softmax each row independently */
    for (uint32_t r = 0; r < rows; r++) {
        tu_softmax_host(golden + r * cols, cols, 0.0f);
    }

    tu_softmax_2d(&sram, 0, rows, cols, 0.0f, true);

    float output[12];
    sram_read_floats(&sram, 0, output, rows * cols);

    for (uint32_t i = 0; i < rows * cols; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-6f, "batched element");
    }

    /* Each row should sum to 1 */
    for (uint32_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; c++)
            sum += output[r * cols + c];
        ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "row sum should be 1");
    }

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 7: Log-softmax ---- */
static void test_log_softmax(void) {
    TEST("log-softmax");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 4;
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float golden[4];
    memcpy(golden, input, sizeof(input));

    sram_write_floats(&sram, 0, input, n);

    tu_log_softmax_host(golden, n, 0.0f);
    tu_log_softmax(&sram, 0, n, 0.0f, true);

    float output[4];
    sram_read_floats(&sram, 0, output, n);

    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-5f, "log-softmax element");
    }

    /* exp(log_softmax) should sum to 1 */
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += expf(output[i]);
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "exp(log_softmax) sum to 1");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 8: Softmax with scaling (attention) ---- */
static void test_scaled_softmax(void) {
    TEST("scaled softmax (attention)");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 4;
    float scale = 1.0f / sqrtf(64.0f);  /* 1/sqrt(d_k) for d_k=64 */
    float input[4] = {8.0f, 16.0f, 24.0f, 32.0f};
    float golden[4];
    memcpy(golden, input, sizeof(input));

    sram_write_floats(&sram, 0, input, n);

    tu_softmax_host(golden, n, scale);
    tu_softmax(&sram, 0, n, scale, true);

    float output[4];
    sram_read_floats(&sram, 0, output, n);

    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-5f, "scaled element");
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += output[i];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "sum to 1");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 9: Masked softmax (causal mask) ---- */
static void test_masked_softmax(void) {
    TEST("masked softmax");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    /* 2x2 matrix, causal mask: position (1,0) masked */
    uint32_t n = 4;
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};  /* [[1,2],[3,4]] */
    float mask[4]  = {0.0f, 0.0f, -1e9f, 0.0f};  /* mask (1,0) */

    sram_write_floats(&sram, 0, input, n);

    tu_softmax_masked(&sram, 0, 2, 2, mask, 0.0f, 0.0f);

    float output[4];
    sram_read_floats(&sram, 0, output, n);

    /* Row 0: softmax([1,2]) → exp(1)/(exp(1)+exp(2)), exp(2)/(exp(1)+exp(2)) */
    float sum0 = expf(1.0f) + expf(2.0f);
    ASSERT_FLOAT_EQ(output[0], expf(1.0f)/sum0, 1e-5f, "masked row0 col0");
    ASSERT_FLOAT_EQ(output[1], expf(2.0f)/sum0, 1e-5f, "masked row0 col1");

    /* Row 1: [3+(-1e9), 4] → [-1e9, 4]
     * max = 4, exp(-1e9-4) ≈ 0, exp(0) = 1
     * output ≈ [0, 1]
     */
    ASSERT_FLOAT_EQ(output[2], 0.0f, 1e-6f, "masked position should be 0");
    ASSERT_FLOAT_EQ(output[3], 1.0f, 1e-6f, "unmasked position should be 1");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 10: All-negative-inf (edge case) ---- */
static void test_all_neg_inf(void) {
    TEST("all negative inf");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 4;
    float input[4] = {-1e9f, -1e9f, -1e9f, -1e9f};
    sram_write_floats(&sram, 0, input, n);
    tu_softmax(&sram, 0, n, 0.0f, true);

    float output[4];
    sram_read_floats(&sram, 0, output, n);

    /* Should output uniform distribution: 1/N each */
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], 0.25f, 1e-5f, "all neg inf -> uniform");
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += output[i];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "sum to 1");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 11: Zero input ---- */
static void test_zero_input(void) {
    TEST("zero input");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 4;
    float input[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    sram_write_floats(&sram, 0, input, n);
    tu_softmax(&sram, 0, n, 0.0f, true);

    float output[4];
    sram_read_floats(&sram, 0, output, n);

    /* exp(0)=1, sum=4, each=0.25 */
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], 0.25f, 1e-6f, "zeros -> 1/N");
    }

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 12: Softmax online mode ---- */
static void test_online_mode(void) {
    TEST("online mode");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 4;
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float golden[4];
    memcpy(golden, input, sizeof(input));

    sram_write_floats(&sram, 0, input, n);

    tu_softmax_host(golden, n, 0.0f);

    /* Use online mode via descriptor */
    tu_softmax_desc_t desc = {
        .mode        = TU_SOFTMAX_ONLINE,
        .data_sram   = &sram,
        .data_offset = 0,
        .elem_count  = n,
        .axis_dim    = 0,
        .scale       = 0.0f,
        .in_place    = true,
        .out_offset  = 0,
    };
    tu_softmax_execute(&desc);

    float output[4];
    sram_read_floats(&sram, 0, output, n);

    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-5f, "online element");
    }

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 13: Out-of-place softmax ---- */
static void test_out_of_place(void) {
    TEST("out-of-place");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t n = 4;
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float golden[4];
    memcpy(golden, input, sizeof(input));

    sram_write_floats(&sram, 0, input, n);
    tu_softmax_host(golden, n, 0.0f);

    tu_softmax_desc_t desc = {
        .mode        = TU_SOFTMAX_STANDARD,
        .data_sram   = &sram,
        .data_offset = 0,
        .elem_count  = n,
        .axis_dim    = 0,
        .scale       = 0.0f,
        .in_place    = false,
        .out_offset  = 64,  /* Write to offset 64 */
    };
    tu_softmax_execute(&desc);

    /* Verify input unchanged */
    float orig[4];
    sram_read_floats(&sram, 0, orig, n);
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(orig[i], input[i], 1e-6f, "input unchanged");
    }

    /* Verify output correct */
    float output[4];
    sram_read_floats(&sram, 64, output, n);
    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-6f, "out-of-place output");
    }

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 14: Large row (256 elements, typical attention head_dim) ---- */
static void test_large_row(void) {
    TEST("large row (256D)");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 65536, "test");

    uint32_t n = 256;
    float *input  = (float *)calloc(n, sizeof(float));
    float *golden = (float *)calloc(n, sizeof(float));

    /* Fill with a range of values */
    for (uint32_t i = 0; i < n; i++) {
        input[i] = (float)i * 0.1f;  /* 0.0, 0.1, 0.2, ..., 25.5 */
    }
    memcpy(golden, input, n * sizeof(float));

    sram_write_floats(&sram, 0, input, n);
    tu_softmax_host(golden, n, 0.0f);
    tu_softmax(&sram, 0, n, 0.0f, true);

    float *output = (float *)calloc(n, sizeof(float));
    sram_read_floats(&sram, 0, output, n);

    for (uint32_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-5f, "large row element");
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += output[i];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-4f, "large row sum to 1");

    free(input); free(golden); free(output);
    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 15: Log-softmax 2D ---- */
static void test_log_softmax_2d(void) {
    TEST("log-softmax 2D");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    uint32_t rows = 2, cols = 3;
    float input[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float golden[6];
    memcpy(golden, input, sizeof(input));

    sram_write_floats(&sram, 0, input, rows * cols);

    for (uint32_t r = 0; r < rows; r++) {
        tu_log_softmax_host(golden + r * cols, cols, 0.0f);
    }

    tu_log_softmax_2d(&sram, 0, rows, cols, 0.0f, true);

    float output[6];
    sram_read_floats(&sram, 0, output, rows * cols);

    for (uint32_t i = 0; i < rows * cols; i++) {
        ASSERT_FLOAT_EQ(output[i], golden[i], 1e-5f, "log-softmax 2D");
    }

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Main ---- */

int main(void) {
    printf("=== TU Softmax Engine Tests ===\n\n");

    test_single_element();
    test_uniform_vector();
    test_known_values();
    test_large_values();
    test_negative_values();
    test_batched_2d();
    test_log_softmax();
    test_scaled_softmax();
    test_masked_softmax();
    test_all_neg_inf();
    test_zero_input();
    test_online_mode();
    test_out_of_place();
    test_large_row();
    test_log_softmax_2d();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
