/*
 * TU DPI-C Integration Tests
 * ============================
 * Gap I1: Validates DPI-C wrapper for SystemVerilog co-simulation.
 *
 * Tests:
 *   1. Init/destroy lifecycle
 *   2. Multi-instance (3 handles simultaneously)
 *   3. Memory write/read round-trip
 *   4. GEMM execution via DPI
 *   5. Elementwise operations
 *   6. Softmax computation
 *   7. LayerNorm computation
 *   8. Performance counters
 *   9. Dataflow switching
 *   10. Async command submission
 *   11. Reset and re-use
 *   12. Error handling (invalid handles, bad params)
 */

#include "bindings/tu_dpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    printf("  [TEST] %-50s ", name); fflush(stdout); \
} while(0)

#define PASS do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_EQ(a, b, msg) do { if ((a) != (b)) { \
    printf("FAIL: %s (got %d, expected %d)\n", msg, (int)(a), (int)(b)); \
    tests_failed++; return; } } while(0)

/* ---- Helper: create FP16 1.0 value ---- */
static uint16_t fp16_one(void) {
    return 0x3C00;
}

/* ---- Test 1: Init/Destroy Lifecycle ---- */

static void test_lifecycle(void) {
    TEST("init and destroy single instance");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);
    ASSERT(h >= 1, "init failed");

    int ret = tu_dpi_destroy(h);
    ASSERT_EQ(ret, TU_DPI_OK, "destroy failed");

    /* Verify handle is now invalid */
    ret = tu_dpi_read_counter(h, TU_DPI_CNT_DMA_BYTES);
    ASSERT(ret < 0, "invalid handle should return error");

    PASS;
}

/* ---- Test 2: Multi-Instance ---- */

static void test_multi_instance(void) {
    TEST("three simultaneous instances");
    int h1 = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);
    int h2 = tu_dpi_init(32, 32, 512, TU_DPI_DF_OS);
    int h3 = tu_dpi_init(8, 8, 128, TU_DPI_DF_RS);

    ASSERT(h1 >= 1 && h2 >= 1 && h3 >= 1, "init failed");
    ASSERT(h1 != h2 && h1 != h3 && h2 != h3, "handles should be unique");

    /* Verify each has correct PE dims */
    int rows, cols;
    tu_dpi_get_pe_dims(h1, &rows, &cols);
    ASSERT_EQ(rows, 16, "h1 rows"); ASSERT_EQ(cols, 16, "h1 cols");

    tu_dpi_get_pe_dims(h2, &rows, &cols);
    ASSERT_EQ(rows, 32, "h2 rows"); ASSERT_EQ(cols, 32, "h2 cols");

    tu_dpi_get_pe_dims(h3, &rows, &cols);
    ASSERT_EQ(rows, 8, "h3 rows"); ASSERT_EQ(cols, 8, "h3 cols");

    tu_dpi_destroy(h1);
    tu_dpi_destroy(h2);
    tu_dpi_destroy(h3);
    PASS;
}

/* ---- Test 3: Memory Round-Trip ---- */

static void test_memory_rw(void) {
    TEST("SRAM write/read round-trip");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    /* Write pattern to W-buffer */
    uint16_t pattern[16];
    for (int i = 0; i < 16; i++) {
        pattern[i] = (uint16_t)(0x3C00 + i); /* 1.0 + offset */
    }

    int ret = tu_dpi_sram_write(h, 0, 0, pattern, 32);
    ASSERT_EQ(ret, TU_DPI_OK, "write W-buffer failed");

    /* Read back and verify */
    uint16_t verify[16] = {0};
    ret = tu_dpi_sram_read(h, 0, 0, verify, 32);
    ASSERT_EQ(ret, TU_DPI_OK, "read W-buffer failed");

    for (int i = 0; i < 16; i++) {
        if (verify[i] != pattern[i]) {
            FAIL("data mismatch in round-trip");
            tu_dpi_destroy(h);
            return;
        }
    }

    /* Verify A and O buffers */
    int a_size = tu_dpi_sram_size(h, 1);
    int o_size = tu_dpi_sram_size(h, 2);
    ASSERT(a_size > 0, "A buffer size should be > 0");
    ASSERT(o_size > 0, "O buffer size should be > 0");

    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 4: GEMM Execution ---- */

static void test_gemm(void) {
    TEST("GEMM identity matrix via DPI");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    int N = 16;

    /* Build identity matrices */
    uint16_t *W = (uint16_t *)calloc(N * N, 2);
    uint16_t *A = (uint16_t *)calloc(N * N, 2);
    for (int i = 0; i < N; i++) {
        W[i * N + i] = fp16_one();
        A[i * N + i] = fp16_one();
    }

    /* Zero O-buffer (FP32, N×N×4 bytes) */
    float *O_zero = (float *)calloc(N * N, 4);

    tu_dpi_sram_write(h, 0, 0, W, N * N * 2);
    tu_dpi_sram_write(h, 1, 0, A, N * N * 2);
    tu_dpi_sram_write(h, 2, 0, O_zero, N * N * 4);

    long long cycles = tu_dpi_gemm(h, N, N, N, 0, 0, 0, 0);
    ASSERT(cycles >= 0, "GEMM returned error");

    /* Read output and verify identity */
    float *O = (float *)calloc(N * N, 4);
    tu_dpi_sram_read(h, 2, 0, O, N * N * 4);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float got = O[i * N + j];
            if (fabsf(got - expected) > 1e-5f) {
                printf("FAIL: O[%d][%d] = %f, expected %f\n", i, j, got, expected);
                tests_failed++;
                free(W); free(A); free(O_zero); free(O);
                tu_dpi_destroy(h);
                return;
            }
        }
    }

    free(W); free(A); free(O_zero); free(O);
    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 5: Elementwise Operations ---- */

static void test_elementwise(void) {
    TEST("ReLU elementwise via DPI");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    int rows = 4, cols = 4;
    float data[16] = {-2.0f, -1.0f, 0.0f, 1.0f,
                      -0.5f, 0.5f, -3.0f, 2.0f,
                      5.0f, -4.0f, 0.1f, -0.1f,
                      0.0f, -2.5f, 3.0f, -1.0f};

    tu_dpi_sram_write(h, 2, 0, data, rows * cols * 4);

    long long cycles = tu_dpi_elementwise(h, 0, 0, rows, cols, 0);
    ASSERT(cycles >= 0, "ReLU returned error");

    /* Read and verify ReLU */
    float result[16];
    tu_dpi_sram_read(h, 2, 0, result, rows * cols * 4);

    for (int i = 0; i < rows * cols; i++) {
        float expected = data[i] > 0 ? data[i] : 0.0f;
        if (fabsf(result[i] - expected) > 1e-5f) {
            printf("FAIL: ReLU[%d] = %f, expected %f\n", i, result[i], expected);
            tests_failed++;
            tu_dpi_destroy(h);
            return;
        }
    }

    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 6: Softmax ---- */

static void test_softmax(void) {
    TEST("softmax via DPI");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    int rows = 2, cols = 4;
    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                      1.0f, 1.0f, 1.0f, 1.0f};

    tu_dpi_sram_write(h, 2, 0, data, rows * cols * 4);

    long long cycles = tu_dpi_softmax(h, 0, rows, cols);
    ASSERT(cycles >= 0, "Softmax returned error");

    /* Read and verify softmax properties */
    float result[8];
    tu_dpi_sram_read(h, 2, 0, result, rows * cols * 4);

    /* Check row 0 sums to ~1.0 */
    float sum0 = 0.0f;
    for (int j = 0; j < cols; j++) sum0 += result[j];
    if (fabsf(sum0 - 1.0f) > 1e-4f) {
        printf("FAIL: softmax row 0 sum = %f, expected 1.0\n", sum0);
        tests_failed++;
        tu_dpi_destroy(h);
        return;
    }

    /* All outputs should be in [0, 1] */
    for (int i = 0; i < rows * cols; i++) {
        if (result[i] < 0.0f || result[i] > 1.0f) {
            printf("FAIL: softmax[%d] = %f, out of [0,1]\n", i, result[i]);
            tests_failed++;
            tu_dpi_destroy(h);
            return;
        }
    }

    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 7: LayerNorm ---- */

static void test_layernorm(void) {
    TEST("layernorm via DPI");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    int rows = 2, cols = 4;
    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                      -2.0f, -1.0f, 0.0f, 1.0f};

    tu_dpi_sram_write(h, 2, 0, data, rows * cols * 4);

    /* Epsilon = 1e-5f */
    int eps_int; memcpy(&eps_int, &(float){1e-5f}, 4);
    long long cycles = tu_dpi_layernorm(h, 0, rows, cols, eps_int);
    ASSERT(cycles >= 0, "LayerNorm returned error");

    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 8: Performance Counters ---- */

static void test_counters(void) {
    TEST("performance counters via DPI");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    int N = 8;
    uint16_t *W = (uint16_t *)calloc(N * N, 2);
    uint16_t *A = (uint16_t *)calloc(N * N, 2);
    for (int i = 0; i < N; i++) {
        W[i * N + i] = fp16_one();
        A[i * N + i] = fp16_one();
    }
    float *O = (float *)calloc(N * N, 4);

    tu_dpi_sram_write(h, 0, 0, W, N * N * 2);
    tu_dpi_sram_write(h, 1, 0, A, N * N * 2);
    tu_dpi_sram_write(h, 2, 0, O, N * N * 4);

    tu_dpi_gemm(h, N, N, N, 0, 0, 0, 0);

    long long calls = tu_dpi_read_counter(h, TU_DPI_CNT_MMA_CALLS);
    ASSERT(calls >= 1, "MMA calls should be at least 1");

    long long flops = tu_dpi_read_counter(h, TU_DPI_CNT_MMA_FLOPS);
    ASSERT(flops > 0, "MMA FLOPS should be > 0");

    long long cycles = tu_dpi_read_counter(h, TU_DPI_CNT_EST_CYCLES);
    ASSERT(cycles > 0, "estimated cycles should be > 0");

    free(W); free(A); free(O);
    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 9: Dataflow Switching ---- */

static void test_dataflow_switch(void) {
    TEST("dataflow switch via DPI");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    char name[64];
    tu_dpi_get_dataflow_name(h, name, sizeof(name));
    ASSERT(strcmp(name, "weight_stationary") == 0, "initial DF should be WS");

    tu_dpi_set_dataflow(h, TU_DPI_DF_OS);
    tu_dpi_get_dataflow_name(h, name, sizeof(name));
    ASSERT(strcmp(name, "output_stationary") == 0, "switch to OS failed");

    tu_dpi_set_dataflow(h, TU_DPI_DF_RS);
    tu_dpi_get_dataflow_name(h, name, sizeof(name));
    ASSERT(strcmp(name, "row_stationary") == 0, "switch to RS failed");

    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 10: Async Command Submission ---- */

static void test_async_cmd(void) {
    TEST("async command submission and sync");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    int N = 8;
    uint16_t *W = (uint16_t *)calloc(N * N, 2);
    uint16_t *A = (uint16_t *)calloc(N * N, 2);
    for (int i = 0; i < N; i++) {
        W[i * N + i] = fp16_one();
        A[i * N + i] = fp16_one();
    }
    float *O = (float *)calloc(N * N, 4);

    tu_dpi_sram_write(h, 0, 0, W, N * N * 2);
    tu_dpi_sram_write(h, 1, 0, A, N * N * 2);
    tu_dpi_sram_write(h, 2, 0, O, N * N * 4);

    int cmd = tu_dpi_submit_gemm(h, N, N, N, 0, 0, 0, 0);
    ASSERT(cmd >= 1, "submit GEMM failed");

    int ret = tu_dpi_wait(h, cmd, 1000);
    ASSERT_EQ(ret, TU_DPI_OK, "wait failed");

    ret = tu_dpi_sync(h);
    ASSERT_EQ(ret, TU_DPI_OK, "sync failed");

    free(W); free(A); free(O);
    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 11: Reset and Re-use ---- */

static void test_reset(void) {
    TEST("reset and re-use");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    /* Run a small GEMM */
    uint16_t W[4] = {fp16_one(), 0, 0, fp16_one()};
    uint16_t A[4] = {fp16_one(), 0, 0, fp16_one()};
    float O[4] = {0};
    tu_dpi_sram_write(h, 0, 0, W, 8);
    tu_dpi_sram_write(h, 1, 0, A, 8);
    tu_dpi_sram_write(h, 2, 0, O, 16);
    tu_dpi_gemm(h, 2, 2, 2, 0, 0, 0, 0);

    long long calls1 = tu_dpi_read_counter(h, TU_DPI_CNT_MMA_CALLS);
    ASSERT(calls1 >= 1, "first GEMM should register");

    /* Reset */
    tu_dpi_reset(h);

    long long calls2 = tu_dpi_read_counter(h, TU_DPI_CNT_MMA_CALLS);
    ASSERT(calls2 == 0, "MMA calls should be 0 after reset");

    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 12: Error Handling ---- */

static void test_error_handling(void) {
    TEST("error handling for invalid inputs");

    /* Invalid handle */
    int ret = tu_dpi_read_counter(999, TU_DPI_CNT_DMA_BYTES);
    ASSERT(ret < 0, "invalid handle should error");

    ret = tu_dpi_sram_write(42, 0, 0, NULL, 0);
    ASSERT(ret < 0, "invalid handle should error");

    /* Invalid parameters */
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);
    ASSERT(h >= 1, "init failed");

    ret = tu_dpi_sram_write(h, 5, 0, NULL, 0); /* invalid region */
    ASSERT(ret < 0, "invalid region should error");

    ret = tu_dpi_set_dataflow(h, 99); /* invalid dataflow */
    ASSERT(ret < 0, "invalid dataflow should error");

    /* Init with bad params */
    int h2 = tu_dpi_init(0, 16, 256, TU_DPI_DF_WS);
    ASSERT(h2 < 0, "zero PE rows should error");

    h2 = tu_dpi_init(300, 16, 256, TU_DPI_DF_WS);
    ASSERT(h2 < 0, "too many PE rows should error");

    tu_dpi_destroy(h);
    PASS;
}

/* ---- Test 13: Summary String ---- */

static void test_summary(void) {
    TEST("summary string generation");
    int h = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);

    int N = 4;
    uint16_t W[16]; uint16_t A[16]; float O[16];
    for (int i = 0; i < 16; i++) { W[i] = A[i] = fp16_one(); O[i] = 0; }
    tu_dpi_sram_write(h, 0, 0, W, 32);
    tu_dpi_sram_write(h, 1, 0, A, 32);
    tu_dpi_sram_write(h, 2, 0, O, 64);
    tu_dpi_gemm(h, N, N, N, 0, 0, 0, 0);

    char summary[256];
    int ret = tu_dpi_get_summary(h, summary, sizeof(summary));
    ASSERT_EQ(ret, TU_DPI_OK, "get_summary failed");
    ASSERT(strlen(summary) > 0, "summary should not be empty");

    tu_dpi_destroy(h);
    PASS;
}

/* ---- Main ---- */

int main(void) {
    printf("\n=== TU DPI-C Integration Tests (Gap I1) ===\n\n");

    test_lifecycle();
    test_multi_instance();
    test_memory_rw();
    test_gemm();
    test_elementwise();
    test_softmax();
    test_layernorm();
    test_counters();
    test_dataflow_switch();
    test_async_cmd();
    test_reset();
    test_error_handling();
    test_summary();

    printf("\n=== Results: %d passed, %d failed ===\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
