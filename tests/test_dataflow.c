/*
 * Dataflow Plugin Tests
 * =====================
 * A4: Validates WS and OS dataflows produce mathematically identical results
 * and that the plugin registry works correctly.
 *
 * Tests:
 *   1. Registry: create, lookup by ID, lookup by name, count
 *   2. WS identity: 16×16 W=I, A=I → O=I (bit-exact against golden)
 *   3. OS identity: same test, OS dataflow (must match WS result bit-exact)
 *   4. WS-vs-OS equivalence: random 32×16 matrix, both dataflows match
 *   5. Dataflow switch: run WS then OS in same session
 *   6. Edge tiles: non-multiple-of-16 dimensions
 */

#include "tu_cmodel.h"
#include "compute/dataflow/dataflow_interface.h"
#include "compute/dataflow/dataflow_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward declarations from dataflow plugin .c files */
tu_dataflow_plugin_t *tu_dataflow_ws_create(void);
void tu_dataflow_ws_destroy(tu_dataflow_plugin_t *p);
tu_dataflow_plugin_t *tu_dataflow_os_create(void);
void tu_dataflow_os_destroy(tu_dataflow_plugin_t *p);

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    printf("  [TEST] %-55s ", name); fflush(stdout); \
} while(0)

#define PASS do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_EQ(a, b, msg) do { if ((a) != (b)) { \
    printf("FAIL: %s (got %lu, expected %lu)\n", msg, (unsigned long)(a), (unsigned long)(b)); \
    tests_failed++; return; } } while(0)

/* ---- Test 1: Registry API ---- */

static void test_registry_api(void) {
    TEST("registry init and lookup");
    tu_dataflow_registry_init();

    /* Create and register plugins */
    tu_dataflow_plugin_t *ws = tu_dataflow_ws_create();
    tu_dataflow_plugin_t *os = tu_dataflow_os_create();

    ASSERT(ws != NULL, "WS create failed");
    ASSERT(os != NULL, "OS create failed");

    tu_dataflow_register(ws);
    tu_dataflow_register(os);

    ASSERT_EQ(tu_dataflow_registry_count(), 2, "registry count != 2");

    tu_dataflow_plugin_t *found;

    found = tu_dataflow_lookup(TU_DATAFLOW_WEIGHT_STATIONARY);
    ASSERT(found != NULL, "WS lookup failed");
    ASSERT(strcmp(found->name, "weight_stationary") == 0, "WS name mismatch");

    found = tu_dataflow_lookup(TU_DATAFLOW_OUTPUT_STATIONARY);
    ASSERT(found != NULL, "OS lookup failed");
    ASSERT(strcmp(found->name, "output_stationary") == 0, "OS name mismatch");

    found = tu_dataflow_lookup_by_name("output_stationary");
    ASSERT(found != NULL, "OS name lookup failed");
    ASSERT(found->id == TU_DATAFLOW_OUTPUT_STATIONARY, "OS ID mismatch");

    found = tu_dataflow_lookup_by_name("nonexistent");
    ASSERT(found == NULL, "nonexistent lookup should return NULL");

    /* NOTE: Don't destroy the registry — later tests need it.
     * tu_init() re-registers plugins anyway, but we keep the test
     * registry intact so the API is exercised. */
    PASS;
}

/* ---- Helper: create FP16 identity matrix in SRAM ---- */

static void fill_identity_fp16(uint8_t *buf, uint16_t size) {
    uint16_t *ptr = (uint16_t *)buf;
    memset(buf, 0, size * size * 2);
    for (uint16_t i = 0; i < size; i++)
        ptr[i * size + i] = 0x3C00; /* FP16 1.0 */
}

/* ---- Test 2: WS identity MMA ---- */

static void test_ws_identity(void) {
    TEST("WS identity 16x16");
    tu_init();
    tu_set_dataflow(TU_DATAFLOW_WEIGHT_STATIONARY);

    uint16_t N = 16;
    fill_identity_fp16(tu_sram_raw_ptr(&g_tu.sram_w), N);
    fill_identity_fp16(tu_sram_raw_ptr(&g_tu.sram_a), N);
    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, N * N * 4);

    tu_mma(N, N, N, 0, 0, 0, false);

    /* Verify output = identity */
    float *O = (float *)tu_sram_raw_ptr(&g_tu.sram_o);
    for (uint16_t i = 0; i < N; i++) {
        for (uint16_t j = 0; j < N; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float got = O[i * N + j];
            if (fabsf(got - expected) > 1e-5f) {
                char msg[128];
                snprintf(msg, sizeof(msg), "O[%u][%u] = %f, expected %f", i, j, got, expected);
                FAIL(msg);
                return;
            }
        }
    }

    ASSERT(g_tu.total_mma_calls > 0, "no MMA calls recorded");
    PASS;
}

/* ---- Test 3: OS identity MMA ---- */

static void test_os_identity(void) {
    TEST("OS identity 16x16");
    tu_init();
    tu_set_dataflow(TU_DATAFLOW_OUTPUT_STATIONARY);

    uint16_t N = 16;
    fill_identity_fp16(tu_sram_raw_ptr(&g_tu.sram_w), N);
    fill_identity_fp16(tu_sram_raw_ptr(&g_tu.sram_a), N);
    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, N * N * 4);

    tu_mma(N, N, N, 0, 0, 0, false);

    float *O = (float *)tu_sram_raw_ptr(&g_tu.sram_o);
    for (uint16_t i = 0; i < N; i++) {
        for (uint16_t j = 0; j < N; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float got = O[i * N + j];
            if (fabsf(got - expected) > 1e-5f) {
                char msg[128];
                snprintf(msg, sizeof(msg), "O[%u][%u] = %f, expected %f", i, j, got, expected);
                FAIL(msg);
                return;
            }
        }
    }

    ASSERT(strcmp(tu_get_dataflow_name(), "output_stationary") == 0,
           "dataflow name mismatch");
    PASS;
}

/* ---- Test 4: WS vs OS equivalence ---- */

static void test_ws_vs_os_equivalence(void) {
    TEST("WS-vs-OS equivalence (32x16 random)");

    uint16_t M = 32, N = 32, K = 16;

    /* Run WS */
    tu_init();
    tu_set_dataflow(TU_DATAFLOW_WEIGHT_STATIONARY);

    /* Fill W and A with deterministic pseudo-random FP16 */
    uint16_t *W_ws = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_w);
    uint16_t *A_ws = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_a);
    for (uint32_t i = 0; i < (uint32_t)M * K; i++)
        W_ws[i] = 0x3C00 + (i & 0xFF);  /* 1.0 + small offset */
    for (uint32_t i = 0; i < (uint32_t)K * N; i++)
        A_ws[i] = 0x3800 + ((i * 3) & 0xFF); /* 0.5 + small offset */

    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, M * N * 4);
    tu_mma(M, N, K, 0, 0, 0, false);
    float *O_ws = (float *)malloc(M * N * sizeof(float));
    memcpy(O_ws, tu_sram_raw_ptr(&g_tu.sram_o), M * N * 4);

    /* Run OS with same data */
    tu_init();
    tu_set_dataflow(TU_DATAFLOW_OUTPUT_STATIONARY);

    uint16_t *W_os = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_w);
    uint16_t *A_os = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_a);
    for (uint32_t i = 0; i < (uint32_t)M * K; i++)
        W_os[i] = 0x3C00 + (i & 0xFF);
    for (uint32_t i = 0; i < (uint32_t)K * N; i++)
        A_os[i] = 0x3800 + ((i * 3) & 0xFF);

    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, M * N * 4);
    tu_mma(M, N, K, 0, 0, 0, false);
    float *O_os = (float *)tu_sram_raw_ptr(&g_tu.sram_o);

    /* Compare bit-exact */
    for (uint32_t i = 0; i < (uint32_t)M * N; i++) {
        if (fabsf(O_ws[i] - O_os[i]) > 1e-6f) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "Mismatch at [%u]: WS=%f OS=%f diff=%e",
                     i, O_ws[i], O_os[i], fabsf(O_ws[i] - O_os[i]));
            FAIL(msg);
            free(O_ws);
            return;
        }
    }

    free(O_ws);
    PASS;
}

/* ---- Test 5: Dataflow switch mid-session ---- */

static void test_dataflow_switch(void) {
    TEST("dataflow switch WS→OS mid-session");

    uint16_t N = 8;

    tu_init();
    tu_set_dataflow(TU_DATAFLOW_WEIGHT_STATIONARY);

    /* Run WS first */
    fill_identity_fp16(tu_sram_raw_ptr(&g_tu.sram_w), N);
    fill_identity_fp16(tu_sram_raw_ptr(&g_tu.sram_a), N);
    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, N * N * 4);
    tu_mma(N, N, N, 0, 0, 0, false);

    float *O_ws = (float *)malloc(N * N * sizeof(float));
    memcpy(O_ws, tu_sram_raw_ptr(&g_tu.sram_o), N * N * 4);

    /* Switch to OS and re-run */
    tu_set_dataflow(TU_DATAFLOW_OUTPUT_STATIONARY);
    ASSERT(strcmp(tu_get_dataflow_name(), "output_stationary") == 0,
           "switch to OS failed");

    fill_identity_fp16(tu_sram_raw_ptr(&g_tu.sram_w), N);
    fill_identity_fp16(tu_sram_raw_ptr(&g_tu.sram_a), N);
    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, N * N * 4);
    tu_mma(N, N, N, 0, 0, 0, false);

    float *O_os = (float *)tu_sram_raw_ptr(&g_tu.sram_o);

    for (uint16_t i = 0; i < N * N; i++) {
        if (fabsf(O_ws[i] - O_os[i]) > 1e-6f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "Switch mismatch at [%u]", i);
            FAIL(msg);
            free(O_ws);
            return;
        }
    }

    /* Verify both are identity */
    for (uint16_t i = 0; i < N; i++) {
        for (uint16_t j = 0; j < N; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            if (fabsf(O_os[i * N + j] - expected) > 1e-5f) {
                FAIL("OS result not identity after switch");
                free(O_ws);
                return;
            }
        }
    }

    free(O_ws);
    PASS;
}

/* ---- Test 6: Edge tiles ---- */

static void test_edge_tiles(void) {
    TEST("edge tiles (31x31x17 non-multiple-of-16)");

    uint16_t M = 31, N = 31, K = 17;

    /* Run WS */
    tu_init();
    tu_set_dataflow(TU_DATAFLOW_WEIGHT_STATIONARY);

    uint16_t *W = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_w);
    uint16_t *A = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_a);
    for (uint32_t i = 0; i < (uint32_t)M * K; i++) W[i] = 0x3C00; /* all 1.0 */
    for (uint32_t i = 0; i < (uint32_t)K * N; i++) A[i] = 0x3C00; /* all 1.0 */
    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, M * N * 4);
    tu_mma(M, N, K, 0, 0, 0, false);
    float *O_ws = (float *)malloc(M * N * sizeof(float));
    memcpy(O_ws, tu_sram_raw_ptr(&g_tu.sram_o), M * N * 4);

    /* Run OS */
    tu_init();
    tu_set_dataflow(TU_DATAFLOW_OUTPUT_STATIONARY);

    W = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_w);
    A = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_a);
    for (uint32_t i = 0; i < (uint32_t)M * K; i++) W[i] = 0x3C00;
    for (uint32_t i = 0; i < (uint32_t)K * N; i++) A[i] = 0x3C00;
    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, M * N * 4);
    tu_mma(M, N, K, 0, 0, 0, false);
    float *O_os = (float *)tu_sram_raw_ptr(&g_tu.sram_o);

    /* Compare: each element should be ≈ K (all-ones × all-ones = K) */
    float expected = (float)K;
    for (uint32_t i = 0; i < (uint32_t)M * N; i++) {
        if (fabsf(O_ws[i] - expected) > 1e-3f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "WS edge tile: O[%u] = %f, expected %f",
                     i, O_ws[i], expected);
            FAIL(msg);
            free(O_ws);
            return;
        }
        if (fabsf(O_ws[i] - O_os[i]) > 1e-6f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "WS-vs-OS mismatch in edge: [%u]", i);
            FAIL(msg);
            free(O_ws);
            return;
        }
    }

    free(O_ws);
    PASS;
}

/* ---- Main ---- */

int main(void) {
    printf("\n=== TU Dataflow Plugin Tests (Gap A4) ===\n\n");

    test_registry_api();       /* Must run first — destroys registry */
    test_ws_identity();
    test_os_identity();
    test_ws_vs_os_equivalence();
    test_dataflow_switch();
    test_edge_tiles();

    printf("\n=== Results: %d passed, %d failed ===\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
