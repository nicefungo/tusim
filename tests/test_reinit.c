/*
 * TinyTU global lifecycle and re-initialization tests.
 */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_core.h"
#include <stdio.h>
#include <string.h>

static int tests_run;
static int tests_pass;

#define TEST(name) do { tests_run++; printf("  %-58s ", name); } while (0)
#define PASS() do { tests_pass++; printf("PASS\n"); } while (0)
#define FAIL(fmt, ...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); } while (0)

static void test_repeated_idle_reinit(void) {
    TEST("Repeated init releases queue-owned allocations");
    tu_runtime_config_t cfg = tu_runtime_config_default();

    for (uint32_t i = 0; i < 64; i++) {
        tu_init_with_config(&cfg);
        uint32_t missing_dep = 0x80000000u + i;
        if (tu_cmdq_submit(g_tu.cmdq, TU_CMD_NOP, NULL, 1,
                           &missing_dep, NULL) <= 0) {
            FAIL("submit failed at iteration %u", i);
            tu_shutdown();
            return;
        }
    }

    tu_shutdown();
    if (g_tu.initialized || g_tu.cmdq != NULL ||
        g_tu.sram_w.banks.data != NULL || g_tu_dma.num_channels != 0) {
        FAIL("shutdown did not clear global state");
        return;
    }
    PASS();
}

static void test_reinit_drains_dma_before_sram_release(void) {
    TEST("Re-init drains accepted DMA before old SRAM release");
    tu_runtime_config_t cfg = tu_runtime_config_default();
    cfg.dma_async_mode = true;
    cfg.dma_num_channels = 1;
    cfg.dma_max_outstanding = 4;
    tu_init_with_config(&cfg);

    uint8_t expected_a[32], expected_b[32], actual_a[32], actual_b[32];
    for (uint32_t i = 0; i < sizeof(expected_a); i++) {
        expected_a[i] = (uint8_t)(0x20u + i);
        expected_b[i] = (uint8_t)(0xa0u + i);
    }
    memset(actual_a, 0, sizeof(actual_a));
    memset(actual_b, 0, sizeof(actual_b));
    memcpy(tu_sram_raw_ptr(&g_tu.sram_o), expected_a, sizeof(expected_a));
    memcpy(tu_sram_raw_ptr(&g_tu.sram_o) + 64, expected_b, sizeof(expected_b));

    tu_dma_descriptor_t *a = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_TU_TO_HOST, &g_tu.sram_o, 0,
        actual_a, 1, sizeof(actual_a));
    tu_dma_descriptor_t *b = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_TU_TO_HOST, &g_tu.sram_o, 64,
        actual_b, 1, sizeof(actual_b));
    if (!a || !b || tu_dma_submit_desc(a) == 0 ||
        tu_dma_submit_desc(b) == 0) {
        FAIL("descriptor setup or submit failed");
        if (a) { a->next = NULL; tu_dma_desc_destroy(a); }
        if (b) { b->next = NULL; tu_dma_desc_destroy(b); }
        tu_shutdown();
        return;
    }

    tu_init_with_config(&cfg);
    int ok = a->completed && b->completed &&
             memcmp(actual_a, expected_a, sizeof(actual_a)) == 0 &&
             memcmp(actual_b, expected_b, sizeof(actual_b)) == 0;

    /* Submission links caller-owned descriptors; detach before individual free. */
    a->next = NULL;
    b->next = NULL;
    tu_dma_desc_destroy(a);
    tu_dma_desc_destroy(b);
    tu_shutdown();

    if (!ok) {
        FAIL("accepted transfer was lost or observed stale SRAM");
        return;
    }
    PASS();
}

static void test_core_queue_teardown(void) {
    TEST("Core destruction releases complete command-queue storage");
    tu_runtime_config_t cfg = tu_runtime_config_default();
    tu_core_t *core = tu_core_create(&cfg);
    if (!core || !core->state.cmdq) {
        FAIL("core creation failed");
        tu_core_destroy(core);
        tu_shutdown();
        return;
    }
    tu_core_destroy(core);
    tu_shutdown();
    PASS();
}

static void test_shutdown_idempotent(void) {
    TEST("Shutdown is idempotent");
    tu_shutdown();
    tu_shutdown();
    if (g_tu.initialized || g_tu.cmdq || g_tu_dma.num_channels != 0) {
        FAIL("state revived after repeated shutdown");
        return;
    }
    PASS();
}

int main(void) {
    printf("TinyTU Lifecycle Tests\n");
    printf("======================\n\n");
    test_repeated_idle_reinit();
    test_reinit_drains_dma_before_sram_release();
    test_core_queue_teardown();
    test_shutdown_idempotent();
    printf("\n  %d/%d tests passed\n", tests_pass, tests_run);
    return tests_pass == tests_run ? 0 : 1;
}
