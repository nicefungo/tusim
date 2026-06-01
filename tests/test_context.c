/*
 * TU Context Manager Tests (Gap E3)
 * ==================================
 *
 * Tests for multi-context execution: allocation, save/restore,
 * switching, scheduling, state isolation, and error handling.
 */

#include "test_framework.h"
#include "tu_cmodel.h"
#include "tu_cmodel/tu_core.h"
#include "tu_cmodel/infra/tu_context.h"

#include <string.h>
#include <stdlib.h>

/* Global test stats (required by test_framework.h) */
tu_test_stats_t g_test_stats;

/* Per-test assertion helper */
#define CHECK(cond, msg) do { if (!(cond)) { FAIL("%s", msg); return; } } while(0)

/* ---- Setup/teardown ---- */
static tu_core_t *g_core = NULL;
static tu_ctx_manager_t *g_mgr = NULL;
static tu_ctx_manager_config_t g_cfg;

static void setup(void) {
    /* tu_core_create handles its own initialization via tu_init_with_config.
     * Do NOT call tu_init() separately — it creates a g_tu state that
     * tu_core_create's save/restore pattern leaves dangling after core destroy. */
    tu_runtime_config_t rt_cfg = tu_runtime_config_default();
    g_core = tu_core_create(&rt_cfg);

    g_cfg = (tu_ctx_manager_config_t){
        .max_contexts      = 4,
        .sched_policy      = TU_CTX_SCHED_ROUND_ROBIN,
        .time_slice_cycles = 0,
        .time_slice_cmds   = 0,
        .switch_overhead   = 100,
        .save_dram_state   = false,
    };
    g_mgr = tu_ctx_manager_create(g_core, &g_cfg);
}

static void teardown(void) {
    if (g_mgr) { tu_ctx_manager_destroy(g_mgr); g_mgr = NULL; }
    if (g_core) { tu_core_destroy(g_core); g_core = NULL; }
}

/* ================================================================
 * Test 1: Context Manager Creation
 * ================================================================ */
static void test_create_manager(void) {
    CHECK(g_mgr != NULL, "ctx manager should be created");
    CHECK(tu_ctx_get_switch_count(g_mgr) == 0, "initial switch count should be 0");
    CHECK(tu_ctx_get_switch_overhead(g_mgr) == 0, "initial switch overhead should be 0");
    PASS();
}

/* ================================================================
 * Test 2: Context Allocation and Free
 * ================================================================ */
static void test_alloc_free(void) {
    int id0 = tu_ctx_alloc(g_mgr);
    CHECK(id0 == 0, "first context should get ID 0");

    int id1 = tu_ctx_alloc(g_mgr);
    CHECK(id1 == 1, "second context should get ID 1");

    int id2 = tu_ctx_alloc(g_mgr);
    CHECK(id2 == 2, "third context should get ID 2");

    int id3 = tu_ctx_alloc(g_mgr);
    CHECK(id3 == 3, "fourth context should get ID 3");

    /* All slots full */
    int id4 = tu_ctx_alloc(g_mgr);
    CHECK(id4 == -1, "fifth alloc should fail (max 4)");

    /* Free and re-allocate */
    tu_ctx_free(g_mgr, 1);
    int id1b = tu_ctx_alloc(g_mgr);
    CHECK(id1b == 1, "re-alloc should reuse freed slot 1");

    PASS();
}

/* ================================================================
 * Test 3: Context State Transitions
 * ================================================================ */
static void test_state_transitions(void) {
    int id0 = tu_ctx_alloc(g_mgr);
    CHECK(id0 == 0, "should allocate ctx 0");

    tu_context_desc_t *ctx = tu_ctx_get(g_mgr, 0);
    CHECK(ctx != NULL, "ctx 0 should be accessible");
    CHECK(ctx->state == TU_CTX_ACTIVE, "first ctx should be ACTIVE");
    CHECK(ctx->ctx_id == 0, "ctx_id should match");
    CHECK(ctx->priority == 128, "default priority should be 128");

    /* Allocate more contexts — they should be READY */
    int id1 = tu_ctx_alloc(g_mgr);
    CHECK(id1 == 1, "ctx 1 allocated");
    tu_context_desc_t *ctx1 = tu_ctx_get(g_mgr, 1);
    CHECK(ctx1->state == TU_CTX_READY, "ctx 1 should be READY");

    PASS();
}

/* ================================================================
 * Test 4: Context Save and Restore (SRAM data isolation)
 * ================================================================ */
static void test_save_restore_sram(void) {
    /* Allocate two contexts */
    int id0 = tu_ctx_alloc(g_mgr);
    int id1 = tu_ctx_alloc(g_mgr);
    CHECK(id0 == 0 && id1 == 1, "two contexts allocated");

    /* Write data to SRAM in active context (ctx 0) */
    float host_w[16];
    float host_a[16];
    for (int i = 0; i < 16; i++) {
        host_w[i] = (float)(i + 1);
        host_a[i] = (float)(i + 100);
    }
    tu_core_dma_load_w(g_core, host_w, 0, 16 * sizeof(float));
    tu_core_dma_load_a(g_core, host_a, 0, 16 * sizeof(float));

    /* Save context 0 */
    int ret = tu_ctx_save(g_mgr);
    CHECK(ret == 0, "save ctx 0 should succeed");

    tu_context_desc_t *ctx0 = tu_ctx_get(g_mgr, 0);
    CHECK(ctx0->state == TU_CTX_READY, "ctx 0 should be READY after save");

    /* Switch to context 1 */
    ret = tu_ctx_restore(g_mgr, 1);
    CHECK(ret == 0, "restore ctx 1 should succeed");

    /* Context 1 should have empty SRAM (initial state was copied at alloc time) */
    float out_w[16], out_a[16];
    tu_sram_region_t *sw = tu_core_get_sram_w(g_core);
    tu_sram_read_bulk(sw, 0, out_w, 16 * sizeof(float));
    tu_sram_region_t *sa = tu_core_get_sram_a(g_core);
    tu_sram_read_bulk(sa, 0, out_a, 16 * sizeof(float));

    /* Switch back to context 0 */
    ret = tu_ctx_save(g_mgr);  /* save ctx 1 */
    CHECK(ret == 0, "save ctx 1 should succeed");
    ret = tu_ctx_restore(g_mgr, 0);
    CHECK(ret == 0, "restore ctx 0 should succeed");

    /* Verify context 0 has our data back */
    tu_sram_read_bulk(sw, 0, out_w, 16 * sizeof(float));
    tu_sram_read_bulk(sa, 0, out_a, 16 * sizeof(float));
    CHECK(out_w[0] == 1.0f, "ctx 0 W[0] should be 1.0 after restore");
    CHECK(out_w[15] == 16.0f, "ctx 0 W[15] should be 16.0 after restore");
    CHECK(out_a[0] == 100.0f, "ctx 0 A[0] should be 100.0 after restore");

    PASS();
}

/* ================================================================
 * Test 5: Context Switching with tu_ctx_switch
 * ================================================================ */
static void test_context_switch(void) {
    int id0 = tu_ctx_alloc(g_mgr);
    int id1 = tu_ctx_alloc(g_mgr);
    CHECK(id0 == 0 && id1 == 1, "two contexts allocated");

    /* Write to SRAM in ctx 0 */
    float w0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tu_core_dma_load_w(g_core, w0, 0, 4 * sizeof(float));

    /* Switch to ctx 1 */
    int ret = tu_ctx_switch(g_mgr, 1);
    CHECK(ret == 0, "switch to ctx 1 should succeed");

    /* Write different data in ctx 1 */
    float w1[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    tu_core_dma_load_w(g_core, w1, 0, 4 * sizeof(float));

    /* Switch back to ctx 0 and verify isolation */
    ret = tu_ctx_switch(g_mgr, 0);
    CHECK(ret == 0, "switch back to ctx 0 should succeed");

    float out[4];
    tu_sram_region_t *sw = tu_core_get_sram_w(g_core);
    tu_sram_read_bulk(sw, 0, out, 4 * sizeof(float));
    CHECK(out[0] == 1.0f, "ctx 0 should have original data");
    CHECK(out[3] == 4.0f, "ctx 0 data should be isolated from ctx 1");

    /* Switch to ctx 1 and verify its data */
    ret = tu_ctx_switch(g_mgr, 1);
    CHECK(ret == 0, "switch back to ctx 1");
    tu_sram_read_bulk(sw, 0, out, 4 * sizeof(float));
    CHECK(out[0] == 10.0f, "ctx 1 should have its data");
    CHECK(out[3] == 40.0f, "ctx 1 data preserved");

    CHECK(tu_ctx_get_switch_count(g_mgr) == 3, "should have 3 switches");

    PASS();
}

/* ================================================================
 * Test 6: Round-robin Scheduling
 * ================================================================ */
static void test_round_robin_sched(void) {
    /* Allocate 3 contexts */
    tu_ctx_alloc(g_mgr);  /* id 0, active */
    tu_ctx_alloc(g_mgr);  /* id 1, ready */
    tu_ctx_alloc(g_mgr);  /* id 2, ready */

    /* Schedule next: should get 1 (after active 0) */
    int next = tu_ctx_schedule_next(g_mgr);
    CHECK(next == 1, "round-robin next after 0 should be 1");

    /* Switch, then schedule again */
    tu_ctx_switch(g_mgr, 1);
    next = tu_ctx_schedule_next(g_mgr);
    CHECK(next == 2, "round-robin next after 1 should be 2");

    tu_ctx_switch(g_mgr, 2);
    next = tu_ctx_schedule_next(g_mgr);
    CHECK(next == 0, "round-robin should wrap to 0");

    PASS();
}

/* ================================================================
 * Test 7: Priority Scheduling
 * ================================================================ */
static void test_priority_sched(void) {
    /* Re-create manager with priority scheduling */
    tu_ctx_manager_destroy(g_mgr);
    g_cfg.sched_policy = TU_CTX_SCHED_PRIORITY;
    g_mgr = tu_ctx_manager_create(g_core, &g_cfg);
    CHECK(g_mgr != NULL, "priority-sched manager created");

    /* Allocate contexts with different priorities */
    int id0 = tu_ctx_alloc(g_mgr);
    tu_ctx_get(g_mgr, 0)->priority = 50;

    int id1 = tu_ctx_alloc(g_mgr);
    tu_ctx_get(g_mgr, 1)->priority = 200;

    int id2 = tu_ctx_alloc(g_mgr);
    tu_ctx_get(g_mgr, 2)->priority = 100;

    (void)id0; (void)id1; (void)id2;

    /* Schedule: should pick highest priority READY ctx */
    int next = tu_ctx_schedule_next(g_mgr);
    CHECK(next == 1, "priority sched should pick ctx 1 (priority 200)");
    CHECK(tu_ctx_get(g_mgr, 1)->priority == 200, "ctx 1 prio verification");

    PASS();
}

/* ================================================================
 * Test 8: Block and Unblock Contexts
 * ================================================================ */
static void test_block_unblock(void) {
    int id0 = tu_ctx_alloc(g_mgr);
    int id1 = tu_ctx_alloc(g_mgr);
    (void)id0;

    /* Block the active context */
    int ret = tu_ctx_block_current(g_mgr);
    CHECK(ret == 0, "block current should succeed");

    tu_context_desc_t *ctx0 = tu_ctx_get(g_mgr, 0);
    CHECK(ctx0->state == TU_CTX_BLOCKED, "ctx 0 should be BLOCKED");

    /* Schedule: should skip blocked context, pick ctx 1 */
    int next = tu_ctx_schedule_next(g_mgr);
    CHECK(next == 1, "should skip blocked ctx 0, pick ctx 1");

    /* Unblock ctx 0 */
    ret = tu_ctx_unblock(g_mgr, 0);
    CHECK(ret == 0, "unblock should succeed");
    CHECK(ctx0->state == TU_CTX_READY, "ctx 0 should be READY after unblock");

    PASS();
}

/* ================================================================
 * Test 9: State Isolation Across Contexts (SRAM + Stats)
 * ================================================================ */
static void test_perf_isolation(void) {
    int id0 = tu_ctx_alloc(g_mgr);
    int id1 = tu_ctx_alloc(g_mgr);

    /* Write data and track DMA bytes in ctx 0 */
    float w[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tu_core_dma_load_w(g_core, w, 0, 4 * sizeof(float));
    uint64_t ctx0_dma_bytes = g_core->state.total_dma_bytes;
    CHECK(ctx0_dma_bytes > 0, "ctx 0 should have DMA bytes");

    /* Switch to ctx 1 — its DMA counter should be 0 (fresh) */
    tu_ctx_switch(g_mgr, (uint32_t)id1);

    uint64_t ctx1_dma_bytes_initial = g_core->state.total_dma_bytes;
    CHECK(ctx1_dma_bytes_initial == 0, "ctx 1 should start with 0 DMA bytes");

    /* Do work in ctx 1 */
    float w1[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    tu_core_dma_load_w(g_core, w1, 0, 4 * sizeof(float));
    uint64_t ctx1_dma_bytes = g_core->state.total_dma_bytes;
    CHECK(ctx1_dma_bytes > 0, "ctx 1 should have its own DMA bytes");

    /* Switch back to ctx 0, verify its counter is restored */
    tu_ctx_switch(g_mgr, (uint32_t)id0);
    uint64_t ctx0_restored = g_core->state.total_dma_bytes;
    CHECK(ctx0_restored == ctx0_dma_bytes,
          "ctx 0 DMA bytes should be restored to pre-switch value");

    (void)id0; (void)id1;
    PASS();
}

/* ================================================================
 * Test 10: Status Printing
 * ================================================================ */
static void test_print_status(void) {
    tu_ctx_alloc(g_mgr);
    tu_ctx_alloc(g_mgr);

    /* Just verify it doesn't crash */
    tu_ctx_print_status(g_mgr, stdout);

    uint64_t count = tu_ctx_get_switch_count(g_mgr);
    CHECK(count == 0, "no switches yet");

    PASS();
}

/* ================================================================
 * Test 11: Null pointer safety
 * ================================================================ */
static void test_null_safety(void) {
    int ret = tu_ctx_alloc(NULL);
    CHECK(ret == -1, "alloc NULL should return -1");

    tu_ctx_free(NULL, 0);  /* Should not crash */

    tu_context_desc_t *ctx = tu_ctx_get(NULL, 0);
    CHECK(ctx == NULL, "get NULL should return NULL");

    ret = tu_ctx_save(NULL);
    CHECK(ret == -1, "save NULL should return -1");

    ret = tu_ctx_restore(NULL, 0);
    CHECK(ret == -1, "restore NULL should return -1");

    ret = tu_ctx_switch(NULL, 0);
    CHECK(ret == -1, "switch NULL should return -1");

    int next = tu_ctx_schedule_next(NULL);
    CHECK(next == -1, "sched NULL should return -1");

    tu_ctx_notify_command(NULL);  /* No crash */
    tu_ctx_notify_cycles(NULL, 100);  /* No crash */
    tu_ctx_print_status(NULL, stdout);  /* No crash */

    PASS();
}

/* ================================================================
 * Test 12: Maximum contexts edge case
 * ================================================================ */
static void test_max_contexts(void) {
    /* Create manager with only 2 slots */
    tu_ctx_manager_destroy(g_mgr);
    g_cfg.max_contexts = 2;
    g_mgr = tu_ctx_manager_create(g_core, &g_cfg);
    CHECK(g_mgr != NULL, "2-slot manager created");

    int id0 = tu_ctx_alloc(g_mgr);
    int id1 = tu_ctx_alloc(g_mgr);
    CHECK(id0 == 0 && id1 == 1, "two contexts allocated");

    int id2 = tu_ctx_alloc(g_mgr);
    CHECK(id2 == -1, "third alloc should fail");

    tu_ctx_free(g_mgr, 0);
    id2 = tu_ctx_alloc(g_mgr);
    CHECK(id2 == 0, "re-alloc after free should work");

    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
    test_stats_init();

    TEST("ctx_manager_create");    setup(); test_create_manager();    teardown();
    TEST("ctx_alloc_free");        setup(); test_alloc_free();        teardown();
    TEST("ctx_state_transitions");  setup(); test_state_transitions(); teardown();
    TEST("ctx_save_restore_sram");  setup(); test_save_restore_sram(); teardown();
    TEST("ctx_context_switch");    setup(); test_context_switch();    teardown();
    TEST("ctx_round_robin_sched"); setup(); test_round_robin_sched(); teardown();
    TEST("ctx_priority_sched");    setup(); test_priority_sched();    teardown();
    TEST("ctx_block_unblock");     setup(); test_block_unblock();     teardown();
    TEST("ctx_perf_isolation");    setup(); test_perf_isolation();    teardown();
    TEST("ctx_print_status");      setup(); test_print_status();      teardown();
    TEST("ctx_null_safety");       setup(); test_null_safety();       teardown();
    TEST("ctx_max_contexts");      setup(); test_max_contexts();      teardown();

    return test_exit();
}
