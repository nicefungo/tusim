/*
 * TinyTU Command Queue Tests
 * ===========================
 * Verifies:
 *   1. Queue creation and basic properties
 *   2. Submit and immediate execution (synchronous mode)
 *   3. Command ID tracking
 *   4. Barrier support
 *   5. Queue overflow behavior
 *   6. Command queue results match direct execution
 *   7. Dependency tracking
 */
#include "tu_cmodel/tu_cmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_pass = 0;

#define TEST(name) do { tests_run++; printf("  %-54s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_pass++; } while(0)
#define FAIL(fmt, ...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); } while(0)

/* ================================================================
 * Test 1: Queue creation and basic properties
 * ================================================================ */
static void test_cmdq_create(void) {
    TEST("Command queue creation");
    tu_command_queue_t *cq = tu_cmdq_create(16, true);
    if (!cq) { FAIL("NULL returned"); return; }
    if (tu_cmdq_get_depth(cq) != 0) { FAIL("depth != 0"); tu_cmdq_destroy(cq); return; }
    PASS();
    tu_cmdq_destroy(cq);
}

/* ================================================================
 * Test 2: Submit MMA via command queue (synchronous mode)
 * ================================================================ */
static void test_cmdq_mma_sync(void) {
    TEST("CMDQ MMA identity 16×16×16");

    tu_runtime_config_t cfg = tu_config_default();
    tu_init_with_config(&cfg);

    /* Load identity matrices */
    fp16_t W[16 * 16], A[16 * 16];
    memset(W, 0, sizeof(W)); memset(A, 0, sizeof(A));
    for (int i = 0; i < 16; i++) {
        W[i * 16 + i] = fp32_to_fp16(1.0f);
        A[i * 16 + i] = fp32_to_fp16(1.0f);
    }

    tu_dma_load_w(W, 0, sizeof(W));
    tu_dma_load_a(A, 0, sizeof(A));

    /* Submit MMA via command queue */
    int cmd_id = tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);
    if (cmd_id <= 0) { FAIL("submit failed, rc=%d", cmd_id); return; }

    /* Verify it completed immediately (sync mode) */
    tu_cmd_status_t st = tu_cmdq_get_status(g_tu.cmdq, (uint32_t)cmd_id);
    if (st != TU_CMD_COMPLETED) { FAIL("status=%d, expected COMPLETED", st); return; }

    /* Read results */
    fp32_t O[16 * 16];
    tu_dma_store_o(O, 0, sizeof(O));

    /* Verify identity */
    int ok = 1;
    for (int i = 0; i < 16 * 16 && ok; i++) {
        int row = i / 16, col = i % 16;
        float expected = (row == col) ? 1.0f : 0.0f;
        if (fabsf(O[i] - expected) > 0.01f) {
            FAIL("O[%d] = %f, expected %f", i, O[i], expected);
            ok = 0;
        }
    }
    if (ok) PASS();

    /* Verify counters */
    uint64_t submitted, completed, faulted;
    tu_cmdq_get_counts(g_tu.cmdq, &submitted, &completed, &faulted);
    if (submitted != 1 || completed != 1 || faulted != 0)
        fprintf(stderr, "\n         (counts: sub=%lu comp=%lu fault=%lu)", submitted, completed, faulted);
}

/* ================================================================
 * Test 3: Command ID monotonicity
 * ================================================================ */
static void test_cmdq_monotonic_ids(void) {
    TEST("Command ID monotonicity");

    tu_runtime_config_t cfg = tu_config_default();
    tu_init_with_config(&cfg);

    int id1 = tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);
    int id2 = tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);
    int id3 = tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);

    if (id1 > 0 && id2 > id1 && id3 > id2) PASS();
    else FAIL("ids: %d %d %d (expected monotonic)", id1, id2, id3);
}

/* ================================================================
 * Test 4: Barrier support
 * ================================================================ */
static void test_cmdq_barrier(void) {
    TEST("Command queue barrier");

    tu_runtime_config_t cfg = tu_config_default();
    tu_init_with_config(&cfg);

    /* Submit commands, barrier between them */
    int id1 = tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);
    int bid  = tu_cmdq_submit_barrier();
    int id2 = tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);

    if (id1 <= 0 || bid <= 0 || id2 <= 0) { FAIL("submit failed"); return; }
    if (bid <= id1) { FAIL("barrier id %d not > cmd1 id %d", bid, id1); return; }
    if (id2 <= bid) { FAIL("cmd2 id %d not > barrier id %d", id2, bid); return; }

    /* In sync mode, all should be completed */
    tu_cmd_status_t s1 = tu_cmdq_get_status(g_tu.cmdq, (uint32_t)id1);
    tu_cmd_status_t sb = tu_cmdq_get_status(g_tu.cmdq, (uint32_t)bid);
    tu_cmd_status_t s2 = tu_cmdq_get_status(g_tu.cmdq, (uint32_t)id2);

    if (s1 == TU_CMD_COMPLETED && sb == TU_CMD_COMPLETED && s2 == TU_CMD_COMPLETED) PASS();
    else FAIL("statuses: cmd1=%d barrier=%d cmd2=%d", s1, sb, s2);
}

/* ================================================================
 * Test 5: Queue overflow behavior
 * ================================================================ */
static void test_cmdq_overflow(void) {
    TEST("Command queue overflow");

    tu_runtime_config_t cfg = tu_config_default();
    tu_init_with_config(&cfg);

    /* Fill the queue */
    int count = 0;
    int rc;
    do {
        rc = tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);
        if (rc > 0) count++;
    } while (rc > 0);

    /* Queue should reject when full */
    if (rc == -1 && count > 0) PASS();
    else FAIL("filled %d commands, last rc=%d", count, rc);
}

/* ================================================================
 * Test 6: Command queue via DMA
 * ================================================================ */
static void test_cmdq_dma(void) {
    TEST("CMDQ DMA load + MMA");

    tu_runtime_config_t cfg = tu_config_default();
    tu_init_with_config(&cfg);

    fp16_t W[16 * 16];
    for (int i = 0; i < 16 * 16; i++)
        W[i] = fp32_to_fp16(0.5f);

    fp16_t A[16 * 8];
    for (int i = 0; i < 16 * 8; i++)
        A[i] = fp32_to_fp16(2.0f);

    /* Submit DMA via command queue */
    int dma1 = tu_cmdq_submit_dma_load(0, 0, W, sizeof(W));
    int dma2 = tu_cmdq_submit_dma_load(1, 0, A, sizeof(A));

    if (dma1 <= 0 || dma2 <= 0) { FAIL("DMA submit failed"); return; }

    /* MMA depends on both DMAs */
    uint32_t deps[] = { (uint32_t)dma1, (uint32_t)dma2 };
    tu_cmd_mma_desc_t mma_desc = { .M = 16, .N = 8, .K = 16,
        .w_offset = 0, .a_offset = 0, .o_offset = 0, .has_bias = false };
    uint32_t mma_id;
    int rc = tu_cmdq_submit(g_tu.cmdq, TU_CMD_MMA, &mma_desc, 2, deps, &mma_id);
    if (rc < 0) { FAIL("MMA submit with deps failed"); return; }

    /* Sync and verify */
    fp32_t O[16 * 8];
    tu_dma_store_o(O, 0, sizeof(O));

    int ok = 1;
    for (int i = 0; i < 16 * 8 && ok; i++) {
        if (fabsf(O[i] - 16.0f) > 0.1f) {
            FAIL("O[%d] = %f, expected 16.0", i, O[i]);
            ok = 0;
        }
    }
    if (ok) PASS();
}

/* ================================================================
 * Test 7: Independent queue (not global)
 * ================================================================ */
static void test_cmdq_standalone(void) {
    TEST("Standalone command queue");
    tu_command_queue_t *cq = tu_cmdq_create(8, true);
    if (!cq) { FAIL("create failed"); return; }

    /* Queue should be empty */
    if (tu_cmdq_get_depth(cq) != 0) { FAIL("depth != 0"); tu_cmdq_destroy(cq); return; }

    /* Submit a NOP */
    uint32_t id;
    int rc = tu_cmdq_submit(cq, TU_CMD_NOP, NULL, 0, NULL, &id);
    if (rc < 0) { FAIL("submit failed"); tu_cmdq_destroy(cq); return; }

    uint64_t sub, comp, fault;
    tu_cmdq_get_counts(cq, &sub, &comp, &fault);
    if (sub == 1 && comp == 1 && fault == 0) PASS();
    else FAIL("counts: sub=%lu comp=%lu fault=%lu", sub, comp, fault);

    tu_cmdq_destroy(cq);
}

/* ================================================================
 * Test 8: Reset behavior
 * ================================================================ */
static void test_cmdq_reset(void) {
    TEST("Command queue reset");

    tu_runtime_config_t cfg = tu_config_default();
    tu_init_with_config(&cfg);

    /* Submit some commands */
    tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);
    tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);

    uint64_t sub, comp, fault;
    tu_cmdq_get_counts(g_tu.cmdq, &sub, &comp, &fault);
    if (sub < 2) { FAIL("expected >=2 submissions, got %lu", sub); return; }

    /* Reset */
    tu_cmdq_reset(g_tu.cmdq);
    tu_cmdq_get_counts(g_tu.cmdq, &sub, &comp, &fault);
    if (sub == 0 && comp == 0 && fault == 0) PASS();
    else FAIL("after reset: sub=%lu comp=%lu fault=%lu", sub, comp, fault);
}

/* ================================================================
 * Test 9: NOP commands
 * ================================================================ */
static void test_cmdq_nop(void) {
    TEST("NOP command execution");
    tu_command_queue_t *cq = tu_cmdq_create(8, true);

    uint32_t id;
    int rc = tu_cmdq_submit(cq, TU_CMD_NOP, NULL, 0, NULL, &id);
    if (rc < 0) { FAIL("submit failed"); tu_cmdq_destroy(cq); return; }

    tu_cmd_status_t st = tu_cmdq_get_status(cq, id);
    if (st == TU_CMD_COMPLETED) PASS();
    else FAIL("status=%d", st);

    tu_cmdq_destroy(cq);
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
    printf("TinyTU Command Queue Tests\n");
    printf("===========================\n\n");

    test_cmdq_create();
    test_cmdq_mma_sync();
    test_cmdq_monotonic_ids();
    test_cmdq_barrier();
    test_cmdq_overflow();
    test_cmdq_dma();
    test_cmdq_standalone();
    test_cmdq_reset();
    test_cmdq_nop();

    printf("\n═══════════════════════════════════════════\n");
    printf("  %d/%d tests passed\n", tests_pass, tests_run);
    printf("═══════════════════════════════════════════\n");
    return tests_pass == tests_run ? 0 : 1;
}
