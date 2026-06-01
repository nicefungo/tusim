/*
 * TU CModel — Cycle-Accurate Model Tests (Gap P2.5)
 * ====================================================
 *
 * Tests:
 *   1. Pipeline tracker: issue/complete, hazard detection
 *   2. Pipeline utilization and stall accounting
 *   3. Bank conflict model: access, bandwidth exhaustion, refill
 *   4. DRAM row buffer: hit/miss/empty states
 *   5. DRAM presets: HBM2, DDR4, DDR5
 *   6. Cycle model: tile execution with all fidelity levels
 *   7. DMA bus arbitration
 *   8. Integration with performance counters
 *   9. Model reset and re-init
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../tu_cmodel/perf/cycle_model.h"
#include "../tu_cmodel/perf/performance_counters.h"
#include "../tu_cmodel/tu_config.h"

static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define TEST(name) do { tests_run++; printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT_TRUE(c, m) do { if(!(c)){FAIL(m);return;} } while(0)
#define ASSERT_EQ(a,b,m) do { if((a)!=(b)){char buf[200];snprintf(buf,200,"%s (%zu!=%zu)",m,(size_t)(a),(size_t)(b));FAIL(buf);return;} } while(0)
#define ASSERT_NEAR(a,b,t,m) do { if(fabs((double)(a)-(double)(b))>(double)(t)){char buf[200];snprintf(buf,200,"%s (%.4f!=%.4f)",m,(double)(a),(double)(b));FAIL(buf);return;} } while(0)

/* ================================================================
 * Test 1: Pipeline Tracker — Basic Issue/Complete
 * ================================================================ */

static void test_pipeline_basic(void) {
    TEST("pipeline — issue and complete one tile");
    tu_pipeline_tracker_t pt;
    tu_cycle_pipeline_init(&pt, 4);

    uint32_t src[] = {100, 200};
    uint32_t dst[] = {300};
    uint64_t stall = tu_cycle_pipeline_issue(&pt, 0, 16, 0, 16, 0, 16,
                                        src, 2, dst, 1, 0);
    ASSERT_EQ(stall, 0, "no stall on first issue");
    ASSERT_EQ(pt.total_issues, 1, "one issue counted");
    ASSERT_TRUE(pt.entries[0].active, "entry should be active");

    uint64_t elapsed = tu_cycle_pipeline_complete(&pt, 100);
    ASSERT_TRUE(elapsed >= 100, "elapsed >= issue-to-complete cycles");
    ASSERT_TRUE(!pt.entries[0].active, "entry deactivated");
    ASSERT_EQ(pt.total_completions, 1, "one completion");

    tu_cycle_pipeline_destroy(&pt);
    PASS();
}

/* ================================================================
 * Test 2: Pipeline Hazard Detection
 * ================================================================ */

static void test_pipeline_raw_hazard(void) {
    TEST("pipeline — RAW hazard detection");
    tu_pipeline_tracker_t pt;
    tu_cycle_pipeline_init(&pt, 4);

    uint32_t src0[] = {10, 20};
    uint32_t dst0[] = {30};
    tu_cycle_pipeline_issue(&pt, 0, 16, 0, 16, 0, 16, src0, 2, dst0, 1, 0);

    /* Second tile reads from register 30 that first tile writes */
    uint32_t src1[] = {30, 40};  /* src1[0] = 30 is RAW on first tile */
    uint32_t dst1[] = {50};
    uint64_t stall = tu_cycle_pipeline_issue(&pt, 1, 16, 1, 16, 1, 16,
                                        src1, 2, dst1, 1, 0);
    ASSERT_TRUE(stall > 0, "RAW hazard should cause stall");

    ASSERT_EQ(pt.total_issues, 2, "two issues");
    ASSERT_TRUE(pt.total_stall_cycles > 0, "stall cycles tracked");

    tu_cycle_pipeline_complete(&pt, 100);
    tu_cycle_pipeline_complete(&pt, 200);

    tu_cycle_pipeline_destroy(&pt);
    PASS();
}

static void test_pipeline_waw_hazard(void) {
    TEST("pipeline — WAW hazard detection");
    tu_pipeline_tracker_t pt;
    tu_cycle_pipeline_init(&pt, 4);

    uint32_t src0[] = {10, 20};
    uint32_t dst0[] = {30};
    tu_cycle_pipeline_issue(&pt, 0, 16, 0, 16, 0, 16, src0, 2, dst0, 1, 0);

    /* Second tile also writes to register 30 */
    uint32_t src1[] = {40, 50};
    uint32_t dst1[] = {30};  /* Same destination = WAW */
    uint64_t stall = tu_cycle_pipeline_issue(&pt, 1, 16, 1, 16, 1, 16,
                                        src1, 2, dst1, 1, 0);
    ASSERT_TRUE(stall > 0, "WAW hazard should cause stall");

    tu_cycle_pipeline_complete(&pt, 100);
    tu_cycle_pipeline_complete(&pt, 200);

    tu_cycle_pipeline_destroy(&pt);
    PASS();
}

static void test_pipeline_utilization(void) {
    TEST("pipeline — utilization tracking");
    tu_pipeline_tracker_t pt;
    tu_cycle_pipeline_init(&pt, 4);

    ASSERT_NEAR(tu_cycle_pipeline_utilization(&pt), 0.0, 0.01, "empty = 0%");

    uint32_t src[] = {1}, dst[] = {2};
    tu_cycle_pipeline_issue(&pt, 0, 16, 0, 16, 0, 16, src, 1, dst, 1, 0);
    ASSERT_NEAR(tu_cycle_pipeline_utilization(&pt), 0.25, 0.01, "1/4 = 25%");

    tu_cycle_pipeline_issue(&pt, 0, 16, 0, 16, 0, 16, src, 1, dst, 1, 0);
    ASSERT_NEAR(tu_cycle_pipeline_utilization(&pt), 0.50, 0.01, "2/4 = 50%");

    tu_cycle_pipeline_issue(&pt, 0, 16, 0, 16, 0, 16, src, 1, dst, 1, 0);
    tu_cycle_pipeline_issue(&pt, 0, 16, 0, 16, 0, 16, src, 1, dst, 1, 0);
    ASSERT_NEAR(tu_cycle_pipeline_utilization(&pt), 1.0, 0.01, "4/4 = 100%");

    tu_cycle_pipeline_complete(&pt, 100);
    ASSERT_NEAR(tu_cycle_pipeline_utilization(&pt), 0.75, 0.01, "after 1 complete = 75%");

    tu_cycle_pipeline_destroy(&pt);
    PASS();
}

/* ================================================================
 * Test 3: Bank Conflict Model
 * ================================================================ */

static void test_bank_model_basic(void) {
    TEST("bank model — basic access");
    tu_bank_model_t bm;
    tu_bank_model_init(&bm, 8, 4, 4, 2, 1);

    /* Single access should succeed */
    uint32_t stall = tu_bank_model_access(&bm, 0, false, 1, 0);
    ASSERT_EQ(stall, 0, "single access should not stall");

    tu_bank_model_destroy(&bm);
    PASS();
}

static void test_bank_model_exhaustion(void) {
    TEST("bank model — bandwidth exhaustion");
    tu_bank_model_t bm;
    tu_bank_model_init(&bm, 4, 4, 4, 2, 1);

    /* First access consumes the only word available */
    tu_bank_model_access(&bm, 0, false, 1, 0);

    /* Second access should stall */
    uint32_t stall = tu_bank_model_access(&bm, 0, false, 1, 0);
    ASSERT_EQ(stall, 2, "exhausted bank should stall 2 cycles (1 word × penalty 2)");

    tu_bank_model_destroy(&bm);
    PASS();
}

static void test_bank_model_refill(void) {
    TEST("bank model — periodic refill");
    tu_bank_model_t bm;
    tu_bank_model_init(&bm, 4, 4, 4, 2, 1);

    /* Consume the budget */
    tu_bank_model_access(&bm, 0, false, 1, 0);

    /* Advance to refill cycle */
    tu_bank_model_tick(&bm, 4);

    /* Now should succeed */
    uint32_t stall = tu_bank_model_access(&bm, 0, true, 1, 4);
    ASSERT_EQ(stall, 0, "after refill, access should succeed");

    tu_bank_model_destroy(&bm);
    PASS();
}

static void test_bank_model_stats(void) {
    TEST("bank model — statistics");
    tu_bank_model_t bm;
    tu_bank_model_init(&bm, 4, 4, 4, 2, 1);

    tu_bank_model_access(&bm, 0, false, 1, 0);
    tu_bank_model_access(&bm, 1, true, 1, 0);
    tu_bank_model_access(&bm, 0, false, 1, 0);  /* stall */

    uint64_t reads, writes, stalls, conf;
    double util;
    tu_bank_model_get_stats(&bm, &reads, &writes, &stalls, &conf, &util);
    ASSERT_EQ(reads, 2, "2 reads recorded");
    ASSERT_EQ(writes, 1, "1 write recorded");
    ASSERT_TRUE(stalls > 0, "stalls recorded");
    ASSERT_TRUE(conf > 0, "conflicts recorded");

    tu_bank_model_destroy(&bm);
    PASS();
}

/* ================================================================
 * Test 4: DRAM Row Buffer Model
 * ================================================================ */

static void test_dram_row_hit(void) {
    TEST("DRAM — row buffer hit");
    tu_dram_channel_t ch;
    tu_dram_channel_init(&ch, TU_DRAM_IDEAL, 1000.0, 32);
    /* This overrides with HBM2 timings for realistic test */
    tu_dram_timing_preset(&ch.timing, TU_DRAM_HBM2, 1000.0, 32);

    /* First access: activate row */
    uint64_t lat1 = tu_dram_access(&ch, 0, false, 256, 0);
    ASSERT_TRUE(lat1 >= ch.timing.tRCD + ch.timing.tCL, "first access = RCD+CL");

    /* Second access same row: hit */
    uint64_t lat2 = tu_dram_access(&ch, 0, false, 256, 10);
    ASSERT_TRUE(lat2 <= lat1, "row hit should be faster than row miss");

    ASSERT_EQ(ch.total_row_hits, 1, "one row hit");
    ASSERT_EQ(ch.total_row_misses, 1, "one row miss (first access)");

    tu_cycle_dram_destroy(&ch);
    PASS();
}

static void test_dram_row_conflict(void) {
    TEST("DRAM — row buffer conflict");
    tu_dram_channel_t ch;
    tu_dram_channel_init(&ch, TU_DRAM_IDEAL, 1000.0, 32);
    tu_dram_timing_preset(&ch.timing, TU_DRAM_HBM2, 1000.0, 32);

    /* Access row 0, column 0 */
    tu_dram_access(&ch, 0, false, 256, 0);

    /* Access row 1, column 0 (same bank, different row) */
    uint64_t addr_row1 = (1ull << 14);  /* row=1, bank=0, col=0 */
    uint64_t lat = tu_dram_access(&ch, addr_row1, false, 256, 10);

    /* Should be RP + RCD + CL */
    ASSERT_TRUE(lat >= ch.timing.tRP + ch.timing.tRCD + ch.timing.tCL,
                "row conflict = RP+RCD+CL");

    tu_cycle_dram_destroy(&ch);
    PASS();
}

static void test_dram_presets(void) {
    TEST("DRAM — preset values");

    tu_dram_timing_t t;
    tu_dram_preset_hbm2(&t, 32);
    ASSERT_EQ(t.num_banks, 8, "HBM2 has 8 banks");
    ASSERT_TRUE(t.tRCD > 0, "HBM2 has non-zero tRCD");

    tu_dram_preset_ddr4(&t, 8);
    ASSERT_EQ(t.num_banks, 16, "DDR4 has 16 banks");
    ASSERT_EQ(t.num_bank_groups, 4, "DDR4 has 4 bank groups");

    tu_dram_preset_ddr5(&t, 8);
    ASSERT_EQ(t.num_banks, 32, "DDR5 has 32 banks");

    tu_dram_preset_hbm3(&t, 32);
    ASSERT_EQ(t.num_banks, 16, "HBM3 has 16 banks");

    tu_dram_preset_ideal(&t, 64);
    ASSERT_EQ(t.tCL, 1, "ideal has CL=1");

    PASS();
}

/* ================================================================
 * Test 5: Cycle Model — Functional Mode
 * ================================================================ */

static void test_cycle_model_functional(void) {
    TEST("cycle model — functional mode");
    tu_cycle_model_t *cm = tu_cycle_model_create(TU_CYCLE_MODEL_FUNCTIONAL, NULL);
    ASSERT_TRUE(cm != NULL, "created");

    uint64_t cycles = tu_cycle_model_execute_tile(cm, 0, 16, 0, 16, 0, 16,
                                                    100, 200, 300);
    ASSERT_EQ(cycles, 0, "functional mode returns 0 cycles");
    ASSERT_EQ(cm->current_cycle, 0, "global cycle unchanged");

    uint64_t dma_cycles = tu_cycle_model_dma_transfer(cm, 0, 1024, true, 0x1000, 0);
    ASSERT_EQ(dma_cycles, 0, "DMA transfer returns 0 in functional mode");

    tu_cycle_model_destroy(cm);
    PASS();
}

/* ================================================================
 * Test 6: Cycle Model — Estimated Mode
 * ================================================================ */

static void test_cycle_model_estimated(void) {
    TEST("cycle model — estimated mode tile");
    tu_cycle_model_t *cm = tu_cycle_model_create(TU_CYCLE_MODEL_ESTIMATED, NULL);
    ASSERT_TRUE(cm != NULL, "created");

    uint64_t cycles = tu_cycle_model_execute_tile(cm, 0, 16, 0, 16, 0, 64,
                                                    100, 200, 300);
    uint64_t expected = TU_PE_PIPELINE_DEPTH * 16  /* fill */
                        + 64                        /* compute */
                        + TU_PE_PIPELINE_DEPTH * 16; /* drain */
    ASSERT_EQ(cycles, expected, "estimated = fill + compute + drain");
    ASSERT_EQ(cm->current_cycle, expected, "global cycle advanced");

    tu_cycle_model_destroy(cm);
    PASS();
}

/* ================================================================
 * Test 7: Cycle Model — Cycle-Accurate Mode
 * ================================================================ */

static void test_cycle_model_accurate(void) {
    TEST("cycle model — cycle-accurate tile execution");
    tu_perf_counters_t perf;
    tu_perf_init(&perf, 1000.0);

    tu_cycle_model_t *cm = tu_cycle_model_create(TU_CYCLE_MODEL_CYCLE_ACCURATE, &perf);
    ASSERT_TRUE(cm != NULL, "created");
    ASSERT_TRUE(cm->pipeline != NULL, "pipeline created");
    ASSERT_TRUE(cm->bank_model != NULL, "bank model created");
    ASSERT_TRUE(cm->dram_channel != NULL, "DRAM channel created");

    uint64_t cycles = tu_cycle_model_execute_tile(cm, 0, 16, 0, 16, 0, 64,
                                                    0x100, 0x200, 0x300);
    ASSERT_TRUE(cycles > 0, "cycle-accurate mode returns positive cycles");
    ASSERT_TRUE(cm->current_cycle > 0, "global cycle advanced");

    /* Pipeline should have one completion */
    ASSERT_TRUE(cm->pipeline->total_issues >= 1, "at least one issue");

    tu_cycle_model_report(cm);
    tu_cycle_model_destroy(cm);
    PASS();
}

static void test_cycle_model_multiple_tiles(void) {
    TEST("cycle model — multiple tiles");
    tu_cycle_model_t *cm = tu_cycle_model_create(TU_CYCLE_MODEL_CYCLE_ACCURATE, NULL);

    /* Issue 4 tiles sequentially */
    uint64_t t1 = tu_cycle_model_execute_tile(cm, 0, 16, 0, 16, 0, 16,
                                                0x100, 0x200, 0x300);
    uint64_t t2 = tu_cycle_model_execute_tile(cm, 0, 16, 16, 16, 0, 16,
                                                0x140, 0x240, 0x340);
    uint64_t t3 = tu_cycle_model_execute_tile(cm, 16, 16, 0, 16, 0, 16,
                                                0x180, 0x280, 0x380);
    uint64_t t4 = tu_cycle_model_execute_tile(cm, 16, 16, 16, 16, 0, 16,
                                                0x1C0, 0x2C0, 0x3C0);

    ASSERT_TRUE(t1 > 0 && t2 > 0 && t3 > 0 && t4 > 0, "all tiles take positive cycles");
    ASSERT_EQ(cm->pipeline->total_issues, 4, "4 issues");
    ASSERT_EQ(cm->pipeline->total_completions, 4, "4 completions");

    tu_cycle_model_destroy(cm);
    PASS();
}

/* ================================================================
 * Test 8: DMA Bus Arbitration
 * ================================================================ */

static void test_dma_arbitration(void) {
    TEST("DMA arbitration — single channel");
    tu_cycle_model_t *cm = tu_cycle_model_create(TU_CYCLE_MODEL_CYCLE_ACCURATE, NULL);

    uint64_t arb = tu_cycle_model_dma_arbitrate(cm, 0, 10);
    ASSERT_EQ(arb, 0, "single channel gets no arbitration stall");

    /* Mark channels 1 and 2 as active (contending for bus) */
    cm->dma_bus_cycles[1] = cm->current_cycle + 10;
    cm->dma_bus_cycles[2] = cm->current_cycle + 10;
    arb = tu_cycle_model_dma_arbitrate(cm, 0, 10);
    ASSERT_TRUE(arb > 0, "contention causes arbitration stall");

    tu_cycle_model_destroy(cm);
    PASS();
}

/* ================================================================
 * Test 9: DMA Transfer in Cycle-Accurate Mode
 * ================================================================ */

static void test_dma_transfer_accurate(void) {
    TEST("DMA transfer — cycle-accurate mode");
    tu_cycle_model_t *cm = tu_cycle_model_create(TU_CYCLE_MODEL_CYCLE_ACCURATE, NULL);

    uint64_t cycles = tu_cycle_model_dma_transfer(cm, 0, 4096, true, 0x10000, 2);
    ASSERT_TRUE(cycles > 0, "DMA transfer takes positive cycles");
    ASSERT_TRUE(cm->current_cycle > 0, "cycle counter advanced");

    /* Channel stats tracked */
    ASSERT_TRUE(cm->dma_bus_cycles[0] > 0, "channel 0 bus cycles tracked");

    tu_cycle_model_destroy(cm);
    PASS();
}

/* ================================================================
 * Test 10: Model Reset
 * ================================================================ */

static void test_model_reset(void) {
    TEST("cycle model — reset");
    tu_cycle_model_t *cm = tu_cycle_model_create(TU_CYCLE_MODEL_CYCLE_ACCURATE, NULL);

    tu_cycle_model_execute_tile(cm, 0, 16, 0, 16, 0, 64, 0x100, 0x200, 0x300);
    ASSERT_TRUE(cm->current_cycle > 0, "cycles accumulated");

    tu_cycle_model_reset(cm);
    ASSERT_EQ(cm->current_cycle, 0, "cycle counter reset");
    ASSERT_EQ(cm->pipeline->total_issues, 0, "pipeline issues reset");
    ASSERT_EQ(cm->dma_bus_stall_cycles, 0, "bus stalls reset");

    tu_cycle_model_destroy(cm);
    PASS();
}

/* ================================================================
 * Test 11: DRAM Statistics
 * ================================================================ */

static void test_dram_stats(void) {
    TEST("DRAM — statistics");
    tu_dram_channel_t ch;
    tu_dram_channel_init(&ch, TU_DRAM_IDEAL, 1000.0, 32);
    tu_dram_timing_preset(&ch.timing, TU_DRAM_HBM2, 1000.0, 32);

    tu_dram_access(&ch, 0, false, 128, 0);      /* miss */
    tu_dram_access(&ch, 0, false, 128, 10);     /* hit */
    tu_dram_access(&ch, 0, true, 64, 20);       /* hit */

    uint64_t acc, hits, misses, stall;
    double hr, bw;
    tu_cycle_dram_get_stats(&ch, &acc, &hits, &misses, &hr, &bw, &stall);
    ASSERT_EQ(acc, 3, "3 accesses");
    ASSERT_EQ(hits, 2, "2 row hits");
    ASSERT_EQ(misses, 1, "1 row miss");
    ASSERT_NEAR(hr, 2.0/3.0, 0.01, "hit rate = 2/3");

    tu_cycle_dram_destroy(&ch);
    PASS();
}

/* ================================================================
 * Test 12: Edge Cases
 * ================================================================ */

static void test_null_safety(void) {
    TEST("null safety — destroy NULL");
    tu_cycle_model_destroy(NULL);
    tu_cycle_pipeline_destroy(NULL);
    tu_bank_model_destroy(NULL);
    tu_cycle_dram_destroy(NULL);
    PASS();

    TEST("null safety — operations on NULL");
    ASSERT_EQ(tu_cycle_model_execute_tile(NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0,
              "tile on NULL returns 0");
    ASSERT_EQ(tu_cycle_model_dma_transfer(NULL, 0, 0, false, 0, 0), 0,
              "DMA on NULL returns 0");
    ASSERT_EQ(tu_cycle_pipeline_complete(NULL, 0), 0, "complete on NULL returns 0");
    ASSERT_EQ(tu_bank_model_access(NULL, 0, false, 0, 0), 0,
              "bank access on NULL returns 0");
    ASSERT_EQ(tu_dram_access(NULL, 0, false, 0, 0), 0,
              "DRAM access on NULL returns 0");
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("\n=== Cycle-Accurate Model Tests ===\n\n");

    /* Pipeline */
    test_pipeline_basic();
    test_pipeline_raw_hazard();
    test_pipeline_waw_hazard();
    test_pipeline_utilization();

    /* Bank conflicts */
    test_bank_model_basic();
    test_bank_model_exhaustion();
    test_bank_model_refill();
    test_bank_model_stats();

    /* DRAM */
    test_dram_row_hit();
    test_dram_row_conflict();
    test_dram_presets();
    test_dram_stats();

    /* Cycle model */
    test_cycle_model_functional();
    test_cycle_model_estimated();
    test_cycle_model_accurate();
    test_cycle_model_multiple_tiles();
    test_model_reset();

    /* DMA */
    test_dma_arbitration();
    test_dma_transfer_accurate();

    /* Edge cases */
    test_null_safety();

    /* Summary */
    printf("\n---\n");
    printf("Tests: %d run, %d passed, %d failed\n",
           tests_run, tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
