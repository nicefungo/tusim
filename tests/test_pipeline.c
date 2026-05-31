/*
 * TU CModel — Software Pipelining Controller Tests
 * ==================================================
 * Gap E2: Tile-level DMA/compute overlap, pipeline stages,
 *         overlap accounting, backpressure, statistics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/tu_sram.h"
#include "../tu_cmodel/dma_descriptor.h"
#include "../tu_cmodel/memory/double_buffer.h"
#include "../tu_cmodel/compute/pipeline_controller.h"

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

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_FALSE(cond, msg) do { \
    if (cond) { FAIL(msg); return; } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    long _a = (long)(a); \
    long _b = (long)(b); \
    if (_a != _b) { FAIL(msg); printf("  expected %ld, got %ld\n", _b, _a); return; } \
} while(0)

#define ASSERT_GT(a, b, msg) do { \
    long _a = (long)(a); \
    long _b = (long)(b); \
    if (_a <= _b) { FAIL(msg); printf("  %ld not > %ld\n", _a, _b); return; } \
} while(0)

#define ASSERT_NEQ(a, b, msg) do { \
    long _a = (long)(a); \
    long _b = (long)(b); \
    if (_a == _b) { FAIL(msg); printf("  unexpected eq: %ld == %ld\n", _a, _b); return; } \
} while(0)

/* ---- Persistent host buffer for DMA descriptors ---- */
static float host_pool[8192];

/* ---- Setup/Teardown ---- */

static tu_sram_region_t g_sram;
static bool sram_ready = false;

static void setup_sram(void) {
    if (!sram_ready) {
        tu_sram_init(&g_sram, 65536, "pipeline_test");
        tu_sram_enable_double_buffer(&g_sram);
        sram_ready = true;
    }
}

static void teardown_sram(void) {
    if (sram_ready) {
        tu_sram_destroy(&g_sram);
        sram_ready = false;
    }
}

static void setup_dma(void) {
    tu_dma_init_full(true, 4, 32);
}

static void setup_pipeline(uint32_t depth) {
    tu_pipeline_init(depth, NULL);
}

/* ---- Test 1: Pipeline initialization ---- */
static void test_init(void) {
    TEST("pipeline initialization");
    tu_pipeline_init(2, NULL);

    ASSERT_TRUE(g_tu_pipeline.initialized, "should be initialized");
    ASSERT_EQ(g_tu_pipeline.depth, 2, "depth should be 2");
    ASSERT_EQ(g_tu_pipeline.active_count, 0, "no active tiles");
    ASSERT_NEQ((long)g_tu_pipeline.slots, 0L, "slots should be allocated");
    ASSERT_EQ(g_tu_pipeline.next_tile_id, 0, "first tile ID should be 0");
    ASSERT_TRUE(tu_pipeline_is_idle(), "should be idle after init");

    tu_pipeline_destroy();
    ASSERT_FALSE(g_tu_pipeline.initialized, "should no longer be initialized");

    PASS();
}

/* ---- Test 2: Default auto-initialization ---- */
static void test_auto_init(void) {
    TEST("auto initialization on first submit");
    tu_pipeline_destroy();

    int tid = tu_pipeline_submit_tile(NULL, NULL, 100, 0, NULL);
    ASSERT_TRUE(tid >= 0, "should auto-init and accept tile");
    ASSERT_EQ(g_tu_pipeline.depth, 1, "auto-init should use depth=1");

    tu_pipeline_destroy();
    PASS();
}

/* ---- Test 3: Pipeline free slots ---- */
static void test_free_slots(void) {
    TEST("free slots tracking");
    setup_pipeline(4);

    ASSERT_EQ(tu_pipeline_free_slots(), 4, "all slots free initially");

    /* Submit tiles with huge compute cost — won't complete during auto-advance */
    tu_pipeline_submit_tile(NULL, NULL, 2000000, 0, NULL);
    ASSERT_EQ(tu_pipeline_free_slots(), 3, "one slot consumed");

    tu_pipeline_submit_tile(NULL, NULL, 2000000, 0, NULL);
    ASSERT_EQ(tu_pipeline_free_slots(), 2, "two slots consumed");

    tu_pipeline_submit_tile(NULL, NULL, 2000000, 0, NULL);
    ASSERT_EQ(tu_pipeline_free_slots(), 1, "three slots consumed");

    tu_pipeline_submit_tile(NULL, NULL, 2000000, 0, NULL);
    ASSERT_EQ(tu_pipeline_free_slots(), 0, "pipeline full");

    /* Fifth should fail — auto-advance can't complete 2M+ cycles */
    int tid4 = tu_pipeline_submit_tile(NULL, NULL, 2000000, 0, NULL);
    ASSERT_EQ(tid4, -1, "fifth tile rejected (pipeline fully stalled)");

    /* Advance enough to complete all tiles */
    g_tu_pipeline.current_cycle += 4000000;
    for (int t = 0; t < 500; t++) {
        tu_pipeline_advance();
    }

    ASSERT_GT(tu_pipeline_free_slots(), 0, "slots should free up after advancing");
    ASSERT_EQ((long)g_tu_pipeline.total_tiles_processed, 4L, "all 4 tiles processed");

    tu_pipeline_destroy();
    PASS();
}

/* ---- Test 4: Pipeline stages progression ---- */
static void test_stage_progression(void) {
    TEST("stage progression DMA_PRELOAD->COMPUTE->DMA_STORE->DONE");
    setup_pipeline(2);

    int tid = tu_pipeline_submit_tile(NULL, NULL, 10, 0, NULL);
    ASSERT_TRUE(tid >= 0, "tile accepted");

    ASSERT_EQ(g_tu_pipeline.slots[0].stage, TU_PIPE_STAGE_COMPUTE,
              "first tile with no load should go directly to COMPUTE");

    for (int t = 0; t < 20; t++) {
        g_tu_pipeline.current_cycle++;
        tu_pipeline_advance();
    }

    ASSERT_EQ((long)g_tu_pipeline.total_tiles_processed, 1L, "tile should be processed");

    tu_pipeline_destroy();
    PASS();
}

/* ---- Test 5: Overlap accounting (depth=2) ---- */
static void test_overlap_depth2(void) {
    TEST("overlap accounting with depth=2");
    setup_sram();
    setup_pipeline(2);

    /* First tile: no load desc (data already in place), compute immediately */
    int tid0 = tu_pipeline_submit_tile(NULL, NULL, 100, 0, &g_sram);
    ASSERT_TRUE(tid0 >= 0, "first tile accepted");

    /* Second tile: DMA load prep for shadow buffer while first computes */
    memset(host_pool, 0x3F, 64 * 4);  /* Fill with ~0.5 in FP32 */
    tu_dma_descriptor_t *load_desc = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &g_sram, 0, host_pool, 4, 64);

    int tid1 = tu_pipeline_submit_tile(load_desc, NULL, 100, 0, &g_sram);
    ASSERT_TRUE(tid1 >= 0, "second tile accepted");

    /* Advance pipeline through all stages */
    for (int t = 0; t < 200; t++) {
        g_tu_pipeline.current_cycle++;
        tu_pipeline_advance();
    }

    tu_pipeline_stats_t stats;
    tu_pipeline_get_stats(&stats);

    ASSERT_EQ((long)stats.total_tiles, 2L, "both tiles processed");
    ASSERT_TRUE(stats.speedup > 1.0, "speedup > 1.0x");

    tu_pipeline_destroy();
    PASS();
}

/* ---- Test 6: Speedup measurable ---- */
static void test_speedup_measurable(void) {
    TEST("measurable speedup with pipelining (depth=2 vs depth=1)");
    setup_sram();
    memset(host_pool, 0x3F, sizeof(host_pool));

    /* Re-init DMA to clear any prior state */
    tu_dma_init_full(true, 4, 32);

    /* Sequential: depth=1 */
    setup_pipeline(1);
    for (int i = 0; i < 4; i++) {
        tu_dma_descriptor_t *load = tu_dma_desc_create_linear(
            0, TU_DMA_DIR_HOST_TO_TU, &g_sram, i * 256, host_pool, 4, 16);
        tu_pipeline_submit_tile(load, NULL, 500, 0, &g_sram);
    }
    for (int t = 0; t < 3000; t++) {
        g_tu_pipeline.current_cycle++;
        tu_pipeline_advance();
    }

    tu_pipeline_stats_t seq_stats;
    tu_pipeline_get_stats(&seq_stats);

    tu_pipeline_destroy();

    /* Re-init DMA and pipeline for second run */
    tu_dma_init_full(true, 4, 32);
    setup_pipeline(2);
    for (int i = 0; i < 4; i++) {
        tu_dma_descriptor_t *load = tu_dma_desc_create_linear(
            0, TU_DMA_DIR_HOST_TO_TU, &g_sram, i * 256, host_pool, 4, 16);
        tu_pipeline_submit_tile(load, NULL, 500, 0, &g_sram);
    }
    for (int t = 0; t < 3000; t++) {
        g_tu_pipeline.current_cycle++;
        tu_pipeline_advance();
    }

    tu_pipeline_stats_t pipe_stats;
    tu_pipeline_get_stats(&pipe_stats);

    ASSERT_EQ((long)seq_stats.total_tiles, 4L, "sequential: all 4 tiles");
    ASSERT_EQ((long)pipe_stats.total_tiles, 4L, "pipelined: all 4 tiles");
    ASSERT_TRUE(pipe_stats.speedup > 1.0, "pipelined speedup > 1.0");

    tu_pipeline_destroy();
    PASS();
}

/* ---- Test 7: Sync and idle detection ---- */
static void test_sync_and_idle(void) {
    TEST("sync and idle detection");
    setup_pipeline(2);

    ASSERT_TRUE(tu_pipeline_is_idle(), "should be idle initially");

    tu_pipeline_submit_tile(NULL, NULL, 100, 0, NULL);
    ASSERT_FALSE(tu_pipeline_is_idle(), "should not be idle with active tile");

    tu_pipeline_sync();
    ASSERT_TRUE(tu_pipeline_is_idle(), "should be idle after sync");

    tu_pipeline_destroy();
    PASS();
}

/* ---- Test 8: Backpressure and stall counting ---- */
static void test_backpressure(void) {
    TEST("backpressure and stall tracking");
    setup_pipeline(1);

    int tid0 = tu_pipeline_submit_tile(NULL, NULL, 2000000, 0, NULL);
    ASSERT_TRUE(tid0 >= 0, "first tile accepted");

    int tid1 = tu_pipeline_submit_tile(NULL, NULL, 2000000, 0, NULL);
    ASSERT_EQ(tid1, -1, "second tile rejected (pipeline full, depth=1)");

    tu_pipeline_stats_t stats;
    tu_pipeline_get_stats(&stats);
    ASSERT_GT((long)stats.total_stalls, 0L, "stalls should be non-zero");

    tu_pipeline_destroy();
    PASS();
}

/* ---- Test 9: Configuration defaults ---- */
static void test_config_defaults(void) {
    TEST("config defaults");
    tu_pipeline_config_t cfg = tu_pipeline_config_default();
    ASSERT_EQ(cfg.max_depth, 2, "default depth should be 2");
    ASSERT_TRUE(cfg.enable_load_overlap, "load overlap enabled by default");
    ASSERT_TRUE(cfg.enable_store_overlap, "store overlap enabled by default");
    ASSERT_FALSE(cfg.enable_triple_overlap, "triple overlap disabled by default");
    ASSERT_TRUE(cfg.model_stalls, "stall modeling on by default");
    PASS();
}

/* ---- Test 10: Statistics completeness ---- */
static void test_statistics_correctness(void) {
    TEST("statistics completeness after multi-tile run");
    setup_sram();
    tu_dma_init_full(true, 4, 32);
    setup_pipeline(2);

    for (int i = 0; i < 3; i++) {
        tu_dma_descriptor_t *load = tu_dma_desc_create_linear(
            0, TU_DMA_DIR_HOST_TO_TU, &g_sram, i * 256, host_pool, 4, 32);
        tu_pipeline_submit_tile(load, NULL, 200, 0, &g_sram);
    }

    /* Flush pipeline to completion */
    g_tu_pipeline.current_cycle += 50000;
    for (int t = 0; t < 50000; t++) {
        tu_pipeline_advance();
    }

    tu_pipeline_stats_t stats;
    tu_pipeline_get_stats(&stats);

    ASSERT_GT((long)stats.total_tiles, 1L, "at least 2 tiles processed");
    ASSERT_GT((long)stats.total_compute_cycles, 0L, "compute cycles recorded");
    ASSERT_GT((long)stats.pipelined_total, 0L, "pipelined total positive");
    ASSERT_TRUE(stats.speedup >= 1.0, "speedup >= 1.0");

    ASSERT_TRUE(stats.load_overlap_pct >= 0.0 && stats.load_overlap_pct <= 100.0,
                "load overlap pct in [0, 100]");

    tu_pipeline_destroy();
    PASS();
}

/* ---- Test 11: Reset clears state ---- */
static void test_reset(void) {
    TEST("reset clears pipeline state");
    setup_pipeline(2);

    tu_pipeline_submit_tile(NULL, NULL, 100, 0, NULL);
    ASSERT_FALSE(tu_pipeline_is_idle(), "active before reset");

    tu_pipeline_reset();
    ASSERT_TRUE(tu_pipeline_is_idle(), "idle after reset");
    ASSERT_EQ((long)g_tu_pipeline.total_tiles_processed, 0L, "no tiles after reset");

    tu_pipeline_destroy();
    PASS();
}

/* ---- Main ---- */

int main(void) {
    printf("\n===========================================\n");
    printf("  Software Pipelining Controller Tests\n");
    printf("  (Gap E2 — Tile-level DMA/Compute Overlap)\n");
    printf("===========================================\n\n");

    setup_dma();

    test_init();
    test_auto_init();
    test_free_slots();
    test_stage_progression();
    test_overlap_depth2();
    test_speedup_measurable();
    test_sync_and_idle();
    test_backpressure();
    test_config_defaults();
    test_statistics_correctness();
    test_reset();

    tu_pipeline_destroy();
    teardown_sram();

    printf("\n-------------------------------------------\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(", %d FAILED\n", tests_failed);
        printf("-------------------------------------------\n");
        return 1;
    }
    printf("\n-------------------------------------------\n");
    return 0;
}
