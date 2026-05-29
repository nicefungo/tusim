/*
 * TU CModel — Double Buffering Tests
 * ====================================
 * Gap A7: Ping-pong buffer management, swap semantics,
 *         DMA overlap modeling, and edge cases.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/tu_sram.h"
#include "../tu_cmodel/memory/double_buffer.h"

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
    if ((a) != (b)) { FAIL(msg); printf("  expected %ld, got %ld\n", (long)(b), (long)(a)); return; } \
} while(0)

/* ---- Test 1: Double buffering not enabled by default ---- */
static void test_default_disabled(void) {
    TEST("default disabled");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    ASSERT_FALSE(tu_sram_is_double_buffered(&sram), "should not be double-buffered by default");
    ASSERT_EQ(tu_sram_get_shadow_ptr(&sram), NULL, "shadow ptr should be NULL when disabled");
    ASSERT_EQ(tu_sram_swap_buffers(&sram), 0UL, "swap should return 0 when disabled");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 2: Enable double buffering ---- */
static void test_enable(void) {
    TEST("enable double buffering");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    int rc = tu_sram_enable_double_buffer(&sram);
    ASSERT_EQ(rc, 0, "enable should succeed");

    ASSERT_TRUE(tu_sram_is_double_buffered(&sram), "should be double-buffered");
    ASSERT_TRUE(tu_sram_get_shadow_ptr(&sram) != NULL, "shadow ptr should be non-NULL");
    ASSERT_TRUE(tu_sram_get_shadow_ptr(&sram) != tu_sram_get_active_ptr(&sram),
                "shadow and active should be different buffers");
    ASSERT_FALSE(tu_sram_is_shadow_dirty(&sram), "shadow should start clean");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 3: Swap buffers ---- */
static void test_swap(void) {
    TEST("swap buffers");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");
    tu_sram_enable_double_buffer(&sram);

    /* Write data to active buffer */
    float val = 42.0f;
    tu_sram_write(&sram, 0, &val);

    /* Swap */
    uint64_t count = tu_sram_swap_buffers(&sram);
    ASSERT_EQ(count, 1UL, "first swap should return 1");

    /* Read from active buffer (should be empty shadow) */
    float read_val = 0.0f;
    tu_sram_read(&sram, 0, &read_val);
    ASSERT_TRUE(read_val == 0.0f, "after swap, active should be empty shadow");

    /* Write to shadow, then swap */
    uint8_t *shadow = tu_sram_get_shadow_ptr(&sram);
    ASSERT_TRUE(shadow != NULL, "shadow should exist");
    /* shadow is now the old active (which had 42.0f at offset 0) */
    /* Actually let me write new data to shadow, then swap */
    float new_val = 99.0f;
    memcpy(shadow + 0, &new_val, sizeof(float));
    tu_sram_notify_shadow_write(&sram, sizeof(float), 10);
    ASSERT_TRUE(tu_sram_is_shadow_dirty(&sram), "shadow should be dirty after write");

    count = tu_sram_swap_buffers(&sram);
    ASSERT_EQ(count, 2UL, "second swap should return 2");
    ASSERT_FALSE(tu_sram_is_shadow_dirty(&sram), "shadow should be clean after swap");

    /* Now active should have 99.0 */
    read_val = 0.0f;
    tu_sram_read(&sram, 0, &read_val);
    ASSERT_TRUE(read_val == 99.0f, "after second swap, active should have new data");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 4: Shadow write notification ---- */
static void test_shadow_write(void) {
    TEST("shadow write notification");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");
    tu_sram_enable_double_buffer(&sram);

    ASSERT_FALSE(tu_sram_is_shadow_dirty(&sram), "should start clean");

    tu_sram_notify_shadow_write(&sram, 1024, 50);
    ASSERT_TRUE(tu_sram_is_shadow_dirty(&sram), "should be dirty after notification");

    tu_db_stats_t stats;
    tu_sram_get_db_stats(&sram, &stats);
    ASSERT_EQ(stats.dma_to_shadow_bytes, 1024UL, "bytes should match");
    ASSERT_EQ(stats.dma_to_shadow_cycles, 50UL, "cycles should match");
    ASSERT_TRUE(stats.enabled, "should be enabled");

    /* Swap should clear dirty */
    tu_sram_swap_buffers(&sram);
    ASSERT_FALSE(tu_sram_is_shadow_dirty(&sram), "dirty cleared by swap");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 5: Multiple swaps ---- */
static void test_multiple_swaps(void) {
    TEST("multiple swaps");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");
    tu_sram_enable_double_buffer(&sram);

    for (int i = 0; i < 10; i++) {
        uint64_t c = tu_sram_swap_buffers(&sram);
        ASSERT_EQ(c, (uint64_t)(i + 1), "swap count should increment");
    }

    /* After even number of swaps, active should be back to primary */
    ASSERT_EQ(tu_sram_get_active_ptr(&sram), sram.banks.data,
              "after 10 swaps, active should be primary again");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 6: Disable double buffering ---- */
static void test_disable(void) {
    TEST("disable");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");
    tu_sram_enable_double_buffer(&sram);

    /* Write to active */
    float val = 77.0f;
    tu_sram_write(&sram, 0, &val);

    /* Disable */
    tu_sram_disable_double_buffer(&sram);
    ASSERT_FALSE(tu_sram_is_double_buffered(&sram), "should be disabled");
    ASSERT_EQ(tu_sram_get_shadow_ptr(&sram), NULL, "shadow should be NULL");

    /* Data from active should be preserved in primary buffer */
    float read_val = 0.0f;
    tu_sram_read(&sram, 0, &read_val);
    ASSERT_TRUE(read_val == 77.0f, "data should be preserved after disable");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 7: Overlapped cycle tracking ---- */
static void test_overlapped_cycles(void) {
    TEST("overlapped cycles");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");
    tu_sram_enable_double_buffer(&sram);

    ASSERT_EQ(tu_sram_get_overlapped_cycles(&sram), 0UL, "start at 0");

    tu_sram_record_overlapped_cycles(&sram, 100);
    ASSERT_EQ(tu_sram_get_overlapped_cycles(&sram), 100UL, "100 overlapped");

    tu_sram_record_overlapped_cycles(&sram, 50);
    ASSERT_EQ(tu_sram_get_overlapped_cycles(&sram), 150UL, "150 total");

    /* Swap should NOT reset overlapped cycles */
    tu_sram_swap_buffers(&sram);
    ASSERT_EQ(tu_sram_get_overlapped_cycles(&sram), 150UL, "overlapped persists across swap");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 8: Stats integrity ---- */
static void test_stats(void) {
    TEST("stats integrity");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");
    tu_sram_enable_double_buffer(&sram);

    /* Simulate a workload: DMA into shadow, compute, swap, repeat */
    for (int tile = 0; tile < 5; tile++) {
        /* DMA tile N+1 into shadow (50 cycles) */
        tu_sram_notify_shadow_write(&sram, 1024, 50);

        /* Compute tile N from active (100 cycles) */
        /* DMA overlaps: 50 cycles saved */
        tu_sram_record_overlapped_cycles(&sram, 50);

        /* Swap */
        tu_sram_swap_buffers(&sram);
    }

    tu_db_stats_t stats;
    tu_sram_get_db_stats(&sram, &stats);

    ASSERT_EQ(stats.swap_count, 5UL, "5 swaps");
    ASSERT_EQ(stats.dma_to_shadow_bytes, 5120UL, "5 × 1024 bytes");
    ASSERT_EQ(stats.dma_to_shadow_cycles, 250UL, "5 × 50 cycles");
    ASSERT_EQ(stats.overlapped_cycles, 250UL, "5 × 50 overlapped");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 9: Active pointer changes on swap ---- */
static void test_active_ptr_swap(void) {
    TEST("active pointer changes");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");
    tu_sram_enable_double_buffer(&sram);

    uint8_t *before = tu_sram_get_active_ptr(&sram);
    tu_sram_swap_buffers(&sram);
    uint8_t *after = tu_sram_get_active_ptr(&sram);

    ASSERT_TRUE(before != after, "active ptr should change after swap");
    /* Shadow and active should swap roles */
    ASSERT_EQ(tu_sram_get_shadow_ptr(&sram), before,
              "old active should now be shadow");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Test 10: Double-enable is idempotent ---- */
static void test_double_enable(void) {
    TEST("double enable idempotent");
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "test");

    int rc1 = tu_sram_enable_double_buffer(&sram);
    ASSERT_EQ(rc1, 0, "first enable should succeed");

    int rc2 = tu_sram_enable_double_buffer(&sram);
    ASSERT_EQ(rc2, 0, "second enable should succeed (idempotent)");

    ASSERT_TRUE(tu_sram_is_double_buffered(&sram), "still double-buffered");

    tu_sram_destroy(&sram);
    PASS();
}

/* ---- Main ---- */

int main(void) {
    printf("=== TU Double Buffering Tests ===\n\n");

    test_default_disabled();
    test_enable();
    test_swap();
    test_shadow_write();
    test_multiple_swaps();
    test_disable();
    test_overlapped_cycles();
    test_stats();
    test_active_ptr_swap();
    test_double_enable();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
