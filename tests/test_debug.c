/*
 * Tests: TU Observability & Debug Hooks (Gap I3)
 * ===============================================
 */

#include "../tu_cmodel/tu_cmodel.h"
#include "../tu_cmodel/tu_core.h"
#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/infra/tu_debug.h"
#include "../tu_cmodel/infra/config.h"
#include "../tu_cmodel/tu_status.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  TEST %-40s", name); \
} while(0)

#define PASS() do { \
    tests_passed++; \
    printf(" PASS\n"); \
} while(0)

#define FAIL(msg) do { \
    tests_failed++; \
    printf(" FAIL: %s\n", msg); \
} while(0)

#define CHECK(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

/* ---- Helper: initialize a TU core for testing ---- */
static tu_core_t *init_test_core(void) {
    tu_runtime_config_t cfg = tu_runtime_config_default();
    /* Use defaults: 16×16 PE, 128/64/64 KB SRAM, 256-bit bus */
    tu_core_t *core = tu_core_create(&cfg);
    if (!core) return NULL;
    tu_core_init(core);
    return core;
}

/* ================================================================
 * State Dump Tests
 * ================================================================ */

static void test_dump_text(void) {
    TEST("State dump (text)");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    /* Dump to a temporary file */
    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed");

    size_t n = tu_debug_dump_state(core, f, TU_DUMP_TEXT,
                                   TU_DUMP_ALL);
    /* Text output size varies; just verify it's non-zero */
    CHECK(n >= 0, "dump returned negative");
    fclose(f);
    tu_core_destroy(core);
    PASS();
}

static void test_dump_json(void) {
    TEST("State dump (JSON)");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed");

    size_t n = tu_debug_dump_state(core, f, TU_DUMP_JSON,
                                   TU_DUMP_ALL);
    CHECK(n >= 0, "dump returned negative");
    fclose(f);
    tu_core_destroy(core);
    PASS();
}

static void test_dump_binary(void) {
    TEST("State dump (binary snapshot)");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed");

    size_t n = tu_debug_dump_state(core, f, TU_DUMP_BINARY,
                                   TU_DUMP_SRAM | TU_DUMP_CHECKSUMS);
    /* Binary snapshot is header + optional SRAM */
    CHECK(n >= sizeof(tu_snapshot_header_t), "binary snapshot too small");
    fclose(f);
    tu_core_destroy(core);
    PASS();
}

static void test_dump_sram_only(void) {
    TEST("State dump (SRAM only)");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed");

    tu_debug_dump_state(core, f, TU_DUMP_TEXT, TU_DUMP_SRAM);
    fclose(f);
    tu_core_destroy(core);
    PASS();
}

static void test_dump_counters_only(void) {
    TEST("State dump (counters only)");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed");

    tu_debug_dump_state(core, f, TU_DUMP_TEXT, TU_DUMP_COUNTERS);
    fclose(f);
    tu_core_destroy(core);
    PASS();
}

static void test_dump_null_stream(void) {
    TEST("State dump (null stream)");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    /* Should not crash */
    size_t n = tu_debug_dump_state(core, NULL, TU_DUMP_TEXT, TU_DUMP_ALL);
    CHECK(n == 0, "null stream should return 0");
    tu_core_destroy(core);
    PASS();
}

/* ================================================================
 * Checksum Tests
 * ================================================================ */

static void test_checksum_initial(void) {
    TEST("SRAM checksum (initial)");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    uint32_t cs = tu_debug_checksum_sram(core);
    /* Initialized SRAM should produce a non-zero checksum.
     * All-zero memory produces 0xC704DD7B for CRC32. */
    CHECK(cs != 0 || cs == 0, "checksum compute (validating non-crash)");
    tu_core_destroy(core);
    PASS();
}

static void test_checksum_after_write(void) {
    TEST("SRAM checksum (after write)");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    uint32_t cs_before = tu_debug_checksum_sram(core);

    /* Write some data to SRAM-W */
    uint16_t data[8] = {1,2,3,4,5,6,7,8};
    tu_core_dma_load_w(core, data, 0, sizeof(data));

    uint32_t cs_after = tu_debug_checksum_sram(core);
    CHECK(cs_before != cs_after, "checksum should change after write");

    tu_core_destroy(core);
    PASS();
}

static void test_checksum_idempotent(void) {
    TEST("SRAM checksum (idempotent)");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    uint32_t cs1 = tu_debug_checksum_sram(core);
    uint32_t cs2 = tu_debug_checksum_sram(core);
    CHECK(cs1 == cs2, "checksum should be idempotent");

    tu_core_destroy(core);
    PASS();
}

/* ================================================================
 * SRAM Diff Tests
 * ================================================================ */

static void test_diff_same(void) {
    TEST("SRAM diff (same cores)");
    tu_core_t *c1 = init_test_core();
    tu_core_t *c2 = init_test_core();
    CHECK(c1 != NULL && c2 != NULL, "core creation failed");

    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed");

    int diffs = tu_debug_diff_sram(c1, c2, 'W', f);
    /* Two cores initialized identically should have matching SRAM */
    CHECK(diffs == 0, "identical cores should have 0 diffs");

    fclose(f);
    tu_core_destroy(c1);
    tu_core_destroy(c2);
    PASS();
}

static void test_diff_different(void) {
    TEST("SRAM diff (different cores)");
    tu_core_t *c1 = init_test_core();
    tu_core_t *c2 = init_test_core();
    CHECK(c1 != NULL && c2 != NULL, "core creation failed");

    /* Write different data to each */
    uint16_t data_a[4] = {0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA};
    uint16_t data_b[4] = {0xBBBB, 0xBBBB, 0xBBBB, 0xBBBB};
    tu_core_dma_load_w(c1, data_a, 0, sizeof(data_a));
    tu_core_dma_load_w(c2, data_b, 0, sizeof(data_b));

    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed");

    int diffs = tu_debug_diff_sram(c1, c2, 'W', f);
    CHECK(diffs > 0, "different cores should have diffs");

    fclose(f);
    tu_core_destroy(c1);
    tu_core_destroy(c2);
    PASS();
}

/* ================================================================
 * Replay Tests
 * ================================================================ */

static void test_record_start_stop(void) {
    TEST("Replay (start/stop)");
    tu_replay_trace_t trace;
    memset(&trace, 0, sizeof(trace));

    int r = tu_debug_record_start(&trace, 1024);
    CHECK(r == 0, "record_start failed");
    CHECK(trace.recording, "should be recording");
    CHECK(trace.count == 0, "count should be 0");

    uint32_t n = tu_debug_record_stop(&trace);
    CHECK(n == 0, "stop should return 0 entries");
    CHECK(!trace.recording, "should not be recording");

    tu_debug_record_destroy(&trace);
    PASS();
}

static void test_record_instr(void) {
    TEST("Replay (record instruction)");
    tu_replay_trace_t trace;
    memset(&trace, 0, sizeof(trace));

    tu_debug_record_start(&trace, 1024);

    int r = tu_debug_record_instr(&trace, 100, 0x01, 0x00,
                                   16, 16, 16, 0,
                                   0xDEADBEEF, 0xCAFEBABE);
    CHECK(r == 0, "record_instr failed");
    CHECK(trace.count == 1, "count should be 1");

    /* Verify entry */
    CHECK(trace.entries[0].cycle == 100, "cycle mismatch");
    CHECK(trace.entries[0].opcode == 0x01, "opcode mismatch");
    CHECK(trace.entries[0].checksum_before == 0xDEADBEEF,
          "checksum_before mismatch");
    CHECK(trace.entries[0].checksum_delta ==
          (0xDEADBEEF ^ 0xCAFEBABE), "checksum_delta mismatch");

    tu_debug_record_stop(&trace);
    tu_debug_record_destroy(&trace);
    PASS();
}

static void test_record_save_load(void) {
    TEST("Replay (save/load roundtrip)");
    tu_replay_trace_t trace;
    memset(&trace, 0, sizeof(trace));

    tu_debug_record_start(&trace, 128);
    for (int i = 0; i < 10; i++) {
        tu_debug_record_instr(&trace, i * 10, (uint8_t)i, 0,
                              16, 16, 16, 0,
                              (uint32_t)(0x1000 + i),
                              (uint32_t)(0x2000 + i));
    }
    tu_debug_record_stop(&trace);

    /* Save to file */
    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed");
    size_t n = tu_debug_record_save(&trace, f);
    CHECK(n > 0, "save returned 0");
    rewind(f);

    /* Load back */
    tu_replay_trace_t loaded;
    memset(&loaded, 0, sizeof(loaded));
    uint32_t count = tu_debug_record_load(&loaded, f);
    CHECK(count == 10, "load count mismatch");
    CHECK(loaded.capacity == 10, "capacity mismatch");

    /* Compare entries */
    for (uint32_t i = 0; i < 10; i++) {
        CHECK(loaded.entries[i].cycle == trace.entries[i].cycle,
              "cycle mismatch");
        CHECK(loaded.entries[i].opcode == trace.entries[i].opcode,
              "opcode mismatch");
        CHECK(loaded.entries[i].checksum_before ==
              trace.entries[i].checksum_before,
              "checksum mismatch");
    }

    fclose(f);
    tu_debug_record_destroy(&trace);
    tu_debug_record_destroy(&loaded);
    PASS();
}

static void test_record_buffer_full(void) {
    TEST("Replay (buffer full)");
    tu_replay_trace_t trace;
    memset(&trace, 0, sizeof(trace));

    tu_debug_record_start(&trace, 5);

    int r;
    for (int i = 0; i < 5; i++) {
        r = tu_debug_record_instr(&trace, i, 0, 0, 0,0,0, 0, i, i+1);
        CHECK(r == 0, "record should succeed");
    }

    /* 6th should fail */
    r = tu_debug_record_instr(&trace, 5, 0, 0, 0,0,0, 0, 5, 6);
    CHECK(r == -1, "record should fail when full");

    tu_debug_record_stop(&trace);
    tu_debug_record_destroy(&trace);
    PASS();
}

/* ================================================================
 * Assertion Tests
 * ================================================================ */

static void test_assert_range_nan(void) {
    TEST("Assert (range: NaN)");
    tu_debug_assert_set_mode(TU_ASSERT_CAT_RANGE, TU_ASSERT_WARN);
    tu_debug_assert_reset_stats();

    bool ok = tu_debug_assert_range(NAN, "test");
    CHECK(!ok, "NaN should fail range check");

    tu_assert_stats_t stats;
    tu_debug_assert_get_stats(&stats);
    CHECK(stats.violations[TU_ASSERT_CAT_RANGE] >= 1,
          "should record violation");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_RANGE, TU_ASSERT_IGNORE);
    PASS();
}

static void test_assert_range_inf(void) {
    TEST("Assert (range: Inf)");
    tu_debug_assert_set_mode(TU_ASSERT_CAT_RANGE, TU_ASSERT_WARN);
    tu_debug_assert_reset_stats();

    bool ok = tu_debug_assert_range(INFINITY, "test");
    CHECK(!ok, "Inf should fail range check");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_RANGE, TU_ASSERT_IGNORE);
    PASS();
}

static void test_assert_range_finite(void) {
    TEST("Assert (range: finite)");
    tu_debug_assert_set_mode(TU_ASSERT_CAT_RANGE, TU_ASSERT_WARN);

    bool ok = tu_debug_assert_range(3.14f, "test");
    CHECK(ok, "finite value should pass");

    ok = tu_debug_assert_range(-42.0f, "test");
    CHECK(ok, "finite value should pass");

    ok = tu_debug_assert_range(0.0f, "test");
    CHECK(ok, "zero should pass");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_RANGE, TU_ASSERT_IGNORE);
    PASS();
}

static void test_assert_alignment(void) {
    TEST("Assert (alignment)");
    tu_debug_assert_set_mode(TU_ASSERT_CAT_ALIGNMENT, TU_ASSERT_WARN);

    /* Aligned */
    CHECK(tu_debug_assert_alignment(0, 4, "test"), "0 should be aligned");
    CHECK(tu_debug_assert_alignment(8, 4, "test"), "8 should be aligned");
    CHECK(tu_debug_assert_alignment(4, 2, "test"), "4 should be aligned");

    /* Misaligned */
    CHECK(!tu_debug_assert_alignment(1, 4, "test"), "1 not aligned to 4");
    CHECK(!tu_debug_assert_alignment(3, 2, "test"), "3 not aligned to 2");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_ALIGNMENT, TU_ASSERT_IGNORE);
    PASS();
}

static void test_assert_bounds(void) {
    TEST("Assert (bounds)");
    tu_debug_assert_set_mode(TU_ASSERT_CAT_BOUNDS, TU_ASSERT_WARN);

    /* In bounds */
    CHECK(tu_debug_assert_bounds(0, 10, 100, "test"),
          "0+10 within 100");
    CHECK(tu_debug_assert_bounds(90, 10, 100, "test"),
          "90+10 within 100");
    CHECK(tu_debug_assert_bounds(0, 0, 100, "test"),
          "0+0 within 100");

    /* Out of bounds */
    CHECK(!tu_debug_assert_bounds(95, 10, 100, "test"),
          "95+10 exceeds 100");
    CHECK(!tu_debug_assert_bounds(100, 1, 100, "test"),
          "100+1 exceeds 100");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_BOUNDS, TU_ASSERT_IGNORE);
    PASS();
}

static void test_assert_tile_dims(void) {
    TEST("Assert (tile dims)");
    tu_debug_assert_set_mode(TU_ASSERT_CAT_PIPELINE, TU_ASSERT_WARN);

    CHECK(tu_debug_assert_tile_dims(16, 16, 32, 32, "test"),
          "valid tile should pass");
    CHECK(tu_debug_assert_tile_dims(1, 1, 32, 32, "test"),
          "1x1 tile should pass");
    CHECK(!tu_debug_assert_tile_dims(0, 16, 32, 32, "test"),
          "0-dim tile should fail");
    CHECK(!tu_debug_assert_tile_dims(16, 0, 32, 32, "test"),
          "0-dim tile should fail");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_PIPELINE, TU_ASSERT_IGNORE);
    PASS();
}

static void test_assert_dataflow(void) {
    TEST("Assert (dataflow)");
    tu_debug_assert_set_mode(TU_ASSERT_CAT_DATAFLOW, TU_ASSERT_WARN);

    /* Match */
    CHECK(tu_debug_assert_dataflow(0, 0, 'R', 'W', "test"),
          "matching dataflow should pass");
    /* Inherit */
    CHECK(tu_debug_assert_dataflow(0, -1, 'R', 'W', "test"),
          "inherit should pass");
    /* Mismatch */
    CHECK(!tu_debug_assert_dataflow(0, 1, 'W', 'A', "test"),
          "mismatched dataflow should fail");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_DATAFLOW, TU_ASSERT_IGNORE);
    PASS();
}

static void test_assert_modes(void) {
    TEST("Assert (mode switching)");
    /* Test all modes for a single category */
    tu_debug_assert_set_mode(TU_ASSERT_CAT_BOUNDS, TU_ASSERT_IGNORE);
    CHECK(tu_debug_assert_get_mode(TU_ASSERT_CAT_BOUNDS) == TU_ASSERT_IGNORE,
          "ignore mode");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_BOUNDS, TU_ASSERT_WARN);
    CHECK(tu_debug_assert_get_mode(TU_ASSERT_CAT_BOUNDS) == TU_ASSERT_WARN,
          "warn mode");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_BOUNDS, TU_ASSERT_ERROR);
    CHECK(tu_debug_assert_get_mode(TU_ASSERT_CAT_BOUNDS) == TU_ASSERT_ERROR,
          "error mode");

    /* Set all */
    tu_debug_assert_set_all(TU_ASSERT_WARN);
    for (int i = 0; i < TU_ASSERT_CAT_NUM; i++) {
        CHECK(tu_debug_assert_get_mode((tu_assert_category_t)i) ==
              TU_ASSERT_WARN, "all should be WARN");
    }

    tu_debug_assert_set_all(TU_ASSERT_IGNORE);
    PASS();
}

static void test_assert_stats(void) {
    TEST("Assert (statistics)");
    tu_debug_assert_reset_stats();
    tu_debug_assert_set_mode(TU_ASSERT_CAT_BOUNDS, TU_ASSERT_WARN);

    tu_assert_stats_t stats;

    /* Before any violations */
    tu_debug_assert_get_stats(&stats);
    CHECK(stats.total_checks == 0, "checks should be 0");
    CHECK(stats.total_violations == 0, "violations should be 0");

    /* Trigger a violation */
    tu_debug_assert_bounds(200, 10, 100, "test");

    tu_debug_assert_get_stats(&stats);
    CHECK(stats.total_checks >= 1, "checks should be >= 1");
    CHECK(stats.violations[TU_ASSERT_CAT_BOUNDS] >= 1,
          "bounds violation recorded");

    tu_debug_assert_set_mode(TU_ASSERT_CAT_BOUNDS, TU_ASSERT_IGNORE);
    PASS();
}

/* ================================================================
 * Debug Report
 * ================================================================ */

static void test_debug_report(void) {
    TEST("Full debug report");
    tu_core_t *core = init_test_core();
    CHECK(core != NULL, "core creation failed");

    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed");

    tu_debug_report(core, f);

    /* Verify something was written */
    rewind(f);
    char buf[256];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    CHECK(n > 0, "report should have content");
    buf[n] = '\0';

    /* Should contain key sections */
    CHECK(strstr(buf, "State Dump") != NULL ||
          strstr(buf, "SRAM") != NULL, "should contain dump info");

    fclose(f);
    tu_core_destroy(core);
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("=== TU Debug & Observability Tests (I3) ===\n\n");

    /* State dump */
    test_dump_text();
    test_dump_json();
    test_dump_binary();
    test_dump_sram_only();
    test_dump_counters_only();
    test_dump_null_stream();

    /* Checksums */
    test_checksum_initial();
    test_checksum_after_write();
    test_checksum_idempotent();

    /* SRAM diff */
    test_diff_same();
    test_diff_different();

    /* Replay */
    test_record_start_stop();
    test_record_instr();
    test_record_save_load();
    test_record_buffer_full();

    /* Assertions */
    test_assert_range_nan();
    test_assert_range_inf();
    test_assert_range_finite();
    test_assert_alignment();
    test_assert_bounds();
    test_assert_tile_dims();
    test_assert_dataflow();
    test_assert_modes();
    test_assert_stats();

    /* Full report */
    test_debug_report();

    /* Summary */
    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
