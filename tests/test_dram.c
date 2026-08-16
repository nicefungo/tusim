/*
 * TU CModel — DRAM Model Test Suite
 * ==================================
 * Tests: creation, type enumeration, access timing, statistics,
 * bandwidth metering, channel contention, custom parameters,
 * estimation correctness.
 */

#include "tu_cmodel/memory/dram_model.h"
#include "tu_cmodel/infra/config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [%d] %s ... ", tests_run, name); \
} while(0)

#define PASS() do { \
    tests_passed++; \
    printf("PASS\n"); \
} while(0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
} while(0)

#define CHECK(cond, msg) do { \
    if (!(cond)) { FAIL(msg); goto done; } \
} while(0)

/* ---- Test 1: Create and destroy all DRAM types ---- */
static void test_create_all_types(void) {
    TEST("Create all DRAM types");
    const char *expected_names[] = {
        "ideal", "HBM2", "HBM2e", "HBM3", "DDR4", "DDR5", "LPDDR5", "custom"
    };

    for (int t = 0; t < TU_DRAM_TYPE_COUNT; t++) {
        tu_dram_model_t *dram = tu_dram_create((tu_dram_type_t)t);
        CHECK(dram != NULL, "create returned NULL");
        CHECK(strcmp(tu_dram_get_name(dram), expected_names[t]) == 0,
              "wrong name");
        tu_dram_destroy(dram);
    }
    PASS();
done:;
}

/* ---- Test 2: Custom DRAM parameters ---- */
static void test_custom_params(void) {
    TEST("Custom DRAM parameters");
    tu_dram_params_t p = {
        .clock_ghz = 2.0,
        .bandwidth_gbps = 500.0,
        .read_latency_cycles = 30,
        .write_latency_cycles = 30,
        .bus_width_bytes = 64,
        .burst_length = 64,
        .channels = 4,
        .banks_per_channel = 8,
        .row_buffer_size = 4096,
        .model_row_conflicts = true
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "my-dram");
    CHECK(dram != NULL, "create_custom failed");
    CHECK(dram->type == TU_DRAM_TYPE_CUSTOM, "wrong type");
    CHECK(dram->params.bandwidth_gbps == 500.0, "wrong BW");
    CHECK(dram->params.channels == 4, "wrong channels");
    CHECK(dram->num_channels == 4, "wrong num_channels");
    CHECK(dram->params.model_row_conflicts == true, "row conflicts not set");
    tu_dram_destroy(dram);
    PASS();
done:;
}

/* ---- Test 3: Ideal DRAM has zero latency ---- */
static void test_ideal_zero_latency(void) {
    TEST("Ideal DRAM zero latency");
    tu_dram_model_t *dram = tu_dram_create(TU_DRAM_TYPE_IDEAL);
    CHECK(dram != NULL, "create failed");

    uint64_t cycles, stall;
    for (int i = 0; i < 100; i++) {
        tu_dram_read(dram, i * 64, 64, &cycles, &stall);
        CHECK(cycles == 0, "ideal read has non-zero cycles");
        CHECK(stall == 0, "ideal read has stall");
        tu_dram_write(dram, i * 64, 64, &cycles, &stall);
        CHECK(cycles == 0, "ideal write has non-zero cycles");
        CHECK(stall == 0, "ideal write has stall");
    }

    tu_dram_stats_t s;
    tu_dram_get_stats(dram, &s);
    CHECK(s.total_reads == 100, "wrong read count");
    CHECK(s.total_writes == 100, "wrong write count");
    CHECK(s.total_read_bytes == 6400, "wrong read bytes");
    CHECK(s.total_write_bytes == 6400, "wrong write bytes");

    tu_dram_destroy(dram);
    PASS();
done:;
}

/* ---- Test 4: Access timing with HBM2 ---- */
static void test_hbm2_timing(void) {
    TEST("HBM2 access timing");
    tu_dram_model_t *dram = tu_dram_create(TU_DRAM_TYPE_HBM2);
    CHECK(dram != NULL, "create failed");

    uint64_t cycles, stall;
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles > 0, "HBM2 read should have cycle cost");
    CHECK(cycles >= dram->params.read_latency_cycles,
          "cycles < latency");

    tu_dram_write(dram, 0, 64, &cycles, &stall);
    CHECK(cycles > 0, "HBM2 write should have cycle cost");

    tu_dram_destroy(dram);
    PASS();
done:;
}

/* ---- Test 5: Statistics accumulation ---- */
static void test_stats_accumulation(void) {
    TEST("Statistics accumulation");
    tu_dram_model_t *dram = tu_dram_create(TU_DRAM_TYPE_DDR5);
    CHECK(dram != NULL, "create failed");

    uint64_t cycles, stall;
    for (int i = 0; i < 50; i++) {
        tu_dram_read(dram, i * 128, 128, &cycles, &stall);
    }
    for (int i = 0; i < 30; i++) {
        tu_dram_write(dram, i * 256, 256, &cycles, &stall);
    }

    tu_dram_stats_t s;
    tu_dram_get_stats(dram, &s);
    CHECK(s.total_reads == 50, "wrong reads");
    CHECK(s.total_writes == 30, "wrong writes");
    CHECK(s.total_read_bytes == 50 * 128, "wrong read bytes");
    CHECK(s.total_write_bytes == 30 * 256, "wrong write bytes");
    CHECK(s.total_read_cycles > 0, "no read cycles");
    CHECK(s.total_write_cycles > 0, "no write cycles");

    tu_dram_destroy(dram);
    PASS();
done:;
}

/* ---- Test 6: Bandwidth estimation ---- */
static void test_bandwidth_estimation(void) {
    TEST("Bandwidth estimation");
    tu_dram_model_t *dram = tu_dram_create(TU_DRAM_TYPE_HBM3);
    CHECK(dram != NULL, "create failed");

    /* 819 GB/s HBM3. At 1 GHz core clock, that's ~819 bytes/cycle. */
    uint64_t est_read = tu_dram_estimate_transfer(dram, 819, true);
    CHECK(est_read > 0, "zero estimate");
    /* Should be roughly 1 cycle (BW) + 40 cycles (latency) = ~41 */
    CHECK(est_read >= 40 && est_read <= 60, "unexpected estimate range");

    uint64_t est_write = tu_dram_estimate_transfer(dram, 819, false);
    CHECK(est_write >= 40 && est_write <= 60, "unexpected write estimate");

    /* Large transfer: 1 MB */
    uint64_t est_large = tu_dram_estimate_transfer(dram, 1048576, true);
    CHECK(est_large > 1000, "large transfer estimate too small");

    tu_dram_destroy(dram);
    PASS();
done:;
}

/* ---- Test 7: Null safety ---- */
static void test_null_safety(void) {
    TEST("Null safety");
    uint64_t cycles = 999, stall = 999;

    tu_dram_read(NULL, 0, 64, &cycles, &stall);
    CHECK(cycles == 0, "null read cycles");
    CHECK(stall == 0, "null read stall");

    cycles = 999; stall = 999;
    tu_dram_write(NULL, 0, 64, &cycles, &stall);
    CHECK(cycles == 0, "null write cycles");
    CHECK(stall == 0, "null write stall");

    CHECK(tu_dram_estimate_transfer(NULL, 100, true) == 0, "null estimate");
    CHECK(tu_dram_get_name(NULL) != NULL, "null name"); /* returns "null" */
    tu_dram_print_stats(NULL, stdout); /* should not crash */

    PASS();
done:;
}

/* ---- Test 8: Reset clears statistics ---- */
static void test_reset(void) {
    TEST("Reset clears statistics");
    tu_dram_model_t *dram = tu_dram_create(TU_DRAM_TYPE_HBM2);
    CHECK(dram != NULL, "create failed");

    uint64_t cycles, stall;
    for (int i = 0; i < 10; i++) {
        tu_dram_read(dram, i * 64, 64, &cycles, &stall);
    }

    tu_dram_reset(dram);

    tu_dram_stats_t s;
    tu_dram_get_stats(dram, &s);
    CHECK(s.total_reads == 0, "reads not reset");
    CHECK(s.total_writes == 0, "writes not reset");
    CHECK(s.total_read_bytes == 0, "read bytes not reset");

    tu_dram_destroy(dram);
    PASS();
done:;
}

/* ---- Test 9: Tick advances cycle counter ---- */
static void test_tick_advances(void) {
    TEST("Tick advances cycle counter");
    tu_dram_model_t *dram = tu_dram_create(TU_DRAM_TYPE_HBM2);
    CHECK(dram != NULL, "create failed");

    CHECK(dram->current_cycle == 0, "initial cycle != 0");
    for (int i = 0; i < 100; i++) {
        tu_dram_tick(dram);
    }
    CHECK(dram->current_cycle == 100, "tick didn't advance");

    tu_dram_destroy(dram);
    PASS();
done:;
}

/* ---- Test 10: Peak BW per cycle calculation ---- */
static void test_peak_bw_per_cycle(void) {
    TEST("Peak BW per cycle");
    tu_dram_model_t *dram = tu_dram_create(TU_DRAM_TYPE_HBM2);
    CHECK(dram != NULL, "create failed");

    /* HBM2 = 256 GB/s, at 1 GHz = 256 bytes/cycle */
    uint64_t peak = tu_dram_peak_bw_per_cycle(dram, 1.0);
    CHECK(peak == 256, "wrong peak BW at 1GHz");

    /* At 2 GHz = 128 bytes/cycle */
    peak = tu_dram_peak_bw_per_cycle(dram, 2.0);
    CHECK(peak == 128, "wrong peak BW at 2GHz");

    tu_dram_destroy(dram);
    PASS();
done:;
}

/* ---- Test 11: Print stats doesn't crash ---- */
static void test_print_stats(void) {
    TEST("Print stats");
    tu_dram_model_t *dram = tu_dram_create(TU_DRAM_TYPE_HBM2);
    CHECK(dram != NULL, "create failed");

    uint64_t cycles, stall;
    tu_dram_read(dram, 0, 1024, &cycles, &stall);
    tu_dram_write(dram, 0, 512, &cycles, &stall);

    printf("\n"); /* newline before stats output */
    tu_dram_print_stats(dram, stdout);

    tu_dram_destroy(dram);
    PASS();
done:;
}

/* ---- Test 12: All types have valid parameters ---- */
static void test_all_types_valid(void) {
    TEST("All types have valid parameters");
    for (int t = 0; t < TU_DRAM_TYPE_COUNT - 1; t++) { /* skip CUSTOM */
        tu_dram_model_t *dram = tu_dram_create((tu_dram_type_t)t);
        CHECK(dram != NULL, "create failed");
        CHECK(dram->params.bandwidth_gbps > 0, "zero BW");
        CHECK(dram->params.channels > 0, "zero channels");
        CHECK(dram->params.banks_per_channel > 0, "zero banks");
        CHECK(dram->params.burst_length > 0, "zero burst");
        if (t != TU_DRAM_TYPE_IDEAL) {
            CHECK(dram->params.read_latency_cycles > 0, "zero latency on non-ideal");
        }
        tu_dram_destroy(dram);
    }
    PASS();
done:;
}

static tu_dram_model_t *make_row_test_dram(tu_dram_row_policy_mode_t policy) {
    tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "row-test");
    if (dram) tu_dram_set_row_policy(dram, policy, 20);
    return dram;
}

static void test_open_page_row_tracking(void) {
    TEST("Open-page row hits and misses");
    tu_dram_model_t *dram = make_row_test_dram(TU_DRAM_ROW_OPEN_PAGE);
    CHECK(dram != NULL, "create failed");
    uint64_t cycles, stall;
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 70, "first access must miss");
    tu_dram_read(dram, 64, 64, &cycles, &stall);
    CHECK(cycles == 50, "same-row access must hit");
    tu_dram_write(dram, 4096, 64, &cycles, &stall);
    CHECK(cycles == 60, "different-row write must miss");
    CHECK(dram->stats.total_row_conflicts == 2, "wrong miss count");
    CHECK(dram->stats.total_row_hits == 1, "wrong hit count");
    tu_dram_destroy(dram);
    PASS();
done:;
}

static void test_split_row_timing(void) {
    TEST("Split row activation and replacement timing");
    tu_dram_model_t *dram = make_row_test_dram(TU_DRAM_ROW_OPEN_PAGE);
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_set_row_policy_timing(dram, TU_DRAM_ROW_OPEN_PAGE, 20, 45),
          "split timing set failed");
    uint64_t cycles = 0, stall = 0;
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 70, "closed-bank activation must use activate cost");
    tu_dram_read(dram, 64, 64, &cycles, &stall);
    CHECK(cycles == 50, "same-row access must hit");
    tu_dram_read(dram, 4096, 64, &cycles, &stall);
    CHECK(cycles == 95, "open-row replacement must use conflict cost");
    CHECK(dram->stats.total_row_empty_misses == 1, "wrong empty-row count");
    CHECK(dram->stats.total_row_replacements == 1, "wrong replacement count");
    CHECK(dram->stats.total_row_conflicts == 2, "compat miss count changed");

    CHECK(tu_dram_set_row_policy(dram, TU_DRAM_ROW_OPEN_PAGE, 17),
          "legacy timing setter failed");
    CHECK(dram->row_miss_penalty_cycles == 17 &&
          dram->row_conflict_penalty_cycles == 17,
          "legacy setter must preserve equal-cost behavior");
    CHECK(!tu_dram_set_row_policy_timing(dram, (tu_dram_row_policy_mode_t)9,
                                         1, 2), "invalid policy accepted");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_adaptive_row_timeout(void) {
    TEST("Adaptive row timeout preserves short reuse and closes after idle");
    tu_dram_model_t *dram = make_row_test_dram(TU_DRAM_ROW_OPEN_PAGE);
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_set_row_policy_timeout(dram, TU_DRAM_ROW_ADAPTIVE_TIMEOUT,
                                         20, 45, 4), "timeout set failed");
    uint64_t cycles = 0, stall = 0;
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 70, "first access must activate");
    for (int i = 0; i < 4; ++i) tu_dram_tick(dram);
    tu_dram_read(dram, 64, 64, &cycles, &stall);
    CHECK(cycles == 50, "reuse at timeout boundary must hit");
    for (int i = 0; i < 5; ++i) tu_dram_tick(dram);
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 70, "reuse after timeout must reactivate");
    CHECK(dram->stats.total_row_hits == 1, "wrong adaptive hit count");
    CHECK(dram->stats.total_row_empty_misses == 2, "wrong adaptive activation count");
    CHECK(dram->stats.total_row_replacements == 0, "timeout became replacement");
    CHECK(dram->stats.total_row_timeout_precharges == 1, "timeout not counted");
    CHECK(!tu_dram_set_row_policy_timeout(dram, TU_DRAM_ROW_ADAPTIVE_TIMEOUT,
                                           20, 45, 0), "zero timeout accepted");
    CHECK(dram->row_policy == TU_DRAM_ROW_ADAPTIVE_TIMEOUT &&
           dram->row_open_timeout_cycles == 4, "failed setter mutated state");
    CHECK(tu_dram_set_row_policy_timeout_domain(
              dram, TU_DRAM_ROW_ADAPTIVE_TIMEOUT, 20, 45,
              TU_DRAM_ROW_TIMEOUT_PHYSICAL_NS, 8.0),
          "physical timeout set failed");
    CHECK(dram->row_timeout_domain == TU_DRAM_ROW_TIMEOUT_PHYSICAL_NS &&
          dram->row_open_timeout_source == 8.0 &&
          dram->row_open_timeout_cycles == 8, "physical timeout at 1 GHz");
    CHECK(tu_dram_configure_core_clock(dram, 2.0) &&
          dram->row_open_timeout_cycles == 16,
          "physical timeout not recomputed at 2 GHz");
    CHECK(!tu_dram_set_row_policy_timeout_domain(
              dram, TU_DRAM_ROW_ADAPTIVE_TIMEOUT, 20, 45,
              (tu_dram_row_timeout_domain_t)9, 8.0),
          "invalid timeout domain accepted");
    CHECK(dram->row_timeout_domain == TU_DRAM_ROW_TIMEOUT_PHYSICAL_NS &&
          dram->row_open_timeout_cycles == 16,
          "failed domain setter mutated state");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_dram_turnaround(void) {
    TEST("Per-channel read/write bus turnaround and config path");
    const tu_dram_params_t two_channel_params = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 2, .banks_per_channel = 1,
        .row_buffer_size = 256, .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = make_row_test_dram(TU_DRAM_ROW_LEGACY);
    CHECK(dram != NULL, "create failed");
    CHECK(dram->turnaround_mode == TU_DRAM_TURNAROUND_NONE,
          "compatibility turnaround default changed");
    CHECK(tu_dram_set_turnaround(dram, TU_DRAM_TURNAROUND_FIXED,
                                 TU_DRAM_TURNAROUND_CORE_CYCLES, 4, 7),
          "fixed turnaround set failed");
    uint64_t cycles = 0, stall = 0;
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 50, "first read paid turnaround");
    tu_dram_write(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 44, "read-to-write cost missing");
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 57, "write-to-read cost missing");
    CHECK(dram->stats.total_turnaround_events == 2 &&
          dram->stats.total_turnaround_cycles == 11,
          "turnaround counters wrong");
    tu_dram_reset(dram);
    tu_dram_write(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 40 && dram->stats.total_turnaround_events == 0,
          "reset did not clear direction history");
    CHECK(tu_dram_set_turnaround(dram, TU_DRAM_TURNAROUND_FIXED,
                                 TU_DRAM_TURNAROUND_PHYSICAL_NS, 3, 8),
          "physical turnaround set failed");
    CHECK(tu_dram_configure_core_clock(dram, 2.0) &&
          dram->read_to_write_turnaround_cycles == 6 &&
          dram->write_to_read_turnaround_cycles == 16,
          "physical turnaround not recomputed");
    CHECK(!tu_dram_set_turnaround(dram, (tu_dram_turnaround_mode_t)9,
                                  TU_DRAM_TURNAROUND_CORE_CYCLES, 1, 1),
          "invalid mode accepted");
    CHECK(dram->turnaround_domain == TU_DRAM_TURNAROUND_PHYSICAL_NS &&
          dram->write_to_read_turnaround_cycles == 16,
          "failed setter mutated turnaround state");
    CHECK(tu_dram_set_turnaround(dram, TU_DRAM_TURNAROUND_BURST_CREDIT,
                                 TU_DRAM_TURNAROUND_CORE_CYCLES, 4, 7),
          "burst-credit turnaround set failed");
    CHECK(dram->turnaround_mode == TU_DRAM_TURNAROUND_BURST_CREDIT,
          "burst-credit mode not retained");
    CHECK(tu_dram_set_turnaround(dram, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT,
                                 TU_DRAM_TURNAROUND_CORE_CYCLES, 4, 7),
          "burst-rounded turnaround set failed");
    CHECK(dram->turnaround_mode == TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT,
          "burst-rounded mode not retained");
    tu_dram_reset(dram);
    for (int i = 0; i < 1001; ++i) tu_dram_tick(dram);
    uint64_t bandwidth_before = dram->bandwidth_available;
    tu_dram_read(dram, 0, 16, &cycles, &stall);
    tu_dram_write(dram, 0, 80, &cycles, &stall);
    CHECK(dram->stats.total_read_bytes == 16 &&
          dram->stats.total_write_bytes == 80,
          "rounded mode changed logical byte counters");
    CHECK(dram->stats.total_read_occupied_bytes == 64 &&
          dram->stats.total_write_occupied_bytes == 128,
          "rounded occupied-byte counters wrong");
    CHECK(dram->pending_read_bytes == 64 && dram->pending_write_bytes == 128 &&
          dram->bandwidth_available == bandwidth_before - 192,
          "rounded occupancy not wired into bandwidth window");
    tu_dram_destroy(dram);
    dram = tu_dram_create_custom(&two_channel_params, "turnaround-channels");
    CHECK(dram != NULL &&
          tu_dram_set_turnaround(dram, TU_DRAM_TURNAROUND_FIXED,
                                 TU_DRAM_TURNAROUND_CORE_CYCLES, 4, 7),
          "two-channel turnaround setup failed");
    tu_dram_read(dram, 0, 64, &cycles, &stall);   /* channel 0: first read */
    tu_dram_write(dram, 64, 64, &cycles, &stall); /* channel 1: first write */
    CHECK(cycles == 40 && dram->stats.total_turnaround_events == 0,
          "direction history leaked across channels");
    tu_dram_write(dram, 0, 64, &cycles, &stall);  /* channel 0: R -> W */
    CHECK(cycles == 44 && dram->stats.total_turnaround_events == 1 &&
          dram->stats.total_turnaround_cycles == 4,
          "channel-local direction change not charged");
    tu_dram_destroy(dram);
    dram = NULL;

    tu_config_t cfg;
    char err[192];
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"type\":\"ddr5\","
        "\"core_clock_ghz\":2,\"turnaround_mode\":\"fixed\","
        "\"turnaround_domain\":\"physical_ns\","
        "\"read_to_write_turnaround\":3,\"write_to_read_turnaround\":8}}}}",
        &cfg, err, sizeof(err)) == 0, "turnaround config parse");
    dram = tu_dram_create_from_config(&cfg);
    CHECK(dram != NULL && dram->turnaround_mode == TU_DRAM_TURNAROUND_FIXED &&
          dram->read_to_write_turnaround_cycles == 6 &&
          dram->write_to_read_turnaround_cycles == 16,
          "turnaround config propagation");
    tu_dram_destroy(dram);
    dram = NULL;
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"turnaround_mode\":\"fixed\"}}}}",
        &cfg, err, sizeof(err)) != 0, "zero fixed turnaround accepted");
    CHECK(strstr(err, "turnaround") != NULL, "wrong turnaround error");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_closed_page_and_reset(void) {
    TEST("Closed-page policy and reset");
    tu_dram_model_t *dram = make_row_test_dram(TU_DRAM_ROW_CLOSED_PAGE);
    CHECK(dram != NULL, "create failed");
    uint64_t cycles, stall;
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 70, "closed read miss");
    tu_dram_read(dram, 64, 64, &cycles, &stall);
    CHECK(cycles == 70, "closed sequential read must miss");
    CHECK(dram->stats.total_row_conflicts == 2, "closed miss count");
    tu_dram_reset(dram);
    CHECK(dram->stats.total_row_conflicts == 0 && dram->stats.total_row_hits == 0,
          "reset row stats");
    tu_dram_destroy(dram);
    PASS();
done:;
}

static void test_legacy_row_compatibility(void) {
    TEST("Legacy row-conflict compatibility");
    tu_dram_model_t *dram = make_row_test_dram(TU_DRAM_ROW_LEGACY);
    CHECK(dram != NULL, "create failed");
    tu_dram_set_row_modeling(dram, true);
    uint64_t cycles, stall;
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 70, "legacy read penalty changed");
    tu_dram_write(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 40, "legacy write must remain unpenalized");
    CHECK(dram->stats.total_row_conflicts == 1, "legacy conflict count");
    CHECK(dram->stats.total_row_hits == 0, "legacy must not synthesize hits");
    tu_dram_destroy(dram);
    PASS();
done:;
}

static void test_row_policy_config_path(void) {
    TEST("Row policy and address mapping canonical config path");
    const char *json = "{\"tu\":{\"memory\":{\"dram\":{"
                       "\"type\":\"ddr5\",\"row_policy\":\"open_page\","
                       "\"address_mapping\":\"xor_interleaved\","
                       "\"row_miss_penalty_cycles\":23,"
                       "\"row_conflict_penalty_cycles\":41}}}}";
    tu_config_t cfg;
    CHECK(tu_config_load_string(json, &cfg, NULL, 0) == 0, "config parse");
    CHECK(cfg.dram_row_policy == TU_DRAM_CONFIG_ROW_OPEN_PAGE, "policy parse");
    CHECK(cfg.dram_address_mapping == TU_DRAM_CONFIG_ADDR_XOR_INTERLEAVED,
          "mapping parse");
    CHECK(cfg.dram_row_miss_penalty_cycles == 23, "penalty parse");
    CHECK(cfg.dram_row_conflict_penalty_cycles == 41, "conflict penalty parse");
    tu_dram_model_t *dram = tu_dram_create_from_config(&cfg);
    CHECK(dram != NULL, "create from config");
    CHECK(dram->row_policy == TU_DRAM_ROW_OPEN_PAGE, "policy propagation");
    CHECK(dram->address_mapping == TU_DRAM_ADDR_XOR_INTERLEAVED,
          "mapping propagation");
    CHECK(dram->row_miss_penalty_cycles == 23, "penalty propagation");
    CHECK(dram->row_conflict_penalty_cycles == 41, "conflict penalty propagation");
    tu_dram_destroy(dram);

    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"type\":\"ddr5\","
        "\"row_policy\":\"open_page\",\"row_miss_penalty_cycles\":31}}}}",
        &cfg, NULL, 0) == 0, "compat config parse");
    CHECK(cfg.dram_row_conflict_penalty_cycles == 0, "absent conflict cost must inherit");
    dram = tu_dram_create_from_config(&cfg);
    CHECK(dram != NULL && dram->row_conflict_penalty_cycles == 31,
          "absent conflict cost did not preserve equal-cost behavior");
    tu_dram_destroy(dram);

    char err[128];
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"row_policy\":\"magic\"}}}}",
        &cfg, err, sizeof(err)) != 0, "invalid policy accepted");
    CHECK(strstr(err, "row_policy") != NULL, "wrong validation error");
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"address_mapping\":\"xor_magic\"}}}}",
        &cfg, err, sizeof(err)) != 0, "invalid mapping accepted");
    CHECK(strstr(err, "address_mapping") != NULL, "wrong mapping validation error");
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"row_conflict_penalty_cycles\":-1}}}}",
        &cfg, err, sizeof(err)) != 0, "invalid conflict penalty accepted");
    CHECK(strstr(err, "row timing") != NULL, "wrong row timing validation error");

    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"type\":\"ddr5\","
        "\"row_policy\":\"adaptive_timeout\",\"row_open_timeout_cycles\":77}}}}",
        &cfg, err, sizeof(err)) == 0, "adaptive policy parse");
    CHECK(cfg.dram_row_policy == TU_DRAM_CONFIG_ROW_ADAPTIVE_TIMEOUT &&
          cfg.dram_row_open_timeout_cycles == 77, "adaptive config value");
    dram = tu_dram_create_from_config(&cfg);
    CHECK(dram != NULL && dram->row_policy == TU_DRAM_ROW_ADAPTIVE_TIMEOUT &&
          dram->row_open_timeout_cycles == 77, "adaptive propagation");
    tu_dram_destroy(dram);
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"row_policy\":\"adaptive_timeout\","
        "\"row_open_timeout_cycles\":0}}}}",
        &cfg, err, sizeof(err)) != 0, "zero adaptive timeout accepted");
    CHECK(strstr(err, "timeout") != NULL, "wrong adaptive timeout error");
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"type\":\"ddr5\","
        "\"core_clock_ghz\":2.0,\"row_policy\":\"adaptive_timeout\","
        "\"row_timeout_domain\":\"physical_ns\",\"row_open_timeout_ns\":8.5}}}}",
        &cfg, err, sizeof(err)) == 0, "physical timeout config parse");
    dram = tu_dram_create_from_config(&cfg);
    CHECK(dram != NULL &&
          dram->row_timeout_domain == TU_DRAM_ROW_TIMEOUT_PHYSICAL_NS &&
          dram->row_open_timeout_source == 8.5 &&
          dram->row_open_timeout_cycles == 17,
          "physical timeout config propagation");
    tu_dram_destroy(dram);
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"row_timeout_domain\":\"dram_cycles\"}}}}",
        &cfg, err, sizeof(err)) != 0, "invalid timeout domain accepted");
    CHECK(strstr(err, "row_timeout_domain") != NULL,
          "wrong timeout-domain error");
    PASS();
done:;
}

static void test_address_mapping_decode(void) {
    TEST("Burst, row, and XOR address mapping decode");
    tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 4, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "mapping-test");
    CHECK(dram != NULL, "create failed");
    uint32_t channel = 99, bank = 99;
    uint64_t row = 99;
    CHECK(tu_dram_decode_address(dram, 256, &channel, &bank, &row),
          "burst decode failed");
    CHECK(channel == 0 && bank == 0 && row == 0, "burst decode mismatch");
    CHECK(tu_dram_decode_address(dram, 64, &channel, &bank, &row),
          "burst stripe decode failed");
    CHECK(channel == 1 && bank == 0 && row == 0, "burst stripe mismatch");

    CHECK(tu_dram_set_row_policy(dram, TU_DRAM_ROW_OPEN_PAGE, 20),
          "open-page set failed");
    uint64_t cycles = 0, stall = 0;
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 70, "initial row access must miss");
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 50, "repeated row access must hit");

    CHECK(tu_dram_set_address_mapping(dram, TU_DRAM_ADDR_ROW_INTERLEAVED),
          "row mapping set failed");
    CHECK(tu_dram_decode_address(dram, 256, &channel, &bank, &row),
          "row decode failed");
    CHECK(channel == 1 && bank == 0 && row == 0, "row decode mismatch");
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 70, "mapping switch did not clear open-row state");

    CHECK(tu_dram_set_address_mapping(dram, TU_DRAM_ADDR_XOR_INTERLEAVED),
          "XOR mapping set failed");
    CHECK(tu_dram_decode_address(dram, 1024, &channel, &bank, &row),
          "XOR decode failed");
    CHECK(channel == 1 && bank == 1 && row == 0, "XOR decode mismatch");
    CHECK(!tu_dram_set_address_mapping(dram, (tu_dram_address_mapping_mode_t)9),
          "invalid mapping accepted");
    dram->address_mapping = (tu_dram_address_mapping_mode_t)9;
    CHECK(!tu_dram_decode_address(dram, 0, &channel, &bank, &row),
          "invalid live mapping decoded with fallback");
    tu_dram_destroy(dram);
    PASS();
done:;
}

static void test_xor_mapping_constraints(void) {
    TEST("XOR mapping power-of-two channel constraint");
    tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 3, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "xor-invalid");
    CHECK(dram != NULL, "create failed");
    CHECK(!tu_dram_set_address_mapping(dram, TU_DRAM_ADDR_XOR_INTERLEAVED),
          "XOR accepted non-power-of-two channels");
    tu_dram_destroy(dram);

    tu_config_t cfg;
    char err[160];
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"channels\":3,"
        "\"address_mapping\":\"xor_interleaved\"}}}}",
        &cfg, err, sizeof(err)) != 0, "config accepted unsupported XOR geometry");
    CHECK(strstr(err, "power-of-two") != NULL, "wrong XOR geometry error");
    PASS();
done:;
}

/* ---- DRAM refresh (JEDEC tREFI/tRFC) ---- */

/* 4-channel / 4-bank / 256 B row custom DRAM for refresh tests.
 * Burst-interleaved decode: burst=addr/64, ch=burst%4, group=(burst/4)/4,
 * bank=group%4. Addresses hitting distinct channels avoid contention noise. */
static tu_dram_model_t *make_refresh_dram(void) {
    tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 4, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    return tu_dram_create_custom(&p, "refresh-test");
}

static void test_refresh_all_bank_fixed(void) {
    TEST("All-bank fixed refresh lockout");
    tu_dram_model_t *dram = make_refresh_dram();
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                              TU_DRAM_REFRESH_SCHEDULING_FIXED,
                              1, 5000, 100, 30, 5000), "set refresh");
    uint64_t cycles = 0, stall = 0;

    /* Prime the bandwidth window (first refill at cycle 1001) so the
     * coarse BW metering cannot mask refresh accounting. First refresh
     * is scheduled at tREFI=5000. */
    for (int i = 0; i < 1001; ++i) tu_dram_tick(dram);  /* T=1001 */
    tu_dram_read(dram, 0, 64, &cycles, &stall);         /* ch0 */
    CHECK(cycles == 50 && stall == 0, "pre-refresh access unperturbed");
    CHECK(dram->stats.total_refresh_events == 0, "no refresh yet");

    for (int i = 0; i < 3999; ++i) tu_dram_tick(dram);  /* T=5000 */
    tu_dram_read(dram, 64, 64, &cycles, &stall);        /* ch1 */
    CHECK(cycles == 150 && stall == 0, "access inside tRFC window stalls");
    CHECK(dram->stats.total_refresh_events == 1, "one refresh fired");
    CHECK(dram->stats.total_refresh_stall_cycles == 100, "refresh stall count");

    for (int i = 0; i < 100; ++i) tu_dram_tick(dram);   /* T=5100 */
    tu_dram_read(dram, 128, 64, &cycles, &stall);       /* ch2 */
    CHECK(cycles == 50 && stall == 0, "access after window unperturbed");

    for (int i = 0; i < 4900; ++i) tu_dram_tick(dram);  /* T=10000 */
    tu_dram_read(dram, 192, 64, &cycles, &stall);       /* ch3 */
    CHECK(cycles == 150 && stall == 0, "second refresh window stalls");
    CHECK(dram->stats.total_refresh_events == 2, "two refreshes fired");
    CHECK(dram->stats.total_refresh_stall_cycles == 200, "cumulative refresh stalls");
    CHECK(dram->stats.total_stall_cycles == 0,
          "refresh lockout kept out of the contention-stall domain");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_refresh_per_bank_staggered(void) {
    TEST("Per-bank staggered refresh isolates banks");
    tu_dram_model_t *dram = make_refresh_dram();
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_set_refresh(dram, TU_DRAM_REFRESH_PER_BANK,
                              TU_DRAM_REFRESH_SCHEDULING_FIXED,
                              1, 4000, 100, 40, 4000), "set refresh");
    uint64_t cycles = 0, stall = 0;

    /* Staggered schedules: bank b first refreshes at (b+1)*4000/4.
     * Priming to 1001 also crosses bank0's refresh at 1000. */
    for (int i = 0; i < 1001; ++i) tu_dram_tick(dram);  /* T=1001 */
    CHECK(dram->stats.total_refresh_events == 1, "bank0 fired at 1000");

    /* addr 2176: burst 34 → ch2, bank2 — not refreshing → unperturbed. */
    tu_dram_read(dram, 2176, 64, &cycles, &stall);
    CHECK(cycles == 50 && stall == 0, "other bank unperturbed during per-bank refresh");

    for (int i = 0; i < 999; ++i) tu_dram_tick(dram);   /* T=2000 */
    CHECK(dram->stats.total_refresh_events == 2, "bank1 fired at 2000");

    /* addr 1024: burst 16 → ch0, bank1 — refreshing → pays tRFCpb remainder. */
    tu_dram_read(dram, 1024, 64, &cycles, &stall);
    CHECK(cycles == 90 && stall == 0, "refreshing bank stalls for tRFCpb");
    CHECK(dram->stats.total_refresh_stall_cycles == 40, "per-bank stall count");

    /* All banks fire exactly once over the first tREFI window. */
    for (int i = 0; i < 2000; ++i) tu_dram_tick(dram);  /* T=4000 */
    CHECK(dram->stats.total_refresh_events == 4, "all four banks refreshed once");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_refresh_deferred_opportunistic(void) {
    TEST("Deferred refresh fires at access or hard deadline");
    tu_dram_model_t *dram = make_refresh_dram();
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                              TU_DRAM_REFRESH_SCHEDULING_DEFERRED,
                              1, 5000, 100, 30, 2000), "set refresh");
    uint64_t cycles = 0, stall = 0;

    for (int i = 0; i < 1001; ++i) tu_dram_tick(dram);  /* T=1001 */
    tu_dram_read(dram, 0, 64, &cycles, &stall);         /* T=1001, ch0 */
    CHECK(cycles == 50 && stall == 0, "pre-schedule access unperturbed");

    /* Schedule at 5000, deadline at 7000. Access at 5100 fires it
     * opportunistically at the access, paying the full tRFC. */
    for (int i = 0; i < 4099; ++i) tu_dram_tick(dram);  /* T=5100 */
    tu_dram_read(dram, 64, 64, &cycles, &stall);        /* ch1 */
    CHECK(cycles == 150 && stall == 0, "opportunistic fire at access");
    CHECK(dram->stats.total_refresh_events == 1, "opportunistic refresh counted");

    /* Next schedule at 10000, deadline at 12000. No access before the
     * deadline → tick fires it at the deadline. */
    for (int i = 0; i < 5000; ++i) tu_dram_tick(dram);  /* T=10100 */
    CHECK(dram->stats.total_refresh_events == 1, "no fire before deadline");
    for (int i = 0; i < 1900; ++i) tu_dram_tick(dram);  /* T=12000: deadline passed */
    CHECK(dram->stats.total_refresh_events == 2, "forced fire at deadline");
    tu_dram_read(dram, 128, 64, &cycles, &stall);       /* ch2, T=12000 */
    CHECK(cycles == 150 && stall == 0, "access overlapping forced refresh stalls");
    CHECK(dram->stats.total_refresh_stall_cycles == 200, "deferred stall total");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_refresh_rate_multiplier(void) {
    TEST("Refresh rate multiplier doubles events at 2x");
    tu_dram_model_t *dram = make_refresh_dram();
    CHECK(dram != NULL, "create failed");

    CHECK(tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                              TU_DRAM_REFRESH_SCHEDULING_FIXED,
                              2, 2000, 50, 30, 2000), "set refresh 2x");
    for (int i = 0; i < 4000; ++i) tu_dram_tick(dram);
    CHECK(dram->stats.total_refresh_events == 4, "2x rate fires every 1000 cycles");
    tu_dram_destroy(dram);
    dram = NULL;

    dram = make_refresh_dram();
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                              TU_DRAM_REFRESH_SCHEDULING_FIXED,
                              1, 2000, 50, 30, 2000), "set refresh 1x");
    for (int i = 0; i < 4000; ++i) tu_dram_tick(dram);
    CHECK(dram->stats.total_refresh_events == 2, "1x rate fires every 2000 cycles");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_refresh_none_compat(void) {
    TEST("Refresh NONE preserves legacy behavior");
    tu_dram_model_t *dram = make_refresh_dram();
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_set_refresh(dram, TU_DRAM_REFRESH_NONE,
                              TU_DRAM_REFRESH_SCHEDULING_FIXED,
                              1, 1000, 100, 30, 1000), "set refresh none");
    uint64_t cycles = 0, stall = 0;
    /* Prime the bandwidth window, then a steady single-address stream. */
    for (int i = 0; i < 1001; ++i) tu_dram_tick(dram);
    for (int i = 0; i < 40; ++i) {
        tu_dram_read(dram, 0, 64, &cycles, &stall);
        if (cycles != 50 || stall != 0) { FAIL("legacy cycles changed"); goto done; }
        for (int t = 0; t < 60; ++t) tu_dram_tick(dram);
    }
    CHECK(dram->stats.total_refresh_events == 0, "no refresh events");
    CHECK(dram->stats.total_refresh_stall_cycles == 0, "no refresh stalls");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_refresh_reset(void) {
    TEST("Reset clears refresh state and counters");
    tu_dram_model_t *dram = make_refresh_dram();
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                              TU_DRAM_REFRESH_SCHEDULING_FIXED,
                              1, 1000, 100, 30, 1000), "set refresh");
    for (int i = 0; i < 1000; ++i) tu_dram_tick(dram);
    CHECK(dram->stats.total_refresh_events == 1, "refresh fired before reset");
    tu_dram_reset(dram);
    CHECK(dram->stats.total_refresh_events == 0, "reset clears events");
    CHECK(dram->current_cycle == 0, "reset rewinds clock");
    uint64_t cycles = 0, stall = 0;
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 50 && stall == 0, "schedule rebuilt after reset");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_refresh_rejects(void) {
    TEST("Refresh setter fails closed on unsupported input");
    tu_dram_model_t *dram = make_refresh_dram();
    CHECK(dram != NULL, "create failed");
    CHECK(!tu_dram_set_refresh(dram, (tu_dram_refresh_mode_t)9,
                               TU_DRAM_REFRESH_SCHEDULING_FIXED,
                               1, 1000, 100, 30, 1000), "bad mode accepted");
    CHECK(!tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                               (tu_dram_refresh_scheduling_t)5,
                               1, 1000, 100, 30, 1000), "bad scheduling accepted");
    CHECK(!tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                               TU_DRAM_REFRESH_SCHEDULING_FIXED,
                               3, 1000, 100, 30, 1000), "bad rate accepted");
    CHECK(!tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                               TU_DRAM_REFRESH_SCHEDULING_DEFERRED,
                               1, 1000, 100, 30, 2000),
          "deferral beyond tREFI accepted");
    CHECK(dram->refresh_mode == TU_DRAM_REFRESH_NONE, "failed setter must not mutate");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_refresh_config_path(void) {
    TEST("Refresh canonical config parse and propagation");
    tu_dram_model_t *dram = NULL;
    const char *json = "{\"tu\":{\"memory\":{\"dram\":{\"type\":\"ddr5\","
                       "\"refresh\":{\"mode\":\"per_bank\",\"scheduling\":\"deferred\","
                       "\"rate\":2,\"trefi_ns\":3900,\"trfc_ns\":280,"
                       "\"trfc_pb_ns\":70,\"max_deferral_ns\":1950}}}}}";
    tu_config_t cfg;
    CHECK(tu_config_load_string(json, &cfg, NULL, 0) == 0, "config parse");
    CHECK(cfg.dram_refresh_mode == TU_DRAM_CONFIG_REFRESH_PER_BANK, "mode parse");
    CHECK(cfg.dram_refresh_scheduling == TU_DRAM_CONFIG_REFRESH_SCHED_DEFERRED,
          "scheduling parse");
    CHECK(cfg.dram_refresh_rate == 2, "rate parse");
    CHECK(cfg.dram_trefi_ns == 3900, "trefi parse");
    CHECK(cfg.dram_refresh_max_deferral_ns == 1950, "deferral parse");

    dram = tu_dram_create_from_config(&cfg);
    CHECK(dram != NULL, "create from config");
    CHECK(dram->refresh_mode == TU_DRAM_REFRESH_PER_BANK, "mode propagation");
    CHECK(dram->refresh_scheduling == TU_DRAM_REFRESH_SCHEDULING_DEFERRED,
          "scheduling propagation");
    CHECK(dram->refresh_rate == 2, "rate propagation");
    CHECK(dram->refresh_trefi_cycles == 3900, "base trefi propagation");
    CHECK(dram->refresh_trfc_pb_cycles == 70, "per-bank lockout propagation");
    CHECK(dram->refresh_max_deferral_cycles == 1950, "deferral propagation");
    tu_dram_destroy(dram);
    dram = NULL;

    /* Zero-initialized callers: mode 0 (NONE) and zero timings mean defaults. */
    tu_config_t zc;
    memset(&zc, 0, sizeof(zc));
    zc.dram_type = TU_DRAM_TYPE_DDR5;
    dram = tu_dram_create_from_config(&zc);
    CHECK(dram != NULL, "zero-init create");
    CHECK(dram->core_clock_ghz == 1.0, "zero-init clock compatibility");
    CHECK(dram->refresh_mode == TU_DRAM_REFRESH_NONE, "zero-init legacy mode");
    CHECK(dram->refresh_rate == 1, "zero rate defaults to 1x");
    CHECK(dram->refresh_trefi_cycles == 7800, "zero trefi defaults to 7800");
    tu_dram_destroy(dram);
    dram = NULL;

    char err[128];
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"refresh\":{\"mode\":\"all_bank\","
        "\"max_deferral_ns\":99999}}}}}",
        &cfg, err, sizeof(err)) != 0, "deferral > tREFI accepted by validation");
    CHECK(strstr(err, "max_deferral") != NULL, "wrong deferral error");
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"refresh\":{\"mode\":\"magic\"}}}}}",
        &cfg, err, sizeof(err)) != 0, "invalid refresh mode accepted");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_refresh_closes_rows(void) {
    TEST("Refresh precharges open row buffers");
    tu_dram_model_t *dram = make_refresh_dram();
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_set_row_policy(dram, TU_DRAM_ROW_OPEN_PAGE, 20),
          "open-page set failed");
    CHECK(tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                              TU_DRAM_REFRESH_SCHEDULING_FIXED,
                              1, 1000, 100, 30, 1000), "set refresh");
    uint64_t cycles = 0, stall = 0;
    tu_dram_read(dram, 0, 64, &cycles, &stall);    /* ch0, bank0, row0: miss */
    CHECK(cycles == 70, "first access misses");
    for (int i = 0; i < 71; ++i) tu_dram_tick(dram);  /* T=71, clear channel */
    tu_dram_read(dram, 256, 64, &cycles, &stall);  /* same ch0/bank0/row0: hit */
    CHECK(cycles == 50, "same-row access hits");
    CHECK(dram->stats.total_row_hits == 1, "one hit before refresh");

    for (int i = 0; i < 929; ++i) tu_dram_tick(dram);  /* T=1000: refresh closes rows */
    tu_dram_read(dram, 512, 64, &cycles, &stall);  /* ch0, bank0, row0, during refresh */
    CHECK(cycles == 170, "refresh lockout plus reopened-row miss");
    CHECK(dram->stats.total_row_hits == 1, "refresh invalidated the open row");
    CHECK(dram->stats.total_row_conflicts == 2, "post-refresh access misses");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_core_clock_cycle_domain(void) {
    TEST("Core clock drives bandwidth and refresh cycle conversion");
    tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "clock-test");
    CHECK(dram != NULL, "create failed");
    CHECK(tu_dram_configure_core_clock(dram, 0.5), "0.5 GHz rejected");
    CHECK(tu_dram_estimate_transfer(dram, 4096, true) == 82,
          "0.5 GHz bandwidth cycles");
    CHECK(tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                              TU_DRAM_REFRESH_SCHEDULING_FIXED,
                              1, 1000, 100, 40, 500), "refresh set");
    CHECK(dram->refresh_trefi_cycles == 500 && dram->refresh_trfc_cycles == 50,
          "0.5 GHz refresh conversion");

    CHECK(tu_dram_configure_core_clock(dram, 2.0), "2 GHz rejected");
    CHECK(tu_dram_estimate_transfer(dram, 4096, true) == 178,
          "2 GHz bandwidth cycles");
    CHECK(dram->refresh_trefi_cycles == 2000 && dram->refresh_trfc_cycles == 200,
          "clock change did not rebuild refresh cycles");
    CHECK(dram->refresh_next[0] == 2000, "clock change did not rebuild schedule");
    CHECK(!tu_dram_configure_core_clock(dram, 0.0), "zero clock accepted");
    CHECK(!tu_dram_configure_core_clock(dram, 11.0), "out-of-range clock accepted");
    CHECK(dram->core_clock_ghz == 2.0, "failed setter mutated clock");

    tu_config_t cfg;
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"dram\":{\"type\":\"ddr5\","
        "\"core_clock_ghz\":1.5,\"refresh\":{\"mode\":\"all_bank\","
        "\"trefi_ns\":1000,\"trfc_ns\":100,\"max_deferral_ns\":1000}}}}}",
        &cfg, NULL, 0) == 0, "clock config parse");
    tu_dram_destroy(dram);
    dram = tu_dram_create_from_config(&cfg);
    CHECK(dram != NULL && dram->core_clock_ghz == 1.5,
          "clock config propagation");
    CHECK(dram->refresh_trefi_cycles == 1500 && dram->refresh_trfc_cycles == 150,
          "config refresh cycle conversion");
    PASS();
done:;
    tu_dram_destroy(dram);
}

static void test_latency_domain(void) {
    TEST("DRAM base latency supports core-cycle and physical-ns domains");
    tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    uint64_t cycles = 0, stall = 0;
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "latency-domain-test");
    CHECK(dram != NULL, "create failed");
    CHECK(dram->latency_domain == TU_DRAM_LATENCY_CORE_CYCLES,
          "custom default changed");
    CHECK(tu_dram_configure_core_clock(dram, 0.5), "0.5 GHz rejected");
    CHECK(dram->params.read_latency_cycles == 50, "core-cycle latency rescaled");
    CHECK(tu_dram_set_latency_domain(dram, TU_DRAM_LATENCY_PHYSICAL_NS,
                                     50.0, 40.0), "physical-ns mode rejected");
    CHECK(dram->params.read_latency_cycles == 25 &&
          dram->params.write_latency_cycles == 20, "0.5 GHz ns conversion");
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    CHECK(cycles == 25, "physical-ns read service did not use converted latency");
    tu_dram_write(dram, 64, 64, &cycles, &stall);
    CHECK(cycles == 20, "physical-ns write service did not use converted latency");
    CHECK(tu_dram_estimate_transfer(dram, 4096, true) == 57,
          "0.5 GHz physical estimate");
    CHECK(tu_dram_configure_core_clock(dram, 2.0), "2 GHz rejected");
    CHECK(dram->params.read_latency_cycles == 100 &&
          dram->params.write_latency_cycles == 80, "clock did not rescale latency");
    CHECK(tu_dram_estimate_transfer(dram, 4096, true) == 228,
          "2 GHz physical estimate");
    CHECK(!tu_dram_set_latency_domain(dram, (tu_dram_latency_domain_t)9,
                                      50.0, 40.0), "unknown domain accepted");
    CHECK(!tu_dram_set_latency_domain(dram, TU_DRAM_LATENCY_PHYSICAL_NS,
                                      -1.0, 40.0), "negative latency accepted");
    CHECK(dram->latency_domain == TU_DRAM_LATENCY_PHYSICAL_NS &&
          dram->params.read_latency_cycles == 100, "failed setter mutated state");

    tu_config_t cfg;
    CHECK(tu_config_load_string(
        "{\"tu\":{\"memory\":{\"latency\":{\"dram_read\":65,\"dram_write\":45},"
        "\"dram\":{\"type\":\"ddr5\",\"latency_domain\":\"physical_ns\","
        "\"core_clock_ghz\":1.5}}}}", &cfg, NULL, 0) == 0,
        "physical-ns config parse");
    tu_dram_destroy(dram);
    dram = tu_dram_create_from_config(&cfg);
    CHECK(dram != NULL && dram->latency_domain == TU_DRAM_LATENCY_PHYSICAL_NS,
          "config domain propagation");
    CHECK(dram->params.read_latency_cycles == 98 &&
          dram->params.write_latency_cycles == 68, "config ns conversion");

    memset(&cfg, 0, sizeof(cfg));
    cfg.dram_type = TU_DRAM_TYPE_DDR5;
    dram = (tu_dram_destroy(dram), tu_dram_create_from_config(&cfg));
    CHECK(dram != NULL && dram->latency_domain == TU_DRAM_LATENCY_CORE_CYCLES,
          "zero-init compatibility domain");
    CHECK(dram->params.read_latency_cycles == 0, "zero-init latency changed");
    PASS();
done:;
    tu_dram_destroy(dram);
}

/* ---- Main ---- */
int main(void) {
    printf("\n=== TU DRAM Model Tests ===\n\n");

    test_create_all_types();
    test_custom_params();
    test_ideal_zero_latency();
    test_hbm2_timing();
    test_stats_accumulation();
    test_bandwidth_estimation();
    test_null_safety();
    test_reset();
    test_tick_advances();
    test_peak_bw_per_cycle();
    test_print_stats();
    test_all_types_valid();
    test_open_page_row_tracking();
    test_split_row_timing();
    test_adaptive_row_timeout();
    test_dram_turnaround();
    test_closed_page_and_reset();
    test_legacy_row_compatibility();
    test_row_policy_config_path();
    test_address_mapping_decode();
    test_xor_mapping_constraints();
    test_refresh_all_bank_fixed();
    test_refresh_per_bank_staggered();
    test_refresh_deferred_opportunistic();
    test_refresh_rate_multiplier();
    test_refresh_none_compat();
    test_refresh_reset();
    test_refresh_rejects();
    test_refresh_config_path();
    test_refresh_closes_rows();
    test_core_clock_cycle_domain();
    test_latency_domain();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
