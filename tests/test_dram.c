/*
 * TU CModel — DRAM Model Test Suite
 * ==================================
 * Tests: creation, type enumeration, access timing, statistics,
 * bandwidth metering, channel contention, custom parameters,
 * estimation correctness.
 */

#include "tu_cmodel/memory/dram_model.h"
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

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
