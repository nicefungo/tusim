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
                       "\"address_mapping\":\"row_interleaved\","
                       "\"row_miss_penalty_cycles\":23}}}}";
    tu_config_t cfg;
    CHECK(tu_config_load_string(json, &cfg, NULL, 0) == 0, "config parse");
    CHECK(cfg.dram_row_policy == TU_DRAM_CONFIG_ROW_OPEN_PAGE, "policy parse");
    CHECK(cfg.dram_address_mapping == TU_DRAM_CONFIG_ADDR_ROW_INTERLEAVED,
          "mapping parse");
    CHECK(cfg.dram_row_miss_penalty_cycles == 23, "penalty parse");
    tu_dram_model_t *dram = tu_dram_create_from_config(&cfg);
    CHECK(dram != NULL, "create from config");
    CHECK(dram->row_policy == TU_DRAM_ROW_OPEN_PAGE, "policy propagation");
    CHECK(dram->address_mapping == TU_DRAM_ADDR_ROW_INTERLEAVED,
          "mapping propagation");
    CHECK(dram->row_miss_penalty_cycles == 23, "penalty propagation");
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
    PASS();
done:;
}

static void test_address_mapping_decode(void) {
    TEST("Burst and row address mapping decode");
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
    CHECK(!tu_dram_set_address_mapping(dram, (tu_dram_address_mapping_mode_t)9),
          "invalid mapping accepted");
    dram->address_mapping = (tu_dram_address_mapping_mode_t)9;
    CHECK(!tu_dram_decode_address(dram, 0, &channel, &bank, &row),
          "invalid live mapping decoded with fallback");
    tu_dram_destroy(dram);
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
    test_open_page_row_tracking();
    test_closed_page_and_reset();
    test_legacy_row_compatibility();
    test_row_policy_config_path();
    test_address_mapping_decode();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
