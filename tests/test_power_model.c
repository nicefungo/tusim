/*
 * TU CModel — Power/Energy Model Tests (Gap E4)
 * ==============================================
 *
 * Tests:
 *   1. tech_node_lookup          — all 6 technology nodes exist
 *   2. tech_node_from_string     — string → node conversion
 *   3. power_model_init          — default initialization
 *   4. power_model_tech_nodes    — all nodes have valid energy params
 *   5. power_mac_recording       — MAC energy by precision
 *   6. power_memory_recording    — memory hierarchy energy
 *   7. power_dram_recording      — DRAM with page hit/miss
 *   8. power_dma_recording       — DMA bus energy
 *   9. power_tick                — clock + leakage accumulation
 *  10. power_total               — total energy computation
 *  11. power_avg_power           — average power in mW
 *  12. power_breakdown           — energy breakdown fractions
 *  13. power_area_estimate       — chip area estimation
 *  14. power_snapshot_diff       — interval profiling
 *  15. power_reset               — reset preserves config
 *  16. power_tech_switch         — mid-run technology node switch
 *  17. power_disable             — disabled model records nothing
 *  18. power_energy_scaling      — energy scales with node (7nm < 45nm)
 *  19. power_config_integration  — tu_power_model_from_config
 *  20. power_numeric_stability   — large values, zero values
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../tu_cmodel/perf/power_model.h"
#include "../tu_cmodel/infra/config.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  TEST %2d: %-50s", tests_run, name); \
} while(0)

#define PASS() do { \
    tests_passed++; \
    printf(" PASS\n"); \
} while(0)

#define FAIL(msg) do { \
    printf(" FAIL — %s\n", msg); \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "%s (got %lld, expected %lld)", msg, \
                 (long long)(a), (long long)(b)); \
        FAIL(buf); return; \
    } \
} while(0)

#define ASSERT_DOUBLE_NEAR(a, b, tol, msg) do { \
    if (fabs((a) - (b)) > (tol)) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "%s (got %.6f, expected %.6f)", msg, a, b); \
        FAIL(buf); return; \
    } \
} while(0)

/* ---- Test 1: All technology nodes exist ---- */
static void test_tech_node_lookup(void) {
    TEST("tech_node_lookup");
    for (int i = 0; i < TU_TECH_NODE_COUNT; i++) {
        const tu_tech_node_energy_t *t = tu_power_get_tech_node((tu_tech_node_t)i);
        ASSERT_TRUE(t != NULL, "tech node not found");
        ASSERT_TRUE(strlen(t->name) > 0, "tech node name empty");
        ASSERT_TRUE(t->pj_per_fp16_mac > 0.0, "fp16 mac energy zero");
        ASSERT_TRUE(t->memory.regfile.pj_per_read > 0.0, "regfile read energy zero");
        ASSERT_TRUE(t->memory.spad.pj_per_write > 0.0, "spad write energy zero");
        ASSERT_TRUE(t->memory.global_buf.pj_per_read > t->memory.spad.pj_per_read,
                    "global buf should be more expensive than spad");
        ASSERT_TRUE(t->pj_per_dram_read > t->pj_per_dma_byte,
                    "DRAM should be more expensive than DMA");
    }
    PASS();
}

/* ---- Test 2: String to tech node conversion ---- */
static void test_tech_node_from_string(void) {
    TEST("tech_node_from_string");
    ASSERT_EQ(tu_power_tech_node_from_string("45nm"), TU_TECH_NODE_45NM, "45nm");
    ASSERT_EQ(tu_power_tech_node_from_string("28nm"), TU_TECH_NODE_28NM, "28nm");
    ASSERT_EQ(tu_power_tech_node_from_string("16nm"), TU_TECH_NODE_16NM, "16nm");
    ASSERT_EQ(tu_power_tech_node_from_string("7nm"),  TU_TECH_NODE_7NM,  "7nm");
    ASSERT_EQ(tu_power_tech_node_from_string("5nm"),  TU_TECH_NODE_5NM,  "5nm");
    ASSERT_EQ(tu_power_tech_node_from_string("3nm"),  TU_TECH_NODE_3NM,  "3nm");
    ASSERT_EQ(tu_power_tech_node_from_string("7"),    TU_TECH_NODE_7NM,  "7");
    ASSERT_EQ(tu_power_tech_node_from_string(NULL),   TU_TECH_NODE_7NM,  "NULL→default");
    ASSERT_EQ(tu_power_tech_node_from_string("garbage"), TU_TECH_NODE_7NM, "garbage→default");
    PASS();
}

/* ---- Test 3: Power model initialization ---- */
static void test_power_model_init(void) {
    TEST("power_model_init");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_5NM, 2000.0);

    ASSERT_EQ(pm.tech_node, TU_TECH_NODE_5NM, "tech node");
    ASSERT_TRUE(pm.enabled, "enabled");
    ASSERT_DOUBLE_NEAR(pm.clock_freq_mhz, 2000.0, 0.01, "clock freq");
    ASSERT_DOUBLE_NEAR(pm.params.pj_per_fp16_mac, 0.14, 0.01, "fp16 mac energy");
    ASSERT_EQ(pm.total_macs, 0, "initial macs");
    ASSERT_EQ(pm.total_cycles, 0, "initial cycles");
    ASSERT_DOUBLE_NEAR(pm.energy_mac_pj, 0.0, 0.001, "initial mac energy");
    PASS();
}

/* ---- Test 4: All tech nodes have consistent energy params ---- */
static void test_power_model_tech_nodes(void) {
    TEST("power_model_tech_nodes");
    for (int i = 0; i < TU_TECH_NODE_COUNT; i++) {
        tu_power_model_t pm;
        tu_power_model_init(&pm, (tu_tech_node_t)i, 1000.0);

        /* Energy values must be positive */
        ASSERT_TRUE(pm.params.pj_per_fp16_mac > 0.0, "fp16 mac <= 0");
        ASSERT_TRUE(pm.params.memory.regfile.pj_per_read > 0.0, "regfile read <= 0");
        ASSERT_TRUE(pm.params.memory.spad.pj_per_read > 0.0, "spad read <= 0");
        ASSERT_TRUE(pm.params.memory.global_buf.pj_per_read > 0.0, "gbuf read <= 0");
        ASSERT_TRUE(pm.params.pj_per_dram_read > 0.0, "dram read <= 0");
        ASSERT_TRUE(pm.params.pj_per_dma_byte > 0.0, "dma byte <= 0");
        ASSERT_TRUE(pm.params.pj_per_clock_tree > 0.0, "clock tree <= 0");
        ASSERT_TRUE(pm.params.static_power_mw_per_mm2 > 0.0, "leakage density <= 0");

        /* Memory hierarchy: RegFile < SPAD < GlobalBuffer */
        ASSERT_TRUE(pm.params.memory.regfile.pj_per_read < pm.params.memory.spad.pj_per_read,
                    "regfile not cheaper than spad");
        ASSERT_TRUE(pm.params.memory.spad.pj_per_read < pm.params.memory.global_buf.pj_per_read,
                    "spad not cheaper than global buf");

        /* FP8 < FP16, INT8 < FP16 */
        ASSERT_TRUE(pm.params.pj_per_fp8_mac < pm.params.pj_per_fp16_mac,
                    "fp8 not cheaper than fp16");
        ASSERT_TRUE(pm.params.pj_per_int8_mac < pm.params.pj_per_fp16_mac,
                    "int8 not cheaper than fp16");
        ASSERT_TRUE(pm.params.pj_per_int4_mac < pm.params.pj_per_int8_mac,
                    "int4 not cheaper than int8");

        /* Area values */
        ASSERT_TRUE(pm.params.area_um2_per_mac > 0.0, "mac area <= 0");
        ASSERT_TRUE(pm.params.area_um2_per_kb_sram > 0.0, "sram area <= 0");
        ASSERT_TRUE(pm.params.nominal_voltage_v > 0.0, "voltage <= 0");
        ASSERT_TRUE(pm.params.frequency_ghz > 0.0, "freq <= 0");
    }
    PASS();
}

/* ---- Test 5: MAC energy recording ---- */
static void test_power_mac_recording(void) {
    TEST("power_mac_recording");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    /* Record 1000 FP16 MACs */
    tu_power_record_mac(&pm, 1000, 0);
    ASSERT_EQ(pm.total_macs, 1000, "total macs");
    ASSERT_DOUBLE_NEAR(pm.energy_mac_pj, 0.20 * 1000.0, 0.5, "fp16 mac energy");

    /* Record 500 INT8 MACs */
    tu_power_record_mac(&pm, 500, 2);
    ASSERT_EQ(pm.total_macs, 1500, "total macs after int8");
    ASSERT_DOUBLE_NEAR(pm.energy_mac_pj, 0.20 * 1000.0 + 0.04 * 500.0, 0.5,
                       "combined mac energy");
    PASS();
}

/* ---- Test 6: Memory hierarchy energy recording ---- */
static void test_power_memory_recording(void) {
    TEST("power_memory_recording");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    /* 1000 regfile reads, 500 writes */
    tu_power_record_regfile_access(&pm, false, 1000);
    tu_power_record_regfile_access(&pm, true, 500);
    ASSERT_EQ(pm.regfile_reads, 1000, "regfile reads");
    ASSERT_EQ(pm.regfile_writes, 500, "regfile writes");
    ASSERT_DOUBLE_NEAR(pm.energy_regfile_read_pj, 0.004 * 1000.0, 0.1, "regfile read energy");
    ASSERT_DOUBLE_NEAR(pm.energy_regfile_write_pj, 0.004 * 500.0, 0.1, "regfile write energy");

    /* 200 spad accesses */
    tu_power_record_spad_access(&pm, false, 100);
    tu_power_record_spad_access(&pm, true, 100);
    ASSERT_DOUBLE_NEAR(pm.energy_spad_read_pj, 0.10 * 100.0, 0.5, "spad read energy");
    ASSERT_DOUBLE_NEAR(pm.energy_spad_write_pj, 0.10 * 100.0, 0.5, "spad write energy");

    /* 50 global buffer accesses */
    tu_power_record_global_buf_access(&pm, false, 30);
    tu_power_record_global_buf_access(&pm, true, 20);
    ASSERT_DOUBLE_NEAR(pm.energy_global_buf_read_pj, 0.24 * 30.0, 0.5, "gbuf read energy");
    PASS();
}

/* ---- Test 7: DRAM energy with page hit/miss ---- */
static void test_power_dram_recording(void) {
    TEST("power_dram_recording");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    /* 64-byte read, page hit */
    tu_power_record_dram_access(&pm, false, 64, true);
    ASSERT_EQ(pm.dram_reads, 1, "dram reads");
    ASSERT_EQ(pm.dram_activates, 0, "dram activates (page hit)");
    ASSERT_DOUBLE_NEAR(pm.energy_dram_read_pj, 130.0, 1.0, "dram read energy (hit)");

    /* 128-byte read, page miss (2 transactions) */
    tu_power_record_dram_access(&pm, false, 128, false);
    ASSERT_EQ(pm.dram_reads, 3, "dram reads after miss");
    ASSERT_EQ(pm.dram_activates, 1, "dram activates");
    ASSERT_DOUBLE_NEAR(pm.energy_dram_activate_pj, 240.0, 1.0, "dram activate energy");

    /* Write with page hit */
    tu_power_record_dram_access(&pm, true, 64, true);
    ASSERT_EQ(pm.dram_writes, 1, "dram writes");
    ASSERT_DOUBLE_NEAR(pm.energy_dram_write_pj, 120.0, 1.0, "dram write energy");
    PASS();
}

/* ---- Test 8: DMA bus energy ---- */
static void test_power_dma_recording(void) {
    TEST("power_dma_recording");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    tu_power_record_dma(&pm, 1024);
    ASSERT_EQ(pm.dma_bytes, 1024, "dma bytes");
    ASSERT_DOUBLE_NEAR(pm.energy_dma_pj, 0.010 * 1024.0, 1.0, "dma energy");
    PASS();
}

/* ---- Test 9: Clock ticking ---- */
static void test_power_tick(void) {
    TEST("power_tick");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    /* Must have area estimate for leakage */
    tu_power_estimate_area(&pm, 32, 32, 65536, 1048576);

    tu_power_tick(&pm, 1000);
    ASSERT_EQ(pm.total_cycles, 1000, "total cycles");
    ASSERT_DOUBLE_NEAR(pm.energy_clock_pj, 0.010 * 1000.0, 0.5, "clock energy");
    ASSERT_TRUE(pm.energy_leakage_pj > 0.0, "leakage energy must be > 0");
    PASS();
}

/* ---- Test 10: Total energy computation ---- */
static void test_power_total(void) {
    TEST("power_total");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    tu_power_record_mac(&pm, 1000, 0);
    tu_power_record_spad_access(&pm, false, 500);
    tu_power_record_dma(&pm, 256);

    tu_power_compute_total(&pm);

    double expected_total = pm.energy_mac_pj + pm.energy_spad_read_pj + pm.energy_dma_pj;
    double diff = fabs(pm.energy_total_pj - expected_total);
    ASSERT_TRUE(diff < 1.0, "total energy mismatch");
    PASS();
}

/* ---- Test 11: Average power calculation ---- */
static void test_power_avg_power(void) {
    TEST("power_avg_power");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    tu_power_estimate_area(&pm, 32, 32, 65536, 1048576);
    tu_power_record_mac(&pm, 10000, 0);
    tu_power_tick(&pm, 10000);

    tu_power_compute_total(&pm);
    double avg_power = tu_power_get_avg_power_mw(&pm);
    ASSERT_TRUE(avg_power > 0.0, "avg power <= 0");
    ASSERT_TRUE(avg_power < 1000.0, "avg power unreasonably high");
    PASS();
}

/* ---- Test 12: Energy breakdown fractions sum to ~1.0 ---- */
static void test_power_breakdown(void) {
    TEST("power_breakdown");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    tu_power_estimate_area(&pm, 32, 32, 65536, 1048576);
    tu_power_record_mac(&pm, 1000, 0);
    tu_power_record_spad_access(&pm, false, 500);
    tu_power_record_dram_access(&pm, false, 128, false);
    tu_power_tick(&pm, 1000);

    tu_power_breakdown_t bd = tu_power_get_breakdown(&pm);
    double sum = bd.fraction_mac + bd.fraction_regfile + bd.fraction_spad
               + bd.fraction_global_buf + bd.fraction_dram + bd.fraction_dma
               + bd.fraction_clock + bd.fraction_leakage;

    ASSERT_DOUBLE_NEAR(sum, 1.0, 0.01, "breakdown fractions don't sum to 1");
    ASSERT_TRUE(bd.fraction_mac > 0.0, "MAC fraction is 0");
    PASS();
}

/* ---- Test 13: Chip area estimation ---- */
static void test_power_area_estimate(void) {
    TEST("power_area_estimate");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    /* 32×32 PE array, 64 KB SPAD, 1 MB GBUF */
    double area = tu_power_estimate_area(&pm, 32, 32, 65536, 1048576);

    /* Area should be in a reasonable range */
    ASSERT_TRUE(area > 0.1, "area too small");
    ASSERT_TRUE(area < 100.0, "area too large");

    /* Check stored area estimate */
    ASSERT_TRUE(pm.estimated_area_mm2 > 0.0, "area not stored");
    ASSERT_DOUBLE_NEAR(pm.estimated_area_mm2, area, 0.001, "stored area mismatch");
    PASS();
}

/* ---- Test 14: Snapshot and diff ---- */
static void test_power_snapshot_diff(void) {
    TEST("power_snapshot_diff");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    tu_power_snapshot_t snap1 = tu_power_snapshot(&pm);

    tu_power_record_mac(&pm, 500, 0);
    tu_power_record_spad_access(&pm, false, 200);
    tu_power_tick(&pm, 500);

    tu_power_snapshot_t snap2 = tu_power_snapshot(&pm);

    tu_power_model_t diff = tu_power_diff(&snap1, &snap2);

    ASSERT_EQ(diff.total_macs, 500, "diff macs");
    ASSERT_EQ(diff.spad_reads, 200, "diff spad reads");
    ASSERT_EQ(diff.total_cycles, 500, "diff cycles");
    ASSERT_TRUE(diff.energy_mac_pj > 0.0, "diff mac energy zero");
    ASSERT_TRUE(diff.energy_spad_read_pj > 0.0, "diff spad energy zero");

    /* Diff should preserve config */
    ASSERT_EQ(diff.tech_node, TU_TECH_NODE_7NM, "diff tech node");
    ASSERT_DOUBLE_NEAR(diff.clock_freq_mhz, 1000.0, 0.01, "diff clock");

    /* Zero diff */
    tu_power_model_t zero_diff = tu_power_diff(&snap1, &snap1);
    ASSERT_EQ(zero_diff.total_macs, 0, "zero diff macs");
    ASSERT_DOUBLE_NEAR(zero_diff.energy_mac_pj, 0.0, 0.001, "zero diff energy");
    PASS();
}

/* ---- Test 15: Reset preserves config ---- */
static void test_power_reset(void) {
    TEST("power_reset");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_5NM, 1500.0);

    tu_power_record_mac(&pm, 100, 0);
    tu_power_tick(&pm, 100);

    ASSERT_EQ(pm.total_macs, 100, "pre-reset macs");

    tu_power_model_reset(&pm);

    ASSERT_EQ(pm.tech_node, TU_TECH_NODE_5NM, "tech node after reset");
    ASSERT_DOUBLE_NEAR(pm.clock_freq_mhz, 1500.0, 0.01, "freq after reset");
    ASSERT_EQ(pm.total_macs, 0, "macs after reset");
    ASSERT_EQ(pm.total_cycles, 0, "cycles after reset");
    ASSERT_DOUBLE_NEAR(pm.energy_mac_pj, 0.0, 0.001, "energy after reset");
    PASS();
}

/* ---- Test 16: Mid-run technology node switch ---- */
static void test_power_tech_switch(void) {
    TEST("power_tech_switch");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_45NM, 1000.0);

    /* Record some MACs at 45nm */
    tu_power_record_mac(&pm, 100, 0);
    double energy_45nm = pm.energy_mac_pj;
    ASSERT_DOUBLE_NEAR(energy_45nm, 1.0 * 100.0, 1.0, "45nm energy");

    /* Switch to 3nm — existing counters preserved, new rates apply */
    tu_power_model_set_tech_node(&pm, TU_TECH_NODE_3NM);
    ASSERT_EQ(pm.tech_node, TU_TECH_NODE_3NM, "tech node switched");

    tu_power_record_mac(&pm, 100, 0);
    double energy_3nm = pm.energy_mac_pj - energy_45nm;
    ASSERT_DOUBLE_NEAR(energy_3nm, 0.10 * 100.0, 1.0, "3nm energy after switch");
    PASS();
}

/* ---- Test 17: Disabled model records nothing ---- */
static void test_power_disable(void) {
    TEST("power_disable");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    tu_power_estimate_area(&pm, 32, 32, 65536, 1048576);
    tu_power_model_set_enabled(&pm, false);

    tu_power_record_mac(&pm, 1000, 0);
    tu_power_record_spad_access(&pm, false, 500);
    tu_power_record_dram_access(&pm, false, 128, true);
    tu_power_record_dma(&pm, 256);
    tu_power_tick(&pm, 1000);

    ASSERT_EQ(pm.total_macs, 0, "macs when disabled");
    ASSERT_EQ(pm.spad_reads, 0, "spad reads when disabled");
    ASSERT_EQ(pm.total_cycles, 0, "cycles when disabled");
    ASSERT_DOUBLE_NEAR(pm.energy_mac_pj, 0.0, 0.001, "mac energy when disabled");
    PASS();
}

/* ---- Test 18: Energy scales with technology node ---- */
static void test_power_energy_scaling(void) {
    TEST("power_energy_scaling");
    tu_power_model_t pm45, pm7, pm3;

    tu_power_model_init(&pm45, TU_TECH_NODE_45NM, 1000.0);
    tu_power_model_init(&pm7,  TU_TECH_NODE_7NM,  1000.0);
    tu_power_model_init(&pm3,  TU_TECH_NODE_3NM,  1000.0);

    tu_power_record_mac(&pm45, 1000, 0);
    tu_power_record_mac(&pm7,  1000, 0);
    tu_power_record_mac(&pm3,  1000, 0);

    ASSERT_TRUE(pm7.energy_mac_pj < pm45.energy_mac_pj,
                "7nm not more efficient than 45nm");
    ASSERT_TRUE(pm3.energy_mac_pj < pm7.energy_mac_pj,
                "3nm not more efficient than 7nm");

    /* 7nm should be roughly 5× more efficient than 45nm */
    double ratio_45_7 = pm45.energy_mac_pj / pm7.energy_mac_pj;
    ASSERT_TRUE(ratio_45_7 > 3.0, "7nm efficiency ratio too low");
    PASS();
}

/* ---- Test 19: Config integration ---- */
static void test_power_config_integration(void) {
    TEST("power_config_integration");
    tu_power_model_t pm;

    /* Small config */
    tu_config_t small_cfg;
    memset(&small_cfg, 0, sizeof(small_cfg));
    small_cfg.pe_rows = 8;
    small_cfg.pe_cols = 8;
    small_cfg.sram_w_size_kb = 32;
    small_cfg.gbuf_size_kb = 256;
    small_cfg.dram_bandwidth_gbps = 100.0;
    small_cfg.counters_enabled = true;

    tu_power_model_from_config(&pm, &small_cfg);
    ASSERT_EQ(pm.tech_node, TU_TECH_NODE_7NM, "small→7nm");
    ASSERT_TRUE(pm.enabled, "small enabled");

    /* Large config */
    tu_config_t large_cfg;
    memset(&large_cfg, 0, sizeof(large_cfg));
    large_cfg.pe_rows = 256;
    large_cfg.pe_cols = 256;
    large_cfg.sram_w_size_kb = 256;
    large_cfg.gbuf_size_kb = 4096;
    large_cfg.dram_bandwidth_gbps = 900.0;
    large_cfg.counters_enabled = true;

    tu_power_model_from_config(&pm, &large_cfg);
    ASSERT_EQ(pm.tech_node, TU_TECH_NODE_5NM, "large→5nm");
    ASSERT_DOUBLE_NEAR(pm.clock_freq_mhz, 2000.0, 1.0, "large freq");
    ASSERT_TRUE(pm.estimated_area_mm2 > 10.0, "large area estimate too small");
    PASS();
}

/* ---- Test 20: Numeric stability ---- */
static void test_power_numeric_stability(void) {
    TEST("power_numeric_stability");
    tu_power_model_t pm;
    tu_power_model_init(&pm, TU_TECH_NODE_7NM, 1000.0);

    /* Zero operations should not crash */
    tu_power_compute_total(&pm);
    ASSERT_DOUBLE_NEAR(pm.energy_total_pj, 0.0, 0.001, "zero total");
    ASSERT_DOUBLE_NEAR(tu_power_get_avg_power_mw(&pm), 0.0, 0.001, "zero power");

    tu_power_breakdown_t bd = tu_power_get_breakdown(&pm);
    /* All fractions should be 0 / 0 = 0 */
    ASSERT_DOUBLE_NEAR(bd.fraction_mac, 0.0, 0.001, "zero breakdown mac");

    /* Very large values */
    tu_power_record_mac(&pm, UINT64_MAX / 2, 0);
    /* Should not crash — energy may overflow double precision but shouldn't crash */
    tu_power_compute_total(&pm);
    PASS();
}

/* ---- Main ---- */

int main(void) {
    printf("\n=== TU Power/Energy Model Tests ===\n\n");

    test_tech_node_lookup();
    test_tech_node_from_string();
    test_power_model_init();
    test_power_model_tech_nodes();
    test_power_mac_recording();
    test_power_memory_recording();
    test_power_dram_recording();
    test_power_dma_recording();
    test_power_tick();
    test_power_total();
    test_power_avg_power();
    test_power_breakdown();
    test_power_area_estimate();
    test_power_snapshot_diff();
    test_power_reset();
    test_power_tech_switch();
    test_power_disable();
    test_power_energy_scaling();
    test_power_config_integration();
    test_power_numeric_stability();

    printf("\nResults: %d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
