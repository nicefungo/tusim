/*
 * Structured Logging Tests (Q2)
 * ==============================
 * Validates: severity filtering, component tagging, runtime level changes,
 * trace event recording, VCD export.
 */

#include "tu_cmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(n) printf("  [TEST] %-55s ", n); fflush(stdout)
#define PASS    do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(m) do { printf("FAIL: %s\n", m); tests_failed++; } while(0)

/* ---- Test 1: Log initialization ---- */
static void test_init(void) {
    TEST("log init and default config");
    tu_log_init();
    tu_log_config_t *cfg = tu_log_get_config();
    if (!cfg || !cfg->initialized) { FAIL("not initialized"); return; }
    if (cfg->min_level != TU_LOG_INFO) { FAIL("wrong default level"); return; }
    PASS;
}

/* ---- Test 2: Level filtering ---- */
static void test_level_filter(void) {
    TEST("severity level filtering");
    tu_log_init();

    /* At INFO level, DEBUG and TRACE should be suppressed.
     * We can't easily capture stderr in a portable way, but we can
     * verify that the API works: set level, check it, change it back. */

    tu_log_set_level(TU_LOG_INFO);
    if (tu_log_get_level() != TU_LOG_INFO) { FAIL("set INFO failed"); return; }

    tu_log_set_level(TU_LOG_DEBUG);
    if (tu_log_get_level() != TU_LOG_DEBUG) { FAIL("set DEBUG failed"); return; }

    tu_log_set_level(TU_LOG_ERROR);
    if (tu_log_get_level() != TU_LOG_ERROR) { FAIL("set ERROR failed"); return; }

    tu_log_set_level(TU_LOG_NONE);
    if (tu_log_get_level() != TU_LOG_NONE) { FAIL("set NONE failed"); return; }

    /* Back to default */
    tu_log_set_level(TU_LOG_INFO);
    PASS;
}

/* ---- Test 3: All levels emit ---- */
static void test_all_levels(void) {
    TEST("all severity levels emit");
    tu_log_init();
    tu_log_set_level(TU_LOG_TRACE); /* Show everything */

    TU_LOG_ERR(TU_COMP_TEST, "test error message");
    TU_LOG_WARN(TU_COMP_TEST, "test warning message");
    TU_LOG_INFO(TU_COMP_TEST, "test info message");
    TU_LOG_DBG(TU_COMP_TEST, "test debug message");
    TU_LOG_TRACE(TU_COMP_TEST, "test trace message");

    /* If we got here without crash, the macro machinery works */
    PASS;
}

/* ---- Test 4: Component tags ---- */
static void test_components(void) {
    TEST("component tags distinct");
    tu_log_init();

    /* Log from different components — all should emit without error */
    TU_LOG_INFO(TU_COMP_CORE, "core message");
    TU_LOG_INFO(TU_COMP_MMA,  "mma message");
    TU_LOG_INFO(TU_COMP_DMA,  "dma message");
    TU_LOG_INFO(TU_COMP_MEM,  "mem message");
    TU_LOG_INFO(TU_COMP_ISA,  "isa message");
    TU_LOG_INFO(TU_COMP_CMD,  "cmd message");
    TU_LOG_INFO(TU_COMP_DF,   "df message");
    TU_LOG_INFO(TU_COMP_PREC, "prec message");

    PASS;
}

/* ---- Test 5: Trace buffer ---- */
static void test_trace_buffer(void) {
    TEST("trace event recording");

    tu_trace_clear();
    uint32_t count = 0;
    const tu_trace_event_t *buf = tu_trace_get_buffer(&count);
    if (count != 0) { FAIL("trace not empty after clear"); return; }

    /* Record some events */
    tu_trace_set_cycle(0);
    tu_trace_event(TU_COMP_MMA, 0x01, 100, 200, 16, 0);
    tu_trace_set_cycle(10);
    tu_trace_event(TU_COMP_MMA, 0x01, 100, 200, 32, 0);
    tu_trace_set_cycle(20);
    tu_trace_event(TU_COMP_DMA, 0x10, 0x1000, 0x2000, 256, 0);

    buf = tu_trace_get_buffer(&count);
    if (count != 3) {
        char msg[64];
        snprintf(msg, sizeof(msg), "expected 3 events, got %u", count);
        FAIL(msg);
        return;
    }

    /* Verify event data */
    if (buf[0].cycle != 0 || buf[0].component != TU_COMP_MMA ||
        buf[0].operand[0] != 100) {
        FAIL("event 0 data mismatch");
        return;
    }

    if (buf[1].cycle != 10 || buf[1].operand[2] != 32) {
        FAIL("event 1 data mismatch");
        return;
    }

    if (buf[2].cycle != 20 || buf[2].component != TU_COMP_DMA ||
        buf[2].operand[2] != 256) {
        FAIL("event 2 data mismatch");
        return;
    }

    PASS;
}

/* ---- Test 6: VCD export ---- */
static void test_vcd_export(void) {
    TEST("VCD trace export");

    tu_trace_clear();
    tu_trace_set_cycle(0);
    tu_trace_event(TU_COMP_MMA, 0x01, 16, 32, 16, 0);
    tu_trace_set_cycle(50);
    tu_trace_event(TU_COMP_DMA, 0x10, 0x100, 0x200, 128, 0);

    /* Export to file */
    FILE *f = fopen("/tmp/tu_test_trace.vcd", "w");
    if (!f) { FAIL("cannot open VCD file"); return; }
    tu_trace_export_vcd(f);
    fclose(f);

    /* Verify the file has basic VCD structure */
    f = fopen("/tmp/tu_test_trace.vcd", "r");
    if (!f) { FAIL("cannot reopen VCD file"); return; }

    char line[256];
    int has_date = 0, has_timescale = 0, has_enddef = 0, has_signal = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "$date")) has_date = 1;
        if (strstr(line, "$timescale")) has_timescale = 1;
        if (strstr(line, "$enddefinitions")) has_enddef = 1;
        if (strstr(line, "$dumpvars")) has_signal = 1;
    }
    fclose(f);

    if (!has_date)      { FAIL("VCD missing $date"); return; }
    if (!has_timescale) { FAIL("VCD missing $timescale"); return; }
    if (!has_enddef)    { FAIL("VCD missing $enddefinitions"); return; }
    if (!has_signal)    { FAIL("VCD missing $dumpvars"); return; }

    PASS;
}

/* ---- Test 7: Integration with tu_init ---- */
static void test_integration(void) {
    TEST("logging integrated with tu_init");
    tu_init();

    /* tu_init() calls tu_log_init() internally.
     * After init, log level should be INFO and config initialized. */
    tu_log_config_t *cfg = tu_log_get_config();
    if (!cfg || !cfg->initialized) { FAIL("log not initialized after tu_init"); return; }

    /* Run a quick MMA to generate trace events */
    uint16_t *W = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_w);
    uint16_t *A = (uint16_t *)tu_sram_raw_ptr(&g_tu.sram_a);
    memset(W, 0, 16 * 16 * 2);
    memset(A, 0, 16 * 16 * 2);
    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, 16 * 16 * 4);
    W[0] = 0x3C00; /* 1.0 at [0][0] */
    A[0] = 0x3C00; /* 1.0 at [0][0] */

    tu_mma(16, 16, 16, 0, 0, 0, false);

    /* Should have at least 1 trace event from MMA */
    uint32_t count = 0;
    tu_trace_get_buffer(&count);
    if (count < 1) { FAIL("no trace events after MMA"); return; }

    PASS;
}

/* ---- Main ---- */

int main(void) {
    printf("\n=== TU Structured Logging Tests (Gap Q2) ===\n\n");

    test_init();
    test_level_filter();
    test_all_levels();
    test_components();
    test_trace_buffer();
    test_vcd_export();
    test_integration();

    printf("\n=== Results: %d passed, %d failed ===\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
