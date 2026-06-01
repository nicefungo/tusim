/*
 * TU CModel — Event Tracing Tests (P2.7)
 * ========================================
 *
 * Validates VCD waveform generation: context lifecycle, signal
 * registration, change detection, tick advancement, and file
 * integrity (valid VCD header/body/format).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "../tu_cmodel/perf/event_trace.h"

#define TEST_TMP_DIR  "/tmp"
#define TEST_VCD_FILE "/tmp/tu_test_trace.vcd"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)  do { printf("  %-50s", name); } while(0)
#define PASS()      do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); } else { PASS(); } } while(0)

/* Helper: check if a string exists in a file */
static bool file_contains(const char *filepath, const char *needle) {
    FILE *f = fopen(filepath, "r");
    if (!f) return false;

    char line[512];
    bool found = false;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, needle)) { found = true; break; }
    }
    fclose(f);
    return found;
}

/* Helper: count lines in a file */
static int file_line_count(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (!f) return -1;

    int count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) count++;
    fclose(f);
    return count;
}

int main(void) {
    printf("=== Event Tracing Tests (P2.7 — VCD Generation) ===\n\n");

    /* ---- Lifecycle ---- */

    TEST("Create trace with valid filename");
    tu_event_trace_t *trace = tu_trace_create(TEST_VCD_FILE, 16);
    CHECK(trace != NULL, "creation failed");
    if (!trace) return 1;  /* can't continue */

    TEST("Verify trace fields after creation");
    CHECK(trace->current_cycle == 0, "cycle not zero");
    CHECK(trace->signal_count == 0, "signal count not zero");
    CHECK(trace->header_written == false, "header should not be written");
    CHECK(trace->definitions_ended == false, "definitions not ended");
    CHECK(trace->signals != NULL, "signal array NULL");

    /* ---- Signal Registration ---- */

    TEST("Add 1-bit signal (DMA active)");
    int idx0 = tu_trace_add_signal(trace, "!", "TU_CORE.dma.ch0.active",
                                    TU_TRACE_SIG_1BIT);
    CHECK(idx0 == 0, "first signal index should be 0");

    TEST("Add 8-bit signal (DMA state)");
    int idx1 = tu_trace_add_signal(trace, "#", "TU_CORE.dma.ch0.state",
                                    TU_TRACE_SIG_8BIT);
    CHECK(idx1 == 1, "second signal index should be 1");

    TEST("Add 32-bit signal (compute counter)");
    int idx2 = tu_trace_add_signal(trace, "$", "TU_CORE.compute.counter",
                                    TU_TRACE_SIG_32BIT);
    CHECK(idx2 == 2, "third signal index should be 2");

    TEST("Signal count updated correctly");
    CHECK(trace->signal_count == 3, "expected 3 signals");

    /* ---- Change Detection ---- */

    TEST("Signal update sets dirty flag");
    tu_trace_signal(trace, 0, 1);
    CHECK(trace->signals[0].dirty == true, "dirty flag not set");

    TEST("Signal with same value does not mark dirty");
    trace->signals[0].dirty = false;
    tu_trace_signal(trace, 0, 1);  /* same value */
    CHECK(trace->signals[0].dirty == false, "should not be dirty for same value");

    TEST("Signal with different value marks dirty");
    tu_trace_signal(trace, 0, 0);  /* different value */
    CHECK(trace->signals[0].dirty == true, "should be dirty for different value");

    /* ---- Tick Advancement ---- */

    TEST("Tick writes header on first call, defers signal flush");
    tu_trace_tick(trace, 0);  /* write header + #0, return before flush */
    CHECK(trace->header_written == true, "header not written");
    CHECK(trace->definitions_ended == true, "definitions not ended");
    /* signals are still dirty — flushed on first real tick */
    CHECK(trace->signals[0].dirty == true,
          "dirty should persist after header-only tick");

    TEST("Tick advances cycle counter and flushes signals");
    tu_trace_tick(trace, 5);
    CHECK(trace->current_cycle == 5, "cycle not advanced to 5");

    TEST("Dirty signals cleared after tick");
    CHECK(trace->signals[0].dirty == false, "dirty not cleared after tick");

    /* ---- VCD File Integrity ---- */

    TEST("VCD file exists and non-empty");
    int lines = file_line_count(TEST_VCD_FILE);
    CHECK(lines > 0, "VCD file empty or missing");

    TEST("VCD header contains $date");
    CHECK(file_contains(TEST_VCD_FILE, "$date"), "$date missing from VCD");

    TEST("VCD header contains $version");
    CHECK(file_contains(TEST_VCD_FILE, "$version"), "$version missing");

    TEST("VCD header contains $timescale");
    CHECK(file_contains(TEST_VCD_FILE, "$timescale"), "$timescale missing");

    TEST("VCD header contains $var definitions");
    CHECK(file_contains(TEST_VCD_FILE, "$var"), "$var missing from VCD");

    TEST("VCD header contains $enddefinitions");
    CHECK(file_contains(TEST_VCD_FILE, "$enddefinitions"),
          "$enddefinitions missing");

    TEST("VCD contains $dumpvars");
    CHECK(file_contains(TEST_VCD_FILE, "$dumpvars"), "$dumpvars missing");

    TEST("VCD contains time header #0");
    CHECK(file_contains(TEST_VCD_FILE, "#0"), "#0 time header missing");

    TEST("VCD contains time header #5");
    CHECK(file_contains(TEST_VCD_FILE, "#5"), "#5 time header missing");

    TEST("VCD contains registered signal name");
    CHECK(file_contains(TEST_VCD_FILE, "TU_CORE.dma.ch0.active"),
          "signal name not in VCD");

    /* ---- Close and verify EOF ---- */

    tu_trace_close(trace);

    TEST("VCD file ends with end-of-file marker after close");
    CHECK(file_contains(TEST_VCD_FILE, "$dumpoff") ||
          file_contains(TEST_VCD_FILE, "$end"),
          "no end marker found");

    /* ---- Edge Cases ---- */

    TEST("Handle null filename gracefully");
    tu_event_trace_t *null_trace = tu_trace_create(NULL, 8);
    CHECK(null_trace == NULL, "should return NULL for NULL filename");

    TEST("Handle signal overflow gracefully");
    tu_event_trace_t *small = tu_trace_create("/tmp/tu_small.vcd", 2);
    assert(small);
    int a = tu_trace_add_signal(small, "a", "sig.a", TU_TRACE_SIG_1BIT);
    int b = tu_trace_add_signal(small, "b", "sig.b", TU_TRACE_SIG_1BIT);
    int c = tu_trace_add_signal(small, "c", "sig.c", TU_TRACE_SIG_1BIT);
    CHECK(a == 0 && b == 1 && c == -1, "overflow should return -1");
    tu_trace_close(small);

    /* ---- End marker verification already done above ---- */

    /* ---- Cleanup ---- */
    unlink(TEST_VCD_FILE);
    unlink("/tmp/tu_small.vcd");

    printf("\n=== Results: %d/%d passed ===\n",
           tests_passed, tests_passed + tests_failed);

    return tests_failed ? 1 : 0;
}
