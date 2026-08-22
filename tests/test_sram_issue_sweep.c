/* Exploration: executable SRAM per-bank grant and refill cadence. */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/infra/config.h"

static int run_row(const char *name, unsigned grants, unsigned window,
                   unsigned accesses, int advance_each,
                   uint64_t expected_stall) {
    char json[512];
    snprintf(json, sizeof(json),
             "{\"tu\":{\"memory\":{\"banking\":{"
             "\"banks\":8,\"bank_width_bytes\":4,"
             "\"words_per_refill\":%u,\"refill_window_cycles\":%u,"
             "\"stall_penalty_cycles\":2}}}}", grants, window);
    tu_config_t cfg;
    char err[160] = {0};
    if (tu_config_load_string(json, &cfg, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL %s parse: %s\n", name, err);
        return 1;
    }
    if (tu_init_from_config(&cfg) != 0 ||
        g_tu.sram_a.banks.bank_count != 8 ||
        g_tu.sram_a.banks.bank_width != 4 ||
        g_tu.sram_a.banks.words_per_cycle != grants ||
        g_tu.sram_a.banks.bw_refill_window != window) {
        fprintf(stderr, "FAIL %s propagation\n", name);
        return 1;
    }

    uint64_t stalls = 0;
    uint64_t word = 0;
    for (unsigned i = 0; i < accesses; ++i) {
        stalls += tu_sram_read(&g_tu.sram_a, 0, &word);
        if (advance_each && i + 1 < accesses)
            tu_sram_advance_cycle(&g_tu.sram_a, 1);
    }
    printf("%-16s %6u %6u %8u %12" PRIu64 "\n",
           name, grants, window, accesses, stalls);
    if (stalls != expected_stall) {
        fprintf(stderr, "FAIL %s stalls=%" PRIu64 " expected=%" PRIu64 "\n",
                name, stalls, expected_stall);
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    puts("SRAM grant/refill sweep: one bank, burst/spaced accesses, penalty=2");
    printf("%-16s %6s %6s %8s %12s\n",
           "config", "grant", "window", "accesses", "stall_cycles");
    failures += run_row("legacy-burst", 1, 4, 4, 0, 6);
    failures += run_row("legacy-spaced", 1, 4, 4, 1, 6);
    failures += run_row("single-port", 1, 1, 4, 1, 0);
    failures += run_row("dual-port", 2, 1, 4, 0, 4);
    failures += run_row("quad-port", 4, 1, 4, 0, 0);

    tu_runtime_config_t legacy_rt = {0};
    legacy_rt.pe_rows = 4;
    legacy_rt.pe_cols = 4;
    legacy_rt.pe_pipeline_depth = 1;
    legacy_rt.sram_w_size = 4096;
    legacy_rt.sram_a_size = 4096;
    legacy_rt.sram_o_size = 4096;
    tu_init_with_config(&legacy_rt);
    if (g_tu.sram_a.banks.bank_count != TU_SRAM_BANKS ||
        g_tu.sram_a.banks.bank_width != TU_SRAM_BANK_WIDTH ||
        g_tu.sram_a.banks.words_per_cycle != TU_SRAM_WORDS_PER_CYCLE ||
        g_tu.sram_a.banks.bw_refill_window != TU_SRAM_BW_WINDOW_CYCLES) {
        fprintf(stderr, "FAIL zero-initialized runtime compatibility\n");
        failures++;
    }

    tu_config_t bad;
    char err[160] = {0};
    if (tu_config_load_string(
            "{\"tu\":{\"memory\":{\"banking\":{\"words_per_refill\":0}}}}",
            &bad, err, sizeof(err)) == 0 ||
        strstr(err, "words_per_refill") == NULL) {
        fprintf(stderr, "FAIL invalid grant did not fail closed: %s\n", err);
        failures++;
    }
    if (tu_config_load_string(
            "{\"tu\":{\"memory\":{\"banking\":{\"refill_window_cycles\":0}}}}",
            &bad, err, sizeof(err)) == 0 ||
        strstr(err, "refill_window_cycles") == NULL) {
        fprintf(stderr, "FAIL invalid window did not fail closed: %s\n", err);
        failures++;
    }

    if (failures) return 1;
    puts("PASS: 5 configurations plus parse rejection gates");
    return 0;
}
