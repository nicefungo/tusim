/*
 * Exploration sweep: deferred-refresh debt and nominal-grid retention.
 *
 * A postponed refresh must not silently reset the retention interval after
 * every late command: repeated reset-after-service scheduling could extend
 * the average command cadence indefinitely.  The live model keeps the next
 * nominal command on the original tREFI grid.  This sweep delays the first
 * access within the allowed window, proves that the next nominal schedule is
 * unchanged, then idles until the second hard deadline.
 */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

#define TREFI 5000u
#define TRFC 100u

static tu_dram_model_t *make_dram(void) {
    tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 4, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    return tu_dram_create_custom(&p, "refresh-debt-sweep");
}

int main(void) {
    static const uint32_t max_defs[] = {500, 2000, 5000};
    int failures = 0;

    puts("Deferred-refresh debt sweep: ALL_BANK, tREFI=5000, tRFC=100");
    printf("%-7s %-7s %-10s %-10s %-10s %-10s %-8s\n",
           "maxdef", "delay", "first_fire", "next_grid", "deadline2",
           "interval2", "service");

    for (size_t m = 0; m < sizeof(max_defs) / sizeof(max_defs[0]); ++m) {
        uint32_t max_def = max_defs[m];
        uint32_t delays[] = {0, max_def / 2, max_def - 1};
        for (size_t d = 0; d < sizeof(delays) / sizeof(delays[0]); ++d) {
            tu_dram_model_t *dram = make_dram();
            if (!dram || !tu_dram_set_refresh(
                    dram, TU_DRAM_REFRESH_ALL_BANK,
                    TU_DRAM_REFRESH_SCHEDULING_DEFERRED,
                    1, TREFI, TRFC, 30, max_def)) {
                fprintf(stderr, "FAIL: fixture maxdef=%u delay=%u\n",
                        max_def, delays[d]);
                tu_dram_destroy(dram);
                failures++;
                continue;
            }

            uint64_t first_fire = TREFI + delays[d];
            while (dram->current_cycle < first_fire) tu_dram_tick(dram);
            uint64_t cycles = 0, stall = 0;
            tu_dram_read(dram, 0, 64, &cycles, &stall);

            uint64_t next_grid = dram->refresh_next[0];
            uint64_t deadline2 = next_grid + max_def;
            while (dram->current_cycle < deadline2) tu_dram_tick(dram);
            uint64_t interval2 = deadline2 - first_fire;

            printf("%-7u %-7u %-10" PRIu64 " %-10" PRIu64
                   " %-10" PRIu64 " %-10" PRIu64 " %-8" PRIu64 "\n",
                   max_def, delays[d], first_fire, next_grid, deadline2,
                   interval2, cycles);

            if (dram->stats.total_refresh_events != 2) {
                fprintf(stderr, "  GATE FAIL: expected 2 events, got %" PRIu64 "\n",
                        dram->stats.total_refresh_events);
                failures++;
            }
            if (next_grid != 2 * TREFI) {
                fprintf(stderr, "  GATE FAIL: next nominal schedule moved to %" PRIu64 "\n",
                        next_grid);
                failures++;
            }
            if (cycles != 50 + TRFC) {
                fprintf(stderr, "  GATE FAIL: first access service=%" PRIu64 "\n",
                        cycles);
                failures++;
            }
            tu_dram_destroy(dram);
        }
    }

    if (failures) {
        fprintf(stderr, "FAIL: %d gate violation(s)\n", failures);
        return 1;
    }
    puts("PASS: deferral preserves the nominal tREFI grid and retires debt by the next deadline");
    return 0;
}
