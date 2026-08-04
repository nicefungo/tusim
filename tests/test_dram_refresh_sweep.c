/*
 * Exploration sweep: DRAM refresh model (JEDEC tREFI/tRFC).
 *
 * Matrix: refresh mode (NONE / ALL_BANK / PER_BANK) × scheduling
 * (FIXED / DEFERRED) × rate (1x/2x/4x) × traffic pattern
 * (steady stream / burst-idle). Reports returned service cycles,
 * refresh events, refresh stall cycles, and (for open-page rows)
 * the row hit/miss interaction. Every row is a fail-closed gate on
 * accounting invariants; the trade-off table is the exploration
 * evidence, not a recommendation.
 *
 * Cycle domain: 1 sim cycle = 1 ns (1 GHz core-clock convention used
 * by the whole module). Refresh timings are JEDEC-like defaults:
 * tREFI=7800 ns, tRFC=350 ns, tRFCpb=90 ns.
 */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef enum { STEADY, BURST_IDLE } pattern_t;

typedef struct {
    const char *name;
    pattern_t   pattern;
} pattern_info_t;

static const pattern_info_t patterns[] = {
    {"steady",     STEADY},
    {"burst_idle", BURST_IDLE},
};

/* 4 channels / 4 banks / 256 B row custom DRAM (latency 50). */
static tu_dram_model_t *make_dram(void) {
    tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 4, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    return tu_dram_create_custom(&p, "refresh-sweep");
}

/* Run the traffic pattern inline in run_row; 40,000-cycle steady stream or
 * 8 × (50 reads + 5000 idle cycles) burst-idle stream. */

static int run_row(const char *mode_name, int mode, const char *sched_name,
                   int sched, uint32_t rate, const char *policy_name,
                   int policy, pattern_t pattern, uint64_t *out_service,
                   uint64_t *out_events, uint64_t *out_rstall,
                   uint64_t *out_hits, uint64_t *out_misses) {
    (void)mode_name; (void)sched_name; (void)policy_name;
    /* names are informational only; the enums drive behavior */
    tu_dram_model_t *dram = make_dram();
    if (!dram) return 1;
    if (policy == TU_DRAM_ROW_OPEN_PAGE &&
        !tu_dram_set_row_policy(dram, TU_DRAM_ROW_OPEN_PAGE, 20)) {
        tu_dram_destroy(dram);
        return 1;
    }
    if (!tu_dram_set_refresh(dram, (tu_dram_refresh_mode_t)mode,
                             (tu_dram_refresh_scheduling_t)sched,
                             rate, 7800, 350, 90, 7800)) {
        tu_dram_destroy(dram);
        return 1;
    }
    uint64_t service = 0;
    if (pattern == STEADY) {
        for (uint64_t i = 0; i < 40000; ++i) {
            uint64_t cycles = 0, stall = 0;
            tu_dram_read(dram, i * 64, 64, &cycles, &stall);
            service += cycles;
            tu_dram_tick(dram);
        }
    } else {
        for (int r = 0; r < 8; ++r) {
            for (int i = 0; i < 50; ++i) {
                uint64_t cycles = 0, stall = 0;
                tu_dram_read(dram, (uint64_t)(r * 50 + i) * 64, 64, &cycles, &stall);
                service += cycles;
                tu_dram_tick(dram);
            }
            for (int i = 0; i < 5000; ++i) tu_dram_tick(dram);
        }
    }
    *out_service = service;
    *out_events  = dram->stats.total_refresh_events;
    *out_rstall  = dram->stats.total_refresh_stall_cycles;
    *out_hits    = dram->stats.total_row_hits;
    *out_misses  = dram->stats.total_row_conflicts;
    tu_dram_destroy(dram);
    return 0;
}

int main(void) {
    int failures = 0;
    puts("DRAM refresh sweep: 4ch/4bank/256B-row custom DRAM, latency=50");
    puts("tREFI=7800, tRFC=350, tRFCpb=90 (ns == sim cycles @ 1 GHz)");
    printf("%-11s %-9s %-9s %-4s %-10s %-9s %9s %8s %8s %6s %6s\n",
           "mode", "sched", "pattern", "rate", "policy", "accesses",
           "service", "events", "rstall", "hits", "miss");

    const struct {
        const char *mode_name; int mode;
        const char *sched_name; int sched;
        uint32_t rate; const char *policy_name; int policy;
    } rows[] = {
        {"none", TU_DRAM_REFRESH_NONE, "fixed", TU_DRAM_REFRESH_SCHEDULING_FIXED, 1, "legacy", TU_DRAM_ROW_LEGACY},
        {"all",  TU_DRAM_REFRESH_ALL_BANK, "fixed", TU_DRAM_REFRESH_SCHEDULING_FIXED, 1, "legacy", TU_DRAM_ROW_LEGACY},
        {"all",  TU_DRAM_REFRESH_ALL_BANK, "deferred", TU_DRAM_REFRESH_SCHEDULING_DEFERRED, 1, "legacy", TU_DRAM_ROW_LEGACY},
        {"per",  TU_DRAM_REFRESH_PER_BANK, "fixed", TU_DRAM_REFRESH_SCHEDULING_FIXED, 1, "legacy", TU_DRAM_ROW_LEGACY},
        {"per",  TU_DRAM_REFRESH_PER_BANK, "deferred", TU_DRAM_REFRESH_SCHEDULING_DEFERRED, 1, "legacy", TU_DRAM_ROW_LEGACY},
        {"all",  TU_DRAM_REFRESH_ALL_BANK, "fixed", TU_DRAM_REFRESH_SCHEDULING_FIXED, 2, "legacy", TU_DRAM_ROW_LEGACY},
        {"all",  TU_DRAM_REFRESH_ALL_BANK, "fixed", TU_DRAM_REFRESH_SCHEDULING_FIXED, 4, "legacy", TU_DRAM_ROW_LEGACY},
        {"all",  TU_DRAM_REFRESH_ALL_BANK, "fixed", TU_DRAM_REFRESH_SCHEDULING_FIXED, 1, "open", TU_DRAM_ROW_OPEN_PAGE},
        {"per",  TU_DRAM_REFRESH_PER_BANK, "fixed", TU_DRAM_REFRESH_SCHEDULING_FIXED, 1, "open", TU_DRAM_ROW_OPEN_PAGE},
    };

    for (size_t pi = 0; pi < sizeof(patterns) / sizeof(patterns[0]); ++pi) {
        for (size_t ri = 0; ri < sizeof(rows) / sizeof(rows[0]); ++ri) {
            uint64_t service = 0, events = 0, rstall = 0, hits = 0, misses = 0;
            if (run_row(rows[ri].mode_name, rows[ri].mode,
                        rows[ri].sched_name, rows[ri].sched, rows[ri].rate,
                        rows[ri].policy_name, rows[ri].policy,
                        patterns[pi].pattern, &service, &events, &rstall,
                        &hits, &misses)) {
                fprintf(stderr, "FAIL: row setup %s/%s\n",
                        rows[ri].mode_name, patterns[pi].pattern == STEADY ? "steady" : "burst_idle");
                failures++;
                continue;
            }
            uint64_t accesses = (patterns[pi].pattern == STEADY) ? 40000 : 400;
            printf("%-11s %-9s %-9s %-4u %-10s %9" PRIu64 " %9" PRIu64
                   " %8" PRIu64 " %8" PRIu64 " %6" PRIu64 " %6" PRIu64 "\n",
                   rows[ri].mode_name, rows[ri].sched_name,
                   patterns[pi].pattern == STEADY ? "steady" : "burst_idle",
                   rows[ri].rate, rows[ri].policy_name, accesses, service,
                   events, rstall, hits, misses);

            /* ---- Accounting gates ---- */
            uint64_t dur = (rows[ri].mode == TU_DRAM_REFRESH_ALL_BANK) ? 350 : 90;
            uint64_t trefi_eff = (rows[ri].mode == TU_DRAM_REFRESH_NONE)
                                     ? 7800 : 7800 / rows[ri].rate;
            uint64_t total_ticks = (patterns[pi].pattern == STEADY) ? 40000 : 40400;
            uint64_t expected = total_ticks / trefi_eff;

            if (rows[ri].mode == TU_DRAM_REFRESH_NONE) {
                if (events != 0 || rstall != 0) {
                    fprintf(stderr, "  GATE FAIL: NONE must have zero refresh cost\n");
                    failures++;
                }
                if (service != accesses * 50) {
                    fprintf(stderr, "  GATE FAIL: NONE service must be accesses*50 "
                            "(%" PRIu64 " != %" PRIu64 ")\n",
                            service, accesses * 50);
                    failures++;
                }
            } else {
                if (rows[ri].mode == TU_DRAM_REFRESH_ALL_BANK) {
                    /* Fixed fires at every k*tREFI; deferred fires once per
                     * period at access-or-deadline — same count. */
                    if (events < expected - 1 || events > expected + 1) {
                        fprintf(stderr, "  GATE FAIL: all-bank events %" PRIu64
                                " not ~%" PRIu64 "\n", events, expected);
                        failures++;
                    }
                } else {
                    /* Per-bank: each bank refreshes once per tREFI. */
                    uint64_t per_expected = 4 * expected; /* 4 banks */
                    if (events < per_expected - 8 || events > per_expected + 8) {
                        fprintf(stderr, "  GATE FAIL: per-bank events %" PRIu64
                                " not ~%" PRIu64 "\n", events, per_expected);
                        failures++;
                    }
                }
                /* Each access inside a refresh window pays the REMAINING
                 * duration, and the dedicated counter is the sum of those
                 * per-access delays (the same convention as row penalties
                 * folded into returned cycles). With ≤1 access per cycle at
                 * most `dur` accesses land in a `dur`-length window, paying
                 * dur + (dur-1) + ... + 1 = dur*(dur+1)/2 per event, so the
                 * correct bound is events × dur×(dur+1)/2 — not events×dur,
                 * which only holds when windows overlap at most one access. */
                uint64_t max_rstall = events * dur * (dur + 1) / 2;
                if (rstall > max_rstall) {
                    fprintf(stderr, "  GATE FAIL: refresh stall %" PRIu64
                            " exceeds events*dur*(dur+1)/2 %" PRIu64 "\n",
                            rstall, max_rstall);
                    failures++;
                }
                if (events > 0 && rstall == 0 &&
                    patterns[pi].pattern == STEADY) {
                    /* With a per-cycle access stream, every refresh window
                     * must delay at least one access. */
                    fprintf(stderr, "  GATE FAIL: steady stream paid no refresh stall\n");
                    failures++;
                }
            }
            if (rows[ri].policy == TU_DRAM_ROW_OPEN_PAGE) {
                if (hits + misses != accesses) {
                    fprintf(stderr, "  GATE FAIL: row accounting incomplete "
                            "(%" PRIu64 " + %" PRIu64 " != %" PRIu64 ")\n",
                            hits, misses, accesses);
                    failures++;
                }
                if (rows[ri].mode != TU_DRAM_REFRESH_NONE && misses < events) {
                    fprintf(stderr, "  GATE FAIL: refresh must force row misses "
                            "(%" PRIu64 " < %" PRIu64 ")\n", misses, events);
                    failures++;
                }
            }
        }
    }

    if (failures) {
        fprintf(stderr, "FAIL: %d gate violation(s)\n", failures);
        return 1;
    }
    puts("PASS: all sweep rows satisfied accounting gates");
    return 0;
}
