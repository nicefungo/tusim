/*
 * Exploration sweep: DRAM refresh scheduling phase alignment.
 *
 * The main refresh sweep showed fixed scheduling paying zero refresh
 * stall on one burst-idle phase while deferred paid +325%. That claim —
 * "neither scheduling dominates; the win is workload/phase-dependent" —
 * rests on two anecdotal phases. This sweep varies the phase of a
 * burst-idle workload across the whole tREFI grid and plots fixed vs
 * deferred all-bank refresh service as a function of phase, quantifying
 * the phase dependence as a continuum instead of two points.
 *
 * Workload: initial idle of `phase` cycles, then 8 × (50 reads at
 * 1/cycle + 5000 idle cycles). Phase ∈ {0, 780, ..., 7800} covers the
 * full tREFI grid. ALL_BANK refresh, tREFI=7800, tRFC=350.
 *
 * Gates: every row must satisfy events ≈ ticks/tREFI and
 * rstall <= events * dur * (dur+1)/2 (per-access remainder counter;
 * at most `dur` accesses can overlap a window at 1 access/cycle).
 */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

#define BURSTS 8
#define BURST_READS 50
#define IDLE_AFTER 5000
#define TREFI 7800
#define TRFC 350

/* 4 channels / 4 banks / 256 B row custom DRAM (latency 50). */
static tu_dram_model_t *make_dram(void) {
    tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 4, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    return tu_dram_create_custom(&p, "refresh-phase-sweep");
}

static int run_row(int sched, uint64_t phase, uint64_t *out_service,
                   uint64_t *out_events, uint64_t *out_rstall,
                   uint64_t *out_overlap) {
    tu_dram_model_t *dram = make_dram();
    if (!dram) return 1;
    if (!tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                             (tu_dram_refresh_scheduling_t)sched,
                             1, TREFI, TRFC, 90, TREFI)) {
        tu_dram_destroy(dram);
        return 1;
    }
    uint64_t service = 0;
    uint64_t overlap = 0;  /* reads that paid a nonzero refresh remainder */
    for (uint64_t t = 0; t < phase; ++t) tu_dram_tick(dram);
    for (int r = 0; r < BURSTS; ++r) {
        for (int i = 0; i < BURST_READS; ++i) {
            uint64_t cycles = 0, stall = 0;
            tu_dram_read(dram, (uint64_t)(r * BURST_READS + i) * 64, 64,
                         &cycles, &stall);
            service += cycles;
            if (cycles > 50) overlap++;
            tu_dram_tick(dram);
        }
        for (int i = 0; i < IDLE_AFTER; ++i) tu_dram_tick(dram);
    }
    *out_service = service;
    *out_events  = dram->stats.total_refresh_events;
    *out_rstall  = dram->stats.total_refresh_stall_cycles;
    *out_overlap = overlap;
    tu_dram_destroy(dram);
    return 0;
}

int main(void) {
    int failures = 0;
    puts("Refresh phase-alignment sweep: 4ch/4bank/256B-row, latency=50");
    puts("ALL_BANK, tREFI=7800, tRFC=350; workload = phase idle + 8 x (50 reads + 5000 idle)");
    printf("%-9s %-8s %9s %8s %8s %8s %8s\n",
           "sched", "phase", "service", "events", "rstall", "overlap",
           "avg/read");

    /* Phase grid across tREFI. tRFC=350 → 350/7800 ≈ 4.5% of the grid is
     * a refresh window, so fixed should pay stall on ~1/22 of phases. */
    for (uint64_t phase = 0; phase <= TREFI; phase += 780) {
        for (int s = 0; s < 2; ++s) {
            const char *sname = (s == 0) ? "fixed" : "defer";
            uint64_t service = 0, events = 0, rstall = 0, overlap = 0;
            if (run_row(s, phase, &service, &events, &rstall, &overlap)) {
                fprintf(stderr, "FAIL: row %s phase %" PRIu64 "\n",
                        sname, phase);
                failures++;
                continue;
            }
            uint64_t total_ticks = phase + BURSTS * (BURST_READS + IDLE_AFTER);
            uint64_t expected = total_ticks / TREFI;
            uint64_t accesses = BURSTS * BURST_READS;
            double avg = (double)service / (double)accesses;
            printf("%-9s %-8" PRIu64 " %9" PRIu64 " %8" PRIu64 " %8" PRIu64
                   " %8" PRIu64 " %8.2f\n",
                   sname, phase, service, events, rstall, overlap, avg);

            /* ---- Accounting gates ---- */
            if (events < expected - 1 || events > expected + 1) {
                fprintf(stderr, "  GATE FAIL: events %" PRIu64
                        " not ~%" PRIu64 "\n", events, expected);
                failures++;
            }
            uint64_t max_rstall = events * TRFC * (TRFC + 1) / 2;
            if (rstall > max_rstall) {
                fprintf(stderr, "  GATE FAIL: rstall %" PRIu64
                        " exceeds %" PRIu64 "\n", rstall, max_rstall);
                failures++;
            }
            /* Every read that paid a remainder must show up in rstall,
             * and overlap can never exceed accesses. */
            if (overlap > accesses) {
                fprintf(stderr, "  GATE FAIL: overlap %" PRIu64
                        " > accesses %" PRIu64 "\n", overlap, accesses);
                failures++;
            }
        }
    }

    if (failures) {
        fprintf(stderr, "FAIL: %d gate violation(s)\n", failures);
        return 1;
    }
    puts("PASS: all phase rows satisfied accounting gates");
    return 0;
}
