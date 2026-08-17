/* Exploration: idle-time credit for DRAM read/write bus turnaround. */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef struct {
    char first;
    uint32_t gap;
    uint32_t bytes;
    tu_dram_turnaround_mode_t mode;
    uint64_t service;
    uint64_t turnaround;
} case_t;

static tu_dram_model_t *make_dram(tu_dram_turnaround_mode_t mode) {
    const tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 10, .write_latency_cycles = 8,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 1,
        .row_buffer_size = 256, .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "turnaround-idle-sweep");
    if (!dram || !tu_dram_set_turnaround(
            dram, mode, TU_DRAM_TURNAROUND_CORE_CYCLES, 3, 8)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    return dram;
}

static int run_case(const case_t *c) {
    tu_dram_model_t *dram = make_dram(c->mode);
    if (!dram) return 1;
    uint64_t first_cycles = 0, second_cycles = 0, stall = 0;
    if (c->first == 'R')
        tu_dram_read(dram, 0, c->bytes, &first_cycles, &stall);
    else
        tu_dram_write(dram, 0, c->bytes, &first_cycles, &stall);
    for (uint32_t i = 0; i < c->gap; ++i) tu_dram_tick(dram);
    if (c->first == 'R')
        tu_dram_write(dram, 0, c->bytes, &second_cycles, &stall);
    else
        tu_dram_read(dram, 0, c->bytes, &second_cycles, &stall);
    uint64_t service = first_cycles + second_cycles;
    uint64_t expected_occupied = 2ULL * c->bytes;
    if (c->mode == TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT)
        expected_occupied = 2ULL * ((c->bytes + 63U) / 64U) * 64U;
    uint64_t occupied = dram->stats.total_read_occupied_bytes +
                        dram->stats.total_write_occupied_bytes;
    tu_dram_stats_t stats;
    tu_dram_get_stats(dram, &stats);
    printf("%c2%c %-12s %3u %5u %7" PRIu64 " %6" PRIu64 " %8" PRIu64 " %7.1f\n",
           c->first, c->first == 'R' ? 'W' : 'R',
           c->mode == TU_DRAM_TURNAROUND_NONE ? "none" :
           (c->mode == TU_DRAM_TURNAROUND_FIXED ? "fixed" :
            (c->mode == TU_DRAM_TURNAROUND_IDLE_CREDIT ? "idle-credit" :
             (c->mode == TU_DRAM_TURNAROUND_BURST_CREDIT ? "burst-credit" : "burst-round"))),
           c->gap, c->bytes, service, dram->stats.total_turnaround_cycles,
           occupied, stats.payload_efficiency * 100.0);
    uint64_t expected_events = c->mode == TU_DRAM_TURNAROUND_NONE ? 0 : 1;
    int fail = service != c->service ||
               dram->stats.total_turnaround_cycles != c->turnaround ||
               dram->stats.total_turnaround_events != expected_events ||
               occupied != expected_occupied ||
               stats.payload_efficiency != (double)(2ULL * c->bytes) /
                                           (double)expected_occupied;
    tu_dram_destroy(dram);
    return fail;
}

int main(void) {
    const case_t cases[] = {
        {'R', 0,  64, TU_DRAM_TURNAROUND_NONE,         18, 0},
        {'R', 0,  64, TU_DRAM_TURNAROUND_FIXED,        21, 3},
        {'R', 0,  64, TU_DRAM_TURNAROUND_IDLE_CREDIT,  21, 3},
        {'R', 10, 64, TU_DRAM_TURNAROUND_IDLE_CREDIT,  21, 3},
        {'R', 11, 64, TU_DRAM_TURNAROUND_IDLE_CREDIT,  20, 2},
        {'R', 13, 64, TU_DRAM_TURNAROUND_IDLE_CREDIT,  18, 0},
        {'W', 0,  64, TU_DRAM_TURNAROUND_NONE,         18, 0},
        {'W', 0,  64, TU_DRAM_TURNAROUND_FIXED,        26, 8},
        {'W', 0,  64, TU_DRAM_TURNAROUND_IDLE_CREDIT,  26, 8},
        {'W', 8,  64, TU_DRAM_TURNAROUND_IDLE_CREDIT,  26, 8},
        {'W', 12, 64, TU_DRAM_TURNAROUND_IDLE_CREDIT,  22, 4},
        {'W', 16, 64, TU_DRAM_TURNAROUND_IDLE_CREDIT,  18, 0},
        {'W', 16, 64, TU_DRAM_TURNAROUND_BURST_CREDIT, 26, 8},
        {'W', 20, 64, TU_DRAM_TURNAROUND_BURST_CREDIT, 22, 4},
        {'W', 24, 64, TU_DRAM_TURNAROUND_BURST_CREDIT, 18, 0},
        {'W', 20, 16, TU_DRAM_TURNAROUND_BURST_CREDIT, 18, 0},
        {'W', 20, 16, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 22, 4},
        {'W', 24, 16, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 18, 0},
        {'W', 20, 80, TU_DRAM_TURNAROUND_BURST_CREDIT, 24, 6},
        {'W', 26, 80, TU_DRAM_TURNAROUND_BURST_CREDIT, 18, 0},
        {'W', 20, 80, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 26, 8},
        {'W', 28, 80, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 22, 4},
        {'W', 32, 80, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 18, 0},
        {'R', 13, 64, TU_DRAM_TURNAROUND_BURST_CREDIT, 21, 3},
        {'R', 21, 64, TU_DRAM_TURNAROUND_BURST_CREDIT, 18, 0},
        {'R', 15, 16, TU_DRAM_TURNAROUND_BURST_CREDIT, 18, 0},
        {'R', 20, 16, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 19, 1},
        {'R', 21, 16, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 18, 0},
    };
    int failures = 0;
    puts("DRAM turnaround idle-credit sweep: R=10, W=8, R2W=3, W2R=8");
    printf("%-3s %-12s %3s %5s %7s %6s %8s %7s\n", "dir", "mode", "gap",
           "bytes", "service", "ta_cyc", "occ_B", "useful%");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
        failures += run_case(&cases[i]);
    if (failures) {
        fprintf(stderr, "FAIL: %d idle-credit rows violated exact gates\n", failures);
        return 1;
    }
    puts("PASS: 28 rows passed exact completion-boundary, occupancy, and payload-efficiency gates");
    return 0;
}
