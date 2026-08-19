/* Exploration: payload-size rounding versus address-span DRAM bursts. */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef struct {
    const char *name;
    char first;
    tu_dram_turnaround_mode_t mode;
    uint64_t addr;
    uint32_t bytes;
    uint64_t expected_service;
    uint64_t expected_first_occupied;
} case_t;

static tu_dram_model_t *make_dram(tu_dram_turnaround_mode_t mode) {
    const tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 10, .write_latency_cycles = 8,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 1,
        .row_buffer_size = 256, .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "burst-alignment-sweep");
    if (!dram || !tu_dram_set_turnaround(
            dram, mode, TU_DRAM_TURNAROUND_CORE_CYCLES, 3, 8) ||
        !tu_dram_set_burst_granules(dram, 64, 64)) {
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
        tu_dram_read(dram, c->addr, c->bytes, &first_cycles, &stall);
    else
        tu_dram_write(dram, c->addr, c->bytes, &first_cycles, &stall);
    for (uint32_t i = 0; i < 20; ++i) tu_dram_tick(dram);
    if (c->first == 'R')
        tu_dram_write(dram, 0, 16, &second_cycles, &stall);
    else
        tu_dram_read(dram, 0, 16, &second_cycles, &stall);

    uint64_t service = first_cycles + second_cycles;
    uint64_t first_occupied = c->first == 'R'
        ? dram->stats.total_read_occupied_bytes
        : dram->stats.total_write_occupied_bytes;
    printf("%-12s %c2%c %4" PRIu64 " %5u %8" PRIu64 " %7" PRIu64
           " %7" PRIu64 "\n",
           c->name, c->first, c->first == 'R' ? 'W' : 'R', c->addr,
           c->bytes, first_occupied, service,
           dram->stats.total_turnaround_cycles);
    int fail = service != c->expected_service ||
               first_occupied != c->expected_first_occupied;
    tu_dram_destroy(dram);
    return fail;
}

int main(void) {
    const case_t cases[] = {
        {"size-aligned", 'R', TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 0, 64, 19, 64},
        {"size-misalign", 'R', TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 1, 64, 19, 64},
        {"span-aligned", 'R', TU_DRAM_TURNAROUND_BURST_SPAN_CREDIT, 0, 64, 19, 64},
        {"span-misalign", 'R', TU_DRAM_TURNAROUND_BURST_SPAN_CREDIT, 1, 64, 21, 128},
        {"span-tail", 'R', TU_DRAM_TURNAROUND_BURST_SPAN_CREDIT, 63, 80, 21, 192},
        {"size-misalign", 'W', TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 1, 64, 22, 64},
        {"span-misalign", 'W', TU_DRAM_TURNAROUND_BURST_SPAN_CREDIT, 1, 64, 26, 128},
        {"exact-control", 'R', TU_DRAM_TURNAROUND_BURST_CREDIT, 63, 80, 21, 80},
    };
    int failures = 0;
    puts("DRAM burst alignment sweep: granule=64 B, gap=20 cycles");
    printf("%-12s %-3s %4s %5s %8s %7s %7s\n",
           "config", "dir", "addr", "bytes", "first_B", "service", "ta_cyc");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
        failures += run_case(&cases[i]);
    if (failures) {
        fprintf(stderr, "FAIL: %d burst-alignment rows violated exact gates\n",
                failures);
        return 1;
    }
    puts("PASS: 8 rows passed address-span occupancy and turnaround gates");
    return 0;
}
