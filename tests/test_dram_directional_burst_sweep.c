/* Exploration: direction-specific DRAM fixed-burst occupancy granules. */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef struct {
    const char *name;
    char first;
    uint32_t read_granule;
    uint32_t write_granule;
    tu_dram_turnaround_mode_t mode;
    uint64_t expected_service;
    uint64_t expected_read_occupied;
    uint64_t expected_write_occupied;
} case_t;

static tu_dram_model_t *make_dram(const case_t *c) {
    const tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 10, .write_latency_cycles = 8,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 1,
        .row_buffer_size = 256, .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "directional-burst-sweep");
    if (!dram || !tu_dram_set_turnaround(
            dram, c->mode, TU_DRAM_TURNAROUND_CORE_CYCLES, 3, 8) ||
        !tu_dram_set_burst_granules(dram, c->read_granule,
                                    c->write_granule)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    return dram;
}

static int run_case(const case_t *c) {
    tu_dram_model_t *dram = make_dram(c);
    if (!dram) return 1;
    uint64_t first_cycles = 0, second_cycles = 0, stall = 0;
    if (c->first == 'R')
        tu_dram_read(dram, 0, 16, &first_cycles, &stall);
    else
        tu_dram_write(dram, 0, 16, &first_cycles, &stall);
    for (uint32_t i = 0; i < 20; ++i) tu_dram_tick(dram);
    if (c->first == 'R')
        tu_dram_write(dram, 0, 16, &second_cycles, &stall);
    else
        tu_dram_read(dram, 0, 16, &second_cycles, &stall);

    tu_dram_stats_t stats;
    tu_dram_get_stats(dram, &stats);
    uint64_t service = first_cycles + second_cycles;
    printf("%-10s %c2%c %4u %5u %7" PRIu64 " %6" PRIu64
           " %7" PRIu64 " %7" PRIu64 " %7.1f\n",
           c->name, c->first, c->first == 'R' ? 'W' : 'R',
           c->read_granule, c->write_granule, service,
           dram->stats.total_turnaround_cycles,
           dram->stats.total_read_occupied_bytes,
           dram->stats.total_write_occupied_bytes,
           stats.payload_efficiency * 100.0);
    int fail = service != c->expected_service ||
               dram->stats.total_read_occupied_bytes != c->expected_read_occupied ||
               dram->stats.total_write_occupied_bytes != c->expected_write_occupied ||
               stats.payload_efficiency != 32.0 /
                   (double)(c->expected_read_occupied + c->expected_write_occupied);
    tu_dram_destroy(dram);
    return fail;
}

int main(void) {
    const case_t cases[] = {
        {"symmetric", 'R', 64, 64, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 19, 64, 64},
        {"symmetric", 'W', 64, 64, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 22, 64, 64},
        {"read-wide", 'R', 128, 32, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 21, 128, 32},
        {"read-wide", 'W', 128, 32, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 18, 128, 32},
        {"write-wide", 'R', 32, 128, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 18, 32, 128},
        {"write-wide", 'W', 32, 128, TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT, 26, 32, 128},
        {"exact-ctrl", 'R', 128, 32, TU_DRAM_TURNAROUND_BURST_CREDIT, 18, 16, 16},
        {"exact-ctrl", 'W', 32, 128, TU_DRAM_TURNAROUND_BURST_CREDIT, 18, 16, 16},
    };
    int failures = 0;
    puts("DRAM directional burst sweep: payload=16 B/direction, gap=20 cycles");
    printf("%-10s %-3s %4s %5s %7s %6s %7s %7s %7s\n",
           "config", "dir", "read", "write", "service", "ta_cyc",
           "read_B", "write_B", "useful%");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
        failures += run_case(&cases[i]);
    if (failures) {
        fprintf(stderr, "FAIL: %d directional-burst rows violated exact gates\n",
                failures);
        return 1;
    }
    puts("PASS: 8 rows passed direction-specific completion, occupancy, and efficiency gates");
    return 0;
}
