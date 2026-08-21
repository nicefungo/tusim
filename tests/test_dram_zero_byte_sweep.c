/* Exploration: zero-byte DRAM request semantics across timing modes. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef struct {
    const char *name;
    tu_dram_turnaround_mode_t mode;
} mode_case_t;

static tu_dram_model_t *make_dram(tu_dram_turnaround_mode_t mode) {
    const tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 10, .write_latency_cycles = 8,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 1,
        .row_buffer_size = 256, .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "zero-byte-sweep");
    if (!dram || !tu_dram_set_row_policy(dram, TU_DRAM_ROW_OPEN_PAGE, 20) ||
        !tu_dram_set_turnaround(dram, mode,
                                TU_DRAM_TURNAROUND_CORE_CYCLES, 3, 8) ||
        !tu_dram_set_burst_granules(dram, 64, 64)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    return dram;
}

static int run_zero_case(const mode_case_t *c, int is_read) {
    tu_dram_model_t *dram = make_dram(c->mode);
    if (!dram) return 1;
    uint64_t cycles = UINT64_MAX, stall = UINT64_MAX;
    if (is_read)
        tu_dram_read(dram, 1, 0, &cycles, &stall);
    else
        tu_dram_write(dram, 63, 0, &cycles, &stall);

    uint64_t estimate = tu_dram_estimate_transfer(dram, 0, is_read != 0);
    uint64_t requests = dram->stats.total_reads + dram->stats.total_writes;
    uint64_t occupied = dram->stats.total_read_occupied_bytes +
                        dram->stats.total_write_occupied_bytes;
    printf("%-12s %c %5" PRIu64 " %5" PRIu64 " %5" PRIu64 " %8" PRIu64
           " %7" PRIu64 " %7" PRIu64 "\n",
           c->name, is_read ? 'R' : 'W', cycles, stall, estimate, requests,
           occupied, dram->stats.total_row_conflicts);
    int fail = cycles != 0 || stall != 0 || estimate != 0 ||
               requests != 0 || occupied != 0 ||
               dram->stats.total_row_conflicts != 0 ||
               dram->channel_last_direction[0] != 0 ||
               dram->channel_available_cycle[0] != 0 ||
               dram->open_rows[0] != UINT64_MAX;
    tu_dram_destroy(dram);
    return fail;
}

static int run_one_byte_control(void) {
    tu_dram_model_t *dram = make_dram(TU_DRAM_TURNAROUND_BURST_SPAN_CREDIT);
    if (!dram) return 1;
    uint64_t cycles = 0, stall = 0;
    for (int i = 0; i < 1001; ++i) tu_dram_tick(dram);
    tu_dram_read(dram, 63, 1, &cycles, &stall);
    printf("%-12s R %5" PRIu64 " %5" PRIu64 " %8" PRIu64
           " %7" PRIu64 " %7" PRIu64 "\n",
           "one-byte", cycles, stall, dram->stats.total_reads,
           dram->stats.total_read_occupied_bytes,
           dram->stats.total_row_conflicts);
    int fail = cycles != 30 || stall != 0 || dram->stats.total_reads != 1 ||
               dram->stats.total_read_bytes != 1 ||
               dram->stats.total_read_occupied_bytes != 64 ||
               dram->stats.total_row_conflicts != 1;
    tu_dram_destroy(dram);
    return fail;
}

int main(void) {
    const mode_case_t modes[] = {
        {"none", TU_DRAM_TURNAROUND_NONE},
        {"fixed", TU_DRAM_TURNAROUND_FIXED},
        {"idle-credit", TU_DRAM_TURNAROUND_IDLE_CREDIT},
        {"exact-burst", TU_DRAM_TURNAROUND_BURST_CREDIT},
        {"size-round", TU_DRAM_TURNAROUND_BURST_ROUND_CREDIT},
        {"span-round", TU_DRAM_TURNAROUND_BURST_SPAN_CREDIT},
    };
    int failures = 0;
    puts("DRAM zero-byte sweep: open-page row model, 64 B granule");
    printf("%-12s %s %5s %5s %5s %8s %7s %7s\n",
           "mode", "D", "cycles", "stall", "est", "requests", "occ_B", "misses");
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        failures += run_zero_case(&modes[i], 1);
        failures += run_zero_case(&modes[i], 0);
    }
    failures += run_one_byte_control();
    if (failures) {
        fprintf(stderr, "FAIL: %d zero-byte rows violated exact gates\n", failures);
        return 1;
    }
    puts("PASS: 12 zero-byte service/estimate rows are no-ops; one-byte control consumes service");
    return 0;
}
