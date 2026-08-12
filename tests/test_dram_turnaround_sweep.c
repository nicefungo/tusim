/* Exploration: per-channel DRAM read/write bus-turnaround alternatives. */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef struct {
    const char *name;
    const char *ops;
} workload_t;

typedef struct {
    uint64_t service;
    uint64_t events;
    uint64_t turnaround;
    uint32_t rtw_cycles;
    uint32_t wtr_cycles;
} result_t;

static tu_dram_model_t *make_dram(double clock_ghz,
                                  tu_dram_turnaround_mode_t mode,
                                  tu_dram_turnaround_domain_t domain,
                                  double rtw, double wtr) {
    const tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 10, .write_latency_cycles = 8,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 1,
        .row_buffer_size = 256, .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "turnaround-sweep");
    if (!dram || !tu_dram_configure_core_clock(dram, clock_ghz) ||
        !tu_dram_set_turnaround(dram, mode, domain, rtw, wtr)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    return dram;
}

static result_t run_ops(tu_dram_model_t *dram, const char *ops) {
    result_t out = {0};
    for (size_t i = 0; ops[i] != '\0'; ++i) {
        uint64_t cycles = 0, stall = 0;
        if (ops[i] == 'R')
            tu_dram_read(dram, 0, 64, &cycles, &stall);
        else
            tu_dram_write(dram, 0, 64, &cycles, &stall);
        out.service += cycles;
    }
    out.events = dram->stats.total_turnaround_events;
    out.turnaround = dram->stats.total_turnaround_cycles;
    out.rtw_cycles = dram->read_to_write_turnaround_cycles;
    out.wtr_cycles = dram->write_to_read_turnaround_cycles;
    return out;
}

static int emit_case(const char *case_name, const char *ops, double clock,
                     tu_dram_turnaround_mode_t mode,
                     tu_dram_turnaround_domain_t domain,
                     double rtw, double wtr, const result_t *expected) {
    tu_dram_model_t *dram = make_dram(clock, mode, domain, rtw, wtr);
    if (!dram) return 1;
    result_t got = run_ops(dram, ops);
    printf("%-14s %3.1f %-11s %5u %5u %7" PRIu64 " %6" PRIu64 " %6" PRIu64 "\n",
           case_name, clock,
           mode == TU_DRAM_TURNAROUND_NONE ? "none" :
           (domain == TU_DRAM_TURNAROUND_CORE_CYCLES ? "fixed-cyc" : "fixed-ns"),
           got.rtw_cycles, got.wtr_cycles, got.service, got.events,
           got.turnaround);
    int fail = got.service != expected->service ||
               got.events != expected->events ||
               got.turnaround != expected->turnaround ||
               got.rtw_cycles != expected->rtw_cycles ||
               got.wtr_cycles != expected->wtr_cycles;
    tu_dram_destroy(dram);
    return fail;
}

int main(void) {
    const workload_t workloads[] = {
        {"read_only", "RRRRRRRR"},
        {"write_only", "WWWWWWWW"},
        {"alternating", "RWRWRWRW"},
        {"read_batch", "RRRRWWWW"},
        {"write_batch", "WWWWRRRR"},
    };
    const result_t none_expected[] = {
        {80,0,0,0,0}, {64,0,0,0,0}, {72,0,0,0,0},
        {72,0,0,0,0}, {72,0,0,0,0},
    };
    const result_t sym_expected[] = {
        {80,0,0,5,5}, {64,0,0,5,5}, {107,7,35,5,5},
        {77,1,5,5,5}, {77,1,5,5,5},
    };
    const result_t asym_expected[] = {
        {80,0,0,3,8}, {64,0,0,3,8}, {108,7,36,3,8},
        {75,1,3,3,8}, {80,1,8,3,8},
    };
    int failures = 0;
    puts("DRAM bus-turnaround sweep: one channel, read=10 cycles, write=8 cycles");
    printf("%-14s %3s %-11s %5s %5s %7s %6s %6s\n",
           "workload", "GHz", "mode", "R2W", "W2R", "service", "events", "ta_cyc");
    for (size_t i = 0; i < sizeof(workloads) / sizeof(workloads[0]); ++i) {
        failures += emit_case(workloads[i].name, workloads[i].ops, 1.0,
                              TU_DRAM_TURNAROUND_NONE,
                              TU_DRAM_TURNAROUND_CORE_CYCLES, 0, 0,
                              &none_expected[i]);
        failures += emit_case(workloads[i].name, workloads[i].ops, 1.0,
                              TU_DRAM_TURNAROUND_FIXED,
                              TU_DRAM_TURNAROUND_CORE_CYCLES, 5, 5,
                              &sym_expected[i]);
        failures += emit_case(workloads[i].name, workloads[i].ops, 1.0,
                              TU_DRAM_TURNAROUND_FIXED,
                              TU_DRAM_TURNAROUND_CORE_CYCLES, 3, 8,
                              &asym_expected[i]);
    }
    const double clocks[] = {0.5, 1.0, 2.0};
    const result_t physical_expected[] = {
        {92,7,20,2,4}, {108,7,36,3,8}, {144,7,72,6,16},
    };
    for (size_t i = 0; i < 3; ++i)
        failures += emit_case("alt_physical", "RWRWRWRW", clocks[i],
                              TU_DRAM_TURNAROUND_FIXED,
                              TU_DRAM_TURNAROUND_PHYSICAL_NS, 3, 8,
                              &physical_expected[i]);
    if (failures) {
        fprintf(stderr, "FAIL: %d turnaround rows violated exact gates\n", failures);
        return 1;
    }
    puts("PASS: 18 rows passed exact service/event/domain gates");
    return 0;
}
