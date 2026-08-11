/* Exploration: adaptive row timeout cycle-domain versus physical-ns domain. */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef struct {
    uint64_t service;
    uint64_t hits;
    uint64_t timeout_precharges;
    uint32_t timeout_cycles;
} result_t;

static tu_dram_model_t *make_dram(double clock_ghz,
                                  tu_dram_row_timeout_domain_t domain) {
    const tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 1,
        .row_buffer_size = 256, .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "timeout-domain-sweep");
    if (!dram || !tu_dram_configure_core_clock(dram, clock_ghz) ||
        !tu_dram_set_row_policy_timeout_domain(
            dram, TU_DRAM_ROW_ADAPTIVE_TIMEOUT, 20, 40, domain, 8.0)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    return dram;
}

static int run_case(double clock_ghz, double gap_ns,
                    const char *domain_name,
                    tu_dram_row_timeout_domain_t domain,
                    const result_t *expected) {
    tu_dram_model_t *dram = make_dram(clock_ghz, domain);
    if (!dram) return 1;
    uint64_t cycles = 0, stall = 0;
    result_t got = {0};
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    got.service += cycles;
    uint32_t gap_cycles = (uint32_t)ceil(gap_ns * clock_ghz);
    for (uint32_t i = 0; i < gap_cycles; ++i) tu_dram_tick(dram);
    tu_dram_read(dram, 0, 64, &cycles, &stall);
    got.service += cycles;
    got.hits = dram->stats.total_row_hits;
    got.timeout_precharges = dram->stats.total_row_timeout_precharges;
    got.timeout_cycles = dram->row_open_timeout_cycles;
    printf("%4.1f %-13s %6.1f %9u %7" PRIu64 " %4" PRIu64 " %7" PRIu64 "\n",
           clock_ghz, domain_name, gap_ns, got.timeout_cycles, got.service,
           got.hits, got.timeout_precharges);
    int fail = got.service != expected->service || got.hits != expected->hits ||
               got.timeout_precharges != expected->timeout_precharges ||
               got.timeout_cycles != expected->timeout_cycles;
    tu_dram_destroy(dram);
    return fail;
}

int main(void) {
    const double clocks[] = {0.5, 1.0, 2.0};
    const double gaps[] = {6.0, 12.0};
    const char *domain_names[] = {"core_cycles", "physical_ns"};
    const tu_dram_row_timeout_domain_t domains[] = {
        TU_DRAM_ROW_TIMEOUT_CORE_CYCLES, TU_DRAM_ROW_TIMEOUT_PHYSICAL_NS
    };
    const result_t expected[2][2][3] = {
        { /* 6 ns gap */
          {{120,1,0,8}, {120,1,0,8}, {140,0,1,8}},
          {{120,1,0,4}, {120,1,0,8}, {120,1,0,16}} },
        { /* 12 ns gap */
          {{120,1,0,8}, {140,0,1,8}, {140,0,1,8}},
          {{140,0,1,4}, {140,0,1,8}, {140,0,1,16}} },
    };
    int failures = 0;
    puts("DRAM row-timeout domain sweep: timeout=8 cycles or 8 ns, base=50, activate=20");
    printf("%4s %-13s %6s %9s %7s %4s %7s\n",
           "GHz", "domain", "gap_ns", "timeout_c", "service", "hits", "timeout");
    for (size_t g = 0; g < 2; ++g)
        for (size_t d = 0; d < 2; ++d)
            for (size_t c = 0; c < 3; ++c)
                failures += run_case(clocks[c], gaps[g], domain_names[d],
                                     domains[d], &expected[g][d][c]);
    if (failures) {
        fprintf(stderr, "FAIL: %d rows violated exact gates\n", failures);
        return 1;
    }
    puts("PASS: 12 rows passed exact timeout conversion and row-state gates");
    return 0;
}
