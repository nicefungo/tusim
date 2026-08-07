/* Exploration: DRAM open/closed-page policy with split activate/conflict costs. */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef enum { SEQUENTIAL, ROW_THRASH, BANK_STREAM } pattern_t;

static uint64_t address_for(pattern_t pattern, unsigned i) {
    switch (pattern) {
    case SEQUENTIAL: return (uint64_t)i * 64;
    case ROW_THRASH: return (uint64_t)(i & 1u) * 65536;
    case BANK_STREAM: return (uint64_t)(i % 16u) * 2048;
    }
    return 0;
}

static int run_case(const char *pattern_name, pattern_t pattern,
                    const char *cost_name, uint32_t activate_cost,
                    uint32_t conflict_cost, const char *policy_name,
                    tu_dram_row_policy_mode_t policy) {
    const tu_dram_params_t params = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 16,
        .row_buffer_size = 2048, .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&params, "row-cost-sweep");
    if (!dram || !tu_dram_set_row_policy_timing(dram, policy,
                                                 activate_cost, conflict_cost)) {
        tu_dram_destroy(dram);
        return 1;
    }

    uint64_t service = 0;
    for (unsigned i = 0; i < 64; ++i) {
        uint64_t cycles = 0, stall = 0;
        tu_dram_read(dram, address_for(pattern, i), 64, &cycles, &stall);
        service += cycles;
    }

    uint64_t hits = dram->stats.total_row_hits;
    uint64_t misses = dram->stats.total_row_conflicts;
    uint64_t empty = dram->stats.total_row_empty_misses;
    uint64_t repl = dram->stats.total_row_replacements;
    printf("%-11s %-7s %-11s %8" PRIu64 " %5" PRIu64 " %5" PRIu64
           " %5" PRIu64 " %5" PRIu64 "\n",
           pattern_name, cost_name, policy_name, service, hits, misses, empty, repl);

    int fail = 0;
    if (hits + misses != 64 || empty + repl != misses) fail = 1;
    uint64_t expected_hits = 0, expected_empty = 64, expected_repl = 0;
    if (policy == TU_DRAM_ROW_OPEN_PAGE) {
        if (pattern == SEQUENTIAL) {
            expected_hits = 62; expected_empty = 2;
        } else if (pattern == ROW_THRASH) {
            expected_empty = 1; expected_repl = 63;
        } else {
            expected_hits = 48; expected_empty = 16;
        }
    }
    if (hits != expected_hits || empty != expected_empty || repl != expected_repl)
        fail = 1;
    uint64_t expected_service = 64u * 50u + empty * activate_cost +
                                repl * conflict_cost;
    if (service != expected_service) fail = 1;
    if (policy == TU_DRAM_ROW_CLOSED_PAGE && (hits != 0 || repl != 0)) fail = 1;

    tu_dram_destroy(dram);
    return fail;
}

int main(void) {
    int failures = 0;
    puts("DRAM row-cost sweep: 1 channel, 16 banks, 2 KiB rows, 64 x 64-byte reads");
    puts("base=50; equal costs=activate 20/conflict 20; split costs=activate 20/conflict 40");
    printf("%-11s %-7s %-11s %8s %5s %5s %5s %5s\n",
           "pattern", "cost", "policy", "service", "hits", "miss", "empty", "repl");

    const struct { const char *name; pattern_t pattern; } patterns[] = {
        {"sequential", SEQUENTIAL},
        {"row_thrash", ROW_THRASH},
        {"bank_stream", BANK_STREAM},
    };
    const struct { const char *name; uint32_t activate, conflict; } costs[] = {
        {"equal", 20, 20},
        {"split", 20, 40},
    };
    for (size_t p = 0; p < sizeof(patterns) / sizeof(patterns[0]); ++p) {
        for (size_t c = 0; c < sizeof(costs) / sizeof(costs[0]); ++c) {
            failures += run_case(patterns[p].name, patterns[p].pattern,
                                 costs[c].name, costs[c].activate, costs[c].conflict,
                                 "open_page", TU_DRAM_ROW_OPEN_PAGE);
            failures += run_case(patterns[p].name, patterns[p].pattern,
                                 costs[c].name, costs[c].activate, costs[c].conflict,
                                 "closed_page", TU_DRAM_ROW_CLOSED_PAGE);
        }
    }
    if (failures) {
        fprintf(stderr, "FAIL: %d invalid rows\n", failures);
        return 1;
    }
    puts("PASS: 12 rows passed complete activation/replacement accounting gates");
    return 0;
}
