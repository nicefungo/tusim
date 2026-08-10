/* Exploration: static open/closed page versus adaptive idle-timeout policy. */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef enum { DENSE_REUSE, SPARSE_REUSE, ROW_THRASH, BURST_PAIRS } pattern_t;

typedef struct {
    uint64_t service, hits, empty, repl, timeout;
} result_t;

static tu_dram_model_t *make_dram(tu_dram_row_policy_mode_t policy) {
    const tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 1,
        .row_buffer_size = 256, .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "timeout-sweep");
    if (!dram || !tu_dram_set_row_policy_timeout(dram, policy, 20, 40, 8)) {
        tu_dram_destroy(dram);
        return NULL;
    }
    return dram;
}

static void idle(tu_dram_model_t *dram, unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) tu_dram_tick(dram);
}

static int run_case(const char *pattern_name, pattern_t pattern,
                    const char *policy_name, tu_dram_row_policy_mode_t policy,
                    const result_t *expected) {
    tu_dram_model_t *dram = make_dram(policy);
    if (!dram) return 1;
    result_t got = {0};
    for (unsigned i = 0; i < 16; ++i) {
        uint64_t addr = 0;
        unsigned gap = 0;
        if (pattern == DENSE_REUSE) gap = (i == 0) ? 0 : 2;
        else if (pattern == SPARSE_REUSE) gap = (i == 0) ? 0 : 16;
        else if (pattern == ROW_THRASH) addr = (uint64_t)(i & 1u) * 256;
        else {
            unsigned pair = i / 2;
            addr = (uint64_t)pair * 256 + (uint64_t)(i & 1u) * 64;
            gap = (i > 0 && (i & 1u) == 0) ? 16 : 0;
        }
        idle(dram, gap);
        uint64_t cycles = 0, stall = 0;
        tu_dram_read(dram, addr, 64, &cycles, &stall);
        got.service += cycles;
    }
    got.hits = dram->stats.total_row_hits;
    got.empty = dram->stats.total_row_empty_misses;
    got.repl = dram->stats.total_row_replacements;
    got.timeout = dram->stats.total_row_timeout_precharges;
    printf("%-12s %-17s %7" PRIu64 " %4" PRIu64 " %5" PRIu64
           " %4" PRIu64 " %7" PRIu64 "\n",
           pattern_name, policy_name, got.service, got.hits, got.empty,
           got.repl, got.timeout);
    int fail = got.service != expected->service || got.hits != expected->hits ||
               got.empty != expected->empty || got.repl != expected->repl ||
               got.timeout != expected->timeout;
    tu_dram_destroy(dram);
    return fail;
}

int main(void) {
    const struct {
        const char *name;
        pattern_t pattern;
        result_t expected[3];
    } cases[] = {
        {"dense_reuse", DENSE_REUSE, {{820,15,1,0,0}, {1120,0,16,0,0}, {820,15,1,0,0}}},
        {"sparse_reuse", SPARSE_REUSE, {{820,15,1,0,0}, {1120,0,16,0,0}, {1120,0,16,0,15}}},
        {"row_thrash", ROW_THRASH, {{1420,0,1,15,0}, {1120,0,16,0,0}, {1420,0,1,15,0}}},
        {"burst_pairs", BURST_PAIRS, {{1100,8,1,7,0}, {1120,0,16,0,0}, {960,8,8,0,7}}},
    };
    const char *policy_names[] = {"open_page", "closed_page", "adaptive_timeout"};
    const tu_dram_row_policy_mode_t policies[] = {
        TU_DRAM_ROW_OPEN_PAGE, TU_DRAM_ROW_CLOSED_PAGE,
        TU_DRAM_ROW_ADAPTIVE_TIMEOUT
    };
    int failures = 0;
    puts("DRAM adaptive row-policy sweep: base=50, activate=20, replace=40, timeout=8 cycles");
    printf("%-12s %-17s %7s %4s %5s %4s %7s\n",
           "pattern", "policy", "service", "hit", "empty", "repl", "timeout");
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c)
        for (size_t p = 0; p < 3; ++p)
            failures += run_case(cases[c].name, cases[c].pattern,
                                 policy_names[p], policies[p],
                                 &cases[c].expected[p]);
    if (failures) {
        fprintf(stderr, "FAIL: %d rows violated exact gates\n", failures);
        return 1;
    }
    puts("PASS: 12 rows passed exact service and row-state accounting gates");
    return 0;
}
