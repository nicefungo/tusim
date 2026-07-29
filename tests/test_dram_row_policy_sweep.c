/* Exploration: DRAM open-page versus closed-page row-buffer policy. */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "infra/config.h"
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
                    const char *policy_name, int policy) {
    tu_config_t cfg;
    tu_config_default(&cfg);
    cfg.dram_type = TU_DRAM_TYPE_DDR5;
    cfg.dram_channels = 1;
    cfg.dram_row_policy = policy;
    cfg.dram_row_miss_penalty_cycles = 20;
    cfg.dram_latency_read = 50;

    tu_dram_model_t *dram = tu_dram_create_from_config(&cfg);
    if (!dram) return 1;
    uint64_t service = 0;
    for (unsigned i = 0; i < 64; ++i) {
        uint64_t cycles = 0, stall = 0;
        tu_dram_read(dram, address_for(pattern, i), 64, &cycles, &stall);
        service += cycles;
    }
    uint64_t hits = dram->stats.total_row_hits;
    uint64_t misses = dram->stats.total_row_conflicts;
    double hit_rate = 100.0 * (double)hits / (double)(hits + misses);
    printf("%-12s %-12s %8" PRIu64 " %6" PRIu64 " %6" PRIu64 " %8.2f\n",
           pattern_name, policy_name, service, hits, misses, hit_rate);
    tu_dram_destroy(dram);
    return (hits + misses == 64 && service >= 3200) ? 0 : 1;
}

int main(void) {
    int failures = 0;
    puts("DRAM row-policy sweep: DDR5, 1 channel, 64 x 64-byte reads");
    puts("Base read=50 cycles, modeled miss penalty=20 cycles");
    printf("%-12s %-12s %8s %6s %6s %8s\n",
           "pattern", "policy", "cycles", "hits", "miss", "hit_%");
    const struct { const char *name; pattern_t pattern; } patterns[] = {
        {"sequential", SEQUENTIAL},
        {"row_thrash", ROW_THRASH},
        {"bank_stream", BANK_STREAM},
    };
    for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); ++i) {
        failures += run_case(patterns[i].name, patterns[i].pattern,
                             "open_page", TU_DRAM_CONFIG_ROW_OPEN_PAGE);
        failures += run_case(patterns[i].name, patterns[i].pattern,
                             "closed_page", TU_DRAM_CONFIG_ROW_CLOSED_PAGE);
    }
    if (failures) {
        fprintf(stderr, "FAIL: %d invalid rows\n", failures);
        return 1;
    }
    puts("PASS: all policy/workload rows produced complete row accounting");
    return 0;
}
