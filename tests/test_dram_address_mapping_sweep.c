/* Exploration: DRAM burst, row, and XOR-interleaved address mappings. */
#include <inttypes.h>
#include <stdio.h>

#include "memory/dram_model.h"

typedef enum { CONTIGUOUS, ROW_STRIDE } pattern_t;

static uint64_t address_for(pattern_t pattern, unsigned i) {
    return pattern == CONTIGUOUS ? (uint64_t)i * 64u : (uint64_t)i * 2048u;
}

static unsigned bit_count(uint64_t value) {
    unsigned count = 0;
    while (value) {
        count += (unsigned)(value & 1u);
        value >>= 1;
    }
    return count;
}

static int run_case(const char *pattern_name, pattern_t pattern,
                    const char *mapping_name,
                    tu_dram_address_mapping_mode_t mapping) {
    tu_dram_params_t params = {
        .clock_ghz = 1.0,
        .bandwidth_gbps = 256.0,
        .read_latency_cycles = 50,
        .write_latency_cycles = 50,
        .bus_width_bytes = 32,
        .burst_length = 64,
        .channels = 4,
        .banks_per_channel = 16,
        .row_buffer_size = 2048,
        .model_row_conflicts = false,
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&params, "mapping-sweep");
    if (!dram || !tu_dram_set_row_policy(dram, TU_DRAM_ROW_OPEN_PAGE, 20) ||
        !tu_dram_set_address_mapping(dram, mapping)) {
        tu_dram_destroy(dram);
        return 1;
    }

    uint64_t service = 0, channel_mask = 0;
    unsigned channel_accesses[4] = {0};
    for (unsigned i = 0; i < 64; ++i) {
        uint64_t addr = address_for(pattern, i);
        uint64_t cycles = 0, stall = 0;
        uint32_t channel = 0;
        if (!tu_dram_decode_address(dram, addr, &channel, NULL, NULL) ||
            channel >= 4) {
            tu_dram_destroy(dram);
            return 1;
        }
        channel_mask |= UINT64_C(1) << channel;
        channel_accesses[channel]++;
        tu_dram_read(dram, addr, 64, &cycles, &stall);
        service += cycles;
    }

    unsigned max_channel = 0;
    for (unsigned c = 0; c < 4; ++c)
        if (channel_accesses[c] > max_channel) max_channel = channel_accesses[c];

    printf("%-11s %-19s %8" PRIu64 " %6" PRIu64 " %6" PRIu64
           " %7u %7u\n",
           pattern_name, mapping_name, service,
           dram->stats.total_row_hits, dram->stats.total_row_conflicts,
           bit_count(channel_mask), max_channel);
    int failed = dram->stats.total_row_hits + dram->stats.total_row_conflicts != 64;
    tu_dram_destroy(dram);
    return failed;
}

int main(void) {
    int failures = 0;
    puts("DRAM address-mapping sweep: 4 channels, 16 banks/channel, 2 KiB rows");
    puts("64 x 64-byte reads, open-page, base=50 cycles, miss penalty=20 cycles");
    printf("%-11s %-19s %8s %6s %6s %7s %7s\n",
           "pattern", "mapping", "service", "hits", "miss", "used_ch",
           "max_ch");
    const struct { const char *name; pattern_t pattern; } patterns[] = {
        {"contiguous", CONTIGUOUS},
        {"stride_2k", ROW_STRIDE},
    };
    for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); ++i) {
        failures += run_case(patterns[i].name, patterns[i].pattern,
                             "burst_interleaved", TU_DRAM_ADDR_BURST_INTERLEAVED);
        failures += run_case(patterns[i].name, patterns[i].pattern,
                             "row_interleaved", TU_DRAM_ADDR_ROW_INTERLEAVED);
        failures += run_case(patterns[i].name, patterns[i].pattern,
                             "xor_interleaved", TU_DRAM_ADDR_XOR_INTERLEAVED);
    }
    if (failures) {
        fprintf(stderr, "FAIL: %d invalid rows\n", failures);
        return 1;
    }
    puts("PASS: all mapping/workload rows produced complete row accounting");
    return 0;
}
