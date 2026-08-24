/*
 * Runtime DMA channel-count, bus-topology, and outstanding-depth exploration.
 *
 * This probes the live async descriptor state machine. Completion time is not
 * an end-to-end GEMM, calibrated AXI, DRAM, or shared-SRAM throughput result.
 */
#include "tu_cmodel/dma_descriptor.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define STREAM_BYTES 4096u
#define STREAMS_MAX 3u
#define STREAM_WORDS ((STREAM_BYTES + TU_SRAM_BANK_WIDTH - 1u) / TU_SRAM_BANK_WIDTH)
#define SRAM_INITIAL_GRANTS (TU_SRAM_BANKS * TU_SRAM_WORDS_PER_CYCLE)
#define EXPECTED_SRAM_STALLS \
    ((STREAM_WORDS > SRAM_INITIAL_GRANTS) ? \
     ((STREAM_WORDS - SRAM_INITIAL_GRANTS) * TU_SRAM_BW_STALL_PENALTY) : 0u)
#define EXPECTED_XFER_CYCLES (TU_LATENCY_DRAM_READ + \
    ((STREAM_BYTES + TU_DMA_BUS_WIDTH_BYTES - 1u) / TU_DMA_BUS_WIDTH_BYTES) + \
    EXPECTED_SRAM_STALLS)

static uint8_t sources[STREAMS_MAX][STREAM_BYTES];

static uint64_t completed_count(uint32_t channels) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < channels; i++)
        total += g_tu_dma.channels[i].total_completed;
    return total;
}

static const char *mode_name(int mode) {
    return mode == TU_DMA_BUS_MODE_SHARED_SERIAL ? "shared_serial" : "independent";
}

static int run_stream_case(uint32_t streams, uint32_t channels, int bus_mode,
                           uint64_t *ticks_out) {
    tu_sram_region_t sram;
    tu_dma_descriptor_t *desc[STREAMS_MAX] = {0};
    tu_sram_init(&sram, STREAMS_MAX * STREAM_BYTES, "dma-sweep");
    tu_dma_init_config(true, channels, STREAMS_MAX, bus_mode);
    if (g_tu_dma.num_channels != channels ||
        g_tu_dma.bus_mode != (tu_dma_bus_mode_t)bus_mode)
        return -1;

    for (uint32_t i = 0; i < streams; i++) {
        memset(sources[i], (int)(0x30u + i), STREAM_BYTES);
        desc[i] = tu_dma_desc_create_linear(
            (uint8_t)(i % channels), TU_DMA_DIR_HOST_TO_TU,
            &sram, i * STREAM_BYTES, sources[i], 1, STREAM_BYTES);
        if (!desc[i] || tu_dma_submit_desc(desc[i]) == 0) return -2;
    }

    uint64_t limit = 100000;
    while (completed_count(channels) < streams &&
           g_tu_dma.current_cycle < limit)
        tu_dma_tick();
    if (completed_count(channels) != streams) return -3;

    uint8_t *raw = (uint8_t *)tu_sram_raw_ptr(&sram);
    for (uint32_t i = 0; i < streams; i++) {
        for (uint32_t j = 0; j < STREAM_BYTES; j++) {
            if (raw[i * STREAM_BYTES + j] != (uint8_t)(0x30u + i))
                return -4;
        }
    }

    uint64_t waves = bus_mode == TU_DMA_BUS_MODE_SHARED_SERIAL ? streams :
                     (streams + channels - 1u) / channels;
    uint64_t expected = 1u + waves * EXPECTED_XFER_CYCLES;
    if (g_tu_dma.current_cycle != expected) {
        fprintf(stderr,
                "FAIL mode=%s streams=%u channels=%u ticks=%lu expected=%lu\n",
                mode_name(bus_mode), streams, channels,
                (unsigned long)g_tu_dma.current_cycle,
                (unsigned long)expected);
        return -5;
    }

    *ticks_out = g_tu_dma.current_cycle;
    for (uint32_t i = 0; i < streams; i++) desc[i]->next = NULL;
    tu_dma_destroy();
    for (uint32_t i = 0; i < streams; i++) tu_dma_desc_destroy(desc[i]);
    tu_sram_destroy(&sram);
    return 0;
}

static int run_capacity_case(uint32_t depth, uint32_t *accepted_out) {
    tu_sram_region_t sram;
    tu_dma_descriptor_t *accepted[4] = {0};
    uint32_t accepted_count = 0;
    tu_sram_init(&sram, 4u * 64u, "dma-depth");
    tu_dma_init_full(true, 1, depth);

    for (uint32_t i = 0; i < 4; i++) {
        tu_dma_descriptor_t *d = tu_dma_desc_create_linear(
            0, TU_DMA_DIR_HOST_TO_TU, &sram, i * 64u,
            sources[0], 1, 64);
        uint32_t id = tu_dma_submit_desc(d);
        if (id > 0) accepted[accepted_count++] = d;
    }
    if (accepted_count != (depth < 4u ? depth : 4u)) return -1;

    tu_dma_flush_all();
    for (uint32_t i = 0; i < accepted_count; i++) accepted[i]->next = NULL;
    tu_dma_destroy();
    for (uint32_t i = 0; i < accepted_count; i++)
        tu_dma_desc_destroy(accepted[i]);
    tu_sram_destroy(&sram);
    *accepted_out = accepted_count;
    return 0;
}

int main(void) {
    printf("DMA live topology sweep (4096 B/stream, %u B/cycle, %u-cycle base, %u SRAM-stall cycles)\n",
           TU_DMA_BUS_WIDTH_BYTES, TU_LATENCY_DRAM_READ,
           (unsigned)EXPECTED_SRAM_STALLS);
    printf("topology streams channels completion_ticks useful_bytes_per_tick\n");
    const int modes[] = {
        TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_BUS_MODE_SHARED_SERIAL
    };
    for (uint32_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        for (uint32_t streams = 1; streams <= 3; streams++) {
            for (uint32_t channels = 1; channels <= 3; channels++) {
                uint64_t ticks = 0;
                int rc = run_stream_case(streams, channels, modes[m], &ticks);
                if (rc != 0) return 10 + (-rc);
                double rate = (double)(streams * STREAM_BYTES) / (double)ticks;
                printf("%13s %7u %8u %16lu %21.3f\n",
                       mode_name(modes[m]), streams, channels,
                       (unsigned long)ticks, rate);
            }
        }
    }

    printf("\nmax_outstanding submitted_before_first_tick accepted\n");
    const uint32_t depths[] = {1, 2, 4};
    for (uint32_t i = 0; i < sizeof(depths) / sizeof(depths[0]); i++) {
        uint32_t accepted = 0;
        int rc = run_capacity_case(depths[i], &accepted);
        if (rc != 0) return 20 + (-rc);
        printf("%15u %27u %8u\n", depths[i], 4u, accepted);
    }

    printf("PASS: topology timing, data movement, and outstanding-depth gates\n");
    return 0;
}
