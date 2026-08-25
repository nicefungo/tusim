/*
 * Shared-serial DMA arbitration exploration.
 *
 * Compares descriptor-boundary round-robin with strict descriptor priority.
 * This is a deterministic queue-selection study, not a preemptive, beat-level,
 * queue-aware DRAM, or end-to-end compute/DMA throughput model.
 */
#include "tu_cmodel/dma_descriptor.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define STREAMS 3u
#define STREAM_BYTES 4096u
#define STREAM_WORDS ((STREAM_BYTES + TU_SRAM_BANK_WIDTH - 1u) / TU_SRAM_BANK_WIDTH)
#define SRAM_INITIAL_GRANTS (TU_SRAM_BANKS * TU_SRAM_WORDS_PER_CYCLE)
#define EXPECTED_SRAM_STALLS \
    ((STREAM_WORDS > SRAM_INITIAL_GRANTS) ? \
     ((STREAM_WORDS - SRAM_INITIAL_GRANTS) * TU_SRAM_BW_STALL_PENALTY) : 0u)
#define XFER_CYCLES (TU_LATENCY_DRAM_READ + \
    ((STREAM_BYTES + TU_DMA_BUS_WIDTH_BYTES - 1u) / TU_DMA_BUS_WIDTH_BYTES) + \
    EXPECTED_SRAM_STALLS)

static uint8_t sources[STREAMS][STREAM_BYTES];

static const char *policy_name(int policy) {
    return policy == TU_DMA_ARB_STRICT_PRIORITY ?
           "strict_priority" : "round_robin";
}

static int run_case(int policy, uint64_t completed[STREAMS]) {
    tu_sram_region_t sram;
    tu_dma_descriptor_t *desc[STREAMS] = {0};
    const uint8_t priority[STREAMS] = {0, 10, 5};
    tu_sram_init(&sram, STREAMS * STREAM_BYTES, "dma-arbitration-sweep");
    tu_dma_init_config_policy(true, STREAMS, STREAMS,
                              TU_DMA_BUS_MODE_SHARED_SERIAL, policy);
    if (g_tu_dma.num_channels != STREAMS ||
        g_tu_dma.arb_policy != (tu_dma_arb_policy_t)policy)
        return -1;

    for (uint32_t i = 0; i < STREAMS; i++) {
        memset(sources[i], (int)(0x40u + i), STREAM_BYTES);
        desc[i] = tu_dma_desc_create_linear(
            (uint8_t)i, TU_DMA_DIR_HOST_TO_TU, &sram,
            i * STREAM_BYTES, sources[i], 1, STREAM_BYTES);
        if (!desc[i]) return -2;
        desc[i]->priority = priority[i];
        if (tu_dma_submit_desc(desc[i]) == 0) return -3;
    }

    while (g_tu_dma.total_transfers < STREAMS &&
           g_tu_dma.current_cycle < 100000u)
        tu_dma_tick();
    while (g_tu_dma.channels[0].total_completed +
           g_tu_dma.channels[1].total_completed +
           g_tu_dma.channels[2].total_completed < STREAMS &&
           g_tu_dma.current_cycle < 100000u)
        tu_dma_tick();

    for (uint32_t i = 0; i < STREAMS; i++)
        completed[i] = desc[i]->cycles_completed;

    const uint64_t first = 1u + XFER_CYCLES;
    const uint64_t second = 1u + 2u * XFER_CYCLES;
    const uint64_t third = 1u + 3u * XFER_CYCLES;
    if (policy == TU_DMA_ARB_ROUND_ROBIN) {
        if (completed[0] != first || completed[1] != second ||
            completed[2] != third) return -4;
    } else {
        if (completed[1] != first || completed[2] != second ||
            completed[0] != third) return -5;
    }
    if (g_tu_dma.current_cycle != third) return -6;

    uint8_t *raw = (uint8_t *)tu_sram_raw_ptr(&sram);
    for (uint32_t i = 0; i < STREAMS; i++) {
        for (uint32_t j = 0; j < STREAM_BYTES; j++) {
            if (raw[i * STREAM_BYTES + j] != (uint8_t)(0x40u + i))
                return -7;
        }
    }

    tu_dma_destroy();
    for (uint32_t i = 0; i < STREAMS; i++) {
        desc[i]->next = NULL;
        tu_dma_desc_destroy(desc[i]);
    }
    tu_sram_destroy(&sram);
    return 0;
}

int main(void) {
    const int policies[] = {
        TU_DMA_ARB_ROUND_ROBIN,
        TU_DMA_ARB_STRICT_PRIORITY
    };
    printf("DMA shared-serial arbitration sweep (priorities ch0/ch1/ch2=0/10/5)\n");
    printf("policy low_ch0_complete critical_ch1_complete medium_ch2_complete batch_complete\n");
    for (uint32_t i = 0; i < sizeof(policies) / sizeof(policies[0]); i++) {
        uint64_t completed[STREAMS] = {0};
        int rc = run_case(policies[i], completed);
        if (rc != 0) {
            fprintf(stderr, "FAIL policy=%s rc=%d\n",
                    policy_name(policies[i]), rc);
            return 10 - rc;
        }
        uint64_t batch = completed[0];
        if (completed[1] > batch) batch = completed[1];
        if (completed[2] > batch) batch = completed[2];
        printf("%15s %16lu %21lu %19lu %14lu\n",
               policy_name(policies[i]),
               (unsigned long)completed[0],
               (unsigned long)completed[1],
               (unsigned long)completed[2],
               (unsigned long)batch);
    }
    printf("PASS: arbitration order, exact completion cycles, and byte movement\n");
    return 0;
}
