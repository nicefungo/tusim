/*
 * Shared-serial DMA arbitration exploration.
 *
 * Compares descriptor-boundary round-robin, strict descriptor priority, and
 * grant-epoch aging priority, including submission versus queue-head aging
 * and configurable priority increments per missed grant.
 * This is a deterministic queue-selection study,
 * not a preemptive, beat-level, queue-aware DRAM, or end-to-end throughput
 * model.
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
    if (policy == TU_DMA_ARB_STRICT_PRIORITY) return "strict_priority";
    if (policy == TU_DMA_ARB_AGING_PRIORITY) return "aging_priority";
    return "round_robin";
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

/* Keep injecting fresh priority-2 work behind channel 1. Strict priority
 * serves all three before the priority-0 descriptor. Aging adds one effective
 * level per missed grant, so the old descriptor ties after two misses and the
 * rotating tie-break selects it before the third fresh high-priority item. */
static int run_sustained_case(int policy, uint64_t *low_complete,
                              uint64_t high_complete[3]) {
    enum { BYTES = 64 };
    static uint8_t low_src[BYTES];
    static uint8_t high_src[3][BYTES];
    tu_sram_region_t sram;
    tu_dma_descriptor_t *low = NULL;
    tu_dma_descriptor_t *high[3] = {0};
    tu_sram_init(&sram, 4u * BYTES, "dma-aging-sweep");
    sram.banks.bw_modeling = false;
    memset(low_src, 0x11, sizeof(low_src));
    memset(high_src, 0x22, sizeof(high_src));
    tu_dma_init_config_policy(true, 2, 4,
                              TU_DMA_BUS_MODE_SHARED_SERIAL, policy);
    low = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram,
                                    0, low_src, 1, BYTES);
    high[0] = tu_dma_desc_create_linear(1, TU_DMA_DIR_HOST_TO_TU, &sram,
                                        BYTES, high_src[0], 1, BYTES);
    if (!low || !high[0]) return -1;
    low->priority = 0;
    high[0]->priority = 2;
    if (!tu_dma_submit_desc(low) || !tu_dma_submit_desc(high[0])) return -2;

    tu_dma_tick();
    for (uint32_t n = 1; n < 3; n++) {
        high[n] = tu_dma_desc_create_linear(1, TU_DMA_DIR_HOST_TO_TU, &sram,
                                            (n + 1u) * BYTES,
                                            high_src[n], 1, BYTES);
        if (!high[n]) return -3;
        high[n]->priority = 2;
        if (!tu_dma_submit_desc(high[n])) return -4;
        while (g_tu_dma.arbitration_epoch < n + 1u &&
               g_tu_dma.current_cycle < 10000u)
            tu_dma_tick();
    }
    while (g_tu_dma.total_transfers < 4u &&
           g_tu_dma.current_cycle < 10000u)
        tu_dma_tick();
    while (g_tu_dma.channels[0].total_completed +
           g_tu_dma.channels[1].total_completed < 4u &&
           g_tu_dma.current_cycle < 10000u)
        tu_dma_tick();
    *low_complete = low->cycles_completed;
    for (uint32_t i = 0; i < 3; i++)
        high_complete[i] = high[i]->cycles_completed;

    if (policy == TU_DMA_ARB_STRICT_PRIORITY) {
        if (!(*low_complete > high_complete[2])) return -5;
    } else if (policy == TU_DMA_ARB_AGING_PRIORITY) {
        if (!(*low_complete < high_complete[2]) ||
            g_tu_dma.arbitration_epoch != 4u) return -6;
    }
    uint8_t *raw = tu_sram_raw_ptr(&sram);
    if (memcmp(raw, low_src, BYTES) != 0) return -7;
    for (uint32_t i = 0; i < 3; i++)
        if (memcmp(raw + (i + 1u) * BYTES, high_src[i], BYTES) != 0)
            return -8;

    tu_dma_destroy();
    low->next = NULL;
    tu_dma_desc_destroy(low);
    for (uint32_t i = 0; i < 3; i++) {
        high[i]->next = NULL;
        tu_dma_desc_destroy(high[i]);
    }
    tu_sram_destroy(&sram);
    return 0;
}

static void init_aging_scope(int scope) {
    tu_dma_init_config_boundary_aging(
        true, 2, 4, TU_DMA_BUS_MODE_SHARED_SERIAL,
        TU_DMA_ARB_AGING_PRIORITY, scope, TU_DMA_BIND_EXPLICIT,
        TU_DMA_BUS_WIDTH_BITS, TU_LATENCY_DRAM_READ, TU_LATENCY_DRAM_WRITE,
        TU_DMA_MAX_BURST_BYTES, TU_DMA_MAX_BURST_BYTES,
        TU_DMA_MAX_BURST_BYTES, 0, 0, 0, false, false,
        TU_DMA_SEGMENT_AGGREGATE, TU_DMA_BASE_PER_DESCRIPTOR,
        TU_DMA_PAYLOAD_PACKED_DESCRIPTOR, TU_DMA_ISSUE_PAYLOAD_SERIALIZED,
        TU_DMA_BOUNDARY_SIZE_ONLY);
}

static void init_aging_rate(uint32_t increment) {
    tu_dma_init_config_boundary_aging_rate(
        true, 2, 8, TU_DMA_BUS_MODE_SHARED_SERIAL,
        TU_DMA_ARB_AGING_PRIORITY, TU_DMA_AGING_FROM_SUBMISSION, increment,
        TU_DMA_BIND_EXPLICIT, TU_DMA_BUS_WIDTH_BITS,
        TU_LATENCY_DRAM_READ, TU_LATENCY_DRAM_WRITE,
        TU_DMA_MAX_BURST_BYTES, TU_DMA_MAX_BURST_BYTES,
        TU_DMA_MAX_BURST_BYTES, 0, 0, 0, false, false,
        TU_DMA_SEGMENT_AGGREGATE, TU_DMA_BASE_PER_DESCRIPTOR,
        TU_DMA_PAYLOAD_PACKED_DESCRIPTOR, TU_DMA_ISSUE_PAYLOAD_SERIALIZED,
        TU_DMA_BOUNDARY_SIZE_ONLY);
}

/* A descriptor queued behind an older same-channel head can either retain its
 * total enqueue wait (submission scope) or start aging only when it reaches
 * the selectable head (queue-head scope). */
static int run_deep_queue_case(int scope, uint64_t *deep_complete,
                               uint64_t *fresh_complete) {
    enum { BYTES = 64 };
    static uint8_t src[3][BYTES];
    tu_sram_region_t sram;
    tu_dma_descriptor_t *lead, *deep, *fresh;
    tu_sram_init(&sram, 3u * BYTES, "dma-aging-scope-sweep");
    sram.banks.bw_modeling = false;
    memset(src, 0x5a, sizeof(src));
    init_aging_scope(scope);
    if (g_tu_dma.aging_scope != (tu_dma_aging_scope_t)scope) return -1;

    lead = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram,
                                     0, src[0], 1, BYTES);
    deep = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram,
                                     BYTES, src[1], 1, BYTES);
    if (!lead || !deep) return -2;
    lead->priority = 2;
    deep->priority = 0;
    if (!tu_dma_submit_desc(lead) || !tu_dma_submit_desc(deep)) return -3;
    tu_dma_tick();

    fresh = tu_dma_desc_create_linear(1, TU_DMA_DIR_HOST_TO_TU, &sram,
                                      2u * BYTES, src[2], 1, BYTES);
    if (!fresh || !tu_dma_submit_desc(fresh)) return -4;
    while (g_tu_dma.total_transfers < 3u && g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();
    while (g_tu_dma.channels[0].total_completed +
           g_tu_dma.channels[1].total_completed < 3u &&
           g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();
    *deep_complete = deep->cycles_completed;
    *fresh_complete = fresh->cycles_completed;
    if (lead->cycles_completed != 53u) return -5;
    if (scope == TU_DMA_AGING_FROM_SUBMISSION) {
        if (*deep_complete != 105u || *fresh_complete != 157u) return -6;
    } else {
        if (*fresh_complete != 105u || *deep_complete != 157u) return -7;
    }
    if (memcmp(tu_sram_raw_ptr(&sram), src, sizeof(src)) != 0) return -8;

    tu_dma_destroy();
    lead->next = deep->next = fresh->next = NULL;
    tu_dma_desc_destroy(lead);
    tu_dma_desc_destroy(deep);
    tu_dma_desc_destroy(fresh);
    tu_sram_destroy(&sram);
    return 0;
}

/* Priority-4 arrivals expose the fairness/latency tuning range. An increment
 * of 1/2/4 lets the old priority-0 descriptor tie after 4/2/1 missed grants. */
static int run_aging_rate_case(uint32_t increment, uint64_t *low_complete,
                               uint64_t *batch_complete) {
    enum { BYTES = 64, HIGH_COUNT = 5 };
    static uint8_t src[HIGH_COUNT + 1][BYTES];
    tu_sram_region_t sram;
    tu_dma_descriptor_t *low = NULL;
    tu_dma_descriptor_t *high[HIGH_COUNT] = {0};
    tu_sram_init(&sram, sizeof(src), "dma-aging-rate-sweep");
    sram.banks.bw_modeling = false;
    memset(src, 0x6b, sizeof(src));
    init_aging_rate(increment);
    if (g_tu_dma.aging_increment != increment) return -1;

    low = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram,
                                    0, src[0], 1, BYTES);
    high[0] = tu_dma_desc_create_linear(1, TU_DMA_DIR_HOST_TO_TU, &sram,
                                        BYTES, src[1], 1, BYTES);
    if (!low || !high[0]) return -2;
    low->priority = 0;
    high[0]->priority = 4;
    if (!tu_dma_submit_desc(low) || !tu_dma_submit_desc(high[0])) return -3;
    tu_dma_tick();

    for (uint32_t n = 1; n < HIGH_COUNT; n++) {
        high[n] = tu_dma_desc_create_linear(
            1, TU_DMA_DIR_HOST_TO_TU, &sram, (n + 1u) * BYTES,
            src[n + 1u], 1, BYTES);
        if (!high[n]) return -4;
        high[n]->priority = 4;
        if (!tu_dma_submit_desc(high[n])) return -5;
        while (g_tu_dma.arbitration_epoch < n + 1u &&
               g_tu_dma.current_cycle < 10000u)
            tu_dma_tick();
    }
    while (g_tu_dma.total_transfers < HIGH_COUNT + 1u &&
           g_tu_dma.current_cycle < 10000u)
        tu_dma_tick();
    while (g_tu_dma.channels[0].total_completed +
           g_tu_dma.channels[1].total_completed < HIGH_COUNT + 1u &&
           g_tu_dma.current_cycle < 10000u)
        tu_dma_tick();

    *low_complete = low->cycles_completed;
    *batch_complete = g_tu_dma.current_cycle;
    if ((increment == 1u && *low_complete != 261u) ||
        (increment == 2u && *low_complete != 157u) ||
        (increment == 4u && *low_complete != 105u) ||
        *batch_complete != 313u ||
        memcmp(tu_sram_raw_ptr(&sram), src, sizeof(src)) != 0)
        return -6;

    tu_dma_destroy();
    low->next = NULL;
    tu_dma_desc_destroy(low);
    for (uint32_t n = 0; n < HIGH_COUNT; n++) {
        high[n]->next = NULL;
        tu_dma_desc_destroy(high[n]);
    }
    tu_sram_destroy(&sram);
    return 0;
}

int main(void) {
    const int policies[] = {
        TU_DMA_ARB_ROUND_ROBIN,
        TU_DMA_ARB_STRICT_PRIORITY,
        TU_DMA_ARB_AGING_PRIORITY
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

    printf("\nsustained policy low_complete high0 high1 high2\n");
    for (uint32_t i = 1; i < sizeof(policies) / sizeof(policies[0]); i++) {
        uint64_t low = 0, high[3] = {0};
        int rc = run_sustained_case(policies[i], &low, high);
        if (rc != 0) {
            fprintf(stderr, "FAIL sustained policy=%s rc=%d\n",
                    policy_name(policies[i]), rc);
            return 30 - rc;
        }
        printf("%15s %12lu %5lu %5lu %5lu\n",
               policy_name(policies[i]), (unsigned long)low,
               (unsigned long)high[0], (unsigned long)high[1],
               (unsigned long)high[2]);
    }
    printf("\naging_scope deep_queued fresh_peer\n");
    const int scopes[] = {
        TU_DMA_AGING_FROM_SUBMISSION, TU_DMA_AGING_FROM_QUEUE_HEAD
    };
    const char *scope_names[] = {"submission", "queue_head"};
    for (uint32_t i = 0; i < 2; i++) {
        uint64_t deep = 0, fresh = 0;
        int rc = run_deep_queue_case(scopes[i], &deep, &fresh);
        if (rc != 0) {
            fprintf(stderr, "FAIL aging_scope=%s rc=%d\n", scope_names[i], rc);
            return 50 - rc;
        }
        printf("%12s %11lu %10lu\n", scope_names[i],
               (unsigned long)deep, (unsigned long)fresh);
    }
    printf("\naging_increment low_priority_complete batch_complete\n");
    const uint32_t increments[] = {1u, 2u, 4u};
    for (uint32_t i = 0; i < 3; i++) {
        uint64_t low = 0, batch = 0;
        int rc = run_aging_rate_case(increments[i], &low, &batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL aging_increment=%u rc=%d\n",
                    increments[i], rc);
            return 70 - rc;
        }
        printf("%15u %21lu %14lu\n", increments[i],
               (unsigned long)low, (unsigned long)batch);
    }
    printf("PASS: exact order/cycles, aging scopes/rates, and byte movement\n");
    return 0;
}
