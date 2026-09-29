/*
 * Shared-serial DMA arbitration exploration.
 *
 * Compares descriptor-boundary round-robin, strict descriptor priority, and
 * aging priority, including submission versus queue-head scope, configurable
 * increments, and missed-grant versus wait-cycle metrics.
 * This is a deterministic queue-selection study,
 * not a preemptive, beat-level, queue-aware DRAM, or end-to-end throughput
 * model.
 */
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
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
    if (policy == TU_DMA_ARB_DEFICIT_ROUND_ROBIN) return "deficit_round_robin";
    return "round_robin";
}

static void init_drr(uint32_t quantum) {
    tu_dma_init_config_boundary_aging_policy_drr(
        true, 2, 8, TU_DMA_BUS_MODE_SHARED_SERIAL,
        TU_DMA_ARB_DEFICIT_ROUND_ROBIN, TU_DMA_AGING_FROM_SUBMISSION,
        TU_DMA_AGING_BY_MISSED_GRANTS, 1u, 0u, 1u, quantum,
        TU_DMA_BIND_EXPLICIT, TU_DMA_BUS_WIDTH_BITS,
        TU_LATENCY_DRAM_READ, TU_LATENCY_DRAM_WRITE,
        TU_DMA_MAX_BURST_BYTES, TU_DMA_MAX_BURST_BYTES,
        TU_DMA_MAX_BURST_BYTES, 0, 0, 0, false, false,
        TU_DMA_SEGMENT_AGGREGATE, TU_DMA_BASE_PER_DESCRIPTOR,
        TU_DMA_PAYLOAD_PACKED_DESCRIPTOR, TU_DMA_ISSUE_PAYLOAD_SERIALIZED,
        TU_DMA_BOUNDARY_SIZE_ONLY);
}

static void init_drr_cost(int cost_mode) {
    tu_dma_init_config_boundary_aging_policy_drr_cost(
        true, 2, 8, TU_DMA_BUS_MODE_SHARED_SERIAL,
        TU_DMA_ARB_DEFICIT_ROUND_ROBIN, TU_DMA_AGING_FROM_SUBMISSION,
        TU_DMA_AGING_BY_MISSED_GRANTS, 1u, 0u, 1u, 64u, cost_mode,
        TU_DMA_BIND_EXPLICIT, 256u, 50u, 50u, 64u, 64u, 64u,
        0u, 0u, 0u, false, false, TU_DMA_SEGMENT_AGGREGATE,
        TU_DMA_BASE_PER_DESCRIPTOR, TU_DMA_PAYLOAD_ALIGN_BURST_COMMAND,
        TU_DMA_ISSUE_PAYLOAD_SERIALIZED, TU_DMA_BOUNDARY_SRAM_ADDRESS);
}

static int run_drr_cost_case(int cost_mode, uint64_t ch0_complete[2],
                             uint64_t *ch1_complete,
                             uint64_t *batch_complete) {
    enum { BYTES = 64, SRAM_BYTES = 224 };
    static uint8_t src[3][BYTES];
    tu_sram_region_t sram;
    tu_dma_descriptor_t *d0 = NULL, *d1 = NULL, *d2 = NULL;
    tu_sram_init(&sram, SRAM_BYTES, "dma-drr-cost-sweep");
    sram.banks.bw_modeling = false;
    memset(src[0], 0x21, BYTES);
    memset(src[1], 0x43, BYTES);
    memset(src[2], 0x65, BYTES);
    init_drr_cost(cost_mode);
    if (g_tu_dma.drr_cost_mode != (tu_dma_drr_cost_mode_t)cost_mode)
        return -1;
    d0 = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU,
                                   &sram, 0u, src[0], 1u, BYTES);
    d1 = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU,
                                   &sram, 64u, src[1], 1u, BYTES);
    d2 = tu_dma_desc_create_linear(1, TU_DMA_DIR_HOST_TO_TU,
                                   &sram, 129u, src[2], 1u, BYTES);
    if (!d0 || !d1 || !d2 || !tu_dma_submit_desc(d0) ||
        !tu_dma_submit_desc(d1) || !tu_dma_submit_desc(d2)) return -2;
    while (g_tu_dma.total_transfers < 3u && g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();
    while (g_tu_dma.channels[0].total_completed +
           g_tu_dma.channels[1].total_completed < 3u &&
           g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();
    ch0_complete[0] = d0->cycles_completed;
    ch0_complete[1] = d1->cycles_completed;
    *ch1_complete = d2->cycles_completed;
    *batch_complete = g_tu_dma.current_cycle;
    if ((cost_mode == TU_DMA_DRR_CHARGE_USEFUL_BYTES &&
         (ch0_complete[0] != 53u || *ch1_complete != 106u ||
          ch0_complete[1] != 158u)) ||
        (cost_mode == TU_DMA_DRR_CHARGE_OCCUPIED_BYTES &&
         (ch0_complete[0] != 53u || ch0_complete[1] != 105u ||
          *ch1_complete != 158u)) ||
        *batch_complete != 158u ||
        memcmp(tu_sram_raw_ptr(&sram), src[0], BYTES) != 0 ||
        memcmp(tu_sram_raw_ptr(&sram) + 64u, src[1], BYTES) != 0 ||
        memcmp(tu_sram_raw_ptr(&sram) + 129u, src[2], BYTES) != 0)
        return -3;
    tu_dma_destroy();
    d0->next = d1->next = d2->next = NULL;
    tu_dma_desc_destroy(d0);
    tu_dma_desc_destroy(d1);
    tu_dma_desc_destroy(d2);
    tu_sram_destroy(&sram);
    return 0;
}

static int run_drr_case(uint32_t quantum, uint64_t small_complete[4],
                        uint64_t *large_complete, uint64_t *batch_complete) {
    enum { SMALL = 64, LARGE = 512, TOTAL = 4 * SMALL + LARGE };
    static uint8_t small_src[4][SMALL];
    static uint8_t large_src[LARGE];
    tu_sram_region_t sram;
    tu_dma_descriptor_t *small[4] = {0}, *large = NULL;
    tu_sram_init(&sram, TOTAL, "dma-drr-sweep");
    sram.banks.bw_modeling = false;
    memset(small_src, 0x3c, sizeof(small_src));
    memset(large_src, 0x7d, sizeof(large_src));
    init_drr(quantum);
    if (g_tu_dma.drr_quantum_bytes != quantum) return -1;
    for (uint32_t i = 0; i < 4; i++) {
        small[i] = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU,
            &sram, i * SMALL, small_src[i], 1, SMALL);
        if (!small[i] || !tu_dma_submit_desc(small[i])) return -2;
    }
    large = tu_dma_desc_create_linear(1, TU_DMA_DIR_HOST_TO_TU,
        &sram, 4 * SMALL, large_src, 1, LARGE);
    if (!large || !tu_dma_submit_desc(large)) return -3;
    while (g_tu_dma.total_transfers < 5u && g_tu_dma.current_cycle < 10000u)
        tu_dma_tick();
    while (g_tu_dma.channels[0].total_completed +
           g_tu_dma.channels[1].total_completed < 5u &&
           g_tu_dma.current_cycle < 10000u)
        tu_dma_tick();
    for (uint32_t i = 0; i < 4; i++) small_complete[i] = small[i]->cycles_completed;
    *large_complete = large->cycles_completed;
    *batch_complete = g_tu_dma.current_cycle;
    const uint64_t expected_large = quantum == 64u ? 275u :
                                    (quantum == 256u ? 171u : 119u);
    const uint64_t expected_small[3][4] = {
        {53u, 105u, 157u, 209u},
        {53u, 105u, 223u, 275u},
        {53u, 171u, 223u, 275u}
    };
    uint32_t row = quantum == 64u ? 0u : (quantum == 256u ? 1u : 2u);
    if (*batch_complete != 275u || *large_complete != expected_large ||
        memcmp(small_complete, expected_small[row], sizeof(expected_small[row])) != 0 ||
        memcmp(tu_sram_raw_ptr(&sram), small_src, sizeof(small_src)) != 0 ||
        memcmp(tu_sram_raw_ptr(&sram) + 4 * SMALL, large_src, LARGE) != 0)
        return -4;
    tu_dma_destroy();
    for (uint32_t i = 0; i < 4; i++) {
        small[i]->next = NULL;
        tu_dma_desc_destroy(small[i]);
    }
    large->next = NULL;
    tu_dma_desc_destroy(large);
    tu_sram_destroy(&sram);
    return 0;
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

static void init_aging_rate(uint32_t increment, uint32_t max_boost) {
    tu_dma_init_config_boundary_aging_policy_cap(
        true, 2, 8, TU_DMA_BUS_MODE_SHARED_SERIAL,
        TU_DMA_ARB_AGING_PRIORITY, TU_DMA_AGING_FROM_SUBMISSION,
        TU_DMA_AGING_BY_MISSED_GRANTS, increment, max_boost, 1u,
        TU_DMA_BIND_EXPLICIT, TU_DMA_BUS_WIDTH_BITS,
        TU_LATENCY_DRAM_READ, TU_LATENCY_DRAM_WRITE,
        TU_DMA_MAX_BURST_BYTES, TU_DMA_MAX_BURST_BYTES,
        TU_DMA_MAX_BURST_BYTES, 0, 0, 0, false, false,
        TU_DMA_SEGMENT_AGGREGATE, TU_DMA_BASE_PER_DESCRIPTOR,
        TU_DMA_PAYLOAD_PACKED_DESCRIPTOR, TU_DMA_ISSUE_PAYLOAD_SERIALIZED,
        TU_DMA_BOUNDARY_SIZE_ONLY);
}

static void init_aging_metric(int metric, uint32_t quantum) {
    tu_dma_init_config_boundary_aging_policy(
        true, 2, 4, TU_DMA_BUS_MODE_SHARED_SERIAL,
        TU_DMA_ARB_AGING_PRIORITY, TU_DMA_AGING_FROM_SUBMISSION, metric,
        1u, quantum, TU_DMA_BIND_EXPLICIT, TU_DMA_BUS_WIDTH_BITS,
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
static int run_aging_rate_case(uint32_t increment, uint32_t max_boost,
                               uint64_t *low_complete,
                               uint64_t *batch_complete) {
    enum { BYTES = 64, HIGH_COUNT = 5 };
    static uint8_t src[HIGH_COUNT + 1][BYTES];
    tu_sram_region_t sram;
    tu_dma_descriptor_t *low = NULL;
    tu_dma_descriptor_t *high[HIGH_COUNT] = {0};
    tu_sram_init(&sram, sizeof(src), "dma-aging-rate-sweep");
    sram.banks.bw_modeling = false;
    memset(src, 0x6b, sizeof(src));
    init_aging_rate(increment, max_boost);
    if (g_tu_dma.aging_increment != increment ||
        g_tu_dma.aging_max_boost != max_boost) return -1;

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
    uint64_t expected_low = max_boost != 0u && max_boost < 4u ? 313u :
                            (increment == 1u ? 261u :
                             (increment == 2u ? 157u : 105u));
    if (*low_complete != expected_low ||
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

/* One long priority-2 transfer separates the metrics. Immediately before it
 * retires, enqueue a fresh priority-2 peer. Missed-grant aging gives the old
 * priority-0 request one step and serves the fresh peer; 64-cycle aging gives
 * it two steps after 178 wait cycles, so the rotating tie serves the old work. */
static int run_aging_metric_case(int metric, uint32_t quantum,
                                 uint64_t *low_complete,
                                 uint64_t *fresh_complete,
                                 uint64_t *batch_complete) {
    enum { LOW_BYTES = 64, LONG_BYTES = 4096, TOTAL_BYTES = 4224 };
    static uint8_t low_src[LOW_BYTES];
    static uint8_t long_src[LONG_BYTES];
    static uint8_t fresh_src[LOW_BYTES];
    tu_sram_region_t sram;
    tu_dma_descriptor_t *low = NULL, *long_desc = NULL, *fresh = NULL;
    tu_sram_init(&sram, TOTAL_BYTES, "dma-aging-metric-sweep");
    sram.banks.bw_modeling = false;
    memset(low_src, 0x31, sizeof(low_src));
    memset(long_src, 0x42, sizeof(long_src));
    memset(fresh_src, 0x53, sizeof(fresh_src));
    init_aging_metric(metric, quantum);
    if (g_tu_dma.aging_metric != (tu_dma_aging_metric_t)metric ||
        g_tu_dma.aging_cycle_quantum != quantum) return -1;

    low = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram,
                                    0, low_src, 1, LOW_BYTES);
    long_desc = tu_dma_desc_create_linear(1, TU_DMA_DIR_HOST_TO_TU, &sram,
                                          LOW_BYTES, long_src, 1, LONG_BYTES);
    if (!low || !long_desc) return -2;
    low->priority = 0;
    long_desc->priority = 2;
    if (!tu_dma_submit_desc(low) || !tu_dma_submit_desc(long_desc)) return -3;
    tu_dma_tick();
    while (g_tu_dma.current_cycle < 178u) tu_dma_tick();

    fresh = tu_dma_desc_create_linear(1, TU_DMA_DIR_HOST_TO_TU, &sram,
                                      LOW_BYTES + LONG_BYTES, fresh_src,
                                      1, LOW_BYTES);
    if (!fresh) return -4;
    fresh->priority = 2;
    if (!tu_dma_submit_desc(fresh)) return -5;
    while (g_tu_dma.total_transfers < 3u && g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();
    while (g_tu_dma.channels[0].total_completed +
           g_tu_dma.channels[1].total_completed < 3u &&
           g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();

    *low_complete = low->cycles_completed;
    *fresh_complete = fresh->cycles_completed;
    *batch_complete = g_tu_dma.current_cycle;
    if ((metric == TU_DMA_AGING_BY_MISSED_GRANTS &&
         (*fresh_complete != 231u || *low_complete != 283u)) ||
        (metric == TU_DMA_AGING_BY_WAIT_CYCLES && quantum <= 64u &&
         (*low_complete != 231u || *fresh_complete != 283u)) ||
        (metric == TU_DMA_AGING_BY_WAIT_CYCLES && quantum > 64u &&
         (*fresh_complete != 231u || *low_complete != 283u)) ||
        *batch_complete != 283u) return -6;
    uint8_t *raw = tu_sram_raw_ptr(&sram);
    if (memcmp(raw, low_src, LOW_BYTES) != 0 ||
        memcmp(raw + LOW_BYTES, long_src, LONG_BYTES) != 0 ||
        memcmp(raw + LOW_BYTES + LONG_BYTES, fresh_src, LOW_BYTES) != 0)
        return -7;

    tu_dma_destroy();
    low->next = long_desc->next = fresh->next = NULL;
    tu_dma_desc_destroy(low);
    tu_dma_desc_destroy(long_desc);
    tu_dma_desc_destroy(fresh);
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
        int rc = run_aging_rate_case(increments[i], 0u, &low, &batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL aging_increment=%u rc=%d\n",
                    increments[i], rc);
            return 70 - rc;
        }
        printf("%15u %21lu %14lu\n", increments[i],
               (unsigned long)low, (unsigned long)batch);
    }
    printf("\naging_max_boost low_priority_complete batch_complete\n");
    const uint32_t caps[] = {0u, 2u, 4u};
    for (uint32_t i = 0; i < 3; i++) {
        uint64_t low = 0, batch = 0;
        int rc = run_aging_rate_case(1u, caps[i], &low, &batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL aging_max_boost=%u rc=%d\n", caps[i], rc);
            return 80 - rc;
        }
        printf("%15u %21lu %14lu\n", caps[i],
               (unsigned long)low, (unsigned long)batch);
    }
    printf("\naging_metric old_low_complete fresh_high_complete batch_complete\n");
    const int metrics[] = {
        TU_DMA_AGING_BY_MISSED_GRANTS, TU_DMA_AGING_BY_WAIT_CYCLES
    };
    const char *metric_names[] = {"grants", "cycles_q64"};
    for (uint32_t i = 0; i < 2; i++) {
        uint64_t low = 0, fresh = 0, batch = 0;
        int rc = run_aging_metric_case(metrics[i], 64u, &low, &fresh, &batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL aging_metric=%s rc=%d\n",
                    metric_names[i], rc);
            return 90 - rc;
        }
        printf("%12s %16lu %19lu %14lu\n", metric_names[i],
               (unsigned long)low, (unsigned long)fresh,
               (unsigned long)batch);
    }
    printf("\naging_quantum_domain clock_ghz effective_cycles old_low_complete fresh_high_complete batch_complete\n");
    const double clocks[] = {0.5, 1.0, 2.0};
    for (uint32_t domain = TU_DMA_CONFIG_AGING_QUANTUM_CORE_CYCLES;
         domain <= TU_DMA_CONFIG_AGING_QUANTUM_PHYSICAL_NS; domain++) {
        for (uint32_t i = 0; i < 3; i++) {
            tu_config_t cfg;
            tu_config_default(&cfg);
            cfg.dma_aging_metric = TU_DMA_CONFIG_AGING_WAIT_CYCLES;
            cfg.dma_aging_cycle_quantum = 64u;
            cfg.dma_aging_quantum_domain = (int)domain;
            cfg.dma_aging_quantum_ns = 64.0;
            cfg.dram_core_clock_ghz = clocks[i];
            if (tu_config_validate(&cfg, NULL, 0) != 0) return 100;
            tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
            uint32_t expected_quantum = domain ==
                    TU_DMA_CONFIG_AGING_QUANTUM_PHYSICAL_NS ?
                (uint32_t)(64.0 * clocks[i]) : 64u;
            if (rt.dma_aging_cycle_quantum != expected_quantum ||
                rt.dma_aging_quantum_domain != (int)domain)
                return 101;
            uint64_t low = 0, fresh = 0, batch = 0;
            int rc = run_aging_metric_case(TU_DMA_AGING_BY_WAIT_CYCLES,
                                           expected_quantum,
                                           &low, &fresh, &batch);
            if (rc != 0) {
                fprintf(stderr, "FAIL aging domain=%u clock=%.1f rc=%d\n",
                        domain, clocks[i], rc);
                return 102 - rc;
            }
            printf("%20s %9.1f %16u %16lu %19lu %14lu\n",
                   domain == TU_DMA_CONFIG_AGING_QUANTUM_PHYSICAL_NS ?
                       "physical_ns" : "core_cycles",
                   clocks[i], expected_quantum, (unsigned long)low,
                   (unsigned long)fresh, (unsigned long)batch);
        }
    }
    printf("\ndrr_quantum small0 small1 small2 small3 large batch\n");
    const uint32_t drr_quantums[] = {64u, 256u, 1024u};
    for (uint32_t i = 0; i < 3; i++) {
        uint64_t small[4] = {0}, large = 0, batch = 0;
        int rc = run_drr_case(drr_quantums[i], small, &large, &batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL drr quantum=%u rc=%d\n", drr_quantums[i], rc);
            return 120 - rc;
        }
        printf("%11u %6lu %6lu %6lu %6lu %5lu %5lu\n",
               drr_quantums[i], (unsigned long)small[0],
               (unsigned long)small[1], (unsigned long)small[2],
               (unsigned long)small[3], (unsigned long)large,
               (unsigned long)batch);
    }
    printf("\ndrr_cost_mode aligned0 aligned1 misaligned occupied_aligned occupied_misaligned batch\n");
    const int drr_cost_modes[] = {
        TU_DMA_DRR_CHARGE_USEFUL_BYTES, TU_DMA_DRR_CHARGE_OCCUPIED_BYTES
    };
    const char *drr_cost_names[] = {"useful_bytes", "occupied_bytes"};
    for (uint32_t i = 0; i < 2; i++) {
        uint64_t aligned[2] = {0}, misaligned = 0, batch = 0;
        int rc = run_drr_cost_case(drr_cost_modes[i], aligned,
                                   &misaligned, &batch);
        if (rc != 0) {
            fprintf(stderr, "FAIL drr cost=%s rc=%d\n",
                    drr_cost_names[i], rc);
            return 140 - rc;
        }
        printf("%14s %8lu %8lu %10lu %16u %19u %5lu\n",
               drr_cost_names[i], (unsigned long)aligned[0],
               (unsigned long)aligned[1], (unsigned long)misaligned,
               64u, 96u, (unsigned long)batch);
    }
    printf("PASS: exact order/cycles, aging and DRR cost alternatives, config conversion, occupied traffic, and byte movement\n");
    return 0;
}
