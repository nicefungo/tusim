/* DMA serialized versus overlapped command-issue/payload exploration. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SPACE 2048u
#define BASE 50u

static uint8_t input[SPACE];
static uint8_t output[SPACE];

static void init_engine(int mode, uint32_t burst_bytes, uint32_t issue_cycles,
                        int segmentation, int payload_scope,
                        uint32_t channels, int binding) {
    tu_dma_init_config_overlap(
        true, channels, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, binding, 256u, BASE, BASE,
        burst_bytes, burst_bytes, burst_bytes, issue_cycles,
        0u, 0u, false, false, segmentation,
        TU_DMA_BASE_PER_DESCRIPTOR, payload_scope, mode);
}

static int propagation_gate(void) {
    tu_config_t cfg;
    char err[192] = {0};
    if (tu_config_load_string(
            "{\"tu\":{\"dma\":{\"issue_payload_mode\":\"overlapped\","
            "\"burst_issue_cycles\":4}}}",
            &cfg, err, sizeof(err)) != 0) return -1;
    if (cfg.dma_issue_payload_mode !=
        TU_DMA_CONFIG_ISSUE_PAYLOAD_OVERLAPPED) return -2;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (rt.dma_issue_payload_mode !=
        TU_DMA_CONFIG_ISSUE_PAYLOAD_OVERLAPPED) return -3;
    tu_init_with_config(&rt);
    if (g_tu_dma.issue_payload_mode != TU_DMA_ISSUE_PAYLOAD_OVERLAPPED)
        return -4;

    if (tu_config_load_string(
            "{\"tu\":{\"dma\":{\"issue_payload_mode\":\"magic\"}}}",
            &cfg, err, sizeof(err)) == 0) return -5;
    return strstr(err, "issue_payload_mode") ? 0 : -6;
}

static int run_linear(int mode, tu_dma_direction_t direction, uint32_t bytes,
                      uint32_t burst_bytes, uint32_t issue_cycles,
                      uint64_t expected_completion, uint64_t *completion_out) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-issue-payload-linear");
    sram.banks.bw_modeling = false;
    memset(tu_sram_raw_ptr(&sram), 0, SPACE);
    memset(output, 0, sizeof(output));
    if (direction == TU_DMA_DIR_TU_TO_HOST)
        memcpy(tu_sram_raw_ptr(&sram), input, bytes);

    init_engine(mode, burst_bytes, issue_cycles, TU_DMA_SEGMENT_AGGREGATE,
                TU_DMA_PAYLOAD_PACKED_DESCRIPTOR, 1, TU_DMA_BIND_EXPLICIT);
    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        0, direction, &sram, 0,
        direction == TU_DMA_DIR_HOST_TO_TU ? (void *)input : (void *)output,
        1, bytes);
    if (!desc || tu_dma_submit_desc(desc) == 0) return -1;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 2000u)
        tu_dma_tick();

    *completion_out = desc->cycles_completed;
    const uint8_t *dst = direction == TU_DMA_DIR_HOST_TO_TU ?
                         tu_sram_raw_ptr(&sram) : output;
    if (*completion_out != expected_completion ||
        g_tu_dma.current_cycle != expected_completion ||
        g_tu_dma.total_bytes != bytes ||
        g_tu_dma.total_occupied_bytes != ((bytes + 31u) / 32u) * 32u ||
        memcmp(dst, input, bytes) != 0) return -2;

    tu_dma_destroy();
    desc->next = NULL;
    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&sram);
    return 0;
}

static int fragmented_gate(int mode, tu_dma_direction_t direction,
                           uint64_t expected_completion) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-issue-payload-fragmented");
    sram.banks.bw_modeling = false;
    memset(tu_sram_raw_ptr(&sram), 0, SPACE);
    memset(output, 0, sizeof(output));
    if (direction == TU_DMA_DIR_TU_TO_HOST)
        memcpy(tu_sram_raw_ptr(&sram), input, SPACE);

    init_engine(mode, 32u, 4u, TU_DMA_SEGMENT_LOGICAL,
                TU_DMA_PAYLOAD_ALIGN_LOGICAL_SEGMENT, 1,
                TU_DMA_BIND_EXPLICIT);
    tu_dma_descriptor_t *desc = tu_dma_desc_create_strided_2d(
        0, direction, &sram, 0,
        direction == TU_DMA_DIR_HOST_TO_TU ? (void *)input : (void *)output,
        8u, 8u, 1u, 6u, 5u);
    if (!desc || tu_dma_submit_desc(desc) == 0) return -1;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 2000u)
        tu_dma_tick();
    if (desc->cycles_completed != expected_completion ||
        g_tu_dma.total_bytes != 30u || g_tu_dma.total_occupied_bytes != 192u)
        return -2;
    const uint8_t *dst = direction == TU_DMA_DIR_HOST_TO_TU ?
                         tu_sram_raw_ptr(&sram) : output;
    for (uint32_t row = 0; row < 6; row++)
        if (memcmp(dst + row * 8u, input + row * 8u, 5u) != 0) return -3;

    tu_dma_destroy();
    desc->next = NULL;
    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&sram);
    return 0;
}

static int projected_binding_gate(int mode, uint8_t expected_channel) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-issue-payload-projection");
    sram.banks.bw_modeling = false;
    init_engine(mode, 128u, 4u, TU_DMA_SEGMENT_LOGICAL,
                TU_DMA_PAYLOAD_ALIGN_LOGICAL_SEGMENT, 2,
                TU_DMA_BIND_EXPLICIT);

    tu_dma_descriptor_t *command_heavy = tu_dma_desc_create_strided_2d(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, input, 8u, 8u, 1u, 6u, 5u);
    tu_dma_descriptor_t *balanced = tu_dma_desc_create_linear(
        1, TU_DMA_DIR_HOST_TO_TU, &sram, 128u, input, 1u, 480u);
    tu_dma_descriptor_t *probe = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 1024u, input, 1u, 16u);
    if (!command_heavy || !balanced || !probe ||
        tu_dma_submit_desc(command_heavy) == 0 ||
        tu_dma_submit_desc(balanced) == 0) return -1;
    g_tu_dma.binding_policy = TU_DMA_BIND_LEAST_PROJECTED_CYCLES;
    g_tu_dma.next_binding_channel = 0;
    if (tu_dma_submit_desc(probe) == 0 || probe->channel != expected_channel)
        return -2;

    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

static int legacy_and_compatibility_gate(void) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-issue-payload-legacy");
    sram.banks.bw_modeling = false;

    init_engine(TU_DMA_ISSUE_PAYLOAD_SERIALIZED, 32u, 4u,
                TU_DMA_SEGMENT_AGGREGATE, TU_DMA_PAYLOAD_PACKED_DESCRIPTOR,
                1, TU_DMA_BIND_EXPLICIT);
    tu_dma_load(TU_DMA_CHAN_W, &sram, 0, input, 96u);
    if (g_tu_dma.estimated_cycles != 65u ||
        memcmp(tu_sram_raw_ptr(&sram), input, 96u) != 0) return -1;

    init_engine(TU_DMA_ISSUE_PAYLOAD_OVERLAPPED, 32u, 4u,
                TU_DMA_SEGMENT_AGGREGATE, TU_DMA_PAYLOAD_PACKED_DESCRIPTOR,
                1, TU_DMA_BIND_EXPLICIT);
    tu_dma_store(TU_DMA_CHAN_O, &sram, 0, output, 96u);
    if (g_tu_dma.estimated_cycles != 62u ||
        memcmp(output, input, 96u) != 0) return -2;

    tu_runtime_config_t rt;
    memset(&rt, 0, sizeof(rt));
    rt.pe_rows = 1;
    rt.pe_cols = 1;
    rt.sram_w_size = 4096;
    rt.sram_a_size = 4096;
    rt.sram_o_size = 4096;
    tu_init_with_config(&rt);
    if (g_tu_dma.issue_payload_mode != TU_DMA_ISSUE_PAYLOAD_SERIALIZED)
        return -3;

    init_engine(TU_DMA_ISSUE_PAYLOAD_SERIALIZED, 32u, 4u,
                TU_DMA_SEGMENT_AGGREGATE, TU_DMA_PAYLOAD_PACKED_DESCRIPTOR,
                1, TU_DMA_BIND_EXPLICIT);
    tu_dma_descriptor_t *zero = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, input, 1, 0);
    if (!zero || tu_dma_submit_desc(zero) == 0) return -4;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 100u)
        tu_dma_tick();
    if (zero->cycles_completed != BASE + 1u || g_tu_dma.total_bytes != 0 ||
        g_tu_dma.total_occupied_bytes != 0) return -4;
    tu_dma_desc_destroy(zero);

    tu_dma_init_config_overlap(
        true, 1, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT, 256u, BASE, BASE,
        32u, 32u, 32u, 4u, 0u, 0u, false, false,
        TU_DMA_SEGMENT_AGGREGATE, TU_DMA_BASE_PER_DESCRIPTOR,
        TU_DMA_PAYLOAD_PACKED_DESCRIPTOR, 2);
    if (g_tu_dma.num_channels != 0) return -5;
    tu_sram_destroy(&sram);
    return 0;
}

int main(void) {
    for (uint32_t i = 0; i < SPACE; i++) input[i] = (uint8_t)(i ^ 0xa5u);
    if (propagation_gate() != 0) {
        fprintf(stderr, "FAIL: config propagation/rejection\n");
        return 1;
    }

    struct row {
        uint32_t bytes;
        uint32_t burst;
        uint32_t issue;
        uint64_t serialized;
        uint64_t overlapped;
    } rows[] = {
        {16u, 32u, 4u, 56u, 56u},
        {96u, 32u, 4u, 66u, 63u},
        {512u, 32u, 4u, 131u, 115u},
        {512u, 128u, 1u, 71u, 68u},
    };

    printf("DMA issue/payload overlap sweep (32-byte interface, 50-cycle base)\n");
    printf("bytes burst issue serialized overlapped reduction\n");
    for (uint32_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        uint64_t serial = 0, overlap = 0;
        if (run_linear(TU_DMA_ISSUE_PAYLOAD_SERIALIZED,
                       TU_DMA_DIR_HOST_TO_TU, rows[i].bytes, rows[i].burst,
                       rows[i].issue, rows[i].serialized, &serial) != 0 ||
            run_linear(TU_DMA_ISSUE_PAYLOAD_OVERLAPPED,
                       TU_DMA_DIR_HOST_TO_TU, rows[i].bytes, rows[i].burst,
                       rows[i].issue, rows[i].overlapped, &overlap) != 0) {
            fprintf(stderr, "FAIL: linear row %u\n", i);
            return 2;
        }
        double reduction = 100.0 * (double)(serial - overlap) / (double)serial;
        printf("%5u %5u %5u %10lu %10lu %8.2f%%\n",
               rows[i].bytes, rows[i].burst, rows[i].issue,
               (unsigned long)serial, (unsigned long)overlap, reduction);
    }

    if (run_linear(TU_DMA_ISSUE_PAYLOAD_OVERLAPPED,
                   TU_DMA_DIR_TU_TO_HOST, 96u, 32u, 4u, 63u,
                   &(uint64_t){0}) != 0 ||
        fragmented_gate(TU_DMA_ISSUE_PAYLOAD_SERIALIZED,
                        TU_DMA_DIR_HOST_TO_TU, 81u) != 0 ||
        fragmented_gate(TU_DMA_ISSUE_PAYLOAD_OVERLAPPED,
                        TU_DMA_DIR_HOST_TO_TU, 75u) != 0 ||
        fragmented_gate(TU_DMA_ISSUE_PAYLOAD_OVERLAPPED,
                        TU_DMA_DIR_TU_TO_HOST, 75u) != 0) {
        fprintf(stderr, "FAIL: store or logical-segment gate\n");
        return 3;
    }
    if (projected_binding_gate(TU_DMA_ISSUE_PAYLOAD_SERIALIZED, 0) != 0 ||
        projected_binding_gate(TU_DMA_ISSUE_PAYLOAD_OVERLAPPED, 1) != 0) {
        fprintf(stderr, "FAIL: queued projected-cycle overlap gate\n");
        return 4;
    }
    if (legacy_and_compatibility_gate() != 0) {
        fprintf(stderr, "FAIL: legacy/default/zero/rejection gate\n");
        return 5;
    }

    printf("PASS: serialized/overlapped issue, read/write, fragmented, live/queued/legacy, config/default/rejection\n");
    return 0;
}
