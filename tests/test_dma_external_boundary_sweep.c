/* DMA explicit external-address and dual-endpoint 4 KiB boundary sweep. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SPACE 8192u
#define BASE_CYCLES 50u

static uint8_t input[SPACE];
static uint8_t output[SPACE];

static void init_engine(int mode, uint32_t channels, int binding) {
    tu_dma_init_config_boundary(
        true, channels, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, binding, 256u, BASE_CYCLES, BASE_CYCLES,
        8192u, 8192u, 8192u, 3u, 0u, 0u, false, false,
        TU_DMA_SEGMENT_LOGICAL, TU_DMA_BASE_PER_DESCRIPTOR,
        TU_DMA_PAYLOAD_ALIGN_BURST_COMMAND,
        TU_DMA_ISSUE_PAYLOAD_SERIALIZED, mode);
}

static int run_linear(int mode, tu_dma_direction_t direction,
                      uint32_t sram_address, uint64_t external_address,
                      uint64_t expected_completion,
                      uint64_t expected_occupied) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-external-boundary");
    sram.banks.bw_modeling = false;
    memset(tu_sram_raw_ptr(&sram), 0, SPACE);
    memset(output, 0, sizeof(output));
    if (direction == TU_DMA_DIR_TU_TO_HOST)
        memcpy(tu_sram_raw_ptr(&sram) + sram_address, input, 64);

    init_engine(mode, 1, TU_DMA_BIND_EXPLICIT);
    void *host = direction == TU_DMA_DIR_HOST_TO_TU ?
                 (void *)input : (void *)output;
    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        0, direction, &sram, sram_address, host, 1, 64);
    if (!desc || !tu_dma_desc_set_external_address(desc, external_address) ||
        tu_dma_submit_desc(desc) == 0) return -1;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 2000) tu_dma_tick();
    if (desc->cycles_completed != expected_completion ||
        g_tu_dma.current_cycle != expected_completion ||
        g_tu_dma.total_occupied_bytes != expected_occupied) return -2;
    const uint8_t *actual = direction == TU_DMA_DIR_HOST_TO_TU ?
                            tu_sram_raw_ptr(&sram) + sram_address : output;
    if (memcmp(actual, input, 64) != 0) return -3;

    tu_dma_destroy();
    desc->next = NULL;
    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&sram);
    return 0;
}

static int missing_metadata_rejection(void) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-external-reject");
    sram.banks.bw_modeling = false;
    init_engine(TU_DMA_BOUNDARY_EXTERNAL_4K, 1, TU_DMA_BIND_EXPLICIT);
    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, input, 1, 64);
    if (!desc || tu_dma_submit_desc(desc) != 0 ||
        g_tu_dma.channels[0].total_submitted != 0) return -1;

    desc = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, input, 1, 64);
    if (!desc) return -2;
    tu_dma_execute_desc(desc);
    if (desc->completed || memcmp(tu_sram_raw_ptr(&sram), input, 64) == 0)
        return -3;
    tu_dma_desc_destroy(desc);

    tu_dma_descriptor_t *head = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, input, 1, 16);
    tu_dma_descriptor_t *tail = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 32, input, 1, 16);
    if (!head || !tail || !tu_dma_desc_set_external_address(head, 0))
        return -4;
    tu_dma_desc_chain(head, tail);
    if (tu_dma_submit_desc(head) != 0 ||
        g_tu_dma.channels[0].total_submitted != 0) return -5;
    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

static int strided_external_gate(void) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-external-strided");
    sram.banks.bw_modeling = false;
    memset(tu_sram_raw_ptr(&sram), 0, SPACE);
    init_engine(TU_DMA_BOUNDARY_EXTERNAL_4K, 1, TU_DMA_BIND_EXPLICIT);
    tu_dma_descriptor_t *desc = tu_dma_desc_create_strided_2d(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, input, 64, 64, 1, 2, 20);
    if (!desc || !tu_dma_desc_set_external_address(desc, 4080) ||
        tu_dma_submit_desc(desc) == 0) return -1;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 2000) tu_dma_tick();
    if (desc->cycles_completed != 63 ||
        g_tu_dma.total_occupied_bytes != 96 ||
        memcmp(tu_sram_raw_ptr(&sram), input, 20) != 0 ||
        memcmp(tu_sram_raw_ptr(&sram) + 64, input + 64, 20) != 0)
        return -2;
    tu_dma_destroy();
    desc->next = NULL;
    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&sram);
    return 0;
}

static int propagation_gate(void) {
    tu_config_t cfg;
    char err[192] = {0};
    if (tu_config_load_string(
            "{\"tu\":{\"dma\":{\"burst_boundary_mode\":\"both_4k\"}}}",
            &cfg, err, sizeof(err)) != 0) return -1;
    if (cfg.dma_burst_boundary_mode !=
        TU_DMA_CONFIG_BURST_BOUNDARY_BOTH_4K) return -2;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (rt.dma_burst_boundary_mode !=
        TU_DMA_CONFIG_BURST_BOUNDARY_BOTH_4K) return -3;
    tu_init_with_config(&rt);
    return g_tu_dma.burst_boundary_mode == TU_DMA_BOUNDARY_BOTH_4K ? 0 : -4;
}

static int projection_reversal_gate(int mode, uint8_t expected_channel) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-external-projection");
    sram.banks.bw_modeling = false;
    init_engine(mode, 2, TU_DMA_BIND_EXPLICIT);
    tu_dma_descriptor_t *linear = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, input, 1, 64);
    tu_dma_descriptor_t *rows = tu_dma_desc_create_strided_2d(
        1, TU_DMA_DIR_HOST_TO_TU, &sram, 256, input, 64, 64, 1, 2, 16);
    tu_dma_descriptor_t *probe = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 512, input, 1, 16);
    if (!linear || !rows || !probe ||
        !tu_dma_desc_set_external_address(linear, 4090) ||
        !tu_dma_desc_set_external_address(rows, 0) ||
        !tu_dma_desc_set_external_address(probe, 512) ||
        tu_dma_submit_desc(linear) == 0 || tu_dma_submit_desc(rows) == 0)
        return -1;
    g_tu_dma.binding_policy = TU_DMA_BIND_LEAST_PROJECTED_CYCLES;
    g_tu_dma.next_binding_channel = 0;
    if (tu_dma_submit_desc(probe) == 0 || probe->channel != expected_channel)
        return -2;
    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

int main(void) {
    for (uint32_t i = 0; i < SPACE; i++) input[i] = (uint8_t)(i ^ 0x5au);
    if (propagation_gate() || missing_metadata_rejection()) {
        fprintf(stderr, "FAIL: propagation or fail-closed metadata gate\n");
        return 1;
    }

    struct row {
        const char *name;
        int mode;
        uint32_t sram_address;
        uint64_t external_address;
        uint64_t completion;
        uint64_t occupied;
    } rows[] = {
        {"size_only",   TU_DMA_BOUNDARY_SIZE_ONLY,    4090u, 4070u, 56u,  64u},
        {"sram_4k",     TU_DMA_BOUNDARY_SRAM_4K,      4090u, 4070u, 60u,  96u},
        {"external_4k", TU_DMA_BOUNDARY_EXTERNAL_4K,  4090u, 4070u, 60u,  96u},
        {"both_4k",     TU_DMA_BOUNDARY_BOTH_4K,      4090u, 4070u, 64u, 128u},
    };

    printf("DMA endpoint 4 KiB sweep (32-byte interface, 8192-byte maximum, 3 issue cycles, 50-cycle base)\n");
    printf("mode sram_addr external_addr completion useful occupied\n");
    for (uint32_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        if (run_linear(rows[i].mode, TU_DMA_DIR_HOST_TO_TU,
                       rows[i].sram_address, rows[i].external_address,
                       rows[i].completion, rows[i].occupied) != 0) {
            fprintf(stderr, "FAIL: load %s\n", rows[i].name);
            return 2;
        }
        printf("%-12s %9u %13lu %10lu %6u %8lu\n",
               rows[i].name, rows[i].sram_address,
               (unsigned long)rows[i].external_address,
               (unsigned long)rows[i].completion, 64u,
               (unsigned long)rows[i].occupied);
    }
    if (run_linear(TU_DMA_BOUNDARY_EXTERNAL_4K, TU_DMA_DIR_TU_TO_HOST,
                   0, 4090, 60, 96) ||
        run_linear(TU_DMA_BOUNDARY_BOTH_4K, TU_DMA_DIR_TU_TO_HOST,
                   4090, 4070, 64, 128) || strided_external_gate()) {
        fprintf(stderr, "FAIL: store or strided direction gate\n");
        return 3;
    }
    if (projection_reversal_gate(TU_DMA_BOUNDARY_SIZE_ONLY, 0) ||
        projection_reversal_gate(TU_DMA_BOUNDARY_EXTERNAL_4K, 1)) {
        fprintf(stderr, "FAIL: queued projection reversal\n");
        return 4;
    }

    printf("PASS: explicit external metadata, independent and combined endpoint boundaries, load/store bytes, occupied traffic, queued timing, config/default/rejection\n");
    return 0;
}
