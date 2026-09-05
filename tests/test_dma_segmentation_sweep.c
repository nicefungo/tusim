/* DMA aggregate-vs-logical burst segmentation exploration and exact gates. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SPACE 512u
#define ISSUE 3u

typedef enum {
    CASE_LINEAR,
    CASE_STRIDED_2D,
    CASE_STRIDED_3D,
    CASE_GATHER
} case_t;

static uint8_t input[SPACE];
static uint8_t output[SPACE];
static const uint32_t gather_index[5] = {0u, 8u, 16u, 24u, 32u};

static int propagation_gate(void) {
    tu_config_t cfg;
    char err[192] = {0};
    if (tu_config_load_string(
            "{\"tu\":{\"dma\":{\"burst_segmentation\":\"logical_segments\"}}}",
            &cfg, err, sizeof(err)) != 0) return -1;
    if (cfg.dma_burst_segmentation != TU_DMA_CONFIG_SEGMENT_LOGICAL) return -2;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (rt.dma_burst_segmentation != TU_DMA_CONFIG_SEGMENT_LOGICAL) return -3;
    tu_init_with_config(&rt);
    if (g_tu_dma.burst_segmentation != TU_DMA_SEGMENT_LOGICAL) return -4;

    if (tu_config_load_string(
            "{\"tu\":{\"dma\":{\"burst_segmentation\":\"packet_magic\"}}}",
            &cfg, err, sizeof(err)) == 0) return -5;
    return strstr(err, "burst_segmentation") ? 0 : -6;
}

static tu_dma_descriptor_t *make_desc(case_t which, tu_sram_region_t *sram) {
    switch (which) {
    case CASE_LINEAR:
        return tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU,
                                         sram, 0, input, 1, 80);
    case CASE_STRIDED_2D:
        return tu_dma_desc_create_strided_2d(0, TU_DMA_DIR_HOST_TO_TU,
                                             sram, 0, input,
                                             32, 32, 1, 4, 20);
    case CASE_STRIDED_3D:
        return tu_dma_desc_create_strided_3d(0, TU_DMA_DIR_HOST_TO_TU,
                                             sram, 0, input,
                                             32, 128, 32, 128,
                                             1, 2, 3, 20);
    case CASE_GATHER:
        memcpy(tu_sram_raw_ptr(sram), input, SPACE);
        return tu_dma_desc_create_gather(0, sram, output, gather_index, 5, 4);
    }
    return NULL;
}

static int verify_bytes(case_t which, const tu_sram_region_t *sram) {
    const uint8_t *raw = tu_sram_raw_ptr((tu_sram_region_t *)sram);
    switch (which) {
    case CASE_LINEAR:
        return memcmp(raw, input, 80) == 0 ? 0 : -1;
    case CASE_STRIDED_2D:
        for (uint32_t r = 0; r < 4; r++)
            if (memcmp(raw + r * 32, input + r * 32, 20) != 0) return -2;
        return 0;
    case CASE_STRIDED_3D:
        for (uint32_t d = 0; d < 2; d++)
            for (uint32_t r = 0; r < 3; r++)
                if (memcmp(raw + d * 128 + r * 32,
                           input + d * 128 + r * 32, 20) != 0) return -3;
        return 0;
    case CASE_GATHER:
        for (uint32_t i = 0; i < 5; i++)
            if (memcmp(output + i * 4, input + gather_index[i], 4) != 0) return -4;
        return 0;
    }
    return -5;
}

static int run_case(case_t which, int mode, uint64_t expected_bursts,
                    uint64_t *completion_out) {
    static const uint32_t bytes[] = {80u, 80u, 120u, 20u};
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-segmentation-sweep");
    sram.banks.bw_modeling = false;
    memset(tu_sram_raw_ptr(&sram), 0, SPACE);
    memset(output, 0, sizeof(output));

    tu_dma_init_config_segmentation(
        true, 1, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 64u, 64u, 64u, ISSUE,
        0u, 0u, false, false, mode);
    if (g_tu_dma.num_channels != 1 ||
        g_tu_dma.burst_segmentation != (tu_dma_burst_segmentation_t)mode)
        return -1;

    tu_dma_descriptor_t *desc = make_desc(which, &sram);
    if (!desc || tu_dma_submit_desc(desc) == 0) return -2;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();

    uint64_t payload = bytes[which] / 32u + (bytes[which] % 32u != 0);
    uint64_t expected = 1u + 50u + payload + expected_bursts * ISSUE;
    *completion_out = desc->cycles_completed;
    if (*completion_out != expected || g_tu_dma.current_cycle != expected)
        return -3;
    if (verify_bytes(which, &sram) != 0) return -4;

    tu_dma_destroy();
    desc->next = NULL;
    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&sram);
    return 0;
}

static int projected_binding_gate(int mode, uint8_t expected_channel) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-segmentation-projection");
    sram.banks.bw_modeling = false;
    tu_dma_init_config_segmentation(
        true, 2, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 64u, 64u, 64u, ISSUE,
        0u, 0u, false, false, mode);

    tu_dma_descriptor_t *segmented = tu_dma_desc_create_strided_2d(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, input, 32, 32, 1, 4, 20);
    tu_dma_descriptor_t *linear = tu_dma_desc_create_linear(
        1, TU_DMA_DIR_HOST_TO_TU, &sram, 128, input, 1, 100);
    tu_dma_descriptor_t *probe = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 256, input, 1, 32);
    if (!segmented || !linear || !probe ||
        tu_dma_submit_desc(segmented) == 0 || tu_dma_submit_desc(linear) == 0)
        return -1;
    g_tu_dma.binding_policy = TU_DMA_BIND_LEAST_PROJECTED_CYCLES;
    g_tu_dma.next_binding_channel = 0;
    if (tu_dma_submit_desc(probe) == 0 || probe->channel != expected_channel)
        return -2;

    /* Pending descriptors are owned and freed by engine teardown. */
    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

static int zero_init_compat_gate(void) {
    tu_runtime_config_t rt;
    memset(&rt, 0, sizeof(rt));
    rt.pe_rows = 1;
    rt.pe_cols = 1;
    rt.sram_w_size = 4096;
    rt.sram_a_size = 4096;
    rt.sram_o_size = 4096;
    tu_init_with_config(&rt);
    return g_tu_dma.burst_segmentation == TU_DMA_SEGMENT_AGGREGATE ? 0 : -1;
}

static int rejection_gate(void) {
    tu_dma_init_config_segmentation(
        true, 1, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 64u, 64u, 64u, ISSUE,
        0u, 0u, false, false, 2);
    return g_tu_dma.num_channels == 0 ? 0 : -1;
}

int main(void) {
    for (uint32_t i = 0; i < SPACE; i++) input[i] = (uint8_t)(i ^ 0x5au);
    if (propagation_gate() != 0) {
        fprintf(stderr, "FAIL: config propagation/rejection\n");
        return 1;
    }

    static const char *names[] = {"linear", "strided_2d", "strided_3d", "gather"};
    static const uint64_t aggregate_bursts[] = {2u, 2u, 2u, 1u};
    static const uint64_t logical_bursts[] = {2u, 4u, 6u, 5u};
    printf("DMA burst segmentation sweep (64 B bursts, 3 issue cycles, 256-bit path, 50-cycle base)\n");
    printf("case aggregate_bursts logical_bursts aggregate_done logical_done\n");
    for (int i = 0; i < 4; i++) {
        uint64_t aggregate_done = 0, logical_done = 0;
        int rc = run_case((case_t)i, TU_DMA_SEGMENT_AGGREGATE,
                          aggregate_bursts[i], &aggregate_done);
        if (rc == 0)
            rc = run_case((case_t)i, TU_DMA_SEGMENT_LOGICAL,
                          logical_bursts[i], &logical_done);
        if (rc != 0) {
            fprintf(stderr, "FAIL: %s rc=%d\n", names[i], rc);
            return 2;
        }
        printf("%-12s %16lu %14lu %14lu %12lu\n", names[i],
               (unsigned long)aggregate_bursts[i],
               (unsigned long)logical_bursts[i],
               (unsigned long)aggregate_done,
               (unsigned long)logical_done);
    }

    if (projected_binding_gate(TU_DMA_SEGMENT_AGGREGATE, 0) != 0 ||
        projected_binding_gate(TU_DMA_SEGMENT_LOGICAL, 1) != 0) {
        fprintf(stderr, "FAIL: queued projected-cycle segmentation\n");
        return 3;
    }
    if (zero_init_compat_gate() != 0 || rejection_gate() != 0) {
        fprintf(stderr, "FAIL: compatibility or invalid-mode gate\n");
        return 4;
    }
    printf("PASS: aggregate/logical modes, linear/2D/3D/gather bytes, live/queued cycles, config/default/rejection\n");
    return 0;
}
