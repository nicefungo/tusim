/* DMA descriptor-vs-logical-segment base-latency exploration and gates. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SPACE 1024u
#define ISSUE 3u
#define BASE 50u

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
            "{\"tu\":{\"dma\":{\"base_latency_scope\":\"logical_segments\"}}}",
            &cfg, err, sizeof(err)) != 0) return -1;
    if (cfg.dma_base_latency_scope !=
        TU_DMA_CONFIG_BASE_PER_LOGICAL_SEGMENT) return -2;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (rt.dma_base_latency_scope !=
        TU_DMA_CONFIG_BASE_PER_LOGICAL_SEGMENT) return -3;
    tu_init_with_config(&rt);
    if (g_tu_dma.base_latency_scope !=
        TU_DMA_BASE_PER_LOGICAL_SEGMENT) return -4;

    if (tu_config_load_string(
            "{\"tu\":{\"dma\":{\"base_latency_scope\":\"packet_magic\"}}}",
            &cfg, err, sizeof(err)) == 0) return -5;
    return strstr(err, "base_latency_scope") ? 0 : -6;
}

static tu_dma_descriptor_t *make_desc(case_t which, tu_sram_region_t *sram,
                                      tu_dma_direction_t direction) {
    switch (which) {
    case CASE_LINEAR:
        return tu_dma_desc_create_linear(0, direction, sram, 0,
                                         direction == TU_DMA_DIR_HOST_TO_TU ?
                                             (void *)input : (void *)output,
                                         1, 80);
    case CASE_STRIDED_2D:
        return tu_dma_desc_create_strided_2d(
            0, direction, sram, 0,
            direction == TU_DMA_DIR_HOST_TO_TU ? (void *)input : (void *)output,
            32, 32, 1, 4, 20);
    case CASE_STRIDED_3D:
        return tu_dma_desc_create_strided_3d(
            0, direction, sram, 0,
            direction == TU_DMA_DIR_HOST_TO_TU ? (void *)input : (void *)output,
            32, 128, 32, 128, 1, 2, 3, 20);
    case CASE_GATHER:
        memcpy(tu_sram_raw_ptr(sram), input, SPACE);
        return tu_dma_desc_create_gather(0, sram, output,
                                         gather_index, 5, 4);
    }
    return NULL;
}

static int verify_bytes(case_t which, tu_dma_direction_t direction,
                        const tu_sram_region_t *sram) {
    const uint8_t *raw = tu_sram_raw_ptr((tu_sram_region_t *)sram);
    const uint8_t *dst = direction == TU_DMA_DIR_HOST_TO_TU ? raw : output;
    const uint8_t *src = input;
    switch (which) {
    case CASE_LINEAR:
        return memcmp(dst, src, 80) == 0 ? 0 : -1;
    case CASE_STRIDED_2D:
        for (uint32_t r = 0; r < 4; r++)
            if (memcmp(dst + r * 32, src + r * 32, 20) != 0) return -2;
        return 0;
    case CASE_STRIDED_3D:
        for (uint32_t d = 0; d < 2; d++)
            for (uint32_t r = 0; r < 3; r++)
                if (memcmp(dst + d * 128 + r * 32,
                           src + d * 128 + r * 32, 20) != 0) return -3;
        return 0;
    case CASE_GATHER:
        for (uint32_t i = 0; i < 5; i++)
            if (memcmp(output + i * 4, input + gather_index[i], 4) != 0)
                return -4;
        return 0;
    }
    return -5;
}

static int run_case(case_t which, int scope, tu_dma_direction_t direction,
                    uint64_t expected_base_count, uint64_t expected_completion,
                    uint64_t *completion_out) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-base-scope-sweep");
    sram.banks.bw_modeling = false;
    memset(tu_sram_raw_ptr(&sram), 0, SPACE);
    memset(output, 0, sizeof(output));
    if (direction == TU_DMA_DIR_TU_TO_HOST)
        memcpy(tu_sram_raw_ptr(&sram), input, SPACE);

    tu_dma_init_config_base_scope(
        true, 1, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, BASE, BASE, 64u, 64u, 64u, ISSUE,
        0u, 0u, false, false, TU_DMA_SEGMENT_LOGICAL, scope);
    if (g_tu_dma.num_channels != 1 ||
        g_tu_dma.base_latency_scope != (tu_dma_base_latency_scope_t)scope)
        return -1;

    tu_dma_descriptor_t *desc = make_desc(which, &sram, direction);
    if (!desc || tu_dma_submit_desc(desc) == 0) return -2;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 2000u)
        tu_dma_tick();

    *completion_out = desc->cycles_completed;
    if (*completion_out != expected_completion ||
        g_tu_dma.current_cycle != expected_completion) return -3;
    if (verify_bytes(which, direction, &sram) != 0) return -4;
    if (scope == TU_DMA_BASE_PER_LOGICAL_SEGMENT &&
        expected_base_count > 1 &&
        *completion_out <= BASE * expected_base_count) return -5;

    tu_dma_destroy();
    desc->next = NULL;
    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&sram);
    return 0;
}

static int projected_binding_gate(int scope, uint8_t expected_channel) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-base-scope-projection");
    sram.banks.bw_modeling = false;
    tu_dma_init_config_base_scope(
        true, 2, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, BASE, BASE, 64u, 64u, 64u, ISSUE,
        0u, 0u, false, false, TU_DMA_SEGMENT_LOGICAL, scope);

    tu_dma_descriptor_t *segmented = tu_dma_desc_create_strided_2d(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, input, 32, 32, 1, 4, 20);
    tu_dma_descriptor_t *linear = tu_dma_desc_create_linear(
        1, TU_DMA_DIR_HOST_TO_TU, &sram, 256, input, 1, 512);
    tu_dma_descriptor_t *probe = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 800, input, 1, 16);
    if (!segmented || !linear || !probe ||
        tu_dma_submit_desc(segmented) == 0 || tu_dma_submit_desc(linear) == 0)
        return -1;
    g_tu_dma.binding_policy = TU_DMA_BIND_LEAST_PROJECTED_CYCLES;
    g_tu_dma.next_binding_channel = 0;
    if (tu_dma_submit_desc(probe) == 0 || probe->channel != expected_channel)
        return -2;

    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

static int compatibility_and_rejection_gate(void) {
    tu_runtime_config_t rt;
    memset(&rt, 0, sizeof(rt));
    rt.pe_rows = 1;
    rt.pe_cols = 1;
    rt.sram_w_size = 4096;
    rt.sram_a_size = 4096;
    rt.sram_o_size = 4096;
    tu_init_with_config(&rt);
    if (g_tu_dma.base_latency_scope != TU_DMA_BASE_PER_DESCRIPTOR) return -1;

    tu_dma_descriptor_t *zero = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &g_tu.sram_w, 0, input, 1, 0);
    if (!zero || tu_dma_submit_desc(zero) == 0 || !zero->completed ||
        zero->cycles_completed != BASE) return -2;
    tu_dma_desc_destroy(zero);

    tu_dma_init_config_base_scope(
        true, 1, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, BASE, BASE, 64u, 64u, 64u, ISSUE,
        0u, 0u, false, false, TU_DMA_SEGMENT_LOGICAL, 2);
    return g_tu_dma.num_channels == 0 ? 0 : -3;
}

int main(void) {
    for (uint32_t i = 0; i < SPACE; i++) input[i] = (uint8_t)(i ^ 0xa5u);
    if (propagation_gate() != 0) {
        fprintf(stderr, "FAIL: config propagation/rejection\n");
        return 1;
    }

    static const char *names[] = {"linear", "strided_2d", "strided_3d", "gather"};
    static const uint64_t segment_counts[] = {1u, 4u, 6u, 5u};
    static const uint64_t descriptor_done[] = {60u, 66u, 73u, 67u};
    static const uint64_t segment_done[] = {60u, 216u, 323u, 267u};
    printf("DMA base-latency scope sweep (logical burst commands, 64 B bursts, 3 issue cycles, 256-bit path, 50-cycle base)\n");
    printf("case segments descriptor_done segment_done\n");
    for (int i = 0; i < 4; i++) {
        uint64_t descriptor = 0, segment = 0;
        int rc = run_case((case_t)i, TU_DMA_BASE_PER_DESCRIPTOR,
                          i == CASE_GATHER ? TU_DMA_DIR_TU_TO_HOST :
                                             TU_DMA_DIR_HOST_TO_TU,
                          1u, descriptor_done[i], &descriptor);
        if (rc == 0)
            rc = run_case((case_t)i, TU_DMA_BASE_PER_LOGICAL_SEGMENT,
                          i == CASE_GATHER ? TU_DMA_DIR_TU_TO_HOST :
                                             TU_DMA_DIR_HOST_TO_TU,
                          segment_counts[i], segment_done[i], &segment);
        if (rc != 0) {
            fprintf(stderr, "FAIL: %s rc=%d\n", names[i], rc);
            return 2;
        }
        printf("%-12s %8lu %15lu %12lu\n", names[i],
               (unsigned long)segment_counts[i],
               (unsigned long)descriptor, (unsigned long)segment);
    }

    /* Store-side strided path uses write latency and the same segment scope. */
    uint64_t store_done = 0;
    if (run_case(CASE_STRIDED_2D, TU_DMA_BASE_PER_LOGICAL_SEGMENT,
                 TU_DMA_DIR_TU_TO_HOST, 4u, 216u, &store_done) != 0) {
        fprintf(stderr, "FAIL: store-side logical base scope\n");
        return 3;
    }
    if (projected_binding_gate(TU_DMA_BASE_PER_DESCRIPTOR, 0) != 0 ||
        projected_binding_gate(TU_DMA_BASE_PER_LOGICAL_SEGMENT, 1) != 0) {
        fprintf(stderr, "FAIL: queued projected-cycle base scope\n");
        return 4;
    }
    if (compatibility_and_rejection_gate() != 0) {
        fprintf(stderr, "FAIL: compatibility or invalid-mode gate\n");
        return 5;
    }
    printf("PASS: descriptor/logical base scopes, linear/2D/3D/gather, read/write bytes, live/queued cycles, config/default/rejection\n");
    return 0;
}
