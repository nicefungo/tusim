/* DMA size-only, SRAM burst-aligned, and SRAM 4 KiB boundary exploration. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SPACE 8192u
#define BASE 50u

static uint8_t input[SPACE];
static uint8_t output[SPACE];
static const uint32_t gather_index[5] = {62u, 126u, 190u, 254u, 318u};

typedef enum { LINEAR_ALIGNED, LINEAR_MISALIGNED, LINEAR_TAIL,
               PAGE_CROSS, STRIDED_2D, STRIDED_3D, SCATTER, GATHER } case_t;

static void init_engine_burst(int mode, uint32_t channels, int binding,
                              uint32_t burst_bytes) {
    tu_dma_init_config_boundary(
        true, channels, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, binding, 256u, BASE, BASE,
        burst_bytes, burst_bytes, burst_bytes, 3u, 0u, 0u, false, false,
        TU_DMA_SEGMENT_LOGICAL, TU_DMA_BASE_PER_DESCRIPTOR,
        TU_DMA_PAYLOAD_ALIGN_BURST_COMMAND,
        TU_DMA_ISSUE_PAYLOAD_SERIALIZED, mode);
}

static void init_engine(int mode, uint32_t channels, int binding) {
    init_engine_burst(mode, channels, binding, 64u);
}

static tu_dma_descriptor_t *make_desc(case_t which, tu_sram_region_t *sram,
                                      tu_dma_direction_t dir, uint8_t channel) {
    void *host = dir == TU_DMA_DIR_HOST_TO_TU ? (void *)input : (void *)output;
    switch (which) {
    case LINEAR_ALIGNED:
        return tu_dma_desc_create_linear(channel, dir, sram, 0, host, 1, 64);
    case LINEAR_MISALIGNED:
        return tu_dma_desc_create_linear(channel, dir, sram, 1, host, 1, 64);
    case LINEAR_TAIL:
        return tu_dma_desc_create_linear(channel, dir, sram, 49, host, 1, 80);
    case PAGE_CROSS:
        return tu_dma_desc_create_linear(channel, dir, sram, 4090, host, 1, 64);
    case STRIDED_2D:
        return tu_dma_desc_create_strided_2d(channel, dir, sram, 48, host,
                                             64, 64, 1, 4, 20);
    case STRIDED_3D:
        return tu_dma_desc_create_strided_3d(channel, dir, sram, 48, host,
                                             64, 256, 64, 256, 1, 2, 3, 20);
    case SCATTER:
        return tu_dma_desc_create_scatter(channel, sram, input,
                                          gather_index, 5, 4);
    case GATHER:
        return tu_dma_desc_create_gather(channel, sram, output,
                                         gather_index, 5, 4);
    }
    return NULL;
}

static int verify_bytes(case_t which, tu_dma_direction_t dir,
                        tu_sram_region_t *sram) {
    uint8_t *raw = tu_sram_raw_ptr(sram);
    uint8_t *dst = dir == TU_DMA_DIR_HOST_TO_TU ? raw : output;
    switch (which) {
    case LINEAR_ALIGNED: return memcmp(dst, input, 64) == 0 ? 0 : -1;
    case LINEAR_MISALIGNED: return memcmp(dst + (dir == TU_DMA_DIR_HOST_TO_TU), input, 64) == 0 ? 0 : -2;
    case LINEAR_TAIL: return memcmp(dst + (dir == TU_DMA_DIR_HOST_TO_TU ? 49 : 0), input, 80) == 0 ? 0 : -3;
    case PAGE_CROSS: return memcmp(dst + (dir == TU_DMA_DIR_HOST_TO_TU ? 4090 : 0), input, 64) == 0 ? 0 : -4;
    case STRIDED_2D:
        for (uint32_t r = 0; r < 4; r++)
            if (memcmp(dst + (dir == TU_DMA_DIR_HOST_TO_TU ? 48 : 0) + r * 64,
                       input + r * 64, 20) != 0) return -5;
        return 0;
    case STRIDED_3D:
        for (uint32_t d = 0; d < 2; d++)
            for (uint32_t r = 0; r < 3; r++)
                if (memcmp(dst + (dir == TU_DMA_DIR_HOST_TO_TU ? 48 : 0) + d * 256 + r * 64,
                           input + d * 256 + r * 64, 20) != 0) return -6;
        return 0;
    case SCATTER:
        for (uint32_t i = 0; i < 5; i++)
            if (memcmp(raw + gather_index[i], input + i * 4, 4) != 0) return -7;
        return 0;
    case GATHER:
        for (uint32_t i = 0; i < 5; i++)
            if (memcmp(output + i * 4, input + gather_index[i], 4) != 0) return -8;
        return 0;
    }
    return -9;
}

static int run_case(case_t which, int mode, tu_dma_direction_t dir,
                    uint64_t expected_done, uint64_t expected_occupied) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-boundary-sweep");
    sram.banks.bw_modeling = false;
    memset(tu_sram_raw_ptr(&sram), 0, SPACE);
    memset(output, 0, sizeof(output));
    if (which == GATHER)
        memcpy(tu_sram_raw_ptr(&sram), input, SPACE);
    else if (dir == TU_DMA_DIR_TU_TO_HOST) {
        uint8_t *raw = tu_sram_raw_ptr(&sram);
        if (which == LINEAR_ALIGNED) memcpy(raw, input, 64);
        else if (which == LINEAR_MISALIGNED) memcpy(raw + 1, input, 64);
        else if (which == LINEAR_TAIL) memcpy(raw + 49, input, 80);
        else if (which == PAGE_CROSS) memcpy(raw + 4090, input, 64);
        else if (which == STRIDED_2D)
            for (uint32_t r = 0; r < 4; r++)
                memcpy(raw + 48 + r * 64, input + r * 64, 20);
        else if (which == STRIDED_3D)
            for (uint32_t d = 0; d < 2; d++)
                for (uint32_t r = 0; r < 3; r++)
                    memcpy(raw + 48 + d * 256 + r * 64,
                           input + d * 256 + r * 64, 20);
    }
    init_engine(mode, 1, TU_DMA_BIND_EXPLICIT);
    tu_dma_descriptor_t *desc = make_desc(which, &sram, dir, 0);
    if (!desc || tu_dma_submit_desc(desc) == 0) return -1;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 2000) tu_dma_tick();
    if (desc->cycles_completed != expected_done ||
        g_tu_dma.current_cycle != expected_done) return -2;
    if (g_tu_dma.total_occupied_bytes != expected_occupied) return -3;
    if (verify_bytes(which, dir, &sram) != 0) return -4;
    tu_dma_destroy();
    desc->next = NULL;
    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&sram);
    return 0;
}

static int propagation_rejection_gate(void) {
    tu_config_t cfg;
    char err[192] = {0};
    if (tu_config_load_string("{\"tu\":{\"dma\":{\"burst_boundary_mode\":\"sram_4k\"}}}",
                              &cfg, err, sizeof(err)) != 0) return -1;
    if (cfg.dma_burst_boundary_mode != TU_DMA_CONFIG_BURST_BOUNDARY_SRAM_4K)
        return -2;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (rt.dma_burst_boundary_mode != TU_DMA_CONFIG_BURST_BOUNDARY_SRAM_4K)
        return -3;
    tu_init_with_config(&rt);
    if (g_tu_dma.burst_boundary_mode != TU_DMA_BOUNDARY_SRAM_4K) return -4;
    if (tu_config_load_string("{\"tu\":{\"dma\":{\"burst_boundary_mode\":\"magic\"}}}",
                              &cfg, err, sizeof(err)) == 0) return -5;
    if (!strstr(err, "burst_boundary_mode")) return -6;
    init_engine(3, 1, TU_DMA_BIND_EXPLICIT);
    return g_tu_dma.num_channels == 0 ? 0 : -7;
}

static int compatibility_gate(void) {
    tu_runtime_config_t rt;
    memset(&rt, 0, sizeof(rt));
    rt.pe_rows = rt.pe_cols = 1;
    rt.sram_w_size = rt.sram_a_size = rt.sram_o_size = SPACE;
    tu_init_with_config(&rt);
    if (g_tu_dma.burst_boundary_mode != TU_DMA_BOUNDARY_SIZE_ONLY) return -1;
    init_engine(TU_DMA_BOUNDARY_SRAM_ADDRESS, 1, TU_DMA_BIND_EXPLICIT);
    uint64_t before = g_tu_dma.estimated_cycles;
    tu_dma_load(TU_DMA_CHAN_W, &g_tu.sram_w, 1, input, 64);
    if (g_tu_dma.estimated_cycles - before != 59) return -2;
    return memcmp(tu_sram_raw_ptr(&g_tu.sram_w) + 1, input, 64) == 0 ? 0 : -3;
}

static int projection_reversal_gate(int mode, uint32_t offset, uint8_t expected) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-boundary-projection");
    sram.banks.bw_modeling = false;
    init_engine(mode, 2, TU_DMA_BIND_EXPLICIT);
    tu_dma_descriptor_t *bounded = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, offset, input, 1, 64);
    tu_dma_descriptor_t *rows = tu_dma_desc_create_strided_2d(
        1, TU_DMA_DIR_HOST_TO_TU, &sram, 256, input, 64, 64, 1, 2, 16);
    tu_dma_descriptor_t *probe = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 2048, input, 1, 16);
    if (!bounded || !rows || !probe ||
        tu_dma_submit_desc(bounded) == 0 || tu_dma_submit_desc(rows) == 0)
        return -1;
    g_tu_dma.binding_policy = TU_DMA_BIND_LEAST_PROJECTED_CYCLES;
    g_tu_dma.next_binding_channel = 0;
    if (tu_dma_submit_desc(probe) == 0 || probe->channel != expected) return -2;
    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

static int independent_4k_gate(int mode, uint64_t expected_done,
                               uint64_t expected_occupied) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, SPACE, "dma-4k-independent");
    sram.banks.bw_modeling = false;
    memset(tu_sram_raw_ptr(&sram), 0, SPACE);
    init_engine_burst(mode, 1, TU_DMA_BIND_EXPLICIT, 8192u);
    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 4090, input, 1, 64);
    if (!desc || tu_dma_submit_desc(desc) == 0) return -1;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 2000) tu_dma_tick();
    if (desc->cycles_completed != expected_done ||
        g_tu_dma.total_occupied_bytes != expected_occupied ||
        memcmp(tu_sram_raw_ptr(&sram) + 4090, input, 64) != 0) return -2;
    tu_dma_destroy();
    desc->next = NULL;
    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&sram);
    return 0;
}

int main(void) {
    for (uint32_t i = 0; i < SPACE; i++) input[i] = (uint8_t)(i ^ 0xa5u);
    if (propagation_rejection_gate() || compatibility_gate()) {
        fprintf(stderr, "FAIL: config/default/legacy/rejection gate\n");
        return 1;
    }
    static const char *names[] = {"linear_aligned", "linear_misaligned", "linear_tail",
                                  "page_cross", "strided_2d", "strided_3d", "scatter", "gather"};
    static const char *modes[] = {"size_only", "sram_address", "sram_4k"};
    static const uint64_t done[][3] = {{56,56,56},{56,60,56},{60,64,60},{56,60,60},
                                       {67,83,67},{75,99,75},{71,91,71},{71,91,71}};
    static const uint64_t occupied[][3] = {{64,64,64},{64,96,64},{96,128,96},{64,96,96},
                                           {128,256,128},{192,384,192},{160,320,160},{160,320,160}};
    static const uint32_t useful[] = {64,64,80,64,80,120,20,20};
    printf("DMA SRAM burst-boundary sweep (32-byte interface, 64-byte bursts, 3 issue cycles, 50-cycle base)\n");
    printf("case mode completion useful occupied\n");
    for (int c = 0; c < 8; c++) {
        tu_dma_direction_t dir = c == GATHER ? TU_DMA_DIR_TU_TO_HOST : TU_DMA_DIR_HOST_TO_TU;
        for (int m = 0; m < 3; m++) {
            int rc = run_case((case_t)c, m, dir, done[c][m], occupied[c][m]);
            if (rc) { fprintf(stderr, "FAIL: %s/%s rc=%d\n", names[c], modes[m], rc); return 2; }
            printf("%-18s %-12s %10lu %6u %8lu\n", names[c], modes[m],
                   (unsigned long)done[c][m], useful[c],
                   (unsigned long)occupied[c][m]);
        }
    }
    if (run_case(LINEAR_MISALIGNED, TU_DMA_BOUNDARY_SRAM_ADDRESS,
                 TU_DMA_DIR_TU_TO_HOST, 60, 96) ||
        run_case(PAGE_CROSS, TU_DMA_BOUNDARY_SRAM_4K,
                 TU_DMA_DIR_TU_TO_HOST, 60, 96)) {
        fprintf(stderr, "FAIL: store direction gate\n"); return 3;
    }
    if (projection_reversal_gate(TU_DMA_BOUNDARY_SIZE_ONLY, 1, 0) ||
        projection_reversal_gate(TU_DMA_BOUNDARY_SRAM_ADDRESS, 1, 1) ||
        projection_reversal_gate(TU_DMA_BOUNDARY_SRAM_4K, 4090, 1)) {
        fprintf(stderr, "FAIL: projected binding reversal\n"); return 4;
    }
    if (independent_4k_gate(TU_DMA_BOUNDARY_SIZE_ONLY, 56, 64) ||
        independent_4k_gate(TU_DMA_BOUNDARY_SRAM_ADDRESS, 56, 64) ||
        independent_4k_gate(TU_DMA_BOUNDARY_SRAM_4K, 60, 96)) {
        fprintf(stderr, "FAIL: independent 4 KiB boundary gate\n"); return 5;
    }
    printf("PASS: all three modes, aligned/misaligned/tail/4KiB crossing, independent 8KiB burst gate, 2D/3D/S-G, load/store, occupied bytes, live/queued timing, config/default/rejection\n");
    return 0;
}
