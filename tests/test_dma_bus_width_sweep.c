/* Runtime DMA bus-width propagation and cycle sweep. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BYTES 4096u
static uint8_t src[BYTES];

static int propagation_gate(void) {
    static const char json[] =
        "{\"tu\":{\"dma\":{\"bus_width_bits\":512}}}";
    tu_config_t cfg;
    char err[160] = {0};
    tu_config_default(&cfg);
    if (tu_config_load_string(json, &cfg, err, sizeof(err)) != 0) return -1;
    if (tu_config_validate(&cfg, err, sizeof(err)) != 0) return -2;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (rt.dma_bus_width_bits != 512u) return -3;
    tu_init_with_config(&rt);
    if (g_tu_dma.bus_width_bytes != 64u) return -4;

    rt.dma_bus_width_bits = 0; /* zero-initialized caller compatibility */
    tu_init_with_config(&rt);
    if (g_tu_dma.bus_width_bytes != TU_DMA_BUS_WIDTH_BYTES) return -5;
    return 0;
}

static int run_width(uint32_t bits, uint64_t *done_out) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, BYTES, "dma-width-sweep");
    sram.banks.bw_modeling = false;
    memset(src, (int)(bits / 8u), sizeof(src));

    tu_dma_init_config_arch(true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
                            TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
                            bits);
    if (g_tu_dma.num_channels != 1 ||
        g_tu_dma.bus_width_bytes != bits / 8u) return -1;
    tu_dma_descriptor_t *d = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, src, 1, BYTES);
    if (!d || tu_dma_submit_desc(d) == 0) return -2;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();

    const uint64_t expected = 1u + TU_LATENCY_DRAM_READ +
                              (BYTES + bits / 8u - 1u) / (bits / 8u);
    *done_out = d->cycles_completed;
    if (*done_out != expected || g_tu_dma.current_cycle != expected) return -3;
    uint8_t *raw = (uint8_t *)tu_sram_raw_ptr(&sram);
    for (uint32_t i = 0; i < BYTES; i++)
        if (raw[i] != src[i]) return -4;

    tu_dma_destroy();
    d->next = NULL;
    tu_dma_desc_destroy(d);
    tu_sram_destroy(&sram);
    return 0;
}

static int rejection_gate(void) {
    tu_dma_init_config_arch(true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
                            TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
                            100u);
    return g_tu_dma.num_channels == 0 && g_tu_dma.bus_width_bytes == 0 ? 0 : -1;
}

int main(void) {
    if (propagation_gate() != 0) {
        fprintf(stderr, "FAIL: config/runtime propagation\n");
        return 1;
    }
    const uint32_t widths[] = {128u, 256u, 512u};
    printf("DMA runtime bus-width sweep (4096 B, SRAM meter disabled)\n");
    printf("width_bits width_bytes completion_cycle payload_cycles\n");
    for (uint32_t i = 0; i < 3; i++) {
        uint64_t done = 0;
        int rc = run_width(widths[i], &done);
        if (rc != 0) {
            fprintf(stderr, "FAIL: width=%u rc=%d\n", widths[i], rc);
            return 2;
        }
        printf("%10u %11u %16lu %14lu\n", widths[i], widths[i] / 8u,
               (unsigned long)done,
               (unsigned long)(done - 1u - TU_LATENCY_DRAM_READ));
    }
    if (rejection_gate() != 0) {
        fprintf(stderr, "FAIL: invalid width did not fail closed\n");
        return 3;
    }
    printf("PASS: parse-to-live propagation, defaults, exact cycles, bytes, rejection\n");
    return 0;
}
