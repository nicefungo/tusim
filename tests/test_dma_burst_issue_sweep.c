/* DMA maximum-burst and per-burst issue-cost propagation/cycle sweep. */
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
        "{\"tu\":{\"dma\":{\"bus_width_bits\":256,"
        "\"max_burst_bytes\":128,\"burst_issue_cycles\":2}}}";
    tu_config_t cfg;
    char err[160] = {0};
    if (tu_config_load_string(json, &cfg, err, sizeof(err)) != 0) return -1;
    if (tu_config_validate(&cfg, err, sizeof(err)) != 0) return -2;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (rt.dma_max_burst_bytes != 128u || rt.dma_burst_issue_cycles != 2u)
        return -3;
    tu_init_with_config(&rt);
    if (g_tu_dma.max_burst_bytes != 128u || g_tu_dma.burst_issue_cycles != 2u)
        return -4;

    memset(&rt, 0, sizeof(rt));
    rt.pe_rows = 1; rt.pe_cols = 1;
    rt.sram_w_size = 4096; rt.sram_a_size = 4096; rt.sram_o_size = 4096;
    tu_init_with_config(&rt);
    if (g_tu_dma.max_burst_bytes != TU_DMA_MAX_BURST_BYTES ||
        g_tu_dma.burst_issue_cycles != 0u)
        return -5;
    return 0;
}

static int run_case(uint32_t burst_bytes, uint32_t issue_cycles,
                    uint64_t *done_out, uint64_t *bursts_out) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, BYTES, "dma-burst-issue-sweep");
    sram.banks.bw_modeling = false;
    memset(src, (int)burst_bytes, sizeof(src));

    tu_dma_init_config_burst(true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
                             TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
                             256u, 50u, 50u, burst_bytes, issue_cycles);
    if (g_tu_dma.num_channels != 1 ||
        g_tu_dma.max_burst_bytes != burst_bytes ||
        g_tu_dma.burst_issue_cycles != issue_cycles)
        return -1;
    tu_dma_descriptor_t *d = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, src, 1, BYTES);
    if (!d || tu_dma_submit_desc(d) == 0) return -2;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 2000u)
        tu_dma_tick();

    const uint64_t bursts = (BYTES + burst_bytes - 1u) / burst_bytes;
    const uint64_t expected = 1u + 50u + BYTES / 32u +
                              bursts * issue_cycles;
    *done_out = d->cycles_completed;
    *bursts_out = bursts;
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

static int legacy_gate(void) {
    uint8_t input[96], output[96] = {0};
    tu_sram_region_t sram;
    for (uint32_t i = 0; i < sizeof(input); i++) input[i] = (uint8_t)i;
    tu_sram_init(&sram, sizeof(input), "dma-burst-legacy");
    sram.banks.bw_modeling = false;
    tu_dma_init_config_burst(false, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
                             TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
                             256u, 50u, 50u, 64u, 2u);
    tu_dma_load(TU_DMA_CHAN_W, &sram, 0, input, sizeof(input));
    if (g_tu_dma.estimated_cycles != 57u) return -1;
    tu_dma_store(TU_DMA_CHAN_O, &sram, 0, output, sizeof(output));
    if (g_tu_dma.estimated_cycles != 114u ||
        memcmp(input, output, sizeof(input)) != 0) return -2;
    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

static int rejection_gate(void) {
    tu_dma_init_config_burst(true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
                             TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
                             256u, 50u, 50u, 96u, 1u);
    if (g_tu_dma.num_channels != 0 || g_tu_dma.max_burst_bytes != 0) return -1;
    tu_dma_init_config_burst(true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
                             TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
                             256u, 50u, 50u, 64u, 1025u);
    return g_tu_dma.num_channels == 0 && g_tu_dma.burst_issue_cycles == 0 ? 0 : -2;
}

int main(void) {
    if (propagation_gate() != 0) {
        fprintf(stderr, "FAIL: config/runtime burst propagation\n");
        return 1;
    }
    const uint32_t burst_bytes[] = {64u, 32u, 64u, 128u};
    const uint32_t issue_cycles[] = {0u, 1u, 1u, 1u};
    printf("DMA burst issue sweep (4096 B, 256-bit path, 50-cycle base)\n");
    printf("burst_bytes issue_cycles burst_count completion_cycle\n");
    for (uint32_t i = 0; i < 4; i++) {
        uint64_t done = 0, bursts = 0;
        int rc = run_case(burst_bytes[i], issue_cycles[i], &done, &bursts);
        if (rc != 0) {
            fprintf(stderr, "FAIL: burst=%u issue=%u rc=%d\n",
                    burst_bytes[i], issue_cycles[i], rc);
            return 2;
        }
        printf("%11u %12u %11lu %16lu\n",
               burst_bytes[i], issue_cycles[i],
               (unsigned long)bursts, (unsigned long)done);
    }
    if (legacy_gate() != 0) {
        fprintf(stderr, "FAIL: legacy load/store burst accounting\n");
        return 3;
    }
    if (rejection_gate() != 0) {
        fprintf(stderr, "FAIL: invalid burst configuration did not fail closed\n");
        return 4;
    }
    printf("PASS: parse-to-live propagation, compatibility, exact cycles, bytes, rejection\n");
    return 0;
}
