/* DMA directional maximum-burst propagation and cycle sweep. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BYTES 96u
static uint8_t input[BYTES];
static uint8_t output[BYTES];

static int propagation_gate(void) {
    static const char json[] =
        "{\"tu\":{\"dma\":{\"max_burst_bytes\":64,"
        "\"read_max_burst_bytes\":128,"
        "\"write_max_burst_bytes\":32,"
        "\"burst_issue_cycles\":2}}}";
    tu_config_t cfg;
    char err[192] = {0};
    if (tu_config_load_string(json, &cfg, err, sizeof(err)) != 0) return -1;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (rt.dma_read_max_burst_bytes != 128u ||
        rt.dma_write_max_burst_bytes != 32u) return -2;
    tu_init_with_config(&rt);
    if (g_tu_dma.read_max_burst_bytes != 128u ||
        g_tu_dma.write_max_burst_bytes != 32u) return -3;

    if (tu_config_load_string(
            "{\"tu\":{\"dma\":{\"max_burst_bytes\":256}}}",
            &cfg, err, sizeof(err)) != 0) return -4;
    rt = tu_config_to_runtime(&cfg);
    tu_init_with_config(&rt);
    if (g_tu_dma.max_burst_bytes != 256u ||
        g_tu_dma.read_max_burst_bytes != 256u ||
        g_tu_dma.write_max_burst_bytes != 256u) return -5;
    return 0;
}

static int run_direction(tu_dma_direction_t direction,
                         uint32_t read_burst, uint32_t write_burst,
                         uint64_t *done_out, uint64_t *bursts_out) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, BYTES, "dma-directional-burst-sweep");
    sram.banks.bw_modeling = false;
    memset(output, 0, sizeof(output));
    if (direction == TU_DMA_DIR_TU_TO_HOST)
        memcpy(tu_sram_raw_ptr(&sram), input, BYTES);

    tu_dma_init_config_directional_burst(
        true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 64u, read_burst, write_burst, 2u);
    if (g_tu_dma.read_max_burst_bytes != read_burst ||
        g_tu_dma.write_max_burst_bytes != write_burst) return -1;

    void *host = direction == TU_DMA_DIR_HOST_TO_TU ? (void *)input : output;
    tu_dma_descriptor_t *d = tu_dma_desc_create_linear(
        0, direction, &sram, 0, host, 1, BYTES);
    if (!d || tu_dma_submit_desc(d) == 0) return -2;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();

    uint32_t selected = direction == TU_DMA_DIR_TU_TO_HOST ?
                        write_burst : read_burst;
    uint64_t bursts = (BYTES + selected - 1u) / selected;
    uint64_t expected = 1u + 50u + (BYTES + 31u) / 32u + bursts * 2u;
    *done_out = d->cycles_completed;
    *bursts_out = bursts;
    if (*done_out != expected || g_tu_dma.current_cycle != expected) return -3;
    if (direction == TU_DMA_DIR_HOST_TO_TU) {
        if (memcmp(tu_sram_raw_ptr(&sram), input, BYTES) != 0) return -4;
    } else if (memcmp(output, input, BYTES) != 0) return -5;

    tu_dma_destroy();
    d->next = NULL;
    tu_dma_desc_destroy(d);
    tu_sram_destroy(&sram);
    return 0;
}

static int legacy_gate(void) {
    tu_sram_region_t sram;
    uint8_t roundtrip[BYTES] = {0};
    tu_sram_init(&sram, BYTES, "dma-directional-burst-legacy");
    sram.banks.bw_modeling = false;
    tu_dma_init_config_directional_burst(
        false, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 64u, 128u, 32u, 2u);
    tu_dma_load(TU_DMA_CHAN_W, &sram, 0, input, BYTES);
    if (g_tu_dma.estimated_cycles != 55u) return -1;
    tu_dma_store(TU_DMA_CHAN_O, &sram, 0, roundtrip, BYTES);
    if (g_tu_dma.estimated_cycles != 114u ||
        memcmp(roundtrip, input, BYTES) != 0) return -2;
    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

static int rejection_gate(void) {
    tu_dma_init_config_directional_burst(
        true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 64u, 96u, 32u, 1u);
    return g_tu_dma.num_channels == 0 &&
           g_tu_dma.read_max_burst_bytes == 0 ? 0 : -1;
}

int main(void) {
    for (uint32_t i = 0; i < BYTES; i++) input[i] = (uint8_t)(i ^ 0x5au);
    if (propagation_gate() != 0) {
        fprintf(stderr, "FAIL: directional burst config propagation\n");
        return 1;
    }

    const uint32_t reads[] = {64u, 128u, 32u};
    const uint32_t writes[] = {64u, 32u, 128u};
    printf("DMA directional burst sweep (96 B, 256-bit path, 50-cycle base, 2 issue cycles)\n");
    printf("read_burst write_burst load_bursts load_done store_bursts store_done\n");
    for (uint32_t i = 0; i < 3; i++) {
        uint64_t load_done = 0, load_bursts = 0;
        uint64_t store_done = 0, store_bursts = 0;
        int rc = run_direction(TU_DMA_DIR_HOST_TO_TU, reads[i], writes[i],
                               &load_done, &load_bursts);
        if (rc == 0)
            rc = run_direction(TU_DMA_DIR_TU_TO_HOST, reads[i], writes[i],
                               &store_done, &store_bursts);
        if (rc != 0) {
            fprintf(stderr, "FAIL: read=%u write=%u rc=%d\n",
                    reads[i], writes[i], rc);
            return 2;
        }
        printf("%10u %11u %11lu %9lu %12lu %10lu\n",
               reads[i], writes[i],
               (unsigned long)load_bursts, (unsigned long)load_done,
               (unsigned long)store_bursts, (unsigned long)store_done);
    }
    if (legacy_gate() != 0) {
        fprintf(stderr, "FAIL: directional legacy accounting\n");
        return 3;
    }
    if (rejection_gate() != 0) {
        fprintf(stderr, "FAIL: invalid directional burst accepted\n");
        return 4;
    }
    printf("PASS: propagation, inheritance, both directions, exact cycles/bytes, legacy, rejection\n");
    return 0;
}
