/* DMA direction-specific burst-issue propagation and cycle sweep. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BYTES 96u
static uint8_t input[BYTES];
static uint8_t output[BYTES];

static int propagation_gate(void) {
    static const char json[] =
        "{\"tu\":{\"dma\":{\"max_burst_bytes\":32,"
        "\"burst_issue_cycles\":2,"
        "\"read_burst_issue_cycles\":0,"
        "\"write_burst_issue_cycles\":4}}}";
    tu_config_t cfg;
    char err[192] = {0};
    if (tu_config_load_string(json, &cfg, err, sizeof(err)) != 0) return -1;
    if (!cfg.dma_read_burst_issue_configured ||
        !cfg.dma_write_burst_issue_configured) return -2;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (!rt.dma_read_burst_issue_configured ||
        rt.dma_read_burst_issue_cycles != 0u ||
        rt.dma_write_burst_issue_cycles != 4u) return -3;
    tu_init_with_config(&rt);
    if (g_tu_dma.burst_issue_cycles != 2u ||
        g_tu_dma.read_burst_issue_cycles != 0u ||
        g_tu_dma.write_burst_issue_cycles != 4u) return -4;
    g_tu.estimated_cycles = 0;
    tu_dma_load_o(input, 0, BYTES);
    if (g_tu.estimated_cycles != 3u ||
        memcmp(tu_sram_raw_ptr(&g_tu.sram_o), input, BYTES) != 0) return -5;

    if (tu_config_load_string(
            "{\"tu\":{\"dma\":{\"burst_issue_cycles\":3}}}",
            &cfg, err, sizeof(err)) != 0) return -6;
    rt = tu_config_to_runtime(&cfg);
    tu_init_with_config(&rt);
    if (g_tu_dma.read_burst_issue_cycles != 3u ||
        g_tu_dma.write_burst_issue_cycles != 3u) return -7;
    return 0;
}

static int projected_binding_gate(void) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, BYTES * 3u, "dma-directional-issue-projected");
    sram.banks.bw_modeling = false;
    memcpy(tu_sram_raw_ptr(&sram), input, BYTES);
    tu_dma_init_config_directional_issue(
        true, 2, 4, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 32u, 32u, 32u, 2u,
        0u, 4u, true, true);
    tu_dma_descriptor_t *store = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_TU_TO_HOST, &sram, 0, output, 1, BYTES);
    tu_dma_descriptor_t *load = tu_dma_desc_create_linear(
        1, TU_DMA_DIR_HOST_TO_TU, &sram, BYTES, input, 1, BYTES);
    tu_dma_descriptor_t *probe = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, BYTES * 2u, input, 1, BYTES);
    if (!store || !load || !probe ||
        tu_dma_submit_desc(store) == 0 || tu_dma_submit_desc(load) == 0)
        return -1;
    g_tu_dma.binding_policy = TU_DMA_BIND_LEAST_PROJECTED_CYCLES;
    g_tu_dma.next_binding_channel = 0;
    if (tu_dma_submit_desc(probe) == 0 || probe->channel != 1u) return -2;
    while ((g_tu_dma.channels[0].total_completed < 1u ||
            g_tu_dma.channels[1].total_completed < 2u) &&
           g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();
    if (g_tu_dma.channels[0].total_completed != 1u ||
        g_tu_dma.channels[1].total_completed != 2u) return -3;
    tu_dma_destroy();
    store->next = NULL; load->next = NULL; probe->next = NULL;
    tu_dma_desc_destroy(store);
    tu_dma_desc_destroy(load);
    tu_dma_desc_destroy(probe);
    tu_sram_destroy(&sram);
    return 0;
}

static int run_direction(tu_dma_direction_t direction,
                         uint32_t read_issue, uint32_t write_issue,
                         uint64_t *done_out) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, BYTES, "dma-directional-issue-sweep");
    sram.banks.bw_modeling = false;
    memset(output, 0, sizeof(output));
    if (direction == TU_DMA_DIR_TU_TO_HOST)
        memcpy(tu_sram_raw_ptr(&sram), input, BYTES);

    tu_dma_init_config_directional_issue(
        true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 32u, 32u, 32u, 2u,
        read_issue, write_issue, true, true);
    if (g_tu_dma.read_burst_issue_cycles != read_issue ||
        g_tu_dma.write_burst_issue_cycles != write_issue) return -1;

    void *host = direction == TU_DMA_DIR_HOST_TO_TU ? (void *)input : output;
    tu_dma_descriptor_t *d = tu_dma_desc_create_linear(
        0, direction, &sram, 0, host, 1, BYTES);
    if (!d || tu_dma_submit_desc(d) == 0) return -2;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();

    uint32_t issue = direction == TU_DMA_DIR_TU_TO_HOST ?
                     write_issue : read_issue;
    uint64_t expected = 1u + 50u + (BYTES + 31u) / 32u + 3u * issue;
    *done_out = d->cycles_completed;
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
    tu_sram_init(&sram, BYTES, "dma-directional-issue-legacy");
    sram.banks.bw_modeling = false;
    tu_dma_init_config_directional_issue(
        false, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 32u, 32u, 32u, 2u,
        0u, 4u, true, true);
    tu_dma_load(TU_DMA_CHAN_W, &sram, 0, input, BYTES);
    if (g_tu_dma.estimated_cycles != 53u) return -1;
    tu_dma_store(TU_DMA_CHAN_O, &sram, 0, roundtrip, BYTES);
    if (g_tu_dma.estimated_cycles != 118u ||
        memcmp(roundtrip, input, BYTES) != 0) return -2;
    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

static int rejection_gate(void) {
    tu_dma_init_config_directional_issue(
        true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
        TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
        256u, 50u, 50u, 32u, 32u, 32u, 2u,
        1025u, 0u, true, true);
    return g_tu_dma.num_channels == 0 &&
           g_tu_dma.read_burst_issue_cycles == 0 ? 0 : -1;
}

int main(void) {
    for (uint32_t i = 0; i < BYTES; i++) input[i] = (uint8_t)(i ^ 0xa5u);
    if (propagation_gate() != 0) {
        fprintf(stderr, "FAIL: directional issue config propagation\n");
        return 1;
    }

    const uint32_t reads[] = {2u, 0u, 4u};
    const uint32_t writes[] = {2u, 4u, 0u};
    printf("DMA directional issue sweep (96 B, 32 B bursts, 256-bit path, 50-cycle base)\n");
    printf("read_issue write_issue load_done store_done\n");
    for (uint32_t i = 0; i < 3; i++) {
        uint64_t load_done = 0, store_done = 0;
        int rc = run_direction(TU_DMA_DIR_HOST_TO_TU, reads[i], writes[i],
                               &load_done);
        if (rc == 0)
            rc = run_direction(TU_DMA_DIR_TU_TO_HOST, reads[i], writes[i],
                               &store_done);
        if (rc != 0) {
            fprintf(stderr, "FAIL: read=%u write=%u rc=%d\n",
                    reads[i], writes[i], rc);
            return 2;
        }
        printf("%10u %11u %9lu %10lu\n",
               reads[i], writes[i],
               (unsigned long)load_done, (unsigned long)store_done);
    }
    if (legacy_gate() != 0) {
        fprintf(stderr, "FAIL: directional legacy accounting\n");
        return 3;
    }
    if (projected_binding_gate() != 0) {
        fprintf(stderr, "FAIL: directional queued projection\n");
        return 4;
    }
    if (rejection_gate() != 0) {
        fprintf(stderr, "FAIL: invalid directional issue accepted\n");
        return 5;
    }
    printf("PASS: explicit zero, inheritance, live/queued/direct/legacy paths, exact cycles/bytes, rejection\n");
    return 0;
}
