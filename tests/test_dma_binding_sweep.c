/* Runtime DMA descriptor channel-binding exploration. */
#include "tu_cmodel/dma_descriptor.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 6u
#define SRAM_BYTES (96u * 1024u)
static uint8_t src[N][12288];

static const char *name(int p) {
    if (p == TU_DMA_BIND_ROUND_ROBIN) return "round_robin";
    if (p == TU_DMA_BIND_LEAST_OUTSTANDING) return "least_outstanding";
    if (p == TU_DMA_BIND_LEAST_BYTES) return "least_bytes";
    if (p == TU_DMA_BIND_LEAST_PROJECTED_CYCLES) return "least_projected_cycles";
    return "explicit";
}

static int run_case(int policy, uint8_t assigned[N], uint64_t done[N],
                    uint64_t *batch) {
    const uint32_t bytes[N] = {12288, 4096, 4096, 4096, 4096, 4096};
    tu_sram_region_t sram;
    tu_dma_descriptor_t *d[N] = {0};
    tu_sram_init(&sram, SRAM_BYTES, "dma-binding-sweep");
    sram.banks.bw_modeling = false; /* isolate queue binding from SRAM meter */
    tu_dma_init_config_full(true, 3, N, TU_DMA_BUS_MODE_INDEPENDENT,
                            TU_DMA_ARB_ROUND_ROBIN, policy);
    if (g_tu_dma.num_channels != 3 ||
        g_tu_dma.binding_policy != (tu_dma_binding_policy_t)policy)
        return -1;

    for (uint32_t i = 0; i < 3; i++) {
        memset(src[i], 0x20 + (int)i, bytes[i]);
        d[i] = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram,
                                         i * 12288u, src[i], 1, bytes[i]);
        if (!d[i] || tu_dma_submit_desc(d[i]) == 0) return -2;
        assigned[i] = d[i]->channel;
    }
    while (g_tu_dma.current_cycle < 307u) tu_dma_tick();
    for (uint32_t i = 3; i < N; i++) {
        memset(src[i], 0x20 + (int)i, bytes[i]);
        d[i] = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram,
                                         i * 12288u, src[i], 1, bytes[i]);
        if (!d[i] || tu_dma_submit_desc(d[i]) == 0) return -3;
        assigned[i] = d[i]->channel;
    }
    while (g_tu_dma.total_transfers < N && g_tu_dma.current_cycle < 10000u)
        tu_dma_tick();
    while (g_tu_dma.channels[0].total_completed +
           g_tu_dma.channels[1].total_completed +
           g_tu_dma.channels[2].total_completed < N &&
           g_tu_dma.current_cycle < 10000u)
        tu_dma_tick();

    *batch = 0;
    for (uint32_t i = 0; i < N; i++) {
        done[i] = d[i]->cycles_completed;
        if (done[i] > *batch) *batch = done[i];
        uint8_t *raw = (uint8_t *)tu_sram_raw_ptr(&sram);
        uint32_t off = i * 12288u;
        for (uint32_t j = 0; j < bytes[i]; j++)
            if (raw[off + j] != (uint8_t)(0x20u + i)) return -4;
    }

    if (policy == TU_DMA_BIND_EXPLICIT) {
        const uint8_t a[N] = {0,0,0,0,0,0};
        const uint64_t t[N] = {435,613,791,969,1147,1325};
        if (memcmp(assigned, a, N) != 0 || memcmp(done, t, sizeof(t)) != 0) {
            fprintf(stderr, "explicit observed a=%u,%u,%u,%u,%u,%u t=%lu,%lu,%lu,%lu,%lu,%lu\n",
                    assigned[0],assigned[1],assigned[2],assigned[3],assigned[4],assigned[5],
                    (unsigned long)done[0],(unsigned long)done[1],(unsigned long)done[2],
                    (unsigned long)done[3],(unsigned long)done[4],(unsigned long)done[5]);
            return -5;
        }
    } else if (policy == TU_DMA_BIND_ROUND_ROBIN) {
        const uint8_t a[N] = {0,1,2,0,1,2};
        const uint64_t t[N] = {435,179,179,613,486,486};
        if (memcmp(assigned, a, N) != 0 || memcmp(done, t, sizeof(t)) != 0)
            return -6;
    } else if (policy == TU_DMA_BIND_LEAST_OUTSTANDING) {
        const uint8_t a[N] = {0,1,2,1,2,0};
        const uint64_t t[N] = {435,179,179,486,486,613};
        if (memcmp(assigned, a, N) != 0 || memcmp(done, t, sizeof(t)) != 0)
            return -7;
    } else if (policy == TU_DMA_BIND_LEAST_BYTES) {
        const uint8_t a[N] = {0,1,2,1,2,1};
        const uint64_t t[N] = {435,179,179,486,486,664};
        if (memcmp(assigned, a, N) != 0 || memcmp(done, t, sizeof(t)) != 0)
            return -8;
    } else {
        const uint8_t a[N] = {0,1,2,1,2,0};
        const uint64_t t[N] = {435,179,179,486,486,613};
        if (memcmp(assigned, a, N) != 0 || memcmp(done, t, sizeof(t)) != 0)
            return -9;
    }

    tu_dma_destroy();
    for (uint32_t i = 0; i < N; i++) {
        d[i]->next = NULL;
        tu_dma_desc_destroy(d[i]);
    }
    tu_sram_destroy(&sram);
    return 0;
}

static int rejection_gates(void) {
    tu_sram_region_t sram;
    uint8_t b = 1;
    tu_sram_init(&sram, 64, "binding-reject");
    tu_dma_init_config_full(true, 3, 4, TU_DMA_BUS_MODE_INDEPENDENT,
                            TU_DMA_ARB_ROUND_ROBIN, 99);
    if (g_tu_dma.num_channels != 0 || g_tu_dma.total_transfers != 0) return -1;
    tu_dma_init_config_full(true, 3, 4, TU_DMA_BUS_MODE_INDEPENDENT,
                            TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT);
    tu_dma_descriptor_t *bad = tu_dma_desc_create_linear(7,
        TU_DMA_DIR_HOST_TO_TU, &sram, 0, &b, 1, 1);
    if (!bad || tu_dma_submit_desc(bad) != 0) return -2;
    if (g_tu_dma.total_transfers != 0 || g_tu_dma.channels[0].queue_depth != 0)
        return -3;
    tu_dma_destroy();
    tu_sram_destroy(&sram);
    return 0;
}

int main(void) {
    const int policies[] = {TU_DMA_BIND_EXPLICIT, TU_DMA_BIND_ROUND_ROBIN,
                            TU_DMA_BIND_LEAST_OUTSTANDING, TU_DMA_BIND_LEAST_BYTES,
                            TU_DMA_BIND_LEAST_PROJECTED_CYCLES};
    printf("DMA channel binding sweep (all descriptors request channel 0)\n");
    printf("policy assignments completion_cycles batch\n");
    for (uint32_t p = 0; p < 5; p++) {
        uint8_t a[N] = {0}; uint64_t d[N] = {0}, batch = 0;
        int rc = run_case(policies[p], a, d, &batch);
        if (rc != 0) { fprintf(stderr, "FAIL %s rc=%d\n", name(policies[p]), rc); return 10-rc; }
        printf("%17s %u,%u,%u,%u,%u,%u %lu,%lu,%lu,%lu,%lu,%lu %lu\n",
               name(policies[p]), a[0],a[1],a[2],a[3],a[4],a[5],
               (unsigned long)d[0],(unsigned long)d[1],(unsigned long)d[2],
               (unsigned long)d[3],(unsigned long)d[4],(unsigned long)d[5],
               (unsigned long)batch);
    }
    if (rejection_gates() != 0) return 20;
    printf("PASS: assignments, exact completion cycles, bytes, and rejection gates\n");
    return 0;
}
