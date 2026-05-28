/* TinyTU DMA Engine — Implementation */
#include "tu_dma.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

tu_dma_engine_t g_tu_dma = {0};

void tu_dma_init(bool async) {
    memset(&g_tu_dma, 0, sizeof(g_tu_dma));
    g_tu_dma.async_mode = async;
}

void tu_dma_load(tu_dma_channel_t ch, tu_sram_region_t *dst, uint32_t offset,
                 const void *host, uint32_t bytes) {
    (void)ch;
    if (offset + bytes > dst->total_size) {
        fprintf(stderr, "DMA load overflow: ch=%d off=%u size=%u/%u\n", ch, offset, bytes, dst->total_size);
        abort();
    }
    memcpy(dst->banks.data + offset, host, bytes);
    g_tu_dma.total_bytes += bytes;
    g_tu_dma.total_transfers++;
    g_tu_dma.estimated_cycles += TU_LATENCY_DRAM_READ;
    g_tu_dma.estimated_cycles += (bytes + TU_DMA_BUS_WIDTH_BYTES - 1) / TU_DMA_BUS_WIDTH_BYTES;
}

void tu_dma_store(tu_dma_channel_t ch, tu_sram_region_t *src, uint32_t offset,
                  void *host, uint32_t bytes) {
    (void)ch;
    if (offset + bytes > src->total_size) {
        fprintf(stderr, "DMA store overflow: ch=%d off=%u size=%u/%u\n", ch, offset, bytes, src->total_size);
        abort();
    }
    memcpy(host, src->banks.data + offset, bytes);
    g_tu_dma.total_bytes += bytes;
    g_tu_dma.total_transfers++;
    g_tu_dma.estimated_cycles += TU_LATENCY_DRAM_WRITE;
    g_tu_dma.estimated_cycles += (bytes + TU_DMA_BUS_WIDTH_BYTES - 1) / TU_DMA_BUS_WIDTH_BYTES;
}

void tu_dma_sync(void) { /* no-op in sync mode */ }
void tu_dma_print_stats(void) {
    fprintf(stderr, "  DMA: %lu bytes, %lu transfers, %lu cycles\n",
            g_tu_dma.total_bytes, g_tu_dma.total_transfers, g_tu_dma.estimated_cycles);
}
