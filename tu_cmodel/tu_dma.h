/* TinyTU DMA Engine */
#ifndef TU_DMA_H
#define TU_DMA_H
#include "tu_config.h"
#include "tu_sram.h"
#include <stdint.h>

typedef enum { TU_DMA_CHAN_W=0, TU_DMA_CHAN_A=1, TU_DMA_CHAN_O=2, TU_DMA_NUM_CHANNELS=3 } tu_dma_channel_t;

typedef struct {
    uint64_t total_bytes;
    uint64_t total_transfers;
    uint64_t estimated_cycles;
    bool     async_mode;
} tu_dma_engine_t;

extern tu_dma_engine_t g_tu_dma;

void tu_dma_init(bool async);
void tu_dma_load(tu_dma_channel_t ch, tu_sram_region_t *dst, uint32_t offset, const void *host, uint32_t bytes);
void tu_dma_store(tu_dma_channel_t ch, tu_sram_region_t *src, uint32_t offset, void *host, uint32_t bytes);
void tu_dma_sync(void);
void tu_dma_print_stats(void);
#endif
