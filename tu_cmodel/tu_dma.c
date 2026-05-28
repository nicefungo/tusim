/*
 * TinyTU DMA Engine — Backward-Compat Implementation
 * ====================================================
 * All functionality delegated to dma_descriptor.c.
 * This file exists for backward compat with existing code that
 * uses tu_dma_load/tu_dma_store/tu_dma_sync/tu_dma_print_stats.
 *
 * For new code, use dma_descriptor.h: tu_dma_desc_create_*,
 * tu_dma_submit_desc, tu_dma_flush_all, etc.
 */
#include "tu_dma.h"

/* All functions (tu_dma_init, tu_dma_load, tu_dma_store,
 * tu_dma_sync, tu_dma_print_stats) are implemented in
 * dma_descriptor.c. This file is intentionally empty. */
