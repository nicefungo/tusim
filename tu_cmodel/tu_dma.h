/*
 * TinyTU DMA Engine — Backward-Compat Header
 * ============================================
 * Thin wrapper around dma_descriptor.h for backward compatibility.
 * All new code should use dma_descriptor.h directly.
 */
#ifndef TU_DMA_H
#define TU_DMA_H

#include "dma_descriptor.h"

/* Backward compat: dma_descriptor.h already provides all types.
 * tu_dma_load/tu_dma_store/tu_dma_sync/tu_dma_print_stats
 * are declared in dma_descriptor.h. */

#endif /* TU_DMA_H */
