/*
 * TU CModel — Double Buffering Extension for SRAM
 * =================================================
 * Gap A7: Adds ping-pong (double) buffering to SRAM regions,
 *         enabling DMA/compute overlap — the key performance
 *         optimization in production systolic accelerators.
 *
 * Architecture:
 *   Each tu_sram_region_t can be configured for double buffering.
 *   When enabled, a shadow buffer of equal size is allocated.
 *   The "active" buffer is what compute reads/writes; the "shadow"
 *   buffer is what DMA writes into for the next tile.
 *
 *   Buffer swap is an atomic pointer exchange (0-cycle in HW).
 *   This models the hardware pattern of flip-flop pointers.
 *
 *   DMA/compute overlap model:
 *     Cycle N:   Compute reads active buffer (tile N)
 *                DMA writes shadow buffer (tile N+1)
 *     Swap:      active ↔ shadow
 *     Cycle N+1: Compute reads active buffer (tile N+1)
 *                DMA writes shadow buffer (tile N+2)
 *
 *   The performance benefit is modeled by subtracting DMA cycles
 *   from the critical path: if DMA takes D cycles and compute takes
 *   C cycles, double-buffered execution takes max(C, D) rather
 *   than C + D.
 *
 *   Swap semantics:
 *     - Swap is instant (0-cycle): just exchanges pointers
 *     - After swap, all subsequent reads see the new active buffer
 *     - The old active buffer becomes available for DMA writes
 *     - Multiple swaps per operation are supported (for tiled execution)
 *
 *   Integration:
 *     - tu_sram_swap_buffers() performs the pointer swap
 *     - tu_dma_load_async_to_shadow() writes to the shadow buffer
 *     - tu_sram_get_active_ptr() and tu_sram_get_shadow_ptr() expose
 *       raw pointers for direct DMA access
 *
 *   Limitations (by design):
 *     - Shadow buffer is same size as primary (required for swap)
 *     - Only one shadow buffer (double, not triple, buffering)
 *     - Swap does not copy data; it's a pure pointer exchange
 *     - Bank bandwidth modeling applies to whichever buffer is active
 */

#ifndef TU_DOUBLE_BUFFER_H
#define TU_DOUBLE_BUFFER_H

#include "../tu_config.h"
#include "../tu_sram.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Double-buffer configuration ---- */

#define TU_DB_MAX_REGIONS    8   /* Max double-buffered regions */

typedef struct tu_double_buffer_t {
    bool        enabled;              /* Double buffering active? */
    uint8_t    *shadow_data;          /* Shadow buffer (same size as primary) */
    uint32_t    buffer_size;          /* Size of each buffer (bytes) */
    uint8_t     active_idx;           /* Which buffer is active: 0=primary, 1=shadow */
    uint64_t    swap_count;           /* Number of swaps performed */
    uint64_t    dma_to_shadow_bytes;  /* Bytes DMA'd into shadow buffer */
    uint64_t    dma_to_shadow_cycles; /* DMA cycles into shadow buffer */
    bool        shadow_dirty;         /* Shadow buffer has been written since last swap */
    uint64_t    overlapped_cycles;    /* Compute cycles saved by overlap */
} tu_double_buffer_t;

/* ---- SRAM Region Extensions ---- */

/*
 * Enable double buffering on an SRAM region.
 *
 * Allocates a shadow buffer of equal size. After this call:
 *   - tu_sram_raw_ptr() returns the active buffer
 *   - DMA targeting the shadow buffer uses tu_sram_get_shadow_ptr()
 *   - tu_sram_swap_buffers() atomically exchanges active and shadow
 *
 * Must be called after tu_sram_init() and before any DMA operations.
 * Cannot be disabled once enabled (models fixed hardware design).
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int tu_sram_enable_double_buffer(tu_sram_region_t *r);

/*
 * Disable double buffering and free shadow buffer.
 * After this, the region behaves as single-buffered.
 * All data in the active buffer is preserved; shadow buffer is discarded.
 */
void tu_sram_disable_double_buffer(tu_sram_region_t *r);

/*
 * Check if a region has double buffering enabled.
 */
bool tu_sram_is_double_buffered(const tu_sram_region_t *r);

/*
 * Atomically swap active and shadow buffers.
 *
 * This is a 0-cycle operation (pointer exchange in hardware).
 * After swap:
 *   - tu_sram_raw_ptr() returns the new active buffer
 *   - The old active buffer becomes the new shadow buffer
 *   - shadow_dirty flag is cleared (new shadow is clean)
 *
 * Returns the number of swaps performed so far (for stats).
 * Returns 0 if double buffering is not enabled.
 */
uint64_t tu_sram_swap_buffers(tu_sram_region_t *r);

/*
 * Get the active buffer's raw data pointer.
 * Equivalent to tu_sram_raw_ptr() but explicit about double-buffer semantics.
 * Returns primary buffer pointer if double buffering is disabled.
 */
uint8_t *tu_sram_get_active_ptr(tu_sram_region_t *r);

/*
 * Get the shadow buffer's raw data pointer.
 *
 * DMA engines should write new data here while compute reads the active buffer.
 * Returns NULL if double buffering is not enabled.
 *
 * IMPORTANT: Writing to the shadow buffer sets the shadow_dirty flag.
 * The swap operation clears this flag, so consumers can check whether
 * the shadow buffer has fresh data.
 */
uint8_t *tu_sram_get_shadow_ptr(tu_sram_region_t *r);

/*
 * Notify that DMA has written `bytes` to the shadow buffer.
 *
 * This updates bandwidth statistics and marks the shadow as dirty.
 * The `cycles` parameter represents the DMA transfer time; these
 * cycles overlap with compute when the active buffer is in use,
 * so they are tracked separately from compute cycles.
 */
void tu_sram_notify_shadow_write(tu_sram_region_t *r,
                                  uint32_t bytes, uint64_t cycles);

/*
 * Check if the shadow buffer has been written since the last swap.
 * When true, swapping would bring fresh data into the active buffer.
 */
bool tu_sram_is_shadow_dirty(const tu_sram_region_t *r);

/*
 * Get double-buffer statistics.
 */
typedef struct {
    bool     enabled;
    uint32_t buffer_size;
    uint64_t swap_count;
    uint64_t dma_to_shadow_bytes;
    uint64_t dma_to_shadow_cycles;
    uint64_t overlapped_cycles;   /* Compute cycles that overlapped with DMA */
    bool     shadow_dirty;
} tu_db_stats_t;

void tu_sram_get_db_stats(const tu_sram_region_t *r, tu_db_stats_t *stats);

/*
 * Print double-buffer statistics to stderr.
 */
void tu_sram_print_db_stats(const tu_sram_region_t *r);

/*
 * Record that `cycles` of compute happened while DMA was active on
 * the shadow buffer. These cycles would have been sequential without
 * double buffering but are overlapped.
 */
void tu_sram_record_overlapped_cycles(tu_sram_region_t *r, uint64_t cycles);

/*
 * Get total overlapped cycles (compute cycles saved by double buffering).
 */
uint64_t tu_sram_get_overlapped_cycles(const tu_sram_region_t *r);

#ifdef __cplusplus
}
#endif

#endif /* TU_DOUBLE_BUFFER_H */
