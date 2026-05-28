/*
 * TinyTU DMA Descriptor Engine
 * =============================
 * Hardware DMA descriptor types, strided transfers, chaining,
 * and completion signaling.
 *
 * Gap DM1/DM2: Synchronous DMA → Async DMA with descriptor queues.
 *
 * Architecture:
 *   DMA descriptors describe data movement between memory levels.
 *   Each descriptor specifies:
 *     - Source and destination (memory level + base address)
 *     - Transfer geometry (linear, strided 2D/3D)
 *     - Chaining (linked-list via next_descriptor pointer)
 *     - Completion signaling (signal ID for interrupt generation)
 *
 *   The DMA engine processes descriptors from a per-channel queue.
 *   In functional mode, descriptors execute immediately.
 *   In async mode, they are enqueued and executed via tu_dma_tick().
 *
 *   Strided transfers are modeled as multiple linear transfers with
 *   proper stride accounting, enabling accurate bandwidth estimation
 *   for non-contiguous memory patterns (e.g., matrix row/column access,
 *   im2col patterns, tiled DMA).
 */

#ifndef TU_DMA_DESCRIPTOR_H
#define TU_DMA_DESCRIPTOR_H

#include "tu_config.h"
#include "tu_sram.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Transfer types ---- */
typedef enum {
    TU_DMA_XFER_LINEAR      = 0,  /* Contiguous byte range */
    TU_DMA_XFER_STRIDED_2D  = 1,  /* 2D with row stride */
    TU_DMA_XFER_STRIDED_3D  = 2,  /* 3D with depth and row strides */
    TU_DMA_XFER_COUNT
} tu_dma_transfer_type_t;

/* ---- Channel types ---- */
typedef enum {
    TU_DMA_CHAN_W = 0,
    TU_DMA_CHAN_A = 1,
    TU_DMA_CHAN_O = 2,
    TU_DMA_NUM_CHANNELS = 3
} tu_dma_channel_t;

/* ---- Transfer direction ---- */
typedef enum {
    TU_DMA_DIR_HOST_TO_TU   = 0,  /* Load: DRAM → SRAM */
    TU_DMA_DIR_TU_TO_HOST   = 1,  /* Store: SRAM → DRAM */
    TU_DMA_DIR_TU_TO_TU     = 2,  /* Internal: SRAM region → SRAM region */
} tu_dma_direction_t;

/* ---- DMA descriptor ---- */
typedef struct tu_dma_descriptor_t {
    /* Descriptor metadata */
    uint32_t                desc_id;        /* Unique descriptor ID */
    tu_dma_transfer_type_t  type;
    tu_dma_direction_t      direction;

    /* Channel assignment */
    uint8_t                 channel;        /* DMA channel (0=W, 1=A, 2=O, 3+=general) */

    /* Source specification */
    tu_sram_region_t       *src_region;     /* Source SRAM region (NULL = host DRAM) */
    uint32_t                src_base;       /* Base byte offset in source memory */
    uint32_t                src_strides[3]; /* Strides: [row_stride, depth_stride, 0] */
    const void             *src_host;       /* Host-side pointer (for host↔TU xfers) */

    /* Destination specification */
    tu_sram_region_t       *dst_region;     /* Destination SRAM region (NULL = host DRAM) */
    uint32_t                dst_base;       /* Base byte offset in destination memory */
    uint32_t                dst_strides[3]; /* Strides: [row_stride, depth_stride, 0] */
    void                   *dst_host;       /* Host-side pointer */

    /* Transfer geometry */
    uint32_t                dims[3];        /* [rows, cols, depth] */
    uint32_t                elem_size;      /* Bytes per element */
    uint32_t                total_bytes;    /* Computed total transfer size */

    /* Chaining */
    struct tu_dma_descriptor_t *next;       /* Linked list: next descriptor (NULL = end of chain) */

    /* Completion */
    uint32_t                signal_id;      /* Completion signal ID (0 = no signal) */

    /* Priority */
    uint8_t                 priority;       /* 0 (lowest) to 255 (highest) */

    /* Status (set by engine) */
    bool                    completed;
    uint64_t                cycles_issued;
    uint64_t                cycles_completed;
} tu_dma_descriptor_t;

/* ---- DMA Channel State ---- */
typedef struct {
    uint8_t                 channel_id;
    tu_dma_descriptor_t    *head;           /* Head of descriptor queue */
    tu_dma_descriptor_t    *tail;           /* Tail (for appending) */
    tu_dma_descriptor_t    *active;         /* Currently executing descriptor */

    uint32_t                queue_depth;    /* Number of pending descriptors */
    uint32_t                max_depth;      /* Max queue depth */

    /* Counters */
    uint64_t                total_submitted;
    uint64_t                total_completed;
    uint64_t                total_bytes;
    uint64_t                total_cycles;
} tu_dma_channel_state_t;

/* ---- DMA Engine (with descriptor support) ---- */
typedef struct {
    tu_dma_channel_state_t  channels[TU_DMA_CHANNELS];
    uint32_t                num_channels;
    bool                    async_mode;
    uint64_t                current_cycle;

    /* Global counters */
    uint64_t                total_bytes;
    uint64_t                total_transfers;
    uint64_t                estimated_cycles;
} tu_dma_engine_t;

/* ---- External engine instance ---- */
extern tu_dma_engine_t g_tu_dma;

/* ---- Lifecycle ---- */

/* Initialize DMA engine (with async support) */
void tu_dma_init_full(bool async, uint32_t num_channels, uint32_t max_queue_depth);

/* Legacy init (backward compat) */
void tu_dma_init(bool async);

/* Destroy engine (free all descriptor chains) */
void tu_dma_destroy(void);

/* ---- Descriptor Construction ---- */

/* Create and initialize a linear transfer descriptor */
tu_dma_descriptor_t *tu_dma_desc_create_linear(
    uint8_t channel,
    tu_dma_direction_t dir,
    tu_sram_region_t *sram, uint32_t sram_offset,
    void *host_ptr,
    uint32_t elem_size,
    uint32_t elem_count);

/* Create a 2D strided transfer descriptor */
tu_dma_descriptor_t *tu_dma_desc_create_strided_2d(
    uint8_t channel,
    tu_dma_direction_t dir,
    tu_sram_region_t *sram, uint32_t sram_base,
    void *host_ptr,
    uint32_t sram_row_stride,    /* Stride between rows in SRAM */
    uint32_t host_row_stride,    /* Stride between rows in host memory */
    uint32_t elem_size,
    uint32_t rows, uint32_t cols);

/* Create a 3D strided transfer descriptor */
tu_dma_descriptor_t *tu_dma_desc_create_strided_3d(
    uint8_t channel,
    tu_dma_direction_t dir,
    tu_sram_region_t *sram, uint32_t sram_base,
    void *host_ptr,
    uint32_t sram_row_stride,
    uint32_t sram_depth_stride,
    uint32_t host_row_stride,
    uint32_t host_depth_stride,
    uint32_t elem_size,
    uint32_t depth, uint32_t rows, uint32_t cols);

/* Free a descriptor (and all chained descriptors) */
void tu_dma_desc_destroy(tu_dma_descriptor_t *desc);

/* Chain two descriptors: head→next = tail. Returns head. */
tu_dma_descriptor_t *tu_dma_desc_chain(tu_dma_descriptor_t *head,
                                        tu_dma_descriptor_t *tail);

/* ---- Submission & Execution ---- */

/* Submit a descriptor (or chain) for execution.
 * In sync mode: executes immediately.
 * In async mode: enqueues on the channel, returns immediately.
 * Returns descriptor ID on success, 0 on failure. */
uint32_t tu_dma_submit_desc(tu_dma_descriptor_t *desc);

/* Execute a single descriptor immediately (regardless of async mode).
 * Used internally; also available for direct execution. */
void tu_dma_execute_desc(tu_dma_descriptor_t *desc);

/* Advance async execution by one tick. Returns # of descriptors completed this tick. */
int tu_dma_tick(void);

/* Flush all pending DMA on all channels */
void tu_dma_flush_all(void);

/* Flush pending DMA on a specific channel */
void tu_dma_flush_channel(uint8_t channel);

/* ---- Legacy API (backward compat) ---- */

/* Simple DMA load: host → SRAM region */
void tu_dma_load(tu_dma_channel_t ch, tu_sram_region_t *dst,
                 uint32_t offset, const void *host, uint32_t bytes);

/* Simple DMA store: SRAM region → host */
void tu_dma_store(tu_dma_channel_t ch, tu_sram_region_t *src,
                  uint32_t offset, void *host, uint32_t bytes);

/* Wait for all pending DMA */
void tu_dma_sync(void);

/* Print DMA statistics */
void tu_dma_print_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* TU_DMA_DESCRIPTOR_H */
