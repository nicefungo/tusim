/*
 * TinyTU DMA Descriptor Engine
 * =============================
 * Hardware DMA descriptor types, strided transfers, chaining,
 * completion signaling, and scatter/gather support (DM3).
 *
 * Gap DM1/DM2: Synchronous DMA → Async DMA with descriptor queues.
 * Gap DM3: Scatter/gather DMA for sparse weight/activation access.
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
    TU_DMA_XFER_SCATTER     = 3,  /* 1-to-N: src contiguous → dst via index list (DM3) */
    TU_DMA_XFER_GATHER      = 4,  /* N-to-1: src via index list → dst contiguous (DM3) */
    TU_DMA_XFER_MULTICAST   = 5,  /* 1-to-N: single src → multiple SRAM dst regions (DM4) */
    TU_DMA_XFER_COUNT
} tu_dma_transfer_type_t;

/* ---- Channel types ---- */
typedef enum {
    TU_DMA_CHAN_W = 0,
    TU_DMA_CHAN_A = 1,
    TU_DMA_CHAN_O = 2,
    TU_DMA_NUM_CHANNELS = 3
} tu_dma_channel_t;

typedef enum {
    TU_DMA_BUS_MODE_INDEPENDENT = 0,
    TU_DMA_BUS_MODE_SHARED_SERIAL = 1
} tu_dma_bus_mode_t;

/* Shared-serial descriptor-boundary arbitration. Round-robin is the
 * compatibility default; strict priority uses descriptor.priority and
 * round-robin tie-breaking. Independent paths do not consume this policy. */
typedef enum {
    TU_DMA_ARB_ROUND_ROBIN = 0,
    TU_DMA_ARB_STRICT_PRIORITY = 1
} tu_dma_arb_policy_t;

/* Descriptor-to-queue binding. Explicit preserves the producer-selected
 * channel. Automatic policies rebind at accepted submission boundaries. */
typedef enum {
    TU_DMA_BIND_EXPLICIT = 0,
    TU_DMA_BIND_ROUND_ROBIN = 1,
    TU_DMA_BIND_LEAST_OUTSTANDING = 2,
    TU_DMA_BIND_LEAST_BYTES = 3,
    TU_DMA_BIND_LEAST_PROJECTED_CYCLES = 4
} tu_dma_binding_policy_t;

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

    /* DM3: Scatter/Gather index list */
    const uint32_t         *index_list;     /* Array of byte offsets (NULL for non-S/G) */
    uint32_t                index_count;    /* Number of entries in index_list */
    uint32_t                index_elem_size;/* Bytes per element in scatter/gather */

    /* DM4: Multicast destination list */
    struct {
        tu_sram_region_t  **regions;        /* Array of destination SRAM regions */
        uint32_t           *offsets;        /* Array of destination offsets */
        uint32_t            count;          /* Number of multicast targets */
    } multicast;

    /* Status (set by engine) */
    bool                    completed;
    uint64_t                cycles_issued;
    uint64_t                cycles_completed;
} tu_dma_descriptor_t;

/* ---- DMA Channel State ---- */
typedef struct {
    uint8_t                 channel_id;
    tu_dma_descriptor_t    *head;
    tu_dma_descriptor_t    *tail;
    tu_dma_descriptor_t    *active;
    uint32_t                queue_depth;
    uint32_t                max_depth;
    uint64_t                total_submitted;
    uint64_t                total_completed;
    uint64_t                total_bytes;
    uint64_t                total_cycles;
} tu_dma_channel_state_t;

/* ---- DMA Engine ---- */
typedef struct {
    /* Fixed model capacity is independent of the selected/default channel
     * count so one binary can compare candidate 1..8-channel designs. */
    tu_dma_channel_state_t  channels[TU_DMA_ENGINE_MAX_CHANNELS];
    uint32_t                num_channels;
    tu_dma_bus_mode_t       bus_mode;
    tu_dma_arb_policy_t     arb_policy;
    uint32_t                next_shared_channel;
    tu_dma_binding_policy_t binding_policy;
    uint32_t                next_binding_channel;
    uint32_t                bus_width_bytes;
    uint32_t                max_burst_bytes;
    uint32_t                read_max_burst_bytes;
    uint32_t                write_max_burst_bytes;
    uint32_t                burst_issue_cycles;
    uint32_t                read_latency_cycles;
    uint32_t                write_latency_cycles;
    bool                    async_mode;
    uint64_t                current_cycle;
    uint64_t                total_bytes;
    uint64_t                total_transfers;
    uint64_t                estimated_cycles;
} tu_dma_engine_t;

extern tu_dma_engine_t g_tu_dma;

/* ---- Lifecycle ---- */
void tu_dma_init_full(bool async, uint32_t num_channels, uint32_t max_queue_depth);
void tu_dma_init_config(bool async, uint32_t num_channels,
                        uint32_t max_queue_depth, int bus_mode);
void tu_dma_init_config_policy(bool async, uint32_t num_channels,
                               uint32_t max_queue_depth, int bus_mode,
                               int arb_policy);
void tu_dma_init_config_full(bool async, uint32_t num_channels,
                             uint32_t max_queue_depth, int bus_mode,
                             int arb_policy, int binding_policy);
void tu_dma_init_config_arch(bool async, uint32_t num_channels,
                             uint32_t max_queue_depth, int bus_mode,
                             int arb_policy, int binding_policy,
                             uint32_t bus_width_bits);
void tu_dma_init_config_timing(bool async, uint32_t num_channels,
                               uint32_t max_queue_depth, int bus_mode,
                               int arb_policy, int binding_policy,
                               uint32_t bus_width_bits,
                               uint32_t read_latency_cycles,
                               uint32_t write_latency_cycles);
void tu_dma_init_config_burst(bool async, uint32_t num_channels,
                              uint32_t max_queue_depth, int bus_mode,
                              int arb_policy, int binding_policy,
                              uint32_t bus_width_bits,
                              uint32_t read_latency_cycles,
                              uint32_t write_latency_cycles,
                              uint32_t max_burst_bytes,
                              uint32_t burst_issue_cycles);
void tu_dma_init_config_directional_burst(bool async, uint32_t num_channels,
                                          uint32_t max_queue_depth, int bus_mode,
                                          int arb_policy, int binding_policy,
                                          uint32_t bus_width_bits,
                                          uint32_t read_latency_cycles,
                                          uint32_t write_latency_cycles,
                                          uint32_t max_burst_bytes,
                                          uint32_t read_max_burst_bytes,
                                          uint32_t write_max_burst_bytes,
                                          uint32_t burst_issue_cycles);
void tu_dma_init(bool async);
void tu_dma_destroy(void);

/* ---- Descriptor Construction ---- */
tu_dma_descriptor_t *tu_dma_desc_create_linear(
    uint8_t channel, tu_dma_direction_t dir,
    tu_sram_region_t *sram, uint32_t sram_offset,
    void *host_ptr, uint32_t elem_size, uint32_t elem_count);
tu_dma_descriptor_t *tu_dma_desc_create_strided_2d(
    uint8_t channel, tu_dma_direction_t dir,
    tu_sram_region_t *sram, uint32_t sram_base,
    void *host_ptr, uint32_t sram_row_stride, uint32_t host_row_stride,
    uint32_t elem_size, uint32_t rows, uint32_t cols);
tu_dma_descriptor_t *tu_dma_desc_create_strided_3d(
    uint8_t channel, tu_dma_direction_t dir,
    tu_sram_region_t *sram, uint32_t sram_base,
    void *host_ptr,
    uint32_t sram_row_stride, uint32_t sram_depth_stride,
    uint32_t host_row_stride, uint32_t host_depth_stride,
    uint32_t elem_size, uint32_t depth, uint32_t rows, uint32_t cols);

/* DM3: Scatter — src contiguous host → scattered SRAM targets via index list */
tu_dma_descriptor_t *tu_dma_desc_create_scatter(
    uint8_t channel, tu_sram_region_t *dst_region,
    const void *src_host, const uint32_t *index_list,
    uint32_t elem_count, uint32_t elem_size);

/* DM3: Gather — scattered SRAM sources via index list → dst contiguous host */
tu_dma_descriptor_t *tu_dma_desc_create_gather(
    uint8_t channel, tu_sram_region_t *src_region,
    void *dst_host, const uint32_t *index_list,
    uint32_t elem_count, uint32_t elem_size);

/* DM4: Multicast — single src → multiple SRAM destinations (1-to-N broadcast) */
tu_dma_descriptor_t *tu_dma_desc_create_multicast(
    uint8_t channel,
    const void *src_host,
    tu_sram_region_t **dst_regions, uint32_t *dst_offsets,
    uint32_t num_destinations, uint32_t elem_size, uint32_t elem_count);

void tu_dma_desc_destroy(tu_dma_descriptor_t *desc);
tu_dma_descriptor_t *tu_dma_desc_chain(tu_dma_descriptor_t *head, tu_dma_descriptor_t *tail);

/* ---- Submission & Execution ---- */
uint32_t tu_dma_submit_desc(tu_dma_descriptor_t *desc);
void tu_dma_execute_desc(tu_dma_descriptor_t *desc);
int tu_dma_tick(void);
void tu_dma_flush_all(void);
void tu_dma_flush_channel(uint8_t channel);

/* ---- Legacy API ---- */
void tu_dma_load(tu_dma_channel_t ch, tu_sram_region_t *dst,
                 uint32_t offset, const void *host, uint32_t bytes);
void tu_dma_store(tu_dma_channel_t ch, tu_sram_region_t *src,
                  uint32_t offset, void *host, uint32_t bytes);
void tu_dma_sync(void);
void tu_dma_print_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* TU_DMA_DESCRIPTOR_H */
