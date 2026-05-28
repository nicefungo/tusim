/*
 * TinyTU DMA Descriptor Engine — Implementation
 * ===============================================
 *
 * Supports:
 *   - Linear, strided 2D, and strided 3D transfers
 *   - Descriptor chaining (linked list)
 *   - Completion signaling
 *   - Per-channel queuing for async execution
 *   - Dual-mode: synchronous (immediate) or async (tick-driven)
 */

#include "dma_descriptor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Global state
 * ================================================================ */

tu_dma_engine_t g_tu_dma = {0};

/* ================================================================
 * Lifecycle
 * ================================================================ */

void tu_dma_init_full(bool async, uint32_t num_channels, uint32_t max_queue_depth) {
    memset(&g_tu_dma, 0, sizeof(g_tu_dma));
    g_tu_dma.async_mode = async;
    g_tu_dma.num_channels = num_channels > 0 ? num_channels : TU_DMA_CHANNELS;
    if (g_tu_dma.num_channels > 8) g_tu_dma.num_channels = 8;

    for (uint32_t i = 0; i < g_tu_dma.num_channels; i++) {
        g_tu_dma.channels[i].channel_id = (uint8_t)i;
        g_tu_dma.channels[i].max_depth = max_queue_depth > 0 ? max_queue_depth : TU_DMA_MAX_OUTSTANDING;
    }
}

void tu_dma_init(bool async) {
    tu_dma_init_full(async, TU_DMA_CHANNELS, TU_DMA_MAX_OUTSTANDING);
}

void tu_dma_destroy(void) {
    /* Free all descriptor chains on all channels */
    for (uint32_t i = 0; i < g_tu_dma.num_channels; i++) {
        tu_dma_channel_state_t *ch = &g_tu_dma.channels[i];

        /* Free pending queue */
        tu_dma_descriptor_t *desc = ch->head;
        while (desc) {
            tu_dma_descriptor_t *next = desc->next;
            free(desc);
            desc = next;
        }

        /* Free active descriptor */
        if (ch->active && ch->active != ch->head) {
            tu_dma_desc_destroy(ch->active);
        }
    }
    memset(&g_tu_dma, 0, sizeof(g_tu_dma));
}

/* ================================================================
 * Descriptor Construction
 * ================================================================ */

static uint32_t next_desc_id(void) {
    static uint32_t id = 0;
    return ++id;
}

static tu_dma_descriptor_t *alloc_descriptor(void) {
    tu_dma_descriptor_t *desc = calloc(1, sizeof(*desc));
    if (!desc) {
        fprintf(stderr, "DMA: failed to allocate descriptor\n");
        return NULL;
    }
    desc->desc_id = next_desc_id();
    return desc;
}

tu_dma_descriptor_t *tu_dma_desc_create_linear(
    uint8_t channel,
    tu_dma_direction_t dir,
    tu_sram_region_t *sram, uint32_t sram_offset,
    void *host_ptr,
    uint32_t elem_size,
    uint32_t elem_count)
{
    tu_dma_descriptor_t *desc = alloc_descriptor();
    if (!desc) return NULL;

    desc->type      = TU_DMA_XFER_LINEAR;
    desc->direction = dir;
    desc->channel   = channel;
    desc->elem_size = elem_size;
    desc->dims[0]   = elem_count;  /* rows = element count for linear */
    desc->dims[1]   = 1;
    desc->dims[2]   = 1;
    desc->total_bytes = elem_size * elem_count;

    if (dir == TU_DMA_DIR_HOST_TO_TU || dir == TU_DMA_DIR_TU_TO_HOST) {
        desc->src_host = (dir == TU_DMA_DIR_HOST_TO_TU) ? host_ptr : NULL;
        desc->dst_host = (dir == TU_DMA_DIR_TU_TO_HOST) ? host_ptr : NULL;
        if (dir == TU_DMA_DIR_HOST_TO_TU) {
            desc->dst_region = sram;
            desc->dst_base   = sram_offset;
        } else {
            desc->src_region = sram;
            desc->src_base   = sram_offset;
        }
    }

    return desc;
}

tu_dma_descriptor_t *tu_dma_desc_create_strided_2d(
    uint8_t channel,
    tu_dma_direction_t dir,
    tu_sram_region_t *sram, uint32_t sram_base,
    void *host_ptr,
    uint32_t sram_row_stride,
    uint32_t host_row_stride,
    uint32_t elem_size,
    uint32_t rows, uint32_t cols)
{
    tu_dma_descriptor_t *desc = alloc_descriptor();
    if (!desc) return NULL;

    desc->type      = TU_DMA_XFER_STRIDED_2D;
    desc->direction = dir;
    desc->channel   = channel;
    desc->elem_size = elem_size;
    desc->dims[0]   = rows;
    desc->dims[1]   = cols;
    desc->dims[2]   = 1;
    desc->total_bytes = elem_size * rows * cols;

    if (dir == TU_DMA_DIR_HOST_TO_TU) {
        desc->dst_region    = sram;
        desc->dst_base      = sram_base;
        desc->dst_strides[0] = sram_row_stride;
        desc->dst_strides[1] = elem_size;         /* contiguous within row */
        desc->src_host      = host_ptr;
        desc->src_strides[0] = host_row_stride;
        desc->src_strides[1] = elem_size;
    } else if (dir == TU_DMA_DIR_TU_TO_HOST) {
        desc->src_region    = sram;
        desc->src_base      = sram_base;
        desc->src_strides[0] = sram_row_stride;
        desc->src_strides[1] = elem_size;
        desc->dst_host      = host_ptr;
        desc->dst_strides[0] = host_row_stride;
        desc->dst_strides[1] = elem_size;
    }

    return desc;
}

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
    uint32_t depth, uint32_t rows, uint32_t cols)
{
    tu_dma_descriptor_t *desc = alloc_descriptor();
    if (!desc) return NULL;

    desc->type      = TU_DMA_XFER_STRIDED_3D;
    desc->direction = dir;
    desc->channel   = channel;
    desc->elem_size = elem_size;
    desc->dims[0]   = depth;
    desc->dims[1]   = rows;
    desc->dims[2]   = cols;
    desc->total_bytes = elem_size * depth * rows * cols;

    if (dir == TU_DMA_DIR_HOST_TO_TU) {
        desc->dst_region     = sram;
        desc->dst_base       = sram_base;
        desc->dst_strides[0] = sram_row_stride;
        desc->dst_strides[1] = sram_depth_stride;
        desc->dst_strides[2] = elem_size;
        desc->src_host       = host_ptr;
        desc->src_strides[0] = host_row_stride;
        desc->src_strides[1] = host_depth_stride;
        desc->src_strides[2] = elem_size;
    } else if (dir == TU_DMA_DIR_TU_TO_HOST) {
        desc->src_region     = sram;
        desc->src_base       = sram_base;
        desc->src_strides[0] = sram_row_stride;
        desc->src_strides[1] = sram_depth_stride;
        desc->src_strides[2] = elem_size;
        desc->dst_host       = host_ptr;
        desc->dst_strides[0] = host_row_stride;
        desc->dst_strides[1] = host_depth_stride;
        desc->dst_strides[2] = elem_size;
    }

    return desc;
}

tu_dma_descriptor_t *tu_dma_desc_chain(tu_dma_descriptor_t *head,
                                        tu_dma_descriptor_t *tail)
{
    if (!head) return tail;
    tu_dma_descriptor_t *cur = head;
    while (cur->next) cur = cur->next;
    cur->next = tail;
    return head;
}

void tu_dma_desc_destroy(tu_dma_descriptor_t *desc) {
    while (desc) {
        tu_dma_descriptor_t *next = desc->next;
        free(desc);
        desc = next;
    }
}

/* ================================================================
 * Descriptor Execution
 * ================================================================ */

/*
 * Execute a linear transfer.
 */
static void execute_linear(const tu_dma_descriptor_t *desc,
                           const uint8_t *src, uint8_t *dst)
{
    memcpy(dst, src, desc->total_bytes);
}

/*
 * Execute a 2D strided transfer. For each row, copy the row data,
 * then advance src and dst by their respective row strides.
 */
static void execute_strided_2d(const tu_dma_descriptor_t *desc,
                               const uint8_t *src, uint8_t *dst)
{
    uint32_t rows      = desc->dims[0];
    uint32_t cols      = desc->dims[1];
    uint32_t elem_size = desc->elem_size;
    uint32_t row_bytes = cols * elem_size;

    for (uint32_t r = 0; r < rows; r++) {
        memcpy(dst, src, row_bytes);
        src += desc->src_strides[0];
        dst += desc->dst_strides[0];
    }
}

/*
 * Execute a 3D strided transfer. For each depth slice, for each row,
 * copy the row data and advance by strides.
 */
static void execute_strided_3d(const tu_dma_descriptor_t *desc,
                               const uint8_t *src, uint8_t *dst)
{
    uint32_t depth     = desc->dims[0];
    uint32_t rows      = desc->dims[1];
    uint32_t cols      = desc->dims[2];
    uint32_t elem_size = desc->elem_size;
    uint32_t row_bytes = cols * elem_size;

    uint32_t src_depth_stride = desc->src_strides[1];
    uint32_t dst_depth_stride = desc->dst_strides[1];
    uint32_t src_row_stride   = desc->src_strides[0];
    uint32_t dst_row_stride   = desc->dst_strides[0];

    for (uint32_t d = 0; d < depth; d++) {
        const uint8_t *src_slice = src;
        uint8_t       *dst_slice = dst;
        for (uint32_t r = 0; r < rows; r++) {
            memcpy(dst_slice, src_slice, row_bytes);
            src_slice += src_row_stride;
            dst_slice += dst_row_stride;
        }
        src += src_depth_stride;
        dst += dst_depth_stride;
    }
}

void tu_dma_execute_desc(tu_dma_descriptor_t *desc) {
    if (!desc || desc->completed) return;

    /* Determine source and destination pointers */
    const uint8_t *src_ptr = NULL;
    uint8_t       *dst_ptr = NULL;

    switch (desc->direction) {
    case TU_DMA_DIR_HOST_TO_TU:
        src_ptr = (const uint8_t *)desc->src_host;
        if (desc->dst_region) {
            dst_ptr = tu_sram_raw_ptr(desc->dst_region) + desc->dst_base;
            /* Validate bounds */
            if (desc->dst_base + desc->total_bytes > desc->dst_region->total_size) {
                fprintf(stderr, "DMA overflow: dst=%s offset=%u + %u > %u\n",
                        desc->dst_region->name,
                        desc->dst_base, desc->total_bytes,
                        desc->dst_region->total_size);
                return;
            }
        } else {
            dst_ptr = (uint8_t *)desc->dst_host;
        }
        break;

    case TU_DMA_DIR_TU_TO_HOST:
        if (desc->src_region) {
            src_ptr = tu_sram_raw_ptr(desc->src_region) + desc->src_base;
            if (desc->src_base + desc->total_bytes > desc->src_region->total_size) {
                fprintf(stderr, "DMA overflow: src=%s offset=%u + %u > %u\n",
                        desc->src_region->name,
                        desc->src_base, desc->total_bytes,
                        desc->src_region->total_size);
                return;
            }
        } else {
            src_ptr = (const uint8_t *)desc->src_host;
        }
        dst_ptr = (uint8_t *)desc->dst_host;
        break;

    case TU_DMA_DIR_TU_TO_TU:
        if (desc->src_region)
            src_ptr = tu_sram_raw_ptr(desc->src_region) + desc->src_base;
        if (desc->dst_region)
            dst_ptr = tu_sram_raw_ptr(desc->dst_region) + desc->dst_base;
        break;

    default:
        fprintf(stderr, "DMA: unknown direction %d\n", desc->direction);
        return;
    }

    if (!src_ptr || !dst_ptr) {
        fprintf(stderr, "DMA: null pointer (src=%p dst=%p)\n",
                (void*)src_ptr, (void*)dst_ptr);
        return;
    }

    /* Execute based on transfer type */
    switch (desc->type) {
    case TU_DMA_XFER_LINEAR:
        execute_linear(desc, src_ptr, dst_ptr);
        break;
    case TU_DMA_XFER_STRIDED_2D:
        execute_strided_2d(desc, src_ptr, dst_ptr);
        break;
    case TU_DMA_XFER_STRIDED_3D:
        execute_strided_3d(desc, src_ptr, dst_ptr);
        break;
    default:
        fprintf(stderr, "DMA: unknown transfer type %d\n", desc->type);
        return;
    }

    /* Update accounting */
    uint64_t transfer_cycles = TU_LATENCY_DRAM_READ;  /* base latency */
    transfer_cycles += (desc->total_bytes + TU_DMA_BUS_WIDTH_BYTES - 1) / TU_DMA_BUS_WIDTH_BYTES;

    desc->completed = true;
    desc->cycles_completed = g_tu_dma.current_cycle + transfer_cycles;

    g_tu_dma.total_bytes += desc->total_bytes;
    g_tu_dma.total_transfers++;
    g_tu_dma.estimated_cycles += transfer_cycles;

    /* Update channel counters */
    if (desc->channel < g_tu_dma.num_channels) {
        tu_dma_channel_state_t *ch = &g_tu_dma.channels[desc->channel];
        ch->total_bytes += desc->total_bytes;
        ch->total_cycles += transfer_cycles;
    }
}

/* ================================================================
 * Submission & Async Execution
 * ================================================================ */

uint32_t tu_dma_submit_desc(tu_dma_descriptor_t *desc) {
    if (!desc) return 0;

    uint32_t ch_id = desc->channel;
    if (ch_id >= g_tu_dma.num_channels) {
        fprintf(stderr, "DMA: invalid channel %u\n", ch_id);
        tu_dma_desc_destroy(desc);
        return 0;
    }

    tu_dma_channel_state_t *ch = &g_tu_dma.channels[ch_id];

    if (ch->queue_depth >= ch->max_depth && ch->max_depth > 0) {
        fprintf(stderr, "DMA: channel %u queue full (depth=%u max=%u)\n",
                ch_id, ch->queue_depth, ch->max_depth);
        tu_dma_desc_destroy(desc);
        return 0;
    }

    /* Enqueue */
    if (!ch->head) {
        ch->head = desc;
        ch->tail = desc;
    } else {
        ch->tail->next = desc;
        ch->tail = desc;
    }
    ch->queue_depth++;
    ch->total_submitted++;

    /* In synchronous mode, execute immediately.
     * Execute the entire chain until the queue is empty. */
    if (!g_tu_dma.async_mode) {
        while (ch->head) {
            tu_dma_descriptor_t *next = ch->head->next;
            tu_dma_execute_desc(ch->head);
            ch->head = next;
            ch->queue_depth--;
            ch->total_completed++;
        }
        ch->tail = NULL;
    }

    return desc->desc_id;
}

int tu_dma_tick(void) {
    g_tu_dma.current_cycle++;
    int completed = 0;

    for (uint32_t i = 0; i < g_tu_dma.num_channels; i++) {
        tu_dma_channel_state_t *ch = &g_tu_dma.channels[i];

        /* If there's an active descriptor, it completes this tick */
        if (ch->active && ch->active->cycles_completed <= g_tu_dma.current_cycle) {
            ch->active = NULL;
            ch->total_completed++;
            completed++;
        }

        /* Dequeue next descriptor if channel is idle */
        if (!ch->active && ch->head) {
            ch->active = ch->head;
            ch->head = ch->head->next;
            ch->queue_depth--;
            tu_dma_execute_desc(ch->active);
        }
    }

    return completed;
}

void tu_dma_flush_channel(uint8_t channel) {
    if (channel >= g_tu_dma.num_channels) return;

    tu_dma_channel_state_t *ch = &g_tu_dma.channels[channel];

    /* Execute all pending descriptors synchronously */
    while (ch->head) {
        tu_dma_descriptor_t *next = ch->head->next;
        tu_dma_execute_desc(ch->head);
        ch->head = next;
        ch->queue_depth--;
        ch->total_completed++;
    }
    ch->tail = NULL;

    /* Execute active descriptor if any */
    if (ch->active) {
        tu_dma_execute_desc(ch->active);
        ch->active = NULL;
        ch->total_completed++;
    }
}

void tu_dma_flush_all(void) {
    for (uint32_t i = 0; i < g_tu_dma.num_channels; i++) {
        tu_dma_flush_channel((uint8_t)i);
    }
}

/* ================================================================
 * Legacy API (backward compat)
 * ================================================================ */

void tu_dma_load(tu_dma_channel_t ch, tu_sram_region_t *dst,
                 uint32_t offset, const void *host, uint32_t bytes)
{
    if (offset + bytes > dst->total_size) {
        fprintf(stderr, "DMA load overflow: ch=%d off=%u size=%u/%u\n",
                ch, offset, bytes, dst->total_size);
        abort();
    }
    memcpy(dst->banks.data + offset, host, bytes);
    g_tu_dma.total_bytes += bytes;
    g_tu_dma.total_transfers++;
    g_tu_dma.estimated_cycles += TU_LATENCY_DRAM_READ;
    g_tu_dma.estimated_cycles += (bytes + TU_DMA_BUS_WIDTH_BYTES - 1) / TU_DMA_BUS_WIDTH_BYTES;
}

void tu_dma_store(tu_dma_channel_t ch, tu_sram_region_t *src,
                  uint32_t offset, void *host, uint32_t bytes)
{
    (void)ch;
    if (offset + bytes > src->total_size) {
        fprintf(stderr, "DMA store overflow: ch=%d off=%u size=%u/%u\n",
                ch, offset, bytes, src->total_size);
        abort();
    }
    memcpy(host, src->banks.data + offset, bytes);
    g_tu_dma.total_bytes += bytes;
    g_tu_dma.total_transfers++;
    g_tu_dma.estimated_cycles += TU_LATENCY_DRAM_WRITE;
    g_tu_dma.estimated_cycles += (bytes + TU_DMA_BUS_WIDTH_BYTES - 1) / TU_DMA_BUS_WIDTH_BYTES;
}

void tu_dma_sync(void) {
    tu_dma_flush_all();
}

void tu_dma_print_stats(void) {
    fprintf(stderr, "  DMA: %lu bytes, %lu transfers, %lu cycles\n",
            g_tu_dma.total_bytes, g_tu_dma.total_transfers,
            g_tu_dma.estimated_cycles);

    for (uint32_t i = 0; i < g_tu_dma.num_channels; i++) {
        tu_dma_channel_state_t *ch = &g_tu_dma.channels[i];
        if (ch->total_submitted > 0) {
            fprintf(stderr, "    ch%u: submitted=%lu completed=%lu bytes=%lu cycles=%lu\n",
                    i, ch->total_submitted, ch->total_completed,
                    ch->total_bytes, ch->total_cycles);
        }
    }
}
