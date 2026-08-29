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
#include "tu_status.h"
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

void tu_dma_init_config_arch(bool async, uint32_t num_channels,
                             uint32_t max_queue_depth, int bus_mode,
                             int arb_policy, int binding_policy,
                             uint32_t bus_width_bits) {
    memset(&g_tu_dma, 0, sizeof(g_tu_dma));
    g_tu_dma.async_mode = async;
    if (bus_mode != TU_DMA_BUS_MODE_INDEPENDENT &&
        bus_mode != TU_DMA_BUS_MODE_SHARED_SERIAL) {
        fprintf(stderr, "DMA: unsupported bus mode %d\n", bus_mode);
        return;
    }
    g_tu_dma.bus_mode = (tu_dma_bus_mode_t)bus_mode;
    if (arb_policy != TU_DMA_ARB_ROUND_ROBIN &&
        arb_policy != TU_DMA_ARB_STRICT_PRIORITY) {
        fprintf(stderr, "DMA: unsupported arbitration policy %d\n", arb_policy);
        return;
    }
    g_tu_dma.arb_policy = (tu_dma_arb_policy_t)arb_policy;
    if (binding_policy != TU_DMA_BIND_EXPLICIT &&
        binding_policy != TU_DMA_BIND_ROUND_ROBIN &&
        binding_policy != TU_DMA_BIND_LEAST_OUTSTANDING &&
        binding_policy != TU_DMA_BIND_LEAST_BYTES &&
        binding_policy != TU_DMA_BIND_LEAST_PROJECTED_CYCLES) {
        fprintf(stderr, "DMA: unsupported binding policy %d\n", binding_policy);
        return;
    }
    g_tu_dma.binding_policy = (tu_dma_binding_policy_t)binding_policy;
    if (bus_width_bits == 0)
        bus_width_bits = TU_DMA_BUS_WIDTH_BITS; /* zero-initialized runtime compatibility */
    if (bus_width_bits < 32 || bus_width_bits > 1024 ||
        (bus_width_bits & (bus_width_bits - 1u)) != 0) {
        fprintf(stderr, "DMA: bus width must be a power of two in [32,1024], got %u\n",
                bus_width_bits);
        return;
    }
    g_tu_dma.bus_width_bytes = bus_width_bits / 8u;
    g_tu_dma.num_channels = num_channels > 0 ? num_channels : TU_DMA_CHANNELS;
    if (g_tu_dma.num_channels > TU_DMA_ENGINE_MAX_CHANNELS) {
        fprintf(stderr, "DMA: channel count %u exceeds model capacity %u\n",
                g_tu_dma.num_channels, TU_DMA_ENGINE_MAX_CHANNELS);
        g_tu_dma.num_channels = 0; /* fail closed; never silently clamp */
        return;
    }

    for (uint32_t i = 0; i < g_tu_dma.num_channels; i++) {
        g_tu_dma.channels[i].channel_id = (uint8_t)i;
        g_tu_dma.channels[i].max_depth = max_queue_depth > 0 ? max_queue_depth : TU_DMA_MAX_OUTSTANDING;
    }
}

void tu_dma_init_config_full(bool async, uint32_t num_channels,
                             uint32_t max_queue_depth, int bus_mode,
                             int arb_policy, int binding_policy) {
    tu_dma_init_config_arch(async, num_channels, max_queue_depth, bus_mode,
                            arb_policy, binding_policy, TU_DMA_BUS_WIDTH_BITS);
}

void tu_dma_init_config_policy(bool async, uint32_t num_channels,
                               uint32_t max_queue_depth, int bus_mode,
                               int arb_policy) {
    tu_dma_init_config_full(async, num_channels, max_queue_depth, bus_mode,
                            arb_policy, TU_DMA_BIND_EXPLICIT);
}

void tu_dma_init_config(bool async, uint32_t num_channels,
                        uint32_t max_queue_depth, int bus_mode) {
    tu_dma_init_config_policy(async, num_channels, max_queue_depth, bus_mode,
                              TU_DMA_ARB_ROUND_ROBIN);
}

void tu_dma_init_full(bool async, uint32_t num_channels, uint32_t max_queue_depth) {
    tu_dma_init_config(async, num_channels, max_queue_depth,
                       TU_DMA_BUS_MODE_INDEPENDENT);
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

/* DM3: Scatter — contiguous host → scattered SRAM via index list */
tu_dma_descriptor_t *tu_dma_desc_create_scatter(
    uint8_t channel, tu_sram_region_t *dst_region,
    const void *src_host, const uint32_t *index_list,
    uint32_t elem_count, uint32_t elem_size)
{
    tu_dma_descriptor_t *desc = (tu_dma_descriptor_t *)calloc(1, sizeof(*desc));
    if (!desc) return NULL;

    static uint32_t next_id = 1000;
    desc->desc_id         = next_id++;
    desc->type            = TU_DMA_XFER_SCATTER;
    desc->direction       = TU_DMA_DIR_HOST_TO_TU;
    desc->channel         = channel;
    desc->dst_region      = dst_region;
    desc->src_host        = src_host;
    desc->index_list      = index_list;
    desc->index_count     = elem_count;
    desc->index_elem_size = elem_size;
    desc->elem_size       = elem_size;
    desc->total_bytes     = elem_count * elem_size;

    return desc;
}

/* DM3: Gather — scattered SRAM via index list → contiguous host */
tu_dma_descriptor_t *tu_dma_desc_create_gather(
    uint8_t channel, tu_sram_region_t *src_region,
    void *dst_host, const uint32_t *index_list,
    uint32_t elem_count, uint32_t elem_size)
{
    tu_dma_descriptor_t *desc = (tu_dma_descriptor_t *)calloc(1, sizeof(*desc));
    if (!desc) return NULL;

    static uint32_t next_id = 2000;
    desc->desc_id         = next_id++;
    desc->type            = TU_DMA_XFER_GATHER;
    desc->direction       = TU_DMA_DIR_TU_TO_HOST;
    desc->channel         = channel;
    desc->src_region      = src_region;
    desc->dst_host        = dst_host;
    desc->index_list      = index_list;
    desc->index_count     = elem_count;
    desc->index_elem_size = elem_size;
    desc->elem_size       = elem_size;
    desc->total_bytes     = elem_count * elem_size;

    return desc;
}

/* DM4: Multicast — single contiguous host source → multiple SRAM destinations */
tu_dma_descriptor_t *tu_dma_desc_create_multicast(
    uint8_t channel,
    const void *src_host,
    tu_sram_region_t **dst_regions, uint32_t *dst_offsets,
    uint32_t num_destinations, uint32_t elem_size, uint32_t elem_count)
{
    if (!dst_regions || !dst_offsets || num_destinations == 0 || !src_host)
        return NULL;

    tu_dma_descriptor_t *desc = calloc(1, sizeof(*desc));
    if (!desc) return NULL;

    static uint32_t next_id = 3000;
    desc->desc_id            = next_id++;
    desc->type               = TU_DMA_XFER_MULTICAST;
    desc->direction          = TU_DMA_DIR_HOST_TO_TU;
    desc->channel            = channel;
    desc->src_host           = src_host;
    desc->elem_size          = elem_size;
    desc->dims[0]            = elem_count;
    desc->dims[1]            = num_destinations;
    desc->dims[2]            = 1;
    desc->total_bytes        = elem_count * elem_size * num_destinations;

    /* Allocate destination arrays */
    desc->multicast.count  = num_destinations;
    desc->multicast.regions = calloc(num_destinations, sizeof(tu_sram_region_t*));
    desc->multicast.offsets = calloc(num_destinations, sizeof(uint32_t));
    if (!desc->multicast.regions || !desc->multicast.offsets) {
        free(desc->multicast.regions);
        free(desc->multicast.offsets);
        free(desc);
        return NULL;
    }
    memcpy(desc->multicast.regions, dst_regions, num_destinations * sizeof(tu_sram_region_t*));
    memcpy(desc->multicast.offsets, dst_offsets, num_destinations * sizeof(uint32_t));

    return desc;
}

void tu_dma_desc_destroy(tu_dma_descriptor_t *desc) {
    while (desc) {
        tu_dma_descriptor_t *next = desc->next;
        free(desc->multicast.regions);
        free(desc->multicast.offsets);
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

/* DM3: Execute scatter — src contiguous → dst scattered via index list */
static void execute_scatter(const tu_dma_descriptor_t *desc,
                            const uint8_t *src, uint8_t *dst_base)
{
    uint32_t elem_size = desc->elem_size;
    for (uint32_t i = 0; i < desc->index_count; i++) {
        uint32_t off = desc->index_list[i];
        memcpy(dst_base + off, src + i * elem_size, elem_size);
    }
}

/* DM3: Execute gather — src scattered via index list → dst contiguous */
static void execute_gather(const tu_dma_descriptor_t *desc,
                           const uint8_t *src_base, uint8_t *dst)
{
    uint32_t elem_size = desc->elem_size;
    for (uint32_t i = 0; i < desc->index_count; i++) {
        uint32_t off = desc->index_list[i];
        memcpy(dst + i * elem_size, src_base + off, elem_size);
    }
}

/* DM4: Execute multicast — single src → multiple SRAM destinations */
static void execute_multicast(const tu_dma_descriptor_t *desc)
{
    const uint8_t *src = (const uint8_t *)desc->src_host;
    uint32_t chunk_bytes = desc->dims[0] * desc->elem_size; /* elem_count * elem_size */

    for (uint32_t d = 0; d < desc->multicast.count; d++) {
        tu_sram_region_t *region = desc->multicast.regions[d];
        uint32_t offset = desc->multicast.offsets[d];
        if (!region) continue;

        /* Validate bounds per destination */
        if (offset + chunk_bytes > region->total_size) {
            fprintf(stderr, "DMA multicast overflow: dst=%s target[%u] offset=%u + %u > %u\n",
                    region->name, d, offset, chunk_bytes, region->total_size);
            continue;
        }

        uint8_t *dst = tu_sram_raw_ptr(region) + offset;
        memcpy(dst, src, chunk_bytes);
    }
}

void tu_dma_execute_desc(tu_dma_descriptor_t *desc) {
    if (!desc || desc->completed) return;

    /* Determine source and destination pointers — declare before any goto */
    const uint8_t *src_ptr = NULL;
    uint8_t       *dst_ptr = NULL;
    tu_sram_region_t *sram_region = NULL;
    bool sram_read = false, sram_write = false;

    /* DM4: Multicast has its own resolution path */
    if (desc->type == TU_DMA_XFER_MULTICAST) {
        execute_multicast(desc);
        goto accounting;
    }

    switch (desc->direction) {
    case TU_DMA_DIR_HOST_TO_TU:
        src_ptr = (const uint8_t *)desc->src_host;
        if (desc->dst_region) {
            dst_ptr = tu_sram_raw_ptr(desc->dst_region) + desc->dst_base;
            sram_region = desc->dst_region;
            sram_write = true;
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
            sram_region = desc->src_region;
            sram_read = true;
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
        if (desc->src_region) {
            src_ptr = tu_sram_raw_ptr(desc->src_region) + desc->src_base;
            sram_read = true;
        }
        if (desc->dst_region) {
            dst_ptr = tu_sram_raw_ptr(desc->dst_region) + desc->dst_base;
            sram_write = true;
        }
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
    case TU_DMA_XFER_SCATTER:
        execute_scatter(desc, src_ptr, dst_ptr);
        break;
    case TU_DMA_XFER_GATHER:
        execute_gather(desc, src_ptr, dst_ptr);
        break;
    case TU_DMA_XFER_MULTICAST:
        /* Handled above via early-exit; should not reach here */
        break;
    default:
        fprintf(stderr, "DMA: unknown transfer type %d\n", desc->type);
        return;
    }

    /* Update accounting */
accounting:
    /* For multicast, account fanout cost: N× the per-destination transfer */
    uint64_t transfer_cycles = TU_LATENCY_DRAM_READ;  /* base latency */
    transfer_cycles += (desc->total_bytes + g_tu_dma.bus_width_bytes - 1u) /
                       g_tu_dma.bus_width_bytes;

    /* M2: Account for SRAM bandwidth stalls */
    uint64_t sram_stall_cycles = 0;
    if (sram_region && sram_region->banks.bw_modeling) {
        uint32_t words = (desc->total_bytes + sram_region->banks.bank_width - 1) / sram_region->banks.bank_width;
        /* Advance cycle to trigger refill */
        tu_sram_advance_cycle(sram_region, transfer_cycles);
        /* Simulate bandwidth consumption per word */
        for (uint32_t i = 0; i < words; i++) {
            uint32_t off = (sram_read ? desc->src_base : desc->dst_base) + i * sram_region->banks.bank_width;
            uint32_t bank = tu_sram_bank_index(sram_region, off);
            tu_sram_bw_bank_t *bw = &sram_region->banks.bw_banks[bank];
            if (bw->words_available > 0) {
                bw->words_available--;
                if (sram_write) bw->writes_served++;
                else            bw->reads_served++;
            } else {
                sram_stall_cycles += sram_region->banks.stall_penalty;
                if (sram_write) bw->write_stalls++;
                else            bw->read_stalls++;
            }
        }
    }
    transfer_cycles += sram_stall_cycles;

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

static uint64_t channel_assigned_bytes(const tu_dma_channel_state_t *ch) {
    uint64_t bytes = ch->active ? ch->active->total_bytes : 0;
    for (const tu_dma_descriptor_t *d = ch->head; d; d = d->next)
        bytes += d->total_bytes;
    return bytes;
}

/* Side-effect-free queue estimate in the descriptor engine's coarse service
 * domain. The live active descriptor uses its exact scheduled completion;
 * queued work uses base latency plus payload serialization. SRAM refill
 * penalties are stateful and intentionally excluded rather than guessed. */
static uint64_t descriptor_coarse_cycles(const tu_dma_descriptor_t *desc) {
    return TU_LATENCY_DRAM_READ +
           (desc->total_bytes + g_tu_dma.bus_width_bytes - 1u) /
           g_tu_dma.bus_width_bytes;
}

static uint64_t channel_projected_cycles(const tu_dma_channel_state_t *ch) {
    uint64_t cycles = 0;
    if (ch->active && ch->active->cycles_completed > g_tu_dma.current_cycle)
        cycles = ch->active->cycles_completed - g_tu_dma.current_cycle;
    for (const tu_dma_descriptor_t *d = ch->head; d; d = d->next)
        cycles += descriptor_coarse_cycles(d);
    return cycles;
}

uint32_t tu_dma_submit_desc(tu_dma_descriptor_t *desc) {
    if (!desc) return 0;

    uint32_t ch_id = desc->channel;
    if (g_tu_dma.binding_policy == TU_DMA_BIND_ROUND_ROBIN) {
        ch_id = g_tu_dma.next_binding_channel;
    } else if (g_tu_dma.binding_policy == TU_DMA_BIND_LEAST_OUTSTANDING &&
               g_tu_dma.num_channels > 0) {
        uint32_t best_count = UINT32_MAX;
        for (uint32_t probe = 0; probe < g_tu_dma.num_channels; probe++) {
            uint32_t i = (g_tu_dma.next_binding_channel + probe) %
                         g_tu_dma.num_channels;
            tu_dma_channel_state_t *candidate = &g_tu_dma.channels[i];
            uint32_t count = candidate->queue_depth +
                             (candidate->active ? 1u : 0u);
            if (count < best_count) {
                best_count = count;
                ch_id = i;
            }
        }
    } else if (g_tu_dma.binding_policy == TU_DMA_BIND_LEAST_BYTES &&
               g_tu_dma.num_channels > 0) {
        uint64_t best_bytes = UINT64_MAX;
        for (uint32_t probe = 0; probe < g_tu_dma.num_channels; probe++) {
            uint32_t i = (g_tu_dma.next_binding_channel + probe) %
                         g_tu_dma.num_channels;
            uint64_t bytes = channel_assigned_bytes(&g_tu_dma.channels[i]);
            if (bytes < best_bytes) {
                best_bytes = bytes;
                ch_id = i;
            }
        }
    } else if (g_tu_dma.binding_policy == TU_DMA_BIND_LEAST_PROJECTED_CYCLES &&
               g_tu_dma.num_channels > 0) {
        uint64_t best_cycles = UINT64_MAX;
        for (uint32_t probe = 0; probe < g_tu_dma.num_channels; probe++) {
            uint32_t i = (g_tu_dma.next_binding_channel + probe) %
                         g_tu_dma.num_channels;
            uint64_t cycles = channel_projected_cycles(&g_tu_dma.channels[i]);
            if (cycles < best_cycles) {
                best_cycles = cycles;
                ch_id = i;
            }
        }
    }
    if (ch_id >= g_tu_dma.num_channels) {
        fprintf(stderr, "DMA: invalid channel %u\n", ch_id);
        tu_dma_desc_destroy(desc);
        return 0;
    }

    tu_dma_channel_state_t *ch = &g_tu_dma.channels[ch_id];

    uint32_t outstanding = ch->queue_depth + (ch->active ? 1u : 0u);
    if (outstanding >= ch->max_depth && ch->max_depth > 0) {
        fprintf(stderr, "DMA: channel %u outstanding limit reached (count=%u max=%u)\n",
                ch_id, outstanding, ch->max_depth);
        tu_dma_desc_destroy(desc);
        return 0;
    }

    /* Commit automatic binding state only after every rejection gate. */
    if (g_tu_dma.binding_policy != TU_DMA_BIND_EXPLICIT) {
        desc->channel = (uint8_t)ch_id;
        g_tu_dma.next_binding_channel = (ch_id + 1u) % g_tu_dma.num_channels;
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

        if (ch->active && ch->active->cycles_completed <= g_tu_dma.current_cycle) {
            ch->active = NULL;
            ch->total_completed++;
            completed++;
        }

        /* Compatibility mode: each channel has an independently serviceable
         * data path and may begin one transfer on the same model tick. */
        if (g_tu_dma.bus_mode == TU_DMA_BUS_MODE_INDEPENDENT &&
            !ch->active && ch->head) {
            ch->active = ch->head;
            ch->head = ch->head->next;
            ch->queue_depth--;
            tu_dma_execute_desc(ch->active);
        }
    }

    /* Shared-serial mode represents multiple descriptor queues feeding one
     * physical data path. Start at most one transfer, using round-robin or
     * strict descriptor priority with a rotating tie-break. */
    if (g_tu_dma.bus_mode == TU_DMA_BUS_MODE_SHARED_SERIAL) {
        bool bus_busy = false;
        for (uint32_t i = 0; i < g_tu_dma.num_channels; i++)
            bus_busy = bus_busy || g_tu_dma.channels[i].active != NULL;

        if (!bus_busy) {
            uint8_t best_priority = 0;
            if (g_tu_dma.arb_policy == TU_DMA_ARB_STRICT_PRIORITY) {
                for (uint32_t i = 0; i < g_tu_dma.num_channels; i++) {
                    tu_dma_descriptor_t *head = g_tu_dma.channels[i].head;
                    if (head && head->priority > best_priority)
                        best_priority = head->priority;
                }
            }
            for (uint32_t probe = 0; probe < g_tu_dma.num_channels; probe++) {
                uint32_t i = (g_tu_dma.next_shared_channel + probe) %
                             g_tu_dma.num_channels;
                tu_dma_channel_state_t *ch = &g_tu_dma.channels[i];
                if (!ch->head) continue;
                if (g_tu_dma.arb_policy == TU_DMA_ARB_STRICT_PRIORITY &&
                    ch->head->priority != best_priority)
                    continue;
                ch->active = ch->head;
                ch->head = ch->head->next;
                ch->queue_depth--;
                tu_dma_execute_desc(ch->active);
                g_tu_dma.next_shared_channel = (i + 1u) % g_tu_dma.num_channels;
                break;
            }
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
        TU_REPORT_ERR(TU_ERR_DMA_OVERFLOW, "DMA load exceeds SRAM capacity");
        return;
    }
    memcpy(dst->banks.data + offset, host, bytes);
    g_tu_dma.total_bytes += bytes;
    g_tu_dma.total_transfers++;
    g_tu_dma.estimated_cycles += TU_LATENCY_DRAM_READ;
    g_tu_dma.estimated_cycles += (bytes + g_tu_dma.bus_width_bytes - 1u) /
                                 g_tu_dma.bus_width_bytes;
}

void tu_dma_store(tu_dma_channel_t ch, tu_sram_region_t *src,
                  uint32_t offset, void *host, uint32_t bytes)
{
    (void)ch;
    if (offset + bytes > src->total_size) {
        fprintf(stderr, "DMA store overflow: ch=%d off=%u size=%u/%u\n",
                ch, offset, bytes, src->total_size);
        TU_REPORT_ERR(TU_ERR_DMA_OVERFLOW, "DMA store exceeds SRAM capacity");
        return;
    }
    memcpy(host, src->banks.data + offset, bytes);
    g_tu_dma.total_bytes += bytes;
    g_tu_dma.total_transfers++;
    g_tu_dma.estimated_cycles += TU_LATENCY_DRAM_WRITE;
    g_tu_dma.estimated_cycles += (bytes + g_tu_dma.bus_width_bytes - 1u) /
                                 g_tu_dma.bus_width_bytes;
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
