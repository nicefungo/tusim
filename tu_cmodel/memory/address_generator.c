/*
 * TU CModel — Hardware Address Generator Implementation (Gap M3)
 *
 * See address_generator.h for architecture documentation.
 */

#include "address_generator.h"
#include "../dma_descriptor.h"
#include "../tu_sram.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * im2col dimension computation
 * ============================================================================ */

void tu_agen_im2col_dims(tu_agen_im2col_t *im2col) {
    if (!im2col) return;

    uint32_t effective_kh = (im2col->kernel_h - 1) * im2col->dilation_h + 1;
    uint32_t effective_kw = (im2col->kernel_w - 1) * im2col->dilation_w + 1;

    im2col->output_h = (im2col->input_h + 2 * im2col->pad_h - effective_kh)
                       / im2col->stride_h + 1;
    im2col->output_w = (im2col->input_w + 2 * im2col->pad_w - effective_kw)
                       / im2col->stride_w + 1;

    if (im2col->output_h == 0) im2col->output_h = 1;
    if (im2col->output_w == 0) im2col->output_w = 1;
}

/* ============================================================================
 * Tiling computation
 * ============================================================================ */

void tu_agen_compute_tiling(tu_agen_tiling_t *tiling) {
    if (!tiling) return;

    for (int d = 0; d < 3; d++) {
        if (tiling->tile_dims[d] == 0) tiling->tile_dims[d] = 1;
        tiling->tiles_per_dim[d] = (tiling->total_dims[d] + tiling->tile_dims[d] - 1)
                                    / tiling->tile_dims[d];
    }

    tiling->num_tiles = tiling->tiles_per_dim[0] *
                        tiling->tiles_per_dim[1] *
                        tiling->tiles_per_dim[2];
}

/* ============================================================================
 * Iterator: strided 2D config
 * ============================================================================ */

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride;
} agen_2d_config_t;

/* ============================================================================
 * Iterator: tiled 2D config
 * ============================================================================ */

typedef struct {
    uint32_t total_rows;
    uint32_t total_cols;
    uint32_t tile_rows;
    uint32_t tile_cols;
    uint32_t num_tile_rows;
    uint32_t num_tile_cols;
} agen_tiled2d_config_t;

/* ============================================================================
 * Iterator initialization
 * ============================================================================ */

int tu_agen_iterator_init(tu_agen_iterator_t *it,
                           tu_agen_mode_t mode,
                           uint32_t base_addr,
                           const void *config) {
    if (!it) return -1;
    memset(it, 0, sizeof(*it));

    it->mode = mode;
    it->base_addr = base_addr;
    it->current_idx = 0;

    switch (mode) {
    case TU_AGEN_MODE_LINEAR: {
        const uint32_t *count = (const uint32_t *)config;
        uint32_t elem_count = count ? count[0] : 0;
        uint32_t elem_size  = count ? count[1] : 1;
        it->elem_size = elem_size;
        it->dims[0] = elem_count;
        it->total_elements = elem_count;
        break;
    }

    case TU_AGEN_MODE_STRIDED_2D: {
        const tu_agen_2d_config_t *cfg = (const tu_agen_2d_config_t *)config;
        if (!cfg) return -1;
        it->dims[0] = cfg->rows;
        it->dims[1] = cfg->cols;
        it->strides[0] = cfg->row_stride;
        it->total_elements = cfg->rows * cfg->cols;
        if (cfg->rows > 0 && cfg->cols > 0) it->elem_size = 4;
        break;
    }

    case TU_AGEN_MODE_STRIDED_3D: {
        const uint32_t *cfg = (const uint32_t *)config;
        if (!cfg) return -1;
        it->dims[0] = cfg[0];
        it->dims[1] = cfg[1];
        it->dims[2] = cfg[2];
        it->strides[0] = cfg[3];
        it->strides[1] = cfg[4];
        it->elem_size = cfg[5] ? cfg[5] : 4;
        it->total_elements = cfg[0] * cfg[1] * cfg[2];
        break;
    }

    case TU_AGEN_MODE_TILED_2D: {
        const agen_tiled2d_config_t *cfg = (const agen_tiled2d_config_t *)config;
        if (!cfg) return -1;
        it->total_dims[0] = cfg->total_rows;
        it->total_dims[1] = cfg->total_cols;
        it->tile_dims[0] = cfg->tile_rows;
        it->tile_dims[1] = cfg->tile_cols;
        it->dims[0] = cfg->num_tile_rows;
        it->dims[1] = cfg->num_tile_cols;
        it->total_elements = cfg->total_rows * cfg->total_cols;
        it->elem_size = 4;
        break;
    }

    case TU_AGEN_MODE_TILED_3D: {
        const uint32_t *cfg = (const uint32_t *)config;
        if (!cfg) return -1;
        it->total_dims[0] = cfg[0];
        it->total_dims[1] = cfg[1];
        it->total_dims[2] = cfg[2];
        it->tile_dims[0] = cfg[3];
        it->tile_dims[1] = cfg[4];
        it->tile_dims[2] = cfg[5];
        it->elem_size = cfg[6] ? cfg[6] : 4;
        it->dims[0] = (cfg[0] + cfg[3] - 1) / cfg[3];
        it->dims[1] = (cfg[1] + cfg[4] - 1) / cfg[4];
        it->dims[2] = (cfg[2] + cfg[5] - 1) / cfg[5];
        it->total_elements = cfg[0] * cfg[1] * cfg[2];
        break;
    }

    case TU_AGEN_MODE_IM2COL: {
        const tu_agen_im2col_t *cfg = (const tu_agen_im2col_t *)config;
        if (!cfg) return -1;
        it->input_h   = cfg->input_h;
        it->input_w   = cfg->input_w;
        it->input_c   = cfg->input_c;
        it->kernel_h  = cfg->kernel_h;
        it->kernel_w  = cfg->kernel_w;
        it->pad_h     = cfg->pad_h;
        it->pad_w     = cfg->pad_w;
        it->stride_h  = cfg->stride_h;
        it->stride_w  = cfg->stride_w;
        it->dilation_h = cfg->dilation_h;
        it->dilation_w = cfg->dilation_w;
        it->elem_size = cfg->elem_size ? cfg->elem_size : 4;

        uint32_t eff_kh = (cfg->kernel_h - 1) * cfg->dilation_h + 1;
        uint32_t eff_kw = (cfg->kernel_w - 1) * cfg->dilation_w + 1;
        it->output_h = (cfg->input_h + 2 * cfg->pad_h - eff_kh) / cfg->stride_h + 1;
        it->output_w = (cfg->input_w + 2 * cfg->pad_w - eff_kw) / cfg->stride_w + 1;
        if (it->output_h == 0) it->output_h = 1;
        if (it->output_w == 0) it->output_w = 1;

        it->total_elements = it->output_h * it->output_w *
                             cfg->kernel_h * cfg->kernel_w * cfg->input_c;
        break;
    }

    case TU_AGEN_MODE_BLOCK_INTERLEAVED: {
        const uint32_t *cfg = (const uint32_t *)config;
        if (!cfg) return -1;
        it->total_elements = cfg[0];
        it->num_banks = cfg[1] ? cfg[1] : 1;
        it->bank_width = cfg[2] ? cfg[2] : 1;
        it->elem_size = cfg[3] ? cfg[3] : 4;
        break;
    }

    case TU_AGEN_MODE_TRANSPOSED: {
        const uint32_t *cfg = (const uint32_t *)config;
        if (!cfg) return -1;
        it->dims[0] = cfg[1];  /* iterate cols first */
        it->dims[1] = cfg[0];  /* then rows */
        it->strides[0] = 1;
        it->strides[1] = cfg[0]; /* column stride = original row count */
        it->elem_size = cfg[2] ? cfg[2] : 4;
        it->total_elements = cfg[0] * cfg[1];
        break;
    }

    default:
        return -1;
    }

    return 0;
}

/* ============================================================================
 * Next address computation
 * ============================================================================ */

uint32_t tu_agen_next(tu_agen_iterator_t *it) {
    if (!it || it->current_idx >= it->total_elements) {
        return UINT32_MAX;
    }

    uint32_t addr;
    uint32_t idx = it->current_idx;
    it->current_idx++;

    switch (it->mode) {
    case TU_AGEN_MODE_LINEAR:
        addr = it->base_addr + idx * it->elem_size;
        break;

    case TU_AGEN_MODE_STRIDED_2D: {
        uint32_t row = idx / it->dims[1];
        uint32_t col = idx % it->dims[1];
        addr = it->base_addr + (row * it->strides[0] + col) * it->elem_size;
        break;
    }

    case TU_AGEN_MODE_STRIDED_3D: {
        uint32_t depth = idx / (it->dims[1] * it->dims[2]);
        uint32_t rem   = idx % (it->dims[1] * it->dims[2]);
        uint32_t row   = rem / it->dims[2];
        uint32_t col   = rem % it->dims[2];
        addr = it->base_addr + (depth * it->strides[0] + row * it->strides[1] + col) * it->elem_size;
        break;
    }

    case TU_AGEN_MODE_TILED_2D: {
        uint32_t total_cols = it->total_dims[1];
        uint32_t tile_rows  = it->tile_dims[0];
        uint32_t tile_cols  = it->tile_dims[1];
        uint32_t num_tile_cols = it->dims[1];

        uint32_t elements_per_tile = tile_rows * tile_cols;
        uint32_t tile_idx = idx / elements_per_tile;
        uint32_t elem_in_tile = idx % elements_per_tile;

        uint32_t tile_row = tile_idx / num_tile_cols;
        uint32_t tile_col = tile_idx % num_tile_cols;

        uint32_t local_row = elem_in_tile / tile_cols;
        uint32_t local_col = elem_in_tile % tile_cols;

        uint32_t global_row = tile_row * tile_rows + local_row;
        uint32_t global_col = tile_col * tile_cols + local_col;

        addr = it->base_addr + (global_row * total_cols + global_col) * it->elem_size;
        break;
    }

    case TU_AGEN_MODE_TILED_3D: {
        uint32_t total_r = it->total_dims[1];
        uint32_t total_c = it->total_dims[2];
        uint32_t tile_d  = it->tile_dims[0];
        uint32_t tile_r  = it->tile_dims[1];
        uint32_t tile_c  = it->tile_dims[2];
        uint32_t num_tile_d = it->dims[0];
        uint32_t num_tile_r = it->dims[1];

        uint32_t elems_per_tile = tile_d * tile_r * tile_c;
        uint32_t tile_idx = idx / elems_per_tile;
        uint32_t elem_in = idx % elems_per_tile;

        uint32_t t_d = tile_idx / (num_tile_r * num_tile_d);
        (void)t_d; /* unused in current linearization */
        uint32_t rem2 = tile_idx % (num_tile_r * num_tile_d);
        uint32_t t_r = rem2 / num_tile_d;
        uint32_t t_c = rem2 % num_tile_d;

        uint32_t e_d = elem_in / (tile_r * tile_c);
        uint32_t rem3 = elem_in % (tile_r * tile_c);
        uint32_t e_r = rem3 / tile_c;
        uint32_t e_c = rem3 % tile_c;

        uint32_t g_d = t_d * tile_d + e_d;
        uint32_t g_r = t_r * tile_r + e_r;
        uint32_t g_c = t_c * tile_c + e_c;

        addr = it->base_addr + (g_d * total_r * total_c + g_r * total_c + g_c) * it->elem_size;
        break;
    }

    case TU_AGEN_MODE_IM2COL: {
        uint32_t per_ic  = it->kernel_h * it->kernel_w;
        uint32_t per_ow  = per_ic * it->input_c;
        uint32_t per_oh  = per_ow * it->output_w;

        uint32_t oh = idx / per_oh;
        uint32_t r1 = idx % per_oh;
        uint32_t ow = r1 / per_ow;
        uint32_t r2 = r1 % per_ow;
        uint32_t ic = r2 % it->input_c;
        uint32_t r3 = r2 / it->input_c;
        uint32_t kh = r3 / it->kernel_w;
        uint32_t kw = r3 % it->kernel_w;

        uint32_t h_in = oh * it->stride_h + kh * it->dilation_h - it->pad_h;
        uint32_t w_in = ow * it->stride_w + kw * it->dilation_w - it->pad_w;

        if (h_in < it->input_h && w_in < it->input_w) {
            addr = it->base_addr + (ic * it->input_h * it->input_w +
                                     h_in * it->input_w + w_in) * it->elem_size;
        } else {
            addr = 0xFFFFFF00;
        }
        break;
    }

    case TU_AGEN_MODE_BLOCK_INTERLEAVED: {
        uint32_t bank = idx % it->bank_width;
        uint32_t elem = idx / it->bank_width;
        addr = it->base_addr + (elem * it->bank_width + bank) * it->elem_size;
        break;
    }

    case TU_AGEN_MODE_TRANSPOSED: {
        uint32_t col = idx / it->dims[1];
        uint32_t row = idx % it->dims[1];
        uint32_t orig_cols = it->strides[1];
        addr = it->base_addr + (row * orig_cols + col) * it->elem_size;
        break;
    }

    default:
        addr = UINT32_MAX;
        break;
    }

    return addr;
}

/* ============================================================================
 * Queries
 * ============================================================================ */

bool tu_agen_has_next(const tu_agen_iterator_t *it) {
    return it && (it->current_idx < it->total_elements);
}

void tu_agen_reset(tu_agen_iterator_t *it) {
    if (it) it->current_idx = 0;
}

uint32_t tu_agen_total(const tu_agen_iterator_t *it) {
    return it ? it->total_elements : 0;
}

/* ============================================================================
 * Bulk address generation
 * ============================================================================ */

uint32_t tu_agen_generate_all(tu_agen_iterator_t *it,
                               uint32_t *addr_buffer,
                               uint32_t buffer_size) {
    if (!it || !addr_buffer) return 0;

    uint32_t count = 0;
    tu_agen_reset(it);
    while (tu_agen_has_next(it) && count < buffer_size) {
        addr_buffer[count] = tu_agen_next(it);
        count++;
    }
    return count;
}

uint32_t tu_agen_generate_ranges(tu_agen_iterator_t *it,
                                  tu_agen_range_t *ranges,
                                  uint32_t max_ranges) {
    if (!it || !ranges || max_ranges == 0) return 0;

    tu_agen_reset(it);
    uint32_t range_count = 0;
    uint32_t total = tu_agen_total(it);

    if (total == 0) return 0;

    uint32_t prev_addr = tu_agen_next(it);
    uint32_t range_start = prev_addr;
    uint32_t range_len = 1;
    uint32_t elem_size = it->elem_size;

    for (uint32_t i = 1; i < total; i++) {
        uint32_t addr = tu_agen_next(it);

        if (addr == prev_addr + elem_size && addr != 0xFFFFFF00) {
            range_len++;
        } else {
            if (range_count < max_ranges) {
                ranges[range_count].base_addr   = range_start;
                ranges[range_count].elem_size   = elem_size;
                ranges[range_count].total_bytes = range_len * elem_size;
                ranges[range_count].stride      = elem_size;
            }
            range_count++;
            range_start = addr;
            range_len   = 1;
        }
        prev_addr = addr;
    }

    if (range_count < max_ranges) {
        ranges[range_count].base_addr   = range_start;
        ranges[range_count].elem_size   = elem_size;
        ranges[range_count].total_bytes = range_len * elem_size;
        ranges[range_count].stride      = elem_size;
    }
    range_count++;

    return range_count;
}

/* ============================================================================
 * im2col single-address computation
 * ============================================================================ */

uint32_t tu_agen_addr_im2col(const tu_agen_im2col_t *im2col,
                              uint32_t base_addr,
                              uint32_t oh, uint32_t ow,
                              uint32_t kh, uint32_t kw,
                              uint32_t ic) {
    if (!im2col) return UINT32_MAX;

    uint32_t h_in = oh * im2col->stride_h + kh * im2col->dilation_h;
    uint32_t w_in = ow * im2col->stride_w + kw * im2col->dilation_w;

    if (h_in < im2col->pad_h || w_in < im2col->pad_w) return UINT32_MAX;
    h_in -= im2col->pad_h;
    w_in -= im2col->pad_w;
    if (h_in >= im2col->input_h || w_in >= im2col->input_w) return UINT32_MAX;

    return base_addr + (ic * im2col->input_h * im2col->input_w +
                         h_in * im2col->input_w + w_in) * im2col->elem_size;
}

/* ============================================================================
 * DMA descriptor creation from address generator patterns
 * ============================================================================ */

tu_dma_descriptor_t *tu_agen_desc_from_range(
    uint8_t channel,
    uint32_t direction,
    tu_sram_region_t *sram,
    const tu_agen_range_t *range,
    void *host_ptr) {

    if (!range || range->total_bytes == 0) return NULL;

    tu_dma_descriptor_t *desc = calloc(1, sizeof(tu_dma_descriptor_t));
    if (!desc) return NULL;

    desc->type      = TU_DMA_XFER_LINEAR;
    desc->direction = direction;
    desc->channel   = channel;
    desc->elem_size = range->elem_size;
    desc->total_bytes = range->total_bytes;
    desc->dims[0]   = range->total_bytes / range->elem_size;
    desc->dims[1]   = 1;
    desc->dims[2]   = 1;

    if (direction == TU_DMA_DIR_HOST_TO_TU) {
        desc->src_host = host_ptr;
        desc->dst_region = sram;
        desc->dst_base   = range->base_addr;
    } else {
        desc->src_region = sram;
        desc->src_base   = range->base_addr;
        desc->dst_host   = host_ptr;
    }

    return desc;
}

tu_dma_descriptor_t *tu_agen_desc_chain_from_iterator(
    tu_agen_iterator_t *it,
    uint8_t channel,
    uint32_t direction,
    tu_sram_region_t *sram,
    void *host_base) {

    if (!it) return NULL;

    tu_agen_range_t ranges[128];
    uint32_t num_ranges = tu_agen_generate_ranges(it, ranges, 128);

    if (num_ranges == 0) return NULL;

    tu_dma_descriptor_t *head = NULL;
    tu_dma_descriptor_t *prev = NULL;
    uint8_t *host_ptr = (uint8_t *)host_base;

    for (uint32_t i = 0; i < num_ranges; i++) {
        tu_dma_descriptor_t *desc = tu_agen_desc_from_range(
            channel, direction, sram, &ranges[i],
            host_ptr + ranges[i].base_addr);

        if (!desc) continue;

        if (!head) head = desc;
        if (prev) prev->next = desc;
        prev = desc;
    }

    return head;
}
