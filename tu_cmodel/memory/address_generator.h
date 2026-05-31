/*
 * TU CModel — Hardware Address Generator (Gap M3)
 * =================================================
 * Hardware address generation for strided/block/im2col/tiled transfers.
 *
 * Production systolic accelerators (TPU, Gemmini, NVIDIA TensorCore)
 * use dedicated hardware address generation units (AGUs) to compute
 * memory addresses for complex access patterns. This avoids burdening
 * the compiler with per-element address calculation and enables the
 * DMA engine to handle arbitrary strided/patterned transfers.
 *
 * This module provides a library of address generation functions that
 * feed the DMA descriptor engine. Addresses can be generated:
 *   - STATIC: pre-compute all addresses into a buffer (for scatter/gather)
 *   - STREAMING: generate addresses one at a time (DMA descriptor metadata)
 *   - BLOCK: compute start+stride+count for bulk DMA descriptors
 *
 * Pattern catalog:
 *   1. Linear (contiguous)     — base + offset
 *   2. Strided 2D              — base + row*row_stride + col*elem_size
 *   3. Strided 3D              — base + depth*depth_stride + row*row_stride + col*elem_size
 *   4. Tiled 2D                — decompose a large 2D matrix into tiles
 *   5. Tiled 3D                — decompose a 3D volume into tiles
 *   6. im2col                  — image-to-column for convolution
 *   7. Block interleaved       — bank-interleaved addressing
 *   8. Transposed              — column-major from row-major storage
 *
 * Gap: M3 — Address generation (P1)
 * Dependencies: dma_descriptor.h, tu_config.h
 */

#ifndef TU_ADDRESS_GENERATOR_H
#define TU_ADDRESS_GENERATOR_H

#include "../tu_config.h"
#include "../dma_descriptor.h"
#include "../tu_sram.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Address generation modes ---- */
typedef enum {
    TU_AGEN_MODE_LINEAR       = 0,  /* Contiguous, base + offset */
    TU_AGEN_MODE_STRIDED_2D   = 1,  /* Row-major 2D with row stride */
    TU_AGEN_MODE_STRIDED_3D   = 2,  /* 3D volume with depth and row strides */
    TU_AGEN_MODE_TILED_2D     = 3,  /* 2D matrix decomposed into tiles */
    TU_AGEN_MODE_TILED_3D     = 4,  /* 3D volume decomposed into tiles */
    TU_AGEN_MODE_IM2COL       = 5,  /* im2col for convolution */
    TU_AGEN_MODE_BLOCK_INTERLEAVED = 6,  /* Bank-interleaved */
    TU_AGEN_MODE_TRANSPOSED   = 7,  /* Column-major access */
} tu_agen_mode_t;

/* ---- Descriptor for a single address range ---- */
typedef struct {
    uint32_t    base_addr;      /* Starting byte address */
    uint32_t    elem_size;      /* Bytes per element */
    uint32_t    total_bytes;    /* Total bytes in this range */
    uint32_t    stride;         /* Stride between consecutive elements (0=contiguous) */
} tu_agen_range_t;

/* ---- Descriptor for a tiled access pattern ---- */
typedef struct {
    uint32_t    num_tiles;      /* Total number of tiles */
    uint32_t    tiles_per_dim[3]; /* Tiles per dimension [depth, rows, cols] */
    uint32_t    tile_dims[3];   /* Elements per tile [depth, rows, cols] */
    uint32_t    total_dims[3];  /* Total elements [depth, rows, cols] */
    uint32_t    elem_size;      /* Bytes per element */
} tu_agen_tiling_t;

/* ---- Descriptor for im2col pattern ---- */
typedef struct {
    uint32_t    input_h;        /* Input height */
    uint32_t    input_w;        /* Input width */
    uint32_t    input_c;        /* Input channels */
    uint32_t    kernel_h;       /* Kernel height */
    uint32_t    kernel_w;       /* Kernel width */
    uint32_t    pad_h;          /* Padding top/bottom */
    uint32_t    pad_w;          /* Padding left/right */
    uint32_t    stride_h;       /* Vertical stride */
    uint32_t    stride_w;       /* Horizontal stride */
    uint32_t    dilation_h;     /* Vertical dilation */
    uint32_t    dilation_w;     /* Horizontal dilation */
    uint32_t    output_h;       /* Output height (computed) */
    uint32_t    output_w;       /* Output width (computed) */
    uint32_t    elem_size;      /* Bytes per element */
} tu_agen_im2col_t;

/* ---- Iterator for streaming address generation ---- */
typedef struct {
    tu_agen_mode_t  mode;
    uint32_t        base_addr;
    uint32_t        elem_size;

    /* Current position */
    uint32_t        current_idx;    /* Linear index */

    /* Dimensions */
    uint32_t        dims[3];        /* [depth, rows, cols] */
    uint32_t        strides[3];     /* Strides for each dim */
    uint32_t        tile_dims[3];   /* Tile dimensions (tiled modes) */
    uint32_t        total_dims[3];  /* Total dimensions */

    /* im2col specific */
    uint32_t        kernel_h, kernel_w;
    uint32_t        pad_h, pad_w;
    uint32_t        stride_h, stride_w;
    uint32_t        dilation_h, dilation_w;
    uint32_t        input_w, input_h, input_c;
    uint32_t        output_h, output_w;

    /* Total elements */
    uint32_t        total_elements;

    /* Block interleaved */
    uint32_t        num_banks;
    uint32_t        bank_width;

} tu_agen_iterator_t;

/* ---- API: Compute Access Pattern Parameters ---- */

/*
 * Compute im2col output dimensions and total elements.
 * Fills output_h, output_w, total_elements in the descriptor.
 */
void tu_agen_im2col_dims(tu_agen_im2col_t *im2col);

/*
 * Compute tiling: given total dimensions and tile dimensions,
 * populate tiles_per_dim and num_tiles.
 */
void tu_agen_compute_tiling(tu_agen_tiling_t *tiling);

/* ---- API: Streaming Address Generator (Iterator) ---- */

/*
 * Initialize an address generator iterator.
 *
 * mode:    which addressing pattern to use
 * base:    base byte address in memory
 * config:  pattern-specific parameters (cast to appropriate type)
 *
 * Returns 0 on success, -1 on invalid parameters.
 */
int tu_agen_iterator_init(tu_agen_iterator_t *it,
                           tu_agen_mode_t mode,
                           uint32_t base_addr,
                           const void *config);

/*
 * Get the next address from the iterator.
 * Returns the byte address, or UINT32_MAX at end of sequence.
 */
uint32_t tu_agen_next(tu_agen_iterator_t *it);

/*
 * Check if the iterator has more addresses.
 */
bool tu_agen_has_next(const tu_agen_iterator_t *it);

/*
 * Reset the iterator to the beginning.
 */
void tu_agen_reset(tu_agen_iterator_t *it);

/*
 * Get the total number of addresses this iterator will produce.
 */
uint32_t tu_agen_total(const tu_agen_iterator_t *it);

/* ---- API: Bulk Address Generation ---- */

/*
 * Generate all addresses into a pre-allocated buffer.
 * buffer must have room for tu_agen_total() entries.
 *
 * Returns the number of addresses generated.
 */
uint32_t tu_agen_generate_all(tu_agen_iterator_t *it,
                               uint32_t *addr_buffer,
                               uint32_t buffer_size);

/*
 * Generate contiguous ranges for DMA descriptor creation.
 * Groups consecutive addresses into ranges with stride=1.
 *
 * ranges:     output buffer for range descriptors
 * max_ranges: capacity of ranges buffer
 *
 * Returns the number of ranges produced.
 */
uint32_t tu_agen_generate_ranges(tu_agen_iterator_t *it,
                                  tu_agen_range_t *ranges,
                                  uint32_t max_ranges);

/* ---- API: Specific Pattern Address Computations ---- */

/*
 * Compute the linear index for a 2D strided access.
 * index = row * row_stride + col
 * addr  = base + index * elem_size
 */
static inline uint32_t tu_agen_addr_2d(uint32_t base, uint32_t row_stride,
                                        uint32_t row, uint32_t col,
                                        uint32_t elem_size) {
    return base + (row * row_stride + col) * elem_size;
}

/*
 * Compute the linear index for a 3D strided access.
 * index = depth * depth_stride + row * row_stride + col
 */
static inline uint32_t tu_agen_addr_3d(uint32_t base,
                                        uint32_t depth_stride, uint32_t row_stride,
                                        uint32_t depth, uint32_t row, uint32_t col,
                                        uint32_t elem_size) {
    return base + (depth * depth_stride + row * row_stride + col) * elem_size;
}

/*
 * Compute the linear index for bank-interleaved access.
 * Each bank stores every num_banks-th element.
 * For bank b, element i at: base + (i * num_banks + b) * elem_size
 */
static inline uint32_t tu_agen_addr_banked(uint32_t base,
                                            uint32_t num_banks, uint32_t bank,
                                            uint32_t element_idx, uint32_t elem_size) {
    return base + (element_idx * num_banks + bank) * elem_size;
}

/*
 * Compute the linear index for a tiled 2D access.
 * Given tile (tile_row, tile_col) of size (tile_h × tile_w)
 * within a matrix of total (total_h × total_w):
 */
static inline uint32_t tu_agen_addr_tile_start(uint32_t base,
                                                uint32_t total_w,
                                                uint32_t tile_row, uint32_t tile_col,
                                                uint32_t tile_h, uint32_t tile_w,
                                                uint32_t elem_size) {
    /* Start of tile = base + (tile_row * tile_h * total_w + tile_col * tile_w) * elem_size */
    uint32_t tile_base = base + (tile_row * tile_h * total_w + tile_col * tile_w) * elem_size;
    return tile_base;
}

/*
 * Compute im2col address for a specific output position and kernel element.
 *
 * For output position (oh, ow) and kernel element (kh, kw):
 *   input_h_pos = oh * stride_h + kh * dilation_h - pad_h
 *   input_w_pos = ow * stride_w + kw * dilation_w - pad_w
 *
 * Returns UINT32_MAX if the position is in padding (address is invalid).
 */
uint32_t tu_agen_addr_im2col(const tu_agen_im2col_t *im2col,
                              uint32_t base_addr,
                              uint32_t oh, uint32_t ow,
                              uint32_t kh, uint32_t kw,
                              uint32_t ic);

/* ---- Convenience: Config Types for Iterator Initialization ---- */

/* strided 2D config */
typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride;
} tu_agen_2d_config_t;

/* tiled 2D config */
typedef struct {
    uint32_t total_rows;
    uint32_t total_cols;
    uint32_t tile_rows;
    uint32_t tile_cols;
    uint32_t num_tile_rows;
    uint32_t num_tile_cols;
} tu_agen_tiled2d_config_t;

/* ---- Convenience: Create DMA Descriptors from Patterns ---- */

/*
 * Create a DMA descriptor for a contiguous range.
 */
tu_dma_descriptor_t *tu_agen_desc_from_range(
    uint8_t channel,
    uint32_t direction,
    tu_sram_region_t *sram,
    const tu_agen_range_t *range,
    void *host_ptr);

/*
 * Create chained DMA descriptors covering all ranges from an iterator.
 */
tu_dma_descriptor_t *tu_agen_desc_chain_from_iterator(
    tu_agen_iterator_t *it,
    uint8_t channel,
    uint32_t direction,
    tu_sram_region_t *sram,
    void *host_base);

#ifdef __cplusplus
}
#endif

#endif /* TU_ADDRESS_GENERATOR_H */
