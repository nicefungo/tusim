/*
 * TU CModel — Pooling Engine
 * ===========================
 * MaxPool and AveragePool 2D operations with configurable
 * kernel size, stride, and padding.
 *
 * Gap O6: Pooling operations (MaxPool, AvgPool).
 * Complements convolution engine for vision model support.
 *
 * Data layout: NCHW (batch × channels × height × width)
 * Data types: FP32 internally, configurable input/output dtype
 *
 * Reference: PyTorch nn.MaxPool2d, nn.AvgPool2d semantics
 */

#ifndef TU_POOLING_ENGINE_H
#define TU_POOLING_ENGINE_H

#include "../tu_config.h"
#include "../tu_sram.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Pooling operation type ---- */
typedef enum {
    TU_POOL_MAX = 0,
    TU_POOL_AVG = 1,
    TU_POOL_COUNT
} tu_pool_type_t;

/* ---- Pooling configuration descriptor ---- */
typedef struct {
    tu_pool_type_t  pool_type;      /* Max or Average */

    /* Input tensor (NCHW layout) */
    uint32_t        batch;          /* N: batch size */
    uint32_t        channels;       /* C: input channels */
    uint32_t        ih;             /* H: input height */
    uint32_t        iw;             /* W: input width */

    /* Kernel */
    uint32_t        kh;             /* Kernel height */
    uint32_t        kw;             /* Kernel width */

    /* Stride */
    uint32_t        sh;             /* Vertical stride */
    uint32_t        sw;             /* Horizontal stride */

    /* Padding (symmetric: applied to both sides) */
    uint32_t        ph;             /* Height padding */
    uint32_t        pw;             /* Width padding */

    /* Asymmetric padding (overrides symmetric when set) */
    bool            asym_padding;    /* Use asymmetric padding */
    uint32_t        ph_top, ph_bottom;
    uint32_t        pw_left, pw_right;

    /* Data type info */
    uint32_t        elem_size;      /* Bytes per element */
    bool            is_float;       /* true = FP32/FP16, false = integer */

    /* Output dimensions (computed) */
    uint32_t        oh;             /* Output height */
    uint32_t        ow;             /* Output width */

    /* Memory pointers */
    tu_sram_region_t *src_region;   /* Input data region */
    uint32_t        src_offset;     /* Byte offset in region */
    tu_sram_region_t *dst_region;   /* Output data region */
    uint32_t        dst_offset;     /* Byte offset in region */

    /* Count-exclude-padding flag for AvgPool */
    bool            count_include_pad;  /* true = include padded zeros in average */
} tu_pool_desc_t;

/* ---- Lifecycle ---- */

/* Compute output dimensions from input + kernel + stride + padding.
 * Returns 0 on success, -1 if dimensions are invalid. */
int tu_pool_compute_dims(tu_pool_desc_t *desc);

/* Validate a pooling descriptor. Returns 0 if valid. */
int tu_pool_validate(const tu_pool_desc_t *desc);

/* ---- Core Operation ---- */

/* Execute a pooling operation.
 * Reads input from src_region, writes output to dst_region.
 * Returns the number of compute cycles consumed, or -1 on error. */
int64_t tu_pool_execute(tu_pool_desc_t *desc);

/* ---- Sub-operations (exposed for testing) ---- */

/* Execute MaxPool on a single channel slice.
 * src: input data (FP32), ih × iw elements, row-major
 * dst: output data (FP32), oh × ow elements, row-major
 * padding: pad value (typically -inf for max, 0 for avg) */
void tu_pool_max_2d(const float *src, float *dst,
                     uint32_t ih, uint32_t iw,
                     uint32_t oh, uint32_t ow,
                     uint32_t kh, uint32_t kw,
                     uint32_t sh, uint32_t sw,
                     uint32_t ph, uint32_t pw,
                     float pad_value);

/* Execute AvgPool on a single channel slice.
 * count_include_pad: if true, divide by kh*kw always;
 *                    if false, divide by actual elements in window */
void tu_pool_avg_2d(const float *src, float *dst,
                     uint32_t ih, uint32_t iw,
                     uint32_t oh, uint32_t ow,
                     uint32_t kh, uint32_t kw,
                     uint32_t sh, uint32_t sw,
                     uint32_t ph, uint32_t pw,
                     bool count_include_pad);

#ifdef __cplusplus
}
#endif

#endif /* TU_POOLING_ENGINE_H */
