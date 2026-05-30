/*
 * TinyTU Convolution Engine — Hardware Convolution Support
 * =========================================================
 * Gap O2: Production-grade convolution for the TU cmodel.
 *
 * Implements Conv2D via im2col + GEMM mapping, the standard approach
 * for systolic-array-based accelerators (Gemmini, Eyeriss, MAERI).
 *
 * The im2col transform unrolls convolution into matrix multiplication:
 *   Conv2D(input[N][C][H][W], weight[K][C][R][S]) →
 *   im2col(input)  →  matrix A [C*R*S][H_out*W_out]
 *   weight         →  matrix B [K][C*R*S]
 *   output         →  matrix C [K][H_out*W_out] (then reshape)
 *
 * Features:
 *   - NHWC input / output format (channel-last) — GPU/TPU standard
 *   - KCRS weight format (output-channels-first)
 *   - Stride, padding, dilation support
 *   - Depthwise and grouped convolution
 *   - Fused ReLU activation on output
 *   - Bias add
 *   - Cycle accounting via im2col DMA + GEMM cycles
 *
 * Reference: Gemmini ISA, NVIDIA cuDNN im2col, Google TPU Conv.
 */

#ifndef TU_CONVOLUTION_ENGINE_H
#define TU_CONVOLUTION_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Convolution Descriptor ---- */

typedef enum {
    TU_CONV_FORMAT_NHWC = 0,  /* [batch, height, width, channels] */
    TU_CONV_FORMAT_NCHW = 1,  /* [batch, channels, height, width] */
} tu_conv_format_t;

typedef enum {
    TU_CONV_ACTIVATION_NONE = 0,
    TU_CONV_ACTIVATION_RELU = 1,
    TU_CONV_ACTIVATION_RELU6 = 2,
} tu_conv_activation_t;

typedef struct {
    /* Tensor dimensions */
    uint32_t  batch;            /* N: batch size (default 1) */
    uint32_t  in_channels;      /* C: input channels */
    uint32_t  in_height;        /* H: input height */
    uint32_t  in_width;         /* W: input width */
    uint32_t  out_channels;     /* K: output channels (filters) */

    /* Kernel */
    uint32_t  kernel_h;         /* R: filter height */
    uint32_t  kernel_w;         /* S: filter width */

    /* Stride */
    uint32_t  stride_h;
    uint32_t  stride_w;

    /* Padding (top, bottom, left, right) */
    uint32_t  pad_t;
    uint32_t  pad_b;
    uint32_t  pad_l;
    uint32_t  pad_r;

    /* Dilation */
    uint32_t  dilation_h;
    uint32_t  dilation_w;

    /* Groups (1 = standard conv, C = depthwise) */
    uint32_t  groups;

    /* Data format */
    tu_conv_format_t  input_format;

    /* Fused activation */
    tu_conv_activation_t  activation;

    /* Bias enabled? */
    bool      has_bias;

    /* Derived dimensions (computed by tu_conv_compute_dims) */
    uint32_t  out_height;
    uint32_t  out_width;
    uint32_t  im2col_rows;     /* C * kernel_h * kernel_w per group */
    uint32_t  im2col_cols;     /* out_height * out_width per group */

} tu_conv_desc_t;

/* ---- Lifecycle ---- */

/* Compute output dimensions and validate the convolution descriptor.
 * Returns 0 on success, -1 on invalid parameters. */
int tu_conv_compute_dims(tu_conv_desc_t *desc);

/* ---- Im2Col ---- */

/*
 * Perform im2col transformation on NHWC input data.
 *
 * Input:  [batch][H][W][C] — channel-last, FP32 or FP16
 * Output: [im2col_rows][im2col_cols] — row-major for GEMM
 *   im2col_rows = C * kernel_h * kernel_w (per group)
 *   im2col_cols = out_h * out_w (per group)
 *
 * If groups > 1: each group gets contiguous rows.
 * The caller allocates output buffer: im2col_rows * im2col_cols * elem_size * groups
 *
 * dtype_size: sizeof(fp16_t)=2 or sizeof(fp32_t)=4 or sizeof(int8_t)=1
 */
void tu_im2col_nhwc(const void *input,
                     void *output,
                     const tu_conv_desc_t *desc,
                     uint32_t elem_size);

/*
 * Perform im2col transformation on NCHW input data.
 * Input layout: [batch][C][H][W]
 */
void tu_im2col_nchw(const void *input,
                     void *output,
                     const tu_conv_desc_t *desc,
                     uint32_t elem_size);

/* ---- Direct Conv2D (Software Reference) ---- */

/*
 * Direct FP32 convolution — golden reference for verification.
 * Computes: output[b][k][oh][ow] = sum_c sum_r sum_s (
 *     input[b][c][oh*sh + r*dh - pt][ow*sw + s*dw - pl] * weight[k][c][r][s]
 * ) + bias[k]
 *
 * Input:  NCHW format, FP32
 * Weight: KCRS format (output-last: [K][C][R][S]), FP32
 * Output: NCHW format, FP32
 */
void tu_conv2d_direct_nchw_fp32(const float *input,
                                 const float *weight,
                                 const float *bias,
                                 float *output,
                                 const tu_conv_desc_t *desc);

/*
 * Same as above, but NHWC format.
 */
void tu_conv2d_direct_nhwc_fp32(const float *input,
                                 const float *weight,
                                 const float *bias,
                                 float *output,
                                 const tu_conv_desc_t *desc);

/* ---- Im2col + GEMM Pipeline (for TU systolic array) ---- */

/*
 * im2col + GEMM convolution using existing tu_mma infrastructure.
 *
 * This function:
 *   1. Calls im2col to transform input
 *   2. Treats weights as matrix B and im2col output as matrix A
 *   3. Calls tu_mma() for each output channel group (K * C*R*S → K *)
 *   4. Applies bias and fused activation
 *
 * The caller provides SRAM offsets and the function manages tiling.
 *
 * Parameters:
 *   im2col_buf:  scratch buffer for im2col output (im2col_rows * im2col_cols * elem_size * groups bytes)
 *   out_buf:     output buffer (K * out_h * out_w * elem_size bytes)
 *   elem_size:   bytes per element (2=FP16, 4=FP32)
 */
void tu_conv2d_im2col_gemm(const void *input,
                            const void *weight,
                            const float *bias,
                            void *output,
                            const tu_conv_desc_t *desc,
                            void *im2col_buf,
                            uint32_t elem_size);

/* ---- Cycle Accounting ---- */

/*
 * Estimate cycles for a full convolution operation.
 * Accounts for: im2col overhead, GEMM cycles (via tu_mma model),
 * bias add, activation overhead.
 */
uint64_t tu_conv_estimate_cycles(const tu_conv_desc_t *desc,
                                  uint16_t pe_rows, uint16_t pe_cols);

#ifdef __cplusplus
}
#endif

#endif /* TU_CONVOLUTION_ENGINE_H */
