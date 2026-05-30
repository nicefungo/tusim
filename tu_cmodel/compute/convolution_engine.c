/*
 * TinyTU Convolution Engine — Implementation
 * ============================================
 * Gap O2: Production-grade convolution via im2col + GEMM for systolic arrays.
 *
 * All convolution modes (standard, depthwise, grouped) are implemented
 * through im2col transformation followed by matrix multiply. This naturally
 * maps to the TU systolic array (GEMM engine).
 */

#include "convolution_engine.h"
#include "../tu_config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ================================================================
 * Dimension Computation
 * ================================================================ */

int tu_conv_compute_dims(tu_conv_desc_t *desc) {
    int64_t oh = ((int64_t)desc->in_height + desc->pad_t + desc->pad_b
                  - desc->dilation_h * (desc->kernel_h - 1) - 1) / desc->stride_h + 1;
    int64_t ow = ((int64_t)desc->in_width + desc->pad_l + desc->pad_r
                  - desc->dilation_w * (desc->kernel_w - 1) - 1) / desc->stride_w + 1;

    if (oh <= 0 || ow <= 0) return -1;
    if (desc->in_channels % desc->groups != 0) return -1;
    if (desc->out_channels % desc->groups != 0) return -1;

    desc->out_height   = (uint32_t)oh;
    desc->out_width    = (uint32_t)ow;
    desc->im2col_rows  = (desc->in_channels / desc->groups) * desc->kernel_h * desc->kernel_w;
    desc->im2col_cols  = desc->out_height * desc->out_width;
    return 0;
}

/* ================================================================
 * Im2Col — NHWC Input
 * ================================================================ */

static void im2col_copy_elem(const void *src, void *dst, uint32_t elem_size) {
    switch (elem_size) {
    case 1: *(uint8_t*)dst  = *(const uint8_t*)src;  break;
    case 2: *(uint16_t*)dst = *(const uint16_t*)src; break;
    case 4: *(uint32_t*)dst = *(const uint32_t*)src; break;
    case 8: *(uint64_t*)dst = *(const uint64_t*)src; break;
    default: memcpy(dst, src, elem_size); break;
    }
}

void tu_im2col_nhwc(const void *input,
                     void *output,
                     const tu_conv_desc_t *desc,
                     uint32_t elem_size) {
    uint32_t C        = desc->in_channels;
    uint32_t H        = desc->in_height;
    uint32_t W        = desc->in_width;
    uint32_t R        = desc->kernel_h;
    uint32_t S        = desc->kernel_w;
    uint32_t sh       = desc->stride_h;
    uint32_t sw       = desc->stride_w;
    uint32_t pt       = desc->pad_t;
    uint32_t pl       = desc->pad_l;
    uint32_t dh       = desc->dilation_h;
    uint32_t dw       = desc->dilation_w;
    uint32_t oh       = desc->out_height;
    uint32_t ow       = desc->out_width;
    uint32_t groups   = desc->groups;
    uint32_t c_per_g  = C / groups;

    uint32_t im2col_row_stride = desc->im2col_cols * elem_size;
    uint32_t input_row_stride  = W * C * elem_size;
    uint32_t group_row_offset  = c_per_g * R * S * desc->im2col_cols * elem_size;

    uint8_t *out8 = (uint8_t *)output;

    for (uint32_t g = 0; g < groups; g++) {
        uint32_t g_c_start = g * c_per_g;
        uint8_t *g_out = out8 + g * group_row_offset;

        for (uint32_t c = 0; c < c_per_g; c++) {
            uint32_t chan = g_c_start + c;
            for (uint32_t r = 0; r < R; r++) {
                for (uint32_t s = 0; s < S; s++) {
                    uint32_t row = c * R * S + r * S + s;
                    uint8_t *row_out = g_out + row * im2col_row_stride;

                    for (uint32_t oy = 0; oy < oh; oy++) {
                        for (uint32_t ox = 0; ox < ow; ox++) {
                            int32_t iy = (int32_t)(oy * sh + r * dh) - (int32_t)pt;
                            int32_t ix = (int32_t)(ox * sw + s * dw) - (int32_t)pl;

                            uint8_t *col_out = row_out + (oy * ow + ox) * elem_size;

                            if (iy >= 0 && iy < (int32_t)H && ix >= 0 && ix < (int32_t)W) {
                                const uint8_t *in_ptr = (const uint8_t *)input
                                    + (uint32_t)iy * input_row_stride
                                    + (uint32_t)ix * C * elem_size
                                    + chan * elem_size;
                                im2col_copy_elem(in_ptr, col_out, elem_size);
                            } else {
                                memset(col_out, 0, elem_size);
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ================================================================
 * Im2Col — NCHW Input
 * ================================================================ */

void tu_im2col_nchw(const void *input,
                     void *output,
                     const tu_conv_desc_t *desc,
                     uint32_t elem_size) {
    uint32_t C        = desc->in_channels;
    uint32_t H        = desc->in_height;
    uint32_t W        = desc->in_width;
    uint32_t R        = desc->kernel_h;
    uint32_t S        = desc->kernel_w;
    uint32_t sh       = desc->stride_h;
    uint32_t sw       = desc->stride_w;
    uint32_t pt       = desc->pad_t;
    uint32_t pl       = desc->pad_l;
    uint32_t dh       = desc->dilation_h;
    uint32_t dw       = desc->dilation_w;
    uint32_t oh       = desc->out_height;
    uint32_t ow       = desc->out_width;
    uint32_t groups   = desc->groups;
    uint32_t c_per_g  = C / groups;

    uint32_t im2col_row_stride = desc->im2col_cols * elem_size;
    uint32_t input_chan_stride = H * W * elem_size;
    uint32_t group_row_offset  = c_per_g * R * S * desc->im2col_cols * elem_size;

    uint8_t *out8 = (uint8_t *)output;

    for (uint32_t g = 0; g < groups; g++) {
        uint32_t g_c_start = g * c_per_g;
        uint8_t *g_out = out8 + g * group_row_offset;

        for (uint32_t c = 0; c < c_per_g; c++) {
            uint32_t chan = g_c_start + c;
            const uint8_t *chan_ptr = (const uint8_t *)input + chan * input_chan_stride;

            for (uint32_t r = 0; r < R; r++) {
                for (uint32_t s = 0; s < S; s++) {
                    uint32_t row = c * R * S + r * S + s;
                    uint8_t *row_out = g_out + row * im2col_row_stride;

                    for (uint32_t oy = 0; oy < oh; oy++) {
                        for (uint32_t ox = 0; ox < ow; ox++) {
                            int32_t iy = (int32_t)(oy * sh + r * dh) - (int32_t)pt;
                            int32_t ix = (int32_t)(ox * sw + s * dw) - (int32_t)pl;

                            uint8_t *col_out = row_out + (oy * ow + ox) * elem_size;

                            if (iy >= 0 && iy < (int32_t)H && ix >= 0 && ix < (int32_t)W) {
                                const uint8_t *in_ptr = chan_ptr
                                    + (uint32_t)iy * W * elem_size
                                    + (uint32_t)ix * elem_size;
                                im2col_copy_elem(in_ptr, col_out, elem_size);
                            } else {
                                memset(col_out, 0, elem_size);
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ================================================================
 * Direct Conv2D — Golden Reference (NCHW, FP32)
 * ================================================================ */

void tu_conv2d_direct_nchw_fp32(const float *input,
                                 const float *weight,
                                 const float *bias,
                                 float *output,
                                 const tu_conv_desc_t *desc) {
    uint32_t N   = desc->batch;
    uint32_t C   = desc->in_channels;
    uint32_t H   = desc->in_height;
    uint32_t W   = desc->in_width;
    uint32_t K   = desc->out_channels;
    uint32_t R   = desc->kernel_h;
    uint32_t S   = desc->kernel_w;
    uint32_t sh  = desc->stride_h;
    uint32_t sw  = desc->stride_w;
    uint32_t pt  = desc->pad_t;
    uint32_t pl  = desc->pad_l;
    uint32_t dh  = desc->dilation_h;
    uint32_t dw  = desc->dilation_w;
    uint32_t OH  = desc->out_height;
    uint32_t OW  = desc->out_width;
    uint32_t grp = desc->groups;
    uint32_t c_per_g = C / grp;
    uint32_t k_per_g = K / grp;

    size_t out_size = (size_t)N * K * OH * OW;
    memset(output, 0, out_size * sizeof(float));

    for (uint32_t g = 0; g < grp; g++) {
        uint32_t g_c_off  = g * c_per_g;
        uint32_t g_k_off  = g * k_per_g;

        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t k = 0; k < k_per_g; k++) {
                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        float sum = 0.0f;
                        uint32_t kg = g_k_off + k;

                        for (uint32_t c = 0; c < c_per_g; c++) {
                            for (uint32_t r = 0; r < R; r++) {
                                for (uint32_t s = 0; s < S; s++) {
                                    int32_t iy = (int32_t)(oh * sh + r * dh) - (int32_t)pt;
                                    int32_t ix = (int32_t)(ow * sw + s * dw) - (int32_t)pl;

                                    if (iy >= 0 && iy < (int32_t)H && ix >= 0 && ix < (int32_t)W) {
                                        uint32_t cg = g_c_off + c;
                                        sum += input[n * C * H * W + cg * H * W + (uint32_t)iy * W + (uint32_t)ix]
                                             * weight[kg * c_per_g * R * S + c * R * S + r * S + s];
                                    }
                                }
                            }
                        }

                        size_t out_idx = n * K * OH * OW + kg * OH * OW + oh * OW + ow;
                        output[out_idx] = sum;

                        if (bias)
                            output[out_idx] += bias[kg];
                    }
                }
            }
        }
    }

    /* Fused activation */
    if (desc->activation == TU_CONV_ACTIVATION_RELU) {
        for (size_t i = 0; i < out_size; i++)
            if (output[i] < 0.0f) output[i] = 0.0f;
    } else if (desc->activation == TU_CONV_ACTIVATION_RELU6) {
        for (size_t i = 0; i < out_size; i++) {
            if (output[i] < 0.0f) output[i] = 0.0f;
            if (output[i] > 6.0f) output[i] = 6.0f;
        }
    }
}

/* ================================================================
 * Direct Conv2D — Golden Reference (NHWC, FP32)
 * ================================================================ */

void tu_conv2d_direct_nhwc_fp32(const float *input,
                                 const float *weight,
                                 const float *bias,
                                 float *output,
                                 const tu_conv_desc_t *desc) {
    uint32_t N   = desc->batch;
    uint32_t C   = desc->in_channels;
    uint32_t H   = desc->in_height;
    uint32_t W   = desc->in_width;
    uint32_t K   = desc->out_channels;
    uint32_t R   = desc->kernel_h;
    uint32_t S   = desc->kernel_w;
    uint32_t sh  = desc->stride_h;
    uint32_t sw  = desc->stride_w;
    uint32_t pt  = desc->pad_t;
    uint32_t pl  = desc->pad_l;
    uint32_t dh  = desc->dilation_h;
    uint32_t dw  = desc->dilation_w;
    uint32_t OH  = desc->out_height;
    uint32_t OW  = desc->out_width;
    uint32_t grp = desc->groups;
    uint32_t c_per_g = C / grp;
    uint32_t k_per_g = K / grp;

    size_t out_size = (size_t)N * OH * OW * K;
    memset(output, 0, out_size * sizeof(float));

    for (uint32_t g = 0; g < grp; g++) {
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t k = 0; k < k_per_g; k++) {
                uint32_t kg = g * k_per_g + k;
                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        float sum = 0.0f;
                        for (uint32_t c = 0; c < c_per_g; c++) {
                            uint32_t cg = g * c_per_g + c;
                            for (uint32_t r = 0; r < R; r++) {
                                for (uint32_t s = 0; s < S; s++) {
                                    int32_t iy = (int32_t)(oh * sh + r * dh) - (int32_t)pt;
                                    int32_t ix = (int32_t)(ow * sw + s * dw) - (int32_t)pl;
                                    if (iy >= 0 && iy < (int32_t)H && ix >= 0 && ix < (int32_t)W) {
                                        sum += input[n * H * W * C + (uint32_t)iy * W * C + (uint32_t)ix * C + cg]
                                             * weight[kg * c_per_g * R * S + c * R * S + r * S + s];
                                    }
                                }
                            }
                        }
                        size_t out_idx = n * OH * OW * K + oh * OW * K + ow * K + kg;
                        output[out_idx] = sum;
                        if (bias) output[out_idx] += bias[kg];
                    }
                }
            }
        }
    }

    /* Fused activation */
    if (desc->activation == TU_CONV_ACTIVATION_RELU) {
        for (size_t i = 0; i < out_size; i++)
            if (output[i] < 0.0f) output[i] = 0.0f;
    } else if (desc->activation == TU_CONV_ACTIVATION_RELU6) {
        for (size_t i = 0; i < out_size; i++) {
            if (output[i] < 0.0f) output[i] = 0.0f;
            if (output[i] > 6.0f) output[i] = 6.0f;
        }
    }
}

/* ================================================================
 * Im2Col + GEMM Pipeline
 * ================================================================ */

static void conv_fp32_im2col_gemm(const float *input_nhwc,
                                   const float *weight_kcrs,
                                   const float *bias,
                                   float *output_nhwc,
                                   const tu_conv_desc_t *desc,
                                   float *im2col_buf) {
    uint32_t N         = desc->batch;
    uint32_t C         = desc->in_channels;
    uint32_t K         = desc->out_channels;
    uint32_t OH        = desc->out_height;
    uint32_t OW        = desc->out_width;
    uint32_t R         = desc->kernel_h;
    uint32_t S         = desc->kernel_w;
    uint32_t groups    = desc->groups;
    uint32_t c_per_g   = C / groups;
    uint32_t k_per_g   = K / groups;
    uint32_t im2col_k  = c_per_g * R * S;   /* rows in im2col matrix */
    uint32_t im2col_n  = OH * OW;           /* cols in im2col matrix */

    for (uint32_t n = 0; n < N; n++) {
        tu_im2col_nhwc(input_nhwc + n * desc->in_height * desc->in_width * C, im2col_buf, desc, sizeof(float));

        float *out_ptr = output_nhwc + n * OH * OW * K;

        for (uint32_t g = 0; g < groups; g++) {
            float *im2col_g = im2col_buf + g * c_per_g * R * S * OH * OW;
            const float *weight_g = weight_kcrs + g * k_per_g * c_per_g * R * S;

            /* GEMM: [k_per_g][im2col_n] += [k_per_g][im2col_k] × [im2col_k][im2col_n] */
            for (uint32_t k = 0; k < k_per_g; k++) {
                const float *w_row = weight_g + k * im2col_k;
                float *o_row = out_ptr + (g * k_per_g + k);

                for (uint32_t col = 0; col < im2col_n; col++) {
                    float sum = bias ? bias[g * k_per_g + k] : 0.0f;
                    const float *a_col = im2col_g + col;
                    for (uint32_t ki = 0; ki < im2col_k; ki++) {
                        sum += w_row[ki] * a_col[ki * im2col_n];
                    }
                    /* NHWC output: [H][W][K] */
                    uint32_t oh = col / OW;
                    uint32_t ow = col % OW;
                    o_row[(oh * OW + ow) * K] = sum;
                }
            }
        }
    }

    /* Fused activation */
    if (desc->activation == TU_CONV_ACTIVATION_RELU) {
        size_t total = (size_t)N * OH * OW * K;
        for (size_t i = 0; i < total; i++)
            if (output_nhwc[i] < 0.0f) output_nhwc[i] = 0.0f;
    } else if (desc->activation == TU_CONV_ACTIVATION_RELU6) {
        size_t total = (size_t)N * OH * OW * K;
        for (size_t i = 0; i < total; i++) {
            if (output_nhwc[i] < 0.0f) output_nhwc[i] = 0.0f;
            if (output_nhwc[i] > 6.0f) output_nhwc[i] = 6.0f;
        }
    }
}

void tu_conv2d_im2col_gemm(const void *input,
                            const void *weight,
                            const float *bias,
                            void *output,
                            const tu_conv_desc_t *desc,
                            void *im2col_buf,
                            uint32_t elem_size) {
    (void)elem_size;
    /* Currently FP32-only; FP16/INT8 handled via wrapper */
    conv_fp32_im2col_gemm(
        (const float *)input,
        (const float *)weight,
        bias,
        (float *)output,
        desc,
        (float *)im2col_buf);
}

/* ---- Cycle Accounting ---- */

uint64_t tu_conv_estimate_cycles(const tu_conv_desc_t *desc,
                                  uint16_t pe_rows, uint16_t pe_cols) {
    (void)desc; (void)pe_rows; (void)pe_cols;

    uint32_t C         = desc->in_channels;
    uint32_t K         = desc->out_channels;
    uint32_t R         = desc->kernel_h;
    uint32_t S         = desc->kernel_w;
    uint32_t OH        = desc->out_height;
    uint32_t OW        = desc->out_width;
    uint32_t groups    = desc->groups;
    uint32_t c_per_g   = C / groups;
    uint32_t k_per_g   = K / groups;

    uint32_t im2col_k = c_per_g * R * S;  /* K dim for GEMM */
    uint32_t im2col_n = OH * OW;          /* N dim for GEMM */

    /* Im2col overhead: reading each input element once.
     * Rough estimate: C * H * W * elem_size / bus_width cycles. */
    uint64_t im2col_cycles = (uint64_t)C * desc->in_height * desc->in_width
                             * sizeof(float) / TU_DMA_BUS_WIDTH_BYTES;

    /* GEMM cycles: per group, [k_per_g][im2col_n] × [im2col_k][im2col_n]
     * Equivalent to MMA(M=k_per_g, N=im2col_n, K=im2col_k)
     * Tiled: (M/pe_rows) × (N/pe_cols) × (K/pe_cols) tiles
     * Each tile: pipeline_depth * pe_cols fill + K compute */
    uint16_t mt = (k_per_g + pe_rows - 1) / pe_rows;
    uint16_t nt = (im2col_n + pe_cols - 1) / pe_cols;
    uint16_t kt = (im2col_k + pe_cols - 1) / pe_cols;
    uint64_t per_group = (uint64_t)mt * nt * kt
                         * (TU_PE_PIPELINE_DEPTH * pe_cols + pe_cols);

    /* Bias + activation: ~im2col_n cycles per group */
    uint64_t bias_act_cycles = (uint64_t)K * im2col_n;

    return im2col_cycles + per_group * groups + bias_act_cycles;
}
