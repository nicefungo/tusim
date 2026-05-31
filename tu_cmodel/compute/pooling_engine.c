/*
 * TU CModel — Pooling Engine Implementation
 * ===========================================
 *
 * Implements MaxPool2D and AvgPool2D with configurable kernel,
 * stride, padding, and data types. Operates on FP32 internally
 * for numerical stability, with input/output conversion handled
 * by the precision layer.
 *
 * Gap O6: Pooling operations (MaxPool, AvgPool)
 */

#include "pooling_engine.h"
#include "../tu_precision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ================================================================
 * Dimension Computation
 * ================================================================ */

int tu_pool_compute_dims(tu_pool_desc_t *desc) {
    if (!desc) return -1;

    /* Input dimensions must be positive */
    if (desc->ih == 0 || desc->iw == 0) {
        fprintf(stderr, "pool: invalid input dimensions %ux%u\n",
                desc->ih, desc->iw);
        return -1;
    }

    /* Kernel dimensions must be positive */
    if (desc->kh == 0 || desc->kw == 0) {
        fprintf(stderr, "pool: invalid kernel %ux%u\n", desc->kh, desc->kw);
        return -1;
    }

    /* Stride must be at least 1 */
    if (desc->sh == 0) desc->sh = 1;
    if (desc->sw == 0) desc->sw = 1;

    /* Output dimensions: OH = floor((IH + 2*PH - KH) / SH) + 1 */
    uint32_t ph_total = desc->asym_padding
        ? (desc->ph_top + desc->ph_bottom)
        : (desc->ph * 2);
    uint32_t pw_total = desc->asym_padding
        ? (desc->pw_left + desc->pw_right)
        : (desc->pw * 2);

    /* OH = ceil((IH + 2*PH - KH + 1) / SH) */
    if (desc->ih + ph_total < desc->kh) {
        fprintf(stderr, "pool: input height %u + padding %u < kernel %u\n",
                desc->ih, ph_total, desc->kh);
        return -1;
    }
    desc->oh = (desc->ih + ph_total - desc->kh) / desc->sh + 1;

    if (desc->iw + pw_total < desc->kw) {
        fprintf(stderr, "pool: input width %u + padding %u < kernel %u\n",
                desc->iw, pw_total, desc->kw);
        return -1;
    }
    desc->ow = (desc->iw + pw_total - desc->kw) / desc->sw + 1;

    return 0;
}

/* ================================================================
 * Validation
 * ================================================================ */

int tu_pool_validate(const tu_pool_desc_t *desc) {
    if (!desc) return -1;
    if (desc->pool_type >= TU_POOL_COUNT) return -1;
    if (!desc->src_region || !desc->dst_region) return -1;
    if (desc->elem_size == 0 || desc->elem_size > 8) return -1;

    /* Check input fits in source region */
    uint32_t input_size = desc->batch * desc->channels *
                          desc->ih * desc->iw * desc->elem_size;
    if (desc->src_offset + input_size > desc->src_region->total_size) {
        fprintf(stderr, "pool: src overflow %u + %u > %u\n",
                desc->src_offset, input_size, desc->src_region->total_size);
        return -1;
    }

    /* Check output fits in destination region */
    uint32_t output_size = desc->batch * desc->channels *
                           desc->oh * desc->ow * desc->elem_size;
    if (desc->dst_offset + output_size > desc->dst_region->total_size) {
        fprintf(stderr, "pool: dst overflow %u + %u > %u\n",
                desc->dst_offset, output_size, desc->dst_region->total_size);
        return -1;
    }

    return 0;
}

/* ================================================================
 * MaxPool 2D
 * ================================================================ */

void tu_pool_max_2d(const float *src, float *dst,
                     uint32_t ih, uint32_t iw,
                     uint32_t oh, uint32_t ow,
                     uint32_t kh, uint32_t kw,
                     uint32_t sh, uint32_t sw,
                     uint32_t ph, uint32_t pw,
                     float pad_value) {
    for (uint32_t oy = 0; oy < oh; oy++) {
        for (uint32_t ox = 0; ox < ow; ox++) {
            float max_val = -INFINITY;
            bool any_valid = false;

            for (uint32_t ky = 0; ky < kh; ky++) {
                for (uint32_t kx = 0; kx < kw; kx++) {
                    /* Map output (oy, ox) + kernel offset to input */
                    int32_t iy = (int32_t)(oy * sh + ky) - (int32_t)ph;
                    int32_t ix = (int32_t)(ox * sw + kx) - (int32_t)pw;

                    if (iy >= 0 && iy < (int32_t)ih &&
                        ix >= 0 && ix < (int32_t)iw) {
                        float val = src[iy * iw + ix];
                        if (val > max_val || !any_valid) {
                            max_val = val;
                            any_valid = true;
                        }
                    } else if (!any_valid && pad_value > max_val) {
                        /* In padded region: use pad_value if it beats current */
                        /* For max pool, we use -inf so pad regions are ignored
                         * unless the entire window is in the pad region */
                    }
                }
            }

            /* If no valid input covered (all padding), use pad_value */
            dst[oy * ow + ox] = any_valid ? max_val : pad_value;
        }
    }
}

/* ================================================================
 * AvgPool 2D
 * ================================================================ */

void tu_pool_avg_2d(const float *src, float *dst,
                     uint32_t ih, uint32_t iw,
                     uint32_t oh, uint32_t ow,
                     uint32_t kh, uint32_t kw,
                     uint32_t sh, uint32_t sw,
                     uint32_t ph, uint32_t pw,
                     bool count_include_pad) {
    for (uint32_t oy = 0; oy < oh; oy++) {
        for (uint32_t ox = 0; ox < ow; ox++) {
            double sum = 0.0;
            uint32_t count = count_include_pad ? (kh * kw) : 0;

            for (uint32_t ky = 0; ky < kh; ky++) {
                for (uint32_t kx = 0; kx < kw; kx++) {
                    int32_t iy = (int32_t)(oy * sh + ky) - (int32_t)ph;
                    int32_t ix = (int32_t)(ox * sw + kx) - (int32_t)pw;

                    if (iy >= 0 && iy < (int32_t)ih &&
                        ix >= 0 && ix < (int32_t)iw) {
                        sum += (double)src[iy * iw + ix];
                        if (!count_include_pad) count++;
                    }
                }
            }

            dst[oy * ow + ox] = (count > 0) ? (float)(sum / (double)count) : 0.0f;
        }
    }
}

/* ================================================================
 * Full Pooling Execution (with memory management)
 * ================================================================ */

int64_t tu_pool_execute(tu_pool_desc_t *desc) {
    if (!desc) return -1;

    /* Compute output dimensions */
    if (tu_pool_compute_dims(desc) != 0) return -1;

    /* Validate */
    if (tu_pool_validate(desc) != 0) return -1;

    /* Get raw pointers */
    uint8_t *src_raw = tu_sram_raw_ptr(desc->src_region) + desc->src_offset;
    uint8_t *dst_raw = tu_sram_raw_ptr(desc->dst_region) + desc->dst_offset;

    uint32_t spatial_in  = desc->ih * desc->iw;
    uint32_t spatial_out = desc->oh * desc->ow;
    uint32_t channel_stride_in  = spatial_in * desc->elem_size;
    uint32_t channel_stride_out = spatial_out * desc->elem_size;
    uint32_t batch_stride_in  = desc->channels * channel_stride_in;
    uint32_t batch_stride_out = desc->channels * channel_stride_out;

    /* Allocate temporary FP32 buffers for one channel */
    float *src_f32 = (float *)malloc(spatial_in * sizeof(float));
    float *dst_f32 = (float *)malloc(spatial_out * sizeof(float));
    if (!src_f32 || !dst_f32) {
        fprintf(stderr, "pool: malloc failed\n");
        free(src_f32);
        free(dst_f32);
        return -1;
    }

    float pad_value = (desc->pool_type == TU_POOL_MAX) ? -INFINITY : 0.0f;

    /* Phase knot between input and output (pipeline fill analogy) */
    /* 1 cycle per element + 1 MAC-equivalent per comparison */
    int64_t total_cycles = 0;
    int64_t ops_per_elem = (desc->pool_type == TU_POOL_MAX) ? 1 : 2; /* max=compare, avg=add+div */

    for (uint32_t b = 0; b < desc->batch; b++) {
        for (uint32_t c = 0; c < desc->channels; c++) {
            /* Copy channel from SRAM to FP32 buffer */
            uint8_t *ch_src = src_raw + b * batch_stride_in + c * channel_stride_in;
            if (desc->is_float && desc->elem_size == 4) {
                memcpy(src_f32, ch_src, spatial_in * 4);
            } else if (desc->is_float && desc->elem_size == 2) {
                /* FP16 → FP32 conversion */
                for (uint32_t i = 0; i < spatial_in; i++) {
                    uint16_t h;
                    memcpy(&h, ch_src + i * 2, 2);
                    src_f32[i] = tu_fp16_to_fp32(h);
                }
            } else {
                /* Default: byte copy to FP32 (for integer types) */
                for (uint32_t i = 0; i < spatial_in; i++) {
                    int32_t val = 0;
                    memcpy(&val, ch_src + i * desc->elem_size,
                           desc->elem_size < 4 ? desc->elem_size : 4);
                    src_f32[i] = (float)val;
                }
            }

            /* Execute pooling */
            if (desc->pool_type == TU_POOL_MAX) {
                tu_pool_max_2d(src_f32, dst_f32,
                                desc->ih, desc->iw,
                                desc->oh, desc->ow,
                                desc->kh, desc->kw,
                                desc->sh, desc->sw,
                                desc->ph, desc->pw,
                                pad_value);
            } else {
                tu_pool_avg_2d(src_f32, dst_f32,
                                desc->ih, desc->iw,
                                desc->oh, desc->ow,
                                desc->kh, desc->kw,
                                desc->sh, desc->sw,
                                desc->ph, desc->pw,
                                desc->count_include_pad);
            }

            /* Copy result back to SRAM */
            uint8_t *ch_dst = dst_raw + b * batch_stride_out + c * channel_stride_out;
            if (desc->is_float && desc->elem_size == 4) {
                memcpy(ch_dst, dst_f32, spatial_out * 4);
            } else if (desc->is_float && desc->elem_size == 2) {
                for (uint32_t i = 0; i < spatial_out; i++) {
                    uint16_t h = tu_fp32_to_fp16(dst_f32[i]);
                    memcpy(ch_dst + i * 2, &h, 2);
                }
            } else {
                for (uint32_t i = 0; i < spatial_out; i++) {
                    int32_t val = (int32_t)roundf(dst_f32[i]);
                    memcpy(ch_dst + i * desc->elem_size, &val,
                           desc->elem_size < 4 ? desc->elem_size : 4);
                }
            }

            /* Cycle accounting */
            total_cycles += (int64_t)(spatial_out * desc->kh * desc->kw * ops_per_elem);
        }
    }

    free(src_f32);
    free(dst_f32);

    /* Pipeline drain: ~kh cycles for the pipeline to flush */
    total_cycles += desc->kh;

    return total_cycles;
}
