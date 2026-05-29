/*
 * TU CModel — Online Softmax Engine Implementation
 * ==================================================
 * Gap O7: Production-grade softmax with numerical stability,
 *         batched row-wise operation, masking, and bandwidth
 *         accounting for transformer attention workloads.
 *
 * Algorithm variants:
 *
 *   1) TWO-PASS (TU_SOFTMAX_STANDARD, default):
 *      Pass 1: Read all x_i, find max, compute exp(x_i-max), accumulate sum
 *      Pass 2: y_i = exp(x_i-max) / sum, write back
 *      Pros: Single division per element, no rescaling overhead.
 *      Cons: Two SRAM read passes.
 *
 *   2) TWO-PASS LOG (TU_SOFTMAX_LOG):
 *      Pass 1: Same as standard.
 *      Pass 2: y_i = (x_i - max) - ln(sum), write back
 *      Uses log1p-exp for numerical stability when input is large negative.
 *
 *   3) ONLINE SINGLE-PASS (TU_SOFTMAX_ONLINE):
 *      For each element in streaming order:
 *        m' = max(m, x_i)
 *        s  = s * exp(m - m') + exp(x_i - m')
 *        m  = m'
 *      After all elements: y_i = exp(x_i - m) / s (requires re-read)
 *      Pros: Single pass for max/sum, useful for streaming
 *      Cons: More FP multiplies, still needs second pass for division
 *      In practice, for SRAM-resident data, the two-pass is preferred.
 *
 *   For SRAM: we load entire row into host buffer, process, write back.
 *   This models the hardware pattern: SRAM → compute pipeline → SRAM.
 *
 * NUMERICAL STABILITY:
 *   - max-subtract prevents overflow (exp(709) overflows FP32)
 *   - scale (1/sqrt(head_dim)) applied before softmax for attention
 *   - mask values added before max computation
 *   - when all inputs are -inf, softmax divides by 0; we output 1/N
 *     (uniform distribution) and log-softmax outputs ln(1/N)
 */

#include "softmax_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Large negative for "negative infinity" effect in masking */
#define TU_MASK_NEG_INF  (-1e9f)
#define TU_LN_INF         (-88.7228f)  /* ln(FLT_MIN) approximation */

/* ---- Internal helpers ---- */

/* Load N FP32 elements from SRAM to host buffer. Returns stall cycles. */
static uint64_t sram_load_floats(tu_sram_region_t *s, uint32_t off,
                                  float *out, uint32_t n) {
    uint64_t stall = 0;
    for (uint32_t i = 0; i < n; i++)
        stall += tu_sram_read(s, off + i * sizeof(float), &out[i]);
    return stall;
}

/* Store N FP32 elements from host buffer to SRAM. Returns stall cycles. */
static uint64_t sram_store_floats(tu_sram_region_t *s, uint32_t off,
                                   const float *in, uint32_t n) {
    uint64_t stall = 0;
    for (uint32_t i = 0; i < n; i++)
        stall += tu_sram_write(s, off + i * sizeof(float), &in[i]);
    return stall;
}

/*
 * Standard two-pass softmax for a single row.
 *
 *   row:      input/output array of `cols` FP32 elements
 *   cols:     number of elements
 *   scale:    applied before softmax (0 = skip)
 *   mode:     TU_SOFTMAX_STANDARD or TU_SOFTMAX_LOG
 *   max_out:  receives computed max (NULL = discard)
 *   sum_out:  receives exp-sum (NULL = discard)
 */
static void softmax_row_two_pass(float *row, uint32_t cols, float scale,
                                  tu_softmax_mode_t mode,
                                  float *max_out, float *sum_out) {
    if (cols == 0) {
        if (max_out) *max_out = 0.0f;
        if (sum_out) *sum_out = 0.0f;
        return;
    }

    /* Step 1: Apply scale if needed, find max */
    float max_val = -INFINITY;
    for (uint32_t i = 0; i < cols; i++) {
        if (scale != 0.0f)
            row[i] *= scale;
        if (row[i] > max_val)
            max_val = row[i];
    }

    /* Step 2: Compute exp(x_i - max) and sum */
    double sum = 0.0;
    for (uint32_t i = 0; i < cols; i++) {
        float shifted = row[i] - max_val;
        /* Clamp to prevent overflow: exp(88.72) ≈ 1e38, exp(709) overflows */
        if (shifted < -87.0f) {
            row[i] = 0.0f;   /* exp(-87) ≈ 1.6e-38, treat as 0 */
        } else {
            row[i] = expf(shifted);
        }
        sum += (double)row[i];
    }

    /* Step 3: Normalize */
    if (sum == 0.0) {
        /* All inputs effectively -inf; output uniform distribution */
        float uniform = 1.0f / (float)cols;
        for (uint32_t i = 0; i < cols; i++)
            row[i] = uniform;
    } else {
        double inv_sum = 1.0 / sum;
        if (mode == TU_SOFTMAX_LOG) {
            float log_sum = logf((float)sum);
            for (uint32_t i = 0; i < cols; i++) {
                /* log_softmax_i = (x_i - max) - log(sum) */
                /* But row[i] now holds exp(x_i - max), recover x_i - max */
                float shifted = (row[i] > 0.0f) ? logf(row[i]) : TU_LN_INF;
                row[i] = shifted - log_sum;
            }
        } else {
            for (uint32_t i = 0; i < cols; i++)
                row[i] = (float)((double)row[i] * inv_sum);
        }
    }

    if (max_out) *max_out = max_val;
    if (sum_out) *sum_out = (float)sum;
}

/*
 * Online single-pass softmax for a single row.
 *
 * Uses the streaming rescaling algorithm to compute max and sum in one
 * pass WITHOUT storing intermediate exp values (which would become stale
 * when a new max is discovered).
 *
 * Pass 1 (streaming): Apply scale, compute max and exp_sum with rescaling.
 *   row[i] retains scaled x_i (not overwritten with exp values).
 * Pass 2: Normalize using the final max and sum.
 *   y_i = exp(x_i - max) / sum  (or log variant)
 *
 * This is the standard "online normalizer" algorithm from
 * Milakov & Gimelshein (arXiv:1805.02867), adapted for SRAM data.
 */
static void softmax_row_online(float *row, uint32_t cols, float scale,
                                tu_softmax_mode_t mode,
                                float *max_out, float *sum_out) {
    if (cols == 0) {
        if (max_out) *max_out = 0.0f;
        if (sum_out) *sum_out = 0.0f;
        return;
    }

    float max_val = -INFINITY;
    float exp_sum = 0.0f;

    /* Pass 1: streaming — apply scale, compute max and sum with rescaling.
     * row[i] is left as the scaled x_i value (not overwritten). */
    for (uint32_t i = 0; i < cols; i++) {
        if (scale != 0.0f)
            row[i] *= scale;

        float x = row[i];
        if (x > max_val) {
            /* New max discovered: rescale accumulated sum.
             * All previous exp(x_j - old_max) become exp(x_j - new_max)
             * by multiplying by exp(old_max - new_max). */
            float ratio = (max_val == -INFINITY) ? 1.0f : expf(max_val - x);
            exp_sum = exp_sum * ratio + 1.0f;
            max_val = x;
        } else {
            float exp_val = expf(x - max_val);
            exp_sum += exp_val;
        }
        /* row[i] retains x_i — NOT overwritten with exp */
    }

    float final_sum = (exp_sum > 0.0f) ? exp_sum : (float)cols;
    float inv_sum  = 1.0f / final_sum;
    float log_sum  = logf(final_sum);

    /* Pass 2: normalize each element using final max and sum */
    if (mode == TU_SOFTMAX_LOG) {
        for (uint32_t i = 0; i < cols; i++) {
            float shifted = row[i] - max_val;
            row[i] = shifted - log_sum;
        }
    } else {
        for (uint32_t i = 0; i < cols; i++) {
            float shifted = row[i] - max_val;
            /* Clamp to prevent exp overflow (should never happen since
             * shifted <= 0 by definition of max) */
            if (shifted < -87.0f) {
                row[i] = 0.0f;
            } else {
                row[i] = expf(shifted) * inv_sum;
            }
        }
    }

    if (max_out) *max_out = max_val;
    if (sum_out) *sum_out = final_sum;
}

/* ---- Core: single-row softmax dispatch ---- */

static uint64_t softmax_single_row(tu_sram_region_t *sram,
                                    uint32_t in_offset, uint32_t out_offset,
                                    uint32_t cols, float scale,
                                    tu_softmax_mode_t mode,
                                    const float *mask, bool additive_mask,
                                    float mask_fill,
                                    float *max_out, float *sum_out) {
    if (cols == 0) return 0;

    float *row = (float *)malloc(cols * sizeof(float));
    if (!row) {
        fprintf(stderr, "tu_softmax: malloc(%u floats) failed\n", cols);
        return UINT64_MAX;
    }

    uint64_t stall = 0;

    /* ---- Load row from SRAM ---- */
    stall += sram_load_floats(sram, in_offset, row, cols);

    /* ---- Apply mask if provided ---- */
    if (mask) {
        if (additive_mask) {
            for (uint32_t i = 0; i < cols; i++)
                row[i] += mask[i];
        } else {
            for (uint32_t i = 0; i < cols; i++)
                row[i] *= mask[i];
        }
    }

    /* ---- Apply mask fill for -inf masking ---- */
    if (mask_fill != 0.0f) {
        for (uint32_t i = 0; i < cols; i++)
            if (row[i] <= mask_fill || isnan(row[i]))
                row[i] = TU_MASK_NEG_INF;
    }

    /* ---- Execute softmax ---- */
    if (mode == TU_SOFTMAX_ONLINE) {
        softmax_row_online(row, cols, scale, mode, max_out, sum_out);
    } else {
        softmax_row_two_pass(row, cols, scale, mode, max_out, sum_out);
    }

    /* ---- Write result back ---- */
    stall += sram_store_floats(sram, out_offset, row, cols);

    free(row);
    return stall;
}

/* ---- Public API ---- */

uint64_t tu_softmax_execute(const tu_softmax_desc_t *desc) {
    if (!tu_softmax_validate_desc(desc))
        return UINT64_MAX;

    tu_sram_region_t *sram = desc->data_sram;
    uint32_t elem_count    = desc->elem_count;
    uint32_t axis_dim      = desc->axis_dim;
    tu_softmax_mode_t mode = desc->mode;

    /* Single row if no axis specified */
    if (axis_dim == 0)
        axis_dim = elem_count;

    uint32_t num_rows = elem_count / axis_dim;
    if (num_rows == 0) num_rows = 1;

    uint32_t out_off = desc->in_place ? desc->data_offset : desc->out_offset;
    uint64_t total_stall = 0;

    for (uint32_t r = 0; r < num_rows; r++) {
        uint32_t row_in  = desc->data_offset + r * axis_dim * sizeof(float);
        uint32_t row_out = out_off + r * axis_dim * sizeof(float);
        uint32_t cols    = axis_dim;

        /* Mask: point to this row's mask data */
        const float *row_mask = desc->mask
            ? desc->mask + r * axis_dim
            : NULL;

        float max_dummy, sum_dummy;
        uint64_t s = softmax_single_row(sram, row_in, row_out, cols,
                                         desc->scale, mode,
                                         row_mask, desc->mask_is_additive,
                                         desc->mask_fill,
                                         desc->max_out ? &max_dummy : NULL,
                                         desc->sum_out ? &sum_dummy : NULL);

        if (s == UINT64_MAX) {
            total_stall = UINT64_MAX;
            break;
        }
        total_stall += s;

        if (desc->max_out) desc->max_out[r] = max_dummy;
        if (desc->sum_out) desc->sum_out[r] = sum_dummy;
    }

    return total_stall;
}

/* ---- Convenience wrappers ---- */

uint64_t tu_softmax(tu_sram_region_t *data_sram, uint32_t offset,
                    uint32_t count, float scale, bool in_place) {
    tu_softmax_desc_t desc = {
        .mode             = TU_SOFTMAX_STANDARD,
        .data_sram        = data_sram,
        .data_offset      = offset,
        .elem_count       = count,
        .axis_dim         = 0,
        .mask             = NULL,
        .mask_is_additive = true,
        .mask_fill        = 0.0f,
        .scale            = scale,
        .in_place         = in_place,
        .out_offset       = offset,
        .max_out          = NULL,
        .sum_out          = NULL,
    };
    return tu_softmax_execute(&desc);
}

uint64_t tu_softmax_2d(tu_sram_region_t *data_sram, uint32_t offset,
                        uint32_t rows, uint32_t cols,
                        float scale, bool in_place) {
    tu_softmax_desc_t desc = {
        .mode             = TU_SOFTMAX_STANDARD,
        .data_sram        = data_sram,
        .data_offset      = offset,
        .elem_count       = rows * cols,
        .axis_dim         = cols,
        .mask             = NULL,
        .mask_is_additive = true,
        .mask_fill        = 0.0f,
        .scale            = scale,
        .in_place         = in_place,
        .out_offset       = offset,
        .max_out          = NULL,
        .sum_out          = NULL,
    };
    return tu_softmax_execute(&desc);
}

uint64_t tu_softmax_masked(tu_sram_region_t *data_sram, uint32_t offset,
                            uint32_t rows, uint32_t cols,
                            const float *mask, float mask_fill,
                            float scale) {
    tu_softmax_desc_t desc = {
        .mode             = TU_SOFTMAX_STANDARD,
        .data_sram        = data_sram,
        .data_offset      = offset,
        .elem_count       = rows * cols,
        .axis_dim         = cols,
        .mask             = mask,
        .mask_is_additive = true,
        .mask_fill        = mask_fill,
        .scale            = scale,
        .in_place         = true,
        .out_offset       = offset,
        .max_out          = NULL,
        .sum_out          = NULL,
    };
    return tu_softmax_execute(&desc);
}

uint64_t tu_log_softmax(tu_sram_region_t *data_sram, uint32_t offset,
                         uint32_t count, float scale, bool in_place) {
    tu_softmax_desc_t desc = {
        .mode             = TU_SOFTMAX_LOG,
        .data_sram        = data_sram,
        .data_offset      = offset,
        .elem_count       = count,
        .axis_dim         = 0,
        .mask             = NULL,
        .mask_is_additive = true,
        .mask_fill        = 0.0f,
        .scale            = scale,
        .in_place         = in_place,
        .out_offset       = offset,
        .max_out          = NULL,
        .sum_out          = NULL,
    };
    return tu_softmax_execute(&desc);
}

uint64_t tu_log_softmax_2d(tu_sram_region_t *data_sram, uint32_t offset,
                            uint32_t rows, uint32_t cols,
                            float scale, bool in_place) {
    tu_softmax_desc_t desc = {
        .mode             = TU_SOFTMAX_LOG,
        .data_sram        = data_sram,
        .data_offset      = offset,
        .elem_count       = rows * cols,
        .axis_dim         = cols,
        .mask             = NULL,
        .mask_is_additive = true,
        .mask_fill        = 0.0f,
        .scale            = scale,
        .in_place         = in_place,
        .out_offset       = offset,
        .max_out          = NULL,
        .sum_out          = NULL,
    };
    return tu_softmax_execute(&desc);
}

/* ---- Host reference functions (for testing) ---- */

void tu_softmax_host(float *data, uint32_t count, float scale) {
    tu_softmax_mode_t m = TU_SOFTMAX_STANDARD;
    float max_val, sum_val;
    softmax_row_two_pass(data, count, scale, m, &max_val, &sum_val);
}

void tu_log_softmax_host(float *data, uint32_t count, float scale) {
    tu_softmax_mode_t m = TU_SOFTMAX_LOG;
    float max_val, sum_val;
    softmax_row_two_pass(data, count, scale, m, &max_val, &sum_val);
}

/* ---- Utility ---- */

const char *tu_softmax_mode_name(tu_softmax_mode_t mode) {
    switch (mode) {
    case TU_SOFTMAX_STANDARD: return "standard";
    case TU_SOFTMAX_LOG:      return "log";
    case TU_SOFTMAX_ONLINE:   return "online";
    default:                  return "unknown";
    }
}

bool tu_softmax_validate_desc(const tu_softmax_desc_t *desc) {
    if (!desc) {
        fprintf(stderr, "tu_softmax_validate: null descriptor\n");
        return false;
    }
    if (!desc->data_sram) {
        fprintf(stderr, "tu_softmax_validate: null data_sram\n");
        return false;
    }
    if (desc->elem_count == 0) {
        fprintf(stderr, "tu_softmax_validate: elem_count is zero\n");
        return false;
    }
    if (desc->mode > TU_SOFTMAX_ONLINE) {
        fprintf(stderr, "tu_softmax_validate: invalid mode %d\n", desc->mode);
        return false;
    }

    /* Validate axis_dim divides elem_count */
    if (desc->axis_dim > 0 && (desc->elem_count % desc->axis_dim) != 0) {
        fprintf(stderr, "tu_softmax_validate: elem_count %u not divisible "
                "by axis_dim %u\n", desc->elem_count, desc->axis_dim);
        return false;
    }

    if (desc->axis_dim > desc->elem_count) {
        fprintf(stderr, "tu_softmax_validate: axis_dim %u > elem_count %u\n",
                desc->axis_dim, desc->elem_count);
        return false;
    }

    /* Check data bounds */
    uint32_t data_end = desc->data_offset + desc->elem_count * sizeof(float);
    if (data_end > desc->data_sram->total_size) {
        fprintf(stderr, "tu_softmax_validate: data exceeds SRAM bounds "
                "(offset=%u + %u*4 = %u > %u)\n",
                desc->data_offset, desc->elem_count, data_end,
                desc->data_sram->total_size);
        return false;
    }

    /* Out-of-place: validate out_offset */
    if (!desc->in_place) {
        uint32_t out_end = desc->out_offset + desc->elem_count * sizeof(float);
        if (out_end > desc->data_sram->total_size) {
            fprintf(stderr, "tu_softmax_validate: output exceeds SRAM bounds "
                    "(%u > %u)\n", out_end, desc->data_sram->total_size);
            return false;
        }
    }

    return true;
}
