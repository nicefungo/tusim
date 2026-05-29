/*
 * TU CModel — Normalization Engine Implementation
 * ==================================================
 * Gap O5: Production-grade LayerNorm and RMSNorm with
 *         online statistics, numerical stability, and
 *         bandwidth-aware SRAM access.
 */

#include "normalization_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Internal helpers ---- */

/* Safe load: read N FP32 elements from SRAM into a host buffer.
 * Returns total stall cycles. */
static uint64_t sram_load_floats(tu_sram_region_t *sram, uint32_t byte_offset,
                                  float *out, uint32_t count) {
    uint64_t total_stall = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t off = byte_offset + i * sizeof(float);
        uint64_t s = 0;
        tu_sram_read(sram, off, &out[i]);
        total_stall += s;
    }
    return total_stall;
}

/* Safe store: write N FP32 elements from a host buffer into SRAM.
 * Returns total stall cycles. */
static uint64_t sram_store_floats(tu_sram_region_t *sram, uint32_t byte_offset,
                                   const float *in, uint32_t count) {
    uint64_t total_stall = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t off = byte_offset + i * sizeof(float);
        uint64_t s = tu_sram_write(sram, off, &in[i]);
        total_stall += s;
    }
    return total_stall;
}

/*
 * NUMERICALLY STABLE LAYERNORM STATISTICS
 *
 * To avoid catastrophic cancellation when computing variance,
 * we use a two-pass algorithm:
 *
 *   Pass 1: Welford-like accumulation with max-shift:
 *     μ_maxshift = μ - max(x) to keep values centered near zero
 *     variance = Σ(x_i - μ)² / N
 *
 *   Actually we use the textbook two-pass for simplicity and correctness:
 *     Pass 1: μ = Σx_i / N
 *     Pass 2: σ² = Σ(x_i - μ)² / N
 *
 *   Then: y_i = (x_i - μ) / √(σ² + ε) * γ_i + β_i
 *
 * For large N, the two-pass method is stable. The subtraction (x_i - μ)
 * may lose precision for very large equal values, but that's inherent
 * to FP32 arithmetic.
 */

/* ---- Core: single-row normalization ---- */

/*
 * Normalize a single row of `cols` elements.
 *
 *   sram:       SRAM region (read source)
 *   offset:     byte offset of first element
 *   out_offset: byte offset for output (if in_place, same as offset)
 *   cols:       elements in this row
 *   gamma:      scale array (NULL = ones)
 *   beta:       bias array (NULL = zeros)
 *   epsilon:    stability constant
 *   mode:       TU_NORM_LAYER_NORM or TU_NORM_RMS_NORM
 *
 * Returns stall cycles.
 */
static uint64_t normalize_row(tu_sram_region_t *sram,
                               uint32_t offset, uint32_t out_offset,
                               uint32_t cols,
                               const float *gamma, const float *beta,
                               float epsilon, tu_norm_mode_t mode,
                               float *mean_out, float *var_out) {
    if (cols == 0) return 0;

    float *row = (float *)malloc(cols * sizeof(float));
    if (!row) {
        fprintf(stderr, "tu_norm: malloc(%u floats) failed\n", cols);
        return UINT64_MAX;
    }

    uint64_t stall = 0;

    /* ---- Pass 1: Read data and compute statistics ---- */
    stall += sram_load_floats(sram, offset, row, cols);

    double sum  = 0.0;
    double sum2 = 0.0;

    if (mode == TU_NORM_RMS_NORM) {
        /* RMSNorm: compute mean of squares */
        for (uint32_t i = 0; i < cols; i++) {
            double x = (double)row[i];
            sum2 += x * x;
        }
        double rms2 = sum2 / (double)cols;
        double inv_rms = 1.0 / sqrt(rms2 + (double)epsilon);

        /* ---- Pass 2: Apply normalization ---- */
        for (uint32_t i = 0; i < cols; i++) {
            double y = (double)row[i] * inv_rms;
            if (gamma) y *= (double)gamma[i];
            row[i] = (float)y;
        }

        if (var_out) *var_out = (float)rms2;

    } else {
        /* LayerNorm: compute mean then variance */
        for (uint32_t i = 0; i < cols; i++) {
            sum += (double)row[i];
        }
        double mean = sum / (double)cols;

        /* Second pass for variance (using already-loaded data) */
        for (uint32_t i = 0; i < cols; i++) {
            double d = (double)row[i] - mean;
            sum2 += d * d;
        }
        double var = sum2 / (double)cols;
        double inv_std = 1.0 / sqrt(var + (double)epsilon);

        /* ---- Pass 2: Apply normalization ---- */
        for (uint32_t i = 0; i < cols; i++) {
            double y = ((double)row[i] - mean) * inv_std;
            if (gamma) y *= (double)gamma[i];
            if (beta)  y += (double)beta[i];
            row[i] = (float)y;
        }

        if (mean_out) *mean_out = (float)mean;
        if (var_out)  *var_out  = (float)var;
    }

    /* ---- Write result back ---- */
    stall += sram_store_floats(sram, out_offset, row, cols);

    free(row);
    return stall;
}

/* ---- Public API ---- */

uint64_t tu_norm_execute(const tu_norm_desc_t *desc) {
    if (!tu_norm_validate_desc(desc)) {
        return UINT64_MAX;
    }

    tu_sram_region_t *sram = desc->data_sram;
    uint32_t elem_count  = desc->elem_count;
    uint32_t axis_dim    = desc->norm_axis_dim;
    tu_norm_mode_t mode  = desc->mode;

    /* If no axis specified, normalize as single row */
    if (axis_dim == 0) {
        axis_dim = elem_count;
    }

    /* Compute rows */
    uint32_t num_rows = elem_count / axis_dim;
    if (num_rows == 0) num_rows = 1;

    /* Extract gamma and beta from SRAM if needed */
    float *gamma_host = NULL;
    float *beta_host  = NULL;

    if (desc->gamma_sram) {
        gamma_host = (float *)malloc(axis_dim * sizeof(float));
        if (!gamma_host) return UINT64_MAX;
        sram_load_floats(desc->gamma_sram, desc->gamma_offset,
                          gamma_host, axis_dim);
    }

    if (desc->beta_sram && mode == TU_NORM_LAYER_NORM) {
        beta_host = (float *)malloc(axis_dim * sizeof(float));
        if (!beta_host) {
            free(gamma_host);
            return UINT64_MAX;
        }
        sram_load_floats(desc->beta_sram, desc->beta_offset,
                          beta_host, axis_dim);
    }

    uint32_t out_off = desc->in_place ? desc->data_offset : desc->out_offset;
    uint64_t total_stall = 0;

    for (uint32_t r = 0; r < num_rows; r++) {
        uint32_t row_off    = desc->data_offset + r * axis_dim * sizeof(float);
        uint32_t row_out    = out_off + r * axis_dim * sizeof(float);
        uint32_t cols       = axis_dim;
        /* Last row might be shorter if axis_dim doesn't divide evenly */
        if (r == num_rows - 1 && axis_dim * num_rows > elem_count) {
            cols = elem_count - r * axis_dim;
        }

        float mean_dummy, var_dummy;
        uint64_t s = normalize_row(sram, row_off, row_out, cols,
                                    gamma_host, beta_host,
                                    desc->epsilon, mode,
                                    desc->mean_out ? &mean_dummy : NULL,
                                    desc->var_out  ? &var_dummy  : NULL);

        if (s == UINT64_MAX) {
            total_stall = UINT64_MAX;
            break;
        }
        total_stall += s;

        /* Store per-row statistics if requested */
        if (desc->mean_out && r < num_rows) {
            desc->mean_out[r] = mean_dummy;
        }
        if (desc->var_out && r < num_rows) {
            desc->var_out[r] = var_dummy;
        }
    }

    free(gamma_host);
    free(beta_host);

    return total_stall;
}

/* ---- Convenience wrappers ---- */

uint64_t tu_layernorm(tu_sram_region_t *data_sram, uint32_t offset,
                      uint32_t elem_count,
                      const float *gamma, const float *beta,
                      float epsilon, bool in_place) {
    tu_norm_desc_t desc = {
        .mode          = TU_NORM_LAYER_NORM,
        .data_sram     = data_sram,
        .data_offset   = offset,
        .elem_count    = elem_count,
        .gamma_sram    = NULL,
        .beta_sram     = NULL,
        .epsilon       = epsilon,
        .norm_axis_dim = 0,
        .in_place      = in_place,
        .out_offset    = offset,
        .mean_out      = NULL,
        .var_out       = NULL,
    };

    /* For convenience, if gamma/beta are host pointers, write them to
     * a temporary location in data SRAM. We allocate temp space at
     * a high offset to avoid collisions. This is a pragmatic choice:
     * in production the compiler would place gamma/beta in SRAM. */
    if (gamma || beta) {
        uint32_t temp_base = data_sram->total_size - elem_count * 2 * sizeof(float);
        if (temp_base < offset + elem_count * sizeof(float)) {
            fprintf(stderr, "tu_layernorm: SRAM too small for gamma/beta temp storage\n");
            return UINT64_MAX;
        }
        if (gamma) {
            sram_store_floats(data_sram, temp_base, gamma, elem_count);
            desc.gamma_sram   = data_sram;
            desc.gamma_offset = temp_base;
        }
        if (beta) {
            sram_store_floats(data_sram, temp_base + elem_count * sizeof(float),
                              beta, elem_count);
            desc.beta_sram   = data_sram;
            desc.beta_offset = temp_base + elem_count * sizeof(float);
        }
    }

    return tu_norm_execute(&desc);
}

uint64_t tu_rmsnorm(tu_sram_region_t *data_sram, uint32_t offset,
                    uint32_t elem_count,
                    const float *gamma, float epsilon, bool in_place) {
    tu_norm_desc_t desc = {
        .mode          = TU_NORM_RMS_NORM,
        .data_sram     = data_sram,
        .data_offset   = offset,
        .elem_count    = elem_count,
        .gamma_sram    = NULL,
        .beta_sram     = NULL,
        .epsilon       = epsilon,
        .norm_axis_dim = 0,
        .in_place      = in_place,
        .out_offset    = offset,
        .mean_out      = NULL,
        .var_out       = NULL,
    };

    /* Store gamma in SRAM if provided */
    if (gamma) {
        uint32_t temp_base = data_sram->total_size - elem_count * sizeof(float);
        if (temp_base < offset + elem_count * sizeof(float)) {
            fprintf(stderr, "tu_rmsnorm: SRAM too small for gamma temp storage\n");
            return UINT64_MAX;
        }
        sram_store_floats(data_sram, temp_base, gamma, elem_count);
        desc.gamma_sram   = data_sram;
        desc.gamma_offset = temp_base;
    }

    return tu_norm_execute(&desc);
}

uint64_t tu_layernorm_2d(tu_sram_region_t *data_sram, uint32_t offset,
                         uint32_t rows, uint32_t cols,
                         const float *gamma, const float *beta,
                         float epsilon) {
    tu_norm_desc_t desc = {
        .mode          = TU_NORM_LAYER_NORM,
        .data_sram     = data_sram,
        .data_offset   = offset,
        .elem_count    = rows * cols,
        .gamma_sram    = NULL,
        .beta_sram     = NULL,
        .epsilon       = epsilon,
        .norm_axis_dim = cols,    /* Normalize each row independently */
        .in_place      = true,
        .out_offset    = offset,
        .mean_out      = NULL,
        .var_out       = NULL,
    };

    /* Use temp storage in SRAM for gamma/beta */
    if (gamma || beta) {
        uint32_t temp_base = data_sram->total_size - cols * 2 * sizeof(float);
        if (temp_base < offset + rows * cols * sizeof(float)) {
            fprintf(stderr, "tu_layernorm_2d: SRAM too small\n");
            return UINT64_MAX;
        }
        if (gamma) {
            sram_store_floats(data_sram, temp_base, gamma, cols);
            desc.gamma_sram   = data_sram;
            desc.gamma_offset = temp_base;
        }
        if (beta) {
            sram_store_floats(data_sram, temp_base + cols * sizeof(float),
                              beta, cols);
            desc.beta_sram   = data_sram;
            desc.beta_offset = temp_base + cols * sizeof(float);
        }
    }

    return tu_norm_execute(&desc);
}

uint64_t tu_rmsnorm_2d(tu_sram_region_t *data_sram, uint32_t offset,
                       uint32_t rows, uint32_t cols,
                       const float *gamma, float epsilon) {
    tu_norm_desc_t desc = {
        .mode          = TU_NORM_RMS_NORM,
        .data_sram     = data_sram,
        .data_offset   = offset,
        .elem_count    = rows * cols,
        .gamma_sram    = NULL,
        .beta_sram     = NULL,
        .epsilon       = epsilon,
        .norm_axis_dim = cols,
        .in_place      = true,
        .out_offset    = offset,
        .mean_out      = NULL,
        .var_out       = NULL,
    };

    if (gamma) {
        uint32_t temp_base = data_sram->total_size - cols * sizeof(float);
        if (temp_base < offset + rows * cols * sizeof(float)) {
            fprintf(stderr, "tu_rmsnorm_2d: SRAM too small\n");
            return UINT64_MAX;
        }
        sram_store_floats(data_sram, temp_base, gamma, cols);
        desc.gamma_sram   = data_sram;
        desc.gamma_offset = temp_base;
    }

    return tu_norm_execute(&desc);
}

/* ---- Utility ---- */

const char *tu_norm_mode_name(tu_norm_mode_t mode) {
    switch (mode) {
    case TU_NORM_LAYER_NORM: return "LayerNorm";
    case TU_NORM_RMS_NORM:   return "RMSNorm";
    case TU_NORM_BATCH_NORM: return "BatchNorm";
    default:                 return "Unknown";
    }
}

bool tu_norm_validate_desc(const tu_norm_desc_t *desc) {
    if (!desc) {
        fprintf(stderr, "tu_norm_validate: null descriptor\n");
        return false;
    }
    if (!desc->data_sram) {
        fprintf(stderr, "tu_norm_validate: null data_sram\n");
        return false;
    }
    if (desc->elem_count == 0) {
        fprintf(stderr, "tu_norm_validate: elem_count is zero\n");
        return false;
    }
    if (desc->mode != TU_NORM_LAYER_NORM &&
        desc->mode != TU_NORM_RMS_NORM &&
        desc->mode != TU_NORM_BATCH_NORM) {
        fprintf(stderr, "tu_norm_validate: invalid mode %d\n", desc->mode);
        return false;
    }
    if (desc->norm_axis_dim > desc->elem_count) {
        fprintf(stderr, "tu_norm_validate: axis_dim %u > elem_count %u\n",
                desc->norm_axis_dim, desc->elem_count);
        return false;
    }
    if (!desc->in_place) {
        /* out_offset validation would need knowledge of available SRAM */
    }

    /* Check data bounds */
    uint32_t data_end = desc->data_offset + desc->elem_count * sizeof(float);
    if (data_end > desc->data_sram->total_size) {
        fprintf(stderr, "tu_norm_validate: data exceeds SRAM bounds "
                "(offset=%u + %u*4 = %u > %u)\n",
                desc->data_offset, desc->elem_count, data_end,
                desc->data_sram->total_size);
        return false;
    }

    return true;
}
