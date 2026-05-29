/*
 * TU CModel — Normalization Engine
 * ===================================
 * Gap O5: LayerNorm and RMSNorm with online statistics computation.
 *
 * Architecture:
 *   Normalization operates on FP32 data resident in SRAM (typically
 *   the output/accumulator buffer after a GEMM). Two-pass algorithms
 *   compute statistics online to avoid storing intermediate data:
 *
 *   LayerNorm (two-pass):
 *     Pass 1: Compute mean μ = Σx_i / N, variance σ² = Σ(x_i - μ)² / N
 *     Pass 2: y_i = (x_i - μ) / √(σ² + ε) · γ + β
 *
 *   RMSNorm (two-pass):
 *     Pass 1: Compute mean-of-squares: rms² = Σx_i² / N
 *     Pass 2: y_i = x_i / √(rms² + ε) · γ  (no bias)
 *
 *   Scale (γ) and bias (β) are per-element FP32 parameters stored
 *   inline or in a separate SRAM region. ε prevents division by zero.
 *
 *   Both ops support:
 *     - Configurable axis: normalize over last dimension (default)
 *       or normalize entire tensor
 *     - Fused scale/bias: avoids separate elementwise pass
 *     - Bandwidth-aware: models SRAM read/write stall cycles
 *     - Numerical stability: max-subtract in LayerNorm variance to
 *       reduce catastrophic cancellation
 */

#ifndef TU_NORMALIZATION_ENGINE_H
#define TU_NORMALIZATION_ENGINE_H

#include "../tu_config.h"
#include "../tu_sram.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Normalization mode ---- */
typedef enum {
    TU_NORM_LAYER_NORM = 0,   /* (x - μ) / √(σ² + ε) · γ + β */
    TU_NORM_RMS_NORM   = 1,   /* x / √(rms² + ε) · γ */
    TU_NORM_BATCH_NORM = 2,   /* (x - μ) / √(σ² + ε) · γ + β (batch axis) */
} tu_norm_mode_t;

/* ---- Descriptor ---- */
typedef struct {
    tu_norm_mode_t  mode;           /* LayerNorm or RMSNorm */

    /* Data location */
    tu_sram_region_t *data_sram;    /* SRAM region containing input data */
    uint32_t         data_offset;   /* Byte offset of data in SRAM */
    uint32_t         elem_count;    /* Number of FP32 elements to normalize */

    /* Per-element parameters (scale γ, bias β) */
    tu_sram_region_t *gamma_sram;   /* SRAM region for gamma (NULL = use default γ=1) */
    uint32_t         gamma_offset;  /* Byte offset of gamma */
    tu_sram_region_t *beta_sram;    /* SRAM region for beta (NULL = use default β=0) */
    uint32_t         beta_offset;   /* Byte offset of beta */

    /* Normalization parameters */
    float            epsilon;       /* Small constant to avoid division by zero (default 1e-5) */
    uint32_t         norm_axis_dim; /* Dimension size for per-row normalization (0 = whole tensor) */

    /* Output configuration */
    bool             in_place;      /* true = write result back to data_sram */
    uint32_t         out_offset;    /* Output offset if !in_place */

    /* Statistics output (optional) */
    float           *mean_out;      /* If non-NULL, store computed mean */
    float           *var_out;       /* If non-NULL, store computed variance/rms² */
} tu_norm_desc_t;

/* ---- API ---- */

/*
 * Execute normalization (LayerNorm or RMSNorm).
 *
 * The engine performs a two-pass algorithm over FP32 data in SRAM:
 *   1. Compute statistics (mean/variance or rms²)
 *   2. Normalize, scale, and bias each element
 *
 * Returns total stall cycles incurred from SRAM bandwidth contention.
 * Returns UINT64_MAX on error (check stderr for details).
 */
uint64_t tu_norm_execute(const tu_norm_desc_t *desc);

/*
 * Convenience: LayerNorm on FP32 data in SRAM.
 *
 *   data_sram:  SRAM containing input data
 *   offset:     byte offset of first element
 *   elem_count: number of FP32 elements
 *   gamma:      scale array (FP32, one per element; NULL = identity)
 *   beta:       bias array (FP32, one per element; NULL = zero)
 *   epsilon:    small constant for numerical stability
 *   in_place:   overwrite input with output
 *
 * Returns total stall cycles.
 */
uint64_t tu_layernorm(tu_sram_region_t *data_sram, uint32_t offset,
                      uint32_t elem_count,
                      const float *gamma, const float *beta,
                      float epsilon, bool in_place);

/*
 * Convenience: RMSNorm on FP32 data in SRAM.
 *
 * Same parameters as tu_layernorm() except beta is not used
 * (RMSNorm has no bias term).
 */
uint64_t tu_rmsnorm(tu_sram_region_t *data_sram, uint32_t offset,
                    uint32_t elem_count,
                    const float *gamma, float epsilon, bool in_place);

/*
 * Per-row LayerNorm: normalize each row of a 2D tensor independently.
 *
 *   data_sram:  SRAM containing row-major FP32 data
 *   offset:     byte offset
 *   rows:       number of rows
 *   cols:       elements per row (normalized dimension)
 *   gamma:      scale per column (FP32, cols elements; NULL = ones)
 *   beta:       bias per column (FP32, cols elements; NULL = zeros)
 *   epsilon:    small constant
 */uint64_t tu_layernorm_2d(tu_sram_region_t *data_sram, uint32_t offset,
                         uint32_t rows, uint32_t cols,
                         const float *gamma, const float *beta,
                         float epsilon);

/*
 * Per-row RMSNorm: normalize each row independently.
 */
uint64_t tu_rmsnorm_2d(tu_sram_region_t *data_sram, uint32_t offset,
                       uint32_t rows, uint32_t cols,
                       const float *gamma, float epsilon);

/* ---- Utility ---- */

/* Get mode name as string */
const char *tu_norm_mode_name(tu_norm_mode_t mode);

/* Validate a descriptor (returns false on error, prints to stderr) */
bool tu_norm_validate_desc(const tu_norm_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif /* TU_NORMALIZATION_ENGINE_H */
