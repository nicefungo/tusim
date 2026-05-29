/*
 * TU CModel — Online Softmax Engine
 * ===================================
 * Gap O7: Online softmax with numerical stability for attention.
 *
 * Architecture:
 *   Softmax operates on FP32 data resident in SRAM (output/accumulator
 *   buffer after GEMM). The algorithm uses max-subtract for numerical
 *   stability, critical for transformer attention where input values
 *   can be large (e.g., Q·K^T with large head_dim).
 *
 *   Algorithm (per-row, numerically stable):
 *     Pass 1: m = max(x_i)           — find maximum
 *             s = Σ exp(x_i - m)     — compute sum of shifted exps
 *     Pass 2: y_i = exp(x_i - m) / s — normalize
 *
 *   For online/streaming scenarios (attention), we support:
 *     - Per-row softmax: each row normalized independently
 *     - Online softmax (single pass with rescaling):
 *       m' = max(m, x_i)
 *       s' = s * exp(m - m') + exp(x_i - m')
 *       This avoids two passes but requires FP32 multiplies per element.
 *       The two-pass implementation is the default as it's faster in
 *       hardware (no per-element rescaling multiplication).
 *
 *   The engine supports fused computation scenarios:
 *     - Softmax only (standalone)
 *     - Log-softmax: ln(softmax(x_i))
 *     - Attention-ready: can be chained after Q·K^T and before ×V
 *
 *   Bandwidth awareness: all reads/writes go through SRAM module
 *   with bank conflict and bandwidth stall accounting.
 */

#ifndef TU_SOFTMAX_ENGINE_H
#define TU_SOFTMAX_ENGINE_H

#include "../tu_config.h"
#include "../tu_sram.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Softmax mode ---- */
typedef enum {
    TU_SOFTMAX_STANDARD  = 0,  /* Standard softmax: y_i = exp(x_i) / Σ exp(x_j) */
    TU_SOFTMAX_LOG       = 1,  /* Log-softmax:  y_i = ln(softmax(x_i)) */
    TU_SOFTMAX_ONLINE    = 2,  /* Online single-pass softmax (rescaling) */
} tu_softmax_mode_t;

/* ---- Descriptor ---- */
typedef struct {
    tu_softmax_mode_t mode;          /* Standard, log, or online */

    /* Data location */
    tu_sram_region_t *data_sram;     /* SRAM region containing input data */
    uint32_t          data_offset;   /* Byte offset of first element */
    uint32_t          elem_count;    /* Total number of FP32 elements */

    /* Row configuration (for batched/attention softmax) */
    uint32_t          axis_dim;      /* Elements per row (0 = treat as single row) */
    /* When axis_dim > 0: elem_count must be a multiple of axis_dim.
     * Each row of axis_dim elements is normalized independently.
     * Example: attention scores [batch*heads, seq_len, seq_len]:
     *   elem_count = batch*heads*seq_len*seq_len
     *   axis_dim = seq_len */

    /* Mask support (for causal / padding masks) */
    const float      *mask;          /* Mask buffer (NULL = no mask) */
    bool              mask_is_additive; /* true = add mask before softmax (typical),
                                           false = multiply mask (less common) */
    float             mask_fill;     /* Value for masked positions (e.g., -inf) */

    /* Scaling (for attention: divide by sqrt(head_dim)) */
    float             scale;         /* Apply scale before softmax (0 = no scaling) */

    /* Output configuration */
    bool              in_place;      /* true = write result back to data_sram */
    uint32_t          out_offset;    /* Output offset if !in_place (must be in same SRAM) */

    /* Statistics output (optional) */
    float            *max_out;       /* Per-row max values (caller allocates: num_rows floats) */
    float            *sum_out;       /* Per-row exp-sum values */
} tu_softmax_desc_t;

/* ---- API ---- */

/*
 * Execute softmax on FP32 data in SRAM.
 *
 * The engine performs stable softmax with max-subtract:
 *   1. For each row: find maximum
 *   2. Compute exp(x_i - max) and accumulate sum
 *   3. Divide each exp(x_i - max) by sum
 *
 * For online mode, uses single-pass rescaling algorithm.
 * For log-softmax, computes ln after normalization.
 *
 * Returns total stall cycles from SRAM bandwidth contention.
 * Returns UINT64_MAX on error (check stderr for details).
 */
uint64_t tu_softmax_execute(const tu_softmax_desc_t *desc);

/*
 * Convenience: softmax on a flat FP32 array in SRAM.
 *
 *   data_sram:  SRAM containing input data
 *   offset:     byte offset of first FP32 element
 *   count:      number of FP32 elements
 *   scale:      pre-softmax scaling factor (0 = none)
 *   in_place:   overwrite input with output
 *
 * Returns total stall cycles.
 */
uint64_t tu_softmax(tu_sram_region_t *data_sram, uint32_t offset,
                    uint32_t count, float scale, bool in_place);

/*
 * Convenience: per-row softmax (batched).
 *
 *   data_sram:  SRAM containing row-major FP32 data
 *   offset:     byte offset
 *   rows:       number of rows
 *   cols:       elements per row
 *   scale:      pre-softmax scaling (0 = none)
 *   in_place:   overwrite input with output
 *
 * Softmax is applied independently to each row.
 * Returns total stall cycles.
 */
uint64_t tu_softmax_2d(tu_sram_region_t *data_sram, uint32_t offset,
                       uint32_t rows, uint32_t cols,
                       float scale, bool in_place);

/*
 * Convenience: softmax with mask (for causal/attention masking).
 *
 *   data_sram:  SRAM containing FP32 data
 *   offset:     byte offset
 *   rows:       number of rows
 *   cols:       elements per row
 *   mask:       mask buffer (row-major FP32, same shape; NULL = no mask)
 *   mask_fill:  value to add at masked positions (e.g., -1e9f for -inf effect)
 *   scale:      pre-softmax scaling (e.g., 1/sqrt(head_dim))
 *
 * Mask values are ADDED to the input before softmax.
 * To zero out a position, add a large negative number.
 * For causal masks, positions j > i get mask_fill.
 * Returns total stall cycles.
 */
uint64_t tu_softmax_masked(tu_sram_region_t *data_sram, uint32_t offset,
                           uint32_t rows, uint32_t cols,
                           const float *mask, float mask_fill,
                           float scale);

/*
 * Convenience: log-softmax.
 *
 * Same as tu_softmax() but returns ln(softmax(x_i)).
 * Useful for numerical stability in cross-entropy loss.
 */
uint64_t tu_log_softmax(tu_sram_region_t *data_sram, uint32_t offset,
                        uint32_t count, float scale, bool in_place);

/*
 * Convenience: per-row log-softmax.
 */
uint64_t tu_log_softmax_2d(tu_sram_region_t *data_sram, uint32_t offset,
                           uint32_t rows, uint32_t cols,
                           float scale, bool in_place);

/* ---- Utility ---- */

/* Get mode name as string */
const char *tu_softmax_mode_name(tu_softmax_mode_t mode);

/* Validate a descriptor (returns false on error, prints to stderr) */
bool tu_softmax_validate_desc(const tu_softmax_desc_t *desc);

/* Compute softmax on a host FP32 buffer (for testing/reference).
 * Uses stable max-subtract. Output overwrites input if in_place.
 * count: number of elements in this row. */
void tu_softmax_host(float *data, uint32_t count, float scale);

/* Compute log-softmax on a host buffer (for testing/reference). */
void tu_log_softmax_host(float *data, uint32_t count, float scale);

#ifdef __cplusplus
}
#endif

#endif /* TU_SOFTMAX_ENGINE_H */
