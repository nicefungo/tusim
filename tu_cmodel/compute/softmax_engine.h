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
 *       The two-pass implementation remains the compatibility default.
 *       Physical latency/area/energy require a separate arithmetic model.
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

/* Exponential-result handling, not global FP/DAZ or final-output flushing. */
typedef enum {
    TU_SOFTMAX_EXP_LEGACY = 0, /* Historical -87 cutoff; online sum uses expf. */
    TU_SOFTMAX_EXP_FTZ = 1,    /* Flush rounded subnormal expf results only. */
    TU_SOFTMAX_EXP_GRADUAL = 2 /* Preserve host expf subnormal results. */
} tu_softmax_exp_policy_t;

/* Order of the existing scale and mask/threshold stages. */
typedef enum {
    TU_SOFTMAX_MASK_THEN_SCALE = 0, /* Compatibility: scale(mask/threshold(x)). */
    TU_SOFTMAX_SCALE_THEN_MASK = 1  /* mask/threshold(scale(x)). */
} tu_softmax_preprocess_order_t;

/* Whole-row SRAM scheduling; elements inside each row stay forward. */
typedef enum {
    TU_SOFTMAX_ROWS_AUTO = 0,    /* Forward unless overlap requires reverse. */
    TU_SOFTMAX_ROWS_FORWARD = 1,
    TU_SOFTMAX_ROWS_REVERSE = 2
} tu_softmax_row_schedule_t;

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
    float             mask_fill;     /* Nonzero threshold: replace values <= it
                                       (or NaN) with fixed -1e9 after mask op. */

    /* Scaling (for attention: divide by sqrt(head_dim)) */
    float             scale;         /* Apply scale before softmax (0 = no scaling) */

    /* Output configuration */
    bool              in_place;      /* true = write result back to data_sram */
    uint32_t          out_offset;    /* Output offset if !in_place (must be in same SRAM) */

    /* Statistics output (optional) */
    float            *max_out;       /* Per-row max values (caller allocates: num_rows floats) */
    float            *sum_out;       /* Per-row exp-sum values */
    tu_softmax_exp_policy_t exp_policy; /* Zero/default preserves old behavior. */
    tu_softmax_preprocess_order_t preprocess_order; /* Zero = mask then scale. */
    tu_softmax_row_schedule_t row_schedule; /* Zero = overlap-safe AUTO. */
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
 * For log-softmax, retains shifted logits and subtracts log(exp-sum),
 * rather than taking log of rounded probabilities. This preserves finite
 * negative tails when exp underflows. Finite FP32 preprocessing and shifted
 * logits are required for this guarantee; NaN/Inf policy is not specified.
 * LOG uses log1p of a residual sum excluding exactly one maximum, preserving
 * small negative outputs without first rounding the exp-sum to one. The
 * selected exponential policy applies before summation, including ONLINE
 * rescaling. LEGACY retains the -87 output/two-pass cutoff and the historical
 * uncut ONLINE sum. FTZ flushes subnormal exponential results, not final
 * probabilities or log outputs; GRADUAL preserves expf results. Neither is
 * a correctly-rounded transcendental guarantee. Host gradual FP arithmetic
 * is required for GRADUAL; global FTZ/DAZ or fast-math builds are unsupported.
 * Optional sum_out retains its mode-specific FP32 rounding. Convenience and
 * host wrappers always select LEGACY. Appending exp_policy requires binary
 * consumers to rebuild; zero-initialized source callers remain compatible.
 *
 * preprocess_order selects MASK_THEN_SCALE (zero/default) or SCALE_THEN_MASK.
 * The mask stage adds/multiplies the supplied float mask, then, if mask_fill
 * is nonzero, replaces values <= mask_fill (or NaN) with fixed -1e9. It is
 * a numeric threshold, NOT a Boolean exclusion mask or a configurable fill.
 * Scaling by zero (including -0) means skip in either order. Thus with no
 * threshold, additive masks compute scale*(x+mask) versus scale*x+mask;
 * every operation rounds separately to FP32 (no fused multiply-add).
 * Choose order to match the producer's score/bias units, not a speed ranking.
 * All convenience/host wrappers retain MASK_THEN_SCALE. Binary consumers must
 * rebuild for the appended selector. No global config or attention-kernel
 * selection is implied. Finite preprocessing is the tested numerical scope;
 * nonfinite/all-masked policies remain separate.
 *
 * row_schedule selects AUTO (zero/default), FORWARD, or REVERSE whole-row
 * traversal. AUTO uses forward except when a higher overlapping output span
 * requires reverse to preserve unread input. A fixed schedule incompatible
 * with multirow overlap rejects before accesses or statistics writes. Exact
 * alias, disjoint spans, and single-row overlap permit either fixed order.
 * Each entire row is buffered before output; element/reduction order within
 * the row is unchanged. Masks and statistics stay indexed by logical row.
 * Mask/max_out/sum_out buffers must not alias SRAM or one another, and must
 * have the documented extents; host-pointer aliasing is not validated.
 * SRAM scalar width must be sizeof(float). No concurrent mutation is allowed.
 * AUTO preserves previous forward behavior for previously safe layouts and
 * corrects higher-address overlapping multirow output (previously corrupted).
 * All wrappers select AUTO; no YAML/JSON/global attention setting is added.
 * Binary descriptor consumers must rebuild. Invalid schedule rejection is
 * atomic; allocation failure after earlier rows is not transactional.
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
 *   mask_fill:  nonzero post-mask threshold; values <= it become fixed -1e9
 *   scale:      pre-softmax scaling (e.g., 1/sqrt(head_dim))
 *
 * Mask values are ADDED to the input before softmax.
 * The caller supplies every mask entry (including any causal pattern); this
 * wrapper does not generate a causal mask. The mask/threshold stage precedes
 * scaling for compatibility. Finite negative sentinels do not guarantee exact
 * exclusion for every scale or all-masked row.
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
