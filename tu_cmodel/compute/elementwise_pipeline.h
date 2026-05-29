/*
 * TinyTU Elementwise Pipeline
 * =============================
 * Fused elementwise operations in the accumulator path.
 *
 * O4: Fused elementwise ops avoid round-trips to DRAM.
 * Operations execute on FP32 data in SRAM (typically the output/
 * accumulator buffer) and can be chained for compound transforms.
 *
 * Supported ops:
 *   Unary: RELU, GELU (tanh approx), SILU (sigmoid approx), SIGMOID,
 *          TANH, EXP, NEG, ABS, SQRT, LOG
 *   Binary: ADD, MUL, SUB, DIV, MIN, MAX
 *
 * Architecture:
 *   The pipeline is a stateless function that reads from one SRAM
 *   region (or buffer), applies a sequence of ops, and writes back.
 *   Multiple ops can be fused into a single pass over the data.
 */

#ifndef TU_ELEMENTWISE_PIPELINE_H
#define TU_ELEMENTWISE_PIPELINE_H

#include "tu_config.h"
#include "tu_sram.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opcodes ---- */

typedef enum {
    TU_EW_NOP      = 0,    /* No-op (pass-through) */
    TU_EW_RELU     = 1,    /* max(0, x) */
    TU_EW_GELU     = 2,    /* GELU: x * Φ(x), tanh approximation */
    TU_EW_SILU     = 3,    /* SiLU / Swish: x * σ(x) */
    TU_EW_SIGMOID  = 4,    /* σ(x) = 1/(1+e^(-x)) */
    TU_EW_TANH     = 5,    /* tanh(x) */
    TU_EW_EXP      = 6,    /* e^x */
    TU_EW_NEG      = 7,    /* -x */
    TU_EW_ABS      = 8,    /* |x| */
    TU_EW_SQRT     = 9,    /* sqrt(x), x≥0 */
    TU_EW_LOG      = 10,   /* ln(x), x>0 */
    TU_EW_ADD      = 11,   /* x + y (binary) */
    TU_EW_MUL      = 12,   /* x * y (binary) */
    TU_EW_SUB      = 13,   /* x - y (binary) */
    TU_EW_DIV      = 14,   /* x / y (binary) */
    TU_EW_MIN      = 15,   /* min(x, y) (binary) */
    TU_EW_MAX      = 16,   /* max(x, y) (binary) */
    TU_EW_COUNT
} tu_ew_opcode_t;

/* ---- Operation descriptor ---- */

/* A single elementwise operation in a chain */
typedef struct {
    tu_ew_opcode_t opcode;
    float          scalar;        /* Scalar operand (for binary ops, the RHS constant) */
    bool           has_scalar;    /* true = use scalar, false = in-place unary */
} tu_ew_op_t;

/* A chain of fused elementwise operations */
#define TU_EW_MAX_OPS  8          /* Max ops in a fused chain */

typedef struct {
    tu_ew_op_t      ops[TU_EW_MAX_OPS];
    uint8_t         num_ops;
    uint32_t        sram_offset;   /* Byte offset in SRAM region */
    uint32_t        elem_count;    /* Number of FP32 elements */
    tu_sram_region_t *sram_region; /* SRAM region to operate on */
    bool            in_place;      /* true = modify in-place, false = write to separate output */
    uint32_t        out_offset;    /* Output offset (if !in_place) */
} tu_ew_desc_t;

/* ---- API ---- */

/*
 * Execute a chain of elementwise operations on FP32 data in SRAM.
 *
 * The pipeline reads FP32 elements from sram_region at sram_offset,
 * applies each op in sequence, and writes results back (in-place
 * or to out_offset).
 *
 * Returns the number of stall cycles incurred from SRAM bandwidth.
 */
uint64_t tu_ew_execute(const tu_ew_desc_t *desc);

/*
 * Convenience: apply a single unary op in-place.
 * Returns stall cycles.
 */
uint64_t tu_ew_apply_unary(tu_sram_region_t *sram, uint32_t offset,
                           uint32_t elem_count, tu_ew_opcode_t op);

/*
 * Convenience: apply a single binary op with scalar in-place.
 * Returns stall cycles.
 */
uint64_t tu_ew_apply_binary_scalar(tu_sram_region_t *sram, uint32_t offset,
                                   uint32_t elem_count, tu_ew_opcode_t op,
                                   float scalar);

/*
 * Apply elementwise add of two tensors (A + B → out).
 * A is at sram_offset, B is at b_offset, result goes to out_offset.
 * All regions must be the same SRAM region.
 */
uint64_t tu_ew_add_tensors(tu_sram_region_t *sram,
                           uint32_t a_offset, uint32_t b_offset,
                           uint32_t out_offset, uint32_t elem_count);

/*
 * Apply a fused activation after MMA: configurable sequence of ops.
 * Common patterns:
 *   - RELU only: {TU_EW_RELU}
 *   - GELU:      {TU_EW_GELU}
 *   - SiLU:      {TU_EW_SILU}
 *   - Add bias + ReLU: {TU_EW_ADD(bias), TU_EW_RELU}
 */
uint64_t tu_ew_apply_fused(tu_sram_region_t *sram, uint32_t offset,
                           uint32_t elem_count, const tu_ew_op_t *ops,
                           uint8_t num_ops);

/* ---- Utility ---- */

/* Get opcode name as string */
const char *tu_ew_opcode_name(tu_ew_opcode_t op);

/* Validate a descriptor (returns false if invalid) */
bool tu_ew_validate_desc(const tu_ew_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif /* TU_ELEMENTWISE_PIPELINE_H */
