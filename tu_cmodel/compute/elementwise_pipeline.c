/*
 * TinyTU Elementwise Pipeline — Implementation
 * ==============================================
 * Fused elementwise ops on FP32 data in SRAM.
 *
 * All math functions use fast approximations where appropriate:
 *   - GELU: tanh approximation (Hendrycks & Gimpel 2016)
 *   - SiLU/Swish: x * sigmoid(x) using fast sigmoid
 *   - Sigmoid, Tanh, Exp: standard math library (libm)
 */

#include "elementwise_pipeline.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Internal: single-element unary ops ---- */

static inline float ew_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float ew_gelu_tanh_approx(float x) {
    /* GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) */
    const float a = 0.7978845608f; /* sqrt(2/π) */
    const float b = 0.044715f;
    float x3 = x * x * x;
    float inner = a * (x + b * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline float ew_silu(float x) {
    /* SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x)) */
    return x / (1.0f + expf(-x));
}

static inline float ew_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float ew_neg(float x) {
    return -x;
}

static inline float ew_abs(float x) {
    return x < 0.0f ? -x : x;
}

/* ---- Execute a single op on one element ---- */

static float ew_apply_unary_op(float x, tu_ew_opcode_t op) {
    switch (op) {
    case TU_EW_NOP:     return x;
    case TU_EW_RELU:    return ew_relu(x);
    case TU_EW_GELU:    return ew_gelu_tanh_approx(x);
    case TU_EW_SILU:    return ew_silu(x);
    case TU_EW_SIGMOID: return ew_sigmoid(x);
    case TU_EW_TANH:    return tanhf(x);
    case TU_EW_EXP:     return expf(x);
    case TU_EW_NEG:     return ew_neg(x);
    case TU_EW_ABS:     return ew_abs(x);
    case TU_EW_SQRT:    return x >= 0.0f ? sqrtf(x) : 0.0f;
    case TU_EW_LOG:     return x > 0.0f ? logf(x) : -INFINITY;
    default:            return x;
    }
}

static float ew_apply_binary_op(float x, float y, tu_ew_opcode_t op) {
    switch (op) {
    case TU_EW_ADD:  return x + y;
    case TU_EW_MUL:  return x * y;
    case TU_EW_SUB:  return x - y;
    case TU_EW_DIV:  return (y != 0.0f) ? x / y : 0.0f;
    case TU_EW_MIN:  return x < y ? x : y;
    case TU_EW_MAX:  return x > y ? x : y;
    default:         return x;
    }
}

/* ---- Core execution ---- */

uint64_t tu_ew_execute(const tu_ew_desc_t *desc) {
    if (!desc || !desc->sram_region || desc->num_ops == 0)
        return 0;

    if (!tu_ew_validate_desc(desc))
        return 0;

    tu_sram_region_t *sram = desc->sram_region;
    uint32_t elem_count = desc->elem_count;
    uint32_t in_offset  = desc->sram_offset;
    uint32_t out_offset = desc->in_place ? desc->sram_offset : desc->out_offset;
    uint8_t  num_ops    = desc->num_ops;

    /* Validate SRAM bounds */
    uint32_t data_bytes = elem_count * sizeof(float);
    if (in_offset + data_bytes > sram->total_size) {
        fprintf(stderr, "EW: input overflow offset=%u + %u > %u\n",
                in_offset, data_bytes, sram->total_size);
        return 0;
    }
    if (out_offset + data_bytes > sram->total_size) {
        fprintf(stderr, "EW: output overflow offset=%u + %u > %u\n",
                out_offset, data_bytes, sram->total_size);
        return 0;
    }

    /* Get raw pointers for fast access (bandwidth accounting done post-hoc) */
    float *in  = (float *)(tu_sram_raw_ptr(sram) + in_offset);
    float *out = (float *)(tu_sram_raw_ptr(sram) + out_offset);

    /* Execute loop */
    for (uint32_t i = 0; i < elem_count; i++) {
        float val = in[i];

        for (uint8_t j = 0; j < num_ops; j++) {
            const tu_ew_op_t *op = &desc->ops[j];
            if (op->has_scalar) {
                val = ew_apply_binary_op(val, op->scalar, op->opcode);
            } else {
                val = ew_apply_unary_op(val, op->opcode);
            }
        }

        out[i] = val;
    }

    /* Bandwidth accounting: 2 words per element (read + write) if in-place,
     * or 1 read + 1 write if separate buffers */
    uint64_t stall = 0;
    if (sram->banks.bw_modeling) {
        uint32_t bw = sram->banks.bank_width;
        uint32_t words_per_element = sizeof(float) / bw;
        uint32_t words = (in_offset != out_offset ? 2 : 1) * elem_count * words_per_element;

        tu_sram_advance_cycle(sram, elem_count); /* approx compute cycles */
        for (uint32_t i = 0; i < words; i++) {
            uint32_t off = (i % 2 == 0 || in_offset == out_offset) ? in_offset : out_offset;
            off += (i / 2) * sizeof(float);
            uint32_t bank = tu_sram_bank_index(sram, off);
            tu_sram_bw_bank_t *bw_b = &sram->banks.bw_banks[bank];
            if (bw_b->words_available > 0) {
                bw_b->words_available--;
                bw_b->writes_served++;
            } else {
                stall += sram->banks.stall_penalty;
                bw_b->write_stalls++;
                sram->banks.stall_cycles += sram->banks.stall_penalty;
            }
        }
    }

    return stall;
}

/* ---- Convenience Functions ---- */

uint64_t tu_ew_apply_unary(tu_sram_region_t *sram, uint32_t offset,
                           uint32_t elem_count, tu_ew_opcode_t op) {
    tu_ew_desc_t desc = {
        .sram_offset = offset,
        .elem_count  = elem_count,
        .sram_region = sram,
        .in_place    = true,
        .num_ops     = 1,
    };
    desc.ops[0].opcode = op;
    desc.ops[0].has_scalar = false;
    return tu_ew_execute(&desc);
}

uint64_t tu_ew_apply_binary_scalar(tu_sram_region_t *sram, uint32_t offset,
                                   uint32_t elem_count, tu_ew_opcode_t op,
                                   float scalar) {
    tu_ew_desc_t desc = {
        .sram_offset = offset,
        .elem_count  = elem_count,
        .sram_region = sram,
        .in_place    = true,
        .num_ops     = 1,
    };
    desc.ops[0].opcode = op;
    desc.ops[0].has_scalar = true;
    desc.ops[0].scalar = scalar;
    return tu_ew_execute(&desc);
}

uint64_t tu_ew_add_tensors(tu_sram_region_t *sram,
                           uint32_t a_offset, uint32_t b_offset,
                           uint32_t out_offset, uint32_t elem_count) {
    /* This is a special case: we need to read from a and b, add, write to out.
     * For simplicity, we temporarily modify the first buffer. */
    float *a = (float *)(tu_sram_raw_ptr(sram) + a_offset);
    float *b = (float *)(tu_sram_raw_ptr(sram) + b_offset);
    float *o = (float *)(tu_sram_raw_ptr(sram) + out_offset);

    /* If out == a, do it in-place on a */
    if (out_offset == a_offset) {
        for (uint32_t i = 0; i < elem_count; i++)
            a[i] += b[i];
    } else if (out_offset == b_offset) {
        for (uint32_t i = 0; i < elem_count; i++)
            b[i] += a[i];
    } else {
        for (uint32_t i = 0; i < elem_count; i++)
            o[i] = a[i] + b[i];
    }

    /* Bandwidth accounting */
    uint64_t stall = 0;
    if (sram->banks.bw_modeling) {
        /* 2 reads + 1 write per element */
        uint32_t words = 3 * elem_count * (sizeof(float) / sram->banks.bank_width);
        tu_sram_advance_cycle(sram, elem_count);
        for (uint32_t i = 0; i < words; i++) {
            uint32_t bank = tu_sram_bank_index(sram, out_offset + i * sram->banks.bank_width);
            tu_sram_bw_bank_t *bw_b = &sram->banks.bw_banks[bank];
            if (bw_b->words_available > 0) {
                bw_b->words_available--;
            } else {
                stall += sram->banks.stall_penalty;
                bw_b->write_stalls++;
                sram->banks.stall_cycles += sram->banks.stall_penalty;
            }
        }
    }
    return stall;
}

uint64_t tu_ew_apply_fused(tu_sram_region_t *sram, uint32_t offset,
                           uint32_t elem_count, const tu_ew_op_t *ops,
                           uint8_t num_ops) {
    tu_ew_desc_t desc = {
        .sram_offset = offset,
        .elem_count  = elem_count,
        .sram_region = sram,
        .in_place    = true,
        .num_ops     = num_ops,
    };
    if (num_ops > TU_EW_MAX_OPS) {
        fprintf(stderr, "EW: too many ops in fused chain (%u > %u)\n",
                num_ops, TU_EW_MAX_OPS);
        return 0;
    }
    memcpy(desc.ops, ops, num_ops * sizeof(tu_ew_op_t));
    return tu_ew_execute(&desc);
}

/* ---- Utility ---- */

const char *tu_ew_opcode_name(tu_ew_opcode_t op) {
    switch (op) {
    case TU_EW_NOP:     return "NOP";
    case TU_EW_RELU:    return "ReLU";
    case TU_EW_GELU:    return "GELU";
    case TU_EW_SILU:    return "SiLU";
    case TU_EW_SIGMOID: return "Sigmoid";
    case TU_EW_TANH:    return "Tanh";
    case TU_EW_EXP:     return "Exp";
    case TU_EW_NEG:     return "Neg";
    case TU_EW_ABS:     return "Abs";
    case TU_EW_SQRT:    return "Sqrt";
    case TU_EW_LOG:     return "Log";
    case TU_EW_ADD:     return "Add";
    case TU_EW_MUL:     return "Mul";
    case TU_EW_SUB:     return "Sub";
    case TU_EW_DIV:     return "Div";
    case TU_EW_MIN:     return "Min";
    case TU_EW_MAX:     return "Max";
    default:            return "UNKNOWN";
    }
}

bool tu_ew_validate_desc(const tu_ew_desc_t *desc) {
    if (!desc) return false;
    if (!desc->sram_region) return false;
    if (desc->num_ops == 0 || desc->num_ops > TU_EW_MAX_OPS) return false;
    if (desc->elem_count == 0) return false;

    uint32_t data_bytes = desc->elem_count * sizeof(float);
    if (desc->sram_offset + data_bytes > desc->sram_region->total_size)
        return false;
    if (!desc->in_place &&
        desc->out_offset + data_bytes > desc->sram_region->total_size)
        return false;

    /* Validate each op */
    for (uint8_t i = 0; i < desc->num_ops; i++) {
        if (desc->ops[i].opcode >= TU_EW_COUNT) return false;
    }

    return true;
}
