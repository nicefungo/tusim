/*
 * TinyTU Precision Module — Implementation
 * D6: Configurable rounding modes wired into all conversion paths.
 */

#include "tu_precision.h"
#include "rounding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward: subnormal mode (defined close to its accessors below) */
static tu_subnormal_mode_t g_subnormal_mode = TU_SUBNORMAL_FLUSH;

/* ================================================================
 * FP16 ↔ FP32 (IEEE 754)
 * ================================================================ */

fp16_t tu_fp32_to_fp16(fp32_t v) {
    if (isnan(v)) return 0x7E00;
    if (isinf(v)) return (v > 0) ? 0x7C00 : 0xFC00;

    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp > 15) return (sign << 15) | 0x7C00;
    if (exp < -25) return sign << 15;

    uint16_t h;
    /* Subnormal handling: flush-to-zero or full IEEE */
    if (exp <= -15) {
        if (g_subnormal_mode == TU_SUBNORMAL_FLUSH)
            return sign << 15;  /* Flush subnormal to zero */
        int shift = -14 - exp;
        uint32_t m = (mant | 0x800000) >> shift;
        if (shift > 0) {
            uint32_t discarded = mant & ((1u << shift) - 1);
            int carry;
            uint32_t rounding_val = (m << shift) | discarded;
            m = tu_round_apply(rounding_val, 10, &carry);
            if (carry) m = 0x200;
        }
        h = (sign << 15) | (m & 0x3FF);
    } else {
        /* Normal path: round 23-bit mantissa to 10-bit FP16 mantissa */
        int carry;
        uint32_t m_rounded = tu_round_fp32_to_mantissa(mant, exp, 10, &carry);
        uint16_t e = (uint16_t)(exp + 15);
        if (carry) {
            m_rounded = 0;
            e++;
        }
        if (e > 31) return (sign << 15) | 0x7C00;
        h = (sign << 15) | (e << 10) | (m_rounded & 0x3FF);
    }
    return h;
}

fp32_t tu_fp16_to_fp32(fp16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign << 31; }
        else { uint32_t m = mant; int e = -14;
            while ((m & 0x400) == 0) { m <<= 1; e--; }
            bits = (sign<<31) | ((uint32_t)(e+127)<<23) | ((m&0x3FF)<<13); }
    } else if (exp == 0x1F) {
        bits = (sign<<31) | (0xFF<<23) | (mant<<13);
    } else {
        bits = (sign<<31) | ((uint32_t)(exp-15+127)<<23) | (mant<<13);
    }
    fp32_t f; memcpy(&f, &bits, 4); return f;
}

fp16_t tu_round_fp32_to_fp16(fp32_t v) { return tu_fp32_to_fp16(v); }

void tu_fp32_to_fp16_buffer(const fp32_t *src, fp16_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = tu_fp32_to_fp16(src[i]);
}

void tu_fp16_to_fp32_buffer(const fp16_t *src, fp32_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = tu_fp16_to_fp32(src[i]);
}

/* ================================================================
 * BF16 (bfloat16) support — public API
 * ================================================================ */

/* Internal: BF16 ↔ FP32 for precision registry (const void* interface) */
static fp32_t prec_bf16_to_fp32(const void *src) {
    uint16_t bits = *(const uint16_t *)src;
    uint32_t fp32 = (uint32_t)bits << 16;
    fp32_t f;
    memcpy(&f, &fp32, 4);
    return f;
}

static void prec_bf16_from_fp32(fp32_t v, void *dst) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    /* Handle NaN/Inf: pass through */
    if ((bits & 0x7F800000) == 0x7F800000) {
        *(uint16_t *)dst = (uint16_t)(bits >> 16);
        return;
    }

    /* BF16 has 7 mantissa bits. Use rounding module. */
    int carry;
    uint32_t r_mant = tu_round_fp32_to_mantissa(mant, exp, 7, &carry);
    uint16_t sign = (bits >> 31) & 1;
    int16_t e = exp + 127;  /* BF16 uses same exponent bias as FP32 */

    if (carry) {
        r_mant = 0;
        e++;
    }

    if (e >= 255) {  /* Infinity */
        *(uint16_t *)dst = (sign << 15) | 0x7F80;
        return;
    }
    if (e <= 0) {  /* Zero or subnormal */
        *(uint16_t *)dst = sign << 15;
        return;
    }

    *(uint16_t *)dst = (sign << 15) | ((uint16_t)e << 7) | (r_mant & 0x7F);
}

/* Convert BF16 → FP32 (exact: BF16 is FP32 with truncated mantissa).
 * BF16: 1 sign, 8 exponent, 7 mantissa bits.
 * FP32: 1 sign, 8 exponent, 23 mantissa bits.
 * Conversion is simply left-shifting by 16 bits. */
fp32_t tu_bf16_to_fp32(bf16_t h) {
    return prec_bf16_to_fp32(&h);
}

/* Convert FP32 → BF16 with configurable rounding. */
bf16_t tu_fp32_to_bf16(fp32_t v) {
    bf16_t result;
    prec_bf16_from_fp32(v, &result);
    return result;
}

void tu_fp32_to_bf16_buffer(const fp32_t *src, bf16_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = tu_fp32_to_bf16(src[i]);
}

void tu_bf16_to_fp32_buffer(const bf16_t *src, fp32_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = tu_bf16_to_fp32(src[i]);
}

/* ================================================================
 * Subnormal Handling Mode
 * ================================================================ */

/* ================================================================
 * Subnormal Mode Accessors
 * ================================================================ */

tu_subnormal_mode_t tu_get_subnormal_mode(void) {
    return g_subnormal_mode;
}

void tu_set_subnormal_mode(tu_subnormal_mode_t mode) {
    g_subnormal_mode = mode;
}

/* ================================================================
 * Precision Type Registry
 * ================================================================ */

static fp32_t prec_fp16_to_fp32(const void *src) { return tu_fp16_to_fp32(*(const fp16_t*)src); }
static void   prec_fp16_from_fp32(fp32_t v, void *dst) { *(fp16_t*)dst = tu_fp32_to_fp16(v); }
static fp32_t prec_fp32_to_fp32(const void *src) { return *(const fp32_t*)src; }
static void   prec_fp32_from_fp32(fp32_t v, void *dst) { *(fp32_t*)dst = v; }

/* ---- FP8 precision registry entries (stub — FP8 not yet implemented) ---- */
static fp32_t prec_fp8_to_fp32(const void *src) {
    /* FP8 not implemented; pass through as uint8 → FP32 for now */
    return (float)(*(const uint8_t*)src);
}
static void prec_fp8_from_fp32(fp32_t v, void *dst) {
    /* Stub: clamp to 0-255 */
    if (v < 0.0f) v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    *(uint8_t*)dst = (uint8_t)v;
}

/* ---- INT8 precision registry entries ---- */
static fp32_t prec_int8_to_fp32(const void *src) {
    tu_quant_params_t qp;
    tu_quant_params_init_int8(&qp);
    return tu_int8_to_fp32(*(const int8_t_t*)src, &qp);
}
static void prec_int8_from_fp32(fp32_t v, void *dst) {
    tu_quant_params_t qp;
    tu_quant_params_init_int8(&qp);
    *(int8_t_t*)dst = tu_fp32_to_int8(v, &qp);
}

/* ---- INT4 precision registry entries ---- */
static fp32_t prec_int4_to_fp32(const void *src) {
    tu_quant_params_t qp;
    tu_quant_params_init_uint4(&qp);
    return tu_uint4_nibble_to_fp32(*(const uint8_t*)src & 0x0F, &qp);
}
static void prec_int4_from_fp32(fp32_t v, void *dst) {
    tu_quant_params_t qp;
    tu_quant_params_init_uint4(&qp);
    *(uint8_t*)dst = tu_fp32_to_uint4_nibble(v, &qp);
}

static tu_precision_desc_t builtin_precisions[] = {
    { TU_PREC_FP16, 2, "fp16", prec_fp16_to_fp32, prec_fp16_from_fp32 },
    { TU_PREC_FP32, 4, "fp32", prec_fp32_to_fp32, prec_fp32_from_fp32 },
    { TU_PREC_BF16, 2, "bf16", prec_bf16_to_fp32, prec_bf16_from_fp32 },
    { TU_PREC_FP8,  1, "fp8",  prec_fp8_to_fp32,  prec_fp8_from_fp32  },
    { TU_PREC_INT8, 1, "int8", prec_int8_to_fp32, prec_int8_from_fp32 },
    { TU_PREC_INT4, 1, "int4", prec_int4_to_fp32, prec_int4_from_fp32 },
};

static tu_precision_desc_t *custom_precisions = NULL;
static int custom_count = 0;

const tu_precision_desc_t *tu_precision_get(tu_precision_t prec) {
    if (prec < TU_PREC_COUNT) return &builtin_precisions[prec];
    for (int i = 0; i < custom_count; i++)
        if (custom_precisions[i].type == prec) return &custom_precisions[i];
    return NULL;
}

void tu_precision_register(const tu_precision_desc_t *desc) {
    custom_precisions = realloc(custom_precisions, (custom_count+1)*sizeof(*custom_precisions));
    custom_precisions[custom_count++] = *desc;
}
