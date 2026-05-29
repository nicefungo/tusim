/*
 * TinyTU Precision Module — Implementation
 */

#include "tu_precision.h"
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
        if (shift > 0 && ((mant >> (shift - 1)) & 1)) {
            uint32_t round_bit = (mant >> (shift - 1)) & 1;
            uint32_t sticky = 0;
            if (shift > 1) sticky = (mant & ((1u << (shift - 1)) - 1)) != 0;
            if (round_bit && (sticky || (m & 1))) m++;
        }
        h = (sign << 15) | (m & 0x3FF);
    } else {
        uint16_t e = (uint16_t)(exp + 15);
        uint32_t round_bit = (mant >> 12) & 1;
        uint32_t sticky = (mant & 0xFFF) != 0;
        uint32_t m_rounded = (mant >> 13);
        if (round_bit && (sticky || (m_rounded & 1))) m_rounded++;
        if (m_rounded >= 0x400) { m_rounded = 0; e++; }
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
    uint32_t rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    if ((rounded & 0x7F800000) > 0x7F800000)
        rounded = (rounded & 0x80000000) | 0x7F800000;
    *(uint16_t *)dst = (uint16_t)(rounded >> 16);
}

/* Convert BF16 → FP32 (exact: BF16 is FP32 with truncated mantissa).
 * BF16: 1 sign, 8 exponent, 7 mantissa bits.
 * FP32: 1 sign, 8 exponent, 23 mantissa bits.
 * Conversion is simply left-shifting by 16 bits. */
fp32_t tu_bf16_to_fp32(bf16_t h) {
    return prec_bf16_to_fp32(&h);
}

/* Convert FP32 → BF16 with round-to-nearest-even.
 * This is the standard bfloat16 conversion used in TPU and NVIDIA hardware. */
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

static tu_precision_desc_t builtin_precisions[] = {
    { TU_PREC_FP16, 2, "fp16", prec_fp16_to_fp32, prec_fp16_from_fp32 },
    { TU_PREC_FP32, 4, "fp32", prec_fp32_to_fp32, prec_fp32_from_fp32 },
    { TU_PREC_BF16, 2, "bf16", prec_bf16_to_fp32, prec_bf16_from_fp32 },
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
