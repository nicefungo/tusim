/*
 * TU TF32 Module — D3: TensorFloat-32 (1-8-10)
 * ===============================================
 *
 * Implements NVIDIA TensorFloat-32, the mixed-precision format used
 * in Ampere (A100), Hopper (H100), and Blackwell GPUs since 2020.
 *
 * TF32 Format (1-8-10, 19 total bits, stored in 32-bit word):
 *   - 1 sign, 8 exponent, 10 mantissa bits
 *   - Exponent bias: 127 (identical to FP32)
 *   - Dynamic range: identical to FP32 (1.1755e-38 to 3.4028e+38)
 *   - Precision: ~3.3 decimal digits (10-bit mantissa vs 23 for FP32)
 *   - Subnormals: same as FP32 (13 low mantissa bits zeroed)
 *   - NaN/Inf: preserved (same exponent bits as FP32)
 *
 * The key insight: TF32 is just FP32 with the 13 least-significant
 * mantissa bits truncated (or rounded). This means TF32 operations
 * use ~1/8 the multiplier area of FP32 MACs while preserving the
 * full dynamic range — critical for training convergence.
 *
 * Reference: NVIDIA A100 Tensor Core GPU Architecture, 2020
 *            "Accelerating AI Training with NVIDIA TF32 Tensor Cores"
 */

#include "tf32.h"
#include "rounding.h"
#include "tu_precision.h"
#include <string.h>
#include <math.h>

/* ================================================================
 * Single-Value Conversions
 * ================================================================ */

float tu_tf32_to_fp32(tf32_t v) {
    /*
     * TF32 is stored as a 32-bit word with 13 low mantissa bits
     * forced to zero. Reinterpreting as FP32 is exact — no conversion
     * needed. This is the whole point: TF32 values ARE valid FP32
     * values with reduced precision.
     */
    uint32_t bits = v;
    /* Mask off any non-TF32 bits for safety (bits 12:0 must be zero) */
    bits &= 0xFFFFE000u;
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

tf32_t tu_fp32_to_tf32(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);

    uint32_t sign = (bits >> 31) & 1;

    /* NaN/Inf: preserve as-is (TF32 keeps full FP32 NaN/Inf encoding) */
    uint32_t exp = (bits >> 23) & 0xFF;
    if (exp == 0xFF) {
        return bits & 0xFFFFE000u;
    }

    /* Zero: preserve sign */
    if (exp == 0 && (bits & 0x007FFFFFu) == 0) {
        return bits & 0x80000000u;
    }

    /*
     * Normal or subnormal: round the 23-bit mantissa to 10 bits.
     * The mantissa occupies bits [22:0] in FP32.
     * We need to round bit 12 (the first bit to be dropped) into
     * bits [22:13] (the 10-bit TF32 mantissa).
     */

    uint32_t mantissa = bits & 0x007FFFFFu;   /* 23-bit mantissa */
    uint32_t tf32_mant = mantissa >> 13;       /* Top 10 bits [22:13] */
    uint32_t round_bit = (mantissa >> 12) & 1; /* Bit 12: first dropped bit */
    uint32_t sticky = mantissa & 0x00000FFFu;  /* Bits [11:0]: sticky bits */

    tu_rounding_mode_t mode = tu_get_rounding_mode();

    switch (mode) {
    case TU_ROUND_RNE:
        /* Round to nearest even: round up if round_bit=1 AND (sticky>0 OR LSB=1) */
        if (round_bit && (sticky > 0 || (tf32_mant & 1))) {
            tf32_mant++;
        }
        break;
    case TU_ROUND_RTZ:
        /* Truncate: no rounding needed */
        break;
    case TU_ROUND_STOCHASTIC: {
        /*
         * Stochastic rounding: probability of rounding up = fraction.
         * The fraction is: (round_bit * 0.5 + sticky/2^12) / 1.0
         * Simplified: if round_bit, at least 50% chance.
         */
        double frac = (double)((round_bit << 12) | (sticky & 0xFFF)) / 8192.0;
        if (tu_stochastic_uniform() < frac) {
            tf32_mant++;
        }
        break;
    }
    default:
        if (round_bit && (sticky > 0 || (tf32_mant & 1))) {
            tf32_mant++;
        }
        break;
    }

    /* Handle mantissa overflow (carry into exponent) */
    if (tf32_mant >= 0x400) {  /* 10-bit mantissa overflow */
        tf32_mant = 0;
        exp++;
    }

    /* Handle exponent overflow → Infinity */
    if (exp >= 0xFF) {
        return (sign << 31) | 0x7F800000u;
    }

    /* Handle subnormal: if exponent was 0 and mantissa overflowed */
    /* (already handled by exp++ logic above) */

    return (sign << 31) | (exp << 23) | (tf32_mant << 13);
}

/* ================================================================
 * Batch Conversions
 * ================================================================ */

void tu_fp32_to_tf32_buffer(const float *src, tf32_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = tu_fp32_to_tf32(src[i]);
    }
}

void tu_tf32_to_fp32_buffer(const tf32_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = tu_tf32_to_fp32(src[i]);
    }
}

/* ================================================================
 * Mixed Precision Conversion
 * ================================================================ */

uint16_t tu_tf32_to_fp16(tf32_t v) {
    return tu_fp32_to_fp16(tu_tf32_to_fp32(v));
}

tf32_t tu_fp16_to_tf32(uint16_t v) {
    return tu_fp32_to_tf32(tu_fp16_to_fp32(v));
}

uint16_t tu_tf32_to_bf16(tf32_t v) {
    return tu_fp32_to_bf16(tu_tf32_to_fp32(v));
}

tf32_t tu_bf16_to_tf32(uint16_t v) {
    return tu_fp32_to_tf32(tu_bf16_to_fp32(v));
}

/* ================================================================
 * Dynamic Range Query
 * ================================================================ */

void tu_tf32_get_range(float *min_normal, float *max_normal) {
    *min_normal = TF32_MIN_NORMAL;
    *max_normal = TF32_MAX_NORMAL;
}
