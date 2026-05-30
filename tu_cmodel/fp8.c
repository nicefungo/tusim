/*
 * TU FP8 Module — D4: FP8 E4M3 and E5M2
 * ======================================
 *
 * Implements the two OCP Microscaling Formats (MX) compliant FP8 types
 * used in NVIDIA Hopper (H100), AMD MI300, and Intel Gaudi accelerators.
 *
 * FP8 E4M3 (1-4-3):
 *   - 1 sign, 4 exponent, 3 mantissa bits
 *   - Exponent bias: 7
 *   - Normal range: [2^-6, 448.0]
 *   - Subnormals: exp=0000, mantissa != 000 (0.001953125 * mantissa/8)
 *   - No infinities (all exponent-15 codes are NaN)
 *   - NaN: s1111_111 (canonical quiet NaN)
 *     s1111_1xx where xx != 11 are signaling NaNs
 *   - Used for: forward-pass inference (higher precision, no inf)
 *
 * FP8 E5M2 (1-5-2):
 *   - 1 sign, 5 exponent, 2 mantissa bits
 *   - Exponent bias: 15
 *   - Normal range: [2^-14, 57344.0]
 *   - Subnormals: exp=00000, mantissa != 00
 *   - Infinity: s11111_00
 *   - NaN: s11111_{01,10,11}
 *   - Used for: backward-pass gradients (wider range, handles overflow)
 *
 * Reference: OCP Microscaling Formats (MX) Specification v1.0
 *            NVIDIA FP8 formats for Deep Learning (arxiv:2209.05433)
 */

#include "fp8.h"
#include "rounding.h"
#include "tu_precision.h"
#include <string.h>
#include <math.h>

/* ================================================================
 * FP8 E4M3 Implementation
 * ================================================================ */

float tu_fp8_e4m3_to_fp32(uint8_t v) {
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp  = (v >> 3) & 0x0F;
    uint32_t mant = v & 0x07;

    if (exp == 0) {
        /* Zero or subnormal */
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        /* Subnormal: (-1)^sign * 2^-6 * mant/8 */
        float result = (float)mant / 8.0f * (1.0f / 64.0f);  /* 2^-6 = 1/64 */
        return sign ? -result : result;
    }

    if (exp == 0x0F) {
        /* NaN (no infinities in E4M3) */
        /* Canonical quiet NaN: s1111_111 */
        /* Return quiet NaN */
        uint32_t nan_bits = (sign << 31) | 0x7FC00000;
        float f;
        memcpy(&f, &nan_bits, 4);
        return f;
    }

    /* Normal: (-1)^sign * 2^(exp-7) * (1 + mant/8) */
    float result = (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, (int)exp - 7);
    return sign ? -result : result;
}

uint8_t tu_fp32_to_fp8_e4m3(float v) {
    if (isnan(v)) {
        uint32_t bits;
        memcpy(&bits, &v, 4);
        return ((bits >> 31) & 1) ? 0xFF : 0x7F;
    }

    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 31) & 1;

    if (v == 0.0f) return sign ? 0x80 : 0x00;

    /* E4M3 max normal is 240. Saturate to NaN above. */
    float absv = sign ? -v : v;
    if (absv >= 240.0f) return sign ? 0xFF : 0x7F;

    /* E4M3 subnormal range: [2^-9, 2^-6) = [0.001953125, 0.015625) */
    if (absv < 0.015625f) {
        /* Scale by 2^6 and multiply by 8 to get 3-bit mantissa */
        float scaled = absv * 64.0f * 8.0f;  /* v * 512 */
        /* v * 512 should give mantissa value in [1, 7] */
        uint32_t im = (uint32_t)(scaled + 0.5f);
        if (tu_get_rounding_mode() == TU_ROUND_RTZ) {
            im = (uint32_t)scaled;
        } else if (tu_get_rounding_mode() == TU_ROUND_STOCHASTIC) {
            float frac = scaled - floorf(scaled);
            im = (uint32_t)scaled;
            if (frac > 0 && tu_stochastic_uniform() < (double)frac) im++;
        }
        if (im == 0) return sign ? 0x80 : 0x00;
        if (im > 7) im = 7;
        return (uint8_t)((sign << 7) | (im & 0x07));
    }

    /* Normal: compute exponent and mantissa for E4M3 (bias=7) */
    /* E4M3 value = 2^(e-7) * (1 + m/8) */
    float log2_val = log2f(absv);
    int e = (int)floorf(log2_val) + 7;  /* encoded exponent = bias + floor(log2) */
    if (e < 1) e = 1;
    if (e > 14) e = 14;

    float remainder = absv / ldexpf(1.0f, e - 7) - 1.0f;  /* fractional part */
    /* remainder in [0, 1), multiply by 8 for mantissa */
    float m_float = remainder * 8.0f;
    uint32_t e4m3_mant;

    tu_rounding_mode_t mode = tu_get_rounding_mode();
    switch (mode) {
    case TU_ROUND_RNE:
        e4m3_mant = (uint32_t)(m_float + 0.5f);
        break;
    case TU_ROUND_RTZ:
        e4m3_mant = (uint32_t)m_float;
        break;
    case TU_ROUND_STOCHASTIC: {
        float frac = m_float - floorf(m_float);
        e4m3_mant = (uint32_t)m_float;
        if (frac > 0 && tu_stochastic_uniform() < (double)frac) e4m3_mant++;
        break;
    }
    default:
        e4m3_mant = (uint32_t)(m_float + 0.5f);
        break;
    }

    /* Handle mantissa overflow */
    if (e4m3_mant >= 8) {
        e4m3_mant = 0;
        e++;
    }

    if (e >= 15) return sign ? 0xFF : 0x7F;
    if (e <= 0) return sign ? 0x80 : 0x00;

    return (uint8_t)((sign << 7) | (e << 3) | (e4m3_mant & 0x07));
}

/* ================================================================
 * FP8 E5M2 Implementation
 * ================================================================ */

float tu_fp8_e5m2_to_fp32(uint8_t v) {
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp  = (v >> 2) & 0x1F;
    uint32_t mant = v & 0x03;

    if (exp == 0) {
        /* Zero or subnormal */
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        /* Subnormal: (-1)^sign * 2^-14 * mant/4 */
        float result = (float)mant / 4.0f * (1.0f / 16384.0f);  /* 2^-14 = 1/16384 */
        return sign ? -result : result;
    }

    if (exp == 0x1F) {
        if (mant == 0) {
            /* Infinity */
            return sign ? -INFINITY : INFINITY;
        }
        /* NaN */
        uint32_t nan_bits = (sign << 31) | 0x7FC00000;
        float f;
        memcpy(&f, &nan_bits, 4);
        return f;
    }

    /* Normal: (-1)^sign * 2^(exp-15) * (1 + mant/4) */
    float result = (1.0f + (float)mant / 4.0f) * ldexpf(1.0f, (int)exp - 15);
    return sign ? -result : result;
}

uint8_t tu_fp32_to_fp8_e5m2(float v) {
    if (isnan(v)) {
        uint32_t bits;
        memcpy(&bits, &v, 4);
        return ((bits >> 31) & 1) ? 0xFD : 0x7D;
    }

    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 31) & 1;

    if (v == 0.0f) return sign ? 0x80 : 0x00;
    if (isinf(v)) return sign ? 0xFC : 0x7C;

    float absv = sign ? -v : v;
    /* E5M2 max normal is 57344. Saturate to infinity above. */
    if (absv > 57344.0f) return sign ? 0xFC : 0x7C;

    /* E5M2 subnormal range: [2^-16, 2^-14) = [1.53e-5, 6.1e-5) */
    if (absv < 6.103515625e-5f) {  /* 2^-14 */
        float scaled = absv * 16384.0f * 4.0f;  /* v * 65536 */
        uint32_t im;
        tu_rounding_mode_t mode = tu_get_rounding_mode();
        switch (mode) {
        case TU_ROUND_RNE:  im = (uint32_t)(scaled + 0.5f); break;
        case TU_ROUND_RTZ:  im = (uint32_t)scaled; break;
        case TU_ROUND_STOCHASTIC: {
            float frac = scaled - floorf(scaled);
            im = (uint32_t)scaled;
            if (frac > 0 && tu_stochastic_uniform() < (double)frac) im++;
            break;
        }
        default: im = (uint32_t)(scaled + 0.5f); break;
        }
        if (im == 0) return sign ? 0x80 : 0x00;
        if (im > 3) im = 3;
        return (uint8_t)((sign << 7) | (im & 0x03));
    }

    /* Normal: compute exponent and mantissa for E5M2 (bias=15) */
    float log2_val = log2f(absv);
    int e = (int)floorf(log2_val) + 15;  /* encoded exponent = bias + floor(log2) */
    if (e < 1) e = 1;
    if (e > 30) e = 30;

    float remainder = absv / ldexpf(1.0f, e - 15) - 1.0f;
    float m_float = remainder * 4.0f;
    uint32_t e5m2_mant;

    tu_rounding_mode_t mode = tu_get_rounding_mode();
    switch (mode) {
    case TU_ROUND_RNE:  e5m2_mant = (uint32_t)(m_float + 0.5f); break;
    case TU_ROUND_RTZ:  e5m2_mant = (uint32_t)m_float; break;
    case TU_ROUND_STOCHASTIC: {
        float frac = m_float - floorf(m_float);
        e5m2_mant = (uint32_t)m_float;
        if (frac > 0 && tu_stochastic_uniform() < (double)frac) e5m2_mant++;
        break;
    }
    default: e5m2_mant = (uint32_t)(m_float + 0.5f); break;
    }

    if (e5m2_mant >= 4) {
        e5m2_mant = 0;
        e++;
    }

    if (e >= 31) return sign ? 0xFC : 0x7C;
    if (e <= 0) return sign ? 0x80 : 0x00;

    return (uint8_t)((sign << 7) | (e << 2) | (e5m2_mant & 0x03));
}

/* ================================================================
 * Batch Conversions
 * ================================================================ */

void tu_fp32_to_fp8_e4m3_buffer(const float *src, uint8_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = tu_fp32_to_fp8_e4m3(src[i]);
}

void tu_fp8_e4m3_to_fp32_buffer(const uint8_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = tu_fp8_e4m3_to_fp32(src[i]);
}

void tu_fp32_to_fp8_e5m2_buffer(const float *src, uint8_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = tu_fp32_to_fp8_e5m2(src[i]);
}

void tu_fp8_e5m2_to_fp32_buffer(const uint8_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = tu_fp8_e5m2_to_fp32(src[i]);
}

/* ================================================================
 * Mixed Precision Conversion
 * ================================================================ */

uint16_t tu_fp8_e4m3_to_fp16(uint8_t v) {
    return tu_fp32_to_fp16(tu_fp8_e4m3_to_fp32(v));
}

uint16_t tu_fp8_e5m2_to_fp16(uint8_t v) {
    return tu_fp32_to_fp16(tu_fp8_e5m2_to_fp32(v));
}
