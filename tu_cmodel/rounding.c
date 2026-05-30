/*
 * TU Rounding Module — D6: Configurable Rounding Modes
 * =====================================================
 *
 * Pluggable rounding infrastructure for all precision types.
 * Supports three modes:
 *
 *   RNE  — Round-to-Nearest-Even (IEEE 754 default)
 *           Ties round to the even mantissa LSB.
 *           Lowest bias, standard for inference.
 *
 *   RTZ  — Round-Toward-Zero (truncation)
 *           Always discard fractional bits.
 *           Zero bias but higher variance. Used in some integer
 *           quantization paths and matches x86/ARM truncation.
 *
 *   STOCHASTIC — Stochastic Rounding
 *           Round up with probability equal to the fractional part.
 *           Unbiased in expectation; critical for training where
 *           repeated truncation would produce systematic drift.
 *           Uses xorshift128+ PRNG for fast, reproducible randomness.
 *
 * Usage:
 *   tu_set_rounding_mode(TU_ROUND_STOCHASTIC);
 *   fp16_t h = tu_fp32_to_fp16(f);  // uses stochastic rounding
 *
 * Architecture:
 *   The rounding module is stateless except for the global mode and
 *   PRNG seed. All conversion functions in tu_precision.c call
 *   tu_round_apply() with their mantissa + target bits. This
 *   replaces the inline RNE-only logic that was previously scattered
 *   through FP16, BF16, and INT8 conversion routines.
 */

#include "rounding.h"
#include <string.h>

/* ---- Global state ---- */
static tu_rounding_mode_t g_rounding_mode = TU_ROUND_RNE;

/* ---- xorshift128+ PRNG state (for stochastic rounding) ---- */
static uint64_t g_prng_state[2] = { 0xDEADBEEFCAFEBABEULL, 0x0123456789ABCDEFULL };

/* ================================================================
 * PRNG: xorshift128+
 * ================================================================ */

void tu_stochastic_set_seed(uint64_t seed) {
    /* Splitmix64 to seed both state words */
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    g_prng_state[0] = z ^ (z >> 31);

    z = g_prng_state[0] + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    g_prng_state[1] = z ^ (z >> 31);
}

static uint64_t prng_next(void) {
    uint64_t s1 = g_prng_state[0];
    const uint64_t s0 = g_prng_state[1];
    g_prng_state[0] = s0;
    s1 ^= s1 << 23;
    g_prng_state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return g_prng_state[1] + s0;
}

/* Generate a uniform double in [0, 1) */
static double prng_uniform(void) {
    /* Use top 53 bits of a 64-bit random for full double precision */
    return (double)(prng_next() >> 11) * 0x1.0p-53;
}

/* ================================================================
 * Rounding Mode Accessors
 * ================================================================ */

void tu_set_rounding_mode(tu_rounding_mode_t mode) {
    g_rounding_mode = mode;
}

tu_rounding_mode_t tu_get_rounding_mode(void) {
    return g_rounding_mode;
}

/* ================================================================
 * Core Rounding Functions
 * ================================================================ */

/*
 * tu_round_apply — round a mantissa to `target_bits` bits.
 *
 * Parameters:
 *   mantissa:    full-precision mantissa (e.g., 23 bits for FP32 input)
 *   target_bits: desired mantissa width (e.g., 10 for FP16, 7 for BF16,
 *                3 for FP8 E4M3, 2 for FP8 E5M2)
 *   carry_out:   set to 1 if rounding overflowed (mantissa >= 2^target_bits),
 *                indicating exponent must be incremented.
 *
 * Returns: rounded mantissa in [0, 2^target_bits).
 *
 * The caller is responsible for shifting the mantissa so that the
 * rounding decision boundary aligns with bit position 0.
 * Convention: the mantissa is passed in with the implicit leading 1
 * (if normalized) in bit position (target_bits), and the MSB of the
 * fractional part in bit position (target_bits - 1).
 */
uint32_t tu_round_apply(uint32_t mantissa, int target_bits, int *carry_out) {
    *carry_out = 0;

    if (target_bits <= 0) return 0;

    /* Shift such that rounding bit is at position 0, guard/sticky below */
    /* mantissa has (target_bits + 1) bits of precision we care about:
     *   bit target_bits:     implicit leading 1 (keep)
     *   bit target_bits-1:   MSB of mantissa (keep)
     *   bit 0:               LSB of mantissa
     *   below bit 0:         fraction to round */
    int shift = target_bits;
    uint32_t integer_part = mantissa >> shift;          /* bits to keep */
    uint32_t fractional = mantissa & ((1u << shift) - 1); /* bits to round */
    uint32_t half = 1u << (shift - 1);                   /* 0.5 in fixed-point */

    uint32_t result = integer_part;

    switch (g_rounding_mode) {
    case TU_ROUND_RNE: {
        /* Round-to-nearest-even:
         *   - If fractional > 0.5: round up
         *   - If fractional < 0.5: round down
         *   - If fractional == 0.5: round to even (LSB of integer = 0) */
        if (fractional > half) {
            result = integer_part + 1;
        } else if (fractional == half) {
            /* Tie: round to even */
            if (integer_part & 1) {
                result = integer_part + 1;
            }
        }
        break;
    }
    case TU_ROUND_RTZ: {
        /* Round-toward-zero: always truncate */
        /* result = integer_part (already set) */
        break;
    }
    case TU_ROUND_STOCHASTIC: {
        /* Stochastic rounding: probability of rounding up = fractional / 2^shift */
        if (fractional > 0) {
            double p = (double)fractional / (double)(1u << shift);
            if (prng_uniform() < p) {
                result = integer_part + 1;
            }
        }
        break;
    }
    default:
        break;
    }

    /* Check for overflow: if result >= 2^target_bits, carry into exponent */
    uint32_t max_val = (1u << target_bits);
    if (result >= max_val) {
        *carry_out = 1;
        result >>= 1;  /* Renormalize: mantissa /= 2, exponent += 1 */
    }

    return result;
}

/*
 * tu_round_fp32_mantissa_to — convenience wrapper for FP32→narrower conversion.
 *
 * Given an FP32 value's mantissa (23 bits, with implicit leading 1 already
 * OR'd in for normals) and a target mantissa width, perform the shift,
 * rounding, and return the result along with any exponent carry.
 *
 * target_bits: desired mantissa bits (including implicit bit position but
 *              the result will have target_bits explicit mantissa bits
 *              for subnormal handling the caller handles the exponent edge).
 */
uint32_t tu_round_fp32_to_mantissa(uint32_t fp32_mantissa, int fp32_exp,
                                    int target_bits, int *carry_out) {
    (void)fp32_exp;  /* reserved for future subnormal-aware rounding */
    *carry_out = 0;

    /* fp32_mantissa has 23 explicit bits. We're converting to target_bits.
     * The mantissa is already normalized with implicit leading 1.
     * We need to round from 23 down to target_bits by shifting right
     * by (23 - target_bits) positions. */
    int discard_bits = 23 - target_bits;
    if (discard_bits <= 0) {
        /* Target is wider or equal — no rounding needed */
        return fp32_mantissa;
    }

    /* Build a fixed-point representation:
     *   The mantissa has (23 - discard_bits) = target_bits bits to keep,
     *   plus discard_bits fractional bits.
     *
     * We shift left by target_bits so the rounding decision is at bit 0,
     * then call tu_round_apply.
     *
     * Simplified approach:
     *   integer_part = mantissa >> discard_bits
     *   fractional   = mantissa & ((1 << discard_bits) - 1)
     *
     * Then apply rounding as if we had target_bits precision.
     */
    uint32_t integer_part = fp32_mantissa >> discard_bits;
    uint32_t fractional   = fp32_mantissa & ((1u << discard_bits) - 1);
    uint32_t half         = 1u << (discard_bits - 1);

    uint32_t result = integer_part;

    switch (g_rounding_mode) {
    case TU_ROUND_RNE:
        if (fractional > half) {
            result = integer_part + 1;
        } else if (fractional == half) {
            if (integer_part & 1) result = integer_part + 1;
        }
        break;
    case TU_ROUND_RTZ:
        break;
    case TU_ROUND_STOCHASTIC:
        if (fractional > 0) {
            double p = (double)fractional / (double)(1u << discard_bits);
            if (prng_uniform() < p) {
                result = integer_part + 1;
            }
        }
        break;
    case TU_ROUND_COUNT:
        break;
    }

    uint32_t max_val = (1u << target_bits);
    if (result >= max_val) {
        *carry_out = 1;
        result >>= 1;
    }

    return result;
}
