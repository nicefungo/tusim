/*
 * TU Rounding Module — D6: Configurable Rounding Modes
 * =====================================================
 *
 * Pluggable rounding infrastructure for all precision conversions.
 *
 * Three modes:
 *   TU_ROUND_RNE        — Round-to-Nearest-Even (IEEE 754)
 *   TU_ROUND_RTZ        — Round-Toward-Zero (truncation)
 *   TU_ROUND_STOCHASTIC — Stochastic (unbiased probabilistic)
 *
 * Usage:
 *   tu_set_rounding_mode(TU_ROUND_STOCHASTIC);
 *   tu_stochastic_set_seed(42);  // optional, for reproducibility
 *   fp16_t h = tu_fp32_to_fp16(f);
 */

#ifndef TU_ROUNDING_H
#define TU_ROUNDING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Rounding Mode Enum ---- */
typedef enum {
    TU_ROUND_RNE        = 0,  /* Round-to-nearest-even (IEEE 754 default) */
    TU_ROUND_RTZ        = 1,  /* Round-toward-zero (truncation) */
    TU_ROUND_STOCHASTIC = 2,  /* Stochastic rounding (unbiased) */
    TU_ROUND_COUNT
} tu_rounding_mode_t;

/* ---- Global Mode Accessors ---- */

/* Set the global rounding mode for all precision conversions */
void tu_set_rounding_mode(tu_rounding_mode_t mode);

/* Get the current global rounding mode */
tu_rounding_mode_t tu_get_rounding_mode(void);

/* ---- Stochastic PRNG Seed ---- */

/* Set the PRNG seed for reproducible stochastic rounding.
 * Default seed is fixed (deterministic across runs until changed). */
void tu_stochastic_set_seed(uint64_t seed);

/* Get a uniform random double in [0, 1) — for custom stochastic logic */
double tu_stochastic_uniform(void);

/* ---- Core Rounding Functions ---- */

/*
 * Round a mantissa to `target_bits` bits using the current mode.
 *
 * Parameters:
 *   mantissa:    value to round, with the rounding decision boundary
 *                aligned so bit 0 is the LSB of the fractional part.
 *   target_bits: desired mantissa width.
 *   carry_out:   set to 1 if rounding overflowed (mantissa >= 2^target_bits).
 *
 * Returns: rounded mantissa in [0, 2^target_bits).
 *
 * This is the low-level primitive called by precision conversion routines.
 */
uint32_t tu_round_apply(uint32_t mantissa, int target_bits, int *carry_out);

/*
 * Convenience: round an FP32 mantissa (23 bits) to a narrower width.
 *
 * fp32_mantissa: 23-bit mantissa of an FP32 value.
 * fp32_exp:      exponent (reserved for subnormal-aware rounding).
 * target_bits:   desired output mantissa width.
 * carry_out:     set to 1 on overflow.
 *
 * Returns: rounded mantissa with `target_bits` bits of precision.
 */
uint32_t tu_round_fp32_to_mantissa(uint32_t fp32_mantissa, int fp32_exp,
                                    int target_bits, int *carry_out);

#ifdef __cplusplus
}
#endif

#endif /* TU_ROUNDING_H */
