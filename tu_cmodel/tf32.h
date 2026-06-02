/*
 * TU TF32 Module — D3: TensorFloat-32 Header
 * ============================================
 *
 * TF32 (TensorFloat-32) is NVIDIA's 19-bit floating-point format:
 *   1 sign, 8 exponent, 10 mantissa
 *
 * It preserves the full FP32 dynamic range (same 8-bit exponent)
 * while using ~1/8 the mantissa precision. This makes it ideal for
 * mixed-precision training — matrix multiplies use TF32 MACs (smaller,
 * faster, lower power) while accumulation stays in FP32.
 *
 * TF32 is stored in a 32-bit word (tf32_t = uint32_t). Bits [12:0]
 * are always zero. Cast to float * to reinterpret as FP32.
 */

#ifndef TU_TF32_H
#define TU_TF32_H

#include "tu_config.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- TF32 type ---- */
typedef uint32_t tf32_t;

/* ---- Single-value conversion ---- */

/* TF32 → FP32: Reinterpret bits (TF32 is valid FP32 with zeroed low bits) */
float tu_tf32_to_fp32(tf32_t v);

/* FP32 → TF32: Round 23-bit mantissa to 10 bits using configured rounding mode */
tf32_t tu_fp32_to_tf32(float v);

/* ---- Batch conversions ---- */

void tu_fp32_to_tf32_buffer(const float *src, tf32_t *dst, size_t n);
void tu_tf32_to_fp32_buffer(const tf32_t *src, float *dst, size_t n);

/* ---- Mixed precision bridges ---- */

/* TF32 → FP16 (via FP32 intermediate) */
uint16_t tu_tf32_to_fp16(tf32_t v);

/* FP16 → TF32 (via FP32 intermediate) */
tf32_t tu_fp16_to_tf32(uint16_t v);

/* TF32 → BF16 (via FP32 intermediate) */
uint16_t tu_tf32_to_bf16(tf32_t v);

/* BF16 → TF32 (via FP32 intermediate) */
tf32_t tu_bf16_to_tf32(uint16_t v);

/* ---- Dynamic range ---- */

/* TF32 has identical range to FP32 */
#define TF32_MAX_NORMAL  3.402823466e38f
#define TF32_MIN_NORMAL  1.175494351e-38f

/* ---- Range query ---- */
void tu_tf32_get_range(float *min_normal, float *max_normal);

#ifdef __cplusplus
}
#endif

#endif /* TU_TF32_H */
