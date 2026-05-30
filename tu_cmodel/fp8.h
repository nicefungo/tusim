/*
 * TU FP8 Module — D4: FP8 E4M3 and E5M2
 * ======================================
 *
 * OCP Microscaling Formats (MX) compliant FP8 types.
 *
 * FP8 E4M3: 1 sign, 4 exponent, 3 mantissa
 *   - Exponent bias: 7
 *   - Normal range: [0.015625, 448.0]
 *   - No infinities (nan-only for overflow)
 *   - Primary use: forward pass (higher precision)
 *
 * FP8 E5M2: 1 sign, 5 exponent, 2 mantissa
 *   - Exponent bias: 15
 *   - Normal range: [6.1e-5, 57344.0]
 *   - Supports infinities
 *   - Primary use: backward pass (wider dynamic range)
 *
 * Reference: OCP MX Formats Spec v1.0, NVIDIA Hopper architecture
 */

#ifndef TU_FP8_H
#define TU_FP8_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* FP8 type: single byte for both E4M3 and E5M2 */
typedef uint8_t fp8_t;

/* ================================================================
 * FP8 E4M3 (1-4-3)
 * ================================================================ */

/* Convert E4M3 → FP32 */
float   tu_fp8_e4m3_to_fp32(uint8_t v);

/* Convert FP32 → E4M3 (uses global rounding mode) */
uint8_t tu_fp32_to_fp8_e4m3(float v);

/* Batch E4M3 conversions */
void tu_fp32_to_fp8_e4m3_buffer(const float *src, uint8_t *dst, size_t n);
void tu_fp8_e4m3_to_fp32_buffer(const uint8_t *src, float *dst, size_t n);

/* ================================================================
 * FP8 E5M2 (1-5-2)
 * ================================================================ */

/* Convert E5M2 → FP32 */
float   tu_fp8_e5m2_to_fp32(uint8_t v);

/* Convert FP32 → E5M2 (uses global rounding mode) */
uint8_t tu_fp32_to_fp8_e5m2(float v);

/* Batch E5M2 conversions */
void tu_fp32_to_fp8_e5m2_buffer(const float *src, uint8_t *dst, size_t n);
void tu_fp8_e5m2_to_fp32_buffer(const uint8_t *src, float *dst, size_t n);

/* ================================================================
 * Cross-Precision Conversions
 * ================================================================ */

/* E4M3 → FP16 (via FP32 intermediate) */
uint16_t tu_fp8_e4m3_to_fp16(uint8_t v);

/* E5M2 → FP16 (via FP32 intermediate) */
uint16_t tu_fp8_e5m2_to_fp16(uint8_t v);

#ifdef __cplusplus
}
#endif

#endif /* TU_FP8_H */
