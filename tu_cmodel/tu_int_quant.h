/*
 * TinyTU INT8/INT4 Quantization Module
 * =====================================
 * Gap D2: Production-grade integer quantization support.
 *
 * Implements INT8 and INT4 data types with:
 *   - Per-tensor affine quantization: value = (q - zero_point) * scale
 *   - INT8 × INT8 → INT32 MAC accumulation
 *   - FP32 → INT8/INT4 quantization with configurable rounding
 *   - INT8/INT4 → FP32 dequantization
 *   - Batch conversion utilities
 *
 * Reference: Google TensorFlow Lite quantization spec, NVIDIA TensorRT INT8.
 */

#ifndef TU_INT_QUANT_H
#define TU_INT_QUANT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Quantized types ---- */
typedef int8_t   int8_t_t;    /* INT8: -128 .. 127 */
typedef uint8_t  uint4_t_t;   /* UINT4 packed: 2 values per byte, low nibble first */

/* ---- Quantization parameters ---- */

/* Per-tensor affine quantization parameters.
 * real_value = (q - zero_point) * scale
 * q = clamp(round(real_value / scale) + zero_point, min, max)
 */
typedef struct {
    float    scale;          /* Quantization step size */
    int32_t  zero_point;     /* Integer zero-point (maps to real 0.0) */
    int32_t  qmin;           /* Minimum quantized value */
    int32_t  qmax;           /* Maximum quantized value */
} tu_quant_params_t;

/* ---- Default quantization parameters ---- */

/* Standard INT8 symmetric/asymmetric params */
#define TU_INT8_QMIN         (-128)
#define TU_INT8_QMAX         127
#define TU_INT8_DEFAULT_SCALE     0.007874016f   /* ~ 1/127 */
#define TU_INT8_DEFAULT_ZP        0              /* Symmetric by default */

/* UINT4 params (0..15) */
#define TU_UINT4_QMIN        0
#define TU_UINT4_QMAX        15
#define TU_UINT4_DEFAULT_SCALE     0.06666667f   /* ~ 1/15 */
#define TU_UINT4_DEFAULT_ZP        8             /* Center at 8 for symmetric feel */

/* ---- Quantization Functions ---- */

/* Initialize default quantization parameters for INT8 (symmetric). */
void tu_quant_params_init_int8(tu_quant_params_t *qp);

/* Initialize default quantization parameters for UINT4. */
void tu_quant_params_init_uint4(tu_quant_params_t *qp);

/*
 * Calibrate quantization parameters from a data buffer.
 * For INT8 symmetric: computes scale = max(|data|) / 127, zero_point = 0.
 * For INT8 asymmetric: computes [min, max] range and appropriate scale/zp.
 * For UINT4: computes scale = (max - min) / 15, zero_point = round(qmin - min/scale).
 */
void tu_quant_params_calibrate_int8_symmetric(const float *data, size_t n,
                                               tu_quant_params_t *qp);
void tu_quant_params_calibrate_int8_asymmetric(const float *data, size_t n,
                                                tu_quant_params_t *qp);
void tu_quant_params_calibrate_uint4(const float *data, size_t n,
                                      tu_quant_params_t *qp);

/* ---- Single-element Conversion ---- */

/* FP32 → INT8 (with clamp and round-to-nearest) */
int8_t_t tu_fp32_to_int8(float v, const tu_quant_params_t *qp);

/* INT8 → FP32 (dequantize) */
float tu_int8_to_fp32(int8_t_t q, const tu_quant_params_t *qp);

/* FP32 → UINT4 (packed: 2 values per byte) */
/* Returns the nibble value (0..15) — caller packs into bytes */
uint8_t tu_fp32_to_uint4_nibble(float v, const tu_quant_params_t *qp);

/* UINT4 nibble → FP32 (dequantize) */
float tu_uint4_nibble_to_fp32(uint8_t nibble, const tu_quant_params_t *qp);

/* ---- Batch Conversion ---- */

/* Quantize FP32 array → INT8 array */
void tu_fp32_to_int8_buffer(const float *src, int8_t_t *dst, size_t n,
                             const tu_quant_params_t *qp);

/* Dequantize INT8 array → FP32 array */
void tu_int8_to_fp32_buffer(const int8_t_t *src, float *dst, size_t n,
                             const tu_quant_params_t *qp);

/* Quantize FP32 array → packed UINT4 array (ceil(n/2) bytes) */
void tu_fp32_to_uint4_buffer(const float *src, uint8_t *dst, size_t n,
                              const tu_quant_params_t *qp);

/* Dequantize packed UINT4 array → FP32 array */
void tu_uint4_to_fp32_buffer(const uint8_t *src, float *dst, size_t n,
                              const tu_quant_params_t *qp);

/*
 * Unpack a UINT4 nibble from packed storage.
 * index: element index (0-based, even → low nibble, odd → high nibble).
 */
uint8_t tu_uint4_unpack(const uint8_t *packed, size_t index);

/* Pack a UINT4 nibble into packed storage. */
void tu_uint4_pack(uint8_t *packed, size_t index, uint8_t nibble);

/* ---- INT MAC Operations ---- */

/*
 * INT8 vector dot product: sum(qa[i] * qb[i]) for i=0..n-1, result in INT32.
 * No quantization parameters yet — the caller applies scale after accumulation.
 * This matches hardware behavior: INT8 MACs accumulate in INT32.
 */
int32_t tu_int8_dot_product(const int8_t_t *a, const int8_t_t *b, size_t n);

/*
 * INT8 MMA tile: O[m][n] += W[m][k_int] × A[k_int][n]
 *   W: INT8 weight matrix [M][K] (row-major)
 *   A: INT8 activation matrix [K][N] (row-major)
 *   O: INT32 output matrix [M][N] (row-major, accumulated)
 *
 * After this call, the caller must apply quantization scale:
 *   output_fp32[i] = O_int32[i] * w_scale * a_scale
 */
void tu_int8_mma_tile(const int8_t_t *W, const int8_t_t *A,
                       int32_t *O,
                       uint16_t M, uint16_t N, uint16_t K);

/* ---- Configuration ---- */

/* Precision type descriptor for INT8 (to register in tu_precision module) */
typedef struct {
    int elem_bytes;
    const char *name;
} tu_int_precision_info_t;

extern const tu_int_precision_info_t tu_int8_precision_info;
extern const tu_int_precision_info_t tu_int4_precision_info;

#ifdef __cplusplus
}
#endif

#endif /* TU_INT_QUANT_H */
