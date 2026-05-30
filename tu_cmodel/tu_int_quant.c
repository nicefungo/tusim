/*
 * TinyTU INT8/INT4 Quantization — Implementation
 * ===============================================
 * Gap D2: Production-grade integer quantization for the TU cmodel.
 *
 * Quantization formula (affine):
 *   q = clamp(round(r / scale) + zero_point, qmin, qmax)
 *   r = (q - zero_point) * scale
 *
 * INT4 is stored as packed nibbles (2 values per byte, low nibble first).
 * INT8 MACs accumulate into INT32 registers.
 */

#include "tu_int_quant.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ================================================================
 * Quantization Parameters
 * ================================================================ */

void tu_quant_params_init_int8(tu_quant_params_t *qp) {
    qp->scale      = TU_INT8_DEFAULT_SCALE;
    qp->zero_point = TU_INT8_DEFAULT_ZP;
    qp->qmin       = TU_INT8_QMIN;
    qp->qmax       = TU_INT8_QMAX;
}

void tu_quant_params_init_uint4(tu_quant_params_t *qp) {
    qp->scale      = TU_UINT4_DEFAULT_SCALE;
    qp->zero_point = TU_UINT4_DEFAULT_ZP;
    qp->qmin       = TU_UINT4_QMIN;
    qp->qmax       = TU_UINT4_QMAX;
}

/* ---- Calibration ---- */

void tu_quant_params_calibrate_int8_symmetric(const float *data, size_t n,
                                               tu_quant_params_t *qp) {
    if (n == 0) { tu_quant_params_init_int8(qp); return; }

    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(data[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) amax = 1.0f; /* all zeros → safe default */

    qp->scale      = amax / 127.0f;
    qp->zero_point = 0;
    qp->qmin       = TU_INT8_QMIN;
    qp->qmax       = TU_INT8_QMAX;
}

void tu_quant_params_calibrate_int8_asymmetric(const float *data, size_t n,
                                                tu_quant_params_t *qp) {
    if (n == 0) { tu_quant_params_init_int8(qp); return; }

    float vmin = data[0], vmax = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] < vmin) vmin = data[i];
        if (data[i] > vmax) vmax = data[i];
    }
    if (vmax == vmin) vmax = vmin + 1.0f;

    qp->qmin       = TU_INT8_QMIN;
    qp->qmax       = TU_INT8_QMAX;
    qp->scale      = (vmax - vmin) / 255.0f;
    qp->zero_point = (int32_t)roundf(TU_INT8_QMIN - vmin / qp->scale);
    if (qp->zero_point < TU_INT8_QMIN) qp->zero_point = TU_INT8_QMIN;
    if (qp->zero_point > TU_INT8_QMAX) qp->zero_point = TU_INT8_QMAX;
}

void tu_quant_params_calibrate_uint4(const float *data, size_t n,
                                      tu_quant_params_t *qp) {
    if (n == 0) { tu_quant_params_init_uint4(qp); return; }

    float vmin = data[0], vmax = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] < vmin) vmin = data[i];
        if (data[i] > vmax) vmax = data[i];
    }
    if (vmax == vmin) vmax = vmin + 1.0f;

    qp->qmin       = TU_UINT4_QMIN;
    qp->qmax       = TU_UINT4_QMAX;
    qp->scale      = (vmax - vmin) / 15.0f;
    qp->zero_point = (int32_t)roundf(TU_UINT4_QMIN - vmin / qp->scale);
    if (qp->zero_point < TU_UINT4_QMIN) qp->zero_point = TU_UINT4_QMIN;
    if (qp->zero_point > TU_UINT4_QMAX) qp->zero_point = TU_UINT4_QMAX;
}

/* ================================================================
 * Single-element Conversion
 * ================================================================ */

static int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int8_t_t tu_fp32_to_int8(float v, const tu_quant_params_t *qp) {
    float scaled = roundf(v / qp->scale) + (float)qp->zero_point;
    return (int8_t_t)clamp_i32((int32_t)scaled, qp->qmin, qp->qmax);
}

float tu_int8_to_fp32(int8_t_t q, const tu_quant_params_t *qp) {
    return ((float)q - (float)qp->zero_point) * qp->scale;
}

uint8_t tu_fp32_to_uint4_nibble(float v, const tu_quant_params_t *qp) {
    float scaled = roundf(v / qp->scale) + (float)qp->zero_point;
    return (uint8_t)clamp_i32((int32_t)scaled, qp->qmin, qp->qmax);
}

float tu_uint4_nibble_to_fp32(uint8_t nibble, const tu_quant_params_t *qp) {
    return ((float)nibble - (float)qp->zero_point) * qp->scale;
}

/* ================================================================
 * Batch Conversion
 * ================================================================ */

void tu_fp32_to_int8_buffer(const float *src, int8_t_t *dst, size_t n,
                             const tu_quant_params_t *qp) {
    for (size_t i = 0; i < n; i++)
        dst[i] = tu_fp32_to_int8(src[i], qp);
}

void tu_int8_to_fp32_buffer(const int8_t_t *src, float *dst, size_t n,
                             const tu_quant_params_t *qp) {
    for (size_t i = 0; i < n; i++)
        dst[i] = tu_int8_to_fp32(src[i], qp);
}

/* ---- UINT4 Packed Storage ---- */

uint8_t tu_uint4_unpack(const uint8_t *packed, size_t index) {
    uint8_t byte = packed[index / 2];
    return (index % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
}

void tu_uint4_pack(uint8_t *packed, size_t index, uint8_t nibble) {
    size_t bi = index / 2;
    if (index % 2 == 0)
        packed[bi] = (packed[bi] & 0xF0) | (nibble & 0x0F);
    else
        packed[bi] = (packed[bi] & 0x0F) | ((nibble & 0x0F) << 4);
}

void tu_fp32_to_uint4_buffer(const float *src, uint8_t *dst, size_t n,
                              const tu_quant_params_t *qp) {
    size_t dst_bytes = (n + 1) / 2;
    memset(dst, 0, dst_bytes);
    for (size_t i = 0; i < n; i++) {
        uint8_t nibble = tu_fp32_to_uint4_nibble(src[i], qp);
        tu_uint4_pack(dst, i, nibble);
    }
}

void tu_uint4_to_fp32_buffer(const uint8_t *src, float *dst, size_t n,
                              const tu_quant_params_t *qp) {
    for (size_t i = 0; i < n; i++) {
        uint8_t nibble = tu_uint4_unpack(src, i);
        dst[i] = tu_uint4_nibble_to_fp32(nibble, qp);
    }
}

/* ================================================================
 * INT MAC Operations
 * ================================================================ */

int32_t tu_int8_dot_product(const int8_t_t *a, const int8_t_t *b, size_t n) {
    int32_t sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

void tu_int8_mma_tile(const int8_t_t *W, const int8_t_t *A,
                       int32_t *O,
                       uint16_t M, uint16_t N, uint16_t K) {
    for (uint16_t m = 0; m < M; m++) {
        for (uint16_t n = 0; n < N; n++) {
            int32_t sum = 0;
            for (uint16_t k = 0; k < K; k++)
                sum += (int32_t)W[m * K + k] * (int32_t)A[k * N + n];
            O[m * N + n] += sum;  /* accumulate into existing output */
        }
    }
}

/* ================================================================
 * Precision Info (for registry)
 * ================================================================ */

const tu_int_precision_info_t tu_int8_precision_info = {
    .elem_bytes = 1,
    .name       = "int8"
};

const tu_int_precision_info_t tu_int4_precision_info = {
    .elem_bytes = 1,  /* 1 byte holds 2 values */
    .name       = "int4"
};
