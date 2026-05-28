/*
 * TinyTU Precision Module
 * ========================
 * IEEE 754 floating-point conversion and arithmetic.
 * Pluggable: supports FP16, FP32. BF16, FP8, INT8 pending.
 */

#ifndef TU_PRECISION_H
#define TU_PRECISION_H

#include "tu_config.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- FP16 type ---- */
typedef uint16_t fp16_t;
typedef float    fp32_t;

/* ---- FP32 ↔ FP16 ---- */
fp16_t tu_fp32_to_fp16(fp32_t v);
fp32_t tu_fp16_to_fp32(fp16_t h);

/* Batch conversion */
void tu_fp32_to_fp16_buffer(const fp32_t *src, fp16_t *dst, size_t n);
void tu_fp16_to_fp32_buffer(const fp16_t *src, fp32_t *dst, size_t n);

/* Round FP32 → FP16 with configurable rounding mode */
fp16_t tu_round_fp32_to_fp16(fp32_t v);

/*
 * Precision type descriptor — for multi-precision support.
 * Each precision type defines: element size, MAC operation, conversion to/from FP32.
 */
typedef enum {
    TU_PREC_FP16 = 0,
    TU_PREC_FP32 = 1,
    TU_PREC_BF16 = 2,
    TU_PREC_FP8  = 3,
    TU_PREC_INT8 = 4,
    TU_PREC_INT4 = 5,
    TU_PREC_COUNT
} tu_precision_t;

typedef struct {
    tu_precision_t type;
    uint8_t        elem_bytes;
    const char    *name;
    /* Convert element at src to FP32 accumulator value */
    fp32_t (*to_fp32)(const void *src);
    /* Convert FP32 accumulator to element, write to dst */
    void   (*from_fp32)(fp32_t val, void *dst);
} tu_precision_desc_t;

/* Lookup precision descriptor */
const tu_precision_desc_t *tu_precision_get(tu_precision_t prec);

/* Register a custom precision type (for extensibility) */
void tu_precision_register(const tu_precision_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif /* TU_PRECISION_H */
