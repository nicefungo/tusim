/*
 * TU Weight Compression — RLE + Frequency-Aware Encoding (Gap M5)
 * ================================================================
 *
 * Compresses weight tensors for reduced memory bandwidth and storage.
 * Supports Run-Length Encoding (RLE) optimized for quantized/dead weights.
 *
 * Gap: M5 — Memory compression (P2)
 * Dependencies: tu_config.h, tu_precision.h
 *
 * Design:
 *   - RLE: Run-length encoding for repeated values (zeros, saturated, identical)
 *   - Quantization-aware: configurable epsilon for near-value merging
 *   - Config-driven: enable/disable, compression type, epsilon
 *   - DMA integration: compress on host→TU transfer, transparent decompression
 */

#ifndef TU_WEIGHT_COMPRESS_H
#define TU_WEIGHT_COMPRESS_H

#include "../tu_config.h"
#include "../tu_precision.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Compression Types ---- */
typedef enum {
    TU_COMPRESS_NONE    = 0,  /* Pass-through, no compression */
    TU_COMPRESS_RLE     = 1,  /* Run-length encoding for repeated values */
    TU_COMPRESS_COUNT
} tu_compress_type_t;

/* ---- RLE Encoding ---- */

/*
 * RLE-encoded data format:
 *   Header:  uint32_t original_elem_count (number of elements before compression)
 *            uint32_t encoded_runs       (number of runs)
 *   Run:     uint16_t value              (FP16 bit pattern)
 *            uint32_t count              (consecutive repetitions of this value)
 *
 * Worst case: no runs (alternating values) → 6 bytes per element (300% overhead)
 * Best case: all identical → 6 bytes for entire tensor (~100% / N compression)
 * Typical: sparse/quantized weights → 2-10x compression
 */

/* Maximum compressed size for worst-case RLE encoding */
#define TU_RLE_MAX_ENCODED_SIZE(elem_count) \
    (sizeof(uint32_t) * 2 + (elem_count) * sizeof(tu_rle_run_t))

/* RLE run entry */
typedef struct {
    uint16_t value;     /* FP16 bit pattern */
    uint32_t count;     /* Number of consecutive occurrences */
} tu_rle_run_t;

/* ---- Compression Config ---- */
typedef struct {
    tu_compress_type_t  type;           /* Compression algorithm */
    float               rle_epsilon;    /* Max difference to merge into a run (0 = exact) */
    bool                enabled;        /* Enable/disable compression */
} tu_compress_config_t;

/* Default config: RLE with exact matching */
extern const tu_compress_config_t tu_compress_config_default;

/* ================================================================
 * Compression API
 * ================================================================ */

/*
 * Compress an FP16 weight tensor using RLE.
 *
 *   src:         Source FP16 data (element_count elements)
 *   element_count: Number of FP16 elements in src
 *   epsilon:     Maximum absolute difference to consider values "equal" for merging
 *                 0.0f = exact matching only; 0.001f = near-value merging
 *   dst:         Output buffer (caller-allocated)
 *   dst_capacity: Size of dst buffer in bytes
 *   compressed_size_out: Written with actual compressed size in bytes
 *
 * Returns 0 on success, -1 if dst_capacity is insufficient.
 */
int tu_compress_rle(const fp16_t *src, uint32_t element_count,
                     float epsilon,
                     uint8_t *dst, uint32_t dst_capacity,
                     uint32_t *compressed_size_out);

/*
 * Decompress an RLE-encoded weight tensor.
 *
 *   src:         Compressed data (from tu_compress_rle)
 *   src_size:    Size of compressed data in bytes
 *   dst:         Output buffer for decompressed FP16 data
 *   dst_capacity: Max FP16 elements the buffer can hold
 *   decompressed_count_out: Written with actual number of elements decompressed
 *
 * Returns 0 on success, -1 on error (corrupt data, insufficient capacity).
 */
int tu_decompress_rle(const uint8_t *src, uint32_t src_size,
                       fp16_t *dst, uint32_t dst_capacity,
                       uint32_t *decompressed_count_out);

/*
 * Get the compression ratio for a compressed buffer.
 * Ratio = original_size / compressed_size.
 * Returns 1.0 if no compression, or 0.0 on invalid input.
 */
float tu_compress_get_ratio(const uint8_t *compressed_data, uint32_t compressed_size);

/*
 * Get the original element count from a compressed buffer header.
 * Returns 0 on invalid input.
 */
uint32_t tu_compress_get_original_count(const uint8_t *compressed_data,
                                         uint32_t compressed_size);

/*
 * Validate a compressed buffer header.
 * Returns true if the header is valid.
 */
bool tu_compress_validate(const uint8_t *compressed_data, uint32_t compressed_size);

/* ---- DMA Integration Helpers ---- */

/*
 * Estimate the maximum compressed size for a given input.
 * Use this to allocate destination buffers.
 * For RLE: worst case is every element is unique → sizeof(header) + N × sizeof(run).
 */
static inline uint32_t tu_compress_max_size(uint32_t element_count) {
    return TU_RLE_MAX_ENCODED_SIZE(element_count);
}

/*
 * Compress FP16 data for DMA transfer (host → TU).
 * Wraps tu_compress_rle with config-driven parameters.
 * Returns 0 on success, -1 on failure.
 */
int tu_compress_for_dma(const fp16_t *src, uint32_t element_count,
                         const tu_compress_config_t *cfg,
                         uint8_t *dst, uint32_t dst_capacity,
                         uint32_t *compressed_size_out);

/*
 * Decompress data after DMA transfer (TU side).
 * Returns 0 on success, -1 on failure.
 */
int tu_decompress_from_dma(const uint8_t *src, uint32_t src_size,
                            const tu_compress_config_t *cfg,
                            fp16_t *dst, uint32_t dst_capacity,
                            uint32_t *decompressed_count_out);

#ifdef __cplusplus
}
#endif

#endif /* TU_WEIGHT_COMPRESS_H */
