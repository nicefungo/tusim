/*
 * TU Weight Compression Implementation (Gap M5)
 * ===============================================
 *
 * RLE encoding for FP16 weight tensors with configurable
 * near-value merging for quantized/dead weights.
 */

#include "weight_compress.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Default config */
const tu_compress_config_t tu_compress_config_default = {
    .type        = TU_COMPRESS_RLE,
    .rle_epsilon = 0.0f,
    .enabled     = true,
};

/* ---- Helpers ---- */

/* Compare two FP16 values within epsilon tolerance (in FP32 space) */
static bool fp16_near_equal(fp16_t a, fp16_t b, float epsilon)
{
    if (a == b) return true;
    if (epsilon <= 0.0f) return false;

    float fa = tu_fp16_to_fp32(a);
    float fb = tu_fp16_to_fp32(b);

    /* Handle NaN: NaN != NaN, never merge NaN runs */
    if (fa != fa || fb != fb) return false;

    float diff = (fa > fb) ? (fa - fb) : (fb - fa);
    return diff <= epsilon;
}

/* ================================================================
 * RLE Compression
 * ================================================================ */

int tu_compress_rle(const fp16_t *src, uint32_t element_count,
                     float epsilon,
                     uint8_t *dst, uint32_t dst_capacity,
                     uint32_t *compressed_size_out)
{
    if (!src || !dst || !compressed_size_out) return -1;
    if (element_count == 0) {
        *compressed_size_out = 0;
        return 0;
    }

    /* Header: uint32_t element_count + uint32_t run_count */
    uint32_t header_size = sizeof(uint32_t) * 2;
    uint32_t required = header_size + element_count * sizeof(tu_rle_run_t);

    if (dst_capacity < required) return -1;

    /* Reserve space for header; write at end */
    uint32_t run_count = 0;

    fp16_t current_val = src[0];
    uint32_t current_run = 1;

    for (uint32_t i = 1; i < element_count; i++) {
        if (fp16_near_equal(src[i], current_val, epsilon) && current_run < UINT32_MAX) {
            current_run++;
        } else {
            /* Write completed run */
            tu_rle_run_t run = { .value = current_val, .count = current_run };
            uint32_t offset = header_size + run_count * sizeof(tu_rle_run_t);
            memcpy(dst + offset, &run, sizeof(tu_rle_run_t));
            run_count++;

            current_val = src[i];
            current_run = 1;
        }
    }

    /* Write final run */
    {
        tu_rle_run_t run = { .value = current_val, .count = current_run };
        uint32_t offset = header_size + run_count * sizeof(tu_rle_run_t);
        memcpy(dst + offset, &run, sizeof(tu_rle_run_t));
        run_count++;
    }

    /* Write header */
    memcpy(dst, &element_count, sizeof(uint32_t));
    memcpy(dst + sizeof(uint32_t), &run_count, sizeof(uint32_t));

    *compressed_size_out = header_size + run_count * sizeof(tu_rle_run_t);
    return 0;
}

/* ================================================================
 * RLE Decompression
 * ================================================================ */

int tu_decompress_rle(const uint8_t *src, uint32_t src_size,
                       fp16_t *dst, uint32_t dst_capacity,
                       uint32_t *decompressed_count_out)
{
    if (!src || !dst || !decompressed_count_out) return -1;

    uint32_t header_size = sizeof(uint32_t) * 2;
    if (src_size < header_size) return -1;

    uint32_t element_count, run_count;
    memcpy(&element_count, src, sizeof(uint32_t));
    memcpy(&run_count, src + sizeof(uint32_t), sizeof(uint32_t));

    if (element_count == 0) {
        *decompressed_count_out = 0;
        return 0;
    }

    /* Validate: enough source data for declared runs */
    uint32_t expected_src = header_size + run_count * sizeof(tu_rle_run_t);
    if (src_size < expected_src) return -1;

    /* Validate: destination capacity */
    if (dst_capacity < element_count) return -1;

    /* Decode runs */
    uint32_t dst_pos = 0;
    for (uint32_t i = 0; i < run_count; i++) {
        tu_rle_run_t run;
        memcpy(&run, src + header_size + i * sizeof(tu_rle_run_t),
               sizeof(tu_rle_run_t));

        if (dst_pos + run.count > dst_capacity) return -1;

        for (uint32_t j = 0; j < run.count; j++) {
            dst[dst_pos + j] = run.value;
        }
        dst_pos += run.count;
    }

    /* Verify exact match with header */
    if (dst_pos != element_count) return -1;

    *decompressed_count_out = dst_pos;
    return 0;
}

/* ================================================================
 * Utilities
 * ================================================================ */

float tu_compress_get_ratio(const uint8_t *compressed_data, uint32_t compressed_size)
{
    if (!compressed_data || compressed_size < 8) return 0.0f;

    uint32_t element_count;
    memcpy(&element_count, compressed_data, sizeof(uint32_t));

    if (element_count == 0) return 1.0f;

    float original_size = (float)(element_count * sizeof(fp16_t));
    float comp_size = (float)compressed_size;

    if (comp_size <= 0.0f) return 0.0f;
    return original_size / comp_size;
}

uint32_t tu_compress_get_original_count(const uint8_t *compressed_data,
                                         uint32_t compressed_size)
{
    if (!compressed_data || compressed_size < 4) return 0;

    uint32_t element_count;
    memcpy(&element_count, compressed_data, sizeof(uint32_t));
    return element_count;
}

bool tu_compress_validate(const uint8_t *compressed_data, uint32_t compressed_size)
{
    if (!compressed_data) return false;

    uint32_t header_size = sizeof(uint32_t) * 2;
    if (compressed_size < header_size) return false;

    uint32_t element_count, run_count;
    memcpy(&element_count, compressed_data, sizeof(uint32_t));
    memcpy(&run_count, compressed_data + sizeof(uint32_t), sizeof(uint32_t));

    /* Zero elements is valid */
    if (element_count == 0 && run_count == 0) return true;

    /* Non-zero elements must have at least 1 run */
    if (run_count == 0 || run_count > element_count) return false;

    /* Check buffer size matches runs */
    uint32_t expected = header_size + run_count * sizeof(tu_rle_run_t);
    return compressed_size >= expected;
}

/* ================================================================
 * DMA Integration
 * ================================================================ */

int tu_compress_for_dma(const fp16_t *src, uint32_t element_count,
                         const tu_compress_config_t *cfg,
                         uint8_t *dst, uint32_t dst_capacity,
                         uint32_t *compressed_size_out)
{
    if (!cfg || !cfg->enabled || cfg->type == TU_COMPRESS_NONE) {
        /* Pass-through: just copy the raw data */
        if (!dst || !compressed_size_out) return -1;
        uint32_t raw_size = element_count * sizeof(fp16_t);
        if (dst_capacity < raw_size) return -1;
        memcpy(dst, src, raw_size);
        *compressed_size_out = raw_size;
        return 0;
    }

    if (cfg->type == TU_COMPRESS_RLE) {
        return tu_compress_rle(src, element_count,
                                cfg->rle_epsilon,
                                dst, dst_capacity,
                                compressed_size_out);
    }

    return -1;  /* Unknown compression type */
}

int tu_decompress_from_dma(const uint8_t *src, uint32_t src_size,
                            const tu_compress_config_t *cfg,
                            fp16_t *dst, uint32_t dst_capacity,
                            uint32_t *decompressed_count_out)
{
    if (!cfg || !cfg->enabled || cfg->type == TU_COMPRESS_NONE) {
        /* Pass-through: just copy */
        if (!dst || !decompressed_count_out) return -1;
        uint32_t elem_count = src_size / sizeof(fp16_t);
        if (dst_capacity < elem_count) return -1;
        memcpy(dst, src, src_size);
        *decompressed_count_out = elem_count;
        return 0;
    }

    if (cfg->type == TU_COMPRESS_RLE) {
        return tu_decompress_rle(src, src_size,
                                  dst, dst_capacity,
                                  decompressed_count_out);
    }

    return -1;
}
