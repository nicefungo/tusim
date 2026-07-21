/*
 * TU Weight Compression — raw, RLE, bitmap sparse, and adaptive streams
 * =================================================================
 * Runtime-configurable weight-stream formats for architecture exploration.
 */
#ifndef TU_WEIGHT_COMPRESS_H
#define TU_WEIGHT_COMPRESS_H

#include "../tu_config.h"
#include "../tu_precision.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tu_config_t;

typedef enum {
    TU_COMPRESS_NONE         = 0,
    TU_COMPRESS_RLE          = 1,
    TU_COMPRESS_ADAPTIVE_RLE = 2, /* Per tensor: framed raw or RLE, whichever is smaller */
    TU_COMPRESS_BITMAP       = 3, /* Exact bitmap plus packed nonzero FP16 values */
    TU_COMPRESS_ADAPTIVE     = 4, /* Per tensor: framed raw/RLE/bitmap minimum */
    TU_COMPRESS_COUNT
} tu_compress_type_t;

/* Stable RLE wire format: uint32 element_count, uint32 run_count, then
 * repeated {uint16 value, uint32 count}. Fields are copied individually so
 * ABI struct padding never enters the stream. */
#define TU_RLE_RUN_BYTES (sizeof(uint16_t) + sizeof(uint32_t))
#define TU_RLE_MAX_ENCODED_SIZE(elem_count) \
    (sizeof(uint32_t) * 2 + (elem_count) * TU_RLE_RUN_BYTES)

typedef struct {
    uint16_t value;
    uint32_t count;
} tu_rle_run_t;

/* Adaptive streams are explicitly framed; a decoder never guesses the codec
 * from payload bytes. Header layout (16 bytes): magic:u32, version:u8,
 * codec:u8, reserved:u16, element_count:u32, payload_bytes:u32. */
#define TU_WEIGHT_FRAME_MAGIC       UINT32_C(0x54555743) /* "CWUT" in LE byte order */
#define TU_WEIGHT_FRAME_VERSION     1u
#define TU_WEIGHT_FRAME_HEADER_BYTES 16u

typedef enum {
    TU_WEIGHT_PAYLOAD_RAW = 0,
    TU_WEIGHT_PAYLOAD_RLE = 1,
    TU_WEIGHT_PAYLOAD_BITMAP = 2
} tu_weight_payload_codec_t;

typedef struct {
    tu_compress_type_t type;
    float              rle_epsilon;
    bool               enabled;
} tu_compress_config_t;

extern const tu_compress_config_t tu_compress_config_default;
tu_compress_config_t tu_compress_config_from_tu_config(const struct tu_config_t *cfg);

int tu_compress_rle(const fp16_t *src, uint32_t element_count,
                    float epsilon, uint8_t *dst, uint32_t dst_capacity,
                    uint32_t *compressed_size_out);
int tu_decompress_rle(const uint8_t *src, uint32_t src_size,
                      fp16_t *dst, uint32_t dst_capacity,
                      uint32_t *decompressed_count_out);
float tu_compress_get_ratio(const uint8_t *compressed_data, uint32_t compressed_size);
uint32_t tu_compress_get_original_count(const uint8_t *compressed_data,
                                        uint32_t compressed_size);
bool tu_compress_validate(const uint8_t *compressed_data, uint32_t compressed_size);

/* Exact bitmap wire format: element_count:u32, nonzero_count:u32,
 * ceil(element_count/8) bitmap bytes, then packed FP16 nonzero values.
 * A set bit means that the corresponding FP16 bit pattern is stored. */
#define TU_BITMAP_HEADER_BYTES 8u
static inline uint32_t tu_compress_bitmap_max_size(uint32_t element_count) {
    uint64_t total = TU_BITMAP_HEADER_BYTES + ((uint64_t)element_count + 7u) / 8u +
                     (uint64_t)element_count * sizeof(fp16_t);
    return total > UINT32_MAX ? UINT32_MAX : (uint32_t)total;
}
int tu_compress_bitmap(const fp16_t *src, uint32_t element_count,
                       uint8_t *dst, uint32_t dst_capacity,
                       uint32_t *compressed_size_out);
int tu_decompress_bitmap(const uint8_t *src, uint32_t src_size,
                         fp16_t *dst, uint32_t dst_capacity,
                         uint32_t *decompressed_count_out);
bool tu_compress_bitmap_validate(const uint8_t *src, uint32_t src_size);

/* Adaptive framed codecs. The legacy adaptive-RLE entry point compares raw
 * and RLE. tu_compress_adaptive compares raw, RLE, and bitmap. Both select a
 * codec only when strictly smaller than the current candidate, so raw wins
 * ties and output never exceeds raw bytes plus the fixed frame. */
static inline uint32_t tu_compress_adaptive_max_size(uint32_t element_count) {
    uint64_t total = TU_WEIGHT_FRAME_HEADER_BYTES +
                     (uint64_t)element_count * sizeof(fp16_t);
    return total > UINT32_MAX ? UINT32_MAX : (uint32_t)total;
}
int tu_compress_adaptive_rle(const fp16_t *src, uint32_t element_count,
                             float epsilon, uint8_t *dst, uint32_t dst_capacity,
                             uint32_t *encoded_size_out,
                             tu_weight_payload_codec_t *selected_codec_out);
int tu_compress_adaptive(const fp16_t *src, uint32_t element_count,
                         float epsilon, uint8_t *dst, uint32_t dst_capacity,
                         uint32_t *encoded_size_out,
                         tu_weight_payload_codec_t *selected_codec_out);
int tu_decompress_adaptive(const uint8_t *src, uint32_t src_size,
                           fp16_t *dst, uint32_t dst_capacity,
                           uint32_t *decompressed_count_out);
bool tu_compress_adaptive_validate(const uint8_t *src, uint32_t src_size);
int tu_compress_adaptive_get_codec(const uint8_t *src, uint32_t src_size,
                                   tu_weight_payload_codec_t *codec_out);

/* Legacy helper remains the RLE worst-case allocator. */
static inline uint32_t tu_compress_max_size(uint32_t element_count) {
    return TU_RLE_MAX_ENCODED_SIZE(element_count);
}

int tu_compress_for_dma(const fp16_t *src, uint32_t element_count,
                        const tu_compress_config_t *cfg,
                        uint8_t *dst, uint32_t dst_capacity,
                        uint32_t *compressed_size_out);
int tu_decompress_from_dma(const uint8_t *src, uint32_t src_size,
                           const tu_compress_config_t *cfg,
                           fp16_t *dst, uint32_t dst_capacity,
                           uint32_t *decompressed_count_out);

#ifdef __cplusplus
}
#endif
#endif
