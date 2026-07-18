/* TU weight compression implementation: raw, RLE, adaptive framed raw/RLE. */
#include "weight_compress.h"
#include "../infra/config.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

const tu_compress_config_t tu_compress_config_default = {
    .type = TU_COMPRESS_RLE, .rle_epsilon = 0.0f, .enabled = true,
};

tu_compress_config_t tu_compress_config_from_tu_config(const struct tu_config_t *cfg)
{
    tu_compress_config_t out = tu_compress_config_default;
    if (!cfg) {
        out.enabled = false;
        out.type = TU_COMPRESS_NONE;
        return out;
    }
    out.enabled = cfg->compression_enabled;
    out.type = (tu_compress_type_t)cfg->compression_type;
    out.rle_epsilon = (float)cfg->compression_rle_epsilon;
    return out;
}

static bool fp16_near_equal(fp16_t a, fp16_t b, float epsilon)
{
    if (a == b) return true;
    if (epsilon <= 0.0f) return false;
    float fa = tu_fp16_to_fp32(a), fb = tu_fp16_to_fp32(b);
    if (isnan(fa) || isnan(fb)) return false;
    return fabsf(fa - fb) <= epsilon;
}

static uint32_t count_runs(const fp16_t *src, uint32_t n, float epsilon)
{
    if (n == 0) return 0;
    uint32_t runs = 1;
    fp16_t current = src[0];
    for (uint32_t i = 1; i < n; i++) {
        if (!fp16_near_equal(src[i], current, epsilon)) {
            runs++;
            current = src[i];
        }
    }
    return runs;
}

int tu_compress_rle(const fp16_t *src, uint32_t element_count,
                    float epsilon, uint8_t *dst, uint32_t dst_capacity,
                    uint32_t *compressed_size_out)
{
    if (!src || !dst || !compressed_size_out || epsilon < 0.0f) return -1;
    if (element_count == 0) {
        *compressed_size_out = 0;
        return 0;
    }
    const uint32_t header_size = sizeof(uint32_t) * 2;
    if (element_count > (UINT32_MAX - header_size) / TU_RLE_RUN_BYTES) return -1;
    uint32_t exact_runs = count_runs(src, element_count, epsilon);
    uint32_t required = header_size + exact_runs * TU_RLE_RUN_BYTES;
    if (dst_capacity < required) return -1;

    uint32_t run_count = 0, current_run = 1;
    fp16_t current_val = src[0];
    for (uint32_t i = 1; i < element_count; i++) {
        if (fp16_near_equal(src[i], current_val, epsilon) && current_run < UINT32_MAX) {
            current_run++;
        } else {
            uint32_t off = header_size + run_count * TU_RLE_RUN_BYTES;
            memcpy(dst + off, &current_val, sizeof(current_val));
            memcpy(dst + off + sizeof(current_val), &current_run, sizeof(current_run));
            run_count++;
            current_val = src[i];
            current_run = 1;
        }
    }
    uint32_t off = header_size + run_count * TU_RLE_RUN_BYTES;
    memcpy(dst + off, &current_val, sizeof(current_val));
    memcpy(dst + off + sizeof(current_val), &current_run, sizeof(current_run));
    run_count++;
    memcpy(dst, &element_count, sizeof(element_count));
    memcpy(dst + sizeof(element_count), &run_count, sizeof(run_count));
    *compressed_size_out = header_size + run_count * TU_RLE_RUN_BYTES;
    return 0;
}

int tu_decompress_rle(const uint8_t *src, uint32_t src_size,
                      fp16_t *dst, uint32_t dst_capacity,
                      uint32_t *decompressed_count_out)
{
    if (!src || !dst || !decompressed_count_out) return -1;
    const uint32_t header_size = sizeof(uint32_t) * 2;
    if (src_size < header_size) return -1;
    uint32_t element_count, run_count;
    memcpy(&element_count, src, sizeof(element_count));
    memcpy(&run_count, src + sizeof(element_count), sizeof(run_count));
    if (element_count == 0) {
        if (run_count != 0) return -1;
        *decompressed_count_out = 0;
        return 0;
    }
    if (run_count == 0 || run_count > element_count ||
        run_count > (UINT32_MAX - header_size) / TU_RLE_RUN_BYTES) return -1;
    uint32_t expected = header_size + run_count * TU_RLE_RUN_BYTES;
    if (src_size < expected || dst_capacity < element_count) return -1;

    uint32_t dst_pos = 0;
    for (uint32_t i = 0; i < run_count; i++) {
        tu_rle_run_t run = {0};
        uint32_t roff = header_size + i * TU_RLE_RUN_BYTES;
        memcpy(&run.value, src + roff, sizeof(run.value));
        memcpy(&run.count, src + roff + sizeof(run.value), sizeof(run.count));
        if (run.count == 0 || run.count > element_count - dst_pos) return -1;
        for (uint32_t j = 0; j < run.count; j++) dst[dst_pos + j] = run.value;
        dst_pos += run.count;
    }
    if (dst_pos != element_count) return -1;
    *decompressed_count_out = dst_pos;
    return 0;
}

float tu_compress_get_ratio(const uint8_t *data, uint32_t size)
{
    if (!data || size < 8) return 0.0f;
    uint32_t n;
    memcpy(&n, data, sizeof(n));
    return n == 0 ? 1.0f : (float)(n * sizeof(fp16_t)) / (float)size;
}

uint32_t tu_compress_get_original_count(const uint8_t *data, uint32_t size)
{
    uint32_t n = 0;
    if (data && size >= sizeof(n)) memcpy(&n, data, sizeof(n));
    return n;
}

bool tu_compress_validate(const uint8_t *data, uint32_t size)
{
    if (!data || size < 8) return false;
    uint32_t n, runs;
    memcpy(&n, data, 4);
    memcpy(&runs, data + 4, 4);
    if (n == 0) return runs == 0 && size == 8;
    if (runs == 0 || runs > n || runs > (UINT32_MAX - 8u) / TU_RLE_RUN_BYTES) return false;
    return size == 8u + runs * TU_RLE_RUN_BYTES;
}

static void frame_write(uint8_t *dst, tu_weight_payload_codec_t codec,
                        uint32_t elements, uint32_t payload_bytes)
{
    uint32_t magic = TU_WEIGHT_FRAME_MAGIC;
    uint16_t reserved = 0;
    memcpy(dst, &magic, 4);
    dst[4] = TU_WEIGHT_FRAME_VERSION;
    dst[5] = (uint8_t)codec;
    memcpy(dst + 6, &reserved, 2);
    memcpy(dst + 8, &elements, 4);
    memcpy(dst + 12, &payload_bytes, 4);
}

static bool frame_read(const uint8_t *src, uint32_t size,
                       tu_weight_payload_codec_t *codec,
                       uint32_t *elements, uint32_t *payload_bytes)
{
    if (!src || size < TU_WEIGHT_FRAME_HEADER_BYTES) return false;
    uint32_t magic;
    uint16_t reserved;
    memcpy(&magic, src, 4);
    memcpy(&reserved, src + 6, 2);
    memcpy(elements, src + 8, 4);
    memcpy(payload_bytes, src + 12, 4);
    *codec = (tu_weight_payload_codec_t)src[5];
    if (magic != TU_WEIGHT_FRAME_MAGIC || src[4] != TU_WEIGHT_FRAME_VERSION ||
        reserved != 0 || (*codec != TU_WEIGHT_PAYLOAD_RAW && *codec != TU_WEIGHT_PAYLOAD_RLE)) return false;
    return *payload_bytes == size - TU_WEIGHT_FRAME_HEADER_BYTES;
}

int tu_compress_adaptive_rle(const fp16_t *src, uint32_t n, float epsilon,
                             uint8_t *dst, uint32_t cap, uint32_t *size_out,
                             tu_weight_payload_codec_t *codec_out)
{
    if (!src || !dst || !size_out || epsilon < 0.0f) return -1;
    if (n > (UINT32_MAX - TU_WEIGHT_FRAME_HEADER_BYTES) / sizeof(fp16_t)) return -1;
    uint32_t raw_bytes = n * (uint32_t)sizeof(fp16_t);
    if (cap < TU_WEIGHT_FRAME_HEADER_BYTES + raw_bytes) return -1;

    uint32_t runs = count_runs(src, n, epsilon);
    uint64_t rle_bytes = n == 0 ? 8u : 8u + (uint64_t)runs * TU_RLE_RUN_BYTES;
    tu_weight_payload_codec_t codec = rle_bytes < raw_bytes ?
        TU_WEIGHT_PAYLOAD_RLE : TU_WEIGHT_PAYLOAD_RAW;
    uint32_t payload = raw_bytes;
    if (codec == TU_WEIGHT_PAYLOAD_RLE) {
        if (tu_compress_rle(src, n, epsilon, dst + TU_WEIGHT_FRAME_HEADER_BYTES,
                            cap - TU_WEIGHT_FRAME_HEADER_BYTES, &payload) != 0) return -1;
    } else if (raw_bytes) {
        memcpy(dst + TU_WEIGHT_FRAME_HEADER_BYTES, src, raw_bytes);
    }
    frame_write(dst, codec, n, payload);
    *size_out = TU_WEIGHT_FRAME_HEADER_BYTES + payload;
    if (codec_out) *codec_out = codec;
    return 0;
}

bool tu_compress_adaptive_validate(const uint8_t *src, uint32_t size)
{
    tu_weight_payload_codec_t codec;
    uint32_t n, payload;
    if (!frame_read(src, size, &codec, &n, &payload)) return false;
    if (codec == TU_WEIGHT_PAYLOAD_RAW)
        return n <= UINT32_MAX / sizeof(fp16_t) && payload == n * sizeof(fp16_t);
    return tu_compress_validate(src + TU_WEIGHT_FRAME_HEADER_BYTES, payload) &&
           tu_compress_get_original_count(src + TU_WEIGHT_FRAME_HEADER_BYTES, payload) == n;
}

int tu_compress_adaptive_get_codec(const uint8_t *src, uint32_t size,
                                   tu_weight_payload_codec_t *codec_out)
{
    uint32_t n, payload;
    if (!codec_out || !frame_read(src, size, codec_out, &n, &payload)) return -1;
    return 0;
}

int tu_decompress_adaptive(const uint8_t *src, uint32_t size,
                           fp16_t *dst, uint32_t cap, uint32_t *count_out)
{
    if (!dst || !count_out || !tu_compress_adaptive_validate(src, size)) return -1;
    tu_weight_payload_codec_t codec;
    uint32_t n, payload;
    if (!frame_read(src, size, &codec, &n, &payload) || cap < n) return -1;
    const uint8_t *body = src + TU_WEIGHT_FRAME_HEADER_BYTES;
    if (codec == TU_WEIGHT_PAYLOAD_RAW) {
        if (payload) memcpy(dst, body, payload);
        *count_out = n;
        return 0;
    }
    return tu_decompress_rle(body, payload, dst, cap, count_out);
}

int tu_compress_for_dma(const fp16_t *src, uint32_t n,
                        const tu_compress_config_t *cfg,
                        uint8_t *dst, uint32_t cap, uint32_t *size_out)
{
    if (!src || !dst || !size_out) return -1;
    if (!cfg || !cfg->enabled || cfg->type == TU_COMPRESS_NONE) {
        if (n > UINT32_MAX / sizeof(fp16_t)) return -1;
        uint32_t raw = n * (uint32_t)sizeof(fp16_t);
        if (cap < raw) return -1;
        if (raw) memcpy(dst, src, raw);
        *size_out = raw;
        return 0;
    }
    if (cfg->type == TU_COMPRESS_RLE)
        return tu_compress_rle(src, n, cfg->rle_epsilon, dst, cap, size_out);
    if (cfg->type == TU_COMPRESS_ADAPTIVE_RLE)
        return tu_compress_adaptive_rle(src, n, cfg->rle_epsilon, dst, cap,
                                        size_out, NULL);
    return -1;
}

int tu_decompress_from_dma(const uint8_t *src, uint32_t size,
                           const tu_compress_config_t *cfg,
                           fp16_t *dst, uint32_t cap, uint32_t *count_out)
{
    if (!src || !dst || !count_out) return -1;
    if (!cfg || !cfg->enabled || cfg->type == TU_COMPRESS_NONE) {
        if (size % sizeof(fp16_t) != 0) return -1;
        uint32_t n = size / sizeof(fp16_t);
        if (cap < n) return -1;
        if (size) memcpy(dst, src, size);
        *count_out = n;
        return 0;
    }
    if (cfg->type == TU_COMPRESS_RLE)
        return tu_decompress_rle(src, size, dst, cap, count_out);
    if (cfg->type == TU_COMPRESS_ADAPTIVE_RLE)
        return tu_decompress_adaptive(src, size, dst, cap, count_out);
    return -1;
}
