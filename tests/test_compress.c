/*
 * TU Weight Compression Tests (Gap M5)
 * ======================================
 *
 * Tests for RLE compression/decompression round-trip correctness,
 * compression ratios, edge cases, config-driven behavior,
 * and DMA integration helpers.
 */

#include "test_framework.h"
#include "tu_cmodel.h"
#include "tu_cmodel/memory/weight_compress.h"
#include "tu_cmodel/infra/config.h"

#include <string.h>
#include <stdlib.h>

tu_test_stats_t g_test_stats;
#define CHECK(cond, msg) do { if (!(cond)) { FAIL("%s", msg); return; } } while(0)

/* ================================================================
 * Test 1: RLE round-trip — all identical values
 * ================================================================ */
static void test_rle_all_same(void) {
    fp16_t src[100];
    fp16_t v = tu_fp32_to_fp16(1.5f);
    for (int i = 0; i < 100; i++) src[i] = v;

    uint32_t max_sz = tu_compress_max_size(100);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;
    int ret = tu_compress_rle(src, 100, 0.0f, comp, max_sz, &comp_sz);
    CHECK(ret == 0, "compress all-same should succeed");
    CHECK(comp_sz < max_sz, "compressed should be smaller than max");
    CHECK(comp_sz == 8 + 6, "all-same: 8B header + 6B run = 14B");

    float ratio = tu_compress_get_ratio(comp, comp_sz);
    CHECK(ratio > 10.0f, "all-same should have high ratio");

    fp16_t dst[100];
    uint32_t dec_count;
    ret = tu_decompress_rle(comp, comp_sz, dst, 100, &dec_count);
    CHECK(ret == 0, "decompress should succeed");
    CHECK(dec_count == 100, "decompressed count should be 100");

    for (int i = 0; i < 100; i++) {
        CHECK(dst[i] == v, "all elements should match original");
    }

    free(comp);
    PASS();
}

/* ================================================================
 * Test 2: RLE round-trip — alternating values (worst case)
 * ================================================================ */
static void test_rle_alternating(void) {
    fp16_t src[10];
    for (int i = 0; i < 10; i++) {
        src[i] = tu_fp32_to_fp16((float)i);
    }

    uint32_t max_sz = tu_compress_max_size(10);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;
    int ret = tu_compress_rle(src, 10, 0.0f, comp, max_sz, &comp_sz);
    CHECK(ret == 0, "compress alternating should succeed");

    fp16_t dst[10];
    uint32_t dec_count;
    ret = tu_decompress_rle(comp, comp_sz, dst, 10, &dec_count);
    CHECK(ret == 0, "decompress alternating should succeed");
    CHECK(dec_count == 10, "count should be 10");

    for (int i = 0; i < 10; i++) {
        CHECK(dst[i] == src[i], "all elements should match");
    }

    free(comp);
    PASS();
}

/* ================================================================
 * Test 3: RLE round-trip — mixed runs
 * ================================================================ */
static void test_rle_mixed(void) {
    /* Pattern: AAA BBB C DDDD EEE (5 runs of 13 elements) */
    fp16_t a = tu_fp32_to_fp16(1.0f);
    fp16_t b = tu_fp32_to_fp16(2.0f);
    fp16_t c = tu_fp32_to_fp16(3.0f);
    fp16_t d = tu_fp32_to_fp16(4.0f);
    fp16_t e = tu_fp32_to_fp16(5.0f);
    fp16_t src[] = {a,a,a, b,b,b, c, d,d,d,d, e,e,e};
    uint32_t N = 13;

    uint32_t max_sz = tu_compress_max_size(N);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;
    CHECK(tu_compress_rle(src, N, 0.0f, comp, max_sz, &comp_sz) == 0, "compress mixed");

    fp16_t dst[13];
    uint32_t dec_count;
    CHECK(tu_decompress_rle(comp, comp_sz, dst, 13, &dec_count) == 0, "decompress mixed");
    CHECK(dec_count == 13, "count 13");

    CHECK(dst[0] == a && dst[2] == a, "run A");
    CHECK(dst[3] == b && dst[5] == b, "run B");
    CHECK(dst[6] == c, "run C");
    CHECK(dst[7] == d && dst[10] == d, "run D");
    CHECK(dst[11] == e && dst[12] == e, "run E");

    free(comp);
    PASS();
}

/* ================================================================
 * Test 4: RLE — zeros and special values
 * ================================================================ */
static void test_rle_zeros(void) {
    fp16_t src[50] = {0}; /* All zeros */
    fp16_t v1 = tu_fp32_to_fp16(42.0f);
    src[10] = v1;
    src[11] = v1;

    uint32_t max_sz = tu_compress_max_size(50);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;
    CHECK(tu_compress_rle(src, 50, 0.0f, comp, max_sz, &comp_sz) == 0, "compress zeros");

    fp16_t dst[50];
    uint32_t dec_count;
    CHECK(tu_decompress_rle(comp, comp_sz, dst, 50, &dec_count) == 0, "decompress zeros");
    CHECK(dec_count == 50, "count 50");
    CHECK(dst[0] == 0, "zero");
    CHECK(dst[10] == v1, "non-zero");
    CHECK(dst[49] == 0, "zero end");

    free(comp);
    PASS();
}

/* ================================================================
 * Test 5: RLE — single element
 * ================================================================ */
static void test_rle_single(void) {
    fp16_t src = tu_fp32_to_fp16(3.14f);
    uint32_t max_sz = tu_compress_max_size(1);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;
    CHECK(tu_compress_rle(&src, 1, 0.0f, comp, max_sz, &comp_sz) == 0, "compress single");

    fp16_t dst;
    uint32_t dec_count;
    CHECK(tu_decompress_rle(comp, comp_sz, &dst, 1, &dec_count) == 0, "decompress single");
    CHECK(dec_count == 1, "count 1");
    CHECK(dst == src, "value match");

    free(comp);
    PASS();
}

/* ================================================================
 * Test 6: RLE — epsilon-based merging
 * ================================================================ */
static void test_rle_epsilon(void) {
    /* Values within 0.01 of each other */
    fp16_t src[6] = {
        tu_fp32_to_fp16(1.000f),
        tu_fp32_to_fp16(1.005f), /* within 0.01 epsilon */
        tu_fp32_to_fp16(1.002f), /* within 0.01 epsilon */
        tu_fp32_to_fp16(2.000f), /* far away */
        tu_fp32_to_fp16(2.003f), /* within epsilon of 2.0 */
        tu_fp32_to_fp16(2.001f), /* within epsilon of 2.0 */
    };

    uint32_t max_sz = tu_compress_max_size(6);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;

    /* With epsilon=0.01: should merge 1.0/1.005/1.002 into one run,
     * and 2.0/2.003/2.001 into another → 2 runs total */
    CHECK(tu_compress_rle(src, 6, 0.05f, comp, max_sz, &comp_sz) == 0, "compress epsilon");

    /* Header (8B) + 2 runs × 6B = 20B */
    CHECK(comp_sz <= 32, "epsilon merge should give few runs");

    fp16_t dst[6];
    uint32_t dec_count;
    CHECK(tu_decompress_rle(comp, comp_sz, dst, 6, &dec_count) == 0, "decompress epsilon");
    CHECK(dec_count == 6, "count 6");

    /* Without epsilon: exact match only — reallocate comp buffer */
    free(comp);
    comp = (uint8_t *)malloc(tu_compress_max_size(6));
    CHECK(tu_compress_rle(src, 6, 0.0f, comp, tu_compress_max_size(6), &comp_sz) == 0, "compress no-epsilon");
    /* Should be more runs since exact match only */
    CHECK(comp_sz >= 20, "no-epsilon should have more runs");

    free(comp);
    PASS();
}

/* ================================================================
 * Test 7: Compression ratio tracking
 * ================================================================ */
static void test_compression_ratio(void) {
    fp16_t sparse[200] = {0};
    fp16_t v = tu_fp32_to_fp16(7.0f);
    sparse[50] = v;
    sparse[51] = v;

    uint32_t max_sz = tu_compress_max_size(200);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;
    tu_compress_rle(sparse, 200, 0.0f, comp, max_sz, &comp_sz);

    float ratio = tu_compress_get_ratio(comp, comp_sz);
    /* 200 elements × 2B = 400B raw. Compressed should be much less. */
    CHECK(ratio > 5.0f, "sparse data should compress well");

    uint32_t orig = tu_compress_get_original_count(comp, comp_sz);
    CHECK(orig == 200, "original count should be 200");

    CHECK(tu_compress_validate(comp, comp_sz), "validate should pass");

    free(comp);
    PASS();
}

/* ================================================================
 * Test 8: Validate corrupt data
 * ================================================================ */
static void test_validate_corrupt(void) {
    /* Too short */
    uint8_t short_buf[2] = {0};
    CHECK(!tu_compress_validate(short_buf, 2), "short buffer invalid");

    /* Run count > element count */
    uint32_t bad_hdr[2] = {5, 10}; /* 5 elements but 10 runs */
    CHECK(!tu_compress_validate((uint8_t*)bad_hdr, 100), "bad runs invalid");

    /* Zero elements, zero runs — valid */
    uint32_t empty_hdr[2] = {0, 0};
    CHECK(tu_compress_validate((uint8_t*)empty_hdr, 8), "empty valid");

    PASS();
}

/* ================================================================
 * Test 9: Config-driven compression (DMA path)
 * ================================================================ */
static void test_dma_integration(void) {
    fp16_t src[50];
    for (int i = 0; i < 50; i++) src[i] = tu_fp32_to_fp16((float)(i / 10));

    /* Config: RLE enabled */
    tu_compress_config_t cfg = tu_compress_config_default;
    cfg.enabled = true;
    cfg.type = TU_COMPRESS_RLE;
    cfg.rle_epsilon = 0.0f;

    uint32_t max_sz = tu_compress_max_size(50);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;
    CHECK(tu_compress_for_dma(src, 50, &cfg, comp, max_sz, &comp_sz) == 0, "compress for dma");

    fp16_t dst[50];
    uint32_t dec_count;
    CHECK(tu_decompress_from_dma(comp, comp_sz, &cfg, dst, 50, &dec_count) == 0, "decompress from dma");
    CHECK(dec_count == 50, "count 50");

    for (int i = 0; i < 50; i++) {
        CHECK(dst[i] == src[i], "dma round-trip match");
    }

    free(comp);
    PASS();
}

/* ================================================================
 * Test 10: Disabled compression (pass-through)
 * ================================================================ */
static void test_disabled(void) {
    fp16_t src[20];
    for (int i = 0; i < 20; i++) src[i] = tu_fp32_to_fp16((float)i);

    tu_compress_config_t cfg = tu_compress_config_default;
    cfg.enabled = false;

    uint32_t max_sz = 20 * sizeof(fp16_t);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;
    CHECK(tu_compress_for_dma(src, 20, &cfg, comp, max_sz, &comp_sz) == 0, "pass-through compress");

    /* Should be same as raw */
    CHECK(comp_sz == 20 * sizeof(fp16_t), "pass-through size = raw size");

    fp16_t dst[20];
    uint32_t dec_count;
    CHECK(tu_decompress_from_dma(comp, comp_sz, &cfg, dst, 20, &dec_count) == 0, "pass-through decompress");
    CHECK(dec_count == 20, "pass-through count");

    for (int i = 0; i < 20; i++) {
        CHECK(dst[i] == src[i], "pass-through match");
    }

    free(comp);
    PASS();
}

/* ================================================================
 * Test 11: Canonical runtime config mapping and validation
 * ================================================================ */
static void test_runtime_config(void) {
    const char *json =
        "{\"tu\":{\"weight_compression\":{"
        "\"enabled\":true,\"type\":\"rle\",\"rle_epsilon\":0.015}}}";
    tu_config_t cfg;
    CHECK(tu_config_load_string(json, &cfg, NULL, 0) == 0, "parse compression config");
    CHECK(cfg.compression_enabled, "compression enabled");
    CHECK(cfg.compression_type == 1, "RLE type parsed");
    CHECK(cfg.compression_rle_epsilon > 0.014 &&
          cfg.compression_rle_epsilon < 0.016, "epsilon parsed");

    tu_compress_config_t codec = tu_compress_config_from_tu_config(&cfg);
    CHECK(codec.enabled && codec.type == TU_COMPRESS_RLE, "runtime mapping");
    CHECK(codec.rle_epsilon > 0.014f && codec.rle_epsilon < 0.016f,
          "runtime epsilon mapping");

    CHECK(tu_config_load_string(
        "{\"weight_compression\":{\"type\":\"huffman\"}}",
        &cfg, NULL, 0) != 0, "reject unsupported codec");
    CHECK(tu_config_load_string(
        "{\"weight_compression\":{\"type\":\"rle\",\"rle_epsilon\":-1}}",
        &cfg, NULL, 0) != 0, "reject negative epsilon");
    PASS();
}

/* ================================================================
 * Test 12: NULL safety
 * ================================================================ */
static void test_null_safety(void) {
    CHECK(tu_compress_rle(NULL, 10, 0.0f, NULL, 100, NULL) == -1, "compress null");
    CHECK(tu_decompress_rle(NULL, 10, NULL, 10, NULL) == -1, "decompress null");

    float ratio = tu_compress_get_ratio(NULL, 10);
    CHECK(ratio == 0.0f, "ratio null");

    uint32_t cnt = tu_compress_get_original_count(NULL, 0);
    CHECK(cnt == 0, "orig count null");

    CHECK(!tu_compress_validate(NULL, 10), "validate null");

    PASS();
}

/* ================================================================
 * Test 12: Large sparse tensor compression
 * ================================================================ */
static void test_large_sparse(void) {
    #define N 1000
    fp16_t *src = (fp16_t *)calloc(N, sizeof(fp16_t));
    fp16_t v = tu_fp32_to_fp16(1.0f);
    /* Sparse: 98% zeros, 2% ones */
    for (int i = 0; i < N; i += 50) src[i] = v;

    uint32_t max_sz = tu_compress_max_size(N);
    uint8_t *comp = (uint8_t *)malloc(max_sz);
    uint32_t comp_sz;
    CHECK(tu_compress_rle(src, N, 0.0f, comp, max_sz, &comp_sz) == 0, "compress large");

    float ratio = tu_compress_get_ratio(comp, comp_sz);
    /* 1000 × 2 = 2000B raw. Sparse should compress well. */
    CHECK(ratio > 3.0f, "large sparse should compress");

    fp16_t *dst = (fp16_t *)calloc(N, sizeof(fp16_t));
    uint32_t dec_count;
    CHECK(tu_decompress_rle(comp, comp_sz, dst, N, &dec_count) == 0, "decompress large");
    CHECK(dec_count == N, "large count");

    for (int i = 0; i < N; i++) {
        CHECK(dst[i] == src[i], "large element match");
    }

    free(src); free(dst); free(comp);
    #undef N
    PASS();
}

/* ================================================================
 * Adaptive framed raw/RLE selection
 * ================================================================ */
static void test_adaptive_selects_raw(void) {
    fp16_t src[64], dst[64];
    for (int i = 0; i < 64; i++) src[i] = (fp16_t)(i + 1);
    uint32_t cap = tu_compress_adaptive_max_size(64), size = 0, count = 0;
    uint8_t *encoded = malloc(cap);
    tu_weight_payload_codec_t codec = TU_WEIGHT_PAYLOAD_RLE;
    CHECK(tu_compress_adaptive_rle(src, 64, 0.0f, encoded, cap, &size, &codec) == 0,
          "adaptive raw encode");
    CHECK(codec == TU_WEIGHT_PAYLOAD_RAW, "incompressible tensor selects raw");
    CHECK(size == TU_WEIGHT_FRAME_HEADER_BYTES + sizeof(src), "raw plus fixed frame only");
    CHECK(tu_compress_adaptive_validate(encoded, size), "raw frame validates");
    CHECK(tu_decompress_adaptive(encoded, size, dst, 64, &count) == 0, "raw frame decode");
    CHECK(count == 64 && memcmp(src, dst, sizeof(src)) == 0, "raw exact round-trip");
    free(encoded);
    PASS();
}

static void test_adaptive_selects_rle(void) {
    fp16_t src[128] = {0}, dst[128];
    uint32_t cap = tu_compress_adaptive_max_size(128), size = 0, count = 0;
    uint8_t *encoded = malloc(cap);
    tu_weight_payload_codec_t codec = TU_WEIGHT_PAYLOAD_RAW, parsed;
    CHECK(tu_compress_adaptive_rle(src, 128, 0.0f, encoded, cap, &size, &codec) == 0,
          "adaptive RLE encode");
    CHECK(codec == TU_WEIGHT_PAYLOAD_RLE, "repeated tensor selects RLE");
    CHECK(size == TU_WEIGHT_FRAME_HEADER_BYTES + 14, "frame plus one-run RLE");
    CHECK(size <= TU_WEIGHT_FRAME_HEADER_BYTES + sizeof(src), "never exceeds framed raw");
    CHECK(tu_compress_adaptive_get_codec(encoded, size, &parsed) == 0 && parsed == codec,
          "codec tag is explicit");
    CHECK(tu_decompress_adaptive(encoded, size, dst, 128, &count) == 0, "RLE frame decode");
    CHECK(count == 128 && memcmp(src, dst, sizeof(src)) == 0, "RLE exact round-trip");
    free(encoded);
    PASS();
}

static void test_adaptive_dma_and_config(void) {
    const char *json = "{\"weight_compression\":{"
                       "\"enabled\":true,\"type\":\"adaptive_rle\",\"rle_epsilon\":0}}";
    tu_config_t runtime;
    CHECK(tu_config_load_string(json, &runtime, NULL, 0) == 0, "parse adaptive config");
    CHECK(runtime.compression_type == 2, "adaptive config enum");
    tu_compress_config_t cfg = tu_compress_config_from_tu_config(&runtime);
    CHECK(cfg.type == TU_COMPRESS_ADAPTIVE_RLE, "adaptive runtime mapping");

    fp16_t src[32] = {0}, dst[32];
    uint8_t encoded[TU_WEIGHT_FRAME_HEADER_BYTES + sizeof(src)];
    uint32_t size = 0, count = 0;
    CHECK(tu_compress_for_dma(src, 32, &cfg, encoded, sizeof(encoded), &size) == 0,
          "adaptive DMA encode");
    CHECK(tu_decompress_from_dma(encoded, size, &cfg, dst, 32, &count) == 0,
          "adaptive DMA decode");
    CHECK(count == 32 && memcmp(src, dst, sizeof(src)) == 0, "adaptive DMA round-trip");
    PASS();
}

static void test_adaptive_rejects_corruption(void) {
    fp16_t src[16] = {0}, dst[16];
    uint8_t encoded[TU_WEIGHT_FRAME_HEADER_BYTES + sizeof(src)];
    uint32_t size = 0, count = 0;
    CHECK(tu_compress_adaptive_rle(src, 16, 0.0f, encoded, sizeof(encoded),
                                   &size, NULL) == 0, "make adaptive frame");
    encoded[4] = 99; /* unsupported version */
    CHECK(!tu_compress_adaptive_validate(encoded, size), "reject unknown version");
    CHECK(tu_decompress_adaptive(encoded, size, dst, 16, &count) == -1,
          "decoder rejects unknown version");
    encoded[4] = TU_WEIGHT_FRAME_VERSION;
    encoded[5] = 99; /* unsupported payload codec */
    CHECK(!tu_compress_adaptive_validate(encoded, size), "reject unknown codec");
    CHECK(!tu_compress_adaptive_validate(encoded, size - 1), "reject truncated payload");
    PASS();
}

static void test_bitmap_round_trip_random_sparse(void) {
    fp16_t src[65] = {0}, dst[65];
    for (int i = 0; i < 65; i += 3) src[i] = tu_fp32_to_fp16((float)(i + 1));
    uint32_t cap = tu_compress_bitmap_max_size(65), size = 0, count = 0;
    uint8_t *encoded = malloc(cap);
    CHECK(tu_compress_bitmap(src, 65, encoded, cap, &size) == 0, "bitmap encode");
    CHECK(size == 8 + 9 + 22 * sizeof(fp16_t), "bitmap exact wire size");
    CHECK(tu_compress_bitmap_validate(encoded, size), "bitmap validates");
    CHECK(tu_decompress_bitmap(encoded, size, dst, 65, &count) == 0, "bitmap decode");
    CHECK(count == 65 && memcmp(src, dst, sizeof(src)) == 0, "bitmap exact round-trip");
    CHECK(!tu_compress_bitmap_validate(encoded, size - 1), "reject truncated bitmap stream");
    uint32_t bad_nnz = 21;
    memcpy(encoded + 4, &bad_nnz, sizeof(bad_nnz));
    CHECK(!tu_compress_bitmap_validate(encoded, size), "reject bitmap popcount mismatch");
    free(encoded);
    PASS();
}

static void test_bitmap_preserves_fp16_bit_patterns(void) {
    fp16_t src[] = {0x0000u, 0x8000u, 0x7e01u, 0x3c00u, 0x0000u};
    fp16_t dst[5];
    uint8_t encoded[64];
    uint32_t size = 0, count = 0;
    CHECK(tu_compress_bitmap(src, 5, encoded, sizeof(encoded), &size) == 0,
          "bitmap special encode");
    CHECK(tu_decompress_bitmap(encoded, size, dst, 5, &count) == 0,
          "bitmap special decode");
    CHECK(count == 5 && memcmp(src, dst, sizeof(src)) == 0,
          "negative zero and NaN payload preserved");
    encoded[8] |= 0x80u;
    CHECK(!tu_compress_bitmap_validate(encoded, size), "reject nonzero padding bits");
    PASS();
}

static void test_adaptive_all_selects_realistic_modes(void) {
    fp16_t random_sparse[128] = {0}, clustered[128] = {0}, dense[128], dst[128];
    for (int i = 0; i < 128; i++) {
        if (i % 3 == 0) random_sparse[i] = tu_fp32_to_fp16((float)(i + 1));
        if (i >= 96) clustered[i] = tu_fp32_to_fp16(2.0f);
        dense[i] = (fp16_t)(i + 1);
    }
    uint8_t encoded[TU_WEIGHT_FRAME_HEADER_BYTES + sizeof(dense)];
    uint32_t size = 0, count = 0;
    tu_weight_payload_codec_t codec;
    CHECK(tu_compress_adaptive(random_sparse, 128, 0.0f, encoded, sizeof(encoded),
                               &size, &codec) == 0 && codec == TU_WEIGHT_PAYLOAD_BITMAP,
          "random sparse selects bitmap");
    CHECK(tu_decompress_adaptive(encoded, size, dst, 128, &count) == 0 &&
          memcmp(random_sparse, dst, sizeof(dst)) == 0, "adaptive bitmap round-trip");
    CHECK(tu_compress_adaptive(clustered, 128, 0.0f, encoded, sizeof(encoded),
                               &size, &codec) == 0 && codec == TU_WEIGHT_PAYLOAD_RLE,
          "clustered sparse selects RLE");
    CHECK(tu_compress_adaptive(dense, 128, 0.0f, encoded, sizeof(encoded),
                               &size, &codec) == 0 && codec == TU_WEIGHT_PAYLOAD_RAW,
          "dense unique selects raw");
    PASS();
}

static void test_bitmap_dma_and_runtime_config(void) {
    tu_config_t runtime;
    CHECK(tu_config_load_string("{\"weight_compression\":{\"enabled\":true,\"type\":\"bitmap\"}}",
                                &runtime, NULL, 0) == 0, "parse bitmap config");
    tu_compress_config_t cfg = tu_compress_config_from_tu_config(&runtime);
    CHECK(cfg.type == TU_COMPRESS_BITMAP, "bitmap runtime mapping");
    fp16_t src[32] = {0}, dst[32];
    src[7] = tu_fp32_to_fp16(3.0f);
    uint8_t encoded[128];
    uint32_t size = 0, count = 0;
    CHECK(tu_compress_for_dma(src, 32, &cfg, encoded, sizeof(encoded), &size) == 0,
          "bitmap DMA encode");
    CHECK(tu_decompress_from_dma(encoded, size, &cfg, dst, 32, &count) == 0 &&
          memcmp(src, dst, sizeof(src)) == 0, "bitmap DMA round-trip");
    CHECK(tu_config_load_string("{\"weight_compression\":{\"enabled\":true,\"type\":\"adaptive\"}}",
                                &runtime, NULL, 0) == 0 && runtime.compression_type == 4,
          "parse adaptive-all config");
    PASS();
}

static void test_decoder_cycle_profiles(void) {
    fp16_t rle_src[128] = {0};
    uint8_t rle[128];
    uint32_t size = 0;
    CHECK(tu_compress_rle(rle_src, 128, 0.0f, rle, sizeof(rle), &size) == 0,
          "make RLE stream");
    tu_compress_config_t cfg = tu_compress_config_default;
    cfg.decoder_enabled = true;
    tu_compress_cycle_stats_t s;
    CHECK(tu_compress_estimate_cycles(rle, size, &cfg, 256, &s) == 0,
          "estimate serial RLE decoder");
    CHECK(s.codec == TU_WEIGHT_PAYLOAD_RLE && s.metadata_units == 1,
          "RLE stream metadata parsed");
    CHECK(s.dma_cycles == 1 && s.decode_cycles == 128 && s.total_cycles == 128 &&
          s.decoder_bound, "serial output lane dominates RLE");
    cfg.decoder_elements_per_cycle = 16;
    cfg.rle_runs_per_cycle = 4;
    CHECK(tu_compress_estimate_cycles(rle, size, &cfg, 256, &s) == 0 &&
          s.decode_cycles == 8 && s.total_cycles == 8, "wide RLE decoder");
    cfg.decoder_overlap_dma = false;
    CHECK(tu_compress_estimate_cycles(rle, size, &cfg, 256, &s) == 0 &&
          s.total_cycles == 9, "non-overlap serializes DMA and decode");
    PASS();
}

static void test_bitmap_and_adaptive_cycle_model(void) {
    fp16_t src[64] = {0};
    for (int i = 1; i < 64; i += 2) src[i] = (fp16_t)(i + 1);
    uint8_t bitmap[160], adaptive[144];
    uint32_t bitmap_size = 0, adaptive_size = 0;
    tu_weight_payload_codec_t selected;
    CHECK(tu_compress_bitmap(src, 64, bitmap, sizeof(bitmap), &bitmap_size) == 0,
          "make bitmap stream");
    CHECK(tu_compress_adaptive(src, 64, 0.0f, adaptive, sizeof(adaptive),
                               &adaptive_size, &selected) == 0 &&
          selected == TU_WEIGHT_PAYLOAD_BITMAP, "adaptive selects bitmap");
    tu_compress_config_t cfg = tu_compress_config_default;
    cfg.type = TU_COMPRESS_BITMAP;
    cfg.decoder_enabled = true;
    cfg.decoder_elements_per_cycle = 16;
    cfg.bitmap_elements_per_cycle = 8;
    tu_compress_cycle_stats_t s;
    CHECK(tu_compress_estimate_cycles(bitmap, bitmap_size, &cfg, 256, &s) == 0 &&
          s.element_count == 64 && s.metadata_units == 64,
          "bitmap metadata parsed");
    CHECK(s.dma_cycles == 3 && s.decode_cycles == 8 && s.total_cycles == 8,
          "bitmap scan width dominates");
    cfg.type = TU_COMPRESS_ADAPTIVE;
    CHECK(tu_compress_estimate_cycles(adaptive, adaptive_size, &cfg, 256, &s) == 0 &&
          s.codec == TU_WEIGHT_PAYLOAD_BITMAP && s.decode_cycles == 8,
          "framed adaptive codec is modeled");
    cfg.decoder_enabled = false;
    CHECK(tu_compress_estimate_cycles(adaptive, adaptive_size, &cfg, 256, &s) == 0 &&
          s.decode_cycles == 0 && s.total_cycles == s.dma_cycles,
          "disabled model preserves payload-only behavior");
    PASS();
}

static void test_decoder_runtime_config(void) {
    const char *json = "{\"weight_compression\":{\"enabled\":true,\"type\":\"adaptive\","
                       "\"decoder_enabled\":true,\"decoder_overlap_dma\":false,"
                       "\"decoder_elements_per_cycle\":16,\"rle_runs_per_cycle\":4,"
                       "\"bitmap_elements_per_cycle\":8}}";
    tu_config_t runtime;
    CHECK(tu_config_load_string(json, &runtime, NULL, 0) == 0,
          "parse decoder throughput config");
    tu_compress_config_t cfg = tu_compress_config_from_tu_config(&runtime);
    CHECK(cfg.decoder_enabled && !cfg.decoder_overlap_dma &&
          cfg.decoder_elements_per_cycle == 16 && cfg.rle_runs_per_cycle == 4 &&
          cfg.bitmap_elements_per_cycle == 8, "map decoder config");
    CHECK(tu_config_load_string("{\"weight_compression\":{\"decoder_elements_per_cycle\":0}}",
                                &runtime, NULL, 0) != 0, "reject zero output width");
    CHECK(tu_config_load_string("{\"weight_compression\":{\"rle_runs_per_cycle\":0}}",
                                &runtime, NULL, 0) != 0, "reject zero RLE width");
    CHECK(tu_config_load_string("{\"weight_compression\":{\"bitmap_elements_per_cycle\":0}}",
                                &runtime, NULL, 0) != 0, "reject zero bitmap width");
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
    test_stats_init();

    TEST("rle_all_same");      test_rle_all_same();
    TEST("rle_alternating");   test_rle_alternating();
    TEST("rle_mixed");         test_rle_mixed();
    TEST("rle_zeros");         test_rle_zeros();
    TEST("rle_single");        test_rle_single();
    TEST("rle_epsilon");       test_rle_epsilon();
    TEST("compression_ratio"); test_compression_ratio();
    TEST("validate_corrupt");  test_validate_corrupt();
    TEST("dma_integration");   test_dma_integration();
    TEST("disabled");           test_disabled();
    TEST("runtime_config");     test_runtime_config();
    TEST("null_safety");       test_null_safety();
    TEST("large_sparse");      test_large_sparse();
    TEST("adaptive_selects_raw"); test_adaptive_selects_raw();
    TEST("adaptive_selects_rle"); test_adaptive_selects_rle();
    TEST("adaptive_dma_config");  test_adaptive_dma_and_config();
    TEST("adaptive_corruption");  test_adaptive_rejects_corruption();
    TEST("bitmap_random_sparse"); test_bitmap_round_trip_random_sparse();
    TEST("bitmap_bit_patterns");  test_bitmap_preserves_fp16_bit_patterns();
    TEST("adaptive_all_modes");   test_adaptive_all_selects_realistic_modes();
    TEST("bitmap_dma_config");    test_bitmap_dma_and_runtime_config();
    TEST("decoder_profiles");     test_decoder_cycle_profiles();
    TEST("bitmap_adaptive_cycles"); test_bitmap_and_adaptive_cycle_model();
    TEST("decoder_runtime_config"); test_decoder_runtime_config();

    return test_exit();
}
