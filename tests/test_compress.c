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
 * Test 11: NULL safety
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
    TEST("null_safety");       test_null_safety();
    TEST("large_sparse");      test_large_sparse();

    return test_exit();
}
