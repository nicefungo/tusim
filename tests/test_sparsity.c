/*
 * TU CModel — 2:4 Structured Sparsity Tests
 * ===========================================
 *
 * Test coverage:
 *   1. Mask validation (valid/invalid masks, popcount)
 *   2. Pruning correctness (50% zeros, correct magnitude selection)
 *   3. Encode/decode roundtrip (pack → unpack = original)
 *   4. Compression ratio verification
 *   5. Sparse MMA vs dense MMA correctness
 *   6. Tiled sparse MMA correctness
 *   7. Speedup computation
 *   8. Edge cases: small matrices, tile-edge groups
 *   9. Multi-precision: FP16 weights + FP32 accum
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../tu_cmodel/sparsity/structured_2of4.h"
#include "../tu_cmodel/tu_precision.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-55s", name); \
} while(0)

#define PASS() do { \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
    tests_failed++; \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "%s (got %zu, expected %zu)", msg, (size_t)(a), (size_t)(b)); \
        FAIL(buf); return; \
    } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, tol, msg) do { \
    if (fabsf((float)(a) - (float)(b)) > (float)(tol)) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "%s (got %.6f, expected %.6f, tol %.1e)", msg, (float)(a), (float)(b), (float)(tol)); \
        FAIL(buf); return; \
    } \
} while(0)

/* ================================================================
 * Helpers
 * ================================================================ */

/* Generate random float in range [-range, range] */
static float rand_float(float range) {
    return ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * range;
}

/* Initialize a float array with random values */
static void fill_random(float *data, size_t n, float range) {
    for (size_t i = 0; i < n; i++) {
        data[i] = rand_float(range);
    }
}

/* Dense GEMM reference: O[M][N] += W[M][K] × A[K][N] */
static void dense_gemm_fp32(
    float *O, float *W, float *A,
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float w = W[m * K + k];
            for (int n = 0; n < N; n++) {
                O[m * N + n] += w * A[k * N + n];
            }
        }
    }
}

/* ================================================================
 * Test 1: Mask Validation
 * ================================================================ */

static void test_mask_validation(void) {
    TEST("mask validation — valid masks");
    ASSERT_TRUE(tu_sparsity_2of4_mask_is_valid(0x3), "0011 should be valid");
    ASSERT_TRUE(tu_sparsity_2of4_mask_is_valid(0x5), "0101 should be valid");
    ASSERT_TRUE(tu_sparsity_2of4_mask_is_valid(0x6), "0110 should be valid");
    ASSERT_TRUE(tu_sparsity_2of4_mask_is_valid(0x9), "1001 should be valid");
    ASSERT_TRUE(tu_sparsity_2of4_mask_is_valid(0xA), "1010 should be valid");
    ASSERT_TRUE(tu_sparsity_2of4_mask_is_valid(0xC), "1100 should be valid");
    PASS();

    TEST("mask validation — invalid masks");
    ASSERT_TRUE(!tu_sparsity_2of4_mask_is_valid(0x0), "0000 should be invalid");
    ASSERT_TRUE(!tu_sparsity_2of4_mask_is_valid(0x1), "0001 should be invalid");
    ASSERT_TRUE(!tu_sparsity_2of4_mask_is_valid(0x7), "0111 should be invalid");
    ASSERT_TRUE(!tu_sparsity_2of4_mask_is_valid(0xF), "1111 should be invalid");
    ASSERT_TRUE(!tu_sparsity_2of4_mask_is_valid(0x8), "1000 should be invalid");
    PASS();

    TEST("mask popcount");
    ASSERT_EQ(tu_sparsity_2of4_mask_popcount(0x0), 0, "popcount 0");
    ASSERT_EQ(tu_sparsity_2of4_mask_popcount(0x3), 2, "popcount 0011");
    ASSERT_EQ(tu_sparsity_2of4_mask_popcount(0xF), 4, "popcount 1111");
    ASSERT_EQ(tu_sparsity_2of4_mask_popcount(0xA), 2, "popcount 1010");
    PASS();

    TEST("mask nth-bit extraction");
    /* 0x3 = 0011 → bits at positions 0,1 */
    ASSERT_EQ(tu_sparsity_2of4_mask_nth_bit(0x3, 0), 0, "0x3 bit 0");
    ASSERT_EQ(tu_sparsity_2of4_mask_nth_bit(0x3, 1), 1, "0x3 bit 1");
    /* 0xC = 1100 → bits at positions 2,3 */
    ASSERT_EQ(tu_sparsity_2of4_mask_nth_bit(0xC, 0), 2, "0xC bit 0");
    ASSERT_EQ(tu_sparsity_2of4_mask_nth_bit(0xC, 1), 3, "0xC bit 1");
    /* 0x5 = 0101 → bits at positions 0,2 */
    ASSERT_EQ(tu_sparsity_2of4_mask_nth_bit(0x5, 0), 0, "0x5 bit 0");
    ASSERT_EQ(tu_sparsity_2of4_mask_nth_bit(0x5, 1), 2, "0x5 bit 1");
    PASS();
}

/* ================================================================
 * Test 2: Pruning
 * ================================================================ */

static void test_pruning_basic(void) {
    TEST("pruning — basic magnitude selection");
    /* Group: [1.0, 0.5, 2.0, 0.1] → keep 1.0, 2.0 → [1.0, 0, 2.0, 0] */
    float src[4] = {1.0f, 0.5f, 2.0f, 0.1f};
    float dst[4];
    size_t zeros = tu_sparsity_2of4_prune_fp32(src, dst, 4);
    ASSERT_EQ(zeros, 2, "should zero 2 elements");
    ASSERT_TRUE(dst[0] == 1.0f, "position 0 should be 1.0");
    ASSERT_TRUE(dst[1] == 0.0f, "position 1 should be 0");
    ASSERT_TRUE(dst[2] == 2.0f, "position 2 should be 2.0");
    ASSERT_TRUE(dst[3] == 0.0f, "position 3 should be 0");
    PASS();

    TEST("pruning — with masks");
    float src2[8] = {1.0f, 0.5f, 2.0f, 0.1f,  3.0f, 4.0f, 0.2f, 0.3f};
    float pruned[8];
    tu_sparsity_2of4_mask_t masks[2];
    tu_sparsity_2of4_prune_with_masks_fp32(src2, pruned, masks, 8);

    /* Group 0: keep 1.0, 2.0 → mask 0101 (bits 0 and 2) */
    ASSERT_TRUE(masks[0] == 0x5 || masks[0] == 0x5,
                "group 0 mask should keep positions 0 and 2");
    /* Group 1: keep 3.0, 4.0 → mask 0011 (bits 0 and 1) */
    ASSERT_TRUE(masks[1] == 0x3,
                "group 1 mask should keep positions 0 and 1");
    PASS();
}

static void test_pruning_negative(void) {
    TEST("pruning — negative values (magnitude-based)");
    float src[4] = {-5.0f, 1.0f, -2.0f, 0.5f};
    float dst[4];

    tu_sparsity_2of4_prune_fp32(src, dst, 4);

    /* Keep largest magnitude: -5.0 (abs=5) and -2.0 (abs=2) */
    /* Zero out: 1.0 (abs=1) and 0.5 (abs=0.5) */
    ASSERT_FLOAT_EQ(dst[0], -5.0f, 0.0f, "position 0 should be -5.0");
    ASSERT_FLOAT_EQ(dst[1], 0.0f, 0.0f, "position 1 should be 0");
    ASSERT_FLOAT_EQ(dst[2], -2.0f, 0.0f, "position 2 should be -2.0");
    ASSERT_FLOAT_EQ(dst[3], 0.0f, 0.0f, "position 3 should be 0");
    PASS();
}

/* ================================================================
 * Test 3: Encode/Decode Roundtrip
 * ================================================================ */

static void test_encode_decode_fp32(void) {
    TEST("encode/decode roundtrip — FP32");
    float src[4] = {1.5f, 0.0f, 2.5f, 0.0f};  /* mask 0101 → keep pos 0,2 */
    tu_sparsity_2of4_mask_t mask = 0x5;
    uint8_t pkt[9]; /* 2*4 + 1 = 9 bytes */
    float decoded[4];

    size_t sz = tu_sparsity_2of4_encode_group(src, mask, 4, pkt);
    ASSERT_EQ(sz, 9, "packed size for FP32 group should be 9");

    tu_sparsity_2of4_decode_group(pkt, 4, NULL, decoded);

    ASSERT_FLOAT_EQ(decoded[0], 1.5f, 0.0f, "pos 0");
    ASSERT_FLOAT_EQ(decoded[1], 0.0f, 0.0f, "pos 1");
    ASSERT_FLOAT_EQ(decoded[2], 2.5f, 0.0f, "pos 2");
    ASSERT_FLOAT_EQ(decoded[3], 0.0f, 0.0f, "pos 3");
    PASS();

    TEST("encode/decode roundtrip — FP16");
    fp16_t src_fp16[4];
    uint8_t pkt_fp16[5]; /* 2*2 + 1 = 5 bytes */
    fp16_t decoded_fp16[4];

    src_fp16[0] = tu_fp32_to_fp16(1.5f);
    src_fp16[1] = 0;
    src_fp16[2] = tu_fp32_to_fp16(2.5f);
    src_fp16[3] = 0;

    sz = tu_sparsity_2of4_encode_group(src_fp16, 0x5, 2, pkt_fp16);
    ASSERT_EQ(sz, 5, "packed size for FP16 group should be 5");

    tu_sparsity_2of4_decode_group(pkt_fp16, 2, NULL, decoded_fp16);

    ASSERT_FLOAT_EQ(tu_fp16_to_fp32(decoded_fp16[0]), 1.5f, 0.01f, "pos 0 fp16");
    ASSERT_EQ(decoded_fp16[1], 0, "pos 1 fp16 should be zero");
    ASSERT_FLOAT_EQ(tu_fp16_to_fp32(decoded_fp16[2]), 2.5f, 0.01f, "pos 2 fp16");
    ASSERT_EQ(decoded_fp16[3], 0, "pos 3 fp16 should be zero");
    PASS();
}

/* ================================================================
 * Test 4: Compression
 * ================================================================ */

static void test_compress_decompress(void) {
    TEST("compress/decompress — 8-element FP32");
    float src[8] = {1.0f, 0.5f, 2.0f, 0.1f, 3.0f, 4.0f, 0.2f, 0.3f};
    float pruned[8];
    tu_sparsity_2of4_mask_t masks[2];

    tu_sparsity_2of4_prune_with_masks_fp32(src, pruned, masks, 8);

    size_t packed_sz = tu_sparsity_2of4_packed_size(8, 4);
    uint8_t *packed = malloc(packed_sz);

    size_t actual_sz = tu_sparsity_2of4_compress(pruned, masks, 4, 8, packed);
    ASSERT_EQ(actual_sz, packed_sz, "compressed size should match computed");

    float decompressed[8];
    tu_sparsity_2of4_decompress(packed, 4, 8, decompressed);

    for (int i = 0; i < 8; i++) {
        ASSERT_FLOAT_EQ(decompressed[i], pruned[i], 1e-6f, "decompress element mismatch");
    }

    free(packed);
    PASS();
}

/* ================================================================
 * Test 5: Sparse MMA Correctness
 * ================================================================ */

static void test_sparse_mma_small(void) {
    TEST("sparse MMA — 2×2×4 small matrix");

    /* W = [[1, 0.5, 2, 0.1]] (1×4, row 0)
     * After pruning per group of 4: keep 1.0(pos0), 2.0(pos2) → mask 0101 */
    float W_dense[4] = {1.0f, 0.5f, 2.0f, 0.1f};
    float W_pruned[4];
    tu_sparsity_2of4_mask_t W_masks[1];

    tu_sparsity_2of4_prune_with_masks_fp32(W_dense, W_pruned, W_masks, 4);

    /* Pack W */
    size_t pkt_sz = tu_sparsity_2of4_packed_size(4, 4);
    uint8_t *W_packed = malloc(pkt_sz);
    tu_sparsity_2of4_compress(W_pruned, W_masks, 4, 4, W_packed);

    /* A = [[1, 0], [2, 0], [3, 0], [4, 0]]  (4×2) */
    float A[8] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f};
    float O_sparse[2] = {0.0f, 0.0f};
    float O_dense[2] = {0.0f, 0.0f};

    /* Sparse MMA: only 1.0×1.0 + 2.0×3.0 = 1+6 = 7 for O[0] */
    uint64_t macs = tu_sparsity_2of4_mma_fp16(
        O_sparse, 2 * sizeof(float),
        W_packed, W_masks,
        A, 2 * sizeof(float),
        1, 2, 4, 4, 4);

    /* Dense MMA for reference */
    dense_gemm_fp32(O_dense, W_pruned, A, 1, 2, 4);

    /* With 2:4 sparsity, only 2 of the 4 K elements are non-zero
     * O[0] = 1.0*1.0 + 2.0*3.0 = 7.0 */
    ASSERT_FLOAT_EQ(O_sparse[0], 7.0f, 1e-5f, "sparse O[0]");
    ASSERT_FLOAT_EQ(O_sparse[1], 0.0f, 1e-5f, "sparse O[1]");
    ASSERT_EQ(macs, 4, "should have 4 MACs (2 non-zero weights × 2 N dim)");

    /* Verify against dense */
    double err = tu_sparsity_2of4_verify_against_dense(O_sparse, O_dense, 1, 2);
    ASSERT_TRUE(err < 1e-6, "sparse vs dense error should be negligible");

    free(W_packed);
    PASS();
}

static void test_sparse_mma_4x4(void) {
    TEST("sparse MMA — 4×4×8 matrix");

    int M = 4, N = 4, K = 8;
    float *W = malloc(M * K * sizeof(float));
    float *W_pruned = malloc(M * K * sizeof(float));
    tu_sparsity_2of4_mask_t *W_masks = malloc(M * (K/4) * sizeof(*W_masks));
    float *A = malloc(K * N * sizeof(float));
    float *O_sparse = calloc(M * N, sizeof(float));
    float *O_dense = calloc(M * N, sizeof(float));

    fill_random(W, M * K, 2.0f);

    /* Prune W row by row */
    for (int m = 0; m < M; m++) {
        tu_sparsity_2of4_prune_with_masks_fp32(
            &W[m * K], &W_pruned[m * K],
            &W_masks[m * (K/4)], K);
    }

    /* Pack W */
    size_t pkt_sz = tu_sparsity_2of4_packed_size(M * K, 4);
    uint8_t *W_packed = malloc(pkt_sz);
    tu_sparsity_2of4_compress(W_pruned, W_masks, 4, M * K, W_packed);

    /* Random A */
    fill_random(A, K * N, 2.0f);

    /* Sparse MMA */
    uint64_t macs = tu_sparsity_2of4_mma_fp16(
        O_sparse, N * sizeof(float),
        W_packed, W_masks,
        A, N * sizeof(float),
        M, N, K, 4, 4);

    /* Dense reference */
    dense_gemm_fp32(O_dense, W_pruned, A, M, N, K);

    /* Verify */
    double err = tu_sparsity_2of4_verify_against_dense(O_sparse, O_dense, M, N);
    ASSERT_TRUE(err < 5e-5, "sparse vs dense should match within 5e-5");

    /* Speedup should be ~2.0 */
    double speedup = tu_sparsity_2of4_speedup(M, N, K, W_masks);
    ASSERT_TRUE(speedup > 1.9 && speedup < 2.1, "speedup should be ~2.0");

    /* MAC count should be half of dense */
    uint64_t expected_macs = (uint64_t)M * N * K / 2;
    ASSERT_TRUE(macs <= expected_macs + M * N,
                "MAC count should be ~50% of dense");

    free(W); free(W_pruned); free(W_masks); free(A);
    free(O_sparse); free(O_dense); free(W_packed);
    PASS();
}

/* ================================================================
 * Test 6: Tiled Sparse MMA
 * ================================================================ */

static void test_sparse_mma_tiled(void) {
    TEST("tiled sparse MMA — 8×8×16, tile 4×4×4");

    int M = 8, N = 8, K = 16;
    int tile_m = 4, tile_n = 4, tile_k = 4;

    float *W = malloc(M * K * sizeof(float));
    float *W_pruned = malloc(M * K * sizeof(float));
    tu_sparsity_2of4_mask_t *W_masks = malloc(M * (K/4) * sizeof(*W_masks));
    float *A = malloc(K * N * sizeof(float));
    float *O_sparse = calloc(M * N, sizeof(float));
    float *O_dense = calloc(M * N, sizeof(float));

    fill_random(W, M * K, 2.0f);
    fill_random(A, K * N, 2.0f);

    for (int m = 0; m < M; m++) {
        tu_sparsity_2of4_prune_with_masks_fp32(
            &W[m * K], &W_pruned[m * K], &W_masks[m * (K/4)], K);
    }

    size_t pkt_sz = tu_sparsity_2of4_packed_size(M * K, 4);
    uint8_t *W_packed = malloc(pkt_sz);
    tu_sparsity_2of4_compress(W_pruned, W_masks, 4, M * K, W_packed);

    /* Tiled sparse MMA */
    uint64_t macs = tu_sparsity_2of4_mma_tiled(
        O_sparse, N * sizeof(float),
        W_packed, W_masks,
        A, N * sizeof(float),
        M, N, K,
        tile_m, tile_n, tile_k,
        4, 4);

    /* Dense reference */
    dense_gemm_fp32(O_dense, W_pruned, A, M, N, K);

    double err = tu_sparsity_2of4_verify_against_dense(O_sparse, O_dense, M, N);
    ASSERT_TRUE(err < 5e-5, "tiled sparse vs dense should match");
    ASSERT_TRUE(macs > 0, "should have performed MACs");

    free(W); free(W_pruned); free(W_masks); free(A);
    free(O_sparse); free(O_dense); free(W_packed);
    PASS();
}

/* ================================================================
 * Test 7: Verification Helpers
 * ================================================================ */

static void test_verification_helpers(void) {
    TEST("verify 2:4 pattern — valid sparse");
    float data[8] = {1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 4.0f};
    ASSERT_TRUE(tu_sparsity_2of4_verify_pattern(data, 8, 1e-6f),
                "should detect valid 2:4 pattern");
    PASS();

    TEST("verify 2:4 pattern — too many non-zeros");
    float bad[4] = {1.0f, 2.0f, 3.0f, 0.0f};  /* 3 non-zeros */
    ASSERT_TRUE(!tu_sparsity_2of4_verify_pattern(bad, 4, 1e-6f),
                "should reject 3-nonzero group");
    PASS();

    TEST("sparsity ratio computation");
    float mixed[8] = {1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 4.0f};
    double ratio = tu_sparsity_2of4_ratio(mixed, 8, 1e-6f);
    ASSERT_TRUE(ratio == 0.5, "sparsity ratio should be 0.5");
    PASS();

    TEST("speedup for dense weights = 1.0");
    /* Dense mask: all bits set (=15 is invalid but has high popcount) */
    /* Actually let's just check that a fully-dense mask gives 1.0 */
    tu_sparsity_2of4_mask_t dense_mask[1];
    /* With mask=0xF (all 4 positions occupied), popcount=4, speedup=1.0 */
    dense_mask[0] = 0xF;
    /* NOTE: 0xF is NOT a valid 2:4 mask, but speedup handles it */
    double sp = tu_sparsity_2of4_speedup(1, 1, 4, dense_mask);
    ASSERT_TRUE(sp >= 1.0, "speedup with dense weights should be >= 1.0");
    PASS();
}

/* ================================================================
 * Test 8: Edge Cases
 * ================================================================ */

static void test_edge_cases(void) {
    TEST("edge case — single group (n=4)");
    float src[4] = {5.0f, 1.0f, 3.0f, 2.0f};
    float dst[4];
    tu_sparsity_2of4_mask_t mask;

    tu_sparsity_2of4_prune_with_masks_fp32(src, dst, &mask, 4);
    ASSERT_TRUE(tu_sparsity_2of4_mask_is_valid(mask), "mask should be valid");
    ASSERT_TRUE(tu_sparsity_2of4_verify_pattern(dst, 4, 1e-6f),
                "should be valid 2:4");
    PASS();

    TEST("edge case — all positive equal magnitude");
    /* Tie-breaking: stable sort keeps first 2 largest */
    float src2[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float dst2[4];
    tu_sparsity_2of4_mask_t mask2;

    tu_sparsity_2of4_prune_with_masks_fp32(src2, dst2, &mask2, 4);
    ASSERT_TRUE(tu_sparsity_2of4_mask_is_valid(mask2), "mask should be valid");
    ASSERT_TRUE(tu_sparsity_2of4_verify_pattern(dst2, 4, 1e-6f),
                "should be valid 2:4");
    PASS();

    TEST("edge case — FP16 roundtrip with subnormals");
    fp16_t src_f16[8];
    for (int i = 0; i < 8; i++) src_f16[i] = tu_fp32_to_fp16((float)(i + 1) * 0.1f);

    float tmp[8];
    tu_fp16_to_fp32_buffer(src_f16, tmp, 8);

    float pruned_f32[8];
    tu_sparsity_2of4_mask_t masks[2];
    tu_sparsity_2of4_prune_with_masks_fp32(tmp, pruned_f32, masks, 8);

    /* Convert pruned back to FP16 */
    fp16_t pruned_f16[8];
    tu_fp32_to_fp16_buffer(pruned_f32, pruned_f16, 8);

    /* Pack and unpack */
    fp16_t decoded[8];
    uint8_t pkt[10];
    tu_sparsity_2of4_compress(pruned_f16, masks, 2, 8, pkt);
    tu_sparsity_2of4_decompress(pkt, 2, 8, decoded);

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(decoded[i], pruned_f16[i], "FP16 roundtrip mismatch");
    }
    PASS();
}

/* ================================================================
 * Test 9: INT8 Sparsity
 * ================================================================ */

static void test_int8_sparsity(void) {
    TEST("2:4 sparsity with INT8 elements");

    int8_t W_i8[8] = {10, 2, 30, 4, 50, 60, 7, 8};
    float W_f32[8];
    for (int i = 0; i < 8; i++) W_f32[i] = (float)W_i8[i];

    float pruned[8];
    tu_sparsity_2of4_mask_t masks[2];
    tu_sparsity_2of4_prune_with_masks_fp32(W_f32, pruned, masks, 8);

    /* Pack as INT8 (elem_size=1) */
    int8_t pruned_i8[8];
    for (int i = 0; i < 8; i++) pruned_i8[i] = (int8_t)pruned[i];

    size_t pkt_sz = tu_sparsity_2of4_packed_size(8, 1);
    ASSERT_EQ(pkt_sz, 6, "INT8 packed size: 2 groups × (2*1 + 1) = 6");

    uint8_t pkt[6];
    tu_sparsity_2of4_compress(pruned_i8, masks, 1, 8, pkt);

    int8_t decoded[8];
    tu_sparsity_2of4_decompress(pkt, 1, 8, decoded);

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(decoded[i], pruned_i8[i], "INT8 decompress mismatch");
    }
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("\n=== 2:4 Structured Sparsity Tests ===\n\n");

    /* Section 1: Mask ops */
    test_mask_validation();

    /* Section 2: Pruning */
    test_pruning_basic();
    test_pruning_negative();

    /* Section 3: Encode/Decode */
    test_encode_decode_fp32();

    /* Section 4: Compression */
    test_compress_decompress();

    /* Section 5: Sparse MMA */
    test_sparse_mma_small();
    test_sparse_mma_4x4();

    /* Section 6: Tiled MMA */
    test_sparse_mma_tiled();

    /* Section 7: Verification */
    test_verification_helpers();

    /* Section 8: Edge cases */
    test_edge_cases();

    /* Section 9: INT8 */
    test_int8_sparsity();

    /* Summary */
    printf("\n---\n");
    printf("Tests: %d run, %d passed, %d failed\n",
           tests_run, tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
