/*
 * TU CModel — 2:4 Structured Sparsity (Gap P2.1)
 * ================================================
 *
 * NVIDIA Ampere-style 2:4 structured sparsity:
 * In every contiguous group of 4 elements, exactly 2 are non-zero.
 * This guarantees 2× compute throughput and ~2× memory compression.
 *
 * Reference: "Accelerating Sparsity in the NVIDIA Ampere Architecture"
 * Reference: Mishra et al., "Accelerating Sparse Deep Neural Networks," 2021
 *
 * Architecture:
 *   1. Pruning: magnitude-based selection per group of 4
 *   2. Compression: store only non-zero values + 4-bit metadata per group
 *   3. Decompression: reconstruct dense tensor from packed format
 *   4. Sparse MMA: skip zero-valued MACs in systolic array
 *
 * Compression format (per group of 4 FP16 elements, dense = 8 bytes):
 *   Packed = 2 × 2 bytes (values) + 1 byte (metadata) = 5 bytes
 *   Ratio: 62.5% of dense for FP16, 56.25% for FP32
 *
 * Dependencies: tu_config.h, tu_precision.h
 */

#ifndef TU_SPARSITY_2OF4_H
#define TU_SPARSITY_2OF4_H

#include "../tu_config.h"
#include "../tu_precision.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 2:4 sparsity metadata: 4-bit mask indicating which of 4 positions
 * are non-zero. Only configurations with exactly 2 bits set are valid.
 *
 * Bit 0 (LSB) = position 0, Bit 1 = position 1, etc.
 * Valid masks: 0011 (3), 0101 (5), 0110 (6), 1001 (9), 1010 (10), 1100 (12)
 */
typedef uint8_t tu_sparsity_2of4_mask_t;

#define TU_2OF4_GROUP_SIZE       4
#define TU_2OF4_NONZEROS         2

/* Number of valid 2-of-4 masks (4 choose 2 = 6) */
#define TU_2OF4_NUM_VALID_MASKS  6

/* Pre-computed table of valid 2-of-4 masks */
extern const tu_sparsity_2of4_mask_t TU_2OF4_VALID_MASKS[TU_2OF4_NUM_VALID_MASKS];

/*
 * Validate a 2:4 mask has exactly 2 bits set.
 */
bool tu_sparsity_2of4_mask_is_valid(tu_sparsity_2of4_mask_t mask);

/*
 * Count set bits in a 4-bit mask.
 */
static inline int tu_sparsity_2of4_mask_popcount(tu_sparsity_2of4_mask_t mask) {
    mask &= 0x0F;
    return ((mask >> 0) & 1) + ((mask >> 1) & 1) +
           ((mask >> 2) & 1) + ((mask >> 3) & 1);
}

/*
 * Extract the index of the n-th set bit (n = 0 or 1) from a 4-bit mask.
 */
int tu_sparsity_2of4_mask_nth_bit(tu_sparsity_2of4_mask_t mask, int n);

/* ================================================================
 * Pruning
 * ================================================================ */

/*
 * Magnitude-based 2:4 pruning.
 * For each group of 4 elements, keep the 2 with largest absolute values.
 *
 * src:      dense input tensor (FP32 for magnitude comparison)
 * dst:      pruned output (same size as src, zeros inserted)
 * n:        total number of elements (must be multiple of 4)
 *
 * Returns: number of elements zeroed out (= n/2 exactly).
 */
size_t tu_sparsity_2of4_prune_fp32(const float *src, float *dst, size_t n);

/*
 * Prune and produce mask + compressed data.
 *
 * src:       dense input tensor (FP32)
 * n:         element count (multiple of 4)
 * pruned:    output — pruned dense tensor (with zeros)
 * masks:     output — 4-bit masks, one per group of 4
 *            (array of n/4 uint8_t, each entry 0-15)
 *
 * Returns: number of groups processed (= n/4).
 */
size_t tu_sparsity_2of4_prune_with_masks_fp32(
    const float *src, float *pruned,
    tu_sparsity_2of4_mask_t *masks, size_t n);

/* ================================================================
 * Compression / Decompression
 * ================================================================ */

/*
 * Compress a pruned dense tensor into 2:4 packed format.
 *
 * For FP16:
 *   dense:   n × 2 bytes = 2n bytes
 *   packed:  (n/4) groupes × (2×2 bytes values + 1 byte metadata) = 1.25n bytes
 *
 * pruned:    pruned dense tensor with exactly 2 non-zeros per group of 4
 * masks:     valid 2:4 masks (one per group)
 * elem_size: bytes per element (2=FP16/BF16, 4=FP32, 1=INT8)
 * n:         element count (multiple of 4)
 * packed:    output packed buffer (caller-allocated, size = n/4 * (2*elem_size + 1))
 *
 * Returns: size of packed buffer in bytes.
 */
size_t tu_sparsity_2of4_compress(
    const void *pruned,
    const tu_sparsity_2of4_mask_t *masks,
    size_t elem_size, size_t n,
    void *packed);

/*
 * Decompress 2:4 packed format to dense.
 *
 * packed:    packed buffer
 * elem_size: bytes per element
 * n:         total elements to decompress (multiple of 4)
 * dense:     output dense buffer (caller-allocated, size = n * elem_size)
 */
void tu_sparsity_2of4_decompress(
    const void *packed,
    size_t elem_size, size_t n,
    void *dense);

/*
 * Packed size in bytes for given element count and element size.
 */
static inline size_t tu_sparsity_2of4_packed_size(size_t n, size_t elem_size) {
    size_t groups = n / TU_2OF4_GROUP_SIZE;
    return groups * (TU_2OF4_NONZEROS * elem_size + 1); /* 2 values + 1 byte metadata */
}

/* ================================================================
 * Metadata Encoding/Decoding
 * ================================================================ */

/*
 * Encode a group of 4 values into packed format at a specific offset.
 *
 * pruned:    pointer to the 4-element group start in the pruned buffer
 * mask:      4-bit valid 2:4 mask
 * elem_size: bytes per element
 * packed:    output buffer (must have room for 2*elem_size + 1 bytes)
 *
 * Returns: number of bytes written.
 */
size_t tu_sparsity_2of4_encode_group(
    const void *pruned,
    tu_sparsity_2of4_mask_t mask,
    size_t elem_size,
    void *packed);

/*
 * Decode a packed group into dense 4-element output.
 *
 * packed:    pointer to packed group data
 * elem_size: bytes per element
 * mask:      output — decoded 4-bit mask
 * dense:     output — 4 decoded elements
 */
void tu_sparsity_2of4_decode_group(
    const void *packed,
    size_t elem_size,
    tu_sparsity_2of4_mask_t *mask,
    void *dense);

/* ================================================================
 * Sparse MMA
 * ================================================================ */

/*
 * 2:4 Sparse GEMM: O[M][N] += W[M][K] × A[K][N]
 *
 * W is stored in 2:4 packed format — every group of 4 K-dimension elements
 * has exactly 2 non-zeros. The MMA skips the zero-valued MACs, achieving
 * approximately 2× effective throughput.
 *
 * Parameters:
 *   O:          output matrix [M][N], FP32, pre-allocated
 *   W_packed:   weight matrix in 2:4 packed format
 *   W_masks:    per-group 4-bit masks (n_groups = ceil(K/4) per row)
 *   A_dense:    activation matrix [K][N], dense, elem_size bytes per element
 *   M, N, K:    matrix dimensions
 *   W_elem_size: bytes per weight element (2=FP16, 1=INT8)
 *   A_elem_size: bytes per activation element
 *   O_stride:   row stride of O in bytes (N * sizeof(fp32_t))
 *   A_stride:   row stride of A in bytes
 *
 * Returns: number of MAC operations performed (K * M * N / 2).
 */
uint64_t tu_sparsity_2of4_mma_fp16(
    fp32_t *O, uint32_t O_stride,
    const void *W_packed, const tu_sparsity_2of4_mask_t *W_masks,
    const void *A_dense, uint32_t A_stride,
    uint16_t M, uint16_t N, uint16_t K,
    size_t W_elem_size, size_t A_elem_size);

/*
 * Tiled sparse MMA — decomposes a large M×N×K operation into tile-sized
 * chunks compatible with the systolic PE array dimensions.
 *
 * Identical semantics to tu_sparsity_2of4_mma_fp16() but adds tiling.
 *
 * tile_m, tile_n, tile_k: tile dimensions (e.g., PE array size)
 *
 * Returns: total MAC operations (counting only non-zero multiplies).
 */
uint64_t tu_sparsity_2of4_mma_tiled(
    fp32_t *O, uint32_t O_stride,
    const void *W_packed, const tu_sparsity_2of4_mask_t *W_masks,
    const void *A_dense, uint32_t A_stride,
    uint16_t M, uint16_t N, uint16_t K,
    uint16_t tile_m, uint16_t tile_n, uint16_t tile_k,
    size_t W_elem_size, size_t A_elem_size);

/*
 * Compute speedup factor: dense MACs / sparse MACs.
 * For valid 2:4 sparse weight matrix, returns 2.0.
 * For dense (invalid masks), returns 1.0.
 */
double tu_sparsity_2of4_speedup(uint16_t M, uint16_t N, uint16_t K,
                                 const tu_sparsity_2of4_mask_t *W_masks);

/* ================================================================
 * Verification Helpers
 * ================================================================ */

/*
 * Verify that a tensor follows 2:4 pattern:
 * every group of 4 has exactly 2 non-zeros.
 *
 * Returns: true if valid 2:4 sparse throughout.
 */
bool tu_sparsity_2of4_verify_pattern(const float *data, size_t n, float epsilon);

/*
 * Compute sparsity ratio (fraction of zero elements).
 */
double tu_sparsity_2of4_ratio(const float *data, size_t n, float epsilon);

/*
 * Compare sparse MMA result against dense reference.
 *
 * Returns: maximum absolute error.
 */
double tu_sparsity_2of4_verify_against_dense(
    const fp32_t *sparse_O, const fp32_t *dense_O,
    uint16_t M, uint16_t N);

#ifdef __cplusplus
}
#endif

#endif /* TU_SPARSITY_2OF4_H */
