/*
 * TU CModel — 2:4 Structured Sparsity Implementation
 * =====================================================
 *
 * Gap: P2.1 — Structured sparsity (2:4)
 *
 * This module implements NVIDIA Ampere-style 2:4 structured sparsity:
 *   - Magnitude-based pruning (keep 2 largest per group of 4)
 *   - Packed compression format (stores only non-zeros + 4-bit mask)
 *   - Sparse MMA (skips zero-valued MACs, 2× effective throughput)
 *   - Tiled execution compatible with systolic array dimensions
 *
 * Compression format (per group of 4 elements, FP16 example):
 *   Byte 0-1:   value at position pos0 (fp16, 2 bytes)
 *   Byte 2-3:   value at position pos1 (fp16, 2 bytes)
 *   Byte 4:     mask (bits 3:0 = which of positions 0-3 are non-zero)
 *
 * Total per group: 5 bytes vs. 8 bytes dense = 62.5% memory
 * Compute: 2 MACs per group vs. 4 dense = 50% MACs → 2× speedup
 */

#include "structured_2of4.h"
#include "../infra/config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ================================================================
 * Valid 2:4 masks table (4 choose 2 = 6 combinations)
 * ================================================================ */

const tu_sparsity_2of4_mask_t TU_2OF4_VALID_MASKS[TU_2OF4_NUM_VALID_MASKS] = {
    0x3,  /* 0011 — positions 0,1 */
    0x5,  /* 0101 — positions 0,2 */
    0x6,  /* 0110 — positions 1,2 */
    0x9,  /* 1001 — positions 0,3 */
    0xA,  /* 1010 — positions 1,3 */
    0xC,  /* 1100 — positions 2,3 */
};

bool tu_sparsity_2of4_mask_is_valid(tu_sparsity_2of4_mask_t mask) {
    return tu_sparsity_2of4_mask_popcount(mask) == TU_2OF4_NONZEROS;
}

int tu_sparsity_2of4_mask_nth_bit(tu_sparsity_2of4_mask_t mask, int n) {
    int count = 0;
    for (int i = 0; i < TU_2OF4_GROUP_SIZE; i++) {
        if (mask & (1 << i)) {
            if (count == n) return i;
            count++;
        }
    }
    return -1; /* Should not happen for valid masks */
}

/* ================================================================
 * Pruning
 * ================================================================ */

/*
 * For each group of 4, select the 2 elements with largest absolute values.
 * Zeros out the other 2.
 */
size_t tu_sparsity_2of4_prune_fp32(const float *src, float *dst, size_t n) {
    if (!src || !dst || n == 0 || n % TU_2OF4_GROUP_SIZE != 0) return 0;
    size_t groups = n / TU_2OF4_GROUP_SIZE;
    size_t zeros = 0;

    for (size_t g = 0; g < groups; g++) {
        size_t base = g * TU_2OF4_GROUP_SIZE;
        float vals[TU_2OF4_GROUP_SIZE];

        memcpy(vals, &src[base], TU_2OF4_GROUP_SIZE * sizeof(float));
        /* Copy all four first, then zero the 2 smallest-magnitude */
        memcpy(&dst[base], vals, TU_2OF4_GROUP_SIZE * sizeof(float));

        /* Find the 2 smallest-magnitude positions (to zero out) */
        float abs_vals[TU_2OF4_GROUP_SIZE];
        int indices[TU_2OF4_GROUP_SIZE] = {0, 1, 2, 3};
        for (int i = 0; i < TU_2OF4_GROUP_SIZE; i++) {
            abs_vals[i] = fabsf(vals[i]);
        }

        /* Sort indices by absolute value (ascending bubble sort — 4 elements) */
        for (int i = 0; i < TU_2OF4_GROUP_SIZE - 1; i++) {
            for (int j = i + 1; j < TU_2OF4_GROUP_SIZE; j++) {
                if (abs_vals[indices[i]] > abs_vals[indices[j]]) {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }

        /* Zero out the 2 smallest-magnitude elements */
        dst[base + indices[0]] = 0.0f;
        dst[base + indices[1]] = 0.0f;
        zeros += 2;
    }

    return zeros;
}

size_t tu_sparsity_2of4_prune_with_masks_fp32(
    const float *src, float *pruned,
    tu_sparsity_2of4_mask_t *masks, size_t n)
{
    if (!src || !pruned || !masks || n == 0 ||
        n % TU_2OF4_GROUP_SIZE != 0) return 0;
    size_t groups = n / TU_2OF4_GROUP_SIZE;

    for (size_t g = 0; g < groups; g++) {
        size_t base = g * TU_2OF4_GROUP_SIZE;
        float vals[TU_2OF4_GROUP_SIZE];
        memcpy(vals, &src[base], TU_2OF4_GROUP_SIZE * sizeof(float));

        /* Copy all values, then zero the 2 smallest-magnitude */
        memcpy(&pruned[base], vals, TU_2OF4_GROUP_SIZE * sizeof(float));

        /* Find indices sorted by absolute value (ascending) */
        float abs_vals[TU_2OF4_GROUP_SIZE];
        int indices[TU_2OF4_GROUP_SIZE] = {0, 1, 2, 3};
        for (int i = 0; i < TU_2OF4_GROUP_SIZE; i++) {
            abs_vals[i] = fabsf(vals[i]);
        }
        for (int i = 0; i < TU_2OF4_GROUP_SIZE - 1; i++) {
            for (int j = i + 1; j < TU_2OF4_GROUP_SIZE; j++) {
                if (abs_vals[indices[i]] > abs_vals[indices[j]]) {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }

        /* Zero the 2 smallest-magnitude elements */
        pruned[base + indices[0]] = 0.0f;
        pruned[base + indices[1]] = 0.0f;

        /* Build mask: set bits for the 2 KEPT positions (indices 2,3) */
        masks[g] = (1u << indices[2]) | (1u << indices[3]);
    }

    return groups;
}

/* ================================================================
 * Compression / Decompression
 * ================================================================ */

size_t tu_sparsity_2of4_encode_group(
    const void *pruned,
    tu_sparsity_2of4_mask_t mask,
    size_t elem_size,
    void *packed)
{
    if (!pruned || !packed || elem_size == 0 ||
        !tu_sparsity_2of4_mask_is_valid(mask)) return 0;
    uint8_t *pkt = (uint8_t *)packed;
    const uint8_t *src = (const uint8_t *)pruned;
    int pos0 = tu_sparsity_2of4_mask_nth_bit(mask, 0);
    int pos1 = tu_sparsity_2of4_mask_nth_bit(mask, 1);

    /* Copy non-zero value at position pos0 */
    memcpy(pkt, src + pos0 * elem_size, elem_size);
    pkt += elem_size;

    /* Copy non-zero value at position pos1 */
    memcpy(pkt, src + pos1 * elem_size, elem_size);
    pkt += elem_size;

    /* Store mask byte */
    *pkt = (uint8_t)(mask & 0x0F);

    return 2 * elem_size + 1;
}

void tu_sparsity_2of4_decode_group(
    const void *packed,
    size_t elem_size,
    tu_sparsity_2of4_mask_t *mask,
    void *dense)
{
    const uint8_t *pkt = (const uint8_t *)packed;
    uint8_t *dst = (uint8_t *)dense;

    if (!packed || !dense || elem_size == 0) return;

    /* Get mask from last byte */
    tu_sparsity_2of4_mask_t m = pkt[2 * elem_size] & 0x0F;
    if (!tu_sparsity_2of4_mask_is_valid(m)) {
        memset(dst, 0, TU_2OF4_GROUP_SIZE * elem_size);
        if (mask) *mask = 0;
        return;
    }
    if (mask) *mask = m;

    /* Zero the entire 4-element dense output */
    memset(dst, 0, TU_2OF4_GROUP_SIZE * elem_size);

    /* Place non-zero values at their positions */
    int pos0 = tu_sparsity_2of4_mask_nth_bit(m, 0);
    int pos1 = tu_sparsity_2of4_mask_nth_bit(m, 1);

    memcpy(dst + pos0 * elem_size, pkt, elem_size);
    memcpy(dst + pos1 * elem_size, pkt + elem_size, elem_size);
}

size_t tu_sparsity_2of4_compress(
    const void *pruned,
    const tu_sparsity_2of4_mask_t *masks,
    size_t elem_size, size_t n,
    void *packed)
{
    if (!pruned || !masks || !packed || elem_size == 0 || n == 0 ||
        n % TU_2OF4_GROUP_SIZE != 0) return 0;
    size_t groups = n / TU_2OF4_GROUP_SIZE;
    uint8_t *pkt = (uint8_t *)packed;
    const uint8_t *src = (const uint8_t *)pruned;
    size_t group_stride = TU_2OF4_GROUP_SIZE * elem_size;
    size_t pkt_group_size = TU_2OF4_NONZEROS * elem_size + 1;

    for (size_t g = 0; g < groups; g++) {
        if (!tu_sparsity_2of4_mask_is_valid(masks[g])) return 0;
    }
    for (size_t g = 0; g < groups; g++) {
        tu_sparsity_2of4_encode_group(
            src + g * group_stride,
            masks[g],
            elem_size,
            pkt + g * pkt_group_size);
    }

    return groups * pkt_group_size;
}

void tu_sparsity_2of4_decompress(
    const void *packed,
    size_t elem_size, size_t n,
    void *dense)
{
    if (!packed || !dense || elem_size == 0 || n == 0 ||
        n % TU_2OF4_GROUP_SIZE != 0) return;
    size_t groups = n / TU_2OF4_GROUP_SIZE;
    const uint8_t *pkt = (const uint8_t *)packed;
    uint8_t *dst = (uint8_t *)dense;
    size_t group_stride = TU_2OF4_GROUP_SIZE * elem_size;
    size_t pkt_group_size = TU_2OF4_NONZEROS * elem_size + 1;

    for (size_t g = 0; g < groups; g++) {
        tu_sparsity_2of4_decode_group(
            pkt + g * pkt_group_size,
            elem_size,
            NULL,  /* mask already in packed data */
            dst + g * group_stride);
    }
}

/* ================================================================
 * Sparse MMA
 * ================================================================ */

/*
 * Helper: read a single element from dense buffer with given element size.
 * Returns FP32 value.
 */
static inline float read_elem(const void *data, size_t elem_size) {
    const uint8_t *p = (const uint8_t *)data;
    if (elem_size == 2) {
        uint16_t h;
        memcpy(&h, p, 2);
        return tu_fp16_to_fp32(h);
    } else if (elem_size == 4) {
        float f;
        memcpy(&f, p, 4);
        return f;
    } else {
        /* INT8: promote to FP32 */
        int8_t i8;
        memcpy(&i8, p, 1);
        return (float)i8;
    }
}

/*
 * Helper: read a packed weight element for a given (row, k_group, k_offset).
 * k_group = which group of 4 in K dimension
 * k_offset = which position within the group (0-3)
 */
static inline float read_weight_packed(
    const void *W_packed, size_t W_elem_size,
    size_t row, size_t K_groups, size_t k_group, size_t k_offset)
{
    /* Each row has K_groups packed groups */
    size_t pkt_group_size = TU_2OF4_NONZEROS * W_elem_size + 1; /* 2*elem_size + 1 */
    size_t row_offset = row * K_groups * pkt_group_size;
    const uint8_t *grp = (const uint8_t *)W_packed + row_offset + k_group * pkt_group_size;

    /* Read mask from last byte of the group */
    tu_sparsity_2of4_mask_t mask = grp[2 * W_elem_size] & 0x0F;

    /* Check if this k_offset is non-zero */
    if (!(mask & (1u << k_offset))) {
        return 0.0f;
    }

    /* Find which stored value corresponds to this position */
    int pos0 = tu_sparsity_2of4_mask_nth_bit(mask, 0);
    int pos1 = tu_sparsity_2of4_mask_nth_bit(mask, 1);

    if ((size_t)pos0 == k_offset) {
        return read_elem(grp, W_elem_size);
    } else if ((size_t)pos1 == k_offset) {
        return read_elem(grp + W_elem_size, W_elem_size);
    }
    return 0.0f; /* Should not reach for valid masks */
}

static bool validate_packed_masks(const void *packed,
                                  const tu_sparsity_2of4_mask_t *masks,
                                  size_t groups, size_t elem_size) {
    const uint8_t *bytes = (const uint8_t *)packed;
    size_t group_bytes = TU_2OF4_NONZEROS * elem_size + 1;
    for (size_t g = 0; g < groups; g++) {
        tu_sparsity_2of4_mask_t embedded = bytes[g * group_bytes + 2 * elem_size] & 0x0F;
        if (!tu_sparsity_2of4_mask_is_valid(masks[g]) || embedded != masks[g])
            return false;
    }
    return true;
}

uint64_t tu_sparsity_2of4_mma_fp16(
    fp32_t *O, uint32_t O_stride,
    const void *W_packed, const tu_sparsity_2of4_mask_t *W_masks,
    const void *A_dense, uint32_t A_stride,
    uint16_t M, uint16_t N, uint16_t K,
    size_t W_elem_size, size_t A_elem_size)
{
    if (!O || !W_packed || !W_masks || !A_dense || M == 0 || N == 0 ||
        K == 0 || K % TU_2OF4_GROUP_SIZE != 0 || W_elem_size == 0 ||
        A_elem_size == 0) return 0;
    size_t K_groups = (K + TU_2OF4_GROUP_SIZE - 1) / TU_2OF4_GROUP_SIZE;
    if (!validate_packed_masks(W_packed, W_masks, (size_t)M * K_groups,
                               W_elem_size)) return 0;
    size_t pkt_group_size = TU_2OF4_NONZEROS * W_elem_size + 1;
    uint64_t mac_count = 0;

    for (uint16_t m = 0; m < M; m++) {
        fp32_t *O_row = (fp32_t *)((uint8_t *)O + m * O_stride);
        const uint8_t *W_row_packed = (const uint8_t *)W_packed + m * K_groups * pkt_group_size;
        const tu_sparsity_2of4_mask_t *W_row_masks = W_masks + m * K_groups;

        for (uint16_t k_group = 0; k_group < K_groups; k_group++) {
            tu_sparsity_2of4_mask_t mask = W_row_masks[k_group];
            size_t group_k_base = k_group * TU_2OF4_GROUP_SIZE;

            /* Only 2 non-zero weights per group — iterate just those 2 */
            for (int nz = 0; nz < TU_2OF4_NONZEROS; nz++) {
                int k_off = tu_sparsity_2of4_mask_nth_bit(mask, nz);
                if (k_off < 0) continue;

                size_t k = group_k_base + k_off;
                if (k >= K) continue;

                /* Read weight value from packed storage */
                float w_val = read_elem(
                    W_row_packed + k_group * pkt_group_size + nz * W_elem_size,
                    W_elem_size);

                if (w_val == 0.0f) continue;

                /* Multiply with all N columns of A at row k */
                const void *A_row = (const uint8_t *)A_dense + k * A_stride;
                for (uint16_t n = 0; n < N; n++) {
                    float a_val = read_elem((const uint8_t *)A_row + n * A_elem_size, A_elem_size);
                    O_row[n] += w_val * a_val;
                    mac_count++;
                }
            }
        }
    }

    return mac_count;
}

uint64_t tu_sparsity_2of4_mma_tiled(
    fp32_t *O, uint32_t O_stride,
    const void *W_packed, const tu_sparsity_2of4_mask_t *W_masks,
    const void *A_dense, uint32_t A_stride,
    uint16_t M, uint16_t N, uint16_t K,
    uint16_t tile_m, uint16_t tile_n, uint16_t tile_k,
    size_t W_elem_size, size_t A_elem_size)
{
    if (!O || !W_packed || !W_masks || !A_dense || M == 0 || N == 0 ||
        K == 0 || K % TU_2OF4_GROUP_SIZE != 0 || tile_m == 0 ||
        tile_n == 0 || tile_k == 0 || W_elem_size == 0 || A_elem_size == 0)
        return 0;
    uint64_t total_macs = 0;
    size_t K_groups = (K + TU_2OF4_GROUP_SIZE - 1) / TU_2OF4_GROUP_SIZE;
    if (!validate_packed_masks(W_packed, W_masks, (size_t)M * K_groups,
                               W_elem_size)) return 0;
    size_t pkt_group_size = TU_2OF4_NONZEROS * W_elem_size + 1;

    for (uint16_t m_tile = 0; m_tile < M; m_tile += tile_m) {
        uint16_t m_count = (m_tile + tile_m <= M) ? tile_m : (M - m_tile);

        for (uint16_t n_tile = 0; n_tile < N; n_tile += tile_n) {
            uint16_t n_count = (n_tile + tile_n <= N) ? tile_n : (N - n_tile);

            for (uint16_t k_tile = 0; k_tile < K; k_tile += tile_k) {
                uint16_t k_count = (k_tile + tile_k <= K) ? tile_k : (K - k_tile);

                /* Compute this tile */
                for (uint16_t m = m_tile; m < m_tile + m_count; m++) {
                    fp32_t *O_row = (fp32_t *)((uint8_t *)O + m * O_stride) + n_tile;
                    const uint8_t *W_row_packed =
                        (const uint8_t *)W_packed + m * K_groups * pkt_group_size;
                    const tu_sparsity_2of4_mask_t *W_row_masks =
                        W_masks + m * K_groups;

                    uint16_t k_start_group = k_tile / TU_2OF4_GROUP_SIZE;
                    uint16_t k_end_group = (k_tile + k_count + TU_2OF4_GROUP_SIZE - 1) / TU_2OF4_GROUP_SIZE;

                    for (uint16_t k_group = k_start_group; k_group < k_end_group; k_group++) {
                        if (k_group >= K_groups) break;

                        tu_sparsity_2of4_mask_t mask = W_row_masks[k_group];
                        size_t group_k_base = k_group * TU_2OF4_GROUP_SIZE;

                        for (int nz = 0; nz < TU_2OF4_NONZEROS; nz++) {
                            int k_off = tu_sparsity_2of4_mask_nth_bit(mask, nz);
                            if (k_off < 0) continue;

                            size_t k = group_k_base + k_off;
                            if (k < (size_t)k_tile || k >= (size_t)(k_tile + k_count)) continue;
                            if (k >= K) continue;

                            float w_val = read_elem(
                                W_row_packed + k_group * pkt_group_size + nz * W_elem_size,
                                W_elem_size);

                            if (w_val == 0.0f) continue;

                            const void *A_row = (const uint8_t *)A_dense + k * A_stride;
                            for (uint16_t n = 0; n < n_count; n++) {
                                float a_val = read_elem(
                                    (const uint8_t *)A_row + (n_tile + n) * A_elem_size,
                                    A_elem_size);
                                O_row[n] += w_val * a_val;
                                total_macs++;
                            }
                        }
                    }
                }
            }
        }
    }

    return total_macs;
}

double tu_sparsity_2of4_speedup(uint16_t M, uint16_t N, uint16_t K,
                                 const tu_sparsity_2of4_mask_t *W_masks)
{
    if (!W_masks || M == 0 || N == 0 || K == 0 ||
        K % TU_2OF4_GROUP_SIZE != 0) return 1.0;
    size_t K_groups = (K + TU_2OF4_GROUP_SIZE - 1) / TU_2OF4_GROUP_SIZE;
    uint64_t dense_macs = (uint64_t)M * N * K;
    uint64_t sparse_macs = 0;

    for (uint16_t m = 0; m < M; m++) {
        const tu_sparsity_2of4_mask_t *row_masks = W_masks + m * K_groups;
        for (size_t kg = 0; kg < K_groups; kg++) {
            int pop = tu_sparsity_2of4_mask_popcount(row_masks[kg]);
            sparse_macs += (uint64_t)pop * N;
        }
    }

    if (sparse_macs == 0) return 1.0;
    return (double)dense_macs / (double)sparse_macs;
}

static uint64_t ceil_div_u64(uint64_t n, uint64_t d) {
    return n / d + (n % d != 0);
}

bool tu_sparsity_2of4_estimate_cycles(
    const struct tu_config_t *cfg,
    uint32_t M, uint32_t N, uint32_t K,
    tu_sparsity_2of4_cycle_stats_t *stats)
{
    if (!cfg || !stats || M == 0 || N == 0 || K == 0 ||
        K % TU_2OF4_GROUP_SIZE != 0 || cfg->pe_rows == 0 ||
        cfg->pe_cols == 0 || cfg->dma_bus_width_bits < 8 ||
        cfg->sparsity_decoder_groups_per_cycle == 0) return false;

    memset(stats, 0, sizeof(*stats));
    uint64_t pe_macs = (uint64_t)cfg->pe_rows * cfg->pe_cols;
    uint64_t fill_drain = cfg->pe_pipeline_depth == 0 ? 0 :
                          (uint64_t)cfg->pe_pipeline_depth * 2u - 1u;
    uint64_t bus_bytes = cfg->dma_bus_width_bits / 8u;
    uint64_t groups = (uint64_t)M * K / TU_2OF4_GROUP_SIZE;
    uint64_t activation_bytes = (uint64_t)K * N * sizeof(fp16_t);
    uint64_t output_bytes = (uint64_t)M * N * sizeof(fp32_t);

    stats->dense_macs = (uint64_t)M * N * K;
    stats->sparse_macs = stats->dense_macs / 2u;
    stats->dense_weight_bytes = (uint64_t)M * K * sizeof(fp16_t);
    stats->sparse_weight_bytes = groups * (2u * sizeof(fp16_t) + 1u);
    stats->dense_dma_cycles = ceil_div_u64(
        stats->dense_weight_bytes + activation_bytes + output_bytes, bus_bytes);
    stats->sparse_dma_cycles = ceil_div_u64(
        stats->sparse_weight_bytes + activation_bytes + output_bytes, bus_bytes);
    stats->dense_compute_cycles = ceil_div_u64(stats->dense_macs, pe_macs) + fill_drain;
    stats->sparse_compute_cycles = ceil_div_u64(stats->sparse_macs, pe_macs) + fill_drain;
    stats->sparse_decode_cycles = ceil_div_u64(
        groups, cfg->sparsity_decoder_groups_per_cycle);
    stats->dense_total_cycles = stats->dense_dma_cycles + stats->dense_compute_cycles;
    uint64_t sparse_core = stats->sparse_compute_cycles > stats->sparse_decode_cycles
                         ? stats->sparse_compute_cycles : stats->sparse_decode_cycles;
    stats->sparse_total_cycles = stats->sparse_dma_cycles + sparse_core;
    stats->selected_2of4 = cfg->sparsity_enabled && cfg->sparsity_2of4;
    stats->selected_total_cycles = stats->selected_2of4
                                 ? stats->sparse_total_cycles
                                 : stats->dense_total_cycles;
    return true;
}

/* ================================================================
 * Verification Helpers
 * ================================================================ */

bool tu_sparsity_2of4_verify_pattern(const float *data, size_t n, float epsilon) {
    size_t groups = n / TU_2OF4_GROUP_SIZE;

    for (size_t g = 0; g < groups; g++) {
        size_t base = g * TU_2OF4_GROUP_SIZE;
        int nz_count = 0;
        for (int i = 0; i < TU_2OF4_GROUP_SIZE; i++) {
            if (fabsf(data[base + i]) > epsilon) nz_count++;
        }
        if (nz_count != TU_2OF4_NONZEROS) return false;
    }
    return true;
}

double tu_sparsity_2of4_ratio(const float *data, size_t n, float epsilon) {
    size_t zeros = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabsf(data[i]) <= epsilon) zeros++;
    }
    return (double)zeros / (double)n;
}

double tu_sparsity_2of4_verify_against_dense(
    const fp32_t *sparse_O, const fp32_t *dense_O,
    uint16_t M, uint16_t N)
{
    double max_err = 0.0;
    for (uint16_t m = 0; m < M; m++) {
        for (uint16_t n = 0; n < N; n++) {
            size_t idx = (size_t)m * N + n;
            double err = fabs((double)sparse_O[idx] - (double)dense_O[idx]);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}
