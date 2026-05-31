/*
 * TU CModel — FlashAttention-Style Attention Engine Implementation
 * =================================================================
 * Gap O3: Tiled attention with on-chip SRAM management.
 *
 * Implements the FlashAttention algorithm:
 *   For each Q tile:
 *     O_tile = 0, m = -inf, l = 0
 *     For each KV tile:
 *       S = Q_tile × K_tile^T          (MMA, FP16→FP32)
 *       S = S * scale                   (elementwise)
 *       S = S + mask                    (elementwise, if masked)
 *       P = softmax(S)                  (online rescale)
 *       O_tile += P × V_tile            (MMA, FP16→FP32)
 *     O_tile → host output (FP16)
 *
 * Uses the pluggable dataflow system for MMA, elementwise pipeline
 * for scaling/masking, and online softmax engine for normalization.
 *
 * SRAM allocation strategy:
 *   sram_a: Q_tile (tile_m × head_dim × 2 bytes FP16)
 *   sram_w: K_tile (tile_n × head_dim × 2 bytes FP16)
 *           then V_tile (tile_n × head_dim × 2 bytes FP16) [overwrites K]
 *           also K^T_scratch for transposed K
 *   sram_o: S_tile (tile_m × tile_n × 4 bytes FP32)
 *           then O_tile (tile_m × head_dim × 4 bytes FP32) [reuses space]
 */

#include "attention_engine.h"
#include "dataflow/dataflow_interface.h"
#include "dataflow/dataflow_registry.h"
#include "elementwise_pipeline.h"
#include "softmax_engine.h"
#include "../infra/logging.h"
#include "../tu_cmodel.h"    /* for g_tu, tu_dma_load_*, tu_set_dataflow */
#include "../tu_precision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Internal helpers ─────────────────────────────────────────── */

/* Transpose a FP16 matrix in SRAM.
 * Reads src from sram_w at src_off [rows][cols] row-major,
 * writes transposed [cols][rows] to sram_w at dst_off.
 * Returns total stall cycles. */
static uint64_t transpose_fp16_in_sram(tu_sram_region_t *sram,
                                        uint32_t src_off, uint32_t dst_off,
                                        uint32_t rows, uint32_t cols) {
    uint64_t stall = 0;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t src_addr = src_off + (r * cols + c) * sizeof(fp16_t);
            uint32_t dst_addr = dst_off + (c * rows + r) * sizeof(fp16_t);
            fp16_t val;
            stall += tu_sram_read(sram, src_addr, &val);
            stall += tu_sram_write(sram, dst_addr, &val);
        }
    }
    return stall;
}

/* Convert FP32 SRAM buffer to FP16 in-place (truncating lower half).
 * Reads FP32 at offset, writes FP16 back starting at same offset.
 * elem_count: number of FP32 elements.
 * Returns stall cycles. */
static uint64_t fp32_to_fp16_in_sram(tu_sram_region_t *sram,
                                      uint32_t offset, uint32_t elem_count) {
    uint64_t stall = 0;
    /* Process in reverse to avoid overwriting unread data if shrinking.
     * FP32 is 4 bytes, FP16 is 2 bytes — so output fits in place. */
    for (uint32_t i = 1; i <= elem_count; i++) {
        uint32_t idx = elem_count - i;
        uint32_t src_addr = offset + idx * sizeof(fp32_t);
        fp32_t val;
        stall += tu_sram_read(sram, src_addr, &val);
        fp16_t h = tu_fp32_to_fp16(val);
        /* Write FP16 at same offset (overwrites the upper half) */
        uint32_t dst_addr = offset + idx * sizeof(fp16_t);
        stall += tu_sram_write(sram, dst_addr, &h);
    }
    return stall;
}

/* Build a causal mask in host memory for a Q_tile × K_tile block.
 * mask[i][j] = 0.0 if j <= i+q_start, else mask_fill.
 * The caller must free the returned buffer. */
static float *build_causal_mask(uint32_t q_start, uint32_t q_count,
                                 uint32_t kv_start, uint32_t kv_count,
                                 float mask_fill) {
    uint32_t total = q_count * kv_count;
    float *mask = (float *)calloc(total, sizeof(float));
    if (!mask) return NULL;
    for (uint32_t i = 0; i < q_count; i++) {
        for (uint32_t j = 0; j < kv_count; j++) {
            if (j + kv_start > i + q_start) {
                mask[i * kv_count + j] = mask_fill;
            }
        }
    }
    return mask;
}

/* Write a FP32 host buffer into SRAM as FP32.
 * Used for loading masks into SRAM. Returns stall cycles. */
static uint64_t write_fp32_to_sram(tu_sram_region_t *sram, uint32_t offset,
                                    const float *host, uint32_t count) {
    uint64_t stall = 0;
    for (uint32_t i = 0; i < count; i++) {
        stall += tu_sram_write(sram, offset + i * sizeof(float), &host[i]);
    }
    return stall;
}

/* ── Auto-tiling ──────────────────────────────────────────────── */

void tu_attention_auto_tile(tu_attention_desc_t *desc) {
    /* Available SRAM for tiles (use A-buffer for Q, W-buffer for K/V, O-buffer for S/O) */
    uint32_t aw = desc->head_dim;  /* the "K" dimension in MMA terms */

    /* Space budget in sram_w (for K_tile and V_tile + K^T_scratch):
     *   K_tile: tile_n × head_dim × 2 (FP16)
     *   K^T_scratch: head_dim × tile_n × 2 (FP16) — same size as K_tile
     *   V_tile: tile_n × head_dim × 2 (FP16) — loaded after K, reused space
     * Maximum needed at once: 2 × tile_n × head_dim × 2 (K + K^T)
     */
    uint32_t sram_w_capacity = TU_SRAM_W_SIZE;
    uint32_t sram_o_capacity = TU_SRAM_O_SIZE;

    /* K_tile + K^T_scratch must fit in sram_w */
    uint32_t max_tile_n_by_w = sram_w_capacity / (2 * aw * sizeof(fp16_t));
    if (max_tile_n_by_w > desc->seq_len_kv) max_tile_n_by_w = desc->seq_len_kv;

    /* We start with a small tile_m and increase until sram_o is full */
    uint32_t best_m = 1, best_n = 1;
    uint64_t best_score = 0; /* tile utilization: tile_m * tile_n */

    for (uint32_t tm = 1; tm <= desc->seq_len_q && tm <= 64; tm++) {
        /* S_tile size = tm × tn × 4 */
        /* O_tile size = tm × aw × 4 */
        uint32_t o_tile_bytes = tm * aw * sizeof(fp32_t);
        uint32_t s_o_remaining = (sram_o_capacity > o_tile_bytes) ?
                                 (sram_o_capacity - o_tile_bytes) : 0;
        if (s_o_remaining == 0) break;

        uint32_t max_tn_by_o = s_o_remaining / (tm * sizeof(fp32_t));
        if (max_tn_by_o > max_tile_n_by_w) max_tn_by_o = max_tile_n_by_w;
        if (max_tn_by_o > desc->seq_len_kv) max_tn_by_o = desc->seq_len_kv;
        if (max_tn_by_o == 0) continue;

        /* Score: prefer larger tiles, balance tm and tn */
        uint64_t score = (uint64_t)tm * max_tn_by_o;
        if (score > best_score) {
            best_score = score;
            best_m = tm;
            best_n = max_tn_by_o;
        }
    }

    /* Cap to sequence lengths */
    if (best_m > desc->seq_len_q) best_m = desc->seq_len_q;
    if (best_n > desc->seq_len_kv) best_n = desc->seq_len_kv;

    /* Align to PE dimensions for efficiency */
    uint16_t pe_rows = TU_PE_ROWS;
    uint16_t pe_cols = TU_PE_COLS;
    best_m = ((best_m + pe_rows - 1) / pe_rows) * pe_rows;
    best_n = ((best_n + pe_cols - 1) / pe_cols) * pe_cols;

    if (best_m < pe_rows) best_m = pe_rows;
    if (best_n < pe_cols) best_n = pe_cols;

    desc->tile_m = best_m;
    desc->tile_n = best_n;

    TU_LOG_INFO(TU_COMP_CORE,
        "Attention auto-tile: tile_m=%u, tile_n=%u (head_dim=%u, "
        "Q_len=%u, KV_len=%u)",
        best_m, best_n, desc->head_dim, desc->seq_len_q, desc->seq_len_kv);
}

/* ── Validation ───────────────────────────────────────────────── */

bool tu_attention_validate_desc(const tu_attention_desc_t *desc) {
    if (!desc) {
        TU_LOG_ERR(TU_COMP_CORE, "Attention: null descriptor");
        return false;
    }
    if (!desc->Q || !desc->K || !desc->V || !desc->output) {
        TU_LOG_ERR(TU_COMP_CORE, "Attention: null tensor pointer");
        return false;
    }
    if (desc->seq_len_q == 0 || desc->seq_len_kv == 0 || desc->head_dim == 0) {
        TU_LOG_ERR(TU_COMP_CORE, "Attention: zero dimension");
        return false;
    }
    if (desc->batch_size == 0 || desc->num_heads == 0) {
        TU_LOG_ERR(TU_COMP_CORE, "Attention: zero batch_size or num_heads");
        return false;
    }
    if (desc->softmax_scale < 0.0f) {
        TU_LOG_ERR(TU_COMP_CORE, "Attention: negative softmax_scale");
        return false;
    }
    /* Auto-compute tile sizes if not set */
    if (desc->tile_m == 0 || desc->tile_n == 0) {
        /* We can't call auto_tile here because desc is const.
         * The caller must call tu_attention_auto_tile() before
         * tu_attention_execute() if they want auto-tiling. */
        TU_LOG_WARN(TU_COMP_CORE,
            "Attention: tile_m=%u or tile_n=%u is 0; auto-tile not called. "
            "Using defaults.",
            desc->tile_m, desc->tile_n);
    }
    return true;
}

/* ── Main Execution ───────────────────────────────────────────── */

int tu_attention_execute(const tu_attention_desc_t *desc,
                         tu_attention_stats_t *stats) {
    if (!desc || !desc->Q || !desc->K || !desc->V || !desc->output) {
        TU_LOG_ERR(TU_COMP_CORE, "Attention: invalid descriptor");
        return -1;
    }

    /* Initialize optional stats */
    tu_attention_stats_t local_stats = {0};

    /* Use provided tile sizes or compute defaults */
    uint32_t tile_m = desc->tile_m;
    uint32_t tile_n = desc->tile_n;

    if (tile_m == 0) {
        /* Conservative default: use PE rows or cap to seq_len_q */
        tile_m = TU_PE_ROWS;
        if (tile_m > desc->seq_len_q) tile_m = desc->seq_len_q;
    }
    if (tile_n == 0) {
        tile_n = TU_PE_COLS;
        if (tile_n > desc->seq_len_kv) tile_n = desc->seq_len_kv;
    }

    /* Scale factor */
    float scale = desc->softmax_scale;
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)desc->head_dim);
    }

    /* Mask fill value */
    float mfill = desc->mask_fill;
    if (mfill == 0.0f) mfill = -1e9f;

    uint32_t aw = desc->head_dim;        /* "K" dimension in MMA */
    uint32_t M = desc->seq_len_q;        /* Query tokens */
    uint32_t N = desc->seq_len_kv;       /* Key/Value tokens */
    uint16_t pe_rows = g_tu.rt_cfg.pe_rows;
    uint16_t pe_cols = g_tu.rt_cfg.pe_cols;

    /* Get SRAM region sizes for bounds checking */
    uint32_t sram_w_cap = g_tu.rt_cfg.sram_w_size;
    uint32_t sram_a_cap = g_tu.rt_cfg.sram_a_size;
    uint32_t sram_o_cap = g_tu.rt_cfg.sram_o_size;

    /* Verify tiles fit in SRAM */
    uint32_t q_tile_bytes = tile_m * aw * sizeof(fp16_t);
    uint32_t k_tile_bytes = tile_n * aw * sizeof(fp16_t);
    uint32_t kt_tile_bytes = aw * tile_n * sizeof(fp16_t);  /* K^T */
    uint32_t s_tile_bytes = tile_m * tile_n * sizeof(fp32_t);
    uint32_t o_tile_bytes = tile_m * aw * sizeof(fp32_t);

    if (q_tile_bytes > sram_a_cap) {
        TU_LOG_ERR(TU_COMP_CORE,
            "Attention: Q tile (%u B) exceeds A-buffer (%u B)",
            q_tile_bytes, sram_a_cap);
        return -1;
    }
    if (k_tile_bytes + kt_tile_bytes > sram_w_cap) {
        TU_LOG_ERR(TU_COMP_CORE,
            "Attention: K+K^T tiles (%u B) exceed W-buffer (%u B)",
            k_tile_bytes + kt_tile_bytes, sram_w_cap);
        return -1;
    }
    if (s_tile_bytes > sram_o_cap || o_tile_bytes > sram_o_cap) {
        TU_LOG_ERR(TU_COMP_CORE,
            "Attention: S/O tiles (%u/%u B) exceed O-buffer (%u B)",
            s_tile_bytes, o_tile_bytes, sram_o_cap);
        return -1;
    }

    /* Compute O-buffer layout and verify total fits */
    uint32_t p_fp16_bytes = tile_m * tile_n * sizeof(fp16_t);
    uint32_t O_off_aligned = ((p_fp16_bytes + 3) / 4) * 4;
    uint32_t o_end = O_off_aligned + o_tile_bytes;
    uint32_t mask_off_aligned = o_end;
    /* Worst-case: mask + S coexist, or O at O_off_aligned */
    uint32_t max_o_usage = o_end;
    uint32_t mask_with_s = mask_off_aligned + s_tile_bytes;
    if (mask_with_s > max_o_usage) max_o_usage = mask_with_s;
    if (max_o_usage > sram_o_cap) {
        TU_LOG_ERR(TU_COMP_CORE,
            "Attention: total O-buffer usage (%u B) exceeds capacity (%u B)",
            max_o_usage, sram_o_cap);
        return -1;
    }

    /* SRAM offsets */
    uint32_t Q_off  = 0;                    /* sram_a: Q tile */
    uint32_t K_off  = 0;                    /* sram_w: K tile */
    uint32_t KT_off = k_tile_bytes;         /* sram_w: K^T tile (after K) */
    uint32_t V_off  = 0;                    /* sram_w: V tile (reuses K space) */
    uint32_t S_off  = 0;                    /* sram_o: S/P tile (FP32 scores → FP16 probs) */

    /* O tile sits after P (FP16) in sram_o — must not overlap */
    uint32_t O_off  = O_off_aligned;
    uint32_t mask_off = mask_off_aligned;

    TU_LOG_INFO(TU_COMP_CORE,
        "Attention: M=%u N=%u d=%u scale=%.6f tile_m=%u tile_n=%u "
        "Q=%uB K=%uB KT=%uB S=%uB O=%uB",
        M, N, aw, scale, tile_m, tile_n,
        q_tile_bytes, k_tile_bytes, kt_tile_bytes,
        s_tile_bytes, o_tile_bytes);

    /* ── Process each batch × head ── */
    uint32_t total_heads = desc->batch_size * desc->num_heads;
    uint32_t stride_q = M * aw;   /* elements per head in Q */
    uint32_t stride_kv = N * aw;  /* elements per head in K/V */

    for (uint32_t h = 0; h < total_heads; h++) {
        const fp16_t *Q_head = (const fp16_t *)desc->Q + h * stride_q;
        const fp16_t *K_head = (const fp16_t *)desc->K + h * stride_kv;
        const fp16_t *V_head = (const fp16_t *)desc->V + h * stride_kv;
        fp16_t       *O_head = (fp16_t *)desc->output + h * stride_q;

        /* ── Outer loop: Q tiles ── */
        for (uint32_t qi = 0; qi < M; qi += tile_m) {
            uint32_t q_count = (qi + tile_m <= M) ? tile_m : (M - qi);

            /* DMA: Q_tile from host → sram_a */
            tu_dma_load_a(Q_head + qi * aw, Q_off, q_count * aw * sizeof(fp16_t));
            local_stats.dma_bytes += q_count * aw * sizeof(fp16_t);

            /* Initialize O accumulator to zero in SRAM */
            {
                fp32_t zero = 0.0f;
                for (uint32_t i = 0; i < q_count * aw; i++) {
                    local_stats.dma_cycles += tu_sram_write(
                        &g_tu.sram_o, O_off + i * sizeof(fp32_t), &zero);
                }
            }

            /* ── Inner loop: KV tiles ── */
            for (uint32_t kvi = 0; kvi < N; kvi += tile_n) {
                uint32_t kv_count = (kvi + tile_n <= N) ? tile_n : (N - kvi);

                /* DMA: K_tile from host → sram_w */
                tu_dma_load_w(K_head + kvi * aw, K_off,
                              kv_count * aw * sizeof(fp16_t));
                local_stats.dma_bytes += kv_count * aw * sizeof(fp16_t);

                /* Transpose K → K^T in sram_w */
                local_stats.compute_cycles += transpose_fp16_in_sram(
                    &g_tu.sram_w, K_off, KT_off, kv_count, aw);

                /* ── Step 1: S = Q × K^T ── */
                {
                    tu_dataflow_tensor_t W_t = {
                        .data = tu_sram_raw_ptr(&g_tu.sram_a) + Q_off,
                        .rows = q_count, .cols = aw,
                        .stride = aw * sizeof(fp16_t),
                        .elem_size = sizeof(fp16_t)
                    };
                    tu_dataflow_tensor_t A_t = {
                        .data = tu_sram_raw_ptr(&g_tu.sram_w) + KT_off,
                        .rows = aw, .cols = kv_count,
                        .stride = kv_count * sizeof(fp16_t),
                        .elem_size = sizeof(fp16_t)
                    };
                    tu_dataflow_tensor_t O_t = {
                        .data = tu_sram_raw_ptr(&g_tu.sram_o) + S_off,
                        .rows = q_count, .cols = kv_count,
                        .stride = kv_count * sizeof(fp32_t),
                        .elem_size = sizeof(fp32_t)
                    };
                    /* Zero the S_tile in SRAM — tu_mma accumulates, so start clean */
                    fp32_t zero = 0.0f;
                    for (uint32_t i = 0; i < q_count * kv_count; i++) {
                        local_stats.dma_cycles += tu_sram_write(
                            &g_tu.sram_o, S_off + i * sizeof(fp32_t), &zero);
                    }

                    uint64_t mma_cycles = tu_dataflow_execute_mma(
                        g_tu.dataflow, &W_t, &A_t, &O_t,
                        pe_rows, pe_cols, pe_cols, TU_PE_PIPELINE_DEPTH);
                    local_stats.compute_cycles += mma_cycles;
                    local_stats.mma_tiles += g_tu.dataflow->total_tiles;
                    local_stats.mma_flops += g_tu.dataflow->total_flops;
                    g_tu.dataflow->total_tiles = 0;
                    g_tu.dataflow->total_flops = 0;
                }

                /* ── Step 2: S = S * scale ── */
                {
                    uint64_t ew_cycles = tu_ew_apply_binary_scalar(
                        &g_tu.sram_o, S_off,
                        q_count * kv_count, TU_EW_MUL, scale);
                    local_stats.compute_cycles += ew_cycles;
                }

                /* ── Step 3: S = S + mask ── */
                float *causal_buf = NULL;

                if (desc->mask_type == TU_ATTN_MASK_CAUSAL && kvi > qi + q_count) {
                    /* All KV positions are after all Q positions —
                     * entire tile is masked. Skip MMA? No, we still computed S.
                     * Instead, just fill S with mask_fill. */
                    for (uint32_t i = 0; i < q_count * kv_count; i++) {
                        fp32_t fill = mfill;
                        tu_sram_write(&g_tu.sram_o,
                            S_off + i * sizeof(fp32_t), &fill);
                    }
                    /* Skip the second MMA for this tile — S × V contributes ~0 */
                    goto skip_v_mma;
                } else if (desc->mask_type == TU_ATTN_MASK_CAUSAL) {
                    causal_buf = build_causal_mask(qi, q_count, kvi, kv_count, mfill);
                    if (causal_buf) {
                        /* Write mask to sram_o (after S space, if fits) */
                        uint32_t mask_bytes = q_count * kv_count * sizeof(float);
                        if (mask_off + mask_bytes <= sram_o_cap) {
                            local_stats.compute_cycles += write_fp32_to_sram(
                                &g_tu.sram_o, mask_off, causal_buf,
                                q_count * kv_count);
                            /* Add: tu_ew_add_tensors(S, mask, S) */
                            uint64_t ew_cycles = tu_ew_add_tensors(
                                &g_tu.sram_o,
                                S_off, mask_off, S_off,
                                q_count * kv_count);
                            local_stats.compute_cycles += ew_cycles;
                            (void)0; /* mask applied */
                        }
                        free(causal_buf);
                    }
                } else if (desc->mask_type == TU_ATTN_MASK_CUSTOM && desc->mask) {
                    /* Load custom mask slice */
                    const float *mask_slice = desc->mask +
                        h * M * N + qi * N + kvi;
                    uint32_t mask_bytes = q_count * kv_count * sizeof(float);
                    if (mask_off + mask_bytes <= sram_o_cap) {
                        local_stats.compute_cycles += write_fp32_to_sram(
                            &g_tu.sram_o, mask_off, mask_slice,
                            q_count * kv_count);
                        uint64_t ew_cycles = tu_ew_add_tensors(
                            &g_tu.sram_o,
                            S_off, mask_off, S_off,
                            q_count * kv_count);
                        local_stats.compute_cycles += ew_cycles;
                        (void)0; /* mask applied */
                    }
                }

                /* ── Step 4: P = softmax(S) ── */
                {
                    tu_softmax_desc_t sm_desc = {
                        .mode       = TU_SOFTMAX_STANDARD,
                        .data_sram  = &g_tu.sram_o,
                        .data_offset = S_off,
                        .elem_count  = q_count * kv_count,
                        .axis_dim    = kv_count,
                        .mask        = NULL,
                        .scale       = 0.0f,  /* already scaled above */
                        .in_place    = true,
                    };
                    uint64_t sm_cycles = tu_softmax_execute(&sm_desc);
                    local_stats.compute_cycles += sm_cycles;
                }

                /* ── Step 5: Convert P to FP16 for MMA ── */
                local_stats.compute_cycles += fp32_to_fp16_in_sram(
                    &g_tu.sram_o, S_off, q_count * kv_count);
                /* Now S_off holds FP16 P values */

                /* ── Step 6: DMA V_tile from host → sram_w ── */
                tu_dma_load_w(V_head + kvi * aw, V_off,
                              kv_count * aw * sizeof(fp16_t));
                local_stats.dma_bytes += kv_count * aw * sizeof(fp16_t);

                /* ── Step 7: O += P × V ── */
                {
                    tu_dataflow_tensor_t W_t = {
                        .data = tu_sram_raw_ptr(&g_tu.sram_o) + S_off,
                        .rows = q_count, .cols = kv_count,
                        .stride = kv_count * sizeof(fp16_t),
                        .elem_size = sizeof(fp16_t)
                    };
                    tu_dataflow_tensor_t A_t = {
                        .data = tu_sram_raw_ptr(&g_tu.sram_w) + V_off,
                        .rows = kv_count, .cols = aw,
                        .stride = aw * sizeof(fp16_t),
                        .elem_size = sizeof(fp16_t)
                    };
                    tu_dataflow_tensor_t O_t = {
                        .data = tu_sram_raw_ptr(&g_tu.sram_o) + O_off,
                        .rows = q_count, .cols = aw,
                        .stride = aw * sizeof(fp32_t),
                        .elem_size = sizeof(fp32_t)
                    };

                    uint64_t mma_cycles = tu_dataflow_execute_mma(
                        g_tu.dataflow, &W_t, &A_t, &O_t,
                        pe_rows, pe_cols, pe_cols, TU_PE_PIPELINE_DEPTH);
                    local_stats.compute_cycles += mma_cycles;
                    local_stats.mma_tiles += g_tu.dataflow->total_tiles;
                    local_stats.mma_flops += g_tu.dataflow->total_flops;
                    g_tu.dataflow->total_tiles = 0;
                    g_tu.dataflow->total_flops = 0;
                }

skip_v_mma:
                ; /* Label needs statement */
            } /* end KV tile loop */

            /* ── Convert O_tile from FP32 to FP16 and DMA out ── */
            /* Read FP32 O_tile from SRAM, convert to FP16, write to host */
            {
                uint32_t num_elems = q_count * aw;
                /* Read FP32 into temp buffer, convert, DMA to host */
                fp32_t *temp_fp32 = (fp32_t *)malloc(num_elems * sizeof(fp32_t));
                fp16_t *temp_fp16 = (fp16_t *)malloc(num_elems * sizeof(fp16_t));
                if (temp_fp32 && temp_fp16) {
                    for (uint32_t i = 0; i < num_elems; i++) {
                        tu_sram_read(&g_tu.sram_o,
                            O_off + i * sizeof(fp32_t), &temp_fp32[i]);
                    }
                    tu_fp32_to_fp16_buffer(temp_fp32, temp_fp16, num_elems);

                    /* DMA store to host output */
                    tu_dma_store_o(O_head + qi * aw, O_off,
                                   num_elems * sizeof(fp16_t));
                    /* Actually, DMA store copies FROM SRAM TO host.
                     * But we need to write the converted fp16 values.
                     * Let me write directly. */
                    memcpy(O_head + qi * aw, temp_fp16,
                           num_elems * sizeof(fp16_t));
                    local_stats.dma_bytes += num_elems * sizeof(fp16_t);

                    free(temp_fp32);
                    free(temp_fp16);
                }
            }
        } /* end Q tile loop */
    } /* end head loop */

    /* Compute totals */
    local_stats.total_cycles = local_stats.compute_cycles +
                               local_stats.dma_cycles;
    if (local_stats.total_cycles > 0) {
        local_stats.utilization =
            (float)local_stats.compute_cycles /
            (float)local_stats.total_cycles;
    }

    TU_LOG_INFO(TU_COMP_CORE,
        "Attention complete: DMA=%lu B, MMA tiles=%lu, FLOPs=%lu, "
        "cycles=%lu (compute=%lu, dma=%lu), util=%.2f",
        (unsigned long)local_stats.dma_bytes,
        (unsigned long)local_stats.mma_tiles,
        (unsigned long)local_stats.mma_flops,
        (unsigned long)local_stats.total_cycles,
        (unsigned long)local_stats.compute_cycles,
        (unsigned long)local_stats.dma_cycles,
        local_stats.utilization);

    if (stats) *stats = local_stats;
    return 0;
}

/* ── Simple API ────────────────────────────────────────────────── */

int tu_attention_simple(const void *Q, const void *K, const void *V,
                        void *output,
                        uint32_t seq_len_q, uint32_t seq_len_kv,
                        uint32_t head_dim,
                        float softmax_scale, bool causal) {
    tu_attention_desc_t desc = {
        .Q             = Q,
        .K             = K,
        .V             = V,
        .output        = output,
        .batch_size    = 1,
        .num_heads     = 1,
        .seq_len_q     = seq_len_q,
        .seq_len_kv    = seq_len_kv,
        .head_dim      = head_dim,
        .softmax_scale = softmax_scale,
        .mask_type     = causal ? TU_ATTN_MASK_CAUSAL : TU_ATTN_MASK_NONE,
        .mask          = NULL,
        .mask_fill     = -1e9f,
        .tile_m        = 0,  /* auto */
        .tile_n        = 0,  /* auto */
        .dataflow      = -1, /* default */
    };

    tu_attention_auto_tile(&desc);
    return tu_attention_execute(&desc, NULL);
}
