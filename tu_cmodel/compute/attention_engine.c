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
 *       Whole row: P = softmax(S)
 *       Streaming: update max/sum, P = exp(S-new_max), rescale O_tile
 *       O_tile += FP16(P) × V_tile      (MMA, FP16→FP32)
 *     Streaming: O_tile /= final_sum
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
 *   sram_o: S/P scratch (tile_m × tile_n × 4 bytes), followed by
 *           O_tile (tile_m × head_dim × 4 bytes), then FP32 mask scratch.
 * S is packed forward to FP16 in its own region; O remains live and disjoint.
 * Multi-KV execution retains FP32 row max/sum after mask scratch, rescales
 * the unnormalized FP32 accumulator before each P*V and divides at the end.
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
            stall += tu_sram_read_bulk(sram, src_addr, &val, sizeof(val));
            stall += tu_sram_write_bulk(sram, dst_addr, &val, sizeof(val));
        }
    }
    return stall;
}

/* Convert FP32 SRAM buffer to packed FP16 in-place.
 * Reads FP32 at offset, writes FP16 back starting at same offset.
 * elem_count: number of FP32 elements.
 * Returns stall cycles. */
static uint64_t fp32_to_fp16_in_sram(tu_sram_region_t *sram,
                                      uint32_t offset, uint32_t elem_count) {
    uint64_t stall = 0;
    /* Shrinking must run forward: output [2*i, 2*i+2) cannot reach
     * the next unread FP32 source at 4*(i+1). Reverse traversal can. */
    for (uint32_t idx = 0; idx < elem_count; idx++) {
        uint32_t src_addr = offset + idx * sizeof(fp32_t);
        fp32_t val;
        stall += tu_sram_read(sram, src_addr, &val);
        fp16_t h = tu_fp32_to_fp16(val);
        /* Explicit width avoids copying a four-byte SRAM word from h. */
        uint32_t dst_addr = offset + idx * sizeof(fp16_t);
        stall += tu_sram_write_bulk(sram, dst_addr, &h, sizeof(h));
    }
    return stall;
}

/* Per-row streaming recurrence: m'=max(m,max(S)), alpha=exp(m-m'),
 * p=exp(S-m'), l'=alpha*l+sum(p), O <- alpha*O before FP16(p)*V.
 * FP32 m/l/O live in SRAM; expf is host FP32. The tile sum uses double,
 * rounded to FP32 at the state update; final division is FP32. Only SRAM
 * stalls are counted here, not a fabricated exp/divide physical latency.
 * Whole-row custom masks normalize P directly without persistent row state.
 * -inf contributes zero; entirely masked rows return zero; NaN/+inf fails.
 */
static bool normalize_attention_tile(uint32_t rows, uint32_t cols, uint32_t dim,
                                    uint32_t score_off, uint32_t output_off,
                                    uint32_t max_off, uint32_t sum_off,
                                    bool streaming, uint64_t *stall) {
    tu_sram_region_t *s = &g_tu.sram_o;
    for (uint32_t r = 0; r < rows; r++) {
        float old_max = -INFINITY, old_sum = 0.0f, hi = -INFINITY;
        if (streaming) {
            *stall += tu_sram_read(s, max_off + r * 4, &old_max);
            *stall += tu_sram_read(s, sum_off + r * 4, &old_sum);
            hi = old_max;
        }
        for (uint32_t j = 0; j < cols; j++) {
            float x;
            *stall += tu_sram_read(s, score_off + (r * cols + j) * 4, &x);
            if (isnan(x) || x == INFINITY) return false;
            hi = fmaxf(hi, x);
        }
        float alpha = old_sum == 0.0f ? 0.0f : expf(old_max - hi);
        double tile_sum = 0.0;
        for (uint32_t j = 0; j < cols; j++) {
            float x;
            uint32_t addr = score_off + (r * cols + j) * 4;
            *stall += tu_sram_read(s, addr, &x);
            float p = hi == -INFINITY || x - hi < -87.0f ? 0.0f : expf(x - hi);
            tile_sum += (double)p;
            *stall += tu_sram_write(s, addr, &p);
        }
        if (streaming) {
            float sum = alpha * old_sum + (float)tile_sum;
            *stall += tu_sram_write(s, max_off + r * 4, &hi);
            *stall += tu_sram_write(s, sum_off + r * 4, &sum);
            for (uint32_t a = 0; a < dim; a++) {
                float out;
                uint32_t addr = output_off + (r * dim + a) * 4;
                *stall += tu_sram_read(s, addr, &out);
                out *= alpha;
                *stall += tu_sram_write(s, addr, &out);
            }
        } else {
            for (uint32_t j = 0; j < cols; j++) {
                float p;
                uint32_t addr = score_off + (r * cols + j) * 4;
                *stall += tu_sram_read(s, addr, &p);
                p = tile_sum == 0.0 ? 0.0f : (float)((double)p / tile_sum);
                *stall += tu_sram_write(s, addr, &p);
            }
        }
    }
    return true;
}

/* ── Auto-tiling ──────────────────────────────────────────────── */

/* Shared configured reservation contract. Validate products before byte
 * scaling so even UINT32_MAX dimensions cannot overflow the sums. */
typedef struct {
    uint32_t q_bytes, k_bytes, s_bytes, o_bytes;
} attention_footprint_t;

static bool attention_footprint(uint32_t tm, uint32_t tn, uint32_t dim,
                                uint32_t n, attention_footprint_t *out) {
    if (!tm || !tn || !dim || !n) return false;
    uint64_t q = (uint64_t)tm * dim, k = (uint64_t)tn * dim;
    uint64_t s = (uint64_t)tm * tn;
    if (q > g_tu.rt_cfg.sram_a_size / sizeof(fp16_t) ||
        k > g_tu.rt_cfg.sram_w_size / (2 * sizeof(fp16_t)) ||
        s > g_tu.rt_cfg.sram_o_size / sizeof(fp32_t) ||
        q > g_tu.rt_cfg.sram_o_size / sizeof(fp32_t)) return false;
    uint64_t o_end = s * sizeof(fp32_t) + q * sizeof(fp32_t);
    uint64_t s_tile_bytes = s * sizeof(fp32_t);
    uint64_t state_bytes = tn < n ? (uint64_t)tm * 2 * sizeof(float) : 0;
    uint64_t max_o_usage = o_end + s_tile_bytes + state_bytes;
    if (max_o_usage > g_tu.rt_cfg.sram_o_size) return false;
    if (out) *out = (attention_footprint_t){q * sizeof(fp16_t),
        k * sizeof(fp16_t), s * sizeof(fp32_t), q * sizeof(fp32_t)};
    return true;
}

void tu_attention_auto_tile(tu_attention_desc_t *desc) {
    if (!desc) return;
    desc->tile_m = desc->tile_n = 0;
    if (!desc->seq_len_q || !desc->seq_len_kv || !desc->head_dim) return;
    uint32_t dim = desc->head_dim, n = desc->seq_len_kv;
    uint64_t max_n = g_tu.rt_cfg.sram_w_size / (4ull * dim);
    if (max_n > n) max_n = n;
    uint64_t best_score = 0;
    /* Retain the bounded 64-query search, not a physical optimum claim.
     * At fixed tm, footprint grows with tn except when full N removes row
     * state. Test that endpoint separately, then solve the streaming budget.
     * Ragged edges are executable; never round an admitted tile upward. */
    for (uint32_t tm = 1; tm <= desc->seq_len_q && tm <= 64; tm++) {
        uint32_t tn = 0;
        if (max_n == n && attention_footprint(tm, n, dim, n, NULL)) {
            tn = n;
        } else if (n > 1) {
            uint64_t row_budget = g_tu.rt_cfg.sram_o_size / tm;
            uint64_t fixed = 4ull * dim + 8;
            uint64_t bound = row_budget > fixed ? (row_budget - fixed) / 8 : 0;
            if (bound > max_n) bound = max_n;
            if (bound >= n) bound = n - 1;
            tn = (uint32_t)bound;
        }
        if (!attention_footprint(tm, tn, dim, n, NULL)) continue;
        uint64_t score = (uint64_t)tm * tn;
        /* Ascending tm plus strict improvement defines ties: smaller tm. */
        if (score > best_score) {
            best_score = score;
            desc->tile_m = tm;
            desc->tile_n = tn;
        }
    }
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
    if (!isfinite(desc->softmax_scale) || desc->softmax_scale < 0.0f) {
        TU_LOG_ERR(TU_COMP_CORE, "Attention: nonfinite or negative softmax_scale");
        return false;
    }
    if (desc->mask_type < TU_ATTN_MASK_NONE || desc->mask_type > TU_ATTN_MASK_CUSTOM ||
        (desc->mask_type == TU_ATTN_MASK_CUSTOM && !desc->mask) ||
        (desc->mask_type == TU_ATTN_MASK_CAUSAL &&
         (isnan(desc->mask_fill) || desc->mask_fill == INFINITY))) return false;
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
    if (!tu_attention_validate_desc(desc)) {
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


    /* One admission producer is shared with auto selection. */
    attention_footprint_t footprint;
    if (!attention_footprint(tile_m, tile_n, aw, N, &footprint)) {
        TU_LOG_ERR(TU_COMP_CORE, "Attention: tile element footprint exceeds SRAM");
        return -1;
    }
    uint32_t q_tile_bytes = footprint.q_bytes;
    uint32_t k_tile_bytes = footprint.k_bytes;
    uint32_t kt_tile_bytes = k_tile_bytes;
    uint32_t s_tile_bytes = footprint.s_bytes;
    uint32_t o_tile_bytes = footprint.o_bytes;


    /* Derive offsets from the already admitted shared footprint. */
    /* O remains live while S is FP32, not only after packing P to FP16.
     * Reserve disjoint score/probability, accumulator and mask regions. */
    uint32_t O_off_aligned = s_tile_bytes;
    uint64_t o_end = (uint64_t)O_off_aligned + o_tile_bytes;
    bool streaming = tile_n < N;

    /* SRAM offsets */
    uint32_t Q_off  = 0;                    /* sram_a: Q tile */
    uint32_t K_off  = 0;                    /* sram_w: K tile */
    uint32_t KT_off = k_tile_bytes;         /* sram_w: K^T tile (after K) */
    uint32_t V_off  = 0;                    /* sram_w: V tile (reuses K space) */
    uint32_t S_off  = 0;                    /* sram_o: S/P tile (FP32 scores → FP16 probs) */

    /* O follows the full FP32 S lifetime, not the smaller packed P. */
    uint32_t O_off  = O_off_aligned;
    uint32_t mask_off = (uint32_t)o_end;
    uint32_t max_off = mask_off + s_tile_bytes;
    uint32_t sum_off = max_off + (streaming ? tile_m * sizeof(float) : 0);

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
            if (streaming) {
                for (uint32_t r = 0; r < q_count; r++) {
                    float hi = -INFINITY, sum = 0.0f;
                    local_stats.compute_cycles += tu_sram_write(&g_tu.sram_o, max_off + r * 4, &hi);
                    local_stats.compute_cycles += tu_sram_write(&g_tu.sram_o, sum_off + r * 4, &sum);
                }
            }
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
                if (desc->mask_type != TU_ATTN_MASK_NONE) {
                    /* Source mask pitch is N, destination pitch is kv_count. */
                    for (uint32_t r = 0; r < q_count; r++) {
                        for (uint32_t j = 0; j < kv_count; j++) {
                            float value = desc->mask_type == TU_ATTN_MASK_CAUSAL ?
                                (kvi + j > qi + r ? mfill : 0.0f) :
                                desc->mask[((size_t)h * M + qi + r) * N + kvi + j];
                            local_stats.compute_cycles += tu_sram_write(&g_tu.sram_o,
                                mask_off + (r * kv_count + j) * 4, &value);
                        }
                    }
                    local_stats.compute_cycles += tu_ew_add_tensors(&g_tu.sram_o,
                        S_off, mask_off, S_off, q_count * kv_count);
                }

                /* ── Step 4: P = softmax(S) ── */
                if (streaming || desc->mask_type == TU_ATTN_MASK_CUSTOM) {
                    if (!normalize_attention_tile(q_count, kv_count, aw, S_off, O_off,
                            max_off, sum_off, streaming, &local_stats.compute_cycles)) return -1;
                } else {
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

            } /* end KV tile loop */

            if (streaming) {
                for (uint32_t r = 0; r < q_count; r++) {
                    float sum;
                    local_stats.compute_cycles += tu_sram_read(&g_tu.sram_o, sum_off + r * 4, &sum);
                    for (uint32_t a = 0; a < aw; a++) {
                        float out;
                        uint32_t addr = O_off + (r * aw + a) * 4;
                        local_stats.compute_cycles += tu_sram_read(&g_tu.sram_o, addr, &out);
                        out = sum == 0.0f ? 0.0f : out / sum;
                        local_stats.compute_cycles += tu_sram_write(&g_tu.sram_o, addr, &out);
                    }
                }
            }

            /* ── Convert O_tile from FP32 to FP16 and DMA out ── */
            /* Read FP32 O_tile from SRAM, convert to FP16, write to host */
            {
                uint32_t num_elems = q_count * aw;
                /* Read FP32 into temp buffer, convert, DMA to host */
                fp32_t *temp_fp32 = (fp32_t *)malloc(num_elems * sizeof(fp32_t));
                fp16_t *temp_fp16 = (fp16_t *)malloc(num_elems * sizeof(fp16_t));
                if (!temp_fp32 || !temp_fp16) {
                    free(temp_fp32); free(temp_fp16);
                    return -1;
                }
                {
                    for (uint32_t i = 0; i < num_elems; i++) {
                        tu_sram_read(&g_tu.sram_o,
                            O_off + i * sizeof(fp32_t), &temp_fp32[i]);
                        if (!isfinite(temp_fp32[i])) {
                            free(temp_fp32); free(temp_fp16);
                            return -1;
                        }
                    }
                    tu_fp32_to_fp16_buffer(temp_fp32, temp_fp16, num_elems);
                    for (uint32_t i = 0; i < num_elems; i++) {
                        if ((temp_fp16[i] & 0x7c00u) == 0x7c00u) {
                            free(temp_fp32); free(temp_fp16);
                            return -1;
                        }
                    }

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
