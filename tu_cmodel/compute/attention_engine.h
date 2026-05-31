/*
 * TU CModel — FlashAttention-Style Attention Engine
 * ===================================================
 * Gap O3: Tiled attention computation for transformer inference.
 *
 * Architecture:
 *   The attention engine implements FlashAttention-style tiled computation:
 *     S = Q × K^T           (MMA, using pluggable dataflow)
 *     S = S * scale         (elementwise scalar multiply)
 *     S = S + mask          (elementwise add, optional)
 *     P = softmax(S)        (online softmax, per-row)
 *     O = P × V             (MMA, using pluggable dataflow)
 *
 *   All computation happens on-chip in SRAM. The engine orchestrates
 *   DMA, MMA, elementwise, and softmax operations, managing tiling
 *   across the M (query), N (key/value), and K (head_dim) dimensions.
 *
 *   Tiling strategy (FlashAttention):
 *     - Q is tiled along the M dimension (query tokens)
 *     - K and V are tiled along the N dimension (key/value tokens)
 *     - The head_dim (K dimension) is processed in one MMA tile
 *       when head_dim ≤ 128 (fits in PE array with tiling)
 *     - For each Q tile: load K tiles sequentially, accumulate
 *       softmax statistics (max, sum), compute weighted V sum
 *
 *   Memory layout:
 *     - Q: [batch*heads, seq_len_q, head_dim] in host DRAM (FP16)
 *     - K: [batch*heads, seq_len_kv, head_dim] in host DRAM (FP16)
 *     - V: [batch*heads, seq_len_kv, head_dim] in host DRAM (FP16)
 *     - Output: [batch*heads, seq_len_q, head_dim] in host DRAM (FP16)
 *
 *   SRAM usage (per tile):
 *     - Q_tile:  tile_m × head_dim × 2 bytes  (FP16)
 *     - K_tile:  tile_n × head_dim × 2 bytes  (FP16)
 *     - V_tile:  tile_n × head_dim × 2 bytes  (FP16)
 *     - S_tile:  tile_m × tile_n × 4 bytes    (FP32, scores)
 *     - O_tile:  tile_m × head_dim × 4 bytes  (FP32, accum)
 *     - scratch: tile_m × 4 bytes (max), tile_m × 4 bytes (sum)
 *
 *   Bandwidth awareness:
 *     All operations go through SRAM with bank conflict and
 *     bandwidth stall accounting. DMA bytes and compute cycles
 *     are aggregated into global counters.
 */

#ifndef TU_ATTENTION_ENGINE_H
#define TU_ATTENTION_ENGINE_H

#include "tu_config.h"
#include "tu_sram.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Mask type ---- */
typedef enum {
    TU_ATTN_MASK_NONE   = 0,  /* No mask */
    TU_ATTN_MASK_CAUSAL = 1,  /* Causal mask: position j > i gets -inf */
    TU_ATTN_MASK_CUSTOM = 2,  /* User-provided mask buffer */
} tu_attn_mask_type_t;

/* ---- Attention descriptor ---- */
typedef struct {
    /* Input tensors (host memory, FP16) */
    const void   *Q;           /* [batch*heads, seq_len_q, head_dim], row-major, FP16 */
    const void   *K;           /* [batch*heads, seq_len_kv, head_dim], row-major, FP16 */
    const void   *V;           /* [batch*heads, seq_len_kv, head_dim], row-major, FP16 */

    /* Output tensor (host memory, FP16) */
    void         *output;      /* [batch*heads, seq_len_q, head_dim], row-major, FP16 */

    /* Dimensions */
    uint32_t      batch_size;  /* Number of sequences in batch */
    uint32_t      num_heads;   /* Number of attention heads */
    uint32_t      seq_len_q;   /* Query sequence length */
    uint32_t      seq_len_kv;  /* Key/Value sequence length */
    uint32_t      head_dim;    /* Dimension per head (typically 64 or 128) */

    /* Scaling */
    float         softmax_scale;  /* 1/sqrt(head_dim) — 0 means auto-compute */

    /* Masking */
    tu_attn_mask_type_t mask_type;
    const float         *mask;         /* Custom mask [batch*heads, seq_len_q, seq_len_kv], FP32 (NULL if not custom) */
    float               mask_fill;     /* Value for masked positions (default -1e9 for -inf effect) */

    /* Tiling parameters (0 = auto-select based on SRAM size) */
    uint32_t      tile_m;       /* Q tile size (query tokens per tile) */
    uint32_t      tile_n;       /* KV tile size (key/value tokens per tile) */

    /* Dataflow selection */
    int           dataflow;     /* Dataflow for MMA (0 = current default) */
} tu_attention_desc_t;

/* ---- Statistics output ---- */
typedef struct {
    uint64_t    dma_bytes;        /* Total DMA bytes transferred */
    uint64_t    mma_tiles;        /* Total MMA tiles executed */
    uint64_t    mma_flops;        /* Total effective FLOPs (MACs × 2) */
    uint64_t    compute_cycles;   /* Estimated compute cycles */
    uint64_t    dma_cycles;       /* Estimated DMA cycles */
    uint64_t    total_cycles;     /* Total estimated cycles */
    float       utilization;      /* Compute utilization (compute/total) */
} tu_attention_stats_t;

/* ---- Attention Statistics ---- */

/* ---- API ---- */

/*
 * Execute FlashAttention: O = softmax(Q × K^T * scale + mask) × V
 *
 * Computes multi-head attention with tiled execution to fit in SRAM.
 * All data movement and computation is tracked via global counters
 * (DMA bytes, MMA FLOPs, estimated cycles).
 *
 * The engine uses the pluggable dataflow system for MMA operations,
 * the elementwise pipeline for scaling and masking, and the online
 * softmax engine for numerically-stable probability computation.
 *
 * Parameters:
 *   desc:  Attention descriptor with tensor pointers and dimensions
 *   stats: Optional output for performance statistics (NULL to skip)
 *
 * Returns:
 *   0 on success, -1 on error (prints details to stderr via logging)
 */
int tu_attention_execute(const tu_attention_desc_t *desc,
                         tu_attention_stats_t *stats);

/*
 * Convenience: single-head attention with auto-tiling.
 *
 * Simplified API for the common case where batch_size=1, num_heads=1.
 * Q, K, V, output are FP16 row-major matrices:
 *   Q: [seq_len_q, head_dim]
 *   K: [seq_len_kv, head_dim]
 *   V: [seq_len_kv, head_dim]
 *   output: [seq_len_q, head_dim]
 *
 * softmax_scale: 0 = auto (1/sqrt(head_dim))
 * causal: true = apply causal mask
 */
int tu_attention_simple(const void *Q, const void *K, const void *V,
                        void *output,
                        uint32_t seq_len_q, uint32_t seq_len_kv,
                        uint32_t head_dim,
                        float softmax_scale, bool causal);

/*
 * Validate an attention descriptor.
 * Returns true if valid, false if invalid (prints error to stderr).
 */
bool tu_attention_validate_desc(const tu_attention_desc_t *desc);

/*
 * Compute auto tile sizes based on SRAM and head_dim.
 * Sets desc->tile_m and desc->tile_n to optimal values.
 * Call before tu_attention_execute() if you want auto-tiling.
 */
void tu_attention_auto_tile(tu_attention_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif /* TU_ATTENTION_ENGINE_H */
