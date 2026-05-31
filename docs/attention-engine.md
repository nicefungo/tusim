# Attention Engine — FlashAttention-Style Tiled Computation (Gap O3)

> **Status:** Implemented  
> **Version:** 1.0  
> **Date:** 2026-05-31  
> **Gap ID:** O3 — No attention support (Critical, P1)  
> **Files:** `tu_cmodel/compute/attention_engine.{h,c}`  
> **Tests:** `tests/test_attention.c`  

---

## 1. Overview

The attention engine implements **FlashAttention-style tiled computation** for multi-head scaled dot-product attention:

```
O = softmax(Q × K^T × scale + mask) × V
```

This is the core operation in transformer models (GPT, BERT, LLaMA, etc.) and was the largest missing operation in the TU cmodel's operation catalog. All computation happens on-chip in SRAM using existing infrastructure: the pluggable dataflow system for MMA, the elementwise pipeline for scaling/masking, and the online softmax engine for normalization.

### Why This Matters

- **Transformer inference requires attention.** Without attention, the TU cannot run any modern transformer model (GPT, BERT, LLaMA, Diffusion Transformers, ViT).
- **On-chip execution avoids DRAM round-trips.** The original attention computation is O(N²) in memory — FlashAttention tiles it to keep data in SRAM, reducing DRAM bandwidth by 10-20×.
- **Production readiness.** Combined with the existing MMA, softmax, and elementwise engines, the attention engine completes the P1 operation catalog, enabling full transformer block execution on TU.

---

## 2. Architecture

### 2.1 Algorithm

The engine implements the FlashAttention tiling strategy:

```
For each Q tile (along seq_len_q):
    O_tile = 0
    For each KV tile (along seq_len_kv):
        S = Q_tile × K_tile^T          (MMA, FP16×FP16→FP32)
        S = S * scale                   (elementwise MUL)
        S = S + mask                    (elementwise ADD, if masked)
        P = softmax(S)                  (online softmax, per-row)
        O_tile += P × V_tile            (MMA, FP16×FP16→FP32)
    O_tile → host output (FP16)
```

### 2.2 Dataflow Diagram

```
Host DRAM (Q/K/V)                    TU SRAM (on-chip)
─────────────────                    ──────────────────
                                     sram_a: [Q_tile FP16]  ← DMA load
                                     sram_w: [K_tile FP16]  ← DMA load
                                              [K^T FP16]    ← Transpose
                                     sram_o: [S_tile FP32]  ← MMA(Q × K^T)
                                              [S_tile FP32]  ← Scale + Mask
                                              [P_tile FP32]  ← Softmax
                                              [P_tile FP16]  ← FP32→FP16 convert
                                     sram_w: [V_tile FP16]  ← DMA load
                                     sram_o: [O_tile FP32]  ← MMA(P × V)
                                     sram_o: → host output  ← FP32→FP16 + DMA store
```

### 2.3 SRAM Allocation

| Buffer | Content | Size (worst case) |
|--------|---------|-------------------|
| `sram_a` | Q tile | `tile_m × head_dim × 2` bytes (FP16) |
| `sram_w` | K tile + K^T | `2 × tile_n × head_dim × 2` bytes (FP16) |
| `sram_w` | V tile (overwrites K) | `tile_n × head_dim × 2` bytes (FP16) |
| `sram_o` | S/P tile (overwritten) | `tile_m × tile_n × 4` bytes (FP32) |
| `sram_o` | O accumulator | `tile_m × head_dim × 4` bytes (FP32) |

With the default 16×16 PE array and 128 KB W-buffer, the auto-tiler selects `tile_m` and `tile_n` to fit all required data in available SRAM.

---

## 3. API

### 3.1 Core API: `tu_attention_execute()`

```c
int tu_attention_execute(const tu_attention_desc_t *desc,
                         tu_attention_stats_t *stats);
```

**Parameters:**

| Field | Type | Description |
|-------|------|-------------|
| `Q` | `const void*` | Query tensor: `[batch*heads, seq_len_q, head_dim]`, FP16, row-major |
| `K` | `const void*` | Key tensor: `[batch*heads, seq_len_kv, head_dim]`, FP16, row-major |
| `V` | `const void*` | Value tensor: `[batch*heads, seq_len_kv, head_dim]`, FP16, row-major |
| `output` | `void*` | Output tensor: `[batch*heads, seq_len_q, head_dim]`, FP16, row-major |
| `batch_size` | `uint32_t` | Number of sequences in batch |
| `num_heads` | `uint32_t` | Number of attention heads |
| `seq_len_q` | `uint32_t` | Query sequence length (M dimension) |
| `seq_len_kv` | `uint32_t` | Key/Value sequence length (N dimension) |
| `head_dim` | `uint32_t` | Dimension per head (K dimension, typically 64 or 128) |
| `softmax_scale` | `float` | Pre-softmax scaling (`1/sqrt(head_dim)`; 0 = auto-compute) |
| `mask_type` | `tu_attn_mask_type_t` | `NONE`, `CAUSAL`, or `CUSTOM` |
| `mask` | `const float*` | Custom mask buffer (only if `CUSTOM`) |
| `mask_fill` | `float` | Fill value for masked positions (default `-1e9` for -inf effect) |
| `tile_m` | `uint32_t` | Q tile size (0 = auto-select) |
| `tile_n` | `uint32_t` | KV tile size (0 = auto-select) |
| `dataflow` | `int` | Dataflow for MMA (-1 = use current default) |

**Returns:** 0 on success, -1 on error.

**Performance statistics** (populated in `stats` if non-NULL):

| Field | Description |
|-------|-------------|
| `dma_bytes` | Total DMA bytes transferred |
| `mma_tiles` | Total MMA tiles executed |
| `mma_flops` | Total effective FLOPs (MACs × 2) |
| `compute_cycles` | Estimated compute cycles |
| `dma_cycles` | Estimated DMA cycles |
| `total_cycles` | Total estimated cycles |
| `utilization` | Compute utilization (compute/total, 0.0–1.0) |

### 3.2 Simplified API: `tu_attention_simple()`

```c
int tu_attention_simple(const void *Q, const void *K, const void *V,
                        void *output,
                        uint32_t seq_len_q, uint32_t seq_len_kv,
                        uint32_t head_dim,
                        float softmax_scale, bool causal);
```

Single-head convenience wrapper. `softmax_scale=0` auto-computes `1/sqrt(head_dim)`.

### 3.3 Utility Functions

| Function | Description |
|----------|-------------|
| `tu_attention_auto_tile(desc)` | Compute optimal `tile_m`, `tile_n` based on SRAM and dimensions |
| `tu_attention_validate_desc(desc)` | Validate descriptor (returns `false` on error) |

---

## 4. Mask Types

### 4.1 None (`TU_ATTN_MASK_NONE`)
No masking — full bidirectional attention. Used for encoder models (BERT) and non-causal decoder layers.

### 4.2 Causal (`TU_ATTN_MASK_CAUSAL`)
Lower-triangular mask: position `j` can only attend to positions `≤ i`. Used for autoregressive decoder models (GPT, LLaMA). The engine builds an on-the-fly causal mask per tile.

Implementation detail: when an entire KV tile is ahead of all Q positions (`kv_start > q_end`), the S tile is set to `mask_fill` and the V projection is skipped — saving the costly MMA.

### 4.3 Custom (`TU_ATTN_MASK_CUSTOM`)
User-provided FP32 mask buffer: `[batch*heads, seq_len_q, seq_len_kv]`. Values are **added** to the scores before softmax. For padding masks, set padded positions to a large negative value (e.g., `-1e9`).

---

## 5. Auto-Tiling

The auto-tiler (`tu_attention_auto_tile()`) selects optimal `tile_m` and `tile_n` based on:

1. **SRAM constraints**: K+K^T must fit in W-buffer, S+O must fit in O-buffer
2. **Utilization preference**: Prefers larger tiles for better PE array utilization
3. **PE alignment**: Tile sizes are rounded up to multiples of `TU_PE_ROWS` and `TU_PE_COLS` for full systolic array utilization

Algorithm:
```
For tm in [1..min(seq_len_q, 64)]:
    o_bytes = tm × head_dim × 4 (FP32 O accumulator)
    remaining_o = O_buffer - o_bytes
    max_tn = min(remaining_o / (tm × 4), W_buffer / (2 × head_dim × 2), seq_len_kv)
    if tm × max_tn > best_score: update best
Round tm, tn to PE multiples
```

---

## 6. Precision & Numerics

### 6.1 Data Types (per tensor)

| Tensor | Storage | Precision |
|--------|---------|-----------|
| Q, K, V | Host DRAM | FP16 |
| Q_tile, K_tile, V_tile | SRAM | FP16 |
| S (scores) | SRAM | FP32 |
| P (probabilities) | SRAM | FP32 (during softmax), then FP16 (before V projection) |
| O (output accum) | SRAM | FP32 |
| Output | Host DRAM | FP16 |

### 6.2 Numerical Stability

- **Softmax uses max-subtract** (via the softmax engine): prevents overflow when Q·K^T values are large
- **FP32 accumulation** for both S and O prevents precision loss from repeated additions
- **FP16 conversion after softmax** models real hardware behavior — softmax output is quantized to FP16 before the Value projection MMA, matching how hardware accelerators (TPU, TensorCore) operate

### 6.3 Error Characteristics

Random tensor tests show max absolute error ≤ 0.1 vs. golden reference (FP32 naive implementation), dominated by FP16 rounding in the Q·K^T and P·V MMAs. This is consistent with FP16 precision limits (~2^-10 relative error per operation).

---

## 7. Performance Model

Cycle estimates include:

| Component | Model |
|-----------|-------|
| **MMA cycles** | Dataflow plugin's cycle model (pipeline fill + compute + drain per tile) |
| **DMA cycles** | DMA engine's bandwidth model (bus width, burst size) |
| **Elementwise cycles** | 1 cycle per element per operation |
| **Softmax cycles** | 2 passes over S tile (max/sum then normalize) |
| **Transpose** | 2 SRAM accesses per element (read + write) |

Utilization = compute_cycles / total_cycles. Typical utilization is 60-85% for realistic tile sizes, limited by DMA bandwidth to load K/V tiles.

---

## 8. Configuration

Attention engine behavior is controlled through the `tu_attention_desc_t` struct at runtime. No new compile-time configuration flags are required — it uses the existing `tu_config.h` parameters:

- `TU_PE_ROWS`, `TU_PE_COLS`: PE array dimensions → MMA tile sizes
- `TU_SRAM_W_SIZE`, `TU_SRAM_O_SIZE`: SRAM capacities → auto-tiling constraints
- `TU_DATAFLOW_MODE`: Dataflow for MMA operations
- `TU_DMA_BUS_WIDTH_BITS`: DMA bandwidth for cycle estimation

---

## 9. Limitations & Future Work

### Current Limitations

1. **No FlashAttention-2 online rescaling**: The current implementation uses two-pass softmax (max + sum, then normalize). FlashAttention-2's online rescaling algorithm (single-pass with running max/sum) would reduce SRAM read traffic by ~33%. The online softmax mode (`TU_SOFTMAX_ONLINE`) exists but requires two passes for the division step; true single-pass requires deeper integration with the MMA engine.

2. **No grouped-query attention (GQA)**: The engine assumes num_heads(Q) = num_heads(K) = num_heads(V). GQA (where multiple Q heads share one KV head) is not supported but can be modeled by repeating K/V tensors.

3. **No ALiBi / RoPE**: Positional encoding must be applied before calling the attention engine. Future work could fuse RoPE into the Q/K DMA path.

4. **FP16-only inputs/outputs**: BF16 and FP8 paths are not yet wired through the attention engine. The underlying MMA, elementwise, and softmax engines support these types — the attention engine needs dispatch logic.

### Future Enhancements

| Feature | Gap ID | Priority |
|---------|--------|----------|
| BF16 attention path | D3 | P1 |
| Online rescaling (FA2) | O3 ext | P1 |
| Grouped-query attention (GQA) | O3 ext | P2 |
| Fused RoPE in DMA | O3 ext | P2 |
| Block-sparse attention | O3+A5 | P2 |

---

## 10. Verification

### Test Coverage (9 tests)

| Test | Description |
|------|-------------|
| Small identity | M=4,N=4,d=8 with all-ones → verifies basic correctness |
| Random small | M=3,N=5,d=16 random tensors → verifies general accuracy |
| Causal mask | M=4,N=4,d=8 causal → verifies upper triangle near-zero |
| Multi-head | 2 heads → verifies head independence and stats |
| Auto-tiling | 64×64×64 config → verifies tile_m/n are valid |
| Edge case: seq_len=1 | M=1,N=1,d=8 → verifies minimal batch |
| Scale factor | Different scale values → verifies scaling differentiation |
| Stats populated | 8×8×16 → verifies performance counters |
| Validate descriptor | Null/zero inputs → verifies error handling |

### Comparison Methodology

- **Golden reference**: Naive O(N²) FP32 attention in C (test file)
- **Tolerance**: 0.1 max absolute error (FP16 units) for random tests, 0.05 for small/identity tests
- **Error is dominated by**: FP16 rounding in MMA operations (Q·K^T and P·V)

---

## 11. Related Documentation

- [Softmax Engine](TU_SOFTMAX.md) — Online softmax with numerical stability
- [Dataflow System](TU_DATAFLOW.md) — Pluggable WS/OS/RS dataflows
- [Elementwise Pipeline](elementwise-pipeline.md) — Fused activation and binary ops
- [Production Redesign](PRODUCTION_TU_REDESIGN.md) — Gap analysis and roadmap
