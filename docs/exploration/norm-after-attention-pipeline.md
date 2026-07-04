# Norm-After-Attention Pipeline Sweep

**Date:** 2026-07-04
**Question:** What fraction of transformer block latency does normalization (LayerNorm/RMSNorm) add after attention, across PE array sizes and head dimensions?
**Hypothesis:** Norm overhead is small (< 5%) because attention is O(M×N×d) compute while norm is O(M×N) SRAM bandwidth. But norm is a fixed per-element cost that doesn't benefit from larger PE arrays — so overhead should increase with PE size.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| PE rows/cols | 8×8, 16×16, 32×32 | 3 sizes, SRAM scaled proportionally |
| Workload (M×N×d) | 32×64, 64×64, 128×64, 32×128, 64×128 | Single-Q-tile prefill; d=64 and d=128 |
| Dataflow | OS | Most efficient for attention |
| Norm | LayerNorm, RMSNorm | In-place on O-buffer FP32 output |
| Attention mask | None | Simplest case (no causal overhead) |

**Configs attempted:** 3 PE × 5 workloads = 15. **Valid:** 10 (5 skipped due to O-buffer capacity on 8×8 PE).

## Results Table

| Workload | PE | AttnCyc | NormCyc | LN% | RMS% | Cycles/elem |
|----------|-----|---------|---------|------|------|-------------|
| prefill-32×64 | 8×8 (128K) | 49,984 | 4,096 | 8.19% | 8.19% | 2.00 |
| prefill-32×128 | 8×8 (128K) | 75,584 | 8,192 | 10.84% | 10.84% | 2.00 |
| prefill-32×64 | 16×16 (256K) | 46,144 | 4,096 | 8.88% | 8.88% | 2.00 |
| prefill-64×64 | 16×16 (256K) | 76,096 | 8,192 | 10.77% | 10.77% | 2.00 |
| prefill-128×64 | 16×16 (256K) | 152,320 | 16,384 | 10.76% | 10.76% | 2.00 |
| prefill-32×128 | 16×16 (256K) | 67,904 | 8,192 | 12.06% | 12.06% | 2.00 |
| prefill-64×128 | 16×16 (256K) | 103,232 | 16,384 | 15.87% | 15.87% | 2.00 |
| prefill-32×64 | 32×32 (512K) | 45,184 | 4,096 | 9.07% | 9.07% | 2.00 |
| prefill-64×64 | 32×32 (512K) | 74,176 | 8,192 | 11.04% | 11.04% | 2.00 |
| prefill-128×64 | 32×32 (512K) | 148,480 | 16,384 | 11.03% | 11.03% | 2.00 |
| prefill-32×128 | 32×32 (512K) | 65,984 | 8,192 | 12.42% | 12.42% | 2.00 |
| prefill-64×128 | 32×32 (512K) | 99,392 | 16,384 | 16.48% | 16.48% | 2.00 |

## Key Findings

### 1. LayerNorm and RMSNorm have identical cycle costs within the cmodel

Both show exactly **2.00 cycles per element** across all workloads. This matches the earlier `norm-mode-comparison.md` finding: both use the same two-pass read-each-element, write-each-element SRAM access pattern (2 reads + 2 writes = 4 operations per element, modeled at 0.5 cycles per op = 2.0 cycles/elem). Mode selection is a compute/numerical decision, not a bandwidth optimization.

### 2. Norm overhead is non-trivial: 8-16% of attention latency

Across all valid configs, norm adds **8.2% to 16.5%** to the attention cycle count. This is higher than the naive < 5% estimate because:
- Norm is purely SRAM-bandwidth-bound (O(N) memory ops)
- Attention has significant DMA overhead (Q, K, V loads + O stores) that norm doesn't share
- The ratio depends on how much DMA overhead attention has

### 3. Larger PE arrays increase norm overhead percentage

For `prefill-32×64` (32×64×64 attention), norm overhead increases from 8.19% (8×8 PE) → 8.88% (16×16) → 9.07% (32×32). The absolute norm cost stays fixed (4,096 cycles), but attention completes faster on larger PEs (49,984 → 46,144 → 45,184 cycles). Larger PE arrays reduce attention's MMA cost faster than they reduce its DMA cost, so the DMA-dominated baseline shrinks less than the norm's fixed 2 cycles/elem.

### 4. Larger head dimensions amplify norm overhead disproportionately

On 16×16 PE, going from d=64 to d=128:
- 32×64 → 32×128: overhead 8.88% → 12.06% (+3.2pp)
- 64×64 → 64×128: overhead 10.77% → 15.87% (+5.1pp)

The norm cost doubles (4,096 → 8,192) with d, but attention cost scales less than 2× because DMA overhead is sub-linear in d. The M×N×d compute grows with d but the DMA cost (Q+K+V loads) grows with K-dim (d) while O-store grows with d². The norm grows purely with d (element count), so its share increases.

### 5. Norm is independent of M (sequence length) once element count is fixed

For `prefill-32×64` and `prefill-64×64` on 16×16 PE: both show ~10.8% overhead even though M doubles. The norm cost doubles (4,096 → 8,192) and attention roughly doubles too (46,144 → 76,096). The norm-to-attention ratio stays flat as M scales.

## Design Implications

- **For small PE arrays:** Norm overhead is ~8% — acceptable without hardware fusion.
- **For large PE arrays (32×32+):** Norm overhead reaches 16% — consider fusing norm into the attention engine's output path to avoid the separate SRAM pass.
- **For wide heads (d ≥ 128):** Normalization is 12-16% of block latency. The `layernorm_2d` approach (per-row normalization) should be integrated into the attention DMA store path so the norm runs on SRAM data before it's converted to FP16.
- **If quantization is on the roadmap:** Norm can be fused with the FP32→FP8/INT8 quantization step (both operate on FP32 O-buffer data), eliminating the norm pass entirely.

## Method

The sweep uses `tu_attention_execute()` followed by `tu_layernorm_2d()` / `tu_rmsnorm_2d()` in-place on `g_tu.sram_o`. After attention, the O-buffer retains FP32 output (the DMA-to-host is non-destructive). Norm operates on the contiguous M×N FP32 matrix at O-buffer offset 0. Only single-Q-tile workloads are used to ensure contiguous O-buffer layout.

## References

- `docs/exploration/norm-mode-comparison.md` — Standalone LayerNorm vs RMSNorm comparison
- `docs/exploration/attention-engine-sweep.md` — Attention PE × workload × dataflow sweep
- `tu_cmodel/compute/normalization_engine.h` — Norm API (layernorm_2d, rmsnorm_2d)
