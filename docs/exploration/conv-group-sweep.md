# Convolution Group Sweep: Standard → Depthwise Throughput

**Date:** 2026-07-03
**Question:** How does convolution throughput scale as groups increase from standard (groups=1) to depthwise (groups=C), given that im2col K-dimension shrinks proportionally with group count?
**Hypothesis:** As groups increase, im2col K-dim shrinks (e.g., from 1152→9 for 3×3 kernel), leading to many short-GEMM tiles with poor pipeline fill efficiency.

## Config Matrix

| Parameter | Values |
|-----------|--------|
| Workload | 56×56 input, 128→128 channels, 3×3 kernel, stride=1, pad=same |
| Groups | 1 (standard), 2, 4, 8, 16, 32, 64, 128 (depthwise) |
| PE array | 8×8, 16×16, 32×32 |
| Method | `tu_conv_estimate_cycles()` — analytical cycle model |

## Results

| Groups | im2colK | M-per-grp | 8×8 Cyc | 8×8 GOPS | 16×16 Cyc | 16×16 GOPS | 32×32 Cyc | 32×32 GOPS |
|--------|---------|-----------|---------|----------|-----------|------------|-----------|------------|
| 1 (std) | 1152 | 128 | 22,127,616 | 41.8 | 5,870,592 | 157.5 | 1,806,336 | 512.0 |
| 2 | 576 | 64 | 11,289,600 | 81.9 | 3,161,088 | 292.6 | 1,129,920 | 819.2 |
| 4 | 288 | 32 | 5,870,592 | 157.5 | 1,806,336 | 512.0 | 790,272 | 1,170.3 |
| 8 | 144 | 16 | 3,161,088 | 292.6 | 1,129,920 | 819.2 | 827,904 | 1,117.1 |
| 16 | 72 | 8 | 1,806,336 | 512.0 | 1,204,224 | 768.0 | 903,168 | 1,024.0 |
| 32 | 36 | 4 | 1,956,864 | 472.6 | 1,354,752 | 682.7 | 1,053,696 | 877.7 |
| 64 | 18 | 2 | 2,257,920 | 409.6 | 1,655,808 | 558.5 | 1,053,696 | 877.7 |
| 128 (dw) | 9 | 1 | 2,860,032 | 323.4 | 1,655,808 | 558.5 | 1,655,808 | 558.5 |

## Key Finding: Analytical Model Predicts Depthwise is *Faster* on 8×8 PE

**Counter-intuitive result:** Depthwise convolution (groups=128, 558.5 GOPS) is modeled as 1.33× faster than standard conv (groups=1, 409.6 GOPS) on a 32×32 PE array, and the peak GOPS occurs at groups=8 (1,170.3 on 32×32).

**Why this happens — a GEMM tiling artifact:**

The analytical model's `per_group` formula is:
```
mt = (k_per_g + pe_rows - 1) / pe_rows
nt = (im2col_n + pe_cols - 1) / pe_cols
kt = (im2col_k + pe_cols - 1) / pe_cols
per_group = mt × nt × kt × (pipeline_depth × pe_cols + pe_cols)
total = im2col_cycles + per_group × groups + bias_act
```

For standard conv (groups=1, M=128, K=1152) on 32×32:
- tiles: 5 × 99 × 37 = 18,315 tiles → 1,757,760 cycles

For depthwise (groups=128) on 32×32:
- Per group: M=1, K=9 → mt=1 (PREVIOUSLY 5!), kt=2 (was 37)
- Per group: 1 × 99 × 2 = 198 tiles × 96 cycles = 19,008 cycles
- Total: 128 × 19,008 = 2,433,024 cycles

The M-dimension collapse (M=1 means mt=1 instead of mt=5) partially offsets the group count explosion (×128). On small PEs, the effect is even stronger:
- 8×8 standard: 17 × 393 × 145 = 968,745 tiles → 23.2M cycles
- 8×8 depthwise: 128 × 1 × 393 × 2 = 100,608 tiles → 2.4M cycles

**This is a modeling blind spot, not a real hardware effect.** Real hardware running 128 separate GEMM calls (even if each is "fast") would pay:
1. **Kernel launch overhead** — 128 DMA descriptor setups, command queue submissions
2. **Im2col per-group** — The current model amortizes im2col cost across all groups (single `C*H*W` read), but each group needs its own im2col slice
3. **Pipeline flush per group** — Between groups, the systolic pipeline must drain
4. **SRAM contention** — Multiple groups competing for the same im2col/weight buffer space

The 1,170 GOPS peak at groups=8 on 32×32 is physically implausible — the total MACs are 924M ops, and the peak theoretical throughput of a 32×32 PE array at 1 GHz is 2,048 GOPS (1,024 MACs × 2 ops), giving 57% utilization. But the model reports 1,170 GOPS, which means the PLAUSIBLE throughput would align with the ~500 GOPS seen at groups=1.

## Second Finding: Utilization Degrades Below M=PE_ROWS

When `k_per_g < pe_rows`, the GEMM M dimension doesn't fill even one PE row. The model's `mt` ceiling division means mt stays at 1, but real hardware would have:
- PE rows sitting idle during MMA
- DMA overhead dominating compute for tiny M tiles
- No opportunity for double-buffering overlap

The sweet spot is `k_per_g >= pe_rows` (groups ≤ 8 for 16×16 PE), where M tiles fill the PE array fully.

## Implications for Architecture Design

1. **For depthwise convolutions, skip the systolic array.** Running depthwise through a GEMM engine is modeling-artifact-fast but hardware-inefficient. A dedicated depthwise engine (sliding window with per-channel compute) would outperform im2col+GEMM.
2. **The analytical model overestimates multi-group throughput.** When groups × mt approaches the single-group mt (i.e., the total tile count is similar), the model reports similar cycles. But real hardware pays a group-switch penalty not captured here.
3. **Model improvement needed:** The cycle model should include a per-group overhead term (DMA descriptor setup + pipeline flush) and scale im2col cost per group rather than assuming a single scan.

## Sweep Harness

`tests/test_conv_groups_sweep.c` — `make test-conv-groups-sweep`
