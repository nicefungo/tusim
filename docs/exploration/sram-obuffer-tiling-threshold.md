# SRAM O-Buffer Sizing: Tiling Threshold and Throughput Impact

**Date:** 2026-06-09
**Question:** How does O-buffer sizing affect throughput for a 128×128×256 GEMM on 16×16 PE? Where is the tiling threshold, and what's the efficiency impact of operating below it?
**Hypothesis:** The O-buffer is the tightest memory constraint because it stores FP32 accumulators (4 B/elem vs 2 B/elem for FP16 weights/activations). Below 64 KB, M-tiling kicks in, adding DMA reload overhead and fill/drain cycles that compound with each tile.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| O-buffer size | 16, 24, 32, 40, 48, 56, 64, 80, 96, 128 KB | Output accumulator buffer |
| W-buffer | 128 KB (fixed) | Weights always fit (64 KB needed) |
| A-buffer | 64 KB (fixed) | Activations exactly fit (64 KB needed) |
| PE array | 16×16 | Sweet spot from prior exploration |
| Dataflow | weight_stationary | Systolic |
| Workload | M=128, N=128, K=256 | 8.39 MFLOPs GEMM |
| Bus width | 256-bit (32 B/cycle) | Default |
| Clock | 1.0 GHz | Default |

**Configs tested:** 10 O-buffer sizes, analytical cycle model validated against cmodel at baseline.

## Buffer Minimums (Tiling-Free Thresholds)

For a tiling-free GEMM (single DMA load per buffer):

| Buffer | Formula | Bytes needed | KB needed | Default | Headroom |
|--------|---------|-------------|-----------|---------|----------|
| W (FP16) | M×K×2 | 65,536 | 64 KB | 128 KB | 2.0× |
| A (FP16) | K×N×2 | 65,536 | 64 KB | 64 KB | **1.0× (exact)** |
| O (FP32) | M×N×4 | 65,536 | 64 KB | 64 KB | **1.0× (exact)** |

**Critical observation:** The O-buffer stores FP32 accumulator values. If O were FP16 (stored as output), only 32 KB would be needed — but the cmodel accumulates in FP32 for numerical accuracy. This doubling makes the O-buffer the binding constraint.

## Results Table (Conservative: A Reloaded per M-Tile)

| O-buf KB | M-tiles | W DMA KB | A DMA KB | O DMA KB | Total DMA KB | DMA cyc | Fill | Compute | Drain | Total cyc | TOPS | Util% |
|----------|---------|----------|----------|----------|-------------|---------|------|---------|-------|-----------|------|-------|
| 16 | 4 | 64.0 | 256.0 | 64.0 | 384.0 | 12,288 | 64 | 16,384 | 16 | 28,752 | 0.292 | 57.0% |
| 24 | 3 | 64.0 | 192.0 | 64.0 | 320.0 | 10,240 | 48 | 16,384 | 16 | 26,688 | 0.314 | 61.4% |
| 32 | 2 | 64.0 | 128.0 | 64.0 | 256.0 | 8,192 | 32 | 16,384 | 16 | 24,624 | 0.341 | 66.5% |
| 40 | 2 | 64.0 | 128.0 | 64.0 | 256.0 | 8,192 | 32 | 16,384 | 16 | 24,624 | 0.341 | 66.5% |
| 48 | 2 | 64.0 | 128.0 | 64.0 | 256.0 | 8,192 | 32 | 16,384 | 16 | 24,624 | 0.341 | 66.5% |
| 56 | 2 | 64.0 | 128.0 | 64.0 | 256.0 | 8,192 | 32 | 16,384 | 16 | 24,624 | 0.341 | 66.5% |
| **64** | **1** | **64.0** | **64.0** | **64.0** | **192.0** | **6,144** | **16** | **16,384** | **16** | **22,560** | **0.372** | **72.6%** |
| 80 | 1 | 64.0 | 64.0 | 64.0 | 192.0 | 6,144 | 16 | 16,384 | 16 | 22,560 | 0.372 | 72.6% |
| 96 | 1 | 64.0 | 64.0 | 64.0 | 192.0 | 6,144 | 16 | 16,384 | 16 | 22,560 | 0.372 | 72.6% |
| 128 | 1 | 64.0 | 64.0 | 64.0 | 192.0 | 6,144 | 16 | 16,384 | 16 | 22,560 | 0.372 | 72.6% |

**Peak TOPS reference:** 0.512 (256 MACs × 2 ops/MAC at 1 GHz).

## Key Findings

### 1. O-buffer tiling threshold is exactly 64 KB for 128×128 output matrices

The M-tiling threshold follows a clean formula:

```
m_per_tile = floor(o_buf_bytes / (N × 4))
```

For N=128 and FP32 O: `m_per_tile = floor(o_buf / 512)`.

| O-buf KB | m_per_tile | M-tiles | DMA penalty |
|----------|-----------|---------|-------------|
| 16 | 32 | 4 | **+100%** (6,144→12,288) |
| 24 | 48 | 3 | +67% |
| 32-56 | 64-112 | 2 | +33% |
| ≥64 | 128 | 1 | **none** |

Below 64 KB, additional M-tiles force A-reloads (the A-buffer is consumed by compute and must be re-DMA'd for each M-tile), which is the dominant cost. The W data total doesn't grow (each M-tile loads its own W-slice).

### 2. Tiling shows a step-function pattern — no continuous degradation

The throughput penalty is NOT gradual. Between 32 KB and 56 KB O-buffer, all configurations produce identical performance (0.341 TOPS) because they all require 2 M-tiles. The exact O-buffer size within this range only affects the balance between the two tiles but not the total compute cycles. The A-reload DMA is the same for both tiles.

**This means SRAM sizing decisions are binary, not continuous:** either you fit in one tile (≥64 KB) or you pay a fixed overhead for an integer number of tiles.

### 3. O-buffer above 64 KB provides zero benefit for this workload

At O=64 KB, the output fits in a single tile and no further M-tiling is needed. Increasing O-buffer to 80, 96, or 128 KB changes nothing — the bottleneck shifts from O-buffer to DMA bus width (the ~20% DMA floor documented in the K-sweep).

**Doubling O-buffer from 64 to 128 KB is wasted silicon** for 128×128 workloads. The extra 64 KB would only be useful for:
- Larger output matrices (M×N ≥ 16,384 elements when N=128 → M ≥ 128; or larger N)
- Double-buffering (ping-pong for DMA/compute overlap on consecutive GEMMs)
- Multi-context execution (holding state for multiple cores/contexts)

### 4. The FP32 accumulator doubles O-buffer pressure vs FP16 data

If the output accumulator were FP16 (2 bytes), the tiling threshold would be 32 KB instead of 64 KB — a 2× reduction. This is the architectural price of mixed-precision: W and A are FP16 (2B), but O accumulates in FP32 (4B) for numerical stability. The O-buffer is always the tightest constraint in any equal-dimension GEMM:

```
O_buf_needed = M × N × 4   (FP32)
W_buf_needed = M × K × 2   (FP16)
A_buf_needed = K × N × 2   (FP16)
```

For square GEMMs (M=N=K), O-buffer needs 2× the bytes of W or A. For typical transformer FFN layers where K=4×M, W-buffer dominates (4× larger). But for balanced GEMMs (M≈N≈K), the O-buffer is the binding constraint.

## Actionable Conclusion

**For the ONNX compiler's hardware target at 128×128 output dimensions, 64 KB O-buffer is the minimum non-negotiable SRAM budget.** Below 64 KB, tiling overhead increases linearly with tile count, reducing throughput by 8-22%.

**Design implications:**

1. **Don't proportionally scale buffers.** The W-buffer at 128 KB has 2× headroom for K up to 512. Budget saved here can be reallocated to O-buffer for larger M or N.

2. **For workloads with M > 128, the O-buffer threshold scales linearly.** M=256 at N=128 needs 128 KB O-buffer. A 256×256 output needs 256 KB O-buffer (4× the default). Transformer FFN layers with d_model=4096 and K=16384 put massive pressure on O-buffer even with tiling.

3. **Double-buffering can mitigate but not eliminate the tiling cost.** With ping-pong O-buffers, DMA of tile N+1 can overlap with compute of tile N — hiding the A-reload latency. This would recover the 33% penalty at 32-56 KB and the 67-100% penalty at 16-24 KB, approximately halving the effective tiling overhead.

4. **FP16 output format would halve the O-buffer constraint.** If numerical requirements allow converting from FP32 accumulator to FP16 before writing to O-buffer, the tiling threshold drops from 64 KB to 32 KB for 128×128. This is a compiler optimization target — many inference workloads have converged gradients and can tolerate FP16 outputs.

## Methodology

Analytical cycle model using validated WS systolic formulas from `weight_stationary.c` and prior exploration validations (PE-array sweep confirmed at 256-bit bus within 0 cycles):

```
m_per_tile = floor(o_buf / (N × 4))
m_tiles = ceil(M / m_per_tile)

Per M-tile:
  w_bytes = m_chunk × K × 2        (W slice for this output row range)
  a_bytes = K × N × 2              (A reloaded per M-tile — consumed by compute)
  o_bytes = m_chunk × N × 4        (O slice for this output row range)
  
  compute = ceil(m_chunk / PE_ROWS) × ceil(N / PE_COLS) × K
  fill = PDEPTH × ceil(N / PE_COLS)
  drain = PDEPTH × ceil(m_chunk / PE_ROWS)

Total DMA = ceil(Σw_bytes / 32) + ceil(Σa_bytes / 32) + ceil(Σo_bytes / 32)
Total cycles = Σfill + Σcompute + Σdrain + DMA
TOPS = (M × N × K × 2) / total / 1000
```

**Conservative assumption (A-reload per M-tile):** The activation matrix is consumed by compute streaming and must be re-DMA'd for each M-tile. An optimistic model with A-buffer residency (no reload) would reduce total DMA to 192 KB regardless of M-tile count, making tiling overhead limited to fill/drain cycles only — a 3-4% effect rather than 8-22%.

## Next Exploration Candidates

1. **Double-buffer quantification:** How much of the tiling overhead does ping-pong buffering hide? If A-buffer has double-buffer, the A-reload for tile N+1 could complete during tile N's compute.
2. **Joint W+A+O sweep:** What if all three buffers are scaled down proportionally? A 32/32/32 KB configuration forces tiling in both M and K dimensions simultaneously.
3. **Larger workload scaling:** At what M×N does the O-buffer requirement exceed practical SRAM budgets (e.g., 128 KB, 256 KB, 512 KB), and what tiling factor results?
4. **K-sweep with constrained O-buffer:** How does the K crossover point (DMA→compute) shift when O-buffer is below the tiling threshold? Smaller K workloads are DMA-dominated already — tiling would compound this.
