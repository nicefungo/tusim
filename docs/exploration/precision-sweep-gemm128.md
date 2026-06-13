# Precision Sweep: FP16 vs BF16 vs INT8 vs FP8 vs TF32 for GEMM 128×128

**Date:** 2026-06-13
**Question:** How does numeric precision affect effective throughput (GFLOPS) for a fixed GEMM workload? When does precision choice matter, and when can it be deferred to accuracy considerations?
**Hypothesis:** Lower-precision types reduce DMA transfer cost (fewer bytes per element) but compute cost is identical across precisions. The benefit is largest at small K where DMA dominates total cycles. At large K, all precisions converge.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| Precision | FP16, BF16, INT8, FP8_E4M3, TF32 | W/A/O element sizes vary |
| PE array | 16×16 | Baseline sweet spot |
| Dataflow | weight_stationary | Systolic, pdepth=2 |
| Workload | M=128, N=128, K={16–4096} | GEMM, 9 values |
| Bus width | 256-bit (32 B/cycle) | Default |
| Clock | 1.0 GHz | Default |

**Configs tested:** 45 (5 precisions × 9 K values), analytical cycle model.

## Element Size Breakdown

| Precision | W (bytes) | A (bytes) | O (bytes) | FP32 accum |
|-----------|-----------|-----------|-----------|------------|
| FP16 | 2 | 2 | 2 | yes |
| BF16 | 2 | 2 | 2 | yes |
| INT8 | 1 | 1 | 4 | yes |
| FP8_E4M3 | 1 | 1 | 2 | yes |
| TF32 | 4 | 4 | 4 | no (native FP32) |

**Critical observation:** INT8's O-buffer stores FP32 accumulators (4 bytes), not INT8 (1 byte). The O-buffer is the largest fixed-size DMA transfer (M×N×4 = 64 KB for INT8 vs M×N×2 = 32 KB for FP16/FP8). This makes INT8 _slower_ than FP8 at small K despite lighter W/A.

## Cycle Model

```
total = fill + compute + drain + dma
fill   = pdepth × ceil(N / cols) = 2 × 8 = 16
compute = ceil(M/rows) × ceil(N/cols) × K = 8 × 8 × K = 64K
drain  = pdepth × ceil(M / rows) = 2 × 8 = 16
dma    = (W_bytes + A_bytes + O_bytes) / bus_width_bytes
```

## Results: Per-Precision Tables

### FP16 / BF16 (W=2B A=2B O=2B) — identical for both

| K | W+KB | A+KB | O+KB | DMA cyc | Comp cyc | Total cyc | GFLOPS | Util% | DMA% |
|---|------|------|------|---------|----------|-----------|--------|-------|------|
| 16 | 4 | 4 | 32 | 1,280 | 1,024 | 2,336 | 0.22 | 43.8 | 54.8 |
| 32 | 8 | 8 | 32 | 1,536 | 2,048 | 3,616 | 0.29 | 56.6 | 42.5 |
| 64 | 16 | 16 | 32 | 2,048 | 4,096 | 6,176 | 0.34 | 66.3 | 33.2 |
| 128 | 32 | 32 | 32 | 3,072 | 8,192 | 11,296 | 0.37 | 72.5 | 27.2 |
| 256 | 64 | 64 | 32 | 5,120 | 16,384 | 21,536 | 0.39 | 76.1 | 23.8 |
| 512 | 128 | 128 | 32 | 9,216 | 32,768 | 42,016 | 0.40 | 78.0 | 21.9 |
| 1024 | 256 | 256 | 32 | 17,408 | 65,536 | 82,976 | 0.40 | 79.0 | 21.0 |
| 2048 | 512 | 512 | 32 | 33,792 | 131,072 | 164,896 | 0.41 | 79.5 | 20.5 |
| 4096 | 1024 | 1024 | 32 | 66,560 | 262,144 | 328,736 | 0.41 | 79.7 | 20.2 |

### INT8 (W=1B A=1B O=4B)

| K | W+KB | A+KB | O+KB | DMA cyc | Comp cyc | Total cyc | GFLOPS | Util% | DMA% |
|---|------|------|------|---------|----------|-----------|--------|-------|------|
| 16 | 2 | 2 | 64 | 2,176 | 1,024 | 3,232 | 0.16 | 31.7 | 67.3 |
| 32 | 4 | 4 | 64 | 2,304 | 2,048 | 4,384 | 0.24 | 46.7 | 52.6 |
| 64 | 8 | 8 | 64 | 2,560 | 4,096 | 6,688 | 0.31 | 61.2 | 38.3 |
| 128 | 16 | 16 | 64 | 3,072 | 8,192 | 11,296 | 0.37 | 72.5 | 27.2 |
| 256 | 32 | 32 | 64 | 4,096 | 16,384 | 20,512 | 0.41 | 79.9 | 20.0 |
| 512 | 64 | 64 | 64 | 6,144 | 32,768 | 38,944 | 0.43 | 84.1 | 15.8 |
| 1024 | 128 | 128 | 64 | 10,240 | 65,536 | 75,808 | 0.44 | 86.4 | 13.5 |
| 2048 | 256 | 256 | 64 | 18,432 | 131,072 | 149,536 | 0.45 | 87.7 | 12.3 |
| 4096 | 512 | 512 | 64 | 34,816 | 262,144 | 296,992 | 0.45 | 88.3 | 11.7 |

### FP8_E4M3 (W=1B A=1B O=2B)

| K | W+KB | A+KB | O+KB | DMA cyc | Comp cyc | Total cyc | GFLOPS | Util% | DMA% |
|---|------|------|------|---------|----------|-----------|--------|-------|------|
| 16 | 2 | 2 | 32 | 1,152 | 1,024 | 2,208 | 0.24 | 46.4 | 52.2 |
| 32 | 4 | 4 | 32 | 1,280 | 2,048 | 3,360 | 0.31 | 61.0 | 38.1 |
| 64 | 8 | 8 | 32 | 1,536 | 4,096 | 5,664 | 0.37 | 72.3 | 27.1 |
| 128 | 16 | 16 | 32 | 2,048 | 8,192 | 10,272 | 0.41 | 79.8 | 19.9 |
| 256 | 32 | 32 | 32 | 3,072 | 16,384 | 19,488 | 0.43 | 84.1 | 15.8 |
| 512 | 64 | 64 | 32 | 5,120 | 32,768 | 37,920 | 0.44 | 86.4 | 13.5 |
| 1024 | 128 | 128 | 32 | 9,216 | 65,536 | 74,784 | 0.45 | 87.6 | 12.3 |
| 2048 | 256 | 256 | 32 | 17,408 | 131,072 | 148,512 | 0.45 | 88.3 | 11.7 |
| 4096 | 512 | 512 | 32 | 33,792 | 262,144 | 295,968 | 0.45 | 88.6 | 11.4 |

### TF32 (W=4B A=4B O=4B)

| K | W+KB | A+KB | O+KB | DMA cyc | Comp cyc | Total cyc | GFLOPS | Util% | DMA% |
|---|------|------|------|---------|----------|-----------|--------|-------|------|
| 16 | 8 | 8 | 64 | 2,560 | 1,024 | 3,616 | 0.14 | 28.3 | 70.8 |
| 32 | 16 | 16 | 64 | 3,072 | 2,048 | 5,152 | 0.20 | 39.8 | 59.6 |
| 64 | 32 | 32 | 64 | 4,096 | 4,096 | 8,224 | 0.26 | 49.8 | 49.8 |
| 128 | 64 | 64 | 64 | 6,144 | 8,192 | 14,368 | 0.29 | 57.0 | 42.8 |
| 256 | 128 | 128 | 64 | 10,240 | 16,384 | 26,656 | 0.31 | 61.5 | 38.4 |
| 512 | 256 | 256 | 64 | 18,432 | 32,768 | 51,232 | 0.33 | 64.0 | 36.0 |
| 1024 | 512 | 512 | 64 | 34,816 | 65,536 | 100,384 | 0.33 | 65.3 | 34.7 |
| 2048 | 1024 | 1024 | 64 | 67,584 | 131,072 | 198,688 | 0.34 | 66.0 | 34.0 |
| 4096 | 2048 | 2048 | 64 | 133,120 | 262,144 | 395,296 | 0.34 | 66.3 | 33.7 |

## Cross-Precision Comparison

| K | FP16 | BF16 | INT8 | FP8 | TF32 | Best | Spread% |
|---|---|---|---|---|---|---|---|
| 16 | 0.22 | 0.22 | 0.16 | **0.24** | 0.14 | FP8 | 63.8 |
| 32 | 0.29 | 0.29 | 0.24 | **0.31** | 0.20 | FP8 | 53.3 |
| 64 | 0.34 | 0.34 | 0.31 | **0.37** | 0.26 | FP8 | 45.2 |
| 128 | 0.37 | 0.37 | 0.37 | **0.41** | 0.29 | FP8 | 39.9 |
| 256 | 0.39 | 0.39 | 0.41 | **0.43** | 0.31 | FP8 | 36.8 |
| 512 | 0.40 | 0.40 | 0.43 | **0.44** | 0.33 | FP8 | 35.1 |
| 1024 | 0.40 | 0.40 | 0.44 | **0.45** | 0.33 | FP8 | 34.2 |
| 2048 | 0.41 | 0.41 | 0.45 | **0.45** | 0.34 | FP8 | 33.8 |
| 4096 | 0.41 | 0.41 | 0.45 | **0.45** | 0.34 | FP8 | 33.6 |

GFLOPS at 1 GHz, 16×16 PE. Peak = 0.51 GFLOPS.

## DMA vs Compute Breakdown at Extremes

| Precision | K=16 DMA% | K=16 Comp% | K=4096 DMA% | K=4096 Comp% |
|-----------|-----------|------------|-------------|--------------|
| FP16/BF16 | 54.8 | 43.8 | 20.2 | 79.7 |
| INT8 | 67.3 | 31.7 | 11.7 | 88.3 |
| FP8_E4M3 | 52.2 | 46.4 | 11.4 | 88.6 |
| TF32 | 70.8 | 28.3 | 33.7 | 66.3 |

## Key Finding

**The O-buffer byte size is the hidden lever.** INT8 has lighter W/A (1 byte each) but heavier O (4 bytes — FP32 accumulator), making it _slower_ than both FP16 and FP8 at K ≤ 128. The O-buffer DMA transfer is fixed at M×N×O_bytes regardless of K — for small K workloads, this fixed cost dominates the savings from lighter weights.

**FP8_E4M3 is the unambiguous throughput champion** at all K values — it combines 1-byte W/A with 2-byte O, minimizing DMA in both dimensions simultaneously.

**At K=16 (small inner product, typical of attention heads):**
- Precision spread = 63.8% — a 1.6× difference between FP8 (0.24 GFLOPS) and TF32 (0.14 GFLOPS)
- FP8 is 5.8% faster than FP16 (2,208 vs 2,336 cycles)
- INT8 is _slower_ than FP16 (3,232 vs 2,336 cycles) due to O-buffer DMA penalty
- TF32 is 54.8% slower than FP16

**At K=4096 (large inner product, typical of LLM FFN layers):**
- Precision spread = 33.6% — still meaningful but no longer a dominant factor
- FP8 delivers 11.4% DMA overhead vs 20.2% for FP16
- TF32 asymptotes at 33.7% DMA — 4-byte elements create an irreducible floor

**Precision choice is a small-K concern.** For training workloads (K > 256), precision choice should optimize for accuracy, not throughput. For edge inference with small inner dimensions, FP8 provides a meaningful throughput advantage over FP16 with no additional hardware complexity beyond the precision converters already present in the cmodel.

**Next exploration candidates:** Conv2D workloads (different DMA patterns), FP8 vs INT8 with quantized O-buffer (hypothetical), precision impact with double-buffering enabled (DMA/compute overlap masks precision differences).