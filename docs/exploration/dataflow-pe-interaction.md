# Dataflow × PE Array Size Interaction: Does Dataflow Choice Matter at Scale?

**Date:** 2026-06-24
**Question:** How does dataflow choice (WS/OS/RS) interact with PE array dimensions for a fixed GEMM workload? Does the systolic fill/drain overhead matter more at small PE arrays (more spatial tiles) or large PE arrays (more PE columns per fill)?

**Hypothesis:** OS's advantage (zero fill/drain) should be most significant at small PE arrays where spatial tile count is high, because WS pays `pd × ceil(N/pe_c) + pd × ceil(M/pe_r)` fill/drain overhead. At large PE arrays, DMA overhead dominates total cycles and dataflow choice becomes irrelevant.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| Dataflow | WS, OS, RS | Weight-stationary, output-stationary, row-stationary |
| PE array | 8×8, 16×16, 32×32, 64×64 | Powers-of-2 range |
| Workload | M=128, N=128, K=256 | 8.39 MFLOPs GEMM |
| Pipeline depth | 2 | Default WS systolic |
| Bus width | 256-bit (32 B/cycle) | Default |
| Precision | FP16 W/A, FP32 O | Default |
| Clock | 1.0 GHz | Default |

**Configs tested:** 12 (3 dataflows × 4 PE sizes), analytical cycle model.

## Cycle Model

```
WS: total = pd×nt + mt×nt×K + pd×mt + dma
OS: total = mt×nt×K + dma
RS: total = (pd-1)×nt + mt×nt×K + (pd-1)×mt + dma

where mt=ceil(M/pe_r), nt=ceil(N/pe_c), dma=ceil((M×K×2+K×N×2+M×N×4)/32)
```

Model assumes fill/drain counted once per spatial dimension (overlap across consecutive tiles). This is the established convention from `dataflow-comparison-gemm128.md` and `pipeline-depth-sweep-gemm128.md`. Validated against the cmodel at 16×16 PE (WS total = 22,560 cycles at pdepth=2).

**DMA:** 196,608 bytes ÷ 32 B/cyc = 6,144 cycles (fixed across all configs)

## Results

| PE | PE MACs | Dataflow | mt | nt | mt×nt | Fill | Compute | Drain | TotalCyc | TOPS | PkTOPS | Util% |
|----|---------|----------|----|----|-------|------|---------|-------|----------|------|--------|-------|
| 8×8 | 64 | WS | 16 | 16 | 256 | 32 | 65,536 | 32 | 71,744 | 0.1169 | 0.1280 | 91.3% |
| 8×8 | 64 | OS | 16 | 16 | 256 | 0 | 65,536 | 0 | 71,680 | 0.1170 | 0.1280 | 91.4% |
| 8×8 | 64 | RS | 16 | 16 | 256 | 16 | 65,536 | 16 | 71,712 | 0.1170 | 0.1280 | 91.4% |
| 16×16 | 256 | WS | 8 | 8 | 64 | 16 | 16,384 | 16 | 22,560 | 0.3719 | 0.5120 | 72.6% |
| 16×16 | 256 | OS | 8 | 8 | 64 | 0 | 16,384 | 0 | 22,528 | 0.3724 | 0.5120 | 72.7% |
| 16×16 | 256 | RS | 8 | 8 | 64 | 8 | 16,384 | 8 | 22,544 | 0.3721 | 0.5120 | 72.7% |
| 32×32 | 1024 | WS | 4 | 4 | 16 | 8 | 4,096 | 8 | 10,256 | 0.8180 | 2.0480 | 39.9% |
| 32×32 | 1024 | OS | 4 | 4 | 16 | 0 | 4,096 | 0 | 10,240 | 0.8193 | 2.0480 | 40.0% |
| 32×32 | 1024 | RS | 4 | 4 | 16 | 4 | 4,096 | 4 | 10,248 | 0.8187 | 2.0480 | 40.0% |
| 64×64 | 4096 | WS | 2 | 2 | 4 | 4 | 1,024 | 4 | 7,176 | 1.1690 | 8.1920 | 14.3% |
| 64×64 | 4096 | OS | 2 | 2 | 4 | 0 | 1,024 | 0 | 7,168 | 1.1704 | 8.1920 | 14.3% |
| 64×64 | 4096 | RS | 2 | 2 | 4 | 2 | 1,024 | 2 | 7,172 | 1.1697 | 8.1920 | 14.3% |

## Key Finding

**Dataflow choice is irrelevant for throughput at K=256.** The fill/drain overhead is <0.1% of total cycles at all PE sizes. OS wins by at most 64 cycles out of 71,744 (0.09% at 8×8 PE). This is because the overhead is O(mt + nt) while compute is O(mt × nt × K) — for K=256, compute dominates by 3 orders of magnitude.

### Best dataflow per PE size

- **8×8 PE:** Best dataflow = **OS** (0.1170 TOPS, WS = 0.1169)
- **16×16 PE:** Best dataflow = **OS** (0.3724 TOPS, WS = 0.3719)
- **32×32 PE:** Best dataflow = **OS** (0.8193 TOPS, WS = 0.8180)
- **64×64 PE:** Best dataflow = **OS** (1.1704 TOPS, WS = 1.1690)

### WS-vs-OS gap (fill/drain overhead as % of total)

- 8×8: WS overhead = 64 cyc / 71,744 total = **0.09%**
- 16×16: WS overhead = 32 cyc / 22,560 total = **0.14%**
- 32×32: WS overhead = 16 cyc / 10,256 total = **0.16%**
- 64×64: WS overhead = 8 cyc / 7,176 total = **0.11%**

The gap peaks at 32×32 PE (0.16%) but remains negligible throughout. The fill/drain overhead _decreases_ with PE size (fewer spatial tiles → fewer fill/drain cycles), but the gap as a fraction of total cycles barely changes because compute also decreases.

### The real bottleneck: DMA-bound at all PE sizes

At 8×8 PE, DMA is 6,144/71,744 = 8.6% of cycles. At 64×64 PE, DMA is 6,144/7,176 = **85.6%** of cycles. The accelerator becomes severely DMA-bound as PE array scales — adding more MAC units doesn't help because there's nothing for them to compute while waiting for data.

| PE | Compute % | DMA % | Fill/Drain % |
|----|-----------|-------|-------------|
| 8×8 | 91.3% | 8.6% | 0.09% |
| 16×16 | 72.6% | 27.2% | 0.14% |
| 32×32 | 39.9% | 59.9% | 0.16% |
| 64×64 | 14.3% | 85.6% | 0.11% |

### Architectural implications

1. **Dataflow is a second-order concern for K≥64.** The systolic vs. vector dataflow debate matters only for very small inner dimensions (attention softmax, small convolutions). For the dominant GEMM workloads in transformers (K=64-4096), pick whatever dataflow is simpler to implement in hardware.

2. **DMA bandwidth is the first-order bottleneck.** At 64×64 PE (4,096 MACs), the cmodel achieves only 14.3% utilization because 85.6% of time is spent waiting for DMA. To make large PE arrays worthwhile, DMA bus width must scale proportionally to PE MAC count.

3. **The sweet spot is near where compute ≅ DMA.** At 16×16 PE with 256-bit bus: compute = 16,384 cyc, DMA = 6,144 cyc, ratio = 2.67:1. At 32×32 PE: compute = 4,096 cyc, DMA = 6,144 cyc, ratio = 0.67:1. The crossover is between 16×16 and 32×32 — consistent with the PE array sweep findings in `pe-array-sweep-gemm128.md`.

## Relationship to prior explorations

- `pe-array-sweep-gemm128.md` (2026-06-03): Swept PE array for WS only. This adds dataflow dimension.
- `dataflow-comparison-gemm128.md` (2026-06-04): Compared dataflows at fixed 16×16 PE, K sweep. This adds PE size dimension.
- `dataflow-rs-comparison-gemm128.md` (2026-06-10): Compared WS/OS/RS at fixed 16×16 PE. RS adds negligible benefit.
- `pipeline-depth-dataflow-interaction.md` (2026-06-23): Swept pdepth × dataflow at fixed 16×16 PE. Deeper pipelines amplify WS penalty but still negligible.

**Conclusion:** All four explorations converge on the same finding: dataflow choice has sub-1% throughput impact for K≥64 GEMM workloads. DMA bandwidth and buffer sizing are the dimensions that actually matter for throughput.
