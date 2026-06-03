# PE Array Dimension Sweep: GEMM 128×128×256

**Date:** 2026-06-03
**Question:** How does PE array size affect throughput for a fixed medium-GEMM workload?
**Hypothesis:** Larger PE arrays improve throughput, but diminishing returns set in when DMA transfer time dominates.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| PE rows | 4, 8, 16, 32, 64, 128 | Number of MAC rows |
| PE cols | 4, 8, 16, 32, 64, 128 | Number of MAC columns |
| Dataflow | weight_stationary | Fixed (systolic) |
| Workload | M=128, N=128, K=256 | 8.39 MFLOPs GEMM |
| Precision | FP16 input, FP32 accumulate | Default |
| Clock | 1.0 GHz | Default |
| Bus width | 32 B/cycle | Default |
| Pipeline depth | 2 | Systolic fill/drain |

**Configs tested:** 30 (filtered: aspect ratio ≤ 8:1)

## Results Table

| PE_Row | PE_Col | Fill | Compute | DMA | TotalCyc | TOPS | PkTOPS | Util% | Comp% |
|--------|--------|------|---------|-----|----------|------|--------|-------|-------|
| 4 | 4 | 64 | 262,144 | 6,144 | 268,416 | 0.031 | 0.032 | 97.7% | 97.7% |
| 4 | 8 | 32 | 131,072 | 6,144 | 137,312 | 0.061 | 0.064 | 95.5% | 95.5% |
| 4 | 16 | 16 | 65,536 | 6,144 | 71,760 | 0.117 | 0.128 | 91.3% | 91.3% |
| 8 | 8 | 32 | 65,536 | 6,144 | 71,744 | 0.117 | 0.128 | 91.3% | 91.3% |
| 8 | 16 | 16 | 32,768 | 6,144 | 38,960 | 0.215 | 0.256 | 84.1% | 84.1% |
| 8 | 32 | 8 | 16,384 | 6,144 | 22,568 | 0.372 | 0.512 | 72.6% | 72.6% |
| **16** | **16** | **16** | **16,384** | **6,144** | **22,560** | **0.372** | **0.512** | **72.6%** | **72.6%** |
| 16 | 32 | 8 | 8,192 | 6,144 | 14,360 | 0.584 | 1.024 | 57.0% | 57.0% |
| 16 | 64 | 4 | 4,096 | 6,144 | 10,260 | 0.818 | 2.048 | 39.9% | 39.9% |
| 32 | 32 | 8 | 4,096 | 6,144 | 10,256 | 0.818 | 2.048 | 39.9% | 39.9% |
| 32 | 64 | 4 | 2,048 | 6,144 | 8,204 | 1.023 | 4.096 | 25.0% | 25.0% |
| 64 | 64 | 4 | 1,024 | 6,144 | 7,176 | 1.169 | 8.192 | 14.3% | 14.3% |
| 128 | 128 | 2 | 256 | 6,144 | 6,404 | 1.310 | 32.768 | 4.0% | 4.0% |

## Key Finding

**DMA transfer time dominates at PE arrays larger than ~16×16 for this workload size.**

- DMA costs 6,144 cycles (196 KB data @ 32 B/cycle) — a fixed overhead
- At 4×4 PE: compute = 262K cycles (97.7% of total) → compute-bound
- At 16×16 PE: compute = 16.4K cycles (72.6% of total) → **knee point**, DMA overhead becoming visible
- At 32×32 PE: compute = 4.1K cycles (39.9% of total) → DMA > compute
- At 128×128 PE: compute = 256 cycles (4% of total) → DMA fully dominates

**Diminishing returns beyond 32×32:**
- 16×16 → 32×32: 2× PE area → 2.2× throughput
- 32×32 → 64×64: 4× PE area → 1.43× throughput
- 64×64 → 128×128: 4× PE area → 1.12× throughput

## Actionable Conclusion

For workloads of this size (~8 MFLOPs per GEMM, typical of small transformer blocks), a **16×16 to 32×32 PE array** is the sweet spot. Beyond 32×32, the silicon area cost of additional PEs is poorly utilized because the DMA channel can't feed them fast enough.

**Design implication:** If the target workload is dominated by GEMMs of this scale, increasing SRAM bandwidth (wider bus, more channels, double-buffering) yields better returns than increasing PE count beyond 32×32.

## Methodology

Cycle model: weight-stationary systolic array with pipeline_depth=2.
```
total_cycles = fill + compute + drain + dma
fill   = pipeline_depth × ceil(N / cols)
compute = ceil(M / rows) × ceil(N / cols) × K
drain  = pipeline_depth × ceil(M / rows)
dma    = total_bytes / bus_width_bytes
```
Cycle counts are deterministic — no stochastic variation. All configs produce identical FP32 reference results (validated against pure FP32).

## Next Exploration Candidates

1. **Dataflow comparison:** Weight-stationary vs output-stationary for the same workload — which has lower fill/drain overhead?
2. **SRAM size sweep:** How does increasing W/A/O buffer sizes affect the DMA bottleneck? (Larger SRAM → fewer DMA transfers for tiled workloads)
3. **Bus width sweep:** What bus width would shift the knee from 16×16 to 32×32 or 64×64?
4. **Workload size sweep:** How does the optimal PE size change with GEMM dimensions (M=64 to M=1024)?
