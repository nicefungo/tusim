# Dataflow Comparison: WS vs OS for GEMM 128×128×K

> **Superseded for comparative evidence (2026-07-26):** this analytical study
> does not match the live per-K-tile dispatcher accounting and predates working
> config-to-runtime selection. See `dataflow-plugin-executable-reaudit.md`.

**Date:** 2026-06-04
**Question:** Does output-stationary dataflow reduce cycle count compared to weight-stationary for typical GEMM workloads? At what K does the choice matter?
**Hypothesis:** OS eliminates systolic fill/drain overhead (2×ceil(N/cols) + 2×ceil(M/rows) cycles), giving an advantage for small-K workloads where fill/drain is a larger fraction of total cycles.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| Dataflow | weight_stationary, output_stationary | Systolic vs vector |
| Workload | M=128, N=128, K={16,32,64,128,256,512,1024} | Varying inner dimension |
| PE array | 16×16 (default) | Sweet spot from previous exploration |
| Precision | FP16 input, FP32 accumulate | Default |
| Bus width | 32 B/cycle | Default |
| Pipeline depth | 2 | Systolic fill/drain (WS only) |

**Configs tested:** 14 (7 K values × 2 dataflows), analytical cycle model.

## Cycle Model Formulas

```
WS total = fill + compute + drain + dma
  fill   = pdepth × ceil(N / cols)
  compute = ceil(M / rows) × ceil(N / cols) × K
  drain  = pdepth × ceil(M / rows)

OS total = compute + dma
  compute = ceil(M / rows) × ceil(N / cols) × K   (same compute, no fill/drain)

dma = total_bytes / bus_width_bytes
total_bytes = (M×K + K×N + M×N) × 2   (FP16 W, A, O)
```

## Results Table

| K | WS Fill | WS Comp | WS Drain | DMA | WS Total | OS Total | WS Overhead | Speedup |
|---|---------|---------|----------|-----|----------|----------|-------------|---------|
| 16 | 16 | 1,024 | 16 | 1,280 | 2,336 | 2,304 | 3.1% | 1.01× |
| 32 | 16 | 2,048 | 16 | 1,536 | 3,616 | 3,584 | 1.6% | 1.01× |
| 64 | 16 | 4,096 | 16 | 2,048 | 6,176 | 6,144 | 0.8% | 1.00× |
| 128 | 16 | 8,192 | 16 | 3,072 | 11,296 | 11,264 | 0.4% | 1.00× |
| 256 | 16 | 16,384 | 16 | 5,120 | 21,536 | 21,504 | 0.2% | 1.00× |
| 512 | 16 | 32,768 | 16 | 9,216 | 42,016 | 41,984 | 0.1% | 1.00× |
| 1024 | 16 | 65,536 | 16 | 17,408 | 82,976 | 82,944 | 0.0% | 1.00× |

**Fixed overhead (fill+drain): 32 cycles for 16×16 PE.** This is a constant, not scaled by K.

## Full PE Array Sweep (K=256, both dataflows)

| PE_Row | PE_Col | WS Fill | WS Comp | WS Drain | DMA | WS Total | OS Total | Overhead |
|--------|--------|---------|---------|----------|-----|----------|----------|----------|
| 4 | 4 | 64 | 262,144 | 64 | 5,120 | 267,392 | 267,264 | 0.0% |
| 4 | 8 | 32 | 131,072 | 64 | 5,120 | 136,288 | 136,192 | 0.1% |
| 4 | 16 | 16 | 65,536 | 64 | 5,120 | 70,736 | 70,656 | 0.1% |
| 8 | 8 | 32 | 65,536 | 32 | 5,120 | 70,720 | 70,656 | 0.1% |
| 8 | 16 | 16 | 32,768 | 32 | 5,120 | 37,936 | 37,888 | 0.1% |
| 8 | 32 | 8 | 16,384 | 32 | 5,120 | 21,544 | 21,504 | 0.2% |
| 16 | 16 | 16 | 16,384 | 16 | 5,120 | 21,536 | 21,504 | 0.2% |
| 16 | 32 | 8 | 8,192 | 16 | 5,120 | 13,336 | 13,312 | 0.3% |
| 16 | 64 | 4 | 4,096 | 16 | 5,120 | 9,236 | 9,216 | 0.5% |
| 32 | 32 | 8 | 4,096 | 8 | 5,120 | 9,232 | 9,216 | 0.4% |
| 32 | 64 | 4 | 2,048 | 8 | 5,120 | 7,180 | 7,168 | 0.6% |
| 64 | 64 | 4 | 1,024 | 4 | 5,120 | 6,152 | 6,144 | 0.8% |
| 128 | 128 | 2 | 256 | 2 | 5,120 | 5,380 | 5,376 | 1.6% |

## Key Finding

**Dataflow choice (WS vs OS) has negligible impact on throughput for this workload class.** For K=256, the fill+drain overhead is only 0.2% of total cycles at 16×16 PE. Even at the extreme (128×128 PE array), overhead peaks at just 1.6%.

**Why this is counterintuitive:** The systolic fill/drain latency is a one-time cost per *tile*, while the K inner dimension creates many MAC operations per tile. The 2× ceil(N/cols) + 2× ceil(M/rows) cost is amortized across K cycles of compute. For any K ≥ 64, the overhead drops below 1%.

**When does dataflow choice matter?**
- **Small K (< 32):** Overhead reaches 3.1% at K=16. For depthwise convolutions (K=1) or attention score computation (K = head_dim, typically 64), WS could have 1-3% overhead.
- **Large M or N relative to PE:** More tiles → more fill/drain events → compound overhead. E.g., M=1024 with 16×16 PE = 64 tiles → 64×16 drain cycles = 1,024 cycles overhead.
- **Attention (Q·K^T):** M=seq_len, N=seq_len, K=head_dim — for short sequences (128) and small heads (64), fill/drain overhead could reach ~5% with WS.

## Actionable Conclusion

**Do not optimize for dataflow choice at this stage.** For the GEMM-heavy workloads typical of transformer FFN layers and convolutions, both WS and OS dataflows produce nearly identical throughput. The dominant bottleneck is **DMA bandwidth** (5,120 cycles for 160 KB at 32 B/cycle), not dataflow pipeline latency.

**Historical design implication (not revalidated by the live dispatcher):** Focus architecture exploration on:
1. **DMA bandwidth** — bus width, number of channels, double-buffered transfers
2. **SRAM sizing** — larger buffers reduce DMA transfer count for tiled workloads
3. **Tiled execution** — pipelining DMA and compute, not dataflow microarchitecture

Dataflow choice becomes relevant only for specific layer types (small-K GEMMs, attention with long sequences). Defer this decision until those workloads are measured on real traces.

## OS Mode Runtime Note

The TU cmodel's output-stationary plugin has an initialization segfault in the current build — OS cycle counts here are analytical. The formulas are validated for WS against the cmodel's actual output (previous exploration verified the WS formulas match the cmodel's perf report). OS formulas follow the same `ceil(M/rows) × ceil(N/cols) × K` compute model documented in `dataflow_interface.h`.

## Methodology

Cycle model: deterministic formulas from cmodel documentation. Fill = pdepth × ceil(N/cols), compute = ceil(M/rows) × ceil(N/cols) × K, drain = pdepth × ceil(M/rows). DMA = total_bytes / bus_width_bytes (ceiling). No stochastic variation.

## Next Exploration Candidates

1. **Bus width sweep:** What bus width shifts the knee from 16×16 to 32×32? (Directly addresses DMA bottleneck.)
2. **SRAM size sweep:** How does larger O-buffer reduce DMA transfer count for tiled outputs?
3. **Double-buffer benefit:** How many DMA cycles does ping-pong buffering hide?
4. **Workload shape sweep:** Vary M while keeping FLOPs constant — how does PE utilization change with matrix aspect ratio?
