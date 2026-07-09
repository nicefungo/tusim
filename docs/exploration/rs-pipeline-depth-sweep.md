# RS Pipeline Depth Sensitivity: Row-Stationary Fill/Drain Scaling

**Date:** 2026-07-09
**Question:** How does the row-stationary dataflow scale with systolic pipeline depth? Does its (pdepth-1) fill/drain reduction provide a meaningful advantage over WS, or is it marginal?
**Hypothesis:** RS's (pdepth-1) fill/drain formula produces a constant absolute savings vs WS regardless of pdepth — the advantage is fixed, not proportional. At shallow pipeline depths, RS approaches OS performance (0.3% overhead at pdepth=1). At deep pipelines, all systolic dataflows are dominated by fill/drain and the RS advantage narrows proportionally.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| Pipeline depth | 1, 2, 4, 8 | Systolic MAC pipeline stages |
| Dataflow | WS, RS, OS | Three registered plugins |
| PE array | 16×16 | Baseline sweet spot |
| Workload | M=128, N=128, K=256 | 8.39 MFLOPs GEMM, 64 spatial tiles |
| Bus width | 256-bit (32 B/cycle) | Default |
| Precision | FP16 W/A, FP32 accumulate | Default |

**Configs tested:** 12 points (3 dataflows × 4 pipeline depths). Analytical cycle model using the per-tile formula validated in `test_dataflow_sweep` at pdepth=2. DMA = 6,144 cycles (196,608 bytes / 32 B/cycle).

## Cycle Model

```
Spatial tiles: 8 M-tiles × 8 N-tiles = 64 (exact multiples, no edge tiles)
nc = 16, mc = 16, K = 256, DMA = 6,144

Per spatial tile:
  WS: pd*nc + K + pd*mc   = 32*pd + 256
  RS: (pd-1)*nc + 1 + K + (pd-1)*mc = 32*(pd-1) + 257
  OS: K = 256

Total = 64 × per_tile + DMA
```

**Validated against:** `test-dataflow-sweep` cmodel output at pdepth=2: WS=20kCyc syst + 6k DMA = 26k, RS=18k + 6k = 24k, OS=16k + 6k = 22k. Analytical model matches within rounding.

## Results

### Full Matrix (Total Cycles Including DMA)

| DF | pdepth | Fill | Drain | Compute | DMA | Total | Overhead vs OS | TOPS | Util% |
|----|--------|------|-------|---------|-----|-------|----------------|------|-------|
| WS | 1 | 1,024 | 1,024 | 16,384 | 6,144 | 24,576 | +9.1% | 0.341 | 66.7 |
| WS | 2 | 2,048 | 2,048 | 16,384 | 6,144 | 26,624 | +18.2% | 0.315 | 61.5 |
| WS | 4 | 4,096 | 4,096 | 16,384 | 6,144 | 30,720 | +36.4% | 0.273 | 53.3 |
| WS | 8 | 8,192 | 8,192 | 16,384 | 6,144 | 38,912 | +72.7% | 0.216 | 42.1 |
| **RS** | **1** | **64** | **0** | **16,384** | **6,144** | **22,592** | **+0.3%** | **0.371** | **72.5** |
| RS | 2 | 1,088 | 1,024 | 16,384 | 6,144 | 24,640 | +9.4% | 0.340 | 66.5 |
| RS | 4 | 3,136 | 3,072 | 16,384 | 6,144 | 28,736 | +27.6% | 0.292 | 57.0 |
| RS | 8 | 7,232 | 7,168 | 16,384 | 6,144 | 36,928 | +63.9% | 0.227 | 44.4 |
| **OS** | **any** | **0** | **0** | **16,384** | **6,144** | **22,528** | **0.0%** | **0.372** | **72.7** |

### RS vs WS: Fill/Drain Savings by Pipeline Depth

| pdepth | WS Fill+Drain | RS Fill+Drain | Savings | Savings % of WS Overhead |
|--------|---------------|---------------|---------|--------------------------|
| 1 | 2,048 | 64 | 1,984 | 96.9% |
| 2 | 4,096 | 2,112 | 1,984 | 48.4% |
| 4 | 8,192 | 6,208 | 1,984 | 24.2% |
| 8 | 16,384 | 14,400 | 1,984 | 12.1% |

### Throughput vs Pipeline Depth (TOPS)

| Dataflow | pd=1 | pd=2 | pd=4 | pd=8 | Loss (1→8) |
|----------|------|------|------|------|------------|
| WS | 0.341 | 0.315 | 0.273 | 0.216 | -36.7% |
| RS | 0.371 | 0.340 | 0.292 | 0.227 | -38.8% |
| OS | 0.372 | 0.372 | 0.372 | 0.372 | 0.0% |

## Key Findings

### 1. RS savings vs WS are ABSOLUTE and CONSTANT — 1,984 cycles regardless of pdepth

The RS formula `(pd-1)*nc + 1` for fill and `(pd-1)*mc` for drain means the fill savings per tile vs WS (`pd*nc`) is exactly `nc - 1 = 15` cycles per tile, and drain savings is `mc = 16` cycles per tile. Across 64 tiles, the total fill savings is 64 × 15 = 960 cycles and drain savings is 64 × 16 = 1,024 cycles, for a constant total of **1,984 cycles** at every pipeline depth.

This is a structural property of the (pd-1) formula, not an approximation. The absolute savings do not grow with pdepth — they're fixed at fabrication time.

### 2. At pdepth=1, RS is a near-perfect systolic approximation of OS

At pdepth=1, RS has exactly 1 cycle of fill per tile (64 cycles total across 64 tiles) and zero drain. This is only **0.3% overhead** vs OS (22,592 vs 22,528 cycles). RS at pdepth=1 delivers 0.371 TOPS vs 0.372 TOPS for OS — effectively identical throughput.

For a hardware team that needs systolic regularity (reduced wiring, simpler clock distribution) but wants OS-level throughput, **RS at pdepth=1 is the optimal systolic configuration**.

### 3. The proportional RS advantage erodes with pipeline depth

As pdepth increases, fill/drain dominates total cycles for both WS and RS. The constant 1,984-cycle savings becomes a shrinking fraction of total overhead:

- At pdepth=1: RS saves 96.9% of WS overhead (RS is nearly OS)
- At pdepth=2: RS saves 48.4% of WS overhead (meaningful but not transformative)
- At pdepth=8: RS saves only 12.1% (both systolic dataflows are crushed by fill/drain)

This means **RS is most valuable in shallow-pipeline designs**. For deep pipelines (pdepth ≥ 4), the choice between WS and RS is secondary to the fundamental systolic-vs-vector decision.

### 4. The pipeline depth scalability ranking is invariant

For all pipeline depths 1-8: **OS > RS > WS**. The ordering never flips. RS is strictly between OS and WS at every pdepth. The only question is the magnitude of the gap, not the ranking.

## Actionable Conclusions

1. **For shallow pipelines (pdepth ≤ 2):** RS is the preferred systolic dataflow. At pdepth=1, it matches OS throughput (0.3% gap) while preserving systolic regularity. At pdepth=2, it cuts WS overhead nearly in half.

2. **For deep pipelines (pdepth ≥ 4):** The RS advantage shrinks to 12-24% of WS overhead. At this point, the dataflow choice matters less than the pipeline depth itself — OS is 28-73% faster than either systolic option. **If pdepth cannot be kept shallow, switch to OS.**

3. **RS's constant savings model generalizes:** For any PE array size and tile count, RS saves exactly `(nc + mc - 1) × ntiles` cycles vs WS regardless of pdepth. This is a hardware invariant that simplifies architecture trade-off analysis.

## Methodology

Analytical cycle model using the per-tile formula from `tests/test_dataflow_sweep.c` line 140:
```c
rs_cyc += ((pd - 1) * nc + 1) + K_WORKLOAD + ((pd - 1) * mc);
```
Validated at pdepth=2 against the cmodel sweep test (RS=18k systolic, 24k total w/ DMA — model matches within rounding). All other pdepth values are analytical extrapolations of this validated formula. Since the formula is structural (arithmetic, not empirical), no additional validation points are needed.

## Prior Work

- `dataflow-rs-comparison-gemm128.md` (2026-06-10): Established RS model at pdepth=2, identified (pd-1) fill/drain advantage
- `pipeline-depth-dataflow-interaction.md` (2026-06-23): Explored WS vs OS pipeline depth sensitivity but did not include RS
- `dataflow-comparison-gemm128.md` (2026-06-04): Original WS vs OS comparison

This exploration fills the gap between the RS comparison (single pdepth point) and the pipeline-depth interaction (WS/OS only), completing the three-dataflow pipeline-depth sensitivity matrix.
