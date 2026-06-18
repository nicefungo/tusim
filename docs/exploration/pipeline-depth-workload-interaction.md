# Pipeline Depth × Workload Size: How Penalty Scales with Problem Size

**Date:** 2026-06-18
**Question:** Does the pdepth penalty (fill/drain overhead) scale with workload size, or is it a fixed-cycle tax? At what workload size does pdepth become negligible?
**Hypothesis:** The pdepth penalty is per-spatial-tile — it scales with `ceil(M/pe_r) × ceil(N/pe_c)`, not with K. Small workloads (fewer K iterations per tile) feel the penalty more because fill/drain is a larger fraction of total compute. Large K workloads amortize the overhead.

## Config Matrix

| Parameter | Values | Description |
|---|---|---|
| Pipeline depth | 1, 2, 4, 8 | Systolic MAC pipeline stages |
| Workload | 64×64×64, 128×128×256, 512×512×1024 | Attention head → LLM FFN |
| PE array | 16×16 | Sweet-spot baseline |
| Dataflow | weight_stationary | Systolic, W preloaded in PEs |
| Bus width | 256-bit (32 B/cycle) | Default |
| Clock | 1.0 GHz | Default |
| Precision | FP16 W/A, FP32 O | Default |

**Configs tested:** 12 (4 pdepth values × 3 workloads), analytical cycle model. Medium workload (128³×256) cross-referenced from prior `pipeline-depth-sweep-gemm128.md`.

## Cycle Model

```
spatial_tiles = ceil(M/pe_rows) × ceil(N/pe_cols)
per_tile      = pd × (pe_rows + pe_cols) + K
total_compute = spatial_tiles × per_tile
DMA           = ceil((M×K×2 + K×N×2 + M×N×4) / 32)
total_cycles  = total_compute + DMA
```

For 16×16 PE: `pe_rows = pe_cols = 16`, so `pd × (16+16) = pd × 32` fill+drain per tile.

## Results Table

### Small Workload: 64×64×64 (0.5M FLOPs, attention head)

spatial_tiles = 4×4 = 16, DMA = 1,024 cycles

| pd | Fill+Drain/Tile | Per Tile | Compute | DMA | Total Cycles | TOPS | Peak TOPS | Util% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 96 | 1,536 | 1,024 | 2,560 | 0.205 | 0.512 | **40.0%** |
| 2 | 64 | 128 | 2,048 | 1,024 | 3,072 | 0.171 | 0.512 | **33.3%** |
| 4 | 128 | 192 | 3,072 | 1,024 | 4,096 | 0.128 | 0.512 | **25.0%** |
| 8 | 256 | 320 | 5,120 | 1,024 | 6,144 | 0.085 | 0.512 | **16.7%** |

**pd=1→8 efficiency loss:** 58.2% (util drops from 40.0% → 16.7%)

### Medium Workload: 128×128×256 (8.4M FLOPs)

spatial_tiles = 8×8 = 64, DMA = 6,144 cycles *(from `pipeline-depth-sweep-gemm128.md`)*

| pd | Fill+Drain/Tile | Per Tile | Compute | DMA | Total Cycles | TOPS | Peak TOPS | Util% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 288 | 18,432 | 6,144 | 24,576 | 0.341 | 0.512 | **66.7%** |
| 2 | 64 | 320 | 20,480 | 6,144 | 26,624 | 0.315 | 0.512 | **61.5%** |
| 4 | 128 | 384 | 24,576 | 6,144 | 30,720 | 0.273 | 0.512 | **53.3%** |
| 8 | 256 | 512 | 32,768 | 6,144 | 38,912 | 0.216 | 0.512 | **42.1%** |

**pd=1→8 efficiency loss:** 36.9% (util drops from 66.7% → 42.1%)

### Large Workload: 512×512×1024 (537M FLOPs, LLM FFN layer)

spatial_tiles = 32×32 = 1,024, DMA = 98,304 cycles

| pd | Fill+Drain/Tile | Per Tile | Compute | DMA | Total Cycles | TOPS | Peak TOPS | Util% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1,056 | 1,081,344 | 98,304 | 1,179,648 | 0.455 | 0.512 | **88.9%** |
| 2 | 64 | 1,088 | 1,114,112 | 98,304 | 1,212,416 | 0.443 | 0.512 | **86.5%** |
| 4 | 128 | 1,152 | 1,179,648 | 98,304 | 1,277,952 | 0.420 | 0.512 | **82.1%** |
| 8 | 256 | 1,280 | 1,310,720 | 98,304 | 1,409,024 | 0.381 | 0.512 | **74.4%** |

**pd=1→8 efficiency loss:** 16.3% (util drops from 88.9% → 74.4%)

## Key Findings

### 1. Pdepth penalty is a fixed per-tile tax — amortized by K, not by M×N

The fill+drain cost is `pd × (pe_rows + pe_cols)` per spatial tile, independent of K. This creates a stark workload-size effect:

| Workload | K | Compute/Tile | FD/Tile (pd=1) | FD% of Tile | Util Drop pd=1→8 |
|---|---|---|---|---|---|
| 64×64×64 | 64 | 64 | 32 | 33.3% | −58.2% |
| 128×128×256 | 256 | 256 | 32 | 11.1% | −36.9% |
| 512×512×1024 | 1024 | 1024 | 32 | 3.0% | −16.3% |

The fill+drain fraction goes as `pd × 32 / (pd × 32 + K)`. For small K, fill/drain dominates; for large K, it vanishes.

### 2. The "when does pdepth matter" threshold

Setting a 5% utilization threshold (i.e., pdepth costs ≤5% of compute cycles):

```
pd × 32 / K ≤ 0.05  →  K ≥ pd × 640
```

| pd | Minimum K for ≤5% overhead |
|----|---------------------------|
| 1 | 640 |
| 2 | 1,280 |
| 4 | 2,560 |
| 8 | 5,120 |

For attention head projections (K=64–128), even pd=1 already costs 25–33% overhead. For LLM FFN layers (K=4096+), pd=8 costs only ~6.25% — negligible.

### 3. DMA dominance shifts with workload size

DMA as fraction of total cycles:

| Workload | DMA Cycles | DMA % (pd=1) | DMA % (pd=8) |
|---|---|---|---|
| 64×64×64 | 1,024 | 40.0% | 16.7% |
| 128×128×256 | 6,144 | 25.0% | 15.8% |
| 512×512×1024 | 98,304 | 8.3% | 7.0% |

For small workloads, DMA is the dominant cost. For large workloads, compute dominates and DMA becomes a rounding error. This flips the optimization target: for attention heads, optimize DMA bandwidth; for LLM layers, optimize MAC throughput.

### 4. Interaction with PE array sizing

Prior explorations showed smaller PE arrays achieve higher utilization. This pdepth analysis adds nuance: for small-K workloads, a smaller PE array (8×8) has more spatial tiles but each tile has the same K — the pdepth cost compounds across more tiles. The optimal PE array for small workloads is a trade-off between fewer tiles (less total fill/drain) and lower peak MACs (lower TOPS ceiling).

## Implications

- **For attention-heavy workloads** (small K): prefer pd=1 or pd=2. Deep pipelines (pd=4+) waste >25% of cycles on fill/drain.
- **For FFN/projection-heavy workloads** (large K): pipeline depth is nearly free. pd=8 is fine.
- **Config-driven hardware** should expose pdepth as a runtime register, not a compile-time constant — different model layers benefit from different pipeline depths.
