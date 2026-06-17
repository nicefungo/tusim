# Pipeline Depth Sweep: GEMM 128×128×256

**Date:** 2026-06-17
**Question:** How does systolic pipeline depth (pdepth) affect throughput and compute utilization across PE array sizes? At what pdepth does fill/drain overhead become negligible relative to compute?
**Hypothesis:** Larger pdepth increases fill/drain overhead linearly. Small PE arrays (more spatial tiles) feel the impact more strongly because they pay fill/drain per spatial tile. At very small K or large PE arrays, pdepth becomes the dominant overhead.

## Config Matrix

| Parameter | Values | Description |
|---|---|---|
| Pipeline depth | 1, 2, 4, 8 | Systolic MAC pipeline stages |
| PE array | 8×8, 16×16, 32×32 | Three key array sizes |
| Dataflow | weight_stationary | Systolic, W preloaded in PEs |
| Workload | M=128, N=128, K=256 | 8.39 MFLOPs GEMM |
| Bus width | 256-bit (32 B/cycle) | Default |
| Clock | 1.0 GHz | Default |
| Precision | FP16 W/A, FP32 O | Default |

**Configs tested:** 12 (4 pdepth values × 3 PE arrays), analytical cycle model.

## Cycle Model

```
Per spatial tile (M-tile × N-tile):
  fill   = pdepth × tile_n             # pipeline fill — once per spatial tile
  drain  = pdepth × tile_m             # pipeline drain — once per spatial tile
Per K-tile within spatial tile:
  compute = tile_k                     # 1 MAC/cycle/PE after pipeline full
Total per spatial tile:
  fill + drain + k_tiles × tile_k
Total:
  spatial_tiles × (pdepth × (tile_m+tile_n) + K)  +  dma
```

Where spatial_tiles = ceil(M/pe_rows) × ceil(N/pe_cols), k_tiles = ceil(K/16), and
dma = ceil((M×K×2 + K×N×2 + M×N×4) / 32) = 6,144 cycles.

## Results Table

| PE | pdepth | Sp.Tiles | Fill+Drain/Tile | FD Total | Comp Total | DMA | Total Cyc | TOPS | Peak TOPS | Util% |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8×8 | 1 | 256 | 16 | 4,096 | 65,536 | 6,144 | 75,776 | 0.1107 | 0.128 | 86.5% |
| 8×8 | 2 | 256 | 32 | 8,192 | 65,536 | 6,144 | 79,872 | 0.1050 | 0.128 | 82.1% |
| 8×8 | 4 | 256 | 64 | 16,384 | 65,536 | 6,144 | 88,064 | 0.0953 | 0.128 | 74.4% |
| 8×8 | 8 | 256 | 128 | 32,768 | 65,536 | 6,144 | 104,448 | 0.0803 | 0.128 | 62.7% |
| 16×16 | 1 | 64 | 32 | 2,048 | 16,384 | 6,144 | 24,576 | 0.3413 | 0.512 | 66.7% |
| 16×16 | 2 | 64 | 64 | 4,096 | 16,384 | 6,144 | 26,624 | 0.3151 | 0.512 | 61.5% |
| 16×16 | 4 | 64 | 128 | 8,192 | 16,384 | 6,144 | 30,720 | 0.2731 | 0.512 | 53.3% |
| 16×16 | 8 | 64 | 256 | 16,384 | 16,384 | 6,144 | 38,912 | 0.2156 | 0.512 | 42.1% |
| 32×32 | 1 | 16 | 64 | 1,024 | 4,096 | 6,144 | 11,264 | 0.7447 | 2.048 | 36.4% |
| 32×32 | 2 | 16 | 128 | 2,048 | 4,096 | 6,144 | 12,288 | 0.6827 | 2.048 | 33.3% |
| 32×32 | 4 | 16 | 256 | 4,096 | 4,096 | 6,144 | 14,336 | 0.5851 | 2.048 | 28.6% |
| 32×32 | 8 | 16 | 512 | 8,192 | 4,096 | 6,144 | 18,432 | 0.4551 | 2.048 | 22.2% |

## Key Findings

### 1. Pipeline depth hits small PE arrays harder in absolute terms

At 8×8 PE (64 MACs), going from pd=1→8 drops utilization from 86.5% to 62.7% — a 23.8pp loss. At 32×32 PE (1024 MACs), the same change drops utilization from 36.4% to 22.2% — a 14.2pp loss. Small arrays generate more spatial tiles (256 for 8×8 vs 16 for 32×32), and each tile pays the fill/drain tax. Pdepth multiplies that tax.

However, in *relative* terms, 32×32 loses 39% of its utilization (36.4→22.2) while 8×8 loses only 28% (86.5→62.7). The larger array is more DMA-bound, so fill/drain is a smaller fraction of the bottleneck — but pdepth still matters.

### 2. Default pdepth=2 is a reasonable compromise

At the 16×16 PE sweet spot (pd=2), fill/drain is 64 cycles per spatial tile vs 256 compute cycles — 20% overhead. Going to pd=1 saves 32 cycles per tile (7.7% improvement in total cycles), but requires a shallower pipeline that may limit clock frequency in real hardware. Going to pd=4 costs an extra 32 cycles per tile with no benefit unless deeper pipelining enables higher clock.

### 3. Marginal cost of pdepth is roughly constant per doubling

| PE | pd1→2 | pd2→4 | pd4→8 |
|---|---:|---:|---:|
| 8×8 | +32 cycles/tile | +32 cycles/tile | +32 cycles/tile |
| 16×16 | +32 cycles/tile | +32 cycles/tile | +32 cycles/tile |
| 32×32 | +64 cycles/tile | +64 cycles/tile | +64 cycles/tile |

Each doubling of pdepth adds pd×(tile_m+tile_n) cycles per spatial tile — a constant marginal cost in cycles. But as a percentage of total, the impact shrinks because total cycles grow (more DMA/compute cycles diluting the overhead).

### 4. Discrepancy with cmodel dispatcher code

The current `dataflow_dispatcher.c` applies `get_fill_cycles`/`get_drain_cycles` **inside the K-tile loop** (lines 72-84), meaning fill/drain is counted per K-tile rather than per spatial tile. For WS dataflow where weights are preloaded in PEs, this is a bug — fill/drain should happen once per spatial tile, not once per K-tile. The analytical model above uses the hardware-accurate spatial-tile accounting.

**Impact:** For this workload (K=256, tile_k=16, so 16 K-tiles per spatial tile), the dispatcher overcounts fill/drain by 16×. This means cmodel-reported cycles are inflated by ~15× fill/drain overhead for WS dataflow. Fixing this would bring cmodel cycle estimates in line with hardware expectations.

## Recommendation

1. **Keep pdepth=2 as default** — balances pipeline depth (for clock frequency headroom) against fill/drain overhead.
2. **Fix the dispatcher** — move fill/drain accounting outside the K-loop for WS dataflow (`dataflow_dispatcher.c`, lines 72-84).
3. **Consider pdepth=1 for small-PE configurations** — if clock frequency can be maintained with a shallower pipeline, the utilization gain (5pp at 16×16, 3pp at 32×32) is free performance.
4. **Revisit if clock frequency depends on pdepth** — if going from pd=2→4 enables 2× clock frequency, the TOPS gain dominates the 8pp utilization loss. This exploration assumes fixed 1 GHz clock.
5. **Cross-reference with OS dataflow** — output-stationary has zero fill/drain overhead, making pdepth irrelevant for OS. For mixed-dataflow architectures, pdepth only matters for systolic (WS/RS) paths.
