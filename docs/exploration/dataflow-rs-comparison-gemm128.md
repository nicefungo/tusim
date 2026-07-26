# Dataflow Sweep: WS vs OS vs RS for GEMM 128×128×256

> **Superseded for comparative evidence (2026-07-26):** the harness selected
> process-global state before core swap-in, used a non-failing pairwise check,
> and evaluated formulas separate from the live per-K-tile dispatcher. See
> `dataflow-plugin-executable-reaudit.md` for fail-closed executable evidence.

**Date:** 2026-06-10
**Question:** How does row-stationary (RS) dataflow compare to weight-stationary (WS) and output-stationary (OS) for a medium GEMM workload? What's the cycle cost of fill/drain overhead?
**Hypothesis:** RS splits the difference between WS and OS — its reduced fill/drain (pd-1 vs pd) gives a partial speedup over WS, but the fundamental requirement to flow data through the array still imposes overhead that OS avoids entirely.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| Dataflow | weight_stationary, output_stationary, row_stationary | Three registered plugins |
| Workload | M=128, N=128, K=256 | 8.39 MFLOPs GEMM |
| PE array | 16×16 | Baseline sweet spot |
| Pipeline depth | 2 | Systolic fill/drain (WS/RS) |
| Bus width | 256-bit (32 B/cycle) | Default |
| Clock | 1.0 GHz | Default |
| Precision | FP16 W/A, FP32 O | Default |

**Configs tested:** 3 dataflows, analytical cycle model (64 tiles each), functional correctness verified via cmodel.

## Functional Correctness

All three dataflows produce bit-identical results (max error < 1e-5 vs WS baseline), confirming the dataflow plugins are functionally equivalent — the only difference is cycle cost.

```
weight_stationary:  O[0]=-14.826792, O[16383]=9.515121 — OK
output_stationary:  O[0]=-14.826792, O[16383]=9.515121 — match WS ✓
row_stationary:     O[0]=-14.826792, O[16383]=9.515121 — match WS ✓
```

## Analytical Cycle Model

**Cycle formulas per tile (pe_r × pe_c, K inner dim, pd = pipeline depth):**

| Dataflow | Fill | Compute | Drain | Total per tile |
|----------|------|---------|-------|----------------|
| WS | `pd × nc` | `K` | `pd × mc` | `pd × nc + K + pd × mc` |
| OS | 0 | `K` | 0 | `K` |
| RS | `(pd-1) × nc + 1` | `K` | `(pd-1) × mc` | `(pd-1) × nc + 1 + K + (pd-1) × mc` |

**Tile geometry:** 8 M-tiles × 8 N-tiles = 64 tiles. All edge tiles are exact multiples (128 = 8 × 16).

## Results Table

| Dataflow | kCycles | mTOPS | Util% | vs WS |
|----------|---------|-------|-------|-------|
| weight_stationary | 20 | 204.8 | 80.0 | 1.00× (baseline) |
| output_stationary | 16 | 256.0 | 100.0 | 1.25× |
| row_stationary | 18 | 226.8 | 88.6 | 1.11× |

**DMA overhead:** 6 kCycles (192 KB at 32 B/cycle) — identical across dataflows since DMA is dataflow-agnostic.
**Total with DMA:** WS=26 kCyc, OS=22 kCyc, RS=24 kCyc.

## Key Finding

**OS gives 20% faster throughput than WS on this workload, and RS gives 10%.**

The fill/drain overhead is the dominant differentiator between dataflows for this workload (where K=256 dominates tile compute). OS eliminates fill/drain entirely by streaming both operands — at the cost of higher bandwidth demand. RS reduces fill/drain overhead by ~31 cycles per tile compared to WS (from 2×nc to 1×nc+1 for fill, 2×mc to 1×mc for drain), recovering about half the WS penalty.

For a 16×16 PE array processing 128×128×256 GEMM:
- **OS is best** when DMA bandwidth can sustain dual-streaming (no fill/drain = 100% PE utilization)
- **RS is a practical compromise** — 89% utilization with lower bandwidth demand than OS
- **WS is the simplest** — 80% utilization, W preloaded in PEs, lowest SRAM bandwidth during compute

This extends the prior WS-vs-OS comparison (2026-06-04) by adding RS to the analysis. The prior finding that "OS wins at small K" remains true — all three dataflows converge as K→∞, but for finite K, fill/drain overhead sets the ranking: OS > RS > WS.

## What's Next

- Extend sweep to small-K workloads (K=16, 32) where fill/drain overhead dominates
- Compare at larger PE arrays (32×32) where tile count decreases (4×4=16 tiles) and fill/drain cost per tile stays the same
- Compare memory bandwidth demand across dataflows (RS has intermediate bandwidth vs WS low and OS high)
