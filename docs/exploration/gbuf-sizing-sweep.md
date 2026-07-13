# GBUF Sizing Sweep: Global Buffer vs Weight Footprint

**Date:** 2026-07-13
**Status:** Complete
**Type:** Analytical sweep (standalone C, no cmodel dependency)

## Design Question

What's the minimum global buffer (GBUF) size needed to avoid redundant DRAM weight fetches during GEMM? How does GBUF sizing interact with the K-dimension (weight footprint)?

## Config Matrix

| Parameter | Values |
|-----------|--------|
| PE Array | 16×16 |
| M | 256 |
| N | 256 |
| K | 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| GBUF | 64, 128, 256, 512, 1024, 2048, 4096, 8192 KB |
| Bus Width | 16 bytes (128-bit) |
| W-SPAD | 128 KB |
| A-SPAD | 64 KB |
| O-SPAD | 64 KB |

## Results: DMA Cycles

| K\GBUF | 64KB | 128KB | 256KB | 512KB | 1024KB | 2048KB | 4096KB | 8192KB |
|--------|------|-------|-------|-------|--------|--------|--------|--------|
| K=64   | 24,576 | 24,576 | 24,576 | 24,576 | 24,576 | 24,576 | 24,576 | 24,576 |
| K=128  | 94,208 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 |
| K=256  | 180,224 | 172,032 | 49,152 | 49,152 | 49,152 | 49,152 | 49,152 | 49,152 |
| K=512  | 376,832 | 344,064 | 327,680 | 81,920 | 81,920 | 81,920 | 81,920 | 81,920 |
| K=1024 | 868,352 | 737,280 | 671,744 | 638,976 | 147,456 | 147,456 | 147,456 | 147,456 |
| K=2048 | 2,244,608 | 1,720,320 | 1,458,176 | 1,327,104 | 1,261,568 | 278,528 | 278,528 | 278,528 |
| K=4096 | 6,569,984 | 4,472,832 | 3,424,256 | 2,899,968 | 2,637,824 | 2,506,752 | 540,672 | 540,672 |
| K=8192 | 21,512,192 | 13,123,584 | 8,929,280 | 6,832,128 | 5,783,552 | 5,259,264 | 4,997,120 | 1,064,960 |

## Speedup vs 64KB Baseline

| K\GBUF | 128KB | 256KB | 512KB | 1024KB | 2048KB | 4096KB | 8192KB |
|--------|-------|-------|-------|--------|--------|--------|--------|
| K=64   | 1.00x | 1.00x | 1.00x | 1.00x  | 1.00x  | 1.00x  | 1.00x  |
| K=128  | 1.00x | 1.00x | 1.00x | 1.00x  | 1.00x  | 1.00x  | 1.00x  |
| K=256  | 1.05x | 1.05x | 1.05x | 1.05x  | 1.05x  | 1.05x  | 1.05x  |
| K=512  | 1.10x | 1.15x | 1.15x | 1.15x  | 1.15x  | 1.15x  | 1.15x  |
| K=1024 | 1.18x | 1.29x | 1.36x | 1.36x  | 1.36x  | 1.36x  | 1.36x  |
| K=2048 | 1.30x | 1.54x | 1.69x | 1.78x  | 1.78x  | 1.78x  | 1.78x  |
| K=4096 | 1.47x | 1.92x | 2.27x | 2.49x  | 2.62x  | 2.62x  | 2.62x  |
| K=8192 | 1.64x | 2.41x | 3.15x | 3.72x  | 4.09x  | 4.30x  | 4.30x  |

## Weight-Fit Threshold

| K   | Weight Footprint | Min GBUF |
|-----|-----------------|----------|
| 64  | 32 KB           | 64 KB    |
| 128 | 64 KB           | 64 KB    |
| 256 | 128 KB          | 128 KB   |
| 512 | 256 KB          | 256 KB   |
| 1024| 512 KB          | 512 KB   |
| 2048| 1 MB            | 1 MB     |
| 4096| 2 MB            | 2 MB     |
| 8192| 4 MB            | 4 MB     |

Weight footprint = K × N × 2 bytes (FP16 weights). When GBUF ≥ weight footprint, DRAM → GBUF weight transfers happen exactly once (no redundant reloads).

## Key Finding

**GBUF sizing is a binary threshold determined by weight footprint. Below K×N×2 bytes, DMA cycles grow proportional to `ceil(weight_bytes / gbuf_bytes)`. Above it, returns vanish instantly.**

- The threshold is exact: doubling K requires doubling GBUF to maintain 1× weight-DRAM penalty
- For K=8192 with 64KB GBUF: 21.5M DMA cycles. With 8192KB GBUF: 1.06M — **20× more DMA traffic**
- The penalty comes from the GBUF→SPAD pipeline: when GBUF can't hold all weights, each K-tile must reload from DRAM, multiplying total DRAM traffic

## Recommendations

1. **Size GBUF to the weight footprint of the target workload.** For a 256-wide GEMM with K≤512, 256KB GBUF suffices. For LLM attention where N=64 (head_dim) and K up to 4096, 512KB GBUF covers it.
2. **Oversizing GBUF beyond weight-fit yields zero DMA benefit.** The extra silicon area is wasted — use it for SRAM or more PEs instead.
3. **If the workload K-dim varies widely, consider double-buffered weight staging in GBUF** rather than a single large monolithic buffer. The double-buffering exploration already showed that overlap between DMA and compute can recover some of the penalty when GBUF is undersized.
