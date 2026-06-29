# Convolution Kernel×Stride×PE Sweep

**Date:** 2026-06-29
**Question:** How does convolution cycle count scale with kernel size and stride across different PE array dimensions?

## Config Matrix

| Parameter | Values |
|-----------|--------|
| Workload | 56×56 input, 128→128 channels (ResNet mid-layer) |
| Kernel | 1×1, 3×3, 5×5, 7×7 |
| Stride | 1, 2 |
| PE array | 8×8, 16×16, 32×32 |
| Method | `tu_conv_estimate_cycles()` — analytical cycle model |

## Results

| Kernel       | PE     | out_HW | im2colK | Total Cycles | GOPS |
|-------------|--------|--------|---------|-------------|------|
| 1×1, s=1    | 8×8    | 56×56  | 128     | 2,860,032   | 35.9 |
| 1×1, s=1    | 16×16  | 56×56  | 128     | 1,053,696   | 97.5 |
| 1×1, s=1    | 32×32  | 56×56  | 128     | 602,112     | 170.7 |
| 3×3, s=1    | 8×8    | 56×56  | 1152    | 22,127,616  | 41.8 |
| 3×3, s=1    | 16×16  | 56×56  | 1152    | 5,870,592   | 157.5 |
| 3×3, s=1    | 32×32  | 56×56  | 1152    | 1,806,336   | 512.0 |
| 5×5, s=1    | 8×8    | 56×56  | 3200    | 60,662,784  | 42.3 |
| 5×5, s=1    | 16×16  | 56×56  | 3200    | 15,504,384  | 165.7 |
| 5×5, s=1    | 32×32  | 56×56  | 3200    | 4,214,784   | 609.5 |
| 7×7, s=1    | 8×8    | 56×56  | 6272    | 118,465,536 | 42.5 |
| 7×7, s=1    | 16×16  | 56×56  | 6272    | 29,955,072  | 168.1 |
| 7×7, s=1    | 32×32  | 56×56  | 6272    | 7,827,456   | 643.3 |
| 3×3, s=2    | 8×8    | 28×28  | 1152    | 5,569,536   | 41.5 |
| 3×3, s=2    | 16×16  | 28×28  | 1152    | 1,505,280   | 153.6 |
| 3×3, s=2    | 32×32  | 28×28  | 1152    | 496,128     | 466.0 |
| 5×5, s=2    | 8×8    | 28×28  | 3200    | 15,203,328  | 42.2 |
| 5×5, s=2    | 16×16  | 28×28  | 3200    | 3,913,728   | 164.1 |
| 5×5, s=2    | 32×32  | 28×28  | 3200    | 1,110,528   | 578.3 |
| 7×7, s=2    | 8×8    | 28×28  | 6272    | 29,654,016  | 42.5 |
| 7×7, s=2    | 16×16  | 28×28  | 6272    | 7,526,400   | 167.3 |
| 7×7, s=2    | 32×32  | 28×28  | 6272    | 2,032,128   | 619.5 |

## Key Finding: GOPS Improves with Larger Kernels on Big PE Arrays

**Larger kernels improve PE utilization**, counterintuitively. On a 32×32 PE array:
- 1×1 kernel (im2col K=128): **171 GOPS** — poor utilization, many short-K GEMM tiles
- 7×7 kernel (im2col K=6272): **643 GOPS** — 3.8× better, deep K tiles fill the pipeline

This happens because im2col converts spatial convolution into GEMM where:
- **M** = output channels per group (=128, fixed)
- **N** = output spatial dims (=H_out×W_out, shrinks with stride)
- **K** = input channels × kernel_h × kernel_w (grows with kernel size)

A deep K dimension creates few, large GEMM tiles that amortize pipeline fill overhead. Shallow K creates many small tiles with poor fill efficiency.

**Stride=2 gives ~3.8× speedup** for all kernel sizes, close to the theoretical 4× from quartering output pixels. The gap is im2col overhead (fixed per input element, independent of stride).

**8×8 PE is always compute-bound** — GOPS plateaus at ~42 regardless of kernel size, meaning GEMM dominates and im2col overhead is negligible. The small PE array can't exploit the deeper K tiles.

## Sweep Harness

`tests/test_conv_sweep.c` — `make test-conv-sweep`
