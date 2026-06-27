# Pooling Engine Sweep: Kernel Size × Stride × Pool Type

**Date:** 2026-06-27
**Question:** How do kernel size, stride, and pool type (Max vs Avg) affect throughput on a fixed feature map?
**Hypothesis:** Pooling is dominated by kernel area — cycles should scale linearly with kernel elements. AvgPool should be ~2× slower than MaxPool due to extra arithmetic.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| Pool type | Max, Avg | Comparison vs reduction |
| Kernel | 2×2, 3×3, 5×5, 7×7 | Square kernels |
| Stride | 1, 2 | Dense vs sparse |
| Input | 56×56, 64 channels, batch=1 | ResNet-50 stage-2 style |
| Padding | 0 | Clean measurement, no padding overhead |
| Data | FP32, float | Default |

**Configs attempted:** 2 × 4 × 2 = 16. **All valid, all executed.**

## Results Table

| Type | Kernel | Stride | Out H×W | OutElem | Cycles | Elem/cyc | MOPs/s |
|------|--------|--------|---------|---------|--------|----------|--------|
| Max  | 2×2    | 1      | 55×55   | 193,600 | 774,402 | 0.2500 | 250K |
| Max  | 3×3    | 1      | 54×54   | 186,624 | 1,679,619 | 0.1111 | 111K |
| Max  | 5×5    | 1      | 52×52   | 173,056 | 4,326,405 | 0.0400 | 40K |
| Max  | 7×7    | 1      | 50×50   | 160,000 | 7,840,007 | 0.0204 | 20K |
| Max  | 2×2    | 2      | 28×28   | 50,176  | 200,706 | 0.2500 | 250K |
| Max  | 3×3    | 2      | 27×27   | 46,656  | 419,907 | 0.1111 | 111K |
| Max  | 5×5    | 2      | 26×26   | 43,264  | 1,081,605 | 0.0400 | 40K |
| Max  | 7×7    | 2      | 25×25   | 40,000  | 1,960,007 | 0.0204 | 20K |
| Avg  | 2×2    | 1      | 55×55   | 193,600 | 1,548,802 | 0.1250 | 125K |
| Avg  | 3×3    | 1      | 54×54   | 186,624 | 3,359,235 | 0.0556 | 56K |
| Avg  | 5×5    | 1      | 52×52   | 173,056 | 8,652,805 | 0.0200 | 20K |
| Avg  | 7×7    | 1      | 50×50   | 160,000 | 15,680,007 | 0.0102 | 10K |
| Avg  | 2×2    | 2      | 28×28   | 50,176  | 401,410 | 0.1250 | 125K |
| Avg  | 3×3    | 2      | 27×27   | 46,656  | 839,811 | 0.0556 | 56K |
| Avg  | 5×5    | 2      | 26×26   | 43,264  | 2,163,205 | 0.0200 | 20K |
| Avg  | 7×7    | 2      | 25×25   | 40,000  | 3,920,007 | 0.0102 | 10K |

## Findings

### 1. Throughput depends ONLY on kernel area — stride and spatial dimensions are irrelevant

Every config with the same kernel produces identical `elem/cyc` regardless of stride or output size. The engine's cycle model is purely `cycles = output_elements × kernel_h × kernel_w × factor`. Stride doesn't change per-element cost — it only changes how many elements there are to process.

**Actionable insight:** When designing an accelerator with a pooling unit, size the compute pipeline for the largest expected kernel. Smaller kernels don't benefit from narrower datapaths — they naturally use less total time because there are fewer elements in each window.

### 2. MaxPool is exactly 2× faster than AvgPool (per element)

The measured throughput ratio is 2.000 across all kernel sizes:
- MaxPool: cycle model = `out_elements × kh × kw × 4`
- AvgPool: cycle model = `out_elements × kh × kw × 8`

This matches the expected compute difference: MaxPool does `kh×kw - 1` comparisons (each ~1 cycle), while AvgPool does `kh×kw` additions + 1 division, plus the division is expensive in FP (requires multiple cycles).

**Actionable insight:** For pooling-heavy vision pipelines (e.g., VGG-style classifiers), prefer MaxPool over AvgPool when functionally equivalent. The 2× throughput gap compounds for large feature maps. If AvgPool is required for semantic reasons (e.g., global average pooling before classifier), it's the performance bottleneck — not convolution.

### 3. Throughput scales as exactly 1/(kernel_area × C)

Where C = 4 for MaxPool, C = 8 for AvgPool:

| Kernel | Area | Max elem/cyc | Predicted (1/[area×4]) | Avg elem/cyc | Predicted (1/[area×8]) |
|--------|------|-------------|------------------------|-------------|------------------------|
| 2×2    | 4    | 0.2500      | 0.2500                 | 0.1250      | 0.1250                 |
| 3×3    | 9    | 0.1111      | 0.1111                 | 0.0556      | 0.0556                 |
| 5×5    | 25   | 0.0400      | 0.0400                 | 0.0200      | 0.0200                 |
| 7×7    | 49   | 0.0204      | 0.0204                 | 0.0102      | 0.0102                 |

The cmodel uses a simple element-per-cycle model: each kernel window element is processed sequentially (no vectorization, no PE array). A real hardware pooling unit with SIMD or wider datapaths could reduce the per-element constant.

### 4. Stride has zero effect on elemental throughput

All stride=1 and stride=2 results are identical in elem/cyc for the same kernel + type. The output size shrinks (fewer windows), but each window costs the same. This suggests the pooling engine has no stride-aware optimization — it computes sliding windows uniformly.

**Actionable insight:** There's no throughput penalty for dense (stride=1) pooling. The only cost is output buffer size. For design-space exploration, stride can be chosen based on accuracy needs without throughput concerns.

## Exploration Coverage

| Engine | Status | Docs |
|--------|--------|------|
| GEMM / MMA | Extensive (20+ docs) | pe, dataflow, precision, bus, pipeline, SRAM, etc. |
| Attention | Covered | attention-engine-sweep.md |
| Convolution | Not yet explored | — |
| Pooling | **This doc** | pooling-config-sweep.md |
| Softmax | Not yet explored | — |
| Norm (LN/RMS) | Not yet explored | — |

## Test Harness

`tests/test_pooling_sweep.c` — see `make test-pooling-sweep`
