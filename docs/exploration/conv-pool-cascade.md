# Conv+Pool Cascade Sweep

**Date:** 2026-07-05
**Question:** What fraction of a conv+pool vision block does pooling add, across conv kernel sizes, PE array dimensions, and pool configurations?
**Hypothesis:** Pool overhead is small (< 5%) for large-kernel convs because conv compute dominates. But for pointwise (1×1) convolutions on large PE arrays, pool overhead could be significant — approaching the 8-16% range seen in norm-after-attention.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| Conv kernel | 1×1, 3×3, 5×5 | Pointwise, standard, and large-kernel conv; stride=1, same padding |
| PE rows/cols | 8×8, 16×16, 32×32 | 3 sizes, analytical conv cycle model |
| Pool kernel | 2×2, 3×3 | Stride=2 downsampling (ResNet standard) |
| Pool type | Max, Avg | MaxPool (1 op/elem) vs AvgPool (2 ops/elem) |
| Workload | 56×56 input, 64→128 channels | ResNet-50 mid-layer representative |

**Method:** Analytical `tu_conv_estimate_cycles()` for conv + functional `tu_pool_execute()` with SRAM-backed data for pool. Conv output (56×56×128) feeds directly into stride-2 pool.

**Configs attempted:** 3 conv × 3 PE × 4 pool = 36. All valid.

## Results Table

| ConvK | PE | Pool | ConvCyc | PoolCyc | TotalCyc | Pool% |
|-------|-----|------|---------|---------|----------|-------|
| 1×1 | 8×8 | Max 2×2s2 | 1,630,720 | 401,410 | 2,032,130 | 19.8% |
| 1×1 | 8×8 | Avg 2×2s2 | 1,630,720 | 802,818 | 2,433,538 | 33.0% |
| 1×1 | 8×8 | Max 3×3s2 | 1,630,720 | 839,811 | 2,470,531 | 34.0% |
| 1×1 | 8×8 | Avg 3×3s2 | 1,630,720 | 1,679,619 | 3,310,339 | 50.7% |
| 1×1 | 16×16 | Max 2×2s2 | 727,552 | 401,410 | 1,128,962 | 35.6% |
| 1×1 | 16×16 | Avg 2×2s2 | 727,552 | 802,818 | 1,530,370 | 52.5% |
| 1×1 | 16×16 | Max 3×3s2 | 727,552 | 839,811 | 1,567,363 | 53.6% |
| 1×1 | 16×16 | Avg 3×3s2 | 727,552 | 1,679,619 | 2,407,171 | 69.8% |
| 1×1 | 32×32 | Max 2×2s2 | 501,760 | 401,410 | 903,170 | 44.4% |
| 1×1 | 32×32 | Avg 2×2s2 | 501,760 | 802,818 | 1,304,578 | 61.5% |
| 1×1 | 32×32 | Max 3×3s2 | 501,760 | 839,811 | 1,341,571 | 62.6% |
| 1×1 | 32×32 | Avg 3×3s2 | 501,760 | 1,679,619 | 2,181,379 | 77.0% |
| 3×3 | 8×8 | Max 2×2s2 | 11,264,512 | 401,410 | 11,665,922 | 3.4% |
| 3×3 | 8×8 | Avg 2×2s2 | 11,264,512 | 802,818 | 12,067,330 | 6.7% |
| 3×3 | 8×8 | Max 3×3s2 | 11,264,512 | 839,811 | 12,104,323 | 6.9% |
| 3×3 | 8×8 | Avg 3×3s2 | 11,264,512 | 1,679,619 | 12,944,131 | 13.0% |
| 3×3 | 16×16 | Max 2×2s2 | 3,136,000 | 401,410 | 3,537,410 | 11.3% |
| 3×3 | 16×16 | Avg 2×2s2 | 3,136,000 | 802,818 | 3,938,818 | 20.4% |
| 3×3 | 16×16 | Max 3×3s2 | 3,136,000 | 839,811 | 3,975,811 | 21.1% |
| 3×3 | 16×16 | Avg 3×3s2 | 3,136,000 | 1,679,619 | 4,815,619 | 34.9% |
| 3×3 | 32×32 | Max 2×2s2 | 1,103,872 | 401,410 | 1,505,282 | 26.7% |
| 3×3 | 32×32 | Avg 2×2s2 | 1,103,872 | 802,818 | 1,906,690 | 42.1% |
| 3×3 | 32×32 | Max 3×3s2 | 1,103,872 | 839,811 | 1,943,683 | 43.2% |
| 3×3 | 32×32 | Avg 3×3s2 | 1,103,872 | 1,679,619 | 2,783,491 | 60.3% |
| 5×5 | 8×8 | Max 2×2s2 | 30,532,096 | 401,410 | 30,933,506 | 1.3% |
| 5×5 | 8×8 | Avg 2×2s2 | 30,532,096 | 802,818 | 31,334,914 | 2.6% |
| 5×5 | 8×8 | Max 3×3s2 | 30,532,096 | 839,811 | 31,371,907 | 2.7% |
| 5×5 | 8×8 | Avg 3×3s2 | 30,532,096 | 1,679,619 | 32,211,715 | 5.2% |
| 5×5 | 16×16 | Max 2×2s2 | 7,952,896 | 401,410 | 8,354,306 | 4.8% |
| 5×5 | 16×16 | Avg 2×2s2 | 7,952,896 | 802,818 | 8,755,714 | 9.2% |
| 5×5 | 16×16 | Max 3×3s2 | 7,952,896 | 839,811 | 8,792,707 | 9.6% |
| 5×5 | 16×16 | Avg 3×3s2 | 7,952,896 | 1,679,619 | 9,632,515 | 17.4% |
| 5×5 | 32×32 | Max 2×2s2 | 2,308,096 | 401,410 | 2,709,506 | 14.8% |
| 5×5 | 32×32 | Avg 2×2s2 | 2,308,096 | 802,818 | 3,110,914 | 25.8% |
| 5×5 | 32×32 | Max 3×3s2 | 2,308,096 | 839,811 | 3,147,907 | 26.7% |
| 5×5 | 32×32 | Avg 3×3s2 | 2,308,096 | 1,679,619 | 3,987,715 | 42.1% |

## Key Findings

### 1. Pool overhead is workload-dependent: 1.3% to 77%

The pool cost is fixed per element count — it's purely SRAM-bandwidth-bound and doesn't benefit from larger PE arrays or different conv kernels. Conv cost scales with `in_c × K² × out_c`, so:

- **5×5 conv on 8×8 PE:** Pool = 1.3% (Max 2×2). Conv dominates at 30.5M cycles vs. 0.4M for pool.
- **1×1 conv on 32×32 PE:** Pool = 77.0% (Avg 3×3). Conv completes in only 0.5M cycles — pool is 3.4× more expensive than conv.

This is the widest overhead range of any cross-engine sweep so far (cf. norm-after-attention at 8-16%).

### 2. Larger PE arrays amplify pool overhead dramatically

For 1×1 conv + Avg 3×3 pool, pool overhead grows from 50.7% (8×8 PE) → 69.8% (16×16) → 77.0% (32×32). The conv cycle count drops 3.2× (1.63M → 0.50M) as PEs scale, but pool cycles are fixed at 1.68M. On a 32×32 array, the pool engine is the bottleneck — not the systolic array.

| ConvK | PE | Max 2×2 Pool% | Trend |
|-------|-----|---------------|-------|
| 1×1 | 8×8 | 19.8% | baseline |
| 1×1 | 16×16 | 35.6% | +15.8pp |
| 1×1 | 32×32 | 44.4% | +8.8pp |
| 3×3 | 8×8 | 3.4% | baseline |
| 3×3 | 16×16 | 11.3% | +7.9pp |
| 3×3 | 32×32 | 26.7% | +15.4pp |

### 3. MaxPool vs AvgPool: consistent 2× cost ratio

AvgPool always costs exactly 2× MaxPool (1 op/elem vs. 2 ops/elem in the cycle model). For large-kernel convs where pool is negligible (< 5%), this doesn't matter. But for pointwise convs, choosing MaxPool over AvgPool saves 13-27 percentage points of overhead.

### 4. 3×3 pool adds ~2.1× cost over 2×2 pool

The 3×3 kernel has 9/4 = 2.25× more reads per output element than 2×2. Combined with slightly fewer output elements (27² vs. 28²), the net is ~2.09×. This is a pure bandwidth cost — a hardware pooling unit with line-buffer reuse could close this gap.

### 5. Pointwise (1×1) convolution is the critical case

MobileNets, EfficientNets, and transformer FFN layers all use 1×1 (pointwise) convolutions extensively, often followed by pooling. On a 32×32 PE array, the pool engine takes 44-77% of the block latency. This suggests:

- **Fused conv-pool hardware** would eliminate the SRAM round-trip for the conv output
- **Dedicated pooling unit** with line-buffer reuse could reduce pool cycles to `oh × ow × channels` (output-store only) instead of `oh × ow × kh × kw × channels`
- Alternatively, implement pooling as a **post-processing pass on the accumulator output** before SRAM writeback — effectively zero-cost

## Harness

`tests/test_conv_pool_cascade.c` — `make test-conv-pool-cascade`
