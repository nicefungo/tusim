# Pooling Engine (O6)

> **Gap:** O6 — Pooling operations (MaxPool, AvgPool)  
> **Status:** Implemented  
> **Version:** 1.0  
> **Date:** 2026-05-31

## Overview

The Pooling Engine provides MaxPool2D and AveragePool2D operations with configurable kernel size, stride, and padding. It complements the convolution engine to enable full vision model support on the TU cmodel.

### Why Pooling Matters

Pooling is a fundamental operation in convolutional neural networks:

1. **Dimensionality reduction** — Reduces spatial dimensions, lowering computation in later layers
2. **Translation invariance** — Small translations in input produce same or similar output
3. **Feature aggregation** — MaxPool captures dominant features; AvgPool smooths and summarizes

Without pooling, ResNet blocks (Conv → BN → ReLU → Pool) cannot be modeled on the TU accelerator. Pooling is the last missing piece of the basic vision op catalog.

## Architecture

### Operation Types

| Operation | Semantics | Use Case |
|-----------|-----------|----------|
| `TU_POOL_MAX` | For each window, output the maximum value | Feature detection, MaxPool layers |
| `TU_POOL_AVG` | For each window, output the mean value | Global average pooling, smoothing |

### Data Layout

All tensors use **NCHW** layout (batch × channels × height × width), consistent with the rest of the TU cmodel.

### Padding Semantics

Two padding modes:

| Mode | MaxPool | AvgPool |
|------|---------|---------|
| **Padding value** | `-∞` (padded regions never win) | `0.0` (padded regions contribute zero) |
| **Count in AvgPool** | N/A | Configurable: `count_include_pad` |

With `count_include_pad = true`: Always divide by `KH × KW` (PyTorch default)  
With `count_include_pad = false`: Divide by actual number of valid elements in window

### Dimension Formula

```
OH = ⌊(IH + 2×PH - KH) / SH⌋ + 1
OW = ⌊(IW + 2×PW - KW) / SW⌋ + 1
```

Where:
- `IH, IW`: Input height/width
- `KH, KW`: Kernel height/width
- `SH, SW`: Stride height/width
- `PH, PW`: Padding height/width (symmetric by default)

Asymmetric padding supported via `asym_padding` + `ph_top`, `ph_bottom`, `pw_left`, `pw_right`.

## API Reference

### Descriptor Setup

```c
tu_pool_desc_t desc = {0};
desc.pool_type  = TU_POOL_MAX;        // or TU_POOL_AVG
desc.batch      = 1;                  // N
desc.channels   = 64;                 // C
desc.ih = 56;  desc.iw = 56;         // Input spatial dims
desc.kh = 3;   desc.kw = 3;          // Kernel
desc.sh = 2;   desc.sw = 2;          // Stride
desc.ph = 1;   desc.pw = 1;          // Padding
desc.elem_size  = 4;                  // 4 = FP32, 2 = FP16
desc.is_float   = true;
desc.src_region = &src_sram;
desc.src_offset = 0;
desc.dst_region = &dst_sram;
desc.dst_offset = 0;

// Compute output dims
tu_pool_compute_dims(&desc);
// Now desc.oh and desc.ow are computed

// Validate
if (tu_pool_validate(&desc) != 0) { /* error */ }

// Execute
int64_t cycles = tu_pool_execute(&desc);
```

### Low-Level Functions

For direct FP32 manipulation (bypassing SRAM):

```c
// MaxPool on a single channel
tu_pool_max_2d(src_f32, dst_f32, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, -INFINITY);

// AvgPool on a single channel
tu_pool_avg_2d(src_f32, dst_f32, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, true);
```

## Cycle Model

The cycle accounting is a simplified functional model:

```
cycles_per_channel = OH × OW × KH × KW × ops_per_element
total_cycles = N × C × cycles_per_channel + KH  (pipeline drain)
```

Where `ops_per_element = 1` for MaxPool (one comparison) and `2` for AvgPool (one add + one divide).

For cycle-accurate modeling (P2.5), this would be refined with:
- Memory bandwidth: SRAM reads/writes per window
- Bank conflicts: strided access patterns
- Pipeline parallelism: overlapping kernel rows

## Test Coverage

| Test | What It Verifies |
|------|-----------------|
| `pool_dim_computation` | Output dimension formulas with/without padding |
| `maxpool_basic` | 4×4→2×2 with 2×2 kernel, stride 2 |
| `maxpool_stride1` | 3×3→2×2 with stride 1 overlapping windows |
| `maxpool_negative` | Correct max selection with all-negative values |
| `avgpool_basic` | 4×4→2×2 average pooling |
| `avgpool_exclude_pad` | `count_include_pad=false` on edge windows |
| `maxpool_with_padding` | Padding with -∞ pad value |
| `avgpool_padding` | Include vs. exclude count with padding |
| `pool_all_equal` | Identity: all-equal input produces correct output |
| `pool_single_element` | 1×1 degenerate case |
| `pool_rectangular_kernel` | Non-square 2×3 kernel |
| `pool_full_execution` | Full SRAM-based execution path |
| `pool_multichannel` | Multi-channel pooling with SRAM |
| `pool_validation` | Descriptor validation (overflow, null pointers) |

## Examples

### ResNet-50 First MaxPool

```c
// ResNet-50: Conv(7×7, s2) → BN → ReLU → MaxPool(3×3, s2)
// Input: 1×64×112×112, Output: 1×64×56×56
tu_pool_desc_t desc = {
    .pool_type = TU_POOL_MAX,
    .batch = 1, .channels = 64,
    .ih = 112, .iw = 112,
    .kh = 3, .kw = 3,
    .sh = 2, .sw = 2,
    .ph = 1, .pw = 1,    // ceil_mode effective
    .elem_size = 2,       // FP16
    .is_float = true,
    .src_region = &conv_output,
    .dst_region = &pool_output,
};
tu_pool_execute(&desc);
// Output: 1×64×56×56
```

### Global Average Pooling

```c
// Inception/ResNet final layer: GAP reduces spatial to 1×1
tu_pool_desc_t desc = {
    .pool_type = TU_POOL_AVG,
    .batch = 1, .channels = 2048,
    .ih = 7, .iw = 7,
    .kh = 7, .kw = 7,    // Kernel = input spatial
    .sh = 1, .sw = 1,
    .ph = 0, .pw = 0,
    .elem_size = 4,      // FP32
    .is_float = true,
    .src_region = &feature_map,
    .dst_region = &gap_output,
};
tu_pool_execute(&desc);
// Output: 1×2048×1×1
```

## Integration Points

1. **ISA**: New opcodes `TU_ISA_POOL` (0x09) with sub-op for Max/Avg in flags
2. **Command Queue**: `tu_cmdq_submit_pool()` command submission
3. **Performance Counters**: `tu_perf_compute_record_op(c, 9/10, ...)` for PoolMax/PoolAvg tracking
4. **Compiler**: ONNX MaxPool/AveragePool/GlobalAveragePool → TU ASM POOL instruction

## File Structure

```
tu_cmodel/compute/
├── pooling_engine.h      — Pooling engine API and types
├── pooling_engine.c      — MaxPool/AvgPool implementation
└── (existing)
    ├── convolution_engine.h/c
    ├── attention_engine.h/c
    ├── elementwise_pipeline.h/c
    ├── normalization_engine.h/c
    └── softmax_engine.h/c
```

## Comparison with PyTorch

| Property | PyTorch `nn.MaxPool2d` | TU Pooling Engine |
|----------|----------------------|-------------------|
| Kernel | ✓ | ✓ |
| Stride | ✓ | ✓ |
| Padding | ✓ (symmetric by default) | ✓ (symmetric + asymmetric) |
| Dilation | ✓ | ✗ (future) |
| `ceil_mode` | ✓ | Via explicit padding config |
| `return_indices` | ✓ | ✗ (cmodel doesn't need) |
| NCHW layout | ✓ | ✓ |
| FP16/FP32 | ✓ | ✓ |
| INT8 | ✗ | ✓ (via FP32 internal + cast) |

## Next Steps

1. **Dilation support** — Add `dilation[2]` to pooling descriptor for dilated pooling
2. **3D Pooling** — MaxPool3D/AvgPool3D for video models
3. **Global pooling shortcut** — Detect KH=IH, KW=IW and use optimized reduction path
4. **ISA integration** — Add POOL instruction to TU ISA and command queue
