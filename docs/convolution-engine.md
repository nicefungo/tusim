# Convolution Engine — TU CModel

> **Gap:** O2 — Hardware convolution support for production accelerator modeling  
> **Date:** 2026-05-30  
> **Status:** Implemented ✅  
> **Tests:** 12/12 passing

---

## 1. Overview

Convolution is the dominant operation in computer vision models (ResNet, EfficientNet, YOLO) and a key workload for any production-grade systolic array accelerator. This feature adds a full convolution engine to the TU cmodel, supporting:

- **Conv2D** with stride, padding, dilation, and groups
- **Im2Col + GEMM** mapping (the standard approach for systolic arrays)
- **Direct convolution** reference for verification
- **Depthwise convolution** (MobileNet, EfficientNet)
- **Grouped convolution** (ResNeXt)
- **Fused ReLU/ReLU6 activation**
- **Cycle estimation** for performance modeling

### Why Im2Col + GEMM?

Systolic arrays are GEMM engines. The im2col transform converts convolution into matrix multiplication:

```
Conv2D(input[N][C][H][W], weight[K][C][R][S])  →
  im2col(input) → matrix A [C·R·S][H_out·W_out]
  weight (flatten) → matrix B [K][C·R·S]
  output ← A^T · B^T  (GEMM)
```

This is the same approach used by:
- **Gemmini** (Berkeley): Hardware im2col in the DMA engine
- **Eyeriss** (MIT): Row-stationary dataflow, im2col implicit
- **Google TPU**: XLA compiler performs im2col in software
- **cuDNN**: All convolution algorithms (im2col, Winograd, FFT) backed by GEMM

---

## 2. Convolution Descriptor

```c
#include "tu_cmodel/compute/convolution_engine.h"

tu_conv_desc_t desc = {
    .batch = 1,                    // N: batch size
    .in_channels = 3,              // C: input channels
    .in_height = 224,              // H: input height
    .in_width = 224,               // W: input width
    .out_channels = 64,            // K: output channels
    .kernel_h = 7,                 // R: filter height
    .kernel_w = 7,                 // S: filter width
    .stride_h = 2,                 // stride
    .stride_w = 2,
    .pad_t = 3, .pad_b = 3,        // padding (top, bottom, left, right)
    .pad_l = 3, .pad_r = 3,
    .dilation_h = 1, .dilation_w = 1,
    .groups = 1,                   // 1 = standard, C = depthwise
    .activation = TU_CONV_ACTIVATION_RELU,
    .has_bias = true,
};
```

### Dimension Formulas

```
output_height = floor((H + pad_t + pad_b - dilation_h * (R - 1) - 1) / stride_h + 1)
output_width  = floor((W + pad_l + pad_r - dilation_w * (S - 1) - 1) / stride_w + 1)

im2col_rows   = (C / groups) * R * S
im2col_cols   = output_height * output_width
```

### Grouped Convolution

When `groups = G`:
- Input channels split into G groups of C/G channels each
- Output channels split into G groups of K/G channels each
- Each group computes an independent convolution
- `groups = C = K` → depthwise convolution

---

## 3. API Reference

### Dimension Computation

```c
// Returns 0 on success, -1 on invalid parameters (kernel > input, etc.)
if (tu_conv_compute_dims(&desc) != 0) {
    // Invalid configuration
}
// After calling: desc.out_height, desc.out_width, desc.im2col_rows, desc.im2col_cols are set
```

### Im2Col Transform

```c
// NHWC input format: [H][W][C] (channel-last, standard for GPUs/TPUs)
float im2col_out[im2col_rows * im2col_cols]; // FP32
tu_im2col_nhwc(input_nhwc, im2col_out, &desc, sizeof(float));

// NCHW input format: [C][H][W] (channel-first)
tu_im2col_nchw(input_nchw, im2col_out, &desc, sizeof(float));
```

### Direct Convolution (Golden Reference)

```c
// NCHW format — FP32 golden reference
tu_conv2d_direct_nchw_fp32(input_nchw, weight_kcrs, bias, output_nchw, &desc);

// NHWC format
tu_conv2d_direct_nhwc_fp32(input_nhwc, weight_kcrs, bias, output_nhwc, &desc);
```

### Im2Col + GEMM Pipeline

```c
// Full pipeline: im2col → GEMM → bias → activation
void *im2col_buf = malloc(total_im2col_bytes);
tu_conv2d_im2col_gemm(input_nhwc, weight_kcrs, bias, output_nhwc, &desc, im2col_buf, sizeof(float));
```

### Cycle Estimation

```c
uint64_t cycles = tu_conv_estimate_cycles(&desc, pe_rows, pe_cols);
// Returns: im2col_DMA_cycles + GEMM_tile_cycles + bias_activation_cycles
```

---

## 4. Data Layouts

### NHWC (Channel-Last) — Default

```
Input:  [H][W][C]
        stride: row = W*C*elem_size, col = C*elem_size, chan = elem_size

Output: [H_out][W_out][K]
```

### NCHW (Channel-First)

```
Input:  [C][H][W]
        stride: chan = H*W*elem_size, row = W*elem_size, col = elem_size

Output: [K][H_out][W_out]
```

### KCRS Weight Format

```
Weight: [K][C][R][S]  (output channels first)
        stride: k = C*R*S*elem_size, c = R*S*elem_size, r = S*elem_size
```

This matches cuDNN, MIOpen, and ONNX weight layouts.

---

## 5. How Im2Col Works

For a 3×3 input, single-channel, 2×2 kernel, stride=1:

```
Input:    1 2 3     Im2Col output (4 rows × 4 cols):
          4 5 6     
          7 8 9     Row 0 (r=0,s=0): [1 2 4 5]  ← top-left patches
                    Row 1 (r=0,s=1): [2 3 5 6]  ← shifted right
                    Row 2 (r=1,s=0): [4 5 7 8]  ← shifted down
                    Row 3 (r=1,s=1): [5 6 8 9]  ← bottom-right patches
```

Each row of the im2col matrix is a flattened kernel window at a specific (kernel_row, kernel_col) offset. Each column is a spatial output position.

The GEMM then multiplies `weight_flat [K][C·R·S]` × `im2col [C·R·S][H_out·W_out]` to produce `output [K][H_out·W_out]`.

### Memory Requirements

For a standard ResNet-50 first layer (224×224×3 input, 7×7 kernel, 64 filters, stride=2):

| Component | Size |
|-----------|------|
| Input | 224 × 224 × 3 × 4 = 602 KB |
| Weight | 64 × 3 × 7 × 7 × 4 = 38 KB |
| Im2Col buffer | 147 × (109 × 109) × 4 = 7.0 MB |
| Output | 64 × 109 × 109 × 4 = 3.0 MB |

The im2col buffer is the largest intermediate. For hardware, this is typically done in tiles (like the TU MMA tiling) to reduce buffer requirements.

---

## 6. GEMM Integration

### Tiling Strategy

The im2col+GEMM pipeline maps to the existing TU MMA engine through tiling:

1. **Im2Col tiles**: The input is tiled into im2col chunks that fit in SRAM
2. **GEMM tiles**: Each im2col chunk feeds a standard `tu_mma(M=K, N=H_out*W_out, K=C·R·S)` call
3. **Accumulation**: Partial results from multiple K-tiles are accumulated

### With INT8

For INT8 quantized convolution:
1. Quantize input and weights per-channel/per-tensor
2. im2col operates on INT8 data
3. INT8 GEMM via `tu_int8_mma_tile()`
4. Dequantize with proper scale multiplication

---

## 7. Verification

### Test Coverage

| Test | What It Verifies |
|------|-----------------|
| Dimension computation | Correct output sizes for stride=1, stride=2 |
| Invalid config detection | Rejects kernel > input, negative output dims |
| Im2Col NHWC 3×3→2×2 | Exact element placement verified |
| Im2Col with padding | Zero-padding correctly applied |
| Direct conv identity (1×1) | Passes through input unchanged |
| Direct conv multi-channel | 2 input channels, 2 kernel channels |
| Strided convolution | Stride=2, 5×5 input, correct downsampling |
| Depthwise convolution | Groups=channels, per-channel scaling |
| Fused ReLU | Negative outputs clamped to zero |
| Cycle estimation | Non-zero cycle estimates for valid configs |
| Im2col+GEMM pipeline | Matches direct reference |

### Accuracy

- **Direct convolution**: Bit-exact with reference (same algorithm, FP32)
- **Im2col+GEMM vs Direct**: Identical results (same FP32 arithmetic, different loop order)
- **FP16** path: TBD — requires integration with tu_mma FP16 path

---

## 8. Cycle Model

### Im2Col Overhead

```
im2col_cycles = (C * H * W * elem_size) / DMA_bus_width_bytes
```

### GEMM Cycles (per group)

```
tiles = ceil(K/g / pe_rows) × ceil(H_out * W_out / pe_cols) × ceil(C·R·S / pe_cols)
per_tile_cycles = pipeline_depth * pe_cols (fill) + pe_cols (compute)
ge_cycles = tiles × per_tile_cycles
```

### Total Estimate

```
total = im2col_cycles + (ge_cycles * groups) + (K * H_out * W_out)
```

This model captures the dominant contributors: data movement (im2col), compute (GEMM), and elementwise (bias/activation).

---

## 9. Performance Examples

### ResNet-50 First Layer

| Parameter | Value |
|-----------|-------|
| Input | 224×224×3 |
| Kernel | 7×7, stride=2, 64 filters |
| Output | 64×109×109 |
| Im2Col rows | 147 (3×7×7) |
| Im2Col cols | 11,881 (109×109) |
| GEMM | [64 × 147] × [147 × 11881] |
| MACs | 64 × 147 × 11881 = 111.7M MACs |

With a 16×16 PE array:
- Tiles: (64/16) × (11881/16) × (147/16) ≈ 4 × 743 × 10 = 29,720 tiles
- Cycle estimate (functional): ~29,720 × 32 ≈ 951K cycles

---

## 10. Limitations & Future Work

- **Hardware im2col in DMA**: Currently im2col is software-only. Hardware address generation (gap: M3) would reduce overhead and enable double-double for tiled im2col.
- **FP16 path**: Im2Col currently assumes FP32 input. The im2col code handles arbitrary elem_size, so FP16 integration is straightforward (gap: integrate with tu_mma FP16).
- **Winograd convolution**: For 3×3 kernels, Winograd F(2,3) reduces MACs by 2.25× (gap: P2 optimization).
- **Conv3D**: 3D convolution for video models (gap: O1, P2).
- **Transposed convolution**: For upsampling/generator models (gap: O1, P2).
- **FP8/INT8 convolution**: Im2col with sub-byte element sizes requires careful nibble/bit handling (gap: D4, P2).

---

## 11. References

- Chetlur et al., "cuDNN: Efficient Primitives for Deep Learning", arXiv 2014
- Geng et al., "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration", DAC 2021
- Chen et al., "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices", JETCAS 2019
- Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks", CVPR 2016 (Winograd)
- CS231n Convolution Notes: https://cs231n.github.io/convolutional-networks/
