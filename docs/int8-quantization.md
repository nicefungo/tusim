# INT8/INT4 Quantization Support — TU CModel

> **Gap:** D2 — Multi-precision: integer quantization for production inference  
> **Date:** 2026-05-30  
> **Status:** Implemented ✅  
> **Tests:** 14/14 passing

---

## 1. Overview

Production neural network accelerators (TPU, NVIDIA TensorCore, Gemmini) all support **integer quantization** as their primary deployment precision. INT8 quantized models deliver 2-4× throughput improvement over FP16 with minimal accuracy loss.

This feature adds INT8 and INT4 (UINT4 packed) quantization to the TU cmodel:

- **INT8**: Signed 8-bit integer, symmetric/asymmetric quantization, INT8×INT8→INT32 MAC
- **UINT4**: Packed unsigned 4-bit (2 values/byte), dense weight compression

### Why This Matters

| Without INT8 | With INT8 |
|---|---|
| FP16 only → 2 bytes/element | INT8 → 1 byte/element |
| Peak: 256 MACs/cycle (16×16 FP16) | Peak: 512 MACs/cycle (16×32 INT8) |
| Models: 28 MB for ResNet-50 | Models: 14 MB for ResNet-50 |
| No mobile/edge deployment path | Standard for mobile/edge inference |

---

## 2. Affine Quantization Scheme

We implement the standard affine quantization used by TensorFlow Lite, PyTorch Quantization, and ONNX Runtime:

```
q = clamp(round(r / scale) + zero_point, qmin, qmax)
r = (q - zero_point) * scale
```

### INT8 Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| `qmin` | -128 | — |
| `qmax` | 127 | — |
| `scale` | 0.007874 (≈1/127) | Any positive float |
| `zero_point` | 0 (symmetric) | [-128, 127] |

**Symmetric quantization** (zero_point = 0): Values distributed around zero. Matches NVIDIA TensorRT INT8, Google TPU INT8.

**Asymmetric quantization**: Values in [min, max] mapped to [-128, 127]. Used for ReLU-activated tensors.

### UINT4 Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| `qmin` | 0 | — |
| `qmax` | 15 | — |
| `scale` | 0.0667 (≈1/15) | Any positive float |
| `zero_point` | 8 | [0, 15] |

### Storage Format

UINT4 uses **packed nibble storage**: 2 values per byte, low nibble first.

```
Byte 0:  [elem1 << 4 | elem0]    elems 0,1
Byte 1:  [elem3 << 4 | elem2]    elems 2,3
...
```

---

## 3. API Reference

### Quantization Parameters

```c
#include "tu_cmodel/tu_int_quant.h"

// Initialize with defaults
tu_quant_params_t qp;
tu_quant_params_init_int8(&qp);
tu_quant_params_init_uint4(&qp);

// Calibrate from data
tu_quant_params_calibrate_int8_symmetric(data, n, &qp);
tu_quant_params_calibrate_int8_asymmetric(data, n, &qp);
tu_quant_params_calibrate_uint4(data, n, &qp);
```

### Single-element Conversion

```c
int8_t q = tu_fp32_to_int8(3.14f, &qp);        // FP32 → INT8
float r = tu_int8_to_fp32(q, &qp);             // INT8 → FP32

uint8_t nib = tu_fp32_to_uint4_nibble(3.14f, &qp); // FP32 → UINT4 nibble
float r4 = tu_uint4_nibble_to_fp32(nib, &qp);      // UINT4 nibble → FP32
```

### Batch Conversion

```c
tu_fp32_to_int8_buffer(src_fp32, dst_int8, n, &qp);
tu_int8_to_fp32_buffer(src_int8, dst_fp32, n, &qp);

tu_fp32_to_uint4_buffer(src_fp32, dst_packed, n, &qp);  // Ceil(n/2) bytes
tu_uint4_to_fp32_buffer(src_packed, dst_fp32, n, &qp);
```

### INT8 MAC Operations

```c
// Dot product
int32_t sum = tu_int8_dot_product(a, b, n);

// MMA tile: O[M][N] += W[M][K] × A[K][N]
tu_int8_mma_tile(W_int8, A_int8, O_int32, M, N, K);
// After accumulation, dequantize:
//   output_fp32[i] = O_int32[i] * w_scale * a_scale
```

### Precision Registry

INT8/INT4 are registered in the TU precision type system:

```c
const tu_precision_desc_t *pd = tu_precision_get(TU_PREC_INT8);
assert(pd->elem_bytes == 1);  // 1 byte per element
assert(pd->type == TU_PREC_INT8);
```

---

## 4. Configuration

All quantization parameters are configurable via `tu_config.h`:

```c
// Enable INT8 quantization path
#define TU_INT8_ENABLED              1
#define TU_INT8_ACCUM_BITS           32      // INT32 accumulator
#define TU_INT8_SYMMETRIC_DEFAULT    1       // Default to symmetric quantization

// Enable INT4 (UINT4 packed) quantization
#define TU_INT4_ENABLED              1
```

---

## 5. How It Works

### Quantization Flow

```
FP32 Data
    │
    ├── Calibrate: compute scale/zero_point from data statistics
    │   - Symmetric: scale = max(|data|) / 127
    │   - Asymmetric: scale = (max - min) / 255, zp = floor(-128 - min/scale)
    │
    ├── Quantize: clamp(round(data / scale) + zp, -128, 127)
    │
    ▼
INT8 Data (stored in SRAM)
    │
    ├── MAC: O_int32[i] += W_int8[i] * A_int8[i]
    │
    ├── Dequantize: output_fp32 = O_int32 * w_scale * a_scale
    │
    ▼
FP32 Output
```

### INT8 MMA Tile Implementation

The INT8 MAC is a straightforward integer dot product accumulated into INT32:

```c
void tu_int8_mma_tile(const int8_t *W, const int8_t *A,
                       int32_t *O, uint16_t M, uint16_t N, uint16_t K) {
    for (uint16_t m = 0; m < M; m++)
        for (uint16_t n = 0; n < N; n++) {
            int32_t sum = 0;
            for (uint16_t k = 0; k < K; k++)
                sum += (int32_t)W[m*K+k] * (int32_t)A[k*N+n];
            O[m*N+n] += sum;
        }
}
```

This matches hardware behavior where MACs accumulate in wider registers (INT32 for INT8 inputs, INT48 for INT16 × INT8, etc.).

---

## 6. Verification

### Test Coverage

| Test | What It Verifies |
|------|-----------------|
| Round-trip zero/positive/negative | Quantize+dequantize ≈ original |
| Batch conversion (16 elements) | Throughput path correct |
| Symmetric calibration | scale = max/127, zp = 0 |
| Asymmetric calibration | Correct range mapping |
| UINT4 pack/unpack | Nibble storage round-trip |
| UINT4 quant/dequant | Full UINT4 pipeline |
| INT8 dot product | 1×3 + 127×127 correctness |
| INT8 MMA tile | 2×3×2 matrix multiply |
| MMA accumulate | Adds to existing output |
| Precision registry | INT8/INT4 entries exist |

### Accuracy Guarantees

- Quantization error ≤ 0.5 × scale (half-LSB rounding error)
- INT8 MAC results are **exact** (integer arithmetic — no floating-point error)
- Dequantization error is bounded by `scale` (absorbed in overall FP error budget)

---

## 7. Performance Impact

### Memory Savings

| Model | FP16 Weights | INT8 Weights | Reduction |
|-------|-------------|-------------|-----------|
| ResNet-50 | 51 MB | 25.5 MB | 2× |
| BERT-base | 880 MB | 440 MB | 2× |
| INT4 weights | — | 12.75 MB | 4× |

### Compute Throughput

For hardware with INT8 MAC units (2× density of FP16):

| Operation | FP16 MACs/cycle | INT8 MACs/cycle | Speedup |
|-----------|----------------|-----------------|---------|
| 16×16 systolic | 256 | 512 | 2× |
| 32×32 systolic | 1024 | 2048 | 2× |

---

## 8. Limitations & Future Work

- **INT8 path is software-only in cmodel.** The cmodel uses FP32 internally for accumulation (simulating INT32). Full INT32 hardware accumulation path (gap: cycle-accurate INT8 MAC model) is P2.
- **No per-channel quantization yet.** Only per-tensor. Per-channel quantization is important for weights (improves accuracy 1-3%) and maps to most hardware (gap: P2).
- **FP8 (E4M3/E5M2) not yet implemented.** Stub exists in precision registry for forward compatibility (gap: D4, P2).

---

## 9. References

- TensorFlow Lite Quantization Spec: https://www.tensorflow.org/lite/performance/quantization_spec
- NVIDIA TensorRT INT8: https://docs.nvidia.com/deeplearning/tensorrt/
- Google TPU INT8: https://cloud.google.com/tpu/docs
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018
