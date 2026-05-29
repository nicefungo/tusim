# BF16 Support & Subnormal Handling

> **Gap coverage:** D1 (multi-precision), D3 (BF16), D7 (subnormal handling)  
> **Status:** Implemented  
> **Date:** 2026-05-29  
> **Files:** `tu_cmodel/tu_precision.{h,c}`, `tests/test_bf16_subnormal.c`

## Overview

This adds two capabilities to the TinyTU cmodel:

1. **BF16 (bfloat16) support** — a truncated FP32 format (1-8-7: 1 sign, 8 exponent, 7 mantissa) used by Google TPU, NVIDIA A100/H100, and Intel Habana for ML training. BF16 preserves the FP32 exponent range while reducing mantissa precision, making it ideal for mixed-precision training.

2. **Configurable subnormal handling** — the ability to switch between full IEEE 754 subnormal support and flush-to-zero (FTZ), matching the behavior of real hardware where FTZ is commonly enabled for performance (TPU, GPU tensor cores).

## BF16 (bfloat16)

### Format

```
BF16:  [15]  [14:7]    [6:0]
        sign  exponent  mantissa
        1     8         7

FP32:  [31]  [30:23]   [22:0]
        sign  exponent  mantissa
        1     8         23
```

BF16 has the **same exponent range as FP32** (8 exponent bits) but only 7 mantissa bits (vs. 23 for FP32). This matters for ML because:

- **Overflow/underflow behavior matches FP32** — no sudden range loss like FP16
- **~2 decimal digits of precision** — sufficient for gradient accumulation where noise averages out
- **Exact conversion to/from FP32** — BF16 → FP32 is simply a left-shift by 16 bits (zero pad mantissa)

### Comparison with FP16

| Property | FP16 | BF16 |
|----------|------|------|
| Format | 1-5-10 | 1-8-7 |
| Exponent range | ±2^15 ≈ 65504 max | ±2^127 ≈ 3.4e38 max |
| Mantissa precision | ~3.3 decimal digits | ~2.1 decimal digits |
| Subnormal range | Down to 2^-24 ≈ 6e-8 | Down to 2^-133 ≈ 9e-41 |
| Training stability | Overflow risk (gradients clip at 65504) | Matches FP32 range |
| Hardware adoption | Universal | TPU, NVIDIA TensorCore, Habana |

### API

```c
#include "tu_cmodel/tu_precision.h"

bf16_t h = tu_fp32_to_bf16(3.14159f);    // FP32 → BF16
fp32_t f = tu_bf16_to_fp32(h);           // BF16 → FP32 (exact)

// Batch conversion
tu_fp32_to_bf16_buffer(src, dst, n);     // FP32[] → BF16[]
tu_bf16_to_fp32_buffer(src, dst, n);     // BF16[] → FP32[]
```

### Conversion Semantics

- **FP32 → BF16:** Round-to-nearest-even (RNE). The bottom 16 bits of the FP32 mantissa are rounded away using the standard `(bits + 0x7FFF + ((bits >> 16) & 1)) >> 16` pattern. Overflow saturates to ±Inf.
- **BF16 → FP32:** Exact. The BF16 value is left-shifted 16 bits and interpreted as FP32. No precision loss — the 7-bit mantissa is padded with 16 zero bits.

### Precision Registry

BF16 is registered as `TU_PREC_BF16` in the precision type system:

```c
const tu_precision_desc_t *bf16 = tu_precision_get(TU_PREC_BF16);
// bf16->name = "bf16", bf16->elem_bytes = 2
// bf16->to_fp32(src) → FP32 value
// bf16->from_fp32(val, dst) → BF16 value
```

This enables BF16 to be used wherever the precision registry is consulted (MMA dispatch, DMA conversion, elementwise pipeline).

### BF16 in the MMA Pipeline

Currently, BF16 inputs go through a two-step conversion:
1. BF16 → FP32 (exact)
2. FP32 → FP16 (quantized with RNE)

The MMA operates in FP16/FP32 as before. A **native BF16 MMA path** (bypassing FP16, keeping BF16 precision in the systolic array) is a future optimization (gap P1.1).

## Subnormal Handling

### Motivation

Real hardware often **flushes subnormals to zero** (FTZ) for performance:
- Subnormal arithmetic is 10-100× slower than normal arithmetic on CPUs
- GPU tensor cores may not support subnormals at all
- TPUs always flush-to-zero

The cmodel must model both behaviors to match different hardware targets and for numerical studies (e.g., does FTZ cause accuracy issues for a given model?).

### Mode

```c
typedef enum {
    TU_SUBNORMAL_FULL = 0,   // Full IEEE 754 subnormal support
    TU_SUBNORMAL_FLUSH = 1,  // Flush-to-zero
} tu_subnormal_mode_t;

// Query and set mode
tu_subnormal_mode_t mode = tu_get_subnormal_mode();
tu_set_subnormal_mode(TU_SUBNORMAL_FULL);
```

**Default:** `TU_SUBNORMAL_FLUSH` (matches common hardware behavior).

### Behavior

| Mode | Input: 1e-6 (FP16 subnormal) | Input: 6.1e-5 (FP16 normal) |
|------|------------------------------|-----------------------------|
| FLUSH | → 0.0 | → ~6.1e-5 (normal) |
| FULL  | → ~9.5e-7 (subnormal preserved) | → ~6.1e-5 (normal) |

### Scope

Subnormal handling currently applies to **FP16 conversion** (`tu_fp32_to_fp16`). The FP32 accumulator path (MMA, elementwise) uses IEEE 754 FP32 throughout, which always has full subnormal support in software (libm/hardware FPU).

### Configuration

```yaml
# tu_config.yaml
tu_config:
  precision:
    subnormal_mode: "flush"  # "flush" | "full"
```

At compile time, `TU_FP16_SUBNORMAL_FLUSH` in `tu_config.h` sets the default. At runtime, `tu_set_subnormal_mode()` overrides.

## Test Coverage

12 new tests in `tests/test_bf16_subnormal.c`:

| Test | What It Verifies |
|------|-----------------|
| BF16 round-trip (1.0, 0.0) | Exact values survive BF16→FP32 |
| BF16 batch conversion (16 el) | Batch API correctness, ~2% relative error for arbitrary floats |
| BF16 NaN/Inf | Special values preserved through BF16 round-trip |
| BF16 precision boundary | Powers of 2 round-trip exactly |
| BF16 precision loss | Non-exact values lose ~0.4% precision (expected) |
| Subnormal default flush | Default mode flushes to zero |
| Subnormal full mode | Full mode preserves tiny values (5e-6) |
| Subnormal mode switch | get/set roundtrip |
| Subnormal boundary | Normal values survive in both modes; subnormals flush in FTZ |
| BF16→FP16 pipeline | BF16→FP32→FP16 preserves key identity matrix values |
| Precision registry | FP16/F32/BF16 entries correct with proper conversion |

## Design Trade-offs

| Decision | Rationale | Alternative |
|----------|-----------|-------------|
| **Global subnormal mode** | Simpler API; matches hardware where FTZ is a processor-wide flag | Per-operation mode (more flexible, complex) |
| **BF16→FP16 for MMA** | Reuses existing FP16 MMA path; BF16 native path later | Direct BF16 MMA (more work, higher priority later) |
| **Static precision registry** | Compile-time dispatch; no vtable overhead | Dynamic registration (flexible, slower) |
| **BF16 as uint16_t alias** | Zero-overhead; matches hardware register width | Opaque struct (safer, less convenient) |

## Future Work

1. **Native BF16 MMA** — systolic array operating directly on BF16 weights/activations, avoiding the FP16 conversion step (P1.1)
2. **BF16 elementwise path** — elementwise ops on BF16 data in SRAM (P1.5 extension)
3. **Stochastic rounding mode** — add `TU_FP16_ROUNDING_STOCHASTIC` alongside RNE and RTZ (D6/P2)
4. **FP8 support** — E4M3 and E5M2 formats for forward and gradient paths (D4/P2)
5. **INT8/INT4 quantization paths** — zero-point + scale quantization with integer MMA (D2/P1)

## References

- IEEE 754-2008: Binary16 and Binary32 interchange formats
- Google TPU: bfloat16 "Brain Floating Point" format — Cloud TPU documentation
- NVIDIA A100: BF16 TensorCore support — CUDA Programming Guide §I.7
- Intel DL Boost: BF16 via AVX-512 — Intel Architecture Instruction Set Extensions
- "Flush-to-zero in Deep Learning: Does It Matter?" — Zamirai et al., 2021
