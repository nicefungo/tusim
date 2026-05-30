# TU FP8 Implementation — D4

> **Gap:** D4 — FP8 E4M3/E5M2 support  
> **Status:** Implemented  
> **Date:** 2026-05-31  
> **Files:** `tu_cmodel/fp8.h`, `tu_cmodel/fp8.c`, `tests/test_fp8.c`

## What This Is

Complete FP8 support for both OCP Microscaling Formats (MX) compliant types used in NVIDIA Hopper (H100), AMD MI300, and Intel Gaudi accelerators. Replaces the previous stub implementation that passed FP8 values through as raw uint8→float.

## Why It Matters

### Gap D4 (from PRODUCTION_TU_REDESIGN.md)

> **Severity:** Medium | **Priority:** P2  
> **Current:** No FP8 support (stub pass-through)  
> **Target:** FP8 E4M3 (forward) and E5M2 (gradient) per NVIDIA Hopper / OCP spec; emerging standard

FP8 is the next-generation quantization format for large-scale ML inference and training. Key adopters:

| Accelerator | FP8 Support | Use Case |
|---|---|---|
| **NVIDIA H100** | E4M3 + E5M2 | Training (E4M3 forward, E5M2 backward) |
| **NVIDIA B200** | E4M3 + E5M2 | 2× throughput vs FP16 |
| **AMD MI300** | E4M3 + E5M2 | ROCm FP8 support |
| **Intel Gaudi 3** | E4M3 + E5M2 | FP8 inference/training |
| **Google TPU v5** | FP8 (proprietary) | Cloud inference |

Without proper FP8 support, the cmodel cannot model modern accelerator workloads. The 8-bit format provides 2× memory bandwidth and compute throughput vs FP16, making it critical for large model deployment (LLMs, diffusion models).

## How It Works

### Two Complementary Formats

```
FP8 E4M3 (1-4-3):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ S │  E3  │  E2  │  E1  │  E0  │  M2  │  M1  │  M0  │
└───┴───┴───┴───┴───┴───┴───┴───┘
  1 sign, 4 exponent (bias=7), 3 mantissa

Range:       [0.015625, 240.0] normal
Subnormals:  [0.001953, 0.015625)
No infinities — overflow → NaN
Primary use: FORWARD PASS (better precision per bit)

FP8 E5M2 (1-5-2):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ S │  E4  │  E3  │  E2  │  E1  │  E0  │  M1  │  M0  │
└───┴───┴───┴───┴───┴───┴───┴───┘
  1 sign, 5 exponent (bias=15), 2 mantissa

Range:       [6.1e-5, 57344.0] normal
Subnormals:  [1.53e-5, 6.1e-5)
Supports infinities
Primary use: BACKWARD PASS (wider dynamic range)
```

### Encoding Tables

#### E4M3

| Bit Pattern | Value | Notes |
|---|---|---|
| `s0000_000` | ±0 | Zero |
| `s0000_xxx` (xxx≠000) | ±2^-6 × xxx/8 | Subnormal |
| `s0001_xxx` … `s1110_xxx` | ±2^(exp-7) × (1+xxx/8) | Normal |
| `s1111_111` | NaN | Canonical quiet NaN |
| `s1111_0xx` … `s1111_110` | NaN | Also NaN |
| Max normal: `0_1110_111` | 240.0 | 2^7 × (1+7/8) |

#### E5M2

| Bit Pattern | Value | Notes |
|---|---|---|
| `s00000_00` | ±0 | Zero |
| `s00000_xx` (xx≠00) | ±2^-14 × xx/4 | Subnormal |
| `s00001_xx` … `s11110_xx` | ±2^(exp-15) × (1+xx/4) | Normal |
| `s11111_00` | ±∞ | Infinity |
| `s11111_{01,10,11}` | NaN | |
| Max normal: `0_11110_11` | 57344.0 | 2^15 × (1+3/4) |

### API

```c
// E4M3
float   tu_fp8_e4m3_to_fp32(uint8_t v);
uint8_t tu_fp32_to_fp8_e4m3(float v);
void    tu_fp32_to_fp8_e4m3_buffer(const float *src, uint8_t *dst, size_t n);
void    tu_fp8_e4m3_to_fp32_buffer(const uint8_t *src, float *dst, size_t n);

// E5M2
float   tu_fp8_e5m2_to_fp32(uint8_t v);
uint8_t tu_fp32_to_fp8_e5m2(float v);
void    tu_fp32_to_fp8_e5m2_buffer(const float *src, uint8_t *dst, size_t n);
void    tu_fp8_e5m2_to_fp32_buffer(const uint8_t *src, float *dst, size_t n);

// Cross-precision
uint16_t tu_fp8_e4m3_to_fp16(uint8_t v);
uint16_t tu_fp8_e5m2_to_fp16(uint8_t v);
```

### Precision Registry Integration

Both formats are registered in the precision descriptor table:

```c
const tu_precision_desc_t *desc = tu_precision_get(TU_PREC_FP8_E4M3);
// desc->to_fp32 and desc->from_fp32 work as expected
```

Existing code that iterates over `TU_PREC_COUNT` types automatically picks up both FP8 variants.

### Rounding Mode Support

All three rounding modes work with FP8 conversion:
- **RNE**: Standard round-to-nearest-even
- **RTZ**: Truncation (matches some INT8 quantization paths)
- **Stochastic**: Unbiased probabilistic rounding (shares PRNG with rounding module)

## Configuration

```c
// tu_config.h:
#define TU_PRECISION_FP8_E4M3   (1 << 3)  /* Forward pass format */
#define TU_PRECISION_FP8_E5M2   (1 << 4)  /* Backward pass format */
```

Runtime selection via precision registry.

## Verification

21 tests covering:
- **E4M3**: Zero (±0), 1.0, max normal (240), min normal (2^-6), subnormal (2^-7), overflow→NaN, negative values (-42)
- **E5M2**: Zero, 1.0, max normal (57344), min normal (2^-14), infinity, NaN, subnormal (2^-15)
- **Registry**: Both formats accessible through precision descriptor table
- **Cross-precision**: E4M3→FP16 pass-through
- **Rounding interaction**: RTZ truncation, stochastic unbiased average
- **Batch**: 100-value buffer roundtrip

Run: `make test-fp8`

## How It Changes the Cmodel

### Before (Stub)

```c
// Pass-through: uint8 value treated as integer
static fp32_t prec_fp8_to_fp32(const void *src) {
    return (float)(*(const uint8_t*)src);  // value 0x38 → 56.0, NOT 1.0!
}
```

### After

```c
// Proper OCP MX format decoding
static fp32_t prec_fp8_e4m3_to_fp32(const void *src) {
    return tu_fp8_e4m3_to_fp32(*(const uint8_t*)src);  // value 0x38 → 1.0 ✓
}
```

## Future Extensions

- **Microscaling (MX) block support**: Shared exponent across blocks (MXFP8, MXFP6, MXFP4)
- **FP8 MMA**: Direct FP8×FP8→FP32 MAC in systolic array (currently FP8 is converted to FP32 first)
- **Hardware-accurate FP8 rounding**: Cycle-accurate encoding for specific silicon implementations
