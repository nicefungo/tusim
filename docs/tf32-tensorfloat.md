# TU CModel — TF32 (TensorFloat-32)

> **Gap ID:** D3 (No BF16/TF32 → TF32 support)
> **Priority:** P1 (High)
> **Date:** 2026-06-02
> **Heartbeat:** Cycle N+2

---

## What Changed

TF32 (TensorFloat-32) has been added as a new precision type in the TU cmodel's multi-precision system. TF32 is NVIDIA's 19-bit floating-point format introduced with Ampere (A100) and used across H100 and Blackwell GPUs for mixed-precision training.

### Key Features

1. **1-8-10 format** — 1 sign, 8 exponent, 10 mantissa bits (stored in 32-bit word)
2. **Full FP32 dynamic range** — Same 8-bit exponent as FP32, no range reduction
3. **~3.3 decimal digits precision** — 10-bit mantissa vs FP32's 23-bit (~7 decimal digits)
4. **FP32 compatible storage** — TF32 values are valid FP32 with 13 low mantissa bits zeroed
5. **All rounding modes** — RNE (default), RTZ (truncate), Stochastic
6. **Mixed precision bridges** — TF32 ↔ FP16, TF32 ↔ BF16 via FP32 intermediate
7. **Precision registry integration** — Registered as `TU_PREC_TF32` in the pluggable precision system

---

## Why This Matters

TF32 fills a critical gap in the precision hierarchy between BF16 and FP32:

| Format | Sign | Exponent | Mantissa | Total Bits | Range | Precision |
|--------|------|----------|----------|------------|-------|-----------|
| FP8 E4M3 | 1 | 4 | 3 | 8 | 2^-6 to 448 | ~1 digit |
| FP8 E5M2 | 1 | 5 | 2 | 8 | 2^-14 to 57344 | ~0.7 digit |
| FP16 | 1 | 5 | 10 | 16 | 6e-8 to 65504 | ~3.3 digits |
| BF16 | 1 | 8 | 7 | 16 | 1e-38 to 3e38 | ~2.1 digits |
| **TF32** | **1** | **8** | **10** | **19** (in 32) | **1e-38 to 3e38** | **~3.3 digits** |
| FP32 | 1 | 8 | 23 | 32 | 1e-38 to 3e38 | ~7.2 digits |

**TF32's superpower:** Full FP32 dynamic range with reduced precision. NVIDIA found that training convergence is insensitive to the mantissa reduction (most gradients are noise-limited, not precision-limited), while the MAC unit area scales with mantissa width squared. TF32 MACs are ~1/4 the area of FP32 MACs while delivering identical training convergence for most workloads.

### Concrete Use Cases

- **Mixed-precision training:** Matrix multiplies in TF32, accumulation in FP32. Matches NVIDIA's "TF32 Tensor Core" mode.
- **Design space exploration:** Evaluate precision/area tradeoffs for custom accelerators.
- **Reference comparison:** Validate TF32 results against BF16 and FP16 to choose the right format for a given workload.
- **Compiler integration:** The ONNX compiler can target TF32 for GEMM operations, trading ~2% accuracy for ~4× MAC density.

---

## How It Works

### Format

```
TF32 (19 effective bits, 32-bit storage):
  bit 31      : sign (0=positive, 1=negative)
  bits 30-23  : exponent (bias=127, same as FP32)
  bits 22-13  : mantissa (10 bits, "1." implied for normal numbers)
  bits 12-0   : always zero (13 bits dropped from FP32)
```

### Conversion: FP32 → TF32

```c
tf32_t tu_fp32_to_tf32(float v);
```

1. Extract FP32 sign, exponent, and 23-bit mantissa
2. Take the top 10 mantissa bits: `mantissa >> 13`
3. Round bit 12 (with sticky bits 11:0) using the configured rounding mode
4. Handle mantissa overflow (carry into exponent)
5. Pack: `(sign << 31) | (exp << 23) | (tf32_mant << 13)`

**Rounding modes:**
- **RNE (default):** Round to nearest even — round up if bit 12=1 AND (sticky > 0 OR LSB of 10-bit mantissa=1)
- **RTZ:** Truncate — always drop low bits
- **Stochastic:** Round up with probability proportional to the fractional value

### Conversion: TF32 → FP32

```c
float tu_tf32_to_fp32(tf32_t v);
```

This is a no-op! TF32 values are already valid FP32 with 13 low mantissa bits forced to zero. Just reinterpret the 32-bit word as float.

### Batch Conversions

```c
void tu_fp32_to_tf32_buffer(const float *src, tf32_t *dst, size_t n);
void tu_tf32_to_fp32_buffer(const tf32_t *src, float *dst, size_t n);
```

### Mixed Precision Bridges

```c
uint16_t tu_tf32_to_fp16(tf32_t v);   // TF32 → FP16
tf32_t   tu_fp16_to_tf32(uint16_t v); // FP16 → TF32
uint16_t tu_tf32_to_bf16(tf32_t v);   // TF32 → BF16
tf32_t   tu_bf16_to_tf32(uint16_t v); // BF16 → TF32
```

---

## Configuration

TF32 is always available as a precision type — no compile-time flag needed.

The rounding mode is configured globally:

```c
tu_set_rounding_mode(TU_ROUND_RNE);        // Default
tu_set_rounding_mode(TU_ROUND_RTZ);        // Truncation
tu_set_rounding_mode(TU_ROUND_STOCHASTIC); // Training
```

### Precision Registry

TF32 is registered as `TU_PREC_TF32` (element size: 4 bytes, name: "tf32"):

```c
const tu_precision_desc_t *desc = tu_precision_get(TU_PREC_TF32);
// desc->to_fp32(&tf32_val)   → float
// desc->from_fp32(3.14f, &d) → tf32_t
```

---

## Verification

### Test Suite: 25/25 tests passing

| Test | What It Verifies |
|------|-----------------|
| 0.0/-0.0 roundtrip | Zero preserves sign |
| 1.0/-1.0/0.5/2.0 roundtrip | Powers of two are exact |
| PI (3.14159) | 10-bit precision gives ~0.1% relative error |
| max normal (~3.4e38) | Full FP32 dynamic range |
| min normal (~1.18e-38) | Small normal numbers |
| subnormal (1e-38) | Subnormal range preserved |
| +Inf/-Inf/NaN | Special values preserved |
| RNE rounding (3.3) | Round-to-nearest-even behavior |
| RTZ truncation (3.7) | Round-toward-zero behavior |
| Stochastic average | ~50% round-up rate for half-bit values |
| Batch 100 values | Bulk conversion correctness |
| TF32 → FP16 bridge | Cross-precision conversion |
| FP16 → TF32 bridge | Cross-precision conversion |
| TF32 → BF16 bridge | Cross-precision conversion |
| BF16 → TF32 bridge | Cross-precision conversion |
| Registry lookup | Correct type, size, name |
| Registry convert | from_fp32 → to_fp32 roundtrip |
| Registry PI | Precision through registry path |
| Range query | min/max normal match FP32 |

### Run Tests

```bash
make test-tf32    # 25/25 tests pass
```

---

## Future Extensions

- **TF32 MAC operations:** Add `tu_mac_tf32()` to the compute engine for native TF32 matrix multiplies (currently TF32 is a storage/transfer format; compute still upcasts to FP32)
- **Configurable TF32 precision modes:** Allow tuning the mantissa width (8-bit, 9-bit, 10-bit) for design space exploration
- **Integration with sparsity:** TF32 + 2:4 structured sparsity for 8× effective throughput

---

## Files

| File | Change |
|------|--------|
| `tu_cmodel/tf32.h` | New — TF32 interface (conversions, bridges, range) |
| `tu_cmodel/tf32.c` | New — TF32 implementation (170 LOC) |
| `tu_cmodel/tu_precision.h` | Add `TU_PREC_TF32 = 7` to precision enum |
| `tu_cmodel/tu_precision.c` | Add TF32 to precision registry builtins |
| `Makefile` | Add `tf32.o` to library, add `test-tf32` target |
| `tests/test_tf32.c` | New — 25 TF32 tests |
| `docs/tf32-tensorfloat.md` | This document |
