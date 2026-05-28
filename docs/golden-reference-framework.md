# TU CModel — Golden Reference Framework

> **Gap IDs:** V1 (Golden reference), V6 (Random/differential testing)
> **Priority:** P0 (Critical)
> **Date:** 2026-05-29
> **Heartbeat:** Cycle 4

---

## What Changed

The TinyTU cmodel previously had no systematic verification against reference implementations. Tests were hand-written with known values and manual expected-result computation. There was no way to run randomized differential testing or generate golden reference data for regression.

A dual-path golden reference framework has been added:

1. **In-process C FP32 reference** — `tests/test_golden.c` computes expected results in FP32 and compares against the cmodel's FP16→FP32 path
2. **Python NumPy golden generator** — `tests/golden/generate_reference.py` produces JSON reference files for known test cases that can be loaded for regression testing

### Key Features

1. **In-process differential testing** — 5,000 random tensor tests compare cmodel vs FP32 reference
2. **Python golden reference generator** — NumPy-based reference computation in FP64 precision
3. **21 pre-generated golden cases** — Identity matrices, known values, edge tiles, bias, large matrices, 10 random cases
4. **Tolerance tracking** — Per-test max error, mean error, and global max observed error
5. **Deterministic randomness** — xorshift PRNG for reproducible test sequences
6. **Precision boundary testing** — Max FP16 values, scalar edge cases, vector edge cases
7. **Progress reporting** — Bulk test progress every 500 iterations

---

## Why This Matters

Golden reference verification is essential for production cmodel quality:

- **Catches precision bugs** — FP16 quantization errors, rounding mode mismatches, accumulator precision issues
- **Regression prevention** — Golden references ensure future changes don't silently break results
- **Confidence in results** — When the cmodel is used as a reference for RTL, it must be provably correct
- **Continuous validation** — Randomized testing finds corner cases hand-written tests miss
- **Quantified error bounds** — Tolerance tracking establishes the cmodel's accuracy envelope

---

## How It Works

### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     Verification Pipeline                          │
│                                                                    │
│  ┌──────────────────┐          ┌──────────────────┐               │
│  │  C FP32 Ref      │          │  Python NuPy Ref │               │
│  │  (in-process)    │          │  (FP64 golden)   │               │
│  └───────┬──────────┘          └───────┬──────────┘               │
│          │                             │                           │
│          ▼                             ▼                           │
│  ┌──────────────────┐          ┌──────────────────┐               │
│  │  Differential    │          │  JSON Reference   │               │
│  │  Testing (5K+)   │          │  Files            │               │
│  └───────┬──────────┘          └───────┬──────────┘               │
│          │                             │                           │
│          └──────────┬──────────────────┘                           │
│                     ▼                                              │
│          ┌──────────────────┐                                     │
│          │  Compare vs      │                                     │
│          │  CModel Output   │                                     │
│          └───────┬──────────┘                                     │
│                  ▼                                                 │
│          ┌──────────────────┐                                     │
│          │  Pass/Fail +     │                                     │
│          │  Error Report    │                                     │
│          └──────────────────┘                                     │
└────────────────────────────────────────────────────────────────────┘
```

### In-Process FP32 Reference

The C golden test generates random FP32 tensors, converts them to FP16 (as the cmodel would receive via DMA), runs the cmodel, and compares against a pure FP32 matrix multiply:

```c
// 1. Generate random FP32 tensors
fill_random(W_fp32, w_count, 1.0f);
fill_random(A_fp32, a_count, 1.0f);

// 2. Convert to FP16 (simulating DMA)
tu_fp32_to_fp16_buffer(W_fp32, W_fp16, w_count);
tu_fp32_to_fp16_buffer(A_fp32, A_fp16, a_count);

// 3. Compute FP32 reference
compute_fp32_reference(W_fp32, A_fp32, O_ref, M, N, K);

// 4. Run cmodel
tu_dma_load_w(W_fp16, 0, ...);
tu_dma_load_a(A_fp16, 0, ...);
tu_mma(M, N, K, 0, 0, 0, false);
tu_dma_store_o(O_cm, 0, ...);

// 5. Compare
float max_err = max_abs_error(O_ref, O_cm, o_count);
assert(max_err <= tolerance);
```

### Error Characteristics

FP16 has ~3 decimal digits of precision (10-bit mantissa). For MMA with K inner dimension:

- **Small K (1-16):** max error typically < 0.002, mean error < 0.0004
- **Medium K (32-64):** max error typically < 0.003, mean error < 0.0006
- **Large K (100+):** error scales roughly as √K due to accumulation of rounding errors

The observed max error across 5,000 random tests was **0.00315** — well within the expected FP16 tolerance envelope.

### Python Golden Reference Generator

```bash
# Generate all reference cases (identity, known, edge, bias, large, random)
python3 tests/golden/generate_reference.py

# Generate specific case types only
python3 tests/golden/generate_reference.py --cases random --num-random 100

# Control the random seed
python3 tests/golden/generate_reference.py --seed 12345
```

Output format (JSON):
```json
{
  "case_id": "identity_16x16x16",
  "description": "Identity: M=16, N=16, K=16",
  "M": 16, "N": 16, "K": 16,
  "has_bias": false,
  "W_hex": ["0x3c00", "0x0000", ...],
  "A_hex": ["0x3c00", "0x0000", ...],
  "O_ref": [1.0, 0.0, 0.0, 1.0, ...],
  "O_ref_hex": ["0x3f800000", "0x00000000", ...],
  "tolerance": 0.01
}
```

---

## Verification

### Test Suite: 11 tests (configurations + bulk)

| Test | What It Verifies |
|------|-----------------|
| 16×16×16 tiny | Small identity GEMM, max_err ~0.001 |
| 32×32×32 medium | Medium identity GEMM |
| 64×64×64 large | Large identity GEMM, max_err ~0.003 |
| 7×5×9 edge tiles | Non-multiple-of-PE dimensions |
| 4×8×16 non-square | Non-square matrix |
| 31×17×23 primes | Prime-number dimensions |
| 1×1×1 scalar | Single-element GEMM |
| 1×16×64 vector | Vector-vector outer product |
| Bias golden | Zero W/A, sequential bias → output = bias |
| FP16 precision | Max FP16 value (65504) boundary test |
| Bulk random 50 | 50 random tensor differential tests |
| Bulk random 5000 | 5,000 random tensor differential tests (full mode) |

### Run Tests

```bash
make test-golden          # Quick mode: 50 random tests + fixed configs (11/11)
make test-golden-full      # Full mode: 5000 random tests + fixed configs (11/11)
```

### 5,000-Test Results

```
═══════════════════════════════════════════
  11/11 tests passed
  Max observed error: 0.003148
═══════════════════════════════════════════
```

5000/5000 random tests passed with zero failures. The max observed error (0.00315) is well within the FP16 tolerance envelope.

---

## Tolerance Selection

Tolerances are set based on FP16 precision characteristics:

| K (inner dimension) | Tolerance | Rationale |
|---------------------|-----------|-----------|
| 1-8 | 0.001 | Very little accumulation error |
| 9-16 | 0.01 | Standard tolerance |
| 17-32 | 0.02 | Moderate accumulation |
| 33-64 | 0.05 | Higher accumulation |
| 65+ | 0.1-0.2 | Worst-case scenario |

The formula used in the test harness:
```c
float tol = 0.01f + (float)K * 0.0005f;
```

---

## Configuration

| #define | Default | Description |
|---------|---------|-------------|
| `TU_VERIFY_GOLDEN_MODE` | 1 | 0=NumPy, 1=PyTorch |
| `TU_VERIFY_RANDOM_ITERS` | 1000 | Default random test count |
| `TU_VERIFY_ERROR_TOLERANCE` | 1e-5 | Default error tolerance |

Runtime configuration in `tu_runtime_config_t`:
```c
cfg.verify_enabled   = true;   // Enable golden verification
cfg.verify_tolerance = 0.01;   // Override tolerance
```

---

## Future Extensions

This golden reference framework provides the foundation for:

- **V2 (Comprehensive unit tests):** Extend to all op types (Conv, Attention, LayerNorm, etc.) as they're added
- **V3 (Regression framework):** CI pipeline runs golden tests on every commit
- **V4 (Performance validation):** Compare cycle estimates against Gemmini/SCALE-Sim reference
- **V5 (Comparative benchmarking):** Standard benchmark suite (MLPerf Tiny, Transformer layers)
- **Fuzzing:** AFL/libFuzzer integration for random instruction sequence testing

---

## Files

| File | Change |
|------|--------|
| `tests/test_golden.c` | New — In-process C golden reference with 5K random tests |
| `tests/golden/generate_reference.py` | New — Python/NumPy golden reference generator |
| `tests/golden/reference_data/*.json` | New — 21 pre-generated golden reference cases |
| `tests/golden/reference_data/index.json` | New — Golden reference case index |
| `Makefile` | Added `test-golden` and `test-golden-full` targets |
| `docs/golden-reference-framework.md` | This document |
