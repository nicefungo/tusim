# TU Configurable Rounding Modes — D6

> **Gap:** D6 — Configurable rounding modes  
> **Status:** Implemented  
> **Date:** 2026-05-31  
> **Files:** `tu_cmodel/rounding.h`, `tu_cmodel/rounding.c`, tests updated

## What This Is

Pluggable rounding mode infrastructure for all TU precision conversions (FP16, BF16, FP8, INT8). Previously, only round-to-nearest-even (RNE) was hard-coded inline in the FP16/BF16 conversion functions. This makes the cmodel's rounding behavior configurable — critical for matching real hardware behavior (some accelerators use truncation) and for training workloads (stochastic rounding eliminates systematic bias from repeated quantization).

## Why It Matters

### Gap D6 (from PRODUCTION_TU_REDESIGN.md)

> **Severity:** Medium | **Priority:** P2  
> **Current:** Round-to-nearest-even only  
> **Target:** Configurable: round-nearest-even, round-toward-zero, stochastic rounding (for training)

Different hardware accelerators use different rounding modes:

| Accelerator | Default Rounding | Rationale |
|---|---|---|
| **NVIDIA TensorCore** | RNE | IEEE 754 compliance, lowest bias for inference |
| **Google TPU** | RNE (v1-v4), RTZ for INT8 quant | INT truncation avoids bias toward larger values |
| **Gemmini (Berkeley)** | Configurable RNE/RTZ | Parameterized for architecture exploration |
| **Training frameworks** | Stochastic | Eliminates bias from repeated quantization in backward pass |

By making rounding configurable, the cmodel can:
1. Match the behavior of a specific target hardware accelerator
2. Support training workloads (stochastic rounding for unbiased gradient quantization)
3. Enable architecture exploration (what rounding mode minimizes accuracy loss at INT4?)

## How It Works

### Three Modes

```
TU_ROUND_RNE        Round-to-Nearest-Even (IEEE 754 default)
                    Ties (exactly 0.5 ULP) round to the even mantissa.
                    Lowest statistical bias. Standard for inference.

TU_ROUND_RTZ        Round-Toward-Zero (truncation)
                    Always discard fractional bits. Zero bias but higher
                    variance. Used in some integer quantization paths.

TU_ROUND_STOCHASTIC Stochastic (probabilistic, unbiased)
                    Round up with probability = fractional_part.
                    Unbiased in expectation. Critical for training.
```

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  Application Code                                   │
│  tu_set_rounding_mode(TU_ROUND_STOCHASTIC);         │
│  fp16_t h = tu_fp32_to_fp16(f);                    │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│  Precision Conversion (tu_precision.c)              │
│  tu_fp32_to_fp16() → tu_round_fp32_to_mantissa()   │
│  prec_bf16_from_fp32() → tu_round_fp32_to_mantissa()│
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│  Rounding Module (rounding.c)                       │
│  ┌──────────────┐ ┌─────────────┐ ┌───────────────┐ │
│  │ RNE Engine   │ │ RTZ Engine  │ │ Stochastic    │ │
│  │              │ │             │ │ Engine        │ │
│  │ tie→even     │ │ always      │ │ PRNG:         │ │
│  │ >half→up     │ │ truncate    │ │ xorshift128+  │ │
│  └──────────────┘ └─────────────┘ └───────────────┘ │
│  Global mode: g_rounding_mode (thread-safe future)  │
└─────────────────────────────────────────────────────┘
```

### API

```c
// Set global rounding mode
void tu_set_rounding_mode(tu_rounding_mode_t mode);

// Get current mode
tu_rounding_mode_t tu_get_rounding_mode(void);

// Seed the stochastic PRNG for reproducibility
void tu_stochastic_set_seed(uint64_t seed);

// Low-level: round a mantissa with current mode
uint32_t tu_round_apply(uint32_t mantissa, int target_bits, int *carry_out);

// Convenience: round FP32 mantissa to narrower width
uint32_t tu_round_fp32_to_mantissa(uint32_t fp32_mantissa, int fp32_exp,
                                    int target_bits, int *carry_out);
```

### Usage Examples

```c
// Inference with truncation (matches some INT8 quantization hardware)
tu_set_rounding_mode(TU_ROUND_RTZ);
fp16_t h = tu_fp32_to_fp16(3.14159f);

// Training with stochastic rounding (unbiased gradients)
tu_set_rounding_mode(TU_ROUND_STOCHASTIC);
tu_stochastic_set_seed(42);  // reproducible training runs
bf16_t grad = tu_fp32_to_bf16(gradient_value);

// Default: RNE for standard inference
tu_set_rounding_mode(TU_ROUND_RNE);
```

### Stochastic Rounding Details

Stochastic rounding uses xorshift128+ (fast, high-quality PRNG). For a value with fractional part `f` in [0, 1):

```
round_up = (rand() < f)
```

This means:
- `f = 0.3` → 30% chance of rounding up, 70% chance of rounding down
- `f = 0.0` → always rounds down (exact integer)
- `f = 0.999` → 99.9% chance of rounding up

Statistically, the expected value equals the original value — no bias.

## How It Changes the Cmodel

### Before (Inline RNE Only)

```c
// Hard-coded RNE in tu_fp32_to_fp16():
uint32_t round_bit = (mant >> 12) & 1;
uint32_t sticky = (mant & 0xFFF) != 0;
uint32_t m_rounded = (mant >> 13);
if (round_bit && (sticky || (m_rounded & 1))) m_rounded++;
```

### After (Configurable via Rounding Module)

```c
// Configurable rounding in tu_fp32_to_fp16():
int carry;
uint32_t m_rounded = tu_round_fp32_to_mantissa(mant, exp, 10, &carry);
```

## Verification

14 tests covering:
- **RNE**: Basic roundtrip, tie-to-even, FP16→FP32 exactness
- **RTZ**: Truncation for both positive and negative values
- **Stochastic**: Unbiased average (10K trials), reproducibility (same seed)
- **BF16**: All three modes across BF16 precision
- **Mode switching**: Consistency after mode changes
- **Edge cases**: Carry overflow to infinity, subnormal flush-to-zero interaction
- **Bulk**: 1000 random values through all modes produce valid FP16

Run: `make test-rounding`

## Configuration

```c
// In tu_config.h:
#define TU_FP16_ROUNDING_RNE          0
#define TU_FP16_ROUNDING_RTZ          1
#define TU_FP16_ROUNDING_STOCHASTIC   2
#define TU_FP16_ROUNDING_MODE         TU_FP16_ROUNDING_RNE  // default
```

Runtime override via `tu_set_rounding_mode()`.

## Future Extensions

- **Per-precision modes**: Different rounding for FP16 vs INT8 (currently one global mode)
- **Stochastic rounding with dither**: Add Gaussian noise for better training convergence
- **Hardware-accelerated PRNG**: Replace software xorshift with modeled hardware LFSR
