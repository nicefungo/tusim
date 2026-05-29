# TU Softmax Engine

> **Gap O7:** Online softmax with numerical stability for transformer attention.
> **Status:** Implemented — 2026-05-30
> **Files:** `tu_cmodel/compute/softmax_engine.{h,c}`, `tests/test_softmax.c`

## Overview

The TU Softmax Engine provides numerically stable softmax, log-softmax, and online softmax operations on FP32 data in SRAM. It is designed specifically for transformer attention workloads where softmax is applied to attention scores (Q·K^T) before multiplying with values (V).

## Why Softmax Matters

Softmax is the critical normalization step in attention mechanisms:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Without numerical stability, large attention scores (common with large `d_k`) cause FP32 overflow in `exp()`, producing NaN outputs. The max-subtract technique solves this:

```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

This is equivalent mathematically but avoids overflow because `exp(x_i - max(x)) ≤ exp(0) = 1`.

## Architecture

### Dataflow

```
SRAM (FP32 data) ──read──→ [Load row into buffer] ──→ [Find max]
                             ↓
                        [Compute exp(x_i - max) & sum]
                             ↓
                        [Normalize: exp/sum or log]
                             ↓
                        [Write result back to SRAM] ──write──→ SRAM
```

### Modes

| Mode | Algorithm | Use Case |
|------|-----------|----------|
| `TU_SOFTMAX_STANDARD` | Two-pass: find max → exp & sum → divide | Default, transformer inference |
| `TU_SOFTMAX_LOG` | Two-pass: find max → exp & sum → ln | Cross-entropy loss, numerical stability |
| `TU_SOFTMAX_ONLINE` | Single-pass rescaling: streaming max & sum → re-read & divide | Streaming data, hardware pipeline |

### Standard Two-Pass Algorithm (Default)

```
Pass 1 (compute max and exp-sum):
  m = max(x_i)                          // Find maximum
  s = Σ_i exp(x_i - m)                  // Sum of shifted exponentials

Pass 2 (normalize):
  y_i = exp(x_i - m) / s                // Standard softmax
  y_i = (x_i - m) - ln(s)               // Log-softmax variant
```

**Numerical properties:**
- `exp(x_i - m)` never exceeds 1.0 (when `x_i == m`)
- `exp(x_i - m)` is at least `exp(min(x) - max(x))` which may underflow to 0 for large ranges
- Underflow is safe: 0/positive_sum = 0
- When all inputs are effectively `-inf`, outputs uniform distribution `1/N`

### Online Single-Pass Algorithm

```
For each x_i in streaming order:
  m' = max(m, x_i)
  s  = s · exp(m - m') + exp(x_i - m')
  m  = m'

After all elements:
  y_i = exp(x_i - m) / s   (requires re-read or compute-time storage)
```

**When to use online mode:** When data arrives in a streaming fashion (e.g., from a systolic array output row-by-row) and you want to avoid storing the entire row for a second pass. In practice, for SRAM-resident data, the two-pass algorithm is simpler and more efficient.

## API Reference

### Core Function

```c
uint64_t tu_softmax_execute(const tu_softmax_desc_t *desc);
```

Executes softmax according to the descriptor. Returns total stall cycles from SRAM bandwidth contention.

### Descriptor

```c
typedef struct {
    tu_softmax_mode_t mode;          // STANDARD, LOG, or ONLINE
    tu_sram_region_t *data_sram;     // SRAM region
    uint32_t          data_offset;   // Byte offset of first FP32 element
    uint32_t          elem_count;    // Total FP32 elements
    uint32_t          axis_dim;      // Elements per row (0 = single row)
    const float      *mask;          // Optional mask buffer (NULL = none)
    bool              mask_is_additive; // Add mask before softmax
    float             mask_fill;     // Value for masked positions
    float             scale;         // Pre-softmax scaling (e.g., 1/√d_k)
    bool              in_place;      // Overwrite input with output
    uint32_t          out_offset;    // Output offset (if !in_place)
    float            *max_out;       // Optional per-row max values
    float            *sum_out;       // Optional per-row exp-sum values
} tu_softmax_desc_t;
```

### Convenience Functions

```c
// Flat softmax (single vector)
uint64_t tu_softmax(sram, offset, elem_count, scale, in_place);

// Batched softmax (2D: each row independent)
uint64_t tu_softmax_2d(sram, offset, rows, cols, scale, in_place);

// Masked softmax (e.g., causal attention mask)
uint64_t tu_softmax_masked(sram, offset, rows, cols, mask, mask_fill, scale);

// Log-softmax variants
uint64_t tu_log_softmax(sram, offset, count, scale, in_place);
uint64_t tu_log_softmax_2d(sram, offset, rows, cols, scale, in_place);

// Host reference (for testing)
void tu_softmax_host(float *data, uint32_t count, float scale);
void tu_log_softmax_host(float *data, uint32_t count, float scale);
```

## Usage Examples

### Basic Softmax

```c
tu_sram_region_t sram;
tu_sram_init(&sram, 4096, "scores");

// Write 4-element vector: [1.0, 2.0, 3.0, 4.0]
float scores[4] = {1.0f, 2.0f, 3.0f, 4.0f};
// ... write to SRAM ...

// Compute softmax in-place
tu_softmax(&sram, 0, 4, 0.0f, true);
// SRAM now contains: [0.032, 0.087, 0.237, 0.644]  (sum = 1.0)
```

### Attention Softmax with Scaling

```c
// After computing Q·K^T, apply softmax with 1/√d_k scaling
float scale = 1.0f / sqrtf(head_dim);  // e.g., 1/√64 = 0.125

// Batched: batch*heads rows, seq_len elements each
tu_softmax_2d(&sram, attn_offset,
              batch_size * num_heads,  // rows
              seq_len,                  // cols
              scale, true);
```

### Causal Masked Attention

```c
// Create causal mask: positions where j > i get -inf
uint32_t seq_len = 128;
for (uint32_t i = 0; i < seq_len; i++)
    for (uint32_t j = 0; j < seq_len; j++)
        mask[i * seq_len + j] = (j > i) ? -1e9f : 0.0f;

// Apply masked softmax
tu_softmax_masked(&sram, attn_offset,
                  seq_len, seq_len,     // square attention matrix
                  mask, 0.0f,           // mask is additive, no extra fill
                  scale);
```

### Log-Softmax for Cross-Entropy Loss

```c
// Compute log-softmax on logits
tu_log_softmax(&sram, logits_offset, num_classes, 0.0f, true);

// Now SRAM contains ln(softmax(logits_i))
// Cross-entropy: loss = -Σ target_i * log_softmax_i
```

## Edge Case Behavior

| Input | Behavior | Output |
|-------|----------|--------|
| All -inf | All sums divide by zero | Uniform distribution: `1/N` each |
| All zeros | `exp(0) = 1` | `1/N` each |
| Very large positive | Dominates softmax | ~1.0 for max, ~0.0 for others |
| NaN in input | Propagated through mask | May produce NaN outputs |
| Scale > 0 | Applied before finding max | Amplifies differences |
| Scale = 0 | No scaling applied | Raw softmax |
| axis_dim = 0 | Treat as single flat vector | One softmax over all elements |

## Precision & Accuracy

- **FP32 accumulation** for sum (via `double` in C implementation)
- **Error tolerance:** < 1e-5 relative error vs. host reference
- **Underflow:** Values with `x_i - max < -87` (exp < 1.6e-38) are flushed to 0
- **Overflow prevention:** Max-subtract ensures `exp()` argument never exceeds 0

## Bandwidth Accounting

The engine models SRAM bandwidth usage:
- **Reads:** 1 word per FP32 element (2 passes = 2N reads for standard mode)
- **Writes:** 1 word per FP32 element (1 pass = N writes)
- **Total:** 3N words for standard two-pass, 3N for online (2 reads + 1 write)
- **Stall cycles:** Accumulated from `tu_sram_read()`/`tu_sram_write()` bandwidth metering

## ISA Mapping

The softmax engine corresponds to these ISA opcodes:

| ISA Opcode | Softmax Mode |
|------------|-------------|
| `TU_ISA_SOFTMAX` (0x33) | `TU_SOFTMAX_STANDARD` |
| `TU_ISA_LOG_SOFTMAX` (0x34) | `TU_SOFTMAX_LOG` |

ISA flags encode:
- `dim0`: number of rows (for batched mode)
- `dim1`: axis dimension (cols per row)
- `immediates[15:0]`: scale as FP16
- `immediates[31:16]`: mask offset in SRAM

## Verification

15 tests cover:
1. Single element (trivial)
2. Uniform vector (all same)
3. Known values (vs. numpy reference)
4. Large values (numerical stability)
5. Negative values
6. Batched 2D
7. Log-softmax
8. Scaled softmax (attention)
9. Masked softmax (causal)
10. All negative-inf (edge case)
11. Zero input
12. Online mode
13. Out-of-place
14. Large row (256D)
15. Log-softmax 2D

## Relationship to Attention

The softmax engine is a building block for the full attention operation (TU_ISA_ATTENTION, Gap O3). The full attention pipeline is:

```
Q·K^T ──→ [scale by 1/√d_k] ──→ [add mask] ──→ SOFTMAX ──→ ·V
```

The softmax engine handles the middle step. The full attention engine (future gap O3) will orchestrate:
1. MMA for Q·K^T
2. Scale + mask application
3. Softmax (using this engine)
4. MMA for P·V
```

## Configuration

No special config needed — softmax uses existing SRAM and bandwidth infrastructure. All parameters are per-operation via the descriptor.

## References

- Milakov & Gimelshein, "Online Normalizer Calculation for Softmax," arXiv:1805.02867
- NVIDIA, "FlashAttention: Fast and Memory-Efficient Exact Attention," Dao et al., NeurIPS 2022
- O3 gap: Full attention engine (`docs/PRODUCTION_TU_REDESIGN.md` §3.3, gap O3)
