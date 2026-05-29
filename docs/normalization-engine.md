# TU CModel — Normalization Engine

> **Gap O5:** LayerNorm and RMSNorm with online statistics computation  
> **Status:** Implemented  
> **Files:** `tu_cmodel/compute/normalization_engine.{h,c}`  
> **Tests:** `tests/test_normalization.c` (11 tests)

---

## Why This Feature

Every modern transformer architecture (GPT, LLaMA, BERT, Vision Transformer) uses normalization layers before or after attention and feed-forward blocks. LayerNorm is in the original Transformer; RMSNorm is used by LLaMA and most recent LLMs for its computational efficiency.

Without normalization support in the cmodel:
- Transformer inference requires CPU fallback for every norm layer, breaking the fused pipeline
- Cannot model the bandwidth savings of fused GEMM+Norm (avoids DRAM round-trip)
- Cannot evaluate end-to-end transformer block performance on the accelerator

This was prioritized as a **P1 gap** alongside elementwise ops (already implemented) because together they complete the "post-GEMM pipeline": MMA → Norm → Elementwise activation → writeback, all in on-chip SRAM.

---

## How It Works

### Two-Pass Algorithm

Both LayerNorm and RMSNorm use a two-pass algorithm optimized for FP32 data resident in SRAM:

**LayerNorm:**
```
Pass 1: μ = Σx_i / N          (mean)
        σ² = Σ(x_i - μ)² / N  (variance)

Pass 2: y_i = (x_i - μ) / √(σ² + ε) · γ_i + β_i
```

**RMSNorm:**
```
Pass 1: rms² = Σx_i² / N      (mean of squares)

Pass 2: y_i = x_i / √(rms² + ε) · γ_i
```

Both passes operate on the same data in SRAM — Pass 1 reads to compute statistics, Pass 2 reads again to apply normalization and writes the result. This is a trade-off: two SRAM read passes instead of storing intermediate values, which keeps the implementation simple and the SRAM footprint minimal.

### Numerical Stability

- **ε (epsilon):** Small constant (default 1e-5) added before division and sqrt to prevent division by zero. Configurable per call.
- **Double-precision accumulation:** Statistics (sum, sum-of-squares) are accumulated in `double` (FP64) to prevent overflow for large tensors. The final normalized values are rounded back to FP32.
- **Catastrophic cancellation:** The textbook two-pass method is used rather than the one-pass Welford algorithm because, for normalization, the second pass is required anyway (to apply the transform), making two-pass variance computation essentially free.

### SRAM Bandwidth Accounting

Every read and write goes through `tu_sram_read()` / `tu_sram_write()`, which returns stall cycles from the banked bandwidth model. The normalization engine accumulates and returns total stall cycles, enabling accurate performance modeling:

```c
uint64_t stall = tu_layernorm(&sram, offset, elem_count, gamma, beta, 1e-5f, true);
// stall = cycles lost to bank conflicts and BW exhaustion during normalization
```

### Per-Row Normalization (2D)

The `_2d` variants normalize each row of a 2D tensor independently — this is the standard usage for transformer hidden states:

```c
// hidden_states: [batch*seq_len, hidden_dim] in SRAM
// Normalize each row over hidden_dim columns
tu_layernorm_2d(&sram, offset, num_rows, hidden_dim, gamma, beta, 1e-5f);
// or
tu_rmsnorm_2d(&sram, offset, num_rows, hidden_dim, gamma, 1e-5f);
```

Each row gets its own mean/variance statistics computed independently.

### Scale and Bias

Gamma (γ, scale) and beta (β, bias) are per-element FP32 vectors. In the convenience API, they are passed as host pointers and temporarily staged in high SRAM addresses. In production, the compiler places them in dedicated SRAM regions and uses the full `tu_norm_desc_t` interface.

---

## API Reference

### Descriptor-Based API

```c
tu_norm_desc_t desc = {
    .mode          = TU_NORM_LAYER_NORM,
    .data_sram     = &my_sram,
    .data_offset   = 0,
    .elem_count    = 1024,
    .gamma_sram    = &gamma_region,   // NULL = identity γ=1
    .beta_sram     = &beta_region,    // NULL = zero β=0
    .epsilon       = 1e-5f,
    .norm_axis_dim = 0,              // 0 = whole tensor, N = per-row with N cols
    .in_place      = true,
    .mean_out      = NULL,           // optional: receive computed mean
    .var_out       = NULL,           // optional: receive computed variance
};
uint64_t stall = tu_norm_execute(&desc);
```

### Convenience API

```c
// Single-row LayerNorm
uint64_t tu_layernorm(sram, offset, elem_count, gamma, beta, epsilon, in_place);

// Single-row RMSNorm
uint64_t tu_rmsnorm(sram, offset, elem_count, gamma, epsilon, in_place);

// Per-row LayerNorm (2D)
uint64_t tu_layernorm_2d(sram, offset, rows, cols, gamma, beta, epsilon);

// Per-row RMSNorm (2D)
uint64_t tu_rmsnorm_2d(sram, offset, rows, cols, gamma, epsilon);
```

### Modes

| Mode | Enum | Formula | Use Case |
|------|------|---------|----------|
| LayerNorm | `TU_NORM_LAYER_NORM` | `(x-μ)/σ·γ+β` | BERT, original Transformer |
| RMSNorm | `TU_NORM_RMS_NORM` | `x/rms·γ` | LLaMA, Mistral, modern LLMs |
| BatchNorm | `TU_NORM_BATCH_NORM` | (reserved) | Vision models (CNNs) |

---

## Integration with Command Queue

The normalization engine integrates with the command queue via ISA opcodes:

| ISA Opcode | Value | Maps To |
|-----------|-------|---------|
| `TU_ISA_LAYER_NORM` | 0x07 | `tu_layernorm()` |
| `TU_ISA_RMS_NORM` | 0x08 | `tu_rmsnorm()` |

The command queue's `tu_cmdq_submit()` accepts these opcodes. The full integration (descriptor packing for the command queue) follows the same pattern as elementwise ops in `command_queue.h`.

---

## Fused Pipeline Pattern

Normalization is typically fused with the preceding GEMM to avoid a DRAM round-trip:

```
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ GEMM │───→│ LayerNorm│───→│  ReLU    │───→│ Writeback│
│(SRAM)│    │  (SRAM)  │    │  (SRAM)  │    │  (SRAM)  │
└──────┘    └──────────┘    └──────────┘    └──────────┘
     ↑                                           │
     └────────── All in on-chip SRAM ─────────────┘
```

The compiler schedules these as a fused sequence, eliminating intermediate DRAM traffic. See `docs/elementwise-pipeline.md` for the elementwise half of this pipeline.

---

## Verification

- **11 unit tests:** identity normalization (μ≈0, σ≈1), constant input edge case, RMSNorm correctness, gamma scaling, beta biasing, per-row 2D LayerNorm, per-row 2D RMSNorm, single-element edge cases, and descriptor validation.
- Numerical tolerance: 1e-4 for mean/std assertions, 1e-4 for per-element comparison against analytically computed expected values.
- All existing tests pass (no regressions).

---

## Future Extensions

- **BatchNorm:** Reserved in the enum; needs running-mean/running-variance tracking for inference.
- **GroupNorm:** Normalize over groups of channels; useful when batch size is small.
- **Online (one-pass) statistics:** Reduce SRAM bandwidth by computing mean/variance in a single pass using Welford's algorithm. Currently trades simplicity for bandwidth.
- **FP16 norm:** Normalize directly on FP16 data without promotion to FP32 (useful for inference where FP16 precision is sufficient).
