# Elementwise Pipeline

> **Gap coverage:** O1 (operation coverage), O4 (fused elementwise ops)  
> **Status:** Implemented  
> **Date:** 2026-05-29  
> **Files:** `tu_cmodel/compute/elementwise_pipeline.{h,c}`

## Overview

The elementwise pipeline provides fused elementwise operations that execute directly on **FP32 data in SRAM**, eliminating round-trips to DRAM for activation functions and residual connections. It is designed to sit in the accumulator output path — after MMA produces results in the O-buffer, the elementwise pipeline applies activation functions (ReLU, GELU, SiLU, etc.) or binary operations (add bias, scale, residual add) **in-place** before DMA stores the results back to host memory.

### Why This Matters

Without the elementwise pipeline, every activation function or residual add requires:
1. DMA store MMA output to DRAM
2. Host-side computation (CPU or GPU)
3. DMA load back into SRAM for the next layer

With the pipeline, these operations are fused into a **single pass** over the data in SRAM, saving both DMA bandwidth and host compute cycles. This mirrors how real accelerators (Gemmini, TPU, NVIDIA TensorCores) fuse activations into their compute pipelines.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  MMA (Systolic Array)                │
│            FP16×FP16 → FP32 accumulate              │
├──────────────────────────────────────────────────────┤
│                  O-Buffer (SRAM)                     │
│               FP32 output of MMA                     │
├──────────────────────────────────────────────────────┤
│           Elementwise Pipeline (NEW)                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │  Op 1   │→│  Op 2   │→│  Op N   │  ← up to 8   │
│  │ (bias)  │  │ (ReLU)  │  │ (scale)  │    fused ops│
│  └─────────┘  └─────────┘  └─────────┘             │
├──────────────────────────────────────────────────────┤
│           DMA Store → Host DRAM                      │
└──────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Single pass, multi-op fusion:** Up to 8 operations can be chained. Each element is read once, transformed through all ops, and written back once. This minimizes SRAM bandwidth consumption.

2. **In-place by default:** Operations default to in-place modification of SRAM data, matching how activations are applied to accumulator outputs. Non-in-place mode is available for tensor-tensor operations (e.g., residual adds).

3. **FP32-only execution:** All operations execute in FP32. This is correct for the accumulator path (MMA accumulates in FP32), and avoids quantization noise from intermediate FP16 round-trips. The final FP16 conversion happens on DMA store.

4. **Stateless, SRAM-bound:** The pipeline does not own buffers — it operates on SRAM regions passed by the caller. This keeps it composable with DMA and MMA.

5. **Bandwidth accounting:** When SRAM bandwidth modeling is enabled (`bw_modeling = true`), elementwise operations account for read/write bandwidth consumption and can generate bank conflict stalls.

## Supported Operations

### Unary Ops

| Opcode | Function | Range Constraints |
|--------|----------|-------------------|
| `TU_EW_RELU` | `max(0, x)` | None |
| `TU_EW_GELU` | `0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))` | None (tanh approximation) |
| `TU_EW_SILU` | `x · σ(x)` = `x / (1 + e⁻ˣ)` | None |
| `TU_EW_SIGMOID` | `σ(x)` = `1 / (1 + e⁻ˣ)` | Saturates at ±∞ |
| `TU_EW_TANH` | `tanh(x)` | Saturates at ±∞ |
| `TU_EW_EXP` | `eˣ` | Overflows for x > ~88.7 |
| `TU_EW_NEG` | `-x` | None |
| `TU_EW_ABS` | `|x|` | None |
| `TU_EW_SQRT` | `√x` | Clamped: x < 0 → 0 |
| `TU_EW_LOG` | `ln(x)` | Clamped: x ≤ 0 → -∞ |

### Binary Ops (with scalar RHS)

| Opcode | Function |
|--------|----------|
| `TU_EW_ADD` | `x + s` |
| `TU_EW_MUL` | `x · s` |
| `TU_EW_SUB` | `x - s` |
| `TU_EW_DIV` | `x / s` (s=0 → 0) |
| `TU_EW_MIN` | `min(x, s)` |
| `TU_EW_MAX` | `max(x, s)` |

### Compound Opcodes (ISA level)

In the ISA, elementwise operations map to dedicated opcodes for common patterns:

| ISA Opcode | Pipeline Ops |
|------------|-------------|
| `TU_ISA_RELU` (0x23) | `TU_EW_RELU` |
| `TU_ISA_GELU` (0x24) | `TU_EW_GELU` |
| `TU_ISA_SILU` (0x25) | `TU_EW_SILU` |
| `TU_ISA_TANH` (0x26) | `TU_EW_TANH` |
| `TU_ISA_SIGMOID` (0x27) | `TU_EW_SIGMOID` |
| `TU_ISA_EXP` (0x28) | `TU_EW_EXP` |
| `TU_ISA_ADD` (0x21) | `TU_EW_ADD` with scalar |
| `TU_ISA_MUL` (0x22) | `TU_EW_MUL` with scalar |
| `TU_ISA_SCALE` (0x29) | `TU_EW_MUL` with scalar |
| `TU_ISA_ELEMENTWISE` (0x20) | Arbitrary fused chain (ops in flags/immediates) |

## API

### Direct Use (C API)

```c
#include "compute/elementwise_pipeline.h"

// --- Apply a single unary op in-place ---
tu_ew_apply_unary(&g_tu.sram_o, offset, elem_count, TU_EW_RELU);

// --- Apply a binary op with scalar in-place ---
tu_ew_apply_binary_scalar(&g_tu.sram_o, offset, elem_count, TU_EW_ADD, 1.0f);

// --- Apply a fused chain ---
tu_ew_op_t ops[2] = {
    {.opcode = TU_EW_ADD, .has_scalar = true, .scalar = bias_value},
    {.opcode = TU_EW_RELU, .has_scalar = false},
};
tu_ew_apply_fused(&g_tu.sram_o, offset, elem_count, ops, 2);

// --- Tensor-tensor add (A + B → out) ---
tu_ew_add_tensors(&g_tu.sram_o, a_offset, b_offset, out_offset, elem_count);

// --- Full descriptor (non-in-place, custom output) ---
tu_ew_desc_t desc = {
    .sram_offset = in_offset,
    .elem_count  = 256,
    .sram_region = &g_tu.sram_o,
    .in_place    = false,
    .out_offset  = out_offset,
    .num_ops     = 1,
};
desc.ops[0].opcode = TU_EW_GELU;
tu_ew_execute(&desc);
```

### Via Command Queue (recommended for production use)

```c
// ReLU via CMDQ — proper dependency tracking, async support
uint8_t ops[1] = { TU_EW_RELU };
tu_cmdq_submit_elementwise(
    0,        // sram_region: 0=O, 1=W, 2=A
    0,        // sram_offset in bytes
    256,      // element count
    ops, 1,   // opcodes + count
    NULL,     // scalars (none needed for unary)
    NULL      // has_scalar flags
);
tu_cmdq_sync_all();
```

### Fused with MMA (common pattern)

```c
// Typical GEMM + activation fused path:
tu_cmdq_submit_mma(16, 16, 16, w_off, a_off, o_off, false);   // MMA
tu_cmdq_submit_barrier();
uint8_t ops[2] = { TU_EW_ADD, TU_EW_RELU };
float scalars[2] = { bias_value, 0.0f };
bool has_scalar[2] = { true, false };
tu_cmdq_submit_elementwise(0, o_off, 256, ops, 2, scalars, has_scalar);  // add bias + ReLU
tu_cmdq_submit_barrier();
tu_cmdq_submit_dma_store(0, o_off, host_ptr, 256 * sizeof(float));       // store results
tu_cmdq_sync_all();
```

## Configuration

Elementwise operations are governed by the `enable_elementwise` feature flag in `tu_config_t`:

```yaml
# tu_config.yaml
tu_config:
  operations:
    enable_elementwise: true   # Enable elementwise pipeline
```

When disabled, the ISA decoder will reject elementwise opcodes. SRAM bandwidth modeling (for stall accounting) is controlled separately via `model_bank_conflicts` and the SRAM bank configuration.

## Numerical Properties

### GELU Approximation

The pipeline uses the **tanh approximation** (Hendrycks & Gimpel 2016), not the exact erf-based GELU:

```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

Maximum absolute error vs. exact GELU: < 1.5×10⁻⁴ for |x| ≤ 5.

### SiLU / Swish

Uses the direct computation `x / (1 + e⁻ˣ)`. No approximation — matches PyTorch `F.silu()`.

### SIGMOID and TANH

These use libm `expf()` and `tanhf()` directly. They are accurate to within 1 ULP on IEEE 754 platforms.

### SQRT and LOG safety

- `TU_EW_SQRT(x)` returns 0 for negative inputs (hardware behavior — no exceptions)
- `TU_EW_LOG(x)` returns `-INFINITY` for x ≤ 0

## Performance Model

When `bw_modeling` is enabled on the SRAM region:

- **In-place ops:** 1 word read + 1 word write per element (2 words total)
- **Non-in-place ops:** 1 read + 1 write per element (2 words total, fixed output offset)
- **Tensor-tensor adds:** 2 reads + 1 write per element (3 words total)
- **Compute cycles:** `elem_count` cycles (1 cycle per element, pipelined)

Bank conflict stalls are computed using the same `tu_sram_bank_index()` function as MMA and DMA, ensuring consistent bandwidth modeling across all operations.

## Integration Points

### With MMA (fused activation)

The elementwise pipeline is called **after** MMA in the accumulator path. A typical fused GEMM+ReLU sequence:

1. MMA writes FP32 results to O-buffer
2. Elementwise pipeline applies ReLU in-place
3. DMA stores FP16-converted results to host

This is what production accelerators do — cf. Gemmini's fused activation, NVIDIA's `mma.sync` with relu flag.

### With ISA

ISA opcodes `0x20`–`0x29` map to elementwise operations. The flag byte encodes:
- Bits [5:4]: which SRAM region (W/A/O)
- Bits [7:6]: op subtype (unary/binary/fused)
- Immediate fields: element count, offset, scalar value

### With Command Queue

`TU_CMD_ELEMENTWISE` commands support dependency tracking and barrier ordering, enabling proper sequencing with MMA and DMA in the CMDQ.

## Design Trade-offs

| Decision | Rationale | Alternative |
|----------|-----------|-------------|
| **FP32-only execution** | Matches accumulator path; avoids quantization noise | Operate in FP16 for lower energy (future option) |
| **In-place default** | Most activations modify accumulator output directly | Separate read/write buffers (safer, higher BW cost) |
| **Single pass, multi-op** | Minimizes SRAM bandwidth | Separate passes (simpler, higher BW) |
| **libm for exp/tanh** | Correctness and speed (libm is SIMD-optimized) | Custom polynomial approximations (smaller code, less accurate) |
| **Up to 8 fused ops** | Covers all practical patterns (bias+scale+act+clip) | Unlimited chain (unnecessary complexity) |

## Future Extensions

1. **FP16 and BF16 elementwise paths** — execute in the input precision to save compute energy (P1.1)
2. **Polynomial approximations** for EXP/TANH/SIGMOID — reduce latency at cost of accuracy (P2)
3. **Hardware lookup tables (LUT)** — emulated table-based activation for matching real silicon (P2)
4. **Stochastic rounding pass** — apply stochastic rounding before FP16 store (P2)

## References

- Hendrycks & Gimpel (2016): "Gaussian Error Linear Units (GELUs)" — tanh approximation
- Gemmini (Berkeley): Fused activation in accumulator path — `gemmini/accumulator.scala`
- Eyeriss v2 (MIT): Elementwise fusion with configurable dataflow
- Google TPU: Software-managed memory with explicit activation scheduling
