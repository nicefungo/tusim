# TinyTU Compiler Stack — Design Document

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ONNX Model (.onnx)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Compiler Frontend (onnx_to_tu.py)                              │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Parse    │→│ Shape        │→│ Op         │→│ Tiling &  │  │
│  │ ONNX     │  │ Inference    │  │ Lowering   │  │ Alloc     │  │
│  └──────────┘  └──────────────┘  └───────────┘  └───────────┘  │
│                                                                  │
│  Supported ops on TU:  Gemm, MatMul                              │
│  Host fallback:        134 other ONNX ops (stubs)                │
│  Shape inference:      14 op types                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼  TU ASM (linear instruction stream)
                             │  ─────────────────────────────────
┌────────────────────────────┴────────────────────────────────────┐
│  Code Generator                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  TU ASM → C (embeds weights, generates tu_* calls)        │   │
│  │  OR: TU ASM → binary → cmodel interprets directly         │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  TinyTU CModel (tu_cmodel.c)                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ DMA      │  │ SRAM     │  │ Systolic │  │ Stats /       │   │
│  │ Engine   │  │ Manager  │  │ Array    │  │ Counters       │   │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Boundaries

### Layer 1: Compiler IR (internal)
After ONNX parsing, the compiler builds an internal graph representation:
```python
node = {
    'op': 'Gemm',
    'name': '/fc/Gemm',
    'inputs': ['input', 'fc.weight', 'fc.bias'],
    'outputs': ['output'],
    'attrs': {'transB': 1, 'alpha': 1.0, 'beta': 1.0}
}
```
Shape inference annotates each tensor with concrete dimensions.

### Layer 2: TU ASM (compiler ↔ cmodel boundary)
A linear, positional instruction stream with explicit SRAM offsets.
**This is the canonical interface.** See [TU_ASM.md](TU_ASM.md) for the full specification.

Example:
```asm
; Gemm: Y[2][32] = X[2][64] @ W[64][32] + bias[32]
; MMA semantics: O[N][M] += W[N][K] × A[K][M]
LOAD_W   @0, 4096            ; W[32][64] → W-buffer offset 0
LOAD_A   @0, 256             ; X^T[64][2] → A-buffer offset 0
LOAD_O   @0, 128             ; bias[32][2] → O-buffer offset 0 (expanded)
MMA      32, 2, 64, 0,0,0, BIAS
SYNC
STORE_O  @0, 256             ; O[32][2] → host
```

For weight-tiled operations:
```asm
; W[768][256] doesn't fit in 64KB tile workspace → 3 tiles
LOAD_A   @0, 512             ; full A[256][2]
; --- Tile 0: K=[0:85) ---
LOAD_W   @32768, 130560      ; W[:,0:85] → tile workspace
MMA      768, 2, 85, 32768, 0, 0, BIAS
; --- Tile 1: K=[85:170) ---
LOAD_W   @32768, 130560
MMA      768, 2, 85, 32768, 170, 0, NOBIAS
; --- Tile 2: K=[170:256) ---
LOAD_W   @32768, 131072
MMA      768, 2, 86, 32768, 340, 0, NOBIAS
SYNC
STORE_O  @0, 6144
```

### Layer 3: C Code Generation (current backend)
The compiler currently emits C code from the ASM. A future ASM interpreter would
bypass this step, allowing the cmodel to execute `.tuasm` files directly.

## TinyTU Hardware Model

### Systolic Array
- **Dimensions:** 16 rows × 16 columns
- **Dataflow:** Weight-stationary — weights are preloaded into the PE array, activations stream through
- **Precision:** FP16 multiply → FP32 accumulate → FP16 round on store
- **Pipeline:** 16 cycles to fill, then 1 MAC/cycle per PE
- **Peak throughput:** 16×16 = 256 MACs/cycle (after pipeline fill)

### SRAM Hierarchy
| Buffer | Size | Purpose | Managed by |
|--------|------|---------|------------|
| W-Buffer | 128 KB | Weight matrices (stationary) | Compiler (alloc + tile) |
| A-Buffer | 64 KB | Activation matrices (streamed) | Compiler (alloc) |
| O-Buffer | 64 KB | Output accumulators (FP32) | Compiler (alloc) |

**Allocation strategy:**
- Lower 64 KB of W-buffer: preloaded static weights (fit at init time)
- Upper 64 KB of W-buffer: tile workspace (reused per tile)
- A-buffer and O-buffer: bump allocator, reused across ops

### DMA Engine
- 256-bit AXI bus → 32 bytes/cycle
- Three independent channels: W (host→W-buffer), A (host→A-buffer), O (O-buffer↔host)
- Synchronous in cmodel, asynchronous in hardware

### MMA Operation
```
O[N][M] += W[N][K] × A[K][M]

Where:
  N = output features (from weight matrix)
  M = batch dimension (flattened B×T)
  K = inner dimension (input features)

Transposition from ONNX Gemm:
  ONNX: Y[M][N] = X[M][K] × B[K][N] + C
  TU:   W = B^T [N][K]     (stored transposed)
        A = X^T [K][M]     (transposed at load time)
        O = Y^T [N][M]     (output, transposed)
```

## Compiler Passes

### Pass 1: ONNX Parsing
- Load model via `onnx.load()`, validate schema
- Extract graph nodes, initializers (weights), value_info (shapes)
- Build producer→consumer edges for topological sort

### Pass 2: Shape Inference
- Resolve dynamic dims from CLI (`--shape B=2,T=1`)
- Forward propagation through 14 op types:
  - MatMul: ND batch matmul → flatten batch dims
  - Elementwise (Add/Sub/Mul/Div/Relu/Erf/Softmax etc.): copy dominant operand shape
  - Reshape: resolve from constant shape tensor
  - Transpose: permute dimensions
  - ReduceMean: remove/reduce axes
  - Concat/Split: axis-dimension arithmetic
- Falls back to `?` for unsupported ops (triggers host fallback)

### Pass 3: Op Lowering
- Gemm/MatMul with static weights → TU MMA
- All other ops → host fallback stub (`host_<op>()`)
- Weight preloading: transpose B→W, embed as C arrays
- Determine tiled vs. single-shot based on W-buffer capacity

### Pass 4: Tiling & Memory Allocation
- **K-dimension tiling:** When W[N][K] exceeds W-buffer workspace:
  - `K_tile = workspace_size / (N × sizeof(fp16))`
  - Split K into `ceil(K / K_tile)` tiles
  - First tile includes bias, subsequent tiles accumulate
- **SRAM allocation:** Bump allocators for each buffer
- **Tile workspace:** Upper 64 KB of W-buffer, overwritten per tile

### Pass 5: Code Generation
- Emit C program with embedded weight blobs
- Generate `tu_transpose_fp16()` + `tu_dma_load_a()` for activation loading
- Generate bias expansion (`np.repeat`) for bias loading
- Generate `tu_mma()` calls with computed offsets

## Precision Model

```
Input (FP16) ──┐
                ├──→ Multiply (FP16×FP16 → FP32 product)
Weight (FP16) ─┘         │
                          ▼
                    Accumulate (FP32 += product)
                          │
                          ▼
                    Output (FP32 in O-buffer)
                          │
                    DMA Store (memcpy as FP32 bytes)
                          │
                          ▼
                    Host (FP32, may round to FP16)
```

**Rounding behavior:**
- FP32 → FP16 on DMA load: round-to-nearest-even, clamp to [-65504, 65504]
- FP32 accumulation: full IEEE 754 precision (23-bit mantissa)
- No rounding on DMA store (raw FP32 bytes)
- Measured error vs. FP16 reference: ≤ 5.4×10⁻⁷ (last-bit differences)

## Limitations & Future Work

| Area | Current | Planned |
|------|---------|---------|
| ASM execution | Compiler emits C, cmodel linked as library | TU ASM interpreter in cmodel |
| Memory reuse | Bump allocator, no free | Liveness analysis, register-style allocation |
| Host fallback | Empty stubs | RISC-V scalar implementations |
| Activation flow | Only input→output works | Intermediate tensor DMA between TU ops |
| Persistent kernel | Separate DMA+MMA per op | Fused chain in single TU invocation |
| Conv support | Not supported | im2col lowering → Gemm |
