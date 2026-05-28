# TinyTU Assembly Language (TU ASM) v0.1

## Overview

TU ASM is the canonical interface between the ONNX compiler and the TinyTU cmodel.
It is a linear, positional assembly language — each instruction operates on explicit
SRAM offsets with no symbolic registers. The cmodel can either interpret `.tuasm`
files directly or the compiler can lower ASM to C.

## Design Rationale

**Why a custom ASM instead of C code?**
1. **Separation of concerns:** Compiler optimizes at the graph level; cmodel executes
   at the instruction level. The ASM is the contract.
2. **Deterministic semantics:** Every instruction has exactly one interpretation.
   No C compiler UB, no optimization flags changing behavior.
3. **Direct cmodel execution:** A future `tu_run_asm()` can interpret ASM without
   a C compiler, enabling fast iteration.
4. **Hardware alignment:** The ASM maps 1:1 to hardware control words.
   Compiler→ASM→Verilog is a natural progression.

**Why positional (offset-based) instead of symbolic (register-based)?**
- TU SRAM is software-managed (like TPU), not hardware-managed (like GPU registers).
- Offsets are the ground truth — the compiler computes them, the cmodel uses them.
- Symbolic names can be a higher-level IR that lowers to positional ASM.

## Instruction Set

### Format

Each instruction is one line:
```
MNEMONIC  [operands...]  [; comment]
```

Operands are decimal integers (except `BIAS`/`NOBIAS` which are tokens).
Comments start with `;` and extend to end of line.

### Instructions

#### LOAD_W — DMA weights into W-buffer
```
LOAD_W  <tu_offset>  <size_bytes>
```
- Transfers `size_bytes` from a pre-embedded weight blob to W-buffer at `tu_offset`.
- The weight blob is identified by context (the ASM file declares weight sections).
- Constraints: `tu_offset + size_bytes ≤ 131072` (128 KB)

#### LOAD_A — DMA activations into A-buffer
```
LOAD_A  <host_ptr_id>  <tu_offset>  <size_bytes>
```
- Transfers activation data from host memory to A-buffer.
- `host_ptr_id` references a named host buffer (e.g., `input`).
- Constraints: `tu_offset + size_bytes ≤ 65536` (64 KB)

#### LOAD_O — DMA bias/initial data into O-buffer
```
LOAD_O  <host_ptr_id>  <tu_offset>  <size_bytes>
```
- Transfers data from host to O-buffer (typically bias values).
- Data is loaded as FP16 and will be expanded to FP32 by the MMA BIAS flag.
- Constraints: `tu_offset + size_bytes ≤ 65536` (64 KB)

#### MMA — Matrix Multiply-Accumulate
```
MMA  <M>  <N>  <K>  <w_off>  <a_off>  <o_off>  <bias_flag>
```
- Computes: `O[N][M] += W[N][K] × A[K][M]`
- `M`: output columns (batch dimension, flattened)
- `N`: output rows (feature dimension)
- `K`: inner dimension
- `w_off`: byte offset of W[N][K] in W-buffer (row-major, FP16)
- `a_off`: byte offset of A[K][M] in A-buffer (row-major, FP16)
- `o_off`: byte offset of O[N][M] in O-buffer (row-major, FP32 accumulators)
- `bias_flag`: `BIAS` or `NOBIAS`
  - `BIAS`: Initialize O from FP16 data at `o_off` before accumulating
  - `NOBIAS`: Accumulate into existing O values
- Tiling: The cmodel internally tiles MMA into 16×16×16 systolic array operations.
  The ASM-level MMA can have arbitrary M, N, K.

#### SYNC — Pipeline drain barrier
```
SYNC
```
- Waits for all pending DMA and MMA operations to complete.
- Drains the systolic pipeline (16 cycles).
- In the functional cmodel, DMA is synchronous, so this primarily accounts
  for pipeline drain cycles in stats.

#### STORE_O — DMA output from O-buffer to host
```
STORE_O  <host_ptr_id>  <tu_offset>  <size_bytes>
```
- Transfers FP32 accumulator data from O-buffer to host memory.
- `host_ptr_id` references a host buffer for the result.
- Constraints: `tu_offset + size_bytes ≤ 65536` (64 KB)

### Pseudo-instructions (for readability, not executed)

#### %weight — Declare an embedded weight section
```
%weight  <name>  <shape[N,K]>  <size_bytes>
<hex_dump...>
%endweight
```
- Declares a named weight blob. Referenced by LOAD_W via context.
- The shape specifies [N, K] = [output_features, input_features].

#### %input / %output — Declare host buffer bindings
```
%input  <name>  <shape>  <size_bytes>
%output <name>  <shape>  <size_bytes>
```
- Declares named host buffers for activation inputs and outputs.

## Complete Example

### Single Linear Layer: 64→32 with bias
```asm
; Model: single_linear
; Gemm: Y[2][32] = X[2][64] @ W[32][64] + bias[32]
; TU: O[32][2] = W[32][64] × X^T[64][2] + bias[32][2]

%weight  fc.weight  [32, 64]  4096
0000 3C00 4000 4400 ...  ; (4096 bytes of FP16 data)
%endweight

%input   input  [2, 64]  256
%output  output  [32, 2]  256

; --- Inference ---
LOAD_W   @0, 4096            ; W[32][64] → W-buffer
LOAD_A   input, @0, 256      ; X^T[64][2] → A-buffer
LOAD_O   bias, @0, 128       ; bias[32][2] expanded → O-buffer
MMA      32, 2, 64, 0, 0, 0, BIAS
SYNC
STORE_O  output, @0, 256     ; O[32][2] → host
```

### Tiled QKV Projection: 256→768 (K-tiling)
```asm
; Gemm: Y[2][768] = X[2][256] @ W[256][768]  (no bias in attention QKV)
; TU: O[768][2] = W[768][256] × X^T[256][2]
; W[768][256] = 393,216 bytes, exceeds 64KB tile workspace
; K_tile = 65536 / (768 × 2) = 42

%weight  qkv.weight  [768, 256]  393216
...
%endweight

%input   norm_out  [2, 256]  512
%output  qkv_out  [768, 2]  6144

; --- Inference ---
LOAD_A   norm_out, @0, 512    ; full X^T[256][2] → A-buffer

; Tile 0: K=[0:42)
LOAD_W   @32768, 64512        ; W[:,0:42] → tile workspace
MMA      768, 2, 42, 32768, 0, 0, NOBIAS

; Tile 1: K=[42:84)
LOAD_W   @32768, 64512
MMA      768, 2, 42, 32768, 84, 0, NOBIAS

; ... (4 more tiles) ...

; Tile 6: K=[252:256)
LOAD_W   @32768, 6144
MMA      768, 2, 4, 32768, 504, 0, NOBIAS

SYNC
STORE_O  qkv_out, @0, 6144
```

## Memory Layout Conventions

### W-Buffer (128 KB)
```
Offset 0 ─────────────────────────────── 64 KB ─────────────────────── 128 KB
│  Preloaded static weights              │  Tile workspace (reused)         │
│  (fc2.weight, small layers...)         │  LOAD_W overwrites per tile      │
└────────────────────────────────────────┴──────────────────────────────────┘
```

### A-Buffer (64 KB)
```
Bump-allocated per operation. Reused across ops.
A[K][M] row-major, FP16, K×M×2 bytes.
```

### O-Buffer (64 KB)
```
Bump-allocated per operation. Contains FP32 accumulators.
O[N][M] row-major, FP32, N×M×4 bytes.
```

## ASM-to-C Lowering (Current Implementation)

The compiler currently lowers TU ASM to C code. Each ASM instruction maps to a C function call:

| ASM | C |
|-----|---|
| `LOAD_W @off, size` | `tu_dma_load_w(weight_var + byte_off, off, size)` |
| `LOAD_A host, @off, size` | `tu_transpose_fp16(host_var, scratch_T, M, K); tu_dma_load_a(scratch_T, off, size)` |
| `LOAD_O host, @off, size` | `tu_dma_load_o(bias_var, off, size)` |
| `MMA M,N,K,w,a,o,f` | `tu_mma(N, M, K, w, a, o, f==BIAS)` |
| `SYNC` | `tu_sync()` |
| `STORE_O host,@off,size` | `tu_dma_store_o(host_var, off, size)` |

Note the M↔N swap: ASM uses `MMA <M> <N>` for readability (M=output cols, N=output rows),
but the C function signature is `tu_mma(M_rows, N_cols, K_inner, ...)`.

## Future: Native ASM Interpreter

A `tu_run_asm("program.tuasm")` function that:
1. Parses the ASM text
2. Loads weight sections
3. Executes instructions sequentially
4. Writes output sections

This eliminates the C compiler dependency and enables cycle-accurate simulation.
