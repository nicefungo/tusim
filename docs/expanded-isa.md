# TU CModel — Expanded Instruction Set Architecture (ISA)

> **Gap ID:** C1 (Limited ISA — 6 instructions → 30+ instruction production ISA)
> **Priority:** P0 (Critical)
> **Date:** 2026-05-29
> **Heartbeat:** Cycle 5b

---

## What Changed

The TinyTU ISA previously had only 6 instructions: `LOAD_W`, `LOAD_A`, `LOAD_O`, `MMA`, `SYNC`, `STORE_O`. These were parsed from text format in `tu_asm.c` and mapped directly to C function calls. There was no binary encoding, no opcode definitions beyond the parser's string matching, and no operation descriptors.

The ISA has been expanded to **68 defined opcodes** across 9 categories, with a fixed **96-bit binary encoding**, full **operation descriptor structs** for all operation types, and backward compatibility with the existing command queue API.

### Key Features

1. **68 defined opcodes** (plus 60 reserved slots for future expansion) across 9 categories:
   - **Control** (7): NOP, HALT, SYNC, BARRIER, FENCE, WAIT, SIGNAL
   - **Matrix** (10): MMA, MMA.BIAS, MMA.FUSED, CONV2D, CONV3D, DepthwiseConv, TransposedConv, ATTENTION, ATTN.QK, ATTN.PV
   - **Elementwise** (11): ELEMENTWISE, ADD, MUL, RELU, GELU, SILU, TANH, SIGMOID, EXP, SCALE
   - **Normalization & Reduction** (9): REDUCE.SUM/MAX/MEAN, SOFTMAX, LOG_SOFTMAX, LAYER_NORM, RMS_NORM, BATCH_NORM, GROUP_NORM
   - **Pooling** (3): POOL.MAX, POOL.AVG, POOL.GLOBAL_AVG
   - **Data Movement** (8): DMA.LOAD, DMA.STORE, DMA.CHAIN, strided LOAD/STORE, SCATTER, GATHER, BROADCAST
   - **Data Layout** (7): TRANSPOSE, PERMUTE, RESHAPE, SLICE, CONCAT, PAD, TILE
   - **Sparsity** (3): SPARSE_MMA, DECOMPRESS, COMPRESS
   - **Configuration** (2): SET_CONFIG, GET_CONFIG

2. **96-bit fixed-width encoding** — Every instruction is exactly 12 bytes:
   ```
   [7:0]   opcode       — operation code
   [15:8]  flags        — precision (bits 2:0), transpose (bits 4:3),
                          activation (bits 6:5), bias (bit 7)
   [31:16] dim0         — M, N, in_channels, seq_len (context-dependent)
   [47:32] dim1         — N, K, out_channels, head_dim
   [63:48] dim2         — K, R, num_heads
   [95:64] immediates   — address offsets, strides, scales
   ```

3. **Operation descriptors** — 9 specialized structs for decoded instruction execution:
   - `tu_mma_op_desc_t` — Matrix multiply (M×N×K, offsets, precision, fused activation)
   - `tu_conv_op_desc_t` — Convolution (NCHW, KCRS, stride, padding, dilation, groups)
   - `tu_attention_op_desc_t` — Attention (Q/K/V offsets, heads, seq_len, causal mask)
   - `tu_ew_op_desc_t` — Elementwise (binary/unary, activation type)
   - `tu_norm_op_desc_t` — Normalization (gamma/beta offsets, epsilon, axes)
   - `tu_softmax_op_desc_t` — Softmax (axis size, log variant)
   - `tu_pool_op_desc_t` — Pooling (kernel, stride, padding, max/avg/global)
   - `tu_dma_op_desc_t` — DMA (host addr, SRAM offset, size, strided geometry)
   - `tu_reduce_op_desc_t` — Reduction (axis, keep_dims, sum/max/mean)
   - `tu_sparse_mma_op_desc_t` — Sparse MMA (compressed weight + metadata offsets)

4. **Backward compatibility** — All existing `TU_CMD_*` opcodes map to `TU_ISA_*` equivalents via `#define` aliases in `command_queue.h`

---

## Why This Matters

The ISA is the **contract** between the compiler and the hardware:

- **Compiler can target a stable interface** — Instead of ad-hoc string parsing, the compiler emits 96-bit binary instructions with defined semantics
- **Operations can be reasoned about** — Query functions (`tu_isa_is_compute_op`, `tu_isa_is_dma_op`, `tu_isa_has_sram_operands`) let the scheduler understand operation characteristics
- **Hardware can be designed against a spec** — The ISA defines what the TU must implement; new implementations (RTL, FPGA) know exactly what to build
- **Extensibility** — 60 reserved opcode slots + flag extension bits allow growth without breaking existing code

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Compiler (onnx_to_tu.py)                       │
│  ┌───────────────┐    ┌───────────────────────┐ │
│  │ ONNX → TU Ops │ →  │ tu_instruction_t[96b] │ │
│  └───────────────┘    └───────────┬───────────┘ │
└───────────────────────────────────┼─────────────┘
                                    │ binary or text
┌───────────────────────────────────▼─────────────┐
│  TU Core                                        │
│  ┌────────────────────────────────────────────┐ │
│  │  Command Queue                             │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │ Decode   │→ │ Schedule │→ │ Execute  │ │ │
│  │  │ flags    │  │ deps     │  │ op_desc  │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘ │ │
│  └────────────────────────────────────────────┘ │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Compute  │  │ Memory   │  │ DMA Engine   │  │
│  │ Engine   │  │ System   │  │              │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## Instruction Encoding Detail

### 96-bit Layout

```
Byte:  0      1       2-3     4-5     6-7     8-11
      [opcode][flags] [dim0]  [dim1]  [dim2]  [immediates]
Bits:  7-0    15-8    31-16   47-32   63-48   95-64
```

### Flag Field (8 bits)

| Bits | Field | Values |
|------|-------|--------|
| 2:0 | Precision | 0=FP16, 1=FP32, 2=BF16, 3=FP8_E4M3, 4=FP8_E5M2, 5=INT8, 6=INT4 |
| 4:3 | Transpose | 0=None, 1=TransposeA, 2=TransposeB, 3=Both |
| 6:5 | Activation | 0=None, 1=ReLU, 2=GELU, 3=SiLU |
| 7 | Bias | 0=No bias, 1=Has bias |

### dim0/dim1/dim2 (context-dependent)

| Opcode | dim0 | dim1 | dim2 |
|--------|------|------|------|
| MMA | M (rows) | N (cols) | K (inner) |
| CONV2D | H (height) | W (width) | R (kernel H) |
| ATTENTION | seq_len_q | seq_len_kv | head_dim |
| POOL | H | W | kernel_h |
| SOFTMAX | axis_size | 0 | 0 |

### immediates (32 bits)

For MMA: `w_offset` (low 16) + `a_offset` (high 16)  
For DMA: full 32-bit SRAM offset  
For CONV: `input_offset` (full 32-bit)

Additional operands beyond 32 bits are stored in subsequent instructions or memory descriptors.

---

## API Reference

### Core Types

```c
typedef enum { ... } tu_isa_opcode_t;       // 68 opcodes
typedef struct { ... } tu_instruction_t;     // 96-bit instruction
typedef struct { ... } tu_op_descriptor_t;   // Unified decoded descriptor
```

### Query Functions

```c
const char *tu_isa_opcode_name(tu_isa_opcode_t opcode);
tu_isa_category_t tu_isa_opcode_category(tu_isa_opcode_t opcode);
bool tu_isa_has_sram_operands(tu_isa_opcode_t opcode);
bool tu_isa_is_compute_op(tu_isa_opcode_t opcode);
bool tu_isa_is_dma_op(tu_isa_opcode_t opcode);
```

### Flag Decoding

```c
void tu_isa_decode_flags(uint8_t flags,
                          uint8_t *precision_out,
                          uint8_t *transpose_out,
                          uint8_t *activation_out,
                          bool *has_bias_out);
```

---

## Opcode Catalog

### Control (0x00–0x0F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| NOP | 0x00 | No operation |
| HALT | 0x01 | Stop execution |
| SYNC | 0x02 | Drain pipeline, sync all units |
| BARRIER | 0x03 | Ordering barrier |
| FENCE | 0x04 | Memory fence |
| WAIT | 0x05 | Wait for signal |
| SIGNAL | 0x06 | Fire signal |

### Matrix (0x10–0x1F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| MMA | 0x10 | Matrix multiply-accumulate |
| MMA.BIAS | 0x11 | MMA with bias |
| MMA.FUSED | 0x12 | MMA with fused activation |
| CONV2D | 0x13 | 2D convolution |
| CONV3D | 0x14 | 3D convolution |
| CONV.DEPTHWISE | 0x15 | Depthwise convolution |
| CONV.TRANSPOSED | 0x16 | Transposed convolution |
| ATTENTION | 0x17 | Fused attention |
| ATTN.QK | 0x18 | Q×K^T only |
| ATTN.PV | 0x19 | P×V only |

### Elementwise (0x20–0x2F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| ELEMENTWISE | 0x20 | Generic elementwise |
| ADD | 0x21 | C = A + B |
| MUL | 0x22 | C = A × B |
| RELU | 0x23 | C = max(0, A) |
| GELU | 0x24 | Gaussian error linear unit |
| SILU | 0x25 | Sigmoid linear unit |
| TANH | 0x26 | Hyperbolic tangent |
| SIGMOID | 0x27 | Sigmoid |
| EXP | 0x28 | Exponential |
| SCALE | 0x29 | C = A × scalar |

### Normalization & Reduction (0x30–0x3F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| REDUCE.SUM | 0x30 | Sum reduction |
| REDUCE.MAX | 0x31 | Max reduction |
| REDUCE.MEAN | 0x32 | Mean reduction |
| SOFTMAX | 0x33 | Online softmax |
| LOG_SOFTMAX | 0x34 | Log-softmax |
| LAYER_NORM | 0x35 | Layer normalization |
| RMS_NORM | 0x36 | RMS normalization |
| BATCH_NORM | 0x37 | Batch normalization |
| GROUP_NORM | 0x38 | Group normalization |

### Pooling (0x40–0x4F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| POOL.MAX | 0x40 | Max pooling |
| POOL.AVG | 0x41 | Average pooling |
| POOL.GLOBAL_AVG | 0x42 | Global average pooling |

### Data Movement (0x50–0x5F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| DMA.LOAD | 0x50 | DRAM → SRAM |
| DMA.STORE | 0x51 | SRAM → DRAM |
| DMA.CHAIN | 0x52 | Descriptor chain |
| DMA.LOAD.STRIDED | 0x53 | 2D/3D strided load |
| DMA.STORE.STRIDED | 0x54 | 2D/3D strided store |
| DMA.SCATTER | 0x55 | Scatter via index list |
| DMA.GATHER | 0x56 | Gather via index list |
| DMA.BROADCAST | 0x57 | 1-to-N broadcast |

### Data Layout (0x60–0x6F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| TRANSPOSE | 0x60 | 2D transpose |
| PERMUTE | 0x61 | N-D permute |
| RESHAPE | 0x62 | Logical reshape |
| SLICE | 0x63 | Extract slice |
| CONCAT | 0x64 | Concatenate |
| PAD | 0x65 | Pad tensor |
| TILE | 0x66 | Tile/repeat tensor |

### Sparsity (0x70–0x7F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| SPARSE_MMA | 0x70 | Sparse MMA (2:4 structured) |
| DECOMPRESS | 0x71 | Decompress sparse weights |
| COMPRESS | 0x72 | Compress weights |

### Configuration (0x7E–0x7F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| SET_CONFIG | 0x7E | Runtime configuration |
| GET_CONFIG | 0x7F | Read configuration |

---

## Backward Compatibility

The existing `command_queue.h` opcode enum (`tu_cmd_opcode_t`) is now a typedef alias for `tu_isa_opcode_t`. All legacy `TU_CMD_*` identifiers map to `TU_ISA_*` via `#define`:

```c
#define TU_CMD_NOP          TU_ISA_NOP         // 0x00
#define TU_CMD_DMA_LOAD     TU_ISA_DMA_LOAD    // 0x50
#define TU_CMD_MMA          TU_ISA_MMA         // 0x10
#define TU_CMD_SYNC         TU_ISA_SYNC        // 0x02
// ... etc
```

**No existing code needs changes.** The command queue continues to work with `tu_cmd_opcode_t` and `TU_CMD_*` constants.

---

## Tests

### Running

```bash
make test-isa
```

### Coverage (9 tests)

| Test | What it verifies |
|------|-----------------|
| Instruction size | `sizeof(tu_instruction_t) == 12` (96 bits) |
| All opcodes named | 68+ named opcodes, reserved slots return "UNKNOWN" |
| Key opcode names | Specific opcode names are correct |
| Category classification | Each opcode maps to correct category |
| Query functions | `is_compute_op`, `is_dma_op`, `has_sram_operands` correct |
| Flag decoding | Precision/transpose/activation/bias extracted correctly |
| Backward compat | `TU_CMD_*` ≡ `TU_ISA_*` |
| Opcodes distinct | No two defined opcodes share a name |
| Descriptor sizes | All operation descriptors fit in reasonable bounds |

---

## Limitations (Future Work)

1. **No binary encoder/decoder yet** — `tu_instruction_t` ↔ bytes serialization deferred (requires endianness handling)
2. **No text format updated** — `tu_asm.c` still uses the old 6-instruction parser; needs update to parse new opcodes
3. **No compiler integration** — `onnx_to_tu.py` still emits old TU ASM text; needs update to emit new ISA
4. **No encoder for immediate fields** — Complex operands (strides, padding, multi-dim shapes) need multi-instruction encoding
5. **Variable-length extension** — 96-bit fixed width may be too small for some descriptors (e.g., attention with mask offset + causal flag + scale)

---

## Files

| File | Purpose |
|------|---------|
| `tu_cmodel/isa/tu_isa.h` | ISA definitions: opcodes, encoding, descriptors (~300 lines) |
| `tu_cmodel/isa/tu_isa.c` | ISA helpers: names, categories, queries, flag decode (~140 lines) |
| `tu_cmodel/command_queue.h` | Updated to use `tu_isa_opcode_t` with backward-compat aliases |
| `tests/test_isa.c` | 9-test ISA verification suite |
| `docs/expanded-isa.md` | This document |
