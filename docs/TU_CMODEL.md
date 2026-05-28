# TinyTU CModel Internals

## Overview

The TinyTU CModel (`tu_cmodel.c`) is a functional simulator for a 16×16 weight-stationary
systolic array accelerator. It computes bit-accurate results (FP16→FP32→FP16 rounding)
and reports performance counters as if running on silicon.

**Design principle:** Functional correctness first, cycle estimates second.
The cmodel computes the *right answer* using the same dataflow the hardware would use,
but without cycle-accurate pipeline modeling.

## API Reference

### Lifecycle
```c
void tu_init(void);         // Reset TU state, zero SRAM, clear counters
void tu_print_stats(void);  // Print DMA bytes, MMA tiles, FLOPS, est. cycles
```

### DMA
```c
void tu_dma_load_w(const void *host_ptr, uint32_t tu_offset, uint32_t size_bytes);
void tu_dma_load_a(const void *host_ptr, uint32_t tu_offset, uint32_t size_bytes);
void tu_dma_load_o(const void *host_ptr, uint32_t tu_offset, uint32_t size_bytes);
void tu_dma_store_o(void *host_ptr, uint32_t tu_offset, uint32_t size_bytes);
```
All DMA operations are synchronous memcpy with bounds checking.
Cycle estimate: `(size_bytes + 31) / 32` cycles (256-bit AXI bus).

### Compute
```c
void tu_mma(uint16_t M, uint16_t N, uint16_t K,
            uint32_t w_offset, uint32_t a_offset, uint32_t o_offset,
            bool has_bias);
```
Computes: `O[N][M] += W[N][K] × A[K][M]`
- **M:** output columns (batch dimension)
- **N:** output rows (feature dimension)
- **K:** inner dimension
- `has_bias=true`: initialize O from FP16 bias data at `o_offset` before accumulation

### Synchronization
```c
void tu_sync(void);  // Drain systolic pipeline (16 cycles)
```

## Systolic Array Implementation

### Weight-Stationary Dataflow

```
         a[0]    a[1]    a[2]    a[3]
          │       │       │       │
          ▼       ▼       ▼       ▼
       ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
       │W[0,0]│→│W[0,1]│→│W[0,2]│→│W[0,3]│→  o[0]
       └─────┘ └─────┘ └─────┘ └─────┘
          │       │       │       │
          ▼       ▼       ▼       ▼
       ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
       │W[1,0]│→│W[1,1]│→│W[1,2]│→│W[1,3]│→  o[1]
       └─────┘ └─────┘ └─────┘ └─────┘
          │       │       │       │
          ▼       ▼       ▼       ▼
          ...     ...     ...     ...     ...

Each PE[j,k]:
  psum[j] += W[j,k] × a[k]    // W stationary, a flows right, psum flows down
```

### Tiling Strategy

A single `tu_mma(M, N, K, ...)` is decomposed into 16×16×16 tiles:

```
for mi in 0..ceil(M/16):
  for ni in 0..ceil(N/16):
    for ki in 0..ceil(K/16):
      load  W_tile[16×16] from W-buffer
      load  A_tile[16×16] from A-buffer
      compute partial sum via systolic array
      accumulate into O_tile
```

Each tile uses the full 16×16 PE array. Edge tiles handle partial dimensions
with bounds checking.

### Cycle Estimate Model

| Operation | Cycles |
|-----------|--------|
| Pipeline fill (per tile) | 16 |
| Compute (per tile) | K_tile (1 MAC/cycle after fill) |
| DMA (per transfer) | ceil(bytes/32) |
| SYNC | 16 |

```
Total estimated cycles = DMA_cycles + n_tiles × (16 + K_tile) + 16
```

This is a simplified model — it does not account for:
- Memory bank conflicts (assumes optimal layout)
- DMA/Compute overlap (assumes serial execution)
- L2 cache effects (assumes all SRAM access)

## FP16 Conversion

### fp32_to_fp16 (IEEE 754 binary16)
```c
fp16_t fp32_to_fp16(fp32_t v);
```
- Handles: normal, subnormal, zero, infinity, NaN
- Rounding: round-to-nearest-even (ties to even)
- Overflow: clamp to ±infinity (0x7C00 / 0xFC00)
- Underflow: flush to zero (subnormals rounded to nearest representable)

### fp16_to_fp32
```c
fp32_t fp16_to_fp32(fp16_t h);
```
- Exact conversion (no precision loss)
- Handles subnormals by normalizing the mantissa

### Bias Handling Detail

Bias values are loaded into O-buffer as FP16. The `has_bias` flag triggers
FP16→FP32 expansion **in reverse order** to avoid self-clobbering:

```
O-buffer layout before bias init:
  [bias[0].lo][bias[0].hi][bias[1].lo][bias[1].hi] ...  (FP16 pairs)

Bias init (reverse iteration):
  for n = N-1 down to 0:
    for m = M-1 down to 0:
      read  FP16 from O[n*M+m]
      write FP32 to O[n*M+m]    // 4 bytes overwrites 2 FP16 slots

Reverse order ensures FP32 writes never clobber unread FP16 bias values
(because lower-address FP16 values were already read in earlier iterations).
```

The bias must be pre-expanded by the compiler: a bias vector of size N
is replicated M times to form an N×M matrix before loading.
This ensures the cmodel's bias init reads adjacent FP16 values.

## SRAM Layout

### W-Buffer (128 KB = 131,072 bytes)
```
Offset 0 ──────────────────────────────────────────── 131072
│  Static weights (lower 64 KB)  │  Tile workspace (upper 64 KB)  │
│  Preloaded at init             │  Overwritten per MMA tile      │
└────────────────────────────────┴────────────────────────────────┘
```

### A-Buffer (64 KB = 65,536 bytes)
```
Row-major FP16: A[K][M], K×M×2 bytes
Bump-allocated per operation, reused across ops.
```

### O-Buffer (64 KB = 65,536 bytes)
```
Row-major FP32: O[N][M], N×M×4 bytes
Bump-allocated per operation. Contains accumulator state.
```

## Performance Counters

```c
typedef struct {
    uint64_t total_dma_bytes;     // Total bytes transferred via DMA
    uint64_t total_mma_calls;     // Number of tu_mma() invocations
    uint64_t total_mma_tiles;     // Total 16×16×16 tiles executed
    uint64_t total_mma_flops;     // Effective FP16 multiply-adds (×2 for FMA)
    uint64_t estimated_cycles;    // Simplified cycle model
} tu_state_t;
```

## Testing

6 unit tests in `tests/test_cmodel.c`:
1. **FP16 round-trip:** fp32→fp16→fp32 identity for 1.0, 0.0, -2.5
2. **Identity MMA:** W=I₁₆, A=I₁₆ → O=I₁₆ (verifies systolic array correctness)
3. **Rectangular GEMM:** W[32][16]=0.5, A[16][8]=2.0 → O[32][8]=16.0
4. **Bias MMA:** W=0, A=0, bias=[0,1,2,...] → output matches bias expansion
5. **Edge tiles:** Non-multiple-of-16 dimensions (implicit in test 3)

All tests pass with exact or near-exact matching (last-bit FP rounding differences).
