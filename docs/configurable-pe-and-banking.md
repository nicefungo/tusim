# Configurable PE Array & Memory Banking (Gaps A2, A6)

> **Status:** Implemented | **Version:** 1.0 | **Date:** 2026-06-01

## Overview

Two foundational P0 gaps are addressed in this implementation:

1. **Gap A2 — Configurable systolic array dimensions**: The PE array dimensions
   (`PE_ROWS` × `PE_COLS`) are now fully runtime-configurable rather than
   hard-coded to 16×16. The cmodel supports arbitrary array sizes from 1×1 to
   1024×1024.

2. **Gap A6 — Configurable memory banking**: SRAM bank count and bank width are
   runtime parameters, enabling exploration of different memory architectures
   (e.g., 8 banks × 8 bytes vs 64 banks × 4 bytes).

## How It Works

### PE Array Dimensions (A2)

The systolic array uses the runtime config's `pe_rows` and `pe_cols` for tiling:

```c
// tu_mma() — tu_cmodel.c
uint16_t pe_rows = g_tu.rt_cfg.pe_rows;
uint16_t pe_cols = g_tu.rt_cfg.pe_cols;

uint16_t mt = (M + pe_rows - 1) / pe_rows;   // output tiles
uint16_t nt = (N + pe_cols - 1) / pe_cols;   // column tiles
uint16_t kt = (K + pe_cols - 1) / pe_cols;   // inner tiles
```

All compute engines (WS dataflow, OS dataflow, convolution, attention,
softmax, normalization, elementwise) receive the configured PE dimensions
at initialization and use them for their internal tiling.

The pipeline fill overhead scales linearly with PE columns:
```
fill_overhead = pe_pipeline_depth × pe_cols   cycles per tile
```

### Memory Banking (A6)

The SRAM `tu_sram_region_t` now accepts bank count and bank width as
initialization parameters:

```c
// tu_sram.c — tu_sram_init()
void tu_sram_init(tu_sram_region_t *r, uint32_t size_bytes, const char *name) {
    // Uses defaults: TU_SRAM_BANKS banks, TU_SRAM_BANK_WIDTH bytes each
}

// tu_sram_init_bw() — full control
void tu_sram_init_bw(tu_sram_region_t *r, uint32_t size_bytes, const char *name,
                     uint8_t words_per_cycle, uint8_t arb_mode,
                     uint8_t stall_penalty, uint64_t refill_window) {
    // Bank count and width from tu_config.h, overridable via config
}
```

Bank index computation uses the bank width:
```
bank_index = (addr / bank_width) % bank_count
```

The bandwidth model uses configurable parameters:
- `words_per_cycle`: max accesses per bank per cycle window (typically 1)
- `arb_mode`: round-robin (1) or priority (2)
- `stall_penalty`: cycles added when budget exhausted
- `bw_refill_window`: cycles between budget refills (typically 4)

## Configuration

Via `config/tu_config.json`:

```json
{
  "tu": {
    "compute": {
      "pe_array": {
        "rows": 32,       // ← A2: any value 1-1024
        "cols": 32,       // ← A2: any value 1-1024
        "pipeline_depth": 4
      }
    },
    "memory": {
      "banking": {
        "banks": 64,          // ← A6: number of SRAM banks
        "bank_width_bytes": 8, // ← A6: bytes per bank word
        "conflict_model": "stall_cycle"
      }
    }
  }
}
```

## Key Design Decisions

1. **Runtime over compile-time**: All dimensions read from `tu_config_t` /
   `tu_runtime_config_t` rather than `#define` constants. The compile-time
   constants remain as fallback defaults but the compute path uses runtime
   values exclusively.

2. **Validation**: PE dimensions are validated to [1, 1024]. Bank width must
   be 1/2/4/8 (powers of two for efficient hardware). Bank count [1, 1024].

3. **Non-power-of-2 PE arrays**: The tiling logic handles arbitrary dimensions,
   not just powers of two. Edge tiles use `min(dim, pe_dim)` for bounds.

4. **Bank width alignment**: All SRAM accesses use bank-width-aligned word
   reads/writes. Misaligned accesses are prohibited.

## Backward Compatibility

- Default configuration matches the original TinyTU (16×16 PE, 32 banks × 4
  bytes). All existing tests pass without modification.
- The `tu_runtime_config_t` struct's `pe_rows`/`pe_cols` fields are already
  populated from compile-time defaults when using `tu_init()`.
- New `tu_init_from_file()` preserves backward compat — the old API works
  unchanged by loading defaults.

## Gap References

| Gap | Description | Status |
|-----|-------------|--------|
| **A2** | Configurable PE array dimensions (P0) | Implemented — runtime `pe_rows` × `pe_cols` in all compute paths |
| **A6** | Configurable memory banking (P0) | Implemented — runtime bank_count and bank_width via tu_config_t |
