# TU CModel — Parametric PE Array & Configuration System

> **Gap IDs:** A1 (Configurability), A2 (Systolic array dimensions), A6 (Memory banking)
> **Priority:** P0 (Critical)
> **Date:** 2026-05-28
> **Heartbeat:** Cycle 1

---

## What Changed

The TinyTU cmodel was previously hard-coded to a 16×16 PE array with a compile-time guard:

```c
#if TU_PE_ROWS != 16 || TU_PE_COLS != 16
#error "Legacy tu_cmodel.h assumes 16x16 PE array. Update config or code."
#endif
```

This guard has been **removed**. The cmodel is now fully parameterized — any PE array dimensions, SRAM sizes, and memory banking factors defined in `config/tu_config.yaml` → `tu_config.h` are used at runtime.

### Key Changes

1. **Removed the 16×16 hard-code guard** from `tu_cmodel.h`
2. **`tu_state_t` now uses modular components:**
   - `tu_sram_region_t` (banked SRAM) instead of flat `uint8_t` byte arrays
   - `tu_dma_engine_t` (DMA engine) for transfer accounting
   - `tu_runtime_config_t` for active configuration
3. **New API:** `tu_init_with_config(const tu_runtime_config_t *cfg)` allows runtime PE dimension overrides without recompilation
4. **`tu_mma()`** uses `g_tu.rt_cfg.pe_rows` and `g_tu.rt_cfg.pe_cols` for all tiling decisions
5. **`tu_print_stats()`** now reports the active PE array dimensions

---

## Why This Matters

Parametrization is the foundation for all other production features:
- **Architecture exploration:** Test 32×32, 64×64, 128×128 arrays without code changes
- **Performance modeling:** Different PE sizes have different utilization, latency, and area tradeoffs
- **Compiler auto-tuning:** The compiler can query PE dimensions to optimize tiling
- **Test coverage:** Single test suite validates all PE configurations

---

## How It Works

### Configuration Pipeline

```
config/tu_config.yaml  →  tu_config.h (compile-time #defines)
                                     ↓
                          tu_runtime_config_t (runtime override)
                                     ↓
                          tu_init_with_config()  →  g_tu.rt_cfg
                                     ↓
                          tu_mma() reads pe_rows/pe_cols from rt_cfg
```

### Runtime Override Example

```c
#include "tu_cmodel/tu_cmodel.h"

int main() {
    // Default: 16×16 PE array
    tu_init();

    // Or: 32×32 PE array
    tu_runtime_config_t cfg = tu_config_default();
    cfg.pe_rows = 32;
    cfg.pe_cols = 32;
    tu_init_with_config(&cfg);

    // All subsequent operations use 32×32 tiling
    tu_mma(64, 64, 64, ...);
}
```

### Tiling Behavior

With `PE_ROWS=R, PE_COLS=C`, `tu_mma(M, N, K)` tiles the computation:

```
M-tiles: ceil(M / R)
N-tiles: ceil(N / C)
K-tiles: ceil(K / C)
```

Edge tiles (partial tiles) are handled correctly — the last tile in each dimension processes only the remaining rows/columns.

---

## Verification

### Tested PE Configurations

| PE Array | Test Matrix | Result |
|----------|------------|--------|
| 16×16 | 16×16×16 identity | PASS |
| 32×32 | 32×32×32 identity | PASS |
| 8×8 | 16×16×16 identity | PASS |
| 16×16 | 32×16×16 identity (tall M) | PASS |
| 4×8 | 16×16×16 identity (non-square PE) | PASS |
| 16×16 | 48×48×48 identity (non-power-of-2) | PASS |
| 16×16 | 20×20×20 identity (non-multiple-of-tile) | PASS |
| 2×2 | 4×4×4 diagonal matrix | PASS |
| 4×3 | 7×5×9 edge tiles | PASS |

### Backward Compatibility

All existing tests pass unchanged:
- FP16 round-trip conversion (including NaN/Inf/subnormal)
- MMA identity, known-value, and bias tests
- ASM interpreter smoke test
- ONNX compiler integration test

### Test Command

```bash
make test-cmodel    # 19/19 tests pass
make test-asm       # ASM interpreter smoke test
```

---

## Memory Banking (Gap A6)

SRAM is now managed via `tu_sram_region_t` which provides:
- **Configurable banking** (`TU_SRAM_BANKS`, default 32)
- **Bank width** (`TU_SRAM_BANK_WIDTH`, default 4 bytes)
- **Conflict detection** (`TU_CONFLICT_MODEL`): `none`, `detect` (log warning), or `stall_cycle` (model penalty)
- **Per-bank statistics:** reads, writes, conflicts, stall cycles
- **Latency accounting:** `TU_LATENCY_SRAM_READ = 1`, `TU_LATENCY_SRAM_WRITE = 1`

Each SRAM region (W-buffer, A-buffer, O-buffer) has independent banking. The bank index for a byte address is:

```c
bank = (addr / bank_width) % bank_count
```

---

## Configuration Reference

All parameters are in `config/tu_config.yaml` with corresponding `#define` constants in `tu_config.h`:

| YAML Path | #define | Default | Description |
|-----------|---------|---------|-------------|
| `compute.pe_array.rows` | `TU_PE_ROWS` | 16 | PE array height |
| `compute.pe_array.cols` | `TU_PE_COLS` | 16 | PE array width |
| `compute.pe_array.pipeline_depth` | `TU_PE_PIPELINE_DEPTH` | 2 | Pipeline stages per MAC |
| `memory.sram.w_buffer_kb` | `TU_SRAM_W_SIZE_KB` | 128 | Weight buffer size |
| `memory.sram.a_buffer_kb` | `TU_SRAM_A_SIZE_KB` | 64 | Activation buffer size |
| `memory.sram.o_buffer_kb` | `TU_SRAM_O_SIZE_KB` | 64 | Output buffer size |
| `memory.banking.banks` | `TU_SRAM_BANKS` | 32 | Number of SRAM banks |
| `memory.banking.bank_width_bytes` | `TU_SRAM_BANK_WIDTH` | 4 | Bytes per bank word |

### Runtime Overridable

The `tu_runtime_config_t` struct allows overriding at runtime:

| Field | Type | Description |
|-------|------|-------------|
| `pe_rows` | `uint16_t` | PE array rows |
| `pe_cols` | `uint16_t` | PE array cols |
| `sram_w_size` | `uint32_t` | W-buffer size (bytes) |
| `sram_a_size` | `uint32_t` | A-buffer size (bytes) |
| `sram_o_size` | `uint32_t` | O-buffer size (bytes) |
| `counters_enabled` | `bool` | Enable performance counters |
| `detailed_stalls` | `bool` | Track per-stall-reason counters |
| `trace_enabled` | `bool` | Enable event tracing |
| `verify_enabled` | `bool` | Enable golden verification |
| `verify_tolerance` | `double` | Verification error tolerance |

---

## Files Modified

| File | Change |
|------|--------|
| `tu_cmodel/tu_cmodel.h` | Removed 16×16 guard, added `tu_init_with_config()`, modernized `tu_state_t` |
| `tu_cmodel/tu_cmodel.c` | Rewritten to use modular SRAM/DMA, parametric PE dimensions |
| `tu_cmodel/tu_sram.c` | Existing banking module (no changes needed) |
| `tu_cmodel/tu_dma.c` | Existing DMA module (no changes needed) |
| `tests/test_cmodel.c` | 7 new parameterized tests, 19 total tests |
| `docs/parametric-pe-array.md` | This document |
