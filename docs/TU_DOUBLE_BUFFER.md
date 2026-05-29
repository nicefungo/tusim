# TU Double Buffering (Ping-Pong Buffers)

> **Gap A7:** Double/triple-buffered scratchpads for DMA/compute overlap.
> **Status:** Implemented — 2026-05-30
> **Files:** `tu_cmodel/memory/double_buffer.{h,c}`, `tests/test_double_buffer.c`
> **Modified:** `tu_cmodel/tu_sram.{h,c}` (db pointer + active-buffer-aware access)

## Overview

Double buffering (ping-pong buffering) enables DMA transfers and systolic array computation to overlap in time. Without double buffering, execution is strictly sequential:

```
Without DB:  DMA(tile1) → Compute(tile1) → DMA(tile2) → Compute(tile2)
             Total time = D + C + D + C = 2(D + C)

With DB:     DMA(tile1) → Compute(tile1) ──────────────→
                          DMA(tile2)    → Compute(tile2)
             Total time ≈ D + max(C, D) + max(C, D) + C ≈ 2·max(C, D)
```

When compute dominates (C > D), double buffering hides all DMA latency. When DMA dominates (D > C), the compute pipeline stalls waiting for data.

This is a fundamental optimization in every production accelerator:
- **Google TPU:** Software-managed double-buffered scratchpads
- **Gemmini:** Double-buffered scratchpad with tile-level software pipelining
- **NVIDIA TensorCore:** Async copy (TMA) overlapping with MMA compute
- **Eyeriss:** Ping-pong global buffer for layer-wise pipelining

## Architecture

### Data Structure

Each `tu_sram_region_t` gains an optional `db` pointer:

```
tu_sram_region_t
├── banks.data      ← primary buffer (always allocated)
├── banks.bw_banks  ← bandwidth model
├── db              ← double-buffer state (NULL = disabled)
│   ├── shadow_data ← shadow buffer (same size as primary)
│   ├── active_idx  ← 0=primary active, 1=shadow active
│   ├── swap_count  ← atomic swap counter
│   └── overlapped_cycles ← compute cycles saved
```

### Swap Semantics

Buffer swap is modeled as an atomic pointer exchange — it takes 0 cycles in hardware (a single register write to toggle a mux).

```
Before swap:  active=primary, shadow=shadow_data (has tile N+1)
After swap:   active=shadow,  shadow=primary (ready for tile N+2)
```

### DMA/Compute Overlap Model

The performance benefit is tracked via `overlapped_cycles`:

1. DMA writes tile N+1 into shadow buffer (`dma_to_shadow_cycles` tracked)
2. Compute reads tile N from active buffer
3. When DMA finishes, `tu_sram_record_overlapped_cycles(min(D, C))` is called
4. `tu_sram_swap_buffers()` makes tile N+1 the active buffer
5. Repeat

The overlapped cycles represent time that would have been serial DMA time but was instead concurrent with compute.

## API Reference

### Enabling/Disabling

```c
// Enable double buffering (allocates shadow buffer)
int tu_sram_enable_double_buffer(tu_sram_region_t *r);

// Disable and free shadow buffer (preserves active data in primary)
void tu_sram_disable_double_buffer(tu_sram_region_t *r);

// Query
bool tu_sram_is_double_buffered(const tu_sram_region_t *r);
```

### Buffer Operations

```c
// Atomic swap: exchanges active and shadow buffers
// Returns new swap count (0 = double buffering not enabled)
uint64_t tu_sram_swap_buffers(tu_sram_region_t *r);

// Get raw pointer to active buffer (for compute)
uint8_t *tu_sram_get_active_ptr(tu_sram_region_t *r);

// Get raw pointer to shadow buffer (for DMA writes)
uint8_t *tu_sram_get_shadow_ptr(tu_sram_region_t *r);
```

### DMA Integration

```c
// Notify that DMA wrote `bytes` to shadow buffer in `cycles`
void tu_sram_notify_shadow_write(tu_sram_region_t *r,
                                  uint32_t bytes, uint64_t cycles);

// Check if shadow has fresh data (written since last swap)
bool tu_sram_is_shadow_dirty(const tu_sram_region_t *r);
```

### Performance Tracking

```c
// Record compute cycles that overlapped with DMA
void tu_sram_record_overlapped_cycles(tu_sram_region_t *r, uint64_t cycles);

// Get total overlapped cycles
uint64_t tu_sram_get_overlapped_cycles(const tu_sram_region_t *r);

// Get full statistics
void tu_sram_get_db_stats(const tu_sram_region_t *r, tu_db_stats_t *stats);
void tu_sram_print_db_stats(const tu_sram_region_t *r);
```

### SRAM Access Transparency

All `tu_sram_read()`/`tu_sram_write()`/`tu_sram_read_bulk()`/`tu_sram_write_bulk()`/`tu_sram_raw_ptr()` calls automatically route to the currently active buffer when double buffering is enabled. Client code does not need to be modified.

## Usage Example

### Tiled GEMM with Double-Buffered Weight Loading

```c
tu_sram_region_t w_buf, a_buf, o_buf;
tu_sram_init(&w_buf, 128*1024, "W-buf");
tu_sram_enable_double_buffer(&w_buf);  // Double-buffer weights

tu_sram_init(&a_buf, 64*1024,  "A-buf");
tu_sram_init(&o_buf, 64*1024,  "O-buf");

for (int tile = 0; tile < num_tiles; tile++) {
    // DMA: load tile N+1 weights into shadow while compute uses tile N
    uint8_t *shadow = tu_sram_get_shadow_ptr(&w_buf);
    uint64_t dma_cycles = dma_load_from_host(shadow, weights[tile + 1], tile_size);
    tu_sram_notify_shadow_write(&w_buf, tile_size, dma_cycles);

    // Compute: use active buffer (has tile N)
    uint64_t compute_cycles = systolic_array_compute(
        tu_sram_get_active_ptr(&w_buf),  // Uses active (tile N)
        tu_sram_raw_ptr(&a_buf),
        tu_sram_raw_ptr(&o_buf));

    // Record overlap: compute and DMA ran concurrently
    uint64_t overlapped = (dma_cycles < compute_cycles) ? dma_cycles : compute_cycles;
    tu_sram_record_overlapped_cycles(&w_buf, overlapped);

    // Swap: tile N+1 becomes active, old active becomes shadow for tile N+2
    tu_sram_swap_buffers(&w_buf);
}
```

### Cycle Accounting

```
Without DB:  total_cycles = Σ(D_i + C_i)           (sequential)
With DB:     total_cycles = Σ(D_i + C_i) - overlapped_cycles
Speedup:     (Σ(D_i + C_i)) / (Σ(D_i + C_i) - overlapped_cycles)
```

## Limitations (by Design)

| Limitation | Rationale |
|------------|-----------|
| Double only (not triple) | Models most common accelerator design; triple buffering adds complexity with diminishing returns |
| Same-size buffers | Required for pointer swap; mismatched sizes would need data copy |
| Swap is 0-cycle | Models hardware mux toggle; real hardware has 1-cycle pipeline bubble |
| No partial swap | Entire buffer swaps atomically; partial swaps would need segment tracking |
| Shadow writes overwrite | DMA to shadow always overwrites; no merge semantics |
| Per-region only | Each SRAM region is independently double-buffered; cross-region coordination is the compiler's responsibility |

## ISA Integration

Future ISA extensions (Gap E2, software pipelining) will expose double buffering:

```
DMA_LOAD_SHADOW  W, tile_N+1    ; Load next tile into shadow buffer
MMA              W, A, O        ; Compute current tile from active
SWAP_BUFFER      W              ; Exchange active ↔ shadow
```

The SWAP_BUFFER instruction is the ISA-level representation of `tu_sram_swap_buffers()`.

## Verification

10 tests covering:
1. Default disabled state
2. Enable double buffering
3. Swap semantics (data visibility)
4. Shadow write notification
5. Multiple swaps
6. Disable (data preservation)
7. Overlapped cycle tracking
8. Statistics integrity
9. Active pointer changes on swap
10. Double-enable idempotency

## Configuration

Double buffering is a runtime feature — no compile-time config flags needed. Enable per-region as needed:

```yaml
# Conceptual config addition (future config.yaml)
memory:
  double_buffered_regions: ["w_buffer"]  # Or: all, none, [w, a, o]
```

## References

- Gap A7: `docs/PRODUCTION_TU_REDESIGN.md` §3.1
- Gap E2: Software pipelining (enabled by double buffering)
- TPU: Jouppi et al., ISCA 2017 — §4.2 "Double-buffered weight FIFO"
- Gemmini: Genc et al., DAC 2021 — §III.C "Double-Buffered Scratchpads"
- NVIDIA H100: White Paper — §3.5 "Asynchronous Tensor Memory Access (TMA)"
