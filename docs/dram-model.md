# TU CModel — DRAM Model

> **Gap ID:** M1 (No DRAM model → Multi-level memory with bandwidth/latency modeling)
> **Priority:** P0 (Critical)
> **Date:** 2026-05-29
> **Heartbeat:** Cycle 5a

---

## What Changed

The TinyTU cmodel previously had no off-chip memory model. DMA transfers to/from host DRAM used simple `ceil(bytes/32)` cycle estimates with no bandwidth modeling, no latency accounting, and no contention detection. This meant performance numbers were aspirational — they had no connection to memory system constraints.

A pluggable DRAM model has been added with support for 7 built-in DRAM types plus custom parameters.

### Key Features

1. **7 built-in DRAM types** — ideal, HBM2, HBM2e, HBM3, DDR4, DDR5, LPDDR5 — each with real-world bandwidth and latency parameters
2. **Custom DRAM** — Arbitrary user-specified `tu_dram_params_t` for design space exploration
3. **Cycle-accurate access modeling** — Every read/write returns cycle cost with stall accounting
4. **Bandwidth metering** — Refill-based bandwidth budget prevents instantaneous infinite throughput
5. **Channel parallelism** — Multi-channel DRAM with per-channel availability tracking and address interleaving
6. **Row buffer conflict modeling** — Optional row-buffer hit/miss penalty for realistic DRAM timing
7. **Transfer estimation** — `tu_dram_estimate_transfer()` for DMA planning without actual execution
8. **Statistics & reporting** — Per-access counters, effective bandwidth, utilization, formatted reports

---

## Why This Matters

A DRAM model is **foundational** for credible performance modeling. Without it:

- **Performance numbers are fiction** — `ceil(bytes/32)` ignores bandwidth limits, latency, and contention
- **No memory bottleneck visibility** — The compiler has no feedback about whether a workload is compute-bound or memory-bound
- **No design space exploration** — Can't ask "what if we had HBM3 vs DDR5?"
- **No DMA engine calibration** — DMA cycle estimates have no physical basis

With the DRAM model:

- The DMA engine can query `tu_dram_estimate_transfer()` for accurate transfer timing
- The compiler can use `tu_dram_get_stats()` to understand bandwidth utilization
- Architecture exploration can swap DRAM types at runtime and see performance impact
- Cycle-accurate mode can model bandwidth contention between concurrent DMA channels

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DMA Engine                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ DMA Ch 0    │  │ DMA Ch 1    │  │ DMA Ch 2    │         │
│  │ (Weights)   │  │ (Activns)   │  │ (Outputs)   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│                    ┌─────▼─────┐                            │
│                    │ DRAM Model │  ◄── DMA queries timing   │
│                    │            │                            │
│                    │ • BW meter │                            │
│                    │ • Channels │                            │
│                    │ • Latency  │                            │
│                    │ • Stats    │                            │
│                    └────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### Component: `tu_dram_model_t`

| Field | Description |
|-------|-------------|
| `type` | DRAM type enum (ideal, HBM2, etc.) |
| `params` | Timing and geometry parameters |
| `stats` | Accumulated access statistics |
| `current_cycle` | Simulator clock (advanced via `tu_dram_tick()`) |
| `bandwidth_available` | Remaining BW budget in current window |
| `channel_available_cycle[]` | Per-channel next-available cycle for contention modeling |

### Bandwidth Metering

Bandwidth is not modeled as "bytes per access" — that would allow infinite instantaneous bandwidth. Instead, a **refill-based meter** tracks bandwidth consumption over a sliding window:

```
Every BW_WINDOW cycles (default 1000):
  bandwidth_available = Peak_BW_GBps × BW_WINDOW / core_freq

On each access:
  bandwidth_available -= access_bytes
  If bandwidth_available < 0: stall until next refill
```

This correctly models that a burst of many small accesses can saturate the DRAM bus.

### Channel Interleaving

For multi-channel DRAM (HBM2 has 8, HBM3 has 16), addresses are interleaved at burst granularity:

```c
channel = (addr / burst_length) % num_channels
```

Each channel tracks its own availability. Concurrent accesses to different channels can proceed in parallel; accesses to the same channel serialize.

---

## API Reference

### Lifecycle

```c
tu_dram_model_t *tu_dram_create(tu_dram_type_t type);
tu_dram_model_t *tu_dram_create_custom(const tu_dram_params_t *params, const char *name);
void tu_dram_destroy(tu_dram_model_t *dram);
void tu_dram_reset(tu_dram_model_t *dram);
```

### Access & Timing

```c
void tu_dram_read(tu_dram_model_t *dram, uint64_t addr, uint32_t num_bytes, uint64_t *cycles, uint64_t *stall);
void tu_dram_write(tu_dram_model_t *dram, uint64_t addr, uint32_t num_bytes, uint64_t *cycles, uint64_t *stall);
uint64_t tu_dram_estimate_transfer(tu_dram_model_t *dram, uint32_t num_bytes, bool is_read);
void tu_dram_tick(tu_dram_model_t *dram);
```

### Information

```c
const char *tu_dram_type_name(tu_dram_type_t type);
const char *tu_dram_get_name(const tu_dram_model_t *dram);
void tu_dram_get_stats(const tu_dram_model_t *dram, tu_dram_stats_t *stats);
void tu_dram_print_stats(const tu_dram_model_t *dram, FILE *out);
uint64_t tu_dram_peak_bw_per_cycle(const tu_dram_model_t *dram, double core_clock_ghz);
```

### Configuration

```c
void tu_dram_set_core_clock(tu_dram_model_t *dram, double core_clock_ghz);
void tu_dram_set_row_modeling(tu_dram_model_t *dram, bool enabled);
```

---

## Built-in DRAM Types

| Type | Bandwidth | Channels | Read Latency | Use Case |
|------|-----------|----------|-------------|----------|
| **Ideal** | ∞ | 1 | 0 | Functional-only testing |
| **HBM2** | 256 GB/s | 8 | 50 cycles | Data center inference (TPUv3, V100) |
| **HBM2e** | 460 GB/s | 8 | 50 cycles | High-end training (A100) |
| **HBM3** | 819 GB/s | 16 | 40 cycles | Next-gen accelerators (H100) |
| **DDR4** | 25.6 GB/s | 1 | 75 cycles | Edge/embedded (DDR4-3200) |
| **DDR5** | 51.2 GB/s | 1 | 65 cycles | Edge/embedded (DDR5-6400) |
| **LPDDR5** | 51.2 GB/s | 1 | 60 cycles | Mobile/edge inference |

All parameters (clock, BW, latency, burst, channels, banks) are taken from public datasheets and industry references.

---

## Configuration

### YAML (`config/tu_config.yaml`)

```yaml
memory:
  dram:
    type: "hbm2"              # ideal | hbm2 | hbm2e | hbm3 | ddr4 | ddr5 | lpddr5 | custom
    bandwidth_gbps: 256.0     # Override peak bandwidth
    model_row_conflicts: false # Enable row buffer hit/miss modeling
    core_clock_ghz: 1.0       # Core clock for BW-per-cycle calculation
```

### C Header (`tu_config.h`)

```c
#define TU_DRAM_TYPE              TU_DRAM_IDEAL
#define TU_DRAM_BANDWIDTH_GBPS    256.0
#define TU_DRAM_CHANNELS          8
#define TU_DRAM_MODEL_ROW_HIT     0
```

---

## Usage Example

```c
#include "tu_cmodel/memory/dram_model.h"

// Create HBM2 model
tu_dram_model_t *dram = tu_dram_create(TU_DRAM_HBM2);

// Time a DMA transfer: load 64 KB of weights from DRAM
uint64_t cycles, stall;
tu_dram_read(dram, 0x10000, 65536, &cycles, &stall);
printf("64 KB DRAM read: %lu cycles (stall: %lu)\n", cycles, stall);

// Estimate without executing
uint64_t est = tu_dram_estimate_transfer(dram, 65536, true);
printf("Estimated: %lu cycles\n", est);

// Print statistics
tu_dram_print_stats(dram, stdout);

// Explore design space: what if we had HBM3?
tu_dram_model_t *hbm3 = tu_dram_create(TU_DRAM_HBM3);
uint64_t est_hbm3 = tu_dram_estimate_transfer(hbm3, 65536, true);
printf("HBM3 would be %.1fx faster\n",
       (double)est / est_hbm3);

tu_dram_destroy(dram);
tu_dram_destroy(hbm3);
```

---

## Integration with DMA Engine

The DMA engine currently estimates transfer cycles with `ceil(bytes / bus_width_bytes)`. Once integrated:

```c
// In tu_dma_execute_desc():
uint64_t xfer_cycles = tu_dram_estimate_transfer(
    g_tu_dram, desc->total_bytes, desc->direction == TU_DMA_DIR_HOST_TO_TU);
desc->cycles_completed = current_cycle + xfer_cycles;
```

This will enable accurate DMA timing, bandwidth contention between channels, and memory-bound vs compute-bound analysis.

---

## Tests

### Running

```bash
make test-dram
```

### Coverage (12 tests)

| Test | What it verifies |
|------|-----------------|
| Create all types | All 8 DRAM types instantiate with correct names |
| Custom parameters | User-specified `tu_dram_params_t` works |
| Ideal zero latency | Ideal DRAM returns zero cycle cost |
| HBM2 timing | Non-ideal types report non-zero cycle cost |
| Statistics accumulation | Read/write counters increment correctly |
| Bandwidth estimation | `tu_dram_estimate_transfer()` returns plausible values |
| Null safety | All API functions handle NULL without crashing |
| Reset | `tu_dram_reset()` clears all statistics |
| Tick advances | `tu_dram_tick()` increments the cycle counter |
| Peak BW per cycle | `tu_dram_peak_bw_per_cycle()` correct at different clocks |
| Print stats | `tu_dram_print_stats()` produces output without crashing |
| All types valid | Every preset type has non-zero BW, channels, banks, burst |

---

## Limitations (Future Work)

1. **No row-buffer tracking state** — Row conflict modeling is currently a flat penalty per access, not actual row-buffer state tracking
2. **No DRAM power model** — Energy per read/write not yet modeled (deferred to Gap E4: power modeling)
3. **No multi-rank support** — Single rank per channel
4. **Not yet integrated with DMA engine** — DMA still uses `ceil(bytes/32)`; integration deferred to next heartbeat
5. **Bandwidth window hardcoded** — 1000-cycle BW metering window; should be configurable

---

## Files

| File | Purpose |
|------|---------|
| `tu_cmodel/memory/dram_model.h` | DRAM model interface |
| `tu_cmodel/memory/dram_model.c` | DRAM model implementation (~320 lines) |
| `config/tu_config.yaml` | Added `memory.dram` section |
| `tu_cmodel/tu_config.h` | Added `TU_DRAM_*` defines |
| `tests/test_dram.c` | 12-test DRAM test suite |
| `docs/dram-model.md` | This document |
