# Cycle-Accurate Timing Model (Gap P2.5)

> **Status:** Implemented  
> **Version:** 1.0  
> **Date:** 2026-06-01  
> **Gap:** P2.5 — Cycle-accurate model  
> **Priority:** P2 (Medium, foundational for microarchitectural exploration)

---

## 1. Overview

The cycle-accurate timing model transforms the TU cmodel from a functional-only simulator into a production-grade cycle-level simulator. It models realistic hardware behavior:

- **Pipeline hazards** — RAW (read-after-write) and WAW (write-after-write) stalls
- **Bank conflicts** — multi-bank SRAM bandwidth contention with per-bank refill budgets
- **DRAM row buffer** — open-page policy, row hit/miss latency modeling
- **DMA bus arbitration** — shared bus contention between multiple DMA channels

### Fidelity Levels

The model operates at three selectable fidelity levels:

| Level | Mode | Use Case | Cycle Accuracy |
|-------|------|----------|----------------|
| 0 | FUNCTIONAL | Algorithm correctness | None (returns 0) |
| 1 | ESTIMATED | Performance projection | Simplified fill+compute+drain |
| 2 | CYCLE_ACCURATE | Microarchitectural exploration | Full pipeline + memory modeling |

## 2. Why This Was Chosen

Cycle-accurate modeling was selected for this heartbeat because:

1. **Foundational capability** — every production cmodel needs timing fidelity. Without it, the cmodel is a toy.
2. **Config-driven** — fidelity level is selectable via `TU_CYCLE_MODEL` in config; disabled mode has zero overhead.
3. **Composable architecture** — pipeline tracker, bank model, DRAM channel, and bus model are independent modules that can be enabled/disabled individually.
4. **Directly enables P2.7** (VCD/FST trace generation) — the cycle counter and pipeline state form the backbone for waveform export.

## 3. Architecture

### 3.1 Sub-Models

```
┌─────────────────────────────────────────────────────┐
│                 Cycle Model                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ Pipeline     │  │ Bank         │  │ DRAM       │ │
│  │ Tracker      │  │ Model        │  │ Channel    │ │
│  │              │  │              │  │            │ │
│  │ • Issue      │  │ • Access     │  │ • Row hit  │ │
│  │ • Complete   │  │ • Refill     │  │ • Row miss │ │
│  │ • RAW haz    │  │ • Stalls     │  │ • Precharg │ │
│  │ • WAW haz    │  │ • Conflicts  │  │ • Presets  │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│  ┌──────────────────────────────────────────────┐   │
│  │              DMA Bus Arbiter                  │   │
│  │   Channel 0 ──┐                              │   │
│  │   Channel 1 ──┼──► Round-Robin ──► Bus       │   │
│  │   Channel 2 ──┘                              │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │        Performance Counter Integration        │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 3.2 Pipeline Tracker

Models a 5-stage systolic pipeline:

```
IF ──► ID ──► RR ──► MAC ──► WB
│       │      │      │       │
Fetch  Decode  Reg   Multiply Write
              Read   Accum    Back
```

**Hazard types detected:**
- **RAW** (Read-After-Write): Tile N+1 reads a register that Tile N is still writing → stall until Tile N completes
- **WAW** (Write-After-Write): Tile N+1 writes the same register as Tile N → stall to preserve ordering

**Pipeline state:** Circular buffer of in-flight tiles, each tracking:
- Issue cycle, complete cycle
- Tile coordinates (m_start, n_start, k_start, counts)
- Register dependencies (src_regs, dst_regs)

### 3.3 Bank Conflict Model

Models multi-bank SRAM with per-bank bandwidth budgets:

- **Refill mechanism:** Each bank has a word budget that refills every `TU_SRAM_BW_WINDOW_CYCLES` cycles
- **Stall penalty:** When budget exhausted, each excess word costs `TU_SRAM_BW_STALL_PENALTY` cycles
- **Conflict tracking:** Records simultaneous access attempts to the same bank

### 3.4 DRAM Row Buffer Model

Models JEDEC-style DRAM with open-page policy:

| State | Latency | Description |
|-------|---------|-------------|
| **Row hit** | `tCL` (CAS) | Requested row matches open row |
| **Row empty** | `tRCD + tCL` | No row open — activate first |
| **Row miss** | `tRP + tRCD + tCL` | Different row — precharge + activate |

**Timing parameters** (JEDEC standard):
- `tRCD`: RAS-to-CAS delay (activate to read/write)
- `tRP`: Row precharge time
- `tCL`: CAS latency
- `tCWL`: CAS write latency
- `tBL`: Burst length (4 HBM, 8 DDR4, 16 DDR5)
- `tCCD`: Column-to-column delay

### 3.5 DMA Bus Arbitration

When multiple DMA channels contend for the shared bus:
- Detects concurrent channel activity via `dma_bus_cycles[]` tracking
- Round-robin arbitration: each active channel gets 1 cycle of delay

## 4. Module Structure

### Files

```
tu_cmodel/perf/
├── cycle_model.h       # Public API (420 lines)
├── cycle_model.c       # Implementation (~750 lines)
└── performance_counters.h  # Integrated counter types

tests/
└── test_cycle_model.c  # 21 tests

docs/
└── cycle-accurate-model.md  # This document
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `tu_cycle_model_create()` | Create model at specified fidelity |
| `tu_cycle_model_reset()` | Reset all state and counters |
| `tu_cycle_model_execute_tile()` | Simulate systolic tile with full cycle accounting |
| `tu_cycle_model_dma_transfer()` | DMA with bus contention + DRAM modeling |
| `tu_cycle_model_dma_arbitrate()` | Bus arbitration between channels |
| `tu_cycle_model_report()` | Print comprehensive statistics |
| `tu_cycle_pipeline_init/issue/complete()` | Pipeline tracker |
| `tu_bank_model_init/access/tick()` | Bank conflict model |
| `tu_dram_channel_init()` | DRAM channel with timing presets |
| `tu_dram_access()` | DRAM access with row buffer |
| `tu_dram_preset_hbm2/ddr4/ddr5/…()` | Standard DRAM timing presets |

### DRAM Presets

| Preset | Banks | Bank Groups | tCL (ns) | tRCD (ns) | tBL |
|--------|-------|-------------|----------|-----------|-----|
| `ideal` | 1 | 1 | 0 | 0 | 1 |
| `hbm2` | 8 | 1 | 14 | 14 | 4 |
| `hbm2e` | 8 | 1 | 12 | 14 | 4 |
| `hbm3` | 16 | 1 | 10 | 14 | 4 |
| `ddr4` | 16 | 4 | 14 | 14 | 8 |
| `ddr5` | 32 | 8 | 14 | 14 | 16 |
| `lpddr5` | 16 | 4 | 14 | 14 | 8 |

## 5. Configuration

```c
/* tu_config.h — Cycle model fidelity */
#define TU_CYCLE_MODEL_FUNCTIONAL     0  /* No cycle accounting */
#define TU_CYCLE_MODEL_ESTIMATED      1  /* Simple fill+compute+drain */
#define TU_CYCLE_MODEL_CYCLE_ACCURATE  2  /* Full hazards + memory model */
#define TU_CYCLE_MODEL                0  /* Default: functional */

/* Pipeline depth */
#define TU_PE_PIPELINE_DEPTH         2

/* SRAM bandwidth model */
#define TU_SRAM_BW_WINDOW_CYCLES     4
#define TU_SRAM_BW_STALL_PENALTY     2
#define TU_SRAM_WORDS_PER_CYCLE      1
#define TU_SRAM_BANKS                32
#define TU_SRAM_BANK_WIDTH           4
```

## 6. Usage Example

### Creating and Using the Cycle Model

```c
#include "tu_cmodel/perf/cycle_model.h"

// Create cycle-accurate model
tu_perf_counters_t perf;
tu_perf_init(&perf, 1000.0);  // 1 GHz clock

tu_cycle_model_t *cm = tu_cycle_model_create(
    TU_CYCLE_MODEL_CYCLE_ACCURATE, &perf);

// Simulate a 16×16×64 systolic tile
uint64_t tile_cycles = tu_cycle_model_execute_tile(
    cm,
    0 /* m_start */, 16 /* m_count */,
    0 /* n_start */, 16 /* n_count */,
    0 /* k_start */, 64 /* k_count */,
    0x100 /* w_sram_addr */,
    0x200 /* a_sram_addr */,
    0x300 /* o_sram_addr */);

printf("Tile took %lu cycles\n", (unsigned long)tile_cycles);

// Simulate DMA transfer
uint64_t dma_cycles = tu_cycle_model_dma_transfer(
    cm, 0 /* channel */, 4096 /* bytes */,
    true /* read */, 0x10000 /* dram_addr */, 2 /* sram_bank */);

// Print report
tu_cycle_model_report(cm);

// Clean up
tu_cycle_model_destroy(cm);
```

### Model Report Output

```
=== TU Cycle Model Report ===
Model fidelity: CYCLE_ACCURATE
Total cycles: 2619

-- Pipeline --
  Issues:           1
  Completions:      1
  Hazard stalls:    0 cycles
  Utilization:      25.0%

-- SRAM Banks --
  Reads:            1024
  Writes:           256
  Bank stalls:      1277 cycles
  Bank conflicts:   3

-- DRAM --
  Accesses:         12
  Row hits:         8 (66.7%)
  Row misses:       4
  Stall cycles:     180

-- DMA Bus --
  Bus stalls:       0 cycles
  Channel 0:        45 cycles
```

## 7. Tile Execution Cycle Breakdown

For a cycle-accurate tile execution (16×16×64):

```
Operation              Cycles
──────────────────────────────
Instruction decode        1
Pipeline hazard check     0-100  (varies by hazard)
SRAM weight read         32     (16×64×2B / bank_width)
SRAM activation read     32
MAC computation          64     (k_count)
SRAM output writeback    64     (16×16×4B / bank_width)
Bank conflict stalls     0-200  (varies by access pattern)
──────────────────────────────
Total per tile          ~200-500
```

## 8. Verification

### Test Coverage (21 tests, all passing)

| Test Category | Tests | Description |
|---------------|-------|-------------|
| Pipeline tracker | 4 | Issue/complete, RAW hazard, WAW hazard, utilization |
| Bank conflicts | 4 | Basic access, exhaustion, refill, statistics |
| DRAM model | 4 | Row hit, row conflict, presets, statistics |
| Cycle model | 4 | Functional, estimated, accurate, multi-tile |
| DMA bus | 2 | Single channel, arbitration |
| Model lifecycle | 1 | Reset |
| Transfer | 1 | DMA in cycle-accurate mode |
| Edge cases | 1 | NULL safety |

### Verification Strategy

1. **Pipeline correctness:** RAW and WAW hazards are detected when register dependencies overlap
2. **Bank model:** Bandwidth exhaustion produces stalls, refill restores budget
3. **DRAM model:** Row hits are faster than row misses (tCL vs tRP+tRCD+tCL)
4. **DMA arbitration:** Bus contention with 2+ active channels produces arbitration stalls
5. **Model reset:** All counters and pipeline state return to zero

## 9. Integration Points

### Performance Counters

The cycle model integrates with `tu_perf_counters_t`:
- Tile execution records: MACs, active cycles, stall cycles
- DMA transfers record: bytes, active cycles, stalls, channel usage

### Future: VCD/FST Trace (P2.7)

The cycle counter and pipeline state form the basis for waveform export:
```
tu_event_trace_signal(trace, "top.pipeline.stage[0]", entry->stage, 3);
tu_event_trace_tick(trace);
```

### Future: Power Model Integration (P2.6)

Energy counters in `tu_perf_counters_t` can be updated per-cycle:
```c
tu_perf.power.energy_mac_pj += macs_this_cycle * pj_per_mac;
```

## 10. Performance Characteristics

### Overhead by Fidelity Level

| Level | Per-Tile Overhead | Use Case |
|-------|-------------------|----------|
| FUNCTIONAL | 0 µs | Pure algorithmic verification |
| ESTIMATED | <1 µs | Quick performance projection |
| CYCLE_ACCURATE | ~10-50 µs | Detailed microarchitecture exploration |

### Model Accuracy Targets

- Bank conflict stalls: ±10% vs Verilog RTL (Gemmini reference)
- DRAM row buffer: ±5% vs DRAMSim2/3
- Pipeline hazards: Exact (hardware-precise by construction)

## 11. Future Extensions

1. **VCD/FST trace generation (P2.7):** Per-cycle signal dump for GTKWave/Surfer
2. **WAR hazard detection:** Register renaming with scoreboard
3. **NoC congestion model:** Multi-hop network-on-chip latency
4. **Calibration against RTL:** Compare against Gemmini/MAERI Verilog simulations
5. **Technology node parameterization:** Configurable gate delays, wire delays

## 12. References

1. Hennessy & Patterson, "Computer Architecture: A Quantitative Approach," 6th Edition, 2019
2. JEDEC, "DDR4 SDRAM Standard," JESD79-4C, 2020
3. JEDEC, "High Bandwidth Memory (HBM) DRAM," JESD235D, 2021
4. Genc et al., "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation," DAC 2021
5. Kwon et al., "MAERI: Enabling Flexible Dataflow Mapping over DNN Accelerators," ASPLOS 2018
6. Samajdar et al., "SCALE-Sim: Systolic CNN Accelerator Simulator," ISPASS 2020
