# Event Tracing — VCD Waveform Generation (P2.7)

> **Gap ID:** P2.7  
> **Status:** ✅ COMPLETE  
> **Date:** 2026-06-01  

---

## Overview

IEEE 1364-2001 VCD (Value Change Dump) waveform generation for hardware-level
debugging and cmodel-vs-RTL comparison. Produces files viewable in GTKWave,
Surfer, and other standard waveform viewers.

**Why it matters:** Without cycle-accurate tracing, debugging systolic array
behavior requires printf-based logging — slow, imprecise, and impossible to
cross-reference with RTL simulations. VCD is the industry standard format
for digital waveform exchange.

## Architecture

```
┌─────────────────────────────────┐
│ tu_config_t                     │
│  .trace_enabled                 │
│  .trace_file                    │
└──────────┬──────────────────────┘
           │ creates on demand
           ▼
┌──────────────────────────────────────┐
│ tu_event_trace_t                     │
│  ┌────────────────────────────────┐  │
│  │ Signal Registry                │  │
│  │  signals[0] → "TU.dma.active"  │  │
│  │  signals[1] → "TU.dma.state"   │  │
│  │  signals[2] → "TU.compute.cnt" │  │
│  │  ...                           │  │
│  └────────────────────────────────┘  │
│                                      │
│  │ tu_trace_signal()  ← change queue│
│  │ tu_trace_tick()    → VCD write   │
└──────────────────┬───────────────────┘
                   │
                   ▼
         /path/to/output.vcd
```

**Key design decisions:**

1. **Change detection** — values written only when they actually change.
   This makes VCD files compact and matches hardware (signals don't
   "re-transmit" unchanged values).

2. **Batched writes** — multiple signal changes in the same cycle are
   grouped under a single `#time` header.

3. **Buffered I/O** — uses `fprintf` with stdio buffering rather than
   raw `write()` calls. Acceptable because VCD files are debug artifacts,
   not performance-critical runtime paths.

4. **Signal ID system** — each signal gets a short ASCII identifier.
   1-bit: single char (e.g., `!`); multi-bit: same number of chars as bits;
   bus values rendered as binary strings.

## API Reference

### Lifecycle

```c
/* Create trace context, open file (truncates if exists) */
tu_event_trace_t *tu_trace_create(const char *filename,
                                   uint32_t max_signals);

/* Close file, free context, write end-of-file marker */
void tu_trace_close(tu_event_trace_t *trace);
```

### Signal Management

```c
/* Register a signal: id (short ASCII), name (hierarchical), width (bits) */
int tu_trace_add_signal(tu_event_trace_t *trace,
                         const char *id, const char *name,
                         tu_trace_signal_width_t width);
/* Returns signal index (0-based), or -1 on overflow/error */

/* Queue a value change — written on next tick */
void tu_trace_signal(tu_event_trace_t *trace,
                      int signal_index, uint64_t value);

/* Advance cycle and flush pending changes to file */
void tu_trace_tick(tu_event_trace_t *trace, uint64_t cycles);
```

### Signal Widths

| Enum | Bits | Typical Use |
|------|------|-------------|
| `TU_TRACE_SIG_1BIT`  | 1  | Enable/active flags |
| `TU_TRACE_SIG_4BIT`  | 4  | State machine states |
| `TU_TRACE_SIG_8BIT`  | 8  | Opcodes, counts |
| `TU_TRACE_SIG_16BIT` | 16 | Addresses |
| `TU_TRACE_SIG_32BIT` | 32 | Counters, accumulators |
| `TU_TRACE_SIG_64BIT` | 64 | Cycle counters, timestamps |

### Predefined Signal IDs

Standard single-character identifiers for common TU signals:

| Macro | ID | Signal |
|-------|-----|--------|
| `TU_TRACE_ID_CYCLE`          | `!` | Simulation cycle |
| `TU_TRACE_ID_DMA_CH0_STATE`  | `#` | DMA channel 0 state |
| `TU_TRACE_ID_DMA_CH1_STATE`  | `$` | DMA channel 1 state |
| `TU_TRACE_ID_DMA_CH2_STATE`  | `%` | DMA channel 2 state |
| `TU_TRACE_ID_COMPUTE_ACTIVE` | `&` | Compute pipeline active |
| `TU_TRACE_ID_COMPUTE_OPCODE` | `'` | MMU opcode |
| `TU_TRACE_ID_SRAM_ACCESS`    | `(` | SRAM access strobe |
| `TU_TRACE_ID_SRAM_BANK`      | `)` | SRAM bank select |
| `TU_TRACE_ID_CMDQ_DEPTH`     | `*` | Command queue depth |
| `TU_TRACE_ID_CMDQ_SUBMIT`    | `+` | Command submit strobe |
| `TU_TRACE_ID_DRAM_ACCESS`    | `,` | DRAM access strobe |
| `TU_TRACE_ID_TILE_COUNT`     | `-` | Tile iteration counter |

## Usage Example

```c
#include "tu_cmodel/perf/event_trace.h"

/* 1. Create trace context */
tu_event_trace_t *trace = tu_trace_create("sim_output.vcd", 64);

/* 2. Register signals */
int sig_active  = tu_trace_add_signal(trace, "!", "TU.dma.ch0.active",
                                       TU_TRACE_SIG_1BIT);
int sig_state   = tu_trace_add_signal(trace, "#", "TU.dma.ch0.state",
                                       TU_TRACE_SIG_8BIT);

/* 3. Simulate */
for (int cycle = 0; cycle < 1000; cycle++) {
    uint8_t dma_state = dma_get_state();

    tu_trace_signal(trace, sig_active,  dma_state != IDLE);
    tu_trace_signal(trace, sig_state,   dma_state);
    tu_trace_tick(trace, cycle);

    dma_step();
}

/* 4. Cleanup (writes $end marker) */
tu_trace_close(trace);

/* 5. View: gtkwave sim_output.vcd */
```

## VCD Output Format

```
$date
    Mon Jun  1 18:00:00 2026
$end
$version
    TU CModel Event Trace v1.0
$end
$timescale 1ns $end

$scope module TU_CORE $end
$var wire 1 ! TU_CORE.dma.ch0.active $end
$var wire 8 # TU_CORE.dma.ch0.state $end
$upscope $end

$enddefinitions $end
$dumpvars
b00000000 #
0!
$end
#0
b00000001 #
1!
#5
b00000010 #
0!
```

## Configuration

Add to `config/tu_config.json`:

```json
{
  "trace": {
    "enabled": false,
    "file": "trace_output.vcd"
  }
}
```

When `enabled` is `false`, all trace calls are no-ops (zero overhead).

## Cycle Accounting

Trace operations are designed for offline debugging and carry no
performance overhead in the cycle model:

- `tu_trace_create()` / `tu_trace_close()`: not on the critical path
- `tu_trace_signal()`: O(1) — single array write + compare
- `tu_trace_tick()`: O(N) where N = number of dirty signals
- When `trace_enabled == false`, all functions are no-ops

## Integration Points

Current instrumentation targets (to be wired incrementally):

| Component | Signals | Priority |
|-----------|---------|----------|
| DMA engine | ch0/1/2 state, active, byte count | High |
| Compute pipeline | opcode, active, cycle count | High |
| SRAM | access strobe, bank select, R/W | Medium |
| Command queue | depth, submit strobe | Medium |
| Dataflow dispatcher | pattern, PE utilization | Low |

## Tradeoffs

| Choice | Rationale |
|--------|-----------|
| VCD not FST | VCD is ASCII, human-readable, universally supported. FST is more compact but requires external library (libfst). |
| fprintf not write() | Buffered I/O is faster for many small writes. Debug-only code, not hot-path. |
| Change detection in trace layer | Pushes comparison cost to trace insertion (amortized O(1)) rather than tick-time sorting. |
| Fixed-size signal registry | No dynamic reallocation at runtime. `max_signals` chosen at init. |
| No signal groups | Simple flat registry. Hierarchy expressed in signal names. |

## Related Components

- **Performance counters** (`tu_cmodel/perf/performance_counters.h`) — aggregate statistics
- **Cycle model** (`tu_cmodel/perf/cycle_model.h`) — timing annotations
- **Config system** (`tu_cmodel/infra/config.h`) — enables/disables tracing
- **Gap P2.6** — CACTI power model (tracing provides the time dimension for energy integration)

## Viewing Traces

```bash
# GTKWave (Linux)
gtkwave sim_output.vcd

# Surfer (Rust, cross-platform)
surfer sim_output.vcd
```

## Tests

```bash
make test-trace    # Run trace-specific tests
make test          # Include in full suite
```

`tests/test_trace.c` — 21 tests covering:
- Context lifecycle (create/destroy)
- Signal registration (1/8/32-bit, overflow handling)
- Change detection (same value → no-op, different → write)
- Tick advancement (header writing, cycle counter)
- VCD file integrity ($date, $version, $timescale, $var, $dumpvars, time headers)
- Null/missing input handling
- Post-close safety
