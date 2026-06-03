# TU Debug & Observability Hooks (Gap I3)

> **Feature:** Production-grade debug infrastructure for TU cmodel
> **Gap ID:** I3 — Observability / debug hooks
> **Priority:** P2 (Medium)
> **Date:** 2026-06-03
> **Files:** `tu_cmodel/infra/tu_debug.h`, `tu_cmodel/infra/tu_debug.c`, `tests/test_debug.c`

---

## 1. Overview

The debug/observability module provides three subsystems critical for microarchitecture bring-up, debugging, and cmodel-vs-RTL discrepancy diagnosis:

1. **State Dump** — Structured dump of any TU core's internal state in text, JSON, or binary format
2. **Deterministic Replay** — Record→save→load→replay instruction traces with bit-exact verification
3. **Invariant Assertions** — Configurable self-checking assertions at key execution points

All features are zero-overhead when disabled and minimal-overhead when enabled.

## 2. Why This Matters

The redesign document identifies observability as critical (gap I3):

> "Internal state dump (partial sums, tile buffers, pipeline stages) for selected ops; deterministic replay from instruction trace; built-in self-checking assertions on invariants (accumulator range, memory alignment); critical for microarchitecture bring-up and cmodel-vs-RTL discrepancy diagnosis"

Without these hooks, debugging cmodel behavior requires recompilation, printf-debugging, and manual inspection. With them, developers can:

- Compare SRAM state between two TU core instances byte-by-byte
- Record an instruction trace from a passing run and replay it against a modified config
- Detect NaN/Inf propagation, misaligned accesses, and bounds violations automatically
- Generate JSON snapshots for automated CI comparison

## 3. State Dump

### 3.1 API

```c
size_t tu_debug_dump_state(const tu_core_t *core, FILE *out,
                           tu_dump_format_t format, uint32_t flags);
```

### 3.2 Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `TU_DUMP_TEXT` | Human-readable ASCII with section headers and per-bank stats | Interactive debugging |
| `TU_DUMP_JSON` | Structured JSON with key-value pairs | Automated CI comparison, Python tooling |
| `TU_DUMP_BINARY` | Compact binary snapshot with header + SRAM contents | Fast save/restore, replay |

### 3.3 Dump Flags

| Flag | Bitmask | Content |
|------|---------|---------|
| `TU_DUMP_SRAM` | 0x0001 | Full W/A/O SRAM contents (hex preview + stats) |
| `TU_DUMP_DMA` | 0x0002 | DMA engine state + per-channel stats |
| `TU_DUMP_CMD_QUEUE` | 0x0004 | Command queue entries + status |
| `TU_DUMP_COUNTERS` | 0x0008 | Performance counters + TOPS estimate |
| `TU_DUMP_PIPELINE` | 0x0010 | Pipeline stage registers |
| `TU_DUMP_DATAFLOW` | 0x0020 | Dataflow plugin state |
| `TU_DUMP_CHECKSUMS` | 0x0040 | CRC32 checksums per SRAM region |
| `TU_DUMP_ALL` | 0xFFFF | Everything |

### 3.4 Binary Snapshot Format

```
Offset  Field              Size    Description
0x00    magic              4       "TUDB" (0x54554442)
0x04    version            4       1
0x08    core_id            4       Which core
0x0C    timestamp          8       Unix timestamp (ns)
0x14    cycle              8       Cycle counter
0x1C    flags              4       Dump flags mask
0x20    pe_rows            4       PE array dimensions
0x24    pe_cols            4
0x28    sram_w_size        4       SRAM region sizes (bytes)
0x2C    sram_a_size        4
0x30    sram_o_size        4
0x34    checksum_w         4       Pre-computed CRC32
0x38    checksum_a         4
0x3C    checksum_o         4
0x40    reserved           16      Future expansion
0x50    [SRAM data]        var     Optional (if TU_DUMP_SRAM set)
```

### 3.5 Checksums

```c
uint32_t tu_debug_checksum_sram(const tu_core_t *core);
```

Returns a combined 32-bit CRC32 checksum (CRC32(W) ^ CRC32(A) ^ CRC32(O)). Useful for fast consistency checks — a change in any bit of any SRAM region changes the combined checksum.

### 3.6 SRAM Diff

```c
int tu_debug_diff_sram(const tu_core_t *core_a, const tu_core_t *core_b,
                       char region, FILE *out);
```

Byte-by-byte comparison of a single SRAM region between two cores. Reports the number of differing bytes, with the first 64 diffs listed individually.

## 4. Deterministic Replay

### 4.1 Concept

The replay system captures every instruction executed by the command queue (opcode, operands, pre/post SRAM checksums) into a compact binary trace. This trace can be:

1. **Saved** to a file for offline analysis
2. **Replayed** against any TU core instance (different config, different machine)
3. **Verified** — checksums are compared after each instruction; mismatches signal non-determinism

### 4.2 API

```c
// Start recording
int  tu_debug_record_start(tu_replay_trace_t *trace, uint32_t capacity);

// Record one instruction
int  tu_debug_record_instr(tu_replay_trace_t *trace,
      uint64_t cycle, uint8_t opcode, uint8_t flags,
      uint16_t dim0, uint8_t dim1, uint8_t dim2,
      uint64_t immediates,
      uint32_t checksum_before, uint32_t checksum_after);

// Stop recording
uint32_t tu_debug_record_stop(tu_replay_trace_t *trace);

// Save/load traces
size_t   tu_debug_record_save(const tu_replay_trace_t *trace, FILE *out);
uint32_t tu_debug_record_load(tu_replay_trace_t *trace, FILE *in);

// Replay against a core
int      tu_debug_replay_execute(tu_core_t *core,
               const tu_replay_trace_t *trace, FILE *out);

// Cleanup
void     tu_debug_record_destroy(tu_replay_trace_t *trace);
```

### 4.3 Trace Entry Format

```c
typedef struct {
    uint64_t cycle;            // Cycle when instruction issued
    uint8_t  opcode;           // TU ISA opcode
    uint8_t  flags;            // Instruction flags
    uint16_t dim0;             // Operand dimensions
    uint8_t  dim1;
    uint8_t  dim2;
    uint16_t reserved;
    uint64_t immediates;       // Immediate operands
    uint32_t checksum_before;  // SRAM checksum before execution
    uint32_t checksum_delta;   // Checksum change (after XOR before)
} tu_replay_entry_t;  // 32 bytes per entry
```

At 1M entries max (configurable via `TU_REPLAY_MAX_ENTRIES`), the trace buffer is ~32 MB.

### 4.4 Usage Example

```c
// Record
tu_replay_trace_t trace = {0};
tu_debug_record_start(&trace, 0);  // 0 = use default max

// ... run workload, call tu_debug_record_instr() for each instruction ...

tu_debug_record_stop(&trace);

// Save
FILE *f = fopen("trace.bin", "wb");
tu_debug_record_save(&trace, f);
fclose(f);

// Later: load and replay
tu_replay_trace_t loaded = {0};
f = fopen("trace.bin", "rb");
tu_debug_record_load(&loaded, f);
fclose(f);

FILE *report = fopen("replay_report.txt", "w");
int mismatches = tu_debug_replay_execute(core, &loaded, report);
fclose(report);

tu_debug_record_destroy(&trace);
tu_debug_record_destroy(&loaded);
```

## 5. Invariant Assertions

### 5.1 Categories

| Category | Enum | Checks |
|----------|------|--------|
| Range | `TU_ASSERT_CAT_RANGE` | Accumulator values: no NaN/Inf |
| Alignment | `TU_ASSERT_CAT_ALIGNMENT` | Memory accesses aligned to data type |
| Bounds | `TU_ASSERT_CAT_BOUNDS` | Reads/writes within allocated regions |
| Pipeline | `TU_ASSERT_CAT_PIPELINE` | Tile dimensions match PE array |
| Dataflow | `TU_ASSERT_CAT_DATAFLOW` | Operation direction matches dataflow mode |
| Dtype | `TU_ASSERT_CAT_DTYPE` | Data type invariants |

### 5.2 Severity Modes

| Mode | Behavior |
|------|----------|
| `TU_ASSERT_IGNORE` | Completely disabled |
| `TU_ASSERT_WARN` | Log warning via TU_LOG_WARN, continue execution (default) |
| `TU_ASSERT_ERROR` | Log error via TU_LOG_ERR, continue execution |
| `TU_ASSERT_ABORT` | Log error and call `abort()` — strict checking |

### 5.3 Configuration

```c
// Set all categories to WARN (default)
tu_debug_assert_set_all(TU_ASSERT_WARN);

// Make range checks strict
tu_debug_assert_set_mode(TU_ASSERT_CAT_RANGE, TU_ASSERT_ABORT);

// Disable alignment checks
tu_debug_assert_set_mode(TU_ASSERT_CAT_ALIGNMENT, TU_ASSERT_IGNORE);

// Query statistics
tu_assert_stats_t stats;
tu_debug_assert_get_stats(&stats);
printf("Total assertions: %u, violations: %u\n",
       stats.total_checks, stats.total_violations);
```

### 5.4 Integration Points

Assertion functions should be called at key execution points:
- After every MAC operation: `tu_debug_assert_range(accumulator_value, "MMA")`
- Before every SRAM access: `tu_debug_assert_alignment(addr, elem_size, "SRAM")`
- Before every DMA transfer: `tu_debug_assert_bounds(addr, size, limit, "DMA")`
- At tile dispatch: `tu_debug_assert_tile_dims(M, N, pe_rows, pe_cols, "Tile")`

## 6. Full Debug Report

```c
void tu_debug_report(const tu_core_t *core, FILE *out);
```

Generates a comprehensive report including:
- Full state dump (text format, all sections)
- Configuration summary
- Assertion violation statistics by category

## 7. Tests

25 tests covering all subsystems:

| Subsystem | Tests | Coverage |
|-----------|-------|----------|
| State dump | 6 | All formats, all flag combinations, null-stream safety |
| Checksums | 3 | Initial state, after write, idempotency |
| SRAM diff | 2 | Identical cores, different cores |
| Replay | 4 | Start/stop, record, save/load roundtrip, buffer full |
| Assertions | 8 | NaN/Inf/finite, alignment, bounds, tile dims, dataflow, modes, stats |
| Full report | 1 | End-to-end report generation |
| **Total** | **25** | **100% pass rate** |

Run: `make test-debug`

## 8. Design Decisions

1. **CRC32 for checksums** — Fast, hardware-friendly, well-understood collision properties. Sufficient for detecting SRAM corruption.

2. **Binary snapshots with versioned header** — Enables forward/backward compatibility as the cmodel evolves. Snapshots from old versions are detected via magic+version mismatch.

3. **`handle_assert` returns void** — The assertion function's return value indicates whether the invariant held (true=pass, false=fail), independent of the configured severity mode. This allows callers to act on violations programmatically while the handler does logging/aborting.

4. **No external dependencies** — Pure C11 + stdio. No JSON library (hand-rolled to avoid dependency bloat), no compression library.

5. **Default: WARN** — Assertions default to warning mode so they don't break existing workflows. Strict mode (ABORT) is opt-in for bring-up and debug sessions.
