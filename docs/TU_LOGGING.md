# TU Structured Logging System (Gap Q2)

> **Priority:** P1 (High) | **Status:** Implemented | **Date:** 2026-05-30
>
> Production-grade structured logging with severity levels, component tagging,
> runtime filtering, file/line annotations, and execution tracing.

---

## 1. What This Is

The structured logging system (Gap Q2) replaces ad-hoc `fprintf(stderr, ...)` calls
with a unified logging framework:

1. **Severity levels:** ERROR, WARNING, INFO, DEBUG, TRACE — filterable at runtime
2. **Component tagging:** Every log message carries a subsystem tag (CORE, MMA, DMA, MEM, ISA, etc.)
3. **Timestamps:** Relative cycle counter shown in all log output
4. **Color output:** ANSI colors for quick visual scanning (green=INFO, yellow=WARN, red=ERROR)
5. **Execution trace:** Buffer of up to 65,536 events with VCD export for waveform viewers
6. **Zero-overhead fast path:** Compile-time level checks via macros

## 2. Why This Was Chosen

**Gap Analysis Context:** The production redesign (docs/PRODUCTION_TU_REDESIGN.md)
identifies Q2 as a P1 priority because:

1. **Debuggability:** Raw `fprintf(stderr, ...)` gives no way to filter noise.
   With severity levels, you can run at ERROR-only in production, INFO for normal
   operation, or TRACE for microarchitecture bring-up.

2. **Observability:** Component tagging lets you focus on one subsystem:
   "show me only DMA events" or "suppress memory noise."

3. **Verification:** The execution trace buffer enables post-hoc analysis:
   "what happened at cycle 4532?" — look at the trace, not the noise.

4. **RTL co-simulation readiness:** VCD export is the standard interchange
   format for hardware verification (GTKWave, Verilator, VCS). When an
   RTL implementation exists, the cmodel trace can be compared signal-by-signal.

5. **Multi-instance ready:** With structured logging, multi-core configs
   can prefix each message with a core ID (future enhancement).

## 3. Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  TU_LOG(comp, level, fmt, ...)                               │
│    │                                                         │
│    ▼                                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Macro check: level <= g_log_cfg.min_level?          │   │
│  │    NO → skip (zero cost)                              │   │
│  │    YES → tu_log_emit()                                │   │
│  └──────────────────────────────────────────────────────┘   │
│    │                                                         │
│    ▼                                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  tu_log_emit():                                       │   │
│  │    [timestamp] COMP  LEVEL  file:line message         │   │
│  │    → stderr (configurable)                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Also: tu_trace_event() → g_trace_buffer[65536]              │
│         tu_trace_export_vcd() → .vcd file (GTKWave)          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Log Output Format

```
[    1234] MMA   INFO  GEMM 32×16×16: 8 tiles, 16384 FLOPS
[    1234] DMA   WARN  Channel stall: bank conflict at addr=0x1000
[    1234] CORE  ERROR Not initialized
```

Fields:
- `[  1234]` — relative cycle counter (8 chars, right-aligned)
- `MMA   ` — component tag (5 chars, left-aligned)
- `INFO  ` — severity level (ANSI colored when enabled)
- `GEMM ...` — message (printf-style format string)

### 3.3 Severity Levels

| Level | Value | Typical Use |
|-------|-------|------------|
| `TU_LOG_NONE` | 0 | Silent — all output suppressed |
| `TU_LOG_ERROR` | 1 | Fatal errors, abort conditions |
| `TU_LOG_WARNING` | 2 | Recoverable issues: bank conflicts, dataflow fallback |
| `TU_LOG_INFO` | 3 | Normal operational: init, config, MMA completion |
| `TU_LOG_DEBUG` | 4 | Detailed: tile boundaries, DMA descriptor submission |
| `TU_LOG_TRACE` | 5 | Per-instruction/per-cycle: every MAC, every DMA word |

### 3.4 Component Tags

| Tag | Enum | Subsystem |
|-----|------|-----------|
| CORE | `TU_COMP_CORE` | Top-level orchestration, init, lifecycle |
| MMA | `TU_COMP_MMA` | Systolic array / compute engine |
| DMA | `TU_COMP_DMA` | DMA engine and descriptors |
| MEM | `TU_COMP_MEM` | Memory system: SRAM, DRAM, allocation |
| ISA | `TU_COMP_ISA` | ISA encoder/decoder, ASM interpreter |
| CMD | `TU_COMP_CMD` | Command queue |
| DF | `TU_COMP_DF` | Dataflow plugins |
| PREC | `TU_COMP_PREC` | Precision: FP16/FP8/BF16/INT8 conversion |
| PERF | `TU_COMP_PERF` | Performance counters and cycle model |

## 4. Configuration

### 4.1 Compile-Time Defaults

```c
// tu_config.h
#define TU_LOG_LEVEL_DEFAULT   TU_LOG_INFO    // Default severity threshold
#define TU_LOG_USE_COLOR       1              // ANSI colors
#define TU_LOG_SHOW_TIMESTAMPS 1              // Cycle timestamps
#define TU_LOG_SHOW_FILE_LINE  0              // Source file:line (DEBUG builds)
#define TU_TRACE_MAX_EVENTS    65536          // Trace buffer capacity
```

### 4.2 Runtime API

```c
// Initialize (called automatically by tu_init())
tu_log_init();

// Change severity threshold
tu_log_set_level(TU_LOG_DEBUG);  // Show debug and above
tu_log_set_level(TU_LOG_TRACE);  // Show everything
tu_log_set_level(TU_LOG_NONE);   // Silent mode
tu_log_set_level(TU_LOG_ERROR);  // Errors only

// Check current level
tu_log_level_t lvl = tu_log_get_level();

// Get config for custom modifications
tu_log_config_t *cfg = tu_log_get_config();
cfg->use_color = false;         // Disable ANSI colors
cfg->show_file_line = true;     // Show source locations
cfg->output = my_log_file;      // Redirect to file
```

### 4.3 Usage in Code

```c
// Error — fatal, unrecoverable
TU_LOG_ERR(TU_COMP_CORE, "not initialized");

// Warning — recoverable, unexpected
TU_LOG_WARN(TU_COMP_DF, "dataflow id=%d not registered", df_id);
TU_LOG_WARN(TU_COMP_MEM, "bank conflict at addr=%08x", addr);

// Info — normal operation
TU_LOG_INFO(TU_COMP_CORE, "Initialized: %u×%u PE, %u KB SRAM", rows, cols, kb);

// Debug — detailed diagnostics
TU_LOG_DBG(TU_COMP_DMA, "DMA descriptor submitted: ch=%d dst=%08x sz=%u",
           ch, dst, size);

// Trace — per-cycle events
TU_LOG_TRACE(TU_COMP_MMA, "tile [%u,%u] k_step=%u acc=%.6f", mi, ni, ki, acc);
```

## 5. Execution Trace

### 5.1 Recording Events

```c
// Record a trace event at the current cycle
tu_trace_event(TU_COMP_MMA, 0x01, M, N, K, 0);

// Advance the cycle counter
tu_trace_set_cycle(current_cycle + tile_cycles);
```

Trace events capture:
- `cycle`: when the event occurred
- `component`: which subsystem
- `opcode`: event type (0x01=MMA start, 0x10=DMA load, 0x11=DMA store, etc.)
- `operand[0..3]`: up to 4 32-bit operands (dimensions, addresses, sizes)

### 5.2 Exporting VCD

```c
FILE *vcd = fopen("tu_trace.vcd", "w");
tu_trace_export_vcd(vcd);
fclose(vcd);
// View with: gtkwave tu_trace.vcd
```

The VCD format is compatible with GTKWave, Surfer, and Verilator for
waveform visualization. Signals: `comp[7:0]`, `op[7:0]`, `op0[31:0]` through `op3[31:0]`.

### 5.3 Integration with tu_mma()

The main `tu_mma()` function records a trace event at operation start:

```c
tu_trace_event(TU_COMP_MMA, 0x01, (uint32_t)M, (uint32_t)N, (uint32_t)K, 0);
```

Future: per-tile trace events, DMA transfer events, command queue events.

## 6. Testing

**Test file:** `tests/test_logging.c`
**Run:** `make test-logging`

| Test | What it validates |
|------|-------------------|
| init and default config | Log config is initialized, default level is INFO |
| severity level filtering | `tu_log_set_level()` and `tu_log_get_level()` round-trip |
| all severity levels emit | ERROR through TRACE macros don't crash; messages appear |
| component tags distinct | All 8 component tags emit without error |
| trace event recording | Events recorded with correct cycle/component/operand data |
| VCD trace export | Export produces valid VCD with $date, $timescale, signal definitions |
| integration with tu_init | tu_init() initializes logging; MMA generates trace events |

Test results: **7/7 pass**.

## 7. Integration with Existing Code

The logging system is integrated into `tu_cmodel.c`:

| Old Code | New Code |
|----------|----------|
| `fprintf(stderr, "TU ERROR: %s overflow...", ...)` | `TU_LOG_ERR(TU_COMP_MEM, "%s overflow...", ...)` |
| `fprintf(stderr, "TU ERROR: not initialized\n")` | `TU_LOG_ERR(TU_COMP_CORE, "not initialized")` |
| `fprintf(stderr, "TU WARNING: dataflow...", ...)` | `TU_LOG_WARN(TU_COMP_DF, "dataflow...", ...)` |
| (none — new) | `TU_LOG_INFO(TU_COMP_CORE, "TinyTU CModel initializing...")` |
| (none — new) | `TU_LOG_INFO(TU_COMP_CORE, "Initialized: %u×%u PE...", ...)` |

Other source files (`tu_sram.c`, `tu_dma.c`, `tu_asm.c`, `command_queue.c`) retain
their existing stderr output for now to minimize disruption. Migration is
straightforward: replace `fprintf(stderr, ...)` with the appropriate `TU_LOG_*`
macro.

## 8. Future Work

### Per-Core Instance Logging
Multi-core TU configurations should tag each message with the originating
core ID: `[c0] MMA INFO ...` vs `[c1] MMA INFO ...`.

### Log File Rotation
For long-running workloads, automatic log file rotation based on size
or cycle count.

### Conditional Trace Recording
Currently all `tu_trace_event()` calls are recorded. Future: enable
trace only for specific op types or address ranges (like hardware
trigger conditions for a logic analyzer).

### Performance Counter Integration
Log performance counters at regular intervals (every 1000 cycles) for
throughput-over-time graphs.

## 9. Related Gaps

- **Q2 (this):** Structured logging with levels, tags, and execution trace
- **E4 (Power modeling):** Log energy counters via the same framework
- **V2 (Comprehensive tests):** Logging enables better test diagnostics
- **I3 (Observability/debug hooks):** Trace buffer is the foundation for internal state dump
- **Q4 (Documentation generation):** Structured log output can feed into automated docs

---

**Author:** Hermes Agent heartbeat | **Gap ID:** Q2 | **Priority:** P1 (High)
