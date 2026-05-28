# TU CModel — Command Queue

> **Gap ID:** E1 (No command queue → Command queue with ordering & deps)
> **Priority:** P0 (Critical)
> **Date:** 2026-05-28
> **Heartbeat:** Cycle 2

---

## What Changed

The TinyTU cmodel previously executed all operations via direct function calls — `tu_mma()`, `tu_dma_load_w()`, etc. There was no command queue, no dependency tracking, and no way to model async execution.

A hardware command queue has been added as `tu_cmodel/command_queue.h` and `tu_cmodel/command_queue.c`, integrated into the `tu_state_t` struct.

### Key Features

1. **Command submission** — Operations (DMA load/store, MMA, barriers, NOP) are submitted as commands with unique IDs
2. **Dependency tracking** — Commands can declare prerequisites; they won't issue until dependencies complete
3. **Barrier support** — Barriers enforce ordering: all prior commands must complete before subsequent ones issue
4. **Completion signaling** — Each command gets a completion signal; wait operations poll for completion
5. **Dual-mode execution** — Synchronous (functional) executes immediately; async mode advances via `tu_cmdq_tick()`
6. **Queue statistics** — Track submitted/completed/faulted counts, queue depth

---

## Why This Matters

A command queue is essential for hardware-accurate execution modeling:
- **Dependency tracking** enables the compiler to express operation ordering constraints
- **Barriers** model hardware synchronization (e.g., DMA must finish before MMA starts)
- **Async execution** enables modeling of DMA/compute overlap and software pipelining
- **Completion signals** are the foundation for interrupt-driven execution models
- **Queue depth** models hardware resource limits

---

## How It Works

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     tu_command_queue_t                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Circular Buffer (capacity = TU_ISA_QUEUE_DEPTH = 16)    │  │
│  │  ┌──────┬──────┬──────┬──────┬─────┬──────┐            │  │
│  │  │ cmd0 │ cmd1 │ cmd2 │ cmd3 │ ... │ cmdN │            │  │
│  │  └──────┴──────┴──────┴──────┴─────┴──────┘            │  │
│  │   head →                                    ← tail      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Sync mode: execute_command() called immediately on submit     │
│  Async mode: tu_cmdq_tick() scans for ready commands           │
└────────────────────────────────────────────────────────────────┘
```

### Command Lifecycle

```
PENDING  ──→  ISSUED  ──→  COMPLETED  (or FAULTED)
   │              │
   │  Dependencies │  execute_command()
   │  not yet met  │
   └──────────────┘
```

### Dependency Graph Example

```c
// MMA depends on both DMA loads finishing first
int dma_w = tu_cmdq_submit_dma_load(0, 0, w_data, w_size);
int dma_a = tu_cmdq_submit_dma_load(1, 0, a_data, a_size);

uint32_t deps[] = { dma_w, dma_a };
int mma = tu_cmdq_submit_mma(16, 16, 16, 0, 0, 0, false);
// (in async mode, use tu_cmdq_submit with explicit deps)
```

### Convenience API

| Function | Description |
|----------|-------------|
| `tu_cmdq_submit_mma(M,N,K,w,a,o,bias)` | Submit MMA, returns cmd_id |
| `tu_cmdq_submit_dma_load(ch,off,ptr,sz)` | Submit DMA load, returns cmd_id |
| `tu_cmdq_submit_dma_store(ch,off,ptr,sz)` | Submit DMA store, returns cmd_id |
| `tu_cmdq_submit_barrier()` | Submit barrier, returns cmd_id |
| `tu_cmdq_sync_all()` | Drain queue (wait all) |
| `tu_get_cmdq()` | Get raw command queue handle |

### Low-Level API

| Function | Description |
|----------|-------------|
| `tu_cmdq_create(capacity, sync)` | Create queue |
| `tu_cmdq_destroy(cq)` | Destroy queue |
| `tu_cmdq_submit(cq, opcode, op, ndeps, deps, out_id)` | Submit raw command |
| `tu_cmdq_barrier(cq)` | Submit barrier |
| `tu_cmdq_wait(cq, cmd_id, timeout)` | Wait for command |
| `tu_cmdq_tick(cq)` | Advance execution (async mode) |
| `tu_cmdq_get_depth(cq)` | In-flight count |
| `tu_cmdq_get_status(cq, id)` | Check command status |
| `tu_cmdq_get_counts(cq, &sub, &comp, &fault)` | Get counters |

### Supported Opcodes

| Opcode | Value | Description |
|--------|-------|-------------|
| `TU_CMD_NOP` | 0 | No operation |
| `TU_CMD_DMA_LOAD` | 1 | DMA host→SRAM |
| `TU_CMD_DMA_STORE` | 2 | DMA SRAM→host |
| `TU_CMD_MMA` | 3 | Matrix multiply-accumulate |
| `TU_CMD_SYNC` | 4 | Pipeline drain |
| `TU_CMD_BARRIER` | 5 | Ordering barrier |
| `TU_CMD_HALT` | 6 | Halt execution |
| (reserved) | 10-16 | Conv, Attention, Elementwise, etc. |

---

## Configuration

The command queue is configured via `tu_config.h`:

| #define | Default | Description |
|---------|---------|-------------|
| `TU_ISA_QUEUE_DEPTH` | 16 | Max commands in flight |
| `TU_ISA_DEP_CHECKING` | 0 | Hardware dependency checking |
| `TU_CYCLE_MODEL` | 0 | 0=functional(sync), 1=estimated, 2=cycle-accurate |

When `TU_CYCLE_MODEL == TU_CYCLE_MODEL_FUNCTIONAL` (default), the queue operates in synchronous mode — commands execute immediately on submission. This matches the current functional-model behavior.

---

## Verification

### Test Suite: 9 tests, all passing

| Test | What It Verifies |
|------|-----------------|
| Queue creation | Create/destroy, initial depth = 0 |
| MMA via CMDQ | Submit MMA, verify result identity matrix |
| Monotonic IDs | Command IDs strictly increase |
| Barrier | Commands before/after barrier all complete |
| Queue overflow | Queue rejects commands when full |
| DMA + MMA with deps | DMA→MMA chain with explicit dependencies |
| Standalone queue | Independent queue, NOP execution |
| Reset | Counters zero after reset |
| NOP execution | NOP completes successfully |

### Run Tests

```bash
make test-cmdq    # 9/9 tests pass
make test-cmodel  # 19/19 tests pass (backward compat)
make test-asm     # ASM interpreter unchanged
```

---

## Future Extensions

This command queue provides the foundation for:

- **Gap E2 (Software pipelining):** DMA tile N+1 while computing tile N — submit DMA and MMA with appropriate dependencies
- **Gap DM1 (Async DMA):** Set `async_mode=true`, use `tu_cmdq_tick()` to advance
- **Gap C1 (Rich ISA):** Add opcodes for Conv, Attention, Elementwise, Softmax, etc.
- **Gap E3 (Multi-context):** Multiple command queues per context
- **Gap DM5 (DMA priority/QoS):** Priority field in command descriptor

---

## Files

| File | Change |
|------|--------|
| `tu_cmodel/command_queue.h` | New — command queue interface and types |
| `tu_cmodel/command_queue.c` | New — implementation (circular buffer, dep tracking, barriers) |
| `tu_cmodel/tu_cmodel.h` | Include command_queue.h, add `cmdq` to state, add convenience API |
| `tu_cmodel/tu_cmodel.c` | Init/destroy cmdq, convenience functions for MMA/DMA/barrier |
| `Makefile` | Build command_queue.o, add `test-cmdq` target |
| `tests/test_command_queue.c` | New — 9 command queue tests |
| `docs/command-queue.md` | This document |
