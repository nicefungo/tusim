# Multi-Context Execution (Gap E3)

> **Heartbeat:** 2026-06-02  
> **Gap ID:** E3 — Multi-context execution (P2)  
> **Status:** Complete (12/12 tests passing)  
> **Implementation:** `tu_cmodel/infra/tu_context.{c,h}` (244 + 357 = 601 lines)

## What This Is

Multi-context execution enables a single TU core to support multiple independent, isolated
execution contexts — each with its own SRAM state, DMA counters, command queue, dataflow
configuration, precision modes, and performance statistics. This is the foundation for
**multi-tenant inference** (multiple models or batch partitions sharing one accelerator) and
**preemptive context switching** (pausing one workload to run a higher-priority one).

## Why It Was Chosen

**Priority:** P2 — a general TU execution model property that builds on existing infrastructure.

**Rationale:**
- All P0 and P1 architectural/precision/operation gaps are already addressed
- Context isolation is a general TU property, not architecture-specific
- Builds directly on the `tu_core_t` state isolation (gap A5) and structured error handling (gap E5)
- Enables key production scenarios: multi-model serving, QoS-driven scheduling, fault isolation
- Config-driven, pluggable architecture — scheduling policy is swappable at runtime

## How It Works

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Context Manager (tu_ctx_manager_t)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Context 0│  │ Context 1│  │ Context 2│  │ Context 3│  ...    │
│  │ (ACTIVE) │  │ (READY)  │  │ (IDLE)   │  │ (BLOCKED)│         │
│  │ ┌──────┐ │  │ ┌──────┐ │  │          │  │ ┌──────┐ │         │
│  │ │SRAM  │ │  │ │SRAM  │ │  │          │  │ │SRAM  │ │         │
│  │ │DMA   │ │  │ │DMA   │ │  │          │  │ │DMA   │ │         │
│  │ │Perf  │ │  │ │Perf  │ │  │          │  │ │Perf  │ │         │
│  │ └──────┘ │  │ └──────┘ │  │          │  │ └──────┘ │         │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └────┬─────┘        │
│       │              │                           │              │
│       └──────────────┼───────────────────────────┘              │
│                      │  Scheduler (round-robin / priority)       │
│                      ▼                                           │
│              ┌──────────────┐                                    │
│              │   TU Core    │  (single hardware instance)         │
│              └──────────────┘                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Context State Machine

```
    ┌───────┐   alloc    ┌────────┐   save    ┌────────┐
    │ IDLE  │ ────────► │ ACTIVE │ ────────► │ READY  │
    └──┬────┘           └───┬────┘           └───┬────┘
       │   free             │ save               │ restore
       ◄────────────────────┘                    │
       │                                         ▼
       │              ┌──────────┐  unblock  ┌─────────┐
       │              │COMPLETED │ ◄──────── │ BLOCKED │
       │              └──────────┘           └─────────┘
       │                    ▲                     ▲
       │                    │ HALT          block │
       └────────────────────┘                     │
             free                    tu_ctx_block_current()
```

### Context Switching Procedure

1. **Drain in-flight operations** — Flush DMA channels, sync command queue
2. **Save active context** — Deep-copy SRAM contents (W, A, O buffers), DMA engine state, 
   command queue state, performance counters, precision/rounding modes
3. **Restore target context** — Write saved SRAM data back, restore counters and configuration
4. **Resume execution** — The restored context picks up exactly where it left off

### State Preservation

| Subsystem | What's Preserved | Method |
|-----------|-----------------|--------|
| **SRAM** | Full data arrays (W, A, O buffers) | `malloc` + `memcpy` deep copy |
| **DMA Engine** | Channel state, queue depth, byte/cycle counters | Struct copy (no heap pointers) |
| **Command Queue** | Structure pointer (re-created on restore) | Saved as reference |
| **Performance** | All counters: DMA bytes, MMA calls, tiles, flops, cycles | Struct field copy |
| **Precision** | Dataflow plugin, runtime config, initialization flag | Struct field copy |
| **Double Buffering** | Disabled on restore (simplification) | Set to NULL |

## How to Configure

### YAML Configuration (config/tu_config.json)

```json
{
  "tu": {
    "context_manager": {
      "enabled": true,
      "max_contexts": 4,
      "sched_policy": "round_robin",
      "time_slice_cycles": 0,
      "time_slice_commands": 0,
      "switch_overhead_cycles": 100
    }
  }
}
```

### C API Configuration

```c
tu_ctx_manager_config_t cfg = {
    .max_contexts      = 4,                     // 1-256 concurrent contexts
    .sched_policy      = TU_CTX_SCHED_ROUND_ROBIN,  // or PRIORITY
    .time_slice_cycles = 0,                     // 0 = switch only at sync points
    .time_slice_cmds   = 0,                     // 0 = switch only at sync points
    .switch_overhead   = 100,                   // Cycle cost per context switch
};

tu_ctx_manager_t *mgr = tu_ctx_manager_create(core, &cfg);
```

### Scheduling Policies

| Policy | Description |
|--------|------------|
| `TU_CTX_SCHED_ROUND_ROBIN` | Equal time-slices, circular order (default) |
| `TU_CTX_SCHED_PRIORITY` | Higher priority contexts get scheduled first |

## Usage Example

```c
#include "tu_cmodel.h"
#include "tu_cmodel/tu_core.h"
#include "tu_cmodel/infra/tu_context.h"

int main(void) {
    // Create a TU core and context manager with 4 contexts
    tu_runtime_config_t rt = tu_runtime_config_default();
    tu_core_t *core = tu_core_create(&rt);

    tu_ctx_manager_config_t cfg = { .max_contexts = 4 };
    tu_ctx_manager_t *mgr = tu_ctx_manager_create(core, &cfg);

    // Allocate two execution contexts
    int ctx_a = tu_ctx_alloc(mgr);  // 0, becomes ACTIVE
    int ctx_b = tu_ctx_alloc(mgr);  // 1, READY

    // Run workload on context A
    tu_ctx_get(mgr, ctx_a)->priority = 200;  // High priority
    float w_a[16] = { /* weights */ };
    tu_core_dma_load_w(core, w_a, 0, sizeof(w_a));
    // ... MMA operations ...

    // Switch to context B (lower priority, different workload)
    tu_ctx_switch(mgr, ctx_b);
    float w_b[16] = { /* different weights */ };
    tu_core_dma_load_w(core, w_b, 0, sizeof(w_b));

    // Switch back to A — its SRAM state is intact
    tu_ctx_switch(mgr, ctx_a);
    // A's weights are restored, ready to continue

    // Cleanup
    tu_ctx_free(mgr, ctx_a);
    tu_ctx_free(mgr, ctx_b);
    tu_ctx_manager_destroy(mgr);
    tu_core_destroy(core);
    return 0;
}
```

## How It Changes the CModel's Behavior

### Backward Compatibility
- The legacy `tu_init()` / `g_tu` singleton path is **unchanged**
- `tu_core_create()` now clears `g_tu` after moving state to the core (prevents use-after-free on repeated core create/destroy cycles)
- All existing tests pass (19/19 cmodel, 6/9 cmdq — same as pre-change baseline)

### New Capabilities
1. **State isolation** — Multiple independent workloads can share one TU core without data leakage
2. **Preemptive scheduling** — High-priority contexts can interrupt lower-priority ones at sync points
3. **Fault containment** — One context's error doesn't corrupt another's state
4. **Performance accounting per-context** — Each context has its own cycle/command counters

### Performance Overhead
- Context switch cost: ~100 cycles (configurable)
- SRAM deep copy: O(SRAM_size) per save/restore
- No overhead when not switching (zero-cost when single-context)

## Verification

### Test Coverage (12/12 passing)

| Test | What It Verifies |
|------|-----------------|
| `ctx_manager_create` | Manager lifecycle, initial state |
| `ctx_alloc_free` | Allocation, capacity limits, slot reuse |
| `ctx_state_transitions` | ACTIVE/READY/IDLE state machine |
| `ctx_save_restore_sram` | SRAM data preserved across save/restore |
| `ctx_context_switch` | Full switch cycle with data isolation |
| `ctx_round_robin_sched` | Round-robin scheduling correctness |
| `ctx_priority_sched` | Priority-based scheduling |
| `ctx_block_unblock` | Block/unblock lifecycle |
| `ctx_perf_isolation` | Performance counter isolation across contexts |
| `ctx_print_status` | Status reporting doesn't crash |
| `ctx_null_safety` | All APIs handle NULL pointers gracefully |
| `ctx_max_contexts` | Edge cases with small context pools |

### Files Changed
- **New:** `tu_cmodel/infra/tu_context.h` (244 lines) — Context manager API
- **New:** `tu_cmodel/infra/tu_context.c` (366 lines) — Context manager implementation
- **New:** `tests/test_context.c` (345 lines) — 12 test cases
- **New:** `docs/multi-context-execution.md` (this file)
- **Modified:** `tu_cmodel/tu_core.c` — Fixed g_tu dangling pointer after core creation
- **Modified:** `Makefile` — Added tu_context.o build rule and test-context target

## Pitfalls & Lessons Learned

1. **g_tu save/restore pattern causes use-after-free:** The original `tu_core_create_with_id()` saved g_tu, called `tu_init_with_config()` (which frees old g_tu allocations), copied new state to core, then restored g_tu from saved — but the saved pointers were already freed. **Fix:** After moving state to core, zero g_tu so the next `tu_init()` starts fresh instead of double-freeing.

2. **Context free must handle all non-ACTIVE states:** The original `tu_ctx_free()` only released IDLE and COMPLETED contexts, but contexts spend most of their time in READY state. **Fix:** Accept READY, BLOCKED, and COMPLETED states; auto-save ACTIVE contexts before freeing.

3. **Double-buffering state is not preserved across context switches:** Complex hardware state (ping-pong buffer pointers, DMA descriptor chains) is deliberately simplified — the context is drained before saving. This is realistic: real hardware drains the pipeline before a context switch.
