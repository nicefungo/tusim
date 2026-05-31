# Multi-Core TU Cluster (Gap A5)

> **Status:** Implemented  
> **Gap:** A5 — Multi-instance / multi-core (P1)  
> **Version:** 1.0  
> **Date:** 2026-06-01  
> **Files:** `tu_cmodel/tu_core.h`, `tu_cmodel/tu_core.c`, `tu_cmodel/tu_cluster.h`, `tu_cmodel/tu_cluster.c`, `tests/test_multicore.c`

---

## Table of Contents

1. [Overview](#overview)
2. [Why This Gap Matters](#why-this-gap-matters)
3. [Architecture](#architecture)
4. [API Reference](#api-reference)
5. [Inter-Core Communication](#inter-core-communication)
6. [Topology Models](#topology-models)
7. [Configuration](#configuration)
8. [SPMD Execution Model](#spmd-execution-model)
9. [Verification](#verification)
10. [Limitations & Future Work](#limitations--future-work)

---

## Overview

The TinyTU cmodel previously operated as a single global instance (`g_tu`), restricting it to one TU core. Production accelerators (TPUv2+, NVIDIA TensorCore clusters) use multi-core SPMD topologies where each core has independent SRAM, DMA, and compute engines connected via an interconnect.

This implementation introduces:

- **`tu_core_t`** — A self-contained TU core that owns all hardware state (SRAM, DMA, command queue, dataflow plugin, performance counters). Multiple cores can coexist without interference.
- **`tu_cluster_t`** — A cluster manager that orchestrates N cores with configurable interconnect topology (ring, mesh) and provides inter-core communication primitives.

### What's New

| Component | Before (TinyTU) | After (This PR) |
|-----------|----------------|-----------------|
| Core instances | Single global `g_tu` | `tu_core_t` with explicit lifecycle |
| Multi-core | Not supported | `tu_cluster_t` with up to 256 cores |
| Topology | N/A | Ring, mesh, none |
| ICC | N/A | Send, broadcast, all-reduce sum, barrier |
| SPMD | N/A | `tu_cluster_spmd_execute` |
| State isolation | N/A | Each core owns independent SRAM banks |

---

## Why This Gap Matters

**A5 was selected as the highest-priority remaining P1 gap** because:

1. **Production accelerators are multi-core.** TPUv2/v3/v4 use SPMD across multiple cores. NVIDIA H100 uses thread block clusters with distributed shared memory. Any cmodel targeting production-grade fidelity must support multi-core.

2. **It's a foundational property.** Multi-core support touches every subsystem — SRAM banking, DMA, command queues, statistics — and must be designed before specialized features like sparsity or power modeling.

3. **It enables design space exploration.** With multi-core support, designers can explore topology tradeoffs (ring vs. mesh), core counts, and inter-core bandwidth requirements — the primary knobs in accelerator architecture.

4. **It's the last remaining P1 architectural gap.** All other P1 gaps (dataflow flexibility, precision types, operations) are complete.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      tu_cluster_t                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Core 0  │  │ Core 1  │  │ Core 2  │  │ Core 3  │  ...   │
│  │tu_core_t│  │tu_core_t│  │tu_core_t│  │tu_core_t│        │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │        │
│  │ │SRAM │ │  │ │SRAM │ │  │ │SRAM │ │  │ │SRAM │ │        │
│  │ │ DMA │ │  │ │ DMA │ │  │ │ DMA │ │  │ │ DMA │ │        │
│  │ │ CmdQ│ │  │ │ CmdQ│ │  │ │ CmdQ│ │  │ │ CmdQ│ │        │
│  │ │ DF  │ │  │ │ DF  │ │  │ │ DF  │ │  │ │ DF  │ │        │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │               │
│  ┌────┴────────────┴────────────┴────────────┴────┐         │
│  │              Interconnect (Ring / Mesh)         │         │
│  │   - Send(src, dst, offset, size)               │         │
│  │   - Broadcast(src, offset, size)               │         │
│  │   - AllReduce(sum)                              │         │
│  │   - Barrier                                     │         │
│  └────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

### Design Decisions

1. **State-isolated cores.** Each `tu_core_t` owns independent `tu_state_t` with separate SRAM banks. Cores never share mutable state — all communication is explicit via ICC messages.

2. **Swap-based compatibility.** The core API temporarily swaps `g_tu` with the core's state for each operation, maintaining full backward compatibility with the existing monolithic code.

3. **Configurable topology.** The interconnect is not hard-coded — ring and mesh are both supported, with the API designed for future extension (crossbar, tree, custom).

4. **Functional ICC model.** Inter-core messages are modeled with hop-distance-based latency, not cycle-accurate NoC simulation. This is sufficient for architectural exploration and matches the cmodel's functional modeling level.

---

## API Reference

### tu_core_t — Single Core

```c
#include "tu_cmodel/tu_core.h"

// Lifecycle
tu_core_t *tu_core_create(const tu_runtime_config_t *cfg);
tu_core_t *tu_core_create_with_id(uint32_t core_id, const tu_runtime_config_t *cfg);
void tu_core_init(tu_core_t *core);
void tu_core_destroy(tu_core_t *core);

// Operations
int tu_core_execute_asm_text(tu_core_t *core, const char *program,
                              const tu_host_buffer_t *buffers, int n_buffers);
void tu_core_sync(tu_core_t *core);

// DMA Convenience
void tu_core_dma_load_w(tu_core_t *core, const void *host_ptr,
                         uint32_t tu_offset, uint32_t size_bytes);
void tu_core_dma_load_a(tu_core_t *core, const void *host_ptr,
                         uint32_t tu_offset, uint32_t size_bytes);
void tu_core_dma_store_o(tu_core_t *core, void *host_ptr,
                          uint32_t tu_offset, uint32_t size_bytes);

// MMA Convenience
void tu_core_mma(tu_core_t *core, uint16_t M, uint16_t N, uint16_t K,
                 uint32_t w_offset, uint32_t a_offset, uint32_t o_offset,
                 bool has_bias);

// Subsystem Access
tu_command_queue_t *tu_core_get_cmdq(tu_core_t *core);
tu_dma_engine_t *tu_core_get_dma(tu_core_t *core);
tu_sram_region_t *tu_core_get_sram_w(tu_core_t *core);
tu_sram_region_t *tu_core_get_sram_a(tu_core_t *core);
tu_sram_region_t *tu_core_get_sram_o(tu_core_t *core);

// Stats
void tu_core_print_stats(const tu_core_t *core);

// Default singleton (backward compat)
tu_core_t *tu_core_default(void);
```

### tu_cluster_t — Multi-Core Manager

```c
#include "tu_cmodel/tu_cluster.h"

// Lifecycle
tu_cluster_t *tu_cluster_create(uint32_t num_cores,
                                 tu_topology_t topology,
                                 uint32_t mesh_rows,
                                 const tu_runtime_config_t *base_config);
void tu_cluster_destroy(tu_cluster_t *cluster);
tu_core_t *tu_cluster_get_core(tu_cluster_t *cluster, uint32_t core_id);

// Inter-Core Communication
int tu_cluster_send(tu_cluster_t *cluster, const tu_icc_message_t *msg);
int tu_cluster_broadcast(tu_cluster_t *cluster, uint32_t src_core_id,
                          uint32_t src_offset, uint32_t dst_offset,
                          uint32_t size_bytes);
int tu_cluster_allreduce_sum_f32(tu_cluster_t *cluster,
                                  uint32_t src_offset, uint32_t dst_offset,
                                  uint32_t num_elements);

// Synchronization
int tu_cluster_barrier(tu_cluster_t *cluster);

// SPMD
int tu_cluster_spmd_execute(tu_cluster_t *cluster,
                             const char **programs,
                             const tu_host_buffer_t **buffers,
                             const int *n_buffers_per);

// Topology
uint32_t tu_cluster_hop_distance(const tu_cluster_t *cluster,
                                  uint32_t src, uint32_t dst);
void tu_cluster_neighbors(const tu_cluster_t *cluster, uint32_t core_id,
                           uint32_t *neighbors, uint32_t *num_neighbors);

// Stats
void tu_cluster_print_stats(const tu_cluster_t *cluster);
```

---

## Inter-Core Communication

### Message Model

ICC messages transfer data between cores' O-SRAM regions. Each message is:

```c
typedef struct {
    uint32_t    src_core_id;        // Source core
    uint32_t    dst_core_id;        // Destination core
    uint32_t    src_offset;         // Byte offset in source O-SRAM
    uint32_t    dst_offset;         // Byte offset in destination O-SRAM
    uint32_t    size_bytes;         // Transfer size
    uint32_t    tag;                // User-defined message tag
    bool        blocking;           // true = wait for completion
    uint64_t    latency_cycles;     // Simulated interconnect latency
} tu_icc_message_t;
```

### Latency Model

ICC latency is computed as `hops × hop_latency` where:
- **Ring:** Distance = min(forward, backward) steps around the ring
- **Mesh:** Distance = Manhattan distance (|row_diff| + |col_diff|)
- **None:** No ICC possible (UINT32_MAX distance)

Default `hop_latency` = 5 cycles per hop (tunable in `tu_cluster_t`).

### All-Reduce

`tu_cluster_allreduce_sum_f32` performs element-wise FP32 sum across all cores:
1. Read from each core's O-SRAM at `src_offset`
2. Sum into accumulator on host
3. Write result to each core's O-SRAM at `dst_offset`

---

## Topology Models

### Ring (1D)

```
Core 0 ←→ Core 1 ←→ Core 2 ←→ Core 3
  ↑                            ↓
  └────────────────────────────┘
```

- Each core has 2 neighbors: `(core_id ± 1) % num_cores`
- Hop distance: `min((dst - src) % N, (src - dst) % N)`
- **Use case:** Simple data parallelism, pipeline parallelism

### Mesh (2D)

```
Core 0 ←→ Core 1 ←→ Core 2
  ↕         ↕         ↕
Core 3 ←→ Core 4 ←→ Core 5
```

- Each core has up to 4 neighbors (N/S/E/W), fewer at edges
- Hop distance: Manhattan distance
- **Use case:** Tensor parallelism, 2D weight/activation sharding

### None (Isolated)

- No inter-core communication
- Each core operates independently
- **Use case:** Batch parallelism, multi-tenant inference

---

## Configuration

The cluster is configured at creation time:

```c
tu_runtime_config_t cfg = tu_config_default();

// 4-core ring cluster
tu_cluster_t *ring = tu_cluster_create(4, TU_TOPOLOGY_RING, 0, &cfg);

// 2×3 mesh cluster
tu_cluster_t *mesh = tu_cluster_create(6, TU_TOPOLOGY_MESH, 2, &cfg);
```

The compile-time config in `tu_config.h` provides stubs:

```c
#define TU_MULTICORE_ENABLED    0   // Set to 1 in future
#define TU_NUM_CORES            1
#define TU_INTERCONNECT_MODE    0   // NONE / RING / MESH
```

Runtime hop latency and ICC buffer sizes are configurable per-cluster via the `tu_cluster_t` struct fields.

---

## SPMD Execution Model

The cluster supports Single Program Multiple Data (SPMD) execution:

```c
const char *programs[4] = {prog0, prog1, prog2, prog3};
const tu_host_buffer_t *buffers[4] = {bufs0, bufs1, bufs2, bufs3};
const int n_bufs[4] = {n0, n1, n2, n3};

tu_cluster_spmd_execute(cluster, programs, buffers, n_bufs);
```

Each core executes its own ASM program independently, but with the guarantee that all programs start simultaneously and barriers synchronize across cores.

---

## Verification

### Test Coverage (13 tests)

| Test | Description | Status |
|------|-------------|--------|
| TEST 1 | Core lifecycle (create/init/destroy) | ✅ |
| TEST 2 | Core DMA load/store isolation | ✅ |
| TEST 3 | Core MMA identity computation | ⚠️ |
| TEST 4 | Core state isolation (separate SRAM) | ✅ |
| TEST 5 | Cluster ring creation (4 cores) | ✅ |
| TEST 6 | Cluster mesh creation (2×3, 6 cores) | ✅ |
| TEST 7 | Hop distance calculation (ring + mesh) | ✅ |
| TEST 8 | Neighbor discovery (ring + mesh) | ✅ |
| TEST 9 | ICC send (point-to-point data transfer) | ✅ |
| TEST 10 | ICC broadcast (1 → all cores) | ✅ |
| TEST 11 | All-reduce FP32 sum (across 4 cores) | ✅ |
| TEST 12 | Barrier synchronization | ✅ |
| TEST 13 | SPMD execution (parallel MMA) | ✅ |

**12/13 pass.** The MMA test (3) has a known initialization-order issue with the dataflow plugin pointer propagation through the legacy swap mechanism — fix tracked for next heartbeat.

### Run tests

```bash
make test-multicore
```

---

## Limitations & Future Work

| Limitation | Priority | Notes |
|------------|----------|-------|
| MMA via core swap has dataflow init issue | P2 | Needs explicit dataflow_plugin init per core instead of relying on g_tu swap |
| No cycle-accurate NoC model | P2 | Current model is hop-based; needs congestion, bandwidth contention |
| No multi-core DMA coordination | P2 | DMA channels are per-core; no cross-core DMA descriptor sharing |
| No SPMD→SIMT model | P3 | SPMD is per-core ASM; SIMT (lockstep across cores) not modeled |
| ICC limited to O-SRAM only | P3 | Should support W-SRAM and A-SRAM transfers |
| No inter-core cache coherence | P3 | Configured via `TU_CACHE_COHERENCE 0`; model not implemented |
| Max 256 cores | — | Configurable via `num_cores` parameter |
| Crossbar/tree topologies not implemented | P3 | Architecture supports extension via `tu_topology_t` |

---

## References

- **Gap:** A5 in `docs/PRODUCTION_TU_REDESIGN.md` (Section 3.1)
- **Design:** Section 5.5 — TU Core (Top-Level Orchestrator)
- **Related:** TPUv2+ multi-core SPMD, NVIDIA H100 thread block clusters
- **Config:** `tu_config.h` — `TU_MULTICORE_ENABLED`, `TU_NUM_CORES`, `TU_INTERCONNECT_MODE`
