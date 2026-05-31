# TU CModel — Software Pipelining Controller

> **Gap ID:** E2 — Tile-level software pipelining
> **Priority:** P1 (High)
> **Date:** 2026-05-31
> **Heartbeat:** Cycle 1
> **Dependencies:** A7 (double buffering), DM1/DM2 (async DMA), E1 (command queue)

---

## What Changed

The TU CModel now has a tile-level software pipelining controller that orchestrates DMA/compute overlap — the key performance optimization in production systolic accelerators (TPU, Gemmini, Eyeriss).

Previously, tile execution was strictly sequential: DMA load → compute → DMA store, with no overlap between phases. The new `pipeline_controller` module coordinates double-buffered scratchpads, async DMA descriptors, and the command queue to overlap DMA transfers with computation.

### New Files

| File | Description |
|------|-------------|
| `tu_cmodel/compute/pipeline_controller.h` | Pipeline controller API and data structures |
| `tu_cmodel/compute/pipeline_controller.c` | Implementation (~470 lines) |
| `tests/test_pipeline.c` | 11 unit tests covering init, stages, overlap, backpressure |

### Modified Files

| File | Change |
|------|--------|
| `Makefile` | Added `pipeline_controller.o` to library, `double_buffer.o` link, `test-pipeline` target |
| `tu_cmodel/memory/double_buffer.h` | Existing — now used as dependency |
| `tu_cmodel/dma_descriptor.h` | Existing — now used as dependency |

---

## Why This Matters

Software pipelining is what turns a systolic array from a useful math unit into a production-grade accelerator:

- **Without pipelining:** Total time = Σ (DMA_load + compute + DMA_store) per tile. The PE array sits idle during DMA, and the DMA engine sits idle during compute.
- **With pipelining (depth=2):** While tile N computes, DMA loads tile N+1 into the shadow buffer. Total time ≈ max(DMA, compute, store) per tile.
- **With pipelining (depth=3):** DMA load for N+2, compute for N+1, and DMA store for N all happen concurrently.

This is the single largest performance lever in systolic accelerator design. Google TPUs, NVIDIA TensorCores, and Gemmini all use variants of this technique.

---

## How It Works

### Architecture

```
Pipeline Controller (tu_pipeline_controller_t)
│
├── Slots[0..depth-1]  ←  Pipeline tiles in various stages
│   │
│   ├── DMA_PRELOAD  →  DMA writes tile data into shadow buffer
│   ├── COMPUTE      →  PE array computes on active buffer
│   ├── DMA_STORE    →  DMA stores results to DRAM
│   └── DONE         →  Tile complete; slot recycled
│
├── Overlap Accounting
│   ├── total_load/compute/store_cycles
│   ├── overlapped_load/store_cycles
│   ├── stall_cycles
│   └── sequential_total (Σ without pipelining)
│
└── Integration Points
    ├── tu_dma_tick()        — Advance DMA engines
    ├── tu_sram_swap_buffers() — Atomic buffer swap (0-cycle)
    └── tu_dma_submit_desc() — Async DMA submission
```

### Pipeline Stages

```
           ┌──────────────┐
           │   DMA_PRELOAD │  ← DMA writes into shadow buffer
           └──────┬───────┘
                  │ DMA complete
           ┌──────▼───────┐
           │   COMPUTE     │  ← PE array reads active buffer
           └──────┬───────┘
                  │ Compute complete
           ┌──────▼───────┐
           │   DMA_STORE   │  ← DMA writes results to DRAM
           └──────┬───────┘
                  │ Store complete
           ┌──────▼───────┐
           │   DONE        │  ← Slot recycled for next tile
           └──────────────┘
```

### Overlap Model

With depth=2 (load overlap):
```
        Tile 0: [===== DMA LOAD =====][====== COMPUTE ======][= STORE =]
        Tile 1:                     [== DMA LOAD ==][====== COMPUTE ======]
                                          ↑ overlap ↑
```

With depth=3 (full overlap):
```
        Tile 0: [== DMA LOAD ==][====== COMPUTE ======][= STORE =]
        Tile 1:               [== DMA LOAD ==][====== COMPUTE ======][= STORE =]
        Tile 2:                             [== DMA LOAD ==][====== COMPUTE ======]
```

### Configuration

```c
tu_pipeline_config_t cfg = tu_pipeline_config_default();
// cfg.max_depth = 2           // Pipeline depth (1=sequential, 2=load overlap, 3=full)
// cfg.enable_load_overlap     // Overlap DMA load with compute (default: true)
// cfg.enable_store_overlap    // Overlap DMA store with compute (default: true)
// cfg.enable_triple_overlap   // Full 3-way overlap (default: false)
// cfg.model_stalls            // Account for pipeline stalls (default: true)

tu_pipeline_init(2, &cfg);
```

### Auto-Advance on Backpressure

When `tu_pipeline_submit_tile()` encounters a full pipeline, it automatically advances the pipeline (up to 1M cycles) to try to free a slot. This models hardware backpressure behavior where the producer stalls briefly until a tile completes. If the pipeline remains full after the timeout, the submission returns -1 indicating true backpressure.

### Statistics

```c
tu_pipeline_stats_t stats;
tu_pipeline_get_stats(&stats);
// stats.depth                  // Configured depth
// stats.total_tiles            // Tiles processed
// stats.total_compute_cycles   // Total compute across all tiles
// stats.overlapped_load_cycles // Load cycles hidden by compute
// stats.speedup                // sequential_total / pipelined_total
// stats.load_overlap_pct       // % of load cycles that overlapped
```

---

## How to Configure

### Via `tu_config.h`

```c
// Currently pipeline depth is set at init time:
tu_pipeline_init(2, NULL);  // Depth 2

// Future: these config knobs could be added to tu_config.h:
#define TU_PIPELINE_DEPTH           2
#define TU_PIPELINE_LOAD_OVERLAP    1
#define TU_PIPELINE_STORE_OVERLAP   1
```

### Runtime API

```c
// Initialize with depth 2 (DMA load overlaps with compute)
tu_pipeline_init(2, NULL);

// Submit tiles (the pipeline auto-advances as needed)
for (int t = 0; t < num_tiles; t++) {
    tu_dma_descriptor_t *load = tu_dma_desc_create_linear(...);
    tu_pipeline_submit_tile(load, store, compute_cycles, cmd_id, &buffer);
}

// Alternatively: tick the pipeline externally
while (!tu_pipeline_is_idle()) {
    g_tu_pipeline.current_cycle++;
    tu_pipeline_advance();
}

// Get stats
tu_pipeline_print_stats();
tu_pipeline_destroy();
```

---

## How It Changes CModel Behavior

| Before | After |
|--------|-------|
| Sequential tile execution (DMA→compute→store) | Overlapped DMA load with compute |
| Fixed-cost cycle estimate: Σ(D+C+S) | Dynamic: max(D, C, S) + non-overlapped tail |
| No backpressure modeling | Pipeline full → stall cycles tracked |
| No overlap statistics | Per-tile overlap accounting, speedup report |

### Cycle Estimate Impact

For a workload with 8 tiles × (50 DMA load + 500 compute + 50 DMA store):
- **Without pipelining:** 8 × 600 = 4,800 cycles
- **With depth=2:** ~8 × max(550, 50) ≈ 4,200 cycles (12.5% improvement)
- **With depth=3:** ~8 × 500 + 100 ≈ 4,100 cycles (14.6% improvement)

The speedup is most pronounced when DMA and compute times are balanced. For convolution workloads with large activations and small kernels, load time dominates and depth=2 provides substantial benefits.

---

## Integration with Other Features

| Feature | Relationship |
|---------|-------------|
| **Double buffering (A7)** | Required for pipelining — shadow buffer receives DMA loads |
| **Async DMA (DM1/DM2)** | Required — non-blocking DMA so compute can run concurrently |
| **Command queue (E1)** | Used to submit compute operations |
| **Memory hierarchy (A3)** | Buffer regions reference the memory system |
| **Performance counters (E4)** | Pipeline overlap stats feed into overall performance report |

---

## Design Decisions

1. **Auto-advance on backpressure:** Avoids burdening the caller with manual advance loops. Models transient backpressure in hardware.
2. **Single global instance:** Matches existing architecture (`g_tu_pipeline`). Multi-core support (A5) will require per-core instances.
3. **DMA engine owns descriptors:** Pipeline does not free submitted DMA descriptors. DMA engine manages descriptor lifetime.
4. **0-cycle buffer swap:** Models hardware flip-flop pointer exchange — atomic and free.
5. **Sequential baseline tracked:** Even for pipelined execution, the unoptimized total is computed for accurate speedup calculation.

---

## References

- **Gap E2:** `docs/PRODUCTION_TU_REDESIGN.md` §3.6
- **Double buffering:** `docs/TU_DOUBLE_BUFFER.md`
- **DMA descriptor engine:** `docs/dma-descriptor-engine.md`
- **Gemmini:** Tile-level software pipelining with double-buffered scratchpads
- **TPU:** Compiler-scheduled DMA/compute overlap via explicit sync instructions
