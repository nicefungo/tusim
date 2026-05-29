# TU CModel — Multi-Level Memory Hierarchy

> **Gap A3:** Flat 3-buffer SRAM → Production-grade multi-level memory hierarchy  
> **Status:** Implemented  
> **Files:** `tu_cmodel/memory/memory_hierarchy.{h,c}`, `tu_cmodel/tu_config.h`  
> **Tests:** `tests/test_memory_hierarchy.c` (10 tests)

---

## Why This Feature

The original TinyTU had three flat, unbanked byte buffers (W: 128 KB, A: 64 KB, O: 64 KB) — no hierarchy, no level distinction, no notion of access cost varying by memory tier. Every production accelerator (TPU, Gemmini, Eyeriss, NVIDIA TensorCore) uses a **multi-level on-chip memory hierarchy**:

| Level | Production Analog | Purpose |
|-------|------------------|---------|
| **RegFile (L0)** | Per-PE registers | 1-cycle access, no banking overhead |
| **LocalSPAD (L1)** | Per-core scratchpad | Banked SRAM, BW-metered, double-buffered |
| **GlobalBuf (L2)** | Shared L2 buffer | Cross-core sharing, wider banks, higher latency |
| **DRAM (L3)** | HBM/DDR off-chip | High capacity, high latency, BW contention |

Without this hierarchy, the cmodel can't model the bandwidth/latency tradeoffs that drive real hardware design decisions — tiling strategies, data placement, and DMA scheduling all depend on understanding which data lives at which level.

This was chosen as a **P0 foundational gap** because every higher-level feature (convolution, attention, multi-core) needs a memory hierarchy to model correctly.

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────┐
│  tu_memory_hierarchy_t                           │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  Level 0: RegFile                         │   │
│  │  • 256 B/PE × (rows×cols) PEs            │   │
│  │  • Zero-latency access (functional mode) │   │
│  │  • Read/write counted, not stored        │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  Level 1: LocalSPAD (existing SRAM)       │   │
│  │  • tu_sram_region_t per buffer (W/A/O)   │   │
│  │  • Banked, BW-metered, stall-accounted   │   │
│  │  • Configurable banks/width/ports         │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  Level 2: GlobalBuf (NEW)                 │   │
│  │  • tu_global_buffer_t                     │   │
│  │  • 1 MB shared, 16 banks × 8 B words     │   │
│  │  • Same banking/BW model as LocalSPAD    │   │
│  │  • Hit/miss tracking                      │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  Level 3: DRAM (delegates)                │   │
│  │  • tu_dram_model_t (HBM2/DDR5/ideal)     │   │
│  │  • Channel-parallel access modeling      │   │
│  │  • Row-buffer hit/miss (optional)         │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  Per-level counters: reads, writes, bytes,       │
│  stall cycles                                    │
└─────────────────────────────────────────────────┘
```

### Level-Aware Access

Every access goes through `tu_mem_hierarchy_read()` or `tu_mem_hierarchy_write()`, which dispatches to the correct backend based on the memory level:

```c
tu_memory_hierarchy_t h;
tu_mem_hierarchy_init(&h);

// Access local scratchpad
tu_sram_region_t spad;
tu_sram_init(&spad, 4096, "my_spad");
tu_mem_hierarchy_write(&h, TU_MEM_LOCAL_SPAD, &spad, 0, data, 64, &stall);

// Access global buffer (no region needed — uses internal GBUF)
tu_mem_hierarchy_read(&h, TU_MEM_GLOBAL_BUF, NULL, 0, buf, 64, &stall);

// Access DRAM
tu_mem_hierarchy_write(&h, TU_MEM_DRAM, NULL, 0, data, 128, &stall);

// Record RegFile activity (zero-latency)
tu_mem_hierarchy_read(&h, TU_MEM_REGFILE, NULL, 0, buf, 16, NULL);
```

### Bandwidth and Stall Modeling

Each level models bandwidth independently:

| Level | BW Model | Stall Source |
|-------|----------|-------------|
| RegFile | Unlimited (1-cycle) | None (port contention in cycle-accurate mode) |
| LocalSPAD | Per-bank refill budget | Bank conflicts, BW exhaustion |
| GlobalBuf | Per-bank refill budget (wider banks) | Bank conflicts, BW exhaustion |
| DRAM | Channel-parallel bandwidth pool | Contention, row-buffer misses |

Stall cycles accumulate per-level in `level_stall_cycles[level]`.

### On-Chip Total

`tu_mem_hierarchy_get_onchip_total()` returns the sum of RegFile + LocalSPAD + GlobalBuf capacities (excludes DRAM). This is used by the compiler for scratchpad allocation decisions.

---

## Configuration

All parameters are compile-time configurable via `tu_config.h`:

```c
/* Register File (Level 0) */
#define TU_MEM_REGFILE_PER_PE       256      // Bytes per PE

/* Global Buffer (Level 2) */
#define TU_MEM_GBUF_SIZE            (1 * 1024 * 1024)  // 1 MB
#define TU_MEM_GBUF_BANKS           16
#define TU_MEM_GBUF_BANK_WIDTH      8        // 64-bit words
```

Runtime per-level configuration is also available via `tu_mem_level_config_t` structs and `tu_mem_hierarchy_set_level_config()`.

---

## How It Changes CModel Behavior

1. **Memory is no longer flat.** Access cost varies by level. A read from RegFile costs 0 stall cycles; a read from DRAM can cost 50+ cycles.

2. **Global Buffer adds an intermediate tier.** Data that doesn't fit in LocalSPAD but is accessed frequently should be placed in GlobalBuf rather than DRAM. The compiler can use this for weight caching, attention KV-cache, or intermediate activations.

3. **Statistics are level-aware.** `tu_mem_hierarchy_print_stats()` shows per-level read/write counts, byte totals, and stall breakdowns — enabling bottleneck analysis.

4. **Backward compatible.** Existing code that uses `tu_sram_region_t` directly (LocalSPAD level) continues to work. The hierarchy wraps (doesn't replace) the SRAM substrate.

---

## Verification

- **10 unit tests** covering init/destroy, level name lookup, LocalSPAD access, GlobalBuf access, RegFile tracking, DRAM delegation, on-chip total calculation, reset behavior, cycle advancement, and statistics printing.
- All 119 existing tests continue to pass (no regressions).
- GlobalBuf hit/miss counters validate that accesses actually reach the intended level.

---

## What's Next

- **A7 (Double Buffering):** Ping-pong buffer management between LocalSPAD halves for DMA/compute overlap. The `double_buffered` field in `tu_mem_level_config_t` is reserved for this.
- **M3 (Address Generation):** Hardware address generation for strided/block/im2col access patterns — builds on the hierarchy by providing level-aware address translation.
- **A5 (Multi-Core):** Each core gets its own LocalSPAD; GlobalBuf is shared. The hierarchy already supports this model.
