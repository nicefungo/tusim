# TU CModel — SRAM Bandwidth Modeling

> **Gap ID:** M2 (Per-bank bandwidth, arbitration delay, bank conflict stall cycles)
> **Priority:** P0 (Critical)
> **Date:** 2026-05-29
> **Heartbeat:** Cycle 6a

---

## What Changed

The TinyTU cmodel previously tracked bank conflicts as a simple counter (`conflicts`) but did not model the actual performance impact: bandwidth contention, arbitration delay, and stall cycles. DMA transfers accounted only for bus width (`bytes/32`) — there was no SRAM-side bandwidth limitation at all.

A per-bank bandwidth metering system has been added to `tu_sram`, with refill-based budget tracking, configurable arbitration, and stall cycle accumulation. DMA transfers now account for SRAM bandwidth stalls in their cycle estimates.

### Key Features

1. **Per-bank bandwidth budget** — Each SRAM bank has a configurable words-per-cycle limit (`TU_SRAM_WORDS_PER_CYCLE`, default 1 = single-ported). A refill window (`TU_SRAM_BW_WINDOW_CYCLES`, default 4) determines how often the budget resets.

2. **Three arbitration modes** — Selectable at compile time:
   - `TU_SRAM_ARB_NONE` (0): No arbitration, all accesses pass (unrealistic, for comparison)
   - `TU_SRAM_ARB_ROUND_ROBIN` (1): Round-robin between contending ports (default)
   - `TU_SRAM_ARB_PRIORITY` (2): Fixed priority (read > write)

3. **Stall cycle accounting** — When bandwidth is exhausted, accesses stall for `TU_SRAM_BW_STALL_PENALTY` cycles (default 2). Stalls are attributed per bank and accumulated globally.

4. **Bandwidth utilization statistics** — Per-bank and aggregate utilization as a percentage of theoretical maximum bandwidth. Hottest banks (by stall count) are reported.

5. **DMA integration** — `tu_dma_execute_desc()` now accounts for SRAM bandwidth consumption, adding stall cycles to transfer estimates.

6. **Cycle management** — `tu_sram_advance_cycle()` triggers periodic refill; called by DMA engine and can be called by the MMA path.

---

## Why This Matters

Without bandwidth modeling:
- **Performance predictions are aspirational** — no SRAM throughput ceiling means infinite bandwidth
- **Bank conflicts are invisible** — they're counted but don't affect timing
- **Architecture exploration is blind** — you can't compare single-ported vs dual-ported SRAM, or evaluate different banking strategies
- **Compiler decisions lack a cost model** — the compiler can't optimize data layout for bank access patterns

With bandwidth modeling:
- **Cycle estimates reflect real memory constraints** — DMA transfers slow down when they hit bank bandwidth limits
- **Bank conflicts have performance consequences** — hot banks accumulate stall cycles
- **Design space exploration** — test 1-port vs 2-port SRAM, different bank counts, different interleaving strategies
- **Compiler feedback** — the cost model can report which banks are contention hotspots

---

## How It Works

### Bandwidth Budget Model

Each SRAM bank maintains a `words_available` counter that is consumed per access and refilled every `TU_SRAM_BW_WINDOW_CYCLES` cycles:

```
Bank 0: [words_avail=1]  ──read──> [words_avail=0]  ──read──> STALL (2 cycles)
Bank 1: [words_avail=1]  ──write─> [words_avail=0]  4 cycles pass → [words_avail=1] (refill)
```

### Arbitration Flow

```
Access request
    │
    ├── BW modeling disabled? ───> Allow (return 0 stall)
    │
    ├── words_available > 0?
    │   ├── YES → Consume one word, allow access (return 0)
    │   └── NO  → Apply stall_penalty, increment stall counter (return penalty)
```

### Configuration

All bandwidth parameters are in `tu_config.h`:

```c
#define TU_SRAM_WORDS_PER_CYCLE 1    // Words per bank per cycle window
#define TU_SRAM_BW_WINDOW_CYCLES 4   // Refill window in cycles
#define TU_SRAM_BW_STALL_PENALTY 2   // Cycles to add per bandwidth stall
#define TU_SRAM_ARB_MODE TU_SRAM_ARB_ROUND_ROBIN  // Arbitration policy
```

Runtime override via `tu_sram_init_bw()`:
```c
tu_sram_region_t sram;
tu_sram_init_bw(&sram, 65536, "test", 
    2,                          // words_per_cycle (dual-ported)
    TU_SRAM_ARB_PRIORITY,       // priority arbitration
    1,                          // stall_penalty
    8);                         // refill_window
```

Disable at runtime:
```c
tu_sram_set_bw_modeling(&sram, false);
```

### DMA Integration

When DMA writes to SRAM (`TU_DMA_DIR_HOST_TO_TU`), each word written consumes bandwidth from the target bank. When DMA reads from SRAM (`TU_DMA_DIR_TU_TO_HOST`), each word read consumes bandwidth. The total stall cycles are added to the transfer's cycle estimate.

### Statistics

```c
// Aggregate utilization across all banks
float util = tu_sram_get_bandwidth_utilization(&sram);

// Per-bank stats
uint64_t reads, writes, r_stalls, w_stalls;
float bank_util;
tu_sram_get_bank_bw_stats(&sram, bank_idx, 
    &reads, &writes, &r_stalls, &w_stalls, &bank_util);
```

The `tu_sram_print_stats()` function now reports:
- Aggregate reads/writes/conflicts/stalls
- Bandwidth utilization percentage
- Top 3 hottest banks by stall count

---

## How It Changes the CModel's Behavior

**Before (no bandwidth model):**
```
DMA 64KB load → 64KB / 32B = 2048 cycles
```

**After (with bandwidth model):**
```
DMA 64KB load → 64KB / 32B = 2048 bus cycles
               + SRAM: 16384 words / 32 banks = 512 words/bank
               With 1 word/4 cycles/bank → 512 × 4 = 2048 bank cycles
               With round-robin arbitration, sequential access pattern:
               If all words hit bank 0 → 16384 × 2 = 32768 stall cycles
               If evenly distributed → minimal stalls
```

This reveals that sequential DMA to a single bank is catastrophically slow, which is realistic hardware behavior. The compiler must distribute data across banks.

---

## Limitations & Future Work

1. **MMA path uses raw pointers** — The systolic array accesses SRAM via `tu_sram_raw_ptr()` for performance. MMA bandwidth accounting is deferred to a future refinement (M2b). Current bandwidth model primarily affects DMA transfers.

2. **No NoC congestion** — Bank-level arbitration doesn't model network-on-chip delays between PEs and SRAM banks. The NoC model (gap M2 extension) would add hop count and congestion to the stall calculation.

3. **No read/write port conflict** — The current model uses a single `words_available` counter per bank. Real hardware may have separate read and write ports with independent bandwidth budgets.

4. **Simplified arbitration** — Round-robin is modeled as simple bandwidth exhaustion; there's no actual round-robin state machine tracking which requester goes next. This is sufficient for performance modeling but insufficient for functional verification.

---

## Files Modified

| File | Change |
|------|--------|
| `tu_cmodel/tu_config.h` | Added `TU_SRAM_WORDS_PER_CYCLE`, `TU_SRAM_ARB_*`, `TU_SRAM_BW_WINDOW_CYCLES`, `TU_SRAM_BW_STALL_PENALTY` |
| `tu_cmodel/tu_sram.h` | Added `tu_sram_bw_bank_t`, bandwidth fields to `tu_sram_bank_t`, new API functions |
| `tu_cmodel/tu_sram.c` | Full rewrite: bandwidth metering, refill, arbitration, stall accounting, statistics |
| `tu_cmodel/dma_descriptor.c` | DMA executor now accounts for SRAM bandwidth stalls |
