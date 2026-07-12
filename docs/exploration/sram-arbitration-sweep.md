# SRAM Arbitration Mode Sweep

**Date:** 2026-07-12
**Type:** Analytical sweep (standalone C, no cmodel dependency)
**Parameter swept:** SRAM bank arbitration policy (NONE / Round-Robin / Priority)

## Motivation

The cmodel SRAM has 32 banks with configurable arbitration (`TU_SRAM_ARB_MODE`): NONE (0), Round-Robin (1), Priority (2). This parameter controls how simultaneous accesses to the same bank are resolved. Previous sweeps have used the default RR mode without exploring whether the choice matters for throughput.

## Config Matrix

| Parameter | Value |
|-----------|-------|
| Banks | 32 |
| Bank width | 4 bytes (1 FP32 word) |
| Words per cycle per bank | 1 |
| Stall penalty | 2 cycles |
| BW refill window | 4 cycles |
| Arbitration modes | NONE (0), RR (1), PRIORITY (2) |

## Results

### Balanced workload (50% reads, 50% writes)

| Ops | Active Banks | NONE stalls | RR stalls | PRIORITY stalls | RR vs NONE |
|-----|-------------|-------------|-----------|-----------------|------------|
| 1000 | 1 | 1,498 | 1,498 | 2,214 | +0.0% |
| 1000 | 2 | 1,496 | 1,496 | 2,264 | +0.0% |
| 1000 | 4 | 1,496 | 1,496 | 2,240 | +0.0% |
| 1000 | 8 | 1,488 | 1,488 | 2,224 | +0.0% |
| 1000 | 16 | 1,488 | 1,488 | 2,214 | +0.0% |
| 1000 | 32 | 1,472 | 1,472 | 2,226 | +0.0% |
| 5000 | 1 | 7,498 | 7,498 | 11,276 | +0.0% |
| 5000 | 8 | 7,488 | 7,488 | 11,256 | +0.0% |
| 5000 | 32 | 7,440 | 7,440 | 11,096 | +0.0% |

### Read-heavy workload (80% reads, 20% writes)

| Ops | Active Banks | NONE stalls | RR stalls | PRIORITY stalls | PRIO vs NONE |
|-----|-------------|-------------|-----------|-----------------|--------------|
| 1000 | 4 | 1,496 | 1,496 | 1,834 | +22.6% |
| 1000 | 16 | 1,488 | 1,488 | 1,742 | +17.1% |
| 5000 | 32 | 7,440 | 7,440 | 8,898 | +19.6% |

## Key Findings

### 1. RR and NONE are identical under sequential access patterns

With 32 banks and sequential access (one bank per cycle, round-robining across active banks), there is never simultaneous contention on the same bank. NONE and RR produce identical stall counts — both are limited purely by the bandwidth refill window (1 word/4 cycles = 75% idle).

**RR only matters when multiple ports hit the same bank in the same cycle** — e.g., dual-ported SRAM with concurrent read+write to the same bank, or DMA + compute simultaneously accessing shared buffers.

### 2. PRIORITY mode penalizes writes

PRIORITY adds a 2-cycle extra penalty to write operations that encounter bank contention. This manifests as:
- +48% stalls for 50/50 r/w workloads
- +20% stalls for 80/20 r/w workloads

PRIORITY is useful when reads are on the critical path (e.g., weight-stationary dataflow where W-buffer reads feed the systolic array continuously while writes are background DMA fills). Under those access patterns, favoring reads reduces compute stalls.

### 3. Arbitration mode is a second-order effect for current cmodel

The cmodel's 32-bank SRAM with `TU_SRAM_WORDS_PER_CYCLE=1` means each bank can serve at most one access per cycle. With sequential, strided access patterns from DMA and systolic array, bank conflicts are rare. The dominant factor in SRAM throughput is the 4-cycle bandwidth refill window, not the arbitration policy.

**Recommendation:** RR is the safe default. PRIORITY should be considered for dual-ported SRAM designs where simultaneous read+write to the same bank is expected (e.g., ping-pong buffer swaps, DMA/compute overlap on the same SRAM region).

## Comparison with Existing Sweeps

This sweep complements the SRAM sizing sweeps (`sram-wa-buffer-sizing.md`, `sram-obuffer-tiling-threshold.md`) by confirming that the arbitration policy choice does not require separate exploration for the single-ported SRAM model. The bandwidth model (refill window + stall penalty) dominates SRAM cycle accounting.
