# Scheduler Policy Sweep: ASAP vs ALAP vs BALANCED

**Date:** 2026-07-14
**Type:** Analytical sweep (cmodel-linked)
**Config:** tu_sched_config_default with policy varied

## Summary

Compared the three scheduling policies (ASAP, ALAP, BALANCED) across five workload topologies to measure scheduling quality differences.

## Configuration Matrix

| Parameter | Values |
|-----------|--------|
| Policy | ASAP, ALAP, BALANCED |
| Workloads | All-Independent, Serial-Chain, Fan-Out, Fan-In, Pipeline-Tiles |
| DMA hoisting | enabled (default) |
| Barrier insertion | enabled (default) |
| Pipeline tiles | enabled (default) |

## Workload Descriptions

1. **All-Independent:** 4 NOPs — no SRAM dependencies, trivial DAG.
2. **Serial-Chain:** DMA_LOAD(W) → MMA → DMA_STORE(O) → HALT. Linear dependency chain, no parallelism possible.
3. **Fan-Out:** One DMA_LOAD(W) feeds 4 independent MMA ops (different O offsets). All 4 MMAs share the same W tile — parallelizable after DMA completes.
4. **Fan-In:** 4 DMA_LOAD ops (2×W, 2×A, distinct offsets) converge to 1 MMA.
5. **Pipeline-Tiles:** 4× (DMA_W + DMA_A → MMA) with distinct per-tile offsets. Classic double-buffering candidate.

## Results

| Topology          | Policy   | Cycles | Barriers | DMA Hoisted | Sched Len |
|-------------------|----------|--------|----------|-------------|-----------|
| All-Independent   | ASAP     | 16     | 0        | 0           | 4         |
| All-Independent   | ALAP     | 16     | 0        | 0           | 4         |
| All-Independent   | BALANCED | 16     | 0        | 0           | 4         |
| Serial-Chain      | ASAP     | 10     | 0        | 0           | 4         |
| Serial-Chain      | ALAP     | 10     | 0        | 0           | 4         |
| Serial-Chain      | BALANCED | 10     | 0        | 0           | 4         |
| Fan-Out           | ASAP     | 21     | 0        | 0           | 6         |
| Fan-Out           | ALAP     | 21     | 0        | 0           | 6         |
| Fan-Out           | BALANCED | 21     | 0        | 0           | 6         |
| Fan-In            | ASAP     | 12     | 0        | 0           | 6         |
| Fan-In            | ALAP     | 12     | 0        | 0           | 6         |
| Fan-In            | BALANCED | 12     | 0        | 0           | 6         |
| Pipeline-Tiles    | ASAP     | 28     | 0        | 0           | 13        |
| Pipeline-Tiles    | ALAP     | 28     | 0        | 0           | 13        |
| Pipeline-Tiles    | BALANCED | 28     | 0        | 0           | 13        |

## Key Finding

**All three scheduling policies produce identical `estimated_cycles`, with zero barriers and zero DMA hoisted across all workloads.** This has two root causes:

1. **Cycle estimation is DAG-bound, not schedule-bound.** `estimated_cycles` is derived from the critical path of the dependency DAG, which is identical regardless of instruction ordering. The policies differ in *which order* independent instructions appear in the output sequence, but the DAG structure (and thus its cycle estimate) is invariant.

2. **DMA hoisting requires explicit barriers in the input.** The hoisting pass (`tu_sched_hoist_dma`) looks for DMA instructions that are ordered *after* barriers even though they have no data dependency on preceding compute ops. Without explicit barriers in the input program, there's nothing to hoist past. In real compiled code, the compiler would insert barriers between tiles — but for a raw instruction sequence without barrier markers, every DMA already appears at its earliest position.

**Engineering implication:** To make scheduling policy choice visible, the scheduler needs a cycle model that accounts for schedule order (not just DAG critical path). DMA hoisting only activates when the input contains suboptimally-placed barriers — a compiler, not a scheduler, concern at this stage.

## Sweep Harness

`tests/test_scheduler_sweep.c` — standalone C, compiles via `make test-scheduler-sweep`.
