# SRAM bandwidth refill phase: advance-boundary sensitivity

**Source baseline:** `c13440ee5102e2f114fd4829349a3028551207a4`. This is an executable exploration, not a new runtime mode or a physical SRAM schedule.

## Question and observation

Can splitting an idle time advance change bank stalls when the actual memory operations occur at exactly the same times? Yes. `tu_sram_advance_cycle` calls `tu_sram_refill_bw`, which replaces each expired bank's budget and sets `last_refill_cycle` to the **current invocation time**, not the last nominal grid boundary. The elementwise and descriptor paths also call `tu_sram_advance_cycle`; this probe does not establish their end-to-end consequences. Scalar/bulk calls consume budget but copy bytes immediately even when returning a stall penalty; a counted stall is not replayed service.

The fail-closed probe `tests/explore_sram_refill_phase.c` uses a 32 B region, one 4 B bank, one word per window, and a 2-cycle penalty. It performs three reads or three writes at the same cycles in both cases. The only change is whether an otherwise idle advance is split at the first nominal boundary. It checks byte results, cycle, per-bank served/stalled events, total stall, and access returns.

| Window | Read/write access cycles | Advance calls | Returned penalties | Served/stalled | Last refill |
|---:|---|---|---|---|---:|
| 4 | 0, 5, 8 | 5, 3 | 0, 0, 2 | 2 / 1 | 5 |
| 4 | 0, 5, 8 | 4, 1, 3 | 0, 0, 0 | 3 / 0 | 8 |
| 8 | 0, 9, 16 | 9, 7 | 0, 0, 2 | 2 / 1 | 9 |
| 8 | 0, 9, 16 | 8, 1, 7 | 0, 0, 0 | 3 / 0 | 16 |

All four rows occur for both directions: **8/8 controls passed**. The complete output is `evidence/2026-11-02-sram-refill-phase/matrix.csv`. With current semantics, calls at cycle 5 or 9 reset the phase, while an additional idle advance at 4 or 8 retains the nominal phase long enough for the last refill. Identical memory access timestamps are therefore not sufficient to reproduce the result unless advance-call boundaries are also fixed. The two paths have identical functional bytes; their differing counter and penalty outcomes are model observations, not measured device latency or throughput.

## Plausible contracts

* **Restart-on-observation (existing compatibility):** on a first advance after expiry, refill once and set the next window from that observation. This is simple for lazy service where the refill process is intentionally tied to a software-visible event. It preserves historical local penalties and may discard phase; extra observation calls can alter later results. No claim is made that calling an idle advance represents a real hardware transaction.
* **Fixed nominal grid (proposed alternative):** anchor phase at initialization/reset and advance by the whole number of elapsed windows; cap stored budget at one grant, not an accumulating token bucket. For a continuously clocked periodic grant, the times 0/5/8 in the window-4 probe would have the same local outcomes for both advance sequences: the grid is independent of idle advance partitioning. This is a specified hypothesis, **not an executed cmodel result**. It needs phase/cycle arithmetic and rollover handling; model granularity still does not provide arbitration or replay.
* **Accumulating token bucket or demand-relative refills:** physically plausible for different controllers, but not equivalent to either above. Bounded credit capacity, consumption, sleep/gating, and restart triggers need explicit contracts and discriminating evidence before adding modes; do not treat unlimited accumulated credit as the fixed-grid proposal.

**Trade-offs:** in a local penalty model, a stable grid can change reported stalls depending on access phase; there is no general latency/throughput winner. A restart policy may be suitable for event-activated control and keeps legacy behavior, while a fixed grid may better represent continuously timed periodic grants. Fixed-grid state/arithmetic and reset/clock-gating checks have expected control, area, switching-energy and verification costs, all **unquantified**; actual throughput, latency and SRAM/DRAM traffic are likewise unquantified. Copies and arithmetic/numerical accuracy are unchanged by this probe. Compiler/runtime sequencing could change the observed metric under the current interface, so reproducible comparisons must record advance calls. Neither mode proves physical request scheduling, traffic, energy, port provisioning, or silicon feasibility.

## Implementation readiness and gates

**READY for a separate implementation work unit**, preserving restart-on-observation as zero/default and exposing fixed-grid as an explicit validated runtime choice. Use an initialization-anchored grid with no credit accumulation; define clock advance overflow and disabled-meter/reset/re-enable semantics explicitly. Wire the actual YAML/JSON/generated, canonical, runtime and region/core paths; fail unsupported values without state mutation and preserve zero-initialized callers. Audit public struct ABI and counter/reset/statistics semantics, including the current utilization denominator/clamp, without silently redefining existing metrics.

Gate both directions and modes at window 1 and larger, irregular and equivalent partitionings, exact boundary and just-before/after it, many idle windows, repeated no-op advance, exhausted/unexhausted budget, bank independence, zero cycles, core state, config parse/default/invalid rejection and legacy calls. Demonstrate identical read/write data and existing default matrix/counters. Test rollover safely and document any finite cycle horizon rather than assuming unsigned wrap is monotonic. Separately trace elementwise and DMA call sites; do not claim their schedules or queued timing have become physically accurate. Include ASan/UBSan and focused, quick and clean build gates. Arbitration, replay, physical ports, calibrated area/power and clock-domain transitions are outside this item.

## Reproduce

A clean archive of the source baseline built under an isolated `/tmp` because the repository clean target removes `/tmp/test_asm`; this preserves unrelated host files.

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make clean && make && make test-quick
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel \
  tests/explore_sram_refill_phase.c ./libtucmodel.a -lm -o "$TMPDIR/explore_sram_refill_phase"
"$TMPDIR/explore_sram_refill_phase"
make test-sram-issue-sweep test-sram-word-meter test-dma
```

Baseline clean build and quick suite passed. The eight-row probe and focused targets passed, including DMA 15/15. Those tests do not validate the proposed mode or unexercised access paths.
