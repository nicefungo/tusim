# SRAM utilization: observation windows and initial grants

**Source baseline:** `0a7df608cfdfa57588d00b9710010fd5f6dd9b27` (2026-11-04). This is an observational exploration; it adds no execution mode and does not redefine either existing API.

## Question and scope

Does the reported fraction distinguish the initial grant from subsequent refill opportunities, and do global and per-bank statistics agree at the start of an observation? The producer in `tu_sram.c` initializes each bank with a grant at cycle 0. The global statistic returns zero at cycle 0, then divides served events by `bank_count * grant * max(1, floor(current_cycle/window))`, clamping to one. The bank statistic uses the same period count per bank, but does not apply the cycle-zero return. The print paths in `tu_sram_print_stats` and `tu_mem_hierarchy_print_stats` consume the global statistic. This exploration does not assert that a point observation is a nonzero-duration physical interval or that every nominal refill is physically available.

The retained static-linked probe performs one 4-byte read or write at cycle 0 and no further accesses. It samples seven points at 0, 1, W−1, W, 2W−1, 2W, and 2W+1; crosses W=4/8, one/two banks and both directions; and verifies bytes, served and stalled counters, and the actual getter results. It also prints an independent **hypothetical** initial-inclusive opportunity fraction, `served / (bank_count * grant * (1 + floor(t/W)))`, and the corresponding bank fraction. For this carefully chosen trace with advances at every boundary, this represents initial plus nominal grants, but the expression is not a validated general-purpose replacement: restart-on-observation can shift phase, disabled meters and context restore break its assumptions, and partial boundary inclusion is a policy choice.

## Observations

| W | Banks | Access | Cycle | Existing global | Existing bank 0 | Hypothetical inclusive global | Hypothetical inclusive bank 0 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 1 | read | 0 | 0 | 1 | 1 | 1 |
| 4 | 1 | read | 1 | 1 | 1 | 1 | 1 |
| 4 | 1 | read | 4 | 1 | 1 | 0.5 | 0.5 |
| 4 | 1 | read | 8 | 0.5 | 0.5 | 0.333333 | 0.333333 |
| 4 | 2 | read | 4 | 0.5 | 1 | 0.25 | 0.5 |

All **56/56 pinned observations** pass; 40 global fractions differ from this particular hypothetical including the cycle-zero convention. At cycle 0, the global getter reports 0 while bank 0 reports 1 in all eight configurations. These are facts about the current getters, not assertions that either convention defines the true physical utilization. The full matrix and result are under `evidence/2026-11-04-sram-utilization/`. There is no modeled change to copied bytes or numeric operations. Per-bank stalls and served counters are not physical completed-service evidence: data copies occur even for penalized calls, and denied calls are not replayed by this scalar meter.

## Alternative measurement contracts and trade-offs

* **Existing compatibility fraction:** minimal state and preserves existing reports, but excludes an explicit initial opportunity from its nominal denominator, saturates above 100%, and has different cycle-zero behavior between the two getters. It remains unchanged.
* **Nominal-grid opportunity fraction:** uses a declared epoch, start/end inclusion and number of grants. This can support reproducible comparisons for an actually fixed-phase, continuously enabled meter. Idle windows reduce the fraction; partial windows and clock-gated time need explicit choices. A free initial grant may count as an opportunity, but a zero-duration sample may instead be explicitly unavailable.
* **Actual observed-grant accounting:** count the initial and each actual lazy refill and compare served events against those grants. This reflects this model's current observation-driven policy rather than a periodic physical clock; re-observation can change the number of refill events. Counters, overflow, reset, disable/re-enable and context ownership need specification.
* **Interval/event metrics:** expose served and attempted events with an explicit elapsed interval and validity; an unclamped rate can reveal incompatible clocks or denominator assumptions. Snapshot/diff requires coherent, monotonic counters and a specified boundary when time is zero. These quantities do not by themselves prove a physical utilization or issue rate.

No option is universally preferable. Nominal and actual-event denominators can diverge, and adding a new separately named metric is safer than silently changing the existing getters. Any increased denominator or reduced fraction is a reporting interpretation, not a physical throughput improvement. Throughput, latency, SRAM/DRAM traffic, area/resources, power/energy and silicon cost are **unquantified**. Extra counters, time-window logic and snapshot state could increase hardware bookkeeping if implemented in hardware, or host model/verification cost if software-only. The probe leaves numerical accuracy and useful bytes unchanged; compiler/runtime consequences are metric interpretation and reproducibility, not a measured execution change.

## Readiness and verification gates

**BLOCKED**, rather than asserting one denominator is universally correct. The existing `sram-refill-phase.md` shows the same access times can have different event histories depending on idle advance partition; `sram-context-refill-state.md` shows restoration can mix another context's bank meter with rolled-back time. Fix or explicitly constrain the clock/meter ownership first. Define initialization and sample boundary inclusion, valid/undefined zero-duration results, enabled/disabled periods, counter resets, overflow/rollover, phase, access-versus-service semantics and context retention. Preserve old getters; gate explicitly named new metrics against an independent trace oracle with exact boundaries, one/multiple banks and grants, empty intervals, partial windows, saturated and idle cases, read/write mixtures, refill partitions, disabled/re-enabled metering, contexts, no mutation, and old report snapshots. Include config only if the underlying architecture behavior changes, not merely to select a reporting calculation. Do not infer physical queueing, replay, or utilization calibration from the resulting scalar.

## Reproduction

From a clean archive of the pinned source, build the archive and use a statically linked probe (the probe itself is introduced by the follow-up commit):

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make clean && make && make test-quick
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel \
  tests/explore_sram_utilization_window.c ./libtucmodel.a -lm -o "$TMPDIR/explore-sram-util"
"$TMPDIR/explore-sram-util"
```

The baseline `make clean` removes fixed `/tmp` paths, including an unrelated existing `/tmp/test_asm`; for this run those paths were redirected to the disposable source archive before invoking the clean/build/quick sequence. The source and quick suite otherwise match the pinned archive. `make clean` in an ordinary checkout is **not** safe while such files are user-owned. The prior eight-row phase and four-row context probes were also rebuilt and rerun; their reported observations matched the previous audit. No production fix or hardware performance benchmark is claimed.
