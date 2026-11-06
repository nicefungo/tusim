# SRAM bank refill phase: shared observation versus independent first use

**Baseline:** `a15294cb88e1ae1c1608c6df6f0716bafdfeb08f`. Pre-spec accounting exploration, not physical SRAM timing. Existing code calls `tu_sram_refill_bw()` for *every* bank after each `tu_sram_advance_cycle()`. With identical initial timestamps these banks are refreshed together even when their access histories differ. The candidate below is a separately computed hypothesis, **not an implemented mode**.

## Experiment

A two-bank region uses four-byte words, one grant per four-cycle window and a two-cycle penalty. After initialization, one direction at a time performs `(cycle, bank)` = `(0,0), (2,1), (4,0), (4,1), (6,0), (6,1)`. At each step the real API advances to the target cycle then performs one read or write. The hypothetical alternative starts each bank's four-cycle window at its first access and refills that bank on a later access after four cycles; the initial grant is not duplicated. It is not an assertion that real hardware follows this policy. The independent reference is in the probe, and expected arrays gate both real and hypothetical results. Complete per-direction evidence is `evidence/2026-11-06-sram-bank-phase/matrix.csv`.

| Cycle, bank | Current local penalty | Hypothetical first-use penalty |
|---|---:|---:|
| 0, 0 | 0 | 0 |
| 2, 1 | 0 | 0 |
| 4, 0 | 0 | 0 |
| 4, 1 | 0 | 2 |
| 6, 0 | 2 | 2 |
| 6, 1 | 2 | 0 |

Both directions yield these same six entries: 12 rows, zero mismatches, and the byte-copy controls pass. Current API advancement at cycle 4 updates both banks' refill timestamps to 4; the alternative anchors bank 1 at cycle 2. The alternatives have equal **sum** of reported local penalties in this trace but redistribute penalties between requests. This does not imply equal physical throughput or latency: scalar accesses still copy data while reporting penalties, without retry or time advancement. The first-use column is not live evidence of an implementation.

## Alternatives and costs

* **Shared observation:** current compatibility behavior; a shared clock/phase makes bookkeeping predictable and avoids separately initialized epochs, but an explicit global observation updates inactive banks. Advance partition sensitivity still applies.
* **Independent first-use phase:** could represent independently enabled bank grant windows and reduces coupling between inactive banks; per-bank initialization/phase state, clock gating or local control and additional verification may be needed. The above trace shifts which request incurs a penalty; it establishes neither a general win nor a hardware implementation.
* **Fixed global nominal grid:** plausible synchronized controller, distinct from current restart-on-observation behavior and from first-use phase. Its separate evidence and unresolved timing ownership are in `sram-refill-phase.md`.

All three can preserve byte values; this probe checks byte copies only, not numerical equivalence across engines. SRAM/DRAM physical traffic, throughput, latency, area, power/energy, compile/runtime scheduling impact and calibration are **unquantified**. Per-bank clocks or epoch state can add control/storage/routing and verification effort; software access ordering and runtime advance calls can influence observed counters. No numerical change or universal choice is recommended.

## Status and reproduction

**BLOCKED:** phase policy cannot safely be made runtime-selectable until region time versus per-context state, reset/enable transitions, counter epochs, rollover and advancement semantics are defined. `sram-context-refill-state.md` shows why merely copying timestamps is insufficient. Validate multi-bank idle histories, read/write/unaligned access, all W/A/O regions and context scopes, large/split advances, zero-cycle calls, toggles, overflow, invalid configuration and legacy-default behavior before promoting any mode. Physical clock gating, requests, arbitration and replay require further contracts.

```sh
make
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel \
  tests/explore_sram_bank_phase.c ./libtucmodel.a -lm -o "$TMPDIR/explore-sram-bank-phase"
"$TMPDIR/explore-sram-bank-phase"
```

The quick-suite baseline originally needed `LD_LIBRARY_PATH` because several recipes selected the shared object despite depending only on the static archive. The companion Makefile repair statically links the three affected quick-suite executables and uses a repository-local ignored ASM binary; it does not change cmodel APIs, configuration, or modeled hardware behavior. Other legacy targets are not certified by this repair.
