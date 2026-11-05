# SRAM meter disable/re-enable transitions

**Baseline:** `f20fb19` (2026-11-05). Observational exploration only; no new runtime policy or altered default. The switch examined is the existing `tu_sram_set_bw_modeling()` API, not a physical SRAM power-gating control.

## Question and executable contract

Does disabling and re-enabling the meter preserve, pause, or reset its grant state, and does the order of re-enable and cycle observation affect the next access? This matters when comparing configurations, because a transient meter setting should not silently be interpreted as physical idle time or as uninterrupted utilization.

At the pinned source, the setter only assigns `bw_modeling`. `tu_sram_advance_cycle` always increments `current_cycle`, but its refill call returns while disabled. Disabled reads/writes still copy bytes and increment the aggregate read/write word count, while per-bank served/stall counters and budget are untouched. Re-enabling does not itself refill. A later `tu_sram_advance_cycle(r, 0)` calls the refill routine, which compares the current timestamp with the retained last-refill timestamp and can refill despite advancing zero cycles. This is current code behavior, not a promised physical device behavior.

The statically linked probe tests reads and writes; one bank, 4-byte words, one grant, penalty 2, window 4 cycles. It first consumes the initial grant, then compares continuously enabled against disabled during a gap of 0, 3, 4 or 8 cycles. The disabled path does one extra byte-verified access during the gap, re-enables, accesses, invokes `advance(0)`, and accesses again. The enabled path has no extra gap access. It checks exact penalties, budgets, timestamps, counters and copied bytes; the extra disabled access is not a matched throughput comparison. The complete 16-row matrix is `evidence/2026-11-05-sram-meter-enable/matrix.csv`; the executable returns nonzero on mismatch.

| Gap | Mode | First access after gap | After `advance(0)` | Bank served / stalled | Aggregate word calls |
|---:|---|---:|---:|---:|---:|
| 0 or 3 | enabled | 2 | 2 | 1 / 2 | 3 |
| 0 or 3 | disabled then enabled | 2 | 2 | 1 / 2 | 4 |
| 4 or 8 | enabled | 0 | 2 | 2 / 1 | 3 |
| 4 or 8 | disabled then enabled | 2 | 0 | 2 / 1 | 4 |

All 16 cases pass identically for read and write. In the disabled path the first access after a four-cycle gap incurs a 2-cycle *reported local penalty*, even though the following zero-cycle observation refills and removes that penalty on the next access. Aggregate counts include the unmetered disabled operation; per-bank counts do not. Scalar access copies proceed despite the reported penalty: there is no retry or completed-service schedule. Therefore neither the table nor a sum of these penalties proves hardware latency or throughput.

## Alternatives and decision value

* **Retained budget; no refill until an explicit advance/refill:** existing compatibility behavior. Cheap state and preserves old reports, but interpretation depends on whether caller invokes an observation immediately after re-enable. Disabled time contributes to the subsequent elapsed-time check, and aggregate versus bank counters describe different populations.
* **Refill on re-enable using elapsed time:** might represent a continuously clocked underlying bank even when instrumentation was off; moves the observed grant to the first post-enable access. It needs a defined boundary, including interaction with same-cycle accesses and stale/rolled-back time. Physical throughput benefit is not established.
* **Freeze the meter clock during disable:** could represent pausing a virtual experiment or fully clock-gated bookkeeping; requires a separate enabled-time epoch and a definition of the actual SRAM grant state. It may improve reproducibility for paused experiments while failing to represent a continuously active bank.
* **Reset or explicitly prime on re-enable:** can provide a deterministic new observation epoch, but resets or changes grant/counter state and can give an artificial immediate grant. A real reset/wakeup may require latency, energy and a power-state contract.

These are plausible instrumentation/control contracts, not mutually interchangeable physical architectures. No mode is added or excluded as slower. Wider implications for throughput, physical latency, SRAM/DRAM traffic, area/resources, power/energy and numerical accuracy are **unquantified**; the probe establishes byte-copy integrity for its inputs, not general arithmetic equivalence. More timers, epoch state and transition logic could increase hardware state/control and verification cost if implemented in hardware; software-only observability affects host cost, reproducibility and API/runtime expectations. Physical clock gating or power state, wake-up energy/latency, port arbitration, retries, compiler placement and calibrated activity are not modeled.

## Readiness

**BLOCKED:** re-enable behavior cannot be promoted to a generally valid runtime architecture choice without specifying time and meter ownership across context switches (`sram-context-refill-state.md`), the refill phase (`sram-refill-phase.md`), disabled counter semantics and observation windows (`sram-utilization-window.md`). Define enabled/disabled transition ordering, initial/expired grants, preservation/reset of aggregate and bank counters, clock rollback/overflow, context retention, and meaning of an access while disabled. Preserve existing behavior by default. Gate both directions, 0/short/exact/long gaps, repeated toggles, per-bank budgets, `advance(0)`, disabled copies, bounded cycles, resets, W/A/O and all context scopes before shipping. Clock-gating/power claims require separate physical contracts. This exploration alone does not establish these dependencies or justify an implementation.

## Reproduce

From a clean archive of the baseline with the test source from the follow-up commit:

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make clean && make && make test-quick
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel \
  tests/explore_sram_meter_enable.c ./libtucmodel.a -lm -o "$TMPDIR/explore-sram-meter-enable"
"$TMPDIR/explore-sram-meter-enable"
```

The make clean recipe removes fixed `/tmp` paths; for this run only, those paths were redirected inside the disposable archive to avoid deleting unrelated files. Baseline clean build/quick and focused SRAM issue, word-meter and DMA suites passed; the archived static library was used for the probe. The `TERMS.md` worktree file was not touched or included.
