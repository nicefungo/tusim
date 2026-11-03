# SRAM refill state across context switches

**Source:** `57e611c` (2026-11-03 audit). This is an observational regression snapshot, not a fixed-grid implementation or a correctness gate that should preserve the behavior.

## Question and execution

The refill-phase alternative in `sram-refill-phase.md` was marked READY assuming region time and per-bank refill state could be tested coherently across a context switch. The current context save copies `current_cycle` but deliberately omits `bw_banks`; restore assigns the saved cycle while retaining the other context's physical bank array. The current refill routine subtracts the retained last-refill timestamp as unsigned arithmetic.

`tests/explore_sram_context_refill.c` creates one core, one four-cycle-window bank with a one-word grant and two FULL_SRAM contexts. Context 0 consumes its grant at cycle 0. The control does not switch. In the switch case, context 1 advances its region to cycle 5, and context 0 is restored to cycle 0. Both cases then invoke zero-cycle advance and issue a second scalar access. Read and write variants check the returned stall and bytes. Both contexts are allocated before the first access.

| Direction | Path | Restored cycle | Last refill before / after zero advance | Budget before / after | Second penalty |
|---|---|---:|---:|---:|---:|
| read | no switch | 0 | 0 / 0 | 0 / 0 | 2 |
| read | switch | 0 | 5 / 0 | 1 / 1 | 0 |
| write | no switch | 0 | 0 / 0 | 0 / 0 | 2 |
| write | switch | 0 | 5 / 0 | 1 / 1 | 0 |

The switch leaves the other context's grant and timestamp in place while restoring context 0's earlier time. At zero advance, unsigned subtraction sees a large elapsed interval and resets the timestamp from 5 to 0. The return/bytes match all four **pinned observations**; no behavior is endorsed as a correctness oracle. Full output is in `evidence/2026-11-03-sram-context-refill/matrix.log`. The original no-context refill-phase probe still passes eight controls.

## Alternatives and trade-offs

* **Context-local meter and time:** snapshot and restore bank grants, phase, timestamps and meter counters as a coherent unit. This supports deterministic virtual timelines, but costs per-context state, switch bookkeeping and verification; the transfer-cost model would need an explicit control-state accounting decision.
* **Physical shared meter and monotonic physical time:** retain both region time and per-bank state across context changes, with a distinct context-local timeline where needed. This can model one continuously running shared SRAM, but context interference is intentional, scheduling and idle/switch-time attribution become explicit, and different behavior must not be called context isolation.
* **Drain/reset on switch:** may simplify saved state, but resetting a grant can create excess capacity unless the reset/restart event and its cost are defined. Neither this nor shared time is automatically wrong or faster.

Fixed-grid versus restart remains a useful separate architecture comparison. Until a coherent context/time contract or an explicit fail-closed unsupported combination is implemented and tested, adding only a phase selector would leave its advertised context behavior unverified. The attempted selector was reverted, and no new runtime mode is claimed. Accumulating credits remain separately blocked on credit capacity and gating rules.

The observation changes local model penalties, not measured physical throughput or latency. Physical SRAM/DRAM traffic, area/resources, power/energy and actual timing are unquantified. Expected costs include snapshot storage or shared-time control; the direction of energy and throughput changes depends on scheduling and implementation. Bytes and numerical operations in these probes are unchanged; compiler/runtime implications are context reproducibility and time/ownership semantics, not a measured compiler cost. Arbitration, physical ports, replay, and clock-gating remain unmodeled.

## Verification and next gate

The future implementation must exercise both policies, both SRAM directions, several banks, FULL/LIVE/CONTROL retention, context switching in both directions, fixed versus restart phase, disabled/re-enabled metering, rollover, budget/counter preservation or explicitly specified reset, core snapshot and strict config rejection. Distinguish retired/executed control-state traffic from SRAM payload. Keep the historical default behavior unless a separately documented compatibility fix is justified. The diagnostic above must not be promoted to a passing gate for the defect.

On a clean source archive, use `make clean && make && make test-quick` with `LD_LIBRARY_PATH` set. The checkout's `make clean` deletes `/tmp/test_asm`; the baseline archive replaced that pathname and `/tmp/gpt_block_tu*` with archive-local paths to avoid deleting unrelated host files. Then:

```sh
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel tests/explore_sram_context_refill.c ./libtucmodel.a -lm -o "$TMPDIR/sram-context-refill"
"$TMPDIR/sram-context-refill"
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel tests/explore_sram_refill_phase.c ./libtucmodel.a -lm -o "$TMPDIR/sram-refill-phase"
"$TMPDIR/sram-refill-phase"
```

Only the source paths above and stated FULL_SRAM/read/write pattern were exercised; no conclusion is made about LIVE or CONTROL retention or physical performance.
