# Multicast atomic-rejection regression contract

**Date:** 2026-10-15 · **Mode:** pre-spec correctness/verification maintenance
**Execution-input baseline:** `b7eebe5daa6b015af484d09482e46a1f4f3f8bb8`
**Question:** can the public multicast descriptor path reject one bad target without copying earlier valid targets, fabricating completion, or changing engine state?

## Findings and implementation

The existing shared span validator already enforced whole-descriptor rejection. The old multicast regression instead required overflow to be skipped and the descriptor marked complete; `docs/multicast-dma.md` repeated that obsolete contract. Baseline `make test-multicast` reproduced 9/10 passing, with the stale completion assertion failing.

This item repairs the executable regression and maintained documentation, **not** the production copy/timing mechanism. No numerical semantics, public API, runtime knob, or default architecture changed. Unsafe skip-and-complete is not preserved as a selectable variant merely to make a test green.

- `tests/test_multicast.c`: replaces that assertion with 18 rejection gates; adds three multi-region exact-end controls; explicitly destroys the caller-owned successful synchronous chain rather than leaking it.
- `Makefile`: links multicast against the current static archive, adds it to `test-quick`, and removes its binary in `clean`. `.gitignore` already covers the binary.
- `tests/verify_multicast_contract.py`: reproducible isolated sanitizer and safe mutation runner. It archives current HEAD, overlays the worktree test/Makefile, and never mutates production source in the checkout.
- `docs/multicast-dma.md`: corrects bounds, ownership, fanout/counter evidence, and metadata-versus-runtime claims.
- `IMPLEMENTATION_BACKLOG.md`: sole READY item becomes DONE.

## Complete verification matrix

| Surface | Targets and invalidity | Cases | Required / observed |
|---|---|---:|---|
| Direct execute | Three 64-byte regions; 8-byte source; invalid target first/middle/last; offset 57 (one-byte overflow) or null region | 6 | No target bytes changed, fresh descriptor incomplete with zero timestamps, entire engine snapshot unchanged |
| Synchronous submit | Same target/invalidity matrix | 6 | Return zero, rejected descriptor consumed, all regions and engine snapshot unchanged |
| Asynchronous submit | Same target/invalidity matrix | 6 | Same rejection contract before admission/binding/issue |
| Direct / sync / async exact-end | 8 bytes to offsets 56/120/248 in 64/128/256-byte regions | 3 | Full-region guard/payload equality, completed descriptor, one transfer, 24 useful bytes, 32 occupied bytes, 51 aggregate service cycles |
| Existing suite cases | Constructor, single/three-target copy, original exact-end, null arguments, destruction, explicit perf recording, 16-target fanout, chain | 9 | Pass |

**Result: 30/30.** Expected rejection diagnostics are retained in raw output. Repeated test labels correspond to the loop matrix above, not duplicated claims.

The 51-cycle control is the default 50-cycle read base plus `ceil(24/32)=1` payload cycle. It is aggregate descriptor service, not calibrated multicast-tree latency; async issue/tick time is a separate convention. Rejection measures no service because no command is admitted.

The safe mutant disables only the direct span-prevalidation guard in a temporary archive, leaving private target bounds checks and submission validation intact. All six direct invalid-target cases then fail because the old partial-copy path marks them completed. The mutant suite reports 24/30, so the gate rejects the documented obsolete behavior. It does not intentionally execute an out-of-bounds copy.

## Alternatives and multi-objective consequences

| Dimension | Whole-descriptor prevalidation (current) | Partial target fault / skip-and-complete |
|---|---|---|
| Why choose it | Command-atomic validation avoids partial corruption; useful for compiler-generated scratchpad transfers | Streaming hardware might detect an error only when reaching a target |
| Throughput / latency | Valid modeled service unchanged; validation's simulator/hardware cost unquantified | Could avoid up-front walking but requires explicitly modeled fault/retirement latency; no valid measurement today |
| Area / resources | Expected address/range checks and target-list access, or equivalent compiler verification; unquantified | Expected per-target status and recovery bookkeeping; unquantified |
| Power / energy | Valid traffic unchanged; checking activity expected to add control energy, unquantified | A copied prefix consumes energy before fault; recovery cost unquantified |
| SRAM / DRAM traffic | Valid useful/occupied traffic unchanged; rejected work copies zero bytes | Partial destination writes would need exact accounting; source-read reuse is not modeled |
| Numerical accuracy | Byte-exact valid copies, no arithmetic changes | Valid prefix could be byte-exact while the tensor is incomplete; cannot be called successful |
| Control complexity | Validate every target before admission/copy | Requires explicit partial/error status, signals, rollback policy, and retirement semantics |
| Verification burden | Invalid first/middle/last, both invalidity classes, three entry surfaces, exact-end controls | Larger error/recovery state space; no executable contract yet |
| Compiler / runtime | Submission failure consumes rejected chain; direct rejection retains caller ownership | Producers need partial-result/retry semantics before this is safe |

Partial faults remain **BLOCKED**, not excluded for being slower. Silent skip-and-complete is **EXCLUDED** as an unsupported success contract. Repeated unicast and physical broadcast trees are physically plausible distribution architectures, but this fix neither compares nor implements their hardware timing. No throughput/area/energy ranking is inferred.

## Backlog re-audit

The backlog now has zero READY candidate rows. Rechecked the multicast span path and SRAM arbitration: the latter still stores `arb_mode` without consuming it in scalar arbitration, and has no simultaneous requester/port API, so its existing BLOCKED status stands. Direct compressed-domain compute and physical dataflow traffic still need their documented token/FIFO and movement/overlap contracts. With one ready independent item at entry, this run follows the one-item/day rule rather than adding an unrelated speculative knob.

The broad suite remains blocked by pre-existing attention numerical failures: deterministic-small and causal-mask cases each report `max_err=inf`; suite result 7/9. A fresh archive of the baseline HEAD reproduces both failures. No attention source was changed, and this run does not claim a full-suite pass.

## Commands and retained evidence

```sh
make clean && make
make test-quick
make test-multicast test-dma test-scatter-gather test-dma-arbitration-sweep
python3 tests/verify_multicast_contract.py
make test
```

Initial and final clean builds and quick suites pass. Focused results: multicast 30/30; DMA 15/15; scatter/gather 15/15; arbitration sweep PASS with exact historical alternatives. Isolated ASan/UBSan/LSan multicast run passes 30/30. The full aggregate stops at attention 7/9, independently reproduced on baseline.

Raw transcripts in `evidence/multicast-atomic-rejection/`:

- `baseline-multicast.log`: original 9/10 stale-contract failure.
- `baseline-attention.log`: clean baseline archive's unrelated 7/9 attention result.
- `build.log`, `quick.log`, `focused.log`, `full.log`: post-change commands, including the complete arbitration trade-off matrix and full-suite blocker.
- `sanitizer.log`, `mutation.log`: isolated sanitizer run and rejected safe mutant.

Untracked user `TERMS.md` is untouched and excluded. No HANDOFF or binary is published. Evidence logs are immutable raw output, not fabricated benchmark data.
