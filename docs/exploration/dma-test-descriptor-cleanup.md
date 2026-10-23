# DMA regression-harness descriptor cleanup

**Audit:** 2026-10-23, baseline `450fb16` and independent-stage-width implementation. **Status:** READY follow-up; not implemented in this heartbeat.

## Reproduced defect and ownership evidence

An extended all-object ASan/UBSan build with leak detection passes both the independent-stage-width sweep (8,425 timing rows / 162 queued gates) and historical common-width sweep (874 rows / nine queue gates), but `make test-dma` exits nonzero. LeakSanitizer reports **3,136 bytes in 14 descriptor allocations**. A fresh unchanged-baseline archive built with the same instrumentation reports the same bytes/allocation count and allocation-site categories. This is not a new stage-width regression or a functional DMA byte error. Ordinary `test-dma` still passes 15/15.

Source inspection establishes why a blanket engine-destroy assumption is unsafe:

- Synchronous `tu_dma_submit_desc()` executes and removes queue heads, decrements depth and clears tail, but does not free the completed descriptor.
- Async tick clears the active pointer after its completion timestamp, without freeing the retired descriptor.
- `tu_dma_destroy()` visits pending heads and current active descriptors only; completed detached descriptors are no longer reachable there.
- Early fixtures in `tests/test_dma.c` keep local descriptor pointers but never destroy them after successful sync execution or async retirement. The three-node sync-chain fixture similarly leaves the head and two reachable nodes allocated. Allocation stacks also identify completion-signal/statistics/outstanding/rotating-priority fixtures.
- Destroy traverses `next`, so freeing a chain head and separately freeing nodes reachable from it is wrong. Active/pending/rejected cases require their own ownership trace; do not add unconditional frees at every function tail.

Retained evidence: `evidence/dma-drr-stage-width-2026-10-23/sanitized-gate.log.gz`, `baseline-sanitized-dma.log.gz`, plus the successful `sanitized-focused.log.gz`. They include complete allocation stacks, not just the summary count.

## Alternatives, costs and verification criterion

1. **Explicit caller/harness cleanup after detachment** (READY): retain the actual legacy lifetime contract; free each detached descriptor or intact chain exactly once, preserve engine-owned pending/active objects, and use single cleanup paths on failed assertions. This repairs the harness without changing production APIs or retirement semantics.
2. **Engine-owned retired list / automatic retirement free** (not selected): could reduce caller bookkeeping but changes ownership, descriptor-query lifetime, storage and API expectations. It needs an explicit lifecycle contract and is not justified merely to remove test leaks.
3. **Suppress leak detection** (excluded as an acceptance strategy): hides the defect rather than verifying cleanup.

The next repair must preserve every existing numerical/byte/timing/rejection assertion and 15/15 normal tests, then pass a fresh all-object ASan/UBSan/LSan `test-dma` with **zero leaks and no double free/use-after-free**. Re-run both DRR scan sweeps and clean build/quick gates. Audit success, rejection, early-return, pending/active and completed paths separately; do not change production ownership to green a test.

Expected gain: bounded host test-harness allocation lifetime. Expected sacrifice: explicit cleanup bookkeeping and broader failure-path verification. Device throughput/latency, physical area/power/energy, payload SRAM/DRAM traffic and numerical accuracy are unchanged by the proposed harness-only repair; no hardware gain is claimed. No runtime architecture knob is appropriate for a leaked test allocation. This independent item remains for the next small-backlog heartbeat, rather than expanding today's stage-width implementation scope.
