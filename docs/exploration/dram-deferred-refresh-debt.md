# Deferred DRAM Refresh: Debt-Preserving Grid vs Reset-After-Service

**Date:** 2026-08-05
**Question:** After a refresh command is postponed, should the next due time remain on the nominal tREFI grid or reset relative to the late command?

## Hypothesis

A controller that resets every next refresh to `late_fire + tREFI` can repeatedly postpone commands and silently reduce the long-term refresh rate. A debt-preserving controller instead leaves the next nominal due time on the original grid. The latter should bound a one-time postponement without allowing repeated deferral to compound, at the cost of less slack before the following command.

## Physically plausible alternatives

| Policy | Hardware rationale | Sacrifice / disposition |
|---|---|---|
| Nominal-grid, debt-preserving | JEDEC-style postpone/pull-in accounting retains the average command cadence. A late command consumes slack; the following nominal command is still due on the original grid. Simple counter plus bounded deadline. | Less scheduling slack after a late command; bursts can see a later latency spike. **Implemented behavior.** |
| Reset after service | Simplifies a generic periodic timer and maximizes spacing after each command. | Repeated late service can stretch the average interval and violate the intended retention contract. **Excluded as physically unsafe for this model unless an explicit refresh-credit/debt mechanism is added.** |
| Credit-counted adaptive postpone/pull-in | Real controllers can track a bounded number of postponed or pulled-in commands and schedule around queued work. This can expose useful latency/throughput trade-offs. | Requires request queues, command arbitration, idle prediction or queue visibility, and explicit JEDEC credit limits. **Blocked:** those contracts are absent. |

The cmodel's existing `fixed` and `deferred` runtime modes remain materially distinct. This exploration does not add a third mode: it validates the required debt semantics inside `deferred` and rejects an unsafe implementation shortcut.

## Executable sweep

Command:

```sh
make test-dram-refresh-debt-sweep
```

Configuration: custom 4-channel/4-bank DRAM, ALL_BANK deferred refresh, tREFI=5,000 cycles, tRFC=100 cycles, read latency=50 cycles. The first access arrives at `5,000 + delay` and opportunistically fires refresh. The model then idles to the second hard deadline. `next_grid` is read from the live model state after the first fire; `service` is the first access's returned service cycles.

| Max deferral | Delay | First fire | Next nominal grid | Second deadline | Interval to second deadline | First service |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0 | 5,000 | 10,000 | 10,500 | 5,500 | 150 |
| 500 | 250 | 5,250 | 10,000 | 10,500 | 5,250 | 150 |
| 500 | 499 | 5,499 | 10,000 | 10,500 | 5,001 | 150 |
| 2,000 | 0 | 5,000 | 10,000 | 12,000 | 7,000 | 150 |
| 2,000 | 1,000 | 6,000 | 10,000 | 12,000 | 6,000 | 150 |
| 2,000 | 1,999 | 6,999 | 10,000 | 12,000 | 5,001 | 150 |
| 5,000 | 0 | 5,000 | 10,000 | 15,000 | 10,000 | 150 |
| 5,000 | 2,500 | 7,500 | 10,000 | 15,000 | 7,500 | 150 |
| 5,000 | 4,999 | 9,999 | 10,000 | 15,000 | 5,001 | 150 |

All nine rows passed executable gates: exactly two refresh events by the second deadline, next nominal schedule exactly `2 × tREFI`, and first access service exactly `read_latency + tRFC = 150`.

## Findings

1. **The live model preserves debt.** `next_grid` is 10,000 in every row, independent of when the first command fires. Source behavior (`refresh_next += tREFI`) is therefore intentional and retention-safe; it must not be replaced by `refresh_next = fire_at + tREFI`.
2. **Late service reduces later slack.** At max-deferral-minus-one, the interval from first fire to the next hard deadline is 5,001 cycles for every max-deferral setting. At zero delay it ranges from 5,500 to 10,000. This is the cost of retaining the nominal average cadence.
3. **The prior exploration prose was wrong, not the implementation.** `dram-refresh-model.md` said the next schedule shifted to `fire + tREFI`; this sweep disproves that statement and the document is corrected in this heartbeat. Existing phase measurements remain executable outputs, but their explanation must use nominal-grid debt rather than schedule reset.
4. **Deferred refresh has no demonstrated throughput advantage in the current no-queue model.** Fixed refresh already fires during idle time through `tu_dram_tick()`. Deferred can move the lockout to an access or to a later deadline, but without queued requests, arbitration, or known future idle it cannot select a demonstrably better issue point. It remains useful as a worst-case latency/jitter alternative and as groundwork for a future queue-aware controller, not as a measured optimization.

## Gain versus sacrifice

- **Throughput:** Payload work and bytes are unchanged. Nominal-grid accounting prevents artificial throughput gains from under-refreshing. Queue-aware overlap is **unquantified**.
- **Latency:** The access that triggers an opportunistic command pays the full 100-cycle tRFC in this matrix. Deferral can relocate latency spikes; it does not remove service work.
- **Area/resources:** Nominal-grid deferred mode needs one due counter and deadline state per refresh slot. Credit-counted adaptive scheduling would require additional counters and queue/controller state; gate count is **unquantified**.
- **Power/energy:** Refresh event count is preserved over the nominal grid, preventing an unrealistically low refresh-energy estimate. Refresh command and DRAM-array energy are not wired into the power model and remain **unquantified**.
- **SRAM/DRAM traffic:** Payload bytes are unchanged. Refresh consumes DRAM command/service opportunities and precharges row state; no SRAM traffic is added.
- **Numerical accuracy:** Unaffected when timing is modeled correctly. A reset-after-service implementation could be physically retention-unsafe, but data decay itself is not modeled.
- **Control complexity:** Fixed is simplest; debt-preserving deferred adds deadline/opportunistic checks. Credit-counted adaptive scheduling is materially more complex and is blocked on a queue contract.
- **Verification burden:** The new nine-row fail-closed sweep protects nominal-grid retention. Existing focused tests protect opportunistic fire, hard deadline, event rate, row precharge, reset, config propagation, and unsupported-input rejection.
- **Compiler/runtime:** Software need not select individual refresh commands. Real-time schedulers care about the relocated worst-case spike; a future compiler-informed controller would need an explicit idle/queue hint contract.

## Fidelity limits

The model uses a 1 GHz cycle domain, one access issued per sweep tick, no request queues, no command-bus arbitration, no pull-in credits, no bank groups, and no calibrated DRAM timing. `total_refresh_stall_cycles` is a sum of per-access remaining lockout, not wall-clock makespan. The sweep validates schedule state and service accounting only; it does not prove JEDEC protocol conformance.

## Actionable conclusion

Keep `fixed` and debt-preserving `deferred`; do not add reset-after-service as an alternative because repeated use can under-refresh. Correct the documentation to match the executable nominal-grid semantics. A genuinely adaptive postpone/pull-in policy remains a high-value but **BLOCKED** candidate until the cmodel has request queues, arbitration, explicit refresh credits, and traces that can validate when moving a command is useful rather than merely shifting latency.
