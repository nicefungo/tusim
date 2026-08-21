# Zero-Byte DRAM Request Semantics

**Date:** 2026-08-20 (planning-estimator consistency follow-up: 2026-08-21)
**Question:** Should a zero-byte request consume base latency, row state, direction state, or a fixed protocol burst?

## Hypothesis and realistic alternatives

A descriptor pipeline can encounter an empty transfer after shape collapse, predication, tail generation, or compiler tiling. Three behaviors are conceivable at different abstraction boundaries:

| Behavior | Plausible boundary | Decision and sacrifice |
|---|---|---|
| Reject the descriptor | ISA/command validation with an error-return contract | Strongest bug detection, but the current `void tu_dram_read/write` service API cannot report rejection |
| Side-effect-free no-op | DRAM service API after an upper layer accepts or elides the descriptor | Preserves composability and cannot invent traffic; may hide an upstream empty-descriptor bug unless that layer counts/rejects it |
| Charge a minimum burst | Only if the interface defines a real command despite zero useful bytes | Rejected for this API: no data command exists to justify row activation or occupied bytes, and the behavior makes address low bits create traffic from an empty span |

The cmodel now defines zero-byte DRAM service calls as side-effect-free no-ops. This is a correctness contract shared by all existing timing modes, not a new architecture knob. A future ISA or DMA descriptor validator may separately reject zero-length commands before they reach DRAM.

## Defect found

Before the service-path change, every non-ideal zero-byte read/write paid base latency and changed request, row, direction, and channel-availability state. In `burst_span_credit`, an unaligned empty request also computed:

```text
ceil(((address mod granule) + 0) / granule) * granule
```

so address 1 could occupy 64 bytes despite carrying no payload. That is not a physically meaningful data transfer.

The 2026-08-21 audit found the same contract remained inconsistent at the planning boundary: `tu_dram_estimate_transfer(dram, 0, direction)` returned the configured read or write base latency for every non-ideal DRAM type. A DMA planner could therefore reserve cycles for work that the service path correctly elided. The estimator now returns zero before latency or bandwidth arithmetic. This is a semantic correction, not a selectable hardware optimization.

## Executable matrix

Command: `make test-dram-zero-byte-sweep`

Configuration: custom one-channel/one-bank DRAM, open-page row policy, 64 B read/write granules, 8 B/cycle bus, 10/8-cycle read/write service, and 3/8-cycle turnaround. The one-byte control primes the coarse bandwidth window before measurement.

| Mode | Read zero bytes | Write zero bytes | Read/write estimate | Requests | Occupied bytes | Row misses |
|---|---:|---:|---:|---:|---:|---:|
| NONE | 0 cycles | 0 cycles | 0 / 0 cycles | 0 | 0 | 0 |
| FIXED | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| IDLE_CREDIT | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| BURST_CREDIT | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| BURST_ROUND_CREDIT | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| BURST_SPAN_CREDIT | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| One-byte span control | 30-cycle read | — | not an estimator equivalence claim | 1 | 64 | 1 |

All 12 zero-byte direction/mode rows also gate a zero planning estimate, zero contention stall, unchanged open-row state, unchanged channel availability, unchanged direction history, and zero pending bytes. The one-byte control proves the service gate does not suppress the smallest nonempty request: it pays 10 base + 20 activation cycles and occupies one 64 B granule. The bulk estimator is intentionally not compared to that 30-cycle result because it has no address, row-state, refresh, or direction-history inputs and therefore represents a different timing boundary for nonempty traffic.

## Gain versus sacrifice

- **Throughput and latency:** Empty service calls and planning estimates now consume zero modeled cycles and cannot delay or reserve following work. This removes fictitious overhead; it is not a throughput optimization for real payloads.
- **Area/resources:** No TU datapath change is implied. Real hardware usually needs a zero-length compare in descriptor validation or command issue; area is unquantified and expected to be negligible relative to the DMA controller.
- **Power/energy:** No data command, row activation, or bus occupancy is modeled for an empty request. Avoided physical control energy is unquantified.
- **SRAM/DRAM traffic:** Both useful and occupied traffic remain exactly zero. One-byte fixed-burst behavior remains unchanged.
- **Numerical accuracy:** Unchanged because no payload exists.
- **Control complexity:** No-op handling is simple at the service boundary. Rejecting upstream descriptors would add error propagation and recovery semantics not present in this API.
- **Verification burden:** Every timing mode and both directions must remain gated because span rounding, row policy, refresh, turnaround, and bandwidth accounting otherwise mutate independent state domains.
- **Compiler/runtime:** Compilers may safely elide empty tiles. If a planner encounters one, the estimate now agrees with service elision instead of reserving base latency. If a command producer emits one, command-level diagnostics remain a separate missing contract.

## Implementation and verification

- `tu_cmodel/memory/dram_model.c`: early zero-byte returns before ideal/non-ideal accounting in read, write, and bulk-estimate paths.
- `tu_cmodel/memory/dram_model.h`: estimator no-work contract documented.
- `tests/test_dram.c`: focused state-invariance regression plus read/write estimator gates. The estimator gate was observed RED (`zero-byte read estimate charged latency`) before the follow-up implementation, then GREEN.
- `tests/test_dram_zero_byte_sweep.c`: 12 mode/direction service-and-estimate gates plus a one-byte service control.
- `Makefile` and `.gitignore`: focused sweep target and artifact lifecycle.

Executed:

```text
make test-dram                 # 2026-08-21 estimator RED: 32/33; then GREEN: 33/33
make test-dram-zero-byte-sweep # 12 zero-byte service/estimate rows + one-byte control
```

Final clean build and quick regression are recorded in the heartbeat report.

## Fidelity limits and conclusion

This establishes DRAM service and side-effect-free bulk-planning semantics only. It does not define whether text ASM, packed ISA, command queue, DMA descriptors, or compiler lowering reject, count, retire, or signal a zero-length command. Those producer/consumer surfaces must be audited separately before claiming command-level behavior. For nonempty requests, `tu_dram_estimate_transfer()` remains a coarse latency-plus-bandwidth planner and is not equivalent to address-aware live service.

Keep side-effect-free no-op semantics in DRAM and preserve all six existing timing alternatives for nonempty requests. Do not add a “minimum burst for zero bytes” mode without a protocol that proves a real command is issued. Consider upstream rejection only when the relevant API can return a status and its retirement/error contract is executable.
