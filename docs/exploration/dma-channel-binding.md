# DMA Descriptor Channel Binding

**Date:** 2026-08-28
**Status:** Implemented and executable
**Question:** When producers do not distribute descriptors across independent DMA queues, should software retain explicit placement, should hardware rotate submissions, or should hardware inspect descriptor count, assigned bytes, or projected service cycles?

## Physically plausible alternatives

| Mode | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| `explicit` | Dedicated W/A/O movers, static compiler scheduling, or a minimal controller where queue identity carries meaning | A skewed or mistaken producer can serialize work on one path |
| `round_robin` | Small, deterministic distributor for homogeneous paths; no queue-count comparison | Ignores unequal service time and can assign new work behind a long transfer |
| `least_outstanding` | Dynamic multi-producer systems where descriptor service times and arrivals vary | Requires active+queued counters, comparison, rotating tie state, and more verification |
| `least_bytes` | Variable-size transfers on interchangeable paths where assigned byte volume is a better service proxy than descriptor count | Requires wide byte accumulation/comparison and can overestimate an active transfer that is nearly complete |
| `least_projected_cycles` | Tick-driven controllers that already retain completion countdowns can account for elapsed active service and coarse queued service | Requires timestamp/countdown arithmetic plus per-descriptor cycle estimates; queued SRAM/backpressure costs are not side-effect-free in this model |

`explicit` is numeric zero and remains the checked-in and zero-initialized compatibility default. Automatic modes rebind at accepted submission time. Least-outstanding counts active plus queued descriptors. Least-bytes sums the original byte sizes of active and queued descriptors. Least-projected-cycles uses the live active descriptor's exact remaining scheduled cycles (`cycles_completed - current_cycle`) plus a side-effect-free queued estimate of base latency and payload serialization. Stateful SRAM refill penalties are deliberately excluded from queued estimates rather than guessed. All minimum policies rotate equal-value ties. Invalid policies and invalid explicit channels fail closed; a rejected submission does not advance automatic-selection state.

## Executable experiment

Command:

```sh
make test-dma-binding-sweep
```

The harness uses three independently serviceable channel paths and six descriptors which all request channel 0. At cycle 0 it submits one 12 KiB and two 4 KiB transfers. At cycle 307, after the two short automatically distributed transfers can retire while the long transfer remains active, it submits three more 4 KiB transfers. SRAM bandwidth metering is disabled in this focused harness to isolate descriptor binding; the live DMA base-latency and 256-bit serialization model remain active.

| Policy | Assigned channels (descriptors 0..5) | Completion cycles (0..5) | Batch completion |
|---|---|---|---:|
| explicit | 0,0,0,0,0,0 | 435,613,791,969,1147,1325 | 1325 |
| round_robin | 0,1,2,0,1,2 | 435,179,179,613,486,486 | 613 |
| least_outstanding | 0,1,2,1,2,0 | 435,179,179,486,486,613 | 613 |
| least_bytes | 0,1,2,1,2,1 | 435,179,179,486,486,664 | 664 |
| least_projected_cycles | 0,1,2,1,2,0 | 435,179,179,486,486,613 | 613 |

In this measured skewed independent-path regime, all automatic policies reduce batch completion versus explicit placement: round-robin, least-outstanding, and least-projected-cycles by 53.7%, and least-bytes by 49.9%. Least-projected-cycles observes only 128 cycles left on channel 0 at cycle 307, so it places the third second-wave descriptor there and matches the 613-cycle batch. Least-bytes still charges that active descriptor's full 12 KiB, sends the descriptor to channel 1, and finishes at cycle 664. Its assigned byte totals are more balanced (12/12/8 KiB versus projected-cycles' 16/8/8 KiB), but its active-work estimate is stale in this late-arrival case. Byte-aware distribution remains useful for early arrivals and for hardware that lacks completion countdowns; projected cycles is not universal because queued SRAM/backpressure service is coarsely estimated.

## Gain versus sacrifice

- **Throughput / batch latency:** Automatic binding exposes already-modeled independent-path concurrency when producer placement is skewed. No benefit is expected for a shared-serial bus with the same finite batch because binding changes queue placement, not payload service capacity.
- **Per-request latency:** Round-robin is deterministic but may queue behind a long transfer. Least-outstanding reacts to live descriptor counts, but count is not remaining work. Least-bytes distinguishes unequal descriptors but counts active work at full size. Least-projected-cycles accounts for elapsed active service and coarse queued service, but can misrank paths when SRAM penalties or heterogeneous path rates dominate. Explicit placement can be best when the compiler knows dependencies, locality, or affinity.
- **Area/resources:** Explicit needs no distributor. Round-robin adds a channel pointer/modulo selection. Least-outstanding adds active/queued count reads and comparators. Least-bytes additionally needs per-queue byte totals and wider comparators. Least-projected-cycles needs countdown/timestamp subtraction, queued service accumulation, and comparators. Physical area, timing, and storage are unquantified.
- **Power/energy:** All modes move identical useful bytes in this experiment. Automatic selection adds expected controller switching; any reduction in leakage from shorter completion is not wired to the power model. Energy is unquantified.
- **SRAM/DRAM traffic:** Byte traffic is identical and byte-exact movement is gated. The experiment does not model shared SRAM ports, a common DRAM bandwidth cap, or queue-aware DRAM service, so distribution must not be interpreted as three physical memory ports.
- **Numerical accuracy:** No arithmetic changes; numerical accuracy is unaffected.
- **Control complexity:** Explicit is simplest. Round-robin requires deterministic state. The minimum policies require a combinational or pipelined reduction and an atomic accepted-submission update. Least-bytes needs overflow-safe byte accounting; projected cycles needs overflow-safe time arithmetic and a clearly scoped service estimator.
- **Verification burden:** Every mode needs exact assignment, completion, movement, default, parser/propagation, invalid-mode, invalid-channel, tie rotation, and rejection-without-state-mutation gates. Dynamic arrivals enlarge interleaving coverage.
- **Compiler/runtime:** Explicit preserves dedicated W/A/O semantics and compiler control. Automatic modes assume interchangeable queues; software loses exact placement unless it selects `explicit`. Runtime telemetry may help choose a policy, but no automatic policy switching is modeled.

## Implementation path

`tu.dma.channel_binding` in JSON/YAML accepts `explicit`, `round_robin`, `least_outstanding`, `least_bytes`, or `least_projected_cycles`. The path is YAML/JSON → generator/header constants → canonical `tu_config_t` → `tu_runtime_config_t` → `tu_init_with_config()` → `tu_dma_init_config_full()` → live submission. `tu_dma_init_config_policy()` remains a compatibility wrapper selecting explicit binding.

## Fidelity limits

The descriptor engine is deterministic and tick-driven. Independent channel mode permits concurrent abstract transfer service but does not prove replicated DRAM interfaces or SRAM ports. Least-outstanding ignores size; least-bytes ignores elapsed active service; least-projected-cycles uses an exact active countdown but queued base+serialization estimates omit stateful SRAM penalties, contention, and heterogeneous rates. No mode uses deadlines, dependencies, locality, priority, energy, or destination contention. The model omits producer issue overhead, descriptor fetch, hardware critical path, queue backpressure replay, path affinity, migration, and host-thread concurrency. Consequently these results support keeping all five alternatives configurable, not selecting one physical design universally.

## Verification

```sh
make test-dma-binding-sweep
make test-dma
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make config-docs
make clean && make
make test-quick
```
