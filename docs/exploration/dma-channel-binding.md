# DMA Descriptor Channel Binding

**Date:** 2026-08-26
**Status:** Implemented and executable
**Question:** When producers do not distribute descriptors across independent DMA queues, should software retain explicit placement, should hardware rotate submissions, or should hardware inspect live queue occupancy?

## Physically plausible alternatives

| Mode | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| `explicit` | Dedicated W/A/O movers, static compiler scheduling, or a minimal controller where queue identity carries meaning | A skewed or mistaken producer can serialize work on one path |
| `round_robin` | Small, deterministic distributor for homogeneous paths; no queue-count comparison | Ignores unequal service time and can assign new work behind a long transfer |
| `least_outstanding` | Dynamic multi-producer systems where descriptor service times and arrivals vary | Requires active+queued counters, comparison, rotating tie state, and more verification |

`explicit` is numeric zero and remains the checked-in and zero-initialized compatibility default. Automatic modes rebind at accepted submission time. Least-outstanding counts active plus queued descriptors and rotates equal-count ties. Invalid policies and invalid explicit channels fail closed; a rejected submission does not advance automatic-selection state.

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

In this measured skewed independent-path regime, either automatic policy reduces batch completion by 53.7% versus explicit placement. Least-outstanding moves the first two second-wave descriptors to the idle paths, lowering those descriptor completions from 613/486 to 486/486 compared with round-robin, while moving the third from 486 to 613. Batch completion is identical. This is a latency distribution trade, not a universal throughput win.

## Gain versus sacrifice

- **Throughput / batch latency:** Automatic binding exposes already-modeled independent-path concurrency when producer placement is skewed. No benefit is expected for a shared-serial bus with the same finite batch because binding changes queue placement, not payload service capacity.
- **Per-request latency:** Round-robin is deterministic but may queue behind a long transfer. Least-outstanding reacts to live descriptor counts, but count is not remaining-byte work; it can still choose poorly for unequal payloads. Explicit placement can be best when the compiler knows dependencies or path affinity.
- **Area/resources:** Explicit needs no distributor. Round-robin adds a channel pointer/modulo selection. Least-outstanding adds active/queued count reads, comparators, and tie state. Physical area is unquantified.
- **Power/energy:** All modes move identical useful bytes in this experiment. Automatic selection adds expected controller switching; any reduction in leakage from shorter completion is not wired to the power model. Energy is unquantified.
- **SRAM/DRAM traffic:** Byte traffic is identical and byte-exact movement is gated. The experiment does not model shared SRAM ports, a common DRAM bandwidth cap, or queue-aware DRAM service, so distribution must not be interpreted as three physical memory ports.
- **Numerical accuracy:** No arithmetic changes; numerical accuracy is unaffected.
- **Control complexity:** Explicit is simplest. Round-robin requires deterministic state. Least-outstanding requires a combinational or pipelined minimum reduction and an atomic accepted-submission update.
- **Verification burden:** Every mode needs exact assignment, completion, movement, default, parser/propagation, invalid-mode, invalid-channel, tie rotation, and rejection-without-state-mutation gates. Dynamic arrivals enlarge interleaving coverage.
- **Compiler/runtime:** Explicit preserves dedicated W/A/O semantics and compiler control. Automatic modes assume interchangeable queues; software loses exact placement unless it selects `explicit`. Runtime telemetry may help choose a policy, but no automatic policy switching is modeled.

## Implementation path

`tu.dma.channel_binding` in JSON/YAML accepts `explicit`, `round_robin`, or `least_outstanding`. The path is YAML/JSON → generator/header constants → canonical `tu_config_t` → `tu_runtime_config_t` → `tu_init_with_config()` → `tu_dma_init_config_full()` → live submission. `tu_dma_init_config_policy()` remains a compatibility wrapper selecting explicit binding.

## Fidelity limits

The descriptor engine is deterministic and tick-driven. Independent channel mode permits concurrent abstract transfer service but does not prove replicated DRAM interfaces or SRAM ports. Binding uses descriptor count, not bytes, predicted completion, deadlines, dependencies, locality, priority, or energy. It does not model producer issue overhead, descriptor fetch, hardware critical path, queue backpressure replay, path affinity, migration, or host-thread concurrency. Consequently these results support keeping all three alternatives configurable, not selecting one physical design universally.

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
