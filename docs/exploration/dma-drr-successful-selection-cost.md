# DMA DRR successful-selection cost

**Date:** 2026-10-09
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

Should deficit round-robin (DRR) selection be modeled as free whenever a descriptor is eligible, or should every successful scheduler decision consume visible control cycles?

Both are physically plausible:

- `drr_select_issue_cycles = 0` preserves the compatibility lower bound. It can represent a combinational selector, a deeply pipelined scheduler whose latency is hidden, or an aggregate model that excludes control latency.
- A nonzero fixed cost models a registered eligibility/deficit decision with queue-count-independent visible latency.
- A nonzero `per_channel` cost models a serial full-queue scan before issue. It reuses the existing `drr_round_cost_mode`; the configured cost is multiplied by the number of configured channels.

This setting is independent of `drr_round_issue_cycles`: unsuccessful rounds add deficit credit and may repeat before any descriptor is eligible, while a successful-selection cost is charged exactly once immediately before each selected descriptor is issued.

## Runtime contract

```yaml
dma:
  arbitration: "deficit_round_robin"
  channels: 8
  drr_select_issue_cycles: 1
  drr_round_cost_mode: "per_channel"
```

`0` is the generated, canonical, zero-initialized, and live-engine default. Values `[0,1024]` are accepted. Unsupported values fail without mutating the accepted live setting. `fixed` and `per_channel` remain the existing scheduler-cost alternatives and apply consistently to unsuccessful rounds and successful selections.

Changing only successful-selection latency does not clear deficits: it changes no credit, service order, continuation, or queue ownership. Changing the shared cost mode still clears deficit and continuation state because it changes the timing contract of both scheduler stages.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Controlled workload:

- shared-serial, non-preemptive DMA path;
- useful-byte DRR, exact charging, 64-byte quantum, unit weights;
- one 256-byte linear load on channel 0;
- no unsuccessful-round cost for the successful-selection rows;
- 256-bit interface, 50-cycle read base, zero visible burst-issue cost;
- SRAM bandwidth metering disabled;
- byte-exact destination comparison.

Measured matrix:

| Successful-selection cycles | Cost mode | Configured channels | Descriptor completion | Batch completion | Added vs zero |
|---:|---|---:|---:|---:|---:|
| 0 | `fixed` | 2 | 59 | 59 | 0 |
| 1 | `fixed` | 2 | 60 | 60 | 1 cycle (+1.69%) |
| 4 | `fixed` | 2 | 63 | 63 | 4 cycles (+6.78%) |
| 1 | `fixed` | 8 | 60 | 60 | 1 cycle (+1.69%) |
| 1 | `per_channel` | 8 | 67 | 67 | 8 cycles (+13.56%) |

All rows transfer the same 256 useful bytes and produce identical SRAM contents. The setting exposes scheduler-control latency; it does not change payload serialization, traffic, arithmetic, or numerical accuracy.

The same sweep separately retains the unsuccessful-round control: with three unsuccessful rounds at one cycle each, fixed completion is 62 cycles for 1/2/4/8 channels, while `per_channel` completion is 62/65/71/83 cycles. This proves the two costs compose without conflating one successful selection with the preceding credit rounds.

## Multi-objective trade-offs

| Dimension | Zero / hidden selection | Fixed visible selection | Per-channel visible selection |
|---|---|---|---|
| Throughput | Lower-bound descriptor rate; optimistic for short descriptors | One configured bubble per descriptor; impact amortizes on long payloads | Queue-count-scaled bubble per descriptor; short-transfer rate can become scheduler-bound |
| Latency | 59 cycles in the measured workload | 60 cycles at cost 1; 63 at cost 4 | 67 cycles for 8 channels at cost 1 |
| Area/resources | May imply combinational or hidden-pipeline selector resources; unquantified | Can represent a registered parallel selector | Can represent one reused scanner plus index/state; expected lower parallel read/comparator demand, unquantified |
| Power/energy | Potentially more concurrent queue-state activity, or omitted control energy | Registered logic adds clocked activity once per descriptor | Repeated queue-state reads add control cycles; net energy versus parallel logic is technology-dependent and unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged | Unchanged; queue probes generate no payload traffic |
| Numerical accuracy | Unchanged; movement remains byte exact | Unchanged | Unchanged |
| Control complexity | Simplest timing abstraction but may hide a real stage | Explicit issue-stage timing and saturation | Iterative scanning, queue-count dependence, and longer-lived scheduler state |
| Verification burden | Compatibility/default gate | Exact per-descriptor additive timing and range rejection | Exact channel-count scaling plus interaction with unsuccessful-round delay |
| Compiler/runtime | No scheduler-aware descriptor-size pressure | Short descriptors pay proportionally more; coalescing may amortize cost | Overprovisioned queue contexts increase latency; software may choose fewer queues or larger descriptors |

No mode is universally preferred. A throughput-oriented mover may spend comparator/read-port resources to hide or bound selection latency. A small implementation may reuse a narrow scanner and accept queue-count-sensitive control bubbles. The cmodel retains all three timing points instead of replacing the compatibility lower bound with the fastest or most conservative row.

## Fidelity limits

- `per_channel` models a full scan of every configured channel, not early termination at the selected queue.
- The same `fixed`/`per_channel` mode applies to unsuccessful rounds and successful selections. Independent stage widths would need a stronger scheduler microarchitecture contract.
- Intermediate scanner widths, clock-gated empty queues, pipelined overlap, queue-state SRAM port conflicts, and producer injection during aggregate delay are unmodeled.
- Selection delay is represented as an atomic cycle advance inside one `tu_dma_tick()` call; no concurrent event can arrive during it.
- Arbitration remains descriptor-boundary and non-preemptive.
- Physical selector area, timing closure, power/energy, finite credits/backpressure, shared-memory contention, and calibration are unquantified.

## Verification

```sh
make test-config-generation
make test-config
make test-dma
make test-dma-arbitration-sweep
make config-docs
make clean && make
make test-quick
```
