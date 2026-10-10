# DMA DRR successful-selection cost

**Date:** 2026-10-10
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

Should deficit round-robin (DRR) selection be modeled as free whenever a descriptor is eligible, or should every successful scheduler decision consume visible control cycles?

Both are physically plausible:

- `drr_select_issue_cycles = 0` preserves the compatibility lower bound. It can represent a combinational selector, a deeply pipelined scheduler whose latency is hidden, or an aggregate model that excludes control latency.
- A nonzero fixed cost models a registered eligibility/deficit decision with queue-count-independent visible latency.
- A nonzero `per_channel` cost models a serial full-queue scan before issue. It reuses the existing `drr_round_cost_mode`; the configured cost is multiplied by the number of configured channels.
- A nonzero `per_visited_channel` cost models an iterative scanner that stops at the selected queue. Successful cost is multiplied by the number of probes from the rotating cursor through the selected channel; unsuccessful rounds still visit every configured channel.

This setting is independent of `drr_round_issue_cycles`: unsuccessful rounds add deficit credit and may repeat before any descriptor is eligible, while a successful-selection cost is charged exactly once immediately before each selected descriptor is issued.

## Runtime contract

```yaml
dma:
  arbitration: "deficit_round_robin"
  channels: 8
  drr_select_issue_cycles: 1
  drr_round_cost_mode: "per_visited_channel"
```

`0` is the generated, canonical, zero-initialized, and live-engine default. Values `[0,1024]` are accepted. Unsupported values fail without mutating the accepted live setting. `fixed`, `per_channel`, and `per_visited_channel` are executable scheduler-cost alternatives. For unsuccessful rounds, both serial modes charge all configured channels because no queue was eligible; for successful selections, early-stop mode charges only probes actually visited.

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
- early-stop controls place the same load on channels 0, 3, and 7 with the cursor initially at channel 0;
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
| 1 | `per_visited_channel`, target 0 (1 probe) | 8 | 60 | 60 | 1 cycle (+1.69%) |
| 1 | `per_visited_channel`, target 3 (4 probes) | 8 | 63 | 63 | 4 cycles (+6.78%) |
| 1 | `per_visited_channel`, target 7 (8 probes) | 8 | 67 | 67 | 8 cycles (+13.56%) |

All rows transfer the same 256 useful bytes and produce identical SRAM contents. The setting exposes scheduler-control latency; it does not change payload serialization, traffic, arithmetic, or numerical accuracy.

The same sweep separately retains the unsuccessful-round control: with three unsuccessful rounds at one cycle each, fixed completion is 62 cycles for 1/2/4/8 channels, while `per_channel` completion is 62/65/71/83 cycles. This proves the two costs compose without conflating one successful selection with the preceding credit rounds.

## Multi-objective trade-offs

| Dimension | Zero / hidden | Fixed visible | Full serial scan | Early-stop serial scan |
|---|---|---|---|---|
| Throughput | Optimistic lower bound for short descriptors | One bubble per descriptor | Queue-count-scaled bubble | Placement/occupancy-sensitive bubble; best case fixed-like, worst case full-scan |
| Latency | 59 cycles | 60 cycles at cost 1 | 67 cycles for 8 channels | 60/63/67 cycles for target positions 0/3/7 |
| Area/resources | May imply combinational or hidden-pipeline resources; unquantified | Registered parallel selector | Reused scanner plus index/state | Same narrow scanner plus early-stop/eligible control; physical delta versus full scan unquantified |
| Power/energy | Control energy omitted or hidden | One registered decision | Every configured queue is read | Reads stop at selection; expected lower average activity for early hits, trace-dependent and unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged | Unchanged | Unchanged; probes are scheduler state, not payload traffic |
| Numerical accuracy | Unchanged | Unchanged | Unchanged | Unchanged; movement remains byte exact |
| Control complexity | Simplest abstraction | Explicit issue-stage timing | Deterministic full traversal | Cursor-relative probe counting and data-dependent latency |
| Verification burden | Compatibility/default gate | Exact additive timing | Exact queue-count scaling | First/middle/last positions, cursor wrap, empty queues, unsuccessful rounds |
| Compiler/runtime | No visible scheduler pressure | Coalescing amortizes fixed cost | Fewer configured queues can lower cost | Queue placement and active-set locality affect latency; software may cluster hot streams near the cursor |

No mode is universally preferred. A throughput-oriented mover may spend comparator/read-port resources to hide or bound selection latency. A timing-predictable narrow implementation may deliberately scan the full queue set. An early-stop scanner can reduce average latency and activity when eligible queues are encountered early, but makes service timing depend on cursor position and queue occupancy. The cmodel retains all four timing points rather than selecting the locally fastest row.

## Fidelity limits

- `per_visited_channel` counts logical probes in the existing loop; it does not model individual queue-state read ports or a multi-probe-per-cycle scanner.
- The same cost-mode selector applies to unsuccessful rounds and successful selections. Independent stage widths would need a stronger scheduler microarchitecture contract.
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
