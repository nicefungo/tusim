# DMA DRR scheduler scan cost

**Date:** 2026-10-08
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

When a deficit round-robin (DRR) descriptor is not yet eligible and the scheduler must complete another credit round, should the modeled control delay be independent of configured queue count or scale with every queue probe?

Both are physically plausible:

- `fixed` charges `drr_round_issue_cycles` once per unsuccessful full round. It represents a parallel eligibility tree, a pipelined scheduler with queue-count-independent visible latency, or an aggregate abstraction whose scan cost is hidden.
- `per_channel` charges the configured delay once for every configured channel in an unsuccessful round. It represents a serial or narrow iterative queue scanner whose latency and activity scale with the number of queue contexts.

The modes do not change DRR credit, service order, descriptor cost, payload timing, or successful-round cost. They refine only the already explicit unsuccessful-round scheduler latency.

## Runtime contract

```yaml
dma:
  arbitration: "deficit_round_robin"
  channels: 8
  drr_quantum_bytes: 64
  drr_round_issue_cycles: 1
  drr_round_cost_mode: "per_channel"
```

`fixed` is numeric zero and remains the YAML, JSON, generated-header, canonical, zero-initialized, and live-engine default. `per_channel` is selectable through the same paths. Unsupported IDs and strings fail without mutating the accepted live mode. A live mode change clears deficits and work-conserving continuation state because accumulated credit cannot be cleanly separated from the old scheduler timing contract.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Controlled workload:

- shared-serial, non-preemptive DMA path;
- useful-byte DRR, exact charging, 64-byte quantum, unit weights;
- one 256-byte linear load on channel 0, requiring three unsuccessful rounds before issue;
- 1 modeled scheduler cycle per unsuccessful round or per configured-channel probe;
- configured channel count swept over 1, 2, 4, and 8; unused queues remain empty but are part of a serial scan;
- 256-bit interface, 50-cycle read base, zero visible burst-issue cost;
- SRAM bandwidth metering disabled;
- byte-exact destination comparison.

Measured matrix:

| Round-cost mode | Configured channels | Descriptor completion | Batch completion | Added vs fixed |
|---|---:|---:|---:|---:|
| `fixed` | 1 | 62 | 62 | 0 |
| `fixed` | 2 | 62 | 62 | 0 |
| `fixed` | 4 | 62 | 62 | 0 |
| `fixed` | 8 | 62 | 62 | 0 |
| `per_channel` | 1 | 62 | 62 | 0 (0.00%) |
| `per_channel` | 2 | 65 | 65 | 3 cycles (+4.84%) |
| `per_channel` | 4 | 71 | 71 | 9 cycles (+14.52%) |
| `per_channel` | 8 | 83 | 83 | 21 cycles (+33.87%) |

For this exact case, fixed scheduling follows `59 + 3 × 1`, while serial scanning follows `59 + 3 × channels × 1`. All rows transfer the same 256 bytes and produce identical SRAM contents. The result exposes queue-count-sensitive control latency; it is not a payload-throughput optimization.

## Multi-objective trade-offs

| Dimension | `fixed` | `per_channel` |
|---|---|---|
| Throughput | Lower deterministic batch time when descriptors need several credit rounds; sustained throughput is uncalibrated | Adds control bubbles proportional to configured queues and unsuccessful rounds; 8-channel measured batch is 33.87% longer |
| Latency | 62 cycles for every measured channel count | 62/65/71/83 cycles for 1/2/4/8 configured channels |
| Area/resources | May imply a parallel comparator/eligibility tree, more read ports, or hidden pipeline resources; size is unquantified | Can represent one reused scanner and narrow control datapath; needs scan index/state but may reduce parallel logic; savings are unquantified |
| Power/energy | Potentially more simultaneous queue-state reads and combinational switching | More clocked control cycles and repeated state reads; net energy direction is technology/workload dependent and unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged; empty queue probes create no payload or protocol traffic |
| Numerical accuracy | Unchanged; movement remains byte exact | Unchanged |
| Control complexity | Parallel reduction or queue-count-independent timing contract | Iterative scan state and queue-count-dependent delay; easier resource sharing but longer control critical path in time |
| Verification burden | Gate constant latency across channel-count changes | Gate exact linear scaling, maximum channel count, saturation, invalid mode rejection, state reset, and one-channel equivalence |
| Compiler/runtime | More queues need not increase scheduler latency under the model | Configuring unused queues can increase latency; runtime should avoid overprovisioning queue contexts when scheduler scans them serially |

A high-throughput mover may spend area on parallel eligibility to decouple queue count from visible latency. A small accelerator may reuse one scanner and accept queue-count-sensitive control bubbles. The cmodel retains both rather than treating the lower-latency implementation as universally best.

## Fidelity limits

- `per_channel` charges every configured channel in each unsuccessful round. It does not skip clock-gated empty queues or stop early after a partial scan.
- The model does not distinguish parallel trees, multi-bank queue-state RAM, two- or four-probe-per-cycle scanners, or pipelined overlap between rounds. Those would require an explicit scheduler-width contract rather than more names for the same two endpoints.
- Successful-round selection has no separate probe latency; the existing setting models only complete unsuccessful rounds.
- Producer injection cannot occur during the aggregate delay inside one `tu_dma_tick()` call.
- Arbitration remains descriptor-boundary and non-preemptive.
- Physical selector area, queue-state SRAM ports, timing, power/energy, finite credits/backpressure, shared-memory contention, and calibration remain unquantified.

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
