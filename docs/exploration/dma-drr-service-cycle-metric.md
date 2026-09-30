# DMA DRR service-cycle cost metric

**Date:** 2026-09-30
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

Should deficit round-robin (DRR) on a shared, non-preemptive DMA path allocate credit by payload bytes, occupied interface bytes, or modeled descriptor service cycles?

The existing byte modes answer different fairness questions:

- `useful_bytes` allocates semantic tensor payload and is independent of placement and protocol framing.
- `occupied_bytes` allocates modeled interface occupancy after partial-beat rounding and burst-boundary splits.
- `service_cycles` allocates the deterministic mover time predicted from direction-specific base latency, payload serialization, burst issue cost, segmentation, base-latency scope, issue/payload overlap, and burst boundaries.

All three are hardware-plausible. Byte accounting favors smaller counters and stable software semantics. Service accounting is useful when equal-sized descriptors occupy the shared mover for unequal durations, but it couples arbitration to more timing configuration and prediction logic.

## Runtime implementation

`dma.drr_cost_mode` now accepts `useful_bytes`, `occupied_bytes`, and `service_cycles`. `useful_bytes` remains numeric zero and the generated, canonical, zero-initialized, and legacy-API default.

`dma.drr_quantum_cycles` is an independent service-credit quantum in `[1, 1048576]`; zero-initialized runtime callers inherit 64 cycles. The existing power-of-two `dma.drr_quantum_bytes` remains unchanged for both byte modes. Keeping separate quanta avoids assigning cycle units to a field named in bytes.

Service cost reuses the same side-effect-free `descriptor_transfer_cycles()` helper used by queued projected-cycle binding. It includes configured base, issue, payload, segmentation, overlap, and boundary terms. It deliberately excludes stateful SRAM refill stalls, DRAM row/refresh/turnaround behavior, and active elapsed service because those costs are not available as a stable pre-issue descriptor estimate.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Discriminating setup:

- two queues behind one shared-serial, non-preemptive path;
- 64-byte byte quantum and 64-cycle service quantum;
- 256-bit interface, 64-byte maximum burst, zero burst-issue cost;
- descriptor-wide base and packed payload; SRAM bandwidth metering disabled;
- channel 0: two aligned 64-byte loads, each costing `50 + 2 = 52` modeled cycles;
- channel 1: one aligned 64-byte store, costing `100 + 2 = 102` modeled cycles;
- useful and occupied bytes are 64 for every descriptor.

Measured matrix:

| Cost mode | Load #0 completion | Load #1 completion | Store completion | Batch completion | Service order |
|---|---:|---:|---:|---:|---|
| `useful_bytes` | 53 | 207 | 155 | 207 | load #0, store, load #1 |
| `service_cycles` | 53 | 105 | 207 | 207 | load #0, load #1, store |

Service-cycle charging improves the second load completion by 49.28% (`207 → 105`) and delays the longer store by 33.55% (`155 → 207`) in this exact queue state. Batch completion, useful bytes, occupied bytes, and byte-exact data are unchanged. This is latency/fairness redistribution, not throughput or traffic reduction.

The earlier alignment experiment remains gated: useful charging serves aligned/misaligned/aligned at cycles `53/106/158`, while occupied charging serves aligned/aligned/misaligned at `53/105/158` because the misaligned descriptor occupies 96 rather than 64 bytes.

## Multi-objective trade-offs

| Dimension | Useful bytes | Occupied bytes | Service cycles |
|---|---|---|---|
| Throughput | Same finite-batch completion here | Same | Same |
| Latency objective | Payload fairness | Interface-pressure fairness | Modeled mover-time fairness |
| Area/resources | Small payload counter/comparator | Adds pre-issue beat/boundary accounting | Adds cycle estimator inputs, wider add/compare/subtract path; exact area unquantified |
| Power/energy | Lowest expected scheduler activity | Better proxy for switched interface lanes | Better proxy for mover occupancy; estimator/control energy unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged | Unchanged |
| Numerical accuracy | Unchanged | Unchanged | Unchanged |
| Control complexity | Placement-independent | Coupled to framing and addresses | Coupled to direction, latency, issue, segmentation, overlap, and boundaries |
| Verification burden | Payload and queue state | Shared occupancy helper | Shared timing helper plus every timing-mode interaction |
| Compiler/runtime | Stable semantic service class | Placement changes service order | Timing configuration changes service order; software needs the exact cost contract |

No mode is universally preferred. A fixed-function design may hard-wire one. The pre-spec cmodel retains all three so architecture studies can choose whether fairness means payload delivered, interface consumed, or modeled mover time.

## Verification and limits

Verified paths:

```sh
make test-config-generation
make test-config
make test-dma
make test-dma-arbitration-sweep
make clean && make
make test-quick
```

The sweep gates exact order/timestamps, byte-exact loads and stores, directional base-latency discrimination, default compatibility, parse-to-live propagation, and unsupported-value rejection.

The model remains descriptor-boundary and non-preemptive. Virtual deficit rounds cost no cycles. Stateful SRAM stalls, queue-aware DRAM service, command credits, backpressure, deadlines, weighted channels, physical counter width, scheduler critical-path timing, area, power, energy, and calibration remain unmodeled. Service-cycle fairness is therefore a deterministic cmodel policy, not a claim of weighted fair queueing on a calibrated memory system.
