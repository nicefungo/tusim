# DMA Deficit Round-Robin: Descriptor Fairness vs Byte Fairness

**Date:** 2026-09-28; occupied-byte follow-up 2026-09-29; service-cycle follow-up 2026-09-30; weighted-service follow-up 2026-10-01
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

Should a shared, non-preemptive DMA path rotate once per descriptor, or allocate service using byte credits when channels carry unequal descriptor sizes?

Plain round-robin is a small and predictable descriptor scheduler, but one 512-byte descriptor receives the same turn as one 64-byte descriptor. Deficit round-robin (DRR) keeps a byte deficit per channel, adds a configurable quantum while visiting a backlogged channel, and admits its head only when the deficit covers the descriptor's useful bytes. Both are physically plausible. Strict and aging priority remain separate alternatives for explicit QoS classes.

## Runtime alternatives

- `round_robin`: zero/default compatibility; one descriptor per rotating turn.
- `strict_priority`: highest descriptor priority, rotating ties.
- `aging_priority`: effective priority rises with configured wait.
- `deficit_round_robin`: configurable byte- or service-cycle-credit scheduling.

`drr_quantum_bytes` is a power of two in `[16,65536]`; the default is 256 bytes. A zero field from a legacy zero-initialized caller inherits 256 bytes. Old initialization APIs also inherit 256 bytes, and existing policies ignore it. `dma.drr_cost_mode` independently selects `useful_bytes` (zero/default compatibility), `occupied_bytes`, or `service_cycles`. Byte modes use `drr_quantum_bytes`; service mode uses the separate `drr_quantum_cycles` so units are not overloaded. All modes use saturating credit arithmetic and reset credit when a queue empties. Service remains descriptor-boundary and non-preemptive. See [`dma-drr-cost-metric.md`](dma-drr-cost-metric.md) and [`dma-drr-service-cycle-metric.md`](dma-drr-service-cycle-metric.md).

`dma.drr_service_mode` independently preserves `interleaved` one-descriptor rotation or enables `work_conserving` residual-credit service. `dma.drr_channel_weights` supplies eight `[1,255]` quantum multipliers; unit weights are default, and legacy all-zero runtime arrays inherit unit weights. Weighted short-window results and costs are in [`dma-weighted-drr-service.md`](dma-weighted-drr-service.md).

## Executable experiment

```sh
make test-dma-arbitration-sweep
```

Two backlogged channels share one serialized path. Channel 0 has four 64-byte descriptors; channel 1 has one 512-byte descriptor. SRAM bandwidth metering is disabled. A 64-byte descriptor costs 52 service cycles (50 base + two 32-byte payload beats); the 512-byte descriptor costs 66 cycles. The first issue tick makes exact completion times one cycle larger.

| DRR quantum (B) | Small completions (cycles) | Large completion | Batch completion | Allocation behavior |
|---:|---|---:|---:|---|
| 64 | 53, 105, 157, 209 | 275 | 275 | Fine byte fairness; the large descriptor waits eight credit visits |
| 256 | 53, 105, 223, 275 | 171 | 275 | Middle point; large work runs after two small descriptors |
| 1024 | 53, 171, 223, 275 | 119 | 275 | Coarse credit admits the large descriptor on its first visit, matching descriptor RR order here |

All rows transfer the same 768 useful bytes and produce byte-identical SRAM contents. Increasing the quantum from 64 to 1024 bytes reduces the large descriptor's completion by 56.73% (275 to 119 cycles) while delaying the second small descriptor by 62.86% (105 to 171 cycles). Batch completion is unchanged. This is latency and fairness allocation, not a throughput or traffic reduction.

## Multi-objective trade-offs

| Dimension | Round-robin | DRR, small quantum | DRR, large quantum |
|---|---|---|---|
| Throughput | Same measured 275-cycle finite batch | Same measured batch | Same measured batch |
| Latency | Predictable descriptor alternation; insensitive to size | Protects streams of small transfers; large heads wait for credit | Reduces large-head wait; approaches descriptor RR when quantum exceeds heads |
| Fairness | Equal descriptor opportunities, unequal byte service | Finer byte-service granularity | Coarser byte-service granularity |
| Area/resources | Cursor and queue-ready logic | Per-channel deficit counters plus saturating add/compare/subtract | Same state width; potentially fewer virtual credit rounds |
| Power/energy | Lowest expected scheduler activity | More counter/comparator activity; unquantified | Less repeated credit accounting for large heads; unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged | Unchanged |
| Numerical accuracy | Unchanged; byte-exact movement | Unchanged | Unchanged |
| Control complexity | Lowest | Credit lifecycle, empty-queue reset, large-head progression | Same mechanisms plus coarse latency granularity |
| Verification burden | Cursor, empty queues, ties | Adds quantum limits, overflow, reset, descriptors larger than quantum, mixed sizes | Same |
| Compiler/runtime | No size tuning | Quantum can favor fine control/gather traffic | Quantum can reduce delay for tensor-scale bursts |

No policy is universally best. A simple fixed-function mover may choose round-robin. A shared mover carrying control transfers and tensor bursts can use DRR to make byte allocation explicit. Priority policies remain appropriate when semantic criticality matters more than byte fairness.

## Implementation/configuration path

1. `config/tu_config.{yaml,json}`: arbitration name and DRR quantum.
2. `scripts/gen_config.py` and `tu_cmodel/tu_config.h`: constants, runtime field, default initializer.
3. `tu_cmodel/infra/config.{h,c}`: canonical enum/field, parser, validation, runtime conversion, generated docs.
4. `tu_cmodel/tu_cmodel.c`: top-level propagation.
5. `tu_cmodel/dma_descriptor.{h,c}`: per-channel deficit state and shared-path selection.
6. `tests/test_generated_config.py`, `tests/test_config.c`, `tests/test_dma.c`, and `tests/test_dma_arbitration_sweep.c`: generation, parsing, rejection, compatibility, exact order/cycles, and byte movement.

## Fidelity limits

- Useful bytes, occupied bytes, and modeled service cycles are executable alternatives. Occupied charging reuses live payload/boundary accounting. Service charging reuses deterministic base+issue+payload timing, but excludes stateful SRAM stalls, DRAM service, and active elapsed work.
- DRR is non-preemptive; a selected long descriptor blocks all channels until retirement.
- Virtual credit rounds consume no modeled cycles. Counter/comparator timing and arbitration energy are unquantified.
- Per-channel weights and residual-credit continuation are executable. Finite command credits, backpressure, shared SRAM/DRAM contention, deadlines, and compiler-assigned service-class policy remain unmodeled.
- The finite matrix proves deterministic ordering and latency allocation, not sustained queue-aware bandwidth guarantees or calibrated hardware timing.

## Verification

```sh
make test-config-generation test-config test-dma test-dma-arbitration-sweep
make config-docs
make clean && make
make test-quick
```
