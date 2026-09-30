# DMA DRR cost metric: useful versus occupied bytes

**Date:** 2026-09-29
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

When a shared, non-preemptive DMA path uses deficit round-robin (DRR), should each descriptor consume credit according to useful payload bytes or according to occupied interface bytes after beat rounding and burst-boundary splits?

Both are physically plausible:

- `useful_bytes` allocates semantic payload bandwidth. It is independent of placement and protocol framing, keeps scheduler inputs small, and preserves the original DRR contract.
- `occupied_bytes` allocates the modeled interface resource. A descriptor that wastes lanes or crosses a command boundary consumes more credit, which can protect aligned/coalesced traffic from inefficient peers.

Neither metric is universally preferable. Useful-byte fairness is attractive when software should not be penalized for hidden adapter behavior. Occupied-byte fairness is attractive when the arbitration objective is scarce bus occupancy and descriptors carry enough address/protocol information to estimate it before issue.

## Runtime alternatives

`dma.drr_cost_mode` accepts:

- `useful_bytes` — zero/default compatibility mode; charges `descriptor.total_bytes`.
- `occupied_bytes` — charges `descriptor_payload_cycles × dma_bus_width_bytes`, using the same payload/burst-boundary helper as live completion and occupied-byte counters.

The mode affects only `deficit_round_robin`; round-robin and priority policies ignore it. DRR remains descriptor-boundary and non-preemptive. The configured quantum is still expressed in bytes.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Controls:

- two queues behind one shared-serial path;
- 64-byte DRR quantum;
- 256-bit (32-byte) interface;
- 64-byte maximum burst;
- burst-command payload alignment and SRAM-address burst boundaries;
- 50-cycle descriptor base latency, zero issue cost, SRAM bandwidth metering disabled;
- channel 0: two aligned 64-byte descriptors, each occupying 64 bytes;
- channel 1: one 64-byte descriptor at SRAM offset 129, occupying 96 bytes because it crosses a 64-byte boundary and creates a one-byte tail command.

Measured matrix:

| Cost mode | Aligned #0 completion | Aligned #1 completion | Misaligned completion | Batch completion | Service order |
|---|---:|---:|---:|---:|---|
| `useful_bytes` | 53 | 158 | 106 | 158 | aligned #0, misaligned, aligned #1 |
| `occupied_bytes` | 53 | 105 | 158 | 158 | aligned #0, aligned #1, misaligned |

Occupied charging improves the second aligned completion by 33.54% in this exact queue state, but delays the misaligned descriptor by 49.06%. Batch completion, 192 useful bytes, functional data movement, and numerical behavior are unchanged. The result is a latency/fairness reallocation, not a throughput gain.

## Multi-objective trade-offs

| Dimension | Useful-byte charging | Occupied-byte charging |
|---|---|---|
| Throughput | Same measured finite-batch completion | Same measured finite-batch completion |
| Latency | Does not penalize placement/protocol amplification | Protects efficient traffic; can delay misaligned or fragmented descriptors |
| Fairness objective | Semantic payload fairness | Modeled interface-occupancy fairness |
| Area/resources | Descriptor byte count already exists | Needs pre-issue occupied-byte calculation from bus width, segmentation, and boundary state; exact cost unquantified |
| Power/energy | Scheduler activity depends only on payload size | Expected to correlate better with switched interface lanes, but physical energy is not modeled |
| SRAM/DRAM traffic | Useful and occupied counters remain unchanged | Same traffic; only arbitration order changes |
| Numerical accuracy | Unchanged | Unchanged |
| Control complexity | Smaller, placement-independent contract | More coupling to address metadata and payload/burst configuration; wider verification surface |
| Verification burden | Payload sizes and queue state | Must prove one shared occupied-byte helper across planning, arbitration, execution, and counters |
| Compiler/runtime | Stable service class regardless of placement | Placement, alignment, and descriptor segmentation can change service order; software may exploit or accidentally trigger it |

## Implementation and compatibility

The full executable path is:

1. `config/tu_config.{yaml,json}` — shipped `drr_cost_mode` default.
2. `scripts/gen_config.py` and generated `tu_cmodel/tu_config.h` — macros, runtime field, default initializer.
3. `tu_cmodel/infra/config.{h,c}` — canonical enum/field, parser, validation, runtime conversion, and generated documentation.
4. `tu_cmodel/tu_cmodel.c` — runtime-to-engine propagation.
5. `tu_cmodel/dma_descriptor.{h,c}` — fail-closed engine selection and shared occupied-byte charging helper.
6. `tests/test_generated_config.py`, `tests/test_config.c`, `tests/test_dma.c`, and `tests/test_dma_arbitration_sweep.c` — generated alternative, parse-to-live propagation, invalid rejection, compatibility, exact order/cycles, occupied traffic, and byte movement.

`useful_bytes` is numeric zero and remains the generated, canonical, zero-initialized, and legacy-initializer behavior. A new additive initializer exposes the cost mode without changing older public signatures. Unsupported IDs fail closed before channels are created.

## Fidelity limits

- Occupied bytes are the cmodel's deterministic command/beat accounting, not measured AXI/DRAM traffic.
- Virtual deficit rounds consume no cycles. Comparator, divider/shift, boundary-evaluation timing, counter area, scheduler power, and energy are unquantified.
- The `service_cycles` follow-up in `dma-drr-service-cycle-metric.md` now includes deterministic base, command-issue, and payload timing as a third runtime cost contract. Stateful SRAM stalls, DRAM row/refresh/turnaround service, and active remaining cycles remain outside that pre-issue estimate.
- Arbitration remains non-preemptive, so neither metric can interrupt a long active descriptor.
- Finite command credits, backpressure, queue-aware DRAM service, per-channel weights, deadlines, and calibrated physical throughput remain unmodeled.

## Verification

```sh
make test-config-generation
make test-config
make test-dma
make test-dma-arbitration-sweep
make clean && make
make test-quick
```
