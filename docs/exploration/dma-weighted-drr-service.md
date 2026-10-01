# DMA weighted DRR service discipline

**Date:** 2026-10-01
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

When deficit round-robin (DRR) assigns unequal channel shares, should the shared non-preemptive DMA path rotate after every eligible descriptor, or let one channel spend its remaining deficit before rotating?

Both choices are physically plausible:

- `interleaved` rotates after one descriptor. It bounds consecutive service, preserves the historical cmodel behavior, and is attractive when latency isolation matters more than realizing a weighted share over a short window.
- `work_conserving` keeps the current DRR visit while the next head fits the residual deficit. Combined with per-channel quantum weights, it realizes coarse weighted service without inventing descriptor preemption.

The model exposes eight channel weights in `[1,255]`. A channel receives `base_quantum × weight` credit at the start of a visit. Unit weights preserve unweighted DRR. The byte or cycle cost contract remains independently selectable through `drr_cost_mode`.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Controls:

- two continuously backlogged queues behind one shared-serial path;
- four 64-byte descriptors per channel;
- useful-byte cost and a 64-byte base quantum;
- channel weights `[1,2]`;
- 256-bit interface, 50-cycle read base, no visible issue cost;
- SRAM bandwidth metering disabled;
- every descriptor costs 52 modeled service cycles and completes one tick after issue starts.

Measured matrix:

| Service mode | Channel 0 completions (weight 1) | Channel 1 completions (weight 2) | Batch completion | Early six-descriptor allocation |
|---|---|---|---:|---|
| `interleaved` | 53, 157, 261, 365 | 105, 209, 313, 417 | 417 | 3:3 descriptors |
| `work_conserving` | 53, 209, 365, 417 | 105, 157, 261, 313 | 417 | 2:4 descriptors |

Work-conserving service improves channel 1's second descriptor completion by 24.88% (`209 → 157`) and fourth completion by 24.94% (`417 → 313`) in this exact finite queue. It delays channel 0's second completion by 33.12% (`157 → 209`) and final completion by 14.25% (`365 → 417`). Batch completion remains 417 cycles, all 512 bytes are identical, and useful/occupied traffic is unchanged. This is weighted latency allocation, not throughput improvement.

## Runtime implementation and compatibility

Configuration:

```yaml
dma:
  arbitration: "deficit_round_robin"
  drr_service_mode: "work_conserving"
  drr_channel_weights: [1, 2, 1, 1, 1, 1, 1, 1]
```

The full path is executable through YAML/generator, shipped JSON, canonical parsing and validation, canonical-to-runtime conversion, top-level DMA initialization, and the shared-path scheduler. `interleaved` plus eight unit weights remains the generated and canonical default. A legacy all-zero runtime weight array inherits unit weights. Mixed zero/nonzero arrays, unsupported modes, wrong array lengths, and out-of-range generated values fail without mutating an accepted engine policy.

In `work_conserving` mode, residual credit can authorize consecutive descriptors from one channel. No extra quantum is granted while that same visit continues. Empty queues clear both deficit and continuation state. All additions and weight multiplication saturate at `UINT64_MAX`.

## Multi-objective trade-offs

| Dimension | Interleaved | Work-conserving weighted DRR |
|---|---|---|
| Throughput | Same measured finite-batch completion | Same measured finite-batch completion; sustained throughput is uncalibrated |
| Latency | Bounds consecutive descriptors and protects peer latency | Improves weighted-channel completion while delaying lower-weight peers |
| Area/resources | Cursor, per-channel deficit, one global quantum | Adds eight weight registers, multiplier/shift-or-add equivalent, continuation channel/state; exact area unquantified |
| Power/energy | Lower expected scheduler state activity | More credit arithmetic and potentially burstier channel activity; physical energy unquantified |
| SRAM/DRAM traffic | Unchanged in the measured matrix | Unchanged; policy changes order only |
| Numerical accuracy | Unchanged; byte-exact movement | Unchanged |
| Control complexity | Mandatory rotation after each descriptor | Must distinguish a new visit from residual-credit continuation and clear state on drain |
| Verification burden | Order, deficit, empty reset | Adds weighted multiplication, saturation, consecutive service, defaults, parser length/range, and invalid-setter atomicity |
| Compiler/runtime | Simple uniform latency expectation | Software assigns channel weights and must understand that ratios are coarse and descriptor-size/cost dependent |

A low-cost or latency-isolating mover may hard-wire interleaving. A QoS-oriented shared mover may choose work-conserving weighted DRR when channels represent stable traffic classes. The pre-spec cmodel retains both rather than selecting the locally faster channel outcome.

## Fidelity limits

- Arbitration remains descriptor-boundary and non-preemptive; one long active descriptor still blocks every channel.
- Weight is a quantum multiplier, not a guaranteed deadline, exact short-window bandwidth ratio, or admission reservation.
- Finite credits, backpressure, queue producer timing, shared SRAM/DRAM contention, stateful DRAM service, and channel affinity are not modeled.
- Counter width, multiplier implementation, scheduler critical-path timing, area, power, energy, and physical calibration are unquantified.
- The finite matrix demonstrates deterministic service allocation only. Sustained weighted fairness needs longer arrival traces and calibrated shared-resource service.

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
