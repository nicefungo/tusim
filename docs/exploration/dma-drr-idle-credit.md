# DMA DRR idle-credit policy

**Date:** 2026-10-02
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

When a deficit-round-robin (DRR) channel drains its queue with unused visit credit, should the shared DMA scheduler discard that residual credit or preserve it for the channel's next burst?

Both contracts are physically plausible:

- `reset` clears deficit when a queue becomes empty. This is the conventional fairness-oriented behavior: an idle channel cannot return with credit earned before its idle interval. It also permits queue-empty teardown to clear scheduler state.
- `retain_residual` keeps only credit already granted while the channel was backlogged. It does **not** accumulate credit during idle time. This can favor intermittent latency-sensitive bursts and avoids an empty-transition register clear, but a returning channel can move ahead of a continuously backlogged peer.

The retained residual is bounded below one weighted quantum when the queue drained after a successful descriptor, so this mode is not an unbounded token bucket. Weight, cost metric, quantum, and interleaved/work-conserving service remain independent settings.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Controls:

- two queues behind one shared-serial, non-preemptive path;
- interleaved useful-byte DRR with a 128-byte quantum and unit weights;
- 256-bit interface, 50-cycle read base, zero visible issue cost;
- SRAM bandwidth metering disabled;
- phase 1 serves one 64-byte descriptor from each channel, leaving 64 bytes of residual credit per visit before the queues become empty;
- phase 2 simultaneously submits a 160-byte channel-0 descriptor and a 64-byte channel-1 descriptor.

Measured matrix:

| Idle policy | Channel 0 second completion | Channel 1 second completion | Batch completion | Phase-2 order |
|---|---:|---:|---:|---|
| `reset` | 213 | 158 | 213 | channel 1, channel 0 |
| `retain_residual` | 161 | 213 | 213 | channel 0, channel 1 |

Retaining residual credit improves the intermittent channel-0 completion by 24.41% (`213 → 161`) and delays the peer by 34.81% (`158 → 213`) in this exact staged workload. Batch completion remains 213 cycles. All 352 bytes are identical, and useful/occupied traffic is unchanged. This is latency allocation across an idle boundary, not throughput improvement.

## Runtime implementation and compatibility

Configuration:

```yaml
dma:
  arbitration: "deficit_round_robin"
  drr_idle_policy: "retain_residual"
```

The setting propagates through YAML generation, the checked-in generated runtime header, shipped JSON, canonical parsing and validation, canonical-to-runtime conversion, top-level initialization, and the live shared-path scheduler. `reset` is numeric zero and remains the generated, canonical, zero-initialized, and legacy-engine default. Unsupported IDs and strings fail without changing an accepted live policy.

The scheduler applies the policy at both empty-queue observation points: after dequeuing a final descriptor and while scanning an empty channel. Continuation state still clears on empty; retention preserves only the per-channel deficit value. Switching a live engine back to `reset` immediately clears residuals on channels with neither queued nor active work; active/backlogged channels retain their current visit accounting until they drain.

## Multi-objective trade-offs

| Dimension | `reset` | `retain_residual` |
|---|---|---|
| Throughput | Same measured batch completion | Same measured batch completion; sustained throughput uncalibrated |
| Latency/fairness | Protects continuously backlogged peers from pre-idle credit | Improves a returning burst when residual plus one quantum admits its head; delays peers in that regime |
| Area/resources | Deficit register plus clear-on-empty control | Same deficit register; removes mandatory empty clear but keeps state live across idle/power-management boundaries; exact area unquantified |
| Power/energy | Empty channels can clear or potentially power-gate scheduler state | Retention state must survive idle; leakage/retention and scheduler activity are unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged; ordering only |
| Numerical accuracy | Unchanged; byte-exact movement | Unchanged |
| Control complexity | Simple lifecycle reset | Must define reset, reconfiguration, and power-state persistence of residual credit |
| Verification burden | Empty transitions and zero state | Adds staged arrivals, retained bounds, reset/re-init, invalid policy atomicity, and interactions with weights/cost modes |
| Compiler/runtime | Stable fairness independent of prior bursts | Software may use channels as persistent traffic classes but must account for history-dependent latency |

A fairness-oriented or aggressively power-gated mover may hard-wire `reset`. A QoS mover with persistent per-channel state may retain residual credit for intermittent classes. The pre-spec cmodel preserves both rather than selecting the lower channel-0 latency row universally.

## Fidelity limits

- Arbitration remains descriptor-boundary and non-preemptive; one long active descriptor blocks every channel.
- Retention does not accrue idle-time credit and is not a deadline, reservation, or guaranteed bandwidth mechanism.
- The cmodel does not model scheduler power gating, context ownership of channel credits, finite command credits, backpressure, producer timing, shared SRAM/DRAM contention, or stateful DRAM service.
- Counter width, reset-network cost, retained-state leakage, scheduler timing, physical area/power/energy, and calibration are unquantified.
- The measured two-phase matrix demonstrates deterministic ordering and completion only; it does not establish sustained weighted fairness.

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
