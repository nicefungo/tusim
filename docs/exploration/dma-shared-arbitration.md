# Shared-Serial DMA Arbitration: Fair, Strict, and Aging Priority

**Date:** 2026-08-25; grant-aging follow-up 2026-09-21
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

When several descriptor queues share one non-preemptive DMA path, should the mover rotate fairly, protect critical traffic strictly, or retain priorities while bounding starvation through age?

## Realistic alternatives

| Runtime `arbitration` | Hardware rationale | Expected benefit | Expected sacrifice |
|---|---|---|---|
| `round_robin` (zero/default) | Small rotating selector for general-purpose W/A/O streams | Every continuously nonempty channel receives a service opportunity; no software priority assignment | Critical work can wait behind lower-value descriptors |
| `strict_priority` | Comparator/select tree over caller-supplied 8-bit priorities | Protects deadline-sensitive loads/stores when priorities are trustworthy | Low-priority starvation is possible; software owns priority correctness |
| `aging_priority` | Same priority selector plus submission/grant epochs and saturating addition | Retains short-term priority preference while old queue heads gain one effective level per missed grant | Weakens strict critical isolation; adds counters, add/saturate logic, state width, and dynamic-arrival verification |

Selection occurs only when the shared path is idle and only at descriptor boundaries. Active work is never preempted. Ties use the rotating round-robin cursor. Independent DMA paths do not consume the policy.

Aging is deliberately measured in **missed shared-bus grants**, not elapsed core cycles:

```text
age_grants        = current_grant_epoch - submission_grant_epoch
effective_priority = min(255, descriptor_priority + age_grants)
```

The grant epoch increments after each shared-path descriptor selection. This makes the fairness trade-off independent of descriptor duration: one long transfer does not produce a larger boost than one short transfer. It does not provide a wall-clock deadline.

## Executable workload matrix

Command:

```sh
make test-dma-arbitration-sweep
```

### Simultaneous finite batch

Three simultaneous 4,096-byte host-to-SRAM descriptors use channels 0/1/2 and priorities 0/10/5. The shared serial path is 32 B/model-cycle with a 50-cycle base read latency and the checked-in SRAM grant model. Each transfer has 2,162 modeled service cycles, including 1,984 SRAM-stall cycles; completion includes the initial issue tick.

| Policy | Low ch0 completion | Critical ch1 completion | Medium ch2 completion | Batch completion |
|---|---:|---:|---:|---:|
| `round_robin` | 2,163 | 4,325 | 6,487 | 6,487 |
| `strict_priority` | 6,487 | 2,163 | 4,325 | 6,487 |
| `aging_priority` | 6,487 | 2,163 | 4,325 | 6,487 |

Strict and aging priority are intentionally identical in this finite high-separation batch: the lower priorities do not wait for enough grants to close gaps of five and ten. Both lower critical completion from 4,325 to 2,163 ticks (**50.0%**) without changing the 6,487-tick batch completion or 12,288 useful bytes. The low-priority descriptor moves from first to last and takes **3.0x** its round-robin completion latency.

### Sustained fresh high-priority arrivals

The discriminating workload queues one priority-0 descriptor on channel 0 and repeatedly adds fresh priority-2 64-byte descriptors to channel 1. SRAM metering is disabled; every descriptor consumes 52 service cycles (50 base + two 32-byte payload beats).

| Policy | Low completion | High 0 | High 1 | High 2 | Service order |
|---|---:|---:|---:|---:|---|
| `strict_priority` | 209 | 53 | 105 | 157 | H0, H1, H2, L |
| `aging_priority` | 157 | 53 | 105 | 209 | H0, H1, L, H2 |

After two missed grants, the old low descriptor reaches effective priority 2 and wins the tie through the rotating cursor. Aging lowers its measured completion from 209 to 157 ticks (**24.9%**) but delays the third fresh high descriptor from 157 to 209 ticks (**33.1%**). Batch completion and useful bytes remain unchanged. This is a latency/fairness exchange, not a throughput optimization.

## Multi-objective interpretation

| Dimension | Round-robin | Strict priority | Aging priority |
|---|---|---|---|
| Throughput | Same measured batches | Same measured batches | Same measured batches; no throughput gain claimed |
| Latency | Predictable service opportunities, but critical work waits | Best critical completion in the measured priority batches; low work can starve | Bounded priority-gap catch-up in grants; critical latency can increase once old work ties |
| Area/resources | Rotating cursor and empty-queue scan | Priority comparators/select tree and descriptor priority bits | Strict logic plus grant epoch, per-descriptor submission epoch, subtract/add/saturate path |
| Power/energy | Expected lowest selector activity | Extra comparisons and priority metadata | Extra counter and arithmetic switching; all magnitudes unquantified |
| SRAM/DRAM traffic | Byte-identical | Byte-identical | Byte-identical; only order changes |
| Numerical accuracy | Unchanged, byte-exact | Unchanged, byte-exact | Unchanged, byte-exact |
| Control complexity | Lowest | Priority assignment and starvation policy | Defines aging domain, saturation, dynamic arrival, tie, wrap, and reset behavior |
| Verification burden | Rotation/wrap/empty queues | Priority order/ties/starvation | All strict gates plus submission timing, fresh arrivals, saturation, and bounded service |
| Compiler/runtime | No annotations required | Must assign meaningful priority | Same metadata; aging can tolerate occasional bad/static assignments but is not a deadline guarantee |

Physical area, timing closure, control energy, sustained producer throughput, queue RAM ports, shared SRAM/DRAM contention, and end-to-end compute overlap are **unquantified**. The current cmodel cannot justify one universal policy.

## Implementation and compatibility

The executable path is:

1. `config/tu_config.yaml` and JSON parsing: `dma.arbitration` accepts `round_robin`, `strict_priority`, or `aging_priority`.
2. `scripts/gen_config.py` and `tu_cmodel/tu_config.h`: generated constants/default runtime field; round-robin remains numeric zero/default.
3. `tu_cmodel/infra/config.{h,c}`: canonical enum, parse, validation, runtime propagation, and generated config documentation.
4. `tu_cmodel/tu_cmodel.c`: forwards the selected policy to the live descriptor engine.
5. `tu_cmodel/dma_descriptor.{h,c}`: descriptors record their accepted submission epoch; shared selection computes saturating effective priority and advances the epoch after a grant.
6. `tests/test_dma.c`, `tests/test_config.c`, `tests/test_generated_config.py`, and `tests/test_dma_arbitration_sweep.c`: defaults, generated alternative, parse-to-live propagation, invalid rejection, exact order/timestamps, dynamic arrivals, and byte movement.

No public constructor signature changed. Existing and zero-initialized callers retain round-robin. Strict priority retains its previous behavior. Unsupported policy IDs fail closed without creating channels.

## Fidelity limits and deferred variants

- Grant aging bounds service relative to a finite 8-bit priority gap under continuously eligible queue heads; it is not an absolute cycle deadline. Descriptor lengths can make wall-clock waiting arbitrarily large.
- Arbitration remains non-preemptive. A new critical descriptor cannot interrupt active work. Beat-level preemption needs progress, replay, ordering, and physical burst contracts.
- Epoch arithmetic uses a 64-bit monotonic model counter; practical wrap is not modeled as a hardware-width design. A physical implementation must choose counter width and wrap-safe comparison.
- Priorities are caller metadata. Compiler/command-queue producers do not automatically classify traffic.
- Weighted round-robin, deficit service, earliest deadline first, and configurable aging rates remain excluded until traces or QoS contracts distinguish them. Adding policy names without a discriminating workload would create mode proliferation.
- Completion ticks use the coarse cmodel SRAM refill domain and are not calibrated AXI/DRAM timings.

## Verification

```sh
make test-config-generation test-config test-dma test-dma-arbitration-sweep
make config-docs
make clean && make
make test-quick
```
