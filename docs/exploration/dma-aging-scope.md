# DMA Priority-Aging Scope: Submission Time vs Queue-Head Eligibility

**Date:** 2026-09-22
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

When grant-epoch aging is used on a shared, non-preemptive DMA path, should a descriptor accumulate age from accepted submission, including time behind older work in its own channel, or only after it becomes the selectable queue head?

The distinction matters for queues deeper than one. The previous model described aging as “missed shared-bus grants,” but its submission timestamp also counted grants that a buried descriptor was not eligible to win. That behavior is a plausible end-to-end waiting-time policy, but it is not the only plausible fairness contract and should not be mislabeled as head-of-line grant aging.

## Runtime alternatives

| `dma.aging_scope` | Hardware interpretation | Why a team might choose it | Main sacrifice |
|---|---|---|---|
| `submission` (zero/default) | Every accepted descriptor stores its enqueue grant epoch; age includes waiting behind its own channel | Protects end-to-end request age and can keep deep producer queues from being penalized twice | A newly exposed deep descriptor can arrive at the arbiter with accumulated priority and overtake a fresh peer immediately |
| `queue_head` | A descriptor's age epoch is reset when it becomes its channel's selectable head | Matches “missed grants while eligible,” isolates inter-channel fairness from private queue depth, and gives fresh peer heads an equal tie opportunity | Buried descriptors receive no fairness credit until promotion, increasing their end-to-end latency under deep same-channel queues |

Both modes use the existing saturating rule:

```text
age_grants = current_grant_epoch - selected_start_epoch
effective_priority = min(255, base_priority + age_grants)
```

The selected start epoch is submission or queue-head eligibility. Arbitration remains descriptor-boundary and non-preemptive. Ties use the rotating cursor. `round_robin`, `strict_priority`, and independent-path operation are unchanged.

## Executable sweep

Command:

```sh
make test-dma-arbitration-sweep
```

Discriminating workload:

1. Channel 0 receives a priority-2 64-byte leader and then a priority-0 descriptor behind it.
2. The leader wins the first shared-path grant.
3. A fresh priority-0 64-byte descriptor is then submitted to channel 1.
4. Each transfer costs 52 cycles: 50 base plus two 32-byte payload beats; SRAM metering is disabled.

| Aging scope | Leader completion | Deep channel-0 completion | Fresh channel-1 completion | Service order after leader |
|---|---:|---:|---:|---|
| `submission` | 53 | 105 | 157 | deep, fresh |
| `queue_head` | 53 | 157 | 105 | fresh, deep |

Submission aging gives the deep descriptor one grant of accumulated age, so it has effective priority 1 when promoted and beats the fresh priority-0 peer. Queue-head aging resets the promoted descriptor to age zero; the equal-priority tie follows the rotating cursor to channel 1. The batch still completes at tick 157, and all 192 useful bytes are byte-identical. This is a per-request latency and fairness trade-off, not a throughput gain.

The original arbitration matrices remain unchanged: simultaneous 4 KiB traffic completes at 6,487 ticks under round-robin, strict, and aging policies, while the sustained priority-2 stream still demonstrates aging's starvation mitigation. The new scope only discriminates descriptors that waited below a queue head.

## Multi-objective interpretation

| Dimension | Submission aging | Queue-head aging |
|---|---|---|
| Throughput | Same measured batch completion | Same measured batch completion; no throughput claim |
| Latency | Lower for old descriptors from deep producer queues in the measured case | Lower for a fresh peer head in the measured case |
| Fairness contract | End-to-end accepted-wait fairness | Eligible-head / inter-channel fairness |
| Area/resources | Per-descriptor submission epoch already required | Per-descriptor eligibility epoch or per-channel head timestamp plus promotion update |
| Power/energy | Timestamp and subtract/add/saturate switching; magnitude unquantified | Similar logic plus promotion-state update; magnitude unquantified |
| SRAM/DRAM traffic | Unchanged, byte-identical | Unchanged, byte-identical |
| Numerical accuracy | Unchanged | Unchanged |
| Control complexity | Simpler timestamp lifecycle | Must define promotion timing precisely, including same-epoch ties and reset |
| Verification burden | Deep queues, dynamic arrivals, saturation, reset | Same plus head promotion and no hidden age below the head |
| Compiler/runtime | Better when accepted-request age carries QoS meaning | Better when channels are independent producer classes and only eligible heads should compete |

Physical selector timing, counter width, area, control energy, queue-RAM ports, sustained producer throughput, shared SRAM/DRAM contention, and end-to-end compute overlap are **unquantified**. Neither mode is universally preferable.

## Implementation and compatibility

The full executable path is wired:

1. `config/tu_config.yaml` and `config/tu_config.json`: `dma.aging_scope` accepts `submission` or `queue_head`.
2. `scripts/gen_config.py` and generated `tu_cmodel/tu_config.h`: constants, runtime field, and default initializer.
3. `tu_cmodel/infra/config.{h,c}`: canonical enum, parser, validation, runtime conversion, and generated config documentation.
4. `tu_cmodel/tu_cmodel.c`: forwards the runtime scope to the descriptor engine.
5. `tu_cmodel/dma_descriptor.{h,c}`: records submission and eligibility epochs and updates eligibility on head promotion.
6. `tests/test_generated_config.py`, `tests/test_config.c`, `tests/test_dma.c`, and `tests/test_dma_arbitration_sweep.c`: generated alternative, parse-to-live propagation, unsupported-value rejection, exact order/timestamps, and byte movement.

`submission` is numeric zero and remains the default for generated, canonical, zero-initialized, and legacy initializer callers. Existing public initializers retain their signatures and submission-aging behavior; the additive `tu_dma_init_config_boundary_aging()` API exposes the new scope. Unsupported scope IDs fail closed without creating channels.

## Fidelity limits

- Epochs count granted descriptors, not elapsed cycles; neither mode provides a wall-clock deadline.
- Selection is non-preemptive. Descriptor length can dominate wall-clock waiting.
- The queue stores a 64-bit model epoch; physical counter width and wrap-safe arithmetic are not selected.
- Priorities remain caller-provided metadata; compiler/runtime classification is not automated.
- The model does not implement weighted/deficit service, earliest-deadline-first, finite command FIFOs, backpressure, or queue-aware DRAM service.
- Completion ticks are deterministic cmodel service values, not calibrated AXI/DRAM measurements.

## Verification

```sh
make test-config-generation test-config test-dma test-dma-arbitration-sweep
make config-docs
make clean && make
make test-quick
```
