# DMA Priority Aging: Missed Grants vs Wait Cycles

**Date:** 2026-09-24
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

Should a non-preemptive shared DMA path age queued work by arbitration opportunities or by elapsed wait time?

The existing policy advanced age once per granted descriptor. That is compact and deterministic, but a 64-byte descriptor and a 4 KiB descriptor both count as one aging step. A controller with a cycle counter can instead promote work according to wall-clock wait. Both are physically plausible and neither is universally preferable.

## Runtime alternatives

`dma.aging_metric` selects:

- `grants` (zero/default compatibility): one age step per shared-bus descriptor grant.
- `cycles`: one age step per `dma.aging_cycle_quantum` elapsed model cycles.

Both metrics use the existing `dma.aging_scope` (`submission` or `queue_head`) and `dma.aging_increment`:

```text
steps(grants) = current_grant_epoch - start_grant_epoch
steps(cycles) = floor((current_cycle - start_cycle) / aging_cycle_quantum)
effective_priority = min(255, base_priority + steps * aging_increment)
```

`aging_cycle_quantum` is configurable in `[1, 1048576]`. The representative sweep uses 64 cycles so a long transfer creates a discriminating time-age result without immediate saturation.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Workload:

1. An old priority-0 64-byte descriptor waits on channel 0.
2. A priority-2 4096-byte descriptor starts on channel 1 and occupies the non-preemptive shared path for 178 cycles (50-cycle base plus 128 payload cycles on a 32-byte interface).
3. At cycle 178, immediately before retirement, a fresh priority-2 64-byte descriptor is queued on channel 1.
4. SRAM bandwidth metering is disabled. Aging uses submission scope, increment 1, and a 64-cycle quantum for cycle mode.

| Metric | Old priority-0 completion | Fresh priority-2 completion | Batch completion | Selected second |
|---|---:|---:|---:|---|
| Missed grants | 283 | 231 | 283 | Fresh high-priority descriptor |
| Wait cycles, quantum 64 | 231 | 283 | 283 | Old low-priority descriptor |

Cycle aging reduces the old request's completion from 283 to 231 cycles (18.37%) in this long-descriptor regime, while delaying the fresh high-priority request from 231 to 283 cycles (22.51%). Batch completion, useful bytes, occupied bytes, and byte-exact SRAM contents are unchanged. This is latency allocation and fairness behavior, not a throughput gain.

## Why hardware teams may choose either mode

| Dimension | Missed-grant aging | Wait-cycle aging |
|---|---|---|
| Throughput | Same measured batch completion | Same measured batch completion |
| Per-request latency | Descriptor-length-independent promotion in grants; long transfers can create long wall-clock waits | Bounds promotion in quantized elapsed cycles after the current non-preemptive transfer retires |
| Fairness | Fairness expressed in service opportunities | Fairness tracks wall-clock waiting more directly |
| Area/resources | Grant epoch plus per-descriptor start epoch; narrower policy concept | Requires cycle timestamps, subtraction, quantum division/comparison, and wider state; exact area unquantified |
| Power/energy | Updates primarily at descriptor grants | Cycle timestamp exists globally, but time-based compare arithmetic may switch at each arbitration; magnitude unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged |
| Numerical accuracy | Unchanged | Unchanged |
| Control complexity | Simpler and insensitive to clock-rate interpretation | Needs a cycle-domain contract and quantum selection; dynamic clock changes require policy decisions |
| Verification burden | Grant order, ties, scope, saturation, epoch wrap | Same plus boundary cycles, quantum floor behavior, clock-domain changes, timestamp wrap |
| Compiler/runtime | Priority decay depends on queue service count | QoS software must tune a cycle quantum against descriptor latency and clock frequency |

A cost-constrained fixed-function mover may choose missed grants. A QoS-oriented system with existing timestamps may choose wait cycles to avoid descriptor-size-dependent starvation. The pre-spec cmodel preserves both.

## Implementation and compatibility

The executable path includes:

1. `config/tu_config.yaml` and `config/tu_config.json`: `aging_metric` and `aging_cycle_quantum`.
2. `scripts/gen_config.py` and `tu_cmodel/tu_config.h`: generated constants, runtime fields, and defaults.
3. `tu_cmodel/infra/config.{h,c}`: canonical enums/fields, parser, validation, runtime conversion, and generated docs.
4. `tu_cmodel/tu_cmodel.c`: top-level runtime propagation.
5. `tu_cmodel/dma_descriptor.{h,c}`: grant/cycle timestamps and saturating effective-priority calculation.
6. `tests/test_generated_config.py`, `tests/test_config.c`, and `tests/test_dma_arbitration_sweep.c`: generated alternative, parser/runtime/live propagation, invalid rejection, exact order/cycles, and byte movement.

`grants` remains enum value zero and the generated/canonical default. Existing initializer APIs continue to select grant aging with a one-cycle placeholder quantum. Zero-initialized runtime fields map to grant aging, increment 1, and quantum 1. Unsupported metric IDs and canonical quantum values outside `[1,1048576]` fail closed.

## Fidelity limits

- Both modes remain descriptor-boundary and non-preemptive. Cycle aging cannot interrupt the long transfer that caused the wait.
- Cycle mode uses TU model cycles, not physical nanoseconds. It is not invariant under clock changes and is not a deadline scheduler.
- Submission scope can credit time spent behind same-channel work; queue-head scope starts the cycle timestamp when a descriptor becomes eligible.
- The model uses 64-bit timestamps and 8-bit priorities. Physical counter widths, wrap handling, comparator timing, and register interfaces are unselected.
- Quantum division is integer floor division in the cmodel. A physical implementation may use a power-of-two shift, down-counter, or periodic tick; area/timing/power are unquantified.
- No preemption, weighted/deficit service, finite command credits, shared-memory contention, compiler priority assignment, or calibrated AXI/DRAM timing is modeled.

## Verification

```sh
make test-config-generation test-config test-dma test-dma-arbitration-sweep
make config-docs
make clean && make
make test-quick
```
