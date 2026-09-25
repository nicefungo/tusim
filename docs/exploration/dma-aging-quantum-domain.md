# DMA Aging Quantum Domain: Core Cycles vs Physical Nanoseconds

**Date:** 2026-09-25

**Mode:** pre-spec exploration

**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

When wait-cycle priority aging is used on a non-preemptive shared DMA path, should its quantum be fixed in TU/core cycles or in physical time?

A synchronous, fixed-frequency controller can use a cycle down-counter with minimal conversion logic. A controller whose fairness policy must retain the same physical meaning across candidate clocks can define the quantum in nanoseconds and convert it with:

```text
effective_quantum_cycles = ceil(aging_quantum_ns * core_clock_ghz)
```

Neither contract is universally preferable. A physical implementation would normally hard-wire one domain, while the pre-spec cmodel keeps both runtime-configurable.

## Runtime alternatives

- `core_cycles` (zero/default compatibility): `dma.aging_cycle_quantum` is used directly. Clock changes alter the represented physical wait.
- `physical_ns`: `dma.aging_quantum_ns` is converted using `memory.dram.core_clock_ghz`; the physical wait target is stable while the cycle count changes.

Both alternatives apply only when `dma.aging_metric` is `cycles`. Grant-based aging remains independent of either quantum. The effective cycle quantum must remain in `[1,1048576]`.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Workload and controls:

1. An old priority-0 64-byte request waits on channel 0.
2. A priority-2 4096-byte request occupies the non-preemptive shared path for 178 cycles.
3. A fresh priority-2 64-byte peer arrives immediately before the long request retires.
4. Aging uses submission scope, increment 1, and either 64 core cycles or 64 physical ns.
5. SRAM bandwidth metering is disabled; each short request takes 52 cycles after selection.

| Domain | Core clock (GHz) | Effective quantum (cycles) | Old completion | Fresh completion | Batch completion |
|---|---:|---:|---:|---:|---:|
| Core cycles | 0.5 | 64 | 231 | 283 | 283 |
| Core cycles | 1.0 | 64 | 231 | 283 | 283 |
| Core cycles | 2.0 | 64 | 231 | 283 | 283 |
| Physical ns | 0.5 | 32 | 231 | 283 | 283 |
| Physical ns | 1.0 | 64 | 231 | 283 | 283 |
| Physical ns | 2.0 | 128 | 283 | 231 | 283 |

The core-cycle policy retains identical arbitration at all three clocks but represents 128/64/32 ns at 0.5/1/2 GHz. The physical policy retains a 64 ns target by converting to 32/64/128 cycles. At 2 GHz that larger cycle quantum no longer promotes the old request past the fresh priority-2 peer: old completion rises 22.51%, while fresh completion falls 18.37%. Batch completion, useful bytes, occupied bytes, and byte-exact SRAM contents are unchanged. This is a fairness and latency-allocation trade-off, not a throughput or traffic gain.

## Multi-objective interpretation

| Dimension | Core-cycle quantum | Physical-ns quantum |
|---|---|---|
| Throughput | Same measured batch completion | Same measured batch completion |
| Per-request latency | Stable arbitration in cycle space; physical fairness becomes more aggressive as clock rises | Stable physical policy; cycle-space ordering can change with clock |
| Area/resources | Counter/down-counter and compare; smallest for a fixed constant | Needs clock-aware conversion or precomputed register programming; exact area unquantified |
| Power/energy | Expected lower policy/config activity | Conversion/config state may add activity; magnitude unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged |
| Numerical accuracy | Unchanged | Unchanged |
| Control complexity | Simple synchronous contract | Requires an explicit clock source, ceil conversion, and transition policy |
| Verification burden | Quantum boundaries and counter wrap | Same plus clock sweeps, fractional conversion, and dynamic-clock behavior |
| Compiler/runtime | Software tunes a cycle count for each frequency | Software can state one physical QoS target but must provide a valid clock contract |

A fixed-function TU at one frequency may choose core cycles to minimize control and verification. A multi-frequency or reusable IP block may choose physical ns so firmware can retain one physical fairness target. The current cmodel quantifies neither physical area nor energy, so it does not select a universal mode.

## Implementation and compatibility

The executable path includes:

1. `config/tu_config.yaml` and `config/tu_config.json`: `aging_quantum_domain` and `aging_quantum_ns`.
2. `scripts/gen_config.py` and `tu_cmodel/tu_config.h`: generated domain/source constants, effective cycle quantum, runtime fields, and defaults.
3. `tu_cmodel/infra/config.{h,c}`: canonical enum/fields, parser, validation, documentation, and `ceil(ns × GHz)` conversion.
4. `tu_cmodel/tu_cmodel.c` and `dma_descriptor.c`: the existing live engine consumes the converted cycle quantum.
5. `tests/test_generated_config.py`, `tests/test_config.c`, and `tests/test_dma_arbitration_sweep.c`: generation, parsing, validation, conversion, live order/cycles, and byte movement.

`core_cycles` remains numeric zero and the shipped/default contract. Existing configurations and zero-initialized runtime callers continue to use the existing cycle quantum. `physical_ns` is additive and fails closed for unsupported domain names, nonpositive sources, or values whose converted quantum exceeds the engine limit.

## Fidelity limits

- The clock source is the existing TU/core cycle-domain setting under `memory.dram.core_clock_ghz`; a distinct DMA clock and CDC are not modeled.
- Conversion occurs during canonical-to-runtime configuration. Dynamic clock changes do not recompute an already initialized DMA engine.
- Service remains descriptor-boundary and non-preemptive; aging cannot interrupt the long descriptor that caused the wait.
- The cmodel uses 64-bit timestamps and integer floor division after converting the quantum. Physical counter width, wrap handling, comparator timing, register programming, and power are unquantified.
- Voltage, frequency-transition latency, timing feasibility, thermal behavior, deadlines, weighted/deficit arbitration, finite credits, shared-memory contention, and calibrated protocol timing remain unmodeled.

## Verification

```sh
make test-config-generation test-config test-dma test-dma-arbitration-sweep
make config-docs
make clean && make
make test-quick
```
