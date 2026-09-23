# DMA Priority-Aging Increment: Slow, Balanced, and Aggressive Fairness

**Date:** 2026-09-23
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

How quickly should an old descriptor close a static-priority gap on a shared, non-preemptive DMA path?

The original aging policy added one priority level per missed descriptor grant. That is a conservative and inexpensive policy, but one fixed rate conflates the existence of starvation protection with how aggressively the hardware trades critical-stream latency for fairness. A programmable or hard-wired increment is physically plausible: a low-cost controller can add one, while a QoS-oriented controller can add a larger constant with the same epoch state and a wider/saturating increment path.

## Runtime alternatives

`dma.aging_increment` is an integer in `[1,255]` and is used only by `aging_priority` arbitration:

```text
age_grants = current_grant_epoch - selected_start_epoch
effective_priority = min(255, base_priority + age_grants * aging_increment)
```

| Representative increment | Hardware motivation | Main sacrifice |
|---:|---|---|
| 1 (zero/default compatibility) | Slow, predictable promotion; protects statically important traffic for more grants | A large priority gap can leave old low-priority work waiting for many descriptor completions |
| 2 | Middle point for mixed latency/fairness goals | More frequent overtaking of newly arrived high-priority work |
| 4 | Aggressive starvation protection for latency-sensitive old requests | Weakens static-priority isolation and can delay more critical arrivals after only one miss |

Every value from 1 through 255 is executable so a physical design can model a hard-wired constant or programmable register. The sweep uses 1/2/4 as discriminating design points, not as a claim that these are the only realizable values.

## Executable sweep

Command:

```sh
make test-dma-arbitration-sweep
```

Workload:

1. A priority-0 64-byte descriptor and a priority-4 64-byte descriptor become ready on separate channels.
2. After each high-priority grant, another fresh priority-4 descriptor is submitted, for five high descriptors total.
3. Transfers cost 52 cycles: 50 base plus two 32-byte payload beats; SRAM metering is disabled.
4. Arbitration uses submission-scope aging, descriptor-boundary non-preemptive service, and rotating ties.

| Increment | Old priority-0 completion | Batch completion | Grants needed to tie priority 4 |
|---:|---:|---:|---:|
| 1 | 261 | 313 | 4 |
| 2 | 157 | 313 | 2 |
| 4 | 105 | 313 | 1 |

Relative to increment 1, increment 2 reduces the old descriptor's measured completion time by 39.85%; increment 4 reduces it by 59.77%. The six-descriptor batch remains 313 cycles and all 384 bytes are byte-identical. The change redistributes per-request latency; it does not increase throughput or reduce traffic.

## Multi-objective interpretation

| Dimension | Lower increment | Higher increment |
|---|---|---|
| Throughput | Same measured batch completion | Same measured batch completion; no throughput gain |
| Latency | Protects fresh statically high-priority descriptors longer | Promotes old low-priority descriptors sooner in the measured sustained-arrival regime |
| Fairness | Weaker/slow starvation mitigation | Stronger/faster starvation mitigation in grants |
| Area/resources | Constant-one increment can simplify arithmetic | Constant or programmable multiply/shift/add/saturate path; exact cost unquantified |
| Power/energy | Less arithmetic switching expected | More promotion arithmetic/control switching expected; magnitude unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged, byte-identical |
| Numerical accuracy | Unchanged | Unchanged |
| Control complexity | Smallest when hard-wired to one | More policy state if software-programmable; validation and saturation logic required |
| Verification burden | Priority gaps, ties, epoch reset, saturation | Same plus rate extremes, overflow-safe saturation, and interaction with scope |
| Compiler/runtime | Stable static-priority separation | Exposes a QoS tuning knob; software must understand that priorities decay faster |

No increment is universally preferable. A hardware team may hard-wire one value after workload analysis even though the pre-spec cmodel keeps the rate configurable.

## Implementation and compatibility

The full executable path is wired through:

1. `config/tu_config.yaml` and `config/tu_config.json`: `dma.aging_increment`.
2. `scripts/gen_config.py` and `tu_cmodel/tu_config.h`: generated constant, runtime field, and default initializer.
3. `tu_cmodel/infra/config.{h,c}`: canonical field/default/parser/validation/runtime conversion and generated config docs.
4. `tu_cmodel/tu_cmodel.c`: runtime-to-engine propagation.
5. `tu_cmodel/dma_descriptor.{h,c}`: saturating effective-priority calculation.
6. `tests/test_generated_config.py`, `tests/test_config.c`, and `tests/test_dma_arbitration_sweep.c`: generated alternative, parser/runtime/live propagation, invalid rejection, exact service points, and byte movement.

Value 1 is the generated and canonical default. A zero-initialized runtime field maps to 1 for compatibility. The existing `tu_dma_init_config_boundary_aging()` API retains one-level behavior; the additive `tu_dma_init_config_boundary_aging_rate()` API carries an explicit rate. Canonical configuration rejects 0 and values above 255.

## Fidelity limits

- Aging is measured in granted descriptors, not cycles. Variable descriptor lengths make wall-clock fairness workload-dependent.
- Service is descriptor-boundary and non-preemptive; a selected long transfer cannot be interrupted.
- The model uses 64-bit epochs and 8-bit priorities. A physical counter width, wrap protocol, comparator timing, and register interface are not selected.
- Constant multipliers may synthesize differently for 1, 2, 4, and arbitrary values; area, timing, and energy are unquantified.
- Priorities and increments are configured externally; compiler/runtime QoS assignment is not automated.
- Shared SRAM/DRAM contention, finite command FIFOs, backpressure, weighted/deficit service, deadlines, and end-to-end compute overlap are unmodeled.
- Completion ticks are deterministic cmodel service values, not calibrated AXI/DRAM measurements.

## Verification

```sh
make test-config-generation test-config test-dma test-dma-arbitration-sweep
make config-docs
make clean && make
make test-quick
```
