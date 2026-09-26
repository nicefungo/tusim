# DMA Aging Boost Cap: Unbounded Fairness vs Priority Isolation

**Date:** 2026-09-26

**Mode:** pre-spec exploration

**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

Should priority aging on a shared, non-preemptive DMA path be allowed to erase any static-priority gap, or should hardware cap the promotion granted to an old descriptor?

Unbounded aging gives eventual service under a continuing stream of finite descriptors. A capped implementation can preserve a minimum amount of static QoS isolation: old background work may become more competitive, but cannot overtake traffic whose base-priority gap exceeds the cap. Both are physically plausible. A controller can omit the cap comparator for the simplest compatibility design, or implement a small programmable/hard-wired saturation limit when priority classes must remain meaningful.

## Runtime alternatives

`dma.aging_max_boost` accepts:

- `0`: unbounded compatibility mode. Promotion is limited only by the 8-bit effective-priority ceiling.
- `1..255`: cap the priority levels added by aging.

The executable effective priority is:

```text
raw_boost = aging_steps * aging_increment
effective_boost = min(raw_boost, aging_max_boost)  # cap omitted when value is 0
effective_priority = min(255, base_priority + effective_boost)
```

This setting is orthogonal to submission/queue-head scope, missed-grant/wait-cycle metrics, increment, and cycle/ns quantum. A physical TU would normally hard-wire or narrowly program these together; the pre-spec cmodel retains independent controls to expose their interactions.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Workload:

1. One old priority-0 64-byte descriptor competes with priority-4 descriptors on another channel.
2. Five high-priority descriptors arrive successively.
3. Aging uses submission scope, missed-grant steps, and increment 1.
4. Each descriptor takes 52 cycles: 50 base cycles plus two 32-byte payload beats; SRAM bandwidth metering is disabled.
5. Arbitration is descriptor-boundary and non-preemptive.

| Maximum boost | Old priority-0 completion | Batch completion | Interpretation |
|---:|---:|---:|---|
| 0 (unbounded) | 261 | 313 | Four missed grants erase the priority-4 gap; rotating tie serves old work before the fifth high descriptor |
| 2 | 313 | 313 | The cap cannot erase the four-level gap; all five high descriptors complete first |
| 4 | 261 | 313 | The cap exactly permits the old request to tie after four misses |

All rows move the same 384 useful bytes and produce byte-identical SRAM contents. Relative to unbounded aging, cap 2 delays the old descriptor by 19.92% (261 to 313 cycles) while preserving priority-4 isolation. Batch completion is unchanged. This is latency allocation and fairness policy, not throughput, traffic, or arithmetic improvement.

## Multi-objective interpretation

| Dimension | Unbounded (`0`) | Finite cap (`1..255`) |
|---|---|---|
| Throughput | Same measured 313-cycle batch | Same measured 313-cycle batch |
| Per-request latency | Old low-priority work eventually closes arbitrary finite gaps; 261 cycles here | Protects sufficiently higher classes; cap 2 delays old work to 313 cycles here |
| Fairness/starvation | Stronger eventual-service property for a continuing stream of finite descriptors | A gap larger than the cap can still starve under sustained arrivals; this is intentional isolation, not fairness completion |
| Area/resources | No configurable cap comparator/mux beyond the existing 8-bit saturation | Needs cap state plus compare/select or fixed saturation; magnitude unquantified |
| Power/energy | Expected less policy logic, but old work may perturb high-priority service earlier | Additional compare/control switching; workload energy impact unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged |
| Numerical accuracy | Unchanged; byte-exact movement | Unchanged; byte-exact movement |
| Control complexity | Simplest starvation-oriented aging contract | Adds a second policy threshold and interactions with increment/metric/scope |
| Verification burden | Epoch/time overflow, saturation, ties | Same plus below/equal/above-gap caps and parse/live propagation |
| Compiler/runtime | Priority labels decay completely with sufficient wait | Software can retain class separation but must understand that low classes may not receive eventual service |

No cap is universally preferred. Unbounded aging is appropriate where starvation avoidance dominates. A finite cap is plausible where real-time, safety, display, or model-critical traffic must retain a minimum class advantage, accepting weaker service guarantees for background transfers.

## Implementation and compatibility

The configuration path is executable through:

1. `config/tu_config.yaml` and `config/tu_config.json`: `dma.aging_max_boost`.
2. `scripts/gen_config.py` and generated `tu_cmodel/tu_config.h`: macro, runtime field, and default initializer.
3. `tu_cmodel/infra/config.{h,c}`: canonical field, default, JSON parser, `[0,255]` validation, runtime conversion, and generated documentation.
4. `tu_cmodel/tu_cmodel.c`: runtime-to-engine propagation.
5. `tu_cmodel/dma_descriptor.{h,c}`: overflow-safe boost arithmetic and optional cap.
6. `tests/test_generated_config.py`, `tests/test_config.c`, and `tests/test_dma_arbitration_sweep.c`: generation, parsing, rejection, live propagation, exact ordering/cycles, and byte movement.

Zero is the generated, canonical, and zero-initialized compatibility value. Existing public DMA initializers retain unbounded behavior. The additive `tu_dma_init_config_boundary_aging_policy_cap()` entry point carries an explicit cap without changing older signatures.

## Fidelity limits

- The experiment is finite. Cap 2 would permit starvation under an indefinitely renewed priority-4 stream; the test deliberately reports isolation rather than claiming fairness.
- Service remains non-preemptive. Neither capped nor unbounded aging can interrupt an active long descriptor.
- Priorities and the cap are externally configured; compiler/runtime priority assignment and admission control are unmodeled.
- The cmodel uses 64-bit epochs/timestamps and 8-bit effective priorities. Physical counter width, wrap protocol, cap register encoding, comparator timing, area, power, and energy are unquantified.
- Shared SRAM/DRAM contention, finite credits, command FIFOs, deadlines, weighted/deficit service, and calibrated protocol timing remain unmodeled.

## Verification

```sh
make test-config-generation test-config test-dma test-dma-arbitration-sweep
make config-docs
make clean && make
make test-quick
```
