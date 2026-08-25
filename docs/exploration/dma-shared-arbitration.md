# Shared-Serial DMA Arbitration: Round-Robin vs Strict Priority

**Date:** 2026-08-25
**Architecture question:** When several DMA descriptor queues share one non-preemptive data path, should the mover preserve fair rotation or let latency-critical descriptors bypass lower-priority queue heads?

## Hypothesis and alternatives

The shared-serial topology already has the simultaneous-request and descriptor-boundary contract needed for an executable arbitration comparison. Two materially distinct policies are preserved:

| Runtime `arbitration` | Hardware rationale | Expected benefit | Expected sacrifice |
|---|---|---|---|
| `round_robin` (zero/default) | Small fair queue selector for general-purpose W/A/O streams | Bounded service opportunity for every continuously nonempty channel; no software priority assignment required | A critical transfer can wait behind lower-value work |
| `strict_priority` | Comparator/select tree using the existing descriptor priority field | Protects critical loads/stores and deadline-sensitive streams | Lower-priority starvation is possible; software must assign priorities correctly; priority comparison and verification state are added |

Selection occurs only when the shared path is idle and only at descriptor boundaries. The active transfer is never preempted. Equal priorities use the existing rotating start point, making ties round-robin. Independent data paths retain simultaneous service and do not consume this policy.

## Executable configuration and workload

Command:

```sh
make test-dma-arbitration-sweep
```

The focused sweep submits three simultaneous 4,096-byte host-to-SRAM descriptors on channels 0/1/2 with priorities 0/10/5. Configuration is shared-serial, 32 B/model-cycle DMA width, 50-cycle base read latency, and the checked-in SRAM grant model. Each transfer has 2,162 modeled service cycles, including 1,984 SRAM-stall cycles; completion includes the initial issue tick. Every row gates exact completion timestamps and byte-exact SRAM contents.

| Policy | Low ch0 completion | Critical ch1 completion | Medium ch2 completion | Batch completion |
|---|---:|---:|---:|---:|
| round_robin | 2,163 | 4,325 | 6,487 | 6,487 |
| strict_priority | 6,487 | 2,163 | 4,325 | 6,487 |

Strict priority reduces the measured critical descriptor latency from 4,325 to 2,163 ticks (**50.0% lower**) by exchanging its place with lower-priority work. It does **not** improve batch completion or payload throughput: both policies finish the same 12,288 useful bytes at tick 6,487. The low-priority descriptor moves from first to last and takes **3.0x** its round-robin completion latency in this finite batch.

This is a scheduling trade-off, not a universally faster mode. Round-robin remains the compatibility default because it provides fairness without a compiler/runtime priority contract.

## Multi-objective interpretation

| Dimension | Round-robin | Strict priority | Current fidelity limit |
|---|---|---|---|
| Throughput | Same measured batch throughput | Same measured batch throughput | One non-preemptive shared path; no beat-level overlap, aggregate DRAM queue, or compute overlap |
| Latency | ch0/ch1/ch2 complete in rotation order | Critical ch1 completes 50.0% earlier in this priority assignment | Result depends on simultaneous readiness, descriptor sizes, initial rotation, and assigned priorities |
| Area/resources | Expected small rotating selector/state | Expected comparator/select logic over ready queue heads plus priority bits | Gate count, timing, descriptor RAM ports, and physical area are unquantified |
| Power/energy | Selector activity expected low | Comparator activity and priority metadata may raise control energy slightly | No DMA arbitration energy counters or physical calibration |
| SRAM/DRAM traffic | 12,288 useful bytes | Identical 12,288 useful bytes | Shared SRAM/DRAM arbitration, overfetch, and physical interface traffic are not modeled here |
| Numerical accuracy | Byte-exact transfer | Byte-exact transfer | No arithmetic or quantization semantics are involved |
| Control complexity | Rotation and empty-queue skipping | Highest-priority search plus rotating tie-break; starvation policy required at system level | Aging, weighted service, deadlines, preemption, cancellation, and bounded starvation are absent |
| Verification burden | Fair rotation and wrap-around | Priority ordering, ties, dynamic arrivals, and starvation scenarios | Focused gates cover discriminating initial-ready selection, exact finite-batch order, invalid mode rejection, and bytes |
| Compiler/runtime | No annotations required | Runtime/compiler can mark critical descriptors through existing `priority` | No automatic priority assignment, QoS API, admission policy, or end-to-end scheduler cost model |

## Implementation

The complete executable path is:

1. `config/tu_config.{yaml,json}`: `dma.arbitration` accepts `round_robin` or `strict_priority`.
2. `scripts/gen_config.py` and `tu_cmodel/tu_config.h`: generated/default constants and runtime field; round-robin is zero/default.
3. `tu_cmodel/infra/config.{h,c}`: canonical enum, parse, validation, canonical-to-runtime propagation, and generated config documentation.
4. `tu_cmodel/tu_cmodel.c`: initializes the live descriptor engine with the selected policy.
5. `tu_cmodel/dma_descriptor.{h,c}`: backward-compatible `tu_dma_init_config()` retains round-robin; `tu_dma_init_config_policy()` adds explicit policy selection; shared arbitration uses descriptor priority with rotating ties.
6. `tests/test_dma.c`, `tests/test_config.c`, and `tests/test_dma_arbitration_sweep.c`: policy behavior, invalid rejection, configuration propagation, exact timing, and byte movement.

## Limits and deferred variants

- Strict priority may starve a low-priority queue under sustained higher-priority arrivals. That is a real sacrifice, not a model defect to hide. Aging, weighted round-robin/deficit service, or deadline scheduling should be added only with producer traces and a concrete starvation/latency question.
- Arbitration is non-preemptive. A newly arrived high-priority descriptor cannot interrupt a long active transfer. Beat-level preemption would require progress/replay state and a physical burst contract.
- Descriptor priorities are caller-assigned metadata. The compiler, command queue, and workload runtime do not yet generate or optimize them.
- Completion ticks include the current coarse SRAM refill model and are not calibrated AXI/DRAM timings. Only relative order and exact behavior of this cmodel configuration are claimed.

## Verification commands

```sh
make test-dma
make test-config
make test-dma-arbitration-sweep
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make clean && make
make test-quick
```
