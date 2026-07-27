# Per-Core Heterogeneous Dataflow Selection

**Date:** 2026-07-27

**Question:** Can a multicore TU retain and execute different WS, OS, and RS selections per core, or does the process-global `g_tu` path overwrite every core with the same mode?

**Hypothesis:** A dedicated core-state setter will let independent core snapshots execute distinct registered plug-ins with byte-identical arithmetic but mode-specific live cycle estimates; unsupported modes must fail without changing the retained selection.

## Realistic alternatives

| Architecture choice | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| Homogeneous fixed dataflow | Lowest mux/control/configuration cost; one compiler mapping and verification target; physically specialized operand network | Poor fit for workloads whose reuse pattern differs from the fixed choice; no runtime adaptation |
| Homogeneous runtime-selectable dataflow | One cluster-wide mode can follow workload phase while preserving symmetric cores and simpler scheduling | Reconfigurable movement/control hardware costs more than a fixed implementation; global phase changes may require synchronization |
| Per-core heterogeneous selection | Different cores can serve convolution-like, GEMM-like, or output-resident kernels concurrently, and a pre-spec model can compare mappings without rebuilding | Per-core mode state, routing/mux complexity, scheduling/load-balancing burden, more verification states; a physical implementation may instead hard-wire heterogeneous core types |

All three remain plausible. This change does not select heterogeneous operation as universally preferable: it makes the existing core-snapshot abstraction honor independent choices so homogeneous and heterogeneous studies are both executable.

## Implementation

The public core API now provides:

```c
int tu_core_set_dataflow(tu_core_t *core, tu_dataflow_id_t mode);
tu_dataflow_id_t tu_core_get_dataflow(const tu_core_t *core);
const char *tu_core_get_dataflow_name(const tu_core_t *core);
```

The setter writes the registered plug-in pointer into `core->state.dataflow`, not process-global `g_tu`. `tu_core_mma()` swaps that retained state into the executing path and then restores it. Unknown or reserved-but-unimplemented IDs return `-1` and leave the core unchanged. The legacy global setter was also aligned with its documented contract: it now rejects unsupported IDs rather than silently reporting success after falling back to WS.

The built-in registry objects remain process-shared and stable-address. Core-level active selection, SRAM, cycle totals, and output state are isolated; plug-in-private lifetime statistics are still process-global and therefore are **not** valid per-core counters. True simultaneous host-thread execution is also unsupported because core execution temporarily swaps process-global `g_tu`.

## Executable sweep

Command:

```sh
make test-multicore-dataflow-sweep
```

Configuration: 16x16 PE, pipeline depth 2, FP16 W/A, FP32 accumulation. Three core snapshots retain WS, OS, and RS respectively. Each row loads identical deterministic operands, executes each core, requires the active mode to remain unchanged, rejects NLR without fallback, and requires all three FP32 output buffers to be byte-identical. Reported cycles are MMA-only deltas from each core's `estimated_cycles`; DMA is excluded.

| Workload | MxNxK | WS cycles | OS cycles | RS cycles | Numerical gate |
|---|---:|---:|---:|---:|---|
| Edge + multiple K tiles | 31x19x17 | 468 | 88 | 276 | byte-identical |
| Square | 64x64x64 | 5,120 | 1,280 | 3,136 | byte-identical |
| Wide small-K | 32x128x16 | 1,280 | 320 | 784 | byte-identical |
| Tall small-K | 128x32x16 | 1,280 | 320 | 784 | byte-identical |

These values reproduce the direct-global live formulas through independent core snapshots. They prove selection isolation and deterministic accounting, not physical dataflow speed. OS's current estimate remains lowest at depth 2 in this matrix because its formula omits physical operand-network bandwidth and has no fill/drain term.

## Gain-versus-sacrifice findings

- **Throughput:** The model can now assign different deterministic estimates to cores in one process. No concurrent makespan or load-balance gain is quantified because calls execute serially and there is no multicore scheduler trace.
- **Latency:** Per-operation MMA estimates differ as shown. Mode-switch latency, pipeline drain at reconfiguration, and cluster synchronization are unmodeled.
- **Area/resources:** Unquantified. Expected direction: per-core runtime selection needs configuration state and selectable operand/psum paths; fixed homogeneous cores should be smallest. Physically specialized heterogeneous cores may avoid muxes but lose fungibility.
- **Power/energy:** Unquantified. Independent stationarity may reduce movement in a suitable workload, but no plug-in emits calibrated RF/SRAM/NoC traffic and the power model is not connected to dataflow events.
- **SRAM/DRAM traffic:** Endpoint tensors are identical in the functional model. Reuse and internal movement differences remain logical labels; no traffic reduction is claimed.
- **Numerical accuracy:** All tested finite normal-valued outputs are byte-identical in FP32 accumulator storage. This does not replace raw-bit precision conformance for NaNs, infinities, signed zero, or subnormals.
- **Control complexity:** Higher for per-core dynamic selection; mode state must survive context/core swaps and unsupported modes must fail closed. Global fixed selection is simplest.
- **Verification burden:** Three modes multiplied by core isolation, edge geometry, multi-K tiling, invalid selection, and state-swap behavior. Process-shared plug-in statistics remain an explicit limitation.
- **Compiler/runtime:** A scheduler may assign kernels to compatible cores or set modes before dispatch. It must account for reconfiguration/synchronization once those costs are modeled. The current API enables policy experiments but does not choose a policy.

## Verification

```sh
make test-multicore
make test-multicore-dataflow-sweep
make test-dataflow
make clean && make
make test-quick
```

Safe claim labels: **core-state integrated**, **executable functional**, and **deterministic uncalibrated estimate**. Physical movement, parallel execution, per-core plug-in statistics, switch cost, and calibrated timing remain open fidelity gaps.
