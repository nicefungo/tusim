# WS / OS / RS Executable Dataflow Re-Audit

**Date:** 2026-07-26
**Question:** Do the advertised WS, OS, and RS alternatives actually execute through the selected runtime path, and what does the live dispatcher—not a parallel analytical formula—report across edge, multi-K-tile, square, and small-K shapes?
**Hypothesis:** The historical core-based sweep can label all rows incorrectly because a global selector is overwritten by core swap-in. Re-running through the direct global path with active-plugin, independent-oracle, and exact-cycle gates will expose the real per-K-tile accounting and runtime-configuration gaps.

## Alternatives and why hardware teams might choose them

| Mode | Plausible motivation | Principal sacrifice |
|---|---|---|
| Weight stationary (WS) | Keep weights near MACs, reduce repeated weight delivery, regular systolic control | Fill/drain latency, PE storage for weights, shape-dependent utilization |
| Output stationary (OS) | Keep FP32 partial sums local and stream operands; attractive for dot-product/vector organizations | Higher simultaneous W/A delivery demand and accumulator storage; more SRAM/interconnect pressure |
| Row stationary (RS) | Reuse filter/weight rows and partial sums; attractive for convolution-like reuse | More mapping/control complexity and workload-dependent reuse; GEMM loop labels do not prove Eyeriss-style physical movement |

All three remain explicit runtime alternatives. No mode is selected as universally best.

## Audit findings and implementation

The prior harness called `tu_set_dataflow()` on process-global state and then executed a `tu_core_t`; core swap-in could replace the selection. It also printed mismatches without making them fatal and compared only against WS, so a shared arithmetic defect could pass. Its separate analytical formulas omitted the live dispatcher's K-tile callback multiplicity and OS `ceil(k_count/4)` term.

This audit changed the executable path:

- canonical config `dataflow` and `pipeline_depth` now propagate into `tu_runtime_config_t` and the active global plugin;
- generated-header output preserves both runtime fields;
- unknown dataflow names and reserved-but-unimplemented NLR fail validation instead of silently becoming WS;
- pipeline depth is validated in `[1,16]` and reaches the live callback;
- dispatcher fill/drain callbacks receive valid edge extents and the requested runtime depth;
- WS/OS/RS use the canonical FP16 decoder instead of three copied local decoders;
- focused dataflow tests and the sweep link `./libtucmodel.a`, preventing stale shared-library selection;
- the sweep fails nonzero on an inactive plugin, any FP32 raw-bit oracle mismatch, or any live-cycle/formula mismatch.

The executable cycle contract is currently per `(M tile, N tile, K tile)` callback:

- WS: `pd*n_valid + k_valid + pd*m_valid`
- OS: `k_valid + ceil(k_valid/4)`
- RS, `pd=1`: `k_valid`
- RS, `pd>1`: `(pd-1)*n_valid + 1 + k_valid + (pd-1)*m_valid`

These are **deterministic uncalibrated estimates**, not measured hardware timing. In particular, OS's quarter-K surcharge is an ad hoc proxy not tied to bytes/ports, and the dispatcher charges fill/drain for every K tile. Both require a named SRAM/interconnect schedule before physical performance conclusions are valid.

## Measured live MMA-cycle matrix

Configuration: 16×16 PE, FP16 W/A, FP32 accumulation, direct global plugin path. The table excludes host/SRAM DMA so it does not mix cycle domains. Every row passed active-plugin, FP32 raw-bit oracle, and exact live-cycle gates.

| Workload | M×N×K | Pipeline | WS | OS | RS |
|---|---:|---:|---:|---:|---:|
| edge + multiple K tiles | 31×19×17 | 2 | 468 | 88 | 276 |
| square | 64×64×64 | 2 | 5,120 | 1,280 | 3,136 |
| wide small-K | 32×128×16 | 2 | 1,280 | 320 | 784 |
| tall small-K | 128×32×16 | 2 | 1,280 | 320 | 784 |

Pipeline sensitivity for 31×19×17:

| Pipeline depth | WS | OS | RS |
|---:|---:|---:|---:|
| 1 | 268 | 88 | 68 |
| 2 | 468 | 88 | 276 |
| 4 | 868 | 88 | 676 |

The matrix is evidence about the current formulas, not proof that OS is physically 4× faster. At depth 1 RS is lower than OS because RS has no fill/drain while OS retains its ad hoc fetch surcharge. This reversal is useful: it demonstrates why the alternatives and depth must remain configurable and why the current scalar cycle rankings cannot choose a physical dataflow.

## Gain-versus-sacrifice interpretation

- **Throughput/latency:** Observable only as live deterministic MMA cycles above. OS is lowest at depth 2 in these shapes; RS is lowest at depth 1. Rankings are formula- and regime-specific, not calibrated throughput.
- **Area/resources:** Unquantified. Expected direction: WS needs stationary-weight storage/distribution, OS needs accumulator residency and dual operand delivery, RS needs row mapping/reuse state.
- **Power/energy:** Unquantified. Stationarity can reduce data movement, but no plugin emits named RF/SRAM/interconnect traffic events, so energy comparisons would be invented.
- **SRAM/DRAM traffic:** Endpoint tensors are functionally read from the same arrays. Claimed WS/OS/RS reuse is logical metadata/commentary only; completed transfer counts at RF/SRAM/NoC boundaries are absent.
- **Numerical accuracy:** For the tested finite normal-valued domain, all modes are raw-bit identical to an independent oracle using canonical FP16 conversion and the dispatcher's per-K-tile FP32 grouping. This is not a full FP16 special-value conformance suite.
- **Control complexity:** Qualitatively lowest for a fixed single mode; a flexible pre-spec implementation carries selection/config verification. RS mapping and OS bandwidth scheduling are expected to be more complex than the current scalar loops show.
- **Verification burden:** Three modes require active-state, oracle, edge, multi-K, configuration, and cycle-contract gates. NLR is rejected until an executable plugin exists.
- **Compiler/runtime:** The compiler must select a supported mode and eventually tile/layout operands to match a real movement schedule. JSON misspellings now fail closed. Runtime selection is global; per-core selection/isolation remains a separate contract.

## Fidelity limits and next decision

Safe labels are **executable functional**, **runtime-integrated on the direct global path**, and **deterministic estimate**. It is not defensible to call the scalar loops physical WS/OS/RS schedules or the timing calibrated.

The next dataflow implementation should not tune these formulas. First define and expose named traffic events (RF reads/writes, SRAM W/A/O reads/writes, operand broadcasts, psum movement) and a bandwidth/overlap contract. Then compare logical reuse and cycle estimates against compiler-emitted tiled traces. Per-core selection also needs a dedicated setter/state-isolation gate before multicore dataflow claims.

## Verification commands

```sh
make test-dataflow
make test-config
make test-dataflow-sweep
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make clean && make
make test-quick
```
