# DRAM Adaptive Row-Timeout Clock Domain

**Date:** 2026-08-11
**Question:** Should an adaptive open-row timeout be defined as fixed TU/core cycles or as a physical nanosecond interval when the core clock changes?

## Hypothesis

A fixed-cycle timeout is a plausible controller contract when timer state and policy are tied directly to a synchronous core pipeline. A physical-ns timeout is plausible when the policy represents DRAM/background retention or energy behavior that should not change merely because the TU/core clock changes. With an 8-unit threshold, fixed cycles should change the physical decision boundary across 0.5/1/2 GHz, while physical ns should preserve it by converting to 4/8/16 core cycles.

## Alternatives and hardware rationale

| Domain | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| `core_cycles` | Small synchronous counter/comparator; easy timing closure and compatibility with cycle-budgeted controller stages | The physical idle interval is 16/8/4 ns at 0.5/1/2 GHz, so a clock change silently changes row-policy behavior |
| `physical_ns` | Preserves one physical idle policy across core-clock choices and supports comparing clocks without confounding timeout semantics | Requires clock-aware conversion/state, rounding, and transition semantics; a physical implementation still realizes a cycle counter at the selected clock |

Both are materially distinct and remain runtime-configurable. `core_cycles` is the zero/default compatibility mode.

## Executable contract

Canonical JSON/YAML under `tu.memory.dram` accepts:

```json
"row_policy": "adaptive_timeout",
"row_timeout_domain": "physical_ns",
"row_open_timeout_ns": 8.0,
"row_open_timeout_cycles": 8,
"core_clock_ghz": 2.0
```

Only the source selected by `row_timeout_domain` is active. `core_cycles` uses `row_open_timeout_cycles`; `physical_ns` uses `row_open_timeout_ns` and computes `ceil(ns × core_clock_ghz)`. The legacy `tu_dram_set_row_policy_timeout()` API remains a core-cycle wrapper. The new `tu_dram_set_row_policy_timeout_domain()` API and canonical parser fail without mutation on unsupported domains, non-finite/out-of-range values, or an effective zero adaptive timeout. `tu_dram_configure_core_clock()` recomputes a physical-ns timeout along with physical latency and refresh timing.

A zero-initialized canonical caller still selects `core_cycles`; non-adaptive policies may retain a zero timeout because the field is inactive. The shipped YAML, JSON, generated constants, canonical struct/default/parser/validation, runtime model, generated config docs, and focused tests all carry the setting.

## Measured sweep

Command: `make test-dram-row-timeout-domain-sweep`

Configuration: one channel, one bank, 256-byte rows, two same-row 64-byte reads, 50-cycle base read latency, 20-cycle closed-bank activation, adaptive threshold of either 8 core cycles or 8 ns. Explicit idle ticks represent 6 ns or 12 ns at each clock. Service is returned row-service only; coarse bandwidth/channel stalls are excluded.

| Core GHz | Domain | Gap ns | Effective timeout cycles | Service cycles | Hits | Timeout precharges |
|---:|---|---:|---:|---:|---:|---:|
| 0.5 | core cycles | 6 | 8 | 120 | 1 | 0 |
| 1.0 | core cycles | 6 | 8 | 120 | 1 | 0 |
| 2.0 | core cycles | 6 | 8 | 140 | 0 | 1 |
| 0.5 | physical ns | 6 | 4 | 120 | 1 | 0 |
| 1.0 | physical ns | 6 | 8 | 120 | 1 | 0 |
| 2.0 | physical ns | 6 | 16 | 120 | 1 | 0 |
| 0.5 | core cycles | 12 | 8 | 120 | 1 | 0 |
| 1.0 | core cycles | 12 | 8 | 140 | 0 | 1 |
| 2.0 | core cycles | 12 | 8 | 140 | 0 | 1 |
| 0.5 | physical ns | 12 | 4 | 140 | 0 | 1 |
| 1.0 | physical ns | 12 | 8 | 140 | 0 | 1 |
| 2.0 | physical ns | 12 | 16 | 140 | 0 | 1 |

All 12 rows are exact fail-closed gates over converted timeout cycles, returned service, row hits, and timeout precharges.

## Findings

1. The physical-ns domain preserves the architectural boundary across all clocks: every 6 ns reuse is a hit and every 12 ns reuse times out. Effective cycle thresholds scale 4→8→16.
2. The fixed-cycle domain represents a different real design, not an error. Its 8-cycle threshold corresponds to 16 ns at 0.5 GHz, 8 ns at 1 GHz, and 4 ns at 2 GHz. Consequently, the same 6 ns reuse changes from a hit at 0.5/1 GHz to a timeout at 2 GHz.
3. In this two-access microbenchmark a timeout adds one 20-cycle activation, changing returned service from 120 to 140 cycles. This is a local row-state latency result, not controller throughput.
4. Neither domain is universally preferable. Cycle-defined policy is simpler and may intentionally scale with synchronous controller opportunity; physical-ns policy avoids confounding a clock sweep when the intended timeout is a fixed external-time assumption.

## Gain versus sacrifice

- **Throughput:** Not quantified; there is no request queue, command scheduler, or overlap model. The sweep gates only returned service.
- **Latency:** Exact local result is 120 cycles for a retained hit versus 140 for timeout+activation. Physical time of those base cycle terms still depends on the separately selected base-latency domain.
- **Area/resources:** Both need per-bank age state/comparison. Physical-ns mode additionally needs clock-derived configuration or programmed threshold conversion; actual counter width, gates, and area are unquantified.
- **Power/energy:** Physical-ns mode preserves a consistent idle-duration policy; fixed cycles close rows earlier in physical time as clock rises. Background/open-row, precharge, activation, and clock-tree energy are not wired, so net energy is unquantified.
- **SRAM/DRAM traffic:** Payload bytes are unchanged. Row command behavior changes (hit versus lazy timeout precharge+activate); command counts and bus occupancy remain unmodeled.
- **Numerical accuracy:** Unchanged; memory data and arithmetic semantics are identical.
- **Control complexity:** Core-cycle mode is the simpler contract. Physical-ns mode adds conversion and clock-change state semantics, but no dynamic DVFS transition protocol is claimed.
- **Verification burden:** Physical mode requires clock sweeps, ceiling-boundary vectors, failed-setter immutability, parser/propagation tests, and clock-change recomputation. Both source fields must remain coherent while only one is active.
- **Compiler/runtime:** Software selecting clocks must know whether timeout behavior should scale in physical time. Per-kernel dynamic changes still require drain/in-flight transition semantics not modeled here.

## Fidelity limits

This remains a deterministic lazy row-state model. It does not model a background PRE command, tRAS/tRP legality, command-bus occupancy, request queues, arbitration, reordering, bank groups, backpressure, or activation/background energy. `current_cycle` advances only through explicit ticks. `ceil(ns × GHz)` maps an aggregate physical interval into TU/core cycles; it does not model a separate DRAM command clock, CDC synchronizer, programmable timer quantization, or dynamic frequency transition. The sweep does not establish a calibrated timeout or makespan.

## Verification

```sh
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.row_timeout.h
make test-dram                              # 31/31
make test-config                            # 29/29
make test-dram-row-timeout-sweep            # 12/12 historical rows
make test-dram-row-timeout-domain-sweep     # 12/12 domain rows
make config-docs
make clean && make
make test-quick
```

## Actionable conclusion

Preserve both timeout domains. Use `core_cycles` for compatibility and synchronous cycle-budgeted controller studies; use `physical_ns` when comparing core clocks under one fixed physical idle policy. Do not interpret the latter as DVFS support or calibrated DRAM timing without voltage, transition, CDC, legal command scheduling, queueing, and energy contracts.
