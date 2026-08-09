# DRAM Base-Latency Domain: Core Cycles vs Physical Nanoseconds

**Date:** 2026-08-09
**Question:** When the TU/core clock changes, should the DRAM model preserve a fixed base-latency cycle count or a fixed physical latency?

## Hypothesis

The prior model exposed read/write latency only as an undocumented cycle term. This is plausible for a compatibility abstraction or a latency budget tied to a fixed-depth core-side pipeline, but it is not plausible for an external-memory latency intended to remain constant in physical time while the TU clock changes. Preserving both contracts as explicit runtime modes should prevent clock sweeps from silently changing the meaning of the latency input.

## Alternatives and hardware rationale

| Mode | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| `core_cycles` | Compatibility with existing cost tables; represents fixed pipeline/controller stages or an abstract cycle budget; simplest deterministic contract | Physical latency shrinks as core clock rises and grows as it falls; unsuitable for a fixed-ns external-memory claim |
| `physical_ns` | Represents a calibrated or assumed external-memory/bridge latency in physical time; remains coherent across TU clock studies | Requires clock-domain conversion and retained source values; cycle count changes on clock reconfiguration |

A real chip may contain both fixed core-side stages and fixed-time off-chip service. The current cmodel has one aggregate base-latency term, so these modes are alternative interpretations of that term, not an additive decomposition. A future calibrated model should separate controller/core cycles, CDC/bridge latency, and DRAM-command timing rather than selecting one aggregate domain.

## Executable implementation

`tu.memory.dram.latency_domain` accepts:

- `core_cycles` (`0`, default and zero-initialized compatibility path)
- `physical_ns` (`1`)

The existing `tu.memory.latency.dram_read` and `dram_write` values are interpreted in the selected domain. `physical_ns` converts with `ceil(ns × core_clock_ghz)` and is recomputed by `tu_dram_configure_core_clock()`. `core_cycles` preserves the historical integer-cast behavior and does not rescale. The model retains source read/write values separately from executable cycle counts.

The setting is wired through YAML/JSON, generated constants, canonical defaults/parser/validation, `tu_dram_create_from_config()`, the public `tu_dram_set_latency_domain()` API, runtime state, transfer estimates, and read/write service. Unknown modes, negative/non-finite/out-of-range values, and conversion overflow fail without mutating active latency state.

## Measured sweep

Command: `make test-dram-core-clock-sweep`

Configuration: custom non-ideal DRAM, fixed 64 GB/s, read source value 50 (cycles or ns), 4 KiB read estimate, core clocks 0.5/1/2 GHz. The table uses the deterministic estimate `base_cycles + ceil(4096 × GHz / 64)`.

| Domain | Clock (GHz) | BW (B/core cycle) | Base (cycles) | 4 KiB estimate (cycles) | Estimate (ns) |
|---|---:|---:|---:|---:|---:|
| `core_cycles` | 0.5 | 128.0 | 50 | 82 | 164.0 |
| `core_cycles` | 1.0 | 64.0 | 50 | 114 | 114.0 |
| `core_cycles` | 2.0 | 32.0 | 50 | 178 | 89.0 |
| `physical_ns` | 0.5 | 128.0 | 25 | 57 | 114.0 |
| `physical_ns` | 1.0 | 64.0 | 50 | 114 | 114.0 |
| `physical_ns` | 2.0 | 32.0 | 100 | 228 | 114.0 |

All six rows pass exact fail-closed gates. The constant 114 ns in `physical_ns` is expected here: 50 ns base plus 64 ns to transfer 4 KiB at 64 GB/s. This is formula validation, not silicon calibration.

## Findings

1. The old cycle-only contract creates a clock-dependent physical-time interpretation: this workload changes from 164 ns at 0.5 GHz to 89 ns at 2 GHz even though external bandwidth is fixed.
2. `physical_ns` preserves the measured estimate at 114 ns across all three clocks while executable cycles rise from 57 to 228.
3. Neither mode is universally correct. `core_cycles` is useful for compatibility and fixed-stage abstractions; `physical_ns` is the defensible choice when the source is an external physical-latency assumption.
4. Selecting `physical_ns` does not make the preset latency values calibrated. It only gives their units and clock conversion an executable contract.

## Gain versus sacrifice

- **Throughput:** No request-level throughput was measured. Physical-ns mode consumes more core cycles at higher clocks, making memory pressure visible to cycle-based schedulers; queue overlap and concurrency remain unmodeled.
- **Latency:** Exact deterministic estimates are shown above. Physical-ns mode preserves the configured physical base term; core-cycles mode preserves the configured cycle budget. End-to-end application latency is unquantified.
- **Area/resources:** A hard-wired single-clock design pays no runtime mode cost. Multi-clock or DVFS-capable hardware needs counters/converters and CDC structures; area is unquantified. The cmodel mode itself represents alternative candidate designs, not a requirement for a runtime hardware mux.
- **Power/energy:** Payload and modeled event counts are unchanged. Longer cycle occupancy can increase leakage/control activity, while a faster clock can increase dynamic/clock-tree power. Voltage, controller activity, and DRAM energy are not wired, so net energy is unquantified.
- **SRAM/DRAM traffic:** Identical payload bytes and address placement. Only cycle conversion changes.
- **Numerical accuracy:** Unchanged; no arithmetic or data bytes change.
- **Control complexity:** `core_cycles` is simplest. `physical_ns` requires a clock contract and rescaling on reconfiguration. Safe in-flight clock transitions, PLL sequencing, and CDC are unsupported.
- **Verification burden:** Both modes, three clocks, read/write conversion, parser/default/error paths, zero-initialized compatibility, failed-setter immutability, generated constants, and clock-change recomputation are gated.
- **Compiler/runtime:** Cost models must know which domain supplied each latency. Physical-ns mode prevents a compiler clock sweep from accidentally treating a fixed external delay as fixed core cycles. There is no modeled transition protocol for changing clock with requests in flight.

## Fidelity limits

The model remains deterministic and uncalibrated. It does not separate controller stages, PHY/CDC delay, command timing, row timing in ns, queueing, arbitration, bank groups, request overlap, voltage/frequency feasibility, or thermal response. Read/write service still excludes payload serialization while `tu_dram_estimate_transfer()` includes a coarse bandwidth term; those APIs must not be presented as the same timing boundary. Row penalties remain explicit core-cycle terms.

## Verification

```sh
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.gen.h
make test-dram                         # 30/30
make test-config                       # 28/28
make test-dram-core-clock-sweep        # 6/6 rows
make config-docs
make clean && make
make test-quick
```

## Actionable conclusion

Preserve both modes. Keep `core_cycles` as zero/default for backward compatibility and fixed-stage studies. Select `physical_ns` only when the latency input represents a physical-time assumption, and label it uncalibrated until tied to a memory/controller contract. Do not infer DVFS feasibility or a universal clock choice from this sweep.
