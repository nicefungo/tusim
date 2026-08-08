# DRAM Core-Clock Cycle-Domain Consistency

**Date:** 2026-08-08
**Question:** Does the advertised DRAM `core_clock_ghz` setting actually preserve fixed physical bandwidth and refresh timing when the TU clock changes?

## Hypothesis

The shipped YAML/JSON exposed `tu.memory.dram.core_clock_ghz`, but the canonical parser ignored it, the public setter was a no-op, bandwidth conversion assumed 1 GHz, and refresh treated 1 ns as one cycle. This made non-1-GHz studies internally inconsistent. For a fixed external bandwidth in GB/s, bytes per TU cycle must fall as the TU clock rises. Physical refresh intervals and lockout durations expressed in ns must grow proportionally in TU cycles.

## Alternatives and hardware rationale

| TU/core clock | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| 0.5 GHz | Timing margin, lower expected dynamic power, simpler physical closure, mature-node or low-power deployment | Lower compute throughput; more external bytes arrive per core cycle but work advances more slowly |
| 1.0 GHz | Compatibility/reference point and moderate implementation target | Neither the timing margin of 0.5 GHz nor peak throughput of 2.0 GHz |
| 2.0 GHz | Latency/throughput target when process, voltage, floorplan, and cooling permit | Harder timing closure; expected higher power, clock-tree cost, verification corners, and fewer DRAM bytes per core cycle |

These are comparison points, not hard-wired operating modes or validated DVFS points. The cmodel preserves a continuous runtime setting in `(0, 10]` GHz. A zero-initialized legacy `tu_config_t` maps `0` to the historical 1 GHz behavior; explicit JSON/YAML zero is rejected.

## Executable implementation

The full configuration path now honors `tu.memory.dram.core_clock_ghz`:

1. YAML/JSON source and generated `TU_DRAM_CORE_CLOCK_GHZ` constant;
2. canonical `tu_config_t` default, parser, and validation;
3. `tu_dram_create_from_config()` propagation;
4. per-instance `core_clock_ghz` state;
5. GB/s-to-bytes/cycle bandwidth metering and transfer estimates;
6. ns-to-cycle conversion for tREFI, tRFC, tRFCpb, and maximum deferral;
7. generated config documentation.

`tu_dram_configure_core_clock()` is the validated API. The historical void `tu_dram_set_core_clock()` remains source-compatible and delegates to it. A clock change resets the coarse bandwidth window and rebuilds the refresh schedule from retained source ns values, avoiding stale 1-GHz-derived cycles.

## Measured sweep

Command: `make test-dram-core-clock-sweep`

Configuration: custom non-ideal DRAM, fixed 64 GB/s external bandwidth, 50-cycle read-latency term, 4 KiB read estimate, tREFI=1000 ns, tRFC=100 ns, all-bank fixed refresh.

| Clock (GHz) | BW (B/core cycle) | 4 KiB estimate (cycles) | Estimate (ns) | tREFI (cycles) | tRFC (cycles) |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 128.0 | 82 | 164.0 | 500 | 50 |
| 1.0 | 64.0 | 114 | 114.0 | 1000 | 100 |
| 2.0 | 32.0 | 178 | 89.0 | 2000 | 200 |

All three rows pass exact fail-closed gates against `ceil(bytes × GHz / GB/s)` and `ceil(ns × GHz)`.

## Findings

1. **The prior non-1-GHz path was a documented no-op.** The same 64 GB/s was incorrectly treated as 64 B/cycle at every clock, and refresh ns were incorrectly copied directly to cycles.
2. **Fixed physical bandwidth now has the expected inverse cycle relation.** From 0.5 to 2.0 GHz, bandwidth changes from 128 to 32 B/core-cycle; the 4 KiB bandwidth component therefore changes from 32 to 128 cycles.
3. **Refresh physical time remains invariant.** tREFI=1000 ns maps to 500/1000/2000 cycles and tRFC=100 ns maps to 50/100/200 cycles at 0.5/1/2 GHz.
4. **The transfer-estimate ns column is not a silicon latency prediction.** Its fixed 50-cycle base-latency term shrinks in physical time as clock rises, causing the printed 164→89 ns trend. The current API specifies that term in cycles and has no calibrated DRAM-command-clock conversion. Only the bandwidth and refresh conversions are validated here.

## Gain versus sacrifice

- **Throughput:** Higher TU clock can raise compute throughput, but fixed DRAM GB/s supplies fewer bytes per core cycle. End-to-end throughput is workload-dependent and was not measured.
- **Latency:** The 4 KiB estimate increases in cycles with clock but falls in reported ns because base latency is fixed in core cycles. Calibrated end-to-end latency is **unquantified**.
- **Area/resources:** Faster clocks typically require deeper pipelining, stronger cells, more buffering, and more clock-tree resources. Area is **unquantified**.
- **Power/energy:** Higher clock generally raises dynamic and clock-tree power; voltage may also need to rise. The DRAM clock setting is not wired to voltage or the power model, so power and energy are **unquantified**.
- **SRAM/DRAM traffic:** Payload bytes are unchanged. Available DRAM bytes/core-cycle scale inversely with clock; refresh command count per physical time is unchanged.
- **Numerical accuracy:** Unaffected.
- **Control complexity:** A hard-wired physical design can avoid runtime clock state. Multi-frequency hardware needs clock/voltage sequencing and CDC contracts not represented here.
- **Verification burden:** Parser/default/error cases, zero-initialized compatibility, direct clock changes, bandwidth estimates, refresh rescaling, generated constants, and three sweep rows are gated.
- **Compiler/runtime:** Cost models must use the same clock as the cmodel when translating GB/s and ns. Runtime DVFS transitions, clock-domain crossings, and rescheduling in-flight requests remain unsupported.

## Fidelity limits

This remains a deterministic service model, not a DRAM controller or DVFS simulator. Read/write latency fields remain uncalibrated cycle terms; DRAM command-clock ratios, PLL/voltage transitions, CDC FIFOs, timing closure, thermal limits, queue overlap, arbitration, bank groups, and refresh energy are not modeled. The coarse bandwidth window is capacity accounting, not a request schedule. Do not combine this sweep with the separate power-model clock knob as if the two were automatically synchronized.

## Verification

```sh
make test-dram                         # 29/29
make test-config                       # 27/27
make test-dram-core-clock-sweep        # 3/3 rows
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.gen.h
make config-docs
make clean && make
make test-quick
```

## Actionable conclusion

Keep core clock runtime-configurable and retain 1 GHz as the compatibility default. Use the setting to maintain consistent physical-bandwidth and refresh conversions across architecture studies. Do not interpret higher clock as universally better: it reduces bytes/core-cycle and brings unmodeled area, power, timing, CDC, and thermal costs. Before making physical latency or DVFS conclusions, add an explicit calibrated contract for read/write latency units and clock-domain relationships.
