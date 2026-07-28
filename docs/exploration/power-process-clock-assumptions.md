# Explicit Process-Node and Clock Assumptions

**Date:** 2026-07-28

**Question:** Can power/area studies select process node and modeled clock explicitly, or are results silently determined by PE-array size and DRAM bandwidth heuristics?

**Hypothesis:** Explicit process and clock settings will make physically distinct implementation assumptions reproducible while retaining `auto` as a backward-compatible exploratory baseline.

## Realistic alternatives

| Choice | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| `auto` | Fast early exploration when process and timing targets are genuinely undecided; preserves historical behavior | Array size and DRAM bandwidth become undocumented proxies for process/clock, so comparisons can confound architecture and implementation assumptions |
| Mature process (45/28/16 nm) | Lower mask/NRE risk, mature IP and yield, long-lifecycle or cost-sensitive edge products, possible higher-voltage/automotive qualification | Larger estimated MAC/SRAM area, higher table energy, and lower nominal clocks in this first-order model |
| Advanced process (7/5/3 nm) | Density, energy, and throughput targets for high-volume mobile/datacenter designs | Higher design/mask cost, process risk, leakage/thermal and physical-design complexity not represented by the scalar tables |
| Explicit lower clock | Timing margin, lower operating power, simpler closure, or thermal/power-cap operation | Higher latency and more leakage energy for a fixed cycle count in the current constant-voltage model |
| Explicit higher clock | Lower latency and higher throughput when timing and delivery permit | Higher average power; real designs may require higher voltage, deeper pipelines, stronger clock/power delivery, or larger cells—none are modeled here |

All six process presets and both `auto`/explicit clock selection remain available. A process node is not a runtime-switchable physical feature in one chip; it is a runtime **cmodel configuration alternative** for comparing candidate implementations without rebuilding the simulator.

## Implementation

The canonical JSON/YAML `power` block now accepts:

```json
"power": {
  "tech_node": "auto",
  "clock_freq_mhz": 0.0
}
```

- `tech_node`: `auto`, `45nm`, `28nm`, `16nm`, `7nm`, `5nm`, or `3nm`.
- `clock_freq_mhz`: `0` keeps the legacy 1 GHz / high-bandwidth 2 GHz heuristic; explicit values are validated in `(0, 10000]` MHz.
- Canonical integer value `0` means AUTO, so old zero-initialized `tu_config_t` callers preserve the prior heuristic. Explicit process selections use values 1–6.
- `tu_power_model_from_config()` consumes both fields. Explicit values override the PE-size and DRAM-bandwidth heuristics.
- The YAML generator emits `TU_POWER_TECH_NODE` and `TU_POWER_CLOCK_FREQ_MHZ`; a temporary-header test proves generation does not drop them.
- Invalid process names and negative/out-of-range clocks fail configuration validation.

## Executable sweep

Command:

```sh
make test-power-assumptions-sweep
```

The sweep uses a fixed, explicit activity contract resembling one FP16 GEMM: 1,048,576 MACs, 8,192 cycles, 65,536 input bytes, and 16,384 FP32 output bytes. It records two RF reads and one RF write per MAC, SPAD endpoint accesses, DRAM endpoint transactions, DMA bytes, clock energy, and area-derived leakage. The dimensions are intentionally held constant so rows isolate table/process and clock assumptions.

| Case | Node | MHz | Latency (us) | Energy (uJ) | Avg power (mW) | Area (mm2) | MAC share | DRAM share | Leakage share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | 45nm | 800 | 10.240 | 4.736467 | 462.546 | 18.237 | 22.14% | 17.08% | 59.14% |
| nominal | 28nm | 1200 | 6.827 | 2.071377 | 303.424 | 11.854 | 32.90% | 25.58% | 39.07% |
| nominal | 16nm | 1500 | 5.461 | 1.096897 | 200.848 | 7.295 | 38.24% | 29.87% | 29.06% |
| nominal | 7nm | 2000 | 4.096 | 0.463788 | 113.229 | 3.647 | 45.22% | 35.33% | 16.11% |
| nominal | 5nm | 2500 | 3.277 | 0.301249 | 91.934 | 2.553 | 48.73% | 37.73% | 9.72% |
| nominal | 3nm | 3000 | 2.731 | 0.205970 | 75.428 | 1.824 | 50.91% | 39.28% | 6.04% |
| fixed 7nm | 7nm | 750 | 10.923 | 0.588289 | 53.859 | 3.647 | 35.65% | 27.85% | 33.86% |
| fixed 7nm | 7nm | 1500 | 5.461 | 0.488688 | 89.481 | 3.647 | 42.91% | 33.53% | 20.38% |
| fixed 7nm | 7nm | 2500 | 3.277 | 0.448848 | 136.977 | 3.647 | 46.72% | 36.50% | 13.31% |

## Gain-versus-sacrifice findings

- **Throughput/latency:** For the fixed 8,192-cycle contract, 7nm latency falls from 10.923 us at 750 MHz to 3.277 us at 2.5 GHz (3.33x). This is only a frequency conversion; the cmodel does not prove that this netlist closes timing at any row.
- **Power/energy:** At fixed 7nm coefficients, average power rises from 53.859 to 136.977 mW as clock rises. Energy falls from 0.588289 to 0.448848 uJ because the same dynamic event counts finish sooner and accumulate less leakage. This is a constant-voltage approximation, **not DVFS**; voltage-dependent dynamic energy (`CV^2`) and frequency/voltage feasibility are absent.
- **Area/resources:** Table area falls from 18.237 mm2 at 45nm to 1.824 mm2 at 3nm for the same abstract 16x16 PE and memories. These are first-order scalar estimates with a fixed 30% overhead, not floorplanned macro/compiler results. Mask cost, yield, routing density, SRAM aspect ratios, and analog/IO area are unquantified.
- **SRAM/DRAM traffic:** Event counts and transferred bytes are identical across rows. Process selection changes per-event table energy, not traffic. No locality gain is attributed to process or clock.
- **Numerical accuracy:** Unchanged; all rows execute the same abstract FP16 activity contract. Voltage-induced timing faults, retention limits, and near-threshold numerical reliability are not modeled.
- **Control complexity:** Explicit static configuration adds negligible modeled control. Real in-chip DVFS would require PLLs, regulators, clock-domain crossings, transition sequencing, and voltage/frequency state control; this implementation does not claim those features.
- **Verification burden:** Every string alternative, AUTO/default compatibility, explicit override, invalid name, invalid clock, header generation, and finite-positive sweep row is gated. Physical energy-table calibration remains separate.
- **Compiler/runtime:** Design-space tooling can now pin process and clock so compiler schedule comparisons do not silently change physical assumptions. Software running on a fabricated TU would normally see a fixed process and a platform-controlled clock policy, not choose the process node per workload.
- **Cost/risk:** Advanced nodes appear favorable in every modeled scalar because the existing tables encode monotonic scaling. Real NRE, schedule, yield, IP availability, reliability, package/IO limits, and supply risk are unquantified; therefore the table cannot support a universal “smallest node wins” recommendation.

## Fidelity limits and actionable conclusion

This work corrects a **configuration provenance problem**, not power-model calibration. Safe labels are: **canonical config integrated**, **activity-driven first-order estimate**, and **uncalibrated process-table comparison**. The existing documentation's claimed silicon/CACTI accuracy was not independently revalidated here.

Use explicit node and clock values whenever comparing architectures. Keep AUTO only for early sketches or compatibility. Do not interpret the 7nm frequency rows as DVFS, and do not use the monotonic node table alone to select a fabrication process. A defensible DVFS study requires voltage-dependent dynamic/leakage coefficients, timing feasibility, transition overhead, thermal response, and power-delivery limits.

## Verification

```sh
make test-config                    # 22/22
make test-power                     # 20/20, statically linked
make test-power-assumptions-sweep   # 9 rows + fail-closed finite metric gate
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make clean && make
make test-quick
```
