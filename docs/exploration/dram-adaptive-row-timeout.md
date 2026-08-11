# DRAM Adaptive Open-Row Timeout

**Date:** 2026-08-10
**Question:** Can a low-complexity idle timeout preserve short row reuse while avoiding expensive open-row replacements after long idle gaps?

## Hypothesis

Static open-page and closed-page policies represent useful endpoints, but neither adapts to temporal locality. A per-bank idle timeout is a physically plausible middle point: keep a row open while reuse is likely, then precharge it after an idle threshold. With split activation/replacement costs, this should match open-page on dense reuse, match closed-page's activation behavior after long idle, and outperform both on bursts that contain short intra-burst reuse but long inter-burst gaps.

## Alternatives and hardware rationale

| Policy | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| `open_page` | Minimal policy logic and maximum row-hit opportunity for streaming/local workloads | A later different-row access pays replacement/precharge cost; an idle open row can retain unnecessary state/energy |
| `closed_page` | Predictable service, simple state, and no stale-row replacement; useful for random or adversarial traffic | Discards all row locality and activates on every access |
| `adaptive_timeout` | Captures short reuse while closing stale rows; common low-complexity alternative to history-based predictors | Requires per-bank age state, timeout comparison, tuning, and more verification; a poor threshold can collapse to either endpoint |

The cmodel retains all three modes. It does not select a universal winner, and a physical implementation may hard-wire one policy.

## Executable contract

Canonical JSON/YAML under `tu.memory.dram` accepts:

```json
"row_policy": "adaptive_timeout",
"row_open_timeout_cycles": 100,
"row_timeout_domain": "core_cycles",
"row_miss_penalty_cycles": 20,
"row_conflict_penalty_cycles": 40
```

`adaptive_timeout` requires a nonzero timeout. `legacy`, `open_page`, and `closed_page` remain unchanged; `legacy` remains the zero/default compatibility path. The executable model stores the last access cycle per channel×bank. On the next access, an open row whose idle age is strictly greater than the threshold is lazily precharged before row classification. Reuse exactly at the threshold remains a hit. Lazy evaluation gives the same next-access row result as an eager timer but does not model a background PRE command or its command-bus occupancy.

The public `tu_dram_set_row_policy_timeout()` API fails without mutation on unsupported modes or a zero adaptive timeout. Policy changes, address-mapping changes, reset, refresh precharge, and channel-state construction preserve coherent row state. `total_row_timeout_precharges` distinguishes adaptive closures from explicit open-row replacements.

## Measured sweep

Command: `make test-dram-row-timeout-sweep`

Configuration: one channel, one bank, 256-byte rows, 64-byte accesses, 50-cycle base read latency, 20-cycle activation, 40-cycle open-row replacement, 8-cycle timeout. Each pattern has 16 reads. Service is returned read service only; coarse channel and bandwidth-window stall are excluded.

| Pattern | Policy | Service cycles | Hits | Empty activations | Replacements | Timeout precharges |
|---|---|---:|---:|---:|---:|---:|
| dense reuse, 2-cycle gaps | open page | 820 | 15 | 1 | 0 | 0 |
| dense reuse, 2-cycle gaps | closed page | 1,120 | 0 | 16 | 0 | 0 |
| dense reuse, 2-cycle gaps | adaptive timeout | 820 | 15 | 1 | 0 | 0 |
| sparse reuse, 16-cycle gaps | open page | 820 | 15 | 1 | 0 | 0 |
| sparse reuse, 16-cycle gaps | closed page | 1,120 | 0 | 16 | 0 | 0 |
| sparse reuse, 16-cycle gaps | adaptive timeout | 1,120 | 0 | 16 | 0 | 15 |
| alternating rows, no gap | open page | 1,420 | 0 | 1 | 15 | 0 |
| alternating rows, no gap | closed page | 1,120 | 0 | 16 | 0 | 0 |
| alternating rows, no gap | adaptive timeout | 1,420 | 0 | 1 | 15 | 0 |
| two-access bursts, 16-cycle inter-burst gap | open page | 1,100 | 8 | 1 | 7 | 0 |
| two-access bursts, 16-cycle inter-burst gap | closed page | 1,120 | 0 | 16 | 0 | 0 |
| two-access bursts, 16-cycle inter-burst gap | adaptive timeout | 960 | 8 | 8 | 0 | 7 |

All 12 rows are exact fail-closed gates over service, hits, activations, replacements, and timeout closures.

## Findings

1. Adaptive timeout preserves every dense-reuse hit and exactly matches open-page service (820 cycles), 26.8% below closed-page for this pattern.
2. With a long gap between same-row accesses, adaptive timeout deliberately gives up open-page's retained-row benefit and matches closed-page service (1,120 cycles). This is not a performance gain in the current cycle model; it represents the physically plausible choice to avoid retaining stale rows.
3. With immediate alternating-row thrash, no timeout expires. Adaptive therefore matches open-page and is 26.8% slower than closed-page because each replacement costs 40 cycles versus a 20-cycle closed-bank activation.
4. For short reuse inside a burst and long idle between bursts, adaptive preserves eight hits but converts seven 40-cycle replacements into 20-cycle activations. It measures 960 cycles: 12.7% below open-page and 14.3% below closed-page in this exact regime.
5. The threshold is workload-sensitive. A shorter threshold approaches closed-page; a very long threshold approaches open-page. Compiler/runtime traces are required to choose a physical default.

## Gain versus sacrifice

- **Throughput:** Request-level throughput is not modeled. Returned service improves only in the measured burst-pair regime; queue overlap and command scheduling could change the effect.
- **Latency:** Exact service results are above. Adaptive helps when useful reuse occurs within the threshold and a different row arrives after it. It hurts relative to open-page when useful same-row reuse arrives after the threshold.
- **Area/resources:** Expected increase versus static policies: one age/timestamp or timer representation per tracked bank plus comparator/control. Bit width, synthesis area, and timer sharing are unquantified.
- **Power/energy:** Earlier precharge may reduce open-row/background energy but adds precharge commands; preserving short hits avoids activation energy. Neither activate/precharge nor background energy is wired, so net energy is unquantified.
- **SRAM/DRAM traffic:** Payload bytes and address placement are identical. DRAM command traffic changes qualitatively (timeout PRE versus later replacement), but command counts are not yet explicit.
- **Numerical accuracy:** Unchanged; data and arithmetic semantics are identical.
- **Control complexity:** Higher than static open/closed policies but lower than history-based predictors or queue-aware adaptive page management. Timeout tuning and clock-domain interpretation are new contracts.
- **Verification burden:** Requires threshold-boundary, long-idle, reset, mapping-change, refresh, failed-setter, parser/validation, and every policy-mode gate. Exact focused tests cover boundary, idle closure, state immutability, and parse-to-runtime propagation.
- **Compiler/runtime:** Offline traces can select a timeout or a hard-wired policy by workload class. Dynamic per-kernel changes would need transition and in-flight request semantics not modeled here.

## Fidelity limits

This is a deterministic row-state service model, not a DRAM controller. Timeout precharge is lazy and has no command-bus occupancy, tRAS/tRP legality, bank-group timing, energy, request queue, arbitration, reordering, or backpressure. `current_cycle` advances only through explicit `tu_dram_tick()` calls; callers that issue accesses without advancing time cannot trigger idle timeout. This original sweep uses the compatibility core-cycle timeout domain; `dram-row-timeout-domain.md` adds the runtime physical-ns alternative and clock conversion. Neither sweep establishes a calibrated timeout or end-to-end makespan.

## Verification

```sh
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.gen.h
make test-dram                         # 31/31
make test-config                       # 28/28
make test-dram-row-timeout-sweep       # 12/12 rows
make config-docs
make clean && make
make test-quick
```

## Actionable conclusion

Preserve open-page, closed-page, and adaptive-timeout as runtime alternatives. Adaptive timeout is valuable for bursty temporal locality, not universally superior. Use traces to set the threshold, keep `legacy` as the compatibility default, and do not infer throughput, power, or calibrated DRAM timing until queues, legal command timing, command traffic, and energy are modeled.
