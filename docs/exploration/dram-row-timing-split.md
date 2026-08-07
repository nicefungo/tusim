# DRAM Row Timing: Closed-Bank Activation vs Open-Row Replacement

**Date:** 2026-08-07
**Question:** Does one generic row-miss penalty hide a realistic open-page versus closed-page trade-off?

## Hypothesis

The prior row model charged one penalty for both (1) activating a precharged bank and (2) replacing a different open row. That compatibility model makes open-page no worse than closed-page: hits are free and every miss costs the same. A physically better deterministic abstraction separates an activation cost from the larger precharge-plus-activate replacement cost. Under that split, open-page should retain a locality advantage but lose under severe same-bank row thrashing; closed-page should trade locality for bounded, predictable activation service.

## Alternatives and hardware rationale

| Alternative | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| Equal-cost compatibility (`conflict=0`, inherits activate) | Minimal model/config state; preserves historical cmodel results and simple analytical studies | Cannot represent extra work for replacing an open row and structurally biases comparisons toward open-page |
| Split activation/replacement cost (`conflict > activate`) | Represents precharge+activate versus activate-only service without pretending to model every DRAM timing command | Adds one timing parameter, state classification, counters, calibration burden, and workload sensitivity |
| Open-page policy | Keeps a row active to exploit tensor/weight stream locality | Requires row tags and policy control; pathological same-bank row alternation pays repeated replacements |
| Closed-page policy | Precharges after each access for predictable service and adversarial/mixed-tenant streams | Gives up same-row hits and spends activation work on every access |

The cmodel preserves both policies and both timing interpretations. A physical controller would normally hard-wire a policy and calibrated timings; the pre-spec model keeps them runtime-configurable for comparison.

## Executable configuration and model

Canonical JSON/YAML under `tu.memory.dram` now accepts:

```json
"row_policy": "open_page",
"row_miss_penalty_cycles": 20,
"row_conflict_penalty_cycles": 40
```

`row_miss_penalty_cycles` is the activate-from-closed cost. `row_conflict_penalty_cycles` is the replace-open-row cost. Canonical value `0` inherits the miss cost, preserving old JSON files, `tu_config_default()` callers that modify only the historical field, and zero-initialized callers. The existing public `tu_dram_set_row_policy()` also assigns both costs equally; the new `tu_dram_set_row_policy_timing()` selects split costs explicitly.

The executable model classifies every explicit-policy miss as:

- **empty miss:** bank has no open row; charge activation cost;
- **hit:** requested row is already open; charge zero row penalty;
- **replacement:** another row is open; charge conflict cost;
- **closed-page access:** always classified as an empty activation because the row is not retained.

Compatibility `total_row_conflicts` remains the aggregate miss count. New `total_row_empty_misses` and `total_row_replacements` counters expose the distinction. Address-mapping changes, row-policy changes, reset, and refresh still clear retained row state.

## Measured sweep

Command: `make test-dram-row-policy-sweep`

Configuration: custom 1-channel, 16-bank DRAM; 2 KiB rows; 64 reads × 64 B; 50-cycle base read. Equal profile = activate 20 / replacement 20. Split profile = activate 20 / replacement 40. `service` is the sum of returned read cycles; it excludes the module's separate coarse bandwidth/channel stall output.

| Pattern | Cost profile | Policy | Service | Hits | Misses | Empty | Replacements |
|---|---|---|---:|---:|---:|---:|---:|
| sequential | equal | open | 3,240 | 62 | 2 | 2 | 0 |
| sequential | equal | closed | 4,480 | 0 | 64 | 64 | 0 |
| sequential | split | open | 3,240 | 62 | 2 | 2 | 0 |
| sequential | split | closed | 4,480 | 0 | 64 | 64 | 0 |
| row thrash | equal | open | 4,480 | 0 | 64 | 1 | 63 |
| row thrash | equal | closed | 4,480 | 0 | 64 | 64 | 0 |
| row thrash | split | open | 5,740 | 0 | 64 | 1 | 63 |
| row thrash | split | closed | 4,480 | 0 | 64 | 64 | 0 |
| bank stream | equal | open | 3,520 | 48 | 16 | 16 | 0 |
| bank stream | equal | closed | 4,480 | 0 | 64 | 64 | 0 |
| bank stream | split | open | 3,520 | 48 | 16 | 16 | 0 |
| bank stream | split | closed | 4,480 | 0 | 64 | 64 | 0 |

All 12 rows pass fail-closed accounting gates: hits+misses=64, empty+replacements=misses, and closed-page service exactly equals 64×(base+activate).

## Findings

1. **The old equal-cost model erased the thrash trade-off.** Open and closed both report 4,480 cycles for row thrash because 63 replacements cost no more than empty activation.
2. **Split costs create a workload-dependent reversal.** With a 40-cycle replacement cost, open-page row thrash rises to 5,740 cycles; closed-page remains 4,480, 21.95% lower than open-page in this measured regime.
3. **Locality still favors open-page.** Sequential open-page is 27.68% lower service than closed-page (3,240 vs 4,480), and bank-stream is 21.43% lower (3,520 vs 4,480). Neither result changes under split costs because those patterns produce no replacements.
4. **No policy is universal.** Open-page wins when rows are reused; closed-page wins when replacement cost is materially above activation cost and accesses alternate rows in one bank. Equal costs remain useful only as compatibility or as a deliberately coarse bound.

## Gain versus sacrifice

- **Throughput:** In the serialized returned-service domain, split timing exposes a 27.68% open-page gain for sequential locality and a 21.95% closed-page gain for the measured row-thrash pattern. Queue-level throughput and bank overlap are **unquantified**.
- **Latency:** Open-page gives 50-cycle hits but 90-cycle replacement accesses in the split profile; closed-page is a predictable 70 cycles/access. Tail distributions are not modeled.
- **Area/resources:** Split timing adds one parameter and two counters to the cmodel. Real open-page hardware needs row tags and policy logic; closed-page can simplify policy state. Gate counts and storage energy are **unquantified**.
- **Power/energy:** Replacements should consume more precharge/activate energy than empty activations; open-page hits should save array-command energy. These events are now observable, but the power model is not wired to them, so energy is **unquantified**.
- **SRAM/DRAM traffic:** Payload bytes and SRAM traffic are unchanged. DRAM command activity differs (activate versus precharge+activate), but command-bus traffic is not modeled.
- **Numerical accuracy:** Unaffected; this is timing/state accounting only.
- **Control complexity:** Open-page requires retained row state and a policy decision. Closed-page is simpler and predictable. Split timing itself adds no adaptive policy.
- **Verification burden:** Exact empty/hit/replacement vectors, old-setter compatibility, parser/validation/propagation, generated-header output, shipped JSON load, refresh precharge behavior, and the 12-row matrix are gated.
- **Compiler/runtime:** Allocators and tilers can benefit from row-aware placement under open-page. Closed-page reduces dependence on placement but loses reuse. The current compiler does not consume these counters or schedule DRAM commands.

## Fidelity limits

This is a deterministic service model, not a JEDEC command scheduler. It does not separately model tRCD, tRP, tRAS, tCCD, bank groups, rank/subchannel timing, request queues, reordering, read/write turnaround, command arbitration, or activate/precharge energy. The two penalties are user-supplied cycle abstractions in the module's 1 GHz domain and are uncalibrated. Returned row service remains separate from the coarse bandwidth/channel stall domain; values are not wall-clock makespan.

## Verification

```sh
make test-dram                         # 28/28, including split timing and compatibility
make test-dram-row-policy-sweep        # 12/12 rows
make test-config                       # 26/26; shipped JSON loaded by real parser
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.gen.h
make config-docs
make clean && make
make test-quick
```

## Actionable conclusion

Keep open-page and closed-page as realistic alternatives, and keep equal-cost timing as the backward-compatible default. Use an explicit replacement cost above activation cost when exploring physically plausible row-policy trade-offs; otherwise the model structurally prevents closed-page from winning. Do not convert the measured 20/40-cycle example into a hardware recommendation: the crossover depends on locality, address mapping, calibrated DRAM timings, queues, and controller scheduling, all of which must be stated for any design conclusion.
