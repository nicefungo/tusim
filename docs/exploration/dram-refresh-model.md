# DRAM Refresh Model: NONE / ALL_BANK / PER_BANK × FIXED / DEFERRED × 1x/2x/4x

**Date:** 2026-08-04
**Question:** What does refresh overhead cost a TU DRAM controller, and when should a designer choose all-bank vs per-bank refresh and fixed vs deferred scheduling?

## Hypothesis

Real DRAM must refresh every tREFI or it loses data. A cmodel without refresh overstates available service and hides a workload-dependent cost that differs strongly between controllers: a simple all-bank REFAB that locks the whole device for tRFC, versus DDR5-style per-bank refresh that locks only the addressed bank for a shorter tRFCpb. Scheduling adds a second axis: firing exactly at k×tREFI (predictable, phase-dependent collisions) versus bounded deferral (opportunistic firing on the first post-schedule access, hiding refresh in idle gaps when the phase cooperates). Neither scheduling policy should dominate across workloads — the win should be phase-dependent — and per-bank refresh should cost much less than all-bank under continuous traffic because only the refreshing bank is locked.

## Realistic alternatives

| Mode | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| `none` (compat) | Zero refresh state/control; legacy behavior bound. Not physically realistic — retained only as the backward-compatible default and refresh-free lower bound | Unmodeled data-retention failure; overstates available bandwidth |
| `all_bank` (REFAB) | Simplest controller: one global schedule, one command, whole-device lockout (~tRFC 350 ns). Plausible for low-end/legacy controllers and ranks without bank-group refresh support | Every refresh stalls ALL banks; collisions with active bursts cause latency variance |
| `per_bank` (DDR5-style) | Staggered per-bank refresh (tRFCpb ~90 ns) locks only one bank, hiding most of the cost behind other-bank traffic; supports fine-grained scheduling | Per-bank schedule/control state, 4× more commands, more complex timing logic |

Scheduling alternatives:

| Mode | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| `fixed` | Issues refresh exactly at k×tREFI — deterministic, trivial controller; cost is a phase-locked function of the access stream | Windows can land inside active bursts; no ability to hide behind idle |
| `deferred` | Bounded postponement (≤ max_deferral, validated ≤ tREFI) fires at the first post-schedule access or the hard deadline; can hide refresh in idle gaps | Postponement risks approaching the retention limit under sustained traffic; deadline logic; access-triggered latency spikes |

Rate multiplier 1x/2x/4x shortens the effective interval to tREFI/rate (JEDEC high-temperature retention). A physical TU hard-wires one refresh policy; the pre-spec cmodel preserves all of them as candidate chips. Zero remains the NONE legacy path, so old configs and zero-initialized callers keep historical behavior.

## Executable model and configuration

Canonical JSON/YAML under `tu.memory.dram` accepts:

```json
"refresh": {
  "mode": "per_bank",          /* none | all_bank | per_bank */
  "scheduling": "deferred",    /* fixed | deferred */
  "rate": 2,                   /* 1 | 2 | 4 */
  "trefi_ns": 7800,            /* JEDEC per-bank interval (DDR4-like default) */
  "trfc_ns": 350,              /* all-bank lockout */
  "trfc_pb_ns": 90,            /* per-bank lockout */
  "max_deferral_ns": 7800      /* hard deadline; validated ≤ tREFI */
}
```

Full path: YAML/JSON → `scripts/gen_config.py` (generates `TU_DRAM_REFRESH_MODE_*`, `TU_DRAM_REFRESH_SCHED_*`, `TU_DRAM_REFRESH_RATE`, `TU_DRAM_TREFI_NS`, `TU_DRAM_TRFC_NS`, `TU_DRAM_TRFC_PB_NS`, `TU_DRAM_REFRESH_MAX_DEFERRAL_NS`) → checked-in header → canonical struct/default/parser/validation (mode ∈ {none, all_bank, per_bank}, scheduling ∈ {fixed, deferred}, rate ∈ {1,2,4} with 0 = default 1x, `max_deferral ≤ tREFI` resolved through zero-means-default) → `tu_dram_set_refresh()` (zero timings normalize to NONE/1x/7800/350/90; unsupported mode/scheduling/rate or deferral > tREFI fail closed without mutating state) → per-bank `refresh_next`/`refresh_until` state.

Model semantics (all in the module's 1 GHz cycle domain, 1 cycle = 1 ns):

- **Lazy catch-up:** a catch-up function brings refresh state to cycle T, called on every access (opportunistic = true) and in `tu_dram_tick` (opportunistic = false, so event counters stay accurate during idle).
- **ALL_BANK** keeps a single global schedule (slot 0), first refresh at tREFI. **PER_BANK** staggers bank b's first refresh at (b+1)·tREFI/B.
- **FIXED** fires exactly at the schedule. **DEFERRED** fires at the earlier of (a) the first access to the bank after the schedule (that access pays the full duration) or (b) the hard deadline schedule + max_deferral (tick fires it even with no traffic). After an opportunistic fire at T the next schedule is T + tREFI (additive, not grid-aligned).
- Firing a refresh **precharges the row buffer**: open-page row tags for the addressed bank(s) on every channel are invalidated, so the next access misses. Refresh catch-up runs BEFORE row-policy accounting.
- Refresh lockout goes into returned `cycles` and the dedicated `total_refresh_stall_cycles` counter; it is deliberately kept OUT of `total_stall_cycles` (same convention as row penalties: refresh is service cost, not contention). The counter is a sum of per-access remainders — with ≤1 access/cycle a `dur`-length window can be paid by up to `dur` accesses summing to `dur(dur+1)/2`, not just `dur`.

## Measured sweep

Command: `make test-dram-refresh-sweep`

Configuration: custom DRAM, 4 channels, 4 banks/channel, 256 B rows, 64 B bursts, latency 50, tREFI=7800, tRFC=350, tRFCpb=90. `service` = sum of returned row-service cycles (latency + refresh lockout + row penalty). `rstall` = `total_refresh_stall_cycles` (per-access remainders). Two patterns: `steady` = 40,000 accesses at 1/cycle; `burst_idle` = 8 × (50 reads + 5,000 idle cycles).

| Mode | Sched | Pattern | Rate | Policy | Accesses | Service | Events | Rstall | Hits | Miss |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| none | fixed | steady | 1 | legacy | 40000 | 2,000,000 | 0 | 0 | 0 | 0 |
| all | fixed | steady | 1 | legacy | 40000 | 2,307,125 | 5 | 307,125 | 0 | 0 |
| all | deferred | steady | 1 | legacy | 40000 | 2,307,125 | 5 | 307,125 | 0 | 0 |
| per | fixed | steady | 1 | legacy | 40000 | 2,020,295 | 20 | 20,295 | 0 | 0 |
| per | deferred | steady | 1 | legacy | 40000 | 2,020,295 | 20 | 20,295 | 0 | 0 |
| all | fixed | steady | 2 | legacy | 40000 | 2,614,250 | 10 | 614,250 | 0 | 0 |
| all | fixed | steady | 4 | legacy | 40000 | 3,228,500 | 20 | 1,228,500 | 0 | 0 |
| all | fixed | steady | 1 | open | 40000 | 2,507,365 | 5 | 307,125 | 29,988 | 10,012 |
| per | fixed | steady | 1 | open | 40000 | 2,220,575 | 20 | 20,295 | 29,986 | 10,014 |
| none | fixed | burst_idle | 1 | legacy | 400 | 20,000 | 0 | 0 | 0 | 0 |
| all | fixed | burst_idle | 1 | legacy | 400 | 20,000 | 5 | 0 | 0 | 0 |
| all | deferred | burst_idle | 1 | legacy | 400 | 85,100 | 4 | 65,100 | 0 | 0 |
| per | fixed | burst_idle | 1 | legacy | 400 | 20,000 | 20 | 0 | 0 | 0 |
| per | deferred | burst_idle | 1 | legacy | 400 | 34,393 | 18 | 14,393 | 0 | 0 |
| all | fixed | burst_idle | 2 | legacy | 400 | 23,775 | 10 | 3,775 | 0 | 0 |
| all | fixed | burst_idle | 4 | legacy | 400 | 23,775 | 20 | 3,775 | 0 | 0 |
| all | fixed | burst_idle | 1 | open | 400 | 22,280 | 5 | 0 | 286 | 114 |
| per | fixed | burst_idle | 1 | open | 400 | 22,440 | 20 | 0 | 278 | 122 |

Readings (service deltas vs `none` in the same pattern):

- **Per-bank refresh is ~15× cheaper than all-bank under steady traffic:** +1.0% service (2,020,295) vs +15.4% (2,307,125). Only the refreshing bank is locked; with 4 banks, roughly a quarter of accesses pay a per-bank window.
- **Rate multiplier scales all-bank refresh cost linearly with events:** 2x → +30.7%, 4x → +61.4% steady service. Deferred and fixed produce identical steady cost because the window is always occupied.
- **Scheduling is phase-dependent — neither dominates:** in burst_idle, FIXED lands every refresh window inside an idle gap (20,000, zero stall, identical to none), while DEFERRED's opportunistic firing pulls each refresh onto the first access of the next burst, paying the full tRFC (85,100, +325%). Under steady traffic both are identical. The deferred mechanism's promise — hiding refresh in idle — fails exactly when the post-schedule idle is followed by a burst, because the first access triggers the refresh it was meant to avoid. This is a model of the real deferred-refresh hazard, not a bug.
- **Open-page interaction:** refresh precharges rows; with 5 all-bank events, 10,012 misses occur (every access in the window misses the invalidated row; misses ≥ events holds). Per-bank open page costs 11.0% over the legacy steady baseline (2,220,575) versus 25.4% for all-bank open.
- **NONE preserves legacy exactly:** 40,000×50 and 400×50 service, zero events, zero refresh stalls.

## Phase-alignment sweep: how phase-dependent is the scheduling trade-off?

The burst-idle rows above compare two phases; to test whether the fixed-vs-deferred reversal is a continuum or an anecdote, the phase sweep shifts the workload start across the full tREFI grid. Command: `make test-dram-refresh-phase-sweep`. Workload = `phase` idle cycles, then 8 × (50 reads at 1/cycle + 5,000 idle). ALL_BANK, tREFI=7800, tRFC=350. `overlap` = reads that paid a nonzero refresh remainder; `avg/read` = service/400.

| Sched | Phase | Service | Events | Rstall | Overlap | Avg/read |
|---|---|---:|---:|---:|---:|---:|
| fixed | 0 | 20,000 | 5 | 0 | 0 | 50.00 |
| defer | 0 | 85,100 | 4 | 65,100 | 200 | 212.75 |
| fixed | 780 | 20,210 | 5 | 210 | 20 | 50.52 |
| defer | 780 | 85,100 | 4 | 65,100 | 200 | 212.75 |
| fixed | 1,560 | 20,000 | 5 | 0 | 0 | 50.00 |
| defer | 1,560 | 85,100 | 4 | 65,100 | 200 | 212.75 |
| fixed | 2,340 | 20,000 | 5 | 0 | 0 | 50.00 |
| defer | 2,340 | 85,100 | 4 | 65,100 | 200 | 212.75 |
| fixed | 3,120 | 20,000 | 5 | 0 | 0 | 50.00 |
| defer | 3,120 | 85,100 | 4 | 65,100 | 200 | 212.75 |
| fixed | 3,900 | 23,775 | 5 | 3,775 | 50 | 59.44 |
| defer | 3,900 | 101,375 | 5 | 81,375 | 250 | 253.44 |
| fixed | 4,680 | 20,000 | 5 | 0 | 0 | 50.00 |
| defer | 4,680 | 101,375 | 5 | 81,375 | 250 | 253.44 |
| fixed | 5,460 | 23,455 | 5 | 3,455 | 10 | 58.64 |
| defer | 5,460 | 88,555 | 5 | 68,555 | 210 | 221.39 |
| fixed | 6,240 | 21,775 | 5 | 1,775 | 50 | 54.44 |
| defer | 6,240 | 101,375 | 5 | 81,375 | 250 | 253.44 |
| fixed | 7,020 | 20,000 | 6 | 0 | 0 | 50.00 |
| defer | 7,020 | 101,375 | 5 | 81,375 | 250 | 253.44 |
| fixed | 7,800 | 36,275 | 6 | 16,275 | 50 | 90.69 |
| defer | 7,800 | 101,375 | 5 | 81,375 | 250 | 253.44 |

Readings:

- **Fixed is a phase sawtooth:** zero refresh stall on 8 of 11 phases (windows land in idle), a 210-cycle tail catch at phase 780, partial-window hits (1,775–3,775) at phases 3,900–6,240, and a full 50-read burst paying 16,275 at phase 7,800 (+81% over baseline). The worst fixed row is still far below deferred's best.
- **Deferred is phase-invariant and always bad here:** 85,100–101,375 (+325% to +407%) at every phase. The reason is structural: after each opportunistic fire at a burst, the next schedule shifts to fire+7,800 and the next deadline to fire+15,600; the workload's bursts recur every 5,050 cycles, so a burst always lands inside the schedule→deadline window and triggers the refresh it was meant to hide. Deferred can only hide refresh when the idle gap following a schedule point is long enough that no access arrives before the deadline — i.e., when idle gaps approach tREFI.
- **Refined conclusion:** for workloads with inter-burst idle shorter than tREFI, deferred scheduling is strictly worse than fixed and cannot hide anything; its phase-invariance is a liability, not a robustness win. Deferred remains plausible only for workloads with ≥ tREFI-scale idle gaps (e.g., power-gated or batch-phase schedules), where it can merge refresh into long idle periods.

## Gain versus sacrifice

- **Throughput (sustained):** Under steady traffic, per-bank refresh loses only ~1% of service cycles versus ~15% for all-bank at 1x, scaling to ~61% at 4x. A queue-aware controller could overlap all-bank refresh with buffered work, which is **unquantified** here (no request queue exists).
- **Latency:** Any access overlapping a window pays the remaining duration (up to tRFC 350 / tRFCpb 90 cycles in this configuration). All-bank refreshes threaten every bank; per-bank only the addressed bank. Deferred shifts the spike from the schedule to the first post-schedule access — worse in burst_idle, potentially better when bursts align after windows.
- **Area/resources:** Per-bank requires per-bank schedule/state (B slots) and B times the refresh commands; all-bank needs one slot and one command. Both are small relative to channel/bank row tags. Physical gate counts are **unquantified**.
- **Power/energy:** More refresh commands (per-bank) raise command-bus and timing-logic toggles; all-bank lockouts serialize but with fewer commands. Activation/precharge energy from forced row misses is **not wired** into the power model; refresh energy itself is **unquantified**.
- **SRAM/DRAM traffic:** Payload bytes are unchanged; refresh consumes command/bandwidth slots only. In the open-page rows, refresh-induced misses add precharge/activate service cycles (visible in service).
- **Numerical accuracy:** Unaffected — refresh is a timing-only construct.
- **Control complexity:** Fixed is trivial (counter per slot); deferred needs a deadline comparison and opportunistic trigger on access; per-bank needs stagger initialization and per-bank slots. All are static, no feedback loops.
- **Verification burden:** Gated here by 9 focused DRAM tests (all-bank fixed lockout, per-bank stagger isolation, deferred opportunistic + deadline fire, 2x rate, NONE legacy, reset rebuild, fail-closed setter, full config parse/propagation, refresh-closes-rows) plus 26 config tests and an 18-row sweep with accounting gates (events bounds, rstall ≤ events·dur(dur+1)/2, hits+misses = accesses, misses ≥ events, steady streams must pay some stall).
- **Compiler/runtime:** Refresh is invisible to software except through observed latency jitter; deferred's phase dependence makes worst-case access latency less predictable, which matters for real-time or tightly scheduled loops. No compiler changes required.

## Fidelity limits

The cmodel has no request queue, reordering, bank groups, rank/subchannel geometry, write draining, arbitration, or adaptive scheduling, so refresh cannot yet be overlapped with buffered work and the sweep reports serialized service only. ns→cycle conversion is the module's documented 1 GHz convention (1:1); DRAM-clock-accurate conversion, refresh energy, activation/decode energy, and calibrated timing are unquantified. `total_refresh_stall_cycles` is a per-access sum-of-delays counter (consistent with how row penalties flow into returned cycles), not a wall-clock lost-time meter. The deferred model's hard deadline is enforced by tick; a real controller would also consider command-bus and power-supply constraints.

## Verification

```sh
make test-dram                         # 27/27 (9 refresh-focused)
make test-dram-refresh-sweep           # 18 rows, all accounting gates
make test-dram-refresh-phase-sweep     # 22 rows (11 phases x fixed/deferred)
make test-config                       # 26/26 (4 refresh-focused)
make clean && make
make test-quick
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.gen.h  # refresh constants present
make config-docs                       # docs/CONFIG_REFERENCE.md refresh rows
```

## Actionable conclusion

Preserve all modes. `none` stays the zero/default compatibility path. Per-bank refresh is the physically realistic choice for a DDR5-era controller and costs ~1% vs ~15% service under steady traffic in this model, at the price of per-bank control state and 4× commands; all-bank remains a valid low-complexity controller. Fixed scheduling is the conservative default: the phase sweep shows it pays zero refresh stall on most phases of a burst-idle workload and its worst phase (+81%) is still far below deferred's phase-invariant +325–407%. Deferred is valuable only for workloads with idle gaps approaching tREFI, where it can genuinely hide refresh — its opportunistic trigger guarantees a fire at the first post-schedule access, so for workloads with sub-tREFI idle gaps it can never hide and is strictly worse than fixed. Rate 2x/4x should be reserved for high-temperature retention requirements — it multiplies refresh cost linearly. Do not claim any mode is universally best: the measured phase sawtooth and the deferred-structural-inferiority regime are exactly the workload dependence a real controller exhibits.
