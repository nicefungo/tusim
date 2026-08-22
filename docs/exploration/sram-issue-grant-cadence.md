# SRAM Per-Bank Grant and Refill Cadence

**Date:** 2026-08-22
**Question:** How should the cmodel represent a throttled SRAM bank, a true single-ported bank, and wider multiported/banked implementations without conflating grant count with time?

## Hypothesis and realistic alternatives

The existing module stored `words_per_cycle`, but replenished that budget only every `bw_refill_window` cycles. With the default grant=1 and window=4, the executable contract is one word **per four-cycle window**, not one word/cycle. The canonical JSON exposed bank count and width but did not parse or propagate either the grant or refill cadence into `g_tu`.

| Alternative | Why hardware might choose it | Principal sacrifice |
|---|---|---|
| 1 grant / 4 cycles (compatibility throttle) | Represents a narrow or time-multiplexed bank, conservative macro timing, or a coarse legacy budget | Cannot be called “single port”; it rejects accesses even on intervening cycles until the window refills |
| 1 grant / 1 cycle (single issue) | Ordinary one-access-per-cycle SRAM macro with simple banking/control | Same-bank concurrent accesses serialize; lower peak bandwidth than multiport/replicated designs |
| 2 grants / 1 cycle (dual issue) | Concurrent DMA/compute or read/write demand can justify a true dual-port macro, replication, or banking behind one logical bank | More bitlines/ports or replicated storage, muxing, area, dynamic power, and verification |
| 4 grants / 1 cycle (wide issue) | High-throughput tensor feeds may provision several lanes or subbanks per logical bank | Largest expected area/routing/energy cost; physical feasibility depends on macro organization |

All remain runtime-selectable. The historical 1/4 setting remains the generated and zero-initialized compatibility default; no local fastest row replaces it.

## Executable configuration path

`tu.memory.banking` now accepts:

```json
{
  "banks": 8,
  "bank_width_bytes": 4,
  "words_per_refill": 2,
  "refill_window_cycles": 1,
  "stall_penalty_cycles": 2
}
```

The canonical parser validates grants and penalty in `[1,255]` and the refill window in `[1,1000000]`. `tu_config_to_runtime()` propagates bank count, grant, penalty, and window; `tu_init_with_config()` initializes W/A/O regions with those live values. Zero-valued fields in a manually zero-initialized `tu_runtime_config_t` inherit checked-in defaults. Top-level engine regions retain the 4-byte checked-in word width because scalar FP16/FP32 callers do not carry an access size; making that old canonical field executable would otherwise over-read or overwrite host objects. The lower-level runtime constructor accepts another width only for callers that provide matching word-sized buffers.

The generator emits the matching constants and runtime fields. The old identifier `sram_words_per_cycle` remains for ABI/source compatibility, but generated documentation now defines it as a grant per refill window. A one-cycle window is required before interpreting it as a physical issue rate.

## Measured matrix

Command: `make test-sram-issue-sweep`

Configuration: 8 logical banks, 4-byte words, one addressed bank, four reads, 2-cycle penalty per request beyond the available grant. “Spaced” advances the SRAM clock one cycle after each request; other rows issue all four at one model cycle.

| Configuration | Grant | Window | Request timing | Stall cycles |
|---|---:|---:|---|---:|
| legacy-burst | 1 | 4 | same cycle | 6 |
| legacy-spaced | 1 | 4 | one per cycle | 6 |
| single-port | 1 | 1 | one per cycle | 0 |
| dual-port | 2 | 1 | same cycle | 4 |
| quad-port | 4 | 1 | same cycle | 0 |

The discriminating result is the spaced pair: a true one-grant-per-cycle bank serves all four requests without stalls, while the compatibility 1/4 budget still charges 6 cycles. For same-cycle demand, increasing grants from one to two halves modeled stall cycles (6 to 4); four grants eliminate them for this exact four-request burst.

These are exact budget-counter results, not calibrated macro latency or end-to-end engine throughput. A stalled access still performs its functional memcpy immediately and returns a penalty; the model does not reschedule it or advance time automatically.

## Gain versus sacrifice

- **Throughput:** Wider/per-cycle provisioning raises the number of same-bank requests admitted before a penalty. End-to-end throughput is unquantified because engines call SRAM serially and do not replay stalled accesses.
- **Latency:** The measured penalty falls from 6 to 0 cycles for the stated patterns. This is a local stall counter, not compatible with every DMA/MMA cycle domain.
- **Area/resources:** Single issue is expected to be cheapest. Dual/quad issue needs ports, replication, subbanking, or wider macro organization; area and routing are unquantified.
- **Power/energy:** More ports and wider issue should increase capacitance and control activity, while fewer stall cycles may reduce leakage time. Neither SRAM access energy nor port-dependent leakage is wired into the power model, so net energy is unquantified.
- **SRAM/DRAM traffic:** SRAM request count and bytes are unchanged. DRAM traffic is unchanged. Wider issue changes service capacity only.
- **Numerical accuracy:** Unchanged; every configuration returns identical bytes.
- **Control complexity:** A fixed one-grant bank is simplest. Wider issue requires conflict detection and port assignment. The current scalar API has no simultaneous requester identities.
- **Verification burden:** Parser rejection, zero/default compatibility, grant/window/penalty and bank-count propagation, and exact stall counts are gated. Physical port collisions, replay, and variable-width engine accesses remain absent.
- **Compiler/runtime:** Software can use the settings for architecture comparison, but compiler scheduling cannot yet target ports or reason about simultaneous DMA/compute request sets.

## Arbitration fidelity correction

The prior `sram-arbitration-sweep.md` was analytical and its source harness is absent from the current checkout. The live `arb_mode` field is stored but never consumed by `sram_bw_consume()`. Therefore NONE/RR/PRIORITY are **not executable arbitration alternatives**, and earlier priority rankings must not be treated as cmodel measurements. Real arbitration needs a batched/multi-request API with requester/port identity, same-cycle conflicts, grant order, replay, and fairness state. This run does not invent those semantics.

## Implementation and verification

Changed paths:

- `config/tu_config.{yaml,json}`, `scripts/gen_config.py`, `tu_cmodel/tu_config.h`
- `tu_cmodel/infra/config.c`
- `tu_cmodel/tu_sram.{c,h}`, `tu_cmodel/tu_cmodel.c`
- `tests/test_sram_issue_sweep.c`, `Makefile`, `.gitignore`

Executed:

```text
make test-sram-issue-sweep
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.sram-issue.h
cc -O2 -Wall -Wextra -std=c11 -x c -fsyntax-only -include /tmp/tu_config.sram-issue.h /dev/null
```

Final clean build and quick regression are recorded in the heartbeat report.

## Conclusion and limits

Use grant/window pairs to preserve the legacy throttle and compare physically plausible single/dual/wide issue capacity. Do not call `words_per_cycle=N` an N-port SRAM unless the refill window is one. Do not use the current `arb_mode` labels for architectural conclusions. Queue/replay timing, simultaneous requester identity, read/write port restrictions, macro latency, area, power, and calibrated SRAM timing remain blocked on explicit contracts and evidence.
