# DRAM Useful-Payload Efficiency

**Date:** 2026-08-17
**Question:** When fixed protocol bursts overfetch partial or tail requests, how should designers distinguish useful application bandwidth from occupied interface bandwidth?

## Hypothesis and realistic alternatives

The model already counted logical request bytes separately from occupied bytes, but its derived bandwidth and utilization fields used only logical bytes. That made fixed-burst overfetch visible as a counter while hiding it from the headline bandwidth metrics. A realistic comparison needs both views rather than redefining one metric:

| Interface contract | Why hardware might choose it | Expected sacrifice |
|---|---|---|
| Exact-byte occupancy (`none`, `fixed`, `idle_credit`, `burst_credit`) | Byte enables, coalescing, or a streaming fabric can occupy only valid bytes | Mask/coalescer state, irregular tails, and potentially more request-formation control |
| Fixed-burst occupancy (`burst_round_credit`) | DRAM-like interfaces use a simple minimum transfer granule | Partial and tail requests consume unused beats, bandwidth budget, and expected dynamic interface energy |

The cmodel now reports useful-payload bandwidth, occupied-interface bandwidth, useful and occupied utilization, and `payload_efficiency = useful_bytes / occupied_bytes`. The historical `effective_*_bandwidth` and `utilization` fields retain their useful-payload semantics for compatibility; new fields add the occupied view.

## Executable model

For an observation interval of `T` core cycles at core clock `f` GHz:

```text
useful_read_BW_GBps   = logical_read_bytes  / T * f
occupied_read_BW_GBps = occupied_read_bytes / T * f
useful_utilization     = (useful_read_BW + useful_write_BW) / peak_BW
occupied_utilization   = (occupied_read_BW + occupied_write_BW) / peak_BW
payload_efficiency     = (logical_read_bytes + logical_write_bytes)
                         / (occupied_read_bytes + occupied_write_bytes)
```

With no observed traffic, payload efficiency is reported as zero rather than inventing a perfect efficiency. Values are not clamped: this coarse invocation-time model can report utilization above 100% when callers issue requests without advancing simulation time. Such a result is an observation-window warning, not physical throughput.

## Measured matrix

Command: `make test-dram-turnaround-idle-sweep`

Configuration: one channel and bank, 64 B protocol burst, 8 B/cycle bus, paired opposite-direction requests. Exact-byte modes use request bytes as occupancy; `burst_round_credit` rounds each request to 64 B.

| Per-request payload | Exact occupied pair | Exact payload efficiency | Fixed-burst occupied pair | Fixed-burst payload efficiency | Lost occupied capacity |
|---:|---:|---:|---:|---:|---:|
| 16 B | 32 B | 100.0% | 128 B | 25.0% | 75.0% |
| 64 B | 128 B | 100.0% | 128 B | 100.0% | 0.0% |
| 80 B | 160 B | 100.0% | 256 B | 62.5% | 37.5% |

All 28 sweep rows fail closed on service cycles, turnaround cycles/events, occupied bytes, and payload efficiency. The focused DRAM suite additionally checks separate read/write useful and occupied GB/s, combined 50.0% efficiency for a 16 B read plus 80 B write (96 useful / 192 occupied), and occupied utilization greater than useful utilization.

## Gain versus sacrifice

- **Throughput:** Fixed bursts consume 4.0× occupied bandwidth for measured 16 B requests and 1.6× for 80 B tails. Sustained throughput remains unquantified because the model has no request queue, arbitration, or per-beat schedule.
- **Latency:** The metrics do not change service timing. Existing completion-boundary gates show that overfetch retains turnaround occupancy longer for partial/tail requests; the new fields expose the matching bandwidth cost.
- **Area/resources:** Fixed-granule transport generally simplifies request sizing and alignment logic. Exact-byte transport needs byte enables, masks, or coalescing buffers. Gate count and buffer area are unquantified.
- **Power/energy:** Lower payload efficiency directionally means more interface switching per useful byte. Physical DRAM/PHY energy is not wired to occupied bytes, so energy remains unquantified.
- **SRAM/DRAM traffic:** Logical tensor bytes do not change. Occupied interface bytes increase only for rounded partial/tail requests. The model does not determine whether overfetched bytes enter SRAM or are discarded by masks.
- **Numerical accuracy:** Unchanged; no payload values or arithmetic semantics change.
- **Control complexity:** Fixed bursts favor simple issue logic and software alignment. Exact-byte/coalesced operation shifts complexity to masks, merging, dependency checks, and buffering.
- **Verification burden:** Both useful and occupied read/write rates, combined efficiency, empty traffic, reset, aligned/sub-burst/tail requests, and both directions must remain gated.
- **Compiler/runtime:** Alignment and coalescing can recover efficiency, but can delay issue, enlarge dependency regions, require safe overfetch, and add buffering. The best policy depends on request-size distributions and tensor layout.

## Implementation and compatibility

- `tu_cmodel/memory/dram_model.h`: adds read/write occupied bandwidth, occupied utilization, and payload efficiency derived fields.
- `tu_cmodel/memory/dram_model.c`: derives and prints both useful and occupied metrics without changing request behavior.
- `tests/test_dram.c`: gates exact derived values and the useful-versus-occupied distinction.
- `tests/test_dram_turnaround_idle_sweep.c`: prints and exactly gates payload efficiency on all 28 existing completion/occupancy rows.

No configuration field, public function signature, numerical behavior, timing rule, or default architecture mode changed. Existing useful-bandwidth fields retain their values and names.

## Fidelity limits

These are deterministic observation metrics, not calibrated throughput or energy. The model omits request queues, injection timing, arbitration, reordering, command/address and data phasing, burst chopping, coalescing, byte masks, cache-line fills, bank groups, ranks, backpressure, PHY timing/energy, and calibration. Occupied utilization can exceed one because accesses do not advance `current_cycle`; users must choose a meaningful observation interval and must not interpret API issue rate as a feasible schedule.

## Verification

Executed for the focused implementation:

```text
make test-dram-turnaround-idle-sweep
make test-dram
```

The final clean build and quick regression are recorded in the heartbeat report.

## Actionable conclusion

Preserve both bandwidth views and both transport contracts. Useful bandwidth answers how quickly tensor payload is presented; occupied bandwidth answers how much modeled interface capacity the contract consumes. For 64 B-aligned requests they coincide. For measured 16 B and 80 B requests, fixed bursts reduce payload efficiency to 25.0% and 62.5%, respectively, in exchange for a simpler fixed-granularity interface contract. No mode is universally preferred: compiler coalescing, request sizes, area/control budgets, and unmodeled physical energy determine the appropriate design.
