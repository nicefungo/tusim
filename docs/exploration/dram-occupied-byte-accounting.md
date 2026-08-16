# DRAM Occupied-Byte Accounting

**Date:** 2026-08-16
**Question:** When fixed protocol bursts enlarge a logical request, should the additional channel occupancy also be observable and consume the DRAM bandwidth window?

## Hypothesis and realistic alternatives

The existing `burst_round_credit` timing mode rounded sub-burst and tail requests when calculating the per-channel completion boundary, but the byte counters and coarse bandwidth window still consumed only requested bytes. That split understated traffic pressure and made the modeled sacrifice of fixed-burst hardware unobservable.

| Alternative | Hardware rationale | Modeled sacrifice |
|---|---|---|
| Requested/exact-byte occupancy | Byte-enabled or coalesced streaming fabrics can transfer only valid bytes | Needs masks/coalescing and potentially irregular final beats |
| Fixed-burst occupancy | DRAM-like interfaces may occupy a complete configured burst for every partial/tail request | Overfetch, additional bus activity, and extra bandwidth-window pressure |

This is not a new independent mode. It makes the existing `burst_round_credit` contract internally coherent: that mode rounds occupancy for completion timing, counters, pending traffic, and coarse bandwidth metering. NONE, FIXED, IDLE_CREDIT, and exact-byte BURST_CREDIT preserve requested-byte accounting and backward-compatible defaults.

## Executable model

For each read or write:

```text
logical_bytes = request_bytes
occupied_bytes =
    ceil(request_bytes / burst_length) * burst_length  [burst_round_credit]
    request_bytes                                      [all other modes]

pending_bytes += occupied_bytes
bandwidth_available -= occupied_bytes
total_{read,write}_bytes += logical_bytes
total_{read,write}_occupied_bytes += occupied_bytes
```

`total_read_bytes` and `total_write_bytes` remain logical API traffic. The new occupied-byte counters expose the modeled channel/bandwidth cost without pretending that all logical bytes are useful payload.

## Measured matrix

Command: `make test-dram-turnaround-idle-sweep`

Configuration: one channel/bank, 64 B protocol burst, 8 B/cycle channel width, two opposite-direction requests per row, read base 10 cycles, write base 8 cycles, R→W=3 cycles, W→R=8 cycles.

| Per-request payload | Exact-byte pair occupancy | Fixed-burst pair occupancy | Fixed-burst traffic increase | Completion behavior |
|---:|---:|---:|---:|---|
| 16 B | 32 B | 128 B | 300% (4.0×) | At W→R gap 20, exact has full credit while rounded retains 4 cycles |
| 64 B | 128 B | 128 B | 0% (1.0×) | Aligned control is identical |
| 80 B | 160 B | 256 B | 60% (1.6×) | Full W→R credit moves from gap 26 exact to gap 32 rounded |

The 28-row sweep now fails closed on service cycles, turnaround events/cycles, and occupied bytes. The focused DRAM suite also primes the bandwidth window for 1,001 cycles, submits 16 B read plus 80 B write, and proves separate logical counters (16/80 B), occupied counters (64/128 B), pending occupancy, and a 192 B bandwidth-budget decrement.

## Gain versus sacrifice

- **Throughput:** The rounded mode now consumes 4.0× the coarse bandwidth budget for measured 16 B requests and 1.6× for 80 B requests. Sustained throughput remains unquantified because the bandwidth window is coarse and there is no request queue or arbitration schedule.
- **Latency:** Completion-boundary results are unchanged: rounded occupancy retains turnaround cost longer for partial/tail traffic. The new accounting also permits later bandwidth-window stalls when such traffic accumulates, but this exploration does not claim a queue-aware makespan.
- **Area/resources:** Fixed-burst accounting corresponds to simple align-up logic; exact-byte transport generally needs byte enables, masks, or a coalescer. Gate count and buffering are unquantified.
- **Power/energy:** More occupied bytes directionally imply more interface switching and potentially DRAM activation energy. The power model is not wired to these counters, so energy is unquantified.
- **SRAM/DRAM traffic:** Logical bytes remain unchanged. Occupied channel bytes rise only in fixed-burst mode for partial/tail requests; aligned transfers are identical.
- **Numerical accuracy:** Unchanged. No data values or arithmetic semantics change.
- **Control complexity:** Fixed bursts simplify the transport contract but encourage request alignment and merging. Exact-byte operation shifts complexity into masks/coalescing.
- **Verification burden:** Separate read/write logical and occupied counters, reset behavior, aligned/sub-burst/tail cases, both directions, and bandwidth-window consumption must remain gated.
- **Compiler/runtime:** Alignment and coalescing can reduce fixed-burst overhead, but may add buffering, overfetch safety checks, dependency tracking, and latency before a request is issued.

## Implementation and compatibility

- `tu_cmodel/memory/dram_model.h`: adds `total_read_occupied_bytes` and `total_write_occupied_bytes`.
- `tu_cmodel/memory/dram_model.c`: uses one occupancy helper for timing, pending traffic, bandwidth consumption, counters, and stats printing.
- `tests/test_dram_turnaround_idle_sweep.c`: reports and gates `occ_B` for all 28 rows.
- `tests/test_dram.c`: gates logical/occupied separation and bandwidth-window consumption.

No public function signature, runtime configuration value, or zero/default behavior changed. Existing logical byte counters retain their semantics.

## Fidelity limits

Occupied bytes are a deterministic model contract, not measured physical DRAM traffic. The model still omits burst chopping, request coalescing, byte masks, command/address and data phasing, read/write data timing, queues, reordering, bank groups, ranks, arbitration, PHY energy, and calibration. The coarse bandwidth window does not model per-beat scheduling or makespan. `effective_*_bandwidth` remains based on logical bytes, so it is useful-payload bandwidth rather than occupied-wire bandwidth.

## Verification

Executed:

```text
make test-dram-turnaround-idle-sweep
make test-dram
```

Final aggregate build and quick-suite commands are recorded in the heartbeat report.

## Actionable conclusion

Preserve both contracts. Exact-byte occupancy is appropriate for byte-enabled/coalesced links; fixed-burst occupancy is appropriate when partial and tail requests consume complete protocol transfers. The cmodel now exposes the traffic cost that buys the simpler fixed-granularity contract rather than reporting only its turnaround timing effect. Neither is universally preferable: request-size distribution and compiler/runtime coalescing determine the regime, while physical energy and queue-aware throughput remain unquantified.
