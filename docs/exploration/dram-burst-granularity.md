# DRAM Serialized Occupancy Granularity

**Date:** 2026-08-15
**Question:** Should a serialized DRAM completion boundary occupy only valid payload beats, or round sub-burst and tail requests to the interface's fixed burst granule?

## Hypothesis and realistic alternatives

The prior `burst_credit` model uses `ceil(payload_bytes / bus_width_bytes)`. That is a plausible byte-stream or coalesced-link contract, but the configured DRAM model also exposes `burst_length`. A fixed-burst interface can occupy a complete transfer granule even when software requests fewer valid bytes or leaves a partial final burst.

| Mode | Why a hardware/model team might choose it | Sacrifice |
|---|---|---|
| `none` | Compatibility lower bound or physically independent read/write paths | Omits shared-bus turnaround |
| `fixed` | Conservative direction-boundary charge without completion timestamps | Cannot credit real idle time |
| `idle_credit` | Simple aggregate-service boundary when payload movement is modeled elsewhere | Can release a serialized bus early |
| `burst_credit` | Exact-byte occupancy for byte enables, coalesced links, or abstract streaming fabrics | Understates fixed minimum-transfer granularity |
| `burst_round_credit` | Fixed protocol bursts for DRAM-like sub-burst and tail transfers | Can overstate occupancy when requests coalesce or masks suppress beats |

All modes remain runtime configurable. `none` remains the zero/default compatibility behavior.

## Executable model

`burst_round_credit` uses the existing residual-turnaround accounting but rounds occupied bytes before serialization:

```text
occupied_bytes = ceil(payload_bytes / burst_length) * burst_length
burst_cycles = ceil(occupied_bytes / bus_width_bytes)
completion = request_cycle + base_service + burst_cycles
residual_turnaround = max(0, programmed_turnaround - max(0, next_request - completion))
```

`burst_credit` retains `occupied_bytes = payload_bytes`. The completion term affects per-channel availability and future idle credit; it is not added to the current request's returned service cycles, preserving the module's established service-versus-contention domains. As of the occupied-byte follow-up, the same rounded byte count also feeds explicit occupied-byte counters, pending traffic, and the coarse bandwidth window; logical request-byte counters remain unchanged.

## Measured matrix

Command: `make test-dram-turnaround-idle-sweep`

Configuration: one channel/bank, 64 B protocol burst, 8 B/cycle channel width, read base 10 cycles, write base 8 cycles, R→W=3 cycles, W→R=8 cycles, no row or refresh effects.

| Direction | Mode | Gap | Payload | Occupied bytes | Service sum | Turnaround cycles |
|---|---|---:|---:|---:|---:|---:|
| W→R | `burst_credit` | 20 | 16 B | 16 B | 18 | 0 |
| W→R | `burst_round_credit` | 20 | 16 B | 64 B | 22 | 4 |
| W→R | `burst_round_credit` | 24 | 16 B | 64 B | 18 | 0 |
| W→R | `burst_credit` | 20 | 80 B | 80 B | 24 | 6 |
| W→R | `burst_credit` | 26 | 80 B | 80 B | 18 | 0 |
| W→R | `burst_round_credit` | 20 | 80 B | 128 B | 26 | 8 |
| W→R | `burst_round_credit` | 28 | 80 B | 128 B | 22 | 4 |
| W→R | `burst_round_credit` | 32 | 80 B | 128 B | 18 | 0 |

The complete sweep contains 28 exact fail-closed rows and retains NONE, FIXED, IDLE_CREDIT, exact-byte BURST_CREDIT, aligned 64 B controls, and rounded-mode controls in both directions. Aligned 64 B transfers are identical in both serialized modes.

## Gain versus sacrifice

- **Latency:** For the measured sub-burst request, protocol rounding retains 4 turnaround cycles at gap 20 where exact-byte occupancy has full credit. For 80 B, the full-credit point moves from gap 26 to 32. These are exact deterministic model results, not calibrated DRAM timings.
- **Throughput:** Unquantified. Rounded occupancy is directionally more conservative for partial requests, but the cmodel has no queue, arbitration, coalescer, or makespan scheduler. Returned service sums must not be presented as sustained throughput.
- **Area/resources:** Exact-byte occupancy requires byte-count/width ceiling logic. Burst rounding additionally requires alignment to `burst_length`; power-of-two bursts can use masks/shifts. Gate count, buffering, and timing are unquantified.
- **Power/energy:** A fixed-burst interface is expected to activate/drive more beats for partial requests, increasing dynamic interface and possibly DRAM energy. The power model is not connected to occupied beats, so no numeric energy claim is made.
- **SRAM/DRAM traffic:** Requested byte counters remain unchanged. Separate occupied-byte counters now expose 4.0× pair traffic for measured 16 B requests and 1.6× for 80 B tails, and rounded occupancy consumes the coarse bandwidth window. These counters are a deterministic interface contract, not calibrated physical DRAM traffic.
- **Numerical accuracy:** Unchanged; data values and arithmetic paths are unaffected.
- **Control complexity:** Exact-byte mode fits byte-enable/coalescing fabrics. Rounded mode is simpler as a fixed protocol contract but shifts optimization pressure to request merging and alignment.
- **Verification burden:** The rounded alternative requires aligned, sub-burst, multi-burst-tail, zero/partial/full-credit, both-direction, parser, generator, default, and failed-setter gates.
- **Compiler/runtime:** Compilers and DMA runtimes can reduce rounded occupancy by aligning and coalescing adjacent requests, at the cost of larger transactions, buffering, dependency handling, and possibly overfetch. No ISA change is required for mode selection in this cmodel.

## Configuration path

```json
"turnaround_mode": "burst_round_credit",
"turnaround_domain": "core_cycles",
"read_to_write_turnaround": 3,
"write_to_read_turnaround": 8
```

The burst granule and channel width come from the selected `tu_dram_params_t`. The canonical JSON parser, YAML generator, checked-in generated header, runtime creation path, generated config reference, and zero/default compatibility path are gated.

## Fidelity limits

This is deterministic channel-occupancy accounting, not a JEDEC command scheduler. It does not model request coalescing, byte masks, cache-line fill semantics, burst chopping, command/address versus data phasing, read/write data timing, bank groups, ranks, queues, reordering, fairness, backpressure, PHY termination/training, physical-byte energy, or calibration. Occupied bytes now feed the existing coarse bandwidth window, but its stall output remains separate from returned service cycles and is not a queue-aware makespan.

## Verification

Executed:

```text
make test-dram-turnaround-idle-sweep
make test-dram
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make config-docs
```

The generated temporary header contains `TU_DRAM_TURNAROUND_MODE_BURST_ROUND_CREDIT 4`; the shipped JSON continues to load through `test-config` with the legacy `none` default.

## Actionable conclusion

Preserve both serialized alternatives. Use `burst_credit` for byte-stream, byte-enabled, or externally coalesced occupancy studies. Use `burst_round_credit` when the modeled interface pays a fixed transfer granule for partial/tail requests. Do not select either universally: the right contract depends on request formation and interface protocol. Occupied traffic and coarse bandwidth pressure are now observable; queue-aware throughput, coalescing, and physical energy remain unquantified. See `dram-occupied-byte-accounting.md`.
