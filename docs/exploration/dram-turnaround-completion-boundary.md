# DRAM Turnaround Completion Boundary

**Date:** 2026-08-14
**Question:** When idle time is credited against a read/write turnaround guard, should channel availability begin after aggregate base service or after a serialized payload burst?

## Hypothesis and alternatives

The existing `idle_credit` mode uses `current_cycle + base_service` as the prior completion boundary. That lightweight abstraction is plausible when payload transfer is overlapped or represented elsewhere, but it can credit time while a conservative shared data bus would still be serializing bytes. A second mode should retain the simple boundary and add a payload-dependent alternative:

| Mode | Why a hardware/model team might choose it | Sacrifice |
|---|---|---|
| `none` | Compatibility lower bound or physically separate paths | Omits shared-bus direction cost |
| `fixed` | Conservative request-order accounting without temporal trust | Overcharges after genuine idle |
| `idle_credit` | Low-complexity aggregate service boundary; payload overlap modeled elsewhere | Can release the channel too early for serialized bursts |
| `burst_credit` | Conservative shared-bus boundary: base service plus exact payload serialization | May overstate occupancy when latency and burst overlap; understates fixed minimum bursts |
| `burst_round_credit` | Fixed-granularity protocol occupancy for sub-burst and tail requests | Can overstate occupancy when byte masks or coalescing avoid full bursts |

`none` remains the zero/default. All alternatives remain runtime selectable. The rounded follow-up is measured in `dram-burst-granularity.md`.

## Executable model

`burst_credit` reuses idle-credit residual subtraction but sets per-channel availability to:

```text
completion = request_cycle + base_service + ceil(payload_bytes / bus_width_bytes)
residual_turnaround = max(0, programmed_turnaround - max(0, next_request - completion))
```

The added serialization term affects the completion boundary and coarse channel stall state; it is not added to the current access's returned service cycles. This preserves the module's existing separation between returned service and coarse contention accounting.

## Measured matrix

Command: `make test-dram-turnaround-idle-sweep`

Configuration: one channel/bank, 8 B/cycle bus, read base 10 cycles, write base 8 cycles, R→W=3, W→R=8, no row/refresh effects.

| Direction | Mode | Gap | Bytes | Service sum | Turnaround cycles |
|---|---|---:|---:|---:|---:|
| W→R | `idle_credit` | 16 | 64 | 18 | 0 |
| W→R | `burst_credit` | 16 | 64 | 26 | 8 |
| W→R | `burst_credit` | 20 | 64 | 22 | 4 |
| W→R | `burst_credit` | 24 | 64 | 18 | 0 |
| W→R | `burst_credit` | 20 | 16 | 18 | 0 |
| R→W | `burst_credit` | 13 | 64 | 21 | 3 |
| R→W | `burst_credit` | 21 | 64 | 18 | 0 |

The full sweep retains the prior NONE/FIXED/IDLE_CREDIT controls and passes 18 exact fail-closed rows.

## Multi-objective findings

- **Latency:** For 64 B, burst serialization is 8 cycles. Full W→R credit therefore moves from gap 16 under base-service credit to gap 24 under burst credit. A 16 B payload needs only 2 serialization cycles and is fully credited by gap 20. These are exact results for this deterministic abstraction, not calibrated DRAM latency.
- **Throughput:** Unquantified. The cmodel has no request queue, arbitration, or overlap scheduler; service sums and availability are not a makespan.
- **Area/resources:** Expected hardware/model increment is payload-size/width accounting plus a later availability timestamp. Constant power-of-two widths can use shifts; gate count and storage are unquantified.
- **Power/energy:** Payload bytes are unchanged. Longer modeled occupancy may imply more PHY active time, but PHY/controller energy is not wired, so numeric energy is unavailable.
- **SRAM/DRAM traffic:** Addresses and bytes are unchanged. Only the temporal boundary differs.
- **Numerical accuracy:** Unchanged.
- **Control complexity:** `burst_credit` needs transfer-size-aware completion. `idle_credit` remains useful where another subsystem owns serialization or overlap.
- **Verification burden:** Both directions, different payload sizes, exact-boundary behavior, zero/partial/full credit, parser, generated constants, default compatibility, and failed-setter behavior are gated.
- **Compiler/runtime:** No ISA or arithmetic change. Scheduling studies must select a boundary matching the intended memory interface; software should not infer queue-aware batching benefits from this model.

## Fidelity limits

The exact-byte serialization term is deliberately conservative and uses `ceil(bytes / bus_width)`. The `burst_round_credit` follow-up additionally rounds occupied bytes to `burst_length`; see `dram-burst-granularity.md`. Neither mode models coalescing, byte masks, burst chopping, command/data phase overlap, read/write data timing, ranks, bank groups, request queues, reordering, fairness, backpressure, PHY training/termination, energy, or calibration. These modes are not JEDEC command schedulers.

## Configuration and verification

```json
"turnaround_mode": "burst_credit",
"turnaround_domain": "core_cycles",
"read_to_write_turnaround": 3,
"write_to_read_turnaround": 8
```

Verified with `make test-dram-turnaround-idle-sweep`, `make test-dram`, `make test-config`, generated-header inspection, `make clean && make`, and `make test-quick`.

## Actionable conclusion

Preserve both completion boundaries. Use `idle_credit` when aggregate service completion is the intended contract or burst occupancy is modeled elsewhere. Use `burst_credit` for a conservative serialized shared-bus study. Neither should be promoted to physical timing without command/data overlap and queue contracts.