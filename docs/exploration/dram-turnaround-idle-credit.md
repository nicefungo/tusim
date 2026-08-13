# DRAM Turnaround Idle-Time Credit

**Date:** 2026-08-13
**Question:** Should a direction change pay the full programmed bus-turnaround cost even when the channel has already been idle long enough to satisfy part or all of the guard interval?

## Hypothesis

The existing `fixed` model is a conservative request-order abstraction: every read/write direction change pays the full directional cost. That is useful for simple accounting and back-to-back traffic, but it overcharges a channel that has been idle after the prior transfer completed. A second physically plausible mode should credit elapsed channel-idle cycles against the guard interval while retaining `none` and full-cost `fixed` behavior.

## Runtime alternatives

| Mode | Why hardware/model teams might choose it | Sacrifice |
|---|---|---|
| `none` | Compatibility lower bound; separate paths; turnaround modeled elsewhere | Optimistic for shared bidirectional buses |
| `fixed` | Conservative, simple per-direction accounting without temporal credit | Overcharges direction changes after idle gaps |
| `idle_credit` | A shared-bus controller can satisfy guard time while no transfer occupies the channel | Depends on a service-completion abstraction; adds subtraction/state reasoning and verification |

All modes retain the existing symmetric/asymmetric costs and `core_cycles`/`physical_ns` domains. `none` remains the zero/default behavior.

## Executable model

For a direction change in `idle_credit` mode:

```text
idle_cycles = max(0, request_cycle - prior_channel_service_completion)
residual_turnaround = max(0, programmed_turnaround - idle_cycles)
```

The implementation uses the existing per-channel `channel_available_cycle`. It does not infer idle time from the previous API call alone. A direction-change event is still counted when the residual reaches zero, while the turnaround-cycle counter records only the residual service cost.

## Measured sweep

Command: `make test-dram-turnaround-idle-sweep`

Configuration: one channel/bank, read latency 10 cycles, write latency 8 cycles, R→W cost 3 cycles, W→R cost 8 cycles, row/refresh effects disabled. `gap` is the number of `tu_dram_tick()` calls between requests.

| Direction | Mode | Gap | Service sum | Turnaround cycles |
|---|---|---:|---:|---:|
| R→W | none | 0 | 18 | 0 |
| R→W | fixed | 0 | 21 | 3 |
| R→W | idle_credit | 0 | 21 | 3 |
| R→W | idle_credit | 10 | 21 | 3 |
| R→W | idle_credit | 11 | 20 | 2 |
| R→W | idle_credit | 13 | 18 | 0 |
| W→R | none | 0 | 18 | 0 |
| W→R | fixed | 0 | 26 | 8 |
| W→R | idle_credit | 0 | 26 | 8 |
| W→R | idle_credit | 8 | 26 | 8 |
| W→R | idle_credit | 12 | 22 | 4 |
| W→R | idle_credit | 16 | 18 | 0 |

All 12 rows are exact fail-closed gates.

## Findings and multi-objective trade-offs

- **Latency:** For back-to-back requests, `idle_credit` exactly matches full-cost `fixed`. Once idle extends past prior service completion, cost falls linearly: W→R is 26 cycles at gap 8, 22 at gap 12, and 18 at gap 16. This is not universally faster hardware; it is more accurate accounting for a temporal contract.
- **Throughput:** Unquantified. There is no queue, arbitration, or overlapping burst scheduler, so service sums are not achieved throughput.
- **Area/resources:** Expected incremental hardware over `fixed` is a timestamp/available-cycle comparison and saturating subtraction per scheduling decision if implemented literally. A real controller may already have this timing state. Gate count is unquantified.
- **Power/energy:** Payload traffic is unchanged. Avoiding artificial wait cycles can reduce modeled idle duration, but PHY switching/termination and controller energy are not wired; numeric energy is unquantified.
- **SRAM/DRAM traffic:** Unchanged bytes and addresses. Only returned channel service changes.
- **Numerical accuracy:** Unchanged.
- **Control complexity:** `none` is simplest; `fixed` needs direction history; `idle_credit` additionally reasons about prior service completion. Queue-aware completion order remains outside scope.
- **Verification burden:** Requires zero, partial, and full credit in both directions; back-to-back equality with `fixed`; default compatibility; parser/validation; reset and per-channel isolation inherited from the turnaround suite.
- **Compiler/runtime:** No numerical or ISA changes. A runtime that inserts independent work between opposite-direction transfers can expose idle credit, but the cmodel does not reorder requests or recommend padding solely to reduce turnaround.
- **Physical realism limits:** `channel_available_cycle` currently represents base returned service, not command/data-burst phase completion from a JEDEC scheduler. The model omits queues, bank timing, ranks, tWTR/tRTW decomposition, burst overlap, arbitration, PHY timing/energy, and calibration.

## Configuration and verification

```json
"turnaround_mode": "idle_credit",
"turnaround_domain": "core_cycles",
"read_to_write_turnaround": 3,
"write_to_read_turnaround": 8
```

Verified with:

```sh
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.idle-credit.h
make test-dram-turnaround-idle-sweep   # 12/12 exact rows
make test-dram                         # 32/32
make test-config                       # 30/30
make clean && make
make test-quick
```

## Actionable conclusion

Preserve all three modes. Use `fixed` for conservative request-order studies or when no trustworthy idle/completion timeline exists. Use `idle_credit` when the model's channel service-completion timestamp is an acceptable boundary. Do not interpret either as calibrated DRAM-command timing without a queue and command/data-bus contract.
