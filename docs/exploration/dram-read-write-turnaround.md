# DRAM Read/Write Bus Turnaround

**Date:** 2026-08-12
**Question:** How much does a shared bidirectional DRAM channel bus penalize mixed read/write traffic, and when can request batching avoid that cost?

## Hypothesis

A compatibility model with no direction-change cost is useful for historical results and fabrics with physically independent read/write paths. A shared bidirectional data bus should charge a per-channel penalty when service changes from read to write or write to read. Symmetric costs are a useful simple-controller abstraction; asymmetric costs represent realistic command/data-bus timing where read-to-write and write-to-read constraints differ. Read-only and write-only streams should be unaffected, alternating traffic should pay on every boundary, and batching should reduce the number of boundaries without changing payload bytes.

## Runtime alternatives

| Alternative | Why hardware might choose it | Main sacrifice |
|---|---|---|
| `none` | Backward compatibility; idealized lower bound; separate physical read/write paths or an upstream model that already includes direction costs | Optimistic for a shared bidirectional DRAM data bus under mixed traffic |
| `fixed`, symmetric | Small deterministic controller model with one common guard interval | Hides directional timing asymmetry and can over/understate one transition |
| `fixed`, asymmetric | Represents distinct read→write and write→read bus constraints | Two calibrated values, more state, tests, and scheduler awareness |
| `core_cycles` domain | Direct synchronous-controller counter; stable cycle budget | Physical interval changes with core clock |
| `physical_ns` domain | Keeps a physical turnaround interval stable across core-clock studies | Clock conversion and transition semantics; still not a DRAM command-clock model |

`none` plus zero costs is the zero/default behavior. All alternatives are runtime-configurable through canonical JSON/YAML and `tu_dram_set_turnaround()`.

## Executable model

Each channel stores its last serviced direction. In `fixed` mode, a direction change contributes the configured converted cost to returned service cycles and to dedicated event/cycle counters:

```text
R after W: base_read + write_to_read_turnaround
W after R: base_write + read_to_write_turnaround
same direction or first access: base latency only
```

The physical-ns domain converts with `ceil(ns × core_clock_ghz)`. Clock changes recompute both costs. The setter and parser fail without fallback for unsupported modes/domains, non-finite or out-of-range values, and a semantically empty `fixed` configuration with both costs zero. Reset and mode changes clear per-channel direction history.

## Measured sweep

Command: `make test-dram-turnaround-sweep`

Configuration: one channel, one bank, row policy and refresh disabled, 64-byte accesses, base read latency 10 cycles, base write latency 8 cycles. Returned service excludes the module's coarse bandwidth-window/channel-contention `stall_out` domain. The sequence is submitted in explicit program order; no request queue or reordering is modeled.

| Workload / operations | Mode (1 GHz) | R→W | W→R | Service cycles | Events | Turnaround cycles |
|---|---|---:|---:|---:|---:|---:|
| 8 reads | none | 0 | 0 | 80 | 0 | 0 |
| 8 reads | symmetric fixed | 5 | 5 | 80 | 0 | 0 |
| 8 reads | asymmetric fixed | 3 | 8 | 80 | 0 | 0 |
| 8 writes | none | 0 | 0 | 64 | 0 | 0 |
| 8 writes | symmetric fixed | 5 | 5 | 64 | 0 | 0 |
| 8 writes | asymmetric fixed | 3 | 8 | 64 | 0 | 0 |
| `RWRWRWRW` | none | 0 | 0 | 72 | 0 | 0 |
| `RWRWRWRW` | symmetric fixed | 5 | 5 | 107 | 7 | 35 |
| `RWRWRWRW` | asymmetric fixed | 3 | 8 | 108 | 7 | 36 |
| `RRRRWWWW` | none | 0 | 0 | 72 | 0 | 0 |
| `RRRRWWWW` | symmetric fixed | 5 | 5 | 77 | 1 | 5 |
| `RRRRWWWW` | asymmetric fixed | 3 | 8 | 75 | 1 | 3 |
| `WWWWRRRR` | none | 0 | 0 | 72 | 0 | 0 |
| `WWWWRRRR` | symmetric fixed | 5 | 5 | 77 | 1 | 5 |
| `WWWWRRRR` | asymmetric fixed | 3 | 8 | 80 | 1 | 8 |

Physical-ns conversion control for `RWRWRWRW`, asymmetric 3 ns / 8 ns:

| Core GHz | Converted R→W / W→R cycles | Service cycles | Events | Turnaround cycles |
|---:|---:|---:|---:|---:|
| 0.5 | 2 / 4 | 92 | 7 | 20 |
| 1.0 | 3 / 8 | 108 | 7 | 36 |
| 2.0 | 6 / 16 | 144 | 7 | 72 |

All 18 rows are exact fail-closed gates.

## Findings

1. Direction costs do not affect unidirectional streams: all read-only and write-only rows are byte- and cycle-identical to `none`.
2. Alternating traffic exposes the omitted behavior. Symmetric 5/5 raises service from 72 to 107 cycles (+48.6%); asymmetric 3/8 raises it to 108 (+50.0%) because seven direction changes occur.
3. Explicit batching reduces seven direction changes to one. `RRRRWWWW` with asymmetric timing is 75 cycles versus 108 alternating, a 30.6% lower service sum for the same operations and bytes. The reversed batch is 80 cycles because it pays the larger W→R cost. This is a sequence-local result, not a claim that a real scheduler can always reorder dependencies safely.
4. Neither symmetric nor asymmetric timing is universally better. The asymmetric setting is lower for a read-then-write batch and higher for write-then-read; the required values depend on the memory interface and timing contract.
5. Physical-ns costs scale 2/4 → 3/8 → 6/16 cycles across 0.5/1/2 GHz. The base latencies in this control remain fixed core cycles, so total physical time is intentionally not invariant; the control proves only turnaround-domain conversion.

## Gain versus sacrifice

- **Throughput:** Not quantified. There is no request queue, scheduler, or overlap model; service sums only show the pressure that repeated direction changes would create.
- **Latency:** Exact for the explicit deterministic sequence under this abstraction. Alternating asymmetric service is 108 cycles; read-first batching is 75 and write-first batching is 80.
- **Area/resources:** `fixed` adds one direction bit per channel, two programmed timing values, comparators/control, and counters. Actual gates/area are unquantified. Separate read/write physical fabrics may justify `none` but cost more pins/wires/PHY resources; those costs are unmodeled.
- **Power/energy:** Batching is expected to reduce bus-direction switching activity, but I/O termination, PHY, command, and controller energy are not wired. No numeric energy claim is made.
- **SRAM/DRAM traffic:** Payload bytes and addresses are unchanged. Only direction-change service is added. Command-bus occupancy and data-bus burst timing are not modeled.
- **Numerical accuracy:** Unchanged; data values and arithmetic are unaffected.
- **Control complexity:** `none` is simplest. Fixed asymmetric timing adds per-channel history. Exploiting batching requires queueing, dependency checks, fairness, and starvation policy not present here.
- **Verification burden:** Requires first-access, same-direction, both direction transitions, per-channel isolation, reset, failed-setter immutability, config propagation, and clock conversion tests.
- **Compiler/runtime:** A compiler or runtime may group compatible reads/writes to reduce transitions, but must preserve dependencies and latency/QoS constraints. Current cmodel callers define order directly; no automatic reordering is claimed.

## Fidelity limits

This is deterministic returned-service accounting, not a DRAM command scheduler. It does not model read/write queues, write-drain thresholds, arbitration, bank timing, command/data burst overlap, tWTR/tRTW decomposition, ranks, bank groups, PHY turnaround, bus training, backpressure, or calibration. The last direction follows API invocation order, not completion order from an out-of-order controller. Costs are per channel and are not included in `tu_dram_estimate_transfer()`, which has no prior-direction context. The coarse bandwidth-window stall remains separate in `stall_out` and `total_stall_cycles`.

## Configuration and verification

Configuration path:

```json
"turnaround_mode": "fixed",
"turnaround_domain": "physical_ns",
"read_to_write_turnaround": 3.0,
"write_to_read_turnaround": 8.0
```

Verification commands:

```sh
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.turnaround.h
make test-dram                       # 32/32
make test-config                     # 30/30
make test-dram-turnaround-sweep      # 18/18 rows
make config-docs
make clean && make
make test-quick
```

## Actionable conclusion

Preserve `none`, symmetric fixed, and asymmetric fixed settings. Use `none` only as a compatibility/ideal lower bound or when direction costs are modeled elsewhere. Use asymmetric settings when an interface timing contract is available. For mixed workloads, direction-aware batching is potentially valuable, but implementing an automatic write-drain scheduler is blocked until the cmodel has request queues, dependency/fairness semantics, and trace evidence.
