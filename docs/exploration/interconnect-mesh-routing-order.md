# Deterministic Mesh Routing Order: XY vs YX

**Date:** 2026-07-25
**Status:** Implemented and verified
**Type:** Cmodel-linked simultaneous traffic-matrix sweep

## Design question

Can a fixed dimension-order route be selected independently of topology, and how much can choosing X-then-Y (XY) versus Y-then-X (YX) change the finite shared-link lower bound for asymmetric compiler placements?

The hypothesis was that neither order is generally preferable: both use minimal Manhattan paths and therefore have the same isolated hop count, but each concentrates an L-shaped fan-in on a different first dimension. A workload aligned with one order should favor it and the transposed workload should favor the other.

## Physically plausible alternatives retained

| Runtime mode | Hardware rationale | Gain regime | Sacrifice |
|---|---|---|---|
| `xy` (default) | Conventional deterministic dimension-order routing; simple local next-hop logic and a static deadlock-avoidance order | Traffic that should spread vertically before converging horizontally, including the measured left-column-to-bottom-row pattern | Can funnel top-row-to-left-column traffic through a shared horizontal/vertical corner |
| `yx` | Equally minimal deterministic order with the axis priority reversed; useful when placement or physical link provisioning favors the other dimension | Traffic that should spread horizontally before converging vertically, including the measured top-row-to-left-column pattern | Produces the transpose of XY's hotspot behavior |

Both modes are intentionally preserved. A physical mesh would normally hard-wire one ordering (or use separate virtual networks); the pre-spec cmodel keeps both so compiler placement and NoC orientation can be co-designed. Adaptive minimal routing is not represented by YX and remains out of scope because it needs queue state, arbitration, virtual-channel, and deadlock contracts.

## Executable contract

- Meshes: square 3×3 and 4×4.
- Switching: cut-through.
- Contention: shared directed-link lower bound.
- Link payload width: 16 B/cycle.
- Router latency: 5 cycles/hop.
- Payload: 4,096 B/message.
- Injection: every message in a row is simultaneous.
- Routes: minimal deterministic XY or YX; links are directional.
- Command: `make test-interconnect-routing-sweep`.

The reported estimate remains:

`max(max_isolated, max_directed_link_serialization + max_route_latency)`

Changing routing order changes directed-link occupancy, not hop count or payload bytes.

## Measured matrix

| Mesh | Traffic matrix | Route | Messages | Estimated cycles | Bottleneck serialization | Bottleneck link |
|---|---|---|---:|---:|---:|---|
| 3×3 | top row → left column | XY | 4 | 1,044 | 1,024 | 0→3 |
| 3×3 | top row → left column | YX | 4 | 532 | 512 | 1→4 |
| 3×3 | left column → bottom row | XY | 4 | 532 | 512 | 0→1 |
| 3×3 | left column → bottom row | YX | 4 | 1,044 | 1,024 | 3→6 |
| 3×3 | all-to-all | XY | 72 | 1,556 | 1,536 | 0→1 |
| 3×3 | all-to-all | YX | 72 | 1,556 | 1,536 | 0→1 |
| 4×4 | top row → left column | XY | 9 | 2,334 | 2,304 | 0→4 |
| 4×4 | top row → left column | YX | 9 | 798 | 768 | 1→5 |
| 4×4 | left column → bottom row | XY | 9 | 798 | 768 | 0→1 |
| 4×4 | left column → bottom row | YX | 9 | 2,334 | 2,304 | 8→12 |
| 4×4 | all-to-all | XY | 240 | 4,126 | 4,096 | 1→2 |
| 4×4 | all-to-all | YX | 240 | 4,126 | 4,096 | 1→2 |

## Findings: gains and sacrifices

- **Throughput/latency:** On the 4×4 top-row-to-left-column fan-in, YX lowers the executable shared-link bound by 65.8% (798 vs 2,334 cycles); for the transposed left-column-to-bottom-row traffic, XY provides the identical 65.8% reduction. On symmetric all-to-all, both are identical. These are lower-bound cycle estimates under simultaneous injection, not silicon throughput.
- **No universal route winner:** The exact reversal under transposition confirms that axis order must follow placement and traffic orientation. Selecting YX globally from the first result would simply move the hotspot.
- **Area/resources:** Fixed XY and fixed YX require comparable coordinate-compare and mux logic. Supporting both at runtime needs a route-mode bit/control path and validation state; physical area is expected to increase slightly but is **unquantified**.
- **Power/energy:** Hop count and endpoint bytes are unchanged. Different link concentration can change dynamic router/link utilization and clock activity, but routed energy is **unquantified** because the power model is not wired to link occupancy.
- **SRAM/DRAM traffic:** Endpoint SRAM/DRAM bytes are unchanged. Router-buffer traffic is **unmodeled**; no claim is made about FIFO reads/writes or retransmissions.
- **Numerical accuracy:** Unchanged; route selection affects timing accounting only and never modifies payload data.
- **Control complexity:** Both fixed orders are simpler than adaptive routing. Runtime-selectable order adds one architectural mode; a deployed design may hard-wire one to reduce control and deadlock-verification scope.
- **Verification burden:** Each mode needs deterministic path tests, asymmetric and transposed traffic tests, config parse/default/error checks, and non-regression on symmetric traffic. Adaptive routing would add fairness and liveness obligations not present here.
- **Compiler/runtime:** Placement should treat mesh orientation as architectural state. Compiler tiling or collective lowering can transpose a communication pattern, select a matching route mode when hardware permits, or choose roots that avoid a fixed-order corner hotspot.

## Implementation

- Compile-time/default constants: `TU_ICC_MESH_ROUTE_XY`, `TU_ICC_MESH_ROUTE_YX`, and `TU_ICC_MESH_ROUTING_MODE`.
- Runtime/canonical field: `icc_mesh_routing_mode`.
- JSON/YAML: `tu.multicore.mesh_routing = "xy" | "yx"`.
- `scripts/gen_config.py` emits the switching, contention, link, router, and mesh-routing constants/runtime fields from the YAML source.
- Cluster state: `mesh_routing_mode`.
- `tu_cluster_estimate_traffic_cycles()` now enumerates either X-first or Y-first minimal routes; the previous XY behavior remains the default.
- Invalid route names and invalid runtime values are rejected. RING behavior is unchanged.

## Verification

- `make test-config`: 20/20 passed; covers JSON parsing, canonical-to-runtime propagation, default-compatible fields, and unsupported routing rejection.
- `make test-multicore`: 16/16 passed; focused 4×4 asymmetric traffic proves XY's 576-cycle versus YX's 192-cycle bottleneck for 1 KiB messages, expected total estimates (606 vs 222), and deterministic bottleneck selection.
- `make test-interconnect-routing-sweep`: reproduced all 12 rows above.
- Final clean build and quick regression are recorded in the heartbeat report.

## Limitations and next questions

This is still deterministic lower-bound routing, not a router simulator. Queue depth, flit size, injection timestamps, arbitration, virtual channels, backpressure, head-of-line blocking, physical wire length, and deadlock behavior remain unmodeled. Non-square and irregular meshes use the same implementation but were not included in this sweep. A next routing study should use compiler or collective traces before adding adaptive routing or finite queues; otherwise queue policies would be optimized against arbitrary synthetic bursts.
