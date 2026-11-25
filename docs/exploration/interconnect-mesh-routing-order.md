# Deterministic Mesh Routing Order: XY vs YX

**Date:** 2026-07-25
**Status:** Implemented and verified
**Type:** Cmodel-linked simultaneous traffic-matrix sweep

## Design question

How does X-then-Y (XY) versus Y-then-X (YX) change the shared-link score for asymmetric placements?

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
- Contention: `shared_link` deterministic route-load score, not a bound.
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

- **Route load:** On 4×4 top-row-to-left-column fan-in the YX score is 798 versus 2,334 for XY. Transposing the traffic reverses the scores; symmetric all-to-all ties. The 65.8% score reduction is not a latency reduction.
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
- JSON/YAML: `tu.multicore.mesh_routing = "xy" | "yx" | "xy_else_yx" | "yx_else_xy"`. The last two change order only if the preferred turn router is absent.
- `scripts/gen_config.py` emits the switching, contention, link, router, and mesh-routing constants/runtime fields from the YAML source.
- Cluster state: `mesh_routing_mode`.
- `tu_cluster_estimate_traffic_cycles()` now enumerates either X-first or Y-first minimal routes; the previous XY behavior remains the default.
- Invalid route names and invalid runtime values are rejected. RING behavior is unchanged.

## Verification

- `make test-config`: 20/20 passed; covers JSON parsing, canonical-to-runtime propagation, default-compatible fields, and unsupported routing rejection.
- `make test-multicore`: 16/16 passed; focused 4×4 asymmetric traffic proves XY's 576-cycle versus YX's 192-cycle bottleneck for 1 KiB messages, expected total estimates (606 vs 222), and deterministic bottleneck selection.
- `make test-interconnect-routing-sweep`: reproduced all 12 rows above.

## Limitations and next questions

These are fixed-route load comparisons, not router simulations. See `interconnect-traffic-floor.md` for the separate necessary-time metric. Queue depth, flit size, injection timestamps, arbitration, virtual channels, backpressure, physical wire length, and deadlock behavior remain unmodeled. Non-square and irregular meshes were not included in this sweep. Trace-driven injection is needed before comparing adaptive routing or finite queues.

## Partial last row: does a transfer have a route?

The complete rectangle above hid an admission discrepancy. In a two-row mesh populated as `0 1 2 / 3 4 –`, the original XY batch estimator rejected `3→2` and `4→2` because the horizontal leg crossed the absent router 5. Direct sends copied the bytes and charged cycles anyway. YX had the same discrepancy in the reverse directions. The pre-fix executable gate observed all four disagreements. Manhattan distance is still the geometric minimum; it is not evidence that a selected route exists.

The single-message estimate, nonempty direct send and traffic estimator now use the same turn check. Since the footprint is a row-major prefix, both straight legs exist exactly when their common turn exists. Strict orders reject a missing turn. The two optional orders try the other dimension first **only** in that case; they never inspect load. There are no hidden transit routers. The existing zero-byte direct send remains an endpoint-validated no-op; the route-planning APIs require a valid route. Broadcast still delivers a sequence of sends; [broadcast admission](interconnect-broadcast-admission.md) compares retaining a prefix with checking the entire destination set before the first send.

| `mesh_routing` | 5 populated routers: admitted / rejected | Full 6 routers: admitted / rejected |
|---|---:|---:|
| `xy` (default) | 23 / 2 | 36 / 0 |
| `yx` | 23 / 2 | 36 / 0 |
| `xy_else_yx` | 25 / 0 | 36 / 0 |
| `yx_else_xy` | 25 / 0 | 36 / 0 |

Every ordered pair including self, one 4-byte message, cut-through at 16 B/cycle plus 5 cycles/hop. Both fallback modes use minimal paths: restoring a connection is **not** accelerating an existing one. The known-route cost stays `5 × hops + 1` for nonlocal sends; a local copy charges no link cycles. Admitted sends move four useful bytes at each SRAM endpoint, with no arithmetic, accuracy change or DRAM request. Changing the preferred order may redistribute link traffic without changing hop count. There is no measured sustained throughput or physical latency.

Strict XY/YX fit a fixed turn rule and predictable placement; on a partial mesh software must avoid or explicitly relay unsupported pairs. A fallback implementation can encode the chosen order at injection to retain those placements, at the expected cost of a turn-existence check or route table and additional verification of mixed orders. Quantities for logic area, energy, timing and any virtual channels are unknown. For a physical buffered design, route connectivity is not proof of progress: channel dependencies need a separate check (Dally & Seitz, *Deadlock-Free Message Routing in Multiprocessor Interconnection Networks*, IEEE Trans. Computers, 1987). Arbitrary holes, relay-only routers, congestion feedback and deadlock-free finite-buffer operation are outside this footprint and resource contract.

`make test-icc-ragged-mesh` statically links the model and runs the JSON-to-cluster path for all four modes. It checks admission, transfer time, destination bytes, message/byte/cycle counters and directed link use against an independent step-by-step oracle. It also checks invalid names/types/IDs and default XY. `./test-icc-ragged-mesh` prints the eight-row matrix above. Rerun `make test-interconnect-routing-sweep test-interconnect-contention-sweep test-icc-traffic-floor` for complete-mesh controls.
