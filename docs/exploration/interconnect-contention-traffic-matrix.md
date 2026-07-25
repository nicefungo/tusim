# Interconnect Shared-Link Contention by Traffic Matrix

**Date:** 2026-07-24
**Status:** Implemented and verified
**Type:** Cmodel-linked simultaneous traffic-matrix sweep

## Design question

How much did the isolated-transfer topology and switching sweeps understate latency when multiple TU cores inject tensor messages onto the same finite directed links, and can a mesh still be called universally preferable to a ring once traffic shape is included?

The hypothesis was that topology conclusions would reverse by traffic matrix: a mesh has more links and shorter average routes, while a bidirectional ring can split traffic around two directions and may avoid a deterministic mesh route's hotspot.

## Alternatives retained

| Runtime mode | Model | Why a hardware/modeling team might choose it | Sacrifice |
|---|---|---|---|
| `ideal_parallel` (default) | Maximum isolated message latency; links do not contend | Backward-compatible lower bound for functional/compiler work, or approximation of a heavily overprovisioned/local fabric | Optimistic for collectives, fan-in, and all-to-all traffic |
| `shared_link` | Maximum of isolated latency and `bottleneck directed-link serialization + longest route latency` | Finite-width shared-link lower bound for early topology and placement studies | Still omits queue order, buffers, arbitration, VCs, and head-of-line blocking, so it is not cycle-accurate |

Both are model-fidelity alternatives rather than a claim that real links can literally have infinite capacity. The default remains `ideal_parallel` to preserve existing behavior. `shared_link` is conservative only relative to the old independent-transfer assumption; because it is a lower bound, a real congested fabric can be slower.

## Routing and timing contract

- Switching: cut-through
- Link payload width: 16 B/cycle
- Router latency: 5 cycles/hop
- Payload: 4,096 B/message
- Injection: all messages in a row are simultaneous
- Ring routing: shortest path, clockwise on equal-distance ties
- Mesh routing: deterministic dimension-order XY
- Links are directional; opposite directions do not contend
- Command: `make test-interconnect-contention-sweep`

For each message, the existing isolated cut-through equation is:

`isolated = hops × router_latency + ceil(bytes / link_bytes_per_cycle)`

For a traffic matrix, the shared-link lower bound is:

`max(max_isolated, max_directed_link_serialization + max_route_latency)`

This equation is exact for the focused one-link overlap/disjoint-link tests but intentionally does not claim packet/flit scheduling accuracy for multi-hop queues.

## Measured matrix

| Cores | Topology | Traffic | Messages | Ideal cycles | Shared-link cycles | Shared / ideal | Bottleneck serialization | Bottleneck link |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 8 | RING | neighbor | 8 | 261 | 261 | 1.00× | 256 | 0→1 |
| 8 | RING | hotspot-to-0 | 7 | 276 | 1,044 | 3.78× | 1,024 | 7→0 |
| 8 | RING | all-to-all | 56 | 276 | 2,580 | 9.35× | 2,560 | 0→1 |
| 8 | MESH 2×4 | neighbor | 8 | 276 | 276 | 1.00× | 256 | 0→1 |
| 8 | MESH 2×4 | hotspot-to-0 | 7 | 276 | 1,044 | 3.78× | 1,024 | 4→0 |
| 8 | MESH 2×4 | all-to-all | 56 | 276 | 2,068 | 7.49× | 2,048 | 1→2 |
| 16 | RING | neighbor | 16 | 261 | 261 | 1.00× | 256 | 0→1 |
| 16 | RING | hotspot-to-0 | 15 | 296 | 2,088 | 7.05× | 2,048 | 15→0 |
| 16 | RING | all-to-all | 240 | 296 | 9,256 | 31.27× | 9,216 | 0→1 |
| 16 | MESH 4×4 | neighbor | 16 | 286 | 286 | 1.00× | 256 | 0→1 |
| 16 | MESH 4×4 | hotspot-to-0 | 15 | 286 | 3,102 | 10.85× | 3,072 | 4→0 |
| 16 | MESH 4×4 | all-to-all | 240 | 286 | 4,126 | 14.43× | 4,096 | 1→2 |

## Findings: gains and sacrifices

- **No universal topology winner:** At 16 cores, the 4×4 mesh lowers the all-to-all shared-link lower bound by 55.4% versus the ring (4,126 vs 9,256 cycles), but it is 48.6% slower for hotspot-to-core-0 traffic (3,102 vs 2,088 cycles). The ring's two shortest-path directions split fan-in; deterministic XY concentrates mesh fan-in on directed link 4→0.
- **Neighbor traffic:** Both topologies remain at the ideal bound because this workload maps one message per bottleneck directed link. The ring is 8.7% lower latency than the 4×4 mesh row here (261 vs 286 cycles) because the synthetic mesh neighbor sequence includes a longer wrap route; placement, not raw topology strength, causes the difference.
- **Contention dominates scale:** The isolated model hides a 31.27× lower-bound penalty for 16-core ring all-to-all and 14.43× for the mesh. These are not measured silicon slowdowns; they are the ratio between two executable cmodel abstractions under the stated simultaneous-injection contract.
- **Throughput/latency:** Mesh link multiplicity helps distributed all-to-all, while ring route splitting helps this single-root fan-in. Exact sustained throughput remains **unquantified** without injection timing and queue service.
- **Area/resources:** A mesh needs more router ports and physical links; a ring needs fewer links but may need wider/faster links to handle global traffic. Link/router area is **unquantified**.
- **Power/energy:** Mesh may reduce hops for distributed traffic but has more routers/links and clocked state. Ring all-to-all drives higher bottleneck serialization. Dynamic and leakage energy are **unquantified** because the power model is not wired to routed-link occupancy.
- **SRAM/DRAM traffic:** Endpoint bytes are unchanged. Router-buffer and retry traffic are **unmodeled**; `bottleneck_link_cycles` observes serialization service, not physical buffer accesses.
- **Numerical accuracy:** Unchanged; this API estimates timing only and does not alter payloads.
- **Control complexity:** `shared_link` exposes why finite arbitration matters but does not implement it. Real cut-through hardware still needs credits, VCs, deadlock avoidance, and fair arbitration.
- **Verification burden:** Deterministic routes make link-load accounting reproducible. Adaptive routing would require path-selection and fairness invariants plus adversarial traffic tests.
- **Compiler/runtime:** Placement and collective selection must be traffic-aware. A compiler should not infer that MESH always wins from hop count, nor map all fan-in through a deterministic root column without considering alternate roots, reduction trees, or route spreading.

## Implementation

- Runtime/canonical config: `icc_contention_mode`
- JSON: `tu.multicore.contention = "ideal_parallel" | "shared_link"`
- API: `tu_cluster_estimate_traffic_cycles()` and `tu_icc_traffic_stats_t`
- The API reports ideal/estimated cycles, bottleneck directed-link service cycles, and bottleneck endpoints.
- `RING` shortest-path and `MESH` XY route enumeration share the same link-load model.
- Unsupported config names, invalid routes, and invalid pointers are rejected.

## Verification

- `make test-multicore`: 15/15 passed. Focused cases prove two 1 KiB messages sharing 0→1 serialize to 133 cycles, two disjoint links overlap at 69 cycles, ideal mode preserves 69 cycles, and invalid routes fail.
- `make test-config`: 20/20 passed. JSON parse, canonical-to-runtime propagation, default behavior, and unsupported contention rejection are covered.
- `make test-interconnect-contention-sweep`: reproduced all 12 rows above from the linked cmodel API.

## Limitations and next questions

`shared_link` is a lower bound, not a finite-queue simulator. It omits injection timestamps, packet headers/flit size, queue depth, arbitration, virtual channels, backpressure, head-of-line blocking, router pipeline occupancy, and deadlock. RING tie-breaking remains fixed; MESH routing now supports deterministic XY and YX as documented in `interconnect-mesh-routing-order.md`. Therefore the results support traffic-aware topology comparison but not a final topology or router microarchitecture decision.

Adaptive minimal routing and ring tie-direction studies remain implementation-worthy only after their arbitration/path-selection contracts are specified. A finite FIFO/backpressure model should follow compiler or collective traces so queue depth is explored against realistic burst timing rather than arbitrary simultaneous traffic alone.
