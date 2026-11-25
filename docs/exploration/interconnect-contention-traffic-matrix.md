# Interconnect Shared-Link Contention by Traffic Matrix

**Date:** 2026-07-24
**Status:** Implemented and verified
**Type:** Cmodel-linked simultaneous traffic-matrix sweep

## Design question

How does finite directed-link load change topology rankings relative to isolated transfers? These estimates do not measure elapsed latency.

The hypothesis was that topology conclusions would reverse by traffic matrix: a mesh has more links and shorter average routes, while a bidirectional ring can split traffic around two directions and may avoid a deterministic mesh route's hotspot.

## Alternatives retained

| Runtime mode | Model | Why a hardware/modeling team might choose it | Sacrifice |
|---|---|---|---|
| `ideal_parallel` (default) | Maximum isolated message latency; links do not contend | Backward-compatible lower bound for functional/compiler work, or approximation of a heavily overprovisioned/local fabric | Optimistic for collectives, fan-in, and all-to-all traffic |
| `shared_link` | Maximum of isolated latency and `bottleneck directed-link serialization + longest route latency` | Compatibility score for early topology and placement comparisons | Disjoint maxima can overstate even a feasible schedule |
| `shared_link_floor` | Maximum of isolated latency and busiest directed-link service | Necessary time under fixed routes, exclusive service and the selected isolated timing | Can be loose; does not schedule packets |

These are reporting alternatives, not hardware speed settings. The default remains `ideal_parallel`. See `interconnect-traffic-floor.md` for the counterexample and tested floor.

## Routing and timing contract

- Switching: cut-through
- Link payload width: 16 B/cycle
- Router latency: 5 cycles/hop
- Payload: 4,096 B/message
- Injection: all messages in a row are simultaneous
- Ring routing in the original sweep: shortest path, clockwise on equal-distance ties (still the default).
- Mesh routing: deterministic dimension-order XY
- Links are directional; opposite directions do not contend
- Command: `make test-interconnect-contention-sweep`

For each message, the existing isolated cut-through equation is:

`isolated = hops × router_latency + ceil(bytes / link_bytes_per_cycle)`

For a traffic matrix, the historical shared-link score is:

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

- **Topology ranking reverses:** At 16 cores, the 4×4 mesh has a smaller all-to-all score (4,126 vs 9,256) but a larger hotspot-to-core-0 score (3,102 vs 2,088). Ring paths split fan-in; XY concentrates mesh traffic on 4→0. Neither score is a completion time.
- **Neighbor traffic:** Both topologies remain at the ideal bound because this workload maps one message per bottleneck directed link. The ring has an 8.7% lower estimated score than the 4×4 mesh row here (261 vs 286 cycles) because the synthetic mesh neighbor sequence includes a longer wrap route; placement, not raw topology strength, causes the difference.
- **Scale:** Shared/ideal score ratios are 31.27× for 16-core ring all-to-all and 14.43× for mesh. These are not slowdowns.
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
- JSON: `tu.multicore.contention = "ideal_parallel" | "shared_link" | "shared_link_floor"`
- API: `tu_cluster_estimate_traffic_cycles()` and `tu_icc_traffic_stats_t`
- The API reports ideal/estimated cycles, bottleneck directed-link service cycles, and bottleneck endpoints.
- `RING` shortest-path and `MESH` XY route enumeration share the same link-load model.
- Unsupported config names, invalid routes, and invalid pointers are rejected.

## Verification

- `make test-multicore`: 15/15 passed. Focused cases prove two 1 KiB messages sharing 0→1 serialize to 133 cycles, two disjoint links overlap at 69 cycles, ideal mode preserves 69 cycles, and invalid routes fail.
- `make test-config`: 20/20 passed. JSON parse, canonical-to-runtime propagation, default behavior, and unsupported contention rejection are covered.
- `make test-interconnect-contention-sweep`: reproduced all 12 rows above from the linked cmodel API.

## Limitations and next questions

`shared_link` is a heuristic, not a bound or a finite-queue simulator. Neither it nor the floor models injection timestamps, headers, queue depth, arbitration, virtual channels, backpressure, head-of-line blocking or router pipeline occupancy. See the ring tie experiment below and the separate mesh routing-order study. No final physical topology recommendation follows.

Adaptive minimal routing needs an arbitration/path-selection contract. A finite FIFO/backpressure model should follow compiler or collective traces so queue depth is explored against realistic burst timing rather than arbitrary simultaneous traffic alone.

## Ring tie direction: does steering matter when hop count is unchanged?

`tu.multicore.ring_tie` selects `clockwise` (historical zero/default), `counterclockwise`, `even_cw` or `odd_cw`. The latter two send ties clockwise for the named source parity and counterclockwise otherwise. Select once from the original source ID; a non-tie always takes the shorter path. YAML builds the default; JSON goes through `tu_config_to_runtime` and `tu_cluster_create` to the directed-link walker. Public structs grew; rebuild consumers. Neither `tu_cluster_send` nor the isolated estimator simulates router traversal.

Two 256-byte packets per snapshot, cut-through, 16 B/cycle/directed link, 2 cycles/hop. Cells report **busiest directed-link service cycles**, not elapsed cycles. The necessary floor is the larger of the isolated column and the selected cell. CW means increasing core ID modulo ring size.

| Ring | Messages | Longest isolated (cycles) | CW | CCW | even_cw | odd_cw |
|---:|---|---:|---:|---:|---:|---:|
| 8 | 0→4, 0→1 | 24 | 32 | 16 | 32 | 16 |
| 8 | 0→4, 0→7 | 24 | 16 | 32 | 16 | 32 |
| 8 | 1→5, 1→2 | 24 | 32 | 16 | 16 | 32 |
| 8 | 1→5, 1→0 | 24 | 16 | 32 | 32 | 16 |
| 8 | 0→4, 1→5 | 24 | 32 | 32 | 16 | 16 |
| 7 | 0→3, 0→1 | 22 | 32 | 32 | 32 | 32 |
| 7 | 0→4, 0→6 | 22 | 32 | 32 | 32 | 32 |
| 2 | 0→1, 1→0 | 18 | 16 | 16 | 16 | 16 |

A fixed preference is predictable and can be hard-wired for a known placement; which direction helps depends on the other packet. Parity can spread adjacent antipodal sources without per-packet history, but couples scheduling to endpoint numbering. A team could hard-wire each injection port independently; a programmable version needs selectable steering and tests for both phases, ring wrap and degenerate rings. In the two-node abstraction both directions to the same neighbor denote one directed endpoint pair, not two parallel physical lanes. Stateful round-robin, per-flow hashing and queue-aware choices need packet sequence, flow-key or occupancy contracts that this snapshot API does not supply.

Endpoint bytes and hop counts are invariant, arithmetic is untouched, and the experiment adds no SRAM/DRAM transfers. Where links are active changes, not their aggregate byte-hop work. Power, steering area, achievable clock, sustained throughput and actual packet completion are unquantified: no queues, injection limits, deadlock/progress protocol or calibrated energy model. Use the matrix for placement comparisons, not a speedup claim.

Reproduce with `make clean && make && make test-icc-ring-ties`. The gate also checks reversed message order, invalid and omitted settings, the shipped JSON, and byte-exact isolated sends and counters. `make test-interconnect-contention-sweep test-interconnect-routing-sweep` reruns the original comparisons.
