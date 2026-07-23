# Interconnect Switching: Cut-Through vs Store-and-Forward

**Date:** 2026-07-23
**Status:** Implemented and verified
**Type:** Analytical sweep plus cmodel-linked functional/config tests

## Design question

For a routed on-chip TU interconnect, when does cut-through forwarding justify its buffering and flow-control complexity relative to store-and-forward switching? The prior topology sweep multiplied full-payload serialization by hop count without naming that assumption, while the executable cluster model charged no payload serialization at all. Both hid a material architecture choice.

## Alternatives retained

| Mode | Timing equation | Why hardware might choose it | Main sacrifice |
|---|---|---|---|
| `legacy_hop_only` | `hops × router_latency` | Backward-compatible idealized lower bound for functional/compiler work where payload timing is intentionally disabled | Physically optimistic; no link bandwidth cost |
| `cut_through` | `hops × router_latency + ceil(bytes/link_width)` | Wormhole/cut-through NoCs pipeline flits across routers; attractive for long routes and tensor payloads | Per-virtual-channel buffers, credit flow control, deadlock analysis, and verification complexity |
| `store_and_forward` | `hops × (router_latency + ceil(bytes/link_width))` | Simple packet-level control and full-packet checking/buffering; plausible for small low-frequency fabrics or bridge boundaries | Full packet buffering and repeated serialization produce high multi-hop latency and buffer energy |

Defaults remain `legacy_hop_only`, 16 B/cycle, and 5 cycles/router, preserving the previous cycle behavior exactly until a physical mode is selected.

## Sweep configuration

- Payloads: 64 B, 1 KiB, 64 KiB
- Route lengths: 1, 3, 7 hops
- Baseline link: 16 B/cycle (128-bit payload)
- Router/link latency: 5 cycles/hop
- Width sensitivity: 8, 16, 32 B/cycle at 64 KiB and 3 hops
- Command: `make test-interconnect-switching-sweep`

## Measured analytical matrix

| Bytes | Hops | Cut-through cycles | Store-forward cycles | SF / CT |
|---:|---:|---:|---:|---:|
| 64 | 1 | 9 | 9 | 1.000× |
| 64 | 3 | 19 | 27 | 1.421× |
| 64 | 7 | 39 | 63 | 1.615× |
| 1,024 | 1 | 69 | 69 | 1.000× |
| 1,024 | 3 | 79 | 207 | 2.620× |
| 1,024 | 7 | 99 | 483 | 4.879× |
| 65,536 | 1 | 4,101 | 4,101 | 1.000× |
| 65,536 | 3 | 4,111 | 12,303 | 2.993× |
| 65,536 | 7 | 4,131 | 28,707 | 6.949× |

### Width sensitivity: 64 KiB, 3 hops

| Link B/cycle | Cut-through cycles | Store-forward cycles | SF / CT |
|---:|---:|---:|---:|
| 8 | 8,207 | 24,591 | 2.996× |
| 16 | 4,111 | 12,303 | 2.993× |
| 32 | 2,063 | 6,159 | 2.985× |

## Findings and trade-offs

- **Latency/throughput:** One-hop traffic is identical in this model. The switching choice matters with both route length and payload size: at 3 hops, cut-through reduces modeled latency by 29.6% for 64 B and 66.6% for 64 KiB; at 7 hops and 64 KiB it is 6.95× lower. These are isolated, contention-free transfer results—not collective throughput claims.
- **Link width:** Doubling 8→16→32 B/cycle nearly halves both modes' large-payload latency. The store-forward/cut-through ratio remains near the hop count in the serialization-dominated regime; width does not remove the switching trade-off.
- **Area/resources:** Cut-through is expected to need flit buffers, virtual channels/credits, and more router state. Store-forward needs packet-sized buffering at each hop. Exact area is **unquantified** because the cmodel has no router synthesis/area model.
- **Power/energy:** Store-forward is expected to incur more intermediate buffer writes/reads; cut-through's flow-control and VC logic adds control energy. Both effects are **unquantified**. Existing NoC hop energy counters are not wired to this timing API.
- **SRAM/DRAM traffic:** Endpoint payload traffic is unchanged. Intermediate router-buffer traffic differs but is **not observed** by SRAM/DRAM counters.
- **Numerical accuracy:** No effect; switching changes timing only and `tu_cluster_send()` remains byte-exact.
- **Control and verification:** Cut-through has greater deadlock, backpressure, credit, and packet-interleaving burden. Store-forward is simpler to reason about but requires capacity for complete packets.
- **Compiler/runtime:** Software must not assume the legacy lower bound represents physical latency. Route-aware placement and payload coalescing benefit both physical modes; cut-through especially rewards long contiguous transfers. The compiler need not change functional packet contents.

## Implementation

- `tu_runtime_config_t` and canonical `tu_config_t` expose `icc_switching_mode`, `icc_link_bytes_per_cycle`, and `icc_router_latency_cycles`.
- JSON keys live under `tu.multicore`: `switching`, `link_bytes_per_cycle`, and `router_latency_cycles`.
- `tu_cluster_estimate_transfer_cycles()` is the shared timing API; `tu_cluster_send()` uses it for stats and destination-core cycle accounting.
- Supported JSON names: `legacy_hop_only`, `cut_through`, `store_and_forward`.
- Invalid modes and zero link width are rejected. Zero router latency is permitted as an explicit ideal-router experiment.

## Verification

- `make test-multicore`: 14/14 passed, including exact mode equations, invalid routes, zero-byte behavior, and byte-exact communication.
- `make test-config`: 20/20 passed, including JSON parse, canonical-to-runtime propagation, unsupported mode rejection, and zero-width rejection.
- `make test-interconnect-switching-sweep`: reproduced all 12 matrix rows above.

A pre-existing focused-test defect was also corrected: multicore MMA now reads the FP32 O-buffer as FP32, and the focused target links the rebuilt static archive. Stable dataflow registry entries prevent reinitializing one core from invalidating plugin pointers held by another.

## Limitations / next question

This is an isolated-transfer model. It does not model finite injection queues, virtual channels, head-of-line blocking, route arbitration, shared-link contention, packet size/header flits, or topology-specific routing. Therefore it does **not** validate the old topology document's universal mesh speedups. The next topology recommendation must use a traffic matrix and finite shared-link contention model; until then RING and MESH remain runtime alternatives without a universal winner.
