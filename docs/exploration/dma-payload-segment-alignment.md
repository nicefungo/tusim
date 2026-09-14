# DMA Payload Packing Boundaries

**Date:** 2026-09-14
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_payload_scope_sweep.c`

## Architecture question

Across which boundaries may a DMA width adapter retain a partial interface beat: an entire descriptor, only one logical row/index, or only one issued burst command?

The previous model distinguished descriptor-wide packing from logical-segment alignment but still allowed two burst commands within one row to share a beat. That is optimistic for a protocol engine that independently frames every burst, especially when the configured maximum burst is narrower than the DMA data path.

## Realistic alternatives

- **`descriptor` (zero/default):** pack all useful bytes in one descriptor into a continuous beat stream. A buffered gather/coalescing front end can justify this mode. It minimizes lane waste but needs partial-beat storage, byte steering, address state, and enough credits to retain data across command and address discontinuities.
- **`logical_segments`:** start a fresh beat for every 2D/3D row or scatter/gather index. A simpler strided sequencer may terminate payload occupancy at logical discontinuities while still packing adjacent burst commands inside a row.
- **`burst_commands`:** start a fresh beat for every burst command. This represents a narrow-burst protocol engine or width converter whose framing state drains at each command boundary. It is the most conservative of these three occupancy contracts and can waste lanes when `max_burst_bytes < bus_width_bytes`.

No mode is universally preferred. A physical TU would normally hard-wire one policy; the pre-spec cmodel retains all three because each corresponds to plausible buffering and protocol choices.

## Executable model

For interface width `W`, useful descriptor bytes `N`, logical segment count `S`, segment bytes `B`, and directional maximum burst bytes `G`:

```text
descriptor cycles = ceil(N / W)
logical cycles    = S * ceil(B / W)
burst cycles      = sum over issued bursts j of ceil(bytes_j / W)
occupied bytes    = payload cycles * W
```

Burst mode uses the same aggregate-versus-logical burst-segmentation contract as command counting. In aggregate segmentation, `N` is split into `G`-byte commands. In logical segmentation, every segment is independently split into `G`-byte commands. The common helper feeds live completion, useful/occupied counters, legacy accounting, and queued least-projected-cycle binding.

Compatibility is preserved: `descriptor` remains enum value zero in generated defaults, canonical defaults, the shipped JSON/YAML, old initializer wrappers, and zero-initialized runtime callers.

Runtime configuration:

```json
{"tu":{"dma":{"payload_scope":"burst_commands"}}}
```

Accepted values are `descriptor`, `logical_segments`, and `burst_commands`.

## Measured matrix

Command:

```sh
make test-dma-payload-scope-sweep
```

Controls: one independent channel, 256-bit/32-byte interface, 16-byte directional burst limit, logical burst segmentation, 50-cycle descriptor base latency, zero burst-issue cost, and disabled SRAM bandwidth metering. Completion includes the asynchronous start tick.

| Descriptor | Useful B | Scope | Completion cycles | Occupied B | Payload efficiency |
|---|---:|---|---:|---:|---:|
| Linear 80 B | 80 | descriptor | 54 | 96 | 83.3% |
| Linear 80 B | 80 | logical_segments | 54 | 96 | 83.3% |
| Linear 80 B | 80 | burst_commands | 56 | 160 | 50.0% |
| Strided 2D, 4×20 B | 80 | descriptor | 54 | 96 | 83.3% |
| Strided 2D, 4×20 B | 80 | logical_segments | 55 | 128 | 62.5% |
| Strided 2D, 4×20 B | 80 | burst_commands | 59 | 256 | 31.3% |
| Strided 3D, 2×3×20 B | 120 | descriptor | 55 | 128 | 93.8% |
| Strided 3D, 2×3×20 B | 120 | logical_segments | 57 | 192 | 62.5% |
| Strided 3D, 2×3×20 B | 120 | burst_commands | 63 | 384 | 31.3% |
| Gather, 5×4 B | 20 | descriptor | 52 | 32 | 62.5% |
| Gather, 5×4 B | 20 | logical_segments | 56 | 160 | 12.5% |
| Gather, 5×4 B | 20 | burst_commands | 56 | 160 | 12.5% |

Burst framing adds no cost over logical alignment for the gather because each 4-byte index is already one logical segment and one burst. It adds two cycles and 64 occupied bytes for the linear case, and doubles logical-aligned occupied bytes for the measured 2D/3D rows because each 20-byte row becomes a 16-byte command plus a 4-byte tail command, each occupying a 32-byte beat.

A queued-policy gate also distinguishes runtime planning. A four-row 20-byte strided descriptor is cheaper than a 112-byte linear descriptor under descriptor and logical scopes, so least-projected binding selects its channel. Burst framing changes them to eight versus seven payload cycles and reverses the selected channel. This proves the setting reaches queued projections rather than only report counters.

## Gain versus sacrifice

| Dimension | Descriptor packing | Logical-segment alignment | Burst-command alignment |
|---|---|---|---|
| Throughput / latency | Lowest isolated payload service for fragmented work; sustained throughput unmodeled | Pays row/index tails | Pays every command tail; measured completion is up to 14.5% above descriptor packing (63 vs 55 cycles) |
| Area / resources | Expected largest partial-beat/coalescing buffers and steering | Less cross-address state; row-local packing remains | Expected simplest command-local framing, but may need more downstream transactions; magnitudes unquantified |
| Power / energy | Less interface occupancy but more packing/search activity | Intermediate lane waste and control | Up to 3× occupied bytes versus descriptor packing in this matrix; expected higher interface switching if inactive lanes are physically driven, but no calibrated energy model |
| SRAM / DRAM traffic | Useful bytes unchanged; occupied-interface bytes minimized | Useful bytes unchanged; row/index tails exposed | Useful bytes unchanged; command tails exposed. Off-chip overfetch and cache-line traffic remain separate |
| Numerical accuracy | Byte-exact and unchanged | Byte-exact and unchanged | Byte-exact and unchanged |
| Control complexity | Cross-command and cross-segment continuity | Segment-local reset plus intra-segment packing | Command-local reset; simplest continuity contract, but more command/payload bookkeeping |
| Verification burden | Must prove ordering across discontinuities and partial-beat retention | Must gate every descriptor shape and segment tail | Must additionally gate directional burst limits, full/tail commands, burst narrower/equal/wider than interface, and interactions with segmentation |
| Compiler / runtime | Software can issue fragmented descriptors without exposing lane waste | Compiler benefits from row padding/coalescing | Compiler should avoid narrow bursts on wide movers or combine commands when legal; projected channel binding can change |

The measured 14.5% completion increase is restricted to small transfers dominated by the fixed 50-cycle base. The occupancy amplification is the stronger signal: burst framing can double or triple modeled interface occupancy without changing useful bytes. Conversely, choosing descriptor packing because it is locally faster would silently assume buffering, credits, and byte steering that a low-area implementation may not contain.

## Implementation paths

- `config/tu_config.yaml`, `config/tu_config.json`: `tu.dma.payload_scope`
- `scripts/gen_config.py`, `tu_cmodel/tu_config.h`: generated constants and runtime default
- `tu_cmodel/infra/config.{h,c}`: canonical enum, parse, validation, runtime propagation, generated docs
- `tu_cmodel/dma_descriptor.{h,c}`: executable enum and common burst-aligned payload helper
- `tests/test_dma_payload_scope_sweep.c`: 12-row live matrix, read/write movement, occupied bytes, projection reversal, defaults, parse, and rejection
- `tests/test_generated_config.py`, `tests/test_config.c`: generated/nondefault and canonical gates

## Fidelity limits

This is deterministic payload-beat occupancy, not AXI/NoC/DRAM protocol simulation. Burst boundaries come from configured byte counts, not source/destination alignment, 4 KiB boundaries, cache lines, pages, or response reordering. The model does not represent byte enables, actual inactive-lane switching, adjacent-index merge, command FIFO depth, finite credits, backpressure, arbitration, memory latency per burst, shared SRAM/DRAM bandwidth, or calibrated area/power. Occupied bytes mean `payload_cycles × DMA bus width`; they are not automatically physical DRAM bytes. The existing ideal issue/payload-overlap mode remains aggregate and does not simulate per-burst pipeline bubbles.

## Verification

```sh
make test-dma-payload-scope-sweep
make test-config-generation test-config test-dma
make config-docs
make clean && make
make test-quick
```
