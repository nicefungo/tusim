# DMA Logical-Segment Payload Alignment

**Date:** 2026-09-07

**Question:** May discontiguous rows or scatter/gather elements share interface beats, or must every logical segment begin a fresh beat?

## Hypothesis and realistic alternatives

The descriptor engine historically serialized payload as `ceil(total_bytes / bus_width)`. That is a valid model for a buffered byte-stream packer, but it can combine the tail of one strided row or indexed element with the beginning of the next despite an address discontinuity.

- **`descriptor` (compatibility default):** all useful bytes in one descriptor are packed into a continuous interface stream. Hardware may realize this with gather/coalescing buffers, byte steering, and enough credits to retain partial beats across logical boundaries. It minimizes exposed beat occupancy for fragmented descriptors, at the cost of datapath/control state and verification.
- **`logical_segments`:** each 2D row, 3D row, or scatter/gather element starts a fresh interface beat. This represents a simpler sequencer or an interface where discontinuities terminate a transaction. It can waste tail lanes and dynamic interface energy but avoids cross-segment packing assumptions.

Linear and multicast descriptors have one modeled logical segment, so the modes are identical for them. Payload alignment is independent of burst-command segmentation and base-latency scope: hardware may restart commands, setup, payload beats, any combination of the three, or none.

## Configuration and cycle model

Runtime JSON/YAML field:

```json
{"tu":{"dma":{"payload_scope":"logical_segments"}}}
```

Accepted values are `descriptor` and `logical_segments`. Zero/default is `descriptor`, preserving generated defaults, canonical defaults, older initializer APIs, and zero-initialized runtime callers.

For bus width `W`, useful descriptor bytes `N`, logical segment count `S`, and equal logical segment size `B`:

- descriptor-packed payload cycles: `ceil(N / W)`
- logical-segment-aligned payload cycles: `S × ceil(B / W)`
- occupied interface bytes: `payload_cycles × W`

`S` is one for linear/multicast, rows for 2D, depth×rows for 3D, and index count for scatter/gather. The model keeps useful-byte counters unchanged and adds separate occupied-byte counters at engine and channel scope. The same payload-cycle helper feeds live service and queued least-projected-cycle binding.

## Executable matrix

Command:

```sh
make test-dma-payload-scope-sweep
```

Configuration: 256-bit/32-byte interface, 50-cycle descriptor base latency, zero burst-issue cost, SRAM bandwidth metering disabled. Completion includes the initial asynchronous issue tick.

| Descriptor | Useful bytes | Descriptor-packed completion | Segment-aligned completion | Packed occupied bytes | Aligned occupied bytes | Payload efficiency, packed / aligned |
|---|---:|---:|---:|---:|---:|---:|
| Linear, 80 B | 80 | 54 | 54 | 96 | 96 | 83.3% / 83.3% |
| Strided 2D, 4×20 B | 80 | 54 | 55 | 96 | 128 | 83.3% / 62.5% |
| Strided 3D, 2×3×20 B | 120 | 55 | 57 | 128 | 192 | 93.8% / 62.5% |
| Gather, 5×4 B | 20 | 52 | 56 | 32 | 160 | 62.5% / 12.5% |

The harness also gates the store direction, exact useful-byte movement, separate useful/occupied counters, malformed config rejection, default compatibility, and queued projected-binding behavior. In a discriminating two-channel case, an 8×5 B strided descriptor is cheaper than a 128 B linear descriptor when packed (2 versus 4 payload cycles), but more expensive when each row starts a beat (8 versus 4); least-projected binding therefore reverses channels.

## Multi-objective interpretation

- **Throughput/latency:** descriptor packing saves one to four cycles in this small isolated matrix and can preserve interface utilization for fragmented requests. The benefit scales with short segments relative to bus width; aligned/full-beat rows are unchanged. This is service sensitivity, not sustained memory throughput.
- **Area/resources:** packing is expected to require partial-beat storage, byte steering, address tracking, and possibly wider credit state. Segment alignment can use a simpler lane-mask/restart path. The cmodel has no physical area model for these resources, so magnitudes are unquantified.
- **Power/energy:** aligned segments occupy 1.33×, 1.50×, and 5.00× the interface bytes of the measured 2D, 3D, and gather cases, which qualitatively increases data-path switching if inactive lanes are physically driven. Packing spends buffer/search/steering energy instead. Neither effect is calibrated.
- **SRAM/DRAM traffic:** useful bytes and copied values are unchanged. Occupied interface bytes are now observable, but DRAM cache-line/burst overfetch, address alignment, and row/channel placement remain separate and unmodeled here.
- **Numerical accuracy:** unchanged; this mode only affects timing and occupied-interface accounting, and byte-exact data movement is gated.
- **Control complexity:** descriptor packing needs continuity state across discontinuities; logical alignment makes segment boundaries explicit and local.
- **Verification burden:** packing must prove lane ordering and partial-beat retention across every descriptor type. Alignment requires per-segment beat rounding and tail-mask corner cases. Both need zero-byte, overflow, and direction coverage; this sweep gates the nonempty uniform-segment contract.
- **Compiler/runtime:** compilers can coalesce or pad rows, choose wider contiguous descriptors, or avoid tiny gathers when segment alignment is selected. Descriptor packing gives software more freedom but only if hardware actually supports cross-boundary byte packing.

No mode is universally selected. A bandwidth-oriented TU may justify packing logic; a low-area mover may intentionally expose segment tails. A physical TU would commonly hard-wire one policy, while the pre-spec cmodel preserves both.

## Fidelity limits

This is deterministic beat occupancy, not AXI/NoC/DRAM protocol simulation. It does not inspect source or destination alignment, split segments at maximum-burst or page boundaries, model byte enables, merge adjacent indices, distinguish command and data channels, apply occupied bytes to SRAM/DRAM contention, or estimate physical area/power. Segment sizes are uniform under the current descriptor formats. Occupied counters represent `payload_cycles × bus_width`, not necessarily off-chip DRAM bytes. Calibration, finite FIFOs/credits, retries, queue backpressure, and command/data overlap remain unmodeled.

## Verification

```sh
make test-dma-payload-scope-sweep test-dma-base-scope-sweep test-dma-segmentation-sweep
make test-config test-dma
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.payload-scope.h
make config-docs
make clean && make
make test-quick
```
