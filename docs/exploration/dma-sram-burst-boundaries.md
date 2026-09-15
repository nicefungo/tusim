# DMA SRAM-Side Burst Boundary Accounting

**Date:** 2026-09-15
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_burst_boundary_sweep.c`

## Architecture question

Should a DMA command stream split transfers only by maximum payload size, or must each command also stop at an aligned maximum-burst boundary in the modeled SRAM address space?

The prior command model used `ceil(bytes / max_burst_bytes)`. That is a useful compatibility abstraction for a bridge that accepts arbitrary unaligned command spans or internally realigns them, but it undercounts commands when a protocol or SRAM-side width adapter prohibits crossing an aligned burst boundary.

## Realistic alternatives

- **`size_only` (zero/default):** split commands only by byte count. This represents an abstract/coalescing bridge, preserves all historical timing, and minimizes exposed commands. Hardware may need realignment buffers, byte steering, or a downstream interface that accepts unaligned spans.
- **`sram_address`:** split every modeled SRAM-side logical segment at aligned directional `max_burst_bytes` boundaries. This represents a simpler boundary-constrained command generator. It exposes extra commands and partial interface beats for unlucky placement.

No mode is universally preferred. A physical TU would usually hard-wire the protocol contract; the pre-spec cmodel retains both so compiler placement and controller complexity can be compared.

## Executable contract

For SRAM address `A`, segment length `N`, and directional burst limit `G`, address-aware mode repeatedly emits:

```text
room  = G - (A mod G)
chunk = min(N, room)
A    += chunk
N    -= chunk
```

Each emitted chunk is one command. Under `payload_scope=burst_commands`, each chunk occupies `ceil(chunk / bus_width_bytes)` interface cycles. One shared 64-bit helper produces command and payload totals for live descriptor completion and queued least-projected-cycle binding. It handles linear, strided 2D/3D, scatter/gather, and multicast SRAM offsets; load and store select destination and source SRAM addresses respectively. The focused matrix gates both scatter and gather as distinct indexed directions. Public constructors still store several products in 32-bit descriptor fields, so malformed descriptors outside validated SRAM/configuration bounds are not an overflow-hardening claim of this work.

Address-aware mode necessarily preserves logical SRAM discontinuities even if `burst_segmentation=aggregate`: one address-bounded command cannot span unrelated rows or indexed elements. Base-latency scope and issue/payload overlap remain independent settings.

Crucially, host pointers are not used as addresses. They are process virtual pointers, not a physical DRAM-address contract. This mode therefore models only the SRAM-side command boundary.

## Measured matrix

Command:

```sh
make test-dma-burst-boundary-sweep
```

Controls: one independent channel, 256-bit/32-byte interface, 64-byte directional burst limits, three issue cycles per command, 50-cycle descriptor base, logical segmentation, burst-command payload alignment, serialized issue/payload, and disabled SRAM bandwidth metering. Completion includes the asynchronous start tick.

| Descriptor / SRAM placement | Useful B | Mode | Completion cycles | Occupied B | Relative latency |
|---|---:|---|---:|---:|---:|
| Linear 64 B at 0 | 64 | size_only | 56 | 64 | 1.00x |
| Linear 64 B at 0 | 64 | sram_address | 56 | 64 | 1.00x |
| Linear 64 B at 1 | 64 | size_only | 56 | 64 | 1.00x |
| Linear 64 B at 1 | 64 | sram_address | 60 | 96 | 1.071x |
| Linear 80 B at 49 | 80 | size_only | 60 | 96 | 1.00x |
| Linear 80 B at 49 | 80 | sram_address | 64 | 128 | 1.067x |
| 2D, 4 x 20 B, base 48, stride 64 | 80 | size_only | 67 | 128 | 1.00x |
| 2D, 4 x 20 B, base 48, stride 64 | 80 | sram_address | 83 | 256 | 1.239x |
| 3D, 2 x 3 x 20 B, base 48, row 64 | 120 | size_only | 75 | 192 | 1.00x |
| 3D, 2 x 3 x 20 B, base 48, row 64 | 120 | sram_address | 99 | 384 | 1.320x |
| Scatter, five 4 B elements at offset 62 mod 64 | 20 | size_only | 71 | 160 | 1.00x |
| Scatter, five 4 B elements at offset 62 mod 64 | 20 | sram_address | 91 | 320 | 1.282x |
| Gather, five 4 B elements at offset 62 mod 64 | 20 | size_only | 71 | 160 | 1.00x |
| Gather, five 4 B elements at offset 62 mod 64 | 20 | sram_address | 91 | 320 | 1.282x |

Aligned traffic is identical. A 64-byte transfer starting at byte 1 becomes 63+1-byte commands, adding one issue and one partial interface beat. Every measured strided/gather element crosses a boundary, doubling command-local occupied bytes and increasing completion by 23.9-32.0% despite unchanged useful bytes.

A queued gate distinguishes planning behavior: a misaligned 64-byte linear queue costs 55 cycles in `size_only` and 59 in `sram_address`, while an aligned two-row 2D queue costs 58 in both. Least-projected binding therefore reverses from the linear queue to the 2D queue. This proves configuration reaches queue estimates; it is not queue-aware memory throughput evidence.

## Gain versus sacrifice

| Dimension | Size only | SRAM address bounded |
|---|---|---|
| Throughput | Optimistic for misaligned/fragmented transfers; may represent a coalescing bridge | Exposes command and lane pressure; sustained throughput remains unmodeled |
| Latency | Lower in affected rows; measured equality for aligned 64 B | +6.7% to +32.0% in affected matrix rows under stated costs |
| Area/resources | Expected realignment/coalescing buffers and steering, or a permissive downstream contract | Simpler boundary checks/counters, but more command queue/credit demand; magnitudes unquantified |
| Power/energy | Fewer command events and occupied beats, but buffer/search activity | Up to 2x occupied interface bytes in fragmented rows; expected higher control/interface activity; no calibrated energy |
| SRAM/DRAM traffic | Useful bytes unchanged; command-local occupancy can be optimistic | Useful bytes unchanged; SRAM-interface occupancy exposed. Physical DRAM traffic is not modeled |
| Numerical accuracy | Byte-exact, unchanged | Byte-exact, unchanged |
| Control complexity | More capable unaligned-span handling is assumed | Modulo/remaining-boundary tracking; simpler protocol legality, more events |
| Verification burden | Must prove hidden realignment/coalescing contract in hardware | Must gate aligned, misaligned, tails, strides, indices, both directions, and directional burst limits |
| Compiler/runtime | Placement appears insensitive within a burst | Alignment/padding and descriptor splitting become actionable; projected channel selection can change |

## Configuration and implementation

`tu.dma.burst_boundary_mode` accepts `size_only` and `sram_address` through YAML/JSON, generated constants/runtime defaults, canonical parsing/validation/conversion, top-level initialization, and live DMA state. Unsupported names and runtime IDs fail closed. Existing initializer wrappers and zero-initialized runtime callers remain `size_only`.

Relevant paths:

- `config/tu_config.{yaml,json}`
- `scripts/gen_config.py`, `tu_cmodel/tu_config.h`
- `tu_cmodel/infra/config.{h,c}`
- `tu_cmodel/dma_descriptor.{h,c}`
- `tu_cmodel/tu_cmodel.c`
- `tests/test_dma_burst_boundary_sweep.c`

## Fidelity limits

This is deterministic SRAM-side command geometry, not AXI, NoC, IOMMU, cache-line, page, or DRAM simulation. It has no physical host/DRAM address, 4 KiB rule, byte enables, read-modify-write, adjacent-index merge, finite FIFO/credits, backpressure, response reordering, shared SRAM/DRAM contention, or calibrated area/power. Occupied bytes are DMA-interface lane occupancy, not automatically off-chip bytes. Address arithmetic uses descriptor SRAM offsets and strides; descriptor constructors still own bounds validity. These limits must remain explicit before using the matrix for system throughput or energy claims.

## Verification

```sh
make test-dma-burst-boundary-sweep
make test-config-generation test-config test-dma
make config-docs
make clean && make
make test-quick
```
