# DMA External and Dual-Endpoint 4 KiB Boundary Accounting

**Date:** 2026-09-18
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_external_boundary_sweep.c`

## Architecture question

Should DMA burst legality depend only on the local SRAM address, only on the external bus address, or on both endpoints?

The previous model could enforce an SRAM-side 4 KiB boundary but deliberately ignored host pointers because they are process virtual addresses. That is correct, but insufficient for studying an external AXI-like interface: a transfer can be locally aligned while its external physical/I/O address crosses 4 KiB, or vice versa. A realistic descriptor therefore needs explicit modeled bus-address metadata rather than deriving placement from a host pointer.

## Realistic alternatives

- **`size_only` (zero/default):** split only at the directional maximum transfer length. This preserves compatibility and represents a permissive bridge or hidden realignment.
- **`sram_4k`:** enforce a 4 KiB boundary only in the local SRAM address space. This represents a locally paged/windowed adapter or the prior protocol abstraction.
- **`external_4k`:** enforce a 4 KiB boundary only on an explicit external bus address. This represents an external AXI-like legality rule while allowing the local SRAM path to span that location.
- **`both_4k`:** stop at whichever endpoint reaches a 4 KiB boundary first. This represents two independently constrained adapters and can require more fragments than either one-sided mode.

The existing **`sram_address`** mode remains a separate maximum-burst-alignment alternative. A physical TU would usually hard-wire its endpoint contracts; the pre-spec cmodel keeps the alternatives runtime-selectable.

## Executable contract

Descriptors now optionally carry:

```text
external_address
external_address_valid
```

`tu_dma_desc_set_external_address()` sets this metadata. Host pointers remain functional copy pointers only and are never interpreted as physical addresses. `external_4k` and `both_4k` reject submission before queue or binding state changes when metadata is absent.

For each logical segment, the command generator starts with:

```text
chunk = min(remaining, directional_max_burst)
```

It then limits `chunk` by the enabled endpoint rooms:

```text
sram_room = 4096 - (sram_address mod 4096)
external_room = 4096 - (external_address mod 4096)
```

`both_4k` uses the minimum of all three limits. Both addresses advance by the emitted chunk. Strided 2D/3D descriptors use the corresponding host stride for external placement; scatter/gather uses contiguous external elements and indexed SRAM elements; multicast reuses the source external span for each destination.

The same helper feeds live completion, occupied-byte accounting, and queued least-projected-cycle binding.

## Measured matrix

Command:

```sh
make test-dma-external-boundary-sweep
```

Controls: one independent channel, 256-bit/32-byte interface, 8192-byte directional maximum, three issue cycles per command, 50-cycle descriptor base, logical segmentation, burst-command payload alignment, serialized issue/payload, disabled SRAM bandwidth metering, and a 64-byte linear load. SRAM starts at 4090; external address starts at 4070, so their 4 KiB boundaries occur 6 and 26 bytes into the transfer.

| Mode | SRAM address | External address | Completion cycles | Useful B | Occupied B |
|---|---:|---:|---:|---:|---:|
| `size_only` | 4090 | 4070 | 56 | 64 | 64 |
| `sram_4k` | 4090 | 4070 | 60 | 64 | 96 |
| `external_4k` | 4090 | 4070 | 60 | 64 | 96 |
| `both_4k` | 4090 | 4070 | 64 | 64 | 128 |

The one-sided modes each emit two commands and happen to tie in this placement, but they split at different addresses. `both_4k` emits three commands (6, 20, and 38 bytes), raising completion 14.3% and occupied interface bytes 2x versus `size_only`. This is not a universal penalty: aligned or same-offset endpoints can collapse to the same command geometry.

Store-direction gates reproduce 60 cycles/96 occupied bytes for external-only crossing and 64/128 for two different endpoint crossings. A two-row strided load with one externally crossing row completes in 63 cycles with 96 occupied bytes and preserves both row payloads. Direct execution, single submission, and descriptor-chain submission all reject missing metadata without copying or queue-state mutation. A queued gate also reverses least-projected binding: the 64-byte linear descriptor is selected over a two-row descriptor in `size_only`, while its external crossing makes the two-row queue preferable in `external_4k`.

## Gain versus sacrifice

| Dimension | `size_only` | One-sided 4 KiB | `both_4k` |
|---|---|---|---|
| Throughput | Optimistic if either endpoint has a no-crossing rule | Exposes one constrained adapter; sustained throughput remains unmodeled | Exposes compounded fragmentation when endpoint offsets differ |
| Latency | 56 cycles in the measured row | 60 cycles, +7.1% | 64 cycles, +14.3% |
| Area/resources | Assumes permissive span handling or hidden steering | One low-address comparator/counter plus split state | Two address trackers/comparators and minimum selection; magnitudes unquantified |
| Power/energy | Lowest visible command/lane activity; hidden steering unmodeled | 1.5x occupied bytes in the measured crossing | 2x occupied bytes; expected higher command/control switching |
| SRAM/DRAM traffic | Useful bytes unchanged | Useful bytes unchanged; one interface boundary exposed | Useful bytes unchanged; both endpoint fragmentations exposed |
| Numerical accuracy | Byte-exact, unchanged | Byte-exact, unchanged | Byte-exact, unchanged |
| Control complexity | Simplest visible accounting, strongest bridge assumption | Must retain and advance one address contract | Must synchronize two address domains and merge more responses |
| Verification burden | Length/tail cases | Explicit metadata, both directions, boundary edges | Independent offsets, coincident boundaries, strides/indices, and failure atomicity |
| Compiler/runtime | Placement appears insensitive | Allocator/descriptor must provide the relevant bus address | Placement of both SRAM and external buffers can affect command pressure |

Area, power, energy, sustained throughput, and response-buffer capacity are not quantified. The directions above are hardware rationale, not silicon measurements.

## Configuration and compatibility

`tu.dma.burst_boundary_mode` now accepts `size_only`, `sram_address`, `sram_4k`, `external_4k`, and `both_4k` through YAML/JSON, generated constants/runtime defaults, canonical parsing/validation/conversion, top-level initialization, and live DMA state.

`size_only` remains zero/default. Existing constructors and callers do not need external metadata unless they select an external-boundary mode. Address zero is valid because presence is represented by a separate boolean. Unsupported mode IDs and missing required metadata fail closed.

## Fidelity limits

This is deterministic command geometry, not a complete AXI, IOMMU, virtual-memory, cache, NoC, or DRAM model. The explicit external address is caller-supplied model metadata; the cmodel does not translate process virtual pointers or validate an OS mapping. The model omits page-table walks, IOMMU/TLB behavior, byte enables, response reordering/merge capacity, finite command credits, backpressure, adjacent-segment coalescing, shared SRAM/DRAM contention, and calibrated area/power. Occupied bytes represent DMA-interface lane occupancy, not automatically off-chip DRAM traffic.

## Verification

```sh
make test-dma-external-boundary-sweep
make test-dma-burst-boundary-sweep
make test-config-generation test-config test-dma
make config-docs
make clean && make
make test-quick
```
