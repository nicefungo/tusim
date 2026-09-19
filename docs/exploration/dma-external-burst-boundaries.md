# DMA External and Dual-Endpoint Burst Boundaries

**Date:** 2026-09-19
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_external_boundary_sweep.c`

## Architecture question

Should maximum-burst alignment and 4 KiB no-crossing rules be evaluated at the local SRAM endpoint, the external bus endpoint, or both?

The model previously offered SRAM maximum-burst alignment and one-/two-sided 4 KiB boundaries. This left an asymmetry: an external cache-line, bridge window, or command adapter could require bursts to end at the directional maximum-burst boundary, but the cmodel could express that alignment only on SRAM. Host pointers cannot repair the gap because they are process virtual addresses, not modeled bus placement.

## Realistic alternatives

- **`size_only` (zero/default):** cap command length but ignore address alignment. This represents a permissive bridge or hidden realignment and preserves historical behavior.
- **`sram_address`:** align commands to the directional maximum-burst boundary in local SRAM. This represents a constrained local adapter.
- **`external_address`:** align commands to the same directional maximum-burst boundary at the explicit external address. This represents an external line/window adapter while allowing local spans.
- **`both_address`:** stop at whichever endpoint reaches its maximum-burst boundary first. This represents two independently constrained adapters.
- **`sram_4k`, `external_4k`, `both_4k`:** retain the independent fixed 4 KiB legality alternatives. These represent paged/windowed or AXI-like endpoint rules and are not aliases for maximum-burst alignment.

A physical TU would normally hard-wire one endpoint contract. The pre-spec cmodel retains all seven alternatives because each encodes a materially different placement/resource assumption.

## Executable contract

Descriptors optionally carry:

```text
external_address
external_address_valid
```

`tu_dma_desc_set_external_address()` sets this metadata. Host pointers remain functional copy pointers only. Every mode that examines the external endpoint rejects direct execution, submission, and descriptor-chain submission before copying or queue/binding mutation when metadata is absent.

For each logical segment, command generation starts with:

```text
chunk = min(remaining, directional_max_burst)
```

It then limits the chunk by each enabled endpoint:

```text
max-burst room = directional_max_burst - (address mod directional_max_burst)
4 KiB room      = 4096 - (address mod 4096)
```

Dual-endpoint modes use the minimum enabled room. Both addresses advance by the emitted chunk. Strided 2D/3D descriptors retain corresponding external strides; scatter/gather uses contiguous external elements and indexed SRAM elements; multicast reuses the source external span for each destination. One helper feeds live completion, occupied-byte accounting, and queued least-projected-cycle binding.

## Measured matrix

Command:

```sh
make test-dma-external-boundary-sweep
```

Controls: one independent channel, 256-bit/32-byte interface, 128-byte directional maximum, three issue cycles per command, 50-cycle descriptor base, logical segmentation, burst-command payload alignment, serialized issue/payload, disabled SRAM bandwidth metering, and a 64-byte linear load.

The maximum-burst rows start SRAM at 126 and external at 110, placing endpoint boundaries 2 and 18 bytes into the transfer. The 4 KiB rows start at 4090 and 4070, placing boundaries 6 and 26 bytes into the transfer.

| Mode | SRAM address | External address | Completion cycles | Useful B | Occupied B |
|---|---:|---:|---:|---:|---:|
| `size_only` | 126 | 110 | 56 | 64 | 64 |
| `sram_address` | 126 | 110 | 60 | 64 | 96 |
| `external_address` | 126 | 110 | 60 | 64 | 96 |
| `both_address` | 126 | 110 | 64 | 64 | 128 |
| `sram_4k` | 4090 | 4070 | 60 | 64 | 96 |
| `external_4k` | 4090 | 4070 | 60 | 64 | 96 |
| `both_4k` | 4090 | 4070 | 64 | 64 | 128 |

The one-sided modes each emit two commands and tie in these placements, though they split at different addresses. Each dual-endpoint mode emits three commands because endpoint offsets differ. Relative to `size_only`, one-sided boundary accounting raises measured completion 7.1% and occupied bytes 50%; dual-endpoint accounting raises completion 14.3% and occupied bytes 100%. These are placement-specific command/lane costs, not universal throughput penalties. Aligned endpoints or coincident offsets can collapse to identical geometry.

Store gates reproduce both external maximum-burst and dual-endpoint results. The existing externally crossing strided load remains byte-exact. A queued gate reverses least-projected binding: the 64-byte linear descriptor is preferred over a two-row descriptor in `size_only`, while its external maximum-burst crossing makes the two-row queue preferable in `external_address`.

## Gain versus sacrifice

| Dimension | `size_only` | One-sided endpoint boundary | Dual-endpoint boundary |
|---|---|---|---|
| Throughput | Optimistic if an endpoint cannot realign | Exposes one source of command fragmentation; sustained rate unmodeled | Exposes compounded fragmentation; sustained rate unmodeled |
| Latency | 56 cycles in the measured control | 60 cycles, +7.1% | 64 cycles, +14.3% |
| Area/resources | Assumes hidden steering/buffering or permissive protocol | One address tracker, comparator, and split state | Two trackers/comparators plus minimum selection and response merge state |
| Power/energy | Lowest visible command/lane activity; hidden realignment cost omitted | 1.5x occupied bytes in the measured crossing; expected more control switching | 2x occupied bytes; expected highest command/control switching |
| SRAM/DRAM traffic | Useful bytes unchanged; physical overfetch hidden | Useful bytes unchanged; one interface's lane occupancy exposed | Useful bytes unchanged; both endpoint fragmentations exposed |
| Numerical accuracy | Byte-exact, unchanged | Byte-exact, unchanged | Byte-exact, unchanged |
| Control complexity | Simplest visible model | Must retain/advance one placement contract | Must synchronize two address domains and merge more fragments |
| Verification burden | Length/tail cases | Explicit metadata, directions, exact boundary edges | Independent/coincident offsets, all descriptor shapes, failure atomicity |
| Compiler/runtime | Placement appears insensitive | Allocator/descriptor supplies the relevant address | Both SRAM and external placement can affect command pressure |

Area, power, energy, sustained throughput, finite credit demand, and response-buffer capacity are qualitative/unquantified. The cmodel does not justify selecting one mode universally.

## Configuration and compatibility

`tu.dma.burst_boundary_mode` accepts:

```text
size_only
sram_address
external_address
both_address
sram_4k
external_4k
both_4k
```

The path is executable through YAML/JSON, generated constants/runtime defaults, canonical parsing/validation/conversion, top-level initialization, and live DMA state. `size_only` remains numeric zero/default. Existing constructors need external metadata only when an external endpoint mode is selected. Address zero remains valid through an explicit validity bit. Unsupported mode IDs fail closed.

## Fidelity limits

This is deterministic command geometry, not a complete AXI, cache, IOMMU, virtual-memory, NoC, or DRAM model. The explicit external address is caller-supplied metadata; the cmodel does not translate host pointers. It omits page-table walks, TLBs, byte enables, response ordering/merge capacity, finite command credits, backpressure, adjacent-segment coalescing, shared SRAM/DRAM contention, sustained queue throughput, and calibrated area/power. Occupied bytes represent DMA-interface lane occupancy, not automatically off-chip DRAM traffic. The maximum-burst boundary is a configurable architecture abstraction, not a claim that every bus protocol requires naturally aligned maximum-length bursts.

## Verification

```sh
make test-dma-external-boundary-sweep
make test-dma-burst-boundary-sweep
make test-config-generation test-config test-dma
make config-docs
make clean && make
make test-quick
```
