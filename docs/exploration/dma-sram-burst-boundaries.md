# DMA SRAM-Side Burst and 4 KiB Boundary Accounting

**Date:** 2026-09-17
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_burst_boundary_sweep.c`

## Architecture question

Should the DMA command generator constrain a transfer only by maximum payload length, split it at every aligned maximum-burst boundary, or independently enforce a 4 KiB protocol boundary?

These are distinct hardware contracts. A maximum burst is a length limit. It does not by itself forbid crossing an address boundary. A local width adapter may require every command to stay inside an aligned maximum-burst region. An AXI-like bridge may instead permit that crossing but prohibit crossing 4 KiB.

## Realistic alternatives

- **`size_only` (zero/default):** split only when the command reaches the directional maximum byte count. This represents a permissive or internally realigning bridge and preserves historical timing. It may require steering/buffering or a downstream protocol that accepts the span.
- **`sram_address`:** split at each aligned directional maximum-burst boundary. This represents a strict local SRAM width adapter or boundary-constrained command generator. Misalignment can increase command count at every burst boundary.
- **`sram_4k`:** enforce the directional maximum length and independently stop at each aligned 4 KiB SRAM-side boundary. This represents an AXI-like no-crossing rule without imposing maximum-burst alignment on every command.

No mode is universally preferred. A physical implementation would normally hard-wire one protocol contract; the pre-spec cmodel retains all three for controller and compiler-placement studies.

## Executable contract

For segment address `A`, remaining bytes `N`, directional maximum `G`, and selected boundary `B`, address-bounded modes repeatedly emit:

```text
room  = B - (A mod B)
chunk = min(N, G, room)
A    += chunk
N    -= chunk
```

`B=G` for `sram_address`; `B=4096` for `sram_4k`. `size_only` retains the historical `ceil(N/G)` command count. Under `payload_scope=burst_commands`, each emitted chunk occupies `ceil(chunk / bus_width_bytes)` interface cycles.

One helper supplies command and payload totals to live completion and queued least-projected-cycle binding. It handles linear, strided 2D/3D, scatter/gather, and multicast SRAM offsets. Loads use destination SRAM addresses; stores use source SRAM addresses. Logical discontinuities remain separate in both address-bounded modes.

Host pointers are deliberately excluded: process virtual pointers are not physical DRAM addresses. The 4 KiB rule is therefore an SRAM-side protocol abstraction, not proof of IOMMU or external AXI address behavior.

## Measured matrix

Command:

```sh
make test-dma-burst-boundary-sweep
```

Controls: one independent channel, 256-bit/32-byte interface, 64-byte directional maximum bursts, three issue cycles per command, 50-cycle descriptor base, logical segmentation, burst-command payload alignment, serialized issue/payload, and disabled SRAM bandwidth metering. Completion includes the asynchronous start tick.

| Descriptor / SRAM placement | Useful B | Mode | Completion cycles | Occupied B |
|---|---:|---|---:|---:|
| Linear 64 B at 0 | 64 | `size_only` | 56 | 64 |
|  |  | `sram_address` | 56 | 64 |
|  |  | `sram_4k` | 56 | 64 |
| Linear 64 B at 1 | 64 | `size_only` | 56 | 64 |
|  |  | `sram_address` | 60 | 96 |
|  |  | `sram_4k` | 56 | 64 |
| Linear 80 B at 49 | 80 | `size_only` | 60 | 96 |
|  |  | `sram_address` | 64 | 128 |
|  |  | `sram_4k` | 60 | 96 |
| Linear 64 B at 4090 | 64 | `size_only` | 56 | 64 |
|  |  | `sram_address` | 60 | 96 |
|  |  | `sram_4k` | 60 | 96 |
| 2D, 4 x 20 B, base 48, stride 64 | 80 | `size_only` | 67 | 128 |
|  |  | `sram_address` | 83 | 256 |
|  |  | `sram_4k` | 67 | 128 |
| 3D, 2 x 3 x 20 B, base 48, row 64 | 120 | `size_only` | 75 | 192 |
|  |  | `sram_address` | 99 | 384 |
|  |  | `sram_4k` | 75 | 192 |
| Scatter, five 4 B elements at offset 62 mod 64 | 20 | `size_only` | 71 | 160 |
|  |  | `sram_address` | 91 | 320 |
|  |  | `sram_4k` | 71 | 160 |
| Gather, same placement | 20 | `size_only` | 71 | 160 |
|  |  | `sram_address` | 91 | 320 |
|  |  | `sram_4k` | 71 | 160 |

The 64-byte matrix distinguishes the alternatives:

- `sram_address` pays for ordinary max-burst misalignment and fragmentation: affected completion rises 6.7-32.0%, and fragmented occupied bytes rise as much as 2x.
- `sram_4k` matches `size_only` away from a 4 KiB crossing but raises the 4090+64 B case from 56 to 60 cycles and from 64 to 96 occupied bytes.
- An independent gate uses an 8192-byte maximum burst. At address 4090, `size_only` and `sram_address` both remain 56 cycles/64 B, while `sram_4k` remains 60 cycles/96 B. This proves the page rule is not an alias for maximum-burst alignment.

Queued gates also distinguish runtime consumption. A misaligned max-burst case reverses least-projected binding only in `sram_address`; a 4 KiB-crossing case reverses it in `sram_4k`. These are deterministic planning inputs, not queue-aware memory throughput.

## Gain versus sacrifice

| Dimension | `size_only` | `sram_address` | `sram_4k` |
|---|---|---|---|
| Throughput | Optimistic for constrained bridges; can represent hidden realignment | Exposes recurring misalignment/fragment pressure | Exposes sparse page-crossing pressure; sustained throughput remains unmodeled |
| Latency | Lowest in affected rows | +6.7-32.0% in this matrix | +7.1% only for the measured page crossing |
| Area/resources | Expected realignment/coalescing buffers and steering, or permissive protocol | Modulo/boundary tracking plus more command credits | 12-bit low-address check/comparator plus split/merge state; magnitudes unquantified |
| Power/energy | Fewer visible commands/beats but hidden steering activity | Up to 2x occupied interface bytes in fragmented rows | +50% occupied bytes in the measured crossing; physical energy uncalibrated |
| SRAM/DRAM traffic | Useful bytes unchanged | Useful bytes unchanged; local interface occupancy exposed | Useful bytes unchanged; local interface occupancy exposed; external traffic not modeled |
| Numerical accuracy | Byte-exact, unchanged | Byte-exact, unchanged | Byte-exact, unchanged |
| Control complexity | Most capable span handling is assumed | Boundary counter at every max-burst region | Independent 4 KiB legality and response merge across split commands |
| Verification burden | Must prove hidden realignment behavior | Must gate alignment, tails, strides, indices, and directional limits | Must gate just-below/at/across 4 KiB plus maximum lengths and both directions |
| Compiler/runtime | Placement appears insensitive | Burst alignment/padding can reduce command pressure | Page-aware allocation/descriptor splitting can avoid rare penalties |

Area, power, energy, and sustained throughput are not quantified by the current cmodel. The expected directions above are qualitative hardware rationale, not measured silicon costs.

## Configuration and implementation

`tu.dma.burst_boundary_mode` accepts `size_only`, `sram_address`, and `sram_4k` through YAML/JSON, generated constants/runtime defaults, canonical parsing/validation/conversion, top-level initialization, and live DMA state. Unsupported names and runtime IDs fail closed. Existing initializer wrappers and zero-initialized runtime callers remain `size_only`.

Relevant paths:

- `config/tu_config.yaml`
- `scripts/gen_config.py`, `tu_cmodel/tu_config.h`
- `tu_cmodel/infra/config.{h,c}`
- `tu_cmodel/dma_descriptor.{h,c}`
- `tu_cmodel/tu_cmodel.c`
- `tests/test_dma_burst_boundary_sweep.c`

## Fidelity limits

This is deterministic SRAM-side command geometry, not AXI, NoC, IOMMU, cache-line, virtual-memory, or DRAM simulation. The fixed 4 KiB boundary is intentionally protocol-specific; alternate page sizes are not modeled because no current producer exposes a physical-address/page contract. The model omits byte enables, read-modify-write, adjacent-index merging, finite FIFO/credits, backpressure, response reordering, shared SRAM/DRAM contention, and calibrated area/power. Occupied bytes are DMA-interface lane occupancy, not automatically off-chip bytes.

## Verification

```sh
make test-dma-burst-boundary-sweep
make test-config-generation test-config test-dma
make config-docs
make clean && make
make test-quick
```
