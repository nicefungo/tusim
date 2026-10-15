# Multicast/Broadcast DMA Descriptor

**Status:** executable host-to-SRAM replication through the descriptor API.
**Contract re-audit:** 2026-10-15. See [atomic-rejection evidence](exploration/multicast-atomic-rejection.md).

## Scope and motivation

A contiguous host source is copied to N SRAM destinations, each with its own region and byte offset. Weight tiles, constants, and shared activations can use one descriptor instead of N producer submissions. This proves descriptor-level replication, not a physical broadcast tree, reduced DRAM reads, or a compiler/ISA integration path.

```c
tu_dma_descriptor_t *tu_dma_desc_create_multicast(
    uint8_t channel,
    const void *src_host,
    tu_sram_region_t **dst_regions, uint32_t *dst_offsets,
    uint32_t num_destinations, uint32_t elem_size, uint32_t elem_count);
```

The constructor copies the destination pointer/offset arrays into owned storage. SRAM regions and the source payload must survive accepted execution. Null array/source arguments and zero destination count are rejected by construction; individual target regions and spans are validated before execution/admission.

## Fail-closed bounds contract

Every destination must satisfy `offset + payload_bytes <= region_size` using the shared overflow-checked span validator. An exact-end span is legal. A null region or one overflowing target invalidates the **whole descriptor**, regardless of its target-list position.

- `tu_dma_execute_desc(desc)` returns `void`: rejection leaves a fresh descriptor incomplete, copies no target, and changes no DMA engine, channel, queue, byte, or cycle accounting. The caller still owns the descriptor.
- `tu_dma_submit_desc(desc)` returns zero on rejection and destroys the rejected chain. Do not inspect or destroy it again. Validation precedes binding, admission, and copying for both synchronous and asynchronous engines.
- Valid synchronous submission leaves the completed descriptor caller-owned. Valid asynchronous work must finish and detach before caller destruction. `tu_dma_desc_destroy(head)` also destroys nodes reachable through `next`.

The private copy helper still contains defensive per-target checks. They do **not** define a supported skip-and-complete policy: public execution/admission validates all targets first. Partial faults need an explicit status, signaling, rollback, and retirement contract before they can be modeled.

## Accounting and fidelity

`total_bytes = elem_count * elem_size * num_destinations` counts replicated destination bytes, not unique host-source bytes. The common descriptor timing helpers apply configured base latency, payload occupancy, burst issue, and scope settings. In default size-only/descriptor-packed/zero-issue mode:

```
modeled service = read_base_cycles + ceil(total_bytes / bus_width_bytes)
occupied bytes  = ceil(total_bytes / bus_width_bytes) * bus_width_bytes
```

For three 8-byte targets at the ends of 64/128/256-byte regions, all three public surfaces produce 24 useful bytes, 32 occupied bytes, and 51 aggregate service cycles with the default 50-cycle base and 32-byte bus. Async completion additionally follows the engine's issue/tick convention; service cycles are not an end-to-end physical latency measurement.

The implementation performs sequential host `memcpy` operations. Its aggregate fanout charge is not a link-level routing, SRAM-port, simultaneous-write, multicast-tree, or source-read-reuse model. No physical broadcast speedup follows from one descriptor.

## Configuration, ISA, and performance counters

`dma.multicast_enabled` is parsed/stored canonical metadata; the constructor does not consult it. It is not an executable enable/disable gate. `TU_ISA_DMA_BROADCAST = 0x57` has a declaration/name and classification helpers, not a demonstrated generic decoder-to-multicast path. These surfaces must be audited separately before claiming ISA or compiler support.

`tu_perf_dma_record_read()` can explicitly increment `dma_transfers_multicast`. The focused performance-counter test calls that recording API itself; automatic wiring from descriptor execution into a separate `tu_perf_counters_t` is not established. Live descriptor accounting is in `g_tu_dma`.

## Verification

```sh
make test-multicast
make test-quick
```

The multicast target links `./libtucmodel.a` explicitly, belongs to both quick and full aggregates, and is removed by `make clean`.

`tests/test_multicast.c` runs 30 cases: existing creation/copy/fanout/chain/counter tests, 18 atomic rejection cases (direct/sync/async × overflow/null × first/middle/last invalid target), and three multi-region exact-end controls. Controls check full-region payload/guard bytes and default accounting. Rejections check every region and a byte-preserved engine-state snapshot. Sanitizers and a safe bypass-validation mutant provide additional evidence in the linked exploration note.

## Alternatives and costs

| Choice | Why hardware might use it | Cost / current status |
|---|---|---|
| Whole-command prevalidation | Prevent partial corruption; simple command-level failure | Geometry walk/comparators and verification; current executable contract. Hardware checking latency, area, and energy unquantified. |
| Per-target partial fault | Streaming issue can discover errors late | Needs per-target/error/completion semantics and recovery; blocked, not silently approximated by skip-and-complete. |
| Repeated unicast | Simple mover and explicit software placement | More descriptors and potentially repeated reads; no calibrated comparison here. |
| Physical multicast tree/bus | Source reuse and parallel distribution for shared operands | Fanout wires, buffers, destination ports, flow control, routing and power; not implemented by this descriptor abstraction. |

Valid traffic, numerical bytes, default architecture, and modeled service are unchanged by the regression repair. Verification coverage increases, but there is no measured throughput gain. Strided multicast, per-destination completion, SRAM-source multicast, shared-memory contention, physical area/power/energy, and calibrated broadcast timing remain outside this contract.
