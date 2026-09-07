# DMA Logical-Segment Burst Accounting

**Date:** 2026-09-05

**Status:** Implemented for live descriptor service and queued projected-cycle binding

**Question:** Should a DMA descriptor's burst-command cost be computed from aggregate useful bytes, or should discontiguous rows and indexed elements begin independent command segments?

## Hypothesis and realistic alternatives

The existing burst model used `ceil(total_bytes / max_burst_bytes)` for every descriptor. That is a useful low-cost abstraction for a mover that coalesces a descriptor into one contiguous command stream, but it lets bytes from different strided rows or scatter/gather indices share a burst. A descriptor sequencer without cross-segment coalescing must issue each discontiguous row or indexed element independently.

| Alternative | Why hardware may choose it | Principal sacrifice |
|---|---|---|
| `aggregate` | Aggressive gather/coalescing hardware, a downstream bridge that accepts rich descriptors, or a compatibility lower bound | Requires buffering/address aggregation to realize physically; understates commands when segments cannot merge |
| `logical_segments` | Simple row/index sequencer that emits commands independently at each discontinuity | More exposed command issue and potentially lower latency/throughput for fragmented descriptors |

`aggregate` remains the zero/default mode. `logical_segments` is runtime-selectable. Linear descriptors are identical in both modes. In logical mode, 2D descriptors segment by row, 3D descriptors by row within each depth plane, and scatter/gather descriptors by indexed element. Multicast remains one aggregate descriptor stream because the current multicast contract does not expose independent destination-command timing.

## Executable sweep

```sh
make test-dma-segmentation-sweep
```

Controls: 64-byte maximum burst, 3 visible issue cycles per burst, 256-bit payload path, 50-cycle base latency, SRAM bandwidth metering disabled. Completion includes the initial async start tick. Payload serialization remains `ceil(total_bytes / 32)` and only command count changes.

| Descriptor | Useful bytes / geometry | Aggregate bursts | Logical bursts | Aggregate completion | Logical completion | Added latency |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 80 B | 2 | 2 | 60 cycles | 60 cycles | 0 |
| Strided 2D | 4 rows × 20 B | 2 | 4 | 60 cycles | 66 cycles | 6 cycles (10.0%) |
| Strided 3D | 2 depths × 3 rows × 20 B | 2 | 6 | 61 cycles | 73 cycles | 12 cycles (19.7%) |
| Gather | 5 indices × 4 B | 1 | 5 | 55 cycles | 67 cycles | 12 cycles (21.8%) |

The test gates byte-exact movement in both modes and proves that queued least-projected-cycle binding consumes the same segmentation-aware estimate. With one 4×20 B strided descriptor and one 100 B linear descriptor waiting on separate channels, aggregate accounting selects the strided channel (59 vs 60 projected cycles), while logical accounting selects the linear channel (65 vs 60). This is a policy-input reversal, not evidence that dynamic binding is universally preferable.

## Implementation and configuration path

`YAML/JSON dma.burst_segmentation → generated constants/runtime field → canonical default/parser/validation → canonical-to-runtime conversion → tu_init_with_config() → g_tu_dma.burst_segmentation → live descriptor and queued projection burst-count helper`.

Accepted values:

- `aggregate` — compatibility/default.
- `logical_segments` — segment-aware issue accounting.

Legacy DMA initializers call the aggregate wrapper. A zero-initialized runtime configuration therefore remains aggregate. Unsupported names and runtime IDs fail closed.

The config generator was smoke-tested to a temporary header. It emits the new macros and runtime field, but still omits unrelated manually integrated macros already present in the checked-in header. The generated file was therefore not installed wholesale; the checked-in header received only the verified additions. Generator drift remains a separate maintenance defect.

## Gain versus sacrifice

- **Throughput:** Logical segmentation exposes command-rate pressure in fragmented descriptors; aggregate mode is an optimistic coalescing bound. The sweep is finite isolated service, not sustained producer/consumer throughput.
- **Latency:** Exact for the implemented base + aggregate payload serialization + per-logical-segment burst issue formula. The largest measured increase is 21.8% for five 4-byte gather elements, under the stated 3-cycle issue cost.
- **Area/resources:** Aggregate behavior is expected to require segment coalescing buffers, address comparison/merge logic, and credits. Logical mode can use a simpler sequencer. Queue entries, gates, SRAM ports, and physical area are unquantified.
- **Power/energy:** Extra commands in logical mode are expected to increase control activity. Aggregate coalescing spends buffer/search energy instead. Neither command nor coalescer energy is parameterized, so net energy is unquantified.
- **SRAM/DRAM traffic:** Useful bytes and byte-exact results are unchanged. The model does not add alignment overfetch or split traffic at address/protocol boundaries; occupied interface bytes are unquantified here.
- **Numerical accuracy:** Unchanged; all rows preserve exact bytes.
- **Control complexity:** Logical mode needs descriptor-type geometry but no cross-segment merge state. Aggregate hardware needs enough reassembly/coalescing capability to justify its lower command count.
- **Verification burden:** Both modes require linear equality, 2D/3D row multiplicity, indexed-element multiplicity, queued-estimate consistency, default compatibility, parse propagation, and fail-closed invalid-mode tests.
- **Compiler/runtime:** A compiler can prefer contiguous packing or combine nearby indices when logical segmentation makes command pressure visible. It must not assume bytes from different rows or indices coalesce unless aggregate mode is selected.

## Fidelity limits

This is deterministic command-count accounting, not a command-queue microarchitecture. Base-latency scope and payload-beat scope are independent runtime choices; see `dma-base-latency-scope.md` and `dma-payload-segment-alignment.md`. Payload can now be packed across the descriptor or rounded per logical segment, but external address alignment and protocol overfetch remain unmodeled. The model does not represent burst-boundary crossing, command FIFO depth, finite credits, command/data overlap, coalescer capacity, segment ordering, DRAM row effects, shared SRAM/DRAM bandwidth, retries, physical area/power, or calibration. Scatter/gather treats each indexed element as a segment even when adjacent indices could be merged; a future coalescer model requires an explicit ordering/window contract and trace evidence.

## Verification

```sh
make test-dma-segmentation-sweep
make test-dma-directional-issue-sweep
make test-dma-burst-issue-sweep
make test-dma
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.segmentation.h
make config-docs
make clean && make
make test-quick
```
