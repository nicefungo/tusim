# DMA descriptor SRAM geometry: an event-contract exploration

**Baseline:** `d9e460b247467dbc868cdcf01334f702be44f34f` (2026-11-01). This is an exploration, not an implemented alternative or a physical bank model.

## Question and executable evidence

Does the current descriptor meter distinguish a packed 2D transfer from an equally sized strided transfer that repeatedly addresses the same SRAM bank? The live direct descriptor path copies each row using its SRAM stride but its separate meter walks `ceil(total_bytes / bank_width)` words starting from the SRAM base, irrespective of row addresses and of the SRAM API `word_metering` setting (`tu_cmodel/dma_descriptor.c`, execution and accounting around lines 1671–1724). The meter adds its local penalty to the descriptor completion estimate; it does not replay stalled copies. This finding concerns the direct 2D path exercised here, not all descriptor types or queue behavior.

`tests/explore_dma_sram_geometry.c` executes 32 fresh-region cases: load/store × SRAM API size-only/address-span × base 0/1 × SRAM row stride 4/8/16/20 B. Every case uses four 4-byte rows, a 4-byte packed host stride, four banks of 4-byte words, one grant per 4-cycle refill window, and a 2-cycle local stall penalty. Host bytes and the full 128-byte SRAM region are checked; live per-bank served-plus-stalled counts are checked against the actual descriptor-size meter. The last column in the preserved matrix is an **independent hypothetical** count of all word addresses intersected once per logical row, with no merging across rows. It is not an executed mode, a physical transaction count, or a stall/latency prediction. All 32 complete with byte checks and modeled completion 51 cycles. In 24 of 32 rows the hypothesis differs from the live bank vector (8 aligned, 16 unaligned).

| Base, SRAM row stride | Live bank events (both directions and API modes) | Per-row span hypothesis |
|---|---|---|
| 0, 4 B | 1/1/1/1 | 1/1/1/1 |
| 0, 8 B | 1/1/1/1 | 2/0/2/0 |
| 0, 16 B | 1/1/1/1 | 4/0/0/0 |
| 0, 20 B | 1/1/1/1 | 1/1/1/1 |
| 1, 4 B | 1/1/1/1 | 2/2/2/2 |
| 1, 8 B | 1/1/1/1 | 2/2/2/2 |
| 1, 16 B | 1/1/1/1 | 4/4/0/0 |
| 1, 20 B | 1/1/1/1 | 2/2/2/2 |

The full direction/mode matrix is `evidence/2026-11-01-dma-sram-geometry/matrix.txt`. One row always means four useful bytes even when adjacent logical rows intersect the same aligned word. In a coalescing design those repeated word events could merge; this diagnostic intentionally does not assume that happens.

## Plausible designs and costs

* **Descriptor-size compatibility meter:** the present aggregate abstraction has low model/control complexity and preserves old cycle and counter behavior. It may be useful for a path abstracted away from bank placement, but cannot support conclusions about strided bank conflicts.
* **Per-logical-segment word-span events:** a simple sequencer can charge each row or indexed element for all intersected words; this exposes placement and misalignment and may add modeled stalls. It needs geometry/address traversal and direction/region state, and without replay is still not a physical completion schedule.
* **Merged/coalesced events:** a buffer or scheduler might combine overlapping or adjacent words and reduce duplicate bank requests, but would require defined buffering, segment ordering, masks, capacity and eligibility rules. It has potentially higher storage, steering and verification cost, and its latency and traffic consequences depend on the workload. Reject/split/replay alternatives likewise need status and retirement semantics.

**Throughput and latency:** current direct descriptor estimate is 51 cycles for this matrix; counterfactual throughput or latency has not been measured. **Area/resources and power/energy:** expected address comparators, coalescing buffers and/or split/replay control depend on the selected contract; unquantified. **Traffic:** all cases move 16 useful host/SRAM bytes, while the row-span column counts hypothetical word events, not measured physical SRAM traffic; DRAM traffic is not modeled here. **Accuracy:** copied bytes match; no numerical arithmetic occurs. **Control and verification:** size-only has the smallest modeled surface; per-segment accounting adds geometry and endpoint cases, while coalescing additionally requires overlap, ordering and finite-capacity rules. **Compiler/runtime:** strides, alignment and packing could affect an eventual bank-aware model; no new compiler option or runtime knob is implemented.

## Disposition and verification

Keep the existing default and SRAM API modes unchanged. Descriptor-aware alternatives remain **BLOCKED**, not excluded: define event granularity and coalescing, read/write and internal-transfer endpoints, multicast fanout, temporal grant/replay, counter ownership, error/retirement and interaction with projected queued timing before implementing. A follow-up must cover 2D/3D, scatter/gather, multicast, overlap, zero/unaligned/end spans, direct/sync/async, both directions, defaults and invalid rejection. This experiment does not establish an implementation-ready universal rule for those cases.

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make clean && make && make test-quick
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel tests/explore_dma_sram_geometry.c ./libtucmodel.a -lm -o "$TMPDIR/explore_dma_sram_geometry"
"$TMPDIR/explore_dma_sram_geometry"
make test-sram-word-meter test-dma test-scatter-gather
```

The baseline clean build and quick aggregate passed; the diagnostic passed 32/32 cases, and focused suites passed. A separate clean archive built every library object with ASan/UBSan and the diagnostic passed with leak detection enabled on exercised paths. These gates do not validate unexercised consumers or calibrated hardware timing.
