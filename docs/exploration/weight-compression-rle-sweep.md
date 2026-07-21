# RLE Weight Compression: Placement Sensitivity and Adaptive Framing

**Date:** 2026-07-18
**Status:** Implemented and verified
**Type:** Cmodel-linked sweep (`make test-weight-compression-sweep`)

> **2026-07-21 follow-up:** the identical-pattern audit in
> [`bitmap-weight-compression.md`](bitmap-weight-compression.md) added explicit
> `bitmap` and three-way `adaptive` (raw/RLE/bitmap) runtime modes. This document
> preserves the original raw/RLE experiment; use the follow-up for the current
> 12-row codec matrix and implementation surface.

## Design question

When is a simple FP16 run-length codec physically plausible, and how can a TU/compiler avoid RLE's severe expansion on unsuitable tensors without ambiguously guessing the payload format? The hypothesis is that **placement of repeated values, not scalar sparsity alone**, determines whether RLE helps, so selection must be per tensor.

## Runtime alternatives retained

- **NONE (default):** raw unframed FP16. Real hardware may hard-wire this for fixed-rate DMA, no decoder area/power, and the lowest control and verification burden.
- **RLE (`type: "rle"`):** legacy explicit RLE stream. This is useful when the compiler guarantees clustered/block-pruned or repeated quantized weights. It minimizes metadata on good tensors but can expand traffic by 3×.
- **ADAPTIVE_RLE (`type: "adaptive_rle"`):** encode each tensor as a versioned 16-byte frame containing either raw FP16 or RLE, whichever payload is smaller. Ties select raw. This is plausible where a compiler/offline packer can inspect weights and the runtime supports two decode paths.
- **Epsilon RLE (`rle_epsilon > 0`):** available for RLE and adaptive selection, but lossy. It may create longer runs after quantization; model-level accuracy authorization remains a compiler/user responsibility.

The canonical JSON block is:

```json
"weight_compression": {
  "enabled": false,
  "type": "none",
  "rle_epsilon": 0.0
}
```

The original experiment supported `none`, `rle`, and `adaptive_rle`. The current
runtime additionally supports `bitmap` and `adaptive`; defaults remain
disabled/NONE, preserving existing raw DMA semantics.

## Explicit adaptive frame

The decoder never infers raw versus RLE from payload contents. The 16-byte frame contains:

| Field | Bytes | Meaning |
|---|---:|---|
| Magic | 4 | `TU_WEIGHT_FRAME_MAGIC` |
| Version | 1 | currently 1; unknown versions rejected |
| Payload codec | 1 | RAW or RLE |
| Reserved | 2 | zero; nonzero rejected |
| Element count | 4 | decompressed FP16 count |
| Payload bytes | 4 | exact payload length |

RLE payloads retain their 8-byte element/run header and portable 6-byte `{FP16 value, uint32 count}` entries. The adaptive encoder first counts runs, then emits RLE only if its payload is strictly smaller than raw. Therefore:

`adaptive_bytes <= raw_bytes + 16-byte frame`

This bound is tested for both selected formats. Corrupt/truncated frames, unknown versions, unknown codecs, inconsistent lengths, and mismatched RLE element counts are rejected.

## Workload and cycle model

- 4,096 FP16 weights (8,192 raw bytes)
- 256-bit / 32-byte DMA bus
- Exact RLE (`epsilon=0`)
- Payload cycles: `ceil(bytes / 32)`
- No descriptor, decode, contention, FIFO, or format-dispatch cycles

Random-zero rows use deterministic LCG placement. Clustered-zero rows put zeros contiguously and organize remaining values in runs of 16.

## Measured trade-off matrix

| Workload | Runs | Raw B/cyc | RLE B/cyc | Adaptive B/cyc | Selected |
|---|---:|---:|---:|---:|---|
| Alternating | 4,096 | 8,192 / 256 | 24,584 / 769 | 8,208 / 257 | RAW |
| Random zero 50% | 3,059 | 8,192 / 256 | 18,362 / 574 | 8,208 / 257 | RAW |
| Clustered zero 50% | 129 | 8,192 / 256 | 782 / 25 | 798 / 25 | RLE |
| Random zero 70% | 2,112 | 8,192 / 256 | 12,680 / 397 | 8,208 / 257 | RAW |
| Clustered zero 70% | 78 | 8,192 / 256 | 476 / 15 | 492 / 16 | RLE |
| Random zero 90% | 827 | 8,192 / 256 | 4,970 / 156 | 4,986 / 156 | RLE |
| Clustered zero 90% | 27 | 8,192 / 256 | 170 / 6 | 186 / 6 | RLE |
| All zero | 1 | 8,192 / 256 | 14 / 1 | 30 / 1 | RLE |

Adaptive selection eliminates the prior 1.55–3.00× RLE traffic expansion on alternating/random 50–70% tensors, paying one extra 32-byte bus beat (257 versus 256 cycles) for the frame. On compressible tensors it preserves most RLE savings. Header alignment costs one cycle for clustered 70% (16 versus 15); other measured RLE rows remain in the same rounded payload-cycle bucket.

## Gain versus sacrifice

| Dimension | NONE | Explicit RLE | ADAPTIVE_RLE |
|---|---|---|---|
| Throughput/latency | Fixed 256 payload cycles here | 1–25 cycles on clustered cases, but 397–769 on poor cases | 1–257; bounded fallback prevents expansion beyond one frame, but never beats bare RLE |
| Area/resources | No codec/FIFO | RLE decoder, count FSM, variable-rate buffering | RLE decoder plus raw bypass, frame parser, mux, and format state; unquantified |
| Power/energy | No decode energy; 8,192 DRAM bytes | Large DRAM savings on long runs, but increased traffic on poor runs; decoder energy unquantified | Avoids poor-case expansion; extra frame traffic and format logic energy unquantified |
| SRAM/DRAM traffic | Fixed raw bytes | 14–24,584 bytes measured | 30–8,208 bytes measured |
| Numerical accuracy | Exact | Exact at epsilon 0; lossy with epsilon > 0 | Same semantics as selected payload; selection itself adds no error |
| Control complexity | Lowest/fixed-rate | Variable-rate decode/backpressure | Highest: version/codec dispatch and two payload paths |
| Verification burden | Lowest | RLE corruption and round-trip | Both paths, selection boundary, frame versioning, corruption, and compatibility |
| Compiler/runtime | No packing | Compiler must know tensor suitability | Offline encoder measures actual payload; runtime carries per-tensor format tag |

### Why all modes remain

Adaptive is not universally “better.” A low-cost inference ASIC may prefer NONE because one extra payload cycle, two decode paths, and variable-rate verification are not justified. A tightly controlled compiler/runtime may choose explicit RLE to omit the 16-byte frame and raw bypass. A flexible accelerator serving heterogeneous models may accept adaptive area/control cost to guarantee bounded traffic per tensor.

## Implementation

- `tu_cmodel/memory/weight_compress.{h,c}`: adaptive type, 16-byte frame, selection, validation, decode, DMA dispatch, bounded allocator helper.
- `tu_cmodel/infra/config.{h,c}`: parse and validate `adaptive_rle`.
- `tests/test_compress.c`: 17 tests covering legacy modes plus adaptive raw/RLE decisions, exact round trips, config/DMA mapping, and corrupt frame rejection.
- `tests/test_weight_compression_sweep.c`: complete raw/RLE/adaptive comparison.

The RLE encoder now accepts an exactly sized output buffer based on actual run count instead of unnecessarily requiring worst-case capacity. Public legacy stream semantics and default architecture behavior are unchanged.

## Verification

```sh
make test-compress                  # 17/17 pass
make test-weight-compression-sweep # reproduces the 8-row matrix
make test-config
make clean && make
make test-quick
```

## Physical-model limitations

Area, decoder energy, decode throughput, FIFO depth, backpressure, DRAM burst fragmentation, frame-fetch latency, and offline packing cost are **unquantified**. Expected directions are documented above, but the payload-only cycle model cannot establish whether format dispatch overlaps DMA or how much decoder throughput is required. Epsilon mode is functionally available; no model-level accuracy result is claimed.
