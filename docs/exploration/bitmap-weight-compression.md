# Bitmap Sparse Weight Streams versus RLE

**Date:** 2026-07-21
**Status:** Implemented and verified
**Type:** Cmodel-linked functional codec and payload-cycle sweep (`make test-weight-compression-sweep`)

## Design question and hypothesis

Can an exact bitmap-plus-packed-values stream cover randomly distributed FP16 zeros, where RLE expands or underperforms, without replacing raw or RLE modes that remain physically useful?

The hypothesis was that a one-bit occupancy map has placement-independent metadata cost and should therefore reduce traffic for random sparsity, while RLE should remain preferable for long zero/value runs. A per-tensor selector should retain raw for dense unique tensors, RLE for clustered tensors, and bitmap for scattered zeros.

## Runtime alternatives retained

| Runtime type | Payload alternatives | Why hardware might choose it |
|---|---|---|
| `none` (default) | Raw FP16 | Fixed-rate DMA, no decoder/FIFO, minimum area/control/verification cost |
| `rle` | Explicit exact/epsilon RLE | Small count FSM and excellent compression for clustered or repeated weights; compiler guarantees suitability |
| `adaptive_rle` | Framed raw or RLE | Backward-compatible two-path selector for systems that do not provision a bitmap decoder |
| `bitmap` | Explicit exact bitmap sparse stream | Predictable metadata and position reconstruction for randomly distributed zeros |
| `adaptive` | Framed raw, RLE, or bitmap | Flexible offline packing for heterogeneous tensors; selects the smallest payload, with raw winning ties |

The conservative default remains disabled/`none`. No prior public mode or numerical default changed.

```json
"weight_compression": {
  "enabled": true,
  "type": "adaptive",
  "rle_epsilon": 0.0
}
```

`rle_epsilon` affects only RLE candidates. Bitmap is bit-exact. With epsilon greater than zero, `adaptive` can select a lossy RLE payload; model-level accuracy authorization remains a compiler/runtime responsibility.

## Bitmap wire format

The portable payload is:

| Field | Size |
|---|---:|
| FP16 element count | 4 bytes |
| Nonzero count | 4 bytes |
| Occupancy bitmap | `ceil(elements / 8)` bytes |
| Packed nonzero FP16 bit patterns | `2 * nonzero_count` bytes |

A set bit indicates a stored FP16 bit pattern. Exact bit patterns are retained, including negative zero and NaN payloads; only positive-zero bit pattern `0x0000` is omitted. Validation checks exact stream length, popcount/nonzero-count agreement, zero padding bits, destination capacity, and count bounds.

The existing 16-byte adaptive frame now accepts an explicit BITMAP codec tag in addition to RAW and RLE. Version remains 1: old streams decode unchanged, while older decoders safely reject the previously unknown bitmap tag.

## Workload and model

- 4,096 FP16 weights; raw size 8,192 bytes
- 256-bit / 32-byte DMA bus
- Exact codecs (`epsilon=0`)
- Payload cycles = `ceil(payload_bytes / 32)`
- Random placement uses a deterministic LCG
- Clustered placement uses one zero prefix and 16-element repeated-value runs
- **Not modeled:** bitmap/RLE decoder throughput, metadata/data alignment, FIFO backpressure, burst fragmentation, area, and decoder energy

## Measured trade-off matrix

The `adapt` column includes the 16-byte frame; explicit RLE and bitmap columns do not.

| Workload | NNZ | Runs | Raw B/cyc | RLE B/cyc | Bitmap B/cyc | Adaptive B/cyc | Selected |
|---|---:|---:|---:|---:|---:|---:|---|
| Alternating zero/value | 2,048 | 4,096 | 8,192 / 256 | 24,584 / 769 | 4,616 / 145 | 4,632 / 145 | BITMAP |
| Random zero 10% | 3,685 | 3,953 | 8,192 / 256 | 23,726 / 742 | 7,890 / 247 | 7,906 / 248 | BITMAP |
| Clustered zero 10% | 3,687 | 232 | 8,192 / 256 | 1,400 / 44 | 7,894 / 247 | 1,416 / 45 | RLE |
| Random zero 30% | 2,881 | 3,700 | 8,192 / 256 | 22,208 / 694 | 6,282 / 197 | 6,298 / 197 | BITMAP |
| Clustered zero 30% | 2,868 | 181 | 8,192 / 256 | 1,094 / 35 | 6,256 / 196 | 1,110 / 35 | RLE |
| Random zero 50% | 2,053 | 3,059 | 8,192 / 256 | 18,362 / 574 | 4,626 / 145 | 4,642 / 146 | BITMAP |
| Clustered zero 50% | 2,048 | 129 | 8,192 / 256 | 782 / 25 | 4,616 / 145 | 798 / 25 | RLE |
| Random zero 70% | 1,243 | 2,112 | 8,192 / 256 | 12,680 / 397 | 3,006 / 94 | 3,022 / 95 | BITMAP |
| Clustered zero 70% | 1,229 | 78 | 8,192 / 256 | 476 / 15 | 2,978 / 94 | 492 / 16 | RLE |
| Random zero 90% | 435 | 827 | 8,192 / 256 | 4,970 / 156 | 1,390 / 44 | 1,406 / 44 | BITMAP |
| Clustered zero 90% | 410 | 27 | 8,192 / 256 | 170 / 6 | 1,340 / 42 | 186 / 6 | RLE |
| All zero | 0 | 1 | 8,192 / 256 | 14 / 1 | 520 / 17 | 30 / 1 | RLE |

For random 10–90% zeros, bitmap reduces measured payload cycles by 3.5–82.8% versus raw and by 66.7–76.3% versus RLE. The gain is regime-specific: clustered rows strongly favor RLE (6–45 adaptive cycles), and an actually dense, unique tensor selects raw. The 16-byte adaptive frame occasionally adds one bus beat but prevents selecting a larger candidate.

## Multi-objective gain versus sacrifice

| Dimension | Raw | RLE | Bitmap | Adaptive all |
|---|---|---|---|---|
| Throughput/latency | Fixed 256 payload cycles here | 1–44 on measured clustered rows; 156–769 on scattered rows | 44–247 on measured random-sparse rows; 17 for all-zero, slower than RLE | 1–248, selecting by bytes; decode throughput unquantified |
| Area/resources | No decoder | Run-count FSM and variable-rate FIFO | Bitmap scanner, popcount/check logic, packed-value merge, variable-rate FIFO | Raw bypass plus both decoders, frame parser, mux/state; highest expected cost |
| Power/energy | No decode energy; highest traffic when sparse | DRAM savings can dominate on runs; decoder energy unquantified | DRAM savings on scattered zeros; per-element bitmap scan energy unquantified | Avoids poor payload choice but pays format dispatch and provisioned multi-codec logic |
| SRAM/DRAM traffic | 8,192 bytes | 14–24,584 bytes | 520–7,894 bytes | 30–7,906 bytes in measured matrix |
| Numerical accuracy | Exact | Exact at epsilon 0; optional lossy epsilon | Exact FP16 bit patterns | Exact at epsilon 0; inherits epsilon-RLE loss if enabled |
| Control complexity | Lowest | Variable run lengths | Lockstep bitmap/value streams and alignment | Highest; three paths and per-tensor format state |
| Verification burden | Lowest | Runs, overflow, malformed counts | Bitmap padding/popcount/packed-value bounds | Every payload path, selection boundaries, framing, compatibility |
| Compiler/runtime | No packing | Prefer clustered/repeated tensors | Prefer scattered exact zeros | Offline packer evaluates candidates and carries explicit codec tag |

Area, power, decoder throughput, SRAM port pressure, physical alignment, and verification effort are **qualitative/unquantified**, not fabricated. A real low-cost ASIC may still hard-wire raw or one codec; the pre-spec cmodel keeps all materially distinct modes for comparison.

## Implementation and verification

- `tu_cmodel/memory/weight_compress.{h,c}`: portable bitmap codec, strict validation, adaptive three-way selection, frame/DMA dispatch.
- `tu_cmodel/infra/config.{h,c}`: runtime `bitmap` and `adaptive` parse/validation; default unchanged.
- `tests/test_compress.c`: 21 tests, including random sparse round-trip, negative-zero/NaN preservation, padding corruption, three-way selection, and config/DMA paths.
- `tests/test_weight_compression_sweep.c`: 12-row complete matrix.

Verified commands:

```sh
make test-compress                  # 21/21 pass
make test-weight-compression-sweep # reproduces matrix above
make test-config                    # 18/18 pass
make clean && make
make test-quick
```

## Remaining model gap

Payload bytes are not end-to-end latency. Before recommending decoder provisioning, add configurable bitmap and RLE decode elements/groups per cycle, FIFO/backpressure behavior, and metadata alignment. This is a READY follow-up only after a defensible shared decoder-cycle abstraction and tests are specified; physical area/power still require external characterization.
