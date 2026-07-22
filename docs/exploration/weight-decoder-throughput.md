# Weight-Stream Decoder Throughput and DMA Overlap

**Date:** 2026-07-22
**Status:** Implemented and verified
**Type:** Cmodel-linked codec/cycle sweep (`make test-weight-compression-sweep`)

## Design question

When do raw, RLE, bitmap, and adaptive weight streams reduce end-to-end load latency once a physically finite decompressor must reconstruct dense FP16 weights, rather than assuming decompression is free?

The hypothesis was that payload-only measurements overstate compression speedups. A decoder narrower than the 256-bit DMA bus's 16 FP16 values/cycle should become the bottleneck; a decoder matching that width should at best recover raw latency; and a wider decoder is needed to turn traffic reduction into latency reduction while materializing dense SRAM data.

## Runtime alternatives retained

The cmodel retains all stream formats (`none`, `rle`, `adaptive_rle`, `bitmap`, `adaptive`) and adds independent runtime decoder provisioning:

```json
"weight_compression": {
  "enabled": true,
  "type": "adaptive",
  "decoder_enabled": true,
  "decoder_overlap_dma": true,
  "decoder_elements_per_cycle": 16,
  "rle_runs_per_cycle": 8,
  "bitmap_elements_per_cycle": 16
}
```

- `decoder_enabled=false` is the backward-compatible payload-only default.
- `decoder_overlap_dma=true` models a streaming DMA→decoder pipeline as `max(DMA, decode)`; `false` models a staged implementation as `DMA + decode`.
- `decoder_elements_per_cycle` is the dense FP16 reconstruction/write width shared by compressed codecs.
- `rle_runs_per_cycle` is run-descriptor issue width.
- `bitmap_elements_per_cycle` is occupancy-map scan width.
- Raw bypass has no decode cost. Invalid zero widths are rejected even when the model is disabled, keeping latent configurations valid.

## Cycle model

For a stream of `N` reconstructed elements:

- `dma_cycles = ceil(encoded_bytes / dma_bus_bytes)` (the adaptive frame is included)
- `output_cycles = ceil(N / decoder_elements_per_cycle)`
- `rle_metadata_cycles = ceil(run_count / rle_runs_per_cycle)`
- `bitmap_metadata_cycles = ceil(N / bitmap_elements_per_cycle)`
- `decode_cycles = max(output_cycles, codec_metadata_cycles)`
- overlapped total = `max(dma_cycles, decode_cycles)`
- staged total = `dma_cycles + decode_cycles`

This is a bounded-throughput pipeline model, not a gate-level decoder. It parses and validates the actual encoded stream, so adaptive codec tags, run counts, element counts, frame bytes, and malformed streams affect executable behavior.

## Workload and decoder configurations

- 4,096 FP16 weights; raw = 8,192 bytes
- 256-bit DMA bus = 32 bytes/cycle = 16 raw FP16 values/cycle
- exact compression (`rle_epsilon=0`)
- streaming overlap enabled for the sweep
- same deterministic random and clustered patterns as `bitmap-weight-compression.md`

| Profile | Dense outputs/cycle | RLE runs/cycle | Bitmap positions/cycle | Hardware motivation and cost direction |
|---|---:|---:|---:|---|
| Serial | 1 | 1 | 1 | Minimum lanes/control; lowest expected area and decoder dynamic power, but cannot feed a wide DMA path |
| Balanced | 8 | 4 | 8 | Moderate lane/FIFO replication for cost-sensitive accelerators; still half the raw bus rate |
| Wide | 16 | 8 | 16 | Matches the 256-bit raw ingress rate; more write ports/banking and metadata logic |
| Extra-wide | 32 | 16 | 32 | Expands compressed data at 2× raw ingress rate; highest lane/FIFO/SRAM-port cost, needed for latency benefit in this dense-reconstruction architecture |

## Complete measured trade-off matrix

Each codec cell is `serial / balanced / wide / extra-wide` total cycles. Raw is always 256 cycles because it bypasses the decoder. `Adaptive format` is selected by encoded byte count, not by the cycle model.

| Workload | NNZ | Runs | Adaptive format | RLE cycles S/B/W/X | Bitmap cycles S/B/W/X | Adaptive cycles S/B/W/X |
|---|---:|---:|---|---:|---:|---:|
| Alternating | 2,048 | 4,096 | Bitmap | 4096 / 1024 / 769 / 769 | 4096 / 512 / 256 / 145 | 4096 / 512 / 256 / 145 |
| Random zero 10% | 3,685 | 3,953 | Bitmap | 4096 / 989 / 742 / 742 | 4096 / 512 / 256 / 247 | 4096 / 512 / 256 / 248 |
| Cluster zero 10% | 3,687 | 232 | RLE | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 247 | 4096 / 512 / 256 / 128 |
| Random zero 30% | 2,881 | 3,700 | Bitmap | 4096 / 925 / 694 / 694 | 4096 / 512 / 256 / 197 | 4096 / 512 / 256 / 197 |
| Cluster zero 30% | 2,868 | 181 | RLE | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 196 | 4096 / 512 / 256 / 128 |
| Random zero 50% | 2,053 | 3,059 | Bitmap | 4096 / 765 / 574 / 574 | 4096 / 512 / 256 / 145 | 4096 / 512 / 256 / 146 |
| Cluster zero 50% | 2,048 | 129 | RLE | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 145 | 4096 / 512 / 256 / 128 |
| Random zero 70% | 1,243 | 2,112 | Bitmap | 4096 / 528 / 397 / 397 | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 128 |
| Cluster zero 70% | 1,229 | 78 | RLE | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 128 |
| Random zero 90% | 435 | 827 | Bitmap | 4096 / 512 / 256 / 156 | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 128 |
| Cluster zero 90% | 410 | 27 | RLE | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 128 |
| All zero | 0 | 1 | RLE | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 128 | 4096 / 512 / 256 / 128 |

The full executable output also labels whether DMA or decoder is the final bound.

## Findings: gains and sacrifices

1. **A narrow decompressor reverses the payload-only conclusion.** Serial codecs take 4,096 cycles for every tested compressed stream—16× raw's 256 cycles—because reconstructing one dense FP16/cycle dominates even a 14-byte all-zero RLE payload. Balanced provisioning takes 512–1,024 cycles (2–4× raw). These are plausible low-area choices when DRAM traffic/energy or capacity matters more than tensor-load latency.
2. **Matching the bus is only a break-even point for dense reconstruction.** The wide 16-element/cycle decoder reaches 256 cycles for useful RLE/bitmap streams, equal to raw, while metadata-heavy RLE remains 397–769 cycles. It can reduce DRAM traffic without improving measured load latency.
3. **Extra-wide expansion unlocks regime-specific latency gains.** At 32 outputs/cycle, adaptive takes 128 cycles on clustered 10–90% zeros and random ≥70% zeros (2× lower latency than raw), 145–248 cycles on less sparse scattered patterns, and remains 769 cycles for an explicitly bad alternating RLE stream. The gain buys 2× the raw ingress reconstruction width plus wider metadata and SRAM write capability.
4. **Formats remain workload-dependent.** RLE is appropriate for clustered/repeated streams; bitmap is appropriate for scattered zeros; raw avoids decoder provisioning and malformed-variable-rate-stream concerns. No format or decoder width is universally best.
5. **Byte-minimum adaptive selection remains defensible but is not latency-optimal by construction.** Its 16-byte frame costs one cycle in random 10% and 50% extra-wide rows (248 vs explicit bitmap 247; 146 vs 145). A future selector could use hardware-profile-aware latency, but current evidence does not justify more runtime complexity because codec choice itself did not change in this matrix.

## Multi-objective assessment

| Dimension | Serial/balanced decoder | Wide decoder | Extra-wide decoder | Raw bypass |
|---|---|---|---|---|
| Throughput/latency | 2–16× slower than raw here | Break-even for useful streams; metadata-heavy RLE slower | 1.03–2× faster for selected measured regimes; bad RLE still slower | Fixed 256 cycles |
| Area/resources | Lowest expected decoder lanes/FIFOs | Approximately raw-bus reconstruction width | Highest expected lanes, FIFOs, routing, SRAM write bandwidth | No codec hardware |
| Power/energy | Lower decoder activity per cycle but longer active time; net unquantified | More parallel switching; DRAM savings may dominate | Highest instantaneous decoder/SRAM switching; shorter active interval | No decode energy, maximum sparse traffic |
| SRAM/DRAM traffic | Same compressed traffic as wider profiles; dense SRAM output unchanged | Same | Same | 8,192 bytes transferred |
| Numerical accuracy | Exact for bitmap/RLE at epsilon 0 | Same | Same | Exact raw; epsilon-RLE loss remains separately configurable |
| Control complexity | Small FSM, easiest timing closure | Wider issue and backpressure | Multi-lane expansion, banking/alignment, hardest timing closure | Lowest |
| Verification burden | Width/rate and malformed stream tests | Adds boundary/parallel-lane hazards | Highest inter-lane, FIFO, bank, and ordering state space | Lowest |
| Compiler/runtime | Must know codec and tolerate load stalls | Can hide most decode under DMA | Can exploit compression latency; provisioning profile should inform packing | No packing/format state |

Physical area, frequency impact, decoder energy, SRAM port conflicts, burst fragmentation, FIFO depth/backpressure, and compute overlap are **unquantified**. Expected directions are qualitative; no fabricated values are used.

## Implementation and verification

- `tu_cmodel/infra/config.{h,c}`: five decoder knobs, defaults, parsing, validation, dump, and generated-doc fields.
- `tu_cmodel/memory/weight_compress.{h,c}`: config mapping, stream-aware cycle stats, RLE/bitmap/adaptive metadata parsing, overlap/serialization modes.
- `tests/test_compress.c`: 24 tests total; new checks cover serial/wide profiles, overlap off, bitmap scan bottlenecks, adaptive frame parsing, disabled compatibility, config mapping, and invalid widths.
- `tests/test_weight_compression_sweep.c`: four hardware profiles × twelve workloads × raw/RLE/bitmap/adaptive.
- `Makefile`: `test-config` now links the archive it depends on, preventing stale shared-library execution after config ABI changes.

Verified commands:

```sh
make test-compress                  # 24/24 pass
make test-config                    # 18/18 pass
make test-weight-compression-sweep # 48 profile/workload rows
```

Final clean build and quick regression are recorded in the heartbeat commit/report.

## Remaining limits and next question

The model reconstructs a dense W-buffer. A physically distinct architecture could feed compressed/sparse weights directly into sparse compute and avoid writing zeros, but that requires a shared compression+sparsity+MMA contract, FIFO/backpressure behavior, and compiler-visible encoding. Do not infer that behavior from this dense reconstruction model. It is a high-value blocked follow-up, not an implemented speedup.
