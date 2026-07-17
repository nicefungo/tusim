# RLE Weight Compression: Placement Sensitivity and Runtime Control

**Date:** 2026-07-17
**Status:** Implemented and verified
**Type:** Cmodel-linked sweep (`make test-weight-compression-sweep`)

## Design question

When is a simple FP16 run-length codec a physically plausible weight-stream option, and when must hardware leave weights uncompressed? The hypothesis was that **placement of repeated values, not scalar sparsity alone**, determines whether RLE saves memory traffic.

## Alternatives retained

- **NONE (default):** raw FP16 weights. A real TU chooses this for dense weights, randomly distributed pruning, minimum decoder area/power, deterministic fixed-rate DMA, and simplest verification/compiler behavior.
- **RLE, exact (`rle_epsilon=0`):** lossless value/count runs. A TU may choose this for clustered/block-pruned or repeated quantized weights where DRAM energy and bandwidth dominate. It costs codec logic, variable-rate buffering, metadata, and decode scheduling.
- **RLE, epsilon > 0:** lossy near-value merging. This can improve run formation after quantization, but changes numerical semantics and adds compare/control cost. It remains configurable but is not the default; accuracy must be validated per model.

The canonical JSON config now exposes:

```json
"weight_compression": {
  "enabled": false,
  "type": "none",
  "rle_epsilon": 0.0
}
```

`tu_compress_config_from_tu_config()` maps this block to the codec API. Validation rejects unknown codecs and negative epsilon. Existing default behavior remains raw/uncompressed.

## Workload and model

- 4,096 FP16 weights (8,192 raw bytes)
- 256-bit / 32-byte DMA bus
- Exact RLE
- Encoded format: 8-byte tensor header plus a portable 6-byte `{FP16 value, uint32 count}` run
- DMA cycles are `ceil(bytes / 32)`; descriptor, decoder, and contention costs are not modeled

Random-zero rows use deterministic LCG placement. Clustered-zero rows put zeros in one contiguous region and organize remaining values in runs of 16. These are deliberately different physical pruning/layout regimes at similar scalar sparsity.

## Measured results

| Workload | Runs | Raw B | RLE B | Raw/RLE | Raw DMA cyc | RLE DMA cyc |
|---|---:|---:|---:|---:|---:|---:|
| Alternating | 4,096 | 8,192 | 24,584 | 0.333× | 256 | 769 |
| Random zero 50% | 3,059 | 8,192 | 18,362 | 0.446× | 256 | 574 |
| Clustered zero 50% | 129 | 8,192 | 782 | 10.476× | 256 | 25 |
| Random zero 70% | 2,112 | 8,192 | 12,680 | 0.646× | 256 | 397 |
| Clustered zero 70% | 78 | 8,192 | 476 | 17.210× | 256 | 15 |
| Random zero 90% | 827 | 8,192 | 4,970 | 1.648× | 256 | 156 |
| Clustered zero 90% | 27 | 8,192 | 170 | 48.188× | 256 | 6 |
| All zero | 1 | 8,192 | 14 | 585.143× | 256 | 1 |

## Gain versus sacrifice

| Dimension | NONE | Exact RLE | Epsilon RLE |
|---|---|---|---|
| Throughput / latency | Fixed 256 payload cycles here | 6–25 cycles for clustered 50–90%; **397–769 cycles** for random/alternating 50–70% | Potentially fewer runs; not measured in this sweep |
| Area/resources | No codec or run buffer | Decoder, count FSM, variable-rate FIFO | Exact-RLE hardware plus FP compare/tolerance path |
| Power/energy | Highest DRAM traffic when runs are long | Expected lower DRAM energy for clustered runs; decoder energy unquantified | Similar, plus compare activity; unquantified |
| SRAM/DRAM traffic | 8,192 B in every case | 170–782 B clustered, but 12,680–24,584 B in poor cases | Data-dependent; unquantified |
| Numerical accuracy | Exact original values | Lossless | Lossy by construction; model-level impact unquantified |
| Control complexity | Lowest, fixed-rate | Variable-rate backpressure and metadata handling | Highest; tolerance semantics must match software |
| Verification burden | Lowest | Corrupt stream, count overflow, and round-trip tests | Also bounded-error and model-accuracy tests |
| Compiler/runtime | No transform | Must select/encode only suitable layouts | Must authorize a numerical policy, not merely compression |

**Conclusion:** RLE is not a generic “sparse weights” optimization. At 70% randomly placed zeros it expands traffic by 55%, while clustered 70% zeros reduce traffic by 17.2×. The pre-spec cmodel therefore retains NONE and RLE rather than selecting a universal winner. A future compiler policy should inspect actual encoded size and select per tensor; an on-chip adaptive fallback would require an explicit framed format and is left in the backlog rather than inferred ambiguously at decode time.

## Correctness fix discovered

The documented stream is 6 bytes per run, but the implementation previously serialized `sizeof(tu_rle_run_t)` (8 bytes on this ABI due to padding). The codec now serializes the two fields explicitly into a stable 6-byte wire representation. This restores the specified format and makes streams independent of host ABI padding.

## Verification

```sh
make test-compress                  # 13/13 pass
make test-weight-compression-sweep # reproduces table
make clean && make
make test-quick
```

## Limitations

The cmodel does not yet quantify codec area, decoder energy, decode throughput, FIFO depth, DRAM burst fragmentation, or compiler transform cost. DMA-cycle values are payload-only lower bounds. Epsilon mode is functionally tested for run formation but this sweep does not claim model-level accuracy.
