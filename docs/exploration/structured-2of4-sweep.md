# Dense vs 2:4 Structured-Sparsity Trade-off

**Date:** 2026-07-20
**Question:** When does a physically plausible 2:4 sparse datapath improve end-to-end GEMM latency, and how much metadata-decoder throughput is required?

## Alternatives retained

- **Dense (default):** no pruning, metadata decoder, sparse lane steering, or compiler packing contract. A hardware team may choose it for numerical fidelity, regular timing, lower verification burden, and smaller control logic.
- **Structured 2:4:** two values retained in every contiguous K-group of four, encoded as two FP16 values plus a one-byte position mask. A team may choose it to reduce weight traffic and useful MAC count when model training can tolerate/enforce the constraint.

Both remain runtime selectable through `sparsity.enabled` and `sparsity.structured_2of4`. `sparsity.decoder_groups_per_cycle` represents decoder provisioning; the dense-compatible default remains disabled.

## Executable model

Command:

```sh
make test-sparsity-sweep
```

Configuration: 16x16 PE array, 256-bit DMA, FP16 weights/activations, FP32 outputs. DMA is serialized with the compute phase; metadata decode overlaps sparse compute. Sparse compute has half the useful MACs. Packed FP16 weights use 5 bytes per four values versus 8 bytes dense (37.5% reduction, not 2x compression).

| Workload MxNxK | Decoder groups/cycle | Dense cycles | 2:4 cycles | Modeled speedup |
|---|---:|---:|---:|---:|
| 64x64x64 | 1 | 2,051 | 1,952 | 1.051x |
| 64x64x64 | 4 / 16 | 2,051 | 1,443 | 1.421x |
| 128x128x128 | 1 / 4 / 16 | 12,291 | 7,811 | 1.574x |
| 512x16x512 | 1 | 34,307 | 77,312 | **0.444x** |
| 512x16x512 | 4 | 34,307 | 28,160 | 1.218x |
| 512x16x512 | 16 | 34,307 | 19,971 | 1.718x |
| 64x512x512 | 1 / 4 / 16 | 88,067 | 54,531 | 1.615x |
| 512x512x512 | 1 / 4 / 16 | 589,827 | 321,539 | 1.834x |

## Findings and costs

- **Throughput/latency:** 2:4 is not universally faster. A one-group/cycle decoder makes the narrow-N case 2.25x slower than dense because each decoded weight group feeds too little N reuse. At four groups/cycle it crosses above dense; at sixteen it reaches 1.718x. Square and wide-N cases amortize decode and achieve 1.42-1.83x, still below the naive 2x claim because activation/output DMA and pipeline costs remain.
- **Area/resources:** dense avoids decoder, metadata buffers, selectors, and sparse issue/control. More decoder groups/cycle should increase decoder replication and routing. Area is **unquantified**.
- **Power/energy:** sparse mode should reduce weight transfers and active multiplies, but decoder and steering consume energy. Net energy is **unquantified**; cycle speedup must not be used as an energy proxy.
- **SRAM/DRAM traffic:** FP16 weight payload falls exactly 37.5%. Activation and output traffic are unchanged. Metadata alignment/burst waste is not modeled.
- **Numerical accuracy:** post-training magnitude pruning changes weights and can reduce model quality; the cmodel verifies sparse execution against the *pruned dense tensor*, not against the original model. Task-level accuracy and retraining recovery are **unquantified**.
- **Control complexity:** 2:4 adds mask validation, decode, lane steering, and malformed-stream behavior. Increasing decode rate increases parallel control/routing pressure.
- **Verification burden:** six legal masks, malformed masks, all supported element widths, K-group alignment, tile boundaries, and decoder backpressure all require coverage. This heartbeat added safe rejection for malformed masks and non-multiple-of-four dimensions.
- **Compiler/runtime:** software must prune or receive compliant weights, pack masks in K-major groups, select sparse mode, provision a compatible decoder rate, and reject unsupported unstructured mode. ISA-level automatic dispatch is not yet modeled; functional sparse MMA remains an explicit API.

## Fidelity and limitations

This is an analytical architecture model linked into the cmodel, not RTL timing. It assumes ideal sparse-lane balance, one-byte metadata per group, compute/decode overlap, and no metadata fetch alignment, SRAM bank contention, command setup, decoder pipeline fill, or compiler packing time. Results are upper bounds outside the explicitly modeled decoder bottleneck. The estimator intentionally rejects `K % 4 != 0` rather than inventing tail semantics.

## Implementation and verification

- Runtime parse/validation: `tu_cmodel/infra/config.{c,h}`, `config/tu_config.{json,yaml}`
- Functional module and cycle estimator: `tu_cmodel/sparsity/structured_2of4.{c,h}`
- Correctness/config tests: `tests/test_sparsity.c` (27 tests)
- Sweep: `tests/test_sparsity_sweep.c`
- Build integration: `make test-sparsity test-sparsity-sweep test-config`
