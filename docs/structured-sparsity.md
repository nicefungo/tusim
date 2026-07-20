# 2:4 Structured Sparsity

> **Status:** Implemented and build-integrated; explicit functional API plus runtime-configured analytical cycle estimator
> **Hardened:** 2026-07-20

## Scope

The module represents two realistic architecture choices:

1. **Dense execution (runtime default):** regular compute with no pruning or sparse-control hardware.
2. **2:4 structured sparse execution:** exactly two retained weights in each contiguous group of four along K.

The sparse path is not described as universally faster. It halves useful MACs and reduces FP16 weight bytes by 37.5%, but end-to-end behavior depends on decoder provisioning, N-dimension reuse, DMA traffic, and omitted physical costs. Measured trade-offs are in [`exploration/structured-2of4-sweep.md`](exploration/structured-2of4-sweep.md).

## Packed format

Each group is serialized explicitly, without relying on C struct layout:

```text
[value 0: elem_size bytes][value 1: elem_size bytes][mask: 1 byte]
```

The low four mask bits identify retained positions. Legal masks have exactly two set bits. Storage per group is:

| Type | Dense | Packed | Weight-byte reduction |
|---|---:|---:|---:|
| FP32 | 16 B | 9 B | 43.75% |
| FP16/BF16 | 8 B | 5 B | 37.5% |
| INT8 | 4 B | 3 B | 25% |

INT4 is not supported by this byte-oriented API. The model rejects malformed masks and non-multiple-of-four element/K counts; no implicit tail padding is defined.

## Runtime configuration

Canonical JSON/YAML fields:

```json
"sparsity": {
  "enabled": false,
  "structured_2of4": false,
  "unstructured": false,
  "metadata_format": "bitmask",
  "decoder_groups_per_cycle": 1
}
```

- Dense is the backward-compatible default.
- 2:4 requires both `enabled=true` and `structured_2of4=true`.
- `decoder_groups_per_cycle` must be positive and controls the analytical decoder-backpressure model.
- Unstructured sparsity is rejected because there is no executable implementation.

Functional sparse MMA remains an explicit API; the existing dense ISA/MMA path does not automatically reinterpret W-buffer bytes as packed data.

## Public API

`tu_cmodel/sparsity/structured_2of4.{h,c}` provides:

- magnitude pruning with deterministic tie behavior;
- mask validation and position decoding;
- packed encode/decode for byte-addressable element sizes;
- flat and tiled sparse GEMM with FP32 accumulation;
- comparison against a dense reference computed from the **pruned** weights;
- `tu_sparsity_2of4_estimate_cycles()`, which reads canonical runtime architecture parameters and reports dense/sparse MACs, bytes, DMA, compute, decode, total, and selected-mode cycles.

The cycle estimator assumes FP16 W/A and FP32 O. DMA is serialized with compute; metadata decode overlaps sparse compute. It is an analytical upper bound, not a cycle-accurate sparse pipeline.

## Numerical semantics

Magnitude pruning modifies model weights. Functional correctness means that packed sparse execution matches dense GEMM using the resulting pruned weights. It does **not** establish accuracy relative to the original dense model. Task-level accuracy, fine-tuning recovery, and alternative pruning policies are outside the current cmodel and remain unquantified.

## Hardware trade-offs

- **Dense:** preserves accuracy and regular scheduling; avoids decoder/metadata/steering area, power, control, compiler contracts, and associated verification.
- **2:4:** can reduce latency and weight traffic for compliant models, especially with sufficient N reuse and decoder throughput; sacrifices weight freedom and adds decode/steering/control.
- **Higher decoder throughput:** helps narrow-N/low-reuse GEMMs but should cost more replicated decode logic, routing, and power. Those physical costs are not quantified.

No mode is universally recommended. A physical TU may hard-wire dense or 2:4; the pre-spec cmodel preserves both for comparison.

## Verification

```sh
make test-sparsity          # 27 functional/config/model tests
make test-sparsity-sweep    # full dense/2:4 workload x decoder matrix
make test-config            # canonical JSON config regression
```

Coverage includes six legal masks, illegal masks, deterministic magnitude pruning, FP32/FP16/INT8 round trips, flat/tiled GEMM differential checks, dense default, JSON parse, inconsistent/unsupported config rejection, decoder bottlenecks, and invalid K/group sizes.

## Known limitations

- No automatic command-queue/ISA sparse opcode dispatch.
- No decoder area, power, pipeline, or SRAM-port model.
- No metadata burst alignment or cache/SRAM contention.
- No sparse-lane imbalance beyond the fixed 2:4 MAC count.
- No task-level model-accuracy measurement.
- No K-tail encoding; K must be divisible by four.

These limitations must accompany performance numbers; the old unconditional “exactly 2x throughput” claim is retired.
