# Rounding Mode Accuracy Sweep: RNE vs RTZ vs Stochastic for GEMM

**Date:** 2026-06-22
**Question:** How does the FP16 rounding mode (RNE, RTZ, Stochastic) affect numerical accuracy for a GEMM 128×128×256 workload? Is the accuracy penalty from non-standard rounding modes significant enough to influence hardware design trade-offs?

**Hypothesis:** RNE (round-to-nearest-even, IEEE 754 default) should produce the lowest error. RTZ (truncation) introduces a systematic negative bias (always rounds toward zero), which should compound across K=256 inner-dimension accumulations. Stochastic rounding should be unbiased in expectation but exhibit higher variance — making it interesting for training (gradient accumulation) but potentially worse for inference accuracy on single runs.

## Config Matrix

| Parameter | Values |
|-----------|--------|
| PE array | 16×16 (256 MACs) |
| Dataflow | Weight-stationary |
| Workload | GEMM M=128, N=128, K=256 (8.4 MFLOPs) |
| Precision | FP16 (W, A, O-accumulator FP32→FP16 store) |
| W/A data | FP64 random uniform [-1, 1], seed=42/99 |
| Golden reference | FP64 GEMM |
| Rounding modes | RNE, RTZ, Stochastic (2 seeds) |

## Results

| Rounding Mode | Max Abs Error | Mean Abs Error | Max Relative Error | Violations >0.5 |
|--------------|---------------|----------------|-------------------|-----------------|
| **RNE** (default) | 5.79×10⁻³ | 1.10×10⁻³ | 8.80 | 0 |
| **RTZ** (truncate) | 1.48×10⁻² | 2.91×10⁻³ | 9.80 | 0 |
| **Stochastic** | 8.08×10⁻³ | 1.58×10⁻³ | 1.42 | 0 |
| **Stochastic** (alt seed) | 9.22×10⁻³ | 1.57×10⁻³ | 36.1 | 0 |

Peak relative errors occur at elements with tiny golden values (near zero), where a small absolute FP16 quantization error produces a large relative ratio. All absolute errors are well within the FP16 MMA tolerance formula `tol = 0.01 + K × 0.0005 = 0.138`.

## Key Finding

**RTZ is 2.6× worse than RNE on both max and mean error.** The systematic downward bias (always rounding toward zero) compounds across the K=256 accumulation dimension. For inference workloads with many accumulation steps (large K or deep networks), RTZ accumulates bias that RNE avoids through its statistical symmetry.

**Stochastic rounding is 1.4× worse than RNE** but shows variance across seeds — it's unbiased in expectation, meaning repeated runs would converge to RNE-level accuracy. This makes stochastic rounding viable for training (where gradient accumulation across batches averages out the noise) but a poor choice for single-pass inference.

**No mode produced catastrophic error** (>0.5 absolute) even with K=256 and random data in [-1,1]. The IEEE 754 FP16 format with 10-bit mantissa has enough precision for this workload scale regardless of rounding mode.

## Design Implication

For a **pre-spec accelerator targeting inference**: use RNE. It's the standard, costs one adder in hardware, and produces the best accuracy. RTZ is not worth the small hardware savings (~half the rounding logic) given the 2.6× accuracy penalty.

For a **training accelerator**: stochastic rounding becomes interesting. The hardware cost of a PRNG per PE is non-trivial, but the unbiased property prevents gradient quantization bias that RNE can introduce in low-precision training. This is a trade-off to evaluate when the training vs. inference decision is made.

## Test Harness

`tests/test_rounding_sweep.c` — standalone sweep program using the TU cmodel's `tu_set_rounding_mode()` API. Compares FP16 GEMM output against FP64 golden reference. Run with `make test-rounding-sweep`.
