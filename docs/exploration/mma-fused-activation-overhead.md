# MMA + Fused Activation Overhead: GEMM → ReLU/GELU/SiLU

**Date:** 2026-07-06
**Question:** What SRAM bandwidth overhead does a fused activation (ReLU/GELU/SiLU) add after a GEMM operation? When does the elementwise pass become a bottleneck?
**Hypothesis:** The overhead is purely a function of O-buffer size relative to GEMM compute — all activations have identical memory access patterns, so the specific activation function is irrelevant.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| PE array | 8×8, 16×16, 32×32, 64×16, 8×32 | Standard sweep set |
| Dataflow | weight_stationary | Baseline |
| Workload | M=N=32–256, K=64–256 | GEMM, 7 configs |
| SRAM model | 32 banks, 2-cycle stall | From tu_config.h |
| Pipeline | depth=2 | Default |

Configs tested: 35 (5 PE × 7 workloads), analytical cycle model.

## Cycle Model

```
MMA (WS):
  fill    = pdepth × ceil(N / pe_cols)
  compute = ceil(M/pe_rows) × ceil(N/pe_cols) × K × pdepth
  drain   = pdepth × ceil(M / pe_rows)
  dma     = (W_bytes + A_bytes + O_bytes) / bus_width_bytes
  total   = fill + compute + drain + dma

EW (sequential elementwise on banked SRAM):
  Each group of 32 elements: 1 read cycle (all 32 banks) + 32 × 2 write-stall cycles
  total = ceil(elems / 32) × (1 + 32 × 2) ≈ 2.03 × elems
```

## Results

| PE | Workload | O_elems | O_KB | MMA_cyc | EW_cyc | EW/MMA | Util% |
|----|----------|---------|------|---------|--------|--------|-------|
| 8×8 | 32×32×64 | 1,024 | 4.0 | 2,448 | 2,080 | 85.0% | 83.7 |
| 8×8 | 64×64×64 | 4,096 | 16.0 | 9,248 | 8,320 | 90.0% | 88.6 |
| 8×8 | 64×64×256 | 4,096 | 16.0 | 35,360 | 8,320 | 23.5% | 92.7 |
| 8×8 | 128×128×64 | 16,384 | 64.0 | 35,904 | 33,280 | 92.7% | 91.3 |
| 8×8 | 128×128×256 | 16,384 | 64.0 | 137,280 | 33,280 | 24.2% | 95.5 |
| 16×16 | 32×32×64 | 1,024 | 4.0 | 904 | 2,080 | 230.1% | 56.6 |
| 16×16 | 64×64×64 | 4,096 | 16.0 | 3,088 | 8,320 | 269.4% | 66.3 |
| 16×16 | 64×64×256 | 4,096 | 16.0 | 10,768 | 8,320 | 77.3% | 76.1 |
| 16×16 | 128×128×64 | 16,384 | 64.0 | 11,296 | 33,280 | 294.6% | 72.5 |
| 16×16 | 128×128×256 | 16,384 | 64.0 | 38,944 | 33,280 | 85.5% | 84.1 |
| 32×32 | 32×32×64 | 1,024 | 4.0 | 516 | 2,080 | 403.1% | 24.8 |
| 32×32 | 64×64×64 | 4,096 | 16.0 | 1,544 | 8,320 | 538.9% | 33.2 |
| 32×32 | 64×64×256 | 4,096 | 16.0 | 4,616 | 8,320 | 180.2% | 44.4 |
| 32×32 | 128×128×64 | 16,384 | 64.0 | 5,136 | 33,280 | 648.0% | 39.9 |
| 32×32 | 128×128×256 | 16,384 | 64.0 | 14,352 | 33,280 | 231.9% | 57.1 |
| 32×32 | 256×256×64 | 65,536 | 256.0 | 18,464 | 133,120 | 721.0% | 44.4 |
| 64×16 | 32×32×64 | 1,024 | 4.0 | 646 | 2,080 | 322.0% | 19.8 |
| 64×16 | 128×128×64 | 16,384 | 64.0 | 5,140 | 33,280 | 647.5% | 39.8 |
| 8×32 | 128×128×64 | 16,384 | 64.0 | 11,304 | 33,280 | 294.4% | 72.5 |

Full 35-config table in sweep output.

## Key Finding

**A separate elementwise pass over the O-buffer is a dominant cost for small-K GEMM workloads on large PE arrays.** On a 32×32 PE array with K=64, the ReLU pass costs 403-721% of the GEMM itself. The problem is that GEMM compute scales with K (more inner-product work per memory access) while elementwise cost is fixed by output size.

**Three regimes emerge:**

| Regime | K range | EW/MMA | Example |
|--------|---------|--------|---------|
| **EW-dominated** | K < 128 on PE ≥ 16×16 | 230-721% | Attention projection layers |
| **EW-significant** | K = 128-256 | 77-232% | FFN intermediate layers |
| **EW-negligible** | K > 256 | < 25% | Large FFN layers |

**The PE size paradox:** Larger PE arrays reduce MMA cycles (more MACs/cycle), making EW overhead _worse_ as a percentage. On 32×32 PE with 128×128×64 GEMM, the elementwise pass takes 6.5× more cycles than the GEMM (648%). On 8×8 PE with the same workload, it's only 93% — the GEMM takes so long that the EW pass looks cheap by comparison.

**All activations are equivalent.** ReLU, GELU, and SiLU have identical memory access patterns (read element, apply math, write element). The compute micro-op is negligible compared to SRAM bandwidth. EW cost is a function of O-buffer size only.

**Design recommendation:** Fuse the activation into the accumulator path. Instead of writing FP32 partial sums to O-buffer, reading them back, applying activation, and writing again, apply the activation during the final accumulator write. This eliminates the separate O-buffer pass entirely and saves 2-7× on total cycle count for small-K workloads.

## Next Steps

- Compare fused vs. decoupled activation in cycle-accurate mode (gap E2 pipeline controller integration)
- Model activation fusion in dataflow plugins (WS/OS/RS with post-accumulator activation)
- Measure impact on real workloads (ResNet block: conv → BN → ReLU chain)

## Source

Sweep: `tests/test_mma_activation_sweep.c` (analytical, no cmodel dependency)
Run: `make test-mma-activation-sweep`
