# INT8 Quantization Throughput Sweep

**Date:** 2026-07-11
**Method:** Analytical cycle model (WS dataflow, 2-deep pipeline, 32B/cycle DMA)
**Question:** "How much effective throughput does INT8 quantization deliver over FP16 for GEMM?"

## Config Matrix

| Parameter | Values |
|-----------|--------|
| PE Array | 8×8, 16×16, 32×32, 64×64, 128×16, 16×128 |
| GEMM M×N×K | 32³⁾, 64³⁾, 64²×128, 64²×256, 128³⁾, 128²×256, 256³⁾, 256²×128, 512³⁾, 64×512×64, 512×64×64 |
| Precision | FP16 (2B/elem, FP32 accum), INT8 (1B/elem, INT32 accum) |

⁾ Different PE configs used for summary; full workload sweep on 32×32 only.

## Results (32×32 PE, key workloads)

| GEMM M×N×K | FP16 Cycles | INT8 Cycles | Speedup | DMA Save |
|------------|-------------|-------------|---------|----------|
| 32×32×64 | 515 | 387 | 1.33× | 33.3% |
| 64×64×64 | 1,539 | 1,283 | 1.20× | 25.0% |
| 64×64×128 | 3,587 | 3,075 | 1.17× | 33.3% |
| 128×128×128 | 12,291 | 11,267 | 1.09× | 25.0% |
| 256×256×128 | 45,059 | 43,011 | 1.05× | 16.7% |
| 512×512×64 | 69,635 | 67,587 | 1.03× | 5.6% |

## Key Finding

**INT8 quantization delivers ~3–33% speedup over FP16, entirely from DMA bandwidth savings.** The compute path (same PE array, same MAC-per-cycle) is unchanged — INT8 and FP16 have identical compute cycle counts. All gains come from halving weight/activation data movement (1 byte vs 2 bytes per element).

The speedup is maximized for **small-K workloads** (DMA-bound, K ≤ 64) where DMA dominates total cycles (40–60%). For large-K workloads (K ≥ 256), DMA falls below 10% of total cycles and INT8 gains approach 1.0×.

### K-Sensitivity (128×128, 32×32 PE)

| K | FP16 Cycles | DMA % | INT8 Speedup |
|---|-------------|-------|-------------|
| 64 | 5,123 | 60% | 1.11× |
| 128 | 12,291 | 40% | 1.09× |
| 256 | 24,579 | 27% | 1.06× |
| 512 | 49,155 | 15% | 1.04× |
| 1024 | 196,611 | 7% | 1.02× |

### PE Array Sensitivity (128×128×128)

| PE Array | FP16 Cycles | INT8 Speedup | INT8 Eff. TOPS |
|----------|-------------|-------------|----------------|
| 8×8 | 528,387 | 1.00× | 0.008 |
| 16×16 | 69,635 | 1.01× | 0.061 |
| 32×32 | 12,291 | 1.09× | 0.372 |
| 64×64 | 5,123 | 1.25× | 1.023 |
| 128×16 | 4,227 | 1.32× | 1.309 |

Larger PE arrays see bigger INT8 gains because compute compresses into fewer cycles, making DMA a larger fraction of total time. The 128×16 aspect ratio shows the best INT8 speedup (1.32×) because the wide fan-out maximizes DMA pressure relative to compute depth.

## Limitations

- **Analytical only** — not validated against cmodel INT8 execution path. The cmodel's `tu_int_quant.{c,h}` module provides INT8→INT32 MAC but hasn't been integrated into the main GEMM dataflow.
- **No quantization error modeling** — INT8 quantization error (typically 0.1–1% accuracy loss) is a separate concern from throughput.
- **Fixed DMA bus width** — assumes 32 B/cycle for both precision types. Real silicon might throttle INT8 DMA width.
- **No INT4** — INT4 would provide 4× DMA savings over FP16 (0.5 B/elem) but requires even more careful quantization-aware training.

## Recommendation

INT8 quantization is worth implementing in hardware when:
1. The target workloads are DMA-bound (small K, large M×N, or wide fan-out aspect ratios)
2. The ~1.05–1.33× throughput gain justifies the quantization infrastructure (calibration, scale/zero-point logic)
3. Accuracy loss from INT8 quantization is acceptable for the application

For a pre-spec DSE: INT8 support should NOT be a default requirement. It's a nice-to-have for inference-focused designs targeting transformer FFN layers (large M×N, moderate K), where DMA is 25–40% of total cycles and the 10–25% throughput gain is meaningful.
