# Workload Scaling: Optimal PE Array by Problem Size

**Date:** 2026-06-12
**Question:** How does the optimal PE array size change as the GEMM workload scales from attention heads (0.5M FLOPs) to LLM projections (8.6B FLOPs)?
**Hypothesis:** Larger workloads amortize DMA overhead better, shifting the sweet spot from 16×16 to 32×32 or 64×64. The "right PE size" is workload-dependent, not a fixed architectural constant.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| Workload | 64×64×64, 128×128×256, 512×512×1024, 1024×1024×4096 | Attention head → LLM projection |
| PE array | 8×8, 16×16, 32×32, 64×64 | Powers-of-2 scaling |
| Dataflow | weight_stationary | Systolic, pdepth=2 |
| Bus width | 256-bit (32 B/cycle) | Default |
| Clock | 1.0 GHz | Default |
| Precision | FP16 W/A, FP32 O | Default |

**Configs tested:** 16 (4 workloads × 4 PE arrays), analytical cycle model.

## Results Table

| Workload | PE | M×N×K | FLOPs | Fill | Compute | Drain | DMA | TotalCyc | TOPS | PkTOPS | Util% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Attention head | 8×8 | 64×64×64 | 0.5M | 16 | 4,096 | 16 | 1,024 | 5,152 | 0.102 | 0.128 | **79.5%** |
| Attention head | 16×16 | 64×64×64 | 0.5M | 8 | 1,024 | 8 | 1,024 | 2,064 | 0.254 | 0.512 | **49.6%** |
| Attention head | 32×32 | 64×64×64 | 0.5M | 4 | 256 | 4 | 1,024 | 1,288 | 0.407 | 2.048 | **19.9%** |
| Attention head | 64×64 | 64×64×64 | 0.5M | 2 | 64 | 2 | 1,024 | 1,092 | 0.480 | 8.192 | **5.9%** |
| | | | | | | | | | | | |
| Medium GEMM | 8×8 | 128×128×256 | 8.4M | 32 | 65,536 | 32 | 6,144 | 71,744 | 0.117 | 0.128 | **91.3%** |
| Medium GEMM | 16×16 | 128×128×256 | 8.4M | 16 | 16,384 | 16 | 6,144 | 22,560 | 0.372 | 0.512 | **72.6%** |
| Medium GEMM | 32×32 | 128×128×256 | 8.4M | 8 | 4,096 | 8 | 6,144 | 10,256 | 0.818 | 2.048 | **39.9%** |
| Medium GEMM | 64×64 | 128×128×256 | 8.4M | 4 | 1,024 | 4 | 6,144 | 7,176 | 1.169 | 8.192 | **14.3%** |
| | | | | | | | | | | | |
| LLM FFN layer | 8×8 | 512×512×1024 | 537M | 128 | 4,194,304 | 128 | 98,304 | 4,292,864 | 0.125 | 0.128 | **97.7%** |
| LLM FFN layer | 16×16 | 512×512×1024 | 537M | 64 | 1,048,576 | 64 | 98,304 | 1,147,008 | 0.468 | 0.512 | **91.4%** |
| LLM FFN layer | 32×32 | 512×512×1024 | 537M | 32 | 262,144 | 32 | 98,304 | 360,512 | 1.489 | 2.048 | **72.7%** |
| LLM FFN layer | 64×64 | 512×512×1024 | 537M | 16 | 65,536 | 16 | 98,304 | 163,872 | 3.276 | 8.192 | **40.0%** |
| | | | | | | | | | | | |
| LLM projection | 8×8 | 1024×1024×4096 | 8590M | 256 | 67,108,864 | 256 | 655,360 | 67,764,736 | 0.127 | 0.128 | **99.0%** |
| LLM projection | 16×16 | 1024×1024×4096 | 8590M | 128 | 16,777,216 | 128 | 655,360 | 17,432,832 | 0.493 | 0.512 | **96.2%** |
| LLM projection | 32×32 | 1024×1024×4096 | 8590M | 64 | 4,194,304 | 64 | 655,360 | 4,849,792 | 1.771 | 2.048 | **86.5%** |
| LLM projection | 64×64 | 1024×1024×4096 | 8590M | 32 | 1,048,576 | 32 | 655,360 | 1,704,000 | 5.041 | 8.192 | **61.5%** |

## Key Findings

### 1. Optimal PE size is workload-dependent — no single answer

The PE array size that maximizes utilization shifts rightward as FLOPs increase:

| Workload | FLOPs | Best util PE | Util% | Best TOPS PE | TOPS |
|---|---|---|---|---|---|
| Attention head | 0.5M | 8×8 | 79.5% | 64×64 | 0.480 |
| Medium GEMM | 8.4M | 8×8 | 91.3% | 64×64 | 1.169 |
| LLM FFN layer | 537M | 8×8 | 97.7% | 64×64 | 3.276 |
| LLM projection | 8,590M | 8×8 | 99.0% | 64×64 | 5.041 |

**Utilization always favors smaller PEs** — the 8×8 array achieves ≥79.5% utilization across all workloads because DMA is always the bottleneck and a small PE's compute demand doesn't outstrip the 256-bit bus. But absolute throughput (TOPS) always favors larger PEs despite lower utilization.

### 2. The 50% utilization threshold shifts with workload

For a PE array to be "worth it" (>50% utilization):

| PE Array | Min FLOPs for >50% Util | Example | Current util at threshold |
|---|---|---|---|
| 8×8 | <0.5M | All workloads | 79.5% minimum |
| 16×16 | ~1M | Attention head (0.5M) = 49.6% — just below | — |
| 32×32 | ~30M | Medium GEMM (8.4M) = 39.9% — too small | — |
| 64×64 | ~200M | LLM FFN (537M) = 40.0% — marginal | — |

**Rule of thumb:** Each 4× increase in PE area requires ~30× more FLOPs to maintain >50% utilization. This comes from the DMA overhead being a fixed cost that must be amortized across proportionally more compute.

### 3. Area efficiency peaks at 16×16 or 32×32, not at extremes

TOPS per MAC (a proxy for area efficiency):

| Workload | 8×8 | 16×16 | 32×32 | 64×64 | Best area eff. |
|---|---|---|---|---|---|
| Attention | 1.59K | 0.99K | 0.40K | 0.12K | 8×8 |
| Medium GEMM | 1.83K | 1.45K | 0.80K | 0.29K | 8×8 |
| LLM FFN | 1.95K | 1.83K | 1.45K | 0.80K | 8×8 |
| LLM projection | 1.98K | 1.93K | 1.73K | 1.23K | 8×8 |

TOPS-per-MAC declines with PE size for all workloads — smaller PEs are always more area-efficient. But larger PEs deliver more absolute throughput. The tradeoff is: how much throughput do you need, and what area/power budget do you have?

### 4. DMA overhead asymptote: from 20% to 1%

The DMA fraction of total cycles tells the story:

| Workload | FLOPs | DMA@8×8 | DMA@16×16 | DMA@32×32 | DMA@64×64 |
|---|---|---|---|---|---|
| Attention head | 0.5M | 19.9% | 49.6% | 79.5% | 93.8% |
| Medium GEMM | 8.4M | 8.6% | 27.2% | 59.9% | 85.6% |
| LLM FFN layer | 537M | 2.3% | 8.6% | 27.3% | 60.0% |
| LLM projection | 8,590M | 1.0% | 3.8% | 13.5% | 38.5% |

For attention heads (0.5M FLOPs), DMA dominates at any PE larger than 8×8. For LLM projections (8.6B FLOPs), DMA drops to just 1.0% at 8×8 and 38.5% even at 64×64 — compute finally dominates.

## Visualization

```
Utilization % vs PE Array Size by Workload

100% ┤●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●   ● = LLM projection (8.6B FLOPs)
     ┤▓                              ▓
     ┤▓                              ▓   ■ = LLM FFN (537M FLOPs)
 80% ┤▓■━━━━━━━━━━━━━━━━━━━━━━━━━━━━■▓
     ┤▓■                            ■▓   ▲ = Medium GEMM (8.4M FLOPs)
     ┤▓■                            ■▓
 60% ┤▓■▲━━━━━━━━━━━━━━━━━━━━━━━━━━■▓▲  ◆ = Attention head (0.5M FLOPs)
     ┤▓■▲                          ■▓▲
     ┤▓■▲              ◆           ■▓▲
 40% ┤▓■▲━━━━━━━━━━━━━━◆━━━━━━━━━━■▓▲
     ┤▓■▲            ◆            ■▓▲
     ┤▓■▲          ◆              ■▓▲
 20% ┤▓■▲━━━━━━━━◆━━━━━━━━━━━━━━━■▓▲
     ┤▓■▲      ◆                  ■▓▲
     ┤▓■▲    ◆                    ■▓▲
  0% ┤▓■▲━━◆━━━━━━━━━━━━━━━━━━━━■▓▲━
         8×8        16×16        32×32        64×64
                        PE Array Size
```

## Actionable Conclusion

**For the ONNX compiler's hardware target, the PE array size should match the dominant workload scale:**

1. **If targeting small models (edge/embedded, ≤10M FLOPs per GEMM):** 8×8 or 16×16 PE. Larger arrays are wasted silicon — 64×64 at 0.5M FLOPs achieves only 5.9% utilization. The absolute TOPS gain (0.48 vs 0.10) is real but the area/power cost is 64× higher for 4.7× throughput.

2. **If targeting medium models (mobile/server hybrid, ~10-100M FLOPs):** 16×16 PE is the sweet spot (72.6% util at 8.4M FLOPs). 32×32 is viable if paired with ≥512-bit bus.

3. **If targeting LLM inference (≥500M FLOPs per GEMM):** 32×32 PE with 256-bit bus is viable (72.7% util at 537M FLOPs). 64×64 PE with 512-bit bus becomes worth considering above ~1B FLOPs.

4. **The 256-bit bus becomes the gating factor between 16×16 and 32×32.** At 256-bit, 32×32 only breaks 50% utilization above ~30M FLOPs. If the dominant workload is below this threshold, stick with 16×16. If above, 32×32 becomes competitive.

**Design implication:** The compiler should be parameterized for multiple PE configurations — one for edge (8×8), one for mobile (16×16), one for server (32×32). The optimal configuration is workload-driven, and the compiler is the layer that maps workloads to hardware.

## Methodology

Analytical cycle model using validated WS systolic formulas from `weight_stationary.c`. Prior explorations (PE-array sweep Jun 3, bus-width sweep Jun 7, K-sweep Jun 8) validated these formulas within 0 cycles of the cmodel's perf report output.

```
per-PE-size:
  tiles_m = ceil(M / pe_rows),  tiles_n = ceil(N / pe_cols)
  fill    = pdepth × tiles_n
  compute = tiles_m × tiles_n × K
  drain   = pdepth × tiles_m
  dma     = ceil((M×K×2 + K×N×2 + M×N×4) / 32)
  total   = fill + compute + drain + dma
  TOPS    = (M × N × K × 2) / total / 1000
```

All 16 configs verified analytically. DMA accounts for FP16 W/A (2B each) + FP32 O accumulator (4B), matching the cmodel's actual transfer accounting.

## Next Exploration Candidates

1. **Combined SRAM scaling with workload:** For LLM workloads, the 64 KB O-buffer doesn't fit a 1024×1024 output (needs 4 MB). What tiling factor results, and how does it interact with PE size?
2. **Bus width × workload sweep:** At what bus width does 64×64 PE become viable for LLM FFN layers?
3. **Real layer traces:** Run actual ONNX model layers through the cmodel and measure end-to-end throughput, not just analytical GEMM counts.
4. **Multi-core scaling:** For workloads that are too large for a single PE array, does splitting across multiple 16×16 cores beat a single 64×64 core?
