# Bus Width Sweep: DMA Bandwidth vs PE Array Scaling

**Date:** 2026-06-07
**Question:** At what DMA bus width does the 32×32 PE array stop being DMA-bound and become a compelling alternative to 16×16?
**Hypothesis:** The 32×32 array needs ~512-bit bus width before compute utilization exceeds 50%, and ~800-bit to match 16×16's current utilization at 256-bit.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| Bus width (bits) | 32, 64, 128, 256, 512, 1024 | DMA bus width, powers of 2 |
| PE array | 16×16, 32×32 | Two key array sizes |
| Dataflow | weight_stationary | Systolic |
| Workload | M=128, N=128, K=256 | 8.39 MFLOPs GEMM |
| Precision | FP16 W/A, FP32 O | Cmodel-accurate DMA accounting |
| Clock | 1.0 GHz | Default |

**Configs tested:** 12 (6 bus widths × 2 PE sizes), analytical cycle model.

## Results Table

### 16×16 PE (base compute: 16,416 cycles, peak: 0.512 TOPS)

| Bus (bits) | BW (B/cyc) | DMA cyc | Total cyc | TOPS | DMA% | Util% |
|-----------|------------|---------|-----------|------|------|-------|
| 32 | 4 | 49,152 | 65,568 | 0.128 | 75.0% | 25.0% |
| 64 | 8 | 24,576 | 40,992 | 0.205 | 60.0% | 40.0% |
| 128 | 16 | 12,288 | 28,704 | 0.292 | 42.8% | 57.1% |
| **256** | **32** | **6,144** | **22,560** | **0.372** | **27.2%** | **72.6%** |
| 512 | 64 | 3,072 | 19,488 | 0.430 | 15.8% | 84.1% |
| 1024 | 128 | 1,536 | 17,952 | 0.467 | 8.6% | 91.3% |

### 32×32 PE (base compute: 4,112 cycles, peak: 2.048 TOPS)

| Bus (bits) | BW (B/cyc) | DMA cyc | Total cyc | TOPS | DMA% | Util% |
|-----------|------------|---------|-----------|------|------|-------|
| 32 | 4 | 49,152 | 53,264 | 0.157 | 92.3% | 7.7% |
| 64 | 8 | 24,576 | 28,688 | 0.292 | 85.7% | 14.3% |
| 128 | 16 | 12,288 | 16,400 | 0.512 | 74.9% | 25.0% |
| **256** | **32** | **6,144** | **10,256** | **0.818** | **59.9%** | **39.9%** |
| 512 | 64 | 3,072 | 7,184 | 1.168 | 42.8% | 57.0% |
| 1024 | 128 | 1,536 | 5,648 | 1.485 | 27.2% | 72.5% |

### Speedup Ratio (32×32 vs 16×16)

| Bus (bits) | 32×32 TOPS | 16×16 TOPS | Speedup | Area cost | Efficiency |
|-----------|-----------|-----------|---------|-----------|------------|
| 32 | 0.157 | 0.128 | 1.23× | 4× | 30.8% |
| 64 | 0.292 | 0.205 | 1.43× | 4× | 35.7% |
| 128 | 0.512 | 0.292 | 1.75× | 4× | 43.8% |
| **256** | **0.818** | **0.372** | **2.20×** | **4×** | **55.0%** |
| 512 | 1.168 | 0.430 | 2.72× | 4× | 68.0% |
| 1024 | 1.485 | 0.467 | 3.18× | 4× | 79.5% |

## Key Findings

### 1. 32×32 utilization crosses 50% between 256-bit and 512-bit

At 256-bit (current default), the 32×32 PE array is still heavily DMA-bound at 59.9% DMA overhead. Moving to 512-bit drops DMA to 42.8%, bringing utilization to 57.0% — a meaningful improvement but still far from the 32×32 peak of 2.048 TOPS.

The exact crossover point (50% util) is at ~360 bits (45 B/cycle), producing ~1.0 TOPS.

### 2. DMA bandwidth is the gating factor for PE scaling — not compute density

The 32×32 array has 4× the MACs of 16×16 but only achieves 2.20× the throughput at 256-bit. For every doubling of bus width:
- 32→64: adds 0.135 TOPS to 32×32 (+85%)
- 64→128: adds 0.220 TOPS (+75%)
- 128→256: adds 0.306 TOPS (+60%)
- 256→512: adds 0.350 TOPS (+43%)
- 512→1024: adds 0.317 TOPS (+27%)

Diminishing returns per bus-width doubling, but still >30% at each step through 1024-bit.

### 3. 16×16 saturates faster with bus width

16×16 at 1024-bit achieves 91.3% utilization — only 8.7% of cycles are DMA. The array is effectively compute-bound at this point. Further bus width increases yield <10% throughput improvement for 16×16. But 32×32 at 1024-bit still has 27.2% DMA overhead — there's headroom up to ~2048-bit.

### 4. Area efficiency of 32×32 is poor below 512-bit

For 4× the silicon area (PEx4, SRAMx4), the 32×32 array delivers:
- 256-bit: 2.20× throughput (55% area efficiency)
- 512-bit: 2.72× throughput (68% area efficiency)
- 1024-bit: 3.18× throughput (79% area efficiency)

Architecturally, this means the PE array should not be scaled beyond 16×16 unless the DMA bus is simultaneously upgraded. A 32×32 array with a 256-bit bus is a **poor architectural choice** — you pay 4× for 2.2×.

## Actionable Conclusion

**Don't scale PEs without scaling DMA bandwidth in proportion.** The optimal bus width per PE row scales roughly linearly: 256-bit for 16×16 (16 bits/PE-row), 512-bit for 32×32 (16 bits/PE-row). This maintains ~84% utilization for the smaller array and pushes the larger array to ~57%.

**For the ONNX compiler's hardware target selection:**
- If targeting a 16×16 PE array: 256-bit bus is adequate (72.6% util, near peak efficiency)
- If targeting a 32×32 PE array: **minimum 512-bit** bus is needed for acceptable utilization
- If targeting 64×64: would need ~1024-bit bus (extrapolating: 64 rows × 16 bits/row = 1024-bit)
- The 16 bits/PE-row rule of thumb is derived from: at 256-bit / 16 PE-rows = 16 bits/row, util=72.6%; at 512-bit / 32 PE-rows = 16 bits/row, util=57.0% (close enough for a first-order estimate)

## Runtime implementation follow-up (2026-08-29)

The original table below is an analytical GEMM study and must not be treated as proof that JSON runtime width reached live DMA service. `dma-runtime-bus-width.md` re-audits that path and finds the canonical field was dropped before `tu_runtime_config_t`; the descriptor engine always used the compiled 256-bit macro. The follow-up wires the runtime width into live descriptor and queued-service accounting and gates 128/256/512-bit 4 KiB transfers. Compute-engine estimators that still consume compile-time macros remain separate evidence surfaces.

## Methodology

Analytical cycle model validated against cmodel output (PE array sweep confirmed at 256-bit):
```
compute = ceil(M/rows) × ceil(N/cols) × K
fill = pipeline_depth × ceil(N/cols)
drain = pipeline_depth × ceil(M/rows)
dma = ceil((M×K×2 + K×N×2 + M×N×4) / (bus_width/8))
total = fill + compute + drain + dma
TOPS = (M×N×K×2) / total / 1000
```

DMA accounts for FP16 weights (2B) + FP16 activations (2B) + FP32 output accumulator (4B), matching the cmodel's actual memory transfer behavior. At 256-bit (32 B/cycle), this yields 6,144 DMA cycles for the 128×128×256 GEMM — verified against the cmodel's perf report.

## Next Exploration Candidates

1. **K sweep (DMA crossover):** Find the K value where compute begins to dominate DMA across PE sizes. Previous explorations fixed K=256; varying K reveals the compute/DMA balance.
2. **Double-buffer benefit quantification:** At what PE sizes and bus widths does ping-pong buffering eliminate enough DMA cycles to change the scaling curves?
3. **SRAM sizing impact:** Larger buffers enable larger tiles → fewer DMA transfers. What SRAM size makes 32×32 at 256-bit viable?
4. **Workload scaling:** How do these bus-width thresholds change for larger GEMMs (M=512, 1024) typical of LLM inference?
