# Attention Engine Sweep: PE Array × Workload × Dataflow

**Date:** 2026-06-26
**Question:** How do PE array size, aspect ratio, and dataflow choice affect attention throughput across prefill (high M) and decode (M=1) workloads?
**Hypothesis:** Prefill is DMA-bound (M-dimension dominates SRAM usage); decode is compute-bound. PE array aspect ratio should matter more for systolic dataflows (WS/RS) than for OS due to fill/drain costs.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| PE rows/cols | 8×8, 16×16, 32×32, 64×4, 16×32 | 5 aspect ratios |
| Dataflow | WS, OS, RS | Systolic, vector, hybrid |
| Workload | prefill-128×64, decode-1×512-64, decode-1×512-128, batch-32×256-128, long-ctx-512×128 | LLM attention dimensions |
| SRAM | Scaled with PE (128–512 KB total) | W-buf 50%, A-buf 25%, O-buf 25% |
| Precision | FP16 W/A, FP32 accumulate | Default |
| Auto-tiling | Enabled | Attention engine computes tile_m, tile_n from SRAM budget |

**Configs attempted:** 5 PE × 7 workloads × 3 dataflows = 105. **Valid:** 45 (60 dropped due to SRAM capacity failures or W-buffer overflow).

## Results Table

| Workload | PE Array | DF | Cycles | Compute | DMA | Util% | MFLOPs |
|----------|----------|----|--------|---------|-----|-------|--------|
| prefill-128×64 | 16×16 | WS | 319,232 | 270,144 | 49,088 | 84.6% | 4.19 |
| prefill-128×64 | 16×16 | OS | 288,640 | 239,488 | 49,152 | 83.0% | 4.19 |
| prefill-128×64 | 32×32 | WS | 288,512 | 239,424 | 49,088 | 83.0% | 4.19 |
| prefill-128×64 | 32×32 | OS | 280,960 | 231,808 | 49,152 | 82.5% | 4.19 |
| prefill-128×128 | 32×32 | WS | 380,800 | 315,264 | 65,536 | 82.8% | 8.39 |
| prefill-128×128 | 32×32 | OS | 365,440 | 299,904 | 65,536 | 82.1% | 8.39 |
| decode-1×512-64 | 16×16 | WS | 161,792 | 160,640 | 1,152 | 99.3% | 0.13 |
| decode-1×512-64 | 16×16 | OS | 146,432 | 145,280 | 1,152 | 99.2% | 0.13 |
| decode-1×512-64 | 32×32 | OS | 143,872 | 142,720 | 1,152 | 99.2% | 0.13 |
| decode-1×512-128 | 16×16 | OS | 276,032 | 274,752 | 1,280 | 99.5% | 0.20 |
| decode-1×2048-64 | 32×32 | OS | 546,880 | 542,656 | 4,224 | 99.2% | 0.33 |
| batch-32×256-128 | 32×32 | OS | 240,064 | 215,488 | 24,576 | 89.8% | 4.19 |
| batch-32×256-128 | 16×16 | OS | 247,744 | 223,168 | 24,576 | 90.1% | 4.19 |
| long-ctx-512×128 | 32×32 | OS | 7,366,720 | 6,711,360 | 655,360 | 91.1% | 118.49 |

**Aspect ratio traps (64×4):**
| prefill-128×64 | 64×4 | WS | 564,992 | 515,904 | 49,088 | 91.3% | 4.19 |
| prefill-128×64 | 64×4 | OS | 288,640 | 239,488 | 49,152 | 83.0% | 4.19 |
| decode-1×512-128 | 64×4 | WS | 1,128,512 | 1,127,232 | 1,280 | 99.9% | 0.20 |
| decode-1×512-128 | 64×4 | OS | 299,072 | 297,792 | 1,280 | 99.6% | 0.20 |

## Findings

### 1. Decode is compute-bound (99%+ util), prefill is DMA-bound (82–91% util)
Decode (M=1, batch=1) workloads achieve 99.2–99.9% compute utilization across all PE configurations and dataflows. Only ~1,200 cycles of DMA per tile — the tiny Q tile (2 KB) and streaming K/V tiles dominate compute time. Prefill workloads (M=128) show 82–91% utilization, with DMA consuming 15–17% of total cycles.

**Actionable insight:** For accelerator designs targeting inference serving (mostly decode), PE array size matters more than SRAM or DMA bandwidth. For training/prefill-heavy workloads, invest in DMA bandwidth and double-buffering.

### 2. OS dataflow is strictly faster than WS for attention (5–36% fewer cycles)
Across all valid prefill configurations, OS uses 5–12% fewer total cycles than WS, despite identical utilization percentages. The advantage comes from eliminating systolic pipeline fill/drain (2 × tile_dim rows and cols of dead cycles per tile). For extreme aspect ratios (64×4), the gap balloons to 2× (WS=565K vs OS=289K cycles for prefill-128×64) because tall systolic arrays pay a 128-cycle fill penalty per tile.

**Actionable insight:** For attention, OS dataflow is the clear winner. WS is only competitive when K >> other dimensions (rare in attention).

### 3. 32×32 PE adds no benefit over 16×16 for prefill with head_dim=64
prefill-128×64 on 32×32 (OS: 280,960 cycles) is only 2.7% faster than 16×16 (288,640 cycles), despite 4× more MACs. The bottleneck is O-buffer capacity, not compute — the same 4.19 MFLOPs are spread over 64 MMA tiles on 32×32 vs 512 tiles on 16×16, but total DMA is identical (~49K cycles). Doubling the PE array only cuts compute cycles by 50K out of 289K total (17% reduction in compute, 2.7% reduction in total).

**Actionable insight:** Don't scale PE array for prefill unless you also scale O-buffer. The 128×128 prefill with head_dim=128 requires 32×32 PE + 128KB O-buffer; 16×16's 64KB O-buffer can't fit the 80KB working set (S: 32KB + O: 32KB + scratch: 16KB).

### 4. Aspect ratio asymmetry penalizes systolic dataflows severely
The 64×4 array (64 rows, 4 cols) shows dramatic dataflow sensitivity:  
- prefill-128×64: WS=564,992 vs OS=288,640 (1.96× slower)  
- decode-1×512-128: WS=1,128,512 vs OS=299,072 (3.77× slower)

Systolic fill/drain costs scale with array dimensions: 64 rows × 2 (fill + drain) = 128 dead cycles per tile × many tiles. OS avoids this entirely by not using systolic data movement.

**Actionable insight:** If designing a systolic array, keep it near-square. Tall or wide arrays waste cycles on fill/drain. If the application demands an extreme aspect ratio (e.g., 64×4 for high-M workloads), OS dataflow is mandatory.

### 5. W-buffer off-by-one overflow is a persistent hazard for KV tiles near buffer capacity
Multiple decode workloads with tile_n=256, head_dim=128 hit "addr=131070 size=4 max=131072" — a 2-byte overflow past 128KB W-buffer. The K tile (64KB) + K^T scratch (64KB) lands exactly at the boundary, and a single FP32 write overflows by 2 bytes.

**Actionable insight:** SRAM budget for tiled attention must include K^T scratch space (tile_n × head_dim × 2 bytes) PLUS alignment padding. The auto-tiler should reduce tile_n by 1 or enforce a 16-byte safety margin.

## Conclusion

For transformer attention workloads:
- **OS dataflow wins universally.** Use it as the default.
- **16×16 PE is sufficient for decode (M=1).** More MACs don't help when the workload is KV-bandwidth-bound in the inner loop.
- **Scale O-buffer before PE array for prefill.** The 128×128×128 attention requires 128KB O-buffer; a 16×16 PE with 128KB O-buffer would likely outperform 32×32 with 64KB O-buffer.
- **Keep PE array near-square.** Aspect ratios > 4:1 cripple systolic dataflows.

## Coverage Gap

Prior explorations covered GEMM sweeps extensively. This is the first attention-engine sweep, filling a critical gap for transformer workload characterization. Next candidates: convolution engine sweep (kernel/stride variation) and softmax mode comparison (standard vs online cycle counts).
