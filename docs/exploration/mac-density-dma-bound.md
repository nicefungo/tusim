# MAC Density × DMA Bandwidth: When Does More Compute Stop Helping?

**Date:** 2026-06-25
**Question:** How many MAC units per PE are worthwhile before DMA bandwidth becomes the hard ceiling? At what MAC density does the accelerator transition from compute-bound to memory-bound?

**Hypothesis:** Below ~4 MACs/PE, each doubling of compute density gives near-linear throughput gains. Above 8 MACs/PE, the DMA floor dominates and additional compute provides diminishing returns — the accelerator is memory-bound regardless of how fast the PEs are.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| MAC units per PE | 1, 2, 4, 8, 16, 32, 64, ∞ | Compute density per PE |
| PE array | 16×16 (256 PEs) | Fixed, from prior sweet-spot findings |
| Dataflow | WS (weight-stationary) | Baseline |
| Pipeline depth | 2 | Default systolic |
| Bus width | 256-bit (32 B/cycle) | Default DMA |
| Clock | 1.0 GHz | Default |
| Workload | M=N=K=64, M=N=K=128, M=N=K=256 | Three GEMM sizes |

**Configs tested:** 24 (8 MAC densities × 3 workloads), analytical cycle model.

## Cycle Model

```
WS: total = pd×nt + ceil(mt×nt×K/mac) + pd×mt + dma
where mt=ceil(M/pe_r), nt=ceil(N/pe_c)
dma = ceil((M×K×2 + K×N×2 + M×N×4) / bus_width_bytes)
bus_width_bytes = 32 (256-bit)
pd = 2
```

The MAC units per PE divide compute cycles: if each PE can do 2 MACs/cycle instead of 1, a K=128 tile completes in 64 cycles instead of 128. Fill/drain overhead (pipeline startup/teardown) is unchanged. DMA is completely unaffected — it's purely an off-chip bandwidth constraint.

## Results: K=64 (small inner dimension, 0.52 MFLOPs)

| MAC/PE | PE MACs | Compute | Fill+Drain | DMA | Total | TOPS | Speedup | Util% |
|--------|---------|---------|------------|-----|-------|------|---------|-------|
| 1 | 256 | 1,024 | 32 | 1,024 | 2,080 | 0.252 | 1.00× | 49.2% |
| 2 | 512 | 512 | 32 | 1,024 | 1,568 | 0.334 | 1.33× | 32.7% |
| 4 | 1,024 | 256 | 32 | 1,024 | 1,312 | 0.400 | 1.59× | 19.5% |
| 8 | 2,048 | 128 | 32 | 1,024 | 1,184 | 0.443 | 1.76× | 10.8% |
| 16 | 4,096 | 64 | 32 | 1,024 | 1,120 | 0.468 | 1.86× | 5.7% |
| 32 | 8,192 | 32 | 32 | 1,024 | 1,088 | 0.482 | 1.91× | 2.9% |
| 64 | 16,384 | 16 | 32 | 1,024 | 1,072 | 0.489 | 1.94× | 1.5% |
| ∞ | ∞ | 0 | 32 | 1,024 | 1,056 | 0.497 | 1.97× | 0.0% |

**Asymptotic max speedup: 1.97×** — you can never more than double throughput regardless of compute.

## Results: K=128 (medium, 4.19 MFLOPs)

| MAC/PE | PE MACs | Compute | Fill+Drain | DMA | Total | TOPS | Speedup | Util% |
|--------|---------|---------|------------|-----|-------|------|---------|-------|
| 1 | 256 | 8,192 | 32 | 4,096 | 12,320 | 0.340 | 1.00× | 66.5% |
| 2 | 512 | 4,096 | 32 | 4,096 | 8,224 | 0.510 | 1.50× | 49.8% |
| 4 | 1,024 | 2,048 | 32 | 4,096 | 6,176 | 0.679 | 2.00× | 33.2% |
| 8 | 2,048 | 1,024 | 32 | 4,096 | 5,152 | 0.814 | 2.39× | 19.9% |
| 16 | 4,096 | 512 | 32 | 4,096 | 4,640 | 0.904 | 2.66× | 11.0% |
| 32 | 8,192 | 256 | 32 | 4,096 | 4,384 | 0.957 | 2.81× | 5.8% |
| 64 | 16,384 | 128 | 32 | 4,096 | 4,256 | 0.985 | 2.90× | 3.0% |
| ∞ | ∞ | 0 | 32 | 4,096 | 4,128 | 1.016 | 2.98× | 0.0% |

**Asymptotic max speedup: 2.98×.**

## Results: K=256 (large, 33.55 MFLOPs)

| MAC/PE | PE MACs | Compute | Fill+Drain | DMA | Total | TOPS | Speedup | Util% |
|--------|---------|---------|------------|-----|-------|------|---------|-------|
| 1 | 256 | 65,536 | 32 | 16,384 | 81,952 | 0.409 | 1.00× | 80.0% |
| 2 | 512 | 32,768 | 32 | 16,384 | 49,184 | 0.682 | 1.67× | 66.6% |
| 4 | 1,024 | 16,384 | 32 | 16,384 | 32,800 | 1.023 | 2.50× | 50.0% |
| 8 | 2,048 | 8,192 | 32 | 16,384 | 24,608 | 1.363 | 3.33× | 33.3% |
| 16 | 4,096 | 4,096 | 32 | 16,384 | 20,512 | 1.636 | 4.00× | 20.0% |
| 32 | 8,192 | 2,048 | 32 | 16,384 | 18,464 | 1.817 | 4.44× | 11.1% |
| 64 | 16,384 | 1,024 | 32 | 16,384 | 17,440 | 1.924 | 4.70× | 5.9% |
| ∞ | ∞ | 0 | 32 | 16,384 | 16,416 | 2.044 | 4.99× | 0.0% |

**Asymptotic max speedup: 4.99×.**

## Key Finding

**The MAC-density sweet spot depends on K-dimension, not M or N.** For small K (attention-style workloads), adding MACs beyond 2/PE is wasted. For large K (dense GEMM), you can justify 4-8 MACs/PE before hitting the DMA wall.

### Speedup efficiency (actual ÷ ideal)

| MAC/PE | K=64 | K=128 | K=256 |
|--------|------|-------|-------|
| 2 | 66% | 75% | 83% |
| 4 | 40% | 50% | 62% |
| 8 | 22% | 30% | 42% |
| 16 | 12% | 17% | 25% |

At K=256, 2 MACs/PE captures 83% of ideal 2× speedup — a good investment. At K=64, even 2 MACs/PE captures only 66% — the DMA floor erodes most of the gain.

### The DMA floor formula

The minimum achievable cycle count (infinite compute) is:

```
floor = pd×(mt+nt) + dma
      = pd×(ceil(M/pe_r) + ceil(N/pe_c)) + ceil((M×K×2 + K×N×2 + M×N×4)/bus_bytes)
```

This is entirely determined by DMA bus width (fixed at 256-bit here) and PE array dimensions. No amount of PE compute can go below this floor. For the 16×16 PE, 256-bit bus configuration:

| Workload | DMA floor (cycles) | % of 1-MAC total |
|----------|-------------------|-------------------|
| 64×64×64 | 1,056 | 50.8% |
| 128×128×128 | 4,128 | 33.5% |
| 256×256×256 | 16,416 | 20.0% |

## Architectural implications

1. **For attention-heavy accelerators (small K): 1-2 MACs/PE is optimal.** Beyond that, DMA dominates and extra MAC area is wasted silicon. The 1.97× asymptote at K=64 means you can never double throughput regardless of MAC count.

2. **For GEMM-dominated accelerators (large K): 4-8 MACs/PE is justifiable.** At K=256, 8 MACs/PE gives 3.33× speedup (42% efficiency). 16 MACs/PE gives only 4.00× (25% efficiency) — the marginal gain from 8→16 is just 1.20×, barely worth the 2× area cost.

3. **DMA bus width is the real lever.** The DMA floor at K=128 is 33.5% of total cycles. Doubling the bus width to 512-bit (64 B/cycle) would halve the DMA floor to 2,064 cycles and raise the K=128 asymptote from 2.98× to ~5.96×. This is explored in `bus-width-sweep-gemm128.md` which found diminishing returns beyond 512-bit for a different reason (W/A-buffer bandwidth saturation).

4. **The MAC/PE vs. bus-width trade-off is the fundamental accelerator design tension.** At 16×16 PE with 256-bit bus, the compute:DMA ratio is 2:1 at K=128. Adding MACs reduces compute cycles but can never reduce DMA. The only path to >3× throughput for K=128 with this PE array is wider DMA.

## Relationship to prior explorations

- `dataflow-pe-interaction.md` (2026-06-24): Found DMA is 85.6% of cycles at 64×64 PE. This doc shows the flip side — even at modest 16×16 PE, DMA limits the benefit of faster compute.
- `bus-width-sweep-gemm128.md` (2026-06-07): Swept bus width and found diminishing returns after 512-bit. Combined with this finding: wider bus enables more MACs/PE.
- `pe-array-sweep-gemm128.md` (2026-06-03): Found 16×16 PE as sweet spot. This doc shows why — larger PE arrays make the DMA problem worse, not better.
- `dma-channel-queue-sweep.md` (2026-06-20): Explored DMA parallelism. More channels help hide latency but don't increase peak bandwidth — the floor is unchanged.

**Conclusion:** The DMA bus width sets a hard floor on cycle count. Increasing MAC density can only close the gap to that floor — it can never break through it. For the default 16×16 PE / 256-bit bus configuration, 2-4 MACs/PE captures most achievable speedup while keeping PE area efficient. The real performance headroom is in wider DMA, not denser compute.
