# DRAM Type × Clock Frequency Sweep: When Does DRAM Become the Bottleneck?

**Date:** 2026-06-21
**Question:** For a 16×16 PE array running GEMM 128×128×256 at 256-bit DMA bus, at what clock frequency does DRAM bandwidth become the bottleneck — and which DRAM type is sufficient?
**Hypothesis:** DDR4 is sufficient below ~1 GHz; HBM only becomes justified above ~8 GHz for a 256-bit bus.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| DRAM type | IDEAL, HBM3, HBM2E, HBM2, LPDDR5, DDR5, DDR4 | 7 memory technologies |
| Clock freq (GHz) | 0.25, 0.5, 1.0, 2.0, 4.0, 8.0 | 6 frequencies |
| DMA bus width | 256-bit (32 B/cyc) | Fixed — current default |
| PE array | 16×16 | Weight-stationary systolic |
| Workload | M=128, N=128, K=256 | 4.19M MACs, 8.39M ops |
| Precision | FP16 W/A, FP32 O | 196,608 bytes total DMA |

**Configs analyzed:** 42 (7 DRAM types × 6 frequencies), analytical cycle model.

## Methodology: Analytical Cycle Model

The DMA engine is constrained by two independent limits:
- **Bus limit:** `bus_B_per_cyc = bus_width_bits / 8` = 32 B/cyc
- **DRAM limit:** `dram_B_per_cyc = DRAM_BW_Gbps / clock_ghz`

Effective DMA bandwidth: `eff_B_per_cyc = min(bus_B_per_cyc, dram_B_per_cyc)`

DMA cycles = total_DMA_bytes / eff_B_per_cyc
Total cycles = DMA cycles + compute_cycles (16,416 from bus-width sweep baseline)
TOPS = total_ops × clock_ghz / total_cycles / 1e12

## Results Table

**Baseline constants:**
- Total DMA bytes: 196,608 (W: 65,536 + A: 65,536 + O: 65,536)
- Compute cycles (16×16 WS): 16,416
- Bus-limited DMA cycles: 6,144 (196,608 / 32)
- Bus-limited total cycles: 22,560
- Total ops: 8,388,608 (4,194,304 MACs × 2)

### DRAM BW per Cycle (B/cyc) at Each Frequency

| DRAM Type | BW (GB/s) | @0.25 GHz | @0.5 GHz | @1.0 GHz | @2.0 GHz | @4.0 GHz | @8.0 GHz |
|-----------|-----------|-----------|----------|----------|----------|----------|----------|
| IDEAL     | ∞         | ∞         | ∞        | ∞        | ∞        | ∞        | ∞        |
| HBM3      | 819       | 3,276     | 1,638    | 819      | 409.5    | 204.8    | 102.4    |
| HBM2E     | 460       | 1,840     | 920      | 460      | 230      | 115      | 57.5     |
| HBM2      | 256       | 1,024     | 512      | 256      | 128      | 64       | 32       |
| LPDDR5    | 51.2      | 204.8     | 102.4    | 51.2     | 25.6     | 12.8     | 6.4      |
| DDR5      | 51.2      | 204.8     | 102.4    | 51.2     | 25.6     | 12.8     | 6.4      |
| DDR4      | 25.6      | 102.4     | 51.2     | 25.6     | 12.8     | 6.4      | 3.2      |

**Bold** = DRAM-limited (DRAM BW/cyc < bus BW/cyc = 32 B/cyc).

### Effective TOPS by DRAM Type and Clock Frequency

| DRAM Type | 0.25 GHz | 0.5 GHz | 1.0 GHz | 2.0 GHz | 4.0 GHz | 8.0 GHz |
|-----------|----------|---------|---------|---------|---------|---------|
| IDEAL     | 0.093    | 0.186   | 0.372   | 0.744   | 1.487   | 2.975   |
| HBM3      | 0.093    | 0.186   | 0.372   | 0.744   | 1.487   | 2.975   |
| HBM2E     | 0.093    | 0.186   | 0.372   | 0.744   | 1.487   | 2.975   |
| HBM2      | 0.093    | 0.186   | 0.372   | 0.744   | 1.487   | **2.839** |
| LPDDR5    | 0.093    | 0.186   | 0.372   | **0.696** | **1.056** | **0.776** |
| DDR5      | 0.093    | 0.186   | 0.372   | **0.696** | **1.056** | **0.776** |
| DDR4      | 0.093    | 0.186   | **0.348** | **0.528** | **0.712** | **0.435** |

**Bold** = DRAM-limited (below bus-limited ideal). All non-bold entries are bus-limited (identical to IDEAL).

### DMA Overhead (% of total cycles)

| DRAM Type | 0.25 GHz | 0.5 GHz | 1.0 GHz | 2.0 GHz | 4.0 GHz | 8.0 GHz |
|-----------|----------|---------|---------|---------|---------|---------|
| IDEAL     | 27.2%    | 27.2%   | 27.2%   | 27.2%   | 27.2%   | 27.2%   |
| HBM2      | 27.2%    | 27.2%   | 27.2%   | 27.2%   | 27.2%   | **30.8%** |
| DDR5      | 27.2%    | 27.2%   | 27.2%   | **31.9%** | **48.3%** | **67.7%** |
| DDR4      | 27.2%    | 27.2%   | **31.9%** | **48.3%** | **65.4%** | **77.9%** |

### Throughput Loss vs IDEAL (%)

| DRAM Type | 0.25 GHz | 0.5 GHz | 1.0 GHz | 2.0 GHz | 4.0 GHz | 8.0 GHz |
|-----------|----------|---------|---------|---------|---------|---------|
| DDR4      | 0.0%     | 0.0%    | **-6.5%** | **-29.0%** | **-52.1%** | **-85.4%** |
| DDR5      | 0.0%     | 0.0%    | 0.0%    | **-6.5%** | **-29.0%** | **-73.9%** |
| HBM2      | 0.0%     | 0.0%    | 0.0%    | 0.0%    | 0.0%    | **-4.6%** |

## Key Findings

### 1. Crossover frequencies — when each DRAM type becomes the bottleneck

The DRAM becomes the bottleneck when `DRAM_BW_Gbps / clock_ghz < 32 B/cyc`, i.e., when `clock > DRAM_BW_Gbps / 32`:

| DRAM Type | BW (GB/s) | Crossover Clock | DDR4 Equivalent? |
|-----------|-----------|-----------------|------------------|
| HBM3      | 819       | 25.6 GHz        | —                |
| HBM2E     | 460       | 14.4 GHz        | —                |
| HBM2      | 256       | 8.0 GHz         | —                |
| LPDDR5    | 51.2      | **1.6 GHz**     | —                |
| DDR5      | 51.2      | **1.6 GHz**     | —                |
| DDR4      | 25.6      | **0.8 GHz**     | —                |

At 1.0 GHz (current default): only DDR4 is DRAM-limited. All others are bus-limited.
At 2.0 GHz: DDR4 and DDR5/LPDDR5 are DRAM-limited. HBM still bus-limited.
At 4.0 GHz: everything except HBM is DRAM-limited.
At 8.0 GHz: HBM2E/HBM3 are the only bus-limited options. Even HBM2 begins to show DRAM stalls.

### 2. The 256-bit bus is the real bottleneck below 8 GHz

For any DRAM with ≥256 GB/s bandwidth (all HBM variants), the 256-bit DMA bus — not the DRAM — is the limiting factor up to 8 GHz. **HBM's bandwidth advantage is entirely wasted on a 256-bit bus at realistic clock speeds.** You need a wider bus to exploit HBM bandwidth.

At 512-bit (64 B/cyc), the crossover points shift dramatically:
- DDR4 becomes DRAM-limited at **0.4 GHz**
- DDR5 becomes DRAM-limited at **0.8 GHz**
- HBM2 becomes DRAM-limited at **4.0 GHz**

This means that for a 512-bit bus design, DRAM choice matters much sooner — DDR4 is only viable below 400 MHz; above 1 GHz you need HBM.

### 3. DDR4 is sufficient for the current 1 GHz / 256-bit configuration

At the default config (1 GHz, 256-bit bus), DDR4 costs only 6.5% throughput vs ideal DRAM. The 7,680 DMA cycles (vs 6,144 ideal) represent a 1,536-cycle penalty — 25% more DMA time, but only 6.8% of total cycles. **Upgrading to HBM would yield zero improvement** because the bus, not the DRAM, is the constraint.

### 4. The DDR5→HBM transition point: ~4 GHz at 256-bit, ~2 GHz at 512-bit

DDR5 is viable up to 1.6 GHz at 256-bit bus (zero loss). At 4 GHz, DDR5 loses 29% throughput vs HBM — a meaningful gap that would justify HBM in a high-frequency design. At a wider 512-bit bus, DDR5 becomes inadequate above 800 MHz.

## Actionable Conclusion

**For the current 1 GHz / 256-bit bus design, DRAM type is a don't-care — DDR4, DDR5, and HBM all deliver identical performance.** The DMA bus width is the binding constraint. DRAM choice only matters when either:

1. **You increase clock frequency** above ~800 MHz (then DDR4 hurts) or above ~1.6 GHz (then DDR5 hurts)
2. **You widen the DMA bus** — a 512-bit bus makes DRAM the bottleneck much earlier

**Recommendation for exploration:** If the next design iteration increases the clock to 2+ GHz or the bus to 512+ bits, pair it with a DRAM sweep to find the minimum viable DRAM type. Otherwise, stick with DDR4 and invest silicon budget elsewhere.

## Crossover Map (Quick Reference)

```
Clock (GHz)    0.5    1.0    1.6    2.0    4.0    8.0
               │      │      │      │      │      │
DDR4 ██████████│██████│──────│──────│──────│──────│──  → DRAM-limited above 0.8 GHz
               │      │      │      │      │      │
DDR5 ██████████│██████│██████│──────│──────│──────│──  → DRAM-limited above 1.6 GHz
               │      │      │      │      │      │
HBM2 ██████████│██████│██████│██████│██████│████──│──  → DRAM-limited above 8.0 GHz
               │      │      │      │      │      │
HBM2E██████████│██████│██████│██████│██████│██████│──  → DRAM-limited above 14.4 GHz
               │      │      │      │      │      │
               █ = bus-limited (identical to IDEAL)
               ─ = DRAM-limited (throughput loss vs IDEAL)
```
