# Softmax Mode Comparison: Standard vs Log vs Online

**Date:** 2026-06-28  
**Question:** How do SRAM stall cycles differ across softmax modes (Standard, Log, Online) for attention-like workloads?

## Config Matrix

| Parameter | Values |
|-----------|--------|
| Mode | Standard, Log, Online |
| Row count | 1, 16, 64, 128, 256 |
| Row dimension (cols) | 64, 128, 256, 512 |
| Data type | FP32 in SRAM |
| Scale | 1.0 (none) |

## Results

| Workload | Elems | Std | Log | Online | Best |
|----------|-------|-----|-----|--------|------|
| 1×64 (single row, small head) | 64 | 256 | 256 | 256 | Standard |
| 1×128 (typical attention head) | 128 | 512 | 512 | 512 | Standard |
| 1×256 (large head_dim) | 256 | 1,024 | 1,024 | 1,024 | Standard |
| 1×512 (very large head_dim) | 512 | 2,048 | 2,048 | 2,048 | Standard |
| 16×64 (batched small) | 1,024 | 4,096 | 4,096 | 4,096 | Standard |
| 16×128 (batched typical) | 2,048 | 8,192 | 8,192 | 8,192 | Standard |
| 64×64 (square attention score) | 4,096 | 16,384 | 16,384 | 16,384 | Standard |
| 128×128 (large square) | 16,384 | 65,536 | 65,536 | 65,536 | Standard |
| 256×128 (seq_len × head_dim) | 32,768 | 131,072 | 131,072 | 131,072 | Standard |

## Finding

**Softmax mode has zero bandwidth differentiation.** All three modes (Standard, Log, Online) produce identical SRAM stall cycles for any given element count. The relationship is strictly linear:

```
stall_cycles = 4 × elem_count
```

This is because all three modes use the same memory access pattern: read each FP32 element, compute, write each FP32 element. The algorithmic differences (two-pass max/subtract vs. single-pass rescaling vs. log-transform) affect only compute complexity, not memory bandwidth.

**Cycle model note:** `tu_softmax_execute()` returns SRAM stall cycles only, not total compute cycles. The compute cost of each mode is not captured. For a total-throughput comparison, a cycle-accurate compute model would be needed.

## Architectural Impact

- **Mode selection is a compute decision, not a bandwidth decision.** Choose Online for streaming/attention pipelines (single pass), Log for numerical stability in cross-entropy, and Standard as the default.
- **Softmax is SRAM-bandwidth-bound.** With 4 cycles per element, a 128×128 attention score matrix (16,384 elems) costs 65,536 stall cycles — comparable to a medium-sized GEMM tile.
- **Reducing stall cycles requires SRAM bandwidth optimization** (wider bus, banking), not mode tuning.

## Next Steps

- Convolution engine kernel/stride sweep (underexplored)
- LayerNorm vs RMSNorm throughput comparison (underexplored)
