# LayerNorm vs RMSNorm — Bandwidth Comparison

**Date:** 2026-06-30
**Question:** Does RMSNorm have a memory-bandwidth advantage over LayerNorm?

## Config Matrix

| Parameter | Values |
|-----------|--------|
| Element count | 256, 512, 768, 1024, 2048, 4096, 8192 |
| Mode | LayerNorm, RMSNorm |
| Epsilon | 1e-5 |
| Gamma/Beta | None (identity) |

All tests single-tensor (not per-row 2D), in-place FP32.

## Results

| ElemCount | Mode | StallCycles | Cycles/Elem | RelCost(vs LN) |
|-----------|------|-------------|-------------|-----------------|
| 256 | LayerNorm | 512 | 2.000 | 1.00× |
| 256 | RMSNorm | 512 | 2.000 | 1.00× |
| 512 | LayerNorm | 1024 | 2.000 | 1.00× |
| 512 | RMSNorm | 1024 | 2.000 | 1.00× |
| 768 | LayerNorm | 1536 | 2.000 | 1.00× |
| 768 | RMSNorm | 1536 | 2.000 | 1.00× |
| 1024 | LayerNorm | 2048 | 2.000 | 1.00× |
| 1024 | RMSNorm | 2048 | 2.000 | 1.00× |
| 2048 | LayerNorm | 4096 | 2.000 | 1.00× |
| 2048 | RMSNorm | 4096 | 2.000 | 1.00× |
| 4096 | LayerNorm | 8192 | 2.000 | 1.00× |
| 4096 | RMSNorm | 8192 | 2.000 | 1.00× |
| 8192 | LayerNorm | 16384 | 2.000 | 1.00× |
| 8192 | RMSNorm | 16384 | 2.000 | 1.00× |

## Finding

**LayerNorm and RMSNorm have identical memory bandwidth profiles.** Both produce exactly `2.0 × elem_count` stall cycles because both are two-pass algorithms: each element is read and written in pass 1 (statistics), then read and written in pass 2 (normalize + scale). The SRAM bandwidth model counts accesses, not compute complexity.

RMSNorm computes only one statistic (rms²) vs LayerNorm's two (mean + variance), but this compute difference doesn't show up in the cycle model — the same number of SRAM reads/writes dominate. Linear scaling from 256→8192 confirms the formula is purely `2 × N`.

**Architectural implication:** LayerNorm vs RMSNorm is a compute/numerical decision, not a bandwidth optimization target. Choose:
- **RMSNorm** when you don't need per-element bias (simpler hardware, fewer adders in statistics unit)
- **LayerNorm** when you need bias or mean-centering (better for certain transformer variants)

This mirrors the softmax mode comparison — algorithmic differences don't translate to bandwidth differences when the memory access pattern is identical.

## Test Harness

`tests/test_norm_sweep.c` — see `make test-norm-sweep`
