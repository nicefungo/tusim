#!/usr/bin/env python3
"""Workload aspect ratio sweep: edge effects from misaligned tile dimensions."""
import math

pe_rows, pe_cols = 16, 16
pipeline_depth = 2
bus_width_bytes = 32
dtype_bytes = 2

# Mix of aligned and misaligned dimensions, M×N covers 256..16K range
workloads = []
for M in [16, 20, 32, 40, 64, 80, 96, 128, 160, 192, 200, 256]:
    for N in [16, 32, 48, 64, 80, 96, 128, 160, 192, 256]:
        # Skip where total FLOPs would be absurdly small or large
        mn = M * N
        if mn < 256 or mn > 65536:
            continue
        workloads.append((M, N, 128))

print(f"**Configs tested:** {len(workloads)}")
print()

# Collect by aspect ratio for summary
ratio_buckets = {}

# Sorted by M then N
workloads.sort()
print("| M | N | Tiles | Full | Edge | Util% | Fill | Comp | Drain | DMA | Total | TOPS |")
print("|---|----|-------|------|------|-------|------|------|-------|-----|-------|------|")

for M, N, K in workloads:
    tiles_m = math.ceil(M / pe_rows)
    tiles_n = math.ceil(N / pe_cols)
    total_tiles = tiles_m * tiles_n

    full_m = M // pe_rows
    full_n = N // pe_cols
    full_tiles = full_m * full_n
    edge_tiles = total_tiles - full_tiles

    actual_macs = sum(
        min(pe_rows, M - mt * pe_rows) * min(pe_cols, N - nt * pe_cols) * K
        for mt in range(tiles_m) for nt in range(tiles_n)
    )
    max_possible_macs = total_tiles * pe_rows * pe_cols * K
    utilization = actual_macs / max_possible_macs

    fill = pipeline_depth * tiles_n
    compute = tiles_m * tiles_n * K
    drain = pipeline_depth * tiles_m

    total_bytes = (M * K + K * N + M * N) * dtype_bytes
    dma = math.ceil(total_bytes / bus_width_bytes)

    total_cycles = fill + compute + drain + dma
    flops = actual_macs * 2
    # TOPS = (FLOPs / 1e12) / (total_cycles / 1e9) = FLOPs / total_cycles * 1e-3
    tops = flops / total_cycles * 1e-3

    print(f"| {M} | {N} | {total_tiles} | {full_tiles} | {edge_tiles} "
          f"| {utilization*100:.1f}% | {fill} | {compute} | {drain} | {dma} "
          f"| {total_cycles} | {tops:.3f} |")

    ratio = round(max(M/N, N/M), 1)
    bucket = ratio_buckets.setdefault(ratio, [])
    bucket.append((utilization * 100, tops, edge_tiles))

print()
print("### Summary by Aspect Ratio")
print()
print("| M:N Ratio | Samples | Avg Util% | Best TOPS | Worst TOPS | Avg Edge Tiles |")
print("|-----------|---------|-----------|-----------|------------|----------------|")
for ratio in sorted(ratio_buckets.keys()):
    bucket = ratio_buckets[ratio]
    avg_u = sum(r[0] for r in bucket) / len(bucket)
    best_t = max(r[1] for r in bucket)
    worst_t = min(r[1] for r in bucket)
    avg_e = sum(r[2] for r in bucket) / len(bucket)
    print(f"| {ratio}:1 | {len(bucket)} | {avg_u:.1f}% | {best_t:.3f} | {worst_t:.3f} | {avg_e:.1f} |")

# Show misalignment penalty
print()
print("### Misalignment Penalty: Perfect vs Partial Tiles")
perfect = [(M,N,util*100,tops) for M,N,K,util,cyc,tops,_,_,_,_ in [
    (0,0,0,0,0,0,0,0,0,0)] if False]  # stub

# Find best perfectly-aligned vs worst edge-heavy
print()
print("**Worst edge utilization:**")
print("| M | N | Tiling | Util% | Edge tiles | TOPS |")
print("|---|---|--------|-------|------------|------|")
worst_cases = []
for M, N, K in workloads:
    tiles_m = math.ceil(M / pe_rows)
    tiles_n = math.ceil(N / pe_cols)
    edge = (tiles_m * tiles_n) - (M//pe_rows)*(N//pe_cols)
    actual_macs = sum(
        min(pe_rows, M - mt*pe_rows) * min(pe_cols, N - nt*pe_cols) * K
        for mt in range(tiles_m) for nt in range(tiles_n)
    )
    max_macs = tiles_m * tiles_n * pe_rows * pe_cols * K
    util = actual_macs / max_macs * 100
    total = 2*pipeline_depth + tiles_m*tiles_n*K + math.ceil((M*K+K*N+M*N)*2/32)
    tops = actual_macs * 2 / total * 1e-3
    worst_cases.append((M, N, f"{tiles_m}×{tiles_n}", util, edge, tops))

worst_cases.sort(key=lambda x: x[3])
for M, N, tiling, util, edge, tops in worst_cases[:8]:
    print(f"| {M} | {N} | {tiling} | {util:.1f}% | {edge} | {tops:.3f} |")
