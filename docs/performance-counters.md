# Performance Counter Infrastructure

> **Gap:** E4 (Power/Energy Model), P2.5 (Cycle-Accurate Model) foundation  
> **Status:** Implemented  
> **Version:** 1.0  
> **Date:** 2026-05-31

## Overview

The Performance Counter Infrastructure provides comprehensive monitoring for the production-grade TU cmodel. It enables accurate cycle counting, utilization tracking, stall analysis, per-operation accounting, and energy estimation — all driven by a single, pluggable counter module.

### Why This Matters

The original TinyTU had 5 ad-hoc counters (`total_mma_calls`, `estimated_cycles`, etc.) that were manually incremented in different parts of the code. There was no systematic way to:

1. **Measure utilization** — Is the compute array actually busy?
2. **Diagnose stalls** — Where is the bottleneck? DMA? SRAM bandwidth? Pipeline bubbles?
3. **Estimate power** — How much energy does each operation consume?
4. **Compare configurations** — Does doubling PE rows actually improve throughput?
5. **Generate reports** — Can we dump a performance summary after every run?

The new infrastructure solves all of these with a single cohesive API.

## Architecture

### Counter Categories

```
tu_perf_counters_t
├── dma        — DMA transfer tracking (bytes, cycles, stalls, per-channel)
├── compute    — Compute engine tracking (MACs, utilization, per-op counts)
├── memory     — Memory hierarchy tracking (reads/writes, bank conflicts, DRAM)
├── power      — Energy estimation (pJ per component, total power)
└── global     — Cycle counter, wall-clock time, enable/disable
```

### Key Design Decisions

1. **Monotonically Increasing:** All counters only increase. This makes snapshot/diff correct by construction — no need to worry about counter reset race conditions.

2. **Pluggable Energy Model:** Energy parameters (`pj_per_mac`, `pj_per_sram_read`, etc.) are configurable per technology node. Default values correspond to a conservative 45nm-like process.

3. **Per-Operation Tracking:** Each compute operation type (FP16 MMA, Conv2D, Attention, Softmax, PoolMax, etc.) has its own counter, enabling per-layer performance analysis.

4. **Snapshot/Diff/Merge:** Take a snapshot before and after a region of interest, compute the differential, and analyze only that region. Merge multi-core counters into aggregate.

## API Reference

### Lifecycle

```c
// Initialize with 1 GHz clock
tu_perf_counters_t perf;
tu_perf_init(&perf, 1000.0);

// Disable during fast-path code
tu_perf_set_enabled(&perf, false);

// Reset counters (preserves energy params)
tu_perf_reset(&perf);
```

### DMA Recording

```c
// Record a DMA read (e.g., weight load from DRAM to SRAM)
tu_perf_dma_record_read(&perf,
    bytes,              // Transfer size
    active_cycles,      // Cycles doing actual transfer
    stall_cycles,       // Cycles stalled waiting
    channel,            // DMA channel (0=W, 1=A, 2=O)
    transfer_type);     // 0=linear, 1=strided_2d, 3=scatter, 4=gather
```

### Compute Recording

```c
// Record an MMA operation
tu_perf_compute_record_mma(&perf,
    macs,               // Total MAC operations
    M, N, K,            // Matrix dimensions
    tiles, edge_tiles,  // Tile counts
    active_cycles,      // Useful compute cycles
    stall_cycles,       // Stalled cycles
    precision_type,     // 1=FP16, 2=BF16, 3=INT8, 4=FP8
    dataflow_mode);     // 0=WS, 1=OS

// Record any compute operation
tu_perf_compute_record_op(&perf,
    op_code,            // 1=MMA, 2=Conv, 3=Attn, 4=Elemwise, 6=Softmax, etc.
    active_cycles,
    stall_cycles,
    flops);
```

### Memory Recording

```c
// Scratchpad access
tu_perf_mem_record_spad_access(&perf, is_write, words, bank_conflicts, stall_cycles);

// DRAM access
tu_perf_mem_record_dram_access(&perf, is_write, bytes, row_hit, stall_cycles);
```

### Power Configuration

```c
// Configure for 7nm technology node
tu_perf_power_config(&perf,
    0.2,    // pJ per MAC (7nm)
    0.1,    // pJ per SRAM read
    0.1,    // pJ per SRAM write
    5.0,    // pJ per DRAM access (HBM2)
    0.01,   // pJ per DMA byte
    0.0001); // pJ leakage per cycle

tu_perf_power_set_enabled(&perf, true);
```

### Reporting

```c
// Full formatted report
tu_perf_print_report(&perf);

// One-line compact summary
tu_perf_print_summary(&perf);

// Compute derived metrics
tu_perf_metrics_t m = tu_perf_compute_metrics(&perf);
printf("Utilization: %.1f%%, Throughput: %.3f TOPS\n",
       m.compute_utilization * 100.0f, m.mac_throughput_tops);
```

## Metric Definitions

| Metric | Formula | Description |
|--------|---------|-------------|
| `compute_utilization` | active_cycles / total_cycles | What fraction of time the compute array is doing useful work |
| `dma_bandwidth_gbps` | (read + write bytes) / seconds / 1e9 | Effective DMA throughput |
| `dram_bandwidth_gbps` | dram_bytes / seconds / 1e9 | Effective DRAM throughput |
| `mac_throughput_tops` | total_macs / seconds / 1e12 | Operations per second (trillions) |
| `mac_efficiency` | effective_macs / peak_macs | How close to theoretical peak |
| `spad_hit_rate` | 1 - bank_conflicts / total_accesses | Fraction of scratchpad accesses without bank conflict |
| `energy_per_mac_pj` | total_energy / total_macs | Average energy per MAC operation |
| `power_mw` | total_energy / seconds / 1e9 * 1e3 | Average power draw in milliwatts |

## Technology Node Energy Parameters

Approximate values for different technology nodes. Source: CACTI 7.0 + published literature.

| Technology | pJ/MAC | pJ/SRAM read | pJ/SRAM write | pJ/DRAM access | Notes |
|-----------|--------|-------------|---------------|----------------|-------|
| 45nm | 1.0 | 0.5 | 0.5 | 20.0 | Conservative baseline (default) |
| 28nm | 0.5 | 0.25 | 0.25 | 12.0 | Typical FPGA process |
| 16nm | 0.3 | 0.15 | 0.15 | 8.0 | TSMC 16FF |
| 7nm | 0.2 | 0.1 | 0.1 | 5.0 | HBM2 integration |
| 5nm | 0.12 | 0.06 | 0.06 | 3.5 | Advanced node |
| 3nm | 0.07 | 0.035 | 0.035 | 2.5 | Leading edge |

## Integration Points

The performance counter infrastructure is designed to be wired into:

1. **DMA Engine** (`dma_descriptor.c`): After each `tu_dma_execute_desc()`, record bytes, channel, transfer type, and cycles.

2. **Compute Engine** (`tu_cmodel.c`): After each `tu_mma()`, record MACs, tiles, precision, and dataflow.

3. **Memory System** (`tu_sram.c`): On each bank access, record reads/writes and bank conflicts.

4. **Command Queue** (`command_queue.c`): On command completion, compute stall cycles as `complete_time - issue_time - min_latency`.

5. **Specialized Engines**: Each engine (convolution, attention, softmax, etc.) records its own op counts through `tu_perf_compute_record_op()`.

## Example Output

```
┌──────────────────────────────────────────────────────────────┐
│          TU CModel — Performance Counter Report             │
├──────────────────────────────────────────────────────────────┤
│ Global                                                       │
│   Total cycles:          1234567                             │
│   Simulated time:       1234.567 µs                          │
│   Clock frequency:       1000 MHz                            │
├──────────────────────────────────┬───────────────────────────┤
│ DMA Engine                       │                           │
│   Read bytes:           262144   │  Write bytes:    131072   │
│   Internal:                   0  │  Stall cycles:     1234   │
│   Linear xfers:             12  │  Strided 2D:          0    │
│   Strided 3D:                0  │  Scatter:             0    │
│   Gather:                    0  │  BW:           0.318 GB/s  │
├──────────────────────────────────┼───────────────────────────┤
│ Compute Engine                   │                           │
│   Total MACs:          1048576   │  Utilization:     84.9 %  │
│   Total tiles:             256   │  Edge tiles:        12    │
│   Active cycles:       1048321   │  Stall:          186246   │
│   Idle cycles:               0   │  Bubbles:             0   │
│   Throughput:         0.849 TOPS │  Efficiency:      66.2 %  │
├──────────────────────────────────┼───────────────────────────┤
│ Memory Hierarchy                 │                           │
│   SPAD reads:            65536   │  SPAD writes:    32768   │
│   Bank conflicts:          128   │  SPAD stalls:      256   │
│   DRAM reads:               12   │  DRAM writes:        6   │
│   Row hits:                 15   │  Row misses:         3   │
│   DRAM BW:          0.318 GB/s   │  SPAD hit rate:   99.8 % │
├──────────────────────────────────┼───────────────────────────┤
│ Power / Energy                   │                           │
│   MAC energy:       1048576.0 pJ │  SRAM read:    32768.0 pJ│
│   SRAM write:        16384.0 pJ │  DRAM:           360.0 pJ │
│   DMA energy:        19660.8 pJ │  Leakage:       1234.6 pJ │
│   Total energy:    1118983.4 pJ │  Avg power:     906.496 mW│
│   Energy/MAC:         1.067 pJ/MAC│                          │
└──────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Wire into DMA engine** — Replace ad-hoc cycle counters in `dma_descriptor.c` with `tu_perf_dma_record_*` calls
2. **Wire into compute engine** — Replace `tu_state_t` counters with `tu_perf_counters_t`
3. **Wire into memory system** — Add `tu_perf_mem_record_*` calls in `tu_sram.c` on bank access
4. **Cycle-accurate model (P2.5)** — Extend with pipeline hazard tracking, NoC congestion
5. **VCD/FST trace export (P2.7)** — Write per-cycle counter values as VCD signals

## File Structure

```
tu_cmodel/perf/
├── performance_counters.h   — Counter types, API declarations
├── performance_counters.c   — Implementation (init, record, snapshot, report)
└── (future)
    ├── cycle_model.h        — P2.5: Pipeline hazard tracking
    ├── power_model.h        — P2.6: CACTI-based energy table
    └── event_trace.h        — P2.7: VCD/FST trace generation
```
