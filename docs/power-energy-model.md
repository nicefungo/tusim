# TU Power & Energy Model (Gap E4)

> **Status:** Implemented  
> **Gap IDs:** E4 (Power/Energy Model), P2.6 (CACTI-based energy estimation)  
> **Files:** `tu_cmodel/perf/power_model.h`, `tu_cmodel/perf/power_model.c`, `tests/test_power_model.c`  
> **Tests:** 20/20 passing

## Overview

The TU Power/Energy Model provides per-component energy estimation across six technology nodes, with CACTI-derived SRAM energy parameters and published silicon data for MAC and DRAM energy. It models the full energy breakdown—compute, memory hierarchy (RegFile → SPAD → GlobalBuffer), DRAM, DMA bus, clock distribution, and leakage—enabling architects to evaluate energy efficiency tradeoffs at design time.

**Why this matters:** Without power modeling, architectural exploration is blind. An accelerator that doubles throughput but quadruples power is a bad trade. The power model closes the loop: architects can now answer *"How many pJ per MAC?"* rather than just *"How many TOPS?"*.

### Design Principles

1. **Configurable technology nodes.** Six presets (45nm → 3nm) with distinct energy tables. Switching nodes mid-simulation updates energy rates without resetting counters.
2. **Per-hierarchy-level granularity.** RegFile, SPAD, and GlobalBuffer each have different energy costs, reflecting the real silicon tradeoff: larger memories cost more energy per access.
3. **Snapshot/diff for interval profiling.** Take before/after snapshots of any code region, diff them, and get energy consumed in that interval.
4. **DRAM page hit/miss distinction.** Row buffer hits vs. activates are modeled separately, capturing the 2–3× energy penalty of page misses.
5. **Zero-overhead when disabled.** Setting `enabled = false` makes all recording calls no-ops.

## Technology Node Energy Tables

Each node provides distinct energy parameters calibrated from CACTI 7.0 (SRAM) and published silicon data (MAC, DRAM). All energies in picojoules (pJ) unless noted.

### Comparative Table

| Parameter | 45nm | 28nm | 16nm | 7nm | 5nm | 3nm |
|-----------|------|------|------|-----|-----|-----|
| **FP16 MAC** | 1.00 pJ | 0.65 pJ | 0.40 pJ | 0.20 pJ | 0.14 pJ | 0.10 pJ |
| **FP8 MAC** | 0.25 pJ | 0.16 pJ | 0.10 pJ | 0.05 pJ | 0.035 pJ | 0.025 pJ |
| **INT8 MAC** | 0.20 pJ | 0.13 pJ | 0.08 pJ | 0.04 pJ | 0.028 pJ | 0.020 pJ |
| **INT4 MAC** | 0.12 pJ | 0.08 pJ | 0.05 pJ | 0.025 pJ | 0.017 pJ | 0.012 pJ |
| **RegFile read** | 0.020 pJ | 0.013 pJ | 0.008 pJ | 0.004 pJ | 0.003 pJ | 0.002 pJ |
| **SPAD read** | 0.50 pJ | 0.33 pJ | 0.20 pJ | 0.10 pJ | 0.07 pJ | 0.05 pJ |
| **GlobalBuf read** | 1.20 pJ | 0.78 pJ | 0.48 pJ | 0.24 pJ | 0.17 pJ | 0.12 pJ |
| **DRAM read (64B)** | 640 pJ | 420 pJ | 260 pJ | 130 pJ | 90 pJ | 64 pJ |
| **DRAM activate** | 1200 pJ | 780 pJ | 480 pJ | 240 pJ | 168 pJ | 120 pJ |
| **DMA per byte** | 0.050 pJ | 0.033 pJ | 0.020 pJ | 0.010 pJ | 0.007 pJ | 0.005 pJ |
| **Clock tree/cycle** | 0.050 pJ | 0.033 pJ | 0.020 pJ | 0.010 pJ | 0.007 pJ | 0.005 pJ |
| **Vdd** | 1.00 V | 0.95 V | 0.85 V | 0.75 V | 0.70 V | 0.65 V |
| **Frequency** | 0.8 GHz | 1.2 GHz | 1.5 GHz | 2.0 GHz | 2.5 GHz | 3.0 GHz |
| **MAC area** | 800 µm² | 520 µm² | 320 µm² | 160 µm² | 112 µm² | 80 µm² |
| **SRAM area/KB** | 12,000 µm² | 7,800 µm² | 4,800 µm² | 2,400 µm² | 1,680 µm² | 1,200 µm² |
| **Static power** | 15 mW/mm² | 10 mW/mm² | 8 mW/mm² | 5 mW/mm² | 3.5 mW/mm² | 2.5 mW/mm² |

### Energy Scaling Trends

```
45nm → 28nm: 0.65× (planar shrink)
28nm → 16nm: 0.40× (FinFET transition, large gain)
16nm → 7nm:  0.20× (EUV, second FinFET generation)
7nm  → 5nm:  0.14× (continued scaling)
5nm  → 3nm:  0.10× (GAA nanosheet)
```

### Key Insight: Memory Hierarchy Energy

The SRAM energy hierarchy reflects physical reality:

```
RegFile (per-PE, 256B):   ~2–20 fJ/word  ← cheapest
SPAD (per-core, 64KB):    ~50–500 fJ/word
GlobalBuf (shared, 1MB):  ~120–1200 fJ/word  ← 2.4× SPAD
DRAM (off-chip):          ~64–640 pJ/access  ← 1000× SPAD per bit!
```

This hierarchy incentivizes data locality: accessing data from RegFile instead of DRAM saves 3–4 orders of magnitude in energy. The cmodel's compiler should exploit this by keeping hot data in SRAM and minimizing DRAM round-trips.

## API Reference

### Technology Node Lookup

```c
// Get energy table for a node
const tu_tech_node_energy_t* tu_power_get_tech_node(TU_TECH_NODE_7NM);

// String → node
tu_tech_node_t tu_power_tech_node_from_string("7nm");   // → TU_TECH_NODE_7NM
tu_tech_node_t tu_power_tech_node_from_string("5");     // → TU_TECH_NODE_5NM

// Node → string
const char* name = tu_power_tech_node_name(TU_TECH_NODE_7NM);  // → "7nm"
```

### Lifecycle

```c
tu_power_model_t pm;

// Initialize with 7nm technology, 2 GHz clock
tu_power_model_init(&pm, TU_TECH_NODE_7NM, 2000.0);

// Change technology mid-simulation
tu_power_model_set_tech_node(&pm, TU_TECH_NODE_5NM);

// Reset energy counters (preserves config)
tu_power_model_reset(&pm);

// Disable to eliminate overhead
tu_power_model_set_enabled(&pm, false);
```

### Energy Recording

```c
// MAC operations (precision-aware)
tu_power_record_mac(&pm, 1000, 0);  // 0=FP16, 1=BF16, 2=INT8, 3=INT4, 4=FP8

// Memory hierarchy (per-word access)
tu_power_record_regfile_access(&pm, false, 500);   // 500 reads
tu_power_record_spad_access(&pm, true, 200);       // 200 writes
tu_power_record_global_buf_access(&pm, false, 50); // 50 reads

// DRAM with page hit/miss
tu_power_record_dram_access(&pm, false, 128, true);   // 128B read, page hit
tu_power_record_dram_access(&pm, false, 64, false);   // 64B read, page miss → activate penalty

// DMA bus
tu_power_record_dma(&pm, 1024);  // 1KB transferred

// Clock advancement (leakage + clock tree energy)
tu_power_tick(&pm, 100);
```

### Derived Metrics

```c
// Compute total energy from components
tu_power_compute_total(&pm);
printf("Total: %.1f pJ\n", pm.energy_total_pj);

// Average power
double mw = tu_power_get_avg_power_mw(&pm);  // → milliwatts

// Energy efficiency
double pj_per_mac = tu_power_get_energy_per_mac(&pm);  // → pJ/MAC

// Energy breakdown (fractions sum to 1.0)
tu_power_breakdown_t bd = tu_power_get_breakdown(&pm);
printf("MAC: %.1f%%, SPAD: %.1f%%, DRAM: %.1f%%\n",
       bd.fraction_mac * 100, bd.fraction_spad * 100, bd.fraction_dram * 100);

// Chip area estimate
double area = tu_power_estimate_area(&pm, 32, 32, 65536, 1048576);
// → mm² for 32×32 PE, 64KB SPAD, 1MB GBUF
```

### Interval Profiling

```c
// Before workload
tu_power_snapshot_t snap1 = tu_power_snapshot(&pm);

// ... run workload ...

// After workload
tu_power_snapshot_t snap2 = tu_power_snapshot(&pm);

// Energy consumed in this interval
tu_power_model_t interval = tu_power_diff(&snap1, &snap2);
printf("Interval energy: %.1f pJ\n", interval.energy_total_pj);
```

### Reporting

```c
// Full formatted report
tu_power_print_report(&pm);

// Compact one-liner
tu_power_print_summary(&pm);
// → [power] 7nm 2000.0 MHz | 123456.7 pJ total | 45.678 mW avg | 0.204 pJ/MAC | 5000 MACs
```

### Configuration Integration

```c
tu_config_t cfg;
tu_config_default(&cfg);
cfg.pe_rows = 64;
cfg.pe_cols = 64;
cfg.counters_enabled = true;

tu_power_model_t pm;
cfg.power_tech_node = TU_POWER_CONFIG_TECH_16NM;
cfg.power_clock_freq_mhz = 750.0;             // 0 keeps legacy clock heuristic
tu_power_model_from_config(&pm, &cfg);
// → Explicit process and clock override historical size/BW heuristics
// → Estimates chip area from PE count + SRAM sizes
```

## How It Changes CModel Behavior

**Before (E4):** Energy counters existed in `tu_perf_counters_t` but with hardcoded 45nm-like parameters (`pj_per_mac = 1.0`) and no technology node configurability. No separate power model module existed.

**After:** A dedicated `tu_power_model_t` with:
- 6 technology node presets with distinct energy parameters
- Per-hierarchy-level memory energy (RegFile ≠ SPAD ≠ GlobalBuf)
- DRAM page hit/miss distinction
- Clock tree + leakage modeling
- Chip area estimation
- Interval profiling via snapshot/diff
- Config-driven auto-selection of tech node

## Configuration

The power model accepts explicit `power.tech_node` (`45nm`, `28nm`, `16nm`, `7nm`, `5nm`, `3nm`) and `power.clock_freq_mhz`. `auto`/0 preserves the historical heuristic:

| PE Array Size | Selected Node | Rationale |
|---------------|---------------|-----------|
| ≤ 16×16 | 7nm | Mobile/edge inference |
| 17–127 | 7nm (default) | Conservative |
| ≥ 128×128 | 5nm | Datacenter/training |
| BW > 500 GB/s | 2 GHz clock | High-bandwidth config |

Prefer the canonical JSON/YAML fields for reproducible studies. `tu_power_model_set_tech_node()` remains available for direct API experiments. Explicit clock selection changes time conversion and leakage duration but does not scale voltage-dependent dynamic energy; it must not be described as a DVFS model. See `exploration/power-process-clock-assumptions.md`.

## Calibration Sources

| Component | Source | Accuracy |
|-----------|--------|----------|
| FP16 MAC @ 7nm | NVIDIA A100 (312 TFLOPS @ 400W) | ±15% |
| FP16 MAC @ 5nm | NVIDIA H100 (989 TFLOPS @ 700W) | ±20% |
| SRAM energy | CACTI 7.0 models at each node | ±25% |
| DRAM energy | JEDEC LPDDR5/HBM2e datasheets | ±20% |
| Leakage | ITRS/IRDS projections | ±30% |
| Area | Published die photos + scaling | ±20% |

**Important caveat:** These are first-order estimates for architectural exploration. Real silicon energy depends on circuit design, voltage/frequency scaling, temperature, process variation, and workload characteristics. The model is intentionally simple—it provides directional accuracy (is A better than B?) rather than absolute precision.

## Testing

20 tests in `tests/test_power_model.c`:

| # | Test | What It Verifies |
|---|------|-----------------|
| 1 | tech_node_lookup | All 6 nodes exist with valid parameters |
| 2 | tech_node_from_string | String → enum conversion, all formats |
| 3 | power_model_init | Default initialization correctness |
| 4 | power_model_tech_nodes | All nodes have consistent energy params |
| 5 | power_mac_recording | MAC energy by precision, accumulation |
| 6 | power_memory_recording | Per-level memory energy accounting |
| 7 | power_dram_recording | DRAM page hit vs. miss distinction |
| 8 | power_dma_recording | DMA bus energy |
| 9 | power_tick | Clock tree + leakage per cycle |
| 10 | power_total | Total energy = sum of components |
| 11 | power_avg_power | Average power in mW |
| 12 | power_breakdown | Fractions sum to 1.0 |
| 13 | power_area_estimate | Chip area from PE + SRAM |
| 14 | power_snapshot_diff | Interval profiling correctness |
| 15 | power_reset | Reset preserves config |
| 16 | power_tech_switch | Mid-run node switch |
| 17 | power_disable | Disabled model records nothing |
| 18 | power_energy_scaling | 7nm < 45nm, 3nm < 7nm |
| 19 | power_config_integration | Config → power model mapping |
| 20 | power_numeric_stability | Zero values, large values, edge cases |

## Future Work

- **P2.9 Integration:** Feed power metrics into comparative benchmarking framework for TOPS/W Pareto analysis
- **CACTI/Accelergy integration:** Replace hardcoded tables with a live CACTI call for arbitrary SRAM configurations
- **Thermal modeling:** Extend from energy to temperature (θJA-based junction temperature estimation)
- **Voltage/frequency scaling:** Explicit clock assumptions now exist, but real DVFS still requires voltage-dependent dynamic/leakage energy, feasible operating points, transition overhead, and thermal/power-delivery constraints
- **Multi-core power:** Per-core energy accounting with inter-core communication energy

## References

- Horowitz, M. "1.1 Computing's energy problem (and what we can do about it)." ISSCC 2014.
- Stillmaker, A. and Baas, B. "Scaling equations for the accurate prediction of CMOS device performance from 180 nm to 7 nm." Integration, 2017.
- CACTI 7.0 — HP Labs, "An integrated cache and memory access time, cycle time, area, leakage, and dynamic power model."
- NVIDIA A100 Tensor Core GPU Architecture. Whitepaper, 2020.
- NVIDIA H100 Tensor Core GPU Architecture. Whitepaper, 2022.
- Jouppi, N. et al. "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings." ISCA 2023.
