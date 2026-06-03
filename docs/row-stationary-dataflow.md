# TU CModel — Row-Stationary Dataflow (A4)

> **Gap ID:** A4 (Dataflow Flexibility — Row-Stationary)
> **Priority:** P1 (High)
> **Date:** 2026-06-03
> **Heartbeat:** Midday shift (14:20), Cycle 1

---

## What Changed

The TU cmodel dataflow plugin system now supports all three major systolic dataflows: Weight-Stationary (WS), Output-Stationary (OS), and **Row-Stationary (RS)**. The RS dataflow, based on the Eyeriss architecture (MIT, ISCA 2016), completes the dataflow trifecta and enables dataflow-aware design space exploration.

### Architecture

Row-Stationary is a 3D-stationary dataflow where each PE stores:
- One **row** of filter weights (W)
- One **row** of activations (A)
- One **row** of partial sums (O)

This maximizes data reuse across all three dimensions simultaneously:

```
PE Array (RS dataflow, GEMM O[M][N] = W[M][K] × A[K][N]):

  PE(0,0): stores W[0][:] → computes O[0][0] = Σ W[0][k]·A[k][0]
  PE(0,1): stores W[0][:] → computes O[0][1] = Σ W[0][k]·A[k][1]
  PE(0,2): stores W[0][:] → computes O[0][2] = Σ W[0][k]·A[k][2]
  PE(1,0): stores W[1][:] → computes O[1][0] = Σ W[1][k]·A[k][0]
  ...

  1D Convolutional Reuse: W row stays stationary in PE row
  2D Spatial Reuse:     A values stream column-wise and are reused across PE rows
  Partial Sum Reuse:     Each O[i][j] accumulates in-place (no psum movement)
```

### Dataflow Comparison

| Property | WS (Weight-Stationary) | OS (Output-Stationary) | RS (Row-Stationary) |
|----------|----------------------|------------------------|---------------------|
| Stationary in PE | W[M][K] tensor | O[M][N] accumulators | 1 row each of W, A, O |
| W reuse | Per K-step (preloaded) | None (streams in) | Per N-step (row reuse) |
| A reuse | None (streams) | None (streams) | Per M-step (column reuse) |
| Pipeline fill | `pd × tile_n` cycles | 0 cycles (vector) | `(pd-1) × tile_n + 1` cycles |
| Pipeline drain | `pd × tile_m` cycles | 0 cycles | `(pd-1) × tile_m` cycles |
| Best for | Deep networks, fixed weights | Large models, varying ops | Conv layers, high data reuse |
| Bandwidth demand | Low (W stationary) | High (W+A streaming) | Medium (W row reuse) |
| Reference | Google TPUv1 | NVIDIA TensorCore, TPUv2+ | Eyeriss v1 (MIT) |

---

## Why This Matters

### Completing the Dataflow Trifecta

Having all three dataflows enables:

1. **Per-layer dataflow selection:** The compiler can choose WS for FC layers, OS for attention, and RS for convolution — maximizing utilization per layer type
2. **Architecture design space exploration:** Compare WS, OS, and RS for any workload without code changes
3. **Eyeriss-compatible research:** RS is the canonical dataflow for academic DNN accelerator research; adding it aligns our cmodel with the research community
4. **Energy-aware optimization:** RS maximizes data reuse, minimizing expensive SRAM/DRAM accesses — critical for edge inference

### Eyeriss Legacy

Eyeriss (Chen et al., ISCA 2016) demonstrated that RS achieves 1.4–2.5× energy efficiency over WS and OS for convolutional neural networks. By implementing RS, our cmodel can reproduce and extend these results for production workloads.

---

## How It Works

### Plugin Architecture

RS follows the same plugin interface as WS and OS:

```c
// tu_cmodel/compute/dataflow/dataflow_interface.h
typedef enum {
    TU_DATAFLOW_WEIGHT_STATIONARY = 0,
    TU_DATAFLOW_OUTPUT_STATIONARY = 1,
    TU_DATAFLOW_ROW_STATIONARY = 2,    // ← New
    TU_DATAFLOW_NO_LOCAL_REUSE = 3,
} tu_dataflow_id_t;
```

The RS plugin registers itself at init time:

```c
// tu_cmodel/tu_cmodel.c — tu_init_with_config()
tu_dataflow_register(tu_dataflow_rs_create());
```

### Loop Order

RS uses a **(m, n, k)** loop order, reflecting row-stationary computation:

```c
// RS: each output element computed completely before moving on
for (m in tile_m):
    for (n in tile_n):
        psum = 0
        for (k in tile_k):
            psum += W[m][k] * A[k][n]
        O[m][n] += psum
```

Compare with WS (systolic): **(k, m, n)** — A streams through column by column
Compare with OS (vector): **(m, n, k)** — same loop order, different reuse model

### Reuse Accounting

RS tracks weight reuse explicitly:

- **w_reuse_hits:** Number of times a weight element was reused from PE-local storage (not re-fetched from SRAM). For an m×n tile, this is `m × n × k - m × k`.
- **w_reuse_misses:** Number of weight fetches from SRAM: `m × k` per tile.

This accounting enables accurate energy modeling — SRAM reads are 10–100× more expensive than PE-local register reads.

### Cycle Model

| Phase | Formula | Rationale |
|-------|---------|-----------|
| Fill | `(pipeline_depth - 1) × tile_n + 1` | W rows are pre-distributed (no streaming fill for W), but A still streams through columns |
| Compute | `k_count` | All PEs MAC in parallel, one cycle per K-step |
| Drain | `(pipeline_depth - 1) × tile_m` | Drain remaining partial results through pipeline |
| **Total (per tile)** | Fill + k_count + Drain | |

---

## Configuration

The RS dataflow is selected via:

```c
// Via config header
#define TU_DATAFLOW_MODE  TU_DATAFLOW_MODE_RS  // Set to row-stationary

// Or at runtime
tu_set_dataflow(TU_DATAFLOW_ROW_STATIONARY);

// Or via config YAML
tu_config:
  pe_array:
    dataflow: "row_stationary"
```

The default remains WS for backward compatibility.

---

## Usage Example

```c
#include "tu_cmodel.h"

int main(void) {
    tu_init();

    // Select Row-Stationary dataflow
    tu_set_dataflow(TU_DATAFLOW_ROW_STATIONARY);

    // Load weight matrix W[128×64] as FP16
    tu_dma_load_w(w_data, 128 * 64 * 2);

    // Load activation matrix A[64×128] as FP16
    tu_dma_load_a(a_data, 64 * 128 * 2);

    // Execute GEMM: O[128][128] = W[128][64] × A[64][128]
    tu_mma(128, 128, 64, 0, 0, 0, false);

    // Read results
    tu_dma_store_o(o_data, 128 * 128 * 4);

    tu_print_stats();
    // Dataflow: row_stationary
    // MMA tiles: ...
    // Est. cycles: ... (RS-specific cycle model)

    return 0;
}
```

---

## Verification

### Test Suite: 9 tests, all passing (3 new for RS)

| Test | What It Verifies |
|------|-----------------|
| Registry API | All 3 plugins (WS/OS/RS) registered, lookup by ID and name |
| WS Identity | 16×16 identity matrix, WS dataflow |
| OS Identity | 16×16 identity matrix, OS dataflow |
| **RS Identity** | **16×16 identity matrix, RS dataflow** |
| **RS Registry Lookup** | **RS found by ID (TU_DATAFLOW_ROW_STATIONARY) and name ("row_stationary")** |
| **RS-vs-WS Equivalence** | **32×32 random matrix, RS result matches WS bit-exact** |
| WS-vs-OS Equivalence | 32×32 random matrix, WS and OS match bit-exact |
| Dataflow Switch | WS→OS mid-session, both produce identity |
| Edge Tiles | 31×31×17 non-multiple-of-16, WS and OS match |

### Run Tests

```bash
make test-dataflow    # 9/9 tests pass
make test-cmodel      # Core regression: 19/19 tests pass
```

---

## Files

| File | Change |
|------|--------|
| `tu_cmodel/compute/dataflow/row_stationary.c` | **New** — RS dataflow plugin (~230 LOC) |
| `tu_cmodel/compute/dataflow/dataflow_interface.h` | Existing — RS already enumerated |
| `tu_cmodel/tu_cmodel.c` | Modified — register RS plugin + forward declarations |
| `tests/test_dataflow.c` | Modified — 3 new RS tests |
| `Makefile` | Modified — row_stationary.o in TU_OBJS + build rule |
| `docs/row-stationary-dataflow.md` | **This document** |

---

## References

1. Chen et al., "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks," ISCA 2016
2. Chen et al., "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks," JSSC 2019
3. Kwon et al., "MAERI: Enabling Flexible Dataflow Mapping over DNN Accelerators via Reconfigurable Interconnects," ASPLOS 2018
4. TU CModel Dataflow Interface: `tu_cmodel/compute/dataflow/dataflow_interface.h`

---

## Next Steps

- **NLR Dataflow:** Implement No-Local-Reuse (feed-forward) as the fourth dataflow
- **Per-layer dataflow selection:** Compiler picks optimal dataflow per ONNX layer
- **Energy comparison:** Compare WS/OS/RS energy per workload using the power model
