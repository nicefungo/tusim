# TU Pluggable Dataflow System (Gap A4)

> **Priority:** P1 (High) | **Status:** Implemented | **Date:** 2026-05-30
>
> Makes the cmodel's dataflow strategy configurable: weight-stationary, output-stationary,
> row-stationary, and no-local-reuse, selectable per-operation via a plugin registry.

---

## 1. What This Is

The pluggable dataflow system (Gap A4) replaces the hard-coded weight-stationary tiling
loop with a registry of swappable dataflow plugins. Each plugin encapsulates:
- How data flows through the PE array (which operand is stationary, which streams)
- Tile-level execution strategy (loop order and parallelization)
- Cycle accounting (fill, compute, drain overheads)

The system supports these dataflows:

| Dataflow | ID | Description | Real-World Reference |
|----------|----|-----------|---------------------|
| **Weight-Stationary** (WS) | 0 | Weights preloaded in PEs, activations stream right, partial sums flow down | Google TPUv1, Gemmini |
| **Output-Stationary** (OS) | 1 | Output/accumulators stay in PEs, weights & activations stream in | TPUv2+, NVIDIA TensorCores |
| **Row-Stationary** (RS) | 2 | Each PE stores 1 row of W/A/psum; maximizes 1D reuse | Eyeriss v1 (reserved) |
| **No Local Reuse** (NLR) | 3 | Feed-forward; all data streams through; minimal PE storage | MAERI (reserved) |

## 2. Why This Was Chosen

**Gap Analysis Context:** The production redesign (docs/PRODUCTION_TU_REDESIGN.md)
identifies A4 as a P1 priority because:

1. **Universality across accelerators:** Every major systolic/GEMM accelerator
   (TPU, Gemmini, Eyeriss, MAERI, NVIDIA) supports at least 2 dataflows.
   A production cmodel must match this flexibility.

2. **Different workloads, different dataflows:** WS excels at GEMM with large K
   (weight reuse), OS excels at convolutions and attention (output stays put,
   avoids systolic fill), RS excels at depthwise convolutions. The compiler
   should select the optimal dataflow per layer.

3. **Design-space exploration:** The cmodel is a tool for architecture research.
   Pluggable dataflows enable "what if" comparisons: same workload, different
   dataflows → different cycle counts, bandwidth demands, utilization.

4. **Foundation for other features:** Convolution (O2), attention (O3), and
   software pipelining (E2) all require dataflow flexibility. This was the
   lowest-level dependency to unlock.

## 3. Architecture

### 3.1 Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  tu_mma(M, N, K, ...)                                       │
│    │                                                         │
│    ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  tu_dataflow_execute_mma()  ← tiling + dispatch         │ │
│  │    │                                                    │ │
│  │    │  for each tile: plugin->execute_tile(...)          │ │
│  │    ▼                                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│    │                                                         │
│    ├── tu_dataflow_plugin_t (WS)                              │
│    │     ws_execute_tile: W stationary, A streams, psum↓      │
│    │                                                         │
│    ├── tu_dataflow_plugin_t (OS)                              │
│    │     os_execute_tile: O stationary, W & A stream           │
│    │                                                         │
│    └── tu_dataflow_plugin_t (RS, NLR)  [future]               │
│                                                               │
│  Registered via: tu_dataflow_register() at init time           │
│  Selected via:   tu_set_dataflow(TU_DATAFLOW_OUTPUT_STATIONARY)│
└──────────────────────────────────────────────────────────────┘
```

### 3.2 File Layout

```
tu_cmodel/compute/dataflow/
├── dataflow_interface.h    # Plugin vtable: execute_tile, get_fill_cycles, etc.
├── dataflow_registry.h     # Static registry: register, lookup, count
├── dataflow_registry.c     # Registry implementation (array of 8 plugin slots)
├── dataflow_dispatcher.c   # tu_dataflow_execute_mma(): tiling + dispatch
├── weight_stationary.c     # WS plugin: original TinyTU dataflow, refactored
├── output_stationary.c     # OS plugin: TPUv2+/TensorCore-style vector dataflow
└── [row_stationary.c]      # RS plugin (reserved, P1.2)
```

### 3.3 Plugin Interface

Every dataflow plugin implements this vtable (from `dataflow_interface.h`):

```c
typedef struct tu_dataflow_plugin_t {
    const char *name;
    tu_dataflow_id_t id;

    void     (*init)(...);
    uint64_t (*execute_tile)(...);   // Execute one tile of MMA
    uint64_t (*get_fill_cycles)(...); // Pipeline fill overhead
    uint64_t (*get_drain_cycles)(...);// Pipeline drain overhead
    uint64_t (*get_compute_cycles)(...);

    uint64_t total_flops, total_tiles, total_cycles;
    void    *impl_data;
} tu_dataflow_plugin_t;
```

### 3.4 Integration with tu_cmodel.c

The existing `tu_mma()` is unchanged for backward compatibility. When
`TU_DATAFLOW_DISPATCH_VIA_PLUGIN` is enabled (default: 1), the function
constructs tensor descriptors and delegates to `tu_dataflow_execute_mma()`:

```c
// tu_cmodel.c: tu_mma()
#if TU_DATAFLOW_DISPATCH_VIA_PLUGIN
    tu_dataflow_tensor_t W_t = { .data = W, .rows = M, .cols = K, ... };
    tu_dataflow_tensor_t A_t = { .data = A, .rows = K, .cols = N, ... };
    tu_dataflow_tensor_t O_t = { .data = O, .rows = M, .cols = N, ... };
    uint64_t df_cycles = tu_dataflow_execute_mma(
        g_tu.dataflow, &W_t, &A_t, &O_t,
        pe_rows, pe_cols, pe_cols, TU_PE_PIPELINE_DEPTH);
#else
    // Legacy inline tiling (fallback)
#endif
```

The legacy path remains as a compile-time fallback.

## 4. Configuration

### 4.1 Compile-Time Default

```c
// tu_config.h
#define TU_DATAFLOW_MODE               TU_DATAFLOW_MODE_WS  // default dataflow
#define TU_DATAFLOW_DISPATCH_VIA_PLUGIN 1                    // use plugin system
```

### 4.2 Runtime Selection

```c
tu_init();

// Select output-stationary
tu_set_dataflow(TU_DATAFLOW_OUTPUT_STATIONARY);
printf("Active: %s\n", tu_get_dataflow_name());  // "output_stationary"

// Switch back to weight-stationary
tu_set_dataflow(TU_DATAFLOW_WEIGHT_STATIONARY);

// Fallback: if requested plugin not registered, falls to WS with warning
tu_set_dataflow(99);  // TU WARNING: dataflow id=99 not registered
```

### 4.3 Adding a New Dataflow

1. Create `my_dataflow.c` implementing the `tu_dataflow_plugin_t` vtable
2. Create a constructor: `tu_dataflow_plugin_t *tu_dataflow_my_create(void)`
3. Register it at init: add `tu_dataflow_register(tu_dataflow_my_create())` to `tu_init_with_config()`
4. Select it at runtime: `tu_set_dataflow(TU_DATAFLOW_MY_ID)`

## 5. Behavior

### 5.1 Weight-Stationary (WS)

```
        a[0]    a[1]    a[2]    a[3]
         │       │       │       │
         ▼       ▼       ▼       ▼
      ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
      │W[0,0]│→│W[0,1]│→│W[0,2]│→│W[0,3]│→  o[0]
      └─────┘ └─────┘ └─────┘ └─────┘
         │       │       │       │
         ▼       ▼       ▼       ▼
      ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
      │W[1,0]│→│W[1,1]│→│W[1,2]│→│W[1,3]│→  o[1]
      └─────┘ └─────┘ └─────┘ └─────┘
```

- **Cycle model:** `pipeline_depth × tile_n` fill + `k_count` compute + `pipeline_depth × tile_m` drain
- **Bandwidth:** Low — weights stationary in PEs after initial load
- **Best for:** Large-K GEMM, weight reuse scenarios

### 5.2 Output-Stationary (OS)

```
Each PE(i,j) holds O[i][j]
For each k:
  W[:,k] broadcast across row → PEs
  A[k,:] broadcast down column → PEs
  PE(i,j): O[i][j] += W[i,k] × A[k,j]
```

- **Cycle model:** 0 fill + `k_count` compute + 0 drain (+ W-fetch overhead)
- **Bandwidth:** Higher — W must be re-fetched each K-step
- **Best for:** Convolutions (im2col), attention, small-K operations
- **Note:** OS has no systolic pipeline latency — better latency for small tiles

### 5.3 Equivalence Guarantee

Both WS and OS produce **bit-identical FP32 accumulator results** for the same
inputs. The only difference is the cycle estimate, reflecting their different
microarchitectural characteristics. This is verified by `test_dataflow.c`.

## 6. Testing

**Test file:** `tests/test_dataflow.c`
**Run:** `make test-dataflow`

| Test | What it validates |
|------|-------------------|
| Registry API | Plugin registration, lookup by ID and name, count |
| WS identity 16×16 | Weight-stationary produces I when W=I, A=I |
| OS identity 16×16 | Output-stationary produces I when W=I, A=I |
| WS-vs-OS equivalence | Random 32×16 matrix — WS and OS produce bit-identical results |
| Dataflow switch | Run WS then switch to OS mid-session; results match |
| Edge tiles | 31×31×17 non-multiple-of-16 dimensions; WS-vs-OS match |

Test results: **6/6 pass**, all producing bit-exact matches between dataflows.

Existing tests (19 cmodel, 9 cmdq, 10 DMA, plus bf16, elementwise, normalization,
softmax, memory hierarchy, golden) all continue to pass unchanged.

## 7. Future Work

### RS (Row-Stationary)
The Eyeriss-style RS dataflow is the next implementation candidate.
Each PE stores one row of filter, activation, and partial sum.
Best for depthwise convolutions.

### NLR (No Local Reuse)
Feed-forward dataflow — all operands stream through, no PE-local storage.
Useful for ultra-low-power PEs and as a baseline for DSE.

### Compiler Integration
The compiler (`onnx_to_tu.py`) should eventually select the optimal
dataflow per layer based on a cost model (WS for large-K GEMM, OS for
convolutions/attention). Currently manual via `tu_set_dataflow()`.

### Cycle Model Calibration
WS fill/drain uses `pipeline_depth × dim`. This should be calibrated against
Gemmini RTL or SCALE-Sim for accuracy.

## 8. Related Gaps

- **A4 (this):** Pluggable dataflow — enables WS, OS, RS, NLR
- **O2 (Convolution):** Benefits from OS dataflow (no systolic fill for small tiles)
- **O3 (Attention):** FlashAttention-style tiling works best with OS or hybrid
- **E2 (Software pipelining):** DMA/compute overlap depends on dataflow-aware scheduling
- **P1.2 (Pluggable dataflow full):** RS and NLR implementations

---

**Author:** Hermes Agent heartbeat | **Gap ID:** A4 | **Priority:** P1 (High)
