# API Documentation System (Gap Q4)

> **Feature:** Doxygen API docs + auto-generated config reference + architecture diagrams
> **Gap ID:** Q4 — Documentation generation
> **Priority:** P2 (Medium)
> **Date:** 2026-06-03

---

## 1. Overview

The documentation system provides three documentation outputs:

1. **Doxygen API Documentation** — HTML API reference with class/struct/function graphs
2. **Auto-Generated Config Reference** — Markdown table of every config field with current values
3. **Architecture Diagrams** — Mermaid diagrams visualizing the cmodel's module structure

## 2. Doxygen API Docs

### 2.1 Configuration

The `Doxyfile` at the project root configures Doxygen to:

- Process all `.h` and `.c` files in `tu_cmodel/`
- Extract documentation from code comments (all documented entities)
- Generate call graphs, caller graphs, include graphs
- Output HTML with search, navigation, and source browsing
- Include `docs/*.md` as related pages
- Suppress warnings for undocumented internals

### 2.2 Generating

```bash
make docs-api
```

This creates `docs/api/html/index.html`. Open in any browser.

**Prerequisites:** `doxygen` and `graphviz` (for call/inheritance graphs)
```bash
sudo apt install doxygen graphviz
```

### 2.3 What Gets Documented

| Module | Headers | Description |
|--------|---------|-------------|
| `tu_core.h` | Core API | Top-level TU core lifecycle, execution, subsystem access |
| `tu_cluster.h` | Multi-core | Cluster creation, inter-core communication, SPMD |
| `tu_config.h` | Configuration | Compile-time `#define` constants + runtime config struct |
| `tu_sram.h` | SRAM | Banked SRAM with bandwidth modeling |
| `tu_precision.h` | Precision | FP16/BF16/TF32/FP8 type definitions + conversion |
| `dma_descriptor.h` | DMA | DMA descriptor types, channels, scatter/gather/multicast |
| `command_queue.h` | Command Queue | Command submission, dependency tracking, barriers |
| `infra/config.h` | Config Loader | JSON config file parsing, validation, runtime config |
| `infra/tu_debug.h` | Debug | State dump, deterministic replay, invariant assertions |
| `infra/logging.h` | Logging | Structured logging with severity levels and tracing |
| `memory/memory_hierarchy.h` | Memory | Multi-level memory (RegFile/SPAD/GBuf/DRAM) |
| `memory/dram_model.h` | DRAM | HBM2/HBM3/DDR4/DDR5 configurations |
| `memory/double_buffer.h` | Double Buffer | Ping-pong buffer management |
| `memory/address_generator.h` | Address Gen | Hardware address generation (strided/im2col) |
| `isa/tu_isa.h` | ISA | Instruction encoding/decoding, opcode definitions |
| `isa/tu_scheduler.h` | Scheduler | DAG-based instruction scheduling |
| `isa/tu_liveness.h` | Liveness | Graph-coloring register allocation |
| `compute/elementwise_pipeline.h` | Elementwise | Fused activation/arithmetic pipeline |
| `compute/normalization_engine.h` | Normalization | LayerNorm, RMSNorm, BatchNorm |
| `compute/softmax_engine.h` | Softmax | Online softmax with numerical stability |
| `compute/convolution_engine.h` | Convolution | Direct conv + im2col with stride/padding/dilation |
| `compute/attention_engine.h` | Attention | FlashAttention-style tiled attention |
| `compute/pooling_engine.h` | Pooling | MaxPool, AvgPool |
| `compute/pipeline_controller.h` | Pipeline | Software pipelining for DMA/compute overlap |
| `perf/performance_counters.h` | Counters | Per-op, per-component performance counters |
| `perf/event_trace.h` | Tracing | VCD waveform generation for GTKWave |
| `perf/power_model.h` | Power | CACTI-derived energy estimation |
| `tu_status.h` | Errors | 40 structured error codes, error handling macros |
| `sparsity/structured_2of4.h` | Sparsity | 2:4 structured sparsity with packed compression |

## 3. Auto-Generated Config Reference

### 3.1 Concept

The `tu_config_emit_docs()` function introspects a `tu_config_t` struct and generates a comprehensive Markdown reference table. This replaces hand-maintained config docs with a single source of truth — the config struct itself.

### 3.2 Generating

```bash
make config-docs
```

This compiles a small C program that calls `tu_config_emit_docs()` with the default config and redirects output to `docs/CONFIG_REFERENCE.md`.

### 3.3 Config Reference Structure

The generated document contains 10 sections:

1. **Compute Engine** — PE array dimensions, dataflow mode, MAC units
2. **Precision & Data Types** — FP16/BF16/TF32/FP8/INT8/INT4 enables, rounding
3. **Memory System** — SRAM sizes/banks, global buffer, DRAM type/bandwidth
4. **DMA Engine** — Bus width, channels, async mode, multicast
5. **ISA & Command Queue** — Instruction width, queue depth
6. **Multi-Core** — Core count, interconnect mode
7. **Performance Model** — Cycle accuracy, counters, trace
8. **Sparsity** — 2:4 structured, unstructured, metadata format
9. **Verification** — Golden reference, tolerance
10. **Derived Values** — Total SRAM, total MACs, peak ops/cycle, DMA bandwidth

### 3.4 Custom Configurations

To generate docs for a non-default config:

```c
#include "tu_cmodel/infra/config.h"

int main(void) {
    tu_config_t cfg;
    tu_config_default(&cfg);

    // Override for a specific deployment
    cfg.pe_rows = 128;
    cfg.pe_cols = 128;
    cfg.dram_type = 3; // HBM3
    cfg.dram_bandwidth_gbps = 900.0;

    tu_config_emit_docs(&cfg, stdout);
    return 0;
}
```

## 4. Architecture Diagrams

### 4.1 Mermaid Architecture Diagram

`docs/architecture-diagram.md` contains a Mermaid graph showing:

- All 5 architecture layers (from the redesign doc's layered architecture)
- Every implemented component with its gap ID reference
- Data flow paths: ONNX → Compiler → ISA → Command Queue → Core → Compute/Memory/DMA
- Cross-cutting concerns: Debug, Error Handling, Logging, Verification

The diagram is rendered natively on GitHub, GitLab, and most markdown viewers.

### 4.2 Keeping Diagrams Current

When adding new components:
1. Add the component to the appropriate layer in `architecture-diagram.md`
2. Tag it with the gap ID (e.g., `(C4)`)
3. If the component introduces new data flows, add edges

## 5. Documentation Inventory

As of this heartbeat, the `docs/` directory contains:

| Document | Description | Gap ID |
|----------|-------------|--------|
| `DESIGN.md` | Original TinyTU design doc | — |
| `TU_CMODEL.md` | Original cmodel documentation | — |
| `TU_ASM.md` | Original ASM documentation | — |
| `PRODUCTION_TU_REDESIGN.md` | Gap analysis + redesign proposal | — |
| `Implementing_a_TU_CModel_Roadmap.md` | Industry practices reference | — |
| `configurable-pe-and-banking.md` | PE array + banking configurability | A1, A2, A6 |
| `runtime-configuration.md` | JSON config loader | A1 |
| `parametric-pe-array.md` | Parametric PE sizing | A2 |
| `memory-hierarchy.md` | Multi-level memory | A3 |
| `TU_DATAFLOW.md` | Pluggable dataflow | A4 |
| `multicore-cluster.md` | Multi-core TU cluster | A5 |
| `TU_DOUBLE_BUFFER.md` | Double buffering | A7 |
| `int8-quantization.md` | INT8/INT4 quantization | D2 |
| `tf32-tensorfloat.md` | TF32 support | D3 |
| `fp8-implementation.md` | FP8 E4M3/E5M2 | D4 |
| `rounding-modes.md` | RNE/RTZ/Stochastic rounding | D6 |
| `bf16-subnormal.md` | BF16 + subnormal handling | D7 |
| `convolution-engine.md` | Convolution engine | O2 |
| `attention-engine.md` | Attention engine | O3 |
| `elementwise-pipeline.md` | Elementwise pipeline | O4 |
| `normalization-engine.md` | Normalization engine | O5 |
| `pooling-engine.md` | Pooling engine | O6 |
| `TU_SOFTMAX.md` | Softmax engine | O7 |
| `bandwidth-modeling.md` | SRAM bandwidth modeling | M2 |
| `dram-model.md` | DRAM model | M1 |
| `dma-descriptor-engine.md` | DMA descriptors | DM1, DM2 |
| `scatter-gather-dma.md` | Scatter/gather DMA | DM3 |
| `multicast-dma.md` | Multicast DMA | DM4 |
| `command-queue.md` | Command queue | E1 |
| `software-pipelining.md` | Software pipelining | E2 |
| `multi-context-execution.md` | Multi-context execution | E3 |
| `performance-counters.md` | Performance counters | E4 |
| `power-energy-model.md` | Power/energy model | E4 |
| `exception-handling.md` | Exception handling | E5 |
| `expanded-isa.md` | Expanded ISA | C1 |
| `compiler-scheduling-pass.md` | Compiler scheduling | C2 |
| `liveness-allocation.md` | Liveness allocator | C3 |
| `golden-reference-framework.md` | Golden reference | V1 |
| `regression-framework.md` | CI/regression | V3 |
| `differential-testing.md` | Differential testing | V6 |
| `comparative-benchmarking.md` | Benchmarking | V5 |
| `cycle-accurate-model.md` | Cycle-accurate model | P2.5 |
| `event-tracing-vcd.md` | VCD event tracing | P2.7 |
| `structured-sparsity.md` | Structured sparsity | P2.1 |
| `TU_LOGGING.md` | Structured logging | Q2 |
| `python-bindings.md` | Python bindings | I2 |
| `debug-observability.md` | Debug & observability | I3 |
| `api-documentation.md` | API docs system | Q4 |
| `architecture-diagram.md` | Architecture diagram | Q4 |
| `CONFIG_REFERENCE.md` | Auto-generated config reference | Q4 |

**Total:** 49 documentation files covering all 50+ gap items.

## 6. Integration with CI

Add to the CI pipeline:

```yaml
# In .github/workflows/ci.yml
- name: Generate config reference
  run: make config-docs

- name: Check docs are up to date
  run: |
    git diff --exit-code docs/CONFIG_REFERENCE.md || \
      echo "Warning: Config reference is stale. Run 'make config-docs'"
```

## 7. Design Decisions

1. **Doxygen for API docs** — Industry standard for C projects. Call/inheritance graphs provide architectural insight without separate tooling.

2. **Programmatic config docs** — Single source of truth. Adding a field to `tu_config_t` is automatically documented by `tu_config_emit_docs()`. No need to update a separate markdown file.

3. **Mermaid for diagrams** — Rendered natively on GitHub/GitLab. No external tooling needed. Text-based so it's version-controllable and reviewable in PRs.

4. **No Sphinx/MkDocs (yet)** — For a C library with ~50 header files, Doxygen provides sufficient API documentation. Sphinx would add value for a multi-language (C + Python) documentation site — deferred to P3 when the ecosystem matures.
