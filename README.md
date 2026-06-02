# tusim

Parametric tensor unit (TU) cycle-model simulator for pre-silicon architecture exploration and compiler co-design.

**tusim** is the TU sub-architecture model of a larger heterogeneous accelerator system (RISC-V host + SU/GPU + TU + special memories). It answers architecture questions *before* the hardware spec locks — sweep PE dimensions, memory hierarchy configurations, precision types, and dataflow policies; measure utilization, bandwidth, and stall breakdowns.

## Architecture

```
┌────────────────────────────────────────────┐
│  Compiler FE (not here)                    │
│  Splits work → Host / SU / TU              │
└──────────────┬─────────────────────────────┘
               │ TU commands
┌──────────────▼─────────────────────────────┐
│  tusim (this repo)                         │
│                                            │
│  tu_cmodel/                                │
│  ├── precision/   FP16/FP32/BF16/FP8/TF32/INT8 │
│  ├── compute/     MMA · Conv · Attention ·    │
│  │                Elementwise · Softmax ·     │
│  │                LayerNorm/RMSNorm · Pool    │
│  ├── dataflow/    WS · OS (pluggable)         │
│  ├── memory/      Banked SRAM · Hierarchy ·   │
│  │                DRAM model · Double-buffer   │
│  ├── dma/         Async descriptors · SG/DMA   │
│  ├── isa/         68-opcode ISA · Scheduler ·  │
│  │                Liveness allocator            │
│  ├── infra/       JSON config · Logging ·      │
│  │                VCD trace · Perf counters     │
│  └── perf/        Cycle model · Energy model    │
│                                            │
│  compiler/onnx_to_tu.py   ONNX→TU demo     │
│  tests/                   30+ test suites   │
│  docs/                    Design docs       │
└────────────────────────────────────────────┘
```

## Quick Start

```bash
# Build the TU library
make

# Run all tests
make test-full

# Run a quick smoke test
make test-quick
```

## Configuration

The TU is fully parametric via JSON config — no recompilation needed:

```json
{
  "tu": {
    "pe_rows": 32,
    "pe_cols": 32,
    "dataflow": "weight_stationary",
    "precision": { "input": "fp16", "accumulate": "fp32" },
    "memory": { "spad_w_kb": 256, "spad_a_kb": 128, "spad_o_kb": 128 }
  }
}
```

Load with `tu_init_from_file("config.json", ...)`.

## Exploration Workflow

```
1. Formulate:  "What SRAM size hides DRAM bandwidth stalls?"
2. Configure:  Create 2-3 JSON configs testing the extremes
3. Measure:    Run workload, collect perf counters
4. Document:   Write findings → docs/exploration/
```

## License

TBD
