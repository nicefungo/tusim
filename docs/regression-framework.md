# TinyTU CI/Regression Framework

> **Gap:** V3 — No regression framework; manual test execution
> **Priority:** High P1
> **Heartbeat:** 2026-05-30 evening

## Overview

The TinyTU production cmodel lacked automated regression testing. All tests were run manually via individual `make test-*` targets, with no unified results reporting, no CI integration, and no automated quality gates. This made it impossible to confidently refactor code or accept contributions without manually verifying every commit.

The regression framework introduces:

1. **`tools/ci_runner.sh`** — Comprehensive CI runner with multiple modes
2. **`tools/test_report.py`** — Test result aggregator and HTML report generator
3. **Unified Makefile targets** — `make test`, `make test-quick`, `make test-random`
4. **GitHub Actions CI template** — `.github/workflows/ci.yml`

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CI Runner (ci_runner.sh)              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Build   │→ │  Unit Tests  │→ │  Integration     │  │
│  │  Phase   │  │  (14 suites) │  │  Tests           │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
│                                       ↓                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Coverage │← │  Extended    │← │  Valgrind        │  │
│  │ (opt)    │  │  Random (opt)│  │  (opt)           │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
│                       ↓                                  │
│              ┌────────────────────┐                     │
│              │  Summary Report    │                     │
│              │  (MD + JSON/HTML)  │                     │
│              └────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## CI Runner Modes

| Mode | Command | Runtime | Use Case |
|------|---------|---------|----------|
| **Quick** | `bash tools/ci_runner.sh --quick` | < 30s | Pre-commit hook |
| **Full** | `bash tools/ci_runner.sh` | 2-5 min | PR checks |
| **Nightly** | `bash tools/ci_runner.sh --random --valgrind` | 10-30 min | Scheduled nightly |
| **Coverage** | `bash tools/ci_runner.sh --coverage` | 3-5 min | Weekly coverage tracking |

## Test Suite Coverage

### Phase 2: Unit Tests (14 suites)

| # | Target | Coverage | Gap |
|---|--------|----------|-----|
| 1 | `test-cmodel` | FP16 roundtrip, MMA identity/GEMM/bias, PE dims, SRAM overflow, edge tiles | A1, A2 |
| 2 | `test-cmdq` | Command queue submit, ordering, dependencies, barriers | E1 |
| 3 | `test-dma` | DMA descriptor submission, async transfers, completion signals | DM1, DM2 |
| 4 | `test-dram` | DRAM models (ideal, HBM2, HBM2e, HBM3, DDR), bandwidth/latency | M1 |
| 5 | `test-isa` | ISA encoding/decoding round-trip, all opcodes, binary format | C1 |
| 6 | `test-golden` | FP32 golden reference comparison, 8 fixed configs + bulk random | V1, V6 |
| 7 | `test-elementwise` | 17 elementwise ops (ReLU, GELU, SiLU, sigmoid, tanh…), fused chains | O1, O4 |
| 8 | `test-bf16` | BF16 conversions, subnormal handling, BF16→FP16 pipeline, precision reg | D1, D3, D7 |
| 9 | `test-memhier` | Multi-level memory (RegFile, LocalSPAD, GlobalBuffer, DRAM), bank conflicts | A3 |
| 10 | `test-norm` | LayerNorm, RMSNorm, online statistics | O5 |
| 11 | `test-dataflow` | Pluggable dataflow (WS, OS), registry, dispatcher | A4 |
| 12 | `test-logging` | Structured logging, severity levels, component tags, VCD export | Q2 |
| 13 | `test-int-quant` | INT8/INT4 quantization, symmetric/asymmetric, per-channel | D2 |
| 14 | `test-conv` | Convolution engine (im2col, direct), stride/padding/dilation | O2 |

### Phase 3: Extended Random (nightly only)

- `test-random`: 5000 MMA FP16 + 2000 MMA BF16 + 1000 elementwise + 500 softmax random tests

### Phase 4: Integration Tests

- ASM interpreter smoke test
- ONNX compiler pipeline (GPT-block example)

## GitHub Actions CI

The `.github/workflows/ci.yml` defines four jobs:

1. **`quick`** — Triggered on every push/PR. Build + smoke test.
2. **`regression`** — Runs after `quick` passes. Full 14-suite regression.
3. **`nightly`** — On schedule or manual dispatch with `extended: true`. Random + Valgrind.
4. **`coverage`** — On push to `main` only. Coverage build + lcov report.

All jobs upload artifacts (logs, reports) for post-mortem analysis.

## Makefile Targets

```makefile
make test              # Full suite: all 14 unit targets + ASM
make test-quick        # Smoke test: cmodel + cmdq + dma + asm
make test-random       # Extended random: 5K MMA + 2K BF16 + 1K elem + 500 softmax
make test-full         # ONNX → compile → run pipeline
```

## Test Report Generator

`tools/test_report.py` parses test output logs and produces:

- **Text mode** (default): Per-test pass/fail with error stats
- **JSON mode** (`--json`): Structured JSON for programmatic consumption
- **HTML mode** (`--html`): Dark-themed HTML report with progress bars

## Configuration

All test parameters are configurable via `tu_config.h`:

```c
#define TU_VERIFY_RANDOM_ITERS        1000
#define TU_VERIFY_ERROR_TOLERANCE     1e-05
```

## Verification

```bash
# Quick smoke (pre-commit)
make test-quick

# Full regression
make test

# Nightly extended
make test-random

# Automated CI
bash tools/ci_runner.sh
bash tools/ci_runner.sh --quick --coverage

# Generate HTML report
python3 tools/test_report.py --html > build/ci_reports/report.html
```

## What This Changes

- **Before:** Manual `make test-cmodel` + `make test-cmdq` + ... individually, no aggregation
- **After:** Single `make test` runs everything, CI automation, HTML reports, cross-platform CI template

## Next Steps

- V2: Add performance regression tracking (compare cycle counts against baseline)
- V5: Add MLPerf Tiny benchmark integration
- Coverage targets: > 95% line, > 90% branch
