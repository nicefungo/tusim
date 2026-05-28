# Production TU CModel — Redesign Document

> **Status:** Design proposal  
> **Version:** 1.0  
> **Date:** 2026-05-28  
> **Target:** Evolve TinyTU cmodel from toy (16×16, 256 KB, FP16) to production-grade parametric simulator

---

## Table of Contents

1. [Current Architecture Review](#1-current-architecture-review)
2. [Production-Grade Accelerator References](#2-production-grade-accelerator-references)
3. [Detailed Gap Analysis](#3-detailed-gap-analysis)
4. [Proposed Scalable Architecture](#4-proposed-scalable-architecture)
5. [Component Catalog with Interfaces](#5-component-catalog-with-interfaces)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Verification Methodology](#7-verification-methodology)

---

## 1. Current Architecture Review

### 1.1 What Exists

The TinyTU cmodel is a functional simulator for a 16×16 weight-stationary systolic array with:

| Component | Specification | File |
|-----------|--------------|------|
| **Systolic Array** | 16×16 weight-stationary, FP16×FP16→FP32 MAC, tiled 16³ | `tu_cmodel.c:205-314` |
| **SRAM** | 256 KB total: W(128 KB) + A(64 KB) + O(64 KB), single-ported byte arrays | `tu_cmodel.c:71-73` |
| **DMA Engine** | 3 channels (W-load, A-load, O-load/store), 256-bit AXI bus, synchronous memcpy | `tu_cmodel.c:156-199` |
| **FP Conversion** | IEEE 754 fp32↔fp16, round-to-nearest-even, subnormal support | `tu_cmodel.c:28-106` |
| **Stats** | 5 counters: dma_bytes, mma_calls, mma_tiles, mma_flops, estimated_cycles | `tu_cmodel.c:76-80` |
| **ASM Interpreter** | Text parser, weight embedding, host buffer bindings, 6 instructions | `tu_asm.c:1-271` |
| **Compiler** | ONNX→TU ASM→C, shape inference (14 ops), K-tiling, bump allocation | `onnx_to_tu.py:1-794` |
| **Tests** | 4 unit tests (FP16, identity, GEMM, bias) + ASM smoke test | `tests/test_cmodel.c`, `tests/test_asm.c` |

### 1.2 Dataflow Model

```
Host DRAM --DMA--> [W-Buffer 128KB] ──┐
                                       ├──> Systolic Array (16×16) --> [O-Buffer 64KB] --DMA--> Host
Host DRAM --DMA--> [A-Buffer  64KB] ──┘
                 Weight-stationary: W preloaded in PEs, A streams right, partial sums flow down
```

**MMA semantics:** `O[N][M] += W[N][K] × A[K][M]`  
Tiled internally: `for mi,ni,ki in tiles: compute 16×16×16 GEMM`

### 1.3 Key Design Decisions (Current State)

1. **Functional, not cycle-accurate:** Computes correct results with a simplified cycle estimate (`16 fill + K compute per tile`)
2. **Single monolithic file:** All SRAM, DMA, and MMA in `tu_cmodel.c` (271 lines of implementation)
3. **Hard-coded dimensions:** `TU_PE_ROWS=16`, `TU_PE_COLS=16`, SRAM sizes as `#define` constants
4. **Single global state:** `g_tu` — no multi-instance support
5. **Synchronous DMA:** All transfers are blocking memcpy (no command queue, no DMA/compute overlap)
6. **Single dataflow:** Weight-stationary only; no support for output-stationary or row-stationary
7. **Single precision:** FP16 in, FP32 accumulate, no INT8/INT4/FP8/BF16
8. **Single operation:** Only GEMM; no convolution, attention, elementwise, or reduction support
9. **No microarchitectural detail:** No memory banking, no double buffering, no pipeline hazards, no NoC model

### 1.4 Strengths

- **Clean, readable ~600 lines** across all core files
- **Correct FP implementation** with IEEE 754 handling
- **Working end-to-end pipeline:** ONNX → compiler → cmodel → results
- **TU ASM provides a clean ISA boundary** between compiler and hardware
- **Good documentation:** DESIGN.md, TU_CMODEL.md, TU_ASM.md

### 1.5 Weaknesses (Toy → Production Gap)

- Zero configurability (hard-coded constants everywhere)
- No support for real hardware features: sparsity, multi-level memory, data types beyond FP16
- No performance modeling fidelity (no bank conflicts, no bandwidth contention, no pipeline stalls)
- No verification against golden reference models
- Single-instance, no multi-TU-cluster support
- No support for convolution, attention, or elementwise fusion
- Compiler limited to Gemm/MatMul only (134 ops are stubs)

---

## 2. Production-Grade Accelerator References

This section establishes what "production-grade" means by surveying real systolic/GEMM accelerators. Each reference provides specific design patterns we should adopt.

### 2.1 Google TPU Family (v1 → v5)

| Property | TPUv1 | TPUv2/v3 | TPUv4 | TPUv5 |
|----------|-------|----------|-------|-------|
| **Array size** | 256×256 (65K MACs) | 128×128 (2 cores) | 128×128 (4 cores) | 128×128 |
| **Dataflow** | Weight-stationary (systolic) | Output-stationary (vector) | Output-stationary | Output-stationary |
| **Precision** | INT8 in, INT32 out | BF16 in, FP32 accumulate | BF16/INT8 | BF16/INT8/FP8 |
| **On-chip memory** | 28 MB unified SRAM | 32 GB HBM (v3) | 32 GB HBM2e | HBM |
| **Sparsity** | None | None | Structured (v4) | Structured + unstructured |
| **Key design pattern** | **Software-managed memory** — compiler controls all data placement; no caches; DMA is explicit | **Systolic → Vector transition** — v2+ use vector units with transpose/reduce permuters; more flexible than pure systolic | **Multi-core scaling** — SPMD across cores, each with own HBM partition; ICI interconnect for cross-chip | **SparseCore** — dedicated sparsity engine separate from dense compute |

**Takeaways for our cmodel:**
- Software-managed scratchpad memory hierarchy (no hardware caches)
- Explicit DMA with double-buffering support
- Support for both systolic (weight-stationary) and vector (output-stationary) dataflows as configuration options
- Multi-core topology with inter-core communication primitives
- Structured sparsity support (2:4, block-sparse)
- BF16 and INT8 data type support

### 2.2 NVIDIA TensorCore Evolution (Volta → Hopper)

| Property | Volta (V100) | Ampere (A100) | Hopper (H100) |
|----------|-------------|---------------|---------------|
| **Array per SM** | 8 TensorCores | 4 TensorCores | 4 TensorCores |
| **Operation** | FP16×FP16+FP16→FP16/FP32 | FP16/BF16/TF32/INT8/INT4/INT1, structured sparsity (2:4) | FP8 (E4M3/E5M2), FP16, BF16, TF32, INT8, 2× sparsity |
| **MMA shape** | m16n16k16 (FP16) | m16n8k16 (FP16), m16n8k32 (INT8) | m16n8k16 (FP16), m16n8k32 (INT8) |
| **Memory hierarchy** | L1/SMEM → L2 → HBM2 | L1/SMEM (164KB) → L2 (40MB) → HBM2e | L1/SMEM (256KB) → L2 (50MB) → HBM3 |
| **Async execution** | Warp-level MMA | Async copy (TMA) + MMA overlap | TMA, thread block clusters, distributed shared memory |
| **Key innovation** | **Warp-level matrix instructions** — tensor ops exposed as PTX `mma.sync`; shared memory used as register file extension | **Structured sparsity** — 2:4 pruning gives 2× throughput via MMA instruction variant; **TF32** — 19-bit mantissa for DL training | **FP8** — two formats (E4M3 for forward, E5M2 for gradients); **TMA** — async tensor memory accelerator with hardware address generation; **DSM** — distributed shared memory across clusters |

**Takeaways for our cmodel:**
- MMA should be configurable to specific tile shapes (m16n8k16, m16n8k32, etc.)
- Multi-precision: FP16, BF16, TF32, INT8, INT4, FP8
- Structured sparsity (2:4) as a compile-time configuration
- Asynchronous DMA descriptors with completion signals
- Memory hierarchy: register file → scratchpad → L2 → DRAM
- Hardware address generation for strided/block transfers

### 2.3 Gemmini (Berkeley)

| Property | Gemmini |
|----------|---------|
| **Array size** | Parameterized: `DIM`×`DIM` |
| **Dataflow** | Weight-stationary (systolic) and output-stationary modes |
| **Precision** | INT8 in, INT32 accumulate; configurable scaling |
| **Memory system** | 3-level: scratchpad (SPAD) → accumulator → DRAM; double-buffered |
| **Key innovation** | **RISC-V RoCC accelerator** — integrated into Rocket/BOOM core via coprocessor interface; **im2col for convolution** — hardware im2col with padding, stride, dilation support; **fused activation** — ReLU, ReLU6, prelu applied on accumulator output; **pipelining** — DMA, preload, compute, store can overlap via tile-level pipeline |

**Takeaways for our cmodel:**
- **RISC-V integration model:** Memory-mapped control registers (like Gemmini's RoCC interface) → our cmodel should provide an MMIO register map
- **Hardware im2col** for convolution (reduce compiler complexity, improve utilization)
- **Fused activation functions** in the accumulator path (ReLU, GELU, SiLU, tanh)
- **Double-buffered scratchpads** for DMA/compute overlap
- **Tile-level software pipelining** — compiler schedules DMA for tile N+1 while array computes tile N

### 2.4 Eyeriss v1/v2 (MIT)

| Property | Eyeriss v1 (2016) | Eyeriss v2 (2019) |
|----------|-------------------|-------------------|
| **Array size** | 12×14 = 168 PEs | 12×14 = 168 PEs |
| **Dataflow** | **Row-stationary (RS)** — maximizes data reuse across all dimensions | **Hierarchical mesh** — 2D mesh of PEs with configurable dataflow |
| **Memory** | 108 KB global buffer + 0.5 KB/PE local storage | 192 KB global buffer + hierarchical NoC |
| **Key innovation** | **RS dataflow** — each PE stores one row of filter, activation, and partial sum; maximizes CONV reuse (1D convolution within PE, 2D across array) | **Flexible NoC** — configurable multicast and unicast patterns; adapts dataflow per layer (RS for CONV, OS for FC, WS for depthwise) |
| **Sparsity** | Gating clock for zero-skipping in activations and weights | Compressed sparse column (CSC) for weights, compressed sparse row (CSR) for activations |

**Takeaways for our cmodel:**
- **Configurable dataflow:** The cmodel should support weight-stationary, output-stationary, and row-stationary as compile-time or runtime configurations
- **Sparsity-aware execution:** Zero-skipping (gating) and compressed-sparse representations
- **Adaptive dataflow selection:** Different layers (CONV vs. FC vs. depthwise) benefit from different dataflows; the compiler should select optimal dataflow per layer
- **Network-on-Chip model:** Multicast/unicast patterns for weight/activation distribution

### 2.5 MAERI (Georgia Tech)

| Property | MAERI |
|----------|-------|
| **Array size** | Configurable virtual array, physical 16×16 |
| **Dataflow** | **Fully flexible** — reconfigurable NoC supports WS, OS, RS, NLR |
| **Key innovation** | **Augmented reduction tree** — configurable adder tree with bypass paths; **flexible NoC** — multi-hop configurable switches; **virtual mapping** — maps logical arrays onto physical PEs with configurable interconnect; **sparsity** — zero-value-aware forwarding |

**Takeaways for our cmodel:**
- **Virtual-to-physical mapping:** Our cmodel should separate logical array dimensions from physical PE array dimensions
- **Configurable reduction topology:** Accumulator paths should be configurable (systolic, reduction tree, or hybrid)
- **NoC flexibility:** Different data distribution patterns for different dataflow modes

### 2.6 Cross-Cutting Patterns Summary

Across all six references, the following patterns recur:

| Pattern | Adopters | Implementation |
|---------|----------|---------------|
| **Multi-level memory hierarchy** | All | RegFile → Local Scratchpad → Global Buffer → DRAM |
| **Double/triple buffering** | TPU, Gemmini, Eyeriss | Enable DMA/compute overlap |
| **Configurable dataflow** | Gemmini, MAERI, Eyeriss v2 | WS, OS, RS via NoC config |
| **Multi-precision** | NVIDIA, TPUv5 | INT8/INT4 → BF16/FP16 → FP8 |
| **Structured sparsity** | NVIDIA Ampere+, TPUv4 | 2:4, block-sparse, zero-gating |
| **Hardware im2col / convolution** | Gemmini | Direct convolution with address generation |
| **Fused activation/normalization** | Gemmini, NVIDIA | Elementwise ops on accumulator output |
| **Software-managed memory** | TPU, Gemmini | No caches; compiler controls placement |
| **Asynchronous command execution** | All | Command queue, DMA descriptors, completion signals |
| **Multi-core scaling** | TPUv2+, NVIDIA clusters | SPMD, inter-core interconnect |

---

## 3. Detailed Gap Analysis

Every difference between the current TinyTU (toy) and what a production cmodel must support.

### 3.1 Architectural Gaps

| # | Category | Current TinyTU | Production Target | Severity | Priority |
|---|----------|---------------|-------------------|----------|----------|
| A1 | **Configurability** | Hard-coded: 16×16 PE, 256KB SRAM, 256-bit bus | Parameterized: arbitrary PE array dims, SRAM sizes, bus widths; YAML/JSON config file at cmodel startup | Critical | P0 |
| A2 | **Systolic array dimensions** | Fixed 16×16 only | Configurable `PE_ROWS`×`PE_COLS` (e.g., 32×32, 64×64, 128×128) with compile-time and runtime configuration | Critical | P0 |
| A3 | **SRAM hierarchy** | 3 flat byte buffers (W, A, O), no banking | Multi-bank SRAM with configurable banking factor, multi-level: RegFile (per-PE), LocalSPAD (per-cluster), GlobalBuffer (shared), DRAM interface | High | P0 |
| A4 | **Dataflow flexibility** | Weight-stationary only | Pluggable dataflow: WS, OS (output-stationary), RS (row-stationary), NLR (no local reuse); selectable per-operation | High | P1 |
| A5 | **Multi-instance / multi-core** | Single global `g_tu` instance | Multi-core TU with inter-core communication; each core has own state, memory, and DMA channels | High | P1 |
| A6 | **Memory banking** | None — flat byte arrays | Multi-bank memories with configurable bank count, stride-based interleaving, bank conflict detection | High | P0 |
| A7 | **Double buffering** | None — single buffer per region | Double/triple-buffered scratchpads for DMA/compute overlap; ping-pong buffer management | High | P1 |

### 3.2 Precision & Data Types

| # | Category | Current TinyTU | Production Target | Severity | Priority |
|---|----------|---------------|-------------------|----------|----------|
| D1 | **FP16 only** | FP16 in, FP32 accumulate, FP16 out | Multi-precision: INT4, INT8, INT16, FP8 (E4M3/E5M2), FP16, BF16, TF32, FP32; configurable per-tensor | Critical | P1 |
| D2 | **No integer support** | No INT8/INT32 path | Dedicated INT path with zero-point quantization, per-channel/per-tensor scale; INT8×INT8→INT32 | High | P1 |
| D3 | **No BF16/TF32** | Only IEEE FP16 | BF16 (1-8-7) for training, TF32 (1-8-10) for mixed precision; critical for ML training workloads | High | P1 |
| D4 | **No FP8** | — | FP8 E4M3 (forward) and E5M2 (gradient) per NVIDIA Hopper / OCP spec; emerging standard | Medium | P2 |
| D5 | **No block floating point** | — | Block FP with shared exponent per group of N elements (e.g., Microsoft MSFP, Flexpoint) | Low | P3 |
| D6 | **Rounding modes** | Round-to-nearest-even only | Configurable: round-nearest-even, round-toward-zero, stochastic rounding (for training) | Medium | P2 |
| D7 | **Subnormal handling** | Full subnormal support (FP16) | Configurable: full subnormal or flush-to-zero; flush-to-zero reduces HW cost; important for matching real silicon behavior | Medium | P1 |

### 3.3 Operation Coverage

| # | Category | Current TinyTU | Production Target | Severity | Priority |
|---|----------|---------------|-------------------|----------|----------|
| O1 | **Only GEMM** | `tu_mma()` only (matrix multiply) | Full op catalog: Conv2D/3D, DepthwiseConv, TransposedConv, Attention (FlashAttention-style), Reduction (sum, max, mean), Elementwise (add, mul, relu, gelu, silu, tanh, sigmoid, exp), Normalization (layernorm, batchnorm, rmsnorm), Pooling (max, avg), Reshape/Transpose/Permute, Scatter/Gather | Critical | P0 |
| O2 | **No convolution** | Not supported | Hardware im2col or direct convolution with stride/padding/dilation support; address generation in DMA; separate Conv ISA | Critical | P1 |
| O3 | **No attention** | Not supported | FlashAttention-style tiling: Q×K^T + Softmax + ×V fused in on-chip memory; critical for transformer inference | Critical | P1 |
| O4 | **No elementwise** | Not supported | Fused elementwise ops in accumulator path; avoids round-trip to DRAM; critical for activation functions, residual adds | High | P1 |
| O5 | **No normalization** | Not supported | LayerNorm/RMSNorm with online statistics computation; fused with preceding GEMM | High | P1 |
| O6 | **No pooling** | Not supported | MaxPool, AvgPool with stride; useful for early vision model layers | Low | P2 |
| O7 | **No softmax** | Not supported | Online softmax (max-subtract + exp + sum + divide); required for attention | High | P1 |

### 3.4 Memory System

| # | Category | Current TinyTU | Production Target | Severity | Priority |
|---|----------|---------------|-------------------|----------|----------|
| M1 | **No DRAM model** | DMA is synchronous memcpy | Multi-level memory: RegFile → SPAD → L2/Global → DRAM (HBM/DDR); bandwidth and latency model per level | Critical | P0 |
| M2 | **No bandwidth modeling** | ceil(bytes/32) — too simple | Per-bank bandwidth, arbitration delay, bank conflict stall cycles, NoC congestion, DRAM row buffer hit/miss | High | P0 |
| M3 | **No address generation** | Simple byte offsets | Hardware address generation for strided/block transfers, im2col patterns, tiled DMA descriptors with stride/dim metadata | High | P1 |
| M4 | **No memory protection** | Bounds check with abort() | Configurable memory protection: access violation handling, MMU-like page tables for multi-tenant isolation | Medium | P3 |
| M5 | **No compression** | Raw FP16 storage | Compressed sparse formats (CSR, CSC, block-CSR), Huffman/run-length for weights | Medium | P2 |
| M6 | **No cache/scratchpad distinction** | Flat scratchpad only | Configurable: scratchpad (software-managed) vs cache (hardware-managed) per region; critical for exploring architecture tradeoffs | Low | P3 |

### 3.5 DMA & Data Movement

| # | Category | Current TinyTU | Production Target | Severity | Priority |
|---|----------|---------------|-------------------|----------|----------|
| DM1 | **Synchronous DMA only** | All DMA is blocking memcpy | Asynchronous DMA with descriptor queues, completion interrupts/signals, DMA/compute overlap via double buffering | Critical | P0 |
| DM2 | **No DMA descriptors** | Raw pointer + offset + size | DMA descriptor: src addr, dst addr, size, stride (for 2D), next-desc pointer (chained), completion signal; complex transfer patterns | High | P0 |
| DM3 | **No scatter/gather** | Contiguous transfers only | Scatter/gather DMA for sparse weight loading, indirect addressing for embedding lookups | Medium | P1 |
| DM4 | **No broadcast/multicast** | Point-to-point only | Multicast DMA for weight broadcast to multiple cores, activation broadcast to PE array columns | Medium | P2 |
| DM5 | **No priority/QoS** | FIFO execution | Multiple DMA channels with priority levels; critical for real-time inference | Low | P3 |

### 3.6 Execution Model

| # | Category | Current TinyTU | Production Target | Severity | Priority |
|---|----------|---------------|-------------------|----------|----------|
| E1 | **No command queue** | Direct function calls | Command queue with submission, ordering, dependency tracking; barriers between commands; out-of-order execution where safe | Critical | P0 |
| E2 | **No pipelining** | Sequential DMA→Compute→DMA | Tile-level software pipelining: DMA tile N+1 while computing tile N; requires double buffering | Critical | P1 |
| E3 | **No multi-context** | Single global state | Multiple execution contexts (inference, training, multi-tenant); context switching with state save/restore | Medium | P2 |
| E4 | **No power modeling** | None | Per-component energy counters: MAC energy, SRAM read/write energy, DMA energy, leakage; configurable technology node energy parameters | Medium | P2 |
| E5 | **No exception handling** | abort() on error | Configurable error handling: precise exceptions, recoverable faults, error injection for testing | Medium | P2 |

### 3.7 Compiler & ISA

| # | Category | Current TinyTU | Production Target | Severity | Priority |
|---|----------|---------------|-------------------|----------|----------|
| C1 | **Limited ISA** | 6 instructions (LOAD_W/A/O, MMA, SYNC, STORE_O) | Rich ISA: ~30+ instructions covering all operations; variable-length encoding; binary format alongside text | Critical | P0 |
| C2 | **No scheduling** | WYSIWYG: ASM order = execution order | Compiler scheduling pass: reorder independent operations, hoist DMA, insert barriers, software pipeline | High | P1 |
| C3 | **No register allocation** | Bump allocator for SRAM | Liveness analysis, graph-coloring allocation for scratchpad, spill/fill insertion | High | P1 |
| C4 | **ONNX only via stubs** | 134 ops are stubs | Full ONNX op coverage on accelerator path: Conv, Attention, LayerNorm, Softmax, etc. | High | P1 |
| C5 | **Single backend** | Emits C code | Multi-backend: C functional, SystemC/TLM cycle-accurate, Verilog RTL, gem5 integration, FPGA bitstream hints | Medium | P2 |
| C6 | **No MLIR integration** | Custom Python frontend | MLIR dialect for TU: `tu.mma`, `tu.conv`, `tu.elementwise`; leverage MLIR passes for tiling, fusion, bufferization | Medium | P2 |
| C7 | **No auto-tiling** | Manual K-tiling only | Auto-tiling for all dimensions (M, N, K, batch, spatial) with cost model; search over tile sizes, dataflows, loop orders | High | P2 |

### 3.8 Verification & Testing

| # | Category | Current TinyTU | Production Target | Severity | Priority |
|---|----------|---------------|-------------------|----------|----------|
| V1 | **No golden reference** | Cmodel is the only implementation | Dual-path verification: reference (NumPy/PyTorch) vs cmodel vs RTL; bit-exact matching for fixed configs | Critical | P0 |
| V2 | **4 unit tests** | Identity, small GEMM, bias, FP16 roundtrip | Comprehensive test suite: randomized tensor tests, coverage of all op types, all data types, all corner cases (edge tiles, zero dims, large dims), fuzzing | Critical | P0 |
| V3 | **No regression framework** | Manual test execution | CI pipeline: build → unit tests → integration tests → performance regression → coverage report | High | P1 |
| V4 | **No performance validation** | Cycle estimates have unknown accuracy | Calibrate cycle model against known implementations (e.g., Gemmini RTL, MAERI RTL); report accuracy bounds | Medium | P2 |
| V5 | **No comparative benchmarking** | None | Standard benchmark suite: MLPerf Tiny, Transformer layers, ResNet-50 blocks; compare against Gemmini, SCALE-Sim, Timeloop | High | P2 |
| V6 | **No random/differential testing** | Hand-written tests only | Random tensor generation, compare against PyTorch FP32 reference, report max/mean error; differential testing across dataflows | High | P1 |

### 3.9 Code Quality & Infrastructure

| # | Category | Current TinyTU | Production Target | Severity | Priority |
|---|----------|---------------|-------------------|----------|----------|
| Q1 | **Monolithic code** | ~600 lines in 2 .c files | Modular architecture: ~20+ files with clear abstraction layers; header-only configuration generation | High | P0 |
| Q2 | **No logging/tracing** | fprintf to stderr | Structured logging with severity levels; execution trace in VCD/FST format; instruction-level trace for debugging | Medium | P1 |
| Q3 | **No plugin system** | Everything compiled together | Plugin architecture: dataflow plugins, precision plugins, memory plugins; selectable at configure time | Medium | P1 |
| Q4 | **No documentation generation** | 3 hand-written markdown files | Doxygen/Sphinx API docs; auto-generated configuration reference; architecture diagrams as code | Medium | P2 |
| Q5 | **C only** | C code, Python compiler | C++ with templates for parameterization; Python bindings via pybind11; modeling infrastructure | Medium | P1 |

---

## 4. Proposed Scalable Architecture

### 4.1 Design Principles

1. **Parametric Everything:** Every hardware dimension, data type, and behavior is a compile-time or runtime parameter — no hard-coded constants
2. **Pluggable Components:** Each subsystem (dataflow, memory, DMA, ISA) has a defined interface; implementations can be swapped without changing the rest
3. **Layered Abstraction:** Clear separation between functional model, cycle model, and microarchitectural model
4. **Hardware-First Semantics:** The cmodel reflects real hardware behavior, not idealized software; includes timing, hazards, and constraints
5. **Verification Built In:** Every component includes self-checking assertions and golden-reference comparison hooks

### 4.2 High-Level Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: Compiler & Tooling                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐  │
│  │ MLIR TU      │  │ Tiling &     │  │ Schedule &     │  │ Code Generation   │  │
│  │ Dialect      │  │ Mapping Opt  │  │ Pipeline Opt   │  │ (C/Verilog/Binary)│  │
│  └──────────────┘  └──────────────┘  └───────────────┘  └────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 4: TU ISA & Command Interface                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  TU ISA (text + binary)  │  Command Queue  │  MMIO Register Map          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3: TU Core (Functional + Cycle Model)                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Systolic     │  │ Vector / SIMD│  │ Elementwise  │  │ Normalization     │  │
│  │ Array Engine │  │ Engine       │  │ Pipeline     │  │ Engine            │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Dataflow Controller (configurable: WS, OS, RS, NLR)                      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2: Memory Subsystem                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Register File│  │ Local        │  │ Global       │  │ DRAM Interface     │  │
│  │ (per-PE)     │  │ Scratchpad   │  │ Buffer / L2  │  │ (HBM/DDR/ideal)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  DMA Descriptor Engine (async, multi-channel, scatter/gather, broadcast)  │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 1: Precision & Numerics                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ FP8/FP16/    │  │ INT4/INT8/   │  │ Conversion & │  │ Stochastic        │  │
│  │ BF16/TF32    │  │ INT16/INT32  │  │ Rounding      │  │ Rounding          │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 0: Infrastructure                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Performance  │  │ Power /      │  │ Logging /     │  │ Configuration     │  │
│  │ Counters     │  │ Energy Model │  │ Tracing       │  │ System            │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Parametric Design

All hardware properties are driven by a configuration specification:

```yaml
# tu_config.yaml — Example production configuration
tu_config:
  # Core parameters
  num_cores: 4                    # Multi-core
  pe_array:
    rows: 32                      # PE array height
    cols: 32                      # PE array width
    dataflow: "weight_stationary" # ws | os | rs | nlr
    mac_pipeline_depth: 4         # Pipeline stages per MAC
    
  # Precision
  precision:
    input_dtype: "fp16"           # fp8_e4m3 | fp8_e5m2 | fp16 | bf16 | tf32 | int8 | int4
    weight_dtype: "fp16"
    accumulate_dtype: "fp32"      # fp32 | int32 | fp16
    output_dtype: "fp16"
    rounding_mode: "rne"          # rne | rtz | stochastic
    
  # Memory hierarchy
  memory:
    regfile_per_pe: 256           # bytes per PE
    local_spad:
      size: 65536                 # 64 KB per core
      banks: 8
      read_ports: 2
      write_ports: 1
      double_buffered: true
    global_buffer:
      size: 1048576               # 1 MB shared
      banks: 16
    dram:
      type: "ideal"               # ideal | hbm2 | hbm3 | ddr5
      bandwidth_gbps: 900         # GB/s
      latency_ns: 100             # Access latency
      
  # DMA
  dma:
    num_channels: 4
    bus_width_bits: 512
    descriptor_queue_depth: 64
    support_scatter_gather: true
    
  # Operations
  operations:
    enable_conv: true
    enable_attention: true
    enable_elementwise: true
    enable_normalization: true
    enable_softmax: true
    
  # Sparsity
  sparsity:
    structured_2_4: true          # NVIDIA-style 2:4 structured sparsity
    block_sparse: false
    zero_gating: true             # Eyeriss-style clock gating
    
  # Performance modeling
  perf_model:
    cycle_accurate: false         # false = functional, true = cycle-accurate
    model_bank_conflicts: true
    model_noc_congestion: false
    power_model: "cacti"          # none | cacti | simple
```

This configuration file is the single source of truth. At build time, it generates:
- `tu_config.h` — C/C++ header with all constants
- `tu_isa_defs.h` — ISA encoding derived from config
- `tu_cmodel` binary with baked-in parameters
- Template-generated Python bindings

### 4.4 Abstraction Layers

```
┌──────────────────────────────────────────────────────────────────┐
│                     TU Core Interface                            │
│  initialize(config) → core_handle                                │
│  submit_command(core_handle, command) → cmd_id                   │
│  wait_completion(core_handle, cmd_id) → status                   │
│  read_counter(core_handle, counter_id) → value                   │
│  dump_state(core_handle) → state_snapshot                        │
└──────────────────────────────────────────────────────────────────┘

The TU Core is the top-level component. Below it, each subsystem 
implements a defined interface:

┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│  ComputeEngine    │  │  MemorySystem     │  │  DMAEngine        │
│  Interface        │  │  Interface        │  │  Interface        │
├───────────────────┤  ├───────────────────┤  ├───────────────────┤
│ • execute(op)     │  │ • read(addr,sz)  │  │ • submit_descriptor│
│ • set_dataflow()  │  │ • write(addr,sz) │  │ • poll_channel()  │
│ • get_util()      │  │ • allocate(sz)   │  │ • get_bandwidth() │
│ • flush_pipeline()│  │ • get_bank_conf() │  │ • configure()    │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

### 4.5 File Structure

```
tu_cmodel/
├── tu_core.h              # Top-level TU core API
├── tu_core.c              # Core orchestration
├── tu_config.h.in         # Configuration template (generated)
├── tu_types.h             # Common types: fp8_t, bf16_t, tu_addr_t, etc.
├── tu_status.h            # Error codes, status types
│
├── compute/
│   ├── compute_engine.h        # Compute engine interface
│   ├── compute_engine.c        # Dispatcher to dataflow-specific engines
│   ├── systolic_array.h        # Systolic array (weight-stationary)
│   ├── systolic_array.c
│   ├── vector_engine.h         # Output-stationary / SIMD
│   ├── vector_engine.c
│   ├── elementwise_pipeline.h  # Fused elementwise ops
│   ├── elementwise_pipeline.c
│   ├── normalization_engine.h  # LayerNorm, RMSNorm, BatchNorm
│   ├── normalization_engine.c
│   ├── softmax_engine.h        # Online softmax
│   ├── softmax_engine.c
│   ├── convolution_engine.h    # Direct conv + im2col
│   ├── convolution_engine.c
│   └── dataflow/
│       ├── dataflow_interface.h    # Pluggable dataflow API
│       ├── weight_stationary.c
│       ├── output_stationary.c
│       ├── row_stationary.c
│       └── dataflow_registry.h
│
├── memory/
│   ├── memory_system.h         # Multi-level memory interface
│   ├── memory_system.c
│   ├── sram_bank.h             # Banked SRAM model
│   ├── sram_bank.c
│   ├── scratchpad.h            # Software-managed scratchpad
│   ├── scratchpad.c
│   ├── global_buffer.h         # Shared L2 buffer
│   ├── global_buffer.c
│   ├── dram_model.h            # DRAM interface (ideal/HBM/DDR)
│   ├── dram_model.c
│   ├── address_generator.h     # Hardware address generation
│   ├── address_generator.c
│   └── allocator.h             # Memory allocator (bump, buddy, liveness-based)
│
├── dma/
│   ├── dma_engine.h            # Async DMA engine
│   ├── dma_engine.c
│   ├── dma_descriptor.h        # DMA descriptor types
│   ├── dma_channel.h           # Per-channel state machine
│   ├── scatter_gather.h        # Scatter/gather support
│   └── broadcast.h             # Multicast DMA
│
├── isa/
│   ├── tu_isa.h                # Instruction set definitions
│   ├── tu_isa_encoder.c        # ISA binary encoding
│   ├── tu_isa_decoder.c        # ISA binary decoding
│   ├── tu_isa_text.h           # Textual ASM format
│   ├── tu_isa_text.c
│   ├── command_queue.h         # Command submission & ordering
│   ├── command_queue.c
│   └── mmio_regs.h             # MMIO register map definition
│
├── precision/
│   ├── fp8.h                   # FP8 E4M3 and E5M2
│   ├── fp8.c
│   ├── fp16.h                  # IEEE FP16 (existing, refactored)
│   ├── fp16.c
│   ├── bf16.h                  # Brain Float 16
│   ├── bf16.c
│   ├── tf32.h                  # TensorFloat-32
│   ├── tf32.c
│   ├── int_quant.h             # INT4/INT8 quantization ops
│   ├── int_quant.c
│   ├── rounding.h              # Rounding mode strategies
│   ├── rounding.c
│   └── stochastic_rounding.h   # Stochastic rounding
│
├── sparsity/
│   ├── sparsity_controller.h   # Sparsity abstraction
│   ├── structured_2of4.h       # 2:4 structured
│   ├── structured_2of4.c
│   ├── zero_gating.h           # Zero-value clock gating
│   └── compressed_storage.h    # CSR/CSC/block-CSR
│
├── perf/
│   ├── performance_counters.h  # Counter infrastructure
│   ├── performance_counters.c
│   ├── cycle_model.h           # Cycle-accurate timing
│   ├── cycle_model.c
│   ├── power_model.h           # Energy estimation
│   ├── power_model.c
│   ├── event_trace.h           # VCD/FST trace generation
│   └── event_trace.c
│
├── infra/
│   ├── logging.h               # Structured logging
│   ├── logging.c
│   ├── config.h                # Configuration loader (YAML/JSON)
│   ├── config.c
│   ├── json_reader.h           # Minimal JSON parser (no external deps)
│   └── random_tensor.h         # Random tensor generation for testing
│
├── asm/
│   ├── tu_asm_interpreter.h    # ASM interpreter (refactored from tu_asm.c)
│   ├── tu_asm_interpreter.c
│   ├── tu_asm_assembler.c      # Text → binary
│   └── tu_asm_disassembler.c   # Binary → text
│
├── bindings/
│   ├── python/
│   │   ├── tu_pybind.cpp        # pybind11 bindings
│   │   ├── tu_config.py         # Python config helpers
│   │   └── tu_visualizer.py     # Visualization utilities
│   └── systemc/
│       └── tu_tlm.cpp           # SystemC/TLM wrapper
│
└── tests/
    ├── test_framework.h         # Unified test harness
    ├── test_compute.c           # Compute engine tests
    ├── test_memory.c            # Memory system tests
    ├── test_dma.c               # DMA engine tests
    ├── test_isa.c               # ISA encode/decode tests
    ├── test_precision.c         # All precision types
    ├── test_convolution.c       # Convolution tests
    ├── test_attention.c         # Attention tests
    ├── test_elementwise.c       # Elementwise tests
    ├── test_sparsity.c          # Sparsity tests
    ├── test_integration.c       # End-to-end tests
    ├── test_random.c            # Random/differential tests
    ├── test_fuzz.c              # Fuzzing harness
    └── golden/
        ├── generate_reference.py  # PyTorch golden reference generator
        └── reference_data/        # Pre-computed golden outputs
```

---

## 5. Component Catalog with Interfaces

### 5.1 Compute Engine Interface

```c
// tu_cmodel/compute/compute_engine.h

typedef enum {
    TU_DATAFLOW_WEIGHT_STATIONARY,
    TU_DATAFLOW_OUTPUT_STATIONARY,
    TU_DATAFLOW_ROW_STATIONARY,
    TU_DATAFLOW_NO_LOCAL_REUSE,
} tu_dataflow_t;

typedef enum {
    TU_OP_MMA_FP16,        // FP16 GEMM
    TU_OP_MMA_BF16,        // BF16 GEMM
    TU_OP_MMA_INT8,        // INT8 GEMM
    TU_OP_MMA_FP8,         // FP8 GEMM
    TU_OP_CONV2D,          // 2D convolution
    TU_OP_CONV3D,          // 3D convolution
    TU_OP_DEPTHWISE_CONV,  // Depthwise convolution
    TU_OP_ATTENTION,       // Q×K^T + Softmax + ×V
    TU_OP_ELEMENTWISE,     // Elementwise binary/unary
    TU_OP_REDUCTION,       // Sum, max, mean
    TU_OP_SOFTMAX,         // Online softmax
    TU_OP_LAYER_NORM,      // Layer normalization
    TU_OP_RMS_NORM,        // RMS normalization
    TU_OP_POOL_MAX,        // Max pooling
    TU_OP_POOL_AVG,        // Average pooling
    TU_OP_TRANSPOSE,       // Data layout transform
} tu_opcode_t;

// Tensor descriptor for compute operations
typedef struct {
    void       *data;          // Pointer to data in memory system
    tu_addr_t   base_addr;     // Physical address in memory hierarchy
    uint32_t    dims[4];       // Up to 4D tensor shape
    uint32_t    strides[4];    // Strides for each dimension
    tu_dtype_t  dtype;         // Data type
    uint32_t    zero_point;    // Quantization zero-point (INT types)
    float       scale;         // Quantization scale
} tu_tensor_desc_t;

// MMA operation descriptor
typedef struct {
    tu_tensor_desc_t  a;       // Left operand
    tu_tensor_desc_t  b;       // Right operand
    tu_tensor_desc_t  c;       // Output / accumulator
    tu_tensor_desc_t  bias;    // Optional bias
    uint32_t          m, n, k; // Dimensions
    bool              transpose_a;
    bool              transpose_b;
    bool              has_bias;
    float             alpha;
    float             beta;
    tu_activation_t   activation; // Fused activation (NONE, RELU, GELU, etc.)
} tu_mma_desc_t;

// Convolution descriptor
typedef struct {
    tu_tensor_desc_t  input;       // NCHW or NHWC
    tu_tensor_desc_t  weight;      // KCRS format
    tu_tensor_desc_t  output;
    tu_tensor_desc_t  bias;
    uint32_t          pad[4];      // Padding: top, bottom, left, right
    uint32_t          stride[2];   // Stride: H, W
    uint32_t          dilation[2]; // Dilation: H, W
    uint32_t          groups;
    tu_activation_t   activation;
} tu_conv_desc_t;

// Attention descriptor
typedef struct {
    tu_tensor_desc_t  q, k, v;     // Query, Key, Value
    tu_tensor_desc_t  output;
    tu_tensor_desc_t  mask;        // Optional attention mask
    uint32_t          batch_size;
    uint32_t          num_heads;
    uint32_t          seq_len_q;
    uint32_t          seq_len_kv;
    uint32_t          head_dim;
    float             softmax_scale;
    bool              causal;      // Causal masking
} tu_attention_desc_t;

// Core compute engine API
typedef struct tu_compute_engine_t tu_compute_engine_t;

// Create a compute engine instance
tu_compute_engine_t* tu_compute_create(const tu_config_t *config,
                                        tu_memory_system_t *mem,
                                        tu_dma_engine_t *dma);

// Destroy
void tu_compute_destroy(tu_compute_engine_t *engine);

// Configure dataflow mode for subsequent operations
tu_status_t tu_compute_set_dataflow(tu_compute_engine_t *engine,
                                     tu_dataflow_t dataflow);

// Execute an operation
tu_status_t tu_compute_execute(tu_compute_engine_t *engine,
                                tu_opcode_t op,
                                const void *op_desc,
                                tu_completion_signal_t *signal);

// Flush pipeline (drain all in-flight operations)
tu_status_t tu_compute_flush(tu_compute_engine_t *engine);

// Get utilization statistics
float tu_compute_get_utilization(const tu_compute_engine_t *engine);

// Get performance counters
void tu_compute_get_counters(const tu_compute_engine_t *engine,
                              tu_compute_counters_t *counters);
```

### 5.2 Memory System Interface

```c
// tu_cmodel/memory/memory_system.h

typedef enum {
    TU_MEM_REGFILE,       // Per-PE register file
    TU_MEM_LOCAL_SPAD,    // Local scratchpad (per-core)
    TU_MEM_GLOBAL_BUFFER, // Global buffer / L2
    TU_MEM_DRAM,          // Off-chip DRAM
} tu_mem_level_t;

typedef struct {
    void       *data;
    tu_addr_t   addr;
    uint32_t    size;
    tu_mem_level_t level;
    uint32_t    bank;       // Which bank (for banked memories)
    uint32_t    port;       // Which read/write port
} tu_mem_access_t;

typedef struct tu_memory_system_t tu_memory_system_t;

// Create memory system from config
tu_memory_system_t* tu_memory_create(const tu_config_t *config);

void tu_memory_destroy(tu_memory_system_t *mem);

// Allocate a region at specified level
tu_status_t tu_memory_allocate(tu_memory_system_t *mem,
                                tu_mem_level_t level,
                                uint32_t size_bytes,
                                uint32_t alignment,
                                tu_addr_t *addr_out);

// Deallocate
tu_status_t tu_memory_deallocate(tu_memory_system_t *mem,
                                  tu_mem_level_t level,
                                  tu_addr_t addr);

// Read/write with cycle accounting
tu_status_t tu_memory_read(tu_memory_system_t *mem,
                            const tu_mem_access_t *access,
                            uint64_t *cycles_out);

tu_status_t tu_memory_write(tu_memory_system_t *mem,
                             const tu_mem_access_t *access,
                             const void *data,
                             uint64_t *cycles_out);

// Bulk DMA-like transfer (for DMA engine use)
tu_status_t tu_memory_transfer(tu_memory_system_t *mem,
                                tu_mem_level_t src_level, tu_addr_t src_addr,
                                tu_mem_level_t dst_level, tu_addr_t dst_addr,
                                uint32_t size_bytes,
                                tu_transfer_strategy_t strategy,
                                uint64_t *cycles_out);

// Get memory bandwidth / latency at a level
void tu_memory_get_stats(const tu_memory_system_t *mem,
                          tu_mem_level_t level,
                          tu_memory_stats_t *stats);

// Check for bank conflicts in a set of accesses
uint32_t tu_memory_check_bank_conflicts(tu_memory_system_t *mem,
                                         const tu_mem_access_t *accesses,
                                         uint32_t num_accesses);

// Double-buffer swap (for ping-pong scratchpads)
tu_status_t tu_memory_swap_buffers(tu_memory_system_t *mem,
                                    tu_mem_level_t level,
                                    uint32_t buffer_id);
```

### 5.3 DMA Engine Interface

```c
// tu_cmodel/dma/dma_engine.h

typedef enum {
    TU_DMA_CHANNEL_W,     // Weight loading
    TU_DMA_CHANNEL_A,     // Activation loading
    TU_DMA_CHANNEL_O,     // Output load/store
    TU_DMA_CHANNEL_GENERAL, // General purpose
} tu_dma_channel_id_t;

typedef enum {
    TU_DMA_TRANSFER_LINEAR,      // Contiguous
    TU_DMA_TRANSFER_STRIDED_2D,  // Row-major 2D with stride
    TU_DMA_TRANSFER_STRIDED_3D,  // 3D block with strides
    TU_DMA_TRANSFER_SCATTER,     // Scatter via index list
    TU_DMA_TRANSFER_GATHER,      // Gather via index list
    TU_DMA_TRANSFER_BROADCAST,   // 1-to-N broadcast
    TU_DMA_TRANSFER_IM2COL,      // im2col address pattern
} tu_dma_transfer_type_t;

// DMA descriptor
typedef struct {
    tu_dma_transfer_type_t  type;
    tu_dma_channel_id_t     channel;
    
    // Source
    tu_mem_level_t  src_level;
    tu_addr_t       src_base;
    uint32_t        src_strides[3];
    
    // Destination
    tu_mem_level_t  dst_level;
    tu_addr_t       dst_base;
    uint32_t        dst_strides[3];
    
    // Dimensions
    uint32_t        dims[3];        // rows, cols, depth
    uint32_t        elem_size;      // bytes per element
    uint32_t        total_bytes;    // Total transfer size
    
    // Chaining
    tu_addr_t       next_descriptor; // 0 = end of chain
    uint32_t        completion_signal_id;
    
    // Priority
    uint8_t         priority;       // 0 (lowest) to 255 (highest)
    
} tu_dma_descriptor_t;

typedef struct tu_dma_engine_t tu_dma_engine_t;

tu_dma_engine_t* tu_dma_create(const tu_config_t *config,
                                tu_memory_system_t *mem);

void tu_dma_destroy(tu_dma_engine_t *dma);

// Submit a single descriptor
tu_status_t tu_dma_submit(tu_dma_engine_t *dma,
                           const tu_dma_descriptor_t *desc);

// Submit a chain of descriptors
tu_status_t tu_dma_submit_chain(tu_dma_engine_t *dma,
                                 tu_addr_t first_desc_addr);

// Poll a channel for completion
tu_status_t tu_dma_poll(tu_dma_engine_t *dma,
                          tu_dma_channel_id_t channel,
                          uint32_t completion_signal_id,
                          bool *done);

// Wait for all pending DMA on a channel
tu_status_t tu_dma_flush(tu_dma_engine_t *dma,
                          tu_dma_channel_id_t channel);

// Get DMA performance counters
void tu_dma_get_counters(const tu_dma_engine_t *dma,
                          tu_dma_counters_t *counters);
```

### 5.4 ISA & Command Queue Interface

```c
// tu_cmodel/isa/tu_isa.h

// Opcode encoding
typedef enum {
    TU_ISA_NOP        = 0x00,
    TU_ISA_MMA        = 0x01,
    TU_ISA_CONV       = 0x02,
    TU_ISA_ATTENTION  = 0x03,
    TU_ISA_ELEMENTWISE= 0x04,
    TU_ISA_REDUCTION  = 0x05,
    TU_ISA_SOFTMAX    = 0x06,
    TU_ISA_LAYERNORM  = 0x07,
    TU_ISA_RMSNORM    = 0x08,
    TU_ISA_POOL       = 0x09,
    TU_ISA_TRANSPOSE  = 0x0A,
    TU_ISA_DMA_LOAD   = 0x10,
    TU_ISA_DMA_STORE  = 0x11,
    TU_ISA_DMA_CHAIN  = 0x12,
    TU_ISA_SYNC       = 0x1F,
    TU_ISA_BARRIER    = 0x20,
    TU_ISA_CONFIG     = 0x7E,
    TU_ISA_HALT       = 0x7F,
} tu_isa_opcode_t;

// Fixed-size instruction encoding (64-bit)
typedef struct {
    uint8_t   opcode;          // 8 bits
    uint8_t   flags;           // 8 bits (precision, transpose, bias, activation)
    uint16_t  dim0;            // 16 bits — context-dependent (M, N, K, etc.)
    uint8_t   dim1;            // 8 bits
    uint8_t   dim2;            // 8 bits
    uint16_t  reserved;        // 16 bits — extension
    // Remaining operands in 32-bit immediate field
    uint64_t  immediates;      // Up to 64 bits of immediate data (offsets, etc.)
} tu_instruction_t;  // 96 bits total, or variable-length with extension

// Command queue
typedef struct tu_command_queue_t tu_command_queue_t;

tu_command_queue_t* tu_cmdq_create(const tu_config_t *config);

void tu_cmdq_destroy(tu_command_queue_t *cmdq);

// Submit a command with dependencies
tu_status_t tu_cmdq_submit(tu_command_queue_t *cmdq,
                            const tu_instruction_t *instr,
                            uint32_t num_deps,
                            const uint32_t *dependency_ids,
                            uint32_t *cmd_id_out);

// Wait for command completion
tu_status_t tu_cmdq_wait(tu_command_queue_t *cmdq,
                           uint32_t cmd_id,
                           uint64_t timeout_cycles);

// Barrier: all prior commands must complete before subsequent ones
tu_status_t tu_cmdq_barrier(tu_command_queue_t *cmdq,
                              uint32_t *barrier_id_out);

// Get queue depth / occupancy
uint32_t tu_cmdq_get_depth(const tu_command_queue_t *cmdq);
```

### 5.5 TU Core (Top-Level Orchestrator)

```c
// tu_cmodel/tu_core.h

typedef struct tu_core_t tu_core_t;

// Create a TU core from configuration
tu_core_t* tu_core_create(const char *config_path);

// Create from in-memory config struct
tu_core_t* tu_core_create_from_config(const tu_config_t *config);

// Initialize (reset state, zero memories, reset counters)
tu_status_t tu_core_init(tu_core_t *core);

// Submit an instruction to the command queue
tu_status_t tu_core_submit(tu_core_t *core,
                            const tu_instruction_t *instr,
                            uint32_t *cmd_id_out);

// Execute an ASM program (text or binary)
tu_status_t tu_core_execute_asm_text(tu_core_t *core,
                                       const char *program,
                                       const tu_host_buffer_t *buffers,
                                       uint32_t num_buffers);

tu_status_t tu_core_execute_asm_binary(tu_core_t *core,
                                         const uint8_t *program,
                                         uint32_t program_size,
                                         const tu_host_buffer_t *buffers,
                                         uint32_t num_buffers);

// Wait for all outstanding commands to complete
tu_status_t tu_core_sync(tu_core_t *core);

// Get comprehensive performance report
void tu_core_get_perf_report(const tu_core_t *core,
                              tu_perf_report_t *report);

// Dump state for debugging (SRAM contents, PE state, counters)
void tu_core_dump_state(const tu_core_t *core,
                         FILE *output);

// Get a subcomponent for direct manipulation (testing/debugging)
tu_compute_engine_t*  tu_core_get_compute(tu_core_t *core);
tu_memory_system_t*   tu_core_get_memory(tu_core_t *core);
tu_dma_engine_t*      tu_core_get_dma(tu_core_t *core);
tu_command_queue_t*   tu_core_get_cmdq(tu_core_t *core);

// Multi-core TU
typedef struct tu_cluster_t tu_cluster_t;

tu_cluster_t* tu_cluster_create(uint32_t num_cores,
                                  const tu_config_t *base_config);

tu_core_t* tu_cluster_get_core(tu_cluster_t *cluster, uint32_t core_id);

// Inter-core communication (ICI-like)
tu_status_t tu_cluster_send(tu_cluster_t *cluster,
                              uint32_t src_core, uint32_t dst_core,
                              tu_addr_t local_addr, uint32_t size_bytes);

// Destroy
void tu_core_destroy(tu_core_t *core);
void tu_cluster_destroy(tu_cluster_t *cluster);
```

### 5.6 Precision Abstraction Layer

```c
// tu_cmodel/precision/rounding.h

// Pluggable precision interface
typedef struct tu_precision_ops_t {
    // Name
    const char *name;
    
    // Convert from FP32 to this type
    void (*from_fp32)(const float *src, void *dst, size_t n);
    
    // Convert from this type to FP32
    void (*to_fp32)(const void *src, float *dst, size_t n);
    
    // MAC operation: dst += a * b
    void (*mac)(const void *a, const void *b, float *dst, size_t n);
    
    // Element size in bytes
    size_t elem_size;
    
    // Dynamic range
    float min_value;
    float max_value;
    
    // Is integer type
    bool is_integer;
    
    // Qunatization parameters (for integer types)
    bool has_scale;
    float default_scale;
    int32_t default_zero_point;
    
} tu_precision_ops_t;

// Built-in precision implementations
extern const tu_precision_ops_t tu_precision_fp8_e4m3;
extern const tu_precision_ops_t tu_precision_fp8_e5m2;
extern const tu_precision_ops_t tu_precision_fp16;
extern const tu_precision_ops_t tu_precision_bf16;
extern const tu_precision_ops_t tu_precision_tf32;
extern const tu_precision_ops_t tu_precision_int8;
extern const tu_precision_ops_t tu_precision_int4;

// Register custom precision
void tu_precision_register(const char *name, const tu_precision_ops_t *ops);

// Look up by name
const tu_precision_ops_t* tu_precision_lookup(const char *name);
```

### 5.7 Performance Counter & Tracing Interface

```c
// tu_cmodel/perf/performance_counters.h

typedef struct {
    // DMA counters
    uint64_t dma_read_bytes;
    uint64_t dma_write_bytes;
    uint64_t dma_read_cycles;
    uint64_t dma_write_cycles;
    uint64_t dma_stall_cycles;        // Stalled waiting for memory
    
    // Compute counters
    uint64_t compute_total_cycles;
    uint64_t compute_active_cycles;    // Non-idle cycles
    uint64_t compute_stall_cycles;     // Stalled waiting for data
    uint64_t total_macs;
    uint64_t total_flops;
    float    compute_utilization;      // active/total ratio
    
    // Memory counters
    uint64_t spad_reads;
    uint64_t spad_writes;
    uint64_t spad_bank_conflicts;
    uint64_t gbuf_reads;
    uint64_t gbuf_writes;
    uint64_t dram_reads;
    uint64_t dram_writes;
    
    // Operation counters (per opcode)
    uint64_t op_counts[32];
    
    // Power estimation (nJ)
    double energy_mac;
    double energy_sram_read;
    double energy_sram_write;
    double energy_dram;
    double energy_dma;
    double energy_leakage;
    double energy_total;
    
} tu_perf_counters_t;

// Event trace for VCD/FST generation
typedef struct tu_event_trace_t tu_event_trace_t;

tu_event_trace_t* tu_event_trace_create(const char *filename);

void tu_event_trace_signal(tu_event_trace_t *trace,
                            const char *hierarchy,
                            uint64_t value,
                            uint8_t width);

void tu_event_trace_tick(tu_event_trace_t *trace);

void tu_event_trace_close(tu_event_trace_t *trace);
```

### 5.8 Configuration System

```c
// tu_cmodel/infra/config.h

typedef struct {
    // Array dimensions
    uint32_t pe_rows;
    uint32_t pe_cols;
    
    // Data types
    tu_dtype_t input_dtype;
    tu_dtype_t weight_dtype;
    tu_dtype_t accumulate_dtype;
    tu_dtype_t output_dtype;
    
    // Memory sizes (bytes)
    uint32_t regfile_per_pe;
    uint32_t local_spad_size;
    uint32_t local_spad_banks;
    bool     local_spad_double_buffered;
    uint32_t global_buffer_size;
    uint32_t global_buffer_banks;
    
    // DRAM
    tu_dram_type_t dram_type;
    float          dram_bandwidth_gbps;
    uint32_t       dram_latency_ns;
    
    // DMA
    uint32_t dma_channels;
    uint32_t dma_bus_width_bits;
    uint32_t dma_queue_depth;
    
    // Features
    bool enable_convolution;
    bool enable_attention;
    bool enable_elementwise;
    bool enable_normalization;
    bool enable_sparsity;
    bool enable_multi_core;
    
    // Model fidelity
    bool cycle_accurate;
    bool model_bank_conflicts;
    bool model_power;
    
    // Derived values (computed by tu_config_derive)
    uint32_t mac_pipeline_depth;
    uint32_t total_sram_size;
    uint32_t peak_macs_per_cycle;
    
} tu_config_t;

// Load from YAML/JSON file
tu_status_t tu_config_load_file(const char *path, tu_config_t *config);

// Load from JSON string
tu_status_t tu_config_load_string(const char *json, tu_config_t *config);

// Compute derived values and validate
tu_status_t tu_config_derive(tu_config_t *config);

// Generate C header from config
void tu_config_emit_header(const tu_config_t *config, FILE *output);

// Generate documentation from config
void tu_config_emit_docs(const tu_config_t *config, FILE *output);
```

---

## 6. Implementation Roadmap

### 6.0 Pre-requisite: CI & Infrastructure (Week 1)

- Set up CMake build system (replace Makefile)
- Set up auto-formatting (clang-format), linting (clang-tidy)
- Set up CI pipeline template (build + test on every commit)
- Write initial `.clang-format`, `.clang-tidy`, `CMakeLists.txt`

### 6.1 P0 — Foundation & Critical Gaps (Weeks 1-4)

**Goal:** Establish the production architecture skeleton and address all "Critical" severity gaps.

| ID | Task | Effort | Dependencies | Deliverable |
|----|------|--------|-------------|-------------|
| **P0.1** | **Refactor into modular file structure** | 2 days | — | `tu_core.h`, `tu_types.h`, new directory layout; existing code moved into `compute/`, `memory/`, `dma/`, `precision/` without changing behavior |
| **P0.2** | **Configuration system** | 3 days | P0.1 | `infra/config.h`, `infra/config.c`; YAML/JSON loader; `tu_config_t` struct; `#define` constants replaced with config lookups; existing functionality preserved |
| **P0.3** | **Parameterized PE array** | 2 days | P0.2 | Configurable `PE_ROWS`×`PE_COLS`; tile size derived from config; existing 16×16 as default |
| **P0.4** | **Banked SRAM model** | 3 days | P0.2 | `memory/sram_bank.h`, bank conflict detection; replace flat byte arrays with multi-bank implementation; configurable banking factor |
| **P0.5** | **Async DMA with descriptor queues** | 4 days | P0.2 | `dma/dma_engine.h`, `dma/dma_descriptor.h`; async submission with completion signals; double-buffering support; replace synchronous memcpy |
| **P0.6** | **Command queue** | 3 days | P0.5 | `isa/command_queue.h`; submit instructions with dependency tracking; out-of-order execution analysis; barrier support |
| **P0.7** | **Expanded ISA + binary encoding** | 3 days | — | 30+ instructions for all P0 operations; 96-bit or variable-length encoding; `isa/tu_isa_encoder.c`, `isa/tu_isa_decoder.c`; text format updated |
| **P0.8** | **Golden reference framework** | 3 days | P0.3 | `tests/golden/generate_reference.py`; PyTorch-based reference for MMA (FP16, BF16, INT8); automated comparison script; bit-exact match for FP paths |
| **P0.9** | **Comprehensive unit tests** | 5 days | P0.3-P0.8 | Parameterized test framework; randomized tensor tests; edge cases (zero dims, max dims, non-multiple-of-tile); all data types; coverage > 90% |

**P0 Acceptance Criteria:**
- Same ONNX models compile and produce identical results
- All existing tests pass with configurable PE sizes (16×16, 32×32)
- Randomized tests with >10K random tensors pass against PyTorch reference
- Debug/Release builds on GCC and Clang

### 6.2 P1 — Feature Completeness (Weeks 5-10)

**Goal:** Support all high-priority operations, data types, and optimization features.

| ID | Task | Effort | Dependencies | Deliverable |
|----|------|--------|-------------|-------------|
| **P1.1** | **Multi-precision support** | 4 days | P0.2 | `precision/fp8.c`, `precision/bf16.c`, `precision/tf32.c`, `precision/int_quant.c`; all FP/INT conversion routines with correct rounding |
| **P1.2** | **Pluggable dataflow** | 5 days | P0.3 | `compute/dataflow/dataflow_interface.h`; WS, OS, RS implementations; configurable per-operation; verified against reference for each dataflow |
| **P1.3** | **Convolution engine** | 5 days | P1.2 | `compute/convolution_engine.c`; hardware im2col + direct conv; stride/padding/dilation; depthwise and grouped conv |
| **P1.4** | **Attention engine** | 5 days | P1.2 | `compute/attention_engine.c`; FlashAttention-style tiling; Q×K^T + softmax + ×V in SRAM; causal masking; multi-head support |
| **P1.5** | **Elementwise + activation pipeline** | 3 days | P1.2 | `compute/elementwise_pipeline.c`; fused ops: ReLU, GELU, SiLU, tanh, sigmoid, exp, add, mul; fused with preceding GEMM |
| **P1.6** | **Normalization engines** | 3 days | P0.3 | `compute/normalization_engine.c`; LayerNorm, RMSNorm, BatchNorm; online statistics computation |
| **P1.7** | **Softmax engine** | 2 days | P1.5 | `compute/softmax_engine.c`; online softmax with max-subtract for numerical stability |
| **P1.8** | **Toy→Production run script** | 2 days | P0.9 | Script to run all TinyTU tests with new cmodel, verify identical results |
| **P1.9** | **Subnormal handling modes** | 1 day | P1.1 | Configurable flush-to-zero vs full subnormal; match real hardware behavior |
| **P1.10** | **Multi-core TU cluster** | 4 days | P0.6 | `tu_cluster_t`; inter-core communication (data transfer, sync); SPMD execution model |
| **P1.11** | **DMA scatter/gather** | 2 days | P0.5 | `dma/scatter_gather.h`; index-based and strided scatter/gather transfer patterns |
| **P1.12** | **Software pipelining in ISA** | 3 days | P0.6 | Double-buffered scratchpad management; DMA tile N+1 while computing tile N; ASM-level pipeline directives |

**P1 Acceptance Criteria:**
- Full ResNet-50 layer (Conv→BN→ReLU→Pool) runs on TU
- Basic Transformer block (QKV projection→Attention→FFN→LayerNorm) runs on TU
- Multi-core configs (2, 4 cores) accelerate batch inference
- All data type paths verified against reference

### 6.3 P2 — Advanced Features & Tooling (Weeks 11-16)

| ID | Task | Effort | Dependencies | Deliverable |
|----|------|--------|-------------|-------------|
| **P2.1** | **Structured sparsity (2:4)** | 4 days | P1.1 | `sparsity/structured_2of4.c`; compressed weight storage; sparse MMA with 2× throughput accounting |
| **P2.2** | **Compiler: scheduling pass** | 4 days | P0.7 | Instruction scheduling with latency modeling; DMA hoisting; operation reordering while honoring dependencies |
| **P2.3** | **Compiler: liveness-based allocation** | 3 days | P2.2 | Graph-coloring style scratchpad allocation; spill/fill insertion when SRAM overflows |
| **P2.4** | **Compiler: full ONNX op coverage** | 5 days | P1.1-P1.7 | All op handlers mapped to TU ops or host fallback; shape inference for all ops |
| **P2.5** | **Cycle-accurate model** | 5 days | P0.4 | `perf/cycle_model.c`; pipelined execution with hazards; bank conflict delays; DMA bandwidth contention; DRAM row buffer hit/miss |
| **P2.6** | **Power/energy model** | 3 days | P2.5 | `perf/power_model.c`; per-component energy counters; CACTI-based SRAM energy; configurable technology node |
| **P2.7** | **Event tracing (VCD/FST)** | 3 days | P2.5 | `perf/event_trace.c`; per-cycle signal traces; compatible with GTKWave/Surfer |
| **P2.8** | **Exception handling model** | 2 days | P0.3 | Precise exceptions; error injection for testability; configurable fault behavior |
| **P2.9** | **Comparative benchmarking** | 4 days | P2.5 | MLPerf Tiny benchmarks; compare against Gemmini, SCALE-Sim; performance/power Pareto analysis |
| **P2.10** | **Python bindings** | 3 days | P0.2 | pybind11 bindings for `tu_core_t`; NumPy tensor interop; Jupyter notebook examples |
| **P2.11** | **SystemC/TLM wrapper** | 3 days | P2.5 | TLM-2.0 loosely-timed model; SystemC sockets for integration into virtual platforms |
| **P2.12** | **Multi-backend code generation** | 4 days | P0.7 | C functional backend; gem5 integration; Verilog testbench generator; FPGA synthesis hints |
| **P2.13** | **MLIR TU dialect** | 5 days | P2.4 | MLIR dialect `tu.mma`, `tu.conv`, etc.; tiling/fusion/bufferization passes; LLVM IR lowering path |

### 6.4 P3 — Research & Extensibility (Weeks 17-20)

| ID | Task | Effort | Dependencies | Deliverable |
|----|------|--------|-------------|-------------|
| **P3.1** | **Block floating point (MSFP/Flexpoint)** | 3 days | P1.1 | Shared exponent per block; configurable block size |
| **P3.2** | **Memory compression** | 3 days | P2.1 | Huffman/run-length weight compression; decompression in DMA path |
| **P3.3** | **Multi-tenant isolation** | 3 days | P1.10 | Memory protection; context switching; QoS guarantees |
| **P3.4** | **DMA QoS & priority** | 2 days | P0.5 | Priority-based channel arbitration; bandwidth reservation |
| **P3.5** | **Architecture search integration** | 4 days | P2.5 | Expose design space parameters; integrate with Timeloop/Accelergy for DSE |
| **P3.6** | **Fault injection framework** | 2 days | P2.8 | Bit-flip injection in SRAM/Pipeline; resilience evaluation |
| **P3.7** | **RISC-V Linux driver model** | 3 days | P1.10 | Device tree bindings; Linux kernel driver skeleton; MMIO register documentation |
| **P3.8** | **Documentation site** | 3 days | — | Sphinx/MkDocs; API reference; architecture guide; getting-started tutorial |
| **P3.9** | **Sparse attention (block-sparse)** | 3 days | P2.1 | Block-sparse attention patterns (sliding window, global tokens) |
| **P3.10** | **Mixed-precision training** | 3 days | P1.1 | FP8/FP16 forward, FP32 master weights; gradient accumulation; loss scaling |

---

## 7. Verification Methodology

### 7.1 Verification Pyramid

```
                       ┌─────────────┐
                       │  System     │  Full models (ResNet, Transformer)
                       │  Tests      │
                      ┌┴─────────────┴┐
                      │  Integration  │  Multi-op pipelines, multi-core
                      │  Tests        │
                     ┌┴───────────────┴┐
                     │  Unit Tests    │  Per-component, per-op, per-dtype
                     │                │
                    ┌┴────────────────┴┐
                    │  Golden Ref     │  PyTorch bit-exact comparison
                    │  Comparison     │
                   ┌┴─────────────────┴┐
                   │  Random / Fuzz  │  Millions of random tensor tests
                   │  Testing        │
                  ┌┴──────────────────┴┐
                  │  Property-Based  │  Invariants, monotonicity, bounds
                  │  Testing         │
                 ┌┴───────────────────┴┐
                 │  Coverage-Driven  │  Line, branch, toggle, FSM coverage
                 │  Verification     │
                 └────────────────────┘
```

### 7.2 Golden Reference Model

```
For every operation, maintain three models:
  1. Cmodel (C/C++)           — our implementation
  2. Golden (PyTorch/NumPy)   — ground truth
  3. RTL model (Verilog)      — future

Comparison strategy:
  - FP paths: bit-exact match required (same rounding behavior)
  - INT paths: exact match (integer arithmetic is deterministic)
  - Softmax/Normalization: relative error < 1e-5 (accumulated FP error)

Golden reference generators:
  tests/golden/
  ├── generate_mma_ref.py        # Matrix multiply reference
  ├── generate_conv_ref.py       # Convolution reference
  ├── generate_attention_ref.py  # Attention reference
  ├── generate_norm_ref.py       # Normalization reference
  ├── generate_softmax_ref.py    # Softmax reference
  └── generate_all.py            # Bulk generator for CI
```

### 7.3 Random Tensor Testing (Differential Testing)

```c
// For each operation, data type, and dimension range:
for (int seed = 0; seed < 10000; seed++) {
    // Generate random tensors with uniform distribution
    generate_random_tensors(&a, &b, seed, dtype, dim_ranges);
    
    // Run cmodel
    tu_compute_execute(&result_cmodel, op, &a, &b);
    
    // Run golden reference (via Python subprocess or linked C reference)
    compute_golden(&result_golden, op, &a, &b);
    
    // Compare
    assert_tensors_equal(result_cmodel, result_golden, tolerance);
}

// Also test corner cases:
// - All zeros, all ones, all max-value, all min-value
// - Zero-sized dimensions, max-sized dimensions
// - Non-multiple-of-tile dimensions
// - Mixed signs, extreme values, NaN/Inf propagation
```

### 7.4 Property-Based Testing

For each operation, define invariants that must always hold:

| Operation | Invariant |
|-----------|-----------|
| **MMA** | Linearity: MMA(A+B, W) = MMA(A,W) + MMA(B,W) within FP tolerance |
| **MMA** | Scaling: MMA(αA, W) = α·MMA(A,W) within FP tolerance |
| **Conv** | Translation equivariance: output shifts accordingly |
| **Softmax** | Output sums to 1.0; all elements in [0, 1] |
| **LayerNorm** | Output mean ≈ 0, std ≈ 1 (for full norm) |
| **ReLU** | Output ≥ 0; ReLU(x) = 0 ⇔ x ≤ 0 |
| **All ops** | Same output for identical inputs (deterministic) |
| **All ops** | No NaN/Inf output for finite inputs |

### 7.5 Regression Test Suite

```
tests/
├── unit/
│   ├── test_fp_conversions.c      # All precision types, round-trip, edge cases
│   ├── test_mma.c                 # All dtype combinations, all dataflows
│   ├── test_mma_sparse.c          # 2:4 sparsity
│   ├── test_conv.c                # All spatial dims, strides, paddings, dilations
│   ├── test_attention.c           # Causal, masked, multi-head
│   ├── test_elementwise.c         # All activation functions
│   ├── test_normalization.c       # LayerNorm, RMSNorm, BatchNorm
│   ├── test_softmax.c             # Online softmax, numerical stability
│   ├── test_memory_banks.c        # Bank conflicts, multi-port, double buffer
│   ├── test_dma.c                 # All transfer types, async behavior
│   ├── test_isa.c                 # Encode/decode round-trip, invalid ops
│   ├── test_command_queue.c       # Ordering, dependency, barrier
│   └── test_config.c              # All config combinations
│
├── integration/
│   ├── test_tiny_mlp.c            # Original TinyTU MLP test
│   ├── test_resnet_block.c        # Conv→BN→ReLU→Conv→Add pipeline
│   ├── test_transformer_block.c   # Attention + FFN block
│   ├── test_fused_ops.c           # GEMM+ReLU fused, GEMM+LayerNorm fused
│   ├── test_multicore.c           # Multi-core SPMD execution
│   └── test_onnx_pipeline.c       # ONNX→compile→run full pipeline
│
├── random/
│   ├── test_random_mma.c          # 10K random MMA tests
│   ├── test_random_conv.c         # 10K random convolution tests
│   └── test_random_attention.c    # 10K random attention tests
│
├── benchmark/
│   ├── bench_mlperf_tiny.c        # MLPerf Tiny benchmarks
│   ├── bench_resnet50.c           # ResNet-50 performance
│   ├── bench_bert_base.c          # BERT-base layer benchmarks
│   └── bench_llama_layer.c        # LLaMA decoder layer
│
└── fuzz/
    ├── fuzz_isa_decoder.c         # AFL/libFuzzer on ISA decoder
    ├── fuzz_asm_parser.c          # AFL on ASM text parser
    └── fuzz_mma.c                 # AFL on MMA with random inputs
```

### 7.6 Coverage Targets

| Metric | Target |
|--------|--------|
| **Line coverage** | > 95% for core compute/memory/dma |
| **Branch coverage** | > 90% |
| **Toggle coverage** | > 85% for configuration bits |
| **Functional coverage** | All opcodes, all dtype combinations, all dataflow modes |
| **Edge case coverage** | All tile-edge conditions, all zero-dimension cases, all overflow/underflow paths |

### 7.7 Continuous Integration Pipeline

```yaml
# .github/workflows/ci.yml (conceptual)
pipeline:
  build:
    - cmake -B build -DCMAKE_BUILD_TYPE=Debug
    - cmake --build build
    - cmake -B build_release -DCMAKE_BUILD_TYPE=Release
    - cmake --build build_release
    
  test:
    - ctest --test-dir build -j4
    - ctest --test-dir build_release -j4
    # Random tests run in CI with reduced iterations
    # Full iteration count runs nightly
    
  benchmark:
    - ./build/bench_mlperf_tiny
    - Compare performance against baseline
    
  coverage:
    - cmake -B build_cov -DCOVERAGE=ON
    - cmake --build build_cov
    - ctest --test-dir build_cov
    - lcov --capture --directory build_cov
    - Check coverage >= 90%
    
  lint:
    - clang-format --dry-run --Werror
    - clang-tidy --warnings-as-errors
    
  nightly:
    - Full random test suite (100K iterations)
    - All configuration combinations matrix
    - Fuzzing runs (1 hour each target)
```

### 7.8 Performance Model Calibration

```
1. Baseline reference: Gemmini RTL simulation (Verilator)
   - Run identical GEMM configs on Gemmini and our cmodel
   - Compare cycle counts; compute error ratio
   - Tune our cycle model to match within ±5%

2. SCALE-Sim / Timeloop comparison:
   - Feed identical architecture parameters to both models
   - Compare utilization, stall cycles, bandwidth utilization
   - Document and explain differences

3. Real silicon (future):
   - When RTL or FPGA implementation exists, calibrate against hardware
   - Continuous calibration as architecture evolves
```

---

## Appendix A: Migration Path from TinyTU

```
Phase 0 (Now):
  TinyTU: monolithic 16×16, FP16-only, synchronous DMA

Phase P0 (Week 4):
  Production TU: Same results, configurable dims, modular code
  └─ TinyTU adapter shim maps old API to new tu_core_t

Phase P1 (Week 10):
  Production TU: All data types, all ops, configurable dataflows
  └─ ONNX compiler upgraded to use new ISA

Phase P2 (Week 16):
  Production TU: Cycle-accurate, sparse, multi-backend
  └─ Drop TinyTU adapter; all tools target new API directly

Backward compatibility contract:
  - tu_init(), tu_mma(), tu_dma_load_w(), etc. remain available
    as thin wrappers around tu_core_t
  - TU ASM text format extended (backward compatible)
  - Existing ONNX models continue to work without changes
```

## Appendix B: Key References

1. **Google TPU:** Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017; "TPU v4," MLSys 2023
2. **NVIDIA TensorCore:** "NVIDIA A100 Tensor Core GPU Architecture," 2020; "NVIDIA H100 Tensor Core GPU Architecture," 2022
3. **Gemmini:** Genc et al., "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration," DAC 2021
4. **Eyeriss:** Chen et al., "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks," ISCA 2016; "Eyeriss v2," JSSC 2019
5. **MAERI:** Kwon et al., "MAERI: Enabling Flexible Dataflow Mapping over DNN Accelerators via Reconfigurable Interconnects," ASPLOS 2018
6. **SCALE-Sim:** Samajdar et al., "SCALE-Sim: Systolic CNN Accelerator Simulator," ISPASS 2020
7. **Timeloop:** Parashar et al., "Timeloop: A Systematic Approach to DNN Accelerator Evaluation," ISPASS 2019
8. **OCP FP8:** "OCP 8-bit Floating Point Specification (OFP8)," Open Compute Project, 2023
