# TU CModel — DPI-C Integration Wrapper (I1)

> **Gap ID:** I1 (DPI-C / SystemC integration for RTL co-simulation)
> **Priority:** P1 (High)
> **Date:** 2026-06-03
> **Heartbeat:** Midday shift (14:20), Cycle 2

---

## What Changed

A production-grade DPI-C wrapper has been added to the TU cmodel, enabling SystemVerilog testbenches to use the cmodel as a golden reference for RTL co-simulation. The wrapper follows IEEE 1800-2017 DPI-C conventions: scalar-only interface, handle-based instances, and shared-library compilation.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  SystemVerilog Testbench                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  import "DPI-C" function int tu_dpi_init(...);         │  │
│  │  import "DPI-C" function void tu_dpi_load_weights(...);│  │
│  │  import "DPI-C" function longint tu_dpi_gemm(...);     │  │
│  │  import "DPI-C" function int tu_dpi_read_counter(...); │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │ DPI-C boundary                      │
├──────────────────────────┼───────────────────────────────────┤
│                          ▼                                    │
│  tu_cmodel/bindings/tu_dpi.c  (C wrapper)                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Instance Pool: up to 16 TU cores                      │  │
│  │  Handle-based API: scalar args → tu_state_t            │  │
│  │  Thread-unsafe (single SV thread)                      │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                    │
│  tu_cmodel/tu_cmodel.c  (TU Core)                            │
└──────────────────────────────────────────────────────────────┘
```

---

## Why This Matters

### RTL Co-Simulation

The DPI-C wrapper enables three verification patterns:

1. **Golden Reference:** The cmodel pre-computes expected outputs. RTL is checked against the golden reference operation-by-operation.
2. **Lockstep Co-Simulation:** Both RTL and cmodel receive identical stimuli. Outputs are compared cycle-by-cycle. Discrepancies are flagged immediately.
3. **Standalone Verification:** The cmodel is exercised from SystemVerilog to validate architecture decisions before RTL exists.

### Industry Alignment

All major NPU/accelerator projects use C models for verification:
- **Synopsys C-Model Flow:** C models provide golden references for RTL verification
- **Google TPU:** Software simulator validates against hardware
- **NVIDIA TensorCore:** Functional model drives architectural exploration
- **Gemmini:** C simulator co-verified against Chisel RTL

---

## How It Works

### Handle-Based Instance Pool

The wrapper manages up to 16 simultaneous TU core instances:

```c
int h1 = tu_dpi_init(16, 16, 256, TU_DPI_DF_WS);  // 16×16 WS
int h2 = tu_dpi_init(32, 32, 512, TU_DPI_DF_OS);  // 32×32 OS
int h3 = tu_dpi_init(8, 8, 128, TU_DPI_DF_RS);    // 8×8 RS
```

Each handle is an opaque integer (1-based). Handle 0 is invalid.

### Scalar-Only Interface

No C structs cross the DPI boundary. All parameters are basic types:
- `int` for scalars, handles, status codes
- `long long` for 64-bit counters
- `void*` for memory buffers (byte arrays)
- `char*` for string outputs

### Memory Model

The wrapper converts DPI-C scalar offsets to TU SRAM operations:

```c
// SV: write activation data
tu_dpi_sram_write(h, 1, 0, a_data, 1024);  // region=1 (A-buffer)

// C: internally
tu_sram_region_t *reg = &g_tu.sram_a;
memcpy(tu_sram_raw_ptr(reg) + offset, src, bytes);
```

### API Surface

| Category | Functions | Description |
|----------|-----------|-------------|
| Lifecycle | `init`, `destroy`, `reset` | Create/release TU instances |
| Memory | `sram_write`, `sram_read`, `sram_size` | Byte-level SRAM access |
| Compute | `gemm`, `elementwise`, `softmax`, `layernorm` | Blocking operations |
| Async | `submit_gemm`, `submit_barrier`, `wait`, `sync` | Non-blocking command queue |
| Counters | `read_counter` (10 counter IDs) | Performance monitoring |
| Config | `set_dataflow`, `get_dataflow_name`, `get_pe_dims`, `get_sram_sizes` | Runtime configuration |

### Counter IDs

| ID | Name | Description |
|----|------|-------------|
| 0 | DMA_BYTES | Total DMA bytes transferred |
| 1 | MMA_CALLS | Number of MMA invocations |
| 2 | MMA_TILES | Number of tiled MMA operations |
| 3 | MMA_FLOPS | Effective FP16 multiply-adds |
| 4 | EST_CYCLES | Estimated cycle count |
| 5 | TOTAL_CYCLES | Total cycle count |
| 6 | COMPUTE_ACTIVE | Active compute cycles |
| 7 | BANK_CONFLICTS | SRAM bank conflicts |
| 8 | SRAM_READS | Total SRAM reads |
| 9 | SRAM_WRITES | Total SRAM writes |
| 10 | UTILIZATION | Compute utilization (scaled ×100) |

---

## SystemVerilog Usage

### Basic GEMM Verification

```systemverilog
module tb_tu_gemm;
    import "DPI-C" function int tu_dpi_init(int rows, int cols, int sram_kb, int df);
    import "DPI-C" function void tu_dpi_sram_write(int h, int region, int offset,
                                                     byte data[], int bytes);
    import "DPI-C" function longint tu_dpi_gemm(int h, int M, N, K,
                                                  int w_off, a_off, o_off, bias);
    import "DPI-C" function void tu_dpi_sram_read(int h, int region, int offset,
                                                    byte data[], int bytes);

    int handle;
    byte w_data[512];   // 16×16 FP16 = 512 bytes
    byte a_data[512];
    byte o_data[1024];  // 16×16 FP32 = 1024 bytes

    initial begin
        // Initialize 16×16 TU with WS dataflow
        handle = tu_dpi_init(16, 16, 256, 0);

        // Load identity matrices
        // ... fill w_data, a_data with FP16 1.0 on diagonal ...
        tu_dpi_sram_write(handle, 0, 0, w_data, 512);
        tu_dpi_sram_write(handle, 1, 0, a_data, 512);

        // Execute GEMM
        $display("Expected cycles: %0d",
            tu_dpi_gemm(handle, 16, 16, 16, 0, 0, 0, 0));

        // Read golden result
        tu_dpi_sram_read(handle, 2, 0, o_data, 1024);

        // Compare against RTL output
        // ...
    end
endmodule
```

### Co-Simulation Pattern

```systemverilog
// Drive identical stimuli to RTL and cmodel
task drive_and_compare(input int M, N, K);
    // Drive RTL
    rtl_dut.load_weights(w_data);
    rtl_dut.load_activations(a_data);
    rtl_dut.start_compute();
    #(compute_latency);

    // Drive cmodel
    int cmodel_cycles = tu_dpi_gemm(cmodel_h, M, N, K, 0, 0, 0, 0);

    // Compare
    byte rtl_out[4096], cmodel_out[4096];
    rtl_dut.read_output(rtl_out);
    tu_dpi_sram_read(cmodel_h, 2, 0, cmodel_out, 4096);

    for (int i = 0; i < 4096; i++) begin
        if (rtl_out[i] !== cmodel_out[i])
            $error("Mismatch at byte %0d: RTL=%02x CModel=%02x",
                   i, rtl_out[i], cmodel_out[i]);
    end

    // Compare cycle counts
    $display("RTL cycles: %0d, CModel cycles: %0d, ratio: %.2f",
             rtl_cycles, cmodel_cycles,
             real'(cmodel_cycles) / real'(rtl_cycles));
endtask
```

---

## Build & Link

```bash
# Build the DPI-C shared library
make libtucmodel.so

# In your SystemVerilog simulator (e.g., VCS, Questa, Xcelium):
# vcs -sv tb.sv -LDFLAGS "-L. -ltucmodel -lm"

# Standalone test (no SV simulator needed)
make test-dpi   # 13/13 tests pass
```

---

## Error Codes

| Code | Value | Meaning |
|------|-------|---------|
| TU_DPI_OK | 0 | Success |
| TU_DPI_ERR_INIT | -1 | Initialization failure |
| TU_DPI_ERR_HANDLE | -2 | Invalid handle |
| TU_DPI_ERR_PARAM | -3 | Invalid parameter |
| TU_DPI_ERR_MEMORY | -4 | Memory bounds violation |
| TU_DPI_ERR_DMA | -5 | DMA transfer error |
| TU_DPI_ERR_BUSY | -6 | Resource busy (timeout) |

---

## Verification

### Test Suite: 13 tests, all passing

| Test | What It Verifies |
|------|-----------------|
| Init/Destroy | Single instance lifecycle |
| Multi-Instance | 3 simultaneous handles (16×16 WS, 32×32 OS, 8×8 RS) |
| Memory R/W | Byte-level SRAM write/read round-trip |
| GEMM | 16×16 identity matrix via DPI |
| Elementwise | ReLU on mixed-sign data |
| Softmax | Per-row softmax, sum-to-1 property |
| LayerNorm | 2-row normalization |
| Performance Counters | MMA calls, FLOPS, cycles counters |
| Dataflow Switch | WS → OS → RS via DPI API |
| Async Commands | Submit GEMM + wait + sync |
| Reset/Re-use | Reset clears counters, re-use works |
| Error Handling | Invalid handles, bad regions, bad params |
| Summary String | Human-readable summary output |

### Run Tests

```bash
make test-dpi    # 13/13 tests pass
```

---

## Files

| File | Change |
|------|--------|
| `tu_cmodel/bindings/tu_dpi.h` | **New** — DPI-C interface (~260 LOC) |
| `tu_cmodel/bindings/tu_dpi.c` | **New** — DPI-C implementation (~400 LOC) |
| `tests/test_dpi.c` | **New** — 13 DPI-C tests (~370 LOC) |
| `Makefile` | Modified — bindings/tu_dpi.o, test-dpi target |
| `docs/dpi-c-integration.md` | **This document** |

---

## Next Steps

- **SystemC/TLM Wrapper (I1 extension):** Build a TLM-2.0 loosely-timed model on top of the DPI-C layer for SystemC virtual platforms
- **RTL Co-Simulation Harness (I5):** Create a Verilator-based co-simulation flow that feeds identical stimuli to both cmodel and RTL
- **DPI-C Streaming Interface:** Add cycle-by-cycle DPI-C functions for streaming data in/out to match RTL timing

## References

1. IEEE 1800-2017 SystemVerilog, Section 35: Direct Programming Interface (DPI)
2. Synopsys, "C-Model Flow for High-Performance Hardware Verification"
3. MathWorks, "HDL Verifier: Cosimulation using DPI-C"
4. Gemmini (Berkeley): Chisel RTL + C simulator co-verification
5. TU CModel Core API: `tu_cmodel/tu_cmodel.h`
