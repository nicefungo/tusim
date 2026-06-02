# Python Bindings for TU CModel (Gap I2)

> **Status:** Implemented  
> **Gap ID:** I2 — Python bindings as first-class integration surface (High, P1)  
> **Files:** `bindings/python/tu_bindings.py`  
> **Tests:** Identity GEMM passing via Python CLI

## Overview

The TU CModel Python bindings expose the full cmodel API to Python via ctypes. This enables:

- **NumPy interop:** Convert NumPy arrays to/from FP16 for direct TU execution
- **Golden comparison:** Run cmodel and PyTorch reference side-by-side in Python
- **Jupyter notebooks:** Explore TU architecture interactively
- **Test frameworks:** Write TU tests in Python using pytest
- **Design space exploration:** Script parameter sweeps across PE sizes, dataflows, and precisions

**Why ctypes over pybind11:** The TU cmodel is pure C. ctypes requires zero C++ compilation, works with any Python 3.7+, and loads the existing `libtucmodel.so` without build system changes. Transition to pybind11 (Gap Q5) remains a future option for richer type safety and performance.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Python User Code                                     │
│  (NumPy, PyTorch, Jupyter, pytest)                    │
├──────────────────────────────────────────────────────┤
│  tu_bindings.py                                       │
│  ├── TUCore          — Main API: init, DMA, MMA, ASM  │
│  ├── HostBuffers     — Named buffer management        │
│  ├── quick_gemm()    — One-call GEMM convenience       │
│  ├── FP16 utils      — float↔fp16, numpy interop     │
│  └── CLI main()      — Quick smoke test               │
├──────────────────────────────────────────────────────┤
│  ctypes (Python stdlib)                               │
├──────────────────────────────────────────────────────┤
│  libtucmodel.so  ←  C shared library                  │
│  (built from all tu_cmodel/*.c + deps)                │
└──────────────────────────────────────────────────────┘
```

## Installation

```bash
# Build the shared library
cd /path/to/tusim
make libtucmodel.so

# No Python package installation needed — just import
python3 -c "import sys; sys.path.insert(0, 'bindings/python'); import tu_bindings as tu"
```

Or for convenience:

```bash
export PYTHONPATH="$PWD/bindings/python:$PYTHONPATH"
```

## Quick Start

### Identity GEMM (CLI smoke test)

```bash
python3 bindings/python/tu_bindings.py
```

Expected output:
```
TU CModel Python Bindings
=========================
  TU core initialized (library: .../libtucmodel.so)
  Running identity GEMM test...
  ✅ Identity GEMM: PASS
  Done.
```

### Basic GEMM from Python

```python
import tu_bindings as tu

# Create TU core with default config
core = tu.TUCore()

# Prepare buffers
bufs = tu.HostBuffers()

# Weight matrix W[4][4] — identity
w_fp16 = tu.numpy_to_fp16_bytes([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# Activation A[4][4] — input vector
a_fp16 = tu.numpy_to_fp16_bytes([
    [1.0, 2.0, 3.0, 4.0],
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0],
])

bufs.add("w", w_fp16)
bufs.add("a", a_fp16)
bufs.add("o", size=4 * 4 * 4)  # Output: 4×4 float32

# Load → Compute → Store
core.dma_load_w(bufs.get("w"), 0, len(w_fp16))
core.dma_load_a(bufs.get("a"), 0, len(a_fp16))
core.mma(4, 4, 4, 0, 0, 0, False)
core.sync()
core.dma_store_o(bufs.get("o"), 0, 4 * 4 * 4)

# Read result
result = bufs.get("o").get_float32(4 * 4)
print(result)  # [1.0, 2.0, 3.0, 4.0, ...]
```

### ASM Program Execution

```python
import tu_bindings as tu

core = tu.TUCore()
bufs = tu.HostBuffers()

# Matrix data in FP16
bufs.add("w", weight_fp16_data)
bufs.add("a", activation_fp16_data)
bufs.add("o", size=1024)  # Output buffer

program = (
    "LOAD_W w 0 512\n"
    "LOAD_A a 0 256\n"
    "MMA 16 16 16 0 0 128\n"
    "STORE_O o 128 256\n"
    "SYNC\n"
)
core.execute_asm(program, bufs)

result = bufs.get("o").get_float32(16)
```

### One-Shot Convenience

```python
import tu_bindings as tu

A = [[1.0, 2.0], [3.0, 4.0]]  # 2×2
B = [[1.0, 0.0], [0.0, 1.0]]  # 2×2 identity
C = tu.quick_gemm(A, B)
# C = [[1.0, 2.0], [3.0, 4.0]]
```

## API Reference

### TUCore

```python
class TUCore:
    def __init__(self, config_path=None, library_path=None)
    """Initialize a TU core instance."""

    def dma_load_w(self, buf: HostBuffer, tu_offset=0, size=None)
    """Load weight buffer from host to TU W-SRAM."""

    def dma_load_a(self, buf: HostBuffer, tu_offset=0, size=None)
    """Load activation buffer from host to TU A-SRAM."""

    def dma_load_o(self, buf: HostBuffer, tu_offset=0, size=None)
    """Load output bias buffer from host to TU O-SRAM."""

    def dma_store_o(self, buf: HostBuffer, tu_offset=0, size=None)
    """Store output buffer from TU O-SRAM to host."""

    def mma(self, M, N, K, w_offset=0, a_offset=0, o_offset=0, has_bias=False)
    """Execute matrix multiply-accumulate: O[N][M] += W[N][K] × A[K][M]"""

    def sync(self)
    """Drain systolic pipeline."""

    def execute_asm(self, program: str, buffers: HostBuffers = None)
    """Execute a TU ASM program text."""

    def reset(self)
    """Reset TU state."""
```

### HostBuffer / HostBuffers

```python
class HostBuffer:
    name: str
    ptr: int       # ctypes pointer (for C interop)
    size: int      # Buffer size in bytes
    def get_bytes() -> bytes
    def get_fp16(count) -> List[float]
    def get_float32(count) -> List[float]

class HostBuffers:
    def add(name, data=None, size=0) -> HostBuffer
    def get(name) -> Optional[HostBuffer]
```

### FP16 Utilities

```python
def fp16_to_float(h: int) -> float
    """IEEE 754 binary16 → float32."""

def float_to_fp16(f: float) -> int
    """float32 → IEEE 754 binary16."""

def numpy_to_fp16_bytes(arr) -> bytes
    """Convert numpy array to FP16 bytes."""

def fp16_bytes_to_numpy(data: bytes, dtype='float32')
    """Convert FP16 bytes to numpy array."""

def quick_gemm(A: List[List[float]], B: List[List[float]],
               config_path=None) -> List[List[float]]
    """One-call GEMM on TU cmodel."""
```

## How It Changes CModel Behavior

**Before (I2):** The cmodel was only callable from compiled C programs. Integration with Python tooling required manual subprocess calls, ad-hoc data serialization, and fragile parsing of stdout. No NumPy interop existed. The ONNX compiler ran Python but couldn't call the cmodel directly.

**After:** The full cmodel API is callable from Python with automatic FP16 conversion and NumPy support. This enables:
- Golden comparison: `assert np.allclose(tu_result, torch_result, atol=1e-3)`
- Architectural exploration: `for pe_rows in [16, 32, 64, 128]: sweep()`
- Jupyter visualization of internal state
- pytest-based test suites for the cmodel
- Integration with ML frameworks and ONNX runtime

## ASM Interpreter

The Python `execute_asm()` method implements a pure-Python ASM interpreter that maps ASM instructions directly to C API calls. It supports the 6 core TU instructions:

| Instruction | Python Mapping |
|-------------|---------------|
| `LOAD_W name offset size` | `dma_load_w(buf[name], offset, size)` |
| `LOAD_A name offset size` | `dma_load_a(buf[name], offset, size)` |
| `LOAD_O name offset size` | `dma_load_o(buf[name], offset, size)` |
| `MMA M N K w_off a_off o_off [BIAS]` | `mma(M, N, K, w_off, a_off, o_off, has_bias)` |
| `STORE_O name offset size` | `dma_store_o(buf[name], offset, size)` |
| `SYNC` | `sync()` |

Extended ISA instructions (CONV, ATTENTION, SOFTMAX, etc.) require linking against the full production cmodel library. These are available when `libtucmodel.so` is built with the production configuration.

## Limitations & Future Work

- **Synchronous only.** All calls block. Async execution requires Python threading or asyncio wrappers.
- **No extended ISA in Python ASM.** The Python ASM interpreter handles the 6 core TinyTU instructions. Extended ops require direct C API calls.
- **No performance counters from Python.** `tu_print_stats()` writes to stdout. Future: wrap `tu_perf_counters_t` in ctypes structs.
- **No power model from Python.** `tu_power_print_report()` writes to stdout. Future: ctypes wrapper for `tu_power_model_t`.
- **pybind11 upgrade path.** For richer type safety, pybind11 bindings (Gap Q5) would provide automatic NumPy buffer protocol support and lower call overhead.

## Integration Examples

### Compare against PyTorch Reference

```python
import torch
import tu_bindings as tu

# Generate random tensors
A = torch.randn(16, 16, dtype=torch.float16)
B = torch.randn(16, 16, dtype=torch.float16)

# PyTorch reference
C_ref = A @ B

# TU cmodel
C_tu = tu.quick_gemm(A.float().tolist(), B.float().tolist())

# Compare
C_tu_tensor = torch.tensor(C_tu)
max_err = (C_ref.float() - C_tu_tensor).abs().max().item()
print(f"Max error: {max_err:.6f}")
assert max_err < 0.01, "TU result diverges from PyTorch!"
```

### Design Space Sweep

```python
import tu_bindings as tu

for pe_size in [16, 32, 64]:
    for dataflow in ["weight_stationary", "output_stationary"]:
        # (configure via config file)
        core = tu.TUCore(f"config/tu_{pe_size}x{pe_size}_{dataflow}.json")
        result = tu.quick_gemm(A, B)
        print(f"{pe_size}×{pe_size} {dataflow}: {result[0][0]:.4f}")
```

## Testing

The Python bindings include a built-in smoke test:

```bash
python3 bindings/python/tu_bindings.py
```

Verifies:
- Library loading and TU core initialization
- Identity GEMM correctness
- FP16 conversion round-trip
- DMA load/store cycle

For comprehensive testing, use the C test suite (`make test`).

## References

- Python ctypes documentation: https://docs.python.org/3/library/ctypes.html
- IEEE 754-2008 binary16 format
- TU CModel DESIGN.md
- Gap I2 specification in `PRODUCTION_TU_REDESIGN.md`
