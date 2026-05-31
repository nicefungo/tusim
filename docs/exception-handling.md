# Exception Handling Framework (Gap E5)

> **Status:** Implemented  
> **Gap:** E5 — Exception/error handling (P2)  
> **Version:** 1.0  
> **Date:** 2026-06-01  
> **Files:** `tu_cmodel/tu_status.h`, `tu_cmodel/tu_status.c`, `tests/test_error_handling.c`

---

## Table of Contents

1. [Overview](#overview)
2. [Why This Gap Matters](#why-this-gap-matters)
3. [Architecture](#architecture)
4. [Error Code Catalog](#error-code-catalog)
5. [Error Reporting Macros](#error-reporting-macros)
6. [Error Behavior Modes](#error-behavior-modes)
7. [Error Injection](#error-injection)
8. [Migration Guide](#migration-guide)
9. [Verification](#verification)
10. [Limitations & Future Work](#limitations--future-work)

---

## Overview

Before this change, the TU cmodel used `abort()` for all error conditions — 5 scattered calls across 3 files that terminated the process immediately. This made the cmodel fragile: any bounds violation, uninitialized access, or invalid parameter would crash without diagnostic context or recovery.

This implementation introduces:

- **40 structured error codes** (`tu_status_t`) covering all subsystem failure modes
- **Configurable error behavior** — LOG (default), ABORT (strict), SILENT (quiet)
- **Error context capture** — file, line, function, message, timestamp
- **Convenience macros** — `TU_ASSERT`, `TU_CHECK`, `TU_RETURN_IF_ERR`, `TU_ERROR_INJECT`
- **Error injection framework** — for testing error recovery paths
- **Replacement of all 5 `abort()` calls** — with logged errors and graceful early returns

---

## Why This Gap Matters

**E5 (P2) was selected because:**

1. **Production readiness requires graceful degradation.** A cmodel that crashes on the first bounds violation is unusable for CI pipelines, fuzzing, or batch evaluation of thousands of ONNX models. Replacement with logged errors enables batch processing where one failure doesn't abort the entire run.

2. **Error injection enables robust testing.** Production hardware has fault tolerance mechanisms. The error injection framework allows testing recovery paths without modifying production code — critical for verifying error handling in DMA, SRAM, and command queue subsystems.

3. **It's a general property.** Unlike specialized features (sparsity, MLIR), error handling touches every subsystem. The framework scales to all future features.

4. **Low implementation cost, high impact.** Converting `abort()` → `TU_REPORT_ERR(...)` is a 1-line change at each site. The macros (`TU_ASSERT`, `TU_CHECK`) reduce boilerplate for future error checks.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Error Handling Flow                       │
│                                                              │
│  Application Code                                            │
│  ┌──────────────────────────────────────────────┐            │
│  │  TU_ASSERT(ptr != NULL, "ptr must exist");   │            │
│  │  TU_CHECK(x < MAX, TU_ERR_OUT_OF_RANGE, ...); │           │
│  │  TU_REPORT_ERR(TU_ERR_INTERNAL, "oops");     │            │
│  └──────────────────┬───────────────────────────┘            │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────┐            │
│  │         tu_report_error()                    │            │
│  │  • Records error in g_last_error              │            │
│  │  • Captures file, line, function, message    │            │
│  │  • Acts based on error mode                  │            │
│  └──────────────────┬───────────────────────────┘            │
│                     │                                        │
│          ┌──────────┼──────────┐                             │
│          ▼          ▼          ▼                              │
│      LOG mode   ABORT mode  SILENT mode                      │
│   fprintf to    fprintf +   return code                      │
│   stderr +      abort()     only                             │
│   return code                                                │
│                                                              │
│  Error Injection (test-only)                                 │
│  ┌──────────────────────────────────────────────┐            │
│  │  tu_error_inject_enable(file, line, code)    │            │
│  │  TU_ERROR_INJECT() at call site              │            │
│  │  → Returns injected code if site matches     │            │
│  └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

---

## Error Code Catalog

### Initialization (1-9)

| Code | Name | Use Case |
|------|------|----------|
| 0 | `TU_OK` | Success |
| 1 | `TU_ERR_NOT_INITIALIZED` | Operation before `tu_init()` |
| 2 | `TU_ERR_ALREADY_INITIALIZED` | Duplicate initialization |
| 3 | `TU_ERR_CONFIG_INVALID` | Bad configuration parameter |

### Parameters (10-19)

| Code | Name | Use Case |
|------|------|----------|
| 10 | `TU_ERR_INVALID_PARAM` | Generic invalid parameter |
| 11 | `TU_ERR_OUT_OF_RANGE` | Value outside valid range |
| 12 | `TU_ERR_NULL_POINTER` | Unexpected NULL |

### Memory (20-29)

| Code | Name | Use Case |
|------|------|----------|
| 20 | `TU_ERR_OUT_OF_MEMORY` | malloc/calloc failure |
| 21 | `TU_ERR_SRAM_OVERFLOW` | Access beyond SRAM capacity |
| 22 | `TU_ERR_SRAM_UNDERFLOW` | Access below SRAM base |
| 23 | `TU_ERR_BANK_CONFLICT` | Unresolvable bank conflict |

### DMA (30-39)

| Code | Name | Use Case |
|------|------|----------|
| 30 | `TU_ERR_DMA_OVERFLOW` | Transfer exceeds buffer |
| 31 | `TU_ERR_DMA_INVALID_CHANNEL` | Bad channel number |
| 32 | `TU_ERR_DMA_INVALID_DESC` | Malformed descriptor |
| 33 | `TU_ERR_DMA_QUEUE_FULL` | Descriptor queue full |
| 34 | `TU_ERR_DMA_TIMEOUT` | Transfer timed out |

### Command Queue (40-49)

| Code | Name | Use Case |
|------|------|----------|
| 40 | `TU_ERR_QUEUE_FULL` | Command queue full |
| 41 | `TU_ERR_QUEUE_EMPTY` | Command queue empty |
| 42 | `TU_ERR_CMD_NOT_FOUND` | Command ID not found |
| 43 | `TU_ERR_CMD_TIMEOUT` | Command timed out |
| 44 | `TU_ERR_DEPENDENCY_CYCLE` | Circular dependency |

### Compute (50-59)

| Code | Name | Use Case |
|------|------|----------|
| 50 | `TU_ERR_COMPUTE_INVALID_OP` | Unknown opcode |
| 51 | `TU_ERR_COMPUTE_DIM_MISMATCH` | Tensor shape mismatch |
| 52 | `TU_ERR_COMPUTE_OVERFLOW` | Numeric overflow |
| 53 | `TU_ERR_COMPUTE_UNDERFLOW` | Numeric underflow |

### Data Types (60-69)

| Code | Name | Use Case |
|------|------|----------|
| 60 | `TU_ERR_DTYPE_UNSUPPORTED` | Unsupported data type |
| 61 | `TU_ERR_DTYPE_CONVERSION` | Conversion failure |
| 62 | `TU_ERR_NAN_ENCOUNTERED` | NaN in data |
| 63 | `TU_ERR_INF_ENCOUNTERED` | Inf in data |

### Interconnect (70-79)

| Code | Name | Use Case |
|------|------|----------|
| 70 | `TU_ERR_ICC_NO_ROUTE` | No route between cores |
| 71 | `TU_ERR_ICC_BUFFER_FULL` | ICC buffer full |
| 72 | `TU_ERR_ICC_TIMEOUT` | ICC transfer timeout |

### Internal (80-89)

| Code | Name | Use Case |
|------|------|----------|
| 80 | `TU_ERR_INTERNAL` | Internal logic error |
| 81 | `TU_ERR_NOT_IMPLEMENTED` | Feature stub |
| 82 | `TU_ERR_ASSERTION_FAILED` | Assertion violation |

### Verification (90-99)

| Code | Name | Use Case |
|------|------|----------|
| 90 | `TU_ERR_VERIFY_MISMATCH` | Golden ref mismatch |
| 91 | `TU_ERR_VERIFY_TOLERANCE` | Error tolerance exceeded |

---

## Error Reporting Macros

### TU_REPORT_ERR

Report an error with automatic file/line/function capture.

```c
if (ptr == NULL)
    return TU_REPORT_ERR(TU_ERR_NULL_POINTER, "buffer pointer is NULL");
```

Output (LOG mode):
```
[TU ERROR] file.c:42 in my_func(): buffer pointer is NULL (code=12: null pointer)
```

### TU_ASSERT

Assert a condition; return `TU_ERR_ASSERTION_FAILED` on failure.

```c
TU_ASSERT(size > 0, "size must be positive");
TU_ASSERT(addr % 4 == 0, "address must be word-aligned");
```

### TU_CHECK

Check a condition and return a specific error code.

```c
TU_CHECK(core != NULL, TU_ERR_NULL_POINTER, "core is null");
TU_CHECK(idx < max_idx, TU_ERR_OUT_OF_RANGE, "index out of bounds");
```

### TU_RETURN_IF_ERR

Forward an error from a called function.

```c
tu_status_t s = tu_dma_submit(dma, desc);
TU_RETURN_IF_ERR(s);
// ... continue on success
```

### TU_ERROR_INJECT

Error injection point — returns injected error if one is active.

```c
tu_status_t my_function(void) {
    TU_ERROR_INJECT();  // Inject error here during testing
    // ... normal code
}
```

---

## Error Behavior Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `TU_ERR_MODE_LOG` | Print error to stderr, return error code | **Default** — CI, development |
| `TU_ERR_MODE_ABORT` | Print error to stderr, call abort() | Strict mode — mimics old behavior |
| `TU_ERR_MODE_SILENT` | Return error code only, no output | Batch processing, fuzzing |

```c
// Set at startup:
tu_set_error_mode(TU_ERR_MODE_LOG);     // default
tu_set_error_mode(TU_ERR_MODE_ABORT);   // strict
tu_set_error_mode(TU_ERR_MODE_SILENT);  // quiet
```

---

## Error Injection

The error injection framework enables testing error recovery paths without modifying production code.

```c
// Inject a DMA timeout at a specific call site
tu_error_inject_enable("dma_descriptor.c", 445, TU_ERR_DMA_TIMEOUT);

// The next call to TU_ERROR_INJECT() at that site returns TU_ERR_DMA_TIMEOUT
// (one-shot: automatically deactivated after firing)

// Disable all injections
tu_error_inject_disable_all();
```

---

## Migration Guide

### Before (old code)

```c
void tu_dma_load(tu_dma_channel_t ch, tu_sram_region_t *dst,
                 uint32_t offset, const void *host, uint32_t bytes) {
    if (offset + bytes > dst->total_size) {
        fprintf(stderr, "DMA load overflow\n");
        abort();  // ❌ Process dies immediately
    }
    memcpy(dst->banks.data + offset, host, bytes);
}
```

### After (new code)

```c
void tu_dma_load(tu_dma_channel_t ch, tu_sram_region_t *dst,
                 uint32_t offset, const void *host, uint32_t bytes) {
    if (offset + bytes > dst->total_size) {
        fprintf(stderr, "DMA load overflow\n");
        TU_REPORT_ERR(TU_ERR_DMA_OVERFLOW, "DMA load exceeds SRAM capacity");
        return;  // ✅ Graceful early return, error is logged
    }
    memcpy(dst->banks.data + offset, host, bytes);
}
```

### For new functions returning tu_status_t

```c
tu_status_t tu_dma_submit_safe(tu_dma_engine_t *dma, const tu_dma_descriptor_t *desc) {
    TU_CHECK(dma != NULL, TU_ERR_NULL_POINTER, "dma engine is null");
    TU_CHECK(desc != NULL, TU_ERR_NULL_POINTER, "descriptor is null");
    TU_CHECK(desc->total_bytes > 0, TU_ERR_INVALID_PARAM, "zero-size transfer");

    // ... processing ...

    return TU_OK;
}
```

---

## Verification

### Test Coverage (9 tests)

| Test | Description | Status |
|------|-------------|--------|
| TEST 1 | All error codes have valid strings | ✅ |
| TEST 2 | `tu_is_ok()`, `tu_is_err()` helpers | ✅ |
| TEST 3 | `TU_REPORT_ERR` context capture | ✅ |
| TEST 4 | Error modes (LOG, ABORT, SILENT) | ✅ |
| TEST 5 | `TU_ASSERT` macro | ✅ |
| TEST 6 | `TU_CHECK` macro | ✅ |
| TEST 7 | `TU_RETURN_IF_ERR` propagation | ✅ |
| TEST 8 | Error injection framework | ✅ |
| TEST 9 | All 40 error codes have strings | ✅ |

**9/9 pass.**

### Run tests

```bash
make test-errors
```

### All 5 abort() calls replaced

| File | Line | Old | New |
|------|------|-----|-----|
| `tu_sram.c` | 132 | `abort()` on SRAM overflow | `TU_REPORT_ERR(TU_ERR_SRAM_OVERFLOW)` + return |
| `tu_cmodel.c` | 122 | `abort()` on DMA bounds | `TU_REPORT_ERR(TU_ERR_SRAM_OVERFLOW)` + return |
| `tu_cmodel.c` | 162 | `abort()` on uninit MMA | `TU_REPORT_ERR(TU_ERR_NOT_INITIALIZED)` + return |
| `dma_descriptor.c` | 613 | `abort()` on DMA load overflow | `TU_REPORT_ERR(TU_ERR_DMA_OVERFLOW)` + return |
| `dma_descriptor.c` | 629 | `abort()` on DMA store overflow | `TU_REPORT_ERR(TU_ERR_DMA_OVERFLOW)` + return |

---

## Limitations & Future Work

| Limitation | Priority | Notes |
|------------|----------|-------|
| No per-core error context | P2 | `g_last_error` is global; multi-core needs per-core error state |
| No error recovery paths | P2 | Currently errors cause early return; recovery (retry, fallback) not implemented |
| No error rate limiting | P3 | Repeated errors in loops could flood stderr |
| No structured log format | P3 | Currently fprintf; JSON/structured logging for CI parsing |
| Void functions can't propagate | P2 | Many functions return void; full return-code migration is ongoing |
| Error injection limited to file+line | P3 | Could support pattern-based or count-based injection |

---

## References

- **Gap:** E5 in `docs/PRODUCTION_TU_REDESIGN.md` (Section 3.6)
- **Related:** Q2 (logging), V6 (differential testing), I3 (observability)
- **Design:** Section 5.9 — Code Quality & Infrastructure
