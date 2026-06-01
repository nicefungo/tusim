# Runtime JSON Configuration System (Gap A1)

> **Status:** Implemented | **Version:** 1.0 | **Date:** 2026-06-01

## Overview

The TU cmodel now supports runtime configuration via JSON files, replacing the
previous compile-time-only `#define` constants in `tu_config.h`. This enables:

- **Design space exploration** — sweep PE array sizes, SRAM capacities, data
  types, and DMA parameters without recompilation
- **Multi-instance operation** — each `tu_core_t` can be initialized with its
  own configuration
- **A/B testing** — compare cycle-accurate vs functional models, different
  dataflows, precision modes, and memory banking strategies

## Architecture

```
config/tu_config.json  ──►  tu_cmodel/infra/json_reader.c  ──►  tu_json_value_t
                                      │
                                      ▼
                          tu_cmodel/infra/config.c  ──►  struct tu_config_t
                                      │
                                      ▼
                          tu_init_from_config()    ──►  tu_runtime_config_t
                                      │
                                      ▼
                          tu_init_with_config()   ──►  tu_state_t (g_tu)
```

### Components

| File | Purpose |
|------|---------|
| `tu_cmodel/infra/json_reader.h/c` | Minimal, dependency-free recursive-descent JSON parser. Handles all standard JSON types including nested objects, arrays, escaped strings, and `\uXXXX` Unicode. ~400 lines of C |
| `tu_cmodel/infra/config.h/c` | Full configuration struct (`tu_config_t`) with 70+ parameters spanning compute, memory, DMA, ISA, multi-core, performance, sparsity, precision, and verification. Loads from JSON, applies defaults, validates constraints |
| `config/tu_config.json` | Canonical JSON configuration file matching the existing `config/tu_config.yaml` |
| `tu_cmodel/tu_config.h` | Legacy compile-time defaults — now used as fallback when no JSON config is loaded. Renamed `tu_config_default()` → `tu_runtime_config_default()` |

## Configuration Parameters

All parameters have sensible defaults matching the original TinyTU specification
(16×16 PE, 256 KB SRAM, FP16, weight-stationary). See `config/tu_config.json`
for the complete schema.

Key sections:
- **compute**: PE array dimensions, dataflow mode, pipeline depth
- **memory**: SRAM buffer sizes, banking configuration, DRAM type/bandwidth
- **dma**: Bus width, channels, async mode, burst size
- **isa**: Instruction width, command queue depth
- **multicore**: Core count, interconnect topology
- **performance**: Cycle model fidelity, counter detail
- **precision**: FP16/BF16/FP8/INT8 control, rounding modes
- **sparsity**: 2:4 structured sparsity, unstructured sparsity
- **verification**: Golden reference format, test iterations, error tolerance

## Usage

### Loading from a JSON file

```c
#include "tu_cmodel.h"
#include "infra/config.h"

int main(void) {
    char err_buf[256];
    if (tu_init_from_file("config/tu_config.json", err_buf, sizeof(err_buf)) != 0) {
        fprintf(stderr, "Config error: %s\n", err_buf);
        return 1;
    }

    // Use normal API: tu_dma_load_w(), tu_mma(), etc.
    // PE array dimensions, SRAM sizes, and all other parameters
    // are driven by the JSON config.
}
```

### Loading from a string

```c
tu_config_t cfg;
tu_config_default(&cfg);
cfg.pe_rows = 32;
cfg.pe_cols = 64;
tu_init_from_config(&cfg);
```

### Programmatic config

```c
tu_config_t cfg;
tu_config_load("my_config.json", &cfg, err_buf, sizeof(err_buf));

// Read back:
printf("PE array: %u×%u\n", cfg.pe_rows, cfg.pe_cols);
printf("SRAM: %u KB\n", cfg.sram_w_size_kb + cfg.sram_a_size_kb + cfg.sram_o_size_kb);
```

## Validation

The config loader validates all parameters before initialization:

- PE rows/cols: [1, 1024]
- SRAM buffer sizes: > 0 KB
- Bank width: must be 1, 2, 4, or 8 bytes
- Bank count: [1, 1024]
- DMA bus width: power of 2 in [32, 1024]
- Command queue depth: > 0

Invalid configurations return clear error messages.

## Backward Compatibility

Existing code using `tu_init()` continues to work unchanged — it initializes
with compile-time defaults from `tu_config.h`. The new `tu_init_from_file()`
and `tu_init_from_config()` are additive.

## Testing

18 tests in `tests/test_config.c`:
- 9 JSON parser tests (primitives, arrays, objects, nesting, errors)
- 1 config defaults test
- 1 config-from-string test
- 1 config-from-file test
- 3 config validation tests (pass, fail on bad rows, fail on bad bank width, fail on bad bus)
- 1 config-to-runtime conversion test
- 1 end-to-end test (init from config → MMA → verify output)

## Gap Reference

This addresses gap **A1** (Configurability) from `docs/PRODUCTION_TU_REDESIGN.md`:
> Parameterized: arbitrary PE array dims, SRAM sizes, bus widths; YAML/JSON
> config file at cmodel startup. Critical priority, P0.

The implementation provides JSON configuration as the primary runtime format,
with compile-time `#define` constants as fallback defaults.
