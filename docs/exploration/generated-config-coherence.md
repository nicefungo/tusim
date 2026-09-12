# Generated TU configuration coherence

**Date:** 2026-09-12
**Mode:** pre-spec model-correctness audit
**Question:** Can `config/tu_config.yaml` still regenerate the checked-in C configuration without deleting executable architecture choices or changing their numeric contracts?

## Hypothesis

A pre-spec cmodel needs one reproducible configuration source. The checked-in header had accumulated manually integrated dataflow, precision, memory-hierarchy, DRAM, logging, sparsity, and runtime fields after the generator was introduced. If regeneration removes those definitions, YAML alternatives are not a safe architecture interface: a routine configuration update can either fail the build or silently change an enum/bit contract.

Three maintenance choices were considered:

| Choice | Why a team might choose it | Gain | Sacrifice |
|---|---|---|---|
| Manual checked-in header | Fastest way to integrate one module | No generator work during feature development | YAML is not authoritative; regeneration can erase unrelated architecture support |
| Partial/additive generation | Limits immediate churn when generator coverage is incomplete | New fields can be added without replacing known-good definitions | Permanent dual ownership; omissions remain easy and exact reproducibility is unavailable |
| Complete deterministic generation | Appropriate when YAML is the declared source of truth | One repeatable architecture snapshot; drift becomes testable | Every compile-time definition must have a YAML source or an explicit generated availability constant; generator changes carry broader verification burden |

The complete deterministic path is implemented. Manual and partial workflows are not runtime hardware modes and therefore are not retained as selectable cmodel alternatives.

## Baseline evidence

Command:

```sh
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.h
diff -u tu_cmodel/tu_config.h /tmp/tu_config.h
```

The baseline generated header omitted **34 checked-in macros**, introduced three obsolete dataflow names, and reported 19 differing macro definitions. Important semantic hazards included:

- deleting the active plug-in dispatch selector and WS/OS/RS/NLR mode names;
- reusing FP8 E5M2 bit positions for INT8 and shifting INT4;
- deleting RegFile/GBUF geometry used by the memory hierarchy;
- deleting DRAM type/bandwidth/channel defaults;
- deleting logging/trace limits and integer-quantization availability definitions;
- reverting checked-in structured-sparsity availability.

Several other differences were numerically equivalent symbolic-vs-literal defaults, but they still showed that the generated artifact was not byte-reproducible.

## Implementation

The YAML source now carries the previously manual architecture inputs:

- dataflow plug-in dispatch and NLR naming;
- RegFile and GBUF geometry;
- logging and trace-buffer defaults;
- INT8/INT4 implementation availability and INT8 accumulator width;
- structured 2:4 module availability.

The generator now emits the complete live macro inventory, including stable precision bits, DRAM presets, stochastic-rounding naming, memory hierarchy, and observability constants. The checked-in `tu_cmodel/tu_config.h` is regenerated from that source.

`tests/test_generated_config.py` and `make test-config-generation` enforce:

1. byte identity between a fresh generated header and the checked-in header;
2. a nondefault configuration covering OS dataflow, 2 MiB GBUF, HBM3, four DRAM channels, debug logging, and stochastic FP16 rounding;
3. top-level C compilation with the generated nondefault header, catching include-order or macro/enum collisions;
4. fail-closed rejection of an unsupported dataflow name.

The target is part of both `make test-quick` and the aggregate `make test`, so future architecture work cannot silently reintroduce generator drift.

## Results

| Gate | Result |
|---|---|
| Default YAML → checked header | Byte-identical |
| Nondefault generated header | OS / 2 MiB GBUF / HBM3 / 4 channels / log level 4 / stochastic rounding asserted |
| Top-level compile | Pass with `-std=c11 -Wall -Wextra -Werror -fsyntax-only` |
| Unsupported dataflow | Rejected |
| Canonical JSON/runtime config | 31/31 tests pass |
| Dataflow | 10/10 tests pass |
| Memory hierarchy | 10/10 tests pass |
| INT8/INT4 quantization | 14/14 tests pass |
| Structured 2:4 | 27/27 tests pass |
| Core cmodel | 19/19 tests pass |

No cycle or throughput benchmark is reported because this correction changes configuration provenance, not an executing datapath. The generated defaults preserve the prior checked-in macro values, and focused engine tests verify no numerical or functional regression.

## Multi-objective interpretation

- **Throughput and latency:** unchanged for the default build. The correction prevents accidental architecture changes during regeneration but adds no modeled cycles.
- **Area/resources:** unchanged in modeled hardware. Generator/test code adds host-side repository surface only.
- **Power/energy:** unchanged and not measured; no activity or power-model equation changed.
- **SRAM/DRAM traffic:** unchanged. GBUF/SRAM/DRAM defaults are now reproducible rather than manually retained.
- **Numerical accuracy:** unchanged in focused cmodel, dataflow, integer, and sparse tests. Stable precision-bit assignments prevent future mode aliasing.
- **Control complexity:** hardware control is unchanged. Build/configuration control becomes simpler because YAML and generated C no longer have split ownership.
- **Verification burden:** increases by one fast generation/compile gate, while reducing the larger burden of manually auditing deleted or renumbered macros after every regeneration.
- **Compiler/runtime implications:** generated enum names and bit positions now remain compatible with current C consumers. This does not prove that every compile-time YAML field is runtime-selectable; canonical JSON propagation and runtime consumption remain separate evidence layers.

## Limitations

The generator remains a minimal parser for this repository's YAML subset, not a general YAML implementation. The gate covers the shipped source plus one discriminating alternative matrix; it does not exhaust every value combination. Compile-time availability fields do not replace runtime configuration tests, and physical timing, area, power, and calibration remain outside this provenance correction.

## Verification commands

```sh
make test-config-generation
make test-config test-cmodel test-dataflow test-memhier test-int-quant test-sparsity
make clean && make
make test-quick
```
