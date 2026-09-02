# DMA Direction-Specific Maximum Burst Alternatives

**Date:** 2026-09-01
**Status:** Implemented for live descriptors, queued projection, and legacy wrappers
**Question:** When read and write movers have different protocol, buffering, or merge constraints, should one common maximum burst remain mandatory, or should the cmodel preserve symmetric, read-wide, and write-wide alternatives?

## Hypothesis and alternatives

The prior implementation made maximum burst size executable but used one value for both loads and stores. That is appropriate for a symmetric mover, but cannot represent several plausible designs:

| Alternative | Why a hardware team may choose it | Principal sacrifice |
|---|---|---|
| Symmetric 64/64 B | Shared read/write address generator, equal buffering, simple software contract | Cannot tune command pressure independently for operand loads and result drains |
| Read-wide 128/32 B | Long contiguous weight/activation fetches, but fine-grained or masked result writes | Larger read credits/buffers; store path issues more commands |
| Write-wide 32/128 B | Fine input gathers but coalesced output/writeback path | Larger write-combine storage; load path issues more commands |

`max_burst_bytes` remains the common compatibility setting. `read_max_burst_bytes` and `write_max_burst_bytes` are optional directional overrides; zero inherits the common value. All nonzero limits must be powers of two from 16 through 65,536 bytes. This sweep fixes both directional issue costs at two cycles so it isolates geometry; the later `dma-directional-issue-cost.md` exploration independently makes issue cost direction-selectable.

## Executable matrix

```sh
make test-dma-directional-burst-sweep
```

The harness performs byte-exact 96-byte loads and stores on a 256-bit path with a 50-cycle directional base, two visible issue cycles per burst, and SRAM metering disabled. Completion includes the initial start tick:

`completion = 1 + 50 + ceil(96 / 32) + ceil(96 / directional_burst) × 2`

| Read max | Write max | Load bursts | Load completion | Store bursts | Store completion |
|---:|---:|---:|---:|---:|---:|
| 64 B | 64 B | 2 | 58 cycles | 2 | 58 cycles |
| 128 B | 32 B | 1 | 56 cycles | 3 | 60 cycles |
| 32 B | 128 B | 3 | 60 cycles | 1 | 56 cycles |

The asymmetric rows exchange two cycles of directional completion in this deliberately short, command-visible request. They do not alter the three payload-serialization cycles or the 96 useful bytes. Read-wide is therefore not “better” than write-wide: each helps the direction provisioned with the larger burst and hurts the opposite direction under the tested issue-cost model.

## Implementation path

`YAML/JSON dma.{max_burst_bytes,read_max_burst_bytes,write_max_burst_bytes,burst_issue_cycles} → generator constants/runtime fields → canonical defaults/parser/validation → canonical-to-runtime conversion → tu_init_with_config() → g_tu_dma → live descriptor service and least-projected-cycle queued estimates`.

Legacy `tu_dma_load()` and `tu_dma_store()` now use the corresponding directional limit. The direct `tu_dma_load_o()` accounting path uses the read limit. Existing initializers route through `tu_dma_init_config_burst()` and inherit one common value; the additive `tu_dma_init_config_directional_burst()` exposes independent values. A zero directional runtime field inherits the common limit, preserving old and memset-zeroed callers.

The checked-in generated header was updated additively after a temporary generator audit found unrelated generator drift: regenerating the entire header would erase existing manually integrated constants. The temporary header nevertheless compiled and contained both new directional fields/macros.

## Gain versus sacrifice

- **Throughput:** A direction with visible command-rate pressure can sustain more useful bytes per command with a larger maximum. This sweep measures isolated completion, not steady-state throughput, queue occupancy, or compute overlap.
- **Latency:** Exact only for the tested base + payload serialization + non-overlapped issue model. The measured 56/58/60-cycle reversal demonstrates directional sensitivity; gains scale with request length and issue cost.
- **Area/resources:** Read-wide can require larger read-data/credit state; write-wide can require larger write-combine, mask, or response state. Separate limits require two registers and direction selection. Physical area is unquantified.
- **Power/energy:** Fewer commands in one direction should reduce address/control toggles per useful byte, while larger buffers can increase leakage and switched capacitance. No energy coefficients are wired, so net energy is unquantified.
- **SRAM/DRAM traffic:** Useful traffic is unchanged and byte exact. DMA maximum burst does not round occupancy or model overfetch; DRAM fixed-burst occupancy is a separate subsystem contract.
- **Numerical accuracy:** Unchanged; all load/store rows are byte-identical.
- **Control complexity:** Symmetric mode permits one shared limit. Directional mode adds selection and independent protocol constraints. Split issue-pipeline costs are modeled separately rather than conflated with this geometry sweep.
- **Verification burden:** Both directions, inheritance, asymmetric reversals, live and legacy paths, queued projections, parser/generator propagation, and invalid values need gates.
- **Compiler/runtime:** Target metadata can expose separate load/store limits for descriptor shaping or output coalescing. The compiler must not assume a read-optimal split is also write-optimal.

## Fidelity limits

Functional copies remain aggregate `memcpy` operations. Burst segmentation affects timing accounting only. The model does not include address alignment, 4 KiB boundaries, per-row segmentation for strided descriptors, masks, write combining, explicit command/data overlap, finite credits, retries, queue backpressure, shared SRAM/DRAM bandwidth, protocol legality, or calibration. Directional issue costs now exist as a separate additive abstraction, but do not close those physical gaps. Area, power, and energy directions above are qualitative expectations, not cmodel measurements.

## Verification

```sh
make test-dma-directional-burst-sweep
make test-dma-burst-issue-sweep
make test-dma
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make config-docs
make clean && make
make test-quick
```
