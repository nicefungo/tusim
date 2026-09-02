# DMA Direction-Specific Burst-Issue Cost Alternatives

**Date:** 2026-09-02
**Status:** Implemented for live descriptors, queued projection, legacy wrappers, and direct O-buffer loads
**Question:** When DMA read and write paths have unequal command pipelines, should one per-burst issue cost remain mandatory, or should the cmodel preserve symmetric, read-overlapped, and write-overlapped alternatives?

## Hypothesis and realistic alternatives

The executable maximum-burst model previously used one `burst_issue_cycles` value for both directions. Separate load/store movers, write-combine logic, masked stores, read prefetch, and unequal CDC/protocol pipelines can make visible command issue asymmetric even when payload width and burst geometry are identical.

| Alternative | Why a hardware team may choose it | Principal sacrifice |
|---|---|---|
| Symmetric 2/2 cycles | Shared address/control pipeline or deliberately uniform timing contract | Cannot represent a pipeline optimized for one dominant direction |
| Read-overlapped 0/4 cycles | Deeply pipelined operand prefetch, with a simpler or serialized result-write command path | More load-side queue/credit logic; stores retain higher command latency |
| Write-overlapped 4/0 cycles | Coalesced output drain or dedicated write command path, with simpler input issue | More write-combine/queue state; loads retain higher command latency |

`burst_issue_cycles` remains the common compatibility setting. `read_burst_issue_cycles` and `write_burst_issue_cycles` are optional directional overrides. Presence bits distinguish an explicit zero-cycle override from an absent field that inherits the common value; this is required to compare a fully overlapped direction against a nonzero common cost without changing memset-zeroed callers.

## Executable matrix

```sh
make test-dma-directional-issue-sweep
```

The harness performs byte-exact 96-byte loads and stores on a 256-bit path with 32-byte bursts, a 50-cycle directional base, and SRAM metering disabled. Each request has three bursts. Completion includes the initial start tick:

`completion = 1 + 50 + ceil(96 / 32) + 3 × directional_issue_cycles`

| Read issue | Write issue | Load completion | Store completion |
|---:|---:|---:|---:|
| 2 cycles/burst | 2 cycles/burst | 60 cycles | 60 cycles |
| 0 cycles/burst | 4 cycles/burst | 54 cycles | 66 cycles |
| 4 cycles/burst | 0 cycles/burst | 66 cycles | 54 cycles |

The asymmetric rows exchange six cycles of directional completion around the symmetric row. They do not change the three payload-serialization cycles, burst count, or 96 useful bytes. Neither asymmetric mode is universally preferable: each lowers latency only for the direction whose issue is modeled as fully overlapped.

## Implementation path

`YAML/JSON dma.{burst_issue_cycles,read_burst_issue_cycles,write_burst_issue_cycles} → generator constants/runtime fields and presence bits → canonical defaults/parser/validation → canonical-to-runtime conversion → tu_init_with_config() → g_tu_dma → live descriptor service and least-projected-cycle queued estimates`.

Legacy `tu_dma_load()` and `tu_dma_store()` select their corresponding directional cost. The direct `tu_dma_load_o()` accounting path selects the read cost. Existing initializers call the additive directional initializer with override-presence false, so both directions inherit the common value. Canonical JSON parsing marks a directional field present even when its value is zero.

## Gain versus sacrifice

- **Throughput:** A fully overlapped issue path can reduce command-rate pressure in a direction with many short bursts. This sweep measures isolated deterministic service, not steady-state issue throughput, command queue occupancy, or compute overlap.
- **Latency:** Exact for the tested base + payload serialization + non-overlapped directional issue formula. The measured 54/60/66-cycle reversal is restricted to 96-byte requests with three 32-byte bursts.
- **Area/resources:** A zero-visible-cost path is expected to need enough pipelining, buffering, or credits to hide issue. Unequal paths may save resources in the less provisioned direction. Registers, queues, ports, and physical area are unquantified.
- **Power/energy:** Hiding issue can require more active queue/control state, while fewer exposed command cycles may reduce control occupancy. Command and buffer energy coefficients are absent, so net energy is unquantified.
- **SRAM/DRAM traffic:** Useful bytes, burst count, and byte-exact data are unchanged. This DMA model does not round occupied traffic; DRAM fixed-burst occupancy remains a separate contract.
- **Numerical accuracy:** Unchanged; every load/store row is byte-identical.
- **Control complexity:** Symmetric hardware needs one timing register and path. Directional mode adds two optional values, selection, and presence state so explicit zero is not confused with inheritance.
- **Verification burden:** Both directions, symmetric inheritance, explicit zero, nonzero asymmetry, live/queued/legacy/direct paths, parser/generator propagation, and invalid values require independent gates.
- **Compiler/runtime:** Target metadata can expose unequal command rates for descriptor sizing and load/store scheduling. Software must not infer that a load-optimized issue path also benefits output drains.

## Fidelity limits

Functional movement remains aggregate `memcpy`; burst segmentation affects timing accounting only. The model does not represent command FIFO depth, command/data overlap explicitly, finite credits, address alignment or protocol boundaries, retries, write combining, masks, queue backpressure, shared SRAM/DRAM limits, producer issue timing, or calibration. A zero visible issue cost is an overlap/lower-bound abstraction, not proof that physical command generation consumes no area, energy, or time.

## Verification

```sh
make test-dma-directional-issue-sweep
make test-dma-burst-issue-sweep
make test-dma-directional-burst-sweep
make test-dma
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.directional-issue.h
make config-docs
make clean && make
make test-quick
```
