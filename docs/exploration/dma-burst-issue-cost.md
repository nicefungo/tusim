# DMA Burst Segmentation and Issue-Cost Alternatives

**Date:** 2026-08-31
**Status:** Implemented for live descriptor service and coarse queued projection
**Question:** Can the advertised maximum DMA burst size represent address/control issue cost, and when does a larger burst reduce descriptor latency enough to justify a more demanding interface contract?

## Re-audit and hypothesis

`tu.dma.max_burst_bytes` existed in YAML, JSON, generated constants, the canonical config, and generated documentation, but it was neither validated nor copied into `tu_runtime_config_t`, and live descriptor service ignored it. Changing the setting therefore had no executable effect.

A maximum burst size cannot change payload serialization by itself: 4,096 bytes still requires 128 cycles on a 32-byte/cycle path. To avoid inventing a hidden cost, the model now separates two runtime settings:

- `max_burst_bytes`: the largest payload represented by one abstract DMA burst;
- `burst_issue_cycles`: non-overlapped address/control issue cost charged per burst.

The zero issue-cost mode is intentionally retained. It represents the historical coarse model or a controller that fully pipelines/overlaps burst issue. Nonzero costs represent controllers where address generation, command acceptance, CDC, or descriptor-side setup consumes visible service cycles.

| Alternative | Why hardware may choose it | Principal sacrifice |
|---|---|---|
| 32 B burst, 1 issue cycle | Small buffers, narrow protocol limit, simple boundary handling | Many commands and high control/CDC service cost on large tensors |
| 64 B burst, 1 issue cycle | Moderate command granularity and compatibility with the shipped burst setting | Intermediate command count and interface state |
| 128 B burst, 1 issue cycle | Amortize issue cost for long contiguous transfers | Larger burst buffering/credit scope, potentially longer blocking and harder alignment integration |
| 64 B burst, 0 issue cycles | Fully overlapped issue or legacy lower-bound abstraction | Cannot expose command-rate pressure; optimistic for serialized issue hardware |

The cmodel accepts power-of-two burst limits from 16 through 65,536 bytes and issue costs from 0 through 1,024 cycles. These bounds support exploration; they are not claims that every combination maps to a legal external protocol.

## Executable matrix

```sh
make test-dma-burst-issue-sweep
```

The harness executes one byte-exact 4,096-byte load on a 256-bit independent path with a 50-cycle read base and SRAM metering disabled. Completion includes the start tick:

`completion = 1 + 50 + ceil(4096 / 32) + ceil(4096 / max_burst) × issue_cycles`

| Max burst | Issue cycles/burst | Burst count | Completion cycle | Change vs zero-cost compatibility |
|---:|---:|---:|---:|---:|
| 64 B | 0 | 64 | 179 | baseline |
| 32 B | 1 | 128 | 307 | +71.5% |
| 64 B | 1 | 64 | 243 | +35.8% |
| 128 B | 1 | 32 | 211 | +17.9% |

At one visible issue cycle per burst, moving from 32 B to 64 B lowers this isolated completion by 20.8%, and moving from 32 B to 128 B lowers it by 31.3%. The gain is command-rate amortization, not added payload bandwidth: every row moves exactly 4,096 useful bytes over the same 32-byte/cycle datapath.

## Implementation path

`YAML/JSON dma.max_burst_bytes + dma.burst_issue_cycles → generated constants → tu_config_t defaults/parser/validation → tu_runtime_config_t → tu_init_with_config() → g_tu_dma → live descriptor service, least-projected-cycle queued estimates, and legacy/direct DMA accounting`.

`tu_dma_init_config_burst()` is the additive full initializer. Existing initialization APIs retain the checked-in 64-byte maximum and zero visible issue cost. A zero runtime maximum selects the checked-in burst default, preserving genuinely zero-initialized callers; explicit zero remains valid for the issue cost.

## Gain versus sacrifice

- **Throughput:** Command-limited sustained throughput may improve with larger bursts or more deeply pipelined issue, but this harness measures one descriptor completion. Shared memory bandwidth, producer timing, and overlap are absent.
- **Latency:** Exact only in the tested base + payload serialization + non-overlapped per-burst issue model. Benefits grow with transfer size and visible issue cost; short requests pay at most one issue term.
- **Area/resources:** Larger bursts are expected to require broader credit windows, larger burst buffers, or more retained address/count state. More aggressive overlap can require command queues or multiple address-generation slots. Physical area is unquantified.
- **Power/energy:** Fewer commands should reduce address/control toggles per useful byte, while larger buffers and wider active state can increase leakage and switched capacitance. No physical energy coefficients are connected, so net energy is unquantified.
- **SRAM/DRAM traffic:** Useful bytes are unchanged. This DMA model does not round payload occupancy up to the maximum burst and therefore does not claim overfetch. DRAM burst occupancy is modeled separately in the DRAM subsystem.
- **Numerical accuracy:** Unchanged; transfer data is byte exact.
- **Control complexity:** Zero-cost mode needs no command-rate state. Nonzero issue cost adds burst counting; real overlap would additionally need command acceptance, credits, and backpressure.
- **Verification burden:** Defaults, zero compatibility, parser propagation, power-of-two validation, exact ceiling behavior, both live and queued estimates, legacy wrappers, and interactions with topology/binding require gates.
- **Compiler/runtime:** A compiler can favor long contiguous descriptors only when target burst and issue settings are exposed. Splitting descriptors can change command count without changing payload bytes; dependency and QoS behavior remain separate contracts.

## Fidelity limits

The implementation segments **timing accounting**, not the functional `memcpy`, into abstract bursts. It does not model address alignment, 4 KiB protocol boundaries, beat masks, separate read/write burst limits, command/data channel overlap, coalescing, retries, finite credits, SRAM row boundaries, DRAM commands, or queue backpressure. Strided/scatter/gather descriptors use aggregate transferred bytes rather than per-row or per-index burst boundaries. The issue term is uncalibrated and should be treated as an architecture-exploration parameter, not a measured AXI or DRAM command latency.

## Verification

```sh
make test-dma-burst-issue-sweep
make test-dma
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make config-docs
make clean && make
make test-quick
```
