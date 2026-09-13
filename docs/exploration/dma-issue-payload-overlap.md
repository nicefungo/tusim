# DMA Command-Issue / Payload Overlap

**Date:** 2026-09-13
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_issue_payload_overlap_sweep.c`

## Architecture question

Should a DMA descriptor serialize burst-command issue with payload movement, or should it use a pipelined command front end that can issue later bursts while earlier payload beats are moving?

Both contracts are physically plausible:

- **`serialized` (default):** command/address work and payload movement occupy one modeled service path. This is conservative and represents a small controller, a blocking bridge, or an implementation whose command and data stages cannot run concurrently.
- **`overlapped`:** command generation and payload movement use independently progressable stages after the descriptor's base setup. This represents a pipelined mover with enough command buffering or credits to keep both stages active.

The existing zero-cycle burst issue setting remains a valid fully hidden lower-cost alternative. The new mode matters only when a nonzero issue cost is selected; it does not replace directional issue-cost, burst-size, segmentation, base-scope, or payload-scope alternatives.

## Executable model

For one descriptor:

```text
base_cycles    = configured read/write base latency × configured base scope
payload_cycles = configured descriptor-packed or logical-segment-aligned beats
issue_cycles   = configured burst count × directional issue cycles per burst

serialized = base_cycles + payload_cycles + issue_cycles
overlapped = base_cycles + first_issue +
             max(payload_cycles, remaining_issue_cycles)
```

The same composition helper feeds live descriptor completion, queued least-projected-cycle binding, and legacy linear load/store accounting. Stateful SRAM refill stalls remain outside queued projection and are added only by live service.

Compatibility is explicit: `serialized` is enum value zero and is selected by generated defaults, canonical defaults, old initializer wrappers, and zero-initialized runtime callers.

## Measured matrix

Configuration: one independent DMA channel, 256-bit / 32-byte payload interface, 50-cycle descriptor base latency, SRAM bandwidth metering disabled. Completion includes the asynchronous start tick. Linear loads are byte-checked.

| Useful bytes | Max burst | Issue cycles/burst | Payload cycles | Issue cycles | Serialized completion | Overlapped completion | Completion reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 4 | 1 | 4 | 56 | 56 | 0.00% |
| 96 | 32 | 4 | 3 | 12 | 66 | 63 | 4.55% |
| 512 | 32 | 4 | 16 | 64 | 131 | 115 | 12.21% |
| 512 | 128 | 1 | 16 | 4 | 71 | 68 | 4.23% |

A fragmented 2D control uses six 5-byte rows, logical burst segmentation, and logical payload alignment. It has 30 useful bytes, 192 occupied interface bytes, six payload cycles, and six 4-cycle command issues. Completion is 81 cycles serialized and 75 cycles overlapped. The store direction produces the same overlapped timing and byte-exact output.

The sweep also gives a discriminating queued-policy case. With 128-byte bursts and four issue cycles:

- channel 0 holds six 5-byte logical rows: coarse service is 80 cycles serialized and 74 overlapped;
- channel 1 holds a 480-byte linear descriptor: coarse service is 81 cycles serialized and 69 overlapped.

Least-projected-cycle binding therefore selects channel 0 in serialized mode but channel 1 in overlapped mode. This proves that queued planning consumes the selected contract rather than only changing the live completion report.

## Gain versus sacrifice

| Dimension | Serialized | Overlapped |
|---|---|---|
| Throughput | Lower when both command and payload costs are material; exact queue-aware throughput is not modeled | Can sustain the slower stage rather than their sum; measured benefit grows when both stages carry appreciable work |
| Descriptor latency | Additive and conservative | First-command issue remains causal; later issue overlaps payload. Reduction is 0–12.21% in the measured linear matrix |
| Area/resources | Expected smaller command state and fewer decoupling buffers | Expected command FIFO/credits, independent stage state, and completion tracking; unquantified |
| Power/energy | Less speculative control/storage activity, but longer active time | More front-end/FIFO/control activity; shorter active time may reduce leakage. Net energy is unquantified |
| SRAM/DRAM traffic | Identical useful and occupied bytes | Identical useful and occupied bytes; this is temporal overlap, not traffic reduction |
| Numerical accuracy | Identical byte movement | Identical byte movement |
| Control complexity | Blocking sequencing and simpler completion reasoning | Concurrent stage progress, buffer-full handling, and stronger ordering/backpressure contracts |
| Verification burden | One additive timeline | Must verify fill/drain, finite credits, stalls, direction changes, cancellation, and completion ordering once those mechanisms exist |
| Compiler/runtime | Conservative estimates; fewer hardware scheduling assumptions | Runtime/compiler must know that command capacity exists; projected channel binding can change even with identical descriptors |

No mode is universally preferred. A low-area MCU-class accelerator or bridge may accept serialized latency. A bandwidth-oriented TU that already provisions descriptor queues and credits may choose overlap, especially for many short bursts. Single-burst transfers cannot benefit because the first command must precede payload. Large bursts with cheap issue remain payload-dominated, so overlap saves less in absolute command work and should not be used to justify extra buffering without workload evidence.

## Implementation paths

- `config/tu_config.yaml`, `config/tu_config.json`: `tu.dma.issue_payload_mode`
- `scripts/gen_config.py`, `tu_cmodel/tu_config.h`: generated compile/runtime defaults
- `tu_cmodel/infra/config.{h,c}`: canonical enum, parsing, validation, runtime conversion, generated docs
- `tu_cmodel/dma_descriptor.{h,c}`: executable enum, compatibility initializer, common cycle composition
- `tu_cmodel/tu_cmodel.c`: top-level runtime propagation
- `tests/test_config.c`: defaults, parse, propagation, and invalid-value gates
- `tests/test_dma_issue_payload_overlap_sweep.c`: measured functional/cycle matrix and fail-closed checks

## Verification

```bash
make test-dma-issue-payload-overlap-sweep
make test-config test-dma
make clean && make
make test-quick
```

The focused sweep gates linear and fragmented transfers, load/store symmetry, useful/occupied bytes, zero-byte behavior, old initializer and zero-runtime compatibility, legacy load/store accounting, queued projected binding, JSON-to-live propagation, and unsupported-mode rejection.

## Fidelity limits

This is deterministic descriptor service accounting, not an AXI/NoC/DRAM queue simulator. It does not model finite command FIFO depth, command/data arbitration, memory response ordering, credit return latency, backpressure, burst alignment or protocol boundaries, command coalescing, shared SRAM/DRAM bandwidth, cancellation, or calibrated area/power. Overlap is ideal after the first command is issued: no later bubbles are introduced. These omissions must be modeled before interpreting the result as sustained system throughput or physical energy.
