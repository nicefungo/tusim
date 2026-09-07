# DMA Base-Latency Scope Exploration

**Date:** 2026-09-06

**Question:** Should a strided or scatter/gather descriptor pay its configured DMA base latency once, or should each discontiguous logical segment restart that latency?

## Hypothesis and realistic alternatives

The existing model charged read/write base latency once per descriptor, even when `burst_segmentation=logical_segments` proved that each row or indexed element issues independent burst commands. That is one plausible controller, but not the only plausible implementation.

- **`descriptor` (compatibility default):** a buffered/pipelined descriptor sequencer amortizes address translation, CDC, arbitration, and setup across all rows or indices. A hardware team may choose this for high strided throughput, accepting descriptor state, address-generation pipelines, buffering, and more coupled verification.
- **`logical_segments`:** every 2D row, 3D row, or scatter/gather element restarts the aggregate base-latency path. A hardware team may choose this conservative model for a small sequencer or for an interface where discontinuities require independent transactions. It sacrifices discontiguous-transfer latency but reduces the assumptions needed to claim descriptor-wide setup overlap.

Linear descriptors have one logical segment, so both modes are intentionally identical. The new scope is independent of burst segmentation: command counting and base-latency restart are separate contracts.

## Configuration and model

Runtime JSON/YAML field:

```json
{"tu":{"dma":{"base_latency_scope":"descriptor"}}}
```

Allowed values are `descriptor` and `logical_segments`. The zero/default value is `descriptor`, preserving generated defaults, `tu_config_default()`, old initializer APIs, and zero-initialized runtime callers.

For a nonempty descriptor with directional base latency `L`:

- descriptor scope: `base_cycles = L`
- logical scope: `base_cycles = L × S`

where `S=1` for linear descriptors, rows for 2D, depth×rows for 3D, and index count for scatter/gather. Payload serialization remains `ceil(total_bytes / bus_width)`, and burst issue cost remains `burst_count × directional_issue_cycles`. The same helper feeds live completion and queued least-projected-cycle binding.

For compatibility, descriptor scope still charges one base latency to a zero-byte DMA descriptor, matching the pre-existing descriptor-engine behavior. Logical-segment scope has zero segments and therefore no base term. Producer-level empty-descriptor rejection remains a separate lifecycle contract.

## Executable matrix

Command:

```sh
make test-dma-base-scope-sweep
```

Configuration: 256-bit path, 64-byte maximum bursts, three issue cycles per burst, 50-cycle base latency, logical burst segmentation, SRAM bandwidth metering disabled. Async completion includes the initial issue tick.

| Descriptor | Useful bytes | Logical segments | Descriptor scope | Logical-segment scope | Added cycles | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 80 | 1 | 60 | 60 | 0 | 1.00× |
| Strided 2D, 4×20 B | 80 | 4 | 66 | 216 | 150 | 3.27× |
| Strided 3D, 2×3×20 B | 120 | 6 | 73 | 323 | 250 | 4.42× |
| Gather, 5×4 B | 20 | 5 | 67 | 267 | 200 | 3.99× |

The harness also gates a strided store, exact byte movement, malformed config rejection, default compatibility, and a queued projected-binding reversal. Under descriptor scope the 4×20 B strided queue is projected below a 512 B linear queue; under logical scope the linear queue becomes the lower projected-cost target.

## Multi-objective interpretation

- **Throughput/latency:** descriptor amortization materially lowers the measured service time of many short discontiguous segments. It does not improve linear traffic and is not a universal throughput result.
- **Area/resources:** descriptor scope is expected to need more persistent sequencer/address state, buffering, and potentially translation/credit resources. Logical scope can represent a smaller restart-oriented controller. No area model is wired, so magnitudes are unquantified.
- **Power/energy:** fewer visible setup restarts should reduce control activity, while the more capable pipelined controller has leakage/clocking cost. Neither effect is quantified.
- **SRAM/DRAM traffic:** useful bytes and copied values are identical. Alignment overfetch and external protocol occupancy remain unmodeled.
- **Numerical accuracy:** unchanged; the modes affect timing only, and byte-exact movement is gated.
- **Control complexity:** descriptor scope has the higher expected implementation complexity; logical scope exposes a simple conservative timing contract.
- **Verification burden:** descriptor-wide overlap requires proving state retention and legal progress across row/index boundaries. Logical restart is easier locally but creates more events and corner cases at scale.
- **Compiler/runtime:** compilers can prefer linear/coalesced descriptors when logical restart cost is selected. Descriptor scope rewards larger strided descriptors but requires the runtime's descriptor limits and controller contract to match the model.

## Limitations

This is deterministic base-plus-serialization-plus-command accounting, not a queue-aware memory controller. It does not model TLB/page walks, address alignment, protocol boundaries, descriptor fetch, finite command credits, SRAM replay, DRAM bank timing, command/data overlap, physical area/power, or calibration. `logical_segments` charges the configured aggregate base term per segment; it does not decompose which physical stages actually restart. Multicast fanout remains outside this scope because the current logical-segmentation contract does not classify destinations as independent segments.

## Verification

```sh
make test-dma-base-scope-sweep test-config test-dma
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make clean && make
make test-quick
```
