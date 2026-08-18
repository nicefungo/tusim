# DRAM Direction-Specific Burst Granularity

**Date:** 2026-08-18
**Question:** Should a rounded fixed-burst interface require the same minimum occupancy granule for reads and writes, or should the cmodel preserve asymmetric read/write contracts?

## Hypothesis and realistic alternatives

One shared 64 B granule is a useful compatibility abstraction, but it is not the only physically plausible contract. A read path may fetch a complete line while a masked/coalesced write path occupies a smaller granule; conversely, write-combining or media-write constraints can make writes coarser than reads. The pre-spec model should preserve all three materially distinct alternatives:

| Alternative | Why a hardware team might choose it | Main sacrifice |
|---|---|---|
| Symmetric granule | One burst contract, simpler controller, interface, compiler rules, and verification | Cannot represent direction-specific masks, coalescers, or media constraints |
| Read-wide / write-narrow | Full-line read datapath plus masked or write-combined partial stores | More write-mask/merge state and direction-dependent software rules |
| Read-narrow / write-wide | Fine read service plus coarse write-combining or fixed media writes | Write buffering, delayed visibility, and larger write amplification |
| Exact-byte control | Byte-enable/coalesced fabric where occupied bytes equal payload bytes | More irregular transfer formation and mask/control complexity |

No mode is selected as universally best. `burst_round_credit` now uses independently configurable `read_burst_bytes` and `write_burst_bytes`; all exact-byte and compatibility turnaround modes continue to count requested bytes.

## Executable model

For rounded mode only:

```text
read_occupied  = ceil(read_payload  / read_burst_bytes)  * read_burst_bytes
write_occupied = ceil(write_payload / write_burst_bytes) * write_burst_bytes
burst_cycles   = ceil(direction_occupied / bus_width_bytes)
payload_efficiency = useful_read_write_bytes / occupied_read_write_bytes
```

The selected direction's occupied bytes feed the same completion boundary, pending-byte counters, coarse bandwidth budget, occupied-bandwidth metrics, and payload-efficiency metric. Granules are nonzero powers of two up to 1 MiB. Canonical zero means “inherit the DRAM preset,” preserving zero-initialized callers; shipped defaults remain symmetric 64 B.

## Measured matrix

Command: `make test-dram-directional-burst-sweep`

Configuration: custom one-channel/one-bank model; 8 B/cycle bus; 10-cycle read and 8-cycle write base service; R→W/W→R costs 3/8 cycles; 16 B request in each direction; 20-cycle issue gap. “Service” is the sum of returned base-plus-turnaround service and excludes the separate coarse bandwidth-window stall domain.

| Contract | Direction | Read / write granule | Service cycles | Residual turnaround | Read / write occupied | Payload efficiency |
|---|---|---:|---:|---:|---:|---:|
| Symmetric | R→W | 64 / 64 B | 19 | 1 | 64 / 64 B | 25.0% |
| Symmetric | W→R | 64 / 64 B | 22 | 4 | 64 / 64 B | 25.0% |
| Read-wide | R→W | 128 / 32 B | 21 | 3 | 128 / 32 B | 20.0% |
| Read-wide | W→R | 128 / 32 B | 18 | 0 | 128 / 32 B | 20.0% |
| Write-wide | R→W | 32 / 128 B | 18 | 0 | 32 / 128 B | 20.0% |
| Write-wide | W→R | 32 / 128 B | 26 | 8 | 32 / 128 B | 20.0% |
| Exact-byte control | R→W | 128 / 32 B configured | 18 | 0 | 16 / 16 B | 100.0% |
| Exact-byte control | W→R | 32 / 128 B configured | 18 | 0 | 16 / 16 B | 100.0% |

All eight rows fail closed on returned service, direction-specific occupied bytes, and payload efficiency. The reversal is intentional: widening the first request's direction retains the channel longer and reduces idle credit for the following direction. The exact-byte controls prove granules do not leak into other turnaround modes.

## Gain versus sacrifice

- **Throughput:** For the measured 16 B pair, symmetric 64 B occupancy uses 128 B; either 128/32 asymmetric contract uses 160 B. Sustainable throughput is unquantified because there is no request queue, arbitration, or beat-level schedule.
- **Latency:** At a 20-cycle gap, read-wide raises measured R→W service from 19 to 21 cycles but lowers W→R to 18 because its write granule is narrow. Write-wide reverses the effect, lowering R→W to 18 and raising W→R to 26. These are deterministic completion-boundary effects, not end-to-end memory latency.
- **Area/resources:** Symmetry can share sizing/control. Narrow writes may require byte masks, merge buffers, or read-modify-write support; wide writes may require combining buffers. Narrow reads may need burst chopping or sub-line return logic. Area is unquantified.
- **Power/energy:** Wider occupied granules directionally increase interface switching and likely DRAM activation/data energy per useful byte. The physical energy model is not wired to occupied bytes, so energy remains unquantified.
- **SRAM/DRAM traffic:** Logical payload is unchanged. Modeled interface occupancy is direction-specific; the model does not determine whether excess bytes enter SRAM, are masked, or arise from internal read-modify-write traffic.
- **Numerical accuracy:** Unchanged; no arithmetic or stored value semantics change.
- **Control complexity:** Asymmetry adds two size registers and direction-dependent rounding. Real masked writes or write combining would add substantially more state than this accounting model represents.
- **Verification burden:** Every direction, aligned/sub-granule/tail requests, exact-byte controls, setter rejection, zero/default inheritance, reset, counters, completion timing, bandwidth budget, and derived metrics require gates.
- **Compiler/runtime:** Tensor layout and DMA coalescing can target each direction independently, but asymmetric contracts require direction-aware alignment, safe-overfetch, mask, and batching rules. Compiler-generated request traces are still needed for workload-level conclusions.

## Implementation and configuration paths

- `tu_cmodel/memory/dram_model.{h,c}`: direction-specific state, validated public setter, selected-direction occupancy, and canonical propagation.
- `tu_cmodel/infra/config.{h,c}`: defaults, JSON parsing, validation, generated docs, and zero-means-preset compatibility.
- `config/tu_config.{yaml,json}`, `scripts/gen_config.py`, `tu_cmodel/tu_config.h`: shipped and generated settings `read_burst_bytes` / `write_burst_bytes`.
- `tests/test_dram.c`: runtime behavior, invalid-setter immutability, bandwidth/statistics wiring, parse-to-runtime propagation, and zero-field inheritance.
- `tests/test_config.c`: parse and invalid-granule rejection.
- `tests/test_dram_directional_burst_sweep.c`: eight-row exact trade-off matrix.

## Fidelity limits

This is occupancy and returned-service accounting, not a JEDEC burst-chop, byte-mask, cache-line-fill, write-combining, or read-modify-write implementation. It omits request queues, arbitration, coalescing, dependencies, command/address and data phasing, ranks, bank groups, backpressure, PHY timing/energy, SRAM side effects, and calibration. Power, area, throughput, and makespan implications are therefore qualitative or unquantified.

## Verification

Focused commands executed:

```text
make test-dram-directional-burst-sweep
make test-dram
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make config-docs
```

The final clean build, quick regression, and compatibility sweep are recorded in the heartbeat report.

## Actionable conclusion

Preserve symmetric, read-wide/write-narrow, read-narrow/write-wide, and exact-byte contracts. The measured 20-cycle-gap matrix proves that asymmetric granularity changes both occupancy efficiency and which direction retains turnaround cost: read-wide penalizes R→W while write-wide penalizes W→R. Hardware teams should choose using request-size/direction distributions, mask/coalescer resources, power goals, and compiler alignment capability—not one local latency row.
