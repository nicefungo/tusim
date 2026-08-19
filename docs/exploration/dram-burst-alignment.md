# DRAM Burst Alignment and Address-Span Occupancy

**Date:** 2026-08-19
**Question:** Should a fixed-burst interface round only request payload size, or count every address-aligned protocol burst touched by an unaligned request?

## Hypothesis and realistic alternatives

The existing `burst_round_credit` model computes `ceil(payload/granule)`. That is valid when DMA requests are guaranteed burst-aligned or when a packet fabric pads by length without an address-aligned fetch boundary. It understates occupancy for a DRAM/cache-line-style interface when a request crosses a granule boundary. Both contracts are physically plausible and useful:

| Mode | Hardware rationale | Main sacrifice |
|---|---|---|
| `burst_credit` | Exact-byte streaming/byte-enable control | Irregular transfer formation, masks, or coalescing |
| `burst_round_credit` | Payload-size-rounded packets, or fixed bursts under an enforced alignment contract | Incorrect for unaligned address-span fetches unless software alignment is guaranteed |
| `burst_span_credit` | Counts every address-aligned fixed burst touched; conservative for line/burst fetch interfaces | Unaligned requests amplify traffic and retain the bus longer; address decode enters occupancy accounting |

The new mode is additive. Numeric zero/default remains `none`, and the former `burst_round_credit` semantics remain unchanged for compatibility.

## Executable model

For direction-specific power-of-two granule `G`, address `A`, and payload bytes `N`:

```text
exact_occupied = N
size_rounded_occupied = ceil(N / G) * G
span_rounded_occupied = ceil(((A mod G) + N) / G) * G
serialized_cycles = ceil(occupied / bus_width_bytes)
```

The selected occupied bytes feed completion-boundary idle credit, logical-versus-occupied counters, pending bytes, the coarse bandwidth budget, occupied bandwidth/utilization, and payload efficiency. Logical request bytes remain unchanged.

## Measured matrix

Command: `make test-dram-burst-alignment-sweep`

Configuration: one channel/bank, 64 B granule, 8 B/cycle bus, read/write base service 10/8 cycles, R→W/W→R turnaround 3/8 cycles, and a 20-cycle issue gap. Service is returned base-plus-turnaround service; the separate coarse bandwidth stall domain is excluded.

| Contract | Direction | Address | Payload | First-request occupied | Pair service | Residual turnaround |
|---|---|---:|---:|---:|---:|---:|
| Size-rounded aligned | R→W | 0 | 64 B | 64 B | 19 | 1 |
| Size-rounded unaligned | R→W | 1 | 64 B | 64 B | 19 | 1 |
| Span-rounded aligned | R→W | 0 | 64 B | 64 B | 19 | 1 |
| Span-rounded unaligned | R→W | 1 | 64 B | 128 B | 21 | 3 |
| Span-rounded tail | R→W | 63 | 80 B | 192 B | 21 | 3 |
| Size-rounded unaligned | W→R | 1 | 64 B | 64 B | 22 | 4 |
| Span-rounded unaligned | W→R | 1 | 64 B | 128 B | 26 | 8 |
| Exact-byte control | R→W | 63 | 80 B | 80 B | 21 | 3 |

All eight rows fail closed on occupancy and service. The discriminating 64 B request at address 1 demonstrates the missing behavior: size rounding charges one 64 B transfer, while address-span rounding correctly touches two. At address 63, an 80 B span touches three bursts (192 B), versus 128 B under size-only rounding and 80 B exact.

## Gain versus sacrifice

- **Throughput:** For the measured unaligned 64 B request, span mode consumes 2× the occupied interface bytes; the 80 B request at address 63 consumes 2.4× payload bytes. Sustainable throughput is unquantified because requests are not queued or scheduled beat by beat.
- **Latency:** Added occupied serialization retains turnaround cost. Measured R→W pair service rises 19→21 cycles and W→R rises 22→26 for an address-1 64 B first request. These are deterministic service-accounting effects, not calibrated end-to-end memory latency.
- **Area/resources:** Span accounting needs low address bits plus add/round logic. Real support for unaligned accesses may also require split-request state, merge buffers, or two response tags. Gate/buffer area is unquantified.
- **Power/energy:** More touched bursts directionally increase interface and likely DRAM/cache-line energy. Physical power is not wired to occupied bytes, so energy remains unquantified.
- **SRAM/DRAM traffic:** Logical tensor payload is unchanged. Modeled interface traffic rises at boundary crossings. The cmodel does not say whether overfetched bytes enter SRAM or are discarded/masked.
- **Numerical accuracy:** Unchanged; payload values and arithmetic semantics are untouched.
- **Control complexity:** Size rounding is simpler and valid with compiler-enforced alignment. Span mode is safer for arbitrary addresses but implies split-boundary handling in real control logic.
- **Verification burden:** Both directions, aligned/misaligned/sub-burst/tail spans, direction-specific granules, exact controls, overflow-safe arithmetic, config parsing, counters, reset, completion timing, and bandwidth accounting require gates.
- **Compiler/runtime:** Alignment-aware DMA tiling can avoid amplification but may require prologue/body/epilogue requests, padding, safe overfetch, or tensor-layout changes. A hard alignment requirement simplifies hardware while restricting allocators and graph lowering.

## Implementation and configuration

- `tu_cmodel/memory/dram_model.{h,c}`: adds `TU_DRAM_TURNAROUND_BURST_SPAN_CREDIT`; occupancy includes `addr mod granule` only in that mode.
- `tu_cmodel/infra/config.{h,c}`: canonical enum, JSON string `burst_span_credit`, validation, and generated reference text.
- `scripts/gen_config.py`, `tu_cmodel/tu_config.h`, `config/tu_config.yaml`: generated/default configuration path.
- `tests/test_dram.c`: runtime retention, misaligned occupancy, and completion-boundary propagation.
- `tests/test_config.c`: executable JSON parse gate.
- `tests/test_dram_burst_alignment_sweep.c`: eight-row trade-off matrix.

## Fidelity limits

This is deterministic occupancy and returned-service accounting. It does not split one logical request into independently decoded channel/bank/row operations. It omits burst chopping, byte masks, coalescing, queues, arbitration, command/address/data phasing, bank groups, ranks, request dependencies, backpressure, response merging, SRAM side effects, physical energy, and calibration. A request whose span crosses a channel or row boundary is therefore charged the right number of granules but still receives one coarse access's decode/service; queue-aware makespan and row-command cost remain unmodeled.

## Verification

Executed:

```text
make test-dram-burst-alignment-sweep
make test-dram
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make config-docs
```

The final clean build, quick regression, and compatibility sweeps are recorded in the heartbeat report.

## Actionable conclusion

Preserve all three occupancy contracts. Use exact bytes for byte-enable/streaming studies, size rounding when hardware or compiler guarantees burst alignment, and address-span rounding when arbitrary addresses can cross fixed protocol boundaries. The measured unaligned cases show that size-only rounding can undercount occupied traffic by 2× for a nominally aligned-size request and by 1.5× versus size rounding for the selected 80 B tail. No mode is universally best: hardware split/merge resources, software alignment guarantees, traffic amplification, and verification budget determine the appropriate contract.
