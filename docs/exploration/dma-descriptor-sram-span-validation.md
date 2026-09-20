# DMA Descriptor SRAM Span Validation

**Date:** 2026-09-20
**Mode:** pre-spec model-correctness audit
**Evidence:** `tests/test_dma.c` — `Descriptor SRAM spans reject out-of-range geometry atomically`

## Architecture question

Which SRAM capacity contract should descriptor execution enforce: aggregate useful bytes, or every address actually touched by linear, strided, indexed, and multicast geometry?

The prior implementation checked `base + total_bytes <= region_size` only during execution. That is valid for contiguous linear transfers but not for descriptors containing holes. A two-row transfer can report 32 useful bytes while touching `[16,32)` and `[56,72)` because of its 40-byte SRAM row stride. Against a 64-byte region, the aggregate check accepts the descriptor even though the second row is out of range. Scatter/gather indices and multicast destination offsets had the same structural gap; asynchronous submission could admit invalid work before the late execution check.

## Alternatives considered

| Contract | Hardware/model rationale | Decision |
|---|---|---|
| Aggregate useful-byte bound | Smallest host-side check; valid only when addressing is contiguous or a separate address generator proves every generated address | **Excluded for general descriptors.** It permits accesses beyond the modeled SRAM region and therefore is not a physically plausible safe execution contract. |
| Exact addressed-span validation | Mirrors a fail-closed command processor, IOMMU, or scratchpad range checker; validates the final row/slice, every S/G index, and every multicast destination | **Implemented.** It is required model correctness, not an optional performance mode. |
| Per-beat runtime fault after partial transfer | Could model hardware that discovers faults while issuing requests | **Blocked.** The cmodel has no partial-fault status, completion-signal, rollback, or retirement contract. Silently copying a prefix would be misleading. |

No runtime knob was added: retaining an unsafe permissive mode would allow undefined host memory access and could not represent a valid hardware alternative without explicit partial-fault semantics.

## Executable contract

Before direct execution, the cmodel now validates the actual SRAM span:

- linear: `base + total_bytes`;
- 2D: `base + (rows - 1) * row_stride + row_bytes`;
- 3D: `base + (depth - 1) * depth_stride + (rows - 1) * row_stride + row_bytes`;
- scatter/gather: every `index[i] + element_bytes`;
- multicast: every `destination_offset[i] + payload_bytes`.

All additions and multiplications use overflow-checked 64-bit arithmetic. Exact-end spans remain legal. Submission validates every descriptor in a chain before channel binding, queue-depth mutation, or `total_submitted` accounting. An invalid chain is rejected atomically and follows the existing rejected-submission ownership rule. Direct execution leaves the descriptor incomplete and all DMA counters unchanged.

The old aggregate execution checks were removed because they could both miss sparse address overruns and reject a legal overlapping-stride descriptor whose useful-byte sum exceeds its addressed extent.

## Verification matrix

The focused gate first failed against the prior implementation, then passed after the validator was added. It covers:

| Shape/surface | Invalid geometry | Required result |
|---|---|---|
| Direct 2D execution | 64-byte region, base 16, two 16-byte rows, 40-byte stride | no copy, incomplete descriptor, zero transfer accounting |
| Async 2D submission | same geometry | return 0; no queue/accounting mutation |
| Async 3D submission | final depth/row exceeds region | reject atomically |
| Scatter | 8-byte element at index 60 in 64-byte SRAM | reject atomically |
| Gather | 8-byte element at index 60 in 64-byte SRAM | reject atomically |
| Indexed metadata mismatch | copy width is 8 bytes but mutable index width is changed to 1 | reject rather than under-validating the copy |
| TU-to-TU | 2D source overruns while destination fits; either required endpoint is null | validate both SRAM spans and reject missing endpoints before admission |
| Multicast | null target or one 8-byte destination at offset 60 in 64-byte SRAM | reject the full descriptor |
| Descriptor chain | valid head plus invalid strided tail | reject the whole chain before copying the head |
| Exact boundary control | rows `[0,16)` and `[48,64)` | accept and copy byte-exactly |

## Gain versus sacrifice

| Dimension | Result |
|---|---|
| Functional correctness | Prevents out-of-region strided, indexed, and multicast SRAM accesses. |
| Throughput/latency | Valid descriptors retain the same modeled transfer cycles and bytes. Host validation adds unmodeled simulator overhead only; no hardware-cycle penalty is claimed. |
| Area/resources | A physical implementation that performs equivalent dynamic checking needs range comparators and address-generation checks. A compiler-proven design could omit them, but then the compiler/ISA verifier owns the same invariant. Area is unquantified. |
| Power/energy | Valid transfer traffic is unchanged. Dynamic hardware checking would add small control activity; unquantified. |
| SRAM/DRAM traffic | Invalid descriptors now produce zero traffic; valid useful and occupied bytes are unchanged. |
| Numerical accuracy | Valid byte movement is unchanged. Invalid accesses no longer corrupt unrelated state. |
| Control complexity | Submission must inspect geometry and a chain before admission. Partial-fault/rollback machinery is deliberately absent. |
| Verification burden | Adds shape-specific boundary, overflow, chain-atomicity, and exact-end controls. |
| Compiler/runtime | Producers receive immediate submission failure for invalid SRAM geometry. Rejection destroys the submitted chain, matching the existing API behavior now documented in `dma_descriptor.h`; borrowed host/index/multicast storage must survive accepted execution. Producers must not rely on aggregate useful bytes as a capacity proof. |

## Fidelity limits

The check validates modeled SRAM offsets only. Host pointers remain opaque process virtual addresses; their allocation bounds and explicit external/bus placement are separate contracts. The model does not implement per-beat faults, page protection, partial completion, rollback, canceled signals, or hardware range-check latency. Descriptor constructor multiplication overflow and host-memory span validation remain separate audit dimensions.

## Verification commands

```sh
make test-dma
make test-dma-burst-boundary-sweep test-dma-external-boundary-sweep
make clean && make
make test-quick
```
