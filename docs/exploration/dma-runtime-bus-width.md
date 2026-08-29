# Runtime DMA Bus-Width Alternatives

**Date:** 2026-08-29
**Status:** Implemented for the live descriptor engine
**Question:** Does the advertised 32–1024-bit DMA width actually reach live descriptor service, and what latency/resource trade-off is exposed by representative 128/256/512-bit datapaths?

## Re-audit and hypothesis

`tu.dma.bus_width_bits` already existed in YAML, JSON, the canonical config, validation, and generated compile-time constants. The canonical-to-runtime conversion dropped it, however, and `dma_descriptor.c` always serialized payloads with `TU_DMA_BUS_WIDTH_BYTES`. A parsed nondefault width therefore did not change the live descriptor engine.

The hardware hypothesis is that narrow, baseline, and wide movers are all plausible candidate chips:

| Width | Why a hardware team might choose it | Principal sacrifice |
|---:|---|---|
| 128 bit | Lower datapath/port/routing width for area-, pin-, or energy-constrained designs | More payload cycles and greater pressure to overlap DMA with compute |
| 256 bit | Existing compatibility point; moderate integration width | Intermediate resource demand and latency |
| 512 bit | Feed larger PE arrays or bandwidth-heavy tiles with fewer serialization cycles | Wider SRAM/NoC/DRAM interfaces, harder timing/routing, more switching and verification |

The cmodel preserves the full validated power-of-two range from 32 through 1024 bits. It does not select one width universally.

## Executable experiment

```sh
make test-dma-bus-width-sweep
```

The harness transfers one 4,096-byte linear descriptor through one independent channel. SRAM bandwidth metering is disabled to isolate the DMA serialization term. The live model uses `50 + ceil(bytes / width_bytes)` service cycles; completion includes the tick on which service starts. It gates JSON parse → canonical config → runtime config → top-level initialization → live engine state, zero-runtime-field compatibility, exact byte movement, exact completion cycles, and invalid-width rejection.

| Width | Bytes/cycle | Payload cycles | Completion cycle |
|---:|---:|---:|---:|
| 128 bit | 16 | 256 | 307 |
| 256 bit | 32 | 128 | 179 |
| 512 bit | 64 | 64 | 115 |

For this isolated transfer, 256 bit lowers completion latency 41.7% versus 128 bit, and 512 bit lowers it 35.8% versus 256 bit (62.5% versus 128 bit). The absolute result is specific to one 4 KiB descriptor, a 50-cycle base term, one independently serviceable abstract path, and disabled SRAM metering. Byte traffic and numerical results are unchanged.

## Implementation and configuration path

`tu.dma.bus_width_bits` now follows:

`YAML/JSON → generated/default header → tu_config_t → tu_runtime_config_t → tu_init_with_config() → tu_dma_init_config_arch() → g_tu_dma.bus_width_bytes → live and queued descriptor service`.

The new architecture-aware initializer validates direct callers. Existing DMA initializer signatures remain compatibility wrappers and retain the checked-in 256-bit default. A zero `tu_runtime_config_t.dma_bus_width_bits` also selects that compile-time default so legacy zero-initialized runtime callers do not acquire an invalid zero-byte divisor. `tu_dma_load_o()` uses the same live width for its direct serialization estimate.

## Gain versus sacrifice

- **Throughput:** Wider service can raise throughput only when DMA serialization is on the critical path and the surrounding SRAM/NoC/DRAM system can sustain the width. This harness measures one isolated completion, not steady-state throughput.
- **Latency:** Exact under the descriptor engine's base-plus-serialization abstraction for the tested transfer. Benefits shrink when fixed latency, SRAM stalls, compute, dependencies, or a shared bottleneck dominate.
- **Area/resources:** Expected direction is higher datapath, mux, FIFO, SRAM-port, and routing cost with width. Physical area and timing are unquantified.
- **Power/energy:** Wider logic may complete sooner but toggles more wires and may require stronger clocking/ports. Dynamic and leakage energy are not wired to this path, so net energy is unquantified.
- **SRAM/DRAM traffic:** Useful bytes remain exactly 4,096. Burst overfetch, DRAM command timing, shared port limits, and fabric flits are not represented by this descriptor-width term.
- **Numerical accuracy:** Unchanged; the transfer is byte exact and performs no arithmetic.
- **Control complexity:** The arithmetic change is small, but a physical wider interface can require segmentation, alignment, lane masks, CDC, and flow control. Those mechanisms are unmodeled.
- **Verification burden:** Every width must retain ceiling behavior for tails, defaults, parse propagation, invalid rejection, byte movement, and interactions with topology/binding. The focused gate covers aligned 4 KiB service; tail/alignment protocol behavior remains a separate contract.
- **Compiler/runtime:** Compilers can use the width for tiling and overlap decisions only when all downstream cycle producers consume the same runtime contract. Explicit channel topology and binding remain independent knobs.

## Fidelity limits

This implementation makes width executable in the descriptor engine rather than claiming a physical AXI/DRAM interface. Independent channels still represent abstract concurrent paths; the width does not prove replicated ports. SRAM refill penalties are stateful and intentionally omitted from queued projected-cycle estimates. Compute-engine analytical estimators that directly use compile-time macros are separate surfaces and are not silently relabeled as runtime-width evidence. No calibration, burst protocol, queue-aware DRAM controller, backpressure, alignment masking, physical area, or physical energy model is added.

## Verification

```sh
make test-dma-bus-width-sweep
make test-dma
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make config-docs
make clean && make
make test-quick
```
