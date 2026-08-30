# DMA Directional Base-Latency Alternatives

**Date:** 2026-08-30
**Status:** Implemented for live descriptor service
**Question:** Do configured DRAM read/write base latencies reach DMA loads and stores independently, including physical-nanosecond conversion?

## Re-audit and hardware hypothesis

The canonical config already accepted separate `memory.latency.dram_read` and `dram_write` values plus `memory.dram.latency_domain`, but canonical-to-runtime conversion dropped them. The descriptor engine charged `TU_LATENCY_DRAM_READ` to every direction, including SRAM-to-host stores. Thus asymmetric controller/PHY contracts could not be explored and projected channel binding also used the wrong base term for stores.

Three physically plausible contracts are retained:

| Mode | Read / write base | Why it may exist | Principal sacrifice |
|---|---:|---|---|
| symmetric | 50 / 50 cycles | Balanced controller or compatibility abstraction | Cannot represent directional pipeline asymmetry |
| read-fast | 30 / 70 cycles | Inference-oriented design prioritizing weight/activation fetch | Slower result/writeback service and possible output backpressure |
| write-fast | 70 / 30 cycles | Streaming producer or checkpoint path prioritizing drains | Slower operand fetch and greater preload pressure |

These labels describe sweep points, not new enums: read and write latency remain independent runtime values in either core-cycle or physical-ns source domains.

## Executable matrix

```sh
make test-dma-directional-latency-sweep
```

One 4,096-byte descriptor runs on a 256-bit independent path with SRAM metering disabled. The live formula is `completion = 1 start tick + directional base + 128 payload cycles`.

| Mode | Load completion | Store completion |
|---|---:|---:|
| symmetric 50/50 | 179 | 179 |
| read-fast 30/70 | 159 | 199 |
| write-fast 70/30 | 199 | 159 |

The asymmetric rows exchange exactly 40 cycles between load and store; they do not change bytes or batch work. A parse gate also proves `15.25 ns read / 35.5 ns write` at 2 GHz converts with `ceil(ns × GHz)` to 31 / 71 live cycles. Zero-initialized runtime callers use an explicit validity bit and retain checked-in 50/50 compatibility defaults; canonical callers may still explicitly model zero latency.

## Implementation path

`YAML/JSON memory latency + domain + core clock → tu_config_t → ceil conversion in tu_config_to_runtime() → tu_runtime_config_t directional cycles → tu_init_with_config() → g_tu_dma → descriptor live service, queued projected-cycle binding, and legacy load/store accounting`.

Existing DMA initialization APIs preserve compile-time defaults. `tu_dma_init_config_timing()` is the additive architecture-aware entry point.

## Gain versus sacrifice

- **Throughput:** Not measured. Directional base latency affects small/request-serial regimes; steady-state throughput may instead be limited by bus width, memory bandwidth, queueing, or overlap.
- **Latency:** Exact only under the tested base-plus-serialization descriptor model. Read-fast lowers the measured load 11.2% versus symmetric while raising store 11.2%; write-fast reverses that exchange.
- **Area/resources:** Asymmetric pipelines may allocate buffers, credits, or controller stages differently. Physical area is unquantified.
- **Power/energy:** Faster directional service can shorten residence time, but controller/PHY energy and leakage are not connected. Net energy is unquantified.
- **SRAM/DRAM traffic:** Exactly 4,096 useful bytes in every row. Burst overfetch, commands, and shared bandwidth are outside this descriptor term.
- **Numerical accuracy:** Unchanged; all transfers are byte-exact.
- **Control complexity:** Two timing registers and direction selection are modest in the cmodel; a physical asymmetry can imply distinct read/write datapaths and CDC behavior.
- **Verification burden:** Both directions, cycle/ns conversion, fractional-ns ceiling, explicit zero, compatibility defaults, and projected estimates need independent gates.
- **Compiler/runtime:** A scheduler can prioritize loads or drains only if it consumes directional timing. Explicit channel binding and QoS remain separate choices.

## Fidelity limits

This is deterministic descriptor-service accounting, not a DRAM queue or calibrated controller. It does not model command/data phasing, write combining, read/write turnaround interaction, burst alignment, backpressure, ranks, shared SRAM/DRAM contention, voltage, physical area, or energy. `TU_TO_TU` retains the read-side compatibility base because the model has no separate internal-copy contract.

## Verification

```sh
make test-dma-directional-latency-sweep
make test-dma
make test-config
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.generated.h
make clean && make
make test-quick
```
