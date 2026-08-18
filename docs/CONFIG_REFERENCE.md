# TU CModel Configuration Reference

> Auto-generated from current configuration.
> Each field shows its **value**, type, and description.

---

## 1. Compute Engine

| Field | Value | Type | Description |
|-------|-------|------|-------------|
| `pe_rows` | 16 | uint16 | PE array height (rows) |
| `pe_cols` | 16 | uint16 | PE array width (columns) |
| `pe_pipeline_depth` | 2 | uint16 | Pipeline stages per MAC |
| `mac_units_per_pe` | 1 | uint16 | MAC units per PE |
| `dataflow_mode` | 0 | int | 0=WS, 1=OS, 2=RS (NLR reserved, not executable) |
| `dataflow_via_plugin` | `true` | bool | Use pluggable dataflow dispatcher |

## 2. Precision & Data Types

| Field | Value | Type | Description |
|-------|-------|------|-------------|
| `fp16_enabled` | `true` | bool | IEEE 754 half-precision (1-5-10) |
| `fp32_enabled` | `true` | bool | IEEE 754 single-precision (accumulator) |
| `bf16_enabled` | `false` | bool | Brain Float 16 (1-8-7) |
| `fp8_e4m3_enabled` | `false` | bool | FP8 E4M3 (OCP, forward pass) |
| `fp8_e5m2_enabled` | `false` | bool | FP8 E5M2 (OCP, backward pass) |
| `int8_enabled` | `true` | bool | INT8 symmetric quantization |
| `int4_enabled` | `true` | bool | INT4 packed quantization |
| `rounding_mode` | 0 | int | 0=RNE, 1=RTZ, 2=Stochastic |
| `subnormal_flush` | `true` | bool | Flush-to-zero (FTZ) for subnormals |
| `saturate` | `true` | bool | Saturate on overflow |

## 3. Memory System

| Field | Value | Type | Description |
|-------|-------|------|-------------|
| `sram_w_size_kb` | 128 | uint32 | Weight buffer size (KB) |
| `sram_a_size_kb` | 64 | uint32 | Activation buffer size (KB) |
| `sram_o_size_kb` | 64 | uint32 | Output/accumulator buffer size (KB) |
| `sram_num_banks` | 32 | uint32 | Number of SRAM banks |
| `sram_bank_width` | 4 | uint32 | Bytes per bank word |
| `sram_words_per_cycle` | 1 | uint32 | Max words per bank per cycle |
| `sram_arb_mode` | 1 | int | Arbitration: 0=None, 1=RR, 2=Priority |
| `sram_conflict_mode` | 1 | int | Conflict: 0=None, 1=Detect, 2=Stall |
| `sram_stall_penalty` | 2 | uint8 | Stall penalty cycles |
| `sram_bw_window_cycles` | 4 | uint64 | Bandwidth refill window |
| `sram_bw_modeling` | `true` | bool | Enable bandwidth modeling |
| `gbuf_size_kb` | 1024 | uint32 | Global buffer size (KB), 0=disabled |
| `gbuf_banks` | 16 | uint32 | Global buffer banks |
| `dram_type` | 0 | int | 0=Ideal, 1=HBM2, 2=HBM2e, 3=HBM3, 4=DDR4, 5=DDR5, 6=LPDDR5 |
| `dram_bandwidth_gbps` | 256.0 | double | DRAM bandwidth (GB/s) |
| `dram_channels` | 8 | uint32 | DRAM channel count |
| `dram_model_row_conflicts` | `false` | bool | Model row buffer hit/miss |

| `dram_row_policy` | 0 | int | 0=Legacy, 1=Open-page, 2=Closed-page, 3=Adaptive-timeout |
| `dram_address_mapping` | 0 | int | 0=Burst-interleaved, 1=Row-interleaved, 2=XOR-interleaved |
| `dram_row_miss_penalty_cycles` | 10 | uint32 | Added activate/precharge penalty per modeled miss |
| `dram_row_conflict_penalty_cycles` | 0 | uint32 | Open-row replacement penalty; 0 inherits row_miss_penalty_cycles |
| `dram_row_open_timeout_cycles` | 100 | uint32 | Idle cycles before adaptive-timeout lazily precharges a row |
| `dram_row_open_timeout_ns` | 100.000 | double | Physical-ns adaptive timeout source |
| `dram_row_timeout_domain` | 0 | int | 0=Fixed TU/core cycles (compat), 1=Physical ns converted at core clock |
| `dram_latency_domain` | 0 | int | 0=Fixed TU/core cycles (compat), 1=Physical ns converted at core clock |
| `dram_latency_read/write` | 50.000 / 50.000 | double | Base read/write latency in selected domain |
| `dram_core_clock_ghz` | 1.000 | double | TU/core clock used for GB/s-to-bytes/cycle and ns-to-cycle conversion |
| `dram_turnaround_mode` | 0 | int | 0=None, 1=Fixed, 2=Idle credit after base service, 3=Exact-byte burst credit, 4=Protocol-burst-rounded credit |
| `dram_turnaround_domain` | 0 | int | 0=Fixed TU/core cycles, 1=Physical ns converted at core clock |
| `dram_read_to_write_turnaround` | 0.000 | double | Read-to-write bus turnaround in selected domain |
| `dram_write_to_read_turnaround` | 0.000 | double | Write-to-read bus turnaround in selected domain |
| `dram_read_burst_bytes` | 64 | uint32 | Rounded-mode read occupancy granule; 0 inherits the DRAM preset |
| `dram_write_burst_bytes` | 64 | uint32 | Rounded-mode write occupancy granule; 0 inherits the DRAM preset |
| `dram_refresh_mode` | 0 | int | 0=None (compat), 1=All-bank, 2=Per-bank (JEDEC tREFI/tRFC) |
| `dram_refresh_scheduling` | 0 | int | 0=Fixed periodic, 1=Deferred (bounded postponement) |
| `dram_refresh_rate` | 1 | uint32 | 1x/2x/4x refresh-rate multiplier (high-temp retention) |
| `dram_trefi_ns` | 7800 | uint32 | JEDEC tREFI per-bank refresh interval (ns) |
| `dram_trfc_ns` | 350 | uint32 | All-bank refresh lockout tRFC (ns) |
| `dram_trfc_pb_ns` | 90 | uint32 | Per-bank refresh lockout tRFCpb (ns) |
| `dram_refresh_max_deferral_ns` | 7800 | uint32 | Deferred hard deadline (ns, ≤ tREFI) |

## 4. DMA Engine

| Field | Value | Type | Description |
|-------|-------|------|-------------|
| `dma_bus_width_bits` | 256 | uint32 | AXI bus width |
| `dma_max_burst_bytes` | 64 | uint32 | Max burst size |
| `dma_num_channels` | 3 | uint32 | DMA channel count |
| `dma_max_outstanding` | 4 | uint32 | Max outstanding descriptors |
| `dma_async_mode` | `false` | bool | Async DMA with descriptor queues |
| `dma_multicast_enabled` | `false` | bool | Multicast/broadcast DMA |
| `compression_enabled` | `false` | bool | Enable weight-stream compression |
| `compression_type` | 0 | int | 0=None, 1=RLE, 2=Adaptive RLE, 3=Bitmap, 4=Adaptive all |
| `compression_rle_epsilon` | 0 | double | Merge tolerance; 0 is lossless |
| `compression_decoder_enabled` | `false` | bool | Include decompressor throughput in stream-cycle estimates |
| `compression_decoder_overlap_dma` | `true` | bool | Pipeline payload DMA and decode; false serializes them |
| `compression_decoder_elements_per_cycle` | 1 | uint32 | Dense FP16 outputs reconstructed per cycle |
| `compression_rle_runs_per_cycle` | 1 | uint32 | RLE run descriptors issued per cycle |
| `compression_bitmap_elements_per_cycle` | 1 | uint32 | Bitmap positions scanned per cycle |

## 5. ISA & Command Queue

| Field | Value | Type | Description |
|-------|-------|------|-------------|
| `isa_instr_width_bits` | 96 | uint32 | Instruction encoding width |
| `isa_queue_depth` | 16 | uint32 | Command queue depth |
| `isa_dep_checking` | `false` | bool | Dependency checking |

## 6. Multi-Core

| Field | Value | Type | Description |
|-------|-------|------|-------------|
| `multicore_enabled` | `false` | bool | Multi-core TU cluster |
| `num_cores` | 1 | uint32 | Core count |
| `interconnect_mode` | 0 | int | 0=None, 1=Ring, 2=Mesh |
| `icc_switching_mode` | 0 | int | 0=Legacy hop-only, 1=Cut-through, 2=Store-and-forward |
| `icc_contention_mode` | 0 | int | 0=Ideal parallel links, 1=Shared-link lower bound |
| `icc_mesh_routing_mode` | 0 | int | 0=Deterministic XY, 1=Deterministic YX |
| `icc_link_bytes_per_cycle` | 16 | uint32 | Physical link payload width |
| `icc_router_latency_cycles` | 5 | uint32 | Per-hop router/link latency |

## 7. Performance Model

| Field | Value | Type | Description |
|-------|-------|------|-------------|
| `cycle_model` | 2 | int | 0=Functional, 1=Estimated, 2=Cycle-Accurate |
| `counters_enabled` | `true` | bool | Performance counters |
| `detailed_stalls` | `false` | bool | Detailed stall breakdown |
| `trace_enabled` | `false` | bool | VCD execution trace |
| `trace_max_events` | 65536 | uint32 | Max trace events |
| `power_tech_node` | 0 | int | 0=Auto, 1=45nm, 2=28nm, 3=16nm, 4=7nm, 5=5nm, 6=3nm |
| `power_clock_freq_mhz` | 0.0 | double | 0=Auto heuristic; otherwise explicit modeled clock MHz |

## 8. Sparsity

| Field | Value | Type | Description |
|-------|-------|------|-------------|
| `sparsity_enabled` | `false` | bool | Sparsity support |
| `sparsity_2of4` | `false` | bool | 2:4 structured sparsity |
| `sparsity_unstructured` | `false` | bool | Unstructured sparsity |
| `sparsity_decoder_groups_per_cycle` | 1 | uint32 | 2:4 metadata groups decoded per cycle |
| `sparsity_metadata_format` | 0 | int | 0=Bitmask, 1=CSR, 2=Coord |

## 9. Verification

| Field | Value | Type | Description |
|-------|-------|------|-------------|
| `golden_reference` | 1 | int | 0=NumPy, 1=PyTorch |
| `random_test_iters` | 1000 | uint32 | Random test iterations |
| `error_tolerance` | 1.000000e-05 | double | Golden comparison tolerance |

## 10. Derived Values

| Metric | Value | Formula |
|--------|-------|--------|
| Total SRAM | 256 KB | W + A + O |
| Total MACs | 256 | rows × cols × mac_units_per_pe |
| Peak Ops/cycle | 512 | MACs × 2 |
| DMA bandwidth | 32.0 GB/s | bus_width / 8 × frequency |

---

*Generated by `tu_config_emit_docs()`. Regenerate with `make config-docs` to reflect current settings.*
