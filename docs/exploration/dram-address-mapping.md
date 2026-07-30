# DRAM Address Mapping: Burst vs Row Interleaving

**Date:** 2026-07-30
**Question:** Should contiguous physical addresses stripe channels at each DRAM burst, or keep one row-sized block on a channel before moving to the next channel?

## Hypothesis

Fine-grained burst interleaving should expose more channels for short contiguous tensors and may preserve locality for some channel-aligned strides, while row-granularity interleaving should keep contiguous row-sized blocks together and can improve open-page locality. Neither mapping should dominate because address stride changes both row reuse and channel balance.

## Realistic alternatives

| Mode | Why hardware might choose it | Principal sacrifice |
|---|---|---|
| `burst_interleaved` | Stripes each 64-byte burst across channels, exposing channel-level parallelism quickly for short contiguous transfers; simple low-order channel decode | Some contiguous spans are fragmented across channels, and power-of-two strides can alias onto one channel |
| `row_interleaved` | Keeps a complete row-buffer-sized block on one channel, favoring open-page locality and simpler page-oriented placement | Short spans may use only one or two channels; row-sized strides rotate channels but discard row reuse |

Both are physically plausible static address decoders. A deployed chip would normally hard-wire one mapping, while the pre-spec cmodel keeps both runtime-selectable. The prior mapping is `burst_interleaved`, which remains the zero/default mode for compatibility.

## Executable model and configuration

Canonical JSON/YAML under `tu.memory.dram` accepts:

```json
"address_mapping": "burst_interleaved"
```

Accepted values are `burst_interleaved` and `row_interleaved`. The full path is executable:

1. YAML/JSON source and `scripts/gen_config.py` constants;
2. canonical `tu_config_t`, defaults, parser, and validation;
3. `tu_dram_create_from_config()` propagation;
4. `tu_dram_decode_address()` channel/bank/row decode;
5. the same decoder consumed by channel contention and open-page row tracking.

The model uses the configured burst length and row-buffer size rather than hardcoded dimensions. Changing mapping clears retained open-row tags so stale state cannot cross decoder modes.

## Sweep

Command:

```sh
make test-dram-address-mapping-sweep
```

Configuration: custom DRAM, 4 channels, 16 banks/channel, 2 KiB row buffers, 64-byte bursts, open-page policy, 50-cycle base read service, 20-cycle row-miss penalty. Each case issues 64 reads of 64 bytes. `service` is the sum of returned row-service cycles only. `used_ch` and `max_ch` are exact address-decode occupancy metrics, not a parallel makespan model.

| Pattern | Mapping | Service cycles | Hits | Misses | Channels used | Max accesses/channel |
|---|---|---:|---:|---:|---:|---:|
| Contiguous 64 B | Burst interleaved | 3,280 | 60 | 4 | 4 | 16 |
| Contiguous 64 B | Row interleaved | 3,240 | 62 | 2 | 2 | 32 |
| 2 KiB stride | Burst interleaved | 3,520 | 48 | 16 | 1 | 64 |
| 2 KiB stride | Row interleaved | 4,480 | 0 | 64 | 4 | 16 |

For this 4 KiB contiguous span, row interleaving lowers modeled row-service cycles by **1.22%**, buying two fewer row misses but using only two of four channels. For the 2 KiB-stride pattern, burst interleaving lowers row-service cycles by **21.43%** through retained row reuse, but all 64 requests alias to one channel. Row interleaving uses all four channels evenly in that case but incurs a row miss on every request. These are mapping- and workload-specific reversals, not universal recommendations.

## Gain versus sacrifice

- **Throughput/latency:** Row service and channel exposure move in opposite directions in the stride case. The current model reports service per request and address occupancy; it does not schedule concurrent requests, so it cannot convert four-channel balance into a trustworthy makespan.
- **Area/resources:** Both modes need static bit selection/divide-modulo decode. Row interleaving additionally depends on row-size grouping. Exact gate area and timing are unquantified; power-of-two physical parameters would reduce operations to bit slicing.
- **Power/energy:** More row hits should reduce activate/precharge energy, while wider channel use may activate more channel logic. DRAM activate energy is not wired to this module, so net energy is unquantified.
- **SRAM/DRAM traffic:** Payload bytes are identical. Only channel/bank/row placement and inferred internal row operations differ; SRAM traffic is unchanged.
- **Numerical accuracy:** Unchanged; address mapping does not alter transferred bytes.
- **Control complexity:** Both are deterministic and substantially simpler than XOR/hash mappings or adaptive remapping. Runtime cmodel selection represents alternative candidate chips, not a free dynamic hardware switch.
- **Verification burden:** Exact decode vectors, both modes, invalid-mode rejection, mode-state reset, canonical parse/validation, generated constants, and row-policy interaction are covered. More mappings would multiply placement and trace validation.
- **Compiler/runtime:** Tensor base alignment, layout, padding, and DMA stride can deliberately avoid channel aliases or preserve rows once the hardware mapping is known. Runtime allocators need the mapping contract; opaque or undocumented mapping makes placement optimizations fragile.
- **Physical limitations:** Rank, bank-group, column-bit permutations, XOR/hash channel selection, subchannel geometry, ECC, remap tables, request queues, arbitration, backpressure, refresh, and calibrated timing are absent. The existing contention counter is a coarse separate domain and is intentionally excluded from the table.

## Verification

```sh
make test-dram                         # 17/17
make test-dram-address-mapping-sweep   # 4 rows, complete row accounting
make test-config                       # 22/22
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.address-mapping.h
make clean && make
make test-quick
```

## Actionable conclusion

Preserve both mappings. Burst interleaving is appropriate when early channel exposure matters and software can avoid aliasing strides; row interleaving is appropriate when contiguous row locality is valuable and tensors are large enough to cover channels. A future XOR/hash mapping is not yet justified: it could reduce stride aliases but would add decode complexity and may destroy predictable row locality. Add it only with compiler traces and a queue-aware concurrency model that can value channel balance honestly.
