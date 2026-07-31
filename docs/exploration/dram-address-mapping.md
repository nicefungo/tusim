# DRAM Address Mapping: Burst, Row, and XOR Interleaving

**Date:** 2026-07-31
**Question:** When should a TU DRAM controller use predictable burst/row placement versus XOR-hashed channel selection?

## Hypothesis

Fine-grained burst interleaving exposes channels early but aliases power-of-two tensor strides. Row interleaving can preserve contiguous page locality but delays channel exposure. A low-cost XOR of the base channel with higher row-group bits should retain the burst mapping's bank/row identity while distributing aliased strides across power-of-two channels. The hash should improve placement balance, not automatically prove lower latency: decode delay, concurrent request scheduling, and arbitration are outside the current model.

## Realistic alternatives

| Mode | Why a hardware team might choose it | Principal sacrifice |
|---|---|---|
| `burst_interleaved` | Low-order burst striping is simple, predictable, and quickly exposes channels for contiguous tensors | Power-of-two strides can repeatedly select one channel |
| `row_interleaved` | Keeps each row-sized block together, helping open-page locality and page-oriented software placement | Short contiguous spans can underuse channels; row-sized strides discard locality |
| `xor_interleaved` | XORs the low-order burst channel with low bits of the higher row group, breaking common stride aliases without changing the decoded bank/row | Extra XOR logic on the address path, less intuitive placement, harder compiler reasoning/debug, and power-of-two channel requirement |

A physical TU would normally hard-wire one decoder. The pre-spec cmodel preserves all three as candidate chips. `burst_interleaved` remains numeric zero and the default, so old configs and zero-initialized callers retain their behavior.

## Executable model and configuration

Canonical JSON/YAML under `tu.memory.dram` accepts:

```json
"channels": 8,
"address_mapping": "xor_interleaved"
```

Accepted mappings are `burst_interleaved`, `row_interleaved`, and `xor_interleaved`. XOR mode requires a nonzero power-of-two channel count; canonical validation and the subsystem setter reject unsupported geometries rather than silently falling back.

For burst and XOR modes:

```text
burst          = address / burst_bytes
base_channel   = burst mod channels
channel_burst  = floor(burst / channels)
row_group      = floor(channel_burst / bursts_per_row)
burst channel  = base_channel
XOR channel    = base_channel XOR (row_group mod channels)
bank           = row_group mod banks_per_channel
row            = floor(row_group / banks_per_channel)
```

Row interleaving assigns complete row-sized groups across channels before bank/row selection. All decode dimensions come from live DRAM parameters. Mapping changes clear retained open-row tags. The full path covers YAML/JSON, generated constants, canonical defaults/parser/validation, `tu_dram_create_from_config()`, public decode, channel contention selection, and open-page row tracking. This heartbeat also wired the pre-existing canonical `dram_channels` field to the JSON parser; previously a JSON `channels` value in the DRAM block was ignored.

## Measured sweep

Command:

```sh
make test-dram-address-mapping-sweep
```

Configuration: custom DRAM, 4 channels, 16 banks/channel, 2 KiB rows, 64-byte bursts, open-page policy, 50-cycle base read service, 20-cycle row-miss penalty. Each row issues 64 reads of 64 bytes. `service` is the sum of returned row-service cycles only. `used_ch` and `max_ch` are exact decoder occupancy, not throughput or makespan.

| Pattern | Mapping | Service cycles | Hits | Misses | Channels used | Max accesses/channel |
|---|---|---:|---:|---:|---:|---:|
| Contiguous 64 B | Burst | 3,280 | 60 | 4 | 4 | 16 |
| Contiguous 64 B | Row | 3,240 | 62 | 2 | 2 | 32 |
| Contiguous 64 B | XOR | 3,280 | 60 | 4 | 4 | 16 |
| 2 KiB stride | Burst | 3,520 | 48 | 16 | 1 | 64 |
| 2 KiB stride | Row | 4,480 | 0 | 64 | 4 | 16 |
| 2 KiB stride | XOR | 3,520 | 48 | 16 | 4 | 16 |

For contiguous traffic, XOR is identical to burst placement in every observed metric; row mode lowers summed row service by **1.22%** through two fewer misses but uses only two channels. For the 2 KiB stride, XOR preserves burst mode's 3,520 service cycles and 48 hits while changing channel occupancy from `64/0/0/0` to `16/16/16/16`. Row mode also balances channels but raises summed service by **27.27% versus burst/XOR** because every request misses. The XOR result is therefore a placement improvement for this exact aliasing pattern, not a measured latency speedup.

## Gain versus sacrifice

- **Throughput:** XOR exposes four channels instead of one for the measured 2 KiB stride, so a queue-aware controller may exploit more parallel service. Sustained throughput is **unquantified** because requests are not concurrently scheduled.
- **Latency:** Returned row-service cycles are unchanged between burst and XOR in both patterns. XOR decode could lengthen an address/control critical path; that delay is **unmodeled**. Row mode wins the measured contiguous service sum but has less channel exposure.
- **Area/resources:** Burst/row decoders can be mostly bit slicing for power-of-two geometry. XOR adds a small bank of XOR gates and routing on channel-select bits. Gate count, timing, and physical routing are **unquantified**.
- **Power/energy:** XOR may activate channels more evenly and avoid queue hot spots, but toggles extra decode logic. Row hits should avoid activation energy. Neither decode nor activation energy is wired into the power model, so net energy is **unquantified**.
- **SRAM/DRAM traffic:** Payload bytes are identical. Bank/row identity is intentionally retained between burst and XOR; only channel placement changes. Router/controller-buffer traffic is **unmodeled**.
- **Numerical accuracy:** Unchanged; all mappings preserve byte addresses and arithmetic semantics.
- **Control complexity:** All modes are static deterministic mappings. XOR requires power-of-two channels and a documented hash contract but no adaptive state, request feedback, or remap table.
- **Verification burden:** Exact discriminating decode vectors, all three modes, mode-state reset, malformed IDs, non-power-of-two rejection, parser/propagation, generated constants, and sweep accounting are gated. XOR also expands allocator/compiler placement cases.
- **Compiler/runtime:** Burst and row placement are easier to reason about explicitly. XOR reduces accidental aliases but makes deliberate channel coloring less intuitive; allocators and profilers need the exact hash. A compiler may still prefer deterministic burst mode when software padding/tiling already controls aliases.

## Fidelity limits

The cmodel has no request queue, reordering, write draining, injection timing, arbitration, backpressure, bank groups, rank/subchannel geometry, refresh, calibrated timing, decode delay, or physical area/power. Channel occupancy is exact placement evidence only. The module's coarse stall counters are a separate accounting domain and are not used to claim makespan. A more complex multi-bit hash, bank XOR, adaptive remapping, or calibrated recommendation remains blocked on real compiler/runtime traces and a queue-aware controller contract.

## Verification

```sh
make test-dram                         # 18/18
make test-dram-address-mapping-sweep   # 6 rows
make test-config                       # 22/22
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.xor.h
make clean && make
make test-quick
```

## Actionable conclusion

Preserve all three mappings. Burst interleaving is the conservative predictable default; row interleaving is plausible when contiguous page locality is more valuable than early channel spread; XOR interleaving is valuable for power-of-two channel systems exposed to aliasing tensor strides and willing to pay modest decode/control and software-observability costs. Do not call XOR faster until concurrent request scheduling converts its balanced placement into measured makespan.
