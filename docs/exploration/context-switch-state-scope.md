# Context-switch state-scope exploration

**Date:** 2026-07-19
**Question:** How should a pre-spec TU trade transparent preemption against context-store traffic and hardware state?

## Hypothesis and alternatives

A fixed 100-cycle switch cost hides the dominant scaling term when scratchpad state is retained. Three materially different designs are plausible and should remain runtime-selectable:

- **`FULL_SRAM`** — save W/A/O scratchpads. A server TU may pay for a wide context-store datapath to provide transparent isolation and arbitrary-boundary preemption.
- **`LIVE_SRAM`** — save configured live prefixes of each region. A compiler-managed TU can reduce state by providing liveness/extents at a safe point. This implementation models contiguous prefixes, not arbitrary dirty blocks.
- **`CONTROL_ONLY`** — retain counters/configuration but no SRAM. A low-area edge TU can switch only where software can reload scratchpads; it does not provide transparent SRAM isolation.

No mode is universally preferable. Scheduling remains independently configurable as round-robin or priority, while zero time-slice values retain synchronization-point-only switching.

## Executable model

`make test-context-switch-sweep` executes the real context manager. Cost is:

```text
switch_cycles = fixed_pipeline_cycles
              + ceil((outgoing_saved_bytes + incoming_saved_bytes)
                     / state_bytes_per_cycle)
```

The sweep uses 100 fixed cycles, 32 B/cycle state bandwidth, W/A/O split 50%/25%/25%, and `LIVE_SRAM` retaining 25% of each region.

| SRAM | Scope | Bytes per context | Switch cycles |
|---:|---|---:|---:|
| 128 KiB | FULL | 131,072 | 8,292 |
| 128 KiB | LIVE 25% | 32,768 | 2,148 |
| 128 KiB | CONTROL | 0 | 100 |
| 256 KiB | FULL | 262,144 | 16,484 |
| 256 KiB | LIVE 25% | 65,536 | 4,196 |
| 256 KiB | CONTROL | 0 | 100 |
| 512 KiB | FULL | 524,288 | 32,868 |
| 512 KiB | LIVE 25% | 131,072 | 8,292 |
| 512 KiB | CONTROL | 0 | 100 |

Full-save bandwidth sensitivity at 256 KiB:

| State datapath | Switch cycles |
|---:|---:|
| 16 B/cycle | 32,868 |
| 32 B/cycle | 16,484 |
| 64 B/cycle | 8,292 |

These are model cycles, not wall-clock memcpy measurements. They include save and restore traffic but not DRAM contention, DMA setup, ECC, dirty-map scans, or context-store queueing.

## Multi-objective trade-off

| Dimension | FULL_SRAM | LIVE_SRAM | CONTROL_ONLY |
|---|---|---|---|
| Throughput | Largest interruption; scales linearly with SRAM | Lower when live fraction is small | Lowest modeled switch interruption, but reload work is omitted |
| Latency/QoS | Transparent but long worst-case pause | Shorter pause; compiler must identify safe state | Fast hardware handoff; end-to-end resume latency may be dominated by reload |
| Area/resources | Largest backing store / transfer datapath demand | Smaller store plus extent registers | Minimal context store |
| Power/energy | Highest state movement | Proportional to retained bytes | Lowest switch traffic; later reload energy unquantified |
| SRAM/DRAM traffic | 2× full SRAM per switch | 2× declared live bytes | No context traffic, but reload traffic unmodeled |
| Numerical accuracy | Unchanged | Unchanged if liveness contract is correct | Unchanged only after correct reload |
| Control complexity | Simple semantics; bulk transfer controller | Extent tracking and safe-point protocol | Simple hardware, more runtime orchestration |
| Verification burden | Full isolation and capacity cases | Prefix boundaries, stale tails, compiler/hardware contract | Reload legality and non-isolation behavior |
| Compiler/runtime | Optional hints only | Must emit valid per-region live extents | Must schedule reloads and avoid transparent preemption claims |

Area and energy are directional only: the cmodel does not currently map retained bytes or bandwidth to gates, SRAM macro count, joules, or frequency loss. Reload latency/traffic for `CONTROL_ONLY` is also unquantified, so its 100-cycle row must not be presented as end-to-end superiority.

## Implementation

- `tu_ctx_save_scope_t`: `FULL_SRAM`, `LIVE_SRAM`, `CONTROL_ONLY`.
- `tu_ctx_manager_config_t`: per-region live byte prefixes and `state_bytes_per_cycle`.
- Runtime validation rejects invalid policies, zero context capacity, and live extents beyond physical SRAM.
- Saved-byte accounting is exposed in each context descriptor.
- `FULL_SRAM` is enum value zero, preserving existing aggregate initializers. A zero state bandwidth preserves the old fixed-only cost; explicit nonzero bandwidth enables the realistic transfer term.
- Functional tests prove full restoration, live-prefix restoration with intentionally unrestored tails, control-only zero retention, byte-cost arithmetic, and invalid configuration rejection.

## Limitations and next decision

The live-state implementation supports one prefix per W/A/O region. Real allocators may need dirty bitmaps, scatter/gather extents, or architectural register state beyond the current `tu_state_t` counters/config. Before adding those, obtain compiler traces of live scratchpad allocations. Without such traces, a bitmap implementation would add mechanism without defensible workload evidence.

## Verification commands

```sh
make test-context                 # 15/15
make test-context-switch-sweep    # matrix above
make clean && make
make test-quick
```
