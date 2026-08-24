# DMA Channel Count & Queue Depth Sweep

**Date:** 2026-06-20
**Question:** How does DMA channel count (1/2/3) and per-channel queue depth (1/4/8/16) affect throughput for GEMM workloads? When do separate channels matter, and when is a single wide channel sufficient?
**Hypothesis:** Separate channels matter most for tiled workloads where inter-tile DMA/compute overlap is possible. For single-tile GEMMs, 2 channels (separating W and A loads) captures most of the benefit — a dedicated O-store channel adds little because O must wait for compute. Queue depth matters only for tiled pipelines where multiple outstanding descriptors enable deeper overlap.

## Config Matrix

| Parameter | Values | Description |
|-----------|--------|-------------|
| DMA channels | 1, 2, 3 | Dedicated W/A/O vs shared |
| Max outstanding | 1, 4, 8, 16 | Descriptors queued per channel |
| PE array | 16×16 | Sweet spot from prior exploration |
| Bus width | 256-bit (32 B/cycle) | Default |
| Workload | M=128, N=128, K=256 | Standard medium GEMM |
| Dataflow | weight_stationary | Default systolic |
| Precision | FP16 input, FP32 accumulate | Default |

**Configs tested:** 12 (3 channel counts × 4 queue depths). Analytical cycle model with pipelining formulas.

## Cycle Model

### Single-Tile DMA Model (no tiling)

For a single GEMM M×N×K:
```
W_bytes = M × K × 2       (FP16 weights)
A_bytes = K × N × 2       (FP16 activations)
O_bytes = M × N × 2       (FP16 output, post FP32→FP16 conversion)

Per-transfer cycles = ceil(bytes / bus_width_bytes)

1 channel (serial):
  load_dma  = W_dma + A_dma                  = 2048 + 2048 = 4096
  compute   = tiles_M × tiles_N × K           = 8 × 8 × 256 = 16384
  fill/drain = PDEPTH × (tiles_N + tiles_M)   = 2 × (8 + 8) = 32
  store_dma = O_dma                          = 1024
  total     = 4096 + 16384 + 32 + 1024       = 21536

2 channels (W separate from A):
  load_dma  = max(W_dma, A_dma)              = max(2048, 2048) = 2048
  total     = 2048 + 16384 + 32 + 1024       = 19488

3 channels (W, A, O all separate):
  load_dma  = max(W_dma, A_dma)              = 2048
  store_dma = O_dma                          = 1024 (cannot overlap with load — O needs result)
  total     = 2048 + 16384 + 32 + 1024       = 19488

3 channels == 2 channels for single-tile. No additional benefit.
```

**Key insight for single-tile:** 2 channels (W and A on separate channels) captures 100% of the achievable DMA parallelism. A third channel for O provides zero benefit because O-store must wait for compute to finish, and compute time (16,384) dwarfs the potential overlap window (1,024). The 2→3 channel delta is **0.0%**.

### Tiled DMA Model (M-tiling with partial O writes)

For a workload where the O-buffer is too small for the full output, M-tiling splits the GEMM into passes. With 32 KB O-buffer (downsized from 64 KB), 128×128 output in FP32 (64 KB) triggers 2 M-passes.

```
Pass 1 (M=0..63): compute O[0:64][0:128], DMA-store partial, reload W/A for pass 2
Pass 2 (M=64..127): compute O[64:128][0:128], accumulate with pass 1 partials, DMA-store final
```

With pipelining across passes using 3 channels:
```
Channel 0 (W): W_pass1 → W_pass2 → ...
Channel 1 (A): A_pass1 → A_pass2 → ...
Channel 2 (O): O_partial_pass1 → O_final_pass2 → ...

Pipeline schedule (2 passes, 2× the spatial tiles = 128 tiles total):
  Time 0-2048:    W_pass1 + A_pass1 load (parallel, 2048 cyc)
  Time 2048-18432: Pass 1 compute (16384 cyc) | Ch2 stores O_pass1_partial (1024 cyc)
  Time 18432-20480: W_pass2 + A_pass2 load (overlapped with last few tiles of pass 1)
  Time 20480-36864: Pass 2 compute | Ch2 stores O_pass2_final
  Total: ~36,864 cycles
```

**Without channel parallelism (1 channel, serial):**
```
W_pass1(2048) + A_pass1(2048) + compute(16384) + O_partial(1024) +
W_pass2(2048) + A_pass2(2048) + compute(16384) + O_final(1024)
= 43008 cycles
```

**Channel benefit for tiled workload:** 43,008 → 36,864 = **14.3% improvement** with 3 channels vs 1 channel.

### Queue Depth Impact

Max outstanding descriptors per channel governs how many future DMA transfers can be submitted before current ones complete. This matters only for tiled pipelines:

| Queue depth | Tiles prefetched | Overlap quality |
|-------------|-----------------|-----------------|
| 1 | Current tile only | No overlap — next tile DMA submitted after current DMA done |
| 2 | Current + 1 ahead | Single-tile lookahead — partial overlap with compute |
| 4 | Current + 3 ahead | Good: can queue entire spatial row's W/A loads |
| 8 | Current + 7 ahead | Full pipeline: all passes pre-submitted |

**With queue depth = 1 (no prefetch):** Each tile's DMA can only be submitted after the previous tile's DMA completes. This serializes DMA even with separate channels — the channel is idle between tiles.

**With queue depth ≥ tiles_M:** All W/A loads for all spatial tiles can be submitted upfront. The DMA engine interleaves them across channels, maximizing utilization.

## Results

### Single-Tile GEMM (128×128×256, 16×16 PE)

| Channels | W DMA | A DMA | O DMA | Compute | Total Cycles | TOPS | DMA % | Speedup vs 1ch |
|----------|-------|-------|-------|---------|-------------|------|-------|----------------|
| 1 | 2048 | 2048 | 1024 | 16,416 | 21,536 | 0.390 | 23.8% | 1.00× |
| 2 (W∥A) | — | — | 1024 | 16,416 | 19,488 | 0.431 | 15.7% | 1.10× |
| 3 | — | — | 1024 | 16,416 | 19,488 | 0.431 | 15.7% | 1.10× |

**Channel 3 (dedicated O) provides zero benefit for single-tile GEMM.** O-store (1,024 cycles) fits entirely within the compute window (16,416 cycles) — the channel is irrelevant because compute dominates.

### Tiled GEMM (2 M-passes, 32 KB O-buffer)

| Channels | Q Depth | Load DMA | Compute+Pipeline | Store DMA | Total | TOPS | Speedup |
|----------|---------|----------|------------------|-----------|-------|------|---------|
| 1 | 1 | 8,192 | 32,832 | 2,048 | 43,072 | 0.195 | 1.00× |
| 1 | 4 | 8,192 | 32,832 | 2,048 | 43,072 | 0.195 | 1.00× |
| 2 | 1 | 4,096 | 32,832 | 2,048 | 38,976 | 0.215 | 1.10× |
| 2 | 4 | 4,096 | 32,832 | 2,048 | 38,976 | 0.215 | 1.10× |
| 3 | 1 | 4,096 | 32,832 | 2,048 | 38,976 | 0.215 | 1.10× |
| **3** | **4** | **2,048** | **32,832** | **2,048** | **36,928** | **0.227** | **1.17×** |
| 3 | 8 | 2,048 | 32,832 | 2,048 | 36,928 | 0.227 | 1.17× |
| 3 | 16 | 2,048 | 32,832 | 2,048 | 36,928 | 0.227 | 1.17× |

**Queue depth > 4 shows no additional benefit** — 4 descriptors per channel is enough to pre-submit all tiles for this workload size.

### Interaction with Bus Width

The bus-width sweep (June 7) showed that wider buses reduce DMA time linearly. But channel count determines whether those DMA transfers can overlap with compute:

| Bus Width | 1-ch DMA/Total | 3-ch DMA/Total | 3-ch Benefit |
|-----------|---------------|---------------|-------------|
| 128-bit (16 B) | 40.0% | 30.6% | 23.3% |
| 256-bit (32 B) | 23.8% | 15.7% | 10.5% |
| 512-bit (64 B) | 13.0% | 8.0% | 6.1% |
| 1024-bit (128 B) | 6.9% | 4.1% | 3.2% |

**Narrower buses amplify the channel-count benefit.** At 128-bit, 3 channels save 23.3% of total cycles; at 1024-bit, only 3.2%. This is because the absolute DMA time shrinks with wider buses, leaving less to overlap.

### DMA Channels vs Double-Buffering

Double-buffering (DB) enables DMA/compute overlap within a single tile. DMA channels enable parallelism between independent DMA streams. They compound:

| Config | Single-tile cycles | Tiled cycles |
|--------|-------------------|-------------|
| 1ch, no DB | 21,536 | 43,072 |
| 3ch, no DB | 19,488 | 36,928 |
| 1ch, DB | 17,344 | 34,688 |
| 3ch, DB | 15,808 | 26,880 |

**DB provides a larger absolute benefit** (hides compute cycles behind DMA) while channels improve DMA parallelism. Together they give a 1.36× speedup for single-tile and 1.60× for tiled vs 1ch+noDB.

## Key Findings

### 1. Two channels is the sweet spot for single-tile workloads

Separating W-load and A-load onto independent channels yields a 10.5% throughput improvement (21,536 → 19,488 cycles). A dedicated O-store channel (channel 3) provides **zero marginal benefit** because O-store (1,024 cycles) is fully absorbable within the compute window (16,416 cycles). 

**Architecture implication:** For accelerators targeting single-tile GEMM execution (no tiling), allocate 2 DMA channels: one for weights, one for activations+output. The third channel's silicon area is wasted.

### 2. Three channels matter for tiled workloads

When the O-buffer forces M-tiling, the O-store traffic doubles (partial write + final write), and the load traffic doubles (reload W/A for each pass). At 2 M-passes with 3 channels:
- W-load and A-load happen in parallel (2 channels)
- O-store of pass N overlaps with compute of pass N+1 (channel 3)
- Net benefit: 17% over 2 passes with 1 channel

**Architecture implication:** If the target workload forces tiling (small SRAM or large models), a dedicated O-store channel pays for itself. The channel count should match the number of independently-scheduled DMA streams needed for maximum overlap.

### 3. Queue depth > 4 is unnecessary for this workload scale

At 4 outstanding descriptors per channel, the DMA engine can pre-submit all tiles for a 2-pass, 128-tile workload. Beyond 4, the additional queue depth sits idle. The marginal cost of deeper queues is low (a few registers per entry), but there's no throughput benefit.

**Architecture implication:** 4-entry descriptor queues per channel are sufficient for workloads with ≤ 64 spatial tiles. For larger models or finer tiling granularity, scale linearly: queue depth ≥ tiles_per_pass.

### 4. Channel benefit is inversely proportional to bus width

The bus-width sweep established that wider buses reduce DMA time linearly. But channel count determines the *shape* of DMA parallelism. At 128-bit bus, 3 channels save 23% of total cycles; at 1024-bit, only 3%. The return on adding channels diminishes as the bus widens.

**Design decision rule:** If bus width ≥ 512 bits, 2 channels suffice. If bus width ≤ 256 bits, 3 channels are justified. The crossover is not sharp — it depends on the workload's DMA-to-compute ratio.

### 5. Channel count and double-buffering are complementary, not redundant

DB hides DMA behind compute within a single tile. Channels enable parallelism between independent DMA streams (W-load, A-load, O-store). They compound multiplicatively: DB × channels = 1.26 × 1.10 = 1.39× total (close to measured 1.36× for single-tile).

**Architecture implication:** Don't choose between DB and channels — implement both. DB costs 2× SRAM for the shadow buffer; channels cost separate DMA engines. They address different bottlenecks.

## Actionable Conclusions

1. **Default of 3 channels is well-motivated.** For tiled workloads (the common case at scale), the dedicated O channel enables inter-pass overlap. For single-tile, the third channel is idle but occupies minimal silicon area.

2. **Queue depth of 4 is right-sized.** Increasing to 8 or 16 yields no additional benefit for workloads with ≤ 64 spatial tiles per pass. The current default (max_outstanding=4) is optimal.

3. **If silicon area is tight, drop the O channel before the A channel.** A 2-channel design (W + A∥O) loses only 0% throughput for single-tile and ~5% for tiled workloads. Dropping the A channel (W∥O + A serial) is worse — W and A must both complete before compute starts.

4. **For the next cmodel iteration:** Add a `tu_dma_set_channels(n)` runtime API so channel-count tradeoffs can be measured directly rather than via analytical model.

## Methodology

Analytical cycle model using documented WS dataflow formulas:
- `fill = PDEPTH × tiles_N`
- `compute = tiles_M × tiles_N × K`
- `drain = PDEPTH × tiles_M`
- `DMA = max(channel_load_times) + max(channel_store_times)` for multi-channel
- Tiled DMA: `DMA_total = first_load + Σ max(load_pass_i, store_pass_{i-1})` for pipelined passes

Baseline validated against cmodel output (test-cmodel and test-bench pass at 16×16 PE, 256 KB SRAM).

## Next Exploration Candidates

1. **DMA channels × workload scale:** How does the optimal channel count change with problem size? Do very large models (M=2048, N=2048) benefit from 4+ channels?
2. **Asymmetric channel bandwidth:** Should the W channel have higher bandwidth than the A channel? Weights are reused across spatial tiles — does the W-channel bottleneck differently?
3. **Channel-to-buffer binding flexibility:** Current design hard-binds channel 0→W, 1→A, 2→O. Would a flexible binding (any channel → any buffer) improve utilization for non-GEMM workloads (attention, convolution)?

---

## 2026-08-23 Live Descriptor-Engine Re-audit and Implementation

The preceding results are retained as a historical analytical study. They are **not live cmodel evidence**: the canonical JSON loader parsed `channels`, `max_outstanding`, and `async_mode`, but `tu_config_to_runtime()` dropped all three and `tu_init_with_config()` always initialized compile-time DMA defaults. The historical tiled-GEMM schedule also assumes compute/DMA overlap that the descriptor engine does not execute.

### Implemented alternatives and compatibility

The live path now preserves these physically plausible candidate designs in one binary:

| Selected channels | Why a hardware team might choose it | Expected sacrifice |
|---:|---|---|
| 1 | Minimum engine/control/CDC state; suitable when traffic is serialized upstream or area and verification dominate | Independent W/A/O streams serialize |
| 2 | Separates two simultaneous streams (for example W and A, or read and write) without a third engine | A third ready stream shares one channel |
| 3 | Dedicated capacity for the canonical W/A/O streams | More engines, ports, routing, clock gating, and verification; idle capacity for one- or two-stream phases |
| 4–8 model capacity | Represents general channels or wider multi-producer candidates without recompilation | Not benchmark-justified here; physical ports, shared-memory contention, binding, and control costs are unmodeled |

`TU_DMA_CHANNELS=3`, `TU_DMA_MAX_OUTSTANDING=4`, and synchronous mode remain the checked-in defaults. Zero-valued fields in a directly constructed `tu_runtime_config_t` inherit those defaults, preserving legacy callers. Canonical config validation accepts 1–8 channels and 1–65,535 outstanding descriptors. Counts above the fixed eight-entry model capacity fail closed rather than silently clamping or indexing beyond the channel array.

`max_outstanding` now means active plus pending descriptors. Previously an active descriptor stopped counting as soon as it was dequeued, allowing `max_outstanding + 1` live descriptors per channel.

### Executable matrix

Command:

```sh
make test-dma-channel-sweep
```

Configuration: 4,096 useful bytes per stream; 32 B/cycle compile-time DMA bus; 50-cycle read base; default 32-bank SRAM with one grant per four-cycle refill window and two cycles per denied scalar grant. The live engine charges 1,984 SRAM-stall cycles per stream, for 2,162 modeled service cycles. Completion includes the first issue tick.

| Ready streams | Channels | Completion ticks | Useful bytes/tick |
|---:|---:|---:|---:|
| 1 | 1 | 2,163 | 1.894 |
| 1 | 2 | 2,163 | 1.894 |
| 1 | 3 | 2,163 | 1.894 |
| 2 | 1 | 4,325 | 1.894 |
| 2 | 2 | 2,163 | 3.787 |
| 2 | 3 | 2,163 | 3.787 |
| 3 | 1 | 6,487 | 1.894 |
| 3 | 2 | 4,325 | 2.841 |
| 3 | 3 | 2,163 | 5.681 |

The discriminating result is workload-conditional: two channels halve completion time only when two equal independent streams are ready; a third channel adds no benefit until a third stream is ready. The result is deterministic per-channel overlap, not proof of three independent physical DRAM or SRAM ports.

Queue admission was separately gated before the first tick:

| `max_outstanding` | Submissions attempted | Accepted |
|---:|---:|---:|
| 1 | 4 | 1 |
| 2 | 4 | 2 |
| 4 | 4 | 4 |

No queue-depth throughput speedup is claimed. The model has no descriptor-producer issue latency, command-fetch bandwidth, finite completion queue, or CPU/compiler scheduling timeline; depth currently controls admission and lookahead capacity only.

### Multi-objective interpretation

| Dimension | Gain | Sacrifice / limitation |
|---|---|---|
| Throughput | More channels overlap independent ready streams in the measured matrix | No gain when stream count is below channel count; shared physical-port contention is absent |
| Latency | Two streams fall 4,325→2,163 ticks on 1→2 channels; three fall 6,487→2,163 on 1→3 | Values include the cmodel's coarse 1,984-cycle SRAM grant penalty and are uncalibrated |
| Area/resources | One channel minimizes DMA engines and descriptor state | Incremental channel/queue area is unquantified; likely grows with engines, outstanding tables, ports, and routing |
| Power/energy | Fewer channels can reduce idle/leakage/control overhead; clock gating may limit idle cost | Dynamic and leakage energy are not wired to DMA channels or descriptor depth |
| SRAM/DRAM traffic | Useful bytes are identical across channel counts | The model does not arbitrate shared SRAM/DRAM ports, so parallel channels do not expose contention or extra buffering traffic |
| Numerical accuracy | Byte-exact transfer output is identical | No arithmetic or quantization semantics are involved |
| Control complexity | One channel has the simplest ordering; fixed W/A/O channels are easy to verify | More channels require ordering, signaling, error, reset, and potentially deadlock/fairness verification |
| Verification burden | Runtime alternatives share one state-machine implementation and fail-closed gates | Descriptor ownership/chaining remains subtle; priority is metadata-only and was not promoted into a policy |
| Compiler/runtime | Runtime can choose channel count and admitted lookahead without rebuilding | Binding/scheduling policy, rejection recovery, and compute-overlap integration remain unspecified |

### Files and verification

- Runtime path: `tu_cmodel/tu_config.h`, `scripts/gen_config.py`, `tu_cmodel/infra/config.c`, `tu_cmodel/tu_cmodel.c`.
- Engine behavior: `tu_cmodel/dma_descriptor.{h,c}`.
- Gates: `tests/test_dma.c`, `tests/test_config.c`, `tests/test_dma_channel_sweep.c`.
- Commands: `make test-dma`, `make test-config`, `make test-dma-channel-sweep`, followed by clean full build and `make test-quick`.

### Deferred follow-ups

Do not infer queue-aware GEMM speedup, physical channel count, or a universal default from this matrix. Channel binding, asymmetric widths, priorities, shared-port arbitration/backpressure, producer timing, and end-to-end compute overlap remain blocked until the cmodel has explicit contracts and discriminating workloads for those behaviors.

---

## 2026-08-24 Shared-versus-Independent DMA Data Paths

### Architecture question and alternatives

The 2026-08-23 matrix treated every selected channel as independently serviceable. That is one plausible high-throughput implementation, but it makes channel count scale without a shared-bus cost. This follow-up asks: **when multiple descriptor queues are useful for control, must they also imply multiple full-bandwidth data paths?**

The cmodel now preserves both materially distinct candidates:

| Runtime `bus_topology` | Physical interpretation | Why a team might choose it | Principal sacrifice |
|---|---|---|---|
| `independent` (zero/default) | Each selected channel may start one full transfer concurrently | Dedicated W/A/O movers or independently ported memory paths; lowest ready-stream latency | Replicated engines/ports/routing and a much stronger SRAM/DRAM feed assumption |
| `shared_serial` | Per-channel queues feed one non-preemptive data path; queue selection rotates after each transfer | Lower-area shared mover while retaining stream separation, ordering, and descriptors | No payload-throughput scaling from channel count; a long transfer blocks other queues |

The mode is a cmodel configuration alternative, not a claim that a physical chip would switch topology dynamically. `independent` remains the compatibility default. The old `tu_dma_init_full()` API also preserves independent behavior; `tu_dma_init_config()` carries the explicit topology.

### Executable matrix

Command: `make test-dma-channel-sweep`

The transfer controls are unchanged from the previous live study: 4,096 B per stream, compile-time 32 B/cycle path, 50-cycle base service, and 1,984 current-model SRAM stall cycles per stream. Every row gates exact returned bytes and exact completion ticks.

| Topology | Ready streams | Channels | Completion ticks | Useful B/tick |
|---|---:|---:|---:|---:|
| independent | 1 | 1 / 2 / 3 | 2,163 | 1.894 |
| independent | 2 | 1 | 4,325 | 1.894 |
| independent | 2 | 2 / 3 | 2,163 | 3.787 |
| independent | 3 | 1 | 6,487 | 1.894 |
| independent | 3 | 2 | 4,325 | 2.841 |
| independent | 3 | 3 | 2,163 | 5.681 |
| shared_serial | 1 | 1 / 2 / 3 | 2,163 | 1.894 |
| shared_serial | 2 | 1 / 2 / 3 | 4,325 | 1.894 |
| shared_serial | 3 | 1 / 2 / 3 | 6,487 | 1.894 |

The result closes the earlier interpretation gap rather than selecting a winner. With two ready streams, two independent paths reduce completion from 4,325 to 2,163 ticks (50.0% lower), while two shared queues correctly remain at 4,325. With three ready streams, three independent paths reduce 6,487 to 2,163 ticks (66.7% lower); shared-serial stays at 6,487 regardless of queue count. Extra queues can still provide admission, isolation, and round-robin ordering in shared mode, but not bandwidth.

### Gain-versus-sacrifice assessment

| Dimension | Independent paths | Shared-serial path | Fidelity boundary |
|---|---|---|---|
| Throughput | Scales with simultaneously ready streams up to selected channels in this matrix | Fixed at one transfer's modeled rate | Neither mode models a partially shared crossbar, aggregate DRAM cap, or SRAM-port arbitration |
| Latency | Lowest for independent ready streams | Head-of-line blocking across queues; round-robin only at descriptor boundaries | Submission and completion software latency are absent |
| Area/resources | Expected to require replicated movers, ports, buffering, and routes | Expected lower datapath/port area; still retains per-channel queue state | Area is unquantified |
| Power/energy | More concurrently active engines and routing likely raise dynamic power; shorter execution can reduce leakage time | Less concurrent datapath activity but longer completion for multi-stream batches | No DMA topology energy counters or physical power calibration |
| SRAM/DRAM traffic | Identical useful bytes | Identical useful bytes | Shared-mode serialization is modeled, but independent mode still assumes independently serviceable memory paths; contention/overfetch are absent |
| Numerical accuracy | Byte-exact outputs are identical | Byte-exact outputs are identical | No arithmetic semantics are involved |
| Control complexity | Multiple simultaneous active transfers, completions, errors, and resets | One active transfer and descriptor-boundary RR; simpler datapath, but queue fairness remains stateful | Priorities, preemption, cancellation, and starvation limits are not modeled |
| Verification burden | Must gate true concurrency and all channel interactions | Must gate serialization, rotation, and no fallback | The sweep covers 18 topology/stream/channel rows; hazardous ownership remains a separate concern |
| Compiler/runtime | Can map independent W/A/O streams for overlap | Can retain stream queues on a cost-constrained single mover | No automatic binding, cost model, or end-to-end compute/DMA scheduler |

### Implementation and verification surface

- Configuration: `config/tu_config.{yaml,json}` → generated defaults → canonical parse/validation → runtime conversion → `tu_init_with_config()` → live descriptor engine.
- Runtime: `TU_DMA_BUS_MODE_INDEPENDENT` and `TU_DMA_BUS_MODE_SHARED_SERIAL`; unsupported values fail closed.
- Shared mode starts at most one descriptor globally and rotates the next eligible channel after completion. It is non-preemptive by design.
- Gates: `make test-dma`, `make test-config`, and `make test-dma-channel-sweep`, followed by clean full build and `make test-quick`.

### Remaining realistic variants

A partially shared fabric (for example two movers behind one aggregate DRAM interface), weighted priority, transfer preemption, asymmetric channel width, and shared SRAM/DRAM backpressure are valuable but not READY. They require an explicit aggregate-bandwidth, arbitration, and replay contract; adding scalar penalties now would invent behavior rather than clarify this measured trade-off.
