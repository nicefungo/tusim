# Realistic-Design Implementation Backlog

**Last audit:** 2026-07-19
**Scope:** Evidence-backed pre-spec alternatives; `PRODUCTION_TU_REDESIGN.md` is reference only.

## Status legend

- **READY:** reproducible evidence, valid model path, actionable tests
- **BLOCKED:** valuable but a dependency/model gap prevents honest implementation
- **DONE:** implemented, tested, and documented
- **EXCLUDED:** strong documented reason not to implement

## Candidates

| Priority | Candidate / alternatives | Hardware rationale and evidence | Gains and sacrifices | Observable now | State / dependency / verification |
|---|---|---|---|---|---|
| P0 | Weight stream: NONE / exact RLE / epsilon RLE | `weight-compression-rle-sweep.md`; real designs may omit a codec for fixed-rate simplicity or add RLE for clustered/block-pruned weights | RLE ranges from 3.0× traffic expansion to 585× reduction in measured patterns; decoder area/power/backpressure and epsilon accuracy costs remain unquantified | Encoded bytes, runs, payload DMA cycles, round-trip correctness, config parse | **DONE 2026-07-17.** Canonical runtime config, validation, portable 6-byte wire format, 13 tests, sweep target. Default NONE. |
| P0 | Per-tensor adaptive raw/RLE selection | `weight-compression-rle-sweep.md`: random 50–70% sparsity expands bare RLE traffic while clustered tensors compress strongly | Bounded fallback avoids expansion beyond a 16-byte frame; costs version/codec metadata, raw bypass, two decode paths, and compiler/runtime format state | Encoded size, selected codec, payload cycles, exact round-trip, corrupt-frame rejection | **DONE 2026-07-18.** Runtime `adaptive_rle`, explicit versioned frame, RAW/RLE selection, DMA/config integration, 17 focused tests. Default remains NONE. |
| P0 | 2:4 structured sparse MMA: dense / 2:4 | `tests/test_sparsity.c` and `docs/structured-sparsity.md` describe physically common 50% structured compute skipping | Potential compute/weight traffic reduction; metadata, muxing, pruning accuracy and compiler legality costs | Test intent exists, but no executable module | **BLOCKED.** Referenced `tu_cmodel/sparsity/structured_2of4.{c,h}` is absent from this checkout. Recover/implement module only with explicit engine-hardening scope, then run M/N/K utilization sweep. |
| P1 | Context switching: FULL_SRAM / LIVE_SRAM / CONTROL_ONLY; no preemption / RR / priority | `context-switch-state-scope.md`; physically plausible retention ranges from dedicated full context store to compiler-managed reload | Full preserves isolation but costs 2× retained bytes/switch; live reduces traffic/storage but requires liveness metadata; control minimizes hardware state but forces SRAM reloads and loses transparent isolation | Retained bytes, save+restore cycles, fixed pipeline cost, functional prefix/full restoration, scheduler behavior | **DONE 2026-07-19.** Runtime scope, per-region live prefixes, transfer BW, validation, 15 focused tests, and executable 3-size/3-scope sweep. FULL with BW=0 preserves legacy fixed-only timing/default semantics. |
| P1 | Interconnect: NONE / RING / MESH | `interconnect-topology-sweep.md` reports payload-only hop model | Mesh reduces modeled hops but costs links/router ports/area/power; prior doc overstates universal crossover because contention is absent | Analytical hops and payload cycles | **BLOCKED for further implementation.** Add contention/router/link-width model before using speedup as a hardware recommendation. Existing runtime modes retained. |
| P2 | Compression codec alternatives beyond RLE (bitmap/block, entropy) | RLE failure on randomly placed zeros suggests bitmap/block formats may be plausible | Better random sparsity behavior versus metadata/decode complexity and format proliferation | No implementation or sweep | **BLOCKED pending exploration.** First compare explicit bitmap/block format against RLE using identical patterns; exclude entropy coding unless decoder throughput can be modeled. |

## Audit notes

- The ready candidates were re-audited against current code and executable evidence. Existing major compute/dataflow/precision/topology sweeps already have runtime modes or documented model limitations.
- No realistic alternative was removed because it lost a local benchmark. NONE remains the backward-compatible compression default despite RLE's large gains on clustered data.
- `docs/exploration/TERMS.md` remains pre-existing untracked user work and was not modified or included in this heartbeat.
- Adaptive compression preserves NONE and explicit RLE as materially useful alternatives: fixed-function ASICs can avoid codec cost, controlled software stacks can omit the frame, and heterogeneous runtimes can choose bounded per-tensor fallback.
- Context retention alternatives are preserved rather than selecting the lowest-latency row. Full save supports transparent preemption; live-prefix save supports compiler/runtime liveness contracts; control-only supports low-area hardware that reloads scratchpads.
- Baseline infrastructure issue fixed: `test-asm` no longer depends on undocumented `/tmp` files, so a clean `make test-quick` is reproducible.
