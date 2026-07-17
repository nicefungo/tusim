# Realistic-Design Implementation Backlog

**Last audit:** 2026-07-17
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
| P0 | Per-tensor adaptive raw/RLE selection | Sweep shows random 50–70% sparsity expands traffic while clustered weights compress strongly; a compiler can choose after encoding | Avoids expansion but adds format tag, policy, metadata, two decode paths, compiler/runtime state | Encoded size and payload cycles | **READY.** Dependency: define a versioned framed stream; do not guess raw vs RLE from payload. Verify exact round-trip and never emit more bytes than raw plus frame. |
| P0 | 2:4 structured sparse MMA: dense / 2:4 | `tests/test_sparsity.c` and `docs/structured-sparsity.md` describe physically common 50% structured compute skipping | Potential compute/weight traffic reduction; metadata, muxing, pruning accuracy and compiler legality costs | Test intent exists, but no executable module | **BLOCKED.** Referenced `tu_cmodel/sparsity/structured_2of4.{c,h}` is absent from this checkout. Recover/implement module only with explicit engine-hardening scope, then run M/N/K utilization sweep. |
| P1 | Context switching: no preemption / RR / priority; variable save scope | `infra/tu_context.*` and `tests/test_context.c` already model state isolation and fixed overhead | Preemption/fairness versus SRAM copy traffic, latency jitter, storage, control and verification burden | Switch count, configured fixed overhead, functional isolation | **READY.** Add executable sweep over SRAM sizes/save policy and replace fixed-only cost with bytes/bandwidth + pipeline state cost. Verify save/restore correctness and monotonic size scaling. |
| P1 | Interconnect: NONE / RING / MESH | `interconnect-topology-sweep.md` reports payload-only hop model | Mesh reduces modeled hops but costs links/router ports/area/power; prior doc overstates universal crossover because contention is absent | Analytical hops and payload cycles | **BLOCKED for further implementation.** Add contention/router/link-width model before using speedup as a hardware recommendation. Existing runtime modes retained. |
| P2 | Compression codec alternatives beyond RLE (bitmap/block, entropy) | RLE failure on randomly placed zeros suggests bitmap/block formats may be plausible | Better random sparsity behavior versus metadata/decode complexity and format proliferation | No implementation or sweep | **BLOCKED pending exploration.** First compare explicit bitmap/block format against RLE using identical patterns; exclude entropy coding unless decoder throughput can be modeled. |

## Audit notes

- All 39 exploration documents were enumerated. Existing major compute/dataflow/precision/topology sweeps already have runtime modes or documented model limitations.
- No realistic alternative was removed because it lost a local benchmark. NONE remains the backward-compatible compression default despite RLE's large gains on clustered data.
- `docs/exploration/TERMS.md` was pre-existing untracked user work and was not modified or included in this heartbeat.
- Baseline infrastructure issue fixed: `test-asm` no longer depends on undocumented `/tmp` files, so a clean `make test-quick` is reproducible.
