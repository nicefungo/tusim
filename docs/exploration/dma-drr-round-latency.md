# DMA DRR virtual-round latency

**Date:** 2026-10-04
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

Should deficit round-robin (DRR) accumulate credit through unsuccessful virtual rounds with no modeled time, or expose a scheduler cycle cost for each complete round that examines all configured queues but cannot yet issue a head descriptor?

Both are physically plausible:

- `drr_round_issue_cycles=0` is the compatibility lower bound. It represents combinational/multi-round credit accumulation, a scheduler whose control time is hidden by another stage, or a model that deliberately excludes arbitration latency.
- A nonzero value represents an iterative scheduler that updates deficits and re-evaluates eligibility over one or more registered cycles per unsuccessful full round. Values 1 and 4 are representative low- and higher-control-latency points, not calibrated implementations.

The cost is charged once after a complete unsuccessful round, not per queue probe and not for the final successful round. Round-robin and priority arbitration do not consume it. This separation avoids conflating queue count with a physical comparator pipeline that is not modeled.

## Runtime contract

```yaml
dma:
  arbitration: "deficit_round_robin"
  drr_quantum_bytes: 64
  drr_round_issue_cycles: 1
```

The accepted range is `[0,1024]`. Zero is the generated, canonical, zero-initialized, and live-engine default. Unsupported values fail without changing the accepted live value. Changing the live setting clears deficits and continuation state so the next decision cannot inherit credits accumulated under a different timing contract.

The setting propagates through YAML/JSON generation, the checked-in generated header and runtime defaults, canonical parsing/validation, canonical-to-runtime conversion, top-level initialization, and the live descriptor scheduler. Elapsed-time addition saturates at `UINT64_MAX`.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Controlled workload:

- shared-serial, non-preemptive DMA path;
- useful-byte DRR, exact charging, 64-byte quantum, unit channel weights;
- one 256-byte linear load on channel 0, requiring four credit grants and therefore three unsuccessful full rounds before issue;
- 256-bit interface, 50-cycle read base, zero visible burst-issue cost;
- SRAM bandwidth metering disabled;
- byte-exact destination comparison.

Measured matrix:

| Cycles / unsuccessful round | Unsuccessful rounds | Descriptor completion | Batch completion | Added vs instant |
|---:|---:|---:|---:|---:|
| 0 | 3 | 59 | 59 | 0 |
| 1 | 3 | 62 | 62 | 3 cycles (+5.08%) |
| 4 | 3 | 71 | 71 | 12 cycles (+20.34%) |

The result follows `completion = 59 + 3 × drr_round_issue_cycles` for this exact workload. Useful and occupied bytes and copied values are identical. The nonzero rows correct an optimistic scheduler-control omission; they are not an optimization.

## Multi-objective trade-offs

| Dimension | Instant (`0`) | Registered/iterative (`>0`) |
|---|---|---|
| Throughput | Lowest deterministic batch time when large descriptors need multiple quanta; sustained throughput is uncalibrated | Adds bubbles proportional to unsuccessful rounds; impact grows when descriptor cost greatly exceeds weighted quantum |
| Latency | 59 cycles in the measured 256-byte case | 62 cycles at 1 and 71 at 4; small descriptors already eligible in the first round pay no added cost |
| Area/resources | May imply wider combinational eligibility/credit logic or hidden pipeline resources; magnitude unquantified | Can represent a smaller iterative selector with registers/counter update; actual gate/register savings unquantified |
| Power/energy | Potentially more combinational switching in one decision; not modeled | More active control cycles and clocked state, but potentially simpler per-cycle logic; net energy unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged; the setting creates no payload, padding, or protocol transaction |
| Numerical accuracy | Unchanged; transfer remains byte exact | Unchanged |
| Control complexity | Simplest timing abstraction, but strongest same-cycle assumption | Requires unsuccessful-round detection, delay accounting, and a stable deficit state across the modeled delay |
| Verification burden | Compatibility and zero-delay gates | Adds exact round counting, timestamp arithmetic, invalid-value immutability, live-state reset, and overflow boundaries |
| Compiler/runtime | Quantum can be tuned for fairness without scheduler-time feedback | Descriptor size and quantum jointly determine control latency; tilers may avoid extreme cost/quantum ratios, but no automatic policy is implemented |

A high-frequency controller may choose an iterative implementation to close timing or reduce combinational fan-in. A small fixed-function design may hard-wire an instant or hidden scheduler if descriptor geometry guarantees few rounds. The cmodel retains both rather than declaring one universally preferable.

## Fidelity limits

- The model charges an aggregate full-round cost. It does not model per-probe timing, queue-count-dependent trees, pipelined overlap between rounds, descriptor fetch, or asynchronous CDC.
- The delay advances the deterministic cmodel clock before descriptor issue; no other producer can inject work during the aggregate delay inside one `tu_dma_tick()` call.
- Arbitration remains descriptor-boundary and non-preemptive.
- The measured result uses one backlogged channel. Multi-channel ordering remains governed by the existing DRR cursor, weights, cost metric, granularity, service, and idle-credit policies.
- Shared SRAM/DRAM contention, finite command credits, backpressure, physical area/power/energy, clock feasibility, and calibration remain unmodeled.

## Verification

```sh
make test-config-generation
make test-config
make test-dma
make test-dma-arbitration-sweep
make config-docs
make clean && make
make test-quick
```
