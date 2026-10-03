# DMA DRR cost granularity

**Date:** 2026-10-03
**Mode:** pre-spec exploration
**Evidence:** `tests/test_dma_arbitration_sweep.c`

## Architecture question

Should deficit round-robin (DRR) charge each descriptor's exact selected cost, or round every charge up to an integer number of configured quanta?

Both contracts are physically plausible:

- `exact` subtracts useful bytes, occupied bytes, or modeled service cycles without quantization. It preserves byte/cycle-sensitive fairness and is the historical cmodel behavior.
- `quantum_rounded` charges `ceil(cost / quantum) × quantum`. This represents a scheduler that accounts in whole service tokens or allocation slots rather than carrying fine-grained descriptor costs. It deliberately treats every nonempty sub-quantum descriptor as one full unit.

The granularity is independent of the existing cost metric. Byte metrics use `drr_quantum_bytes`; service-cycle charging uses `drr_quantum_cycles`. Zero-cost descriptors remain zero-cost.

## Executable experiment

Command:

```sh
make test-dma-arbitration-sweep
```

Controls:

- two queues behind one shared-serial, non-preemptive path;
- work-conserving DRR, useful-byte charging, 64-byte quantum, unit weights;
- channel 0 has four 16-byte descriptors; channel 1 has one 64-byte descriptor;
- 256-bit interface, 50-cycle read base, zero visible issue cost;
- SRAM bandwidth metering disabled.

Measured matrix:

| Cost granularity | Channel 0 completions | Channel 1 completion | Batch completion |
|---|---|---:|---:|
| `exact` | 52, 103, 154, 205 | 257 | 257 |
| `quantum_rounded` | 52, 155, 206, 257 | 104 | 257 |

Exact charging lets channel 0 spend one 64-byte visit on four 16-byte descriptors, so the peer waits until cycle 257. Quantum rounding charges each 16-byte descriptor as 64 bytes and rotates after the first one, improving the peer's completion by 59.53% (`257 → 104`). It delays channel 0's second completion by 50.49% (`103 → 155`) and fourth completion by 25.37% (`205 → 257`). Batch completion remains 257 cycles. All 128 bytes are identical and useful/occupied traffic is unchanged. This is service-allocation granularity, not a throughput or traffic gain.

## Runtime implementation and compatibility

Configuration:

```yaml
dma:
  arbitration: "deficit_round_robin"
  drr_cost_granularity: "quantum_rounded"
```

The setting propagates through YAML generation, the checked-in generated runtime header, shipped JSON, canonical parsing and validation, canonical-to-runtime conversion, top-level initialization, and the live scheduler. `exact` is numeric zero and remains the generated, canonical, zero-initialized, and live-engine default.

Unsupported IDs and strings fail without changing the accepted mode. A live granularity change clears all deficit and continuation state: exact residual credit has no stable interpretation after switching to quantum units, and retaining it would make the first post-change decision history-dependent on the old accounting contract.

## Multi-objective trade-offs

| Dimension | `exact` | `quantum_rounded` |
|---|---|---|
| Throughput | Same measured batch completion | Same measured batch completion; sustained throughput uncalibrated |
| Latency/fairness | Preserves descriptor-size distinctions; a burst of tiny descriptors can spend one visit | Gives each nonempty sub-quantum descriptor one full allocation unit; protects peers from tiny-descriptor bursts but delays the tiny-descriptor channel |
| Area/resources | Fine byte/cycle deficit value and exact subtraction | Token-scale accounting can permit narrower counters when implemented directly in units, but the current cmodel still stores 64-bit scaled credits; physical savings unquantified |
| Power/energy | More fine-grained arithmetic state in a plausible implementation | Potentially simpler token arithmetic, but more rotations/selections for tiny descriptors; net energy unquantified |
| SRAM/DRAM traffic | Unchanged | Unchanged; only service order changes |
| Numerical accuracy | Unchanged; byte-exact movement | Unchanged |
| Control complexity | Exact cost producer and subtraction | Ceiling conversion plus unit contract; power-of-two quanta can use shifts, but arbitrary implementation timing is unmodeled |
| Verification burden | Exact mixed-size costs and residuals | Adds sub-quantum, aligned, multi-quantum, every cost metric, zero cost, mode-switch reset, and overflow boundaries |
| Compiler/runtime | Descriptor coalescing and size directly influence share | Software must understand that splitting one transfer into several sub-quantum descriptors consumes several allocation units |

A fine-grained QoS mover may hard-wire exact charging. A command-slot or token-oriented controller may choose quantum-rounded charging to prevent many tiny descriptors from consuming one visit, accepting descriptor-fragmentation sensitivity. The pre-spec cmodel preserves both.

## Fidelity limits

- Arbitration is descriptor-boundary and non-preemptive; neither mode interrupts an active transfer.
- Quantum rounding affects scheduler cost only. It does not pad payload, create protocol traffic, or change completion service cycles.
- The cmodel stores scaled 64-bit deficits in both modes, so it does not quantify the counter-width reduction that a token-native RTL implementation might achieve.
- Finite command credits, backpressure, producer timing, shared SRAM/DRAM contention, stateful DRAM service, deadlines, physical area/power/energy, and calibration remain unmodeled.
- The matrix demonstrates deterministic finite-queue allocation, not sustained weighted fairness.

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
