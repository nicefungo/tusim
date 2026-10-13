# DMA DRR grouped scan width

**Date:** 2026-10-13. **Mode:** pre-spec exploration, uncalibrated aggregate scheduler timing.

## Question and evidence-backed scope

The successful-selection study retained hidden/fixed, full-serial, and early-stop-serial alternatives, but charged every queue probe serially. Can a partially parallel selector represent an intermediate resource/latency point without changing DRR credit or service order?

Yes: `dma.drr_scan_width` selects **1..8 cursor-relative queue positions per scan group**. Zero inherits width 1. Widths need not be powers of two and can exceed the configured queue count: the last group simply has fewer valid lanes. A single scanner is plausible for a small area budget; 2/3/4-way evaluation trades more queue-state read/comparison resources for fewer visible control groups; all-queue evaluation may suit short-descriptor/high-rate movers. None is universally preferable. This is a cost model, not an RTL implementation or proof of read-port feasibility.

Audit also revisited SRAM arbitration, compressed-domain MMA delivery, and physical dataflow traffic. They remain blocked on simultaneous requester/port, codec/token/FIFO, and physical movement/overlap contracts respectively; no READY item was established from those existing candidates. The intermediate DRR width omission was explicitly identified in the preceding scan-cost studies and is now the one implemented item.

## Contract and implementation

```json
{"tu":{"dma":{
  "channels":8, "async_mode":true, "bus_topology":"shared_serial",
  "arbitration":"deficit_round_robin", "drr_quantum_bytes":64,
  "drr_round_cost_mode":"per_visited_channel",
  "drr_round_issue_cycles":1, "drr_select_issue_cycles":1,
  "drr_scan_width":2
}}}
```

Let `C` be configured queues, `W` effective scan width, and `P` probes from the cursor through the selected queue (including empty/ineligible positions). Define `groups(n) = ceil(n/W)`.

- `fixed`: unsuccessful-round and successful-selection cost multipliers remain 1, independent of width.
- `per_channel`: both cost multipliers are `groups(C)`.
- `per_visited_channel`: unsuccessful rounds use `groups(C)`, successful selection uses `groups(P)`.
- Issue-cycle settings now price a scan **group**, not each individual lane, when width > 1. Width 1 exactly preserves historical formulas.
- Width changes preserve deficits, continuation, cursor, and descriptor ownership. No admission, fairness order, DRR charging, or payload-service formula is changed. The existing ordered algorithm supplies lane priority; later lanes in the winning group do not speculatively commit quantum grants.
- Fixed/zero control costs, independent paths, and non-DRR policies retain their behavior. A new setter accepts 0..8 and rejects unsupported values without mutation. Canonical JSON requires an integer; negative, oversized, fractional, Boolean, null, and string values fail validation. Existing initializer signatures are unchanged; public struct layouts grow, so binary consumers must rebuild.

Full path: shipped YAML/JSON → generator/default header → canonical config/default/parser/validation → runtime conversion/default → top-level initializer → live DMA engine and shared scan-group helper. Generated config reference reports the new field. The generated-header gate checks byte identity, nondefault width 3 with a top-level compile, and invalid YAML widths.

## Measured matrix

`make test-dma-drr-scan-width` executes 874 timing rows and nine simultaneous-queue order gates. Each timing row loads or stores 256 bytes exactly, with a 256-bit interface, 50-cycle base, no visible burst-command cost, SRAM metering disabled, unit DRR weights and a 64-byte quantum. Three unsuccessful deficit rounds precede selection. Both round and selection cost are one cycle per group. The baseline completion is 59 cycles before these control costs. Completion and retirement/batch cycle are checked independently against an arithmetic oracle.

| Width | Fixed | Early-stop target 0 | Early-stop target 3 | Early-stop target 7 | Full scan |
|---:|---:|---:|---:|---:|---:|
| 1 | 63 | 84 | 87 | 91 | 91 |
| 2 | 63 | 72 | 73 | 75 | 75 |
| 3 | 63 | 69 | 70 | 71 | 71 |
| 4 | 63 | 66 | 66 | 67 | 67 |
| 8 | 63 | 63 | 63 | 63 | 63 |

All entries are **model core cycles**, not nanoseconds or measured hardware latency. In the last-target case, widths 2/3/4/8 reduce modeled completion by 17.58%/21.98%/26.37%/30.77% versus width 1. Three failed rounds dominate even when the selected queue is close to the cursor. Width 8 equals the fixed-cost row here but is not semantically identical hardware: parallel queue-state resources are not modeled by fixed timing.

The complete matrix retains widths 0..8, all three cost modes, cursor positions 0 and 6, every target 0..7, both directions, wrap-around, group tails and zero-cost controls. Nine queued gates admit channel 7 and channel 0 together from cursor 6, verify channel 7 precedes channel 0 at every effective width, and compare exact completion, batch time and bytes. Raw transcript: `evidence/dma-drr-scan-width-2026-10-13/scan.txt`.

## Gains and sacrifices

| Dimension | One-wide scanner | Partially/all-queue parallel groups |
|---|---|---|
| Throughput | More visible control bubbles for short descriptors | Fewer modeled bubbles; sustained/physical throughput unquantified |
| Latency | Predictable lane rate but cursor-dependent early stop | Measured reduction above; can remain memory/base-latency dominated |
| Area/resources | Reused queue-state path/comparator | Expected additional state-read ports/replicas, comparators, selection and grant-mask wiring; unquantified |
| Power/energy | Fewer concurrent lanes, more scan groups | More simultaneous switching, fewer clocked groups; net energy direction is trace/implementation dependent and unquantified |
| SRAM/DRAM traffic | 256 useful and occupied bytes per timing row | Identical payload traffic; queue-state reads are not payload SRAM traffic |
| Numerical accuracy | Byte-exact movement | Identical; no arithmetic change |
| Control complexity | Serial cursor and update control | Ordered intra-group winner/conditional credit writes and tail-lane masking; modeled only in aggregate |
| Verification burden | Existing serial/default gates | Adds width, non-divisible tails, wrap, direction, config, state-preservation and simultaneous-ready gates |
| Compiler/runtime | Placement influences early-stop distance | Width is an architecture input; placement/descriptor splitting still affects cost, with no automatic compiler policy |

The gain buys expected hardware resources and verification effort. Choosing width 8 just because its local row is shortest would violate the purpose of this cmodel; all widths and existing cost modes are retained.

## Fidelity limits and verification

Both scheduler stages share one width. Independent stage widths, queue-state read conflicts, speculative lane updates, finite command credits, backpressure, pipelined selection/data overlap, producer arrivals during the atomic tick delay, physical area/power, timing closure and calibration are unmodeled. Width does not prove any such mechanism. Shared service remains descriptor-boundary and non-preemptive.

Executed gates:

```sh
make test-config-generation test-config test-dma
make test-dma-arbitration-sweep test-dma-drr-scan-width
make config-docs
make clean && make
make test-quick
make test
```

Focused config (31/31), DMA (15/15), historical arbitration sweep, generated-header gate, the new 874-row/nine-queue-gate sweep, clean full build, and quick suite pass. A fresh archive of the exact staged snapshot independently passes quick/config/arbitration gates; a separate clean rebuild of that snapshot with AddressSanitizer, UndefinedBehaviorSanitizer and leak detection passes the new sweep without sanitizer diagnostics. The broader `make test` stops at attention: deterministic-small and causal-mask tests report infinite error (7/9 pass in the modified checkout). A fresh archive of pre-change `124eda0` reproduces both infinite-error failures and additionally fails its all-ones test (6/9 pass in that run). This is a pre-existing attention blocker, not a green full-suite claim; attention engine changes are outside this heartbeat's scope. The subsequent `make -k test` continues through the aggregate and also reproduces two deterministic pre-existing failures: `test-agen` uses the original row count as the transposed row-major pitch (3 instead of 4 in its 3×4 fixture), and `test-multicast` still expects an overflowing descriptor to be marked completed despite the already implemented fail-closed span contract. Both fail identically in the clean baseline archive. Attention failures vary between invocations, so no stable defect count or attribution is inferred from a single run. Address-pitch correction and multicast regression-contract repair are recorded as two READY follow-ups; attention hardening remains blocked on explicit instruction. Full and baseline transcripts are retained alongside scan evidence.
