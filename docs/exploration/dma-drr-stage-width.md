# DMA DRR independent round and selection scan widths

**Date:** 2026-10-23. **Mode:** pre-spec architecture exploration. **Evidence level:** executable, uncalibrated aggregate scheduler-control timing; not RTL or physical throughput.

## Question, rationale, and alternatives

The grouped scanner study (`dma-drr-scan-width.md`) forced unsuccessful deficit-round scans and successful descriptor selection to share one width. Does that coupling hide useful provisioning alternatives when workloads need different numbers of unsuccessful rounds?

The existing executable algorithm already separates those timing producers. A repeated credit-update scan can plausibly use multiple update lanes while a final winner/issue path remains narrow; conversely a simple round engine can feed a wider eligibility/winner path when most requests already have enough credit. A common-width, potentially time-multiplexed scanner remains a simpler alternative. These are architecture hypotheses represented by separate **costing widths**, not claims that this C loop implements distinct physical parallel stages. State-read/write ports, inter-stage storage, muxing, timing closure and resource sharing require a separate physical contract. Widths are not an additive area budget.

At entry there was no READY implementation row. Re-audited SRAM requester arbitration, direct compressed-domain MMA, physical dataflow traffic and padded-im2col DMA: their missing requester/port, token/FIFO, movement/overlap and tagged-fill/retirement contracts remain dependencies. This heartbeat executes one new bounded exploration and implements its useful omitted alternative, rather than manufacturing a feature from the redesign checklist.

## Runtime contract and complete configuration path

```json
{"tu":{"dma":{
  "channels":8, "async_mode":true, "bus_topology":"shared_serial",
  "arbitration":"deficit_round_robin", "drr_quantum_bytes":64,
  "drr_round_cost_mode":"per_visited_channel",
  "drr_round_issue_cycles":1, "drr_select_issue_cycles":1,
  "drr_scan_width":1, "drr_round_scan_width":8,
  "drr_select_scan_width":1
}}}
```

- `drr_scan_width` retains the common width: 0 inherits 1; explicit 1..8.
- New `drr_round_scan_width` and `drr_select_scan_width` each accept 0..8. **0 dynamically follows the common width**, including subsequent common-width setter calls; 1..8 overrides only that stage. Both default to 0, preserving existing JSON, generated/runtime defaults, memset-initialized canonical callers and old initializer surfaces.
- Let `C` be configured queues, `P` cursor-relative probes including the selected queue, `R/S` effective round/selection widths and `g(n,w)=ceil(n/w)`. `fixed` uses multiplier 1 for both costs; `per_channel` uses `g(C,R)` / `g(C,S)`; `per_visited_channel` uses `g(C,R)` / `g(P,S)`. Empty/ineligible queue positions still count. Width can exceed C; the last group is clipped. No power-of-two restriction.
- Only aggregate delay accounting changes. DRR visits, credits, continuation, cursor, queue ownership and payload service are unchanged. Later lanes after the winning position do not receive speculative credits.
- `tu_dma_set_drr_stage_scan_widths(round, select)` validates the pair before changing either field. Invalid values fail without any engine mutation. Valid changes preserve queued/active descriptors and do not retime an already active descriptor. No host-thread-safety or atomic arrival-processing claim is made.
- Independent paths, non-DRR arbitration and synchronous direct execution do not consume this async-scanner timing. Zero-cost and fixed-cost behavior remain available.
- Full path: YAML + shipped JSON → generator/macros/runtime struct/defaults → canonical defaults/parser/validation → runtime conversion → top-level initialization → live engine → existing shared group helper with stage-specific width. Config docs report both fields. Public struct layouts grow: binary consumers must rebuild; existing function signatures remain intact.
- JSON and YAML reject negative/oversized/fractional/Boolean/null/string widths. A supporting generator fix preserves quoted scalar types rather than converting quoted `"2"` into an integer. Shipped-header identity is unchanged except the new definitions/fields/defaults. The nondefault generated-header test now **executes** its default-constructor probe, instead of merely compiling its return expression.

## Measured matrix and workload reversal

All entries below are **model core cycles to completion/retirement of one isolated descriptor**, not nanoseconds or silicon measurements. Eight queues, cursor 0, target 7, 32-byte interface, 50-cycle directional base, no burst issue cost, unit weights, 64-byte useful-byte DRR quantum, one cycle per scan group, early-stop cost mode, SRAM bandwidth metering disabled. A 256 B descriptor needs three unsuccessful rounds; 32 B is immediately eligible. Useful and occupied payloads stay exactly 256 B / 32 B respectively; both load and store directions are checked byte-exactly.

| Round width | Select width | 256 B, three failed rounds | 32 B, zero failed rounds |
|---:|---:|---:|---:|
| 1 | 1 | 91 | 60 |
| 1 | 3 | 86 | 55 |
| 1 | 8 | 84 | 53 |
| 3 | 1 | 76 | 60 |
| 3 | 3 | 71 | 55 |
| 3 | 8 | 69 | 53 |
| 8 | 1 | 70 | 60 |
| 8 | 3 | 65 | 55 |
| 8 | 8 | 63 | 53 |

The 8/1 alternative reduces the 256 B example's completion 91→70 cycles (23.08%) but buys **nothing** for immediately eligible 32 B requests. The 1/8 alternative reduces that 32 B case 60→53 (11.67%), yet loses to 8/1 for three-round work (84 versus 70). This reversal identifies a useful independent provisioning dimension; it does not establish equal hardware cost between 8/1 and 1/8. Wide/wide is shortest in these rows but may demand more parallel control resources; no universal recommendation is made.

The full executable matrix retains every 0..8 override pair, all three cost modes, every target, cursor wrap, widths greater than a three-queue configuration, non-divisible tails, unequal stage issue costs, zero-cost controls and both directions: **8,425 timing rows**. Additional **162 simultaneous-ready queue/reconfiguration gates** verify order, exact timestamps, ordered credit updates, full-engine state preservation, active-request no-retiming and bytes. Three discriminating live-inheritance gates change the common width after submission and verify completion at 70/84/71 cycles. Separate controls check unsupported settings and unchanged independent/non-DRR/synchronous paths.

## Gain versus sacrifice across design objectives

| Dimension | Common/narrow provisioning | Independent or wider provisioning |
|---|---|---|
| Throughput | More visible control bubbles can limit short-request issue | Fewer modeled bubbles in the relevant regime; sustained throughput, overlap and physical clock are unquantified |
| Latency | Larger round/selection terms; predictable shared provisioning | Measured reversal above; optimizing the wrong stage yields little or no benefit |
| Area/resources | Potential reuse of one state-read/compare/update path | Expected extra lane/port, mux, mask and possibly stage-storage resources; allocation/sharing and physical area unquantified |
| Power/energy | Less parallel switching, potentially more clocked scan groups | More concurrent switching but fewer groups; net power/energy is workload/implementation dependent and unquantified |
| SRAM/DRAM traffic | Identical useful/occupied payload | No extra payload traffic; queue-state accesses are not counted as payload SRAM or DRAM bytes |
| Numerical accuracy | Byte-exact movement | Unchanged, no arithmetic or quantization approximation |
| Control complexity | Common-width state and one cost parameter | Two independently inherited overrides and ordered conditional updates; physical stage implementation unmodeled |
| Verification | Existing common-width matrix | Cross-stage tails, zero inheritance, live changes, exact order/timing and atomic rejection enlarge the verification surface |
| Compiler/runtime | Existing descriptor/channel placement | Architecture configuration may reflect expected round frequency; no automatic placement, quantum tuning or dispatch policy is added |

## Verification and limits

Baseline `450fb16` and final live checkout each passed `make clean && make`, `make test-quick`, config (31/31), DMA (15/15), arbitration/channel/binding/bus-width sweeps, convolution (12/12) and scatter/gather (15/15). The historical common-width sweep still passes 874 timing rows and nine queue-order gates. The initial quick command needed `LD_LIBRARY_PATH="$PWD"` to locate the existing shared library; setting that existing runtime prerequisite fixed the invocation without changing build/default architecture behavior.

The retained pre-implementation tracer **fails baseline with 91 rather than 70 cycles**, then passes the implementation at 70 with exact bytes. Fresh staged-snapshot build/quick/focused execution and a separate clean all-object ASan/UBSan/LSan rebuild also pass the new exercised paths. Commands and byte-preserved compressed raw logs are retained in `evidence/dma-drr-stage-width-2026-10-23/`; `MANIFEST.sha256` covers retained artifacts. The snapshot metadata records its exact Git input tree; no handoff or user-owned `TERMS.md` is included.

`make -k test` is **not green**: attention is the only failed target in both baseline and final runs. Baseline aggregate passes 7/9 attention tests (identity/scale failures); final passes 6/9 (identity/deterministic-small/causal failures). Three further unchanged-baseline runs pass 6/9, 7/9, 7/9 and reproduce the latter failure set. These observations establish a pre-existing variable attention blocker, not root cause or complete failure-set equivalence. No attention implementation was changed.

The model still charges control delays atomically inside one tick. Producer injection within a scan, actual parallel state access/write conflicts, finite credits/FIFOs, selection/data overlap, shared-memory contention, physical layout, power/energy and calibration remain unmodeled. Width selection alone does not prove those mechanisms or admit a physical clock. The extended sanitizer invocation also exposed a historical test-harness leak: `test-dma` lost 3,136 bytes in 14 descriptor allocations on both unchanged baseline and final implementation. That was a READY independent follow-up at the end of the 2026-10-23 heartbeat, while both width sweeps separately passed sanitizers. **Resolved 2026-10-24:** harness-only cleanup now makes ordinary DMA 15/15 sanitizer-clean, with 49 descriptor OOM, 45 real consuming-rejection, pending-assertion and active-hold controls. Production ownership/config/timing remain unchanged. See `dma-test-descriptor-cleanup.md` and its complete retained evidence; the historical 2026-10-23 raw logs remain unmodified.

Reproduce from the repository root:

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make clean && make
make test-dma-drr-stage-width test-dma-drr-scan-width
make test-config-generation test-config test-dma
make test-dma-arbitration-sweep test-dma-channel-sweep test-dma-binding-sweep test-dma-bus-width-sweep
make test-conv test-scatter-gather
make test-quick
make -k test # known variable attention failures; retain its nonzero status
```
