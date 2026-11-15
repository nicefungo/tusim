# Log-softmax: preserve the small negative normalizer

**Date:** 2026-11-15. **Mode:** pre-spec numerical-model correction.
**Baseline:** `20c1c73faeb588350286f40188fe74bd41a1d176`.
**Question:** after fixing finite negative tails, does LOG still lose the
small negative output of a dominant logit, and can it retain that information
without changing probability modes, storage, or SRAM accounting?

## Finding and bounded correction

For input `[0,-20]`, baseline LOG returns `[0,-20]`. The independent
`-log1pl(expl(-20))` reference for the dominant element is
`-2.06115362031e-9`: baseline relative error is 1. This is not FP32 output
underflow. Casting the accumulated exp-sum to FP32 before `logf` rounds it to
one. Simply switching to `log(double_sum)` is insufficient: addition to one
also loses much smaller contributions in double precision.

LOG now tracks the position of **one** maximum and accumulates the other
existing FP32 exponentials in a separate double residual. It evaluates
`log1p(residual)` and subtracts the FP32 result from retained shifted logits.
Excluding exactly one maximum matters for tied maxima; using the wrong
position matters for permuted rows. The original sum is still accumulated in
its original order for `sum_out` compatibility. A returned FP32 sum of 1 may
therefore coexist with a nonzero log-normalizer, intentionally.

This is a correction to the existing LOG contract, not a new architecture
setting. Existing descriptor modes STANDARD=0, LOG=1, ONLINE=2, scale/mask
ordering, host/flat/2D APIs, ABI, YAML/JSON configuration and default mode are
unchanged. LOG result bits can change, including ordinary values. The existing
`shifted < -87` exponential cutoff is **not changed**. No guarantee is made for
relative error after that cutoff, nonfinite inputs, preprocessing/subtraction
overflow, arbitrary aliasing, or all-masked rows.

## Complete measured matrix

The retained before/after transcripts have **30 rows**: all three modes at gaps
`{0,12,16,17,20,40,80,87,88,104}`, input `[0,-gap]`, no mask/scale, in-place FP32,
128-byte default SRAM, no time advance after prefill. Selected LOG rows:

| Gap | Before: dominant output | After: dominant output | Independent reference |
|---:|---:|---:|---:|
| 17 | 0 | -4.13993781478e-8 | -4.13993763309e-8 |
| 20 | 0 | -2.06115369217e-9 | -2.06115362031e-9 |
| 40 | 0 | -4.24835413114e-18 | -4.24835425529e-18 |
| 80 | 0 | -1.80485132858e-35 | -1.80485138785e-35 |
| 87 | 0 | -1.64581145375e-38 | -1.64581143108e-38 |
| 88 | 0 | 0 | -6.0546018954e-39 |
| 104 | 0 | 0 | -6.81355682155e-46 |

The last two rows deliberately expose the retained cutoff boundary rather than
being passed off as mathematical conformance. Every row still performs two
FP32 reads and two FP32 writes and returns **8 local SRAM penalty cycles**.
This is not physical compute latency or throughput.

## Alternatives and multi-objective trade-offs

| Choice / dimension | Gain, motivation and sacrifice |
|---|---|
| STANDARD probability output | Existing default for probability/attention-weight consumers. No output change in the compatibility gate. It can round a dominant probability to one; taking a later log is not a substitute for LOG. |
| ONLINE probability output | Existing runtime streaming-rescale organization retained. Its FP32 accumulation order differs and it still buffers an entire row here. Neither physical streaming execution nor a universal performance ranking is established. |
| LOG residual normalizer | Preserves small negative outputs for log-domain consumers in the tested finite normal-exponential range; retains large negative tails. Adds an anchor index, a double residual accumulator and LOG-side accumulation/control, and uses a double `log1p` instead of float `logf`. |
| Rounded-sum LOG approximation | Simpler arithmetic/state might be a hardware choice only with an explicit approximation/error contract. Silent rounding to zero under the current LOG API is not retained as a selectable correctness bug. A named bounded approximation remains a separate proposal, not an implemented variant. |
| Throughput / latency | Physical values **unquantified**: no log-unit issue, pipeline, sharing or calibration model. Extra host work does not imply a measured device slowdown; no speedup is claimed. Local penalties are unchanged. |
| Area / resources | No additional row allocation or SRAM reservation. One host index and one residual scalar are added; a corresponding hardware design could need extra accumulation/selection or reuse existing resources. Physical area is **unquantified**. |
| Power / energy | More accumulation and a changed log implementation may increase arithmetic/control activity; reuse and scheduling could alter that. Net power/energy **unquantified**, not inferred from accuracy. |
| SRAM / DRAM traffic | Exact SRAM event counts and local penalties unchanged. No DRAM requests occur in this standalone gate; actual physical traffic/energy are **not measured**. |
| Numerical accuracy | Broad near-zero gate: maximum relative error `1.03862458607e-7`, below the `4e-7` bound with **no absolute tolerance floor**. Baseline has 7,665 failed assertions. Not a correctly-rounded or all-domain guarantee. |
| Control / verification | Anchor tracking must work for ties and every row order. Separate normalizer/output and compatibility-statistics contracts require independent oracles and mutation controls. Physical control cost **unquantified**. |
| Compiler / runtime | Existing APIs and formats retained; no lowering or compiler configuration changed. Exact LOG snapshots can need updates. `sum_out` is diagnostic FP32 data, not a lossless representation of the internal log-normalizer. |

## Executed verification

- Clean baseline `make clean && make && make test-quick`: pass.
- New gate: **4,880 cases / 2,433,342 assertions**, zero failures.
  - 4,680 descriptor cases: six widths `{1,2,3,8,65,257}`, thirteen gaps
    including representable neighbours of 17, five scales (including zero-skip
    and negative), three mask choices, two placements and unique/tied maxima.
    Three rows put maxima first, middle and last.
  - 200 host/flat/2D wrapper cases; NULL statistics, disabled metering, full SRAM
    outside-output guards, counter/return agreement and empty host controls.
  - Independent `expl`/`log1pl` oracle; a naive `log(sum(exp))` oracle would
    itself lose the gap-80 control. Unexpected nonfinite values fail first.
- Complete baseline archive plus the same new gate: exit 1, **7,665 failures**.
- Three safe mutants rejected: double-sum `log` (3,930 failures), excluding all
  tied maxima (484,860), and always using anchor zero (361,430). Their nonzero
  exits must come with numerical-failure summaries, not compilation failures.
- Existing finite-range gate: **4,520 cases / 305,801 assertions**, pass.
  STANDARD/ONLINE output-inclusive fingerprints remain
  `7835006ea9ecf509` / `cafd1af0bdab8391`; all modes' separate statistics/access/
  penalty fingerprints are unchanged. LOG output-inclusive fingerprint changes
  intentionally to `2b3e6f53a5424e9d`.
- Final clean build, quick and full `make test`: pass; legacy softmax **15/15**,
  overflow gate and historical mode/attention sweeps pass. Both prior sweep
  stdout tables are byte-identical; this does not revalidate their historical
  cross-engine timing interpretation.
- All-object **ASan/UBSan/LSan** build: new gate and matrix, existing range gate,
  legacy softmax and overflow paths pass. This does not sanitize every full-suite
  path. The deliberate tied-max mutant produces an unused-anchor compiler warning;
  it is not a production warning or a failed compile.

```sh
make clean && make
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
make test-softmax-log-nearzero test-softmax-log-range test-softmax test-softmax-overflow
./test-softmax-log-nearzero --matrix
python3 tests/check_softmax_log_nearzero_mutants.py
make test-softmax-sweep test-softmax-attention-sweep
```

The new target is statically linked and integrated into quick/full/clean/ignore.
[Evidence](evidence/softmax-log-nearzero/README.md) retains the matrices, raw test
logs, baseline pin, source hashes and commands. The final staged snapshot is
independently rebuilt before publication; its log is retained there as well.

## Backlog audit and next bounded design question

Rechecked current normalization wrapper scratch arithmetic, context SRAM
snapshot/time ownership, scalar arbitration mode consumption, and im2col padding
sentinel/range handling. Their recorded contracts remain blocked; none was
silently declared implemented or excluded for being slower.

The gap-88 control supplies evidence for **one READY follow-up**: runtime
exponential-range choices for this standalone softmax engine:

1. Legacy `-87` cutoff, preserving exact default behavior (not falsely named FTZ).
2. Flush subnormal exponential results to zero, retaining normal results.
3. Gradual-underflow exponential results, preserving representable small values.

These are alternative numerical datapaths, not universally ranked performance
settings. Flushing can simplify small-value handling; gradual support retains
information at potential normalization/control cost. Physical costs, throughput
and energy must remain unquantified until an arithmetic timing/resource model
exists. No such new policy is implemented in this run.

**Evidence/rationale:** the local matrix proves the missing small-value regime.
NVIDIA's first-party *Floating Point and IEEE 754*, CUDA 12.9 archive, sections
4.4 and 5.1, independently documents selectable subnormal support/flush behavior
and warns against assuming correctly rounded transcendental libraries. This
establishes hardware plausibility, not TU timing or CUDA-equivalent semantics.
Primary text inspected 2026-11-15; archive locator:
`https://docs.nvidia.com/cuda/archive/12.9.0/floating-point/index.html#compiler-flags`.

**Acceptance plan:** preserve a zero/default legacy descriptor policy and all
wrapper/host defaults; validate/reject unsupported values without side effects;
route every selected exponential, including online rescaling, through one
policy-aware mechanism while preserving legacy online behavior. Gate neighbours
of -87, the normal/subnormal boundary, minimum subnormal and underflow, ties,
row order, STANDARD/LOG/ONLINE, denominator/output consistency, scale/masks,
statistics, byte/penalty invariance and build/config propagation for the actual
runtime API. Use absolute/ULP classification at subnormal outputs, not the present
normal-range relative bound. General nonfinite/mask semantics and attention's
independent kernel are excluded from that bounded follow-up, not claimed covered.
