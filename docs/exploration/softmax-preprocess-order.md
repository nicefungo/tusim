# Standalone softmax: scale/mask stage ordering

**Date:** 2026-11-17. **Mode:** pre-spec numerical datapath alternatives.
**Baseline:** `04d3c02310ceaf0cf549ed78dd1b2d1470e9e78e`.
**Question:** can a designer model both score-unit masks scaled with scores and
output-logit-unit biases added after scaling, without silently changing callers?

## Evidence and implemented alternatives

The existing descriptor executed mask, numeric-threshold replacement, then
scale. A finite additive mask and scale generally do not commute. The initial
executable tracer, scores `[2,0]`, mask `[0,-2]`, scale `0.5`, returned
`0.880797088146` for the first probability; scale-before-mask requires
`0.952574126822` mathematically. This is a **missing optional organization**, not
a violation of the old API. The new path returns `0.952574133873` in FP32.

The first-party PyTorch v2.9.0 `torch/nn/functional.py` reference formulation for
`scaled_dot_product_attention` scales QK scores before adding its attention
bias. Source retrieved 2026-11-17:
`https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/functional.py`.
Retrieved file SHA-256:
`5e96bb0880a4c6f4c0ae34c00a894e152412e68f6ffd073d2b442907eedeb5fa`.
This establishes a concrete producer contract for the second ordering, **not**
full PyTorch conformance, hardware implementation equivalence or measured speed.

`tu_softmax_desc_t.preprocess_order` now selects:

| Value | Organization | Realistic motivation / sacrifice |
|---:|---|---|
| 0 | `TU_SOFTMAX_MASK_THEN_SCALE` | A score-unit additive correction followed by one temperature/scale stage; preserves all previous callers. An output-logit-unit bias would need compiler transformation, which can change rounding or be impossible under the zero-as-skip convention. |
| 1 | `TU_SOFTMAX_SCALE_THEN_MASK` | Scale scores upstream, then apply an output-logit-unit attention bias; permits a bias magnitude independent of scale. Adds ordering/units metadata and selectable control; is not equivalent for callers expecting the entire biased score to be scaled. |

Define `S(x)` as one FP32 multiply by nonzero scale, or identity for either sign
of zero. Define `M(x)` as the existing optional additive/multiplicative float
mask, followed by the existing threshold stage. The paths are exactly `S(M(x))`
and `M(S(x))`, with individual FP32 rounding; no fused multiply-add is specified.
Both feed the unchanged STANDARD, LOG and ONLINE kernels and all three
exponential policies. Scaling is applied exactly once.

**Threshold compatibility:** `mask_fill` was misleadingly documented as a fill
value. The actual existing operation is: if nonzero, replace post-mask values
`<= mask_fill` (or NaN) with fixed `-1e9`. The operation is retained, and the
header now describes it. The threshold travels with the mask stage in either
ordering. It can operate without a mask pointer. It is neither a Boolean
exclusion mask nor an all-masked policy. Negative scale can reverse a finite
sentinel's effect; this change intentionally does not redefine that contract.
The masked wrapper does not construct a causal mask; the caller supplies it.

```c
tu_softmax_desc_t d = {0};
d.data_sram = sram;
d.elem_count = count;
d.in_place = true;
d.mask = additive_bias;
d.mask_is_additive = true;
d.scale = 0.5f;
d.preprocess_order = TU_SOFTMAX_SCALE_THEN_MASK;
uint64_t local_penalty = tu_softmax_execute(&d);
```

This is the **live C descriptor configuration path**, not a YAML/JSON setting.
Invalid selectors reject before allocation, SRAM access or statistics mutation.
Zero/default, designated-initializer callers and all existing wrappers retain
mask-before-scale. Binary descriptor consumers must rebuild and source callers
must initialize the whole descriptor. Attention's separate streaming kernel and
compiler emission are unchanged; no global selection is implied.

## Measured complete matrix

[evidence/softmax-preprocess-order/matrix.csv](evidence/softmax-preprocess-order/matrix.csv)
contains all **30 rows**: two orderings × STANDARD/LOG/ONLINE × five settings.
Every row uses scores `[2,0]`, additive mask `[0,-2]`, default 128-byte SRAM,
in-place FP32, a prefill and no cycle advance. Representative STANDARD output:

| Scale | Threshold | Mask then scale, y0 | Scale then mask, y0 |
|---:|---:|---:|---:|
| 0 (skip) | disabled | 0.982013761997 | 0.982013761997 |
| 0.5 | disabled | 0.880797088146 | 0.952574133873 |
| 2 | disabled | 0.999664664268 | 0.997527360916 |
| -0.5 | disabled | 0.119202919304 | 0.731058597565 |
| 0.5 | -1 | 1 | 1 |

In the last row, LOG still distinguishes ordering: the second result is
`-500000000` versus `-1000000000`, because one organization scales the fixed
sentinel. Identical probability output is not proof of identical preprocessing.
All 30 rows have **2 reads, 2 writes and 8 local SRAM penalty cycles**. The larger
matrix has identical per-bank/penalty fingerprints between orderings:
`ac88a20a04774d9b`. These are scalar local penalties, **not device latency**.

| Dimension | Gain, sacrifice and fidelity limit |
|---|---|
| Numerical semantics / accuracy | Each ordering matches a different producer equation; neither is universally more accurate. Multiplicative masks commute in real arithmetic but can differ under sequential FP32 rounding. Finite threshold behavior is preserved, not promoted to exact exclusion. |
| Throughput / latency | Same observed SRAM events/penalties; physical throughput and arithmetic latency **unquantified**. No pipelined multiplier/adder timing or overlap model exists here. |
| Area / resources | Same allocated row storage and SRAM footprint; one descriptor selector and an extra host preprocessing loop in the new mode. A fixed implementation may wire one order; selectable hardware may require routing/mux/control. Resource area is **unquantified**. |
| Power / energy | Same logical multiply/mask operation count where enabled. Stage placement can change routing/register activity; neither an energy gain nor a definite net direction is established. Physical power/energy **unquantified**. |
| SRAM / DRAM traffic | Input/output SRAM events unchanged. Masks are host-pointer reads outside the SRAM meter; no mask DMA or standalone DRAM transaction is modeled. Physical memory traffic **unquantified** beyond those logical events. |
| Control / verification | Validation plus two stage orders across three softmax and three exponential modes; threshold and zero-as-skip rules increase verification obligations. No stochastic arithmetic or PRNG introduced. |
| Compiler / runtime | Producer must specify score/bias units and select the matching order. Default preserves prior behavior; binary clients rebuild. Compiler lowering and independent attention selection are not implemented. |

No third permutation is needed for this two-stage question. Fused arithmetic,
Boolean masks, different threshold placement independent of the mask, nonfinite
and all-masked policies require distinct numerical/error and compatibility
contracts; they are not excluded because of speed. Arbitrary SRAM/host aliasing,
non-four-byte scalar SRAM and preprocessing overflow are outside this gate.
The 2026-11-18 [row-schedule follow-up](softmax-row-schedule.md) separately adds
safe contiguous input/output overlap and fixed forward/reverse admission;
host mask/statistics aliasing remains unsupported.

## Verification executed

- Baseline `make clean && make` and `make test-quick`: pass. Minimal tracer RED
  before production changes, then GREEN after selector wiring.
- New focused gate: **9,145 cases / 513,010 assertions**, zero failures. Covers
  two orders, three output modes, three exponential policies, four widths,
  seven scales (including signed zero, unity, negative and non-binary), absent/
  additive/multiplicative masks, three threshold settings, two placements and
  three rows per execution. Includes 72 invalid-selector cases and the masked
  wrapper. Exact guards, descriptor/mask preservation, optional statistics,
  deterministic replay, disabled meter and complete per-bank fingerprints pass.
- Oracle independently rounds FP32 preprocessing using double intermediates
  and uses `expl`/`log1pl`; it does not call the production host helper. Finite
  output classification and exact expected zero precede relative comparisons.
  Tested two-pass relative tolerance is `8e-7`; ONLINE uses
  `(2*cols+4)*FLT_EPSILON` for bounded FP32 accumulation, not universal accuracy.
- Complete baseline archive rebuilt with this harness and only the new field
  omitted: **24,330 assertions fail**, exit 1. Invalid-new-selector cases cannot
  be exercised against the old API. This demonstrates missing optional behavior.
- Legacy-only **4,536 cases / 256,284 assertions** pass on old and new builds;
  output/statistics/penalty fingerprints and summaries are byte-identical.
  Prior exponential 81-row matrix, mode-sweep stdout and attention-sweep stdout
  are byte-identical. No historical cross-engine timing claims are recertified.
- Five safe source mutants rejected: ignored selection (24,330 failures),
  changed default (24,334), doubled scale (59,094), zero treated as multiplication
  (30,024), and omitted new-path threshold (12,750).
- Final clean build, quick and full aggregate, focused softmax/overflow/range/
  near-zero/exponential gates, and mode/attention sweeps pass.
- Separate complete-library ASan/UBSan/LSan build: new gate, complete matrix,
  exponential public/private, near-zero/range, legacy softmax and overflow pass.
  This is exercised-path sanitizer evidence, not a full sanitizer-suite claim.
- Fresh staged-source archive clean build, quick and full aggregates pass.
  Existing unrelated config/logging unused-symbol warnings remain.

```sh
make clean && make
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
make test-softmax-preprocess-order
./test-softmax-preprocess-order --matrix
./test-softmax-preprocess-order --legacy
```

## Backlog audit

There were no READY rows. Re-audited context cycle/bank-state restoration,
SRAM arbitration mode consumption, normalization scratch pre-admission, and
im2col padding/range sentinels; their existing contract dependencies remain.
This one new finite-order exploration yielded a directly implementable runtime
alternative and was completed in the same run. No READY row remains in the
updated audited backlog. Untracked `TERMS.md` is preserved and excluded.
