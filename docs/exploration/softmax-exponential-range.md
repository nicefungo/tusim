# Standalone softmax: selectable exponential-result range

**Date:** 2026-11-16. **Mode:** pre-spec numerical datapath alternatives.
**Baseline:** `607b891a375b6bf48c97514790d82cc1dd726807`.
**Question:** can a designer compare the historical cutoff, exponential-result
FTZ and gradual underflow without replacing defaults or claiming a timing win?

## Implemented contract and selection

`tu_softmax_desc_t.exp_policy` is a per-execution C runtime setting:

| Value | Mode | Why retain it / sacrifice |
|---:|---|---|
| 0 | `TU_SOFTMAX_EXP_LEGACY` | Reproducible existing model; two-pass/output exponentials below -87 are zero. ONLINE accumulation historically used uncut `expf`, and still does. This cutoff also discards some normal results: it is **not** result FTZ. |
| 1 | `TU_SOFTMAX_EXP_FTZ` | Flush rounded **exponential** subnormal results, potentially simplifying a physical exponential datapath; retains normal results beyond the old cutoff but loses representable tiny contributions. |
| 2 | `TU_SOFTMAX_EXP_GRADUAL` | Retain host `expf` subnormal results for probability tails and near-zero LOG outputs; a physical implementation may need extra small-result handling. It still rounds every exponential to FP32 before accumulation. |

All three policies apply to STANDARD, LOG and ONLINE. A single helper is the
only `expf` producer, including ONLINE rescaling and sum contributions. New
policies use the same exponential-result rule in denominator and output paths;
this does **not** make order-dependent ONLINE FP32 accumulation mathematically
identical to the double two-pass sum. No global FP environment is changed.
FTZ is **not** DAZ, full-pipeline FTZ, a shifted-input cutoff, or final-output
flushing: division of a retained normal exponential can produce a subnormal
probability, which is retained. LOG keeps finite shifted logits even when their
exponentials vanish.

```c
/* Initialize every field, then supply valid SRAM, shape and offsets. */
tu_softmax_desc_t d = {0};
d.mode = TU_SOFTMAX_LOG;
d.data_sram = sram;
d.elem_count = count;
d.in_place = true;
d.exp_policy = TU_SOFTMAX_EXP_GRADUAL;
uint64_t local_penalty = tu_softmax_execute(&d);
```

Invalid policies reject before allocation, SRAM accesses or statistics writes.
Zero initialization and all existing host/flat/2D/masked wrappers retain LEGACY.
The descriptor field is appended: **binary clients must rebuild**, and source
clients must initialize the whole descriptor (designated/zero initialization).
This is the engine's actual runtime API, not a YAML/JSON setting. No generator,
canonical global configuration or compiler lowering claims are made. Attention's
existing standalone call remains zero/default; its independent streaming kernel
is unchanged and does not inherit this setting.

## Measured trade-off matrix

The [evidence directory](evidence/softmax-exponential-range/README.md) preserves
all **81 rows**: three policies × three output modes × nine gaps. Input is
`[0,-gap]`, no mask/scale, in-place FP32, 128-byte default SRAM, prefilled with no
time advance. Selected dominant LOG outputs:

| Gap | LEGACY | FTZ | GRADUAL |
|---:|---:|---:|---:|
| 87.25 | 0 | -1.28175929753e-38 | -1.28175929753e-38 |
| 88 | 0 | 0 | -6.0546014852e-39 |
| 100 | 0 | 0 | -3.78350585368e-44 |
| 103.25 | 0 | 0 | -1.40129846432e-45 |
| 104 | 0 | 0 | 0 |

At gap 88 the mathematical dominant LOG output is -6.0546018954e-39.
At gap 104, the exponential is below half the minimum binary32 subnormal:
GRADUAL is not arbitrary precision. Many individually underflowed contributions
can therefore remain lost before the residual sum; widening exponential
precision would be a separate arithmetic/error contract, not a silently better
version of this policy.

Every matrix row retains two reads, two writes and **8 local SRAM penalty
cycles**. These are neither arithmetic latency nor device throughput. The broad
matrix also produces identical bank-event/penalty fingerprints for all nine
policy/mode combinations (`cba3f3812425c025`).

| Dimension | Gain / sacrifice and fidelity boundary |
|---|---|
| Accuracy | FTZ restores the normal-exponential band discarded by -87; GRADUAL additionally preserves tested subnormal contributions. Legacy preserves old output bits. No universally correct choice without the consumer's range/error requirement. |
| Throughput / latency | Physical values **unquantified**: no exponential-unit pipeline, denormal-assist or calibrated arithmetic timing model. Local penalties are unchanged; no speedup claim. |
| Area / resources | Same row allocation and SRAM footprint; one descriptor selector plus host branch/classification. Hardware may hard-wire a policy; gradual support could require small-result normalization/rounding, while a selector costs control. Physical area **unquantified**. |
| Power / energy | Flushing could avoid small-result handling in a physical unit; gradual/selectable hardware can add activity. This C implementation still calls host `expf` before classification. Net power/energy **unquantified**, not inferred from policy names. |
| SRAM / DRAM traffic | Same checked SRAM read/write and bank-event traces; no standalone DRAM requests. Physical port traffic/energy **unquantified**. |
| Control / verification | Explicit enum, early validation, centralized exponential producer, legacy ONLINE exception, and three-mode cross-product increase contract/test surface. No stochastic policy or PRNG is involved. |
| Compiler / runtime | Per-descriptor selection can track a kernel's numerical requirement. Existing wrappers/default callers remain unchanged; binary descriptor consumers rebuild. No compiler emission integration was added. |

Hardware plausibility follows the first-party subnormal/flush rationale recorded
in [the motivating exploration](softmax-log-nearzero.md#backlog-audit-and-next-bounded-design-question).
The implementation is not claimed equivalent to CUDA FTZ or an IEEE-correctly-
rounded transcendental unit. Host gradual arithmetic is required; global FTZ/DAZ,
fast-math, arbitrary rounding environments, nonfinite/all-masked semantics,
preprocessing overflow, non-4-byte scalar SRAM and arbitrary aliasing are outside
this gate. Do not infer support merely because an enum is accepted.

## Verification executed

- Clean baseline build and `test-quick`: pass. Initial gap-88 tracer fails on
  baseline (0 versus -6.0546014852e-39) before production edits.
- Public gate: **43,231 cases / 7,931,138 assertions**, zero failures. Includes
  43,200 combinations: five row widths, twenty boundary gaps, four scales,
  no/additive/multiplicative finite mask, two placements, unique/tied maxima,
  three output modes, three policies; three rows place maxima first/middle/last.
  Also 24 invalid-policy cases and seven wrapper paths.
- Independent `expl`/`log1pl` reference includes explicit FP32 exponential
  rounding. Boundary neighbours cover -87, normal/subnormal transition, minimum
  subnormal and half-minimum underflow. Normal two-pass values use relative
  5e-7; ONLINE uses `(2*cols+4)*FLT_EPSILON` for bounded FP32 sequential arithmetic,
  not a universal tolerance. Subnormal checks use <=2 ULP with nonzero/sign
  checks, measured maximum **0 ULP**; expected zero is checked exactly.
- Private-helper supplement: **7,931,534 assertions**, zero failures. Tiny
  denominator changes can round away in public FP32 sums >=1; the supplemental
  gate compiles the actual implementation into its translation unit and tests
  both exponential call contexts. It is not a copied oracle. A source gate
  checks both ONLINE call sites and rejects two bypass controls.
- Old source rebuilt from the complete baseline archive: **480,240 assertion
  failures**, exit 1. `TU_TEST_LEGACY_SOURCE` deliberately omits the new field
  when compiling the same gate against the old API; this measures the missing
  alternatives, not an old default-contract violation. Invalid-new-enum cases
  are necessarily omitted there (43,207 cases / 7,930,970 checks).
- Six safe mutants rejected numerically: ignored selection, FTZ-as--87,
  omitted FTZ, flushed GRADUAL, changed legacy ONLINE sum, and final-probability
  flushing. The last two detect otherwise easy contract confusions.
- Existing 4,880-case near-zero and 4,520-case range gates pass; all three
  output-inclusive legacy fingerprints unchanged. Prior near-zero, mode sweep
  and attention-sweep stdout tables are byte-identical. Historical cross-engine
  timing interpretations are not recertified.
- Final `make clean && make`, quick and full aggregates pass; focused legacy
  softmax and overflow gates pass. Both new targets are static and integrated
  into quick/full/clean/ignore. All-object ASan/UBSan/LSan focused run and matrix
  pass. Existing logging/config unused-symbol warnings remain; the deliberately
  changed legacy ONLINE mutant has an expected unused-parameter warning.

```sh
make clean && make
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
make test-softmax-exp-policy test-softmax-exp-policy-internal
./test-softmax-exp-policy --matrix
python3 tests/check_softmax_exp_policy_mutants.py
make test-softmax-log-nearzero test-softmax-log-range test-softmax test-softmax-overflow
make test-softmax-sweep test-softmax-attention-sweep
```

## Backlog audit

The sole READY item at baseline is implemented here. Rechecked context SRAM
cycle-versus-bank-state restoration, scalar SRAM arbitration consumption,
normalization wrapper scratch placement, and im2col padding sentinels: their
recorded contract dependencies remain, not ready alternatives rejected for speed.
No new speculative engine or global configuration knob was added. See
[IMPLEMENTATION_BACKLOG.md](IMPLEMENTATION_BACKLOG.md).

**2026-11-17 follow-up:** [scale/mask stage ordering](softmax-preprocess-order.md)
adds two independent runtime preprocessing organizations while retaining all
three exponential policies. The original 81-row exponential matrix remains
byte-identical under the default ordering; this does not extend nonfinite,
all-masked or independent attention-kernel guarantees.
