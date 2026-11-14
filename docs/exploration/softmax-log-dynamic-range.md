# Log-softmax: retain finite shifted logits

**Date:** 2026-11-14. **Mode:** pre-spec exploration / numerical-model correction.
**Baseline:** `335b7bc9948a1e6caf7cc8b2f4fd065fe30a0ff4`.
**Question:** does a finite log-domain output survive the existing exponential
cutoff, and what changes in storage, access accounting and other modes if the
engine retains logits instead of recovering them from exponentials?

## Finding and implementation

The old `softmax_row_two_pass()` overwrote each shifted input with its FP32
exponential. Values below -87 were replaced by zero. LOG then recovered the
shift with `logf(exp_value)`, or substituted **-88.7228** for a zero exponential.
For `[0, -1000]`, both the SRAM engine and `tu_log_softmax_host()` therefore
returned a tail of **-88.7228012085**, not -1000. Existing host-vs-engine tests
shared this implementation and could not expose the error.

For finite preprocessed inputs and representable FP32 differences, the corrected
LOG path retains `x_i - max(x)` in the existing row allocation, uses a temporary
exponential for the unchanged sum, and finally subtracts `logf((float)sum)`.
There is no extra row allocation or SRAM access. The exponential cutoff and
normalizer precision are retained; this is a finite-tail correction, **not**
full floating-point conformance or a relative-accuracy guarantee near zero.

Selection remains the existing runtime descriptor `mode=TU_SOFTMAX_LOG` and
host/flat/2D convenience APIs. STANDARD=0 remains the default; LOG=1 and ONLINE=2
remain available. No JSON/YAML field, ABI, descriptor layout, error convention,
mask/scale ordering, or memory-layout contract changes. LOG outputs intentionally
change where exponential recovery lost information; ordinary LOG values can also
differ in final bits because the exp/log round trip is removed.

## Complete experiment matrix

The checked-in [before](evidence/softmax-log-range/red-matrix.log) and
[after](evidence/softmax-log-range/green-matrix.log) transcripts contain **21 rows**:
STANDARD/LOG/ONLINE × gaps {1, 80, 87, 88, 100, 200, 1000}, input `[0,-gap]`, no
mask or scaling, in-place FP32, 128-byte default SRAM, no cycle advance after
prefill. The LOG rows are:

| Gap | Old tail | Corrected tail | Independent double oracle |
|---:|---:|---:|---:|
| 1 | -1.31326162815 | -1.31326162815 | -1.31326168752 |
| 80 | -80 | -80 | -80 |
| 87 | -87 | -87 | -87 |
| 88 | -88.7228012085 | -88 | -88 |
| 100 | -88.7228012085 | -100 | -100 |
| 200 | -88.7228012085 | -200 | -200 |
| 1000 | -88.7228012085 | -1000 | -1000 |

Every matrix row performs **two FP32 reads and two FP32 writes**, returning
**8 local SRAM penalty cycles** in this prefilled fixture, before and after.
These penalties are not compute latency, concurrent makespan or throughput.
STANDARD and ONLINE still output zero tails once the retained cutoff is crossed;
that compatibility behavior does not imply a log-domain tail should saturate.
The prior softmax and attention-followed-by-softmax sweep stdout tables are
byte-identical (both versions retained). Their historical cross-engine timing
interpretations are not revalidated by this equality.

## Realistic choices and gain versus sacrifice

| Choice / dimension | Motivation, observable benefit, and cost or limitation |
|---|---|
| STANDARD probability output | Appropriate when the next consumer needs weights/probabilities; keeps the existing two-pass host arithmetic and compatibility cutoff. Cannot preserve tiny probabilities below that cutoff and must not be used as an exact intermediate for log outputs. |
| ONLINE probability output | Represents a streaming max/sum organization with rescaling state; retained as a runtime alternative. It has different FP32 accumulation order and control/arithmetic work. This SRAM implementation still buffers a row; it is not a physical streaming-pipeline measurement. |
| LOG with retained shifted inputs | Appropriate for log-domain consumers; preserves finite negative tails and removes per-element log recovery. Reuses the current row storage. Requires mode-dependent interpretation of row contents and corresponding numerical verification; it does not create a calibrated hardware log pipeline. |
| Explicit clipped log approximation | A bounded-range format could be a legitimate separate design with a specified floor and error budget. The old undocumented -88.7228 substitution is **excluded as incorrect for the existing LOG API**, not because it is slower. No validated clipped-format contract exists here. |
| Throughput / latency | **Unquantified physically.** Fewer host `logf` calls are visible in source, but no arithmetic-unit scheduling, issue rate or calibration supports a cycle-speedup claim. Local SRAM penalties are unchanged. |
| Area / resources | Existing host row allocation is unchanged. A hardware implementation might reuse row storage and avoid a per-element log stage, or share one log unit anyway. Ports, muxes, pipeline reuse and actual area remain **unquantified**. |
| Power / energy | Fewer log evaluations could reduce arithmetic activity in a matching implementation; changed data values also alter switching. Net power/energy is **unquantified**, not inferred from host operation counts. |
| SRAM / DRAM traffic | Executed SRAM word-event counts are unchanged; input/output representation remains FP32. This standalone gate performs no DRAM requests, so physical SRAM/DRAM traffic and memory-energy changes are **not measured**. |
| Numerical accuracy | Broad gate's LOG maximum absolute error falls from 125911.277199 to 0.00122453406948, over values reaching large finite magnitudes; the magnitude-aware bound below is the acceptance rule. STANDARD/ONLINE output fingerprints are unchanged. No claim of correct rounding, universal relative accuracy or special-value conformance. |
| Control / verification | Existing mode select now chooses shifted-logit versus exponential row contents. Verification adds an independent oracle, cutoff-boundary and large-gap vectors, masks, negative scales, placements and all public LOG surfaces. Physical control cost is **unquantified**. |
| Compiler / runtime | Existing APIs and storage requirements are unchanged. Consumers needing log probabilities should select LOG, not take a log of clipped STANDARD outputs. Existing exact-output fixtures for defective LOG tails need updating; there is no compiler lowering change in this patch. |

## Verification and reproduction

`tests/test_softmax_log_range.c` does not use the production host helper as its
oracle. It reproduces only existing FP32 mask/scale preprocessing, then evaluates
max, exponentials, sum and log in independent double arithmetic. Checks reject
unexpected NaN/Inf before testing `8e-6 + 3e-7*abs(reference)` output tolerance.
This tolerates FP32 shift/output rounding and ONLINE accumulation in the sampled
finite domain; it intentionally does not certify tiny near-zero LOG results.

- **4,320 descriptor cases:** 8 patterns × 6 widths {1,2,3,8,31,64} × 5 scales
  {0(skip),1,0.5,-1,2} × absent/additive/multiplicative masks × in/disjoint-out
  placement × all three modes; three rows each, including changed input order.
- **200 host/wrapper cases:** 40 host and 160 flat/2D wrapper cases. Existing
  wrapper `in_place=false` still uses the same offset; disjoint output is tested
  through descriptors, not falsely attributed to that wrapper flag.
- **305,801 assertions:** numerical outputs and row normalization; exact maxima;
  normalizer values; untouched SRAM outside output; read/write counts; local
  penalty-counter agreement; NULL statistics; disabled metering; empty host no-op.
- Baseline complete archive linked against the new test: **36,760 failed checks**,
  exit 1. Corrected build: **zero failures**, exit 0. Compact matrix: 4 failed
  checks before, zero after.
- STANDARD and ONLINE output/statistics/stall FNV-64 fingerprints are respectively
  `7835006ea9ecf509` and `cafd1af0bdab8391`, unchanged before/after. All modes'
  separate max/sum/access/penalty fingerprints also match. These are same-host
  regression fingerprints, not portable cryptographic provenance; files have a
  separate SHA-256 manifest.
- Baseline `make clean && make && make test-quick` passed. Final clean build,
  quick and full `make test` aggregates passed, as did legacy softmax **15/15**,
  overflow admission and both historical sweeps. All-object
  **ASan/UBSan/LSan** focused LOG-range, legacy and overflow suites and the full
  compact matrix passed. This does not sanitize every full-suite path.

```sh
make clean && make
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
make test-softmax-log-range test-softmax test-softmax-overflow
./test-softmax-log-range --matrix
make test-softmax-sweep test-softmax-attention-sweep
```

The new target statically links `./libtucmodel.a`, belongs to quick/full/clean,
and is gitignored. [Evidence](evidence/softmax-log-range/README.md) includes raw
before/after matrices, gate summaries, compatibility sweeps and sanitizer output.

## Remaining boundaries

Nonfinite inputs, preprocessing/subtraction overflow, all-masked semantics,
relative accuracy near zero, host mask/statistics capacity, threshold-fill
semantics, arbitrary overlapping row placements, allocation-failure atomicity and
non-4-byte SRAM scalar access remain outside this gate. The current normalizer
still uses FP32 exponentials, a cutoff, a double sum, and conversion to FP32 before
`logf`; retaining logits does not make those choices exact. No speculative mode
or hidden numerical fallback was added to conceal these boundaries.

The daily backlog audit found no pre-existing READY row. Normalization wrapper
scratch/aliasing, context-local versus shared SRAM time, requester-aware SRAM
arbitration, and im2col padding/fill still need their recorded contracts. This run
closes one independently verified numerical-model defect, not those subsystems.
