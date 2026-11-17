# Softmax preprocessing-order evidence — 2026-11-17

Baseline input: `04d3c02310ceaf0cf549ed78dd1b2d1470e9e78e` (`main`).
The containing implementation commit preserves the final inputs and this run.
`inputs.sha256` pins the authored executable inputs; `SHA256SUMS` covers this
directory except itself. Large raw transcripts are deterministically gzipped;
no binaries or unrelated handoff files are retained.

## Evidence inventory

- `baseline.log.gz`: initial clean build and quick aggregate, exit 0.
- `order-tracer.c`, `red.log`: minimal pre-implementation capability tracer,
  exit 1; reports 0.880797088146 versus scale-first reference 0.952574126822.
- `old-build.log.gz`, `old-compile.log.gz`, `old-red.log.gz`: complete baseline
  archive rebuilt, new harness compiled with `TU_TEST_OLD_ORDER` (new field
  omitted), exit 1 / 24,330 assertions fail. Invalid-new-selector tests cannot
  be compiled against that old API. This is optional-capability evidence,
  not a claim that baseline default semantics were defective.
- `focused.log.gz`: 9,145 cases / 513,010 assertions, exit 0.
- `matrix.csv`: all 30 small matrix rows plus fail-closed summary.
- `old-legacy.txt`, `new-legacy.txt`: byte-identical summaries and fingerprints
  across 4,536 legacy cases / 256,284 assertions.
- `old-`/`new-exp-matrix.csv`, `old-`/`new-mode-sweep.txt`, and
  `old-`/`new-attention-sweep.txt`: each stdout pair is byte-identical. This does
  not validate historical physical or cross-engine timing interpretations.
- Five named mutation logs and compiler logs: ignored selection, changed
  default, double scaling, zero multiplication and threshold omission; each
  exits 1 with numerical assertion failures. `verify_orders.py` preserves the
  exact safe transformations and execution commands; mutations are outside
  the production tree. Unused-selector warnings in forced-branch mutants are
  expected, not present in the production implementation.
- `final-build.log.gz`, `final-quick.log.gz`, `final-full.log.gz`, and
  `final-sweeps.log.gz`: live clean build and requested regression runs, exit 0.
- `sanitizer-build.log.gz`, `sanitizer.log.gz`, `sanitizer-matrix.log.gz`:
  complete-library ASan/UBSan rebuild, leak detection enabled; focused new,
  exponential public/private, near-zero/range, legacy/overflow and matrix pass.
  This is exercised-path coverage, not full-suite sanitizer certification.
- `staged-verification.log.gz`: exact staged-source archive fresh clean build,
  quick and full suites, exit 0. It records the input tree; adding this transcript
  and its checksum afterwards changes evidence files only, not executed inputs.

The primary-source source URL/version and SHA-256 are in the parent exploration
note. No full third-party source is redistributed here. Host compiler version
and architecture are recorded in the staged verification transcript.

## Reproduce current gates

Run from the checkout with ordinary gradual FP arithmetic, no fast-math/FTZ/DAZ:

```sh
make clean && make
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
./test-softmax-preprocess-order --matrix
./test-softmax-preprocess-order --legacy
```

For old-source, sanitizer and safe-mutation runs, use a new scratch directory;
set `TUSIM_ROOT` to the checkout and `TUSIM_EVIDENCE` outside it. The baseline
commit must be available locally. The helper uses fixed subdirectory names;
use a fresh evidence directory rather than mixing builds from another source.

```sh
export TUSIM_ROOT="$PWD"
export TUSIM_EVIDENCE="$(mktemp -d "${TMPDIR:?}/tusim-order-XXXXXX")"
./test-softmax-preprocess-order --legacy > "$TUSIM_EVIDENCE/new-legacy.txt"
python3 docs/exploration/evidence/softmax-preprocess-order/verify_orders.py baseline
python3 docs/exploration/evidence/softmax-preprocess-order/verify_orders.py sanitizer
python3 docs/exploration/evidence/softmax-preprocess-order/verify_orders.py mutants
```

The runner requires only a C compiler, GNU make, Python, Git and tar. Expected
old/mutant exit 1 is distinguished from compilation or sanitizer failure; each
must contain actual assertion failures. The live static library must be freshly
built before mutation linking. Artifact checksums can be checked with
`sha256sum -c SHA256SUMS` from this directory; check `inputs.sha256` from the
repository root.

## Limits

Masks are host-pointer traffic outside the SRAM meter. Finite preprocessing,
FP32 scalar SRAM and nonaliasing fixtures only are certified here. Neither a
numerical match nor eight local SRAM penalty cycles measures physical
throughput, arithmetic latency, area, power or energy. Default architecture,
existing exponential policies and independent attention code are preserved.
Untracked `docs/exploration/TERMS.md` is excluded and unchanged.
