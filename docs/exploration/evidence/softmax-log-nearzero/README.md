# Softmax near-zero evidence

Run date: 2026-11-15 (CST). Baseline input commit:
`20c1c73faeb588350286f40188fe74bd41a1d176`.

These are actual retained executions, not reconstructed numerical output.
`SHA256SUMS` covers this directory's logs, metadata, README and the claim-bearing
source/test/build/document files relative to the repository root. It excludes
itself. Large logs are losslessly gzip-compressed; `gzip -dc` recovers their
original bytes. Raw `.log` files are excluded only from whitespace checking to
preserve diagnostic output, while all authored source/docs are checked. The
baseline is repository-reachable; corrected inputs are preserved in
the commit containing this evidence. No claim is made that this commit was the
already-existing HEAD during execution.

## Contents and interpretation

- `baseline.log.gz`: required clean baseline build and quick suite (lossless gzip).
- `archive-baseline-build.log`: independent archive static-library build.
- `red-probe.log`: original two-value failing probe (exit 1).
- `red-gate.log`, `red-matrix.log`: same expanded test against the archived
  baseline library, both exit 1; no baseline source edits.
- `green-gate.log`, `green-matrix.log`: corrected gate and all 30 mode/gap rows,
  both exit 0. Matrix check count includes cutoff-preservation controls beyond
  the normal-range numerical guarantee; it is not all-domain conformance.
- `old-range.log`, `green-range.log`: unchanged-mode and accounting fingerprints
  from the independently-oracled previous range gate. LOG output bits change.
- `old-mode-sweep.log`, `new-mode-sweep.log`, `old-attention-sweep.log`,
  `new-attention-sweep.log`: byte-identical old/new stdout tables. Initialization
  diagnostics are in the complete make logs. Equality is not a physical-timing
  endorsement of historical sweep labels.
- `old-compatibility.log`: baseline range gate and prior sweep build/execution.
- `final-tests.log.gz`: corrected clean build, quick/full aggregates and focused
  softmax tests/sweeps, shell exit 0. Existing unrelated compiler warnings are
  retained, not silently erased.
- `sanitized.log`: independent all-object ASan/UBSan build with leak detection,
  new gate/matrix, previous range gate, legacy softmax and overflow gate, exit 0.
- `mutants.log`: three safely generated numeric mutants rejected by the new
  gate. The tied-max mutant has an expected unused-index warning; successful
  compilation is required before its numerical nonzero exit is accepted.
- `staged-tests.log.gz`: fresh exact-index archive build, quick, focused and mutant
  verification. Excludes unrelated untracked `TERMS.md`.
- `environment.txt`: source baseline, platform/compiler and initial unrelated
  file checksum; no credentials or environment-variable dump.

## Reproduction

From a corrected checkout:

```sh
make clean && make
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
make test-softmax-log-nearzero test-softmax-log-range test-softmax test-softmax-overflow
./test-softmax-log-nearzero --matrix
python3 tests/check_softmax_log_nearzero_mutants.py
make test-softmax-sweep test-softmax-attention-sweep
sha256sum -c docs/exploration/evidence/softmax-log-nearzero/SHA256SUMS
```

The mutation script uses `TMPDIR` for generated source/executables and never
modifies tracked files. The main tool environment exports TMPDIR under the
Hermes scratch directory; set it to an owned scratch directory elsewhere.

For RED, archive the baseline commit into an empty scratch directory, run
`make libtucmodel.a` there, and compile the corrected checkout's
`tests/test_softmax_log_nearzero.c` against that baseline archive and include
paths with `cc -O2 -Wall -Wextra -std=c11 ... -lm`. Run the resulting executable
both normally and with `--matrix`; each must exit 1 with numerical assertions,
not a build error. The retained baseline logs record the actual run.

For sanitizers, build **every object** in a fresh corrected source extraction:

```sh
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
make CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' \
LDFLAGS='-lm -fsanitize=address,undefined' \
test-softmax-log-nearzero test-softmax-log-range test-softmax test-softmax-overflow
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
./test-softmax-log-nearzero --matrix
```

Do not reuse normal objects with sanitizer link flags. These runs cover the
named paths, not every full-suite execution. The independent oracle uses host
long-double functions, with a precision-width guard and finite-only scope;
no portable correct-rounding guarantee is inferred.
