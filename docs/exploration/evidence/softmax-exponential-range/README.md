# Exponential-range evidence — 2026-11-16

Baseline input: `607b891a375b6bf48c97514790d82cc1dd726807` (`main`).
The containing implementation commit preserves the final sources and this run;
`inputs.sha256` pins the authored inputs without a self-referential commit hash.
`SHA256SUMS` covers this directory except itself. Logs are unchanged raw output,
compressed with deterministic gzip headers where large. No binaries are retained.

## Files and interpretation

- `baseline.log.gz`: original clean build + quick suite, before edits (pass).
- `red.log`: minimal gap-88 tracer, baseline exit 1 before production edits.
- `old-source-build.log.gz`, `old-source-red.log.gz`: complete baseline archive
  rebuilt; new harness compiled with `-DTU_TEST_LEGACY_SOURCE` (no new field).
  Exit 1 / 480,240 failures proves missing optional behavior, not a violation of
  the historical default API. Invalid-new-policy tests cannot run on that API.
- `final-build.log.gz`, `final-quick.log.gz`, `final-full.log.gz`: final live
  clean build, quick and full aggregate (exit 0).
- `matrix.csv`: full 81-row three-policy × three-mode × nine-gap output plus
  fail-closed summary; not a device-latency benchmark.
- `mutants.log.gz`: six safe numerical mutations rejected with exit 1 and
  assertion failures; two ONLINE helper-bypass structural controls rejected.
- `sanitizer.log.gz`, `sanitizer-matrix.csv`: complete library rebuilt with
  ASan/UBSan and leak detection; focused new/public/internal, prior LOG, legacy
  softmax, overflow and matrix paths pass. This is not a full-suite sanitizer run.
- `old-source-focused.log.gz`, `final-sweeps.log.gz`: focused compatibility runs.
- `old-` / `new-nearzero-matrix.csv`, `old-` / `new-mode-sweep.txt`, and
  `old-` / `new-attention-sweep.txt`: each pair is byte-identical. Historical
  cross-engine cycle interpretations remain outside this acceptance gate.
- `staged-verification.log.gz`: fresh archive of the staged source tree,
  clean build, quick, full and mutation gates. The transcript records the
  staged tree used; subsequently adding this transcript/manifests changes
  evidence content, not the checked executable inputs.

## Reproduce

From the containing commit, using the ordinary gradual floating-point
host environment (no fast-math/FTZ/DAZ):

```sh
make clean && make
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
./test-softmax-exp-policy --matrix
python3 tests/check_softmax_exp_policy_mutants.py
make test-softmax-log-nearzero test-softmax-log-range test-softmax test-softmax-overflow
make test-softmax-sweep test-softmax-attention-sweep
```

For independent sanitizer execution, extract this commit to a new directory
under `$TMPDIR`, then run there (do not mix sanitized and ordinary objects):

```sh
make clean
make CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-lm -fsanitize=address,undefined'
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 make CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-lm -fsanitize=address,undefined' test-softmax-exp-policy test-softmax-exp-policy-internal test-softmax-log-nearzero test-softmax-log-range test-softmax test-softmax-overflow
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ./test-softmax-exp-policy --matrix
```

For baseline capability comparison, extract the pinned baseline, copy only the
current `tests/test_softmax_exp_policy.c` into that extraction, build its full
library, and compile/run:

```sh
cc -O2 -Wall -Wextra -std=c11 -I. -Itu_cmodel -DTU_TEST_LEGACY_SOURCE tests/test_softmax_exp_policy.c ./libtucmodel.a -lm -o test-exp-red
./test-exp-red
```

Expected exit is 1 with a numerical-failure summary, not a compile or sanitizer
error. `TU_TEST_EXP_INTERNAL` separately includes the actual production source
to exercise the otherwise externally unobservable tiny ONLINE helper outputs;
link order ensures the archive does not replace that definition.

Host: Linux aarch64, GCC 11.4.0. Existing full-build unused-symbol warnings are
in logging/config; one intentional mutant has an unused-parameter warning.
No calibrated throughput, physical area, power or energy data is present.
Untracked `docs/exploration/TERMS.md` is excluded and was not edited.
