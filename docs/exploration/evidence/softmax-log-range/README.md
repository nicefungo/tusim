# Softmax finite-range evidence

Baseline: `335b7bc9948a1e6caf7cc8b2f4fd065fe30a0ff4`. The containing commit
provides the corrected source and complete test source. This directory contains
actual raw output; `red-*` failures are intentional regression evidence, not
successful conformance tests. See `../../softmax-log-dynamic-range.md` for scope
and physical-model limits.

- `red-full.log`, `green-full.log`: the same independent test linked against
  the complete baseline archive and corrected archive, respectively. Exit codes
  1 and 0; 36,760 versus zero failures over 305,801 checks / 4,520 cases.
- `red-matrix.log`, `green-matrix.log`: all three modes × seven finite input gaps.
  Exit codes 1 and 0; four versus zero failed checks over 21 rows / 63 checks.
- `old-sweep.log`, `new-sweep.log`: original softmax sweep stdout, identical.
- `old-attention-sweep.log`, `new-attention-sweep.log`: original cross-engine
  sweep stdout, identical. These are compatibility snapshots, not endorsements
  of the historical combined timing formula. Runtime diagnostics are in the
  local full build logs, not these stdout tables.
- `sanitizer-tests.log`, `sanitizer-matrix.log`: all production archive objects
  and test code compiled with ASan/UBSan, leak checking enabled, halt-on-error;
  all commands exited 0. Instrumentation applies to exercised paths only.
- `environment.txt`: baseline pin, actual host/compiler, and pre-documentation
  diff inventory. It is an observation snapshot, not a final file list.
- `SHA256SUMS`: byte hashes of these raw outputs plus the corrected engine,
  header and independent test. Run `sha256sum -c` from the repository root.
  It intentionally does not hash itself or assert its own future commit ID.

## Reproduce corrected behavior

```sh
make test-softmax-log-range
./test-softmax-log-range --matrix
sha256sum -c docs/exploration/evidence/softmax-log-range/SHA256SUMS
```

## Reproduce baseline rejection without modifying the worktree

From the corrected repository root:

```sh
ROOT="$PWD"
mkdir -p /home/zxy/.hermes/cache/scratch
OLD=$(mktemp -d /home/zxy/.hermes/cache/scratch/log-range-baseline.XXXXXX)
git archive 335b7bc9948a1e6caf7cc8b2f4fd065fe30a0ff4 | tar -x -C "$OLD"
make -C "$OLD" libtucmodel.a
cc -O2 -Wall -Wextra -std=c11 -I"$ROOT" -I"$OLD/tu_cmodel" \
  "$ROOT/tests/test_softmax_log_range.c" "$OLD/libtucmodel.a" -lm \
  -o "$OLD/log-range-old"
"$OLD/log-range-old"
# Expected exit 1: 36,760 failed checks, not a build failure.
"$OLD/log-range-old" --matrix
# Expected exit 1: four failed numerical checks.
```

## Sanitizer reproduction

In a separate clean extraction of the corrected commit (do not mix sanitized
and ordinary objects), run:

```sh
make CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDFLAGS='-lm -fsanitize=address,undefined' libtucmodel.a
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  make CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDFLAGS='-lm -fsanitize=address,undefined' \
  test-softmax-log-range test-softmax test-softmax-overflow
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ./test-softmax-log-range --matrix
```

Raw build/aggregate logs are additionally retained outside Git at
`/home/zxy/Workplace/reports/tusim-20261114-softmax/`; repository evidence above
is sufficient to replay the numerical result without those machine-local logs.
