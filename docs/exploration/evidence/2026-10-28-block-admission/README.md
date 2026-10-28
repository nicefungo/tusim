# Retained evidence: block-interleaved admission

Execution-input baseline: `6359360` (full OID in `run.json`). The correction,
new test, public contract, build integration and report are preserved by the
commit containing this directory; no self-referential future commit is embedded.

## Commands and logs

All commands ran from repository root unless `make -C` names the fresh archive.
Shell sequences used `set -e`; failure of any required command stopped its gate.
Existing shared-linked tests used `LD_LIBRARY_PATH=$PWD` (with any prior value
appended). Scratch paths were under the Hermes `TMPDIR`, never system `/tmp`.

- `baseline.log.gz`: baseline `make clean && make && make test-quick`.
- `final-live.log.gz`: final `make clean && make && make test-quick && make test
  && make test-config test-conv test-scatter-gather`.
- `final-probes.log.gz`: final `./test-agen-block-bounds --probes`.
- `old-source-final-harness.log.gz`: the final harness linked with a separately
  compiled baseline `address_generator.c` object and the rebuilt static archive.
  **Expected exit 1**, 30,487 failures; invalid geometries are not dereferenced.
- `sanitizer-build.log.gz` / `sanitizer-tests.log.gz`: baseline Git archive with
  exactly the corrected `Makefile`, address-generator source/header and final
  test overlaid. Every library object was rebuilt with the flags below; the
  six listed targets pass with leak detection. This is exercised-path coverage.
- `mutants.log.gz`: `python3 docs/exploration/evidence/2026-10-28-block-admission/verify_mutants.py`.
  Its control exits 0; four safe mutants exit 1 with completed FAIL summaries,
  not crashes. Each source object is linked before `./libtucmodel.a` so the
  address-generator member in the archive cannot silently override the probe.
- `analyzer.log.gz`: GCC `-fanalyzer` compilation of the changed production
  source and final test; empty means no diagnostics, both commands exited 0.
- `staged-snapshot.log.gz`: fresh exact-index source snapshot clean build, quick,
  configuration, convolution, scatter/gather and new focused gate.

Sanitizer flags:

```
CFLAGS=-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS=-lm -fsanitize=address,undefined
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
UBSAN_OPTIONS=halt_on_error=1
```

Sanitizer targets: `test-agen-block-bounds test-agen test-agen-affine-bounds
 test-agen-tiled-sweep test-agen-dma-chain test-agen-range-admission`.

Baseline-source rejection reproduction after a normal build:

```sh
git show 6359360:tu_cmodel/memory/address_generator.c > "$TMPDIR/baseline_address_generator.c"
cc -O2 -Wall -Wextra -std=c11 -I. -Itu_cmodel -Itu_cmodel/memory -c "$TMPDIR/baseline_address_generator.c" -o "$TMPDIR/baseline_address_generator.o"
cc -O2 -Wall -Wextra -std=c11 -I. -Itu_cmodel tests/test_agen_block_bounds.c "$TMPDIR/baseline_address_generator.o" ./libtucmodel.a -lm -o "$TMPDIR/baseline-block-gate"
"$TMPDIR/baseline-block-gate"
```

The last command must exit **1**, with its own completed FAIL summary.
`run.json` records observed status, parsed summaries, commands and preservation
scope. `INPUT_SHA256SUMS` binds authored inputs; `SHA256SUMS` binds the compressed
logs, runner, metadata and input manifest. Verify using `sha256sum -c` from repo
root for the input manifest and from this directory for the evidence manifest.

The staged log predates inclusion of the retained log payloads, which have no
build/runtime role. After staging those payloads and final manifests, the final
complete index is separately rebuilt and tested before commit; that last command
transcript remains in the run's Hermes scratch directory to avoid a recursive
"a log must contain its own final staged tree" requirement.
