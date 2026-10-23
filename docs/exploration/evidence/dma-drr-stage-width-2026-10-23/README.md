# Independent DRR scan-stage width evidence

Run date: 2026-10-23. Baseline: `450fb16` (full OID in `verification.json`). These are real execution logs, not synthesized benchmark rows. Gzip decompression preserves the original log bytes; gzip timestamps are fixed to zero for deterministic packaging.

## Inputs and provenance

- `source-input.patch.gz`: exact staged diff from baseline for tree `251cc2e9f41f435c1cb37e9567cdc1106429c03e`, used by the all-object sanitizer build. The final source/test/config implementation is identical to this tree.
- `final-input.patch.gz`: exact staged diff for tree `071ac0a69c4df1ca27dd47e6dc94a8ba9b37a744`, used by final snapshot build/quick/focused execution. It adds only documentation corrections and the descriptor-cleanup follow-up to the first input tree.
- Both patches passed `git apply --check` against the untouched baseline archive. They retain claim-bearing source and docs without depending on an uncommitted workspace remaining on disk. The original baseline remains reachable through repository history.
- `verification.json`: exact input trees, per-input SHA-256, decompressed log hashes, observed command exit statuses and test totals. Input-tree archives exclude the user's untracked `TERMS.md` and all handoff material. The subsequent evidence-only additions are not part of the execution-input trees.
- `MANIFEST.sha256`: hashes of retained artifacts (excluding itself). Verify from this directory with `sha256sum -c MANIFEST.sha256`.

## Actual verification commands

Each source tree was extracted using `git archive <tree> | tar -x -C <scratch-directory>`, then built from its own root. `LD_LIBRARY_PATH` named that exact root, not another checkout. All scratch directories were under `/home/zxy/.hermes/cache/scratch/tusim-20261023/`.

Normal baseline, final live and final snapshot checks:

```sh
make clean && make
make test-quick
make test-config test-dma-arbitration-sweep test-dma-channel-sweep test-dma-binding-sweep test-dma-bus-width-sweep test-conv test-scatter-gather
```

The baseline and final live checkouts additionally ran `make -k test`: both returned 2, with attention the sole failed target. Baseline attention varied across repeated unchanged executions; see all retained aggregate and `baseline-attention-*.log.gz` logs rather than assuming one fixed failure set.

The preserved `tracer.c` was compiled separately against baseline and final static archives:

```sh
cc -O2 -Wall -Wextra -std=c11 -I. -Itu_cmodel -o <scratch-executable> <tracer.c> ./libtucmodel.a -lm
```

Its baseline exit is 1 (91 rather than expected 70 cycles); final exit is 0 (70 cycles, exact bytes). Full final timing output is in `stage-final.log.gz`: 8,425 rows and 162 queued/reconfiguration gates. The three extra dynamic-inheritance and unchanged-path controls must also pass for the executable to return zero.

Sanitizer builds rebuilt **all library objects** from clean archives:

```sh
make clean
make 'CFLAGS=-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' 'LDFLAGS=-lm -fsanitize=address,undefined'
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
make 'CFLAGS=-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' 'LDFLAGS=-lm -fsanitize=address,undefined' test-dma-drr-stage-width test-dma-drr-scan-width
```

The final command returned 0; `sanitized-focused.log.gz` contains both passing sweep summaries. An earlier extended command also included `test-dma` and returned 2: preserve `sanitized-gate.log.gz`, which shows the two passing sweeps followed by historical harness leaks. A separate unchanged-baseline all-object sanitizer build and `test-dma` reproduces **3,136 bytes / 14 allocations with identical allocation-site tuples** (`baseline-sanitized-dma.log.gz`). No sanitizer suppression was used; the harness cleanup is READY, not fixed.

`review-initial.json` preserves the independent review's actual initial blocker: the evidence had not yet been copied when that review ran. `review-final.json` records the subsequent passing independent read-only review after the artifacts, hashes, patches and claimed results were inspected. That review does not turn the known broader attention or historical DMA-harness failures into passing tests.
