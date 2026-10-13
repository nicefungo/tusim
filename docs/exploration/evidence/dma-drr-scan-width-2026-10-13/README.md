# DRR scan-width execution evidence

Inputs: pre-change baseline `124eda0` plus the source/config/test changes in the containing commit. This evidence is a development-run bundle, not a self-hashed release or textbook evidence seal.

- `scan.txt`: complete stdout/stderr from `make test-dma-drr-scan-width`.
- `focused.txt`: generated config, config, DMA, existing arbitration, new scan sweep and config-docs run.
- `build.txt`, `quick.txt`: clean full build and final quick smoke suite.
- `full.txt`, `full-keep-going.txt`: broader aggregate attempts; nonzero status is expected from the reported pre-existing failures, not treated as a pass.
- `baseline-attention.txt`, `baseline-broader.txt`: fresh `git archive 124eda0` build observations; attention errors vary between invocations, while transposed-address and multicast-overflow gates reproduce.
- `staged-build.txt`, `staged-tests.txt`: clean build and quick/config/arbitration gates from the exact staged snapshot archive.
- `sanitized.txt`: clean ASan/UBSan rebuild and scan-width sweep with leak detection enabled.
- `SHA256SUMS`: immutable transcript hashes. No raw log was edited to remove whitespace or diagnostics.

The full aggregate is not green. The new scan sweep and quick/focused gates are green. See `../../dma-drr-scan-width.md` for interpretation, exact cost formulas and limits.
