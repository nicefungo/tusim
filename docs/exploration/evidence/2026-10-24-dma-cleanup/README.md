# DMA harness lifetime evidence — 2026-10-24

- `verification.json`: baseline revision, tested index tree, complete source hashes, real command exit statuses, sanitizer controls, matrix comparison and full-suite limitations.
- `logs.tar.gz`: byte-preserved raw output and execution metadata. `live/` contains baseline/final/index build and test logs plus the exact local orchestrator; `campaign-complete/` contains the accepted sanitizer/fault campaign; `campaign-interrupted/` retains the earlier tool-timeout attempt without treating partial coverage as acceptance.
- `MANIFEST.sha256`: SHA-256 of the two artifacts above and this README. `verification.json.archived_raw_sha256` separately covers every uncompressed archive member.

The execution input HEAD is `f8fdd9674a077ef877f22b921083c06574a865fb`, with the changed harness and new test-only files identified by `input_sha256`. The tested index tree includes the authored source/tests/docs, before adding this evidence bundle; adding evidence does not alter execution inputs. Do not interpret the baseline HEAD as an assertion that the tested harness was unchanged.

## Accepted observations

Baseline all-object ASan/UBSan/LSan reproduces 3,136 bytes leaked in 14 descriptor allocations. Final ordinary/sanitized DMA passes 15/15 without leaks, UAF, double-free or UBSan reports. The complete external-source campaign exits 0 after 100 recorded commands: 49 descriptor OOM cases, 45 real consuming rejections (33 newly detected failures / 12 already-negative), and pending-assertion/active-hold controls, plus its build and normal controls. Injected test exits are intentionally nonzero and distinct from sanitizer errors.

Both DRR width sweeps pass (874 / 8,425 timing rows; nine / 162 queue gates), and all 9,316 numeric quick-suite rows match baseline exactly. The normal static/shared libraries and attention executable have identical baseline/final hashes. Clean baseline/final/index build and quick gates, config, DMA arbitration/channel/binding/bus-width, scatter/gather, convolution and GCC analyzer checks pass.

## Non-acceptance and limits

The first foreground campaign exceeded the tool's 420-second wait and has no accepted final summary. A tracked rerun completed; its final summary and source/log hashes were verified. Partial logs are not success.

`make -k test` exits 2 in both baseline and final builds, only at the pre-existing variable attention target. Both report 7/9 but fail different assertions. Identical binaries do not diagnose that defect or prove a stable failure set. No full-suite green or device-performance improvement is claimed.

The OOM campaign targets descriptor objects, not SRAM or multicast metadata allocation. Fixture cleanup is for the known acyclic test graphs, not malformed-graph safety, production lifetime redesign or universal caller ownership. Production config, ABI, numerical semantics and default timing are unchanged.

## Verify or reproduce

From this directory:

```sh
sha256sum -c MANIFEST.sha256
```

From the repository root, use a new output directory outside the checkout:

```sh
python3 tests/test_dma_cleanup.py --output "$TMPDIR/tusim-dma-cleanup-recheck"
```

Allow several minutes and use a tracked worker if the tool's foreground limit is shorter. The runner writes its selected external directory only and preserves the normal checkout build. No credentials, build binaries, user-owned `TERMS.md`, HANDOFF or session-continuation files are included.
