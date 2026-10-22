# Standalone tile-grid planner: exact geometry without wrapped work

**Date:** 2026-10-22. **Mode:** pre-spec exploration-to-implementation.
**Question:** Does the public `tu_agen_compute_tiling()` utility correctly
represent small, ragged, oversized and extreme tile grids independently of
iterator/address-capacity admission?

## Evidence and implementation

Baseline `58f0f143ca1c7681402d3bc09909da3de63f8dd9` contains the public declaration,
compiled library object and one ordinary 1x64x64/1x16x16 test. A C-source call
inventory finds only the utility definition and that test, not an established
compiler, queue or device-execution consumer. The previously corrected tiled
iterator computes its own grid and does not call this utility.

Eight safe integer-only probes find five failures in the baseline; the expanded
independent oracle finds **676,546 failed checks**. Examples:

| Total dimensions | Tile dimensions | Baseline result | Correct planner result |
|---|---|---|---|
| 1 / 1 / UINT32_MAX | 1 / 1 / 2 | third axis 0, total 0 | third axis and total 2,147,483,648 |
| 65537 / 65537 / 1 | 1 / 1 / 1 | total 131,073 | unrepresentable total: 0; exact axes retained |
| UINT32_MAX in all axes | 1 / 1 / 1 | total UINT32_MAX | unrepresentable total: 0; exact axes retained |
| UINT32_MAX in all axes | UINT32_MAX in all axes | grid 0 / 0 / 0 | grid 1 / 1 / 1, total 1 |

The first defect is overflowing ceiling division; the second is an unchecked
three-axis product, whose mathematical value can need 96 bits. The correction
uses quotient/remainder ceiling arithmetic and division-guarded multiplication.
All axes are computed before empty-grid detection and product validation. This
keeps nonempty-axis metadata correct even if another axis is empty.

### Compatibility and runtime contract

- Existing `tu_agen_tiling_t`, void API, struct layout and direct descriptor
  configuration remain unchanged; no JSON/YAML knob is added for invalid math.
- Zero tile dimensions still normalize to one; zero total dimensions stay empty.
- `tiles_per_dim[d]` is always exact for normalized positive tile dimensions.
- `num_tiles` is exact through UINT32_MAX inclusive. Larger products return zero,
  never a wrapped plausible work count. Empty grids also return zero; any zero
  axis distinguishes empty from overflow with all axes nonzero. The void API
  does not provide an explicit status code. Callers needing diagnostics must
  inspect axes; a status-returning API is not silently introduced here.
- `total_dims` and `elem_size` are preserved; NULL remains a no-op.
- This is metadata planning, **not** tensor-byte/SRAM/host-capacity admission.
  A huge tensor covered by one huge tile is a representable grid, even when it
  cannot be addressed or stored by the uint32 iterator. Iterator admission is
  deliberately stricter and remains unchanged. Width zero is untouched metadata
  here, not a promise that a zero-width transfer is executable.

## Complete bounded workload matrix

Total tensor 3x5x7, 4-byte element metadata; compact clipped tile origins, not
padded storage. Every row covers exactly 105 logical elements. Counts are
planner/iterator geometry, **not** DMA commands, memory transactions or cycles.

| Requested tile D/R/C | Normalized tile | Grid D/R/C | Tiles |
|---|---|---|---:|
| 1 / 1 / 1 | 1 / 1 / 1 | 3 / 5 / 7 | 105 |
| 1 / 3 / 7 | 1 / 3 / 7 | 3 / 2 / 1 | 6 |
| 2 / 3 / 4 | 2 / 3 / 4 | 2 / 2 / 2 | 8 |
| 3 / 5 / 7 | 3 / 5 / 7 | 1 / 1 / 1 | 1 |
| 8 / 8 / 8 | 8 / 8 / 8 | 1 / 1 / 1 | 1 |
| 0 / 0 / 0 | 1 / 1 / 1 | 3 / 5 / 7 | 105 |

The supported alternatives remain runtime descriptor choices, not a selected
winning tile. Small tiles can match small working sets and hardware lanes;
large tiles can amortize repeated tile control but require larger working sets.
The count alone does not establish either effect in the live device model.

## Gain versus sacrifice and fidelity

| Dimension | Small / narrow tiles | Large / oversized tiles | What this change establishes |
|---|---|---|---|
| Throughput | Potentially better shape/working-set fit, more tile control | Potentially more amortization, possibly more inactive lanes | No device rate measured; exact work counts only |
| Latency | Can shorten scheduling granularity but multiply setup events | Can reduce setup count but delay a large work unit | Unquantified; no setup/queue schedule added |
| Area/resources | May permit smaller scratch/compute slices | May require more live storage/lanes if physically provisioned | Unquantified; a planner value does not instantiate hardware |
| Power/energy | Smaller active resources may help, more control events may hurt | More resources may switch/leak, amortized control may help | Net direction unquantified; no energy wiring/calibration |
| SRAM/DRAM traffic | Reuse or fragmentation depends on mapping and consumers | Larger working sets may help reuse or force capacity spills | Logical coverage unchanged; physical bytes/reuse unmeasured |
| Numerical accuracy | No arithmetic-format or reduction policy change | Same | Integer geometry corrected; numerical semantics unchanged |
| Control complexity | More tile boundaries; ragged clipping must be correct | Fewer boundaries; larger counters/capacity checks may be needed | Added constant host validation; cost unmeasured |
| Verification | Tail, empty, normalization and overflow cases matter | Extreme tiles/counts and independent capacity checks matter | Exact wide oracle, replay/idempotence and bounded enumeration |
| Compiler/runtime | Tile selection must fit working sets and execution contract | Must not infer device admissibility from planner success | Existing API retained; overflow becomes explicit non-work metadata |

Unchecked wrap is not retained as a faster architecture. A wider-count planner
could support larger grids, but requires an explicit wider public contract and
consumers; it is not added merely to enlarge the feature surface. Physical
padded layout remains separate from compact clipped traversal and needs stride,
capacity and mask contracts.

## Verification actually executed

- Initial `make clean && make` succeeds. The first quick invocation fails to
  locate `libtucmodel.so`; setting `LD_LIBRARY_PATH` to the repository directory
  resolves this environment prerequisite without a code change. Baseline live
  and clean-archive quick suites then pass.
- Two test-first tracer bullets fail before their respective ceiling/product
  corrections and pass afterward. The final eight-probe baseline control has
  five failures; the full baseline gate exits 1 with 676,546 failed checks.
- Final live clean build, `make test-quick`, focused gates and independently
  extracted final-overlay archive clean build/quick/focused gates pass.
- **1,030,014 planner cases:** 568,587 representable nonempty, 271,002 empty,
  190,425 overflowing products; **zero failures**. Includes the million-row
  edge cross-product, 30,000 seeded cases, exact UINT32_MAX count, >64-bit
  products, zero normalization, stale-output overwrite and idempotence.
- **64,007 bounded cases** independently enumerate clipped tile origins and
  coverage, and compare exact grid axes against the existing tiled iterator.
- Config **31/31**, convolution **12/12**, scatter/gather **15/15** pass.
- Fresh all-object ASan/UBSan static build with leak detection passes the planner,
  address, tiled, affine, im2col point/iterator, chain (including OOM) and range
  gates. This certifies exercised paths, not all public address helpers.
- Broad `make test` remains non-green: both baseline and final archive runs stop
  at attention **7/9**, with deterministic-small and causal infinite-error
  failures. The matching observations do not prove a stable failure set or its
  root cause; prior audits record variability. No aggregate-pass claim is made.

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make clean && make
make test-quick
make test-agen-tiling-plan test-config test-conv test-scatter-gather
make test
```

Retained raw logs, exact commands/return codes, input hashes and the executed
runner are in `evidence/2026-10-22-tiling-plan/`. Reproduce the archive experiment
from the corrected checkout with:

```sh
python3 docs/exploration/evidence/2026-10-22-tiling-plan/run_audit.py \
  "$PWD" 58f0f143ca1c7681402d3bc09909da3de63f8dd9 \
  "$TMPDIR/tiling-plan-reproduction-new"
```

The output directory must not already exist. The archive runner extracts regular
files/directories, omitting unrelated external symlink fixtures. Baseline is the
recorded commit; final/sanitized inputs are that tree plus five explicitly hashed
source/test/build overlays. The exact staged publication tree is separately
built and tested before commit. Logs are compressed without editing raw bytes.

## Backlog audit and scope boundary

No READY item existed at entry; this is the one new bounded exploration and its
verified implementation. SRAM arbitration still lacks requester/port/replay
semantics; compressed-domain compute lacks a shared token/FIFO contract; physical
WS/OS/RS movement lacks events/overlap evidence; padded im2col-to-DMA lacks tagged
fill/retirement semantics. They remain BLOCKED, not rejected for being slower.
Block-interleaved address initialization still lacks the affine span gate, and
its current next-address expression algebraically collapses to linear traversal
while `num_banks` does not participate. It is not certified by this planner gate;
choosing a physical bank-layout replacement needs an explicit mapping contract.
No additional READY implementation candidate was established in this audit.

The narrow correction stays on `main`. User-owned untracked `TERMS.md` remains
untouched/excluded; no handoff or compiled artifact belongs to this publication.
