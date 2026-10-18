# Affine iterator admission: count and full byte-span bounds

## Question and evidence (2026-10-18)

Can the runtime LINEAR, STRIDED_2D, STRIDED_3D and TRANSPOSED iterator modes
represent a tensor view without wrapping its element count, wrapping an address,
or colliding with the exhaustion sentinel?
Baseline: `a7cd792d4a4f53041118c5ceb71450dfb7b1a428`.
There were no READY backlog items at entry. This single new correctness
exploration found four safe admission counterexamples, all accepted by baseline:

| Mode | Input | Baseline | Required |
|---|---|---|---|
| Linear | base=UINT32_MAX−7, count=2, width=4 | accepts; last element includes reserved byte | reject |
| Strided 2D | 2×1, row stride=UINT32_MAX elements, width=4 | accepts wrapped span | reject |
| Strided 3D | 2×1×1, depth stride=UINT32_MAX elements, width=4 | accepts wrapped span | reject |
| Transposed | 65536×65536, width=4 | accepts count wrapped to zero | reject |

RED probes only initialize/query iterators: no invalid address is dereferenced,
no descriptor is created, and no huge sequence is traversed. All four initially
fail admission assertions; the same probes pass after the shared validation.

## Implemented runtime contract

- Preserve all four existing modes, public APIs/struct layouts, traversal order,
  default widths and valid address sequences. No new JSON/YAML knob: invalid
  arithmetic is not a useful hardware alternative.
- Count must fit uint32_t. The complete addressed element extent must end at
  or before UINT32_MAX **exclusive**; UINT32_MAX itself remains exhaustion,
  not a usable tensor byte. This matches existing compact-tile admission.
- Positive/nonnegative strides are in **elements**. Zero strides and overlapping
  views remain legal: they can express broadcast/reused windows, not necessarily
  a permutation. Do not require row stride ≥ column count or disjoint planes.
- Nonempty LINEAR requires positive width. Its old explicit zero-width input
  was degenerate repeated-address behavior and now fails; NULL config remains
  empty with width 1, and empty explicit zero-width inputs remain accepted.
  STRIDED_2D stays fixed at 4 B; STRIDED_3D and TRANSPOSED retain size 0 → 4 B.
- Any zero dimension is empty, irrespective of otherwise huge strides/base.
  Invalid admission returns −1 and leaves an empty, safely exhausted iterator.
- Validate before assigning an executable element count. Shared subtraction of
  per-axis spans from a byte-derived element budget avoids count, product and
  multi-axis sum overflow without requiring wider-than-64-bit production math.
- This proves uint32 representability, **not** actual SRAM or host allocation
  capacity. DMA span validation remains mandatory. Inline single-address helpers,
  IM2COL and BLOCK_INTERLEAVED are outside this correction and not certified.

For nonempty D×R×C with element width S, depth stride Sd and row stride Sr:

```
count = D * R * C <= UINT32_MAX
exclusive_end = base + ((D-1)*Sd + (R-1)*Sr + C) * S <= UINT32_MAX
```

The test oracle uses independent 128-bit arithmetic; valid small views use nested
loops, not the implementation's index decoder. Huge accepted views sample only
first/last/exhaustion, never billions of iterations.

## Realistic alternatives and complete layout matrix

All rows use base 13 and 4 B elements. Strides below are elements. Counts are
logical address runs, **not** physical commands, DRAM traffic or cycle estimates.

| Layout / runtime mode | Shape | Strides depth/row | Elements | Logical contiguous runs |
|---|---|---|---:|---:|
| Dense / STRIDED_2D | 3×4 | — / 4 | 12 | 1 |
| Padded / STRIDED_2D | 3×4 | — / 8 | 12 | 3 |
| Overlapping / STRIDED_2D | 3×4 | — / 2 | 12 | 3 |
| Broadcast rows / STRIDED_2D | 3×4 | — / 0 | 12 | 3 |
| Column-first / TRANSPOSED | 3×4 | row-major pitch 4 | 12 | 12 |
| Padded planes / STRIDED_3D | 2×3×4 | 32 / 8 | 24 | 6 |

Dense storage simplifies continuity and minimizes layout holes. Padding can
serve alignment/bank/layout requirements but increases allocation footprint and
may fragment transfers. Overlap/broadcast can reuse stored input windows but
requires alias-aware consumers; stores may overwrite the same bytes repeatedly.
Transpose can match operand orientation without changing underlying storage but
increases logical fragmentation here. No layout is selected as universally best.
Wrapping admission is excluded as unsafe; a wider-address machine is plausible
but needs a different address/sentinel ABI and consumers, not silent modulo wrap.

| Dimension | Gain / sacrifice / model limit |
|---|---|
| Throughput and latency | Valid device timing is unchanged. Initialization adds constant-time host checks. Physical AGU issue rate, critical path and fault latency are unquantified. |
| Area/resources | A physical fail-closed AGU may need bounds/carry comparators and error state; host validation is not an area estimate. Padding costs layout capacity; broadcast can reuse storage. Quantitative hardware costs are unmodeled. |
| Power/energy | Host checks add activity; physical checking/control and layout switching costs are unquantified, with no calibrated power coupling. No numerical energy claim. |
| SRAM/DRAM traffic | Valid generated addresses and logical payload sequences are preserved. Runs/aliasing are observable; cache reuse, burst traffic, masks and physical overfetch are not inferred. |
| Numerical accuracy | No conversions or arithmetic changes. Invalid geometries now reject instead of silently becoming different views; aliasing semantics remain caller responsibility. |
| Control complexity | One shared bounded validator; no new mode product. Empty/default/rejection and overlapping-view contracts are explicit. |
| Verification | Four modes, dimensions, six widths, six bases, five strides, large products and exact boundaries; independent wide oracle plus sequence/reset/bulk/range gates. |
| Compiler/runtime | Callers must check init status and distinguish representable addresses from allocated capacity. Existing layout modes/defaults stay available; zero-width nonempty linear callers must supply a real width. Compiler emission is not newly integrated. |

## Verification actually executed

- Baseline and final `make clean && make`, then `make test-quick`: PASS.
- `make test-agen-affine-bounds`: **18,042 cases**, **8,146 rejected** as
  independently expected; NULL/unknown/default gates and six-row matrix PASS.
- Focused `test-agen` **14/14**, tiled sweep **17,069 cases**, iterator-chain
  capacity **12 rows** and allocation-failure **514 cases**, scatter/gather
  **15/15**, config **31/31**, convolution **12/12**: PASS.
- A fresh archive overlaid with the exact changed implementation/test/build
  inputs, fully rebuilt with ASan/UBSan and leak detection, passes affine,
  address, tiled and chain targets. This covers exercised paths, not all library
  behavior. No stale shared library is involved in the new static-linked gate.
- `make test` still stops at attention **7/9**: deterministic-small and causal
  mask both report `max_err=inf`. A separately clean archive of baseline
  `a7cd792` reproduces those same two failures in its full aggregate. Isolated
  baseline attention initially reports 8/9 (scale-differentiation failure);
  repeated runs later report baseline 7/9 and modified 6/9 (an additional
  identity-style numerical failure). Thus failure membership is execution-
  sensitive, not a stable defect-set fingerprint. Raw repeat logs are retained;
  the full suite is **not green**, and no unrelated attention fix is bundled.
- A fresh exact-index archive also builds and passes quick, configuration,
  convolution and scatter/gather tests before commit.

Reproduction:

```sh
make clean && make
make test-quick
make test-agen-affine-bounds test-agen test-agen-tiled-sweep test-agen-dma-chain
make test-scatter-gather test-config test-conv
make test
```

In a fresh archive, build all objects consistently for sanitizer checks:

```sh
make libtucmodel.a CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer'
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 make test-agen-affine-bounds test-agen test-agen-tiled-sweep test-agen-dma-chain CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-lm -fsanitize=address,undefined'
```

Retained compressed raw logs are in `evidence/affine-iterator-address-bounds/`.
`INPUT_SHA256SUMS` binds implementation/test/build inputs and `SHA256SUMS` binds
logs and that input manifest. The baseline hash above identifies execution input;
this publication preserves the resulting evidence without self-referential hashes.

## Backlog audit and boundaries

Completed this one newly evidenced correctness item. Existing blocked SRAM
arbitration still stores `arb_mode` without consuming it for requester grants;
requester identity, simultaneous port admission, replay and fairness contracts
remain dependencies. Physical dataflow movement and compressed-domain MMA feed
retain movement/overlap and token/FIFO contracts. No evidence supports removing
slower plausible layouts. The user-owned untracked `TERMS.md` remains untouched
and excluded from publication.
