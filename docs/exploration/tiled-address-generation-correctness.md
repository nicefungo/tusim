# Compact tiled address generation: rectangular grids and clipped tails

## Question and baseline evidence (2026-10-16)

Can the existing runtime tiled iterators enumerate each element of a row-major
matrix/volume exactly once when tile-grid dimensions differ or tails are ragged?
This is a prerequisite for comparing tile locality, not a throughput benchmark.
The audited baseline is `8b3e3cb`.

Two independent defects were reproduced before production edits:

- **2D tails:** shape 3×5, tile 2×3, four-byte elements, base `0x123`:
  ordinal 8 should address 323 but returned 311. Dividing the logical element
  ordinal by a fixed full-tile volume treats ragged tiles as padded slots.
- **3D rectangular grids:** shape 2×3×8, tile 1×1×2, same size/base:
  ordinal 4 should address 307 but returned 323. The tile-grid decomposition
  used the depth-grid count where the column-grid count was needed. Even
  fully divisible volumes could therefore walk the wrong coordinates.

`tests/verify_agen_tiled.py` recompiles the original address-generator source
against the final harness: it must exit 1 with **both** discriminating failures.
It then instruments the current harness and address-generator source with
ASan/UBSan (leak detection enabled) and requires the complete gate to pass.
Unexercised library subsystems are not claimed sanitizer-verified by this run.

## Implemented runtime contract

Both tiled modes retain compact row-major **storage**, tile-major **traversal**,
row-major tile-grid order and row-major local order. Boundary tiles are clipped;
no padded addresses or padding values are emitted. Initialization and `next`
share one geometry/ordinal model through a constant-work, allocation-free
3D decomposition; 2D uses a singleton depth. The helper first selects a clipped
depth slab, then row slab, then column tile, and finally local coordinates.
Unlike a skip-over-padded-slots loop, oversized tiles do not increase host
iteration work per emitted element.

- `TU_AGEN_MODE_TILED_2D`: existing `tu_agen_tiled2d_config_t`, fixed 4 B elements.
  Grid counts of zero are derived; supplied nonzero grid counts must equal
  ceiling division of tensor dimensions by tile dimensions.
- `TU_AGEN_MODE_TILED_3D`: existing `uint32_t[7] =
  {D,R,C,tile_D,tile_R,tile_C,element_bytes}`; size zero retains 4 B compatibility.
- Positive tile dimensions are required, even for an empty tensor. Oversized
  tiles are legal. Any zero tensor dimension yields a valid empty iterator.
- Element-count overflow, byte-span overflow, zero tile dimensions, and
  inconsistent explicit 2D grid metadata return `-1` before an iterable state
  is installed. `UINT32_MAX` is reserved for exhaustion; the exclusive byte
  end may equal it, but no tensor byte may occupy it.
- No public struct layout, function signature, enum value, YAML/JSON setting,
  rounding policy, default architecture, DMA service formula or valid element
  count changes. Previously correct fully divisible 2D traversal is retained.
  Previously invalid/wrapping geometries are now explicitly rejected.

These are existing executable API configurations, not newly advertised JSON
knobs. Row-first, strided, transposed and tiled alternatives remain available.
There is no compiler-emission or top-level compute-engine integration claim.

## Complete reported geometry matrix

All rows use four-byte elements and nonzero base `0x123`. Counts below are
**logical contiguous address runs**, produced by the actual range generator;
they are not physical DRAM commands, occupied bursts or calibrated cycles.
The first two rows are also printed as initial discrimination controls.

| Rank | Shape D,R,C | Tile D,R,C | Elements | Contiguous runs |
|---|---|---|---:|---:|
| 2 | 1,6,6 | 1,2,3 | 36 | 10 |
| 2 | 1,3,5 | 1,2,3 | 15 | 4 |
| 2 | 1,3,5 | 1,8,8 | 15 | 1 |
| 3 | 2,3,8 | 1,1,2 | 48 | 1 |
| 3 | 3,5,4 | 2,3,3 | 60 | 28 |
| 3 | 5,3,7 | 2,2,4 | 105 | 26 |
| 2 | 1,3,5 | 1,1,1 | 15 | 1 |
| 2 | 1,3,5 | 1,3,5 | 15 | 1 |
| 3 | 3,5,4 | 1,5,4 | 60 | 1 |
| 3 | 3,5,4 | 8,8,8 | 60 | 1 |
| 3 | 3,5,4 | 1,2,2 | 60 | 19 |

The same 3×5 tensor has one or four contiguous runs depending on traversal;
the same 3×5×4 volume has one, 19 or 28. Fewer runs are not automatically a
better accelerator design: a consumer's operand tile, live SRAM capacity,
reuse and compute ordering can require the more fragmented traversal.

## Realistic alternatives and gain-versus-sacrifice limits

| Dimension | Full-width/slab or whole-tensor tiles | Narrow/multidimensional compact tiles | What this cmodel verifies |
|---|---|---|---|
| Throughput | Can expose longer contiguous runs | Can match compute tiling and reuse, but fragment runs | Exact addresses/runs only; no timing or throughput ranking |
| Latency | Fewer logical runs may lower downstream command overhead | Smaller working sets may reduce time to first usable tile | First-tile readiness, queues and calibrated latency unquantified |
| Area/resources | A whole-tile consumer may need more live SRAM/staging | Smaller tiles can reduce live storage but need boundary/control state | No allocator or physical AGU/port/area model in this gate |
| Power/energy | Coalescing may reduce control activity, larger stores may cost leakage | Reuse can reduce traffic, fragmentation may raise control/switching | Direction is workload-dependent; physical energy unquantified |
| SRAM/DRAM traffic | Logical tensor bytes unchanged | Logical tensor bytes unchanged, address order/runs differ | One visit per element; physical overfetch/reloads/bank traffic unquantified |
| Numerical accuracy | No arithmetic change | No arithmetic change | Correct operand permutation; no arithmetic/rounding promise |
| Control complexity | Full-width/divisible shapes simplify boundaries | Ragged multidimensional tiles need clipping and correct grid decomposition | Shared ordinal helper and fail-closed geometry validation |
| Verification burden | Divisible square-only tests miss defects | Rectangular grids, tails, oversized/empty cases enlarge the state space | Independent nested-loop oracle plus sanitizer and old-source rejection |
| Compiler/runtime | Must arrange capacity/layout and matching consumer order | Must provide valid tile geometry and handle clipped tail work | Existing API modes retained; compiler-to-DMA integration not established |

Incorrect padded-slot enumeration is **EXCLUDED** as an implementation of this
compact contract, not because it loses a benchmark. A genuinely padded physical
layout remains a plausible separate contract, **BLOCKED** here on explicit
allocation strides/capacity, padding-value/masking semantics and consumer
agreement. It cannot be represented honestly by emitting out-of-tensor
addresses while retaining `total_elements = D*R*C`. No padded hardware option
is silently replaced by a faster mode.

## Verification and retained evidence

The six-loop independent oracle checks exact order, bounds, alignment,
uniqueness, complete coverage, total, repeated exhaustion without advancement,
reset replay, full/partial bulk replay and flattened contiguous-range equality.
The executed sweep covers **17,069 permutation cases**: 12 3D shapes × 216 tile
shapes × five element-size settings; 64 2D shapes × 64 tile shapes; two initial
controls and 11 reported rows. It includes ragged tails in every axis,
rectangular grids, vectors, empty tensors, oversized tiles, and zero-default
size. Separate gates cover invalid geometry, both 2D grid fields, count/span
overflow, the exact address-space end, and tiles with `UINT32_MAX` dimensions.

```sh
make clean && make
make test-quick
make test-agen-tiled-sweep
make test-agen test-conv test-scatter-gather test-config
python3 tests/verify_agen_tiled.py --baseline-ref 8b3e3cb
make test
```

Results: clean baseline and final builds/quick suites PASS; tiled sweep PASS;
address suite 14/14; convolution 12/12; scatter/gather 15/15; config 31/31.
The sanitized exercised address path passes, and the old source is rejected.
The new target links `./libtucmodel.a`, is in quick/full aggregates and clean,
and its binary is ignored.

An archive of the exact candidate index source/test tree was cleanly
rebuilt in a fresh directory and passed quick, address, convolution,
scatter/gather and config gates, excluding untracked user work. The retained
`snapshot.log` records that run. An independent fresh-context static review
also passed; `review.json` is its verdict, not an independent execution or
physical-model certification.

The full aggregate still stops at attention **7/9**: deterministic-small and
causal-mask report `max_err=inf`. An independent clean archive/build of
`8b3e3cb` reproduces the same two failures. Attention was inspected but not
modified; this work does not establish its root cause or claim full-suite green.

Immutable raw logs are retained in `evidence/tiled-address-generation/`:
`baseline.log`, `build.log`, `quick.log`, `tiled-sweep.log`, `focused.log`,
`sanitizer.log`, `full.log`, `baseline-attention.log`, and
`baseline-archive-build.log`. `SHA256SUMS` binds their contents.
It also binds `snapshot.log` and `review.json`; `INPUT_SHA256SUMS` records the
authored C/header/test/runner/build inputs relative to the repository root.

## Backlog audit and scope boundary

The safe range-capacity probe reports **256 required / 128 stored**, with an
unchanged guard, for a transposed 16×16 tensor. The producer bounds writes
correctly; source inspection of `tu_agen_desc_chain_from_iterator` finds a
private 128-entry array followed by iteration to the unbounded returned count.
The unsafe consumer is deliberately **not invoked**. This establishes a
separate P0 READY capacity-handling correction, with an actionable plan:
stream/allocate the complete range set or reject overflow atomically, then
verify >128-range movement/ownership and injected allocation failures. Do not
silently truncate a successful tensor transfer. It remains the next daily
independent item rather than being bundled into tiled geometry.

**2026-10-17 closure:** this separate item is now implemented and verified;
see `iterator-dma-range-capacity.md` for complete dynamic range storage,
atomic allocation-failure cleanup, host-offset/ownership scope, 12 capacity
rows and 514 injected allocation-failure cases. The paragraph above records
the 2026-10-16 discovery state, not the current implementation status.

SRAM arbitration still stores `arb_mode` without a requester-aware scalar
consumer; its existing simultaneous-request/port dependency remains BLOCKED.
Direct compressed-domain compute and physical dataflow remain subject to their
existing token/FIFO and movement/overlap contracts. No new mode is added merely
to fill the gap checklist. User-owned untracked `TERMS.md` is untouched.
