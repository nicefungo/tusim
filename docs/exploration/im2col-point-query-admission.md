# Random-access im2col: checked point queries

## Question and executable evidence (2026-10-21)

Can `tu_agen_addr_im2col()` safely query a convolution source point independently
of the streaming iterator, including oversized expansions and large padded
coordinates? Baseline: `69330281ac53c1af9fb0764e5d971806e1b0ae5c`.
No READY implementation item existed at entry; this is one bounded correctness
exploration, not a new compute engine or a feature-checklist exercise.

The previous geometry gate explicitly excluded this public helper. Source
inventory finds its definition/declaration and one ordinary 5x5 test call;
there is no proved compiler/queue execution edge. Eleven safe RED probes all
failed on a clean baseline archive: invalid channels/kernel indices were
accepted, an out-of-domain output index wrapped to address zero, incomplete
source/element extents were admitted, and zero width/stride/kernel/dilation
returned apparently usable addresses. No generated address was dereferenced.
The full new gate rejects baseline with **5,530 mismatches**, including a valid
coordinate sum above uint32_t that the old helper incorrectly rejected.

## Alternatives / contract

| Alternative | Hardware rationale and disposition |
|---|---|
| Checked random-access point query | Supports compiler planning, validation and independent point requests without materializing an expanded tensor. Implemented on the existing API; it recomputes output axes and checks all five index domains and the complete NCHW source byte extent. |
| Checked streaming iterator | Stateful sequential feed avoids caller-side tuple sequencing. Already present and retained; its expanded entry count must fit its uint32_t progress counter. It is not substituted for point queries. |
| Unchecked private producer | Could be safe behind a proof-bearing compiler boundary, but this public descriptor has no such proof. Wrapped addresses and malformed controls are excluded, not preserved as a faster architecture mode. |
| Materialized expansion / host prepadding / tagged on-device fill | Can simplify feed or eliminate external zeros at different storage, traffic and control costs. Not implemented by this helper. Tagged layout/fill/ownership/retirement semantics remain dependencies. |

The helper uses existing runtime C descriptor fields: rectangular kernels,
independent stride/dilation/padding, channels, base and element width. No new
ABI, struct field, opcode, JSON/YAML knob or global default is introduced.
`im2col_output_dims()` and the existing complete source-extent checker are
reused; coordinates are widened before multiplication and padding subtraction.
Checked output/kernel indices bound their sum by the padded input axis.

- Cached output dimensions are ignored and never mutated.
- Kernel/stride/dilation and **element width must be positive**. The helper had
  no documented zero-width default: its old zero-width collapse to `base` now
  rejects. Iterator initialization independently retains its historical
  `elem_size=0 -> 4 B` behavior; no new implicit helper default is invented.
- Source exclusive end may equal `UINT32_MAX`; the sentinel byte itself cannot
  be data. Even an early point rejects when the full source view is invalid.
- `UINT32_MAX` remains the helper result for padding, and now explicitly for
  invalid/empty/out-of-domain queries. This API cannot distinguish these cases.
  There is no copy, fill, allocation, DMA issue or accounting side effect.
- Point queries **do not** inherit the iterator's expanded-count restriction.
  A 1x1x1 source with 65536x65536 kernel, 65535 padding and unit stride/dilation
  has a valid source point at output (0,0), kernel (65535,65535). The helper
  returns base 13 while iterator initialization correctly rejects the expansion.
- Ordinary representable positive-width point results and padding behavior are
  unchanged. Malformed inputs deliberately change to fail-closed behavior.

## Complete displayed matrix

Input 4x5, 2-byte elements, base 13, unit stride/dilation. Counts are logical
queries, **not DMA commands, SRAM/DRAM traffic, cycles or throughput**. The full
bounded matrix also varies H=1..4, W=1..5, channels=1..2, rectangular kernels,
stride, dilation, padding, and widths 1/2/4, with five namespace-bound probes
per shape. A separate valid width-3 query proves non-power-of-two support.

| Channels | Kernel | H/W padding | Output | Queries | Source queries | Padding queries |
|---:|---|---|---|---:|---:|---:|
| 1 | 1x1 | 0/0 | 4x5 | 20 | 20 | 0 |
| 1 | 1x1 | 1/1 | 6x7 | 42 | 20 | 22 |
| 1 | 3x3 | 0/0 | 2x3 | 54 | 54 | 0 |
| 1 | 3x3 | 1/1 | 4x5 | 180 | 130 | 50 |
| 2 | 1x1 | 0/0 | 4x5 | 40 | 40 | 0 |
| 2 | 1x1 | 1/1 | 6x7 | 84 | 40 | 44 |
| 2 | 3x3 | 0/0 | 2x3 | 108 | 108 | 0 |
| 2 | 3x3 | 1/1 | 4x5 | 360 | 260 | 100 |

Padding and kernel/stride/dilation alter the operator, not a free same-workload
optimization. Repeated source queries may exploit reuse or require traffic;
this integer-address gate does not choose a placement/reuse architecture.

## Gains, sacrifices and fidelity

| Dimension | What is established / what is not |
|---|---|
| Throughput / latency | Correct rejection and correct valid wide-coordinate addressing; no measured device speedup. Repeated shape/extent checks add constant host work per point; host time is unmeasured. Streaming can amortize admission but requires sequencing state. |
| Area / resources | Software admission or a bounds-aware AGU is plausible. Physical comparator/multiplier widths, ports and area are unquantified; host uint64 arithmetic is not a prescribed hardware datapath. Materialization requires expanded storage; point/stream feed requires control. |
| Power / energy | Checks add host activity; hardware check/control energy, fill costs, leakage and net energy are unquantified and not wired into the power model. |
| SRAM / DRAM traffic | Valid logical source selection is preserved; invalid requests no longer appear usable. No memory movement is executed here. Reuse, burst fragmentation, prepadding/materialization bytes and external traffic remain unquantified. |
| Numerical accuracy | No arithmetic precision/rounding change. Correct addresses matter for output correctness, but neither convolution arithmetic nor padded zero-fill execution is certified. |
| Control complexity | Reuses checked shape and source extent producers; validates five index domains and widens coordinate formation. Existing ambiguous UINT32_MAX result remains; no queue/fault protocol added. |
| Verification burden | Independent signed/128-bit oracle, complete bounded sequences, namespace edges, source/full-element bounds, cached metadata immutability and oversized expansion need distinct tests. |
| Compiler / runtime | Callers must treat UINT32_MAX as invalid/padding and provide positive width. Existing descriptors retain all valid alternatives. No compiler dispatch, host allocation bounds, SRAM capacity, fill or retirement integration is claimed. |

## Verification and retained evidence

- Initial live baseline `make clean && make`, `make test-quick`: PASS.
- Clean baseline archive build/quick: PASS; the eleven-probe RED control exits
  1 with 11 failures, full point gate exits 1 with 5,530 mismatches.
- Final live clean build/quick and independently extracted final archive
  build/quick: PASS. New target is statically linked and integrated into
  quick/full/clean/ignore paths.
- Point gate: **109,415 checks; 46,401 valid points; 63,014 invalid/padding;
  84,948 point-to-iterator comparisons; zero failures**. Includes 10,000 seeded
  adversarial tuples, NULL/empty/error, zero-control/width cases, exact
  sentinel boundaries, valid wide-coordinate carry, huge source/element widths,
  stale cached outputs and a valid point in an unrepresentable expansion.
- Existing im2col gate: **32,924 cases**; config **31/31**, convolution **12/12**,
  scatter/gather **15/15**: PASS.
- Fresh all-object ASan/UBSan archive with leak detection: point, im2col,
  affine, tiled, iterator-chain and public-range gates PASS. These are
  exercised-path results, not certification of every address API.
- Broad `make test` is **not green**: baseline attention **7/9**, final archive
  **6/9**. Three reruns of the unchanged baseline attention binary give
  **7/9, 8/9, 7/9**, with varying deterministic-small/causal infinity failures
  and scale differentiation. Final additionally fails identity tolerance.
  All attempts are retained; no fixed failure-set equivalence or attention
  root-cause attribution is claimed, and no attention code is changed.

```sh
make clean && make
make test-quick
make test-agen-im2col-point test-agen-im2col-bounds test-config test-conv test-scatter-gather
make test
```

Raw compressed logs, command return codes, executed orchestration script and
complete C/header input hashes are under
`evidence/2026-10-21-im2col-point/`. The runner executed from the recorded
baseline HEAD with the explicitly listed source/test/Makefile overlays; it
extracts regular files/directories, omitting external symlink fixtures unused
by these C tests. Before publication the exact staged snapshot is independently
built and tested again. Checksums bind logs without editing their raw bytes.

## Backlog and scope

SRAM `arb_mode` still has assignments but no live requester arbitration;
compressed streams still reconstruct dense weights with max/sum decoder-DMA
estimation; WS/OS/RS physical movement has no new event/overlap evidence. These
remain BLOCKED, not exclusions based on speed. Padded iterator ranges retain
an untagged sentinel that can alias an address; the point helper's invalid
result does not make range-to-DMA zero-fill safe. No other READY item was
established. Block-interleaved admission, general tiling utility and unchecked
inline helpers are not certified by this gate.

Narrow, reversible changes stay on `main`. User-owned untracked `TERMS.md` is
untouched/excluded; no handoff or generated binary is published.
