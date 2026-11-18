# Softmax row scheduling and overlapping SRAM placement

**Date:** 2026-11-18. **Mode:** pre-spec functional/storage exploration.
**Baseline:** `0839141726b7dfa8630341694bfc143979fc15f9`.
**Question:** can row-buffered softmax reuse overlapping input/output storage
without destroying unread rows, while retaining fixed-direction implementations?

## Evidence and contract

The baseline admits overlapping spans but always traverses rows forward.
A two-row `[0,0; 0,2]` tensor at byte 0, written to byte 8, returns the second
row's first probability as **0.5**, rather than the independent reference
**0.119202922022**: the first output overwrites the second input before its read.
The corrected AUTO path returns **0.119202919304** in FP32. The minimal tracer
failed before the production edit and passed afterward.

Buffer reuse is a real compiler constraint, not a throughput-only objective.
The first-party MLIR *Bufferization* documentation describes reducing allocation
and copying while avoiding writes that destroy values needed later (sections
“Goals of Bufferization” and “Destination-Passing Style”, retrieved 2026-11-18,
`https://mlir.llvm.org/docs/Bufferization/`). That motivates this study; it does
not establish a TU pipeline, MLIR integration, or a hardware performance result.

The engine already stages a **complete FP32 row** before writing that row.
Consequently a safe whole-row direction suffices for equal-length contiguous
input/output spans. No tensor-wide snapshot or new arithmetic kernel is needed.

`tu_softmax_desc_t.row_schedule` is the live runtime configuration:

| Value | Mode | Motivation and sacrifice |
|---:|---|---|
| 0 | `TU_SOFTMAX_ROWS_AUTO` | Choose forward unless higher-address overlapping output requires reverse. Enables compiler-selected overlapping placement with row-sized scratch; costs direction selection and both schedule paths. |
| 1 | `TU_SOFTMAX_ROWS_FORWARD` | A fixed ascending sequencer can omit bidirectional selection and suit an ascending consumer. Rejects higher-address multirow overlap instead of corrupting input; compiler must provide a compatible placement. |
| 2 | `TU_SOFTMAX_ROWS_REVERSE` | A fixed descending sequencer can suit descending layouts/consumers and higher-address overlap. Rejects lower-address multirow overlap; producer/consumer compatibility is a software obligation, not modeled overlap speed. |

AUTO is zero/default. Previously safe layouts retain the old forward sequence:
exact in-place, disjoint spans, and lower-address overlap. Higher-address
multirow overlap is deliberately corrected, not retained as an incorrect mode.
A one-row operation may use either direction because its full input is buffered.
Exact alias and disjoint spans also admit either fixed direction. Byte-misaligned
spans are supported with the existing four-byte scalar accesses and memcpy.
Within-row element and reduction order are unchanged. Masks and statistics use
**logical row indices**, never execution-step indices.

Unsupported enum IDs and incompatible fixed schedules reject in descriptor
validation before allocation, reads, writes, counters, or statistics mutation.
Existing overflow/span checks precede overlap admission. In-place ignores
`out_offset`, including its historical otherwise-invalid values. The field is
appended: binary consumers must rebuild, and source callers must initialize the
whole descriptor. Wrappers use AUTO. No YAML/JSON knob, compiler lowering, or
selection of attention's independent normalization kernel is claimed.

**Boundary:** mask/max/sum host buffers must have the documented sizes and must
not alias SRAM or each other; host-pointer aliasing is not validated. SRAM scalar
word width must be four bytes; concurrent buffer mutation is unsupported.
Invalid-schedule rejection is atomic, but malloc failure after earlier completed
rows is still nontransactional. Nonfinite/all-masked semantics are unchanged.

## Complete measured trade-off matrix

[evidence/softmax-row-schedule/matrix.csv](evidence/softmax-row-schedule/matrix.csv)
contains **45 rows**: three schedules × three softmax modes × five placements.
The fixture has three rows of four FP32 elements, a 1024-byte default SRAM region,
input offset 384, additive masks, scale 0.5, legacy exponentials, size-only word
metering, input prefill, and no modeled cycle advance. Outputs and statistics
are checked against an independent long-double mathematical reference.

All accepted rows perform **12 scalar reads and 12 scalar writes**, with a
**16-byte host row buffer**. The table below applies to STANDARD, LOG and ONLINE;
the CSV preserves each numerical output-mode row rather than reporting a winner.
“Union” counts occupied input/output SRAM bytes, not a measured allocation size.

| Output minus input, bytes | Input/output union, B | AUTO | FORWARD | REVERSE | Accepted local SRAM penalties |
|---:|---:|---|---|---|---:|
| -48 | 96 | forward | forward | reverse | 24 |
| -4 | 52 | forward | forward | reject | 46 |
| 0 | 48 | forward | forward | reverse | 48 |
| +4 | 52 | reverse | reject | reverse | 46 |
| +48 | 96 | forward | forward | reverse | 24 |

The four-byte shifted overlaps use **45.83% fewer union bytes** than disjoint
placement; exact alias uses **50% fewer**. This is footprint reuse, not reduced
logical access traffic. The local penalties increase from 24 to 46 or 48 because
placement reuses banks whose grants the input prefill has consumed. They are
scalar accounting sums in an unadvanced fixture, **not device latency or
throughput**, and do not imply a physical penalty of the same magnitude.
Changing row direction at the same legal placement does not change these totals.

| Dimension | Gain / sacrifice / fidelity limit |
|---|---|
| Throughput and latency | No physical ranking. Same logical accesses and row arithmetic; direction may affect a real streaming producer/consumer, but injection, overlap, queues and calibrated timing are unmodeled. |
| Area / resources | Overlapping placement reduces input/output union footprint in this fixture; row scratch remains 16 B. Fixed direction can simplify control; AUTO needs direction/overlap selection and reverse indexing. Physical SRAM sizing, selector area and ports are **unquantified**. |
| Power / energy | Same logical read/write count; fewer reserved bytes might permit a smaller provisioned SRAM, while bidirectional control may add switching. Neither static nor dynamic net energy is modeled; **unquantified**. |
| SRAM / DRAM traffic | Every accepted case reads/writes 48 logical SRAM bytes per direction. Physical SRAM activity differs from union size. Masks/stats are host-side; no mask DMA or standalone DRAM traffic is measured. |
| Numerical accuracy | Within-row equations and reduction order unchanged; storage corruption removed. Fixed/AUTO output rows agree for the same inputs, and the independent oracle passes. No new rounding or stochastic mode. |
| Control complexity | Fixed direction restricts admitted placement; AUTO adds a range decision and selectable row indexing. Full-row staging, not scalar reverse writing, is essential. |
| Verification burden | Three schedules across all output/exponential/preprocess modes, masks, one/multirow, byte shifts and two SRAM word-meter modes; explicit read/write trace and pre-admission immutability gates. |
| Compiler / runtime | Compiler must preserve required live input values and select an admissible schedule/placement. Output, masks and stats remain logical-row ordered after completion. No compiler integration or asynchronous row-ready API is implemented. |

**Other alternatives:** a tensor-wide snapshot could preserve a chosen forward
schedule even with destructive forward overlap, but requires O(tensor-size)
storage and a separate scratch/traffic/producer-timing contract. It is deferred,
not rejected for being slower: adding unmetered host memory would not establish
that hardware organization. Partial-write/fault recovery likewise needs explicit
status/retirement semantics. The current small abstraction preserves both fixed
hardware-plausible sequencers and a selectable-direction implementation.

## Verification executed

- Baseline and final `make clean && make`, `make test-quick`: pass. Final full
  `make test` passes with the local shared-library directory explicitly supplied.
  Only the same pre-existing logging/config unused-symbol build warnings remain.
- Focused gate: **28,542 cases / 1,748,646 checks**, **2,604 expected rejections**,
  **zero failures**. Includes independent per-byte read-after-write hazard
  simulation, widths 1/2/5/17, one/three rows, eleven signed byte displacements,
  all three output and exponential modes, both preprocessing orders,
  absent/additive/multiplicative masks and both SRAM word meters.
- GNU linker wrapping records actual `tu_sram_read/write` order. Exact full-region
  byte guards, descriptor/mask preservation, logical per-row statistics, disabled
  meter replay, in-place ignored offset, axis-zero whole row, unknown selectors,
  and rejected-call byte/counter/statistics immutability pass.
- The complete old source archive rebuilt with this harness and only the new
  descriptor field omitted fails **163,224 checks**, exit 1. After automatic
  traversal alone, the explicit-policy tracer fails **77,496 checks** until the
  fixed schedules and admission rules are implemented.
- Legacy-safe AUTO subset: **8,214 cases / 530,562 checks**, zero failures on
  baseline and final source. Output/statistics/per-bank accounting fingerprint
  and summary are byte-identical: `ebb9a2108e8b59c8`.
- Six safe source mutants are rejected: always-forward (139,788 failed checks),
  ignored fixed reverse (96,924), skipped hazard admission (23,328), skipped enum
  admission (108), physical-step mask indexing (49,146), and physical-step max
  indexing (8,718). Mutations exist only in isolated scratch copies.
- Complete-library ASan/UBSan/LSan build: new gate/matrix plus preprocessing,
  exponential public/private, near-zero LOG, finite-range LOG, legacy softmax and
  overflow suites pass. This is exercised-path evidence, not a claim that the
  entire aggregate was run under sanitizers.
- Prior standalone softmax and softmax-after-attention sweeps pass; their complete
  paired outputs match the rebuilt baseline. Historical cross-engine timing
  interpretations are not recertified by this comparison.

```sh
make clean && make
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
make test-softmax-row-schedule
./test-softmax-row-schedule --matrix
./test-softmax-row-schedule --legacy
```

The focused target links `./libtucmodel.a` explicitly and is integrated into
quick/full/clean/ignore paths. Full logs and verification/mutation drivers are
retained outside the checkout at
`/home/zxy/Workplace/archives/tusim-20261118-softmax-row-schedule/`.

## Backlog audit

The initial table contained no READY candidate. Source re-audit confirmed SRAM
arbitration is still stored metadata without a requester/grant consumer;
context save/restore restores cycle metadata without per-bank grant snapshots;
im2col padding still emits its untagged sentinel; normalization wrappers still
stage parameters before descriptor validation. Those documented dependencies
remain blocked rather than converted into speculative implementations.

This one new placement/schedule exploration found a correctness defect and a
small validated runtime alternative family, both completed in this run. No READY
item remains in the updated audited table. Unrelated untracked `TERMS.md` is
preserved and excluded from the publication.
