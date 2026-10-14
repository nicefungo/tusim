# Transposed address-generation correctness

## Question and executable contract

Can column-first traversal visit every element of a rectangular row-major tensor exactly once? The baseline `ba22664` failed its existing 3×4 gate: the second address was 268 instead of 272 at base 256 and four bytes/element. Initialization retained the original row count as pitch; the consumer requires the original column count.

The corrected contract is `address = base + (row * cols + col) * element_bytes`, traversing columns outside rows. The runtime API remains `tu_agen_iterator_init(..., TU_AGEN_MODE_TRANSPOSED, base, uint32_t[3]{rows, cols, element_bytes})`. Zero element size retains the four-byte default. No YAML/JSON setting is added: this is an existing executable API mode, not a new architecture knob.

## Matrix and findings

The functional gate covers seven shapes × five size settings: 3×4, 4×3, 4×4, 1×7, 7×1, 0×5, 5×0; sizes 1/2/4/8 and zero-default; nonzero base 0x123. It checks exact order, bounds, unique/complete coverage, total, repeated exhaustion without state advance, reset replay, and bulk replay.

Representative source-formula address-index accounting (not timing measurements):

| Shape | Elements | Old unique indices | Old out-of-tensor addresses | Correct unique indices | Correct out-of-tensor addresses |
|---|---:|---:|---:|---:|---:|
| 3×4 | 12 | 10 | 0 | 12 | 0 |
| 4×3 | 12 | 12 | 3 | 12 | 0 |
| 4×4 | 16 | 16 | 0 | 16 | 0 |
| 1×7 | 7 | 7 | 0 | 7 | 0 |
| 7×1 | 7 | 7 | 6 | 7 | 0 |

Square-only tests cannot discriminate the defect. The expanded test linked against the pre-fix source fails with 12/14 tests passing; the corrected static archive passes 14/14, including all 35 matrix cases. The original gate was observed RED before production modification (12/13), then GREEN (13/13).

## Realistic alternatives and costs

Row-first strided traversal and column-first traversal are both retained as runtime API choices. Row-first can favor contiguous fetch/coalescing; column-first can match a transposed operand consumer without physically rearranging storage, but may fragment accesses and increase bank/command pressure. The incorrect pitch is not retained as an optional mode: it is neither another storage contract nor a valid permutation.

- **Throughput/latency:** no speedup claimed; this iterator exposes addresses, not calibrated service timing. Hardware locality effects depend on downstream ports, burst rules, and scheduling.
- **Area/resources:** the pitch correction requires no new modeled state or datapath; actual AGU/mux/transpose-buffer costs are unquantified.
- **Power/energy:** unquantified; changed address activity and locality can change downstream energy in either direction.
- **SRAM/DRAM traffic:** each valid element is requested once; issued element count is unchanged. Physical overfetch, coalescing and bank traffic are not measured here.
- **Numerical accuracy:** no arithmetic/rounding change; correct operand addressing removes duplicates, omissions and out-of-tensor accesses.
- **Control/verification:** existing mode/ABI unchanged; 35 extra permutation cases and a quick-suite dependency increase host verification coverage.
- **Compiler/runtime:** producers must supply original rows/columns, not dimensions of physically transposed storage. This does not prove compiler-to-DMA integration.

`test-agen` now links `./libtucmodel.a` explicitly, preventing an archive-only rebuild from running a stale shared library, and participates in `test-quick`. No unrelated addressing modes, scalar overflow policy, DMA range-chain semantics, or default architecture settings are changed.

## Verification (2026-10-14)

- Baseline `make clean && make && make test-quick`: PASS.
- Baseline `make test-agen`: FAIL, 12/13; wrong rectangular pitch.
- Corrected `make test-agen`: PASS, 14/14 including 35 cases.
- Expanded regression compiled with `git show HEAD:tu_cmodel/memory/address_generator.c` from pre-fix HEAD and the current static archive: expected exit 1, 12/14. This independently confirms the added gate detects the original code.
- Post-change `make clean && make && make test-quick`: PASS, including the 874-row DRR scan-width regression and nine queued-order gates.
- `make test-conv`: PASS, 12/12.
- `make test-multicast`: still FAIL, 9/10; stale overflow-completion assertion is the separate READY backlog item, not a regression or a changed contract in this fix.

The full aggregate is not claimed green; its previously documented attention failures are outside this bounded correction. Multicast is deliberately left as the remaining daily item because two READY candidates were present and are independent.
