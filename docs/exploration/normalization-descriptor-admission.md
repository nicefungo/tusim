# Normalization descriptor admission and row partitioning

**Input revision:** `fbdc1e836eb44954fd1ccd6133540c53f7e08aee`. This is a pre-spec software-model correctness exploration, not a calibrated normalization hardware benchmark. `tests/explore_norm_descriptor.c` is an executable **snapshot probe**, not a conformance gate: its final zero failure count only checks the observations selected by that probe. Full before/after probe output is retained in `evidence/2026-11-08-norm-descriptor/`.

## Question and live observations

Can a caller depend on descriptor acceptance to mean that all declared SRAM spans are safe and all declared elements are normalized? Both LayerNorm and RMSNorm are public runtime modes. The probe initializes five FP32 values `[1,3,5,7,9]`, a 128-byte SRAM, a distinct output prefilled with `99`, and axis length 2. It compares four and five elements. The 4-element control legitimately leaves the fifth sentinel untouched. With **5 elements**, both modes also report success while leaving that fifth output and the third statistics slot untouched. The runtime computes `num_rows = elem_count / axis_dim`; the commented last-row branch cannot catch the remainder with this floor count. Softmax's own validator rejects the same 5/2 partition. This observed result applies to the descriptor path and this fixture; it does not establish the intended ragged-row semantics.

The same baseline accepts a 2-element output at offset 124 in a 128-byte region, a wrapped input offset `UINT32_MAX-3`, and 2-element gamma or beta at offset 124. These were **validator-only** probes: unsafe descriptors were not executed. Exact-end output offset 120 is accepted. The final validator continues to accept the exact-end control and rejects each of those four invalid examples. The final row-partition observations remain unchanged: this patch does **not** fix or endorse dropped tails.

A separate gate covers the 3 accepted descriptor mode values, in-place/out-of-place, whole-tensor/explicit axis and 9 offsets for each of data, output, gamma and beta: 432 cases. It uses a 64-bit extent oracle, checks invalid execution is rejected before bytes, statistics or counters change, and runs six valid numerical/statistical controls. The same gate linked against the input revision fails **137 of 906 validation-only checks**; final execution passes **1,926 checks**, plus five large-count/axis-extent checks included in that total. Valid examples include an exact-end output and axis-sized parameter arrays for multiple rows. The baseline was not run on unsafe descriptors. The test retains the current implementation's `TU_NORM_BATCH_NORM` behavior as a compatibility observation, **not** as evidence of actual batch normalization: source executes the non-RMS row algorithm and only loads beta for LayerNorm. RMS `mean_out` is also not certified: source does not initialize its row mean before the caller copies it. Both require separate contracts.

## Implementation and alternatives

`tu_norm_validate_desc()` now checks the input, consumed output, gamma and beta byte spans using 64-bit arithmetic before allocations or SRAM access. Parameters use the effective axis width, not the full tensor size. The in-place output field and beta for modes that do not consume it retain their prior ignored semantics. Exact end-of-region spans remain valid. Invalid descriptors return the existing `UINT64_MAX` error through `tu_norm_execute`; valid mode selection, data values, timing formula, ABI, YAML and runtime defaults are unchanged. There is no useful runtime mode for unsafe validation or wrapped addresses.

* **Whole-tensor normalization** (`axis_dim=0`) suits reductions across one vector, but increases the reduction's live range and parameter span. **Regular per-row normalization** bounds independent reductions and can support parallel row scheduling, but repeats row setup and needs an explicit shape/axis contract. Both already exist as runtime descriptor choices.
* For a **ragged final row**, rectangular rejection prevents silent truncation and avoids an ambiguous number of per-row host statistics. A short final row could support packed variable lengths but needs an output, parameter, mask and statistics-capacity ABI. Explicit pad/mask can retain regular hardware but needs padding storage/traffic and defined exclusion from the reduction. None is universally optimal; no ragged policy was added. A fail-closed rectangular admission fix is READY separately; full ragged execution remains blocked on those contracts.
* **Checked pre-admission** prevents partially performed invalid operations; in physical hardware comparable range checking could require comparator/control resources. **Per-beat partial fault** might permit streaming but needs error/retirement, partial-write and retry semantics. Unchecked access is excluded as an unsafe model contract. This implementation is host-side validation, not a simulated range-check unit.

For valid inputs, the bounds check adds host validation work; its wall-clock overhead was not measured. No modeled SRAM/DRAM bytes, numerical arithmetic or modeled engine stalls were changed. Host checks do not establish device throughput, latency, area, power or energy; physical comparators and routing could add cost. Changes to padding or ragged handling could alter memory traffic, numerical statistics and compiler/runtime shape metadata. Any new error behavior requires compiler/runtime propagation and test coverage; fail-closed behavior deliberately changes malformed-descriptor acceptance.

## Sweep and limits

The existing static-linked `tests/test_norm_sweep.c` ran against both revisions: the complete 14-row LayerNorm/RMSNorm output is byte-identical, retained as `norm-sweep-baseline.txt` and `norm-sweep-final.txt`. For 256/512/768/1024/2048/4096/8192 elements, both modes report respectively 512/1024/1536/2048/4096/8192/16384 **local SRAM stall cycles**. These values are not total engine cycles, physical latency, or evidence that the two arithmetic pipelines cost the same. The sweep has no independent numerical oracle; the new valid controls do.

This fix does not certify the convenience wrappers, whose temporary-address arithmetic and multiplication occur before descriptor validation; overlapping input/output, parameter aliasing, non-4-byte SRAM bank widths, host pointer/statistics capacity, allocation-failure rollback or the ragged/BatchNorm/RMS-mean contracts. Softmax bounds and other engines were not repaired here. Scalar SRAM error reporting itself is separate from descriptor-level admission.

## Verification and reproduction

```sh
make clean && make -j4
make test-norm-descriptor-bounds
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel \
  tests/explore_norm_descriptor.c ./libtucmodel.a -lm -o "$TMPDIR/norm-probe"
"$TMPDIR/norm-probe"
```

Also run: `make test-norm test-softmax test-attention test-norm-attention-sweep` with the same library path. A separate HEAD archive with the changed normalization files and test overlaid was rebuilt with ASan/UBSan, and the focused gate passed with leak detection enabled. The full aggregate's shared-library targets need the explicit library path; the focused target statically links the rebuilt archive. Logs for the baseline rejection gate, focused final gate and exercised sanitizer gate are retained with the matrices. Tests establish behavior only for the exercised configurations and do not prove physical timing or arithmetic conformance in general.
