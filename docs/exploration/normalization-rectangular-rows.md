# Normalization rectangular-row admission

**Input revision:** `863a10bd8209a7ce80b3f67cb9efd51b0d6da971`. This is a functional model-correctness change, not a physical performance result. The earlier `normalization-descriptor-admission.md` describes the bounds-only revision and its live five-element/two-column dropped-tail observation.

## Contract and alternatives

A nonzero `norm_axis_dim` now must divide `elem_count`, before parameter loads, output writes, counters or statistics are touched. Axis zero continues to mean one whole-tensor row; all nonzero divisible axes continue to describe regular rows. Rejection uses the existing validation `false` and execute `UINT64_MAX` results. This deliberately changes previously accepted malformed descriptors, not the default architecture, ABI, runtime configuration, valid numerical algorithm or valid modeled stall formula. The unreachable purported short-final-row branch has been removed.

* **Regular rows / pre-admission:** useful for regular accelerator reductions; avoids missing outputs and ambiguous per-row statistics. Unsupported shapes fail, so the compiler/runtime must pad, bucket, split, or handle ragged inputs elsewhere. Validation adds unmeasured host work; comparable physical checking may cost comparators/control.
* **Short final row:** can represent packed variable-length work with less padding, but needs a documented scale/bias extent, row width and output/statistics capacity ABI and partial-row reduction semantics. Its dynamic lengths/control and verification are expected to cost resources; no latency or traffic advantage has been measured. It remains blocked on those contracts, not rejected for performance.
* **Padded/masked row:** may keep regular processing and suit SIMD lanes, but requires defined masked statistics, padding storage or metadata and producer/consumer rules. It can increase traffic and needs mask/control logic; precise cost and numerics are unquantified. It also remains blocked.

The unchecked floor-division behavior is excluded as incorrect: it reports success without processing all declared elements. The accepted BatchNorm enum is tested only for existing compatibility, not as a claim that BatchNorm is correctly implemented. RMS `mean_out` remains outside the tested/defined contract.

## Measured matrix and fidelity

`tests/test_norm_descriptor_bounds.c` now includes 1,080 descriptor combinations: three accepted mode values, in/out-of-place, element counts 1–12 and axis values 0–14. The independent rectangular oracle gates acceptance, pre-access immutability for invalid cases, and valid output/statistics against double-precision reference calculations. Twenty-four valid and four zero-dimension 2D-wrapper controls are also exercised. The complete focused gate, including the earlier 432 bounds cases, performs **23,002 checks, zero failures**. The input-revision library with the same test in safe validation-only mode fails **258 of 1,986 checks**; unsafe malformed descriptors are not executed on the baseline.

The complete 14-row static-linked existing normalization sweep is retained as `evidence/2026-11-09-norm-rectangular/norm-sweep.txt`; it is byte-identical to `evidence/2026-11-08-norm-descriptor/norm-sweep-baseline.txt`. For each of 256, 512, 768, 1024, 2048, 4096 and 8192 elements, LayerNorm and RMSNorm respectively report 512, 1024, 1536, 2048, 4096, 8192 and 16384 **local SRAM stall cycles**, not total engine cycles. These regular inputs show compatibility, not equal physical arithmetic latency. Full red, final and sanitizer-focused transcripts are retained compressed alongside the table.

No reduction in modeled data bytes or valid numerical changes is intended. Physical SRAM/DRAM traffic, device throughput and latency, area, power/energy, and host validation overhead are not measured. More flexible ragged modes would affect accuracy/rounding, compiler metadata, control and verification depending on their eventual contracts; no quantitative estimates are asserted. Scratch placement and wrapper multiplication/overflow can occur before descriptor admission; the small 2D controls do not certify those separate paths. Overlap, non-4-byte SRAM, host statistics capacity, RMS mean, and BatchNorm semantics also remain unverified.

## Reproduction

From a clean source tree with the change applied:

```sh
make clean && make -j4
make test-norm-descriptor-bounds
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test-norm test-softmax test-norm-attention-sweep
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel tests/test_norm_sweep.c ./libtucmodel.a -lm -o "$TMPDIR/norm-sweep-static"
"$TMPDIR/norm-sweep-static"
```

The focused test was also compiled and run with all library objects built using `-fsanitize=address,undefined -fno-omit-frame-pointer`, with leak detection and halt-on-error enabled. No physical-device or calibration claims follow from these tests.
