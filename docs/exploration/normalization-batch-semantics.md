# Normalization mode semantics: BatchNorm label versus execution

**Exploration input revision:** `aa29882ad097c71128b2707cdfff308219e5e6a4`. This is a reproducible semantic audit, not a BatchNorm implementation or a physical performance comparison. The existing public mode remains unchanged. **Do not use `TU_NORM_BATCH_NORM` as a BatchNorm oracle:** it accepts the descriptor but executes row-wise mean/variance normalization and ignores beta.

## Question and source-to-consumer audit

Can the existing mode represent per-channel batch statistics or inference-time precomputed statistics? The enum/header describes a batch axis, but `tu_norm_validate_desc` accepts all three modes; `tu_norm_execute` loads beta only for LayerNorm; and `normalize_row` takes the RMS branch only for RMSNorm, taking the same row mean/variance branch for the other two. The interface has a row width but no channel/layout shape, running mean/variance inputs, training/inference selector, or running-statistics update contract. Search of the checked-in C sources found no other production call path that dispatches `TU_NORM_BATCH_NORM`; the descriptor is publicly callable. This audit does not establish usage outside this repository.

The existing 1,080-shape descriptor regression includes the enum as a compatibility case. It establishes the existing row behavior, not BatchNorm conformance. Changing acceptance or output would therefore be an observable API/semantic change, not a harmless optimization.

## Executable discriminating matrix

The static-linked `tests/explore_norm_batch_semantics.c` exercises all 24 combinations of mode (LayerNorm/RMSNorm/BatchNorm label), axis (whole 4-element tensor or 2-element row), in-place/out-of-place and beta absent/present. Input is row-major `N=2,C=2`: `[[1,10],[3,14]]`; gamma `[2,0.5]` and beta `[0.25,-0.5]` repeat as needed; epsilon is `0.01` stored as FP32. The independent row oracle explicitly represents the *observed legacy behavior*, including BatchNorm's missing beta. A second independent oracle uses per-channel population variance across N and optional beta. Both reject non-finite comparisons. The test intentionally **passes when it detects this known semantic mismatch**; it is a pinned diagnostic snapshot, not a conformance gate and must be updated if the API is intentionally fixed.

For the axis-2/in-place/beta-present row, legacy `BatchNorm` returns approximately `[-1.999506, 0.499877, -1.999670, 0.499917]`. The independent per-channel training reference returns approximately `[-1.740074, -0.999376, 2.240074, -0.000624]`: maximum absolute difference `4.239744`. Its output is also identical to the same legacy BatchNorm case without beta. All 24 rows match their independent legacy row-oracle within `1e-6` (observed maximum zero at printed precision), and all eight BatchNorm-label rows differ from the independent per-channel reference by more than 0.1. See `evidence/2026-11-10-norm-batch/matrix.txt` for the complete matrix and per-row local stall outputs.

All shown executions report eight local SRAM stall cycles for these four elements; the helper drops parameter-load stalls and the counter does not model total arithmetic latency, so this is **not** evidence that the alternatives have equal execution time. It does not model a real BatchNorm path at all. This single well-conditioned FP32 example is discriminating but not a general numerical accuracy study.

## Realistic alternatives and costs

| Alternative | Motivation / regime | Sacrifices and dependencies |
|---|---|---|
| Legacy row behavior | Existing software may rely on it despite the misleading label; retaining a clearly named compatibility mode avoids silent result changes. | Not per-channel BatchNorm; beta ignored; callers must not treat it as such. Renaming/versioning and compiler migration require care. |
| Inference BatchNorm using stored channel mean/variance and gamma/beta | Suits fixed-statistics inference; per-channel affine operations can also be folded into a preceding operator when numerical and quantization contracts permit. | Requires per-channel parameter storage or folded weights, layout/axis and parameter lifetime contracts; folding changes quantization/rounding and may increase weight-management work. SRAM traffic, area, energy and physical throughput are unmeasured. |
| Training/statistics BatchNorm | Calculates statistics across batch and specified spatial axes and may update running statistics; supports data-dependent training or calibration. | Requires a reduction/accumulation schedule, batch/spatial and synchronization semantics, FP accumulation/rounding, variance convention, epsilon and running-statistics rules. Cross-tile/cores need state and synchronization; expected control/storage/verification burden is greater. Physical cost, latency and accuracy are unquantified. |
| Explicit rejection | Prevents new callers from receiving mislabeled results without requiring an unimplemented operator. | Breaks existing callers/tests unless introduced with an explicit compatibility mechanism/version and migration. Does not provide working BatchNorm. |

No speed winner is asserted. Area/resource, power/energy, SRAM/DRAM traffic, physical throughput and latency remain unquantified; wider reductions and persistent state are expected to increase storage, control and verification requirements, while inference folding can trade separate traffic for parameter/precision constraints. Numerical differences are central and must be specified; compiler/runtime must carry layout, axis and parameter identity and respect mode/version. No alternative is excluded because it may be slower.

## Readiness and verification boundary

A true BatchNorm mode is **blocked**, not implemented: define training versus inference, layout and reduction axes, shape and parameter bounds, epsilon/variance/rounding/special-value policy, beta and statistics behavior, and cross-tile state/synchronization. Establish a backward-compatible legacy selector or intentional migration before altering the existing public enum. A future implementation must compare against independent references on multiple shapes, channels and tiles, both placements, gamma/beta combinations, fixed and varying statistics, edge/special values, invalid descriptors, explicit legacy behavior, config propagation where applicable, and exact measured counter domains. Wrapper scratch/alias issues and RMS `mean_out` remain separate.

## Reproduction

From the revision above, with the probe copied into `tests/`:

```sh
make clean && make -j2
make test-quick
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel tests/explore_norm_batch_semantics.c ./libtucmodel.a -lm -o "$TMPDIR/norm-batch-semantics"
"$TMPDIR/norm-batch-semantics"
make test-norm-descriptor-bounds
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test-norm
```

The probe also passed with **all library objects** and the probe built with AddressSanitizer and UndefinedBehaviorSanitizer, leak detection enabled. `test-norm` without the local shared-library path failed to load `libtucmodel.so`; the documented retry passed. The clean baseline was run in a `git archive` checkout to preserve unrelated work. `TERMS.md` was not changed.
