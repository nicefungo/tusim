# Normalization 2D wrapper shape-count overflow

**Input revision:** `a80363f`. This experiment examines one wrapper admission boundary, not general wrapper safety or physical normalization performance.

## Question and executable evidence

Do `tu_layernorm_2d` and `tu_rmsnorm_2d` preserve the requested `rows × cols` shape before constructing a descriptor and staging host gamma/beta? At the input revision the multiplication occurred in 32-bit unsigned arithmetic, and the wrapped count was used in the descriptor and scratch-overlap comparison. For example, `2147483649 × 2` wraps to two elements. On a 256-byte SRAM fixture both wrappers returned success and normalized only those two elements instead of rejecting the unrepresentable request. In some other overflow cases the wrappers returned an error only **after** staging gamma/beta, changing SRAM and meter counters.

The focused static-linked test compares five overflow shapes × two operators × parameters absent/present (20 cases), then checks 48 small valid rectangles against an independent double-precision numerical oracle. All input/parameter arrays in this experiment are bounded; invalid shapes use columns 2–4, and no invalid huge tensor is allocated. The complete baseline and fixed status/byte/meter matrix is in `evidence/2026-11-11-norm-wrapper-count/`. The original source fails 32 of 348 checks; fixed source passes all 348. Each invalid case checks rejection, all 256 SRAM bytes, and read/write/stall counters. These are API-meter values, **not physical execution cycles**.

## Alternatives and trade-offs

| Alternative | Motivation and applicable regime | Gains and sacrifices |
|---|---|---|
| Checked 32-bit descriptor count (implemented) | Existing ABI has a uint32 count; reject a product above its range before any parameter staging. | Avoids false success and side effects for this specific overflow. Adds a wide host multiply/comparison; valid small examples retain their arithmetic. Does not enable tensors whose count exceeds the ABI. |
| Wider-count/segmented execution | Large-tensor systems may divide work into representable tiles or adopt a wider descriptor. | Could support large logical shapes, but needs a defined tensor/SRAM extent, per-row offsets, parameter reuse, multi-command scheduling, statistics and error/retirement semantics. Expected more compiler/runtime and verification work; not implemented. |
| Unchecked wrap | No legitimate representation of the requested tensor under the current ABI. | Excluded as incorrect: silently substitutes another shape and can mutate SRAM even if later rejected. |

No throughput or latency gain is claimed: validation overhead was not benchmarked. Expected hardware comparator, width-conversion, tiling/control and buffer resources, physical area, power/energy, SRAM/DRAM traffic, and physical timing remain unquantified. For valid shapes this change does not intentionally alter data movement, numerical rounding, architecture defaults or configuration; invalid-shape side effects disappear. A wider/segmented path would change compiler/runtime responsibility and require additional functional, numerical, capacity, concurrency and error-path verification.

## Implemented boundary and unresolved risks

Both wrappers use the same uint64 product check and return `UINT64_MAX` before descriptor construction or scratch writes if the product exceeds `UINT32_MAX`. Zero/empty and representable products continue through the old descriptor path; the existing empty-shape test covers the no-parameter case. No mode, ABI or config knob was added for an erroneous wrapped shape.

This is **not** full wrapper validation: scratch-size/offset multiplication and subtraction can still overflow for other representable shapes, `data_sram` may be dereferenced before validation when parameters are supplied, parameter pointers and extents are host contracts, and partial data/output/parameter overlap needs a policy. RMS optional `mean_out` and legacy BatchNorm semantics are separate unresolved issues. Wider count alone does not establish physical SRAM capacity, or make a huge workload executable. Do not interpret this test as certification of these paths.

## Reproduction

```sh
make clean && make -j4
make test-norm-wrapper-count
make test-quick
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
```

For RED, compile the same focused test statically against the archived input revision's freshly built `libtucmodel.a`. To preserve unrelated checkout state, the baseline and final build/quick/full suite were run in isolated `git archive` trees with the task payload overlaid. The test was additionally exercised with all library objects and the test built with AddressSanitizer and UndefinedBehaviorSanitizer and leak detection enabled.
