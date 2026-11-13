# Softmax overflow admission — checked uint64 endpoints and wrapper products

**Date:** 2026-11-13
**Status:** Implemented and gated (`make test-softmax-overflow`, in `test` and `test-quick`).
**Scope:** Functional/model correctness only — no hardware datapath, cycle, traffic, accuracy, area, power, or numerical-semantics change for valid shapes.

## Question

Do `uint32` endpoint and product computations in the softmax engine wrap before comparison, admitting fictitious work as success? A second audit question: is the softmax functional suite actually executable in the current build?

## Findings (baseline RED, 4/4 false successes)

Safe, statically linked probe (`SRAM` pre-admission rejects the accesses, so no OOB host access occurs — the engine only *reports* success):

| Case | Input | Baseline result | Root cause |
|---|---|---|---|
| data-offset wrap | `data_offset=0xFFFFFFF8, elem_count=2` | returns stall cycles (false success), row is uninitialized because reads are OOB-rejected | `data_offset + elem_count*4` wraps to 0 before the `> total_size` compare |
| out-offset wrap | `out_offset=0xFFFFFFFC, elem_count=1, !in_place` | false success, write silently rejected | same wrap in the output endpoint check |
| elem_count byte-extent wrap | `elem_count=0x40000000` (`*4 = 2^32`) | false success | same |
| 2D wrapper product wrap | `tu_softmax_2d/masked/log_softmax_2d(rows=0x10001, cols=0x10000)` | false success: normalizes the wrapped 65536-element prefix of a 4 Gi-element tensor | `rows * cols` wraps to 0x10000 in the wrapper; the wrapped count is divisible by `cols` and fits a 256 KiB region, so no downstream check catches it |

The wrapper case is the dangerous one: every existing descriptor check (divisibility, bounds) passes for the wrapped prefix, so the failure mode is a **silent prefix normalization of a huge tensor**, not an error.

Dormant-suite finding: `tests/test_softmax.c` (15 functional tests) had **no Makefile build rule**; `test-softmax` was a dead `.PHONY` name only referenced by `clean`. The suite existed but was not executable in any aggregate.

## Alternatives considered

- **Checked uint64 admission (implemented):** compute `data_end`/`out_end` as `uint64_t` in `tu_softmax_validate_desc`; reject `rows*cols > UINT32_MAX` in `tu_softmax_2d`, `tu_softmax_masked`, `tu_log_softmax_2d` before descriptor construction. Rejection happens before any SRAM access or partial write. Constant host-check cost.
- **Wider/segmented execution:** BLOCKED — needs new extent, scheduling, and error contracts; it is a capability question, not a validation fix.
- **Unchecked:** EXCLUDED as unsafe — silent prefix normalization and garbage output from OOB-rejected reads are not physically meaningful alternatives.

## Trade-offs / unquantified dimensions

The fix changes no bytes, stalls, or results for valid shapes (verified below). Validation adds unmeasured host-side compare work; in physical hardware an analogous span check costs comparator/address state (area/power/latency unquantified, direction: small constant). No physical throughput, latency, energy, SRAM/DRAM traffic, or accuracy claim is made — this is admission correctness only.

## Implementation

- `tu_cmodel/compute/softmax_engine.c`: 64-bit endpoint arithmetic in `tu_softmax_validate_desc` (plus corrected `%llu` diagnostics); shared `softmax_2d_count_ok()` guard in the three 2D wrappers.
- `tests/test_softmax_overflow.c`: 6 rejection rows (3 descriptor wrap modes, 3 wrapper product wraps) + valid controls (4×8 numerics with row-sum≈1 and monotonicity, out-of-place path, exact-fit full-region boundary, plain non-wrapping overflow).
- Makefile: revived `test-softmax` (statically linked per the focused-target stale-`.so` pitfall) and added it to `test`; added `test-softmax-overflow` to `test` and `test-quick` and `clean`; converted `test-softmax-sweep` / `test-softmax-attention-sweep` from `-L. -ltucmodel` (rpath-broken, needed `LD_LIBRARY_PATH`) to explicit `./libtucmodel.a` static linkage.

## Verification

```
make clean && make -j8                 # 0 errors/warnings in changed files
make test-softmax-overflow             # 11/11 PASS (baseline probe: 4/4 false success)
make test-softmax                      # 15/15 functional suite (revived)
make test-softmax-sweep                # table unchanged
make test-softmax-attention-sweep      # unchanged
make test-quick                        # pass
LD_LIBRARY_PATH=. make test            # all aggregates pass
```

ASan+UBSan builds of both softmax suites pass. The full aggregate still requires `LD_LIBRARY_PATH=.` for unrelated legacy shared-linked targets (documented 2026-11-07); the softmax targets no longer do.

## Remaining boundaries

`mask_fill`-as-threshold masking semantics (`row[i] <= mask_fill → -1e9`) is a separate semantic question, not touched. Mask pointer/count host extent, streaming/segmented execution, and command-level error propagation remain outside this gate.
