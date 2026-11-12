# Normalization SRAM penalty accounting

**Date:** 2026-11-12. **Baseline:** `fb8a3e7bd3a7dc1e961b7b321dbf6cc86d9b8635`.
This is a functional accounting correction, not a calibrated hardware performance study.

## Question and evidence

Does the normalization return value include the SRAM-local penalties produced by all the scalar accesses it actually executes? The baseline discarded every input, gamma and beta `tu_sram_read()` return while still accumulating the write returns. The underlying SRAM bank counters nevertheless recorded the read penalties. This made the documented total internally inconsistent, including when all accesses used a single SRAM region. It does not establish how real hardware schedules those accesses.

A statically linked independent serial-event oracle covers LayerNorm and RMSNorm; shared versus three separate regions; gamma/beta absent, either alone, or both (RMS ignores beta); in-place/out-of-place; one/four banks; and one/two/four grants per bank per four-cycle window. It models the exact executed access sequence, not a physical timeline. Each fixture has two rows of two FP32 values, starts with full grants, uses two penalty cycles per denied event, and does not advance time. All 192 combinations compare returned penalties, each region's bank-counter penalty and FP64-derived output. Two further controls verify disabled metering and exclude a pre-existing historical meter count. The unmodified baseline fails 110 of 1,542 assertions; the correction passes all 1,542. The full baseline and corrected 192-row matrices and legacy sweep are in `evidence/2026-11-12-norm-stall/`.

Representative out-of-place rows, 1 bank and 1 grant, in penalty cycles (not elapsed cycles):

| Mode | Parameter locations | Gamma/beta | Previous return | Corrected return | Data / gamma / beta region penalties |
|---|---|---|---:|---:|---|
| LayerNorm | separate | none | 8 | 14 | 14 / 0 / 0 |
| LayerNorm | separate | both | 8 | 18 | 14 / 2 / 2 |
| LayerNorm | shared | both | 8 | 22 | 22 / 0 / 0 |
| RMSNorm | separate | both | 8 | 16 | 14 / 2 / 0 |
| RMSNorm | shared | both | 8 | 18 | 18 / 0 / 0 |

The shared/separate rows deliberately change the resource mapping, so their difference is a property of this no-advance meter, not an experimentally established latency trade-off. One-bank, one-grant versus four-bank, four-grant without parameters yields 14 versus 0 for the same input and output in this fixture. Wider provisioning potentially reduces conflicts, but needs more ports/subbanks/routing and may increase area and energy; that physical cost is unquantified. Separate parameter storage may distribute accesses but adds memory instances, area, routing and control; shared storage can save resources but concentrates accesses. Neither is universally better. Data and coefficient reads may affect physical SRAM traffic; this fix changes no accesses or DRAM traffic, and accuracy is unchanged in tested finite examples.

The old 14-row `tests/test_norm_sweep.c` pre-fills SRAM by metered writes and never advances the meter. For 256 through 8192 elements both modes previously returned `2*N`; after the correction both return `4*N` local penalty cycles. This follows its already-exhausted budget and current sequential call sequence; the fixture does **not** establish a general 4-cycle normalization cost, a bandwidth, an execution latency, or an equal arithmetic cost. Historical pipeline percentage/fusion conclusions were already withdrawn and remain invalid.

## Contract, alternatives, and boundaries

`tu_norm_execute()` now sums returns from consumed gamma/beta reads, data reads, and data writes. RMSNorm continues to ignore beta. No mode, data representation, JSON/YAML setting, exported signature or SRAM meter behavior changes. This intentionally corrects the observable return for callers; code treating the previous return as a stable benchmark must update its baselines. The unchanged error sentinel is `UINT64_MAX`. Wrapper parameter *staging writes* happen before `tu_norm_execute()` and are not included; the wrappers' larger scratch-placement and alias contracts remain separate.

An aggregate sum represents the existing serial **software call accounting** contract; hardware may instead have parallel bank/port service, overlap or explicit replay and per-cycle arbitration. Such alternatives are valuable, but a `max` of arbitrary regional sums is not a valid parallel makespan model without event timing, dependencies, port identities and arbitration. They are blocked on those contracts, not excluded for performance. Reproducing the old omission as a new selectable hardware mode is excluded: it is a reporting defect, not a physically justified design.

No physical throughput, latency, area, power/energy, bandwidth, compiler overhead, or verification cost is quantified. Parallel engines may require independent ports, buffering, synchronization and routing; their energy and area direction depend on implementation. More explicit timing increases model/control complexity and verification combinations. A compiler/runtime must preserve parameter layouts and not equate local penalty sums to end-to-end schedules; no compiler interface was changed. The correction leaves numerical formats and operation ordering intact in tested cases. SRAM clock advancement, context state, contention, wrapper safety, RMS mean, BatchNorm semantics, partial overlaps and host error rollback are not certified here.

## Verification

```sh
make clean && make -j4
make test-quick
make test-norm-stall-accounting
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test-norm test-norm-descriptor-bounds test-norm-wrapper-count test-norm-sweep test-norm-attention-sweep
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" make test
```

The legacy norm targets link the shared library, hence the library path. The new gate links the archive explicitly. An isolated source archive rebuilt all objects with ASan/UBSan and ran the gate with leak detection: 1,542 checks passed. Baseline and final full matrices were generated by the same gate against archives differing in the normalization implementation. The baseline sweep uses the same source and build flags with the baseline normalization object, and the corrected sweep uses the rebuilt archive. These are exercised-path checks, not a general correctness proof.
