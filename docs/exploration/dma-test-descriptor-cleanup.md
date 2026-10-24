# DMA regression-harness descriptor cleanup

**Audit:** 2026-10-24. **Status:** DONE (test-harness lifetime only). **Baseline:** `f8fdd9674a077ef877f22b921083c06574a865fb`. The original 2026-10-23 finding was the sole READY backlog item; the small-backlog rule selects this one independent repair, not a new architecture feature.

## Reproduced defect and ownership evidence

A fresh unchanged-baseline archive, with **every library object** and the harness built using ASan/UBSan and leak detection, reproduces **3,136 bytes leaked in 14 descriptor allocations**. Ordinary baseline DMA tests pass 15/15. The final harness passes 15/15 under the same instrumentation with no sanitizer diagnostics. This is host fixture lifetime, not incorrect DMA payload bytes or a device-performance improvement.

Source tracing in `tu_cmodel/dma_descriptor.c` distinguishes:

- Synchronous submission removes executed queue heads without freeing them. The three-node fixture retains an intact detached chain.
- Async execution sets `completed` when copying at issue, while the active slot persists until the completion timestamp. Retirement then clears that pointer without freeing the descriptor. **`completed` is not an ownership/retirement test.**
- Rejected submission consumes and destroys the complete submitted chain; every caller alias must become invalid, not just the head.
- Engine destruction sees pending and active descriptors, not detached completed descriptors. Destruction traverses `next`, so separately freeing linked successors without severing links double-frees them.

The baseline leak stacks identify the linear, strided, chain, async, FP32, signal, statistics, outstanding and rotating-priority fixtures. Earlier retained evidence remains in `evidence/dma-drr-stage-width-2026-10-23/`; this audit independently reproduces the same totals at the current baseline.

## Alternatives and decision

| Alternative | Why it might be chosen | Gain | Sacrifice / disposition |
|---|---|---|---|
| Explicit detached-caller fixture cleanup | Preserve the existing API and query lifetime while making tests own their allocations | Eliminates the reproduced harness leaks; supports rejection/assertion/OOM paths | Extra host ownership bookkeeping and queue-membership scans; implemented only for the ordinary acyclic fixture graphs |
| Engine-owned retired list / automatic retirement free | A separately designed runtime may want centralized ownership | Could simplify some callers | Changes descriptor-query lifetime, retained storage and ownership contracts; not selected merely to green tests, and not implemented |
| Leak suppression | Sometimes used temporarily to triage unrelated diagnostics | Hides a report, not a leak | Excluded as an acceptance strategy; leak detection stays enabled |

No runtime architecture knob is appropriate for leaked test allocations. No production source/header, YAML/JSON setting, generated header, numerical format, public API or default architecture behavior changes.

## Implementation and failure paths

`tests/test_dma.c` adds narrowly scoped helpers:

1. `submit_owned()` clears a pointer consumed by rejection.
2. `destroy_detached()` checks active/pending membership before freeing a caller-owned node; it severs `next` before destruction. It deliberately leaves engine-owned work for engine teardown.
3. `cleanup_descriptors()` releases fixture arrays before engine destruction and clears their aliases.

Single cleanup labels now cover failed submission and the pre-tick-data assertion, with engine teardown before SRAM/host lifetime ends. Chain construction checks all allocations before linking and clears every alias after consuming rejection. Arbitration fixtures retain pointers to the previously lost retired descriptor. Statistics and outstanding fixtures release detached allocations; binding fixtures no longer dereference NULL/rejected descriptors during cleanup. Negative-span fixtures always execute allocation-consuming rejection calls even after an earlier failure, rather than short-circuiting them. NULL allocation is not accepted as evidence of real descriptor rejection.

The helpers are **not** a general ownership API or a validator for corrupted descriptor graphs. These fixtures do not put an active chain and its pending successors into a multiply owned teardown graph. Production ownership redesign, malformed graphs, arbitrary lifetime races and host-thread safety remain outside this repair.

## Verification and complete evidence

New `tests/dma_cleanup_faults.c` contains **test-only link wrappers**, and `tests/test_dma_cleanup.py` creates a fresh external source snapshot, builds every library object with instrumentation, statically links the harness, and retains input hashes, full logs, commands and statuses. Wrappers never fabricate an ID: injected submission errors change direction to an unsupported value and use the real admission/rejection/destruction path.

| Gate | Result |
|---|---|
| Baseline `make clean && make`; `LD_LIBRARY_PATH="$PWD" make test-quick` | PASS |
| Fresh baseline all-object sanitizer DMA run | Expected RED: 3,136 B / 14 allocations |
| Final clean build and quick suite | PASS |
| Final ordinary and all-object sanitized DMA | 15/15; no leak, double-free, UAF or UBSan report |
| Descriptor-object OOM, one allocation ordinal per process | 49/49 controls finish, report a failed test, and remain sanitizer-clean |
| Real consuming rejection, one submission ordinal per process | 45/45 controls finish sanitizer-clean: 33 cause newly detected test failure; 12 remain successful because rejection was already expected |
| Pre-tick data assertion failure with work still pending | Expected 14/15 exit 1; cleanup sanitizer-clean |
| Hold active after first issue, suppress later ticks | Expected 13/15 exit 1; active/pending teardown sanitizer-clean |
| Historical common DRR scan-width sweep | 874 timing rows / nine queue gates PASS, including fresh sanitizers |
| Independent DRR stage-width sweep | 8,425 timing rows / 162 queue/reconfiguration gates PASS, including fresh sanitizers |
| Config; scatter/gather; convolution | 31/31; 15/15; 12/12 PASS |
| Arbitration / channel / binding / bus-width sweeps | PASS |
| GCC `-fanalyzer -Wall -Wextra -Werror` on the changed C harness | PASS |

The complete baseline/final quick-suite numeric row streams are identical: **9,316 rows**, SHA-256 `a39c89c8995b23dacac0c961fde443b46401cea1e93ad6eb17347981fa8e5229`. This includes the full DRR matrices rather than just their fastest rows. `matrix-comparison.json` also proves the normal static/shared libraries and attention executable are byte-identical to the clean baseline build.

For continuity, the full nine-row representative stage-width matrix remains below. Eight queues, cursor 0, target 7, 32 B interface, 50-cycle base, useful-byte quantum 64 B, unit weights, one cycle per scan group, `per_visited_channel`, no burst-issue cost, SRAM bandwidth metering disabled. Entries are **uncalibrated model core cycles through retirement**, not ns or silicon timings; both directions remain byte-exact. The complete 8,425-row matrix is in the retained raw logs.

| Round width | Select width | 256 B, three failed rounds | 32 B, zero failed rounds |
|---:|---:|---:|---:|
| 1 | 1 | 91 | 60 |
| 1 | 3 | 86 | 55 |
| 1 | 8 | 84 | 53 |
| 3 | 1 | 76 | 60 |
| 3 | 3 | 71 | 55 |
| 3 | 8 | 69 | 53 |
| 8 | 1 | 70 | 60 |
| 8 | 3 | 65 | 55 |
| 8 | 8 | 63 | 53 |

These values are unchanged by cleanup. Round-wide/select-narrow helps repeated credit rounds; round-narrow/select-wide helps immediately eligible work. Wider costing widths retain their expected port/lane/control-resource sacrifices; equal physical cost and a universally best width are not established.

`make -k test` remains **non-green solely at attention** in both fresh baseline and final runs, each 7/9 but with different cases: baseline deterministic-small/causal infinite errors; final identity error 0.377808 and identical-output scale differentiation. The two attention executables and production libraries have identical hashes. This is a pre-existing variable numerical blocker, not fixed-set equivalence, diagnosed root cause, or an all-tests-pass claim. No attention code or assertion was changed.

Raw evidence is bundled under `evidence/2026-10-24-dma-cleanup/`. A first foreground fault campaign was interrupted by the tool's 420-second limit after partial coverage and is retained as **incomplete, not accepted**. The tracked rerun exited 0 and validated all 100 recorded build/test/control commands. Input and log hashes match the final sources. Fresh staged-snapshot build/quick/focused results are recorded separately in the bundle.

## Multi-objective limits

| Dimension | Result / limit |
|---|---|
| Throughput and device latency | No production timing change or speedup; all 9,316 compared numeric rows unchanged. Additional host cleanup scan cost is unquantified. |
| Area/resources | Test allocations have bounded lifetime; no physical area estimate follows. The alternative retired-list design would need separate storage/lifetime contracts. |
| Power/energy | Physical device power/energy unchanged by these source edits; no physical measurement. Reduced leaked host allocations do not establish a numerical energy saving. |
| SRAM/DRAM traffic | No production payload path changes; no extra modeled bytes or fabricated traffic improvement. Queue-membership scans are host test bookkeeping, not modeled TU SRAM traffic. |
| Numerical accuracy | Original byte/FP32 fixtures preserved; no arithmetic/rounding approximation. |
| Control complexity | More explicit caller-versus-engine cleanup state; no device scheduler-control change. |
| Verification burden | Adds reproducible all-object sanitizer and OOM/rejection/pending/active controls. Descriptor-object OOM does not certify SRAM or multicast-metadata allocation failure, corrupted graphs, or every engine caller. |
| Compiler/runtime | ABI, config/defaults and compiler output unchanged; production callers still need their actual lifetime contract. |

## Reproduction

From the repository root, with a **new, external** output directory:

```sh
make clean && make
LD_LIBRARY_PATH="$PWD" make test-quick
python3 tests/test_dma_cleanup.py --output "$TMPDIR/tusim-dma-cleanup-check"
```

The sanitizer campaign takes several minutes; use a sufficiently bounded tracked worker instead of assuming the platform's foreground wait permits the entire campaign. It writes only its selected external output directory. The normal checkout build remains untouched by that campaign. Compiler sanitizer support and GNU-compatible linker `--wrap` are prerequisites; unsupported toolchains fail rather than silently skipping verification.
