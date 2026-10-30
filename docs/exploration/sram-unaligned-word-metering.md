# SRAM unaligned word metering: compatibility and address-span hypotheses

Date: 2026-10-30. Executed baseline: `b9bb04ca375b2e3b668b382ea656f5cafebdd9aa`.

## Question and reproducible observation

How should a byte-range access consume the existing per-bank grant meter when it starts inside a word and crosses a word boundary? For both scalar and bulk calls the compatibility implementation meters a word at each `addr + i * bank_width`, for `ceil(bytes/bank_width)` words. The bank index is `(address / bank_width) % bank_count`. Actual functional `memcpy` is byte-accurate and independent of metering. Therefore the existing metered bank set can differ from the banks containing the copied bytes.

The executable diagnostic `tests/explore_sram_unaligned_meter.c` uses 64-byte SRAM, four banks, 4-byte words, one grant per four-cycle window and 2-cycle stall penalty. Each of 80 rows starts with fresh state; read and write, scalar (4-byte) and bulk (0, 1, 2, 4, 5, 16, 17 bytes), offsets 0, 1, 3, 4 and 15 are exercised. It independently counts live compatibility events and calculates the distinct word addresses touched by the byte span. All copies, untouched output bytes, per-bank event counts, local stall totals and aggregate operation counts matched the executable compatibility contract; **22 of 80** span estimates differ from live meter events. Complete rows, not just selected examples, are in `evidence/2026-10-30-sram-unaligned/matrix.txt`.

| API, direction | Address | Bytes | Live compatibility bank counts [0,1,2,3] | Touched-word hypothesis [0,1,2,3] | Live stall cycles |
|---|---:|---:|---|---|---:|
| scalar read | 0 | 4 | [1,0,0,0] | [1,0,0,0] | 0 |
| scalar read | 1 | 4 | [1,0,0,0] | [1,1,0,0] | 0 |
| bulk read | 3 | 2 | [1,0,0,0] | [1,1,0,0] | 0 |
| bulk read | 1 | 16 | [1,1,1,1] | [2,1,1,1] | 0 |
| bulk write | 15 | 16 | [1,1,1,1] | [1,1,1,2] | 0 |

The second bank-count column is an **independent calculation**, not a second implemented or executed hardware mode. If that hypothetical word-span accounting used the present simultaneous grant meter, the 16-byte rows above would charge 2 stall cycles rather than 0. This counterfactual is arithmetic, not a measured end-to-end latency or throughput gain/loss. Scalar access length is always `bank_width`, not a separately specified element width.

## Alternatives and trade-offs

1. **Size-only compatibility (implemented):** treat the copy as request-sized beats anchored at the original address. This can be a useful abstract mover or compatibility accounting, potentially representing a width adapter behind an unspecified byte-stream interface. It is not evidence that the physically touched word banks were served, nor that crossing has no physical cost. It preserves existing counters and model cycle results.
2. **Address-span word accounting (proposed, not implemented):** count each aligned word intersecting `[addr, addr+bytes)` exactly once, bank it by that word address, and keep all accesses in the same existing model cycle. It can represent a word-interleaved SRAM interface with subword masking and split accesses; crossing increases bank activity and may produce extra modeled stall on wrapped banks. It does not determine real multi-port scheduling, read-modify-write behavior, or whether operations run in parallel.
3. **Hardware-dependent unaligned rejection or explicit split micro-operations:** plausible on specific interfaces but **blocked** on status propagation, fault/retirement/atomicity, port/replay, byte-mask and compiler alignment contracts. Do not silently reject valid accesses or prescribe a physical policy from the local meter alone.

**Throughput/latency:** only compatibility-path local stalls are measured. Span counts predict additional meter events, not physical completion times; present stalled calls copy immediately. **Area/resources:** alignment shifters, byte enables, extra bank ports or split state may be needed; values unquantified. **Power/energy:** extra word activity could cost dynamic energy; merging/byte masks may alter it; no calibrated estimate. **Traffic:** byte copies are identical for accepted accesses; word-bank activity differs; no physical SRAM bus or DRAM traffic measured. **Numerical accuracy:** copied bytes are unchanged; no arithmetic is performed. **Control/verification:** the span contract needs overflow-safe endpoint math, width-dependent bank mapping, all scalar/bulk read/write paths, zero length, boundaries, multiple turns and invalid-setter tests; it is not an arbitration implementation. **Compiler/runtime:** alignment can avoid splits; exposing mode and selecting safe model assumptions needs propagation and no-op/default tests. The existing scalar API copies a full word, so an access-size-aware ABI is a separate prerequisite for arbitrary-width engine access, not supplied here.

## Implementation status and next gate

This run **did not implement** the alternative. The candidate is READY for a separately verified implementation: preserve size-only as zero/default; expose address-span via canonical JSON/YAML/generated-header/runtime configuration and validate unsupported modes; use one overflow-safe meter shared by scalar/bulk read/write; preserve byte copies and bounds rejection; verify exact per-bank served/stalled counts and data for both alternatives, defaults, malformed config, multiple widths/bank wraps, no-op, and independent queued/live consumers. Confirm what the aggregate `reads`/`writes` counters mean and document the intentional opt-in difference. Audit direct runtime initialization and core snapshots rather than assuming JSON propagation alone. Compare each mode on the complete matrix and rerun clean builds and regressions. No claim is made that all consumers use this SRAM meter: raw-pointer and other bypasses need separate audit.

## Commands and evidence

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make clean && make && make test-quick
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel tests/explore_sram_unaligned_meter.c ./libtucmodel.a -lm -o "$TMPDIR/explore_sram_unaligned_meter"
"$TMPDIR/explore_sram_unaligned_meter"
cc -O1 -g -Wall -Wextra -Werror -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -I. -Itu_cmodel tests/explore_sram_unaligned_meter.c tu_cmodel/tu_sram.c tu_cmodel/tu_status.c -lm -o "$TMPDIR/explore_sram_unaligned_san"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 "$TMPDIR/explore_sram_unaligned_san"
```

The normal and instrumented diagnostic both completed **80 cases, 0 failures** on exercised paths. The initial baseline quick run without `LD_LIBRARY_PATH` failed to load `libtucmodel.so`; the full baseline rerun with the local library path passed. No numerical or physical-performance improvements are claimed.
