# SRAM bounds admission: precheck versus partial faults

Date: 2026-10-29. Baseline: `0b8d82110e74f688ad6e2ffc667826100dfb8f39`.

## Question and observations

Does SRAM's existing overflow reporting actually prevent invalid data movement,
including 32-bit address wrap? `tu_sram_read`, `write`, `read_bulk`, and
`write_bulk` called a void `bounds_check`, which reported an error but returned
only to its caller; each access then proceeded with metering and `memcpy`.
Moreover `addr + size` was evaluated in uint32 arithmetic. This is a correctness
and memory-safety issue, not an architecture-speed sweep.

The diagnostic allocates 32 host bytes but deliberately advertises a 16-byte
region, so the invalid addresses below remain inside the allocation. It uses four
4-byte banks, one grant per four-cycle refill window and a 2-cycle stall penalty;
each case starts with fresh state. The complete matrix is 2 access shapes × 2
read/write directions × 3 offsets, each with length 4:

| API | Direction | Offset 12, exact end | Offset 14, crossing end | Offset 16, starts at end |
|---|---|---|---|---|
| scalar | read | move; no error; 1 read | **baseline:** move, error 21, 1 read; **fixed:** no move, error 21, 0 reads | same as crossing |
| scalar | write | move; no error; 1 write | **baseline:** move, error 21, 1 write; **fixed:** no move, error 21, 0 writes | same as crossing |
| bulk | read | move; no error; 1 read | **baseline:** move, error 21, 1 read; **fixed:** no move, error 21, 0 reads | same as crossing |
| bulk | write | move; no error; 1 write | **baseline:** move, error 21, 1 write; **fixed:** no move, error 21, 0 writes | same as crossing |

Every row returned zero stall; this return is a penalty, **not** status. A
separate non-dereferencing arithmetic control shows `UINT32_MAX-1 + 4` wraps to
2 in uint32 arithmetic, so the old comparison against capacity 16 incorrectly
accepts it. The full fixed gate exercises this large address safely after the
precheck; the old gate deliberately excludes it.

## Contract and alternatives

- **Precheck, implemented:** compare `addr <= capacity` and `size <= capacity -
  addr`, report `TU_ERR_SRAM_OVERFLOW`, and return before meter/copy. Valid
  scalar/bulk moves and their existing cycle accounting are unchanged. A zero-byte
  bulk span at or within the endpoint is a no-op; beyond it is rejected.
- **Partial, beat-by-beat fault:** physically plausible, potentially useful for
  streaming or protection hardware, but **blocked** on explicitly specified
  completed prefix, returned status, caller retirement, counter, retry and rollback
  semantics. It is not excluded for its speed.
- **Unchecked/wrapping memory access:** excluded as unsafe under the current
  region contract, not a selectable performance mode.

The function ABI is retained: a rejected access returns **0 stall cycles**, and
sets the existing last-error object (or aborts in error-abort mode). This avoids
inventing an incompatible sentinel that existing callers could sum as cycles,
but it is not full error propagation. Many compute/cluster callers add or ignore
stall values without checking last-error; upstream calls may continue or report
success after a rejected access, and earlier work is not rolled back. The
existing last-error implementation uses process-global state despite its header's
thread-local comment. Do not claim atomic commands, end-to-end failure reporting,
thread safety or general API safety from this fix. Null/invalid pointers, failed
allocations, object lifetime, host-buffer extents and raw pointer access remain
separate contracts. A status-bearing API and audited producer/consumer paths are
needed before claiming command-level rejection.

## Multi-objective trade-offs

- **Throughput/latency:** valid model cycles do not change; precheck adds
  unbenchmarked host work. Hardware comparator/control delay and fault timing are
  unquantified. No throughput improvement is claimed.
- **Area/resources and energy:** expected range/carry comparators, error handling
  and optional fault-state logic differ by implementation; physical area, power and
  energy are unquantified. Partial-fault handling additionally needs state.
- **SRAM/DRAM traffic:** valid tested bytes and counters remain the same;
  rejected accesses no longer perform a copy or consume a bank budget. This is
  not a measurement of physical DRAM traffic or energy.
- **Numerics:** valid bytes are exact and arithmetic/rounding is unchanged; invalid
  accesses no longer read or write unintended bytes.
- **Control/verification:** subtraction guards integer overflow and avoids partial
  local side effects; error modes, upstream propagation, pointer validity and
  multi-operation atomicity need separate end-to-end verification.
- **Compiler/runtime:** validate tensor/region extents, clear and inspect the
  existing error state where appropriate, and do not treat zero stall as success.
  A status API requires coordinated consumers and compatibility design.

## Verification and reproduction

Baseline clean build and `LD_LIBRARY_PATH=$PWD make test-quick` passed. The new
safe regression against the baseline SRAM object produced **32 failures / 195
checks**, without touching outside its host allocation. The fixed full regression
passed **279 checks**, including both directions/APIs, crossing and exact end,
UINT32 wrapping, zero-byte endpoints, full backing/output immutability and per-
bank accounting. Directly instrumented SRAM and status sources with
ASan/UBSan/LSan passed this gate; coverage is limited to these paths. Clean
build, quick suite, full aggregate, SRAM cadence sweep, memory hierarchy,
multicore, attention, convolution and scatter/gather passed. No runtime option,
JSON/YAML setting, struct layout or public function signature changed.

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make clean && make
make test-sram-bounds test-quick test
make test-sram-issue-sweep test-memhier test-multicore test-attention test-conv test-scatter-gather
cc -O2 -Wall -Wextra -std=c11 -I. -Itu_cmodel tests/explore_sram_bounds.c ./libtucmodel.a -lm -o "$TMPDIR/explore-sram-bounds"
"$TMPDIR/explore-sram-bounds"
cc -O1 -g -Wall -Wextra -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -I. -Itu_cmodel tests/test_sram_bounds.c tu_cmodel/tu_sram.c tu_cmodel/tu_status.c -lm -o "$TMPDIR/sram-bounds-sanitized"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 "$TMPDIR/sram-bounds-sanitized"
```

The baseline and fixed diagnostic outputs and the safe RED/full GREEN summaries
are retained in `evidence/2026-10-29-sram-bounds/`. The diagnostic observes the
historical defect; it is not a regression gate requiring that defect to persist.
