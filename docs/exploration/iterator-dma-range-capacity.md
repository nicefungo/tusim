# Iterator-to-DMA range capacity and allocation atomicity

## Question and baseline (2026-10-17)

Can the existing iterator-to-descriptor convenience API construct a complete
transfer when an address traversal needs more than 128 contiguous ranges?
Baseline: `0c11194849320e000beba4510e23007172167b32`.

`tu_agen_generate_ranges` returns the **required** count while storing only the
bounded prefix. The former consumer allocated 128 entries on its stack but
iterated to that required count. A transposed 16×16, four-byte tensor requires
256 ranges; this was an out-of-bounds consumer, not an inadequate producer.
A second defect skipped failed descriptor allocations and returned a partial
chain as success. RED gates rejected the baseline with an invalid descriptor
span and a non-NULL allocation-failure result respectively. The historical
unsafe baseline capacity call is not a recommended reproduction command;
retained RED logs document it, and source inspection establishes the overread.

## Implemented contract

- Keep the existing public API, struct layout, iterator modes, range-count
  semantics, host offsets and DMA timing. No new JSON/YAML knob is introduced:
  this is host construction correctness, not a new hardware architecture.
- Count using one scratch range, allocate the complete range array with an
  overflow check, reset/replay through the same producer, then construct one
  descriptor per range. Temporary storage is freed before returning the chain.
- On any allocation failure, free the whole partial chain and array and return
  NULL. Construction never submits work, copies bytes or returns partial success.
  Iterator progress is not rolled back; NULL still conflates empty and failure
  because the public pointer-only API is unchanged.
- Reject NULL iterator/SRAM/host, zero element width, and directions other than
  host→TU or TU→host before traversal/allocation. An internal SRAM-to-SRAM
  transfer needs a separate endpoint contract, not this host-pointer helper.
- `host_base` is the origin of the same byte-addressed layout as SRAM;
  descriptor host pointers remain `host_base + range.base_addr`, including a
  nonzero iterator base. It is **not** a packed transpose/output buffer.
- Caller owns an unsubmitted successful chain and destroys it once with
  `tu_dma_desc_destroy(head)`. This gate executes each descriptor directly;
  it does not establish new async queue/chaining/retirement semantics.
- SRAM capacity validation remains in the DMA consumer. Caller must supply
  sufficiently large host storage; its allocation extent is absent from this
  API. im2col padding-sentinel fill semantics are not supplied by this change.

## Alternatives and multi-objective trade-offs

| Alternative | Why choose it | Gain | Sacrifice / disposition |
|---|---|---|---|
| Complete two-pass range array | Existing replayable producer, simple audit and complete admission artifact | Supports fragmented tensors without a fixed 128-range ceiling; reuses one producer | Two address passes, O(R) temporary storage plus R descriptors; implemented |
| Streaming chain construction | Very fragmented workloads or constrained host temporary storage | Could eliminate the temporary range array and one replay | Needs an incremental range/iterator interface or a refactor of producer grouping; deferred as a host implementation alternative, not a missing TU mode |
| Explicit bounded atomic rejection | Fixed descriptor store or software resource budget | Predictable maximum resource use | Rejects otherwise valid layouts; a public capacity/status contract is needed before adding a hardware admission mode; blocked, not rejected for speed |
| Silent truncation / unchecked fixed buffer | No valid complete-transfer rationale | None under this API contract | Excluded: data loss / memory unsafety, never an optional faster mode |

This change makes no physical-throughput or latency claim. Host construction
has an extra pass and allocation; those costs are not timed here. Device latency
and steady-state throughput formulas are unchanged for the same valid chain.
Host resource cost increases with R; physical descriptor-store/AGU area is
**unquantified**, because host allocations are not a hardware sizing model.
Power/energy are **unquantified**: more host allocation/replay activity is
expected, but no physical switching/leakage model is attached. Useful SRAM and
host payload bytes are exact; physical DRAM bursts, overfetch, queues and
contention are not measured. Numerical accuracy is unchanged (byte copies,
no conversion). Control complexity adds count/replay and all-or-nothing cleanup;
verification adds capacity, allocation-position and direction axes. Runtime
must preserve layout-sized host storage and free complete chains; compiler
emission and hardware descriptor-capacity negotiation remain outside scope.
No locally fastest traversal or hardware alternative is removed.

## Complete capacity matrix

Four-byte elements; nonempty workloads use byte base 12. Each row runs in
**both directions**, checks every descriptor's independent expected address,
host offset, exact count/bytes, direct completion, complete output and guards.
The iterator is exhausted by the bounded-prefix probe before chain creation,
so successful construction also gates reset/replay.

| Layout | Shape | Ranges | Useful bytes / direction |
|---|---|---:|---:|
| Linear | empty | 0 | 0 |
| Linear | 1×4 | 1 | 16 |
| Transposed | 2×2 | 4 | 16 |
| Transposed | 2×64 | 128 | 512 |
| Transposed | 16×16 | 256 | 1024 |
| Transposed | 3×129 | 387 | 1548 |

All 12 rows PASS. The complete 16×16 construction has one range-array and
256 descriptor allocations. Inject failure at every one of these 257 positions
in both directions: **514 cases PASS**, always NULL, no host/SRAM mutation and
no DMA byte accounting. ASan/UBSan with leak detection pass these exercised
paths using a separately fully instrumented library. Six invalid-input guards
also pass before allocation or iterator advancement. This does not certify
unexercised library paths or physical hardware correctness.

## Verification and evidence

Executed commands:

```sh
make clean && make
make test-quick
make test-agen-dma-chain test-agen test-agen-tiled-sweep test-scatter-gather test-config test-conv
make test
```

Baseline and final clean builds/quick suites PASS. A fresh archive of the staged
source/test/build snapshot also passes build, quick, scatter/gather, config and
convolution gates, excluding the untracked user file; `snapshot.log` retains
that execution. Focused results: address
14/14, compact tiled sweep 17,069 cases, scatter/gather 15/15, configuration
31/31, convolution 12/12. New chain target is statically linked and integrated
into quick/full/clean/ignore paths; its test-only `--wrap=calloc` injector does
not change production allocation APIs.

Sanitizer reproduction in a fresh source archive/copy (never mix instrumented
objects with a normal shared library):

```sh
make libtucmodel.a CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer'
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 make test-agen-dma-chain CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-lm -fsanitize=address,undefined'
```

The full aggregate still stops at attention **7/9**, with deterministic-small
and causal-mask `max_err=inf`. A separately clean archive of baseline `0c11194`
reproduces exactly those two failures; no full-suite-green claim is made.
Attention hardening is not silently bundled into this memory-safety item.

Raw baseline/final build, quick, focused, RED, sanitizer and broad-suite logs
are retained under `evidence/iterator-dma-range-capacity/`; `SHA256SUMS` binds
those logs and `INPUT_SHA256SUMS` binds the changed source/test/build inputs.

## Backlog audit

Completed the one READY item, following the small-backlog one-item/day rule.
SRAM `arb_mode` remains stored but not read by grant arbitration; requester,
port, simultaneous-admission and replay contracts are still prerequisites.
Physical dataflow movement and direct compressed-domain feed retain their
movement/overlap and token/FIFO dependencies. No unrelated engine feature or
speculative knob was added. Untracked user `TERMS.md` is preserved/excluded.
