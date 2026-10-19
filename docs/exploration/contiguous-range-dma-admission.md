# Contiguous range → DMA descriptor admission

## Question and observed defect (2026-10-19)

Do the public single-range constructor and iterator-chain wrapper enforce the
same basic host-transfer contract? Baseline input is
`1cad91e8fc44a786adf5c67c1d552075f1861faf`. No READY candidate remained at entry,
so this run performed one bounded producer-admission exploration, rather than
adding a speculative engine or changing a hardware default.

The chain wrapper already rejects NULL SRAM/host, zero element width and
unsupported direction. The public `tu_agen_desc_from_range()` checked only the
range pointer and nonzero bytes, divided by `elem_size` without a guard, and
interpreted **every direction other than HOST_TO_TU as a host store**. For
fractional elements, its integer `dims[0]` did not describe `total_bytes`.
The retained safe RED test fails because TU_TO_TU is accepted as a host store;
it destroys that unsubmitted descriptor and never executes invalid movement.
Zero-width division is a source-audited undefined operation, not a claimed
observed crash. RED stops before reaching that vector.

## Alternatives and implementation

| Admission approach | Why it could exist / decision |
|---|---|
| Caller-only trust | A closed compiler-generated pipeline may guarantee valid descriptors and omit redundant hardware checks. The public C helper has no such type/precondition guarantee. Trust is **not** retained as an unsafe runtime mode. |
| Fail-closed host construction | Reject malformed requests before allocation; the only complete current API contract. Existing NULL return supports failure without a new ABI, timing selector or numerical policy. Implemented. |
| Construct then executor-reject | Still appropriate for SRAM span/channel admission, which depends on execution/submission context. It cannot make constructor-side division by zero safe. Preserved for validly shaped but out-of-SRAM ranges. |
| Partial copy or fault-bearing descriptor | Plausible for a real faulting controller, but BLOCKED on explicit fault/status, signaling, retirement and rollback semantics. Not emulated by silent truncation or misrouting. |

The public constructor now requires non-NULL SRAM, host pointer and range,
positive byte count and element width, integral element count, and exactly
HOST_TO_TU or TU_TO_HOST direction. Rejection returns NULL **before allocation**.
Valid calls still allocate one caller-owned, unsubmitted LINEAR descriptor.

`host_ptr` points to the first byte, unlike the chain helper's host-layout
origin. The range's stride field does not request strided movement here;
strided transfers use the separate descriptor API. No new stride support is
claimed. SRAM region capacity is **not** checked at construction: downstream
span validation remains in force. Host allocation size is still caller-owned.
No change to public structs, generated config, JSON/YAML defaults or cycle
accounting is required; undefined malformed requests are not an architecture
alternative worth exposing as a knob.

## Complete exercised movement matrix

Each cell is a separately executed 16-byte transfer in a 64-byte SRAM and host
buffer. All buffer bytes, including unmodified surroundings, are checked.
Base 1 deliberately permits byte-misaligned data; base 48 ends exactly at 64.
Widths are descriptor element metadata, not new numerical formats.

| Direction | Element width (bytes) | Bases tested (bytes) | Elements / transfer | Exact movement |
|---|---:|---|---:|---|
| Host → TU | 1 | 0, 1, 8, 48 | 16 | all pass |
| Host → TU | 2 | 0, 1, 8, 48 | 8 | all pass |
| Host → TU | 4 | 0, 1, 8, 48 | 4 | all pass |
| Host → TU | 8 | 0, 1, 8, 48 | 2 | all pass |
| TU → host | 1 | 0, 1, 8, 48 | 16 | all pass |
| TU → host | 2 | 0, 1, 8, 48 | 8 | all pass |
| TU → host | 4 | 0, 1, 8, 48 | 4 | all pass |
| TU → host | 8 | 0, 1, 8, 48 | 2 | all pass |

Additional gates cover TU_TO_TU/unknown directions, NULL required pointers,
zero width/bytes, 15 bytes with width 4, width 32 exceeding a 16-byte range,
allocation failure in both directions, and constructor-success/executor-reject
for `[60,76)` against a 64-byte SRAM. Invalid construction neither allocates nor
copies; out-of-region execution neither copies nor accounts nor completes.
Valid construction does not change DMA byte counters; execution adds exactly 16.

## Multi-objective interpretation

| Dimension | Gain, sacrifice, and fidelity limit |
|---|---|
| Throughput / latency | Eliminates malformed construction and its undefined arithmetic/misrouting. Valid transfer timing is unchanged. Adds constant host validation; no host benchmark or calibrated physical issue-rate/fault-latency claim. |
| Area / resources | No modeled datapath/storage change. A physical checker may need direction/range/width comparators and status state; rejecting non-power-of-two fractional widths can require more than a mask. Hardware area is unquantified. |
| Power / energy | Host checks add work but reject invalid allocations. Physical check/control energy and net energy are unquantified; no automatic power-model coupling. |
| SRAM / DRAM traffic | All valid byte sequences are preserved. Tests gate useful bytes and full buffers; no physical DRAM burst, cache, mask or bus-traffic reduction is inferred. |
| Numerical accuracy | Raw copy only; no rounding, conversion or arithmetic semantics change. Integral metadata prevents fractional-element inconsistency. |
| Control / verification | One shared public entry-point guard avoids duplicate mechanisms. Adds allocation-order, no-copy/no-accounting, direction, OOM and metadata regression obligations. |
| Compiler / runtime | Callers must handle NULL. Existing valid construction/chain/ownership and downstream capacity checking remain unchanged. Malformed formerly accepted calls must fix their inputs, not rely on a silent store fallback. No compiler-dispatch integration is newly claimed. |

## Verification and retained evidence

- Baseline and final `make clean && make`, followed by `make test-quick`: PASS.
- `make test-agen-range-admission`: **32 movement rows, 4,452 assertions**, plus
  the rejection/OOM/executor boundary cases above: PASS. Static archive linkage
  prevents an older shared library from satisfying this focused target.
- Config **31/31**, convolution **12/12**, scatter/gather **15/15**: PASS.
- Fresh consistently ASan/UBSan-built archive: range admission, iterator-chain
  (including **514** injected allocation failures), affine **18,042** cases and
  tiled **17,069** cases pass with leak detection. Only exercised paths certified.
- Broad `make test`: **not green**. Both baseline clean archive and final live
  tree stop at attention **7/9**, deterministic-small and causal-mask tests
  reporting `max_err=inf`. No attention fix or new engine is bundled.
- The exact staged tree is separately archived, rebuilt and run through quick
  plus config/convolution/scatter-gather gates before publication.

Reproduce functional gates:

```sh
make clean && make
make test-quick
make test-agen-range-admission test-agen-dma-chain
make test-config test-conv test-scatter-gather
make test
```

For sanitizer testing, use a fresh archive and build **all** objects with:

```sh
make libtucmodel.a CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer'
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 make test-agen-range-admission test-agen-dma-chain test-agen-affine-bounds test-agen-tiled-sweep CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-lm -fsanitize=address,undefined'
```

Raw RED/GREEN, compressed build/quick/focused/full/sanitizer logs and command
records are retained under `evidence/2026-10-19-range-admission/`. The verification
runner archives committed baseline inputs and overlays only the named modified
implementation/build/test paths for the sanitizer run. External symlink fixtures
are omitted from disposable archives; none is consumed by these C engine gates.
Input and log checksums bind the exact tested sources and retained transcripts.

## Backlog audit / scope

SRAM `arb_mode` is still assigned but not consumed as a named-requester grant
policy in `tu_sram.c`; simultaneous-port/replay semantics remain BLOCKED.
Compression timing still models dense reconstruction plus max/sum DMA overlap,
not direct compressed-domain compute; token/FIFO/consumer contracts remain
BLOCKED. Physical WS/OS/RS movement is not certified by this constructor gate.
No slower plausible architectural alternative was removed or new READY item
established in these audited candidates. IM2COL, block-interleaving and inline
address helpers remain explicitly outside the previous affine admission gate;
this run does not certify their geometry. Existing attention failures remain an
engine-hardening issue, not an authorized new-engine implementation task.

Changed files are `address_generator.c/.h`, the new focused test, build/ignore
integration (including the existing unignored `test-elementwise` binary), this
note and the backlog/evidence. Untracked user-owned `TERMS.md` stays untouched
and excluded. Integration branch `main` is retained for this narrow reversible
correction; no competing tracked edits existed at entry.
