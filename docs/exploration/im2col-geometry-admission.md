# im2col geometry admission and coordinate representability

## Question / evidence (2026-10-20)

Can the public im2col dimension helper and streaming iterator safely represent
empty, strided, dilated, padded and large-coordinate convolution views without
fabricating output entries or wrapping source addresses? Baseline:
`95d7349c54b30e1d3fad95d1e8ba0bdfca64d59a`. No READY candidate existed at entry;
this is one bounded pre-spec correctness exploration, not an engine expansion.
The existing affine/tiled gates explicitly did not certify im2col.

Safe RED observations (no generated address was dereferenced):

- A 2x2 image / 3x3 kernel / no padding produced **1x1**, rather than an empty
  output. Unsigned subtraction and the forced-one fallback were both wrong.
- After the shape correction, a 65536x65536 / 1x1 view was accepted with an
  overflowed expanded count. The full regression rejected that acceptance.
- After admission correction, input H=2, padding H=`UINT32_MAX`, stride
  H=`UINT32_MAX`, and a 1x1 kernel still mapped a negative coordinate to real
  input through unsigned wrap. Wide coordinate arithmetic rejects that alias.
- Zero stride is a source-audited division-by-zero hazard, **not** a claimed
  observed crash. The baseline RED test stops safely before reaching it.

## Alternatives and resulting contract

| Choice | Rationale / decision |
|---|---|
| Closed producer guarantees / unchecked arithmetic | A private compiler pipeline could prove geometry upstream. The existing public untyped descriptor does not carry that proof. Undefined arithmetic, wrapped addressing and fictitious work are excluded as runtime alternatives. |
| Fail-closed iterator admission | Implemented through the existing status-returning API. Positive kernel/stride/dilation; bounded output axes/count and complete nonempty NCHW source extent. Rejection leaves no traversable entries. |
| Empty output | Implemented: an empty input axis or an oversized effective kernel makes that axis zero; zero channels yields zero entries. This is not an error fallback that invents a 1x1 operation. |
| Fault-bearing / partial execution | BLOCKED on status, signal, retirement and rollback semantics. Not emulated by continuing after wrap. |
| Materialized zero padding / tagged padding / host-prepadded storage | Physically plausible alternatives, but **not implemented here**. Current iterator padding is an untagged legacy sentinel, not an executable fill operation; a shared producer/consumer contract is required. |

`im2col_output_dims()` centralizes wide effective-kernel/padding arithmetic.
The public void helper writes only output dimensions; invalid controls or axes
that cannot fit uint32_t clear both outputs. Because zero also represents an
empty shape, callers needing status must use `tu_agen_iterator_init()`.

Iterator initialization checks the five-factor expanded entry count without
multiplication overflow and reuses the existing complete-byte-extent check for
nonempty source tensors. `UINT32_MAX` stays reserved for exhaustion. Exact
exclusive end `UINT32_MAX` is allowed, but that sentinel byte is not data.
No source-span check is needed when no entry can be generated; invalid control
fields and unrepresentable output axes still reject even on empty requests.

Coordinates are formed in uint64_t and checked before subtracting padding.
For a nonempty admitted view, output and kernel dimensions have a product at
most UINT32_MAX; their zero-based index sum is therefore bounded, so the two
index-times-stride/dilation products and their sum fit uint64_t. Source input
extent bounds the eventual NCHW byte-address computation.

Existing C runtime descriptor settings retain rectangular kernels, independent
H/W padding, strides and dilation, channels, base and element width. Size zero
still means 4 bytes. Order remains `oh,ow,kh,kw,ic`, over channel-first NCHW
storage; cached output fields are ignored/recomputed. No ABI, opcode, numerical
format, cycle formula, global default or JSON/YAML knob is added. Ordinary valid
views retain their sequences; malformed and formerly fictitious empty views
change deliberately and are explicitly regression-tested.

## Full measured shape matrix

Runtime API, base 13, 2-byte elements, input 4x5 spatial dimensions, no padding,
dilation 1. Entries and contiguous logical runs below are **address-iterator
observations**, not physical commands, cycles, SRAM/DRAM traffic or throughput.

| Channels | Kernel HxW | Stride H/W | Output HxW | Entries | Logical runs |
|---:|---|---|---|---:|---:|
| 1 | 1x1 | 1/1 | 4x5 | 20 | 1 |
| 1 | 1x1 | 2/2 | 2x3 | 6 | 6 |
| 1 | 3x3 | 1/1 | 2x3 | 54 | 18 |
| 1 | 3x3 | 2/2 | 1x2 | 18 | 6 |
| 2 | 1x1 | 1/1 | 4x5 | 40 | 40 |
| 2 | 1x1 | 2/2 | 2x3 | 12 | 12 |
| 2 | 3x3 | 1/1 | 2x3 | 108 | 108 |
| 2 | 3x3 | 2/2 | 1x2 | 36 | 36 |

The same 4x5x2 input, 2x3 kernel and H/W stride 1/2 additionally measures:

| H/W padding | H/W stride | H/W dilation | Output | Entries | Padding entries | Logical runs |
|---|---|---|---|---:|---:|---|
| 0/0 | 1/2 | 1/1 | 3x2 | 72 | 0 | 72 |
| 1/1 | 1/2 | 1/1 | 5x3 | 180 | 68 | not certified for ranges |
| 0/0 | 1/2 | 2/1 | 2x2 | 48 | 0 | 48 |
| 0/0 | 2/2 | 2/1 | 1x2 | 24 | 0 | 24 |

The channel-inner sequence on channel-first storage fragments even a 1x1 view
at two channels. That can be useful as a logical compute-feed order but does
not prove an efficient memory schedule. Changing traversal order would change
the consumer's expanded layout; no alternative order is silently substituted.
Stride/dilation alter the operator being represented, not a free optimization
for the same numerical workload. All existing valid choices remain available.

## Gain versus sacrifice / fidelity

| Dimension | Interpretation |
|---|---|
| Throughput / latency | Correct admission removes false work and wrapped coordinates. No new device cycle or throughput claim. Constant host checks and wider per-entry arithmetic may increase simulator cost; host speed is unmeasured. |
| Area / resources | Validation may be performed by software or by a physical AGU with bounds logic; wide host arithmetic does not prescribe a 64-bit hardware datapath. Checker/state area is unquantified. |
| Power / energy | Extra host comparisons/arithmetic, potentially fewer invalid allocations/operations downstream. Physical dynamic/leakage and net energy are unquantified and not wired into the power model. |
| SRAM / DRAM traffic | Repeated im2col entries and logical fragmentation are observable. Reuse, actual bursts, cache/line overfetch, banking, fill traffic, buffer placement and external DRAM bytes are unquantified. Materialization can require extra storage; streaming needs address/control/feed state. |
| Numerical accuracy | No floating-point arithmetic/conversion/rounding changes. Correct source selection is essential to results; this gate does not certify convolution arithmetic or padded zero-fill execution. |
| Control complexity | One checked shape producer feeds both helper and iterator; count/extent admission and padding-before-subtraction add checks. No queue, replay or fault protocol is invented. |
| Verification burden | Empty/error distinction, exact-end/count limits, shape/count/coordinate overflow, zero controls, asymmetric shapes, multi-channel order and deterministic replay require separate gates. |
| Compiler / runtime | Callers must honor init status and empty shapes. Existing C descriptors remain runtime-selectable. Alternate feed orders, prepadding and tagged fills need explicit shared layouts and consumer contracts. No compiler dispatch integration is claimed. |

## Verification / reproducibility

- Baseline and final `make clean && make`, then `make test-quick`: PASS.
- `make test-agen-im2col-bounds`: **32,924 cases**, **7,377 rejected**;
  **25,535** accepted bounded sequence/reset/bulk replay cases. Additional large
  accepted cases are admission-only; the exact UINT32_MAX-entry linear-shaped
  view samples first/last and stable exhaustion without a huge allocation.
- Test-only independent 128-bit shape/admission and signed-coordinate oracle;
  five-factor products are saturated after four factors to avoid even a
  test-oracle 128-bit overflow. Includes 4,096 deterministic adversarial mixes,
  six zero-control cases, NULL, default size, exact end and stale output fields.
- No invalid address is dereferenced. Range replay covers padding-free cases
  only. A partial bulk request and repeated exhaustion preserve state semantics.
- Config **31/31**, convolution **12/12**, scatter/gather **15/15**: PASS.
- Fresh all-object ASan/UBSan static archive, leak detection enabled: im2col,
  affine, tiled, iterator-chain and public range-admission gates PASS. These are
  exercised-path results, not certification of all address APIs.
- Broad `make test`: **not green**. Latest clean archived baseline is **7/9**
  and the final tree is **6/9**. Eight additional executions of the **unchanged
  baseline attention binary** range from **6/9 to 8/9**: identity tolerance,
  deterministic-small/causal errors (including infinity), and scale
  differentiation failures vary between executions. Thus a fixed failure set
  or numerical equality is not claimed; the baseline itself is unstable.
  Every repeat is retained, including the less-failing attempts. No
  attention-engine change or root-cause attribution is bundled.

```sh
make clean && make
make test-quick
make test-agen-im2col-bounds test-config test-conv test-scatter-gather
make test
```

The new focused target links `./libtucmodel.a` explicitly and is included in
quick/full/clean/ignore integration. Build, quick, focused, full, iterative RED,
archive-baseline and sanitizer logs plus command/input/checksum records are
retained in `evidence/2026-10-20-im2col-admission/`. The archive verifier omits
external symlink fixtures; these C gates do not consume them. The exact staged
snapshot is independently rebuilt and tested before publication.

## Audit boundary / remaining dependencies

SRAM `arb_mode` is still only assigned in initialization; named simultaneous
requester/port/replay semantics remain BLOCKED. Compression still estimates
dense reconstruction plus max/sum DMA overlap, not direct compressed-domain
compute; token/FIFO/consumer contracts remain BLOCKED. Physical WS/OS/RS
movement is not certified by im2col correctness. These audited candidates did
not acquire a READY implementation plan.

Padding still returns `0xFFFFFF00`, distinct from iterator exhaustion and from
the single-address helper's `UINT32_MAX` padding result. The sentinel can also
alias a representable source address; ranges do not carry a validity/fill tag.
**Do not treat padded iterator -> range -> DMA as verified zero-fill.** Tagged
entries, omitted DMA work plus a fill stage, and host-prepadding remain viable
alternatives, BLOCKED on a coherent layout/status/ownership contract rather
than rejected for being slower. The independent `tu_agen_addr_im2col()` helper,
block-interleaved mode, general tiling utility and arbitrary manual mutation of
iterator state are not certified by this patch.

**2026-10-21 follow-up:** the independent single-address helper is now checked
and separately gated; see [point-query admission](im2col-point-query-admission.md).
This does not extend certification to padded range-to-DMA zero-fill,
block-interleaved admission, general tiling utility or unchecked inline helpers.

The narrow reversible correction stays on `main`; no competing tracked edits
existed. User-owned untracked `TERMS.md` is preserved and excluded. No handoff
artifacts or build binaries are published.
