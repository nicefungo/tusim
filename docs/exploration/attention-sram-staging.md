# Attention SRAM staging: exact element widths and live scratch

**2026-10-26 follow-up:** the cross-KV defect below is corrected by
[`attention-kv-normalization.md`](attention-kv-normalization.md). The historical
exit-1 `--kv-probe` observation applies to staging checkpoint `d2a2194`; the
current command is a passing exact-0.25 regression. The current Python runner
also verifies cross-KV behavior and eight mutants. Original 2026-10-25 evidence
and its then-unresolved scope remain unchanged below. Auto-tiler feasibility is
now a separately reproduced READY item, not fixed by staging or normalization.

**Audit:** 2026-10-25. **Mode:** pre-spec exploration; one new question after the prior backlog had no READY item. **Baseline:** `025a947` (full OID in evidence). **Status:** staging correction DONE; cross-KV normalization remains a separate READY correctness item.

## Question and reproduced causes

Can the existing FP16 attention engine preserve finite numerical results and SRAM bounds across query-tile sizes and WS/OS/RS without inventing an alignment margin? The motivating evidence was the variable attention failure in the broad aggregate, plus the historical `attention-engine-sweep.md` boundary report. Baseline clean build and quick suite pass, but its attention suite fails 7/9 (deterministic-small and causal infinite errors).

The all-object ASan/UBSan smoke probe locates a **four-byte write into the two-byte `fp16_t val`** in `transpose_fp16_in_sram()`: scalar SRAM accesses copy `bank_width=4`, not the C object's size. The transpose writes also read four bytes from that two-byte object; probability packing has the same over-read. This is an actual host memory-safety defect, not solely a noisy timing estimate. Normal-run variability is consistent with that UB; it does not establish that every historical attention failure had exactly the same cause.

Two independent live-data hazards remain after correcting element widths:

1. Shrinking FP32 to FP16 **backwards** overwrites lower-index FP32 values before reading them. Forward traversal is safe: destination `[2i,2i+2)` cannot reach the next unread source at `4(i+1)`.
2. Placing O after **packed FP16 P** overlaps O with the larger **live FP32 S** before packing. O stays live while scores are calculated/normalized, so it must follow the full score region. Mask scratch is separately reserved after O.

A fresh final-harness baseline numerical-only run fails 59,058 of 66,960 checks. The final implementation passes the full expanded gate; independently restoring each of the three bugs makes it fail again. Do not use the baseline UB-sensitive failure count as a reproducible fingerprint of all incorrect output values.

## Implementation and compatibility

- `tu_cmodel/compute/attention_engine.c`: explicit-sized bulk accesses for every FP16 transpose/packing object; forward in-place packing; disjoint FP32 S / FP32 O / FP32 mask storage; wide element-product checks before byte narrowing; documented descriptor validation before execution.
- Correct O-buffer admission is `2 * (tile_m * tile_n * 4) + tile_m * head_dim * 4` bytes. The score region also holds the smaller packed P. Mask storage is conservatively reserved even when masking is disabled, preserving the existing always-reserved-mask strategy.
- Configured footprints reject **before movement or stats mutation** if they do not fit. The exact 65,536-byte O-buffer control succeeds; a 66,048-byte control rejects. An exact 1,024-byte K+KT W-buffer control succeeds with **no headroom**.
- Existing descriptor tile sizes and global WS/OS/RS selection remain runtime alternatives. No new YAML/JSON field, struct layout, public signature, default dataflow, canonical converter or rounding policy is introduced. Binary callers still see the same ABI. Valid outputs change where the old staging was corrupt; former overlapping scratch configurations may now reject. Undefined/corrupt behavior is not a compatibility mode.
- Both attention gates link the static archive explicitly. The new staging gate belongs to quick/full/clean/gitignore integration. The legacy scale-differentiation assertion now rejects nonfinite error rather than accepting infinity as a difference.

## Realistic alternatives and measured trade-offs

Small query tiles reduce live SRAM demand and can fit resource-constrained implementations, but reread K/V more often. Larger query tiles reuse K/V across more queries, at the cost of more live scratch and expected SRAM resources. WS can represent stationary-weight systolic organization, OS stationary output/control, and RS another reuse schedule, but current functional plug-ins and estimate formulas do **not** quantify physical movement, ports, area or energy. All remain selectable; lower estimate is not a universal hardware winner.

Representative **single-KV-tile** workload: two heads, Q length 5, KV length 8, head dimension 8, scale 0.5, no mask, default 16×16 PE / 256 KiB SRAM, FP16 Q/K/V/P/output and FP32 scores/accumulation. Inputs are bounded deterministic dyadic values. Logical DMA bytes count Q/K/V loads and output payload; these are not DRAM-channel occupancy or a measured external-memory trace. Scratch is per current query tile, not multiplied by sequential heads. The full 720-row matrix is retained in the evidence logs.

| Query tile | Active dataflow | Reserved O SRAM B | Logical DMA B | Compute counter | DMA counter | Max absolute error |
|---:|---|---:|---:|---:|---:|---:|
| 1 | WS | 96 | 2,880 | 3,656 | 288 | 0.000106917133 |
| 3 | WS | 288 | 1,344 | 1,832 | 240 | 0.000106917133 |
| 5 | WS | 480 | 832 | 1,256 | 256 | 0.000106917133 |
| 1 | OS | 96 | 2,880 | 3,336 | 288 | 0.000106917133 |
| 3 | OS | 288 | 1,344 | 1,680 | 240 | 0.000106917133 |
| 5 | OS | 480 | 832 | 1,160 | 256 | 0.000106917133 |
| 1 | RS | 96 | 2,880 | 3,496 | 288 | 0.000106917133 |
| 3 | RS | 288 | 1,344 | 1,756 | 240 | 0.000106917133 |
| 5 | RS | 480 | 832 | 1,208 | 256 | 0.000106917133 |

The two counter columns preserve their module labels for reproducibility. They mix scalar SRAM-stall and MMA estimate accounting, and **must not be added/interpreted as calibrated latency, utilization, throughput or speedup**. The traffic/resource comparison is meaningful: query tile 5 reduces logical DMA from 2,880 to 832 bytes while increasing reserved O scratch from 96 to 480 bytes. No physical area/energy ratio is inferred from either number.

| Objective | Gain / sacrifice / fidelity limit |
|---|---|
| Throughput / latency | Removes UB and numerical corruption. Smaller/larger query tiles retain workload-dependent reuse/working-set choices. Physical timings and throughput are unquantified; old corrupt baselines cannot support speedup claims. |
| Area / resources | Correct scratch reservation is larger than the overlapping fiction. Wider tiles plausibly require more live storage/ports; phase-reused storage would need a different proven liveness schedule. Physical area is unquantified. |
| Power / energy | Fewer logical K/V loads can reduce transfer activity, while larger live SRAM and wider control may cost leakage/switching. No net energy or power measurement/model is established. |
| SRAM / DRAM traffic | Sized FP16 accesses stop unintended neighboring writes. Logical DMA counts are executable; external DRAM traffic, burst occupancy, NoC movement and physical bank scheduling remain unquantified. |
| Accuracy | Bounded single-KV-tile maximum absolute error is 0.000256468932 over the full matrix; uniform power-of-two-key controls require exact `0x3400` output. P is still quantized to FP16; no general FP32-equivalence or format-conformance claim. |
| Control | Explicit element widths, forward packing, lifetime-correct offsets and fail-closed geometry; no new architecture-mode explosion. Physical width adapters/masks are not modeled. |
| Verification | Adds an independent FP64 attention oracle, finite checks, exact uniforms, active-plugin checks, output canaries, admission immutability, full instrumentation and three independently rejected mutations. Allocation failures and arbitrary descriptors are not certified. |
| Compiler / runtime | ABI/config/defaults unchanged; tile selection must respect the larger honest footprint. Internal SRAM output offset is not a public pipeline boundary. Cross-engine users need an explicit output-layout contract, not offset 0 assumptions. |

A separate P region or phase-reused S/O layout is not needed to correct the current algorithm. A new physical buffering mode would need explicit storage lifetime/port/movement evidence before being promoted; the current overlapping layout is excluded because it destroys live data, not because it is slower.

## Verification

`tests/test_attention_staging.c` covers N=1/2/3/4/5/8/17/33, d=1/3/8/17, query tiles 1/3/5, two heads, active WS/OS/RS, no-mask/causal, odd counts, query tails and exact uniform controls. The independent oracle computes dot/softmax/output in double precision. The 0.002 absolute bound is only for these bounded inputs and the FP16 probability/output path; unexpected NaN/Inf fails first.

- Final matrix: **720 numerical cases, 67,034 total checks, zero failures**, including admission and exact-tail controls.
- Baseline ASan: expected RED stack-buffer-overflow. Final **every-object** ASan/UBSan/LSan: matrix and legacy attention pass, no sanitizer diagnostics.
- Isolated mutants restoring word-width copies, reverse packing, or packed-P O placement are all rejected. Word-width rejection is ASan; the other two fail numerical/admission assertions without sanitizer errors.
- Legacy attention: **9/9**. Final clean build, quick suite and **`make -k test` full aggregate pass**. Config 31/31, convolution 12/12, scatter/gather 15/15, softmax 11/11, norm, global dataflow sweep and shared DMA arbitration also pass. These gates do not certify missing cross-KV behavior.
- Fresh staged-snapshot build/quick/full and focused gates are required before publication; exact command results are retained with the batch evidence.

Reproduce from the repository root:

```sh
make clean && make
LD_LIBRARY_PATH="$PWD" make test-quick
LD_LIBRARY_PATH="$PWD" make -k test
make test-attention-staging test-attention
python3 tests/test_attention_staging.py --output "$TMPDIR/tusim-attention-staging-check"
```

The Python runner requires compiler sanitizer support, `ar`, and Python `tarfile` data-filter support. It rejects an existing/in-checkout output path, archives only library build inputs (unrelated example symlinks are not needed), records input hashes/full commands/statuses, and never changes the checkout build. Preliminary runner attempts stopped on an unrelated external example symlink and a missing copied test header; those incomplete attempts are not acceptance. Successful runs use the corrected restricted snapshot and explicit header input.

## Explicitly unresolved behavior

`./test-attention-staging --kv-probe` is a **diagnostic expected to exit 1**, never a passing acceptance gate. Q=K=0, V=0.25, M=1, N=4, d=1 and two KV tiles of 2 return **0.5 instead of 0.25**: each KV tile is normalized separately and then summed. There is no running cross-KV max/sum or output rescaling. This is now a reproducible READY item; the one-item/day small-backlog rule leaves it for the next work unit. Options must preserve correct single-tile behavior while comparing a properly specified streaming/global-normalization implementation against a whole-row baseline; independently normalized summation is not a valid alternative.

Custom multi-row mask slicing, auto-tiler/runtime-capacity coherence, allocation-failure status, descriptor `dataflow` consumption, large tensor/head-stride overflow and cross-engine output/timing contracts are **not certified** here. They remain audit/dependency questions, not newly claimed READY implementations. Historical attention/norm/softmax pipeline performance recommendations are explicitly withdrawn: their offset-zero consumers address scratch, and their timing domains are incompatible. Physical dataflow movement, requester-aware SRAM arbitration, compressed-domain MMA feed and padded im2col fill retain their prior dependencies.

Evidence: `evidence/2026-10-25-attention-staging/` contains raw baseline/final/mutation logs, complete matrices, source hashes, commands/statuses and the batch manifest; no compiled artifacts or source-tree copies are published.
