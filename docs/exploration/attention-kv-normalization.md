# Whole-row versus streaming-KV attention normalization

**Audit:** 2026-10-26. **Mode:** pre-spec, correctness-first. **Input baseline:** `d2a2194b1e4177923d5006e1a63e3c04478d3bf7` (the separately verified staging checkpoint). **Status:** DONE for the bounded functional and accounting contracts below, not physically calibrated timing.

**2026-10-27 follow-up:** the separately scoped auto-tiler defect below is now fixed and gated; see `attention-auto-tiling-feasibility.md`. Historical 2026-10-26 evidence remains unchanged. The current runner expects `--auto-probe` to succeed with 15x512 and additionally gates runtime-capacity selection; the prior expected-failure statement applies only to the pinned baseline.

## Question, alternatives and RED evidence

Can KV tiles reduce live score storage while still computing one globally normalized attention row? The prior staging gate's `--kv-probe` supplies Q=K=0, V=0.25, M=1/N=4/d=1 and tile_n=2. The baseline returns **0.5**, not **0.25**, because it adds two separately normalized tile outputs. That summation is mathematically incorrect, not a legitimate low-area hardware alternative.

Preserve both valid organizations through the **existing runtime descriptor**:

- `tile_n >= seq_len_kv`: whole-row normalization. One complete score row is available, normalized probabilities are packed to FP16, and P×V accumulates in FP32. It avoids persistent cross-tile normalizer state but requires enough score/mask scratch for the configured tile.
- `tile_n < seq_len_kv`: streaming max/sum plus output rescaling. Smaller score tiles can fit constrained SRAM and represent a sequential KV engine, at the expense of persistent row state, repeated tile control, rescale accesses and final normalization. Intermediate KV widths remain valid, not just the smallest or largest choice.
- `tile_m` independently controls query reuse/footprint. WS/OS/RS remain selectable through the **active global** `tu_set_dataflow` path. The descriptor's historical `dataflow` member is still not consumed; no claim of physical stationarity is made.

No new JSON/YAML field, enum, public struct layout, signature, default dataflow or converter policy is needed. The configuration path under test is descriptor → footprint admission → actual KV loop → normalization/MMA. Existing canonical config tests remain relevant for global initialization, not proof of an invented attention JSON knob.

## Numerical contract and implementation

For each query row retain running maximum m, normalizer l and unnormalized output U. Initially m=-inf, l=0 and U=0. For a scaled/masked FP32 score tile S:

```
m_next = max(m, max(S))
alpha  = 0 if l == 0 else expf(m - m_next)
p_j    = expf(S_j - m_next)
l_next = FP32(alpha*l + FP32(sum_in_double(p_j)))
U      = FP32(alpha*U) + MMA(FP16(p), FP16(V))
```

After all KV tiles, output is `FP16(FP32(U/l))`. A zero normalizer produces zero. Scores of -inf contribute exactly zero; `m_next == -inf` is handled without evaluating -inf-minus--inf. Like the retained whole-row softmax, exponentials with shifted score below -87 are zeroed. FP16 packing uses the existing canonical conversion; MMA executes the selected built-in plug-in. This is a **mixed-precision host functional model**, including a double per-tile sum, not a claim about an FP32-only physical reduction tree, exp approximation unit or specified floating-point association in RTL.

Whole-row no-mask/causal execution retains its original standard-softmax path. Whole-row custom masks use the same row helper without persistent state, normalizing before FP16 packing. Custom mask source indexing uses `[head, full_query_row, full_N]`, not compact tile pitch. Mask generation/loading no longer allocates a temporary causal host array. Empty mask tiles still pass through the defined recurrence and MMA; the prior whole-tile causal V-skip shortcut is removed rather than assigning probabilities independently. This deliberately computes and counts all logical QK/PV work, including masked pairs; sparsity/causal pruning would need its own executable alternative and accounting.

NaN/+inf scores on the new normalization path, nonfinite FP32 output, and nonfinite FP16 export return -1. Finite FP32 does not by itself guarantee finite FP16: the exact uniform N=27 / V=`0x7bff` control exposes overflow from represented probability mass above one. Export now rejects before copying that output tile, without silently saturating or changing the converter. Both output temporary-allocation failures free any surviving buffer and return -1. This does **not** establish a complete allocation-failure contract for the standalone softmax or all subsystems.

Invalid scale/mask descriptors and oversized footprints reject before DMA, SRAM access, output or supplied-stat mutation. Runtime numerical/allocation errors are not transactional: earlier output tiles and internal SRAM/counters may already have changed; the supplied stats are assigned only on complete success. Host buffer extents/aliasing, arbitrary large head/stride products and unsupported engine dimensions remain caller/audit boundaries, not newly certified safety properties.

## Honest resource and accounting contract

For configured query/KV tile sizes Tm/Tn and head dimension d:

```
A bytes = 2*Tm*d
W bytes = 4*Tn*d                 # K plus transposed K; V reuses K space
O bytes = 8*Tm*Tn + 4*Tm*d       # FP32 score, accumulator, mask
          + (8*Tm if Tn<N else 0) # persistent FP32 max and sum
```

The state is included in fail-closed O admission. An exact 1,024-byte streaming scratch configuration passes; one byte less rejects before movement. These are configured reservation sizes, not measured silicon area, bank-port count or a peak host-memory footprint. The two temporary export buffers remain host allocations.

Logical DMA counts Q once per query tile, K/V once per KV traversal, and the final FP16 output. For the no-pruning contract, it is independent of KV partition width at fixed query tiling. MMA FLOPs are exactly `4 * batch * heads * M * N * d` (two FLOPs/MAC, both products), including masked pairs; scalar normalization/rescaling operations are **not** included in that MMA counter. New row-state/accumulator SRAM accesses contribute to the existing stall counter. Host exp/max/reduction/divide/control operations have no calibrated latency or energy charge. The historical `compute_cycles`, `dma_cycles`, their sum and utilization ratio mix domains; none is a validated physical throughput/makespan/energy metric.

### Full representative gain-versus-cost matrix

Workload: batch=2, heads=2, M=5, N=33, d=8, query tile=3, no mask, scale=0.5, bounded deterministic dyadic Q/K/V (pattern 1 in the test). Default 16×16 PE and SRAM configuration. Actual query tail=2. All rows perform the same 9,088 logical DMA bytes and 21,120 MMA FLOPs. A reservation is 48 bytes. Columns labelled counter retain their executable names only.

| Active mode | KV tile | W reserve B | O reserve B | Logical DMA B | Compute counter | DMA counter | Max absolute output error vs FP64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WS | 1 | 32 | 144 | 9,088 | 46,702 | 1,544 | 0.0000447441022 |
| WS | 3 | 96 | 192 | 9,088 | 24,912 | 1,516 | 0.0000447441022 |
| WS | 33 | 1,056 | 888 | 9,088 | 16,080 | 1,576 | 0.0000571622811 |
| OS | 1 | 32 | 144 | 9,088 | 40,102 | 1,544 | 0.0000447441022 |
| OS | 3 | 96 | 192 | 9,088 | 22,360 | 1,516 | 0.0000447441022 |
| OS | 33 | 1,056 | 888 | 9,088 | 15,048 | 1,576 | 0.0000571622811 |
| RS | 1 | 32 | 144 | 9,088 | 43,534 | 1,544 | 0.0000447441022 |
| RS | 3 | 96 | 192 | 9,088 | 23,680 | 1,516 | 0.0000447441022 |
| RS | 33 | 1,056 | 888 | 9,088 | 15,552 | 1,576 | 0.0000571622811 |

For the same workload and KV tile=3, query tile=1 changes logical DMA to **21,760 B** while reducing per-query-tile A/W/O reservations to **16/96/64 B**; query tile=3 uses **48/96/192 B**. This is a storage-versus-KV-reload trade-off. Narrower KV tiles at fixed query tiling reduce live W/O storage, **not** total useful input bytes in this implementation. Smaller error in one bounded row does not make the streaming precision association universally more accurate.

| Objective | Gain, sacrifice and measurement boundary |
|---|---|
| Throughput / latency | Correct multi-KV output replaces an invalid sum. Narrow tiles incur more dispatch/state work; wide tiles need more scratch. Counter deltas are reproducible but physical latency/throughput gains are **unquantified**. |
| Area / resources | Streaming reserves 8 bytes/query row of max/sum but can substantially reduce score/K scratch. Wider tiles may need more storage, banking and ports; area and timing feasibility are **unquantified**. |
| Power / energy | Smaller working sets may reduce storage capacity/leakage, while rescaling/control/repeated reads add activity. Query reuse reduces logical loads. No net power/energy or physical SRAM/DRAM energy is established. |
| SRAM / DRAM traffic | Logical DMA and SRAM accesses are executable; cross-KV row state is no longer free. External DRAM burst occupancy, NoC traffic, RF movement, port conflicts and a complete physical traffic schedule are **unquantified**. |
| Numerical accuracy | Independent bounded FP64 oracle, finite checks and partition comparisons. Streaming FP16 unnormalized weights and whole-row FP16 probabilities have different rounding histories; exact general bit-invariance is **not** promised. Uniform power-of-two controls are exact. |
| Control complexity | Whole-row avoids cross-tile state; streaming needs max/sum lifetime, rescaling and final divide. Current selection follows descriptor tile shape, not automatic mode switching or preemption/context integration. |
| Verification burden | Adds changing/decreasing maxima, extreme gaps, empty/all-masked tiles, custom source pitch, tails, heads/batches, state capacity, allocation/error paths and independently rejected mutants. Unsupported dimensions/allocators/parallel host threads remain outside this gate. |
| Compiler / runtime | Existing descriptors/ABI/defaults retained. Tilers must honor the actual scratch formula and distinguish numerical partition tolerance from bit identity. Auto-tiler feasibility and public downstream output-layout contracts remain separate work. |

## Verification and reproducibility

- Baseline clean build/quick pass on the supplied staging worktree; its exact staged snapshot also passes clean build/quick/full before the staging checkpoint commit. The original uniform cross-KV probe exits 1 and returns 0.5. The initial 2,565-case bounded matrix fails 210,279 checks on that baseline (pre-admission expansion), not a claimed canonical fingerprint for every defect.
- Final cross-KV matrix: **2,565 cases, 457,309 checks, zero failures**. N=1/4/5/8/17/33, d=1/3/8, KV tiles=1/2/3/7/33, query tiles=1/3, two batches×two heads, active WS/OS/RS, no-mask/causal/custom, exact uniform controls, monotone increasing/decreasing/extreme score patterns, ragged tails, all-masked rows and initially empty KV tiles. Max absolute error **0.000361148989**; max whole-row/partition difference **0.00048828125**, below the stated 0.002 bound for bounded V and tested shapes. These are not arbitrary-input or format-conformance tolerances.
- Prior staging: **720 cases / 67,034 checks**, unchanged raw-output digest **`32ed8b45db42dbbd`** and identical complete 720-row error/counter tuples versus the checkpoint library. Legacy attention **9/9**. Former uniform KV probe now returns exactly **0.25** and exits 0.
- Fresh all-object ASan/UBSan/LSan passes both matrices, legacy attention and the two explicitly injected export-allocation failures. Three historical staging mutants plus five independent KV mutants (lost prior denominator, missing O rescale, missing final divide, compact custom source pitch, omitted state reservation) are rejected. The state-reservation mutant is caught by admission assertions; no out-of-bounds invocation is needed.
- Final clean `make`, `make test-quick`, keep-going `make -k test`, config, convolution, scatter/gather, softmax, normalization and direct-global dataflow sweep pass. The six-row query-reuse sweep also passes with unchanged KV width. GCC `-fanalyzer -Wall -Wextra -Werror` on the changed engine passes. The exact staged snapshot is rebuilt and tested before publication. Matrix emitters are line-buffered beside stderr logs, and the runner checks all 2,565 unique complete rows rather than accepting a summary over an interleaved transcript.
- One intermediate campaign stopped on a **bad test premise**, not accepted production evidence: N=49 uniform FP16 probabilities round the mass below one, so they do not exercise FP16 export overflow. The corrected independently calculated N=27 control was run RED before the export check was restored; the accepted campaign has a distinct run ID. Superseded logs are retained and not labelled passing.

```sh
make clean && make
LD_LIBRARY_PATH="$PWD" make test-quick
LD_LIBRARY_PATH="$PWD" make -k test
make test-attention-kv test-attention-staging test-attention
./test-attention-staging --kv-probe
python3 tests/test_attention_staging.py --output "$TMPDIR/tusim-attention-online-check"
```

The Python output path must not exist and must be outside the checkout. It archives library inputs at the recorded base HEAD, overlays the explicitly hashed changed inputs, builds every object with sanitizers, records commands/return codes/log hashes and rejects all eight mutants. Raw accepted logs, full matrices and manifests live in `evidence/2026-10-26-attention-online/`. Untouched raw transcripts may contain tool-emitted trailing spaces; authored files pass the whitespace gate with immutable logs excluded.

## Historical next item (now completed) and retained dependencies

At the **2026-10-26 pin**, `./test-attention-kv --auto-probe` is an expected exit-1 diagnostic, separate from that day's passing normalization acceptance. Default SRAM and M=16/N=512/d=64 select 16x512, needing 69,632 O bytes and rejecting against 65,536; manual 8x512 needs 34,816 and succeeds exactly. The 2026-10-27 implementation now shares runtime A/W/O/mask/state admission, selects feasible 15x512, avoids upward PE rounding and handles no-fit/zero/overflow safely. Explicit alternatives and historical zero-field execute fallback remain. Both auto and manual probe paths now assert exact output and return success; the complete resource/traffic matrix does not claim an automatic physical optimum. See `attention-auto-tiling-feasibility.md` for the closed item and new evidence.

SRAM requester arbitration remains metadata-only without a batched requester/grant/replay contract. Compressed-domain feed still lacks a codec→MMA token/FIFO contract; physical dataflow/NoC timing lacks movement/arbitration/calibration; im2col padding still exposes an untagged sentinel rather than a certified fill consumer. None is made READY by attention correctness. Huge host extents, allocation-error propagation through legacy softmax, explicit attention precision variants and cross-engine output/timing integration still require separate audited contracts; auto-tile geometry/zero/overflow is covered by the follow-up rather than left pending here.
