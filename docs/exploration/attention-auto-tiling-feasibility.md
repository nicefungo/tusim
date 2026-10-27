# Attention auto-tiling: runtime feasibility, not a physical optimum

**Audit:** 2026-10-27. **Mode:** pre-spec, correctness-first. **Input HEAD:** `65262e4`.
**Status:** DONE for configured tile reservation and bounded selection. No new engine, precision mode, JSON/YAML field or public ABI.

## Question and counterexample

Can automatic selection return a tile that passes the engine's actual A/W/O reservation check, without erasing explicit hardware-plausible organizations?

At M=16, N=512, d=64 and default A/W/O=65,536/131,072/65,536 B, the baseline chooses **16x512**, reserves **69,632 O bytes**, and rejects a feasible workload. Explicit **8x512** succeeds. A fresh archive of the input HEAD reproduces `--auto-probe` exit 1. The old helper budgets only one score region, ignores runtime capacities and A storage, and rounds upward after budgeting. These are model/control defects, not legitimate low-resource alternatives.

## Executable contract

`tu_attention_auto_tile()` and `tu_attention_execute()` now share a private footprint producer:

```
A = 2*Tm*d
W = 4*Tn*d                      # K plus transposed K; V reuses storage
O = 8*Tm*Tn + 4*Tm*d
    + (8*Tm when Tn<N)          # FP32 row maximum and denominator
```

The O contract deliberately reserves score, accumulator and mask scratch even for an unmasked descriptor. Products are checked against capacity before byte scaling and summing. All capacities come from initialized `g_tu.rt_cfg`, not compile-time size macros. Offsets remain naturally FP32-aligned; no unsupported PE-alignment constraint is invented.

The deterministic **bounded heuristic** searches query rows `1..min(M,64)`. At each query width it evaluates the complete-N endpoint separately, then solves the streaming budget and revalidates with the shared producer. Complete-N removes persistent state, so treating the endpoint like an ordinary streaming width can miss a feasible/larger candidate. The helper retains the historical tile-area scoring intent, with ties choosing smaller Tm, but removes unsafe upward PE rounding. It is **not** a traffic, latency, accuracy, utilization or physical-area optimizer. The query bound limits host search cost, not explicit architecture support.

Calling the helper overwrites both tile fields and nothing else. NULL is a no-op; zero dimensions, uninitialized capacity or no fit return `0/0` without divide-by-zero, movement, counter or SRAM access. Its void ABI is unchanged. Callers can inspect `0/0`; subsequent execute still uses its historical zero-field compile-time PE fallback, which fails admission when no tile can fit. The convenience API inherits the repaired selection and safe no-fit rejection.

**Compatibility:** explicit tiles, including oversized/ragged tiles and query widths above 64, retain their existing reservation/execute semantics. Calling execute directly with one or both zero fields retains the historical fallback rather than silently invoking the new search. Existing runtime WS/OS/RS selection remains through active global state; `desc.dataflow` is not promoted into a consumed setting. Automatic choices intentionally change, and different KV partitions can have different FP16 rounding histories; no general auto-versus-old bit identity is promised.

## Full representative trade-off matrix

Workload: batch=heads=1, M=16/N=512/d=64, Q=K=0, V=0.25, unmasked FP16 input/output and FP32 accumulation. All successful rows return **exact 0.25** and perform **2,097,152 MMA FLOPs**, counting two operations/MAC for both products. The following resource/byte rows execute in all three active plug-ins (WS 16x16, OS 8x16, RS 16x4); this deliberately tests ragged/runtime PE support, **not** comparable physical cycle performance. Full per-mode records are retained in the evidence CSV.

| Selection | Query x KV tile | A reserve B | W reserve B | O reserve B | Logical DMA B |
|---|---:|---:|---:|---:|---:|
| Explicit low query state | 1x512 | 128 | 131,072 | 4,352 | 2,101,248 |
| Explicit whole row | 8x512 | 1,024 | 131,072 | 34,816 | 266,240 |
| New default-memory auto / explicit | 15x512 | 1,920 | 131,072 | 65,280 | 266,240 |
| Explicit narrower KV | 16x128 | 2,048 | 32,768 | 20,608 | 135,168 |
| Explicit intermediate KV | 16x256 | 2,048 | 65,536 | 36,992 | 135,168 |
| Auto with doubled runtime capacities | 16x512 | 2,048 | 131,072 | 69,632 | 135,168 |

The last row uses A/W/O=131,072/262,144/131,072 B, unlike the default-memory rows. Automatic 15x512 **does not minimize logical traffic**: explicit 16x128 fits default memory, needs streaming row state and more KV iterations, but keeps all queries together and halves KV traversals. Explicit 8x512 uses less O storage than auto with identical logical bytes. These counterexamples are why all descriptor choices remain available rather than declaring the automatic row universally best.

Additional executed controls: constrained A selects 1x8; constrained W selects 3x1; A/W/O=80/96/192 B selects 3x3 for M=5/N=33/d=8. A 65-query workload auto-selects 64x8, whereas explicit 65x8 is legal and moves 2,336 rather than 2,592 logical bytes at the cost of 6,240 rather than 6,144 O bytes. Exact 20-byte O endpoint admission passes and 19 bytes rejects before movement. Separate A and W one-byte-short controls reject; oversized explicit tiles, partial-zero fallback, causal/custom masks and the convenience API are exercised.

## Multi-objective interpretation

| Objective | Gain / sacrifice and boundary |
|---|---|
| Throughput / latency | Removes avoidable rejection. Smaller query tiles repeat K/V loads; narrower KV tiles add dispatch/normalization/state work. Physical latency, throughput and clock feasibility are **unquantified**; mixed engine counters are not a makespan. |
| Area / resources | Narrow query/KV reservations can fit smaller scratch; complete rows avoid cross-tile row state. Wider parallel working sets may need more banks/ports/wiring. Reservations are executable bytes, not synthesized area. Physical area is **unquantified**. |
| Power / energy | Less allocated storage can reduce capacity/leakage needs; repeated loads and streaming rescale add activity. Net SRAM/DRAM/controller energy and power are **unquantified**, not inferred from tile area. |
| SRAM / DRAM traffic | Exact logical DMA bytes and live SRAM accesses are observable. External bursts, RF/NoC movement, bank-port schedules, cache effects and physical overfetch are **unquantified**. |
| Numerical accuracy | Explicit-path 2,565 cross-KV and 720 staging cases retain identical complete error/counter tuples; staging raw digest is unchanged. Automatic partitions can change FP16 association, so bounded independent-oracle tests remain necessary. No converter or rounding policy changes. |
| Control complexity | Shared admission eliminates drift; host search is bounded to 64 query candidates and at most one full plus one streaming candidate each. Hardware may hard-wire tile sizes or let a compiler choose; this is a host planner, not a synthesized scheduler. |
| Verification burden | Adds runtime capacity, endpoint state discontinuity, overflow/zero/no-fit, deterministic ties, descriptor/TU immutability and actual execute checks. Huge tensor host extents and full allocation-failure coverage are not certified. |
| Compiler / runtime | Call auto after global initialization, check no-fit, or supply explicit tiles. Existing fields and ABI/default dataflow remain. The simple API's selected partition changes intentionally; runtime must not interpret tile-area score as a physical recommendation. |

## Verification and reproducibility

- **Baseline:** `make clean && make -j4`, `LD_LIBRARY_PATH="$PWD" make test-quick` pass. Fresh archived input HEAD reproduces auto 16x512 rejection and explicit 8x512 success. Initial focused RED reports two failed feasibility/success assertions; repaired first GREEN selects 15x512 with 65,280 O bytes and exact output.
- **Auto gate:** **90,747 metadata cases**, **85,765 no-fit cases**, **69 live executions**, **507,611 checks**, zero failures. Metadata fixtures use an independent brute-force `__uint128_t` reservation oracle; large capacities are metadata-only snapshots, never allocations or hardware execution. Includes zero, near-UINT32_MAX, full endpoint, search-bound, repeatability and non-tile descriptor/TU-state immutability. Live fixtures allocate matching real buffers and assert output guards, bytes/FLOPs and no-fit output/stats/counter immutability. **57 complete unique successful rows** are retained.
- **Compatibility:** all **2,565 KV** and **720 staging** ordered numerical-error/counter tuples match the baseline quick transcript exactly; staging output digest remains `32ed8b45db42dbbd`. Legacy attention **9/9**; no broad arbitrary-input equivalence claim.
- **Sanitizers:** rebuilt every library object with ASan/UBSan/LSan, then ran auto, KV, staging, legacy and both export-allocation failures. **66 recorded commands**, all expected outcomes validated. Five new metadata-only mutants (compile-time W, ignored A, missed full endpoint, missing mask budget, changed ties) are rejected without executing unsafe requests; all eight previous staging/KV mutants remain rejected.
- **Final:** clean build, quick and keep-going full aggregate pass. Config, convolution, scatter/gather, softmax, normalization and direct-global dataflow sweep pass. GCC `-fanalyzer -Wall -Wextra -Werror` passes the changed engine. A fresh exact staged snapshot is rebuilt and tested before publication; its results are retained separately from the live checkout.

```sh
make clean && make -j4
LD_LIBRARY_PATH="$PWD" make test-quick
LD_LIBRARY_PATH="$PWD" make -k test
make test-attention-autotile test-attention-kv test-attention-staging test-attention
./test-attention-kv --auto-probe
python3 tests/test_attention_staging.py --output "$TMPDIR/tusim-auto-audit-new"
```

The last output path must be new and outside the checkout. The runner records the input HEAD plus exact overlaid source hashes, commands, return codes and raw-log hashes. Retained logs, compatibility comparisons, the complete trade-off CSV and manifests are in `evidence/2026-10-27-attention-autotile/`. No binaries, unrelated terminology draft or HANDOFF is published.

## Backlog boundary

Today's sole READY item is closed; four standalone BLOCKED rows remain: requester-aware SRAM arbitration, additional compression formats without a concrete hypothesis, compressed-domain MMA feed, and tagged im2col padding movement. Scoped physical dataflow/interconnect/arbitration/calibration dependencies also remain blocked; no new contract was inferred from this planner fix. Large attention host-stride/head-count products, allocator propagation, precision variants and cross-engine output/timing integration remain separate audit boundaries, not silently certified features.
