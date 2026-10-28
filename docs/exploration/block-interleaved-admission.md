# Block-interleaved iterator: byte-span admission and banking fidelity

## Question and executable evidence (2026-10-28)

Does the remaining runtime `BLOCK_INTERLEAVED` iterator reject unrepresentable
byte views, and do its bank settings actually select a different memory mapping?
Baseline: `6359360` (the exact full input commit is retained in the evidence
manifest). No READY item existed at entry, so this is one new bounded
correctness/fidelity exploration, not a gap-list feature campaign.

Five safe baseline probes fail: an element including reserved byte `UINT32_MAX`,
a wrapping address, `count=1073741824` with 4-byte elements, a first address equal
to exhaustion, and an oversized final element. Exact-exclusive-end and empty
controls already pass. No rejected address is dereferenced. The completed final
harness linked with the baseline address-generator object reports **30,487
admission mismatches**; the corrected implementation reports zero.

Source audit also finds a fidelity limit:

```
bank = index % block_width
element = index / block_width
address = base + (element * block_width + bank) * element_size
        = base + index * element_size
```

`num_banks` is retained but not consumed by this iterator's address producer.
This may be a valid **linear logical stream before a downstream bank decoder**,
but it is not evidence of bank-major storage, bank selection, conflict reduction,
or concurrent bank service. The separate unchecked `tu_agen_addr_banked()` helper
is not this iterator's consumer or an integration proof.

## Implemented contract and compatibility

- Reuse `affine_geometry_valid()` before publishing an executable count. For a
  nonempty view, `base + count * normalized_size <= UINT32_MAX`; that is the
  **exclusive** byte end. Byte `UINT32_MAX` is reserved, never data.
- Runtime input remains `uint32_t[4] = {count,banks,block_width,size}`. Preserve
  zero banks/block width → 1, zero size → 4 B, arbitrary nonzero metadata values,
  unaligned bases, and every valid historical linear address sequence.
- Count zero remains empty at every base/width, including maximal values.
  Invalid admission returns −1 with an empty, safely exhausted iterator.
- No new public API, struct layout, YAML/JSON field, precision policy, default
  architecture, or DMA timing changes. Invalid modulo wrap is a defect, not a
  realistic optional architecture. This is uint32 representability, **not**
  SRAM/host allocation admission; downstream capacity checks still apply.
- The public header now labels the iterator's actual metadata-only bank scope.
  Actual mapping alternatives remain blocked rather than invented or replaced
  by whichever presumed layout looks fastest.

## Complete measured metadata matrix

Base 256, 17 elements, 4 B each. All rows generate the exact same linear address
sequence; no bank-occupancy or physical-cycle counter is produced by this probe.
Each setting also passes real load/store descriptor movement at unaligned base
13, with the entire 128-byte host/SRAM regions checked, including guard bytes.

| Banks | Block width | Logical bytes | Logical contiguous runs | Load/store bytes per direction |
|---:|---:|---:|---:|---:|
| 1 | 1 | 68 | 1 | 68 |
| 4 | 1 | 68 | 1 | 68 |
| 4 | 2 | 68 | 1 | 68 |
| 4 | 7 | 68 | 1 | 68 |
| 32 | 4 | 68 | 1 | 68 |

These are compatibility observations, **not five implemented banking designs**.
All settings remain accepted; there is no measured performance winner.

## Realistic alternatives, gains, sacrifices, and fidelity limits

| Alternative | Why a hardware team might choose it | Sacrifice / current state |
|---|---|---|
| Contiguous logical addresses plus downstream element/block bank decode | Simple tensor/DMA address continuity; bank placement can be centralized in SRAM rather than changing compiler-visible storage | Conflict behavior depends on the actual decoder, access stride and simultaneous requests. This iterator models the linear sequence only; no physical banking claim. |
| Explicit bank-major/blocked physical layout and matching producer/consumer translation | Can make selected per-bank streams local and fit kernel-specific bank/port schedules | Can fragment other streams, require padding/layout transforms, and burden compiler allocation. **BLOCKED:** define logical-to-physical byte/bank mapping, tails, capacity and all consumers before implementation. |
| Wider address/count ABI | Larger memories and views without modulo aliasing | Requires new sentinel, address fields, DMA consumers and tests. **BLOCKED** on an end-to-end ABI; not silently emulated by wrap. |
| Fail-closed checking versus partial runtime fault | Pre-admission prevents invalid work and side effects; partial faults could support dynamic protection | This change implements pre-admission only. Partial faults require explicit status, completed-prefix, rollback and retirement semantics; no fault-latency claims. |

| Dimension | Measured gain / sacrifice / unquantified boundary |
|---|---|
| Throughput / latency | Invalid views no longer become executable wrapped work. Valid modeled service is unchanged; constant host admission work is added but not benchmarked. Physical AGU issue rate, critical path, bank concurrency and fault latency are **unquantified**. |
| Area / resources | Reuses one host validator without new mode state. Physical bounds/carry comparators and error state could cost resources; mapping alternatives could add decoding/layout storage. Hardware costs are **unquantified**. |
| Power / energy | Checking adds host activity; physical check/decoder switching and possible banking savings are **unquantified**. No fabricated energy or speedup number. |
| SRAM / DRAM traffic | Valid logical sequences and tested 68-byte copies are unchanged. No inference about physical bank traffic, cache reuse, protocol bursts, overfetch or DRAM bandwidth. |
| Numerical accuracy | Raw byte movement is exact; no conversion, rounding or arithmetic semantics change. Invalid geometry now errors instead of selecting unintended bytes. |
| Control complexity | One existing shared geometry validator; explicit empty/default/error contracts. Physical mapping/partial-fault protocols remain separate, avoiding an unverified mode product. |
| Verification burden | Full extent, sentinel, huge/empty views, normalization, metadata independence, sequence/exhaustion/reset/bulk/range replay and both movement directions are gated. No certification of unchecked inline helpers or host allocation sizes. |
| Compiler / runtime | Check init status; do not interpret bank metadata as an executed permutation or a timing policy. Physical layouts require coordinated allocator, producer/consumer and capacity contracts. No new compiler integration is claimed. |

## Verification executed

- Baseline `make clean && make && make test-quick`: PASS.
- Final `make clean && make && make test-quick && make test`: PASS.
- Final additional config **31/31**, convolution **12/12**, scatter/gather
  **15/15**: PASS.
- New static-linked `make test-agen-block-bounds`: **46,860 geometry cases**,
  **30,487 expected rejections**, **10,424 bounded complete sequences**,
  **1,803,905 checks**, zero failures; includes ten real 68-byte load/store rows.
  Huge accepted views sample first/last/exhaustion without huge traversal.
- Independent 128-bit test oracle; deterministic boundary/random cases; input
  immutability, NULL handling and normalized metadata checked. Rejected iterators
  cannot emit bulk/range work or modify supplied output buffers.
- Same final harness with baseline source: exit 1 and **30,487 mismatches**.
  The unmodified control passes and four safe mutants fail via completed harness
  summaries: ignored base, checking only the last starting byte, rejecting exact
  exclusive ends, and a wrong zero-width default. Crashes do not count as kills.
- Fresh archive with **every library object** instrumented: ASan/UBSan/LSan PASS
  for the new gate, legacy address tests, affine bounds, tiled traversal, complete
  DMA-chain construction including 514 OOM cases, and range admission. Coverage
  is limited to these exercised paths, not all library behavior.
- GCC `-fanalyzer` on changed production source and new test: no diagnostics.

Reproduce from a built checkout (some existing test targets use the shared
library, so retain its runtime search path):

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make clean && make
make test-quick
make test
make test-config test-conv test-scatter-gather
./test-agen-block-bounds --probes
python3 docs/exploration/evidence/2026-10-28-block-admission/verify_mutants.py
```

Sanitizer rebuild in a fresh source archive:

```sh
make libtucmodel.a CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer'
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 make test-agen-block-bounds test-agen test-agen-affine-bounds test-agen-tiled-sweep test-agen-dma-chain test-agen-range-admission CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-lm -fsanitize=address,undefined'
```

Raw compressed logs, exact input checksums, baseline commit, commands and mutation
runner are retained in `evidence/2026-10-28-block-admission/`. The fresh exact-index
build/quick/focused verification is recorded separately there before publication.

## Backlog re-audit

Completed the one newly evidenced admission correction. The bank-layout contract
is newly explicit BLOCKED work, not an implementation of physical banking.
SRAM arbitration still lacks simultaneous requester/port/grant/replay contracts;
compression reconstructs dense data rather than feeding a specified token/FIFO
MMA path; dataflow labels/counters do not prove physical RF/SRAM/NoC events;
padded im2col still lacks a tagged zero-fill DMA consumer. Those dependencies
are not waived, and no new READY item is established. The unrelated untracked
`TERMS.md` remains untouched and excluded from publication.
