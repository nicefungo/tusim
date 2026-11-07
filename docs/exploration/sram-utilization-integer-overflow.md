# SRAM nominal-utilization arithmetic at large counts

**Baseline:** `b4f8bd426aa0e424bccbb1a3f67256d23478f6f1`. This is a host-side model-observability correctness study, not a physical SRAM speed measurement. It is separate from the unresolved question of what utilization denominator ought to mean (`sram-utilization-window.md`).

## Question and experiment

Can the existing nominal-opportunity metric retain its historical definition without corrupting the arithmetic when intermediate integer products or sums exceed 64 bits? We compare the original wrapped arithmetic against the same formula evaluated with wide intermediates. `tests/explore_sram_utilization_overflow.c` makes one real read, then advances the meter by a synthetic large number of cycles; window=1, four-byte words. These extreme idle intervals deliberately stress representation and are not a realistic workload or a calibration claim. Full six-row matrices for baseline and correction are under `evidence/2026-11-07-sram-utilization-overflow/`.

| Banks × grants | Observed cycle | Baseline global | Corrected global | Baseline bank 0 | Corrected bank 0 |
|---|---:|---:|---:|---:|---:|
| 2 × 1 | 10 | 0.05 | 0.05 | 0.1 | 0.1 |
| 2 × 1 | 2^63 | 0 | 5.42101086e-20 | 1.08420217e-19 | 1.08420217e-19 |
| 2 × 1 | 2^63+1 | 0.5 | 5.42101086e-20 | 1.08420217e-19 | 1.08420217e-19 |
| 1 × 2 | 10 | 0.05 | 0.05 | 0.05 | 0.05 |
| 1 × 2 | 2^63 | 0 | 5.42101086e-20 | 0 | 5.42101086e-20 |
| 1 × 2 | 2^63+1 | 0.5 | 5.42101086e-20 | 0.5 | 5.42101086e-20 |

The four-byte value and served count remain correct in every row. The baseline wraps the capacity product to zero or two; the latter falsely reports 50% instead of a tiny fraction. Sums across read/write and banks can wrap independently; the regression injects counter values to exercise these arithmetic states, not to claim a realizable access history.

## Alternatives and trade-offs

* **Legacy wrapping arithmetic:** cheapest existing host computation, but invalid results at representational boundaries; excluded as a correctness contract.
* **Checked sums/products with a wide floating fallback (implemented):** keeps the existing integer/float path when representable, and evaluates the *same* ratio in `long double` only after overflow is detected. It preserves the zero-cycle/disabled behavior and clamping of the existing getters. It adds host checks and an overflow-only wide division; host overhead was not measured. `long double` precision is platform-dependent; this is a floating report, not exact rational storage.
* **Different interval/event-based metrics:** could yield a more meaningful physical utilization definition for a specified interval, but require time, context, refill, enable/reset and boundary contracts. They remain blocked; this correction does not silently change the denominator.

No configurable architecture mode is added: allowing silent arithmetic wrap would not describe a defensible hardware alternative. Neither this correction nor the probe changes modeled SRAM/DRAM bytes, numerical tensor arithmetic, request ordering or modeled stall penalties. Physical throughput, latency, area, power/energy, actual traffic, and compiler impact are unquantified. Wider counter/clock implementations could cost registers, software/RTL control and verification; the present change is host-side reporting only. Callers must not treat this nominal metric as a measured interval rate.

Individual live `uint64_t` counters can themselves wrap before the getter sees them; a `current_cycle` wrap or context rollback also remains outside this correction. Counter lifetime/rollover, interval epochs, context ownership and physical service/retry semantics require explicit contracts. Existing callers and ABI are unchanged.

## Verification

* The independent `__uint128_t` oracle gate covers 2,304 injected-state combinations: 1/2/4/8 banks, 1/2/3/255 grants, windows 1/4/17, eight time boundaries including zero and `UINT64_MAX`, six read/write counter patterns, global and every-bank getter, clamping and disabled-global behavior. Counter injection is an arithmetic stress test. At baseline it reports 3,464 mismatches; corrected code reports zero. The earlier 56-row live utilization-window matrix is byte-identical before/after.
* Baseline and final `make clean && make && make test-quick` pass in isolated source trees. Final full `make test` passes with `LD_LIBRARY_PATH` pointed at that tree. Without it, the legacy `test-isa` target cannot locate `libtucmodel.so`; this is an existing full-suite linkage issue, not a passed no-environment run.
* Focused all-object ASan/UBSan build and run pass. These exercised paths do not certify individual counter rollover, context switching or other consumers.

Reproduce in the repository after building:

```sh
make test-sram-utilization-overflow
cc -O2 -Wall -Wextra -Werror -std=c11 -I. -Itu_cmodel tests/explore_sram_utilization_overflow.c ./libtucmodel.a -lm -o "$TMPDIR/sram-util-overflow"
"$TMPDIR/sram-util-overflow"
```

For the historical matrix, build the baseline commit separately and run the same probe with `--baseline`; do not run that expected-old-behavior mode against the corrected library.
