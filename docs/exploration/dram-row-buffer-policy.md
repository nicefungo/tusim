# DRAM Row-Buffer Policy: Open Page vs Closed Page

**Date:** 2026-07-29
**Question:** How much does a realistic DRAM row-buffer policy change service time for contiguous tensor reads versus row-thrashing or bank-streaming address patterns?

## Hypothesis

An open-page controller should reduce activate/precharge penalties when tensor DMA reuses a row, while a closed-page controller should be insensitive to locality and avoid retaining stale row state. The benefit should disappear for alternating-row traffic.

## Realistic alternatives

| Mode | Why hardware might choose it | Principal sacrifice |
|---|---|---|
| `open_page` | Tensor DMA often streams contiguous tiles, so retaining a bank's row can exploit spatial/temporal locality | Requires per-bank open-row state and a policy for conflicts; adversarial or mixed traffic can thrash rows, and fairness/predictability are harder |
| `closed_page` | Predictable behavior for random, mixed-tenant, or low-locality traffic; simpler controller policy and no stale open row | Pays activation/precharge cost on every modeled access and discards useful locality |
| `legacy` | Preserves old cmodel behavior and zero-initialized callers; useful only as a compatibility/reference baseline | The optional historical boolean charges every read and no writes, without tracking actual rows; it is not a physical row policy |

The pre-spec cmodel preserves all three. `legacy` remains the default solely for compatibility; it is not selected as an architectural recommendation.

## Executable model and configuration

Canonical JSON/YAML under `tu.memory.dram` now accepts:

```json
"row_policy": "legacy",
"row_miss_penalty_cycles": 10
```

Accepted policies are `legacy`, `open_page`, and `closed_page`. The miss penalty is validated in `[0, 1,000,000]` cycles. `tu_dram_create_from_config()` propagates the canonical DRAM type, bandwidth, channels, read/write latency, legacy boolean, row policy, and penalty into the executable model. Generated and checked-in headers expose matching constants.

The explicit model tracks one open row per channel/bank. Address mapping is deterministic burst interleaving across channels, then row-sized column groups across banks. Open-page accesses hit only when the mapped bank retains the requested row. Closed-page accesses are always misses. Reads and writes both participate in explicit row policy; the compatibility mode preserves the old read-only behavior.

## Sweep

Command:

```sh
make test-dram-row-policy-sweep
```

Configuration: DDR5 preset, one channel, 64 reads × 64 B, base read latency 50 cycles, row-miss penalty 20 cycles. The table sums the API's returned service cycles; it does **not** add the model's separate contention-stall counter.

| Pattern | Policy | Service cycles | Hits | Misses | Hit rate |
|---|---|---:|---:|---:|---:|
| sequential 64 B | open page | 3,240 | 62 | 2 | 96.88% |
| sequential 64 B | closed page | 4,480 | 0 | 64 | 0.00% |
| alternating rows | open page | 4,480 | 0 | 64 | 0.00% |
| alternating rows | closed page | 4,480 | 0 | 64 | 0.00% |
| bank stream | open page | 3,520 | 48 | 16 | 75.00% |
| bank stream | closed page | 4,480 | 0 | 64 | 0.00% |

For this exact sequential pattern, open page lowers modeled service cycles by 27.7% versus closed page. For the bank stream it lowers cycles by 21.4%. Alternating rows eliminates the benefit: both policies take 4,480 cycles. These are workload/configuration-specific first-order results, not universal controller rankings.

## Gain versus sacrifice

- **Throughput/latency:** Open page helps only where address locality survives the channel/bank mapping. It has no modeled benefit under row thrash. Closed page gives stable worst-case service in all measured patterns.
- **Area/resources:** Open page needs an open-row tag per bank and comparison/update logic; closed page can omit retained-row state. Area is not quantified.
- **Power/energy:** Open-page hits should avoid some activate/precharge energy; closed page performs more row operations. The current power model has only a generic DRAM-activate event and is not wired to this module, so energy savings are unquantified.
- **SRAM/DRAM traffic:** Payload bytes are identical. Only DRAM internal row operations and service cycles differ; SRAM traffic is unchanged.
- **Numerical accuracy:** Unchanged; both policies transfer identical bytes.
- **Control complexity:** Open page needs row-state tracking and conflict policy. Closed page is simpler and more deterministic. Scheduling, fairness, and adaptive timeout policies are unmodeled.
- **Verification burden:** Both read and write paths, row hits, conflicts, reset behavior, canonical parse/validation, propagation, and malformed policy rejection are gated. The deterministic mapping is documented and tested, but not calibrated to a specific memory controller.
- **Compiler/runtime:** Compilers can improve open-page locality through tensor placement, tiling, and request ordering. Closed page reduces dependence on those choices. The cmodel does not yet expose address-mapping selection or compiler trace replay.
- **Physical limitations:** tRCD and tRP are collapsed into one configurable penalty. Command buses, tRAS/tRC/tCCD, refresh, bank groups, request queues, reordering, write draining, controller page timeout, and calibrated DRAM timing are absent. Returned service cycles and contention stalls remain separate accounting fields.

## Verification

```sh
make test-dram                       # 16/16
make test-dram-row-policy-sweep      # 6 rows, complete hit/miss accounting
python3 scripts/gen_config.py config/tu_config.yaml -o /tmp/tu_config.row-policy.h
make clean && make
make test-quick
```

## Actionable conclusion

Preserve open-page and closed-page as explicit architecture alternatives. Open page is valuable for contiguous tensor traffic but cannot be called better for adversarial/mixed access; closed page trades locality gains for simpler, predictable behavior. A future controller-policy study should add adaptive close/timeout and request reordering only after defining queue, timing, fairness, and trace contracts.