# When can a shared-link estimate rule out a schedule?

2026-11-20. The `tu.multicore.contention` selector now separates three questions. `ideal_parallel` (0, unchanged default) reports the longest isolated transfer, useful for independent paths but blind to sharing. `shared_link` (1) retains the historical placement score, not a lower bound. `shared_link_floor` (2) reports a necessary time for a batch on fixed routes and independently serviced directed links. No message or API layout changes. YAML generation and JSON-to-cluster configuration use the same values; the shipped JSON retains the ideal default.

Let `I` be maximum isolated time, `L` the maximum sum of packet serialization times on any directed link, and `R` the largest nonempty route's router delay. With no pre-start service, a batch cannot finish before its longest isolated transfer, nor can an exclusive link supply more than one cycle of service per cycle. Thus `max(I,L)` is a lower bound **in the selected model**, assuming fixed widths, independently rounded packet lengths, and no cross-packet packing, adaptive rerouting or multicast merging. In legacy hop-only mode there is no payload term. The old `max(I,L+R)` adds delays that need not occur on the same route. Neither estimate simulates a schedule.

## Counterexample and controls

The directed-link assumptions and results here use the compatibility `full` duplex mode. For a reversible shared link the necessary service term instead aggregates both directions of the same physical pair; see [link duplex](interconnect-link-duplex.md) for the opposed and disjoint controls.

2×4 XY mesh, 16 B/cycle, 5 cycles/hop. Two 1,024 B messages use 0→1; a separate 16 B message uses 4→5→6→7. The two short-route messages could run completely serially in 69+69=138 cycles while the other row finishes in 16 cycles (cut-through) or 18 (store-and-forward). This is a constructive **upper** bound under the assumptions above, not a measured makespan. The shared score of 143 exceeds it. Use the floor of 128 to reject a 127-cycle budget, but do not infer that 128 cycles is achievable.

| Batch | Switching | Ideal | Shared score | Shared floor |
|---|---|---:|---:|---:|
| Three nonempty messages | Hop-only | 15 | 15 | 15 |
| Three nonempty messages | Cut-through / store-and-forward | 69 | 143 | 128 |
| One 1,024 B message plus empty long route | Hop-only | 5 | 5 | 5 |
| Same mixed batch | Cut-through / store-and-forward | 69 | 69 | 69 |
| Empty batch or one zero-byte long route | All three | 0 | 0 | 0 |

Previously `shared_link` charged the empty route 15 cycles and inflated the mixed cut-through batch from 69 to 79. A zero-byte item now contributes neither service nor route delay; invalid destinations still fail even on empty payloads. Positive-payload compatibility scores are unchanged. This is estimator behavior, not a new zero-byte send/retirement protocol.

## Design use and costs

A compiler can use ideal time to screen independent paths, shared scores to reproduce historical placement comparisons, and the floor to rule out infeasible budgets. It may not prune schedules using the heuristic or select a replacement router by minimizing the floor. The estimator already enumerates directed links in all modes; the floor reuses those totals with one maximum, adding no modeled router state, payload movement, SRAM/DRAM traffic or numerical change. No new physical performance, area or energy gain follows. A fabric with dedicated paths may avoid sharing at a cost in ports/wires; shared links may save resources but require arbitration and buffering. Sustained throughput, actual latency, area, power and energy for either are **unquantified**. Verification must cover both which term dominates and disjoint maxima; wiring a real scheduler would also require injection timing, arbitration, queues and backpressure.

`make test-icc-traffic-floor` exercises actual JSON parsing, runtime conversion, cluster construction and the estimator. `./test-icc-traffic-floor > matrix.csv` reproduces `evidence/icc-traffic-floor/matrix.csv` (36 rows). The older 12-row traffic and 12-row routing matrices remain in their originating documents, with their score interpretation corrected.
