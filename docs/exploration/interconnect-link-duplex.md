# Does reverse-direction traffic share the same link capacity?

The old estimator always counted `a→b` and `b→a` independently. That describes full duplex, but cannot represent a narrower, reversible connection. `tu.multicore.link_duplex` now selects `full` (zero, compatibility) or `half` in the ring/mesh traffic estimator. The YAML generator, shipped JSON, canonical validation, runtime conversion and cluster constructor carry the choice. Unknown strings, non-string JSON and unsupported runtime values are rejected. Consumers embedding the public configuration structs should rebuild.

## Measurement

Four-core ring and 2×2 XY mesh; two simultaneous messages, cut-through, 16 bytes/service cycle, two router cycles/hop. All nonlocal paths below are one hop. Per-packet service rounds up independently. The table shows identical results on both topologies; the [20-row CSV](evidence/icc-duplex/matrix.csv) keeps them separate. Each cell in the last two columns is **peak service / necessary batch floor**, in model cycles. Neither number is an executed schedule.

| Traffic | Longest isolated | Full | Half |
|---|---:|---:|---:|
| Opposed 256 B each on one edge | 18 | 16 / 18 | 32 / 32 |
| Same direction, 256 B each | 18 | 32 / 32 | 32 / 32 |
| Disjoint pairs, 256 B each | 18 | 16 / 18 | 16 / 18 |
| Opposed, 33 B and 1 B | 5 | 3 / 5 | 4 / 5 |
| Self-transfer and empty remote packet | 0 | 0 / 0 | 0 / 0 |

Full duplex takes the maximum directed service; half sums both directions **within each neighbor pair** before taking the maximum. Half reports the bottleneck as sorted IDs, not a winning transmission direction. The floor remains `max(longest isolated, peak service)`; historical `shared_link` still yields a heuristic, and `ideal_parallel` still reports isolated time. Legacy hop-only switching has no serialized bytes. A two-node ring still represents one neighbor pair, not parallel physical cables.

## Design decision

Independent lanes buy simultaneous capacity for exchanges; the example rules out a sub-32-cycle budget on the shared lane, but does not prove full duplex finishes in 18 cycles. For mostly unidirectional phases, a shared lane may save wiring/drivers without changing the measured floor. It requires direction selection, and a physical implementation must verify exclusion, turnaround and fairness. Full duplex may need more wires, ports and switching capacitance; shared lanes may instead spend energy changing direction or waiting. Area, energy, physical latency and sustained throughput are unquantified. Width here is held fixed **per active direction**, not at an equal wire or energy budget. No route, payload, SRAM/DRAM traffic or arithmetic changes. Placement software can screen a layout with the chosen resource contract; no new instruction or compiler integration is claimed. Time-varying lane partitions, queues and turnaround penalties need a direction-ordering/time contract and are not inferred from a batch.

Run `make test-icc-duplex`; `./test-icc-duplex` emits the CSV. The focused gate also checks a bidirectionally shared middle hop on a 2×3 mesh across both duplex choices, all three switching policies and all three estimate modes; it verifies sent bytes and isolated accounting separately. Before wiring the option, the half-duplex opposed and tail controls failed on both topologies.
