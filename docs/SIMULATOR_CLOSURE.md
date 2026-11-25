# Simulator closure: November 19–25, 2026

Seven increments extend the functional model and its accounting from base `7a094ff`. Defaults remain native FP32 requests, ideal parallel links, XY mesh order, clockwise ring ties, full duplex and sequential broadcast. Rebuild C consumers: the public descriptor, configuration and cluster structures grew.

| Date | Change | What the focused regression distinguishes |
|---|---|---|
| Nov 19 | [Softmax SRAM requests](exploration/softmax-sram-request-granularity.md): native word, sized element or contiguous row | Same FP32 result and useful bytes; different metered word counts and local penalties. |
| Nov 20 | [ICC traffic floor](exploration/interconnect-traffic-floor.md): separate a necessary time from the historical shared-link score | On disjoint maxima, isolated time is 69, the heuristic is 143 and the floor is 128 model cycles. A floor is not a feasible schedule. |
| Nov 21 | [Ragged mesh](exploration/interconnect-mesh-routing-order.md): direct send and estimators agree on missing turns | Each strict order admits 23 of 25 pairs on the five-router mesh; fallback admits all 25 without shortening a path. |
| Nov 22 | [Ring ties](exploration/interconnect-contention-traffic-matrix.md): choose a direction at injection for equal shortest paths | Source-parity steering can split a shared link without changing hop count or isolated cost. |
| Nov 23 | [Duplex](exploration/interconnect-link-duplex.md): meter directions independently or share service on each edge | Opposed 256-byte flows raise the floor from 18 to 32 cycles in half duplex; disjoint flows stay at 18. |
| Nov 24 | [Broadcast](exploration/interconnect-broadcast-admission.md): sequential versus preflight admission | An unavailable route leaves a delivered prefix or no changes, respectively. Valid routes deliver identical bytes. |
| Nov 25 | [FP32 all-reduce](exploration/allreduce-admission.md): admit all participants and reserve scratch before reading | Valid disjoint, in-place and overlapping layouts preserve the sum and logical traffic accounting. |

## Verification boundary

The complete engineering snapshot was previously recovered and tested against this base. This public selection retains the implementation but uses narrower functional regressions. Its checks are the seven focused targets below plus generated-configuration consistency; it does not inherit a full-suite or sanitizer pass from the recovered snapshot.

```sh
make test-softmax-sram-access test-icc-traffic-floor test-icc-ragged-mesh \
     test-icc-ring-ties test-icc-duplex test-icc-broadcast-admission \
     test-allreduce-layouts test-config-generation
```

The targeted run passed: 280 admitted softmax cases, 36 traffic-floor rows, 244 mesh mode/pair checks, 32 ring-tie rows, 20 duplex rows plus 18 multihop combinations, four broadcast routing cases, and eight valid all-reduce layouts. The four published CSVs matched freshly generated output byte for byte.

These are simulator results, not hardware qualification. Link service and local penalties do not establish elapsed latency, power or sustained throughput. Queueing and routing progress, SRAM grant ownership and replay, padded DMA retirement, normalization scratch and BatchNorm semantics, and recovery after partial collective completion remain separate contracts. The optional ONNX demo is outside this validation.
