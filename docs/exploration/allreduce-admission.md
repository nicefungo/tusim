# FP32 all-reduce admission and layout

The functional sum validates the count, participants and both SRAM spans on every core, then reserves scratch before the first metered read. An admission failure returns `-1` without endpoint traffic or collective credit. On success the helper sums in core-ID order, reads all inputs before publishing any output, and credits logical messages and bytes. Broadcast admission policy does not change this contract.

`make test-allreduce-layouts` checks disjoint exact-end placements in both directions, in-place output and partial overlap, on one and three initialized cores. For three two-element inputs `[1,2]`, `[2,4]`, `[3,6]`, every participant receives `[6,12]`; each reads and writes two words. The collective credits two messages and 16 bytes, not the traffic of a simulated reduction tree. Zero charged link cycles do not mean zero latency. No concurrent visibility or recovery after partial completion is modeled.
