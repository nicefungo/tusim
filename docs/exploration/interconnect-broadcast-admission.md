# Broadcast admission

`tu.multicore.broadcast_admission` selects when a serial broadcast validates its destinations. `sequential` (default) issues individual sends in core-ID order and retains a completed prefix on failure. `preflight` plans every send and reserves scratch before the first copy. Once admitted, both issue the same copies in the same order. Neither provides concurrent visibility, multicast timing, or rollback after issue.

The [four routing cases](evidence/icc-broadcast-admission/matrix.csv) use five endpoints on a two-row mesh. With strict XY, two destinations are reachable before the first missing turn. Sequential delivery changes those two endpoints (8 bytes, 17 charged cycles); preflight changes none. With `xy_else_yx` all four peers receive the value (16 bytes, 39 charged cycles) under either admission policy. These are sums of individual send charges, not elapsed time.

`make test-icc-broadcast-admission` checks endpoint images, read/write counts, cluster accounting and destination cycle charges without modifying SRAM geometry. The contract assumes stable initialized participants during a call.
