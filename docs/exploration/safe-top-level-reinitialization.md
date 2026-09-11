# Safe Top-Level TU Re-initialization

**Date:** 2026-09-11
**Question:** What lifecycle contract prevents repeated architecture sweeps from leaking host resources or invalidating DMA descriptors that still reference the old SRAM?

## Hypothesis and evidence

`tu_init_with_config()` previously destroyed only the three SRAM regions before clearing `g_tu`. Every initialization allocated a new command queue, including its command array and signal registry, without destroying the prior queue. A prior LeakSanitizer run reported 2,144 bytes leaked by repeated sweep initialization. The same ordering cleared the DMA singleton during its next initialization without first resolving accepted descriptors whose SRAM pointers referred to the regions about to be freed.

Source audit separated the ownership contracts:

- A command queue owns its command slots, per-command dependency arrays, and signal registry; `tu_cmdq_destroy()` is the complete release operation.
- DMA descriptors are caller-owned after successful submission. The DMA engine temporarily links pending descriptors and references one active descriptor per channel, but completed descriptors are not retired by freeing them.
- `tu_dma_flush_all()` executes accepted pending/active descriptors and detaches all engine references without freeing caller-owned descriptors.
- The default top-level command queue is functional/synchronous (`TU_CYCLE_MODEL_FUNCTIONAL`), so accepted commands have executed before re-initialization even though completed records remain in queue storage.

## Lifecycle alternatives considered

| Alternative | Why hardware/software might choose it | Gain | Sacrifice | Status here |
|---|---|---|---|---|
| Drain accepted work | Architectural sweeps and graceful reset need deterministic completion and must not invalidate host/SRAM references | Preserves every accepted DMA transfer and gives a simple ownership boundary | Reconfiguration latency includes remaining DMA service; stale work may be undesirable after an error | **Implemented for DMA before SRAM release** |
| Cancel accepted work | Fault recovery or low-latency context replacement may value reset latency over completion | Bounded reset latency | Requires a canceled terminal state, completion signaling, host-pointer lifetime rules, and caller-visible error retirement; none exist in the current descriptor ABI | **Excluded from this patch; blocked on status/retirement contract** |
| Reject reconfiguration while busy | Strict runtimes can force callers to quiesce explicitly | Avoids implicit work loss and hidden reset latency | Requires a status-returning reconfiguration API; the public `tu_init_with_config()` is `void` | **Deferred; API migration required** |

The implementation does not expose these as hardware runtime modes. Re-initialization is a host lifecycle operation rather than a TU datapath alternative, and only drain has a complete executable contract in the current API. Adding nominal modes without canceled status or a return code would be metadata-only.

## Implemented contract

`tu_shutdown()` is now public and idempotent. It:

1. flushes accepted DMA descriptors while their old SRAM regions are valid;
2. clears DMA engine references without taking ownership of completed descriptors;
3. destroys complete global command-queue storage;
4. destroys W/A/O SRAM; and
5. clears `g_tu`.

`tu_init_with_config()` calls this teardown before allocating the replacement state. Per-core destruction and re-initialization now use `tu_cmdq_destroy()` instead of freeing only the queue's outer struct.

## Executable verification

`tests/test_reinit.c` gates four boundaries:

1. 64 repeated initializations, each with a queue command that allocates a dependency array;
2. two accepted asynchronous SRAM-to-host DMA descriptors that are still pending when re-initialization begins—both complete byte-exactly before old SRAM release;
3. complete per-core queue teardown; and
4. repeated idempotent shutdown with cleared global pointers/state.

Commands:

```sh
make test-reinit
make clean && make \
  CFLAGS='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDFLAGS='-fsanitize=address,undefined -lm' test-reinit
make clean && make && make test-quick
```

The sanitizer run passes 4/4 tests with no AddressSanitizer, UndefinedBehaviorSanitizer, or LeakSanitizer report.

## Multi-objective interpretation

- **Functional correctness:** accepted DMA work is preserved, and descriptors never observe freed SRAM.
- **Throughput:** steady-state compute/DMA throughput is unchanged. This is lifecycle correctness, not a datapath speedup.
- **Reconfiguration latency:** drain latency grows with accepted pending/active DMA service. The current model does not report this as a separate reset counter.
- **Area/resources:** no modeled TU datapath area changes. A physical drain-capable reset controller would need outstanding-work tracking already implied by the DMA queues; incremental gates/state are unquantified.
- **Power/energy:** completed work consumes its normal transfer energy instead of being discarded. Reset/control energy is not modeled.
- **SRAM/DRAM traffic:** accepted traffic completes; no additional payload is introduced. A cancel policy could avoid obsolete traffic but is not representable safely today.
- **Numerical accuracy:** unchanged; the byte-exact DMA gate covers data preservation, not arithmetic.
- **Control complexity:** drain reuses existing flush behavior and is lower-risk than introducing cancellation and error retirement.
- **Verification burden:** lifecycle ordering, ownership, idempotence, and sanitizer checks become permanent gates. Cancellation would require substantially broader signaling and double-free/UAF tests.
- **Compiler/runtime implications:** callers may invoke `tu_shutdown()` explicitly; existing callers retain the `void tu_init_with_config()` API and now receive graceful DMA drain semantics. Runtimes needing cancel/reject must wait for a status-bearing lifecycle API.

## Limitations

- The top-level queue is built in functional mode in this checkout. Reconfiguration semantics for genuinely deferred command-queue execution are not claimed; its completion/signaling/retirement state machine needs a separate audit before enabling non-functional top-level modes.
- The DMA flush is synchronous model execution, not a cycle-by-cycle reset handshake or timeout.
- Caller-owned descriptors remain linked as they were during queueing; callers that free descriptors individually must detach `next`, matching existing tests. A future ownership API should remove this sharp edge.
- Host-thread safety, concurrent reconfiguration, device reset registers, cancellation interrupts, and physical reset power/timing are unmodeled.
