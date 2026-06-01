# Multicast/Broadcast DMA Engine (DM4)

> **Status:** Implemented  
> **Date:** 2026-06-01  
> **Gap:** DM4 — Broadcast/Multicast DMA  
> **Files:** `tu_cmodel/dma_descriptor.h`, `tu_cmodel/dma_descriptor.c`, `config/tu_config.json`

## Overview

The Multicast/Broadcast DMA engine extends the existing DMA descriptor system with a 1-to-N transfer type. A single contiguous source (typically host DRAM) is replicated to multiple SRAM destinations in a single DMA descriptor. This is essential for:

- **Weight broadcasting** in multi-core architectures: same weight tile sent to multiple TU cores
- **Activation broadcasting** across PE columns in systolic arrays
- **Constant distribution** (bias terms, scale factors) to multiple SRAM regions

## Why This Matters

In the gap analysis (DM4), broadcast/multicast DMA was identified as a **Medium (P2)** priority feature. It's a general TU property applicable to any systolic array architecture:

1. **Multi-core scaling**: Without broadcast, sending identical weights to N cores requires N separate DMA descriptors, consuming N× bandwidth and N× command queue slots. With multicast, it's one descriptor.

2. **TPU-style weight broadcast**: Google TPU and NVIDIA TensorCores both use weight broadcasting internally. Our cmodel must model this accurately to produce realistic cycle estimates for multi-core configurations.

3. **Pluggable architecture**: The multicast transfer type (`TU_DMA_XFER_MULTICAST`) integrates seamlessly into the existing DMA type system. It's a new transfer type alongside linear, strided, scatter, and gather.

## How It Works

### Transfer Semantics

```
Host DRAM (single source buffer)
     │
     ├──> SRAM Region W, offset 0      (Core 0 weight buffer)
     ├──> SRAM Region A, offset 256    (Core 1 weight buffer)
     ├──> SRAM Region O, offset 512    (Core 2 weight buffer)
     └──> ... N targets
```

Each destination receives an identical copy of the source data.

### Descriptor Structure

```c
// In tu_dma_descriptor_t, new multicast fields:
struct {
    tu_sram_region_t  **regions;    // Array of destination SRAM regions
    uint32_t           *offsets;    // Array of destination offsets
    uint32_t            count;      // Number of multicast targets
} multicast;
```

The descriptor reuses `dims[0]` for element count and `dims[1]` for number of destinations.

### Cycle Accounting

Multicast transfers account for fanout cost:
- `total_bytes = elem_count × elem_size × num_destinations`
- Base latency: `TU_LATENCY_DRAM_READ` (50 cycles)
- Transfer cycles: `total_bytes / TU_DMA_BUS_WIDTH_BYTES`
- This correctly models the N× bandwidth consumption of fanning out to N targets

### Bounds Checking

Each destination is independently bounds-checked. If a destination overflows its SRAM region, a warning is printed and that destination is skipped (other destinations proceed). This is consistent with the DMA engine's approach for other transfer types.

## API

### Constructor

```c
tu_dma_descriptor_t *tu_dma_desc_create_multicast(
    uint8_t channel,
    const void *src_host,           // Source data (host memory)
    tu_sram_region_t **dst_regions, // Array of N destination SRAM regions
    uint32_t *dst_offsets,          // Array of N destination byte offsets
    uint32_t num_destinations,      // Number of targets (N)
    uint32_t elem_size,             // Bytes per element
    uint32_t elem_count);           // Number of elements to write to each target
```

Returns NULL if:
- Any pointer argument is NULL
- `num_destinations` is 0
- Allocation fails

### Destruction

`tu_dma_desc_destroy()` automatically frees the multicast regions and offsets arrays.

### Chaining

Multicast descriptors can be chained with other descriptor types. Example:

```c
// Chain: linear load → multicast broadcast → linear store
tu_dma_desc_chain(linear_load, multicast);
tu_dma_desc_chain(multicast, linear_store);
tu_dma_submit_desc(linear_load);
tu_dma_flush_all();
```

## Configuration

### JSON Config

```json
{
  "dma": {
    "multicast_enabled": true
  }
}
```

When `multicast_enabled` is false, the constructor still works — the feature is always available. The config flag gates whether the ISA-level `DMA.BROADCAST` instruction is recognized by the decoder (future use).

### Performance Counters

A new counter tracks multicast transfers:

```c
tu_dma_counters_t.dma_transfers_multicast  // Count of multicast descriptors executed
```

## ISA Integration

The ISA already defines `TU_ISA_DMA_BROADCAST = 0x57` for future use. When the ISA decoder encounters this opcode, it can construct a multicast descriptor from the instruction's immediate fields.

The ISA encoding reserves `dim1` for the number of destinations and `immediates` for the destination offset list address.

## Verification

### Test Suite

10 tests in `tests/test_multicast.c`:

| Test | Description |
|------|-------------|
| Multicast create | Validates descriptor construction and field values |
| Multicast single target | 1-to-1 copy (degenerate case) |
| Multicast three regions | Writes to 3 distinct SRAM regions |
| Bounds exact fit | Data placed at region boundary |
| Bounds overflow | Graceful skip of overflowed target |
| Null inputs | All failure modes return NULL |
| Destroy frees | Valgrind-clean memory management |
| Performance counter | Correct counter increment |
| 16-target fanout | Large-scale broadcast |
| Chained multicast | Multicast in a descriptor chain |

### Run

```bash
make test-multicast
```

## Tradeoffs & Design Decisions

### Why Not a Separate Engine?

Multicast is implemented as a transfer type within the existing DMA engine, not a separate subsystem. Rationale:

1. **Unified lifecycle**: Same submission, completion, and accounting code
2. **Descriptor chains**: Can mix multicast with other transfer types
3. **Performance counters**: Integrated into existing counter infrastructure
4. **Future extensibility**: Adding `TU_DMA_XFER_MULTICAST_STRIDED` would follow the same pattern

### Fanout Model

The cycle model treats each destination as a concurrent write. In real hardware, a broadcast bus would write to all destinations simultaneously, so the bandwidth cost is N× per-element but with near-zero latency overhead for additional targets. Our model conservatively charges N× full transfer cycles, which is the worst-case sequential model. A future refinement could model a true broadcast bus with 1× cycle cost.

### Limitations

- **No strided multicast**: Only contiguous source → contiguous destinations. Strided multicast (broadcasting into non-contiguous destination patterns) would require a new transfer type.
- **Host-source only**: Multicast currently only supports host→SRAM direction. SRAM→SRAM multicast and SRAM→host broadcast are not supported.
- **No completion signal per-destination**: All targets share one completion signal. Per-destination tracking would require the signal system to track multiple completions.

## Relationship to Other Components

| Component | Relationship |
|-----------|-------------|
| **Multi-core cluster** (`tu_cluster.c`) | `tu_cluster_broadcast()` uses software-level ICC. DMA multicast provides hardware-level fanout. Both are complementary. |
| **Output-stationary dataflow** | OS dataflow broadcasts activations down columns. DMA multicast models the hardware that would perform this distribution. |
| **Scatter/Gather (DM3)** | Scatter writes different elements to different addresses; multicast writes the SAME elements to different addresses. |
| **ISA (DM4)** | `TU_ISA_DMA_BROADCAST = 0x57` is reserved. The ISA decoder can construct multicast descriptors from this opcode. |

## Next Steps

- [ ] Implement `TU_ISA_DMA_BROADCAST` decoder in `tu_isa.c` to construct multicast descriptors from ISA instructions
- [ ] Add strided multicast variant (`TU_DMA_XFER_MULTICAST_STRIDED`) for non-contiguous destination patterns
- [ ] Model true broadcast bus (1× cycle cost for N targets) as an alternative fanout accounting mode
- [ ] Add completion signal tracking per destination
