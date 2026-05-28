# TU CModel — DMA Descriptor Engine

> **Gap IDs:** DM1 (Async DMA), DM2 (DMA Descriptors)
> **Priority:** P0 (Critical)
> **Date:** 2026-05-29
> **Heartbeat:** Cycle 3

---

## What Changed

The TinyTU DMA engine was previously synchronous memcpy with flat byte arrays — three functions (`tu_dma_load_w`, `tu_dma_load_a`, `tu_dma_store_o`) that did blocking `memcpy` into SRAM. There were no DMA descriptors, no strided transfers, no chaining, and no async execution support.

A full DMA descriptor engine has been added as `tu_cmodel/dma_descriptor.h` and `tu_cmodel/dma_descriptor.c`. The existing `tu_dma.h` / `tu_dma.c` now delegate to the new engine for backward compatibility.

### Key Features

1. **DMA descriptors** — Structured descriptors with type, direction, geometry, and chaining
2. **Strided transfers** — Linear, 2D strided, and 3D strided transfer patterns
3. **Descriptor chaining** — Linked-list of descriptors executed in sequence
4. **Completion signaling** — Each descriptor gets a signal ID for interrupt generation
5. **Async execution** — Per-channel descriptor queues with `tu_dma_tick()` for cycle-driven execution
6. **Dual mode** — Synchronous (immediate execute) or async (tick-driven)
7. **Per-channel statistics** — Submitted, completed, bytes, and cycles per DMA channel

---

## Why This Matters

DMA descriptors are the foundation for hardware-accurate memory movement modeling:

- **Strided transfers** enable tiled DMA patterns (loading matrix rows with stride, extracting sub-tiles)
- **Descriptor chaining** enables complex multi-step transfers without CPU intervention
- **Async execution** is a prerequisite for DMA/compute overlap and software pipelining (Gap E2)
- **Completion signaling** provides the foundation for interrupt-driven execution
- **Per-channel queues** model real hardware DMA channel behavior with independent queue depths

---

## How It Works

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      tu_dma_engine_t (global)                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Channel 0 (W)     │ Channel 1 (A)     │ Channel 2 (O)       │ │
│  │ ┌───┐ ┌───┐ ┌───┐ │ ┌───┐ ┌───┐      │ ┌───┐               │ │
│  │ │d0 │→│d1 │→│d2 │ │ │d0 │→│d1 │      │ │d0 │               │ │
│  │ └───┘ └───┘ └───┘ │ └───┘ └───┘      │ └───┘               │ │
│  │  head     →  tail  │  head  →  tail   │  head = tail        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  Sync mode: execute immediately on submit                        │
│  Async mode: dispatch via tu_dma_tick() per cycle                │
└──────────────────────────────────────────────────────────────────┘
```

### Descriptor Structure

```c
typedef struct tu_dma_descriptor_t {
    uint32_t                desc_id;        // Unique ID
    tu_dma_transfer_type_t  type;           // LINEAR | STRIDED_2D | STRIDED_3D
    tu_dma_direction_t      direction;      // HOST→TU | TU→HOST | TU→TU

    uint8_t                 channel;        // DMA channel (0=W, 1=A, 2=O)

    tu_sram_region_t       *src_region;     // Source SRAM (NULL = host)
    uint32_t                src_base;       // Base byte offset in source
    uint32_t                src_strides[3]; // Row, depth strides for source

    const void             *src_host;       // Host-side pointer
    tu_sram_region_t       *dst_region;     // Destination SRAM
    uint32_t                dst_base;
    uint32_t                dst_strides[3];
    void                   *dst_host;

    uint32_t                dims[3];        // [depth, rows, cols]
    uint32_t                elem_size;      // Bytes per element
    uint32_t                total_bytes;    // Computed total

    struct tu_dma_descriptor_t *next;       // Chain: next descriptor
    uint32_t                signal_id;      // Completion signal ID
    uint8_t                 priority;       // 0-255

    bool                    completed;       // Set by engine on completion
} tu_dma_descriptor_t;
```

### Transfer Types

| Type | Description | Example Use |
|------|-------------|------------|
| `TU_DMA_XFER_LINEAR` | Contiguous byte range | Loading a dense weight matrix |
| `TU_DMA_XFER_STRIDED_2D` | 2D with row stride | Extracting columns from a row-major matrix |
| `TU_DMA_XFER_STRIDED_3D` | 3D with depth + row strides | Loading a sub-volume from a 3D tensor |

### API Examples

**Linear transfer (host → SRAM):**
```c
tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
    0,                           // channel 0 (W)
    TU_DMA_DIR_HOST_TO_TU,       // direction
    &sram_w, 0,                  // dest SRAM region + offset
    host_data,                   // host pointer
    sizeof(fp16_t),              // element size
    256                          // element count
);
tu_dma_submit_desc(desc);        // Execute (sync) or enqueue (async)
```

**Strided 2D transfer (column extraction):**
```c
// Extract a 4×4 sub-matrix from a 4×8 matrix in SRAM
tu_dma_descriptor_t *desc = tu_dma_desc_create_strided_2d(
    2, TU_DMA_DIR_TU_TO_HOST,
    &sram, 8, host_buf,         // SRAM region + base offset, host dest
    32,                          // SRAM row stride (8 elems × 4 bytes)
    16,                          // Host row stride (4 elems × 4 bytes)
    4,                           // elem_size
    4, 4);                       // rows, cols
tu_dma_submit_desc(desc);
```

**Descriptor chaining:**
```c
tu_dma_descriptor_t *d0 = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram, 0,   buf0, 1, 64);
tu_dma_descriptor_t *d1 = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram, 128, buf1, 1, 64);
tu_dma_descriptor_t *d2 = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU, &sram, 256, buf2, 1, 64);

tu_dma_desc_chain(d0, d1);      // d0 → d1 → d2
tu_dma_desc_chain(d0, d2);

tu_dma_submit_desc(d0);         // Executes all three in order
```

**Async execution:**
```c
tu_dma_init_full(true, 3, 8);  // async=true

tu_dma_submit_desc(desc);       // Enqueued, not executed
for (int i = 0; i < 200; i++) {
    tu_dma_tick();              // Advance one cycle, execute if ready
}
tu_dma_flush_all();             // Drain all channels
```

---

## Configuration

DMA engine parameters in `tu_config.h`:

| #define | Default | Description |
|---------|---------|-------------|
| `TU_DMA_BUS_WIDTH_BITS` | 256 | AXI bus width |
| `TU_DMA_BUS_WIDTH_BYTES` | 32 | Derived: bus width in bytes |
| `TU_DMA_MAX_BURST_BYTES` | 64 | Max burst size |
| `TU_DMA_CHANNELS` | 3 | Number of DMA channels |
| `TU_DMA_MAX_OUTSTANDING` | 4 | Max queue depth per channel |
| `TU_DMA_ASYNC_MODE` | 0 | 0=sync, 1=async |
| `TU_LATENCY_DRAM_READ` | 50 | DRAM read latency cycles |
| `TU_LATENCY_DRAM_WRITE` | 50 | DRAM write latency cycles |

---

## Verification

### Test Suite: 10 tests, all passing

| Test | What It Verifies |
|------|-----------------|
| Linear host→SRAM | 128-byte linear transfer, byte-exact match |
| Linear SRAM→host | 256-byte linear transfer, byte-exact match |
| Strided 2D | Extract 4×4 column from 4×8 matrix, correct striding |
| Strided 3D | Extract 2×2×2 cube from 4×4×4 volume |
| Descriptor chain | 3 linked descriptors execute in order |
| Async mode | Queue → tick → verify data not loaded until ticked |
| FP32 matrix | 16×16 FP32 matrix (1024 bytes), exact match |
| Legacy API | `tu_dma_load` / `tu_dma_store` backward compat |
| Completion signal | Signal ID assigned and descriptor marked complete |
| Transfer stats | Byte and transfer counters are correct |

### Run Tests

```bash
make test-dma    # 10/10 tests pass
```

---

## Backward Compatibility

All existing code using the legacy API continues to work:

```c
tu_dma_load_w(host_ptr, offset, size);   // Still works
tu_dma_load_a(host_ptr, offset, size);   // Still works
tu_dma_store_o(host_ptr, offset, size);  // Still works
```

These functions now delegate to the new descriptor engine internally. No changes needed to existing tests or generated code.

---

## Future Extensions

This DMA descriptor engine provides the foundation for:

- **Gap DM3 (Scatter/gather):** Add `TU_DMA_XFER_SCATTER` and `TU_DMA_XFER_GATHER` with index lists
- **Gap DM4 (Broadcast/multicast):** Add `TU_DMA_XFER_BROADCAST` for 1-to-N transfers
- **Gap E2 (Software pipelining):** DMA tile N+1 while computing tile N — descriptor chaining + async execution
- **Gap M3 (Address generation):** Hardware im2col patterns via strided 3D transfers

---

## Files

| File | Change |
|------|--------|
| `tu_cmodel/dma_descriptor.h` | New — DMA descriptor types, engine API, strided transfers |
| `tu_cmodel/dma_descriptor.c` | New — Implementation: construction, execution, chaining, async tick |
| `tu_cmodel/tu_dma.h` | Reduced — now delegates to dma_descriptor.h |
| `tu_cmodel/tu_dma.c` | Reduced — thin backward-compat shim |
| `Makefile` | Added `dma_descriptor.o` and `test-dma` target |
| `tests/test_dma.c` | New — 10 DMA descriptor engine tests |
| `docs/dma-descriptor-engine.md` | This document |
