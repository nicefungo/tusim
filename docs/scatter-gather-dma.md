# TU CModel — Scatter/Gather DMA Engine

> **Gap ID:** DM3 (Scatter/gather DMA)
> **Priority:** P1 (High)
> **Date:** 2026-06-02
> **Heartbeat:** evening shift

---

## What Changed

The DMA engine now supports scatter and gather transfer patterns via index lists, enabling sparse data movement — a critical capability for sparse weight loading, embedding table lookups, and gathering scattered partial results.

### Key Features

1. **Scatter (1-to-N):** Copies contiguous host data to scattered SRAM locations specified by an index list
2. **Gather (N-to-1):** Copies scattered SRAM locations (indexed) into a contiguous host buffer
3. **Arbitrary element sizes:** Supports 1-byte (`uint8_t`) through 8-byte (`double`) and beyond via `elem_size`
4. **Non-contiguous offsets:** Index lists are arbitrary byte offsets — no alignment or spacing constraints
5. **Sparse preservation:** Only indexed locations are modified; untouched SRAM bytes remain intact
6. **Zero-copy semantics:** Index list is referenced, not copied; caller retains ownership
7. **Full DMA integration:** Channel routing, cycle accounting, completion signaling, and performance counters all work transparently
8. **Round-trip consistency:** Scatter → Gather produces identical data for any valid index list

---

## Why This Matters

Scatter/gather DMA is essential for production-grade accelerators:

- **Sparse weight loading:** Load nonzero weights at scattered SRAM addresses from a compressed dense buffer
- **Embedding lookups:** Gather embedding vectors at token-specific indices from a large embedding table
- **Indirect tensor access:** Indexed reads/writes for gather/scatter in attention, beam search, and dynamic routing
- **Sparse output assembly:** Scatter partial results from tiled computation into their final output positions
- **Hardware accuracy:** Real DMA engines (NVMe, RDMA, GPU copy engines) support scatter/gather lists; modeling them is essential for performance fidelity

### Relationship to Sparsity

Scatter/gather DMA is the data-movement counterpart to structured sparsity (P2.1). Where 2:4 sparsity prunes weights to 50% density, scatter DMA enables loading only the surviving nonzero elements without touching zeroed positions. Together, they enable end-to-end sparse inference:

```
Compressed weights (dense, nonzeros only)
    │
    ▼  scatter DMA (index list = nonzero positions)
Systolic Array SRAM (scattered weights at computed offsets)
    │
    ▼  compute
Partial results (scattered in output SRAM)
    │
    ▼  gather DMA (index list = output positions)
Dense host output
```

---

## How It Works

### Scatter (Host → SRAM)

```
Host (contiguous):  [A][B][C][D]
Index list:          [0, 128, 256, 512]
                          │    │    │    │
                          ▼    ▼    ▼    ▼
SRAM (scattered):    [A]...[B]...[C]...[D]
                     offset 0    128   256   512
```

**Algorithm:**
```c
for i in 0..index_count:
    memcpy(sram_base + index_list[i], host_base + i * elem_size, elem_size)
```

- Each element `i` from the contiguous host buffer is copied to SRAM at `index_list[i]`
- Only the `elem_size` bytes at each target offset are touched
- Non-targeted SRAM locations are unaffected

### Gather (SRAM → Host)

```
SRAM (scattered):    [A]...[B]...[C]...[D]
Index list:          [0, 128, 256, 512]
                          │    │    │    │
                          ▼    ▼    ▼    ▼
Host (contiguous):   [A][B][C][D]
```

**Algorithm:**
```c
for i in 0..index_count:
    memcpy(host_base + i * elem_size, sram_base + index_list[i], elem_size)
```

- Each element at `index_list[i]` in SRAM is copied to position `i` in the contiguous host buffer
- Elements are packed densely in order of the index list

---

## API Reference

### Scatter Descriptor Creation

```c
tu_dma_descriptor_t *tu_dma_desc_create_scatter(
    uint8_t channel,                    // DMA channel (0=W, 1=A, 2=O)
    tu_sram_region_t *dst_region,       // Target SRAM region
    const void *src_host,               // Contiguous source data (host)
    const uint32_t *index_list,         // Array of destination byte offsets
    uint32_t elem_count,                // Number of elements
    uint32_t elem_size);                // Bytes per element
```

**Behavior:**
- Creates a `TU_DMA_XFER_SCATTER` descriptor with `TU_DMA_DIR_HOST_TO_TU` direction
- `index_list` is borrowed (caller retains ownership — must outlive the descriptor)
- `total_bytes` = `elem_count * elem_size`
- `index_count` = `elem_count`
- Returns `NULL` on allocation failure

### Gather Descriptor Creation

```c
tu_dma_descriptor_t *tu_dma_desc_create_gather(
    uint8_t channel,                    // DMA channel (0=W, 1=A, 2=O)
    tu_sram_region_t *src_region,       // Source SRAM region
    void *dst_host,                     // Destination buffer (host)
    const uint32_t *index_list,         // Array of source byte offsets
    uint32_t elem_count,                // Number of elements
    uint32_t elem_size);                // Bytes per element
```

**Behavior:**
- Creates a `TU_DMA_XFER_GATHER` descriptor with `TU_DMA_DIR_TU_TO_HOST` direction
- Data flows from scattered SRAM locations → contiguous host buffer
- Same index list borrowing semantics as scatter

### Execution

```c
void tu_dma_execute_desc(tu_dma_descriptor_t *desc);
```

Scatter and gather are synchronous (immediate execute) when `async_mode=false`. They use the standard `tu_dma_execute_desc` path with full cycle accounting, bandwidth modeling, and completion signaling.

---

## Performance Model

Scatter/gather transfers contribute to the standard DMA performance counters:

| Counter | Meaning |
|---------|---------|
| `dma_transfers_scatter` | Number of scatter transfers executed |
| `dma_transfers_gather` | Number of gather transfers executed |
| `total_bytes` | Bytes transferred (elem_count × elem_size) |
| `estimated_cycles` | Base latency + (total_bytes / bus_width) + SRAM stall cycles |

Cycle accounting is identical to linear transfers: one base DRAM latency plus bus-width-scaled transfer cycles, with per-bank SRAM bandwidth contention where enabled.

**Key insight:** Scatter/gather can cause more bank conflicts than linear transfers because indices may target the same SRAM bank repeatedly. The bandwidth model captures this correctly — each element is an independent word access that consumes one word from the target bank's budget.

---

## Usage Examples

### Scatter: Loading sparse weights

```c
/* Compressed dense buffer: 3 nonzero weights at positions 5, 12, 27 */
float dense_weights[3] = {0.5f, -0.3f, 0.8f};
uint32_t weight_positions[3] = {5 * sizeof(float), 12 * sizeof(float), 27 * sizeof(float)};

tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
    TU_DMA_CHAN_W,      /* weight channel */
    &g_tu.sram_w,       /* weight SRAM */
    dense_weights,
    weight_positions,
    3,                  /* 3 nonzero elements */
    sizeof(float));

tu_dma_submit_desc(desc);
/* Weights at SRAM offsets 20, 48, 108 are now populated */
```

### Gather: Extracting embedding vectors

```c
/* Embedding table in SRAM with 1000 vectors of 64 floats */
/* Gather vectors for token IDs [42, 7, 999] */

uint32_t gather_indices[3];
float gathered_embeddings[3 * 64];  /* 3 vectors of 64 floats */

/* Compute byte offsets for each token's embedding vector */
for (int i = 0; i < 3; i++) {
    gather_indices[i] = token_ids[i] * 64 * sizeof(float);
}

tu_dma_descriptor_t *desc = tu_dma_desc_create_gather(
    TU_DMA_CHAN_A,      /* activation channel */
    &g_tu.sram_a,       /* activation SRAM (where embeddings live) */
    gathered_embeddings,
    gather_indices,
    3 * 64,             /* 3 vectors × 64 floats */
    sizeof(float));
tu_dma_submit_desc(desc);
```

### Scatter+gather round-trip (sparse sparsification)

```c
/* Moves selected elements from src to dst through SRAM */
/* This pattern is common in attention: gather Q/K/V slices, compute, scatter output */

/* Step 1: Scatter source data to SRAM */
tu_dma_descriptor_t *s = tu_dma_desc_create_scatter(
    ch, &region, src_data, write_indices, N, elem_sz);
tu_dma_execute_desc(s);
tu_dma_desc_destroy(s);

/* Step 2: Compute on the scattered data (external function) */
compute_on_sram_slice(&region, write_indices, N);

/* Step 3: Gather results back to host */
tu_dma_descriptor_t *g = tu_dma_desc_create_gather(
    ch, &region, dst_data, read_indices, N, elem_sz);
tu_dma_execute_desc(g);
tu_dma_desc_destroy(g);
```

---

## Edge Cases & Safety

| Case | Behavior |
|------|----------|
| **Empty index list** (count=0) | Descriptor created normally; execution is a no-op; total_bytes=0 |
| **NULL index list** with count=0 | Valid — produces empty descriptor |
| **NULL index list** with count>0 | Undefined behavior — caller must ensure valid list |
| **Index out of SRAM bounds** | Not checked in scatter/gather path; caller must ensure indices are valid |
| **Overlapping indices** | Allowed — later writes overwrite earlier ones (last-write-wins) |
| **Non-aligned indices** | Allowed — byte-level memcpy, no alignment requirements |
| **Host ↔ SRAM direction** | Scatter is always host→SRAM; Gather is always SRAM→host |
| **Descriptor chaining** | Supported via `tu_dma_desc_chain()` — scatter/gather can appear anywhere in a chain |

---

## Configuration

Scatter/gather is always available — no feature flag is needed. The existing DMA and SRAM configuration controls apply:

```yaml
# tu_config.yaml (relevant sections)
dma:
  channels: 3               # Channels available for scatter/gather
  async_mode: false         # false = synchronous (immediate execute)

memory:
  sram:
    banking:
      conflict_model: "detect"  # Bank conflicts tracked during scatter/gather
```

For performance counter integration, scatter (type=3) and gather (type=4) transfers are automatically classified in `tu_perf_dma_record_read/write`.

---

## Relationship to Other Gaps

| Gap | Relationship |
|-----|-------------|
| **DM1 (Async DMA)** | Scatter/gather use the same async submission path as linear transfers |
| **DM2 (DMA Descriptors)** | Scatter/gather are transfer types within the unified descriptor system |
| **DM4 (Multicast)** | Different pattern: multicast is 1→N with identical data; scatter is 1→N with positional placement |
| **M2 (Bandwidth modeling)** | Per-element bank access enables accurate conflict modeling for sparse address patterns |
| **P2.1 (Structured sparsity)** | Scatter DMA is the transport layer for loading compressed sparse weights |
| **E2 (Software pipelining)** | Scatter/gather descriptors can be pipelined with compute via double buffering |
| **O3 (Attention)** | Gather enables embedding lookups; scatter enables sparse attention output assembly |

---

## Verification

**Test file:** `tests/test_scatter_gather.c` (15 tests)

| # | Test | What It Verifies |
|---|------|-----------------|
| 1 | Scatter descriptor creation | Correct type, direction, count, element size |
| 2 | Scatter sequential offsets | Elements placed at expected sequential positions |
| 3 | Scatter sparse offsets | Elements at non-contiguous positions; untargeted bytes untouched |
| 4 | Scatter single element | Single-element scatter with double type |
| 5 | Scatter empty index list | No-op execution, zero total_bytes |
| 6 | Scatter cycle accounting | Byte and transfer counters increment correctly |
| 7 | Gather descriptor creation | Correct type, direction, dst_host pointer |
| 8 | Gather sequential offsets | Elements read from sequential SRAM positions |
| 9 | Gather sparse offsets | Elements read from sparse SRAM positions |
| 10 | Gather single element | Single-element gather with float |
| 11 | Gather cycle accounting | Byte and transfer counters valid after gather |
| 12 | Scatter+gather round-trip | 16 int16 elements round-trip with 64-byte stride |
| 13 | Scatter large types | double (8-byte) elements including ±1e308 |
| 14 | Scatter 1-byte elements | uint8_t elements, non-targeted bytes verified untouched |
| 15 | Channel routing | All 3 DMA channels accept scatter descriptors |
