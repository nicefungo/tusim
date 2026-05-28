/*
 * TinyTU CModel — Toy Tensor Unit Functional Model
 * ==================================================
 *
 * Target: RISC-V host with MMIO-mapped TU accelerator.
 * This cmodel is a *functional* model — it computes correct results
 * using the same dataflow semantics as the hardware, but in software.
 * Stats are reported as if running on real silicon.
 *
 * Architecture
 * ------------
 *  ┌──────────────────────────────────────────┐
 *  │  Host Interface (MMIO)                    │
 *  │  CTRL | STATUS | CMD_QUEUE | DMA regs    │
 *  ├──────────────────────────────────────────┤
 *  │  DMA Engine (Host DRAM ↔ SRAM)           │
 *  ├──────────────────────────────────────────┤
 *  │  Scratchpad SRAM — 256 KB total           │
 *  │  ┌──────────┬──────────┬──────────┐      │
 *  │  │ W-Buffer │ A-Buffer │ O-Buffer │      │
 *  │  │  128 KB  │  64 KB   │  64 KB   │      │
 *  │  └──────────┴──────────┴──────────┘      │
 *  ├──────────────────────────────────────────┤
 *  │  16×16 Systolic Array                     │
 *  │  • Weight-stationary dataflow             │
 *  │  • FP16 multiply → FP32 accumulate        │
 *  │  • Round-to-nearest-even on store         │
 *  │  • 16-cycle pipeline fill per tile        │
 *  └──────────────────────────────────────────┘
 *
 * Operation
 * ---------
 *   1. Host DMAs weights & activations into TU SRAM
 *   2. Host issues MMA commands — tiled 16×16×16 GEMM
 *   3. TU streams A through PE array, accumulates into O
 *   4. Host DMAs results back to DRAM
 *
 * Precision
 * ---------
 *   Input/weights:  FP16 (IEEE 754 half, 1-5-10)
 *   Accumulation:   FP32
 *   Output:         FP16 (rounded on store)
 */

#ifndef TINY_TU_CMODEL_H
#define TINY_TU_CMODEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ---- FP16 type (IEEE 754 binary16) ---- */
/* (Delegated to tu_precision.h — included here for backward compat) */
#include "tu_precision.h"
#include "tu_config.h"
#include "tu_sram.h"
#include "tu_dma.h"
#include "command_queue.h"

/* Backward-compat aliases */
static inline fp16_t fp32_to_fp16(fp32_t v) { return tu_fp32_to_fp16(v); }
static inline fp32_t fp16_to_fp32(fp16_t h) { return tu_fp16_to_fp32(h); }

/* ---- TU Configuration ---- */
/* Configuration is driven entirely by tu_config.h.
 * PE array dimensions, SRAM sizes, and all hardware parameters
 * are compile-time configurable with runtime overrides available
 * via tu_runtime_config_t. */

/* ---- TU State ---- */
typedef struct {
    /* SRAM regions (using banked SRAM module) */
    tu_sram_region_t  sram_w;     /* weight buffer */
    tu_sram_region_t  sram_a;     /* activation buffer */
    tu_sram_region_t  sram_o;     /* output buffer */

    /* DMA engine */
    tu_dma_engine_t   dma;        /* DMA engine state */

    /* Command queue */
    tu_command_queue_t *cmdq;     /* Hardware command queue */

    /* Performance counters */
    uint64_t  total_dma_bytes;
    uint64_t  total_mma_calls;
    uint64_t  total_mma_tiles;
    uint64_t  total_mma_flops;    /* effective FP16 multiply-adds */
    uint64_t  estimated_cycles;

    /* Active runtime configuration */
    tu_runtime_config_t  rt_cfg;

    bool      initialized;
} tu_state_t;

/* ---- External TU Instance ---- */
extern tu_state_t g_tu;

/* ---- Re-initialize with a new runtime config ---- */
void tu_init_with_config(const tu_runtime_config_t *cfg);

/* ---- API ---- */

/* Initialize TU — reset SRAM, clear stats. Call once at startup. */
void tu_init(void);

/* Print accumulated performance counters to stderr. */
void tu_print_stats(void);

/*
 * DMA: host DRAM ↔ TU SRAM
 * ------------------------
 * These model DMA transfers. In real HW these would be programmed via
 * MMIO registers and run asynchronously. Here they are synchronous
 * memcpy with stat accounting.
 *
 * tu_dma_load_w : host_ptr → weight_buffer[tu_offset]
 * tu_dma_load_a : host_ptr → activation_buffer[tu_offset]
 * tu_dma_store_o: output_buffer[tu_offset] → host_ptr
 */
void tu_dma_load_w(const void *host_ptr, uint32_t tu_offset, uint32_t size_bytes);
void tu_dma_load_a(const void *host_ptr, uint32_t tu_offset, uint32_t size_bytes);
void tu_dma_store_o(void *host_ptr, uint32_t tu_offset, uint32_t size_bytes);
void tu_dma_load_o(const void *host_ptr, uint32_t tu_offset, uint32_t size_bytes);

/*
 * MMA: Matrix Multiply-Accumulate
 * -------------------------------
 *   O[M][N] += W[M][K] × A[K][N]
 *
 * All operands point into TU SRAM at given offsets.
 * Tiling is handled internally: the systolic array processes 16×16 tiles.
 *
 * Parameters:
 *   M, N, K   — matrix dimensions (in elements, not bytes)
 *   w_offset  — offset of W[M][K] in weight buffer (row-major)
 *   a_offset  — offset of A[K][N] in activation buffer (row-major)
 *   o_offset  — offset of O[M][N] in output buffer (row-major, FP32 accumulators)
 *   bias      — if true, add bias at o_offset before MMA (O = bias first, then accumulate)
 *               bias is FP16, same layout as O[M][N], at o_offset
 *
 * Constraints (asserted):
 *   M, N, K must be multiples of 16, or this is the final tile (bounds checked).
 *   Offsets + sizes must fit within respective SRAM regions.
 */
void tu_mma(uint16_t M, uint16_t N, uint16_t K,
            uint32_t w_offset, uint32_t a_offset, uint32_t o_offset,
            bool has_bias);

/*
 * SYNC: barrier — wait for all pending DMA and MMA to complete.
 * In this functional model it's a no-op (everything is synchronous),
 * but the cycle counter accounts for pipeline drain.
 */
void tu_sync(void);

/*
 * Command Queue API
 * -----------------
 * Submit commands to the hardware command queue instead of calling
 * tu_mma/tu_dma_* directly. Commands are tracked with dependency
 * ordering and completion signaling.
 */

/* Get the TU's command queue (for direct cmdq API access) */
tu_command_queue_t *tu_get_cmdq(void);

/* Submit an MMA command through the queue.
 * Returns command ID (>0) on success, -1 if queue full. */
int tu_cmdq_submit_mma(uint16_t M, uint16_t N, uint16_t K,
                       uint32_t w_offset, uint32_t a_offset, uint32_t o_offset,
                       bool has_bias);

/* Submit a DMA load command. Returns command ID or -1. */
int tu_cmdq_submit_dma_load(uint8_t channel, uint32_t sram_offset,
                            const void *host_ptr, uint32_t size_bytes);

/* Submit a DMA store command. Returns command ID or -1. */
int tu_cmdq_submit_dma_store(uint8_t channel, uint32_t sram_offset,
                             void *host_ptr, uint32_t size_bytes);

/* Submit a barrier. Returns barrier command ID or -1. */
int tu_cmdq_submit_barrier(void);

/* Wait for all submitted commands to complete (drain the queue). */
void tu_cmdq_sync_all(void);

/*
 * Low-level helpers (exposed for direct use by generated code).
 */

/* Convert a host FP32 array to FP16 in-place or to a separate buffer. */
void tu_fp32_to_fp16_buffer(const fp32_t *src, fp16_t *dst, size_t n);

/* Convert TU output (FP32 accumulators) back to FP16. */
void tu_fp32_to_fp16_buffer(const float *src, fp16_t *dst, size_t n);

/* Round FP32 accumulator to FP16 (round-to-nearest-even, clamp to fp16 range). */
fp16_t tu_round_fp32_to_fp16(fp32_t v);

/* Compute number of 16×16 tiles for a dimension. */
static inline uint16_t tu_tiles(uint16_t dim) {
    return (dim + 15) / 16;
}

/*
 * TU ASM Interpreter
 * ------------------
 * Parses and executes a .tuasm program directly.
 *
 * The program is a null-terminated string containing TU ASM instructions.
 * Host buffers are provided as an array of {name, ptr, size} bindings.
 * Weight sections are embedded in the ASM text (%weight ... %endweight).
 *
 * Returns 0 on success, non-zero on parse or execution error.
 */

/* A named host buffer binding for ASM execution. */
typedef struct {
    const char *name;      /* matches %input / %output names in ASM */
    void       *data;      /* buffer pointer */
    uint32_t    size;      /* buffer size in bytes */
} tu_host_buffer_t;

/*
 * Execute a TU ASM program.
 *
 *   program:     null-terminated ASM source
 *   buffers:     array of host buffer bindings
 *   n_buffers:   number of entries in buffers[]
 *
 * On success, output buffers contain results (DMA'd via STORE_O).
 * Stats can be read via tu_print_stats() after execution.
 */
int tu_run_asm(const char *program, const tu_host_buffer_t *buffers, int n_buffers);

#endif /* TINY_TU_CMODEL_H */
