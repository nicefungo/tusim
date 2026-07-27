/*
 * TU Core — Single-Core Instance API (Gap A5)
 * ============================================
 *
 * Wrapping the existing tu_state_t into a self-contained tu_core_t
 * with explicit lifecycle management. This is the prerequisite for
 * multi-core clustering: each core owns its own SRAM, DMA engine,
 * command queue, dataflow plugin, and performance counters.
 *
 * Backward-compatible: g_tu and existing tu_*() APIs continue to
 * work when compiled without TU_MULTICORE_ENABLED.
 *
 * Gap: A5 — Multi-instance / multi-core (P1)
 * Dependencies: tu_cmodel.h, tu_config.h
 */

#ifndef TU_CORE_H
#define TU_CORE_H

#include "tu_cmodel.h"
#include "tu_config.h"
#include "compute/dataflow/dataflow_interface.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Forward declaration ---- */
typedef struct tu_core_t tu_core_t;

/* ================================================================
 * tu_core_t — One complete TU compute core
 * ================================================================
 *
 * Each core is a fully independent TU instance with its own:
 *   - SRAM banks (W, A, O) with banking and bandwidth modeling
 *   - DMA engine with descriptor queues
 *   - Command queue with dependency tracking
 *   - Configurable dataflow plugin
 *   - Performance counters and logging
 *   - Runtime configuration
 */
struct tu_core_t {
    /* Core identity */
    uint32_t        core_id;        /* Unique within a cluster (0 = default) */

    /* The core state — owns all hardware resources */
    tu_state_t      state;

    /* Lifecycle */
    bool            initialized;

    /* Inter-core communication buffer (for cluster messaging) */
    void           *icc_buffer;     /* Receive buffer for inter-core messages */
    uint32_t        icc_buffer_size;
};

/* ---- Lifecycle ---- */

/*
 * Create a TU core from a runtime configuration.
 * Allocates and initializes all hardware resources.
 * Returns NULL on failure.
 */
tu_core_t *tu_core_create(const tu_runtime_config_t *cfg);

/*
 * Create a TU core with a specific core ID.
 * The core_id is used for cluster identification.
 */
tu_core_t *tu_core_create_with_id(uint32_t core_id,
                                   const tu_runtime_config_t *cfg);

/*
 * Initialize (or re-initialize) a core.
 * Resets state, zeros SRAM, clears counters.
 */
void tu_core_init(tu_core_t *core);

/*
 * Destroy a core and free all resources.
 */
void tu_core_destroy(tu_core_t *core);

/* ---- Operations ---- */

/*
 * Execute a TU ASM program (text format).
 * Returns 0 on success, non-zero on error.
 */
int tu_core_execute_asm_text(tu_core_t *core,
                              const char *program,
                              const tu_host_buffer_t *buffers,
                              int n_buffers);

/*
 * Wait for all outstanding commands to complete.
 * Drains the command queue.
 */
void tu_core_sync(tu_core_t *core);

/* ---- Subsystem Access (for testing/debugging) ---- */

/* Get the command queue */
tu_command_queue_t *tu_core_get_cmdq(tu_core_t *core);

/* Get the DMA engine */
tu_dma_engine_t *tu_core_get_dma(tu_core_t *core);

/* Get the SRAM region */
tu_sram_region_t *tu_core_get_sram_w(tu_core_t *core);
tu_sram_region_t *tu_core_get_sram_a(tu_core_t *core);
tu_sram_region_t *tu_core_get_sram_o(tu_core_t *core);

/* ---- DMA Convenience (forwarded to core's DMA engine) ---- */

void tu_core_dma_load_w(tu_core_t *core, const void *host_ptr,
                         uint32_t tu_offset, uint32_t size_bytes);
void tu_core_dma_load_a(tu_core_t *core, const void *host_ptr,
                         uint32_t tu_offset, uint32_t size_bytes);
void tu_core_dma_store_o(tu_core_t *core, void *host_ptr,
                          uint32_t tu_offset, uint32_t size_bytes);

/* ---- MMA Convenience (forwarded to core's compute engine) ---- */

/*
 * Select/query the dataflow retained by this core's state snapshot.
 * Selection is isolated from g_tu and from other cores. Unsupported IDs are
 * rejected without changing the active mode.
 */
int tu_core_set_dataflow(tu_core_t *core, tu_dataflow_id_t dataflow_id);
tu_dataflow_id_t tu_core_get_dataflow(const tu_core_t *core);
const char *tu_core_get_dataflow_name(const tu_core_t *core);

void tu_core_mma(tu_core_t *core,
                 uint16_t M, uint16_t N, uint16_t K,
                 uint32_t w_offset, uint32_t a_offset, uint32_t o_offset,
                 bool has_bias);

/* ---- Stats ---- */
void tu_core_print_stats(const tu_core_t *core);

/* ---- Global default core (backward compatibility) ---- */
extern tu_core_t *g_default_core;

/*
 * Get or create the default singleton core.
 * This preserves the legacy g_tu API.
 */
tu_core_t *tu_core_default(void);

#ifdef __cplusplus
}
#endif

#endif /* TU_CORE_H */
