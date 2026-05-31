/*
 * TU Core — Single-Core Instance Implementation (Gap A5)
 * =======================================================
 */

#include "tu_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Global default core (backward compatibility) ---- */
tu_core_t *g_default_core = NULL;

/* ---- Lifecycle ---- */

tu_core_t *tu_core_create_with_id(uint32_t core_id,
                                   const tu_runtime_config_t *cfg) {
    tu_core_t *core = calloc(1, sizeof(tu_core_t));
    if (!core) return NULL;

    core->core_id = core_id;

    /* Initialize the core's state from config.
     * We reuse tu_init_with_config by temporarily swapping g_tu. */
    tu_state_t saved;
    memcpy(&saved, &g_tu, sizeof(tu_state_t));

    tu_init_with_config(cfg);

    /* Move the initialized state into our core */
    memcpy(&core->state, &g_tu, sizeof(tu_state_t));
    core->initialized = true;

    /* Restore previous g_tu */
    memcpy(&g_tu, &saved, sizeof(tu_state_t));

    TU_LOG_INFO(TU_COMP_CORE, "tu_core_t [%u] created", core_id);

    return core;
}

tu_core_t *tu_core_create(const tu_runtime_config_t *cfg) {
    return tu_core_create_with_id(0, cfg);
}

void tu_core_init(tu_core_t *core) {
    if (!core) return;

    /* Destroy previous state if initialized */
    if (core->initialized) {
        tu_sram_destroy(&core->state.sram_w);
        tu_sram_destroy(&core->state.sram_a);
        tu_sram_destroy(&core->state.sram_o);
        free(core->state.cmdq);
        free(core->icc_buffer);
    }

    tu_state_t saved;
    memcpy(&saved, &g_tu, sizeof(tu_state_t));

    tu_init();

    /* Move initialized state */
    memcpy(&core->state, &g_tu, sizeof(tu_state_t));
    core->initialized = true;

    /* Restore */
    memcpy(&g_tu, &saved, sizeof(tu_state_t));

    TU_LOG_INFO(TU_COMP_CORE, "tu_core_t [%u] re-initialized", core->core_id);
}

void tu_core_destroy(tu_core_t *core) {
    if (!core) return;

    if (core->initialized) {
        tu_sram_destroy(&core->state.sram_w);
        tu_sram_destroy(&core->state.sram_a);
        tu_sram_destroy(&core->state.sram_o);
        free(core->state.cmdq);
        core->initialized = false;
    }

    free(core->icc_buffer);

    TU_LOG_INFO(TU_COMP_CORE, "tu_core_t [%u] destroyed", core->core_id);

    if (core == g_default_core) {
        g_default_core = NULL;
    }

    free(core);
}

/* ---- Default core ---- */

tu_core_t *tu_core_default(void) {
    if (!g_default_core) {
        tu_runtime_config_t cfg = tu_config_default();
        g_default_core = tu_core_create(&cfg);
    }
    return g_default_core;
}

/* ---- State swap helper ---- */

static void core_swap_in(tu_core_t *core, tu_state_t *saved) {
    memcpy(saved, &g_tu, sizeof(tu_state_t));
    memcpy(&g_tu, &core->state, sizeof(tu_state_t));
}

static void core_swap_out(tu_core_t *core, tu_state_t *saved) {
    memcpy(&core->state, &g_tu, sizeof(tu_state_t));
    memcpy(&g_tu, saved, sizeof(tu_state_t));
}

/* ---- Operations ---- */

int tu_core_execute_asm_text(tu_core_t *core,
                              const char *program,
                              const tu_host_buffer_t *buffers,
                              int n_buffers) {
    if (!core || !core->initialized) return -1;

    tu_state_t saved;
    core_swap_in(core, &saved);

    int result = tu_run_asm(program, buffers, n_buffers);

    core_swap_out(core, &saved);
    return result;
}

void tu_core_sync(tu_core_t *core) {
    if (!core || !core->initialized) return;

    tu_state_t saved;
    core_swap_in(core, &saved);

    tu_cmdq_sync_all();

    core_swap_out(core, &saved);
}

/* ---- Subsystem Access ---- */

tu_command_queue_t *tu_core_get_cmdq(tu_core_t *core) {
    return core ? core->state.cmdq : NULL;
}

tu_dma_engine_t *tu_core_get_dma(tu_core_t *core) {
    return core ? &core->state.dma : NULL;
}

tu_sram_region_t *tu_core_get_sram_w(tu_core_t *core) {
    return core ? &core->state.sram_w : NULL;
}

tu_sram_region_t *tu_core_get_sram_a(tu_core_t *core) {
    return core ? &core->state.sram_a : NULL;
}

tu_sram_region_t *tu_core_get_sram_o(tu_core_t *core) {
    return core ? &core->state.sram_o : NULL;
}

/* ---- DMA Convenience ---- */

void tu_core_dma_load_w(tu_core_t *core, const void *host_ptr,
                         uint32_t tu_offset, uint32_t size_bytes) {
    if (!core || !core->initialized) return;
    tu_state_t saved;
    core_swap_in(core, &saved);
    tu_dma_load_w(host_ptr, tu_offset, size_bytes);
    core_swap_out(core, &saved);
}

void tu_core_dma_load_a(tu_core_t *core, const void *host_ptr,
                         uint32_t tu_offset, uint32_t size_bytes) {
    if (!core || !core->initialized) return;
    tu_state_t saved;
    core_swap_in(core, &saved);
    tu_dma_load_a(host_ptr, tu_offset, size_bytes);
    core_swap_out(core, &saved);
}

void tu_core_dma_store_o(tu_core_t *core, void *host_ptr,
                          uint32_t tu_offset, uint32_t size_bytes) {
    if (!core || !core->initialized) return;
    tu_state_t saved;
    core_swap_in(core, &saved);
    tu_dma_store_o(host_ptr, tu_offset, size_bytes);
    core_swap_out(core, &saved);
}

/* ---- MMA Convenience ---- */

void tu_core_mma(tu_core_t *core,
                 uint16_t M, uint16_t N, uint16_t K,
                 uint32_t w_offset, uint32_t a_offset, uint32_t o_offset,
                 bool has_bias) {
    if (!core || !core->initialized) return;
    tu_state_t saved;
    core_swap_in(core, &saved);
    tu_mma(M, N, K, w_offset, a_offset, o_offset, has_bias);
    core_swap_out(core, &saved);
}

/* ---- Stats ---- */

void tu_core_print_stats(const tu_core_t *core) {
    if (!core || !core->initialized) return;

    tu_state_t saved;
    memcpy(&saved, &g_tu, sizeof(tu_state_t));
    memcpy(&g_tu, &core->state, sizeof(tu_state_t));

    printf("===== TU Core [%u] Stats =====\n", core->core_id);
    tu_print_stats();

    memcpy(&g_tu, &saved, sizeof(tu_state_t));
}
