/*
 * TU Context Manager Implementation (Gap E3)
 * ===========================================
 *
 * Multi-context execution: save/restore full core state, schedule
 * multiple independent execution contexts on a single TU core.
 *
 * Context switch procedure:
 *   1. Drain in-flight operations (flush DMA, sync command queue)
 *   2. Save active context: SRAM data, DMA stats, cmdq state, perf counters
 *   3. Restore next context: write back saved state
 *   4. Resume execution from where the restored context left off
 */

#include "tu_context.h"
#include "../tu_status.h"
#include "../tu_precision.h"
#include "../rounding.h"
#include "../perf/performance_counters.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Internal: Deep-copy SRAM region data ---- */
static int ctx_copy_sram_data(tu_sram_region_t *dst, const tu_sram_region_t *src,
                              uint32_t copy_bytes)
{
    if (!dst || !src || !src->banks.data) return -1;

    /* Allocate only bytes retained by this policy; metadata still describes
     * the physical region capacity. */
    dst->banks.data = NULL;
    if (copy_bytes > 0) {
        dst->banks.data = (uint8_t *)malloc(copy_bytes);
        if (!dst->banks.data) return -1;
        memcpy(dst->banks.data, src->banks.data, copy_bytes);
    }

    /* Copy bank metadata */
    dst->banks.size        = src->banks.size;
    dst->banks.bank_count  = src->banks.bank_count;
    dst->banks.bank_width  = src->banks.bank_width;
    dst->banks.reads       = src->banks.reads;
    dst->banks.writes      = src->banks.writes;
    dst->banks.conflicts   = src->banks.conflicts;
    dst->banks.stall_cycles = src->banks.stall_cycles;

    /* Bandwidth state: skip per-bank bw_banks (recreated on restore) */
    dst->banks.bw_banks = NULL;
    dst->banks.bw_modeling       = src->banks.bw_modeling;
    dst->banks.words_per_cycle   = src->banks.words_per_cycle;
    dst->banks.arb_mode          = src->banks.arb_mode;
    dst->banks.stall_penalty     = src->banks.stall_penalty;
    dst->banks.bw_refill_window  = src->banks.bw_refill_window;
    dst->banks.current_cycle     = src->banks.current_cycle;

    return 0;
}

/* ---- Internal: Free context-local SRAM data ---- */
static void ctx_free_sram_data(tu_sram_region_t *r)
{
    if (r && r->banks.data) {
        free(r->banks.data);
        r->banks.data = NULL;
    }
    if (r && r->banks.bw_banks) {
        free(r->banks.bw_banks);
        r->banks.bw_banks = NULL;
    }
}

/* ---- Internal: Save full hardware state from core into context ---- */
static uint32_t ctx_scope_bytes(tu_ctx_save_scope_t scope, uint32_t live,
                                uint32_t capacity)
{
    if (scope == TU_CTX_SAVE_FULL_SRAM) return capacity;
    if (scope == TU_CTX_SAVE_LIVE_SRAM) return live;
    return 0;
}

static int ctx_save_full_state(tu_ctx_manager_t *mgr, tu_context_desc_t *ctx,
                               tu_state_t *state)
{
    tu_sram_region_t *sw = &state->sram_w;
    tu_sram_region_t *sa = &state->sram_a;
    tu_sram_region_t *so = &state->sram_o;

    tu_sram_region_t *dw = &ctx->hw_state.sram_w;
    tu_sram_region_t *da = &ctx->hw_state.sram_a;
    tu_sram_region_t *d_o = &ctx->hw_state.sram_o;

    /* Free previous save data if re-saving */
    ctx_free_sram_data(dw);
    ctx_free_sram_data(da);
    ctx_free_sram_data(d_o);

    ctx->saved_w_bytes = ctx_scope_bytes(mgr->save_scope, mgr->live_w_bytes,
                                         sw->banks.size);
    ctx->saved_a_bytes = ctx_scope_bytes(mgr->save_scope, mgr->live_a_bytes,
                                         sa->banks.size);
    ctx->saved_o_bytes = ctx_scope_bytes(mgr->save_scope, mgr->live_o_bytes,
                                         so->banks.size);
    ctx->saved_sram_bytes = (uint64_t)ctx->saved_w_bytes + ctx->saved_a_bytes +
                            ctx->saved_o_bytes;

    if (ctx_copy_sram_data(dw, sw, ctx->saved_w_bytes) != 0) return -1;
    if (ctx_copy_sram_data(da, sa, ctx->saved_a_bytes) != 0) { ctx_free_sram_data(dw); return -1; }
    if (ctx_copy_sram_data(d_o, so, ctx->saved_o_bytes) != 0) { ctx_free_sram_data(dw); ctx_free_sram_data(da); return -1; }

    /* Copy region names (pointers to static strings, safe) */
    dw->name = sw->name;
    da->name = sa->name;
    d_o->name = so->name;

    /* Double-buffer state: skip for now (disable on restore) */
    dw->db = NULL;
    da->db = NULL;
    d_o->db = NULL;

    /* Save SRAM total sizes */
    dw->total_size = sw->total_size;
    da->total_size = sa->total_size;
    d_o->total_size = so->total_size;

    /* Save DMA engine state (struct copy, no heap pointers to deep-copy) */
    memcpy(&ctx->hw_state.dma, &state->dma, sizeof(tu_dma_engine_t));

    /* Save command queue: struct copy only (cmdq is a pointer, save the
     * pointer value for later re-creation — the actual buffer is heap
     * allocated and will be recreated on restore via tu_cmdq_create) */
    ctx->hw_state.cmdq = state->cmdq;

    /* Save dataflow plugin pointer (restore will re-select by name) */
    ctx->hw_state.dataflow = state->dataflow;

    /* Save performance counters */
    ctx->hw_state.total_dma_bytes  = state->total_dma_bytes;
    ctx->hw_state.total_mma_calls  = state->total_mma_calls;
    ctx->hw_state.total_mma_tiles  = state->total_mma_tiles;
    ctx->hw_state.total_mma_flops  = state->total_mma_flops;
    ctx->hw_state.estimated_cycles = state->estimated_cycles;

    /* Save runtime config */
    memcpy(&ctx->hw_state.rt_cfg, &state->rt_cfg, sizeof(tu_runtime_config_t));

    ctx->hw_state.initialized = state->initialized;

    return 0;
}

/* ---- Internal: Restore saved hardware state from context into core ---- */
static int ctx_restore_full_state(tu_state_t *state, const tu_context_desc_t *ctx)
{
    const tu_sram_region_t *sw = &ctx->hw_state.sram_w;
    const tu_sram_region_t *sa = &ctx->hw_state.sram_a;
    const tu_sram_region_t *so = &ctx->hw_state.sram_o;

    tu_sram_region_t *rw = &state->sram_w;
    tu_sram_region_t *ra = &state->sram_a;
    tu_sram_region_t *ro = &state->sram_o;

    /* Core SRAM stays physically allocated. Restore only retained bytes;
     * CONTROL_ONLY and non-live tails must be reloaded by software. */
    if (ctx->saved_w_bytes)
        memcpy(rw->banks.data, sw->banks.data, ctx->saved_w_bytes);
    if (ctx->saved_a_bytes)
        memcpy(ra->banks.data, sa->banks.data, ctx->saved_a_bytes);
    if (ctx->saved_o_bytes)
        memcpy(ro->banks.data, so->banks.data, ctx->saved_o_bytes);

    /* Restore bank metadata */
    rw->banks.size        = sw->banks.size;
    rw->banks.bank_count  = sw->banks.bank_count;
    rw->banks.bank_width  = sw->banks.bank_width;
    rw->banks.reads       = sw->banks.reads;
    rw->banks.writes      = sw->banks.writes;
    rw->banks.conflicts   = sw->banks.conflicts;
    rw->banks.stall_cycles = sw->banks.stall_cycles;
    rw->banks.bw_modeling       = sw->banks.bw_modeling;
    rw->banks.words_per_cycle   = sw->banks.words_per_cycle;
    rw->banks.arb_mode          = sw->banks.arb_mode;
    rw->banks.stall_penalty     = sw->banks.stall_penalty;
    rw->banks.bw_refill_window  = sw->banks.bw_refill_window;
    rw->banks.current_cycle     = sw->banks.current_cycle;
    rw->total_size = sw->total_size;
    rw->name = sw->name;
    rw->db = NULL;  /* Double-buffering not restored across context switches */

    ra->banks.size        = sa->banks.size;
    ra->banks.bank_count  = sa->banks.bank_count;
    ra->banks.bank_width  = sa->banks.bank_width;
    ra->banks.reads       = sa->banks.reads;
    ra->banks.writes      = sa->banks.writes;
    ra->banks.conflicts   = sa->banks.conflicts;
    ra->banks.stall_cycles = sa->banks.stall_cycles;
    ra->banks.bw_modeling       = sa->banks.bw_modeling;
    ra->banks.words_per_cycle   = sa->banks.words_per_cycle;
    ra->banks.arb_mode          = sa->banks.arb_mode;
    ra->banks.stall_penalty     = sa->banks.stall_penalty;
    ra->banks.bw_refill_window  = sa->banks.bw_refill_window;
    ra->banks.current_cycle     = sa->banks.current_cycle;
    ra->total_size = sa->total_size;
    ra->name = sa->name;
    ra->db = NULL;

    ro->banks.size        = so->banks.size;
    ro->banks.bank_count  = so->banks.bank_count;
    ro->banks.bank_width  = so->banks.bank_width;
    ro->banks.reads       = so->banks.reads;
    ro->banks.writes      = so->banks.writes;
    ro->banks.conflicts   = so->banks.conflicts;
    ro->banks.stall_cycles = so->banks.stall_cycles;
    ro->banks.bw_modeling       = so->banks.bw_modeling;
    ro->banks.words_per_cycle   = so->banks.words_per_cycle;
    ro->banks.arb_mode          = so->banks.arb_mode;
    ro->banks.stall_penalty     = so->banks.stall_penalty;
    ro->banks.bw_refill_window  = so->banks.bw_refill_window;
    ro->banks.current_cycle     = so->banks.current_cycle;
    ro->total_size = so->total_size;
    ro->name = so->name;
    ro->db = NULL;

    /* Restore DMA engine state */
    memcpy(&state->dma, &ctx->hw_state.dma, sizeof(tu_dma_engine_t));

    /* Restore command queue (keep existing pointer, but update state)
     * The cmdq buffer contents are not preserved across switch — the
     * context was drained before save. Keep the existing pointer. */
    /* state->cmdq stays as-is (already allocated by core) */

    /* Restore dataflow plugin */
    state->dataflow = ctx->hw_state.dataflow;

    /* Restore performance counters */
    state->total_dma_bytes  = ctx->hw_state.total_dma_bytes;
    state->total_mma_calls  = ctx->hw_state.total_mma_calls;
    state->total_mma_tiles  = ctx->hw_state.total_mma_tiles;
    state->total_mma_flops  = ctx->hw_state.total_mma_flops;
    state->estimated_cycles = ctx->hw_state.estimated_cycles;

    /* Restore runtime config */
    memcpy(&state->rt_cfg, &ctx->hw_state.rt_cfg, sizeof(tu_runtime_config_t));

    state->initialized = ctx->hw_state.initialized;

    return 0;
}

/* ================================================================
 * Public API
 * ================================================================ */

tu_ctx_manager_t *tu_ctx_manager_create(tu_core_t *core,
                                         const tu_ctx_manager_config_t *cfg)
{
    if (tu_ctx_manager_config_validate(core, cfg) != 0) return NULL;

    tu_ctx_manager_t *mgr = (tu_ctx_manager_t *)calloc(1, sizeof(tu_ctx_manager_t));
    if (!mgr) return NULL;

    mgr->core = core;
    mgr->max_contexts = cfg->max_contexts;
    mgr->sched_policy = cfg->sched_policy;
    mgr->time_slice_cycles = cfg->time_slice_cycles;
    mgr->time_slice_cmds = cfg->time_slice_cmds;
    mgr->switch_fixed_cycles = cfg->switch_overhead;
    mgr->state_bytes_per_cycle = cfg->state_bytes_per_cycle;
    mgr->save_scope = cfg->save_scope;
    mgr->live_w_bytes = cfg->live_w_bytes;
    mgr->live_a_bytes = cfg->live_a_bytes;
    mgr->live_o_bytes = cfg->live_o_bytes;

    /* Allocate context descriptors */
    mgr->contexts = (tu_context_desc_t *)calloc(mgr->max_contexts,
                                                 sizeof(tu_context_desc_t));
    if (!mgr->contexts) {
        free(mgr);
        return NULL;
    }

    /* Initialize all contexts to IDLE */
    for (uint32_t i = 0; i < mgr->max_contexts; i++) {
        mgr->contexts[i].ctx_id = i;
        mgr->contexts[i].state = TU_CTX_IDLE;
        mgr->contexts[i].priority = 128;  /* Default middle priority */
    }

    mgr->active_count = 0;
    mgr->total_switches = 0;
    mgr->total_cycles_stolen = 0;

    return mgr;
}

int tu_ctx_manager_config_validate(const tu_core_t *core,
                                   const tu_ctx_manager_config_t *cfg)
{
    if (!core || !cfg || cfg->max_contexts == 0 ||
        (int)cfg->sched_policy < 0 || cfg->sched_policy >= TU_CTX_SCHED_COUNT ||
        (int)cfg->save_scope < 0 || cfg->save_scope >= TU_CTX_SAVE_SCOPE_COUNT) return -1;
    if (cfg->save_scope == TU_CTX_SAVE_LIVE_SRAM &&
        (cfg->live_w_bytes > core->state.sram_w.banks.size ||
         cfg->live_a_bytes > core->state.sram_a.banks.size ||
         cfg->live_o_bytes > core->state.sram_o.banks.size)) return -1;
    return 0;
}

void tu_ctx_manager_destroy(tu_ctx_manager_t *mgr)
{
    if (!mgr) return;

    /* Free context-local SRAM data for each context */
    for (uint32_t i = 0; i < mgr->max_contexts; i++) {
        tu_context_desc_t *ctx = &mgr->contexts[i];
        if (ctx->state != TU_CTX_IDLE) {
            ctx_free_sram_data(&ctx->hw_state.sram_w);
            ctx_free_sram_data(&ctx->hw_state.sram_a);
            ctx_free_sram_data(&ctx->hw_state.sram_o);
        }
    }

    free(mgr->contexts);
    free(mgr);
}

int tu_ctx_alloc(tu_ctx_manager_t *mgr)
{
    if (!mgr) return -1;

    /* Find an IDLE slot */
    for (uint32_t i = 0; i < mgr->max_contexts; i++) {
        if (mgr->contexts[i].state == TU_CTX_IDLE) {
            tu_context_desc_t *ctx = &mgr->contexts[i];
            memset(ctx, 0, sizeof(tu_context_desc_t));
            ctx->ctx_id = i;
            ctx->state = TU_CTX_READY;
            ctx->priority = 128;

            /* Snapshot current core state as initial context state */
            if (ctx_save_full_state(mgr, ctx, &mgr->core->state) != 0) {
                ctx->state = TU_CTX_IDLE;
                return -1;
            }

            /* If this is the first context, make it active */
            if (mgr->active_count == 0) {
                ctx->state = TU_CTX_ACTIVE;
                mgr->active_ctx_id = i;
            }

            mgr->active_count++;
            return (int)i;
        }
    }

    return -1;  /* No free slots */
}

void tu_ctx_free(tu_ctx_manager_t *mgr, uint32_t ctx_id)
{
    if (!mgr || ctx_id >= mgr->max_contexts) return;

    tu_context_desc_t *ctx = &mgr->contexts[ctx_id];

    /* If freeing the active context, save it first to release hardware */
    if (ctx->state == TU_CTX_ACTIVE) {
        tu_ctx_save(mgr);
    }

    /* Now free any context that's not still ACTIVE (shouldn't happen after save) */
    if (ctx->state == TU_CTX_ACTIVE) return;

    /* Free any context in READY, BLOCKED, COMPLETED, or IDLE state */
    ctx_free_sram_data(&ctx->hw_state.sram_w);
    ctx_free_sram_data(&ctx->hw_state.sram_a);
    ctx_free_sram_data(&ctx->hw_state.sram_o);
    memset(ctx, 0, sizeof(tu_context_desc_t));
    ctx->ctx_id = ctx_id;
    ctx->state = TU_CTX_IDLE;
    if (mgr->active_count > 0) mgr->active_count--;
}

tu_context_desc_t *tu_ctx_get(tu_ctx_manager_t *mgr, uint32_t ctx_id)
{
    if (!mgr || ctx_id >= mgr->max_contexts) return NULL;

    tu_context_desc_t *ctx = &mgr->contexts[ctx_id];
    if (ctx->state == TU_CTX_IDLE) return NULL;
    return ctx;
}

/* ================================================================
 * Context Switching
 * ================================================================ */

int tu_ctx_save(tu_ctx_manager_t *mgr)
{
    if (!mgr) return -1;

    uint32_t active_id = mgr->active_ctx_id;
    if (active_id >= mgr->max_contexts) return -1;

    tu_context_desc_t *ctx = &mgr->contexts[active_id];
    if (ctx->state != TU_CTX_ACTIVE) return -1;

    /* Drain in-flight operations before saving */
    tu_core_sync(mgr->core);

    /* Save the hardware state */
    int ret = ctx_save_full_state(mgr, ctx, &mgr->core->state);
    if (ret != 0) return ret;
    mgr->pending_save_bytes = ctx->saved_sram_bytes;

    /* Move context to READY (or COMPLETED) */
    ctx->state = TU_CTX_READY;
    ctx->total_cycles += mgr->core->state.estimated_cycles - ctx->last_switch_cycle;

    return 0;
}

int tu_ctx_restore(tu_ctx_manager_t *mgr, uint32_t ctx_id)
{
    if (!mgr || ctx_id >= mgr->max_contexts) return -1;

    tu_context_desc_t *ctx = &mgr->contexts[ctx_id];
    if (ctx->state != TU_CTX_READY) return -1;

    /* Restore the hardware state */
    int ret = ctx_restore_full_state(&mgr->core->state, ctx);
    if (ret != 0) return ret;

    /* Mark as active */
    ctx->state = TU_CTX_ACTIVE;
    ctx->switch_count++;
    ctx->last_switch_cycle = mgr->core->state.estimated_cycles;
    mgr->active_ctx_id = ctx_id;

    /* Account for fixed drain/control cost plus save and restore traffic. */
    mgr->total_switches++;
    uint64_t transfer_bytes = mgr->pending_save_bytes + ctx->saved_sram_bytes;
    uint64_t transfer_cycles = 0;
    if (mgr->state_bytes_per_cycle > 0)
        transfer_cycles = (transfer_bytes + mgr->state_bytes_per_cycle - 1) /
                          mgr->state_bytes_per_cycle;
    mgr->total_cycles_stolen += mgr->switch_fixed_cycles + transfer_cycles;
    mgr->pending_save_bytes = 0;

    /* Reset time-slice counters */
    mgr->slice_cycles_used = 0;
    mgr->slice_cmds_used = 0;

    return 0;
}

int tu_ctx_switch(tu_ctx_manager_t *mgr, uint32_t ctx_id)
{
    if (!mgr) return -1;

    /* Save current context (if any is active) */
    uint32_t active_id = mgr->active_ctx_id;
    if (active_id < mgr->max_contexts &&
        mgr->contexts[active_id].state == TU_CTX_ACTIVE) {
        int ret = tu_ctx_save(mgr);
        if (ret != 0) return ret;
    }

    /* Restore target context */
    return tu_ctx_restore(mgr, ctx_id);
}

int tu_ctx_request_switch(tu_ctx_manager_t *mgr)
{
    if (!mgr) return -1;

    int next_id = tu_ctx_schedule_next(mgr);
    if (next_id < 0) return -1;

    return tu_ctx_switch(mgr, (uint32_t)next_id);
}

/* ================================================================
 * Scheduling
 * ================================================================ */

int tu_ctx_schedule_next(tu_ctx_manager_t *mgr)
{
    if (!mgr) return -1;

    switch (mgr->sched_policy) {
    case TU_CTX_SCHED_ROUND_ROBIN: {
        /* Start search from after the current active context */
        uint32_t start = (mgr->active_ctx_id + 1) % mgr->max_contexts;
        for (uint32_t i = 0; i < mgr->max_contexts; i++) {
            uint32_t id = (start + i) % mgr->max_contexts;
            if (mgr->contexts[id].state == TU_CTX_READY) {
                return (int)id;
            }
        }
        break;
    }
    case TU_CTX_SCHED_PRIORITY: {
        int best_id = -1;
        uint8_t best_prio = 0;
        for (uint32_t i = 0; i < mgr->max_contexts; i++) {
            if (mgr->contexts[i].state == TU_CTX_READY &&
                mgr->contexts[i].priority > best_prio) {
                best_prio = mgr->contexts[i].priority;
                best_id = (int)i;
            }
        }
        return best_id;
    }
    default:
        break;
    }

    return -1;  /* No ready context */
}

bool tu_ctx_slice_expired(const tu_ctx_manager_t *mgr)
{
    if (!mgr) return false;

    if (mgr->time_slice_cycles > 0 &&
        mgr->slice_cycles_used >= mgr->time_slice_cycles) {
        return true;
    }

    if (mgr->time_slice_cmds > 0 &&
        mgr->slice_cmds_used >= mgr->time_slice_cmds) {
        return true;
    }

    return false;
}

int tu_ctx_block_current(tu_ctx_manager_t *mgr)
{
    if (!mgr) return -1;

    int ret = tu_ctx_save(mgr);
    if (ret != 0) return ret;

    uint32_t active_id = mgr->active_ctx_id;
    if (active_id < mgr->max_contexts) {
        mgr->contexts[active_id].state = TU_CTX_BLOCKED;
    }

    return 0;
}

int tu_ctx_unblock(tu_ctx_manager_t *mgr, uint32_t ctx_id)
{
    if (!mgr || ctx_id >= mgr->max_contexts) return -1;

    if (mgr->contexts[ctx_id].state == TU_CTX_BLOCKED) {
        mgr->contexts[ctx_id].state = TU_CTX_READY;
    }

    return 0;
}

void tu_ctx_notify_command(tu_ctx_manager_t *mgr)
{
    if (!mgr) return;
    mgr->slice_cmds_used++;
}

void tu_ctx_notify_cycles(tu_ctx_manager_t *mgr, uint64_t cycles)
{
    if (!mgr) return;
    mgr->slice_cycles_used += cycles;
}

/* ================================================================
 * Statistics & Debugging
 * ================================================================ */

void tu_ctx_print_status(const tu_ctx_manager_t *mgr, FILE *out)
{
    if (!mgr || !out) return;

    fprintf(out, "═══════════════════════════════════════════\n");
    fprintf(out, "  TU Context Manager Status\n");
    fprintf(out, "═══════════════════════════════════════════\n");
    fprintf(out, "  Max contexts:     %u\n", mgr->max_contexts);
    fprintf(out, "  Active contexts:  %u\n", mgr->active_count);
    fprintf(out, "  Active ctx ID:    %u\n", mgr->active_ctx_id);
    fprintf(out, "  Total switches:   %lu\n",
            (unsigned long)mgr->total_switches);
    fprintf(out, "  Switch overhead:  %lu cycles\n",
            (unsigned long)mgr->total_cycles_stolen);
    fprintf(out, "  Sched policy:     %s\n",
            mgr->sched_policy == TU_CTX_SCHED_ROUND_ROBIN ? "round-robin" :
            mgr->sched_policy == TU_CTX_SCHED_PRIORITY ? "priority" : "unknown");
    fprintf(out, "  Time-slice:       %lu cyc / %u cmds\n",
            (unsigned long)mgr->time_slice_cycles, mgr->time_slice_cmds);

    fprintf(out, "\n  Contexts:\n");
    fprintf(out, "  %4s  %-12s  %8s  %10s  %10s  %8s\n",
            "ID", "State", "Priority", "Cycles", "Commands", "Switches");

    for (uint32_t i = 0; i < mgr->max_contexts; i++) {
        const tu_context_desc_t *ctx = &mgr->contexts[i];
        const char *state_str = "???";
        switch (ctx->state) {
            case TU_CTX_IDLE:      state_str = "IDLE"; break;
            case TU_CTX_ACTIVE:    state_str = "ACTIVE"; break;
            case TU_CTX_READY:     state_str = "READY"; break;
            case TU_CTX_BLOCKED:   state_str = "BLOCKED"; break;
            case TU_CTX_COMPLETED: state_str = "COMPLETED"; break;
        }

        fprintf(out, "  %4u  %-12s  %8u  %10lu  %10lu  %8lu\n",
                ctx->ctx_id, state_str, ctx->priority,
                (unsigned long)ctx->total_cycles,
                (unsigned long)ctx->total_commands,
                (unsigned long)ctx->switch_count);
    }

    fprintf(out, "═══════════════════════════════════════════\n");
}

uint64_t tu_ctx_get_switch_count(const tu_ctx_manager_t *mgr)
{
    return mgr ? mgr->total_switches : 0;
}

uint64_t tu_ctx_get_switch_overhead(const tu_ctx_manager_t *mgr)
{
    return mgr ? mgr->total_cycles_stolen : 0;
}
