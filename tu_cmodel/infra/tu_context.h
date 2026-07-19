/*
 * TU Context Manager — Multi-Context Execution (Gap E3)
 * =====================================================
 *
 * Enables multiple independent execution contexts on a single TU core.
 * Each context holds a complete snapshot of the core's hardware state:
 * SRAM contents, DMA engine state, command queue state, dataflow config,
 * precision/rounding modes, and performance counters.
 *
 * Context switching occurs at synchronization boundaries (BARRIER, SYNC,
 * HALT) or after a configurable number of commands / cycles.
 *
 * Gap: E3 — Multi-context execution (P2)
 * Dependencies: tu_core.h, tu_config.h, tu_status.h
 *
 * Design:
 *   - Config-driven: number of contexts, scheduling policy, time-slice
 *   - Context save/restore is a full snapshot (memcpy the state struct
 *     + deep-copy heap-allocated resources like cmdq buffers)
 *   - Scheduler: round-robin (default) or priority-based
 *   - Thread-safe context switching (no concurrent access to same core)
 */

#ifndef TU_CONTEXT_H
#define TU_CONTEXT_H

#include "../tu_core.h"
#include "config.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Scheduling Policies ---- */
typedef enum {
    TU_CTX_SCHED_ROUND_ROBIN  = 0,  /* Equal time-slices, circular order */
    TU_CTX_SCHED_PRIORITY     = 1,  /* Higher priority = more time-slices */
    TU_CTX_SCHED_COUNT
} tu_ctx_sched_policy_t;

/* FULL is zero so existing aggregate initializers retain historical behavior. */
typedef enum {
    TU_CTX_SAVE_FULL_SRAM    = 0, /* Preserve all W/A/O SRAM */
    TU_CTX_SAVE_LIVE_SRAM    = 1, /* Preserve configured live prefixes only */
    TU_CTX_SAVE_CONTROL_ONLY = 2, /* Preserve control state; software reloads SRAM */
    TU_CTX_SAVE_SCOPE_COUNT
} tu_ctx_save_scope_t;

/* ---- Context State ---- */
typedef enum {
    TU_CTX_IDLE      = 0,  /* Available, not in use */
    TU_CTX_ACTIVE    = 1,  /* Currently executing */
    TU_CTX_READY     = 2,  /* Waiting to be scheduled */
    TU_CTX_BLOCKED   = 3,  /* Waiting on external event (DMA completion, barrier) */
    TU_CTX_COMPLETED = 4,  /* Finished execution (HALT received) */
} tu_ctx_state_t;

/* ---- Context Descriptor ---- */
typedef struct tu_context_desc_t {
    uint32_t            ctx_id;          /* Unique context ID (0-based) */
    tu_ctx_state_t      state;           /* Current execution state */
    uint8_t             priority;        /* 0=lowest, 255=highest (for priority sched) */
    
    /* Saved hardware state */
    tu_state_t          hw_state;        /* Full SRAM/DMA/cmdq snapshot */
    
    /* Execution accounting */
    uint64_t            total_cycles;    /* Total cycles consumed */
    uint64_t            total_commands;  /* Total commands executed */
    uint64_t            switch_count;    /* Number of times this context was switched in */
    uint64_t            last_switch_cycle; /* Cycle counter at last switch-in */
    uint64_t            saved_sram_bytes;  /* Bytes physically retained */
    uint32_t            saved_w_bytes;
    uint32_t            saved_a_bytes;
    uint32_t            saved_o_bytes;
    
    /* Context-local configuration overrides */
    bool                has_config_override;
    tu_config_t         config_override;  /* Per-context config (subset of global) */
    
    /* Client data (for host integration) */
    void               *user_data;       /* Opaque host pointer */
} tu_context_desc_t;

/* ---- Context Manager ---- */
typedef struct tu_ctx_manager_t {
    tu_core_t          *core;            /* Underlying TU core (shared) */
    
    /* Context slots */
    uint32_t            max_contexts;    /* Maximum number of contexts */
    tu_context_desc_t  *contexts;        /* Array of context descriptors */
    uint32_t            active_count;    /* Currently allocated contexts */
    uint32_t            active_ctx_id;   /* Currently executing context ID */
    
    /* Scheduling */
    tu_ctx_sched_policy_t sched_policy;
    uint64_t            time_slice_cycles;  /* Max cycles before forced switch (0=no preemption) */
    uint32_t            time_slice_cmds;    /* Max commands before forced switch (0=no preemption) */
    uint64_t            slice_cycles_used;  /* Cycles used in current slice */
    uint32_t            slice_cmds_used;    /* Commands used in current slice */
    
    /* Statistics */
    uint64_t            total_switches;     /* Total context switches performed */
    uint64_t            total_cycles_stolen;/* Cycles lost to context switch overhead */
    uint64_t            switch_fixed_cycles;/* Pipeline/control save cost */
    uint32_t            state_bytes_per_cycle; /* 0 preserves legacy fixed-only timing */
    tu_ctx_save_scope_t  save_scope;
    uint32_t            live_w_bytes;
    uint32_t            live_a_bytes;
    uint32_t            live_o_bytes;
    uint64_t            pending_save_bytes;
} tu_ctx_manager_t;

/* ---- Config-driven context manager config ---- */
typedef struct {
    uint32_t            max_contexts;        /* Max concurrent contexts (default: 4) */
    tu_ctx_sched_policy_t sched_policy;      /* Scheduling policy (default: round-robin) */
    uint64_t            time_slice_cycles;   /* 0 = switch only at sync points */
    uint32_t            time_slice_cmds;     /* 0 = switch only at sync points */
    uint64_t            switch_overhead;     /* Fixed pipeline/control cycles */
    bool                save_dram_state;     /* Whether to include DRAM in state snapshot */
    tu_ctx_save_scope_t save_scope;          /* Full, live-prefix, or control-only */
    uint32_t            live_w_bytes;        /* LIVE: retained W-buffer prefix */
    uint32_t            live_a_bytes;        /* LIVE: retained A-buffer prefix */
    uint32_t            live_o_bytes;        /* LIVE: retained O-buffer prefix */
    uint32_t            state_bytes_per_cycle; /* Save/restore datapath BW; 0=fixed-only */
} tu_ctx_manager_config_t;

int tu_ctx_manager_config_validate(const tu_core_t *core,
                                   const tu_ctx_manager_config_t *cfg);

/* ================================================================
 * Context Manager API
 * ================================================================ */

/*
 * Create a context manager for a TU core.
 * Allocates context slots and initializes the scheduler.
 * Returns NULL on failure.
 */
tu_ctx_manager_t *tu_ctx_manager_create(tu_core_t *core,
                                         const tu_ctx_manager_config_t *cfg);

/*
 * Destroy the context manager and free all context slots.
 * Does NOT destroy the underlying core.
 */
void tu_ctx_manager_destroy(tu_ctx_manager_t *mgr);

/*
 * Allocate a new execution context.
 * Returns context ID (≥0) on success, -1 if no slots available.
 * The context starts in IDLE state with a snapshot of the current
 * core state as its initial state.
 */
int tu_ctx_alloc(tu_ctx_manager_t *mgr);

/*
 * Free an execution context.
 * Context must be in IDLE or COMPLETED state.
 */
void tu_ctx_free(tu_ctx_manager_t *mgr, uint32_t ctx_id);

/*
 * Get a context descriptor by ID.
 * Returns NULL if ctx_id is invalid.
 */
tu_context_desc_t *tu_ctx_get(tu_ctx_manager_t *mgr, uint32_t ctx_id);

/* ================================================================
 * Context Switching (Core Operations)
 * ================================================================ */

/*
 * Save the currently active context's hardware state.
 * Snapshot: SRAM (W/A/O buffers), DMA engine state, command queue
 * state, dataflow config, precision modes, and performance counters.
 *
 * Called internally before switching away from a context.
 * Exposed for testing and manual context management.
 */
int tu_ctx_save(tu_ctx_manager_t *mgr);

/*
 * Restore a context's saved hardware state into the core.
 * Writes back: SRAM contents, DMA engine state, command queue,
 * dataflow plugin, precision/rounding modes, performance counters.
 *
 * Called internally when switching to a context.
 * Exposed for testing and manual context management.
 */
int tu_ctx_restore(tu_ctx_manager_t *mgr, uint32_t ctx_id);

/*
 * Switch to a specific context.
 * Saves the current context (if any), restores the target context,
 * and updates the scheduler state.
 *
 * Returns 0 on success, non-zero on failure.
 */
int tu_ctx_switch(tu_ctx_manager_t *mgr, uint32_t ctx_id);

/*
 * Trigger a context switch at the next synchronization point.
 * Called by the host to request a context switch after the current
 * operation completes.
 *
 * Returns 0 on success.
 */
int tu_ctx_request_switch(tu_ctx_manager_t *mgr);

/* ================================================================
 * Scheduling
 * ================================================================ */

/*
 * Schedule the next context according to the configured policy.
 * Round-robin: next READY context in circular order.
 * Priority: next READY context with highest priority.
 *
 * If no context is READY, returns -1 (caller should wait).
 */
int tu_ctx_schedule_next(tu_ctx_manager_t *mgr);

/*
 * Check if the current time-slice has expired.
 * Returns true if a context switch is due (time or command count
 * exceeded). The caller should call tu_ctx_schedule_next().
 */
bool tu_ctx_slice_expired(const tu_ctx_manager_t *mgr);

/*
 * Mark the current context as blocked (waiting on external event).
 * Similar to save + set BLOCKED state. The context won't be
 * scheduled again until explicitly unblocked.
 */
int tu_ctx_block_current(tu_ctx_manager_t *mgr);

/*
 * Mark a blocked context as ready.
 * The context becomes eligible for scheduling again.
 */
int tu_ctx_unblock(tu_ctx_manager_t *mgr, uint32_t ctx_id);

/*
 * Notify the scheduler that a command was completed in the active
 * context. Updates the command count for time-slice tracking.
 * Should be called after each command queue submission.
 */
void tu_ctx_notify_command(tu_ctx_manager_t *mgr);

/*
 * Notify the scheduler of elapsed cycles in the active context.
 * Updates the cycle count for time-slice tracking.
 * Should be called periodically (e.g., per-tile or per-DMA-transfer).
 */
void tu_ctx_notify_cycles(tu_ctx_manager_t *mgr, uint64_t cycles);

/* ================================================================
 * Statistics & Debugging
 * ================================================================ */

/*
 * Print a formatted context manager status report.
 * Shows: active context, all context states, scheduling stats.
 */
void tu_ctx_print_status(const tu_ctx_manager_t *mgr, FILE *out);

/*
 * Get the total number of context switches performed.
 */
uint64_t tu_ctx_get_switch_count(const tu_ctx_manager_t *mgr);

/*
 * Get the total cycles lost to context switch overhead.
 */
uint64_t tu_ctx_get_switch_overhead(const tu_ctx_manager_t *mgr);

#ifdef __cplusplus
}
#endif

#endif /* TU_CONTEXT_H */
