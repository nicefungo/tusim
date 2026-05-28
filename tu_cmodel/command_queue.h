/*
 * TinyTU Command Queue
 * =====================
 * Hardware command queue with submission, dependency tracking,
 * barrier support, and completion signaling.
 *
 * Gap E1: No command queue → Command queue with ordering & deps.
 *
 * Architecture:
 *   The command queue is a circular buffer of command descriptors.
 *   Each command carries:
 *     - An opcode and operand descriptor
 *     - A list of prerequisite command IDs (dependencies)
 *     - A completion signal that fires when the command retires
 *
 *   Commands execute in submission order but can retire out-of-order
 *   when dependencies allow. Barriers force strict ordering.
 *
 *   In functional mode (TU_CYCLE_MODEL_FUNCTIONAL), commands execute
 *   immediately on submission (synchronous). In estimated/cycle-accurate
 *   modes, execution is deferred and tracked via the queue.
 */

#ifndef TU_COMMAND_QUEUE_H
#define TU_COMMAND_QUEUE_H

#include "tu_config.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Command opcodes ---- */
typedef enum {
    TU_CMD_NOP        = 0,
    TU_CMD_DMA_LOAD   = 1,
    TU_CMD_DMA_STORE  = 2,
    TU_CMD_MMA        = 3,
    TU_CMD_SYNC       = 4,
    TU_CMD_BARRIER    = 5,
    TU_CMD_HALT       = 6,
    /* Reserved for future expansion */
    TU_CMD_CONV       = 10,
    TU_CMD_ATTENTION  = 11,
    TU_CMD_ELEMENTWISE= 12,
    TU_CMD_SOFTMAX    = 13,
    TU_CMD_LAYERNORM  = 14,
    TU_CMD_RMSNORM    = 15,
    TU_CMD_POOL       = 16,
} tu_cmd_opcode_t;

/* ---- Command status ---- */
typedef enum {
    TU_CMD_PENDING    = 0,  /* Waiting for dependencies */
    TU_CMD_ISSUED     = 1,  /* Issued to execution unit */
    TU_CMD_COMPLETED  = 2,  /* Finished successfully */
    TU_CMD_FAULTED    = 3,  /* Errored */
} tu_cmd_status_t;

/* ---- Completion signal ---- */
typedef struct {
    uint32_t signal_id;
    bool     fired;
    uint64_t cycle_completed;
} tu_completion_signal_t;

/* ---- DMA descriptor (embedded in command) ---- */
typedef struct {
    uint8_t   channel;       /* DMA channel (0=W, 1=A, 2=O) */
    bool      is_store;      /* true = store (to host), false = load (from host) */
    uint32_t  sram_offset;   /* Byte offset within SRAM region */
    uint32_t  size_bytes;    /* Transfer size */
    void     *host_ptr;      /* Host-side buffer pointer */
} tu_cmd_dma_desc_t;

/* ---- MMA descriptor (embedded in command) ---- */
typedef struct {
    uint16_t  M, N, K;       /* Matrix dimensions */
    uint32_t  w_offset;      /* Weight buffer offset */
    uint32_t  a_offset;      /* Activation buffer offset */
    uint32_t  o_offset;      /* Output buffer offset */
    bool      has_bias;
} tu_cmd_mma_desc_t;

/* ---- Command descriptor ---- */
typedef struct {
    uint32_t        cmd_id;         /* Unique command ID (monotonically increasing) */
    tu_cmd_opcode_t opcode;
    tu_cmd_status_t status;

    /* Operand descriptor (union of all op types) */
    union {
        tu_cmd_dma_desc_t  dma;
        tu_cmd_mma_desc_t  mma;
    } op;

    /* Dependency tracking */
    uint32_t        num_deps;       /* Number of prerequisite commands */
    uint32_t       *dep_ids;        /* Array of command IDs this depends on */
    uint32_t        max_deps;       /* Allocated size of dep_ids */

    /* Completion */
    uint32_t        signal_id;      /* Completion signal ID (0 = none) */
    uint64_t        cycle_submitted;
    uint64_t        cycle_completed;

    /* Chaining */
    bool            is_barrier;     /* This command acts as a barrier */
} tu_command_t;

/* ---- Command Queue ---- */
typedef struct {
    tu_command_t   *commands;       /* Circular buffer of commands */
    uint32_t        capacity;       /* Max commands in queue */
    uint32_t        head;           /* Next slot to submit to */
    uint32_t        tail;           /* Next slot to retire from */
    uint32_t        count;          /* Number of commands in flight */

    uint32_t        next_cmd_id;    /* Monotonically increasing */
    uint32_t        next_signal_id; /* Monotonically increasing */

    uint64_t        total_submitted;
    uint64_t        total_completed;
    uint64_t        total_faulted;

    /* Completion signal registry */
    tu_completion_signal_t *signals;
    uint32_t        signal_capacity;
    uint32_t        signal_count;

    /* Barrier tracking */
    uint32_t        last_barrier_id; /* Most recent barrier command */

    /* Cycle counter (for estimated/accurate modes) */
    uint64_t        current_cycle;
    bool            synchronous;     /* true = execute immediately (functional mode) */

} tu_command_queue_t;

/* ---- API ---- */

/* Create command queue with given capacity */
tu_command_queue_t *tu_cmdq_create(uint32_t capacity, bool synchronous);

/* Destroy command queue and free all resources */
void tu_cmdq_destroy(tu_command_queue_t *cq);

/* Reset the queue (clear all pending commands, reset counters) */
void tu_cmdq_reset(tu_command_queue_t *cq);

/*
 * Submit a command.
 *
 *   cq:       command queue
 *   opcode:   operation to perform
 *   op:       operand descriptor (type depends on opcode)
 *   num_deps: number of prerequisite command IDs
 *   dep_ids:  array of prerequisite command IDs (NULL if num_deps=0)
 *   cmd_id_out: receives the assigned command ID (NULL if not needed)
 *
 * Returns 0 on success, -1 if queue is full.
 */
int tu_cmdq_submit(tu_command_queue_t *cq,
                   tu_cmd_opcode_t opcode,
                   const void *op_desc,
                   uint32_t num_deps,
                   const uint32_t *dep_ids,
                   uint32_t *cmd_id_out);

/*
 * Submit a barrier. All previously submitted commands must complete
 * before any subsequently submitted commands can issue.
 * Returns barrier command ID, or -1 on error.
 */
int tu_cmdq_barrier(tu_command_queue_t *cq);

/*
 * Wait for a specific command to complete.
 *   cmd_id: the command to wait for
 *   timeout_cycles: max cycles to wait (0 = no timeout)
 * Returns: 0 on completion, -1 on timeout, -2 if cmd_id not found
 */
int tu_cmdq_wait(tu_command_queue_t *cq, uint32_t cmd_id, uint64_t timeout_cycles);

/*
 * Wait for all pending commands to complete (drain the queue).
 */
void tu_cmdq_sync(tu_command_queue_t *cq);

/*
 * Advance the queue: check for dependency-satisfied commands and
 * retire completed ones. In synchronous mode, this is a no-op.
 * In async modes, call this periodically to advance execution.
 * Returns the number of commands retired this tick.
 */
int tu_cmdq_tick(tu_command_queue_t *cq);

/* Get queue depth (number of in-flight commands) */
uint32_t tu_cmdq_get_depth(const tu_command_queue_t *cq);

/* Get command status by ID. Returns TU_CMD_COMPLETED if not found. */
tu_cmd_status_t tu_cmdq_get_status(const tu_command_queue_t *cq, uint32_t cmd_id);

/* Print queue statistics */
void tu_cmdq_print_stats(const tu_command_queue_t *cq);

/* Get total submitted/completed/faulted counts */
void tu_cmdq_get_counts(const tu_command_queue_t *cq,
                        uint64_t *submitted,
                        uint64_t *completed,
                        uint64_t *faulted);

#ifdef __cplusplus
}
#endif

#endif /* TU_COMMAND_QUEUE_H */
