/*
 * TinyTU Command Queue — Implementation
 * =======================================
 *
 * Circular-buffer command queue with dependency tracking and
 * barrier support. In synchronous (functional) mode, commands
 * execute immediately on submission. In async modes, execution
 * is deferred and advanced via tu_cmdq_tick().
 */
#include "command_queue.h"
#include "tu_cmodel.h"
#include "compute/elementwise_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Internal helpers
 * ================================================================ */

static bool deps_satisfied(const tu_command_queue_t *cq, const tu_command_t *cmd) {
    for (uint32_t i = 0; i < cmd->num_deps; i++) {
        uint32_t dep_id = cmd->dep_ids[i];
        /* Scan backwards for the dependency */
        bool found = false;
        for (uint32_t j = 0; j < cq->capacity; j++) {
            tu_command_t *dep = &cq->commands[j];
            if (dep->cmd_id == dep_id) {
                if (dep->status != TU_CMD_COMPLETED) return false;
                found = true;
                break;
            }
        }
        /* If dep not found in queue, assume it already completed */
        if (!found) continue;
    }
    return true;
}

static bool barrier_clear(const tu_command_queue_t *cq) {
    /* Check if any earlier barrier is still pending */
    /* A barrier blocks all commands submitted after it */
    for (uint32_t i = 0; i < cq->capacity; i++) {
        tu_command_t *cmd = &cq->commands[i];
        if (cmd->cmd_id == 0) continue;
        if (cmd->is_barrier && cmd->status != TU_CMD_COMPLETED) {
            return false;
        }
    }
    return true;
}

/* Execute a single command (called from submit in sync mode, or tick in async mode) */
static void execute_command(tu_command_queue_t *cq, tu_command_t *cmd) {
    cmd->status = TU_CMD_ISSUED;
    cmd->cycle_submitted = cq->current_cycle;

    switch (cmd->opcode) {
    case TU_CMD_NOP:
        break;

    case TU_CMD_DMA_LOAD:
        if (cmd->op.dma.is_store) {
            tu_dma_store_o(cmd->op.dma.host_ptr, cmd->op.dma.sram_offset,
                          cmd->op.dma.size_bytes);
        } else {
            if (cmd->op.dma.channel == 0)
                tu_dma_load_w(cmd->op.dma.host_ptr, cmd->op.dma.sram_offset,
                             cmd->op.dma.size_bytes);
            else if (cmd->op.dma.channel == 1)
                tu_dma_load_a(cmd->op.dma.host_ptr, cmd->op.dma.sram_offset,
                             cmd->op.dma.size_bytes);
            else
                tu_dma_load_o(cmd->op.dma.host_ptr, cmd->op.dma.sram_offset,
                             cmd->op.dma.size_bytes);
        }
        break;

    case TU_CMD_DMA_STORE:
        tu_dma_store_o(cmd->op.dma.host_ptr, cmd->op.dma.sram_offset,
                      cmd->op.dma.size_bytes);
        break;

    case TU_CMD_MMA:
        tu_mma(cmd->op.mma.M, cmd->op.mma.N, cmd->op.mma.K,
               cmd->op.mma.w_offset, cmd->op.mma.a_offset,
               cmd->op.mma.o_offset, cmd->op.mma.has_bias);
        break;

    case TU_CMD_SYNC:
        tu_sync();
        break;

    case TU_CMD_BARRIER:
        /* No-op — barriers only affect scheduling, not execution */
        break;

    case TU_CMD_HALT:
        break;

    case TU_CMD_ELEMENTWISE: {
        /* Select SRAM region */
        tu_sram_region_t *sram;
        switch (cmd->op.ew.sram_region) {
        case 1: sram = &g_tu.sram_w; break;
        case 2: sram = &g_tu.sram_a; break;
        default: sram = &g_tu.sram_o; break;
        }

        /* Build op array from packed descriptors */
        tu_ew_op_t ops[TU_EW_MAX_OPS];
        uint8_t scalar_idx = 0;
        for (uint8_t i = 0; i < cmd->op.ew.num_ops && i < TU_EW_MAX_OPS; i++) {
            ops[i].opcode = (tu_ew_opcode_t)cmd->op.ew.ops[i];
            if (scalar_idx < 4 && cmd->op.ew.has_scalar[scalar_idx]) {
                ops[i].has_scalar = true;
                ops[i].scalar = cmd->op.ew.scalars[scalar_idx];
                scalar_idx++;
            } else {
                ops[i].has_scalar = false;
                ops[i].scalar = 0.0f;
            }
        }

        tu_ew_apply_fused(sram, cmd->op.ew.sram_offset,
                          cmd->op.ew.elem_count, ops, cmd->op.ew.num_ops);
        break;
    }

    default:
        fprintf(stderr, "TU CMDQ: unknown opcode %d\n", cmd->opcode);
        cmd->status = TU_CMD_FAULTED;
        cq->total_faulted++;
        return;
    }

    cmd->status = TU_CMD_COMPLETED;
    cmd->cycle_completed = cq->current_cycle;
    cq->total_completed++;

    /* Fire completion signal if any */
    if (cmd->signal_id > 0) {
        for (uint32_t i = 0; i < cq->signal_count; i++) {
            if (cq->signals[i].signal_id == cmd->signal_id) {
                cq->signals[i].fired = true;
                cq->signals[i].cycle_completed = cq->current_cycle;
                break;
            }
        }
    }
}

/* ================================================================
 * Public API
 * ================================================================ */

tu_command_queue_t *tu_cmdq_create(uint32_t capacity, bool synchronous) {
    tu_command_queue_t *cq = calloc(1, sizeof(*cq));
    if (!cq) return NULL;

    if (capacity < 4) capacity = 4;
    cq->capacity = capacity;
    cq->commands = calloc(capacity, sizeof(tu_command_t));
    cq->synchronous = synchronous;
    cq->next_cmd_id = 1;  /* Start at 1 so 0 means "no command" */
    cq->next_signal_id = 1;

    /* Pre-allocate signal registry */
    cq->signal_capacity = capacity;
    cq->signals = calloc(cq->signal_capacity, sizeof(tu_completion_signal_t));

    return cq;
}

void tu_cmdq_destroy(tu_command_queue_t *cq) {
    if (!cq) return;
    /* Free per-command dependency arrays */
    for (uint32_t i = 0; i < cq->capacity; i++) {
        free(cq->commands[i].dep_ids);
    }
    free(cq->commands);
    free(cq->signals);
    free(cq);
}

void tu_cmdq_reset(tu_command_queue_t *cq) {
    if (!cq) return;
    for (uint32_t i = 0; i < cq->capacity; i++) {
        free(cq->commands[i].dep_ids);
        memset(&cq->commands[i], 0, sizeof(tu_command_t));
    }
    cq->head = cq->tail = cq->count = 0;
    cq->next_cmd_id = 1;
    cq->total_submitted = cq->total_completed = cq->total_faulted = 0;
    cq->current_cycle = 0;
    cq->last_barrier_id = 0;
    cq->signal_count = 0;
}

int tu_cmdq_submit(tu_command_queue_t *cq,
                   tu_cmd_opcode_t opcode,
                   const void *op_desc,
                   uint32_t num_deps,
                   const uint32_t *dep_ids,
                   uint32_t *cmd_id_out) {
    if (!cq) return -1;
    if (cq->count >= cq->capacity) return -1; /* Queue full */

    uint32_t slot = cq->head;
    tu_command_t *cmd = &cq->commands[slot];

    /* Free old dependency array if reusing slot */
    free(cmd->dep_ids);
    memset(cmd, 0, sizeof(*cmd));

    cmd->cmd_id = cq->next_cmd_id++;
    cmd->opcode = opcode;
    cmd->status = TU_CMD_PENDING;
    cmd->is_barrier = (opcode == TU_CMD_BARRIER);

    /* Copy operand descriptor */
    if (op_desc) {
        switch (opcode) {
        case TU_CMD_DMA_LOAD:
        case TU_CMD_DMA_STORE:
            memcpy(&cmd->op.dma, op_desc, sizeof(tu_cmd_dma_desc_t));
            break;
        case TU_CMD_MMA:
            memcpy(&cmd->op.mma, op_desc, sizeof(tu_cmd_mma_desc_t));
            break;
        case TU_CMD_ELEMENTWISE:
            memcpy(&cmd->op.ew, op_desc, sizeof(tu_cmd_ew_desc_t));
            break;
        default:
            break;
        }
    }

    /* Copy dependencies */
    cmd->num_deps = num_deps;
    if (num_deps > 0 && dep_ids) {
        cmd->max_deps = num_deps;
        cmd->dep_ids = malloc(num_deps * sizeof(uint32_t));
        memcpy(cmd->dep_ids, dep_ids, num_deps * sizeof(uint32_t));
    }

    /* Assign completion signal */
    cmd->signal_id = cq->next_signal_id++;

    /* Track barrier */
    if (cmd->is_barrier) cq->last_barrier_id = cmd->cmd_id;

    /* Advance head (circular) */
    cq->head = (cq->head + 1) % cq->capacity;
    cq->count++;
    cq->total_submitted++;

    if (cmd_id_out) *cmd_id_out = cmd->cmd_id;

    /* In synchronous mode, execute immediately.
     * In async mode, tick once to advance the queue (picks up this
     * command if its dependencies are satisfied). This ensures
     * tu_cmdq_submit_mma() and other convenience wrappers work
     * correctly in all cycle-model modes. */
    if (cq->synchronous) {
        execute_command(cq, cmd);
    } else {
        tu_cmdq_tick(cq);
    }

    return (int)cmd->cmd_id;  /* Return command ID on success */
}

int tu_cmdq_barrier(tu_command_queue_t *cq) {
    if (!cq) return -1;
    uint32_t barrier_id;
    int rc = tu_cmdq_submit(cq, TU_CMD_BARRIER, NULL, 0, NULL, &barrier_id);
    if (rc < 0) return rc;

    /* In synchronous mode, make all subsequent commands depend on this barrier */
    /* This is enforced in tick() for async mode */
    return (int)barrier_id;
}

int tu_cmdq_wait(tu_command_queue_t *cq, uint32_t cmd_id, uint64_t timeout_cycles) {
    if (!cq) return -2;

    uint64_t start = cq->current_cycle;

    while (1) {
        /* Check if command completed */
        bool found = false;
        for (uint32_t i = 0; i < cq->capacity; i++) {
            if (cq->commands[i].cmd_id == cmd_id) {
                found = true;
                if (cq->commands[i].status == TU_CMD_COMPLETED) return 0;
                if (cq->commands[i].status == TU_CMD_FAULTED) return -3;
                break;
            }
        }

        if (!found) {
            /* Command already retired, treat as completed */
            return 0;
        }

        /* Advance execution */
        if (!cq->synchronous) tu_cmdq_tick(cq);

        /* Check timeout */
        if (timeout_cycles > 0 && (cq->current_cycle - start) >= timeout_cycles)
            return -1;
    }
}

void tu_cmdq_sync(tu_command_queue_t *cq) {
    if (!cq) return;
    if (cq->synchronous) return; /* Nothing to do */

    /* Drain the queue: tick until all commands retire */
    while (cq->count > 0) {
        tu_cmdq_tick(cq);
    }
}

int tu_cmdq_tick(tu_command_queue_t *cq) {
    if (!cq) return 0;
    cq->current_cycle++;

    int retired = 0;

    /* Scan for PENDING commands whose deps are satisfied and barrier is clear */
    for (uint32_t i = 0; i < cq->capacity; i++) {
        tu_command_t *cmd = &cq->commands[i];
        if (cmd->cmd_id == 0) continue;
        if (cmd->status != TU_CMD_PENDING) continue;

        /* Check barrier: commands after a barrier must wait */
        if (cq->last_barrier_id > 0 && cmd->cmd_id > cq->last_barrier_id) {
            if (!barrier_clear(cq)) continue;
        }

        /* Check dependencies */
        if (!deps_satisfied(cq, cmd)) continue;

        /* Execute */
        execute_command(cq, cmd);
        retired++;

        /* In sync mode, remove immediately from count */
        if (cq->synchronous) {
            cmd->cmd_id = 0; /* Mark slot as free */
            cq->count--;
            cq->tail = (cq->tail + 1) % cq->capacity;
        }
    }

    return retired;
}

uint32_t tu_cmdq_get_depth(const tu_command_queue_t *cq) {
    return cq ? cq->count : 0;
}

tu_cmd_status_t tu_cmdq_get_status(const tu_command_queue_t *cq, uint32_t cmd_id) {
    if (!cq) return TU_CMD_COMPLETED;
    for (uint32_t i = 0; i < cq->capacity; i++) {
        if (cq->commands[i].cmd_id == cmd_id)
            return cq->commands[i].status;
    }
    return TU_CMD_COMPLETED; /* Not found = already retired */
}

void tu_cmdq_print_stats(const tu_command_queue_t *cq) {
    if (!cq) return;
    fprintf(stderr,
        "  CMDQ: submitted=%lu completed=%lu faulted=%lu depth=%u/%u sync=%s\n",
        cq->total_submitted, cq->total_completed, cq->total_faulted,
        cq->count, cq->capacity, cq->synchronous ? "yes" : "no");
}

void tu_cmdq_get_counts(const tu_command_queue_t *cq,
                        uint64_t *submitted,
                        uint64_t *completed,
                        uint64_t *faulted) {
    if (!cq) { *submitted = *completed = *faulted = 0; return; }
    *submitted = cq->total_submitted;
    *completed = cq->total_completed;
    *faulted = cq->total_faulted;
}
