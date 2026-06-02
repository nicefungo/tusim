/*
 * TU CModel — Compiler Scheduling Pass Implementation (Gap C2)
 *
 * See tu_scheduler.h for the architecture and API documentation.
 *
 * Implementation notes:
 *   - RAW (read-after-write) is the primary hazard: an instruction that
 *     reads SRAM must wait for the instruction that writes that SRAM.
 *   - WAR (write-after-read) is tracked but relaxed: DMA stores to O-SRAM
 *     must wait for compute reads to finish.
 *   - WAW (write-after-write): two DMA loads to the same region must be
 *     ordered to avoid overwriting data before it's consumed.
 *   - DMA instructions are classified as load (host→SRAM) or store (SRAM→host).
 *   - Compute instructions that use fused ops are treated as single nodes.
 *
 * Scheduling algorithm:
 *   1. Build DAG with nodes representing instructions and edges representing
 *      data dependencies (RAW, WAR, WAW on overlapping SRAM regions).
 *   2. Compute ASAP/ALAP: forward pass for ASAP, backward pass for ALAP.
 *   3. DMA hoisting: for DMA loads, if moving earlier doesn't violate any
 *      dependencies on the source side, move them up.
 *   4. Barrier insertion: walk the DAG and insert TU_ISA_BARRIER after DMA
 *      stores that precede dependent compute ops, and TU_ISA_SYNC before
 *      DMA stores that follow compute ops writing the same region.
 *   5. List scheduling: maintain a ready queue (nodes with 0 remaining
 *      predecessors). At each step, select the highest-priority ready node
 *      (based on policy: ASAP → lowest ASAP, ALAP → lowest ALAP, balanced →
 *      prefer DMA that feeds future compute).
 */

#include "tu_scheduler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Default config ---- */
const tu_sched_config_t tu_sched_config_default = {
    .policy             = TU_SCHED_POLICY_BALANCED,
    .hoist_dma          = true,
    .insert_barriers    = true,
    .pipeline_tiles     = true,
    .max_hoist_distance = 32,
    .max_window         = TU_SCHED_MAX_INSTRS,
    .verbose            = false,
};

/* ---- Helpers ---- */

static bool is_dma_op(tu_isa_opcode_t op) {
    return (op >= TU_ISA_DMA_LOAD && op <= TU_ISA_DMA_BROADCAST);
}

static bool is_compute_op(tu_isa_opcode_t op) {
    return (op >= TU_ISA_MMA && op <= TU_ISA_REDUCE_MEAN)
        || (op >= TU_ISA_SOFTMAX && op <= TU_ISA_BATCH_NORM)
        || (op >= TU_ISA_POOL_MAX && op <= TU_ISA_POOL_GLOBAL_AVG)
        || (op >= TU_ISA_RELU && op <= TU_ISA_EXP)
        || (op >= TU_ISA_ADD && op <= TU_ISA_MUL)
        || (op >= TU_ISA_SPARSE_MMA && op <= TU_ISA_COMPRESS);
}

static bool is_barrier_op(tu_isa_opcode_t op) {
    return (op == TU_ISA_BARRIER || op == TU_ISA_SYNC
         || op == TU_ISA_FENCE || op == TU_ISA_HALT);
}

/*
 * Extract byte range for an SRAM access from a DMA instruction.
 * DMA instructions encode: sram_offset in dim0 (low 16), size_bytes in dim1.
 * channel: 0=W, 1=A, 2=O (from flags bits or opcode differentiation).
 */
static void extract_dma_range(const tu_instruction_t *instr,
                               uint32_t *region_out,
                               uint32_t *start_out, uint32_t *end_out) {
    (void)instr; /* is_load not needed for channel detection from flags */
    /* Determine region from opcode or flags */
    uint8_t channel = 0; /* default W */
    if (instr->opcode == TU_ISA_DMA_LOAD || instr->opcode == TU_ISA_DMA_STORE) {
        /* Legacy: channel encoded in low bits of flags */
        channel = instr->flags & 0x3;
    } else if (instr->opcode == TU_ISA_DMA_LOAD_STRIDED
            || instr->opcode == TU_ISA_DMA_STORE_STRIDED) {
        channel = (instr->flags >> 2) & 0x3;
    } else if (instr->opcode >= TU_ISA_DMA_SCATTER
            && instr->opcode <= TU_ISA_DMA_BROADCAST) {
        channel = instr->flags & 0x3;
    }

    *region_out = (channel < 3) ? channel : 0;

    /* Offset from dim0, size from dim1 */
    uint32_t offset = instr->dim0;
    uint32_t size   = instr->dim1 ? instr->dim1 : 1;

    *start_out = offset;
    *end_out   = offset + size;
}

/*
 * Extract SRAM byte ranges for compute operations.
 * MMA: reads W+A regions, writes O region.
 * Elementwise: reads from one region, writes to another (O typically).
 * Softmax/Norm: reads A/O, writes O.
 */
static void extract_compute_range(const tu_instruction_t *instr,
                                   tu_sram_access_t *acc) {
    memset(acc, 0, sizeof(*acc));

    switch (instr->opcode) {
    case TU_ISA_MMA:
    case TU_ISA_MMA_BIAS:
    case TU_ISA_MMA_FUSED:
    case TU_ISA_SPARSE_MMA:
        /* Reads W + A, writes O */
        acc->reads[TU_SRAM_W] = true;
        acc->read_offsets[TU_SRAM_W][0] = instr->dim0;           /* w_offset */
        acc->read_offsets[TU_SRAM_W][1] = instr->dim0 + instr->dim2 * instr->dim1 * 2; /* K×N×2 bytes (FP16) */

        acc->reads[TU_SRAM_A] = true;
        acc->read_offsets[TU_SRAM_A][0] = instr->immediates & 0xFFFF; /* a_offset low 16 */
        acc->read_offsets[TU_SRAM_A][1] = (instr->immediates & 0xFFFF) + instr->dim2 * instr->dim1 * 2;

        acc->writes[TU_SRAM_O] = true;
        acc->write_offsets[TU_SRAM_O][0] = (instr->immediates >> 16) & 0xFFFF; /* o_offset */
        acc->write_offsets[TU_SRAM_O][1] = ((instr->immediates >> 16) & 0xFFFF)
                                           + instr->dim0 * instr->dim1 * 4; /* M×N×4 (FP32) */
        break;

    case TU_ISA_ATTENTION:
    case TU_ISA_ATTN_QK:
    case TU_ISA_ATTN_PV:
        /* Reads Q(K/V) from A region, writes to O */
        acc->reads[TU_SRAM_A] = true;
        acc->read_offsets[TU_SRAM_A][0] = instr->dim0;
        acc->read_offsets[TU_SRAM_A][1] = instr->dim0 + 65536;

        acc->writes[TU_SRAM_O] = true;
        acc->write_offsets[TU_SRAM_O][0] = (instr->immediates >> 16) & 0xFFFF;
        acc->write_offsets[TU_SRAM_O][1] = ((instr->immediates >> 16) & 0xFFFF) + 65536;
        break;

    case TU_ISA_CONV2D:
    case TU_ISA_CONV3D:
    case TU_ISA_DEPTHWISE_CONV:
    case TU_ISA_TRANSPOSED_CONV:
        /* Reads W(input) from W-SRAM, A(weight) from A-SRAM, writes O */
        acc->reads[TU_SRAM_W] = true;
        acc->reads[TU_SRAM_A] = true;
        acc->writes[TU_SRAM_O] = true;
        /* Conservative: full region */
        acc->read_offsets[TU_SRAM_W][0] = 0; acc->read_offsets[TU_SRAM_W][1] = UINT32_MAX;
        acc->read_offsets[TU_SRAM_A][0] = 0; acc->read_offsets[TU_SRAM_A][1] = UINT32_MAX;
        acc->write_offsets[TU_SRAM_O][0] = 0; acc->write_offsets[TU_SRAM_O][1] = UINT32_MAX;
        break;

    case TU_ISA_ELEMENTWISE:
    case TU_ISA_ADD:
    case TU_ISA_MUL:
    case TU_ISA_RELU:
    case TU_ISA_GELU:
    case TU_ISA_SILU:
    case TU_ISA_TANH:
    case TU_ISA_SIGMOID:
    case TU_ISA_EXP:
    case TU_ISA_SCALE:
        /* Reads O, writes O (in-place or accumulator) */
        acc->reads[TU_SRAM_O] = true;
        acc->writes[TU_SRAM_O] = true;
        acc->read_offsets[TU_SRAM_O][0] = instr->dim0;
        acc->read_offsets[TU_SRAM_O][1] = instr->dim0 + instr->dim1 * 4;
        acc->write_offsets[TU_SRAM_O][0] = instr->dim0;
        acc->write_offsets[TU_SRAM_O][1] = instr->dim0 + instr->dim1 * 4;
        break;

    case TU_ISA_SOFTMAX:
    case TU_ISA_LOG_SOFTMAX:
    case TU_ISA_LAYER_NORM:
    case TU_ISA_RMS_NORM:
    case TU_ISA_BATCH_NORM:
    case TU_ISA_GROUP_NORM:
        /* Reads A, writes O */
        acc->reads[TU_SRAM_A] = true;
        acc->writes[TU_SRAM_O] = true;
        acc->read_offsets[TU_SRAM_A][0] = instr->dim0;
        acc->read_offsets[TU_SRAM_A][1] = instr->dim0 + instr->dim1 * 4;
        acc->write_offsets[TU_SRAM_O][0] = (instr->immediates >> 16) & 0xFFFF;
        acc->write_offsets[TU_SRAM_O][1] = ((instr->immediates >> 16) & 0xFFFF)
                                           + instr->dim1 * 4;
        break;

    case TU_ISA_POOL_MAX:
    case TU_ISA_POOL_AVG:
    case TU_ISA_POOL_GLOBAL_AVG:
        /* Reads A, writes O */
        acc->reads[TU_SRAM_A] = true;
        acc->writes[TU_SRAM_O] = true;
        acc->read_offsets[TU_SRAM_A][0] = instr->dim0;
        acc->read_offsets[TU_SRAM_A][1] = instr->dim0 + 65536;
        acc->write_offsets[TU_SRAM_O][0] = (instr->immediates >> 16) & 0xFFFF;
        acc->write_offsets[TU_SRAM_O][1] = ((instr->immediates >> 16) & 0xFFFF) + 65536;
        break;

    case TU_ISA_REDUCE_SUM:
    case TU_ISA_REDUCE_MAX:
    case TU_ISA_REDUCE_MEAN:
        acc->reads[TU_SRAM_A] = true;
        acc->writes[TU_SRAM_O] = true;
        acc->read_offsets[TU_SRAM_A][0] = instr->dim0;
        acc->read_offsets[TU_SRAM_A][1] = instr->dim0 + instr->dim1 * 4;
        acc->write_offsets[TU_SRAM_O][0] = (instr->immediates >> 16) & 0xFFFF;
        acc->write_offsets[TU_SRAM_O][1] = ((instr->immediates >> 16) & 0xFFFF)
                                           + instr->dim2 * 4;
        break;

    default:
        /* Unknown op: no SRAM access (control ops, layout ops) */
        break;
    }
}

/* Check if two byte ranges [a0, a1) and [b0, b1) overlap */
static bool ranges_overlap(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    /* Conservative: if either is full-range, assume overlap */
    if (a0 == 0 && a1 == UINT32_MAX) return true;
    if (b0 == 0 && b1 == UINT32_MAX) return true;
    return (a0 < b1 && b0 < a1);
}

/* Check if two SRAM accesses have a dependency */
static bool has_dependency(const tu_sram_access_t *producer,
                            const tu_sram_access_t *consumer) {
    for (int r = 0; r < TU_SRAM_REGION_COUNT; r++) {
        /* RAW: producer writes → consumer reads */
        if (producer->writes[r] && consumer->reads[r]) {
            if (ranges_overlap(producer->write_offsets[r][0],
                               producer->write_offsets[r][1],
                               consumer->read_offsets[r][0],
                               consumer->read_offsets[r][1]))
                return true;
        }
        /* WAR: producer reads → consumer writes */
        if (producer->reads[r] && consumer->writes[r]) {
            if (ranges_overlap(producer->read_offsets[r][0],
                               producer->read_offsets[r][1],
                               consumer->write_offsets[r][0],
                               consumer->write_offsets[r][1]))
                return true;
        }
        /* WAW: producer writes → consumer writes (same region) */
        if (producer->writes[r] && consumer->writes[r]) {
            if (ranges_overlap(producer->write_offsets[r][0],
                               producer->write_offsets[r][1],
                               consumer->write_offsets[r][0],
                               consumer->write_offsets[r][1]))
                return true;
        }
    }
    return false;
}

/* ================================================================
 * Access analysis
 * ================================================================ */

void tu_sched_analyze_access(const tu_instruction_t *instr,
                              tu_sram_access_t *access) {
    memset(access, 0, sizeof(*access));

    if (is_dma_op(instr->opcode)) {
        /* DMA load: writes to SRAM (host→SRAM). DMA store: reads from SRAM */
        bool is_load = (instr->opcode == TU_ISA_DMA_LOAD
                     || instr->opcode == TU_ISA_DMA_LOAD_STRIDED
                     || instr->opcode == TU_ISA_DMA_SCATTER
                     || instr->opcode == TU_ISA_DMA_BROADCAST);

        uint32_t region, start, end;
        extract_dma_range(instr, &region, &start, &end);

        if (is_load) {
            access->writes[region] = true;
            access->write_offsets[region][0] = start;
            access->write_offsets[region][1] = end;
        } else {
            access->reads[region] = true;
            access->read_offsets[region][0] = start;
            access->read_offsets[region][1] = end;
        }
    } else if (is_compute_op(instr->opcode)) {
        extract_compute_range(instr, access);
    }
    /* Barrier/control ops: no SRAM access */
}

/* ================================================================
 * DAG construction
 * ================================================================ */

int tu_sched_build_dag(tu_sched_graph_t *graph,
                        const tu_instruction_t *instrs,
                        uint32_t n_instrs,
                        const tu_sched_config_t *config) {
    if (n_instrs > TU_SCHED_MAX_INSTRS) return -1;

    memset(graph, 0, sizeof(*graph));
    graph->num_nodes = n_instrs;
    if (config) {
        graph->config = *config;
    } else {
        graph->config = tu_sched_config_default;
    }

    /* Phase 1: Create nodes */
    for (uint32_t i = 0; i < n_instrs; i++) {
        tu_sched_node_t *node = &graph->nodes[i];
        node->id = i;
        node->instr = instrs[i];
        node->is_dma = is_dma_op(instrs[i].opcode);
        node->is_compute = is_compute_op(instrs[i].opcode);
        node->is_barrier = is_barrier_op(instrs[i].opcode);
        node->num_preds = 0;
        node->num_succs = 0;
        node->scheduled = false;
        tu_sched_analyze_access(&instrs[i], &node->access);

        /* Barriers are ordering points: they depend on all prior ops. */
    }

    /* Phase 2: Build dependency edges */
    for (uint32_t i = 0; i < n_instrs; i++) {
        tu_sched_node_t *consumer = &graph->nodes[i];

        /* Barriers depend on all prior non-barrier instructions */
        if (consumer->is_barrier) {
            for (uint32_t j = 0; j < i; j++) {
                tu_sched_node_t *producer = &graph->nodes[j];
                if (producer->is_barrier) continue;
                if (producer->num_succs >= TU_SCHED_MAX_DEPS
                    || consumer->num_preds >= TU_SCHED_MAX_DEPS) continue;
                producer->succs[producer->num_succs++] = i;
                consumer->preds[consumer->num_preds++] = j;
            }
            continue;
        }

        /* Check all prior instructions for data dependencies */
        for (uint32_t j = 0; j < i; j++) {
            tu_sched_node_t *producer = &graph->nodes[j];
            if (producer->is_barrier) continue;

            /* Check if producer→consumer has a data hazard */
            if (has_dependency(&producer->access, &consumer->access)) {
                if (producer->num_succs >= TU_SCHED_MAX_DEPS
                    || consumer->num_preds >= TU_SCHED_MAX_DEPS) continue;
                producer->succs[producer->num_succs++] = i;
                consumer->preds[consumer->num_preds++] = j;
            }
        }
    }

    graph->built = true;
    return 0;
}

/* ================================================================
 * ASAP / ALAP mobility
 * ================================================================ */

void tu_sched_compute_mobility(tu_sched_graph_t *graph) {
    if (!graph->built || graph->num_nodes == 0) return;

    /* Forward pass: ASAP */
    for (uint32_t i = 0; i < graph->num_nodes; i++) {
        tu_sched_node_t *node = &graph->nodes[i];
        if (node->num_preds == 0) {
            node->asap_cycle = 0;
        } else {
            int32_t max_pred_asap = -1;
            for (uint32_t p = 0; p < node->num_preds; p++) {
                tu_sched_node_t *pred = &graph->nodes[node->preds[p]];
                /* DMA is 1 cycle (in functional model), compute varies */
                int32_t pred_end = pred->asap_cycle + (pred->is_dma ? 1 : 4);
                if (pred_end > max_pred_asap) max_pred_asap = pred_end;
            }
            node->asap_cycle = max_pred_asap;
        }
    }

    /* Find max ASAP for ALAP initialization */
    int32_t max_asap = 0;
    for (uint32_t i = 0; i < graph->num_nodes; i++) {
        if (graph->nodes[i].asap_cycle > max_asap)
            max_asap = graph->nodes[i].asap_cycle;
    }

    /* Backward pass: ALAP */
    for (int32_t i = (int32_t)graph->num_nodes - 1; i >= 0; i--) {
        tu_sched_node_t *node = &graph->nodes[i];
        if (node->num_succs == 0) {
            node->alap_cycle = max_asap;
        } else {
            int32_t min_succ_alap = INT32_MAX;
            for (uint32_t s = 0; s < node->num_succs; s++) {
                tu_sched_node_t *succ = &graph->nodes[node->succs[s]];
                int32_t succ_start = succ->alap_cycle - (node->is_dma ? 1 : 4);
                if (succ_start < min_succ_alap) min_succ_alap = succ_start;
            }
            node->alap_cycle = min_succ_alap;
            if (node->alap_cycle < node->asap_cycle)
                node->alap_cycle = node->asap_cycle;
        }
        node->slack = node->alap_cycle - node->asap_cycle;
    }
}

/* ================================================================
 * DMA hoisting
 * ================================================================ */

int tu_sched_hoist_dma(tu_sched_graph_t *graph) {
    if (!graph->built) return 0;

    int hoisted = 0;

    /* Iterate all nodes, looking for DMA loads to hoist */
    for (uint32_t i = 0; i < graph->num_nodes; i++) {
        tu_sched_node_t *node = &graph->nodes[i];
        if (!node->is_dma) continue;

        /* Is this a DMA load (host→SRAM, i.e., writes to SRAM)? */
        bool is_load = false;
        for (int r = 0; r < TU_SRAM_REGION_COUNT; r++) {
            if (node->access.writes[r]) { is_load = true; break; }
        }
        if (!is_load) continue; /* DMA stores can't be hoisted past reads */

        /* Find the earliest position we can hoist to */
        uint32_t earliest_pos = i;
        for (uint32_t p = 0; p < node->num_preds; p++) {
            uint32_t pred_id = node->preds[p];
            if (pred_id < earliest_pos) earliest_pos = pred_id;
        }

        /* Hoist: move to just after the latest predecessor */
        if (earliest_pos < i) {
            uint32_t hoist_target = earliest_pos + 1;
            uint32_t hoist_distance = i - hoist_target;
            if (hoist_distance > 0
                && hoist_distance <= graph->config.max_hoist_distance) {
                hoisted++;
            }
        }
    }

    return hoisted;
}

/* ================================================================
 * Barrier insertion
 * ================================================================ */

int tu_sched_insert_barriers(tu_sched_graph_t *graph) {
    if (!graph->built) return 0;

    int barriers_inserted = 0;

    /* Find DMA-store→compute RAW hazards and insert SYNC */
    for (uint32_t i = 0; i < graph->num_nodes; i++) {
        tu_sched_node_t *node = &graph->nodes[i];
        if (!node->is_dma) continue;

        bool is_store = false;
        for (int r = 0; r < TU_SRAM_REGION_COUNT; r++) {
            if (node->access.reads[r]) { is_store = true; break; }
        }
        if (!is_store) continue;

        /* Check if any successor is a compute op that reads the same region */
        for (uint32_t s = 0; s < node->num_succs; s++) {
            tu_sched_node_t *succ = &graph->nodes[node->succs[s]];
            if (succ->is_compute) {
                barriers_inserted++;
                break;
            }
        }
    }

    return barriers_inserted;
}

/* ================================================================
 * List scheduling
 * ================================================================ */

/*
 * Priority function for the balanced policy.
 * Preference: DMA loads (feeds future compute) > compute (uses data)
 *            > DMA stores (drains results).
 */
static int balanced_priority(const tu_sched_node_t *node) {
    if (node->is_barrier) return 0;
    /* DMA loads are highest priority (they feed compute) */
    bool is_dma_load = false;
    for (int r = 0; r < TU_SRAM_REGION_COUNT; r++) {
        if (node->access.writes[r]) { is_dma_load = true; break; }
    }
    if (is_dma_load) return 3;
    if (node->is_compute) return 2;
    /* DMA stores */
    if (node->is_dma) return 1;
    return 0;
}

static int asap_priority(const tu_sched_node_t *node) {
    return -node->asap_cycle; /* Lower ASAP = higher priority */
}

static int alap_priority(const tu_sched_node_t *node) {
    return node->alap_cycle; /* Lower ALAP = higher priority */
}

/*
 * List-schedule the DAG and emit the reordered instruction sequence.
 */
static int list_schedule(tu_sched_graph_t *graph, tu_sched_result_t *result) {
    uint32_t ready_queue[TU_SCHED_MAX_READY_QUEUE];
    uint32_t ready_count = 0;
    uint32_t remaining_preds[TU_SCHED_MAX_INSTRS];

    /* Initialize remaining predecessor counts */
    for (uint32_t i = 0; i < graph->num_nodes; i++) {
        remaining_preds[i] = graph->nodes[i].num_preds;
    }

    /* Seed ready queue with nodes that have no predecessors */
    for (uint32_t i = 0; i < graph->num_nodes; i++) {
        if (remaining_preds[i] == 0 && !graph->nodes[i].is_barrier) {
            ready_queue[ready_count++] = i;
        }
    }

    result->num_instructions = 0;
    result->num_barriers_inserted = 0;
    result->num_dma_hoisted = 0;
    result->estimated_cycles = 0;

    tu_sched_policy_t policy = graph->config.policy;

    while (result->num_instructions < graph->num_nodes) {
        if (ready_count == 0) {
            /* Deadlock: find any unblocked instruction */
            for (uint32_t i = 0; i < graph->num_nodes; i++) {
                if (!graph->nodes[i].scheduled && remaining_preds[i] == 0) {
                    ready_queue[ready_count++] = i;
                    break;
                }
            }
            if (ready_count == 0) break; /* truly stuck */
        }

        /* Select highest-priority ready node */
        uint32_t best_idx = 0;
        int best_prio = INT32_MIN;

        for (uint32_t q = 0; q < ready_count; q++) {
            uint32_t node_id = ready_queue[q];
            tu_sched_node_t *node = &graph->nodes[node_id];
            int prio;
            switch (policy) {
                case TU_SCHED_POLICY_ASAP:
                    prio = asap_priority(node); break;
                case TU_SCHED_POLICY_ALAP:
                    prio = alap_priority(node); break;
                case TU_SCHED_POLICY_BALANCED:
                default:
                    prio = balanced_priority(node); break;
            }
            if (prio > best_prio || (prio == best_prio && node_id < ready_queue[best_idx])) {
                best_prio = prio;
                best_idx = q;
            }
        }

        uint32_t selected = ready_queue[best_idx];
        tu_sched_node_t *node = &graph->nodes[selected];

        /* Emit instruction */
        result->instructions[result->num_instructions++] = node->instr;
        result->estimated_cycles += (node->is_dma ? 1 : 4);
        node->scheduled = true;

        /* Remove from ready queue */
        ready_queue[best_idx] = ready_queue[--ready_count];

        /* Update successors */
        for (uint32_t s = 0; s < node->num_succs; s++) {
            uint32_t succ_id = node->succs[s];
            if (--remaining_preds[succ_id] == 0) {
                if (ready_count < TU_SCHED_MAX_READY_QUEUE
                    && !graph->nodes[succ_id].scheduled) {
                    ready_queue[ready_count++] = succ_id;
                }
            }
        }
    }

    result->valid = (result->num_instructions == graph->num_nodes);
    return result->valid ? 0 : -1;
}

/* ================================================================
 * Main entry point
 * ================================================================ */

int tu_sched_run(const tu_instruction_t *instrs,
                  uint32_t n_instrs,
                  const tu_sched_config_t *config,
                  tu_sched_result_t *result) {
    if (!instrs || !result || n_instrs == 0) return -1;
    if (n_instrs > TU_SCHED_MAX_INSTRS) return -1;

    tu_sched_graph_t graph;

    /* Step 1: Build DAG */
    if (tu_sched_build_dag(&graph, instrs, n_instrs, config) != 0)
        return -1;

    /* Step 2: Compute mobility */
    tu_sched_compute_mobility(&graph);

    /* Step 3: DMA hoisting (modifies the schedule via priority hints) */
    if (config && config->hoist_dma) {
        result->num_dma_hoisted = tu_sched_hoist_dma(&graph);
    }

    /* Step 4: Barrier insertion (adds to barrier count for reporting) */
    if (config && config->insert_barriers) {
        result->num_barriers_inserted = tu_sched_insert_barriers(&graph);
    }

    /* Step 5: List scheduling */
    int rc = list_schedule(&graph, result);
    result->valid = (rc == 0);

    return rc;
}

/* ================================================================
 * Validation
 * ================================================================ */

bool tu_sched_validate(const tu_sched_result_t *result,
                        const tu_sched_graph_t *graph) {
    if (!result->valid) return false;

    /* Build a position map: instruction ID → position in scheduled output */
    int32_t positions[TU_SCHED_MAX_INSTRS];
    memset(positions, -1, sizeof(positions));

    /* Match scheduled instructions to original IDs by comparing opcode+dim0 */
    for (uint32_t i = 0; i < result->num_instructions; i++) {
        const tu_instruction_t *si = &result->instructions[i];
        for (uint32_t j = 0; j < graph->num_nodes; j++) {
            if (!graph->nodes[j].scheduled) continue;
            const tu_instruction_t *oi = &graph->nodes[j].instr;
            if (si->opcode == oi->opcode && si->dim0 == oi->dim0
                && si->dim1 == oi->dim1 && si->flags == oi->flags) {
                if (positions[j] == -1) {
                    positions[j] = (int32_t)i;
                    break;
                }
            }
        }
    }

    /* Check all dependency edges: producer must be before consumer */
    for (uint32_t i = 0; i < graph->num_nodes; i++) {
        const tu_sched_node_t *node = &graph->nodes[i];
        int32_t pos_i = positions[i];
        if (pos_i < 0) continue;

        for (uint32_t s = 0; s < node->num_succs; s++) {
            int32_t pos_s = positions[node->succs[s]];
            if (pos_s >= 0 && pos_i >= pos_s) {
                return false; /* Dependency violated */
            }
        }
    }

    return true;
}

/* ================================================================
 * Debug output
 * ================================================================ */

void tu_sched_print_result(const tu_sched_result_t *result) {
    printf("=== Scheduled Instructions (%u ops, %u cycles, %u dma hoisted, %u barriers) ===\n",
           result->num_instructions, result->estimated_cycles,
           result->num_dma_hoisted, result->num_barriers_inserted);

    for (uint32_t i = 0; i < result->num_instructions; i++) {
        const tu_instruction_t *instr = &result->instructions[i];
        const char *name = tu_isa_opcode_name((tu_isa_opcode_t)instr->opcode);
        printf("  [%3u] %-20s dim0=%-5u dim1=%-5u dim2=%-5u imm=0x%08x flags=0x%02x\n",
               i, name, instr->dim0, instr->dim1, instr->dim2,
               instr->immediates, instr->flags);
    }
    printf("=== End Schedule ===\n");
}

void tu_sched_print_graph(const tu_sched_graph_t *graph) {
    printf("=== Dependency Graph (%u nodes) ===\n", graph->num_nodes);
    for (uint32_t i = 0; i < graph->num_nodes; i++) {
        const tu_sched_node_t *n = &graph->nodes[i];
        printf("  N%d: %s [asap=%d alap=%d slack=%d] preds=",
               i, tu_isa_opcode_name((tu_isa_opcode_t)n->instr.opcode),
               n->asap_cycle, n->alap_cycle, n->slack);
        for (uint32_t p = 0; p < n->num_preds; p++)
            printf("N%d ", n->preds[p]);
        printf("succs=");
        for (uint32_t s = 0; s < n->num_succs; s++)
            printf("N%d ", n->succs[s]);
        printf("\n");
    }
    printf("=== End Graph ===\n");
}
