/*
 * TU CModel — Compiler Scheduling Pass (Gap C2)
 * ==============================================
 *
 * Dependency-graph-based instruction scheduler for the TU ISA.
 * Reorders independent operations to maximize DMA/compute overlap
 * while respecting data dependencies and barrier constraints.
 *
 * Key transformations:
 *   1. DAG construction — parse instruction sequence, build dependency DAG
 *      based on SRAM region read/write conflicts.
 *   2. DMA hoisting — move DMA loads as early as possible (before compute
 *      ops they don't depend on).
 *   3. Barrier insertion — insert SYNC/BARRIER instructions where needed
 *      to prevent DMA/compute hazards.
 *   4. Priority scheduling — schedule compute ops that produce data for
 *      other ops before less-critical ones (ASAP/ALAP weighting).
 *   5. Pipeline optimization — interleave DMA for tile N+1 with compute
 *      for tile N when double buffering is available.
 *
 * Gap: C2 — Compiler scheduling pass (P1, High)
 * Architecture: The scheduler operates on an array of tu_instruction_t
 *   entries, producing a reordered sequence. It is a standalone pass
 *   that can be invoked before binary encoding.
 *
 * Design:
 *   - Dependency graph: nodes are instructions, edges are read-after-write
 *     (RAW), write-after-read (WAR), and write-after-write (WAW) hazards.
 *   - SRAM-aware: tracks which SRAM regions (W, A, O) each instruction
 *     reads/writes, plus the byte ranges.
 *   - Config-driven: scheduling aggressiveness, DMA hoisting depth,
 *     and barrier placement are configurable.
 *   - Max 256 instruction window (configurable via TU_SCHED_MAX_INSTRS).
 */

#ifndef TU_SCHEDULER_H
#define TU_SCHEDULER_H

#include "tu_isa.h"
#include "../tu_config.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Configuration ---- */

#define TU_SCHED_MAX_INSTRS       256   /* Max instructions per scheduling window */
#define TU_SCHED_MAX_DEPS          16   /* Max dependencies per instruction */
#define TU_SCHED_MAX_READY_QUEUE   256   /* Max ready queue depth */

/* Scheduling policy */
typedef enum {
    TU_SCHED_POLICY_ASAP        = 0,  /* As-soon-as-possible (greedy) */
    TU_SCHED_POLICY_ALAP        = 1,  /* As-late-as-possible (reduce live ranges) */
    TU_SCHED_POLICY_BALANCED    = 2,  /* Balance DMA/compute overlap */
    TU_SCHED_POLICY_COUNT
} tu_sched_policy_t;

/* Scheduling configuration */
typedef struct {
    tu_sched_policy_t   policy;             /* Scheduling policy */
    bool                hoist_dma;           /* Hoist DMA loads before compute */
    bool                insert_barriers;     /* Auto-insert SYNC/BARRIER instructions */
    bool                pipeline_tiles;      /* Interleave DMA for tile N+1 with compute for tile N */
    uint32_t            max_hoist_distance;  /* Max instructions to hoist DMA ahead */
    uint32_t            max_window;          /* Max instructions to schedule at once */
    bool                verbose;             /* Print scheduling decisions */
} tu_sched_config_t;

/* Default config: balanced, DMA hoisting on, barriers on, pipeline on */
extern const tu_sched_config_t tu_sched_config_default;

/* ---- SRAM Access Tracking ---- */

typedef enum {
    TU_SRAM_W = 0,
    TU_SRAM_A = 1,
    TU_SRAM_O = 2,
    TU_SRAM_REGION_COUNT = 3
} tu_sram_region_id_t;

typedef struct {
    bool        reads[TU_SRAM_REGION_COUNT];   /* Reads from this region */
    bool        writes[TU_SRAM_REGION_COUNT];   /* Writes to this region */
    uint32_t    read_offsets[TU_SRAM_REGION_COUNT][2];  /* [start, end) byte range for reads */
    uint32_t    write_offsets[TU_SRAM_REGION_COUNT][2]; /* [start, end) byte range for writes */
} tu_sram_access_t;

/* ---- Dependency Graph ---- */

/* Forward declaration */
typedef struct tu_sched_node_t tu_sched_node_t;
typedef struct tu_sched_graph_t tu_sched_graph_t;

/* A node in the dependency graph */
struct tu_sched_node_t {
    uint32_t            id;              /* Node index (position in original sequence) */
    tu_instruction_t    instr;           /* The instruction */
    tu_sram_access_t    access;          /* SRAM access pattern */
    bool                is_dma;          /* Is this a DMA instruction? */
    bool                is_compute;      /* Is this a compute instruction? */
    bool                is_barrier;      /* Is this a barrier/sync? */

    /* Dependencies */
    uint32_t            num_preds;       /* Number of predecessors (must execute before) */
    uint32_t            preds[TU_SCHED_MAX_DEPS];  /* Predecessor node IDs */
    uint32_t            num_succs;       /* Number of successors (must wait for this) */
    uint32_t            succs[TU_SCHED_MAX_DEPS];  /* Successor node IDs */

    /* Scheduling state */
    bool                scheduled;       /* Already emitted to output */
    int32_t             asap_cycle;      /* Earliest schedule cycle */
    int32_t             alap_cycle;      /* Latest schedule cycle */
    int32_t             slack;           /* alap - asap */
};

/* Dependency graph */
struct tu_sched_graph_t {
    tu_sched_node_t     nodes[TU_SCHED_MAX_INSTRS];
    uint32_t            num_nodes;
    bool                built;
    tu_sched_config_t   config;
};

/* ---- Scheduling Result ---- */

typedef struct {
    tu_instruction_t    instructions[TU_SCHED_MAX_INSTRS];
    uint32_t            num_instructions;
    uint32_t            num_barriers_inserted;
    uint32_t            num_dma_hoisted;
    uint32_t            estimated_cycles;     /* Estimated execution cycles */
    bool                valid;
} tu_sched_result_t;

/* ================================================================
 * API
 * ================================================================ */

/*
 * Analyze an instruction's SRAM access pattern.
 * Parses opcode and operand fields to determine which SRAM regions
 * are read/written and their byte ranges.
 */
void tu_sched_analyze_access(const tu_instruction_t *instr,
                              tu_sram_access_t *access);

/*
 * Build a dependency DAG from an instruction sequence.
 *
 *   graph:    output graph (caller-allocated)
 *   instrs:   input instruction sequence
 *   n_instrs: number of instructions
 *   config:   scheduling configuration
 *
 * Returns 0 on success, -1 if too many instructions.
 */
int tu_sched_build_dag(tu_sched_graph_t *graph,
                        const tu_instruction_t *instrs,
                        uint32_t n_instrs,
                        const tu_sched_config_t *config);

/*
 * Compute ASAP and ALAP cycle estimates for all nodes in the DAG.
 * Also calculates slack = ALAP - ASAP for each node.
 * Must be called after tu_sched_build_dag().
 */
void tu_sched_compute_mobility(tu_sched_graph_t *graph);

/*
 * Hoist DMA instructions earlier in the schedule.
 * Moves DMA loads before compute ops they don't depend on,
 * respecting max_hoist_distance.
 * Returns the number of DMA ops hoisted.
 */
int tu_sched_hoist_dma(tu_sched_graph_t *graph);

/*
 * Insert barrier instructions where needed to prevent hazards.
 * Barriers are inserted between DMA and compute ops that share
 * SRAM regions without explicit synchronization.
 * Returns the number of barriers inserted.
 */
int tu_sched_insert_barriers(tu_sched_graph_t *graph);

/*
 * Run the full scheduling pass on an instruction sequence.
 *
 *   instrs:   input instruction sequence
 *   n_instrs: number of instructions
 *   config:   scheduling configuration (NULL = default)
 *   result:   output scheduled sequence (caller-allocated)
 *
 * Returns 0 on success, -1 on error.
 *
 * This is the main entry point. It builds the DAG, computes mobility,
 * hoists DMA, inserts barriers, then emits the scheduled sequence
 * according to the selected policy.
 */
int tu_sched_run(const tu_instruction_t *instrs,
                  uint32_t n_instrs,
                  const tu_sched_config_t *config,
                  tu_sched_result_t *result);

/*
 * Print a human-readable representation of the schedule.
 */
void tu_sched_print_result(const tu_sched_result_t *result);

/*
 * Print the dependency graph (for debugging).
 */
void tu_sched_print_graph(const tu_sched_graph_t *graph);

/*
 * Validate that a scheduled sequence respects all data dependencies.
 * Returns true if valid, false if a dependency is violated.
 */
bool tu_sched_validate(const tu_sched_result_t *result,
                        const tu_sched_graph_t *graph);

#ifdef __cplusplus
}
#endif

#endif /* TU_SCHEDULER_H */
