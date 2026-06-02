/*
 * TU CModel — Liveness-Based Scratchpad Allocator (Gap C3)
 * =========================================================
 *
 * Graph-coloring register allocator adapted for SRAM scratchpad
 * allocation. Computes live ranges for virtual registers, builds
 * an interference graph, colors it, and inserts spill/fill DMA
 * when physical SRAM capacity is exceeded.
 *
 * Architecture:
 *   This is the second compiler pass (after scheduling pass, C2).
 *   It takes a scheduled instruction sequence and resolves virtual
 *   SRAM offsets into physical addresses, inserting spill code
 *   when needed.
 *
 * Key concepts:
 *   - Virtual Register: a logical SRAM allocation (byte range in W/A/O)
 *     that holds a tensor tile or intermediate result.
 *   - Live Range: the instruction interval [first_def, last_use] during
 *     which a virtual register holds live data.
 *   - Interference Graph: nodes = virtual registers, edges = overlapping
 *     live ranges (two VRegs cannot share physical SRAM if live simultaneously).
 *   - Coloring: assign physical SRAM offsets to virtual registers such
 *     that interfering VRegs don't overlap in physical space.
 *   - Spilling: when coloring fails (not enough physical SRAM), select a
 *     victim VReg to evict to DRAM, insert DMA spill/fill instructions.
 *
 * Gap: C3 — Liveness-based scratchpad allocation (P1, High)
 * Dependencies: C2 (scheduler), DMA engine, SRAM model
 *
 * Design principles:
 *   - Per-region allocation: W, A, and O scratchpads are allocated
 *     independently (they are separate physical memories).
 *   - Conservative liveness: live range = [first write, last read] for
 *     each virtual register. A write followed by reads means the register
 *     is live until the last read.
 *   - Spill cost heuristic: based on access frequency within the live
 *     range — frequently accessed VRegs are less likely to be spilled.
 *   - Config-driven: scratchpad capacity, spill strategy, and allocation
 *     order are configurable.
 */

#ifndef TU_LIVENESS_H
#define TU_LIVENESS_H

#include "tu_isa.h"
#include "tu_scheduler.h"
#include "../tu_config.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Configuration ---- */

#define TU_LIVE_MAX_VREGS       128   /* Max virtual registers per region */
#define TU_LIVE_MAX_INTERFERENCES 256 /* Max edges in interference graph */
#define TU_LIVE_MAX_SPILL_SLOTS  16   /* Max concurrent spill slots */
#define TU_LIVE_MAX_ALLOC_TRIES  10   /* Max attempts before giving up */

/* Allocation strategy for physical SRAM */
typedef enum {
    TU_ALLOC_FIRST_FIT   = 0,  /* First-fit: place at lowest available offset */
    TU_ALLOC_BEST_FIT    = 1,  /* Best-fit: minimize fragmentation */
    TU_ALLOC_WORST_FIT   = 2,  /* Worst-fit: maximize remaining large gaps */
    TU_ALLOC_COUNT
} tu_alloc_strategy_t;

/* Spill candidate selection */
typedef enum {
    TU_SPILL_FIFO        = 0,  /* First-in-first-out (oldest live range) */
    TU_SPILL_LRU         = 1,  /* Least-recently-used (furthest next use) */
    TU_SPILL_LARGEST     = 2,  /* Largest live range (free most space) */
    TU_SPILL_LEAST_ACCESSED = 3, /* Least frequently accessed */
    TU_SPILL_COUNT
} tu_spill_strategy_t;

/* Liveness allocator configuration */
typedef struct {
    uint32_t            w_capacity;         /* W-SRAM capacity in bytes */
    uint32_t            a_capacity;         /* A-SRAM capacity in bytes */
    uint32_t            o_capacity;         /* O-SRAM capacity in bytes */
    tu_alloc_strategy_t alloc_strategy;     /* Physical placement strategy */
    tu_spill_strategy_t spill_strategy;     /* Victim selection strategy */
    uint32_t            safety_margin;      /* Bytes reserved for spill/fill DMA descriptors */
    bool                enable_spilling;    /* Allow spilling when SRAM is full */
    bool                verbose;            /* Print allocation decisions */
} tu_live_config_t;

extern const tu_live_config_t tu_live_config_default;

/* ---- Virtual Register ---- */

typedef enum {
    TU_VREG_W = 0,  /* W scratchpad */
    TU_VREG_A = 1,  /* A scratchpad */
    TU_VREG_O = 2,  /* O scratchpad */
} tu_vreg_region_t;

typedef struct {
    uint32_t            id;             /* Unique virtual register ID */
    tu_vreg_region_t    region;         /* Which scratchpad region */
    uint32_t            size_bytes;     /* Byte size of this allocation */

    /* Live range: [first_def, last_use) instruction indices */
    int32_t             first_def;      /* Instruction that first writes this VReg */
    int32_t             last_use;       /* Last instruction that reads this VReg */

    /* Spill state */
    bool                spilled;        /* Evicted to DRAM */
    uint32_t            spill_slot;     /* DRAM spill slot index */
    uint32_t            access_count;   /* Number of reads within live range */

    /* Physical assignment */
    uint32_t            physical_offset; /* Assigned SRAM byte offset (UINT32_MAX = unassigned) */
} tu_vreg_t;

/* ---- Interference Graph ---- */

typedef struct {
    tu_vreg_t          *vregs[TU_LIVE_MAX_VREGS];
    uint32_t            num_vregs;

    /* Adjacency matrix: interference[v1][v2] = true if v1 and v2 interfere */
    bool               *interference;   /* Flattened matrix: [i * num_vregs + j] */

    /* Coloring result */
    uint32_t            num_physical_slots;
    bool                colored;
} tu_interference_graph_t;

/* ---- Liveness Analysis Result ---- */

typedef struct {
    tu_vreg_t           vregs[TU_LIVE_MAX_VREGS];
    uint32_t            num_vregs;

    /* Per-region interference graphs */
    tu_interference_graph_t graph_w;
    tu_interference_graph_t graph_a;
    tu_interference_graph_t graph_o;

    /* Spill statistics */
    uint32_t            num_spills;     /* Total spill/fill pairs inserted */
    uint32_t            spill_bytes;    /* Total bytes spilled */
} tu_liveness_result_t;

/* ---- Allocated Instruction Sequence ---- */

typedef struct {
    tu_instruction_t    instructions[TU_SCHED_MAX_INSTRS * 2]; /* Extra space for spills */
    uint32_t            num_instructions;
    uint32_t            peak_w_usage;   /* Peak W-SRAM usage in bytes */
    uint32_t            peak_a_usage;   /* Peak A-SRAM usage in bytes */
    uint32_t            peak_o_usage;   /* Peak O-SRAM usage in bytes */
    bool                valid;
} tu_allocated_sequence_t;

/* ================================================================
 * API
 * ================================================================ */

/*
 * Analyze liveness for an instruction sequence.
 * Extracts virtual register definitions and uses from each instruction,
 * computes live ranges (first_def to last_use for each VReg).
 *
 *   instrs:    input instruction sequence (from scheduler or parser)
 *   n_instrs:  number of instructions
 *   result:    output liveness analysis (caller-allocated)
 *
 * Returns 0 on success, -1 on too many VRegs.
 */
int tu_live_analyze(const tu_instruction_t *instrs,
                     uint32_t n_instrs,
                     tu_liveness_result_t *result);

/*
 * Build interference graphs for each SRAM region.
 * Two virtual registers interfere if their live ranges overlap:
 * they are both live at some instruction index.
 *
 * Must be called after tu_live_analyze().
 */
void tu_live_build_interference(tu_liveness_result_t *result);

/*
 * Color the interference graph: assign physical SRAM offsets to
 * virtual registers such that interfering VRegs get non-overlapping
 * physical ranges.
 *
 * Uses greedy coloring with the configured allocation strategy.
 * If spilling is enabled, VRegs that can't be colored are spilled
 * to DRAM.
 *
 * Must be called after tu_live_build_interference().
 */
void tu_live_color(tu_liveness_result_t *result,
                    const tu_live_config_t *config);

/*
 * Apply the allocation to the instruction sequence: resolve all
 * virtual SRAM offsets to physical offsets, and insert spill/fill
 * DMA instructions where VRegs were evicted.
 *
 * Must be called after tu_live_color().
 *
 * Returns 0 on success.
 */
int tu_live_apply(tu_liveness_result_t *result,
                   const tu_instruction_t *input_instrs,
                   uint32_t n_input,
                   const tu_live_config_t *config,
                   tu_allocated_sequence_t *output);

/*
 * Full allocation pass: analyze → build interference → color → apply.
 * Convenience wrapper for the 4-step pipeline.
 *
 * Returns 0 on success, -1 on error.
 */
int tu_live_allocate(const tu_instruction_t *instrs,
                      uint32_t n_instrs,
                      const tu_live_config_t *config,  /* NULL = default */
                      tu_allocated_sequence_t *output);

/*
 * Print liveness analysis results.
 */
void tu_live_print_result(const tu_liveness_result_t *result);

/*
 * Print the interference graph (for debugging).
 */
void tu_live_print_interference(const tu_interference_graph_t *graph);

#ifdef __cplusplus
}
#endif

#endif /* TU_LIVENESS_H */
