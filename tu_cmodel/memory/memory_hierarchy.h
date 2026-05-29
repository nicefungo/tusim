/*
 * TU CModel — Multi-Level Memory Hierarchy
 * ==========================================
 * Gap A3: Flat 3-buffer SRAM → Multi-level hierarchy with
 *         configurable banking and level-aware access costs.
 *
 * Architecture:
 *   Four canonical levels mirroring production accelerators
 *   (TPU, Gemmini, Eyeriss):
 *
 *     Level 0 — REGFILE     : Per-PE register file (~256 B/PE)
 *                               1-cycle access, no banking overhead
 *     Level 1 — LOCAL_SPAD  : Per-core scratchpad (banked SRAM)
 *                               Configurable banks, BW metering
 *     Level 2 — GLOBAL_BUF  : Shared L2 buffer (across cores)
 *                               Higher latency, wider banking
 *     Level 3 — DRAM        : Off-chip (delegates to dram_model)
 *                               HBM/DDR, high latency, BW contention
 *
 *   Every tu_sram_region_t is tagged with a level. The hierarchy
 *   provides unified access with per-level cost modeling.
 *
 *   Global buffer is a new addition — a shared, banked SRAM that
 *   sits between per-core scratchpads and DRAM. It is modeled as
 *   a tu_sram_region_t with TU_MEM_GLOBAL_BUFFER level and
 *   configurable size/banks independent of per-core SPADs.
 */

#ifndef TU_MEMORY_HIERARCHY_H
#define TU_MEMORY_HIERARCHY_H

#include "../tu_config.h"
#include "../tu_sram.h"
#include "dram_model.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Memory Levels ---- */
typedef enum {
    TU_MEM_REGFILE      = 0,  /* Per-PE register file */
    TU_MEM_LOCAL_SPAD   = 1,  /* Per-core local scratchpad */
    TU_MEM_GLOBAL_BUF   = 2,  /* Shared global buffer / L2 */
    TU_MEM_DRAM         = 3,  /* Off-chip DRAM */
    TU_MEM_NUM_LEVELS   = 4
} tu_mem_level_t;

/* ---- Per-Level Configuration ---- */
typedef struct {
    tu_mem_level_t  level;
    const char     *name;             /* Human-readable */
    uint32_t        size_bytes;       /* Total capacity */
    uint32_t        bank_count;       /* Number of banks */
    uint32_t        bank_width;       /* Bytes per bank word */
    uint32_t        read_latency;     /* Base read latency in cycles */
    uint32_t        write_latency;    /* Base write latency in cycles */
    uint32_t        words_per_cycle;  /* BW: words per bank per cycle */
    uint32_t        bw_window_cycles; /* BW refill window */
    uint32_t        stall_penalty;    /* Cycles added per BW stall */
    bool            double_buffered;  /* Ping-pong buffer support */
} tu_mem_level_config_t;

/* ---- Global Buffer (Level 2) ---- */
typedef struct {
    tu_sram_region_t    sram;          /* Underlying banked SRAM */
    tu_mem_level_config_t config;       /* Snapshot of config */
    uint64_t            total_hits;     /* Access statistics */
    uint64_t            total_misses;   /* Accesses that missed Gbuf */
} tu_global_buffer_t;

/* ---- Register File Model (Level 0) ---- */
typedef struct {
    uint32_t    size_per_pe;      /* Bytes per PE */
    uint32_t    num_pes;          /* PE count (rows × cols) */
    uint64_t    total_reads;      /* Aggregated stats */
    uint64_t    total_writes;
    /* In functional mode, RegFile is unlimited (no allocation tracking).
     * In cycle-accurate mode, per-PE allocation is tracked. */
    bool        track_allocation;
} tu_regfile_model_t;

/* ---- Memory Hierarchy (top-level) ---- */
typedef struct {
    /* Per-level backends */
    tu_regfile_model_t   regfile;       /* Level 0 */
    /* Level 1 uses existing tu_sram_region_t arrays (per-region) */
    tu_global_buffer_t   gbuf;          /* Level 2 */
    tu_dram_model_t     *dram;          /* Level 3 */

    /* Level-aware access tracking */
    uint64_t            level_reads[TU_MEM_NUM_LEVELS];
    uint64_t            level_writes[TU_MEM_NUM_LEVELS];
    uint64_t            level_bytes_read[TU_MEM_NUM_LEVELS];
    uint64_t            level_bytes_written[TU_MEM_NUM_LEVELS];
    uint64_t            level_stall_cycles[TU_MEM_NUM_LEVELS];

    /* Configuration */
    tu_mem_level_config_t level_configs[TU_MEM_NUM_LEVELS];
    bool                initialized;

    /* Cycle counter */
    uint64_t            current_cycle;
} tu_memory_hierarchy_t;

/* ---- Lifecycle ---- */

/* Initialize the full memory hierarchy from config.
 * Creates RegFile model, Global Buffer, and DRAM model.
 * Local SPADs are initialized separately via tu_sram_init. */
void tu_mem_hierarchy_init(tu_memory_hierarchy_t *h);

/* Set a custom configuration for a specific level.
 * Call BEFORE tu_mem_hierarchy_init() or re-init. */
void tu_mem_hierarchy_set_level_config(tu_memory_hierarchy_t *h,
                                        tu_mem_level_t level,
                                        const tu_mem_level_config_t *config);

/* Destroy and free all resources. */
void tu_mem_hierarchy_destroy(tu_memory_hierarchy_t *h);

/* Reset all statistics, keep configuration. */
void tu_mem_hierarchy_reset(tu_memory_hierarchy_t *h);

/* ---- Access API (level-aware) ---- */

/*
 * Read `bytes` from a memory region at the specified level.
 *
 *   h:       hierarchy instance
 *   level:   which memory level (REGFILE / SPAD / GBUF / DRAM)
 *   region:  the tu_sram_region_t (for SPAD/GBUF levels; NULL for REGFILE/DRAM)
 *   addr:    byte offset within the level
 *   out:     buffer to receive data
 *   bytes:   number of bytes to read
 *   stall_out: receives total stall cycles (can be NULL)
 *
 * Returns 0 on success, -1 on bounds violation.
 */
int tu_mem_hierarchy_read(tu_memory_hierarchy_t *h,
                           tu_mem_level_t level,
                           tu_sram_region_t *region,
                           uint32_t addr,
                           void *out, uint32_t bytes,
                           uint64_t *stall_out);

/*
 * Write `bytes` to a memory region at the specified level.
 * Same parameter semantics as tu_mem_hierarchy_read().
 */
int tu_mem_hierarchy_write(tu_memory_hierarchy_t *h,
                            tu_mem_level_t level,
                            tu_sram_region_t *region,
                            uint32_t addr,
                            const void *data, uint32_t bytes,
                            uint64_t *stall_out);

/* Convenience: read a single word at the given level. */
int tu_mem_hierarchy_read_word(tu_memory_hierarchy_t *h,
                                tu_mem_level_t level,
                                tu_sram_region_t *region,
                                uint32_t addr,
                                void *out, uint32_t word_bytes,
                                uint64_t *stall_out);

/* Convenience: write a single word at the given level. */
int tu_mem_hierarchy_write_word(tu_memory_hierarchy_t *h,
                                 tu_mem_level_t level,
                                 tu_sram_region_t *region,
                                 uint32_t addr,
                                 const void *data, uint32_t word_bytes,
                                 uint64_t *stall_out);

/* ---- Global Buffer API ---- */

/* Initialize (or re-initialize) the global buffer. */
void tu_gbuf_init(tu_global_buffer_t *gbuf, const tu_mem_level_config_t *config);

/* Destroy the global buffer. */
void tu_gbuf_destroy(tu_global_buffer_t *gbuf);

/* Access the global buffer's SRAM region directly. */
tu_sram_region_t *tu_gbuf_get_sram(tu_global_buffer_t *gbuf);

/* Check if an address range fits entirely within the global buffer. */
bool tu_gbuf_contains(const tu_global_buffer_t *gbuf,
                      uint32_t addr, uint32_t bytes);

/* ---- RegFile API ---- */

/* Initialize the register file model. */
void tu_regfile_init(tu_regfile_model_t *rf,
                     uint32_t size_per_pe, uint32_t num_pes);

/* Record a RegFile access with cycle accounting.
 * In functional mode, RegFile accesses are zero-latency.
 * In cycle-accurate mode, models limited ports per PE. */
void tu_regfile_record_read(tu_regfile_model_t *rf, uint32_t count);
void tu_regfile_record_write(tu_regfile_model_t *rf, uint32_t count);

/* ---- Statistics & Reporting ---- */

/* Get a human-readable level name. */
const char *tu_mem_level_name(tu_mem_level_t level);

/* Print hierarchical memory statistics. */
void tu_mem_hierarchy_print_stats(const tu_memory_hierarchy_t *h, FILE *out);

/* Get per-level byte count. */
uint64_t tu_mem_hierarchy_get_bytes(const tu_memory_hierarchy_t *h,
                                     tu_mem_level_t level);

/* Get per-level stall cycles. */
uint64_t tu_mem_hierarchy_get_stalls(const tu_memory_hierarchy_t *h,
                                      tu_mem_level_t level);

/* Get total size across all on-chip levels (excludes DRAM). */
uint64_t tu_mem_hierarchy_get_onchip_total(const tu_memory_hierarchy_t *h);

/* ---- Cycle Management ---- */

/* Advance all memory hierarchy cycles (triggers BW refill). */
void tu_mem_hierarchy_tick(tu_memory_hierarchy_t *h, uint64_t cycles);

/* Get current cycle. */
uint64_t tu_mem_hierarchy_get_cycle(const tu_memory_hierarchy_t *h);

#ifdef __cplusplus
}
#endif

#endif /* TU_MEMORY_HIERARCHY_H */
