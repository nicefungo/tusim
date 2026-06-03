/*
 * TU Observability & Debug Hooks — Gap I3
 * =========================================
 *
 * Production-grade debug infrastructure for the TU cmodel.
 * Addresses the observability gap between the functional cmodel
 * and what's needed for microarchitecture bring-up and cmodel-vs-RTL
 * discrepancy diagnosis.
 *
 * Three subsystems:
 *
 *   1. STATE DUMP — Structured dump of any TU core's internal state:
 *      SRAM contents (W/A/O buffers), partial sums, tile buffers,
 *      pipeline stage registers, DMA descriptor queues, command queue
 *      entries, performance counters, and dataflow plugin state.
 *      Output: human-readable text, JSON, or binary snapshot.
 *
 *   2. DETERMINISTIC REPLAY — Record every instruction executed
 *      (opcode + operands + result checksum) into a compact binary
 *      trace. Replay the same trace against any TU core instance and
 *      verify bit-exact reproduction. Critical for debugging
 *      non-deterministic or config-dependent behavior.
 *
 *   3. INVARIANT ASSERTIONS — Built-in self-checking assertions
 *      that run at key execution points:
 *        - Accumulator range: no NaN/Inf unless explicitly allowed
 *        - Memory alignment: all SRAM accesses are properly aligned
 *        - Buffer bounds: no reads/writes beyond allocated regions
 *        - Pipeline consistency: tile dimensions match PE array size
 *        - Dataflow invariants: data direction matches selected mode
 *      Assertions are configurable (always/never/sampling).
 *
 * Design principles:
 *   - Zero-overhead when disabled (compile-time TU_DEBUG_ENABLED)
 *   - Minimal overhead when enabled (O(1) per checkpoint)
 *   - No external dependencies (pure C11 + stdio)
 *   - Self-describing snapshots (header + version for compatibility)
 *   - Thread-safe (lock-free per-core snapshots)
 */

#ifndef TU_DEBUG_H
#define TU_DEBUG_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Compile-time enable/disable ---- */
#ifndef TU_DEBUG_ENABLED
#define TU_DEBUG_ENABLED 1
#endif

/* ---- Forward declarations ---- */
typedef struct tu_core_t tu_core_t;

/* ================================================================
 * 1. STATE DUMP
 * ================================================================ */

/* Dump output format */
typedef enum {
    TU_DUMP_TEXT   = 0,  /* Human-readable ASCII */
    TU_DUMP_JSON   = 1,  /* JSON with key-value pairs */
    TU_DUMP_BINARY = 2,  /* Compact binary snapshot for replay */
} tu_dump_format_t;

/* What to include in the dump */
typedef enum {
    TU_DUMP_SRAM        = 0x0001,  /* Full W/A/O SRAM contents */
    TU_DUMP_DMA         = 0x0002,  /* DMA engine state + descriptor queues */
    TU_DUMP_CMD_QUEUE   = 0x0004,  /* Command queue entries */
    TU_DUMP_COUNTERS    = 0x0008,  /* Performance counters */
    TU_DUMP_PIPELINE    = 0x0010,  /* Pipeline stage registers */
    TU_DUMP_DATAFLOW    = 0x0020,  /* Dataflow plugin state */
    TU_DUMP_CHECKSUMS   = 0x0040,  /* SRAM region checksums */
    TU_DUMP_ALL         = 0xFFFF,  /* Everything */
} tu_dump_flags_t;

/* Binary snapshot header (for TU_DUMP_BINARY format) */
#define TU_SNAPSHOT_MAGIC   0x54554442  /* "TUDB" */
#define TU_SNAPSHOT_VERSION 1

typedef struct {
    uint32_t    magic;          /* TU_SNAPSHOT_MAGIC */
    uint32_t    version;        /* TU_SNAPSHOT_VERSION */
    uint32_t    core_id;        /* Which core this snapshot is from */
    uint64_t    timestamp;      /* Snapshot timestamp (ns since epoch) */
    uint64_t    cycle;          /* Cycle counter at snapshot time */
    uint32_t    flags;          /* tu_dump_flags_t mask */
    uint32_t    pe_rows;        /* PE array dimensions at snapshot time */
    uint32_t    pe_cols;
    uint32_t    sram_w_size;    /* SRAM region sizes (bytes) */
    uint32_t    sram_a_size;
    uint32_t    sram_o_size;
    uint32_t    checksum_w;     /* Pre-computed checksums (CRC32) */
    uint32_t    checksum_a;
    uint32_t    checksum_o;
    uint32_t    reserved[4];    /* For future expansion */
} tu_snapshot_header_t;

/*
 * Dump a TU core's internal state to a file/stream.
 *
 * core:   the TU core to dump (NULL = default core)
 * out:    output stream (stdout, a file, etc.)
 * format: output format (text, JSON, binary)
 * flags:  what to include (bitmask of tu_dump_flags_t)
 *
 * Returns the number of bytes written, or 0 on error.
 */
size_t tu_debug_dump_state(const tu_core_t *core,
                           FILE *out,
                           tu_dump_format_t format,
                           uint32_t flags);

/*
 * Dump just the SRAM region checksums (fast, no large data transfer).
 * Useful as a lightweight consistency check between runs.
 *
 * Returns a 32-bit combined checksum, or 0 on error.
 */
uint32_t tu_debug_checksum_sram(const tu_core_t *core);

/*
 * Compare two SRAM regions byte-by-byte and report differences.
 *
 * core_a, core_b:  two cores to compare (or NULL for default)
 * region:          'W', 'A', or 'O'
 * out:             where to write the diff report
 *
 * Returns the number of differing bytes, or -1 on error.
 */
int tu_debug_diff_sram(const tu_core_t *core_a,
                       const tu_core_t *core_b,
                       char region,
                       FILE *out);

/* ================================================================
 * 2. DETERMINISTIC REPLAY
 * ================================================================ */

/* A single instruction trace entry */
typedef struct {
    uint64_t    cycle;          /* Cycle when instruction issued */
    uint8_t     opcode;         /* TU ISA opcode */
    uint8_t     flags;          /* Instruction flags */
    uint16_t    dim0;           /* Operand dimensions */
    uint8_t     dim1;
    uint8_t     dim2;
    uint16_t    reserved;
    uint64_t    immediates;     /* Immediate operands */
    uint32_t    checksum_before;/* SRAM checksum before execution */
    uint32_t    checksum_delta; /* Checksum change (after - before) */
} tu_replay_entry_t;

/* Maximum trace length (configurable at compile time) */
#ifndef TU_REPLAY_MAX_ENTRIES
#define TU_REPLAY_MAX_ENTRIES 1048576  /* 1M entries (~64 MB) */
#endif

/* Replay trace buffer */
typedef struct {
    tu_replay_entry_t  *entries;
    uint32_t            count;
    uint32_t            capacity;
    bool                recording;
    uint64_t            start_cycle;
} tu_replay_trace_t;

/*
 * Start recording an instruction trace.
 *
 * trace: pointer to a tu_replay_trace_t (caller-allocated)
 * capacity: max entries (0 = use TU_REPLAY_MAX_ENTRIES)
 *
 * Returns 0 on success, -1 if already recording.
 */
int tu_debug_record_start(tu_replay_trace_t *trace, uint32_t capacity);

/*
 * Record a single instruction into the trace buffer.
 * Called automatically by the command queue / ISA interpreter
 * when recording is active.
 *
 * Returns 0 on success, -1 if buffer full.
 */
int tu_debug_record_instr(tu_replay_trace_t *trace,
                          uint64_t cycle,
                          uint8_t opcode, uint8_t flags,
                          uint16_t dim0, uint8_t dim1, uint8_t dim2,
                          uint64_t immediates,
                          uint32_t checksum_before,
                          uint32_t checksum_after);

/*
 * Stop recording and finalize the trace.
 * Returns the total number of recorded entries.
 */
uint32_t tu_debug_record_stop(tu_replay_trace_t *trace);

/*
 * Save a recorded trace to a file (binary format).
 * Returns the number of bytes written.
 */
size_t tu_debug_record_save(const tu_replay_trace_t *trace, FILE *out);

/*
 * Load a trace from a file into a trace buffer.
 * Returns the number of entries loaded, 0 on error.
 */
uint32_t tu_debug_record_load(tu_replay_trace_t *trace, FILE *in);

/*
 * Replay a recorded trace against a TU core.
 *
 * Each instruction is re-issued in order. After each instruction,
 * the SRAM checksum is compared against the recorded checksum.
 * Mismatches are reported to `out`.
 *
 * Returns the number of mismatches, or -1 on fatal error.
 */
int tu_debug_replay_execute(tu_core_t *core,
                            const tu_replay_trace_t *trace,
                            FILE *out);

/*
 * Free resources associated with a trace buffer.
 */
void tu_debug_record_destroy(tu_replay_trace_t *trace);

/* ================================================================
 * 3. INVARIANT ASSERTIONS
 * ================================================================ */

/* Assertion severity */
typedef enum {
    TU_ASSERT_IGNORE  = 0,  /* Disabled */
    TU_ASSERT_WARN    = 1,  /* Log warning, continue */
    TU_ASSERT_ERROR   = 2,  /* Log error, return error code */
    TU_ASSERT_ABORT   = 3,  /* Log error, abort() */
} tu_assert_mode_t;

/* Assertion categories (individually configurable) */
typedef enum {
    TU_ASSERT_CAT_RANGE      = 0,  /* Accumulator range (NaN/Inf) */
    TU_ASSERT_CAT_ALIGNMENT  = 1,  /* Memory alignment */
    TU_ASSERT_CAT_BOUNDS     = 2,  /* Buffer bounds */
    TU_ASSERT_CAT_PIPELINE   = 3,  /* Pipeline consistency */
    TU_ASSERT_CAT_DATAFLOW   = 4,  /* Dataflow invariants */
    TU_ASSERT_CAT_DTYPE      = 5,  /* Data type invariants */
    TU_ASSERT_CAT_NUM        = 6,
} tu_assert_category_t;

/*
 * Configure assertion behavior.
 *
 * category: which assertion category
 * mode:     how to handle violations
 *
 * Default: all categories = TU_ASSERT_WARN
 */
void tu_debug_assert_set_mode(tu_assert_category_t category,
                              tu_assert_mode_t mode);

/*
 * Set all categories at once.
 */
void tu_debug_assert_set_all(tu_assert_mode_t mode);

/*
 * Get the current mode for a category.
 */
tu_assert_mode_t tu_debug_assert_get_mode(tu_assert_category_t category);

/* ---- Assertion check functions (called by cmodel internals) ---- */

/*
 * Check that an FP32 accumulator value is in the valid range
 * (not NaN, not Inf, or within configured bounds).
 *
 * Returns true if the assertion passes, false if it fails
 * (and triggers the configured mode).
 */
bool tu_debug_assert_range(float value, const char *context);

/*
 * Check that an SRAM address is properly aligned for the data type.
 *
 * addr:     byte address within the SRAM region
 * elem_size: bytes per element (2 for FP16, 4 for FP32, etc.)
 *
 * Returns true if aligned, false otherwise.
 */
bool tu_debug_assert_alignment(uint32_t addr, uint32_t elem_size,
                               const char *context);

/*
 * Check that an SRAM access is within the allocated region bounds.
 *
 * addr:  byte address
 * size:  access size in bytes
 * limit: region size in bytes
 *
 * Returns true if in bounds, false otherwise.
 */
bool tu_debug_assert_bounds(uint32_t addr, uint32_t size,
                            uint32_t limit, const char *context);

/*
 * Check that a tile operation matches the PE array dimensions.
 *
 * tile_M, tile_N: tile dimensions
 * pe_rows, pe_cols: PE array dimensions
 *
 * Returns true if consistent, false otherwise.
 */
bool tu_debug_assert_tile_dims(uint16_t tile_M, uint16_t tile_N,
                               uint16_t pe_rows, uint16_t pe_cols,
                               const char *context);

/*
 * Check that a dataflow operation direction matches the configured
 * dataflow mode.
 *
 * dataflow: configured dataflow (WS=0, OS=1, RS=2, NLR=3)
 * op_dataflow: operation-requested dataflow (-1 = inherit)
 * direction: 'R' (read) or 'W' (write)
 * buffer: 'W', 'A', or 'O'
 *
 * Returns true if consistent, false otherwise.
 */
bool tu_debug_assert_dataflow(int dataflow, int op_dataflow,
                              char direction, char buffer,
                              const char *context);

/* ---- Statistics ---- */

/*
 * Get assertion violation counts.
 */
typedef struct {
    uint32_t violations[TU_ASSERT_CAT_NUM];  /* Per-category count */
    uint32_t total_checks;                    /* Total assertions checked */
    uint32_t total_violations;               /* Total violations */
} tu_assert_stats_t;

void tu_debug_assert_get_stats(tu_assert_stats_t *stats);
void tu_debug_assert_reset_stats(void);

/* ================================================================
 * 4. CONVENIENCE: Full debug report
 * ================================================================ */

/*
 * Generate a comprehensive debug report for a TU core.
 *
 * Includes:
 *   - Text state dump (all SRAM, DMA, CMDQ, counters)
 *   - SRAM checksums
 *   - Assertion statistics
 *   - Configuration summary
 *
 * Writes to `out`.
 */
void tu_debug_report(const tu_core_t *core, FILE *out);

#ifdef __cplusplus
}
#endif

#endif /* TU_DEBUG_H */
