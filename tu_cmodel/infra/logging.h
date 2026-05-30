/*
 * TU Structured Logging System
 * =============================
 * Q2: Production-grade structured logging with severity levels, component
 * tagging, runtime filtering, file/line annotations, and execution tracing.
 *
 * Design principles:
 *   - Zero-allocation in the fast path (compile-time level checks)
 *   - Per-component log categories (TU_CORE, TU_MMA, TU_DMA, TU_MEM, TU_ISA, TU_DF)
 *   - Runtime severity threshold (set via tu_log_set_level())
 *   - Structured output: timestamps, component, level, file:line, message
 *   - Execution trace: instruction-level event log for VCD/FST export
 *
 * Usage:
 *   TU_LOG(TU_CORE, TU_INFO, "PE array: %u×%u", pe_rows, pe_cols);
 *   TU_LOG_ERR(TU_DMA, "Transfer overflow: addr=%08x size=%u", addr, sz);
 *   TU_LOG_DBG(TU_MEM, "Bank conflict: addr1=%08x addr2=%08x", a1, a2);
 */

#ifndef TU_LOGGING_H
#define TU_LOGGING_H

#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Log Severity Levels ---- */

typedef enum {
    TU_LOG_NONE    = 0,  /* Silent — no output */
    TU_LOG_ERROR   = 1,  /* Fatal/unrecoverable errors */
    TU_LOG_WARNING = 2,  /* Recoverable issues, unexpected conditions */
    TU_LOG_INFO    = 3,  /* Normal operational messages */
    TU_LOG_DEBUG   = 4,  /* Detailed diagnostic info */
    TU_LOG_TRACE   = 5,  /* Per-instruction/per-cycle tracing */
} tu_log_level_t;

/* ---- Component / Subsystem Tags ---- */

typedef enum {
    TU_COMP_CORE   = 0,  /* tu_core: init, lifecycle, top-level */
    TU_COMP_MMA    = 1,  /* Systolic array / compute engine */
    TU_COMP_DMA    = 2,  /* DMA engine and descriptors */
    TU_COMP_MEM    = 3,  /* Memory system: SRAM, DRAM, alloc */
    TU_COMP_ISA    = 4,  /* ISA encoder/decoder, ASM interpreter */
    TU_COMP_CMD    = 5,  /* Command queue */
    TU_COMP_DF     = 6,  /* Dataflow plugins */
    TU_COMP_PREC   = 7,  /* Precision: FP16/FP8/BF16/INT8 conversion */
    TU_COMP_TEST   = 8,  /* Test harness */
    TU_COMP_PERF   = 9,  /* Performance counters */
    TU_COMP_COUNT
} tu_log_component_t;

/* ---- Log Configuration ---- */

typedef struct {
    tu_log_level_t   min_level;       /* Minimum severity to emit */
    bool             use_color;       /* ANSI color codes in output */
    bool             show_timestamps; /* Prefix with relative timestamp */
    bool             show_component;  /* Show component tag */
    bool             show_file_line;  /* Show source file:line */
    FILE            *output;          /* Output stream (default: stderr) */
    uint64_t         start_cycle;     /* Cycle counter at init (for timestamps) */
    bool             trace_enabled;   /* Enable execution trace recording */
    char             trace_file[256]; /* Trace output filename */
    bool             initialized;
} tu_log_config_t;

/* ---- Public API ---- */

/* Initialize the logging system with default config */
void tu_log_init(void);

/* Initialize with custom config */
void tu_log_init_config(const tu_log_config_t *config);

/* Set minimum severity level at runtime */
void tu_log_set_level(tu_log_level_t level);

/* Get current minimum level */
tu_log_level_t tu_log_get_level(void);

/* Get the global log config (for inspection/modification) */
tu_log_config_t *tu_log_get_config(void);

/* Core log function — called by macros, not directly */
void tu_log_emit(tu_log_component_t comp, tu_log_level_t level,
                 const char *file, int line,
                 const char *fmt, ...)
#ifdef __GNUC__
    __attribute__((format(printf, 5, 6)))
#endif
    ;

/* ---- Convenience Macros ---- */

#define TU_LOG(comp, level, ...) do { \
    tu_log_config_t *_cfg = tu_log_get_config(); \
    if (_cfg->initialized && (level) <= _cfg->min_level) \
        tu_log_emit((comp), (level), __FILE__, __LINE__, __VA_ARGS__); \
} while(0)

#define TU_LOG_ERR(comp, ...)   TU_LOG((comp), TU_LOG_ERROR,   __VA_ARGS__)
#define TU_LOG_WARN(comp, ...)  TU_LOG((comp), TU_LOG_WARNING, __VA_ARGS__)
#define TU_LOG_INFO(comp, ...)  TU_LOG((comp), TU_LOG_INFO,    __VA_ARGS__)
#define TU_LOG_DBG(comp, ...)   TU_LOG((comp), TU_LOG_DEBUG,   __VA_ARGS__)
#define TU_LOG_TRACE(comp, ...) TU_LOG((comp), TU_LOG_TRACE,   __VA_ARGS__)

/* ---- Execution Trace (for VCD/FST generation) ---- */

/*
 * Record an execution event in the trace buffer.
 * These events can later be exported as VCD signal changes.
 *
 * Events record: cycle, component, opcode, and up to 4 operands.
 */
typedef struct {
    uint64_t   cycle;         /* Cycle counter when event occurred */
    uint8_t    component;     /* TU_COMP_* */
    uint8_t    opcode;        /* Operation or event type */
    uint16_t   flags;         /* Event-specific flags */
    uint32_t   operand[4];    /* Up to 4 operands (addresses, values, sizes) */
} tu_trace_event_t;

/* Maximum events in the trace buffer */
#define TU_TRACE_MAX_EVENTS 65536

/* Record a trace event */
void tu_trace_event(uint8_t component, uint8_t opcode,
                    uint32_t op0, uint32_t op1,
                    uint32_t op2, uint32_t op3);

/* Get the trace buffer and count */
const tu_trace_event_t *tu_trace_get_buffer(uint32_t *count_out);

/* Clear the trace buffer */
void tu_trace_clear(void);

/* Export trace as VCD signal change records */
void tu_trace_export_vcd(FILE *output);

/* Get the current cycle counter for trace events */
uint64_t tu_trace_get_cycle(void);

/* Set the current cycle counter (called by the cycle model) */
void tu_trace_set_cycle(uint64_t cycle);

#ifdef __cplusplus
}
#endif

#endif /* TU_LOGGING_H */
