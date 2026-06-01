/*
 * TU CModel — Event Tracing (VCD/FST)
 * ====================================
 *
 * Generates Value Change Dump (VCD) files compatible with GTKWave,
 * Surfer, and other waveform viewers. VCD is the IEEE 1364-2001
 * industry-standard format for digital waveform exchange.
 *
 * Gap P2.7: Event tracing for cmodel-vs-RTL comparison and
 * microarchitecture debugging.
 *
 * Architecture:
 *   - Signal registry with unique identifier strings
 *   - Cycle-accurate change logging (only writes when values change)
 *   - VCD header generation with module hierarchy
 *   - Configurable via tu_config_t (trace_enabled, trace_file)
 *
 * VCD Format:
 *   $date / $version / $timescale → $var definitions →
 *   $dumpvars → #time value_changes → ...
 */

#ifndef TU_EVENT_TRACE_H
#define TU_EVENT_TRACE_H

#include "../tu_config.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Signal Widths ---- */

typedef enum {
    TU_TRACE_SIG_1BIT = 1,
    TU_TRACE_SIG_4BIT = 4,
    TU_TRACE_SIG_8BIT = 8,
    TU_TRACE_SIG_16BIT = 16,
    TU_TRACE_SIG_32BIT = 32,
    TU_TRACE_SIG_64BIT = 64,
} tu_trace_signal_width_t;

/* ---- Trace Context ---- */

typedef struct tu_event_trace_t {
    FILE       *file;              /* Output file handle */
    uint64_t    current_cycle;     /* Current simulation cycle */
    uint32_t    signal_count;      /* Number of registered signals */
    uint32_t    max_signals;       /* Capacity */
    bool        header_written;
    bool        definitions_ended;
    char        filename[256];

    /* Signal registry */
    struct {
        char       *id;            /* VCD identifier string (e.g., "!") */
        char       *name;          /* Hierarchical name (e.g., "TU_CORE.dma.ch0.state") */
        uint8_t     width;         /* Signal width in bits */
        uint64_t    last_value;    /* Last written value (for change detection) */
        bool        dirty;         /* Value changed since last dump */
    } *signals;
} tu_event_trace_t;

/* ---- Lifecycle ---- */

/*
 * Create a trace context and open the output file.
 * If the file exists, it is truncated.
 * Returns the trace context (caller owns), or NULL on failure.
 */
tu_event_trace_t *tu_trace_create(const char *filename, uint32_t max_signals);

/*
 * Close the trace file and free the context.
 * Writes the VCD end-of-file marker.
 */
void tu_trace_close(tu_event_trace_t *trace);

/* ---- Signal Management ---- */

/*
 * Register a new trace signal.
 * id: unique short identifier (e.g., "!", "#", "$", "%", "a", "b", ...)
 *     Use printable ASCII characters, 1 char per bit for multi-bit signals.
 * name: hierarchical name (e.g., "TU_CORE.dma.ch0.active")
 * width: bit width of the signal
 * Returns: signal index (0-based), or -1 on error
 */
int tu_trace_add_signal(tu_event_trace_t *trace,
                         const char *id, const char *name,
                         tu_trace_signal_width_t width);

/*
 * Queue a signal value change. The change is written to the file
 * on the next tu_trace_tick() call (or immediately if cycle has advanced).
 */
void tu_trace_signal(tu_event_trace_t *trace,
                      int signal_index, uint64_t value);

/*
 * Advance the simulation cycle and write all pending signal changes
 * to the VCD file.
 */
void tu_trace_tick(tu_event_trace_t *trace, uint64_t cycles);

/* ---- Quick API (no context needed) ---- */

/*
 * Check if tracing is globally enabled (from config).
 */
bool tu_trace_is_enabled(void);

/* ---- Predefined Signal IDs ---- */

/* Standard signal identifier characters */
#define TU_TRACE_ID_CYCLE          "!"
#define TU_TRACE_ID_DMA_CH0_STATE  "#"
#define TU_TRACE_ID_DMA_CH1_STATE  "$"
#define TU_TRACE_ID_DMA_CH2_STATE  "%"
#define TU_TRACE_ID_COMPUTE_ACTIVE "&"
#define TU_TRACE_ID_COMPUTE_OPCODE "'"
#define TU_TRACE_ID_SRAM_ACCESS    "("
#define TU_TRACE_ID_SRAM_BANK      ")"
#define TU_TRACE_ID_CMDQ_DEPTH     "*"
#define TU_TRACE_ID_CMDQ_SUBMIT    "+"
#define TU_TRACE_ID_DRAM_ACCESS    ","
#define TU_TRACE_ID_TILE_COUNT     "-"

#ifdef __cplusplus
}
#endif

#endif /* TU_EVENT_TRACE_H */
