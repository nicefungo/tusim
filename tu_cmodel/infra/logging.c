/*
 * TU Structured Logging — Implementation
 * =======================================
 * Q2: Production-grade structured logging with severity filtering,
 * component tagging, file/line annotations, and execution tracing.
 */

#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Global State ---- */

static tu_log_config_t g_log_cfg = {
    .min_level      = TU_LOG_INFO,
    .use_color      = true,
    .show_timestamps = true,
    .show_component  = true,
    .show_file_line  = false,  /* Only in DEBUG/TRACE modes */
    .output          = NULL,   /* Set to stderr at init */
    .start_cycle     = 0,
    .trace_enabled   = false,
    .trace_file      = "",
    .initialized     = false,
};

/* ---- Component Name Map ---- */

static const char *g_comp_names[] = {
    [TU_COMP_CORE] = "CORE",
    [TU_COMP_MMA]  = "MMA",
    [TU_COMP_DMA]  = "DMA",
    [TU_COMP_MEM]  = "MEM",
    [TU_COMP_ISA]  = "ISA",
    [TU_COMP_CMD]  = "CMD",
    [TU_COMP_DF]   = "DF",
    [TU_COMP_PREC] = "PREC",
    [TU_COMP_TEST] = "TEST",
    [TU_COMP_PERF] = "PERF",
};

/* ---- Level Name Map ---- */

static const char *g_level_names[] = {
    [TU_LOG_NONE]    = "",
    [TU_LOG_ERROR]   = "ERROR",
    [TU_LOG_WARNING] = "WARN",
    [TU_LOG_INFO]    = "INFO",
    [TU_LOG_DEBUG]   = "DEBUG",
    [TU_LOG_TRACE]   = "TRACE",
};

/* ---- ANSI Color Codes ---- */

static const char *g_level_colors[] = {
    [TU_LOG_NONE]    = "",
    [TU_LOG_ERROR]   = "\033[1;31m",  /* Bold red */
    [TU_LOG_WARNING] = "\033[1;33m",  /* Bold yellow */
    [TU_LOG_INFO]    = "\033[1;32m",  /* Bold green */
    [TU_LOG_DEBUG]   = "\033[1;36m",  /* Bold cyan */
    [TU_LOG_TRACE]   = "\033[1;35m",  /* Bold magenta */
};
#define COLOR_RESET "\033[0m"

/* ---- Execution Trace ---- */

static tu_trace_event_t g_trace_buffer[TU_TRACE_MAX_EVENTS];
static uint32_t         g_trace_count = 0;
static uint64_t         g_trace_cycle = 0;

/* Helper: print 8-bit value as binary string */
static void fprint_b8(FILE *f, uint8_t v) {
    for (int i = 7; i >= 0; i--)
        fputc((v & (1u << i)) ? '1' : '0', f);
}

/* Helper: print 32-bit value as binary string */
static void fprint_b32(FILE *f, uint32_t v) {
    for (int i = 31; i >= 0; i--)
        fputc((v & (1u << i)) ? '1' : '0', f);
}

/* ---- Lifecycle ---- */

void tu_log_init(void) {
    tu_log_config_t defaults = {
        .min_level       = TU_LOG_INFO,
        .use_color       = true,
        .show_timestamps = true,
        .show_component  = true,
        .show_file_line  = false,
        .output          = stderr,
        .start_cycle     = 0,
        .trace_enabled   = false,
        .trace_file      = "",
        .initialized     = true,
    };
    g_log_cfg = defaults;
}

void tu_log_init_config(const tu_log_config_t *config) {
    if (config) {
        g_log_cfg = *config;
        if (!g_log_cfg.output) g_log_cfg.output = stderr;
    } else {
        tu_log_init();
    }
    g_log_cfg.initialized = true;
}

void tu_log_set_level(tu_log_level_t level) {
    g_log_cfg.min_level = level;
}

tu_log_level_t tu_log_get_level(void) {
    return g_log_cfg.min_level;
}

tu_log_config_t *tu_log_get_config(void) {
    return &g_log_cfg;
}

/* ---- Core Log Emitter ---- */

void tu_log_emit(tu_log_component_t comp, tu_log_level_t level,
                 const char *file, int line,
                 const char *fmt, ...) {
    if (!g_log_cfg.initialized) return;
    if (!g_log_cfg.output) return;

    FILE *out = g_log_cfg.output;

    /* Timestamp (relative cycles) */
    if (g_log_cfg.show_timestamps) {
        uint64_t rel = g_trace_cycle - g_log_cfg.start_cycle;
        fprintf(out, "[%8lu] ", (unsigned long)rel);
    }

    /* Component tag */
    if (g_log_cfg.show_component && comp < TU_COMP_COUNT) {
        fprintf(out, "%-5s ", g_comp_names[comp]);
    }

    /* Severity level */
    const char *lname = (level <= TU_LOG_TRACE) ? g_level_names[level] : "???";
    if (g_log_cfg.use_color && level <= TU_LOG_TRACE) {
        fprintf(out, "%s%-5s%s ", g_level_colors[level], lname, COLOR_RESET);
    } else {
        fprintf(out, "%-5s ", lname);
    }

    /* File:line (only for DEBUG/TRACE or when explicitly enabled) */
    if (g_log_cfg.show_file_line && file) {
        /* Show only the filename, not the full path */
        const char *fname = strrchr(file, '/');
        fname = fname ? fname + 1 : file;
        fprintf(out, "%s:%-4d ", fname, line);
    }

    /* Message */
    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);

    fprintf(out, "\n");
    fflush(out);
}

/* ---- Execution Trace ---- */

void tu_trace_event(uint8_t component, uint8_t opcode,
                    uint32_t op0, uint32_t op1,
                    uint32_t op2, uint32_t op3) {
    if (g_trace_count >= TU_TRACE_MAX_EVENTS) return;

    tu_trace_event_t *evt = &g_trace_buffer[g_trace_count++];
    evt->cycle     = g_trace_cycle;
    evt->component = component;
    evt->opcode    = opcode;
    evt->flags     = 0;
    evt->operand[0] = op0;
    evt->operand[1] = op1;
    evt->operand[2] = op2;
    evt->operand[3] = op3;
}

const tu_trace_event_t *tu_trace_get_buffer(uint32_t *count_out) {
    if (count_out) *count_out = g_trace_count;
    return g_trace_buffer;
}

void tu_trace_clear(void) {
    g_trace_count = 0;
    g_trace_cycle = 0;
}

uint64_t tu_trace_get_cycle(void) {
    return g_trace_cycle;
}

void tu_trace_set_cycle(uint64_t cycle) {
    g_trace_cycle = cycle;
}

/*
 * Export trace as VCD (Value Change Dump) format.
 * Compatible with GTKWave, Surfer, and other waveform viewers.
 *
 * VCD format:
 *   $date ... $end
 *   $version ... $end
 *   $timescale 1ns $end
 *   $scope module tu_core $end
 *     $var ... $end
 *   $upscope $end
 *   $enddefinitions $end
 *   #0
 *   $dumpvars
 *   ...
 *   #100
 *   1! 0" ...
 */
void tu_trace_export_vcd(FILE *output) {
    if (!output) return;

    const char *comp_names[] = {
        [TU_COMP_MMA] = "mma",  [TU_COMP_DMA] = "dma",
        [TU_COMP_MEM] = "mem",  [TU_COMP_ISA] = "isa",
        [TU_COMP_CMD] = "cmd",  [TU_COMP_DF]  = "df",
    };

    /* Header */
    fprintf(output, "$date\n  %s\n$end\n", "Hermes Agent TU CModel Trace");
    fprintf(output, "$version\n  TU CModel Trace v1.0\n$end\n");
    fprintf(output, "$timescale 1ns $end\n");

    /* Define signals */
    fprintf(output, "$scope module tu_core $end\n");
    fprintf(output, "  $var wire 8  comp  component $end\n");
    fprintf(output, "  $var wire 8  op    opcode $end\n");
    fprintf(output, "  $var wire 32  op0   operand0 $end\n");
    fprintf(output, "  $var wire 32  op1   operand1 $end\n");
    fprintf(output, "  $var wire 32  op2   operand2 $end\n");
    fprintf(output, "  $var wire 32  op3   operand3 $end\n");
    fprintf(output, "$upscope $end\n");
    fprintf(output, "$enddefinitions $end\n");

    /* Dump initial values */
    fprintf(output, "#0\n");
    fprintf(output, "$dumpvars\n");
    fprintf(output, "  b00000000 comp\n");
    fprintf(output, "  b00000000 op\n");
    fprintf(output, "  b00000000000000000000000000000000 op0\n");
    fprintf(output, "  b00000000000000000000000000000000 op1\n");
    fprintf(output, "  b00000000000000000000000000000000 op2\n");
    fprintf(output, "  b00000000000000000000000000000000 op3\n");
    fprintf(output, "$end\n");

    /* Emit events */
    uint64_t last_cycle = 0;
    for (uint32_t i = 0; i < g_trace_count; i++) {
        tu_trace_event_t *evt = &g_trace_buffer[i];

        if (evt->cycle > last_cycle) {
            fprintf(output, "#%lu\n", (unsigned long)evt->cycle);
            last_cycle = evt->cycle;
        }

        fputs("  b", output);
        fprint_b8(output, evt->component);
        fputs(" comp\n  b", output);
        fprint_b8(output, evt->opcode);
        fputs(" op\n  b", output);
        fprint_b32(output, evt->operand[0]);
        fputs(" op0\n  b", output);
        fprint_b32(output, evt->operand[1]);
        fputs(" op1\n  b", output);
        fprint_b32(output, evt->operand[2]);
        fputs(" op2\n  b", output);
        fprint_b32(output, evt->operand[3]);
        fputs(" op3\n", output);
    }

    fprintf(output, "#%lu\n", (unsigned long)g_trace_cycle);
}
