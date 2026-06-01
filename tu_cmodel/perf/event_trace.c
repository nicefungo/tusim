/*
 * TU CModel — Event Tracing Implementation (VCD/FST)
 * ====================================================
 *
 * Generates IEEE 1364-2001 VCD files. Key design decisions:
 *
 * 1. CHANGE-DETECTION: Values are only written when they actually change.
 *    This keeps VCD files compact and matches hardware behavior.
 *
 * 2. BATCHED WRITES: Multiple signal changes in the same cycle are
 *    batched under a single #time header.
 *
 * 3. VCD ID SYSTEM: Each signal gets a short ASCII identifier string.
 *    1-bit: single char (e.g., "!"), multi-bit: one char per bit,
 *    bus values are written as binary strings.
 *
 * 4. BUFFERED I/O: Uses fprintf with file buffering for performance.
 */

#define _GNU_SOURCE
#include "event_trace.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

/* ---- Internal helpers ---- */

/* Generate a timestamp string for the VCD header */
static void write_vcd_header(tu_event_trace_t *t) {
    if (t->header_written) return;

    time_t now = time(NULL);
    char *date_str = ctime(&now);
    /* Strip newline from ctime */
    date_str[strcspn(date_str, "\n")] = '\0';

    fprintf(t->file, "$date\n  %s\n$end\n", date_str);
    fprintf(t->file, "$version\n  TU CModel v2.0 — Event Trace\n$end\n");
    fprintf(t->file, "$timescale 1 ns $end\n");

    /* Module hierarchy — single scope for simplicity */
    fprintf(t->file, "$scope module TU_CORE $end\n");

    /* Variable declarations */
    for (uint32_t i = 0; i < t->signal_count; i++) {
        const char *type = (t->signals[i].width == 1) ? "wire" : "reg";
        fprintf(t->file, "$var %s %d %s %s $end\n",
                type, t->signals[i].width,
                t->signals[i].id, t->signals[i].name);
    }

    fprintf(t->file, "$upscope $end\n");
    fprintf(t->file, "$enddefinitions $end\n");

    /* Dump initial values (all zeros) */
    fprintf(t->file, "$dumpvars\n");
    for (uint32_t i = 0; i < t->signal_count; i++) {
        if (t->signals[i].width == 1) {
            fprintf(t->file, "0%s\n", t->signals[i].id);
        } else {
            /* Multi-bit: write binary string of '0's */
            fprintf(t->file, "b");
            for (uint8_t w = 0; w < t->signals[i].width; w++)
                fputc('0', t->file);
            fprintf(t->file, " %s\n", t->signals[i].id);
        }
    }
    fprintf(t->file, "$end\n");

    /* First timestamp */
    fprintf(t->file, "#0\n");

    t->header_written = true;
    t->definitions_ended = true;
}

static void write_signal_change(tu_event_trace_t *t, uint32_t sig_idx) {
    if (!t->signals[sig_idx].dirty) return;

    uint64_t val = t->signals[sig_idx].last_value;
    uint8_t width = t->signals[sig_idx].width;
    const char *id = t->signals[sig_idx].id;

    if (width == 1) {
        fprintf(t->file, "%c%s\n", (val & 1) ? '1' : '0', id);
    } else {
        /* Write binary string, MSB first */
        fprintf(t->file, "b");
        for (int b = width - 1; b >= 0; b--) {
            fputc((val & (1ULL << b)) ? '1' : '0', t->file);
        }
        fprintf(t->file, " %s\n", id);
    }

    t->signals[sig_idx].dirty = false;
}

/* ---- Public API ---- */

tu_event_trace_t *tu_trace_create(const char *filename, uint32_t max_signals) {
    tu_event_trace_t *t = calloc(1, sizeof(*t));
    if (!t) return NULL;

    t->file = fopen(filename, "w");
    if (!t->file) {
        free(t);
        return NULL;
    }

    /* Line-buffered — each fprintf line hits disk for test inspection */
    setvbuf(t->file, NULL, _IOLBF, 0);

    t->max_signals = max_signals > 0 ? max_signals : 64;
    t->signals = calloc(t->max_signals, sizeof(t->signals[0]));
    if (!t->signals) {
        fclose(t->file);
        free(t);
        return NULL;
    }

    t->current_cycle = 0;
    t->signal_count = 0;
    strncpy(t->filename, filename, sizeof(t->filename) - 1);

    return t;
}

void tu_trace_close(tu_event_trace_t *trace) {
    if (!trace) return;

    /* Write VCD end marker */
    if (trace->file) {
        /* Flush any pending changes at current cycle */
        bool any_dirty = false;
        for (uint32_t i = 0; i < trace->signal_count; i++) {
            if (trace->signals[i].dirty) {
                any_dirty = true;
                break;
            }
        }
        if (any_dirty && trace->header_written) {
            for (uint32_t i = 0; i < trace->signal_count; i++) {
                write_signal_change(trace, i);
            }
        }

        /* Advance to final timestamp */
        if (trace->header_written) {
            fprintf(trace->file, "#%lu\n", (unsigned long)trace->current_cycle);
        }

        fclose(trace->file);
    }

    /* Free signal strings */
    for (uint32_t i = 0; i < trace->signal_count; i++) {
        free(trace->signals[i].id);
        free(trace->signals[i].name);
    }
    free(trace->signals);
    free(trace);
}

int tu_trace_add_signal(tu_event_trace_t *trace,
                         const char *id, const char *name,
                         tu_trace_signal_width_t width) {
    if (!trace || !id || !name) return -1;
    if (trace->signal_count >= trace->max_signals) return -1;

    uint32_t idx = trace->signal_count;
    trace->signals[idx].id = strdup(id);
    trace->signals[idx].name = strdup(name);
    trace->signals[idx].width = (uint8_t)width;
    trace->signals[idx].last_value = 0;
    trace->signals[idx].dirty = false;

    trace->signal_count++;
    return (int)idx;
}

void tu_trace_signal(tu_event_trace_t *trace,
                      int signal_index, uint64_t value) {
    if (!trace || signal_index < 0 || (uint32_t)signal_index >= trace->signal_count)
        return;

    /* Only mark dirty if value actually changed */
    if (trace->signals[signal_index].last_value != value) {
        trace->signals[signal_index].last_value = value;
        trace->signals[signal_index].dirty = true;
    }
}

void tu_trace_tick(tu_event_trace_t *trace, uint64_t cycles) {
    if (!trace || !trace->file) return;

    /* Write header on first tick, regardless of cycle delta */
    if (!trace->header_written) {
        write_vcd_header(trace);
        return;  /* header includes #0 timestamp */
    }

    /* No pending changes and no cycle advance → nothing to do */
    bool has_changes = false;
    for (uint32_t i = 0; i < trace->signal_count; i++) {
        if (trace->signals[i].dirty) { has_changes = true; break; }
    }
    if (!has_changes && cycles == 0) return;

    /* Flush pending changes from current cycle */
    for (uint32_t i = 0; i < trace->signal_count; i++) {
        write_signal_change(trace, i);
    }

    /* Advance time */
    if (cycles > 0) {
        trace->current_cycle += cycles;
        fprintf(trace->file, "#%lu\n", (unsigned long)trace->current_cycle);
    }
}

/* ---- Global toggle for quick disable ---- */

static bool g_trace_enabled = false;

bool tu_trace_is_enabled(void) {
    return g_trace_enabled;
}
