/*
 * TinyTU ASM Interpreter
 * =======================
 * Parses and executes .tuasm programs directly against the TU cmodel.
 *
 * Usage:
 *   tu_host_buffer_t bufs[] = {
 *       {"input",  input_data,  input_size},
 *       {"output", output_data, output_size},
 *   };
 *   tu_run_asm(program_text, bufs, 2);
 *   tu_print_stats();
 */

#include "tu_cmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* ---- Weight blob management ---- */

#define TU_ASM_MAX_WEIGHTS 16

typedef struct {
    char     name[64];
    uint8_t *data;
    uint32_t size;
    uint32_t capacity;
} tu_asm_weight_t;

/* ---- Parser state ---- */

typedef struct {
    tu_asm_weight_t        weights[TU_ASM_MAX_WEIGHTS];
    int                    n_weights;
    const tu_host_buffer_t *host_bufs;
    int                    n_host_bufs;
    const char  *cursor;
    int          line_num;
} tu_asm_state_t;

/* ---- Helpers ---- */

static void tu_asm_error(tu_asm_state_t *s, const char *fmt, ...) {
    fprintf(stderr, "TU ASM error line %d: ", s->line_num);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
}

static void tu_asm_skip_whitespace(tu_asm_state_t *s) {
    while (*s->cursor == ' ' || *s->cursor == '\t' || *s->cursor == ',') s->cursor++;
}

static void tu_asm_skip_line(tu_asm_state_t *s) {
    while (*s->cursor && *s->cursor != '\n') s->cursor++;
    if (*s->cursor == '\n') s->cursor++;
    s->line_num++;
}

static int tu_asm_parse_int(tu_asm_state_t *s, uint32_t *out) {
    tu_asm_skip_whitespace(s);
    if (*s->cursor < '0' || *s->cursor > '9') return -1;
    char *end;
    *out = (uint32_t)strtoul(s->cursor, &end, 10);
    s->cursor = end;
    return 0;
}

static int tu_asm_parse_name(tu_asm_state_t *s, char *out, size_t out_sz) {
    tu_asm_skip_whitespace(s);
    const char *start = s->cursor;
    while (*s->cursor && *s->cursor != ' ' && *s->cursor != '\t'
           && *s->cursor != '\n' && *s->cursor != ',' && *s->cursor != ';')
        s->cursor++;
    size_t len = (size_t)(s->cursor - start);
    if (len == 0 || len >= out_sz) return -1;
    memcpy(out, start, len);
    out[len] = '\0';
    return 0;
}

static void *tu_asm_find_buffer(tu_asm_state_t *s, const char *name, uint32_t *size_out) {
    for (int i = 0; i < s->n_host_bufs; i++) {
        if (strcmp(s->host_bufs[i].name, name) == 0) {
            if (size_out) *size_out = s->host_bufs[i].size;
            return s->host_bufs[i].data;
        }
    }
    for (int i = 0; i < s->n_weights; i++) {
        if (strcmp(s->weights[i].name, name) == 0) {
            if (size_out) *size_out = s->weights[i].size;
            return s->weights[i].data;
        }
    }
    return NULL;
}

static int tu_asm_parse_hex_byte(tu_asm_state_t *s, uint8_t *out) {
    tu_asm_skip_whitespace(s);
    char hex[3] = {0};
    for (int i = 0; i < 2; i++) {
        char c = s->cursor[i];
        if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')))
            return -1;
        hex[i] = c;
    }
    *out = (uint8_t)strtoul(hex, NULL, 16);
    s->cursor += 2;
    return 0;
}

static void tu_asm_weight_append(tu_asm_weight_t *w, uint8_t b) {
    if (w->size >= w->capacity) {
        uint32_t new_cap = w->capacity ? w->capacity * 2 : 4096;
        w->data = (uint8_t *)realloc(w->data, new_cap);
        w->capacity = new_cap;
    }
    w->data[w->size++] = b;
}

/* ---- Instruction handlers ---- */

static int tu_asm_exec_load_w(tu_asm_state_t *s) {
    uint32_t tu_off, size_bytes;
    if (tu_asm_parse_int(s, &tu_off) < 0) { tu_asm_error(s, "expected offset"); return -1; }
    if (tu_asm_parse_int(s, &size_bytes) < 0) { tu_asm_error(s, "expected size"); return -1; }

    /* Use last declared weight section that has enough data */
    void *wdata = NULL;
    for (int i = s->n_weights - 1; i >= 0; i--) {
        if (s->weights[i].size >= size_bytes) {
            wdata = s->weights[i].data;
            break;
        }
    }
    if (!wdata) { tu_asm_error(s, "no matching weight section"); return -1; }
    tu_dma_load_w(wdata, tu_off, size_bytes);
    return 0;
}

static int tu_asm_exec_load_a(tu_asm_state_t *s) {
    char name[64]; uint32_t tu_off, size_bytes;
    if (tu_asm_parse_name(s, name, sizeof(name)) < 0) { tu_asm_error(s, "expected buffer name"); return -1; }
    if (tu_asm_parse_int(s, &tu_off) < 0) { tu_asm_error(s, "expected offset"); return -1; }
    if (tu_asm_parse_int(s, &size_bytes) < 0) { tu_asm_error(s, "expected size"); return -1; }
    void *buf = tu_asm_find_buffer(s, name, NULL);
    if (!buf) { tu_asm_error(s, "buffer '%s' not found", name); return -1; }
    tu_dma_load_a(buf, tu_off, size_bytes);
    return 0;
}

static int tu_asm_exec_load_o(tu_asm_state_t *s) {
    char name[64]; uint32_t tu_off, size_bytes;
    if (tu_asm_parse_name(s, name, sizeof(name)) < 0) { tu_asm_error(s, "expected buffer name"); return -1; }
    if (tu_asm_parse_int(s, &tu_off) < 0) { tu_asm_error(s, "expected offset"); return -1; }
    if (tu_asm_parse_int(s, &size_bytes) < 0) { tu_asm_error(s, "expected size"); return -1; }
    void *buf = tu_asm_find_buffer(s, name, NULL);
    if (!buf) { tu_asm_error(s, "buffer '%s' not found", name); return -1; }
    tu_dma_load_o(buf, tu_off, size_bytes);
    return 0;
}

static int tu_asm_exec_mma(tu_asm_state_t *s) {
    uint32_t M, N, K, w_off, a_off, o_off;
    char flag_str[16];
    if (tu_asm_parse_int(s, &M) < 0) { tu_asm_error(s, "expected M"); return -1; }
    if (tu_asm_parse_int(s, &N) < 0) { tu_asm_error(s, "expected N"); return -1; }
    if (tu_asm_parse_int(s, &K) < 0) { tu_asm_error(s, "expected K"); return -1; }
    if (tu_asm_parse_int(s, &w_off) < 0) { tu_asm_error(s, "expected w_offset"); return -1; }
    if (tu_asm_parse_int(s, &a_off) < 0) { tu_asm_error(s, "expected a_offset"); return -1; }
    if (tu_asm_parse_int(s, &o_off) < 0) { tu_asm_error(s, "expected o_offset"); return -1; }
    if (tu_asm_parse_name(s, flag_str, sizeof(flag_str)) < 0) { tu_asm_error(s, "expected BIAS/NOBIAS"); return -1; }

    bool has_bias = (strcmp(flag_str, "BIAS") == 0);
    if (!has_bias && strcmp(flag_str, "NOBIAS") != 0) {
        tu_asm_error(s, "expected BIAS or NOBIAS, got '%s'", flag_str);
        return -1;
    }
    tu_mma((uint16_t)M, (uint16_t)N, (uint16_t)K, w_off, a_off, o_off, has_bias);
    return 0;
}

static int tu_asm_exec_store_o(tu_asm_state_t *s) {
    char name[64]; uint32_t tu_off, size_bytes;
    if (tu_asm_parse_name(s, name, sizeof(name)) < 0) { tu_asm_error(s, "expected buffer name"); return -1; }
    if (tu_asm_parse_int(s, &tu_off) < 0) { tu_asm_error(s, "expected offset"); return -1; }
    if (tu_asm_parse_int(s, &size_bytes) < 0) { tu_asm_error(s, "expected size"); return -1; }
    void *buf = tu_asm_find_buffer(s, name, NULL);
    if (!buf) { tu_asm_error(s, "buffer '%s' not found", name); return -1; }
    tu_dma_store_o(buf, tu_off, size_bytes);
    return 0;
}

/* ---- Public API ---- */

int tu_run_asm(const char *program, const tu_host_buffer_t *buffers, int n_buffers) {
    tu_asm_state_t s = {0};
    s.cursor = program;
    s.line_num = 1;
    s.host_bufs = buffers;
    s.n_host_bufs = n_buffers;

    tu_init();

    while (*s.cursor) {
        tu_asm_skip_whitespace(&s);
        if (*s.cursor == '\n') { s.cursor++; s.line_num++; continue; }
        if (*s.cursor == '\0') break;
        if (*s.cursor == ';') { tu_asm_skip_line(&s); continue; }

        /* %weight section */
        if (strncmp(s.cursor, "%weight", 7) == 0) {
            s.cursor += 7;
            tu_asm_weight_t *w = &s.weights[s.n_weights];
            memset(w, 0, sizeof(*w));
            if (tu_asm_parse_name(&s, w->name, sizeof(w->name)) < 0) {
                tu_asm_error(&s, "%%weight: expected name"); goto fail;
            }
            tu_asm_skip_line(&s);

            while (*s.cursor) {
                tu_asm_skip_whitespace(&s);
                if (*s.cursor == '\n') { s.cursor++; s.line_num++; continue; }
                if (*s.cursor == '\0') break;
                if (strncmp(s.cursor, "%endweight", 10) == 0) {
                    s.cursor += 10; tu_asm_skip_line(&s); break;
                }
                if (*s.cursor == ';') { tu_asm_skip_line(&s); continue; }
                uint8_t b;
                if (tu_asm_parse_hex_byte(&s, &b) == 0) {
                    tu_asm_weight_append(w, b);
                } else {
                    s.cursor++; /* skip non-hex */
                }
            }
            if (s.n_weights < TU_ASM_MAX_WEIGHTS) s.n_weights++;
            continue;
        }

        /* %input / %output — skip declarations */
        if (strncmp(s.cursor, "%input", 6) == 0 || strncmp(s.cursor, "%output", 7) == 0) {
            tu_asm_skip_line(&s); continue;
        }

        /* Instructions */
        int rc = 0;
        if      (strncmp(s.cursor, "LOAD_W",  6) == 0) { s.cursor += 6; rc = tu_asm_exec_load_w(&s); }
        else if (strncmp(s.cursor, "LOAD_A",  6) == 0) { s.cursor += 6; rc = tu_asm_exec_load_a(&s); }
        else if (strncmp(s.cursor, "LOAD_O",  6) == 0) { s.cursor += 6; rc = tu_asm_exec_load_o(&s); }
        else if (strncmp(s.cursor, "STORE_O", 7) == 0) { s.cursor += 7; rc = tu_asm_exec_store_o(&s); }
        else if (strncmp(s.cursor, "MMA",     3) == 0) { s.cursor += 3; rc = tu_asm_exec_mma(&s); }
        else if (strncmp(s.cursor, "SYNC",    4) == 0) { s.cursor += 4; tu_sync(); rc = 0; }
        else {
            tu_asm_error(&s, "unknown instruction: %.20s", s.cursor);
            rc = -1;
        }
        if (rc < 0) goto fail;
        tu_asm_skip_line(&s);
    }

    for (int i = 0; i < s.n_weights; i++) free(s.weights[i].data);
    return 0;

fail:
    for (int i = 0; i < s.n_weights; i++) free(s.weights[i].data);
    return -1;
}
