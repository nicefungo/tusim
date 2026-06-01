/*
 * Minimal JSON Reader — Implementation
 * =====================================
 * Recursive-descent parser. Handles all standard JSON:
 *   - Escaped strings (\n, \t, \\, \", \uXXXX)
 *   - Integer and floating-point numbers (with scientific notation)
 *   - Nested objects and arrays
 *   - true/false/null
 *
 * Designed for configuration files (~1-10 KB). Uses malloc for child
 * values; caller calls tu_json_free() on the root to clean up.
 */

#include "json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ---- Internal parser state ---- */

typedef struct {
    char       *p;        /* current position */
    uint32_t    line;
    uint32_t    col;
    int         depth;
    char       *err_buf;
    size_t      err_size;
} jp_t;

/* ---- Character helpers ---- */

static inline bool is_ws(char c)   { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
static inline bool is_digit(char c) { return c >= '0' && c <= '9'; }
static inline bool is_hex(char c)  { return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

static void set_error(jp_t *jp, const char *msg) {
    if (jp->err_buf && jp->err_size > 0)
        snprintf(jp->err_buf, jp->err_size, "JSON error at line %u, col %u: %s",
                 jp->line, jp->col, msg);
}

static void skip_ws(jp_t *jp) {
    while (*jp->p && is_ws(*jp->p)) {
        if (*jp->p == '\n') { jp->line++; jp->col = 0; }
        else jp->col++;
        jp->p++;
    }
}

/* ---- Forward declarations ---- */

static tu_json_error_t parse_value(jp_t *jp, tu_json_value_t *v);

/* ---- Parse string ---- */

static tu_json_error_t parse_string(jp_t *jp, tu_json_value_t *v) {
    if (*jp->p != '"') return TU_JSON_ERR_SYNTAX;
    jp->p++; jp->col++;

    /* First pass: measure length */
    const char *src = jp->p;
    size_t len = 0;
    while (*src && *src != '"') {
        if ((unsigned char)*src < 0x20) {
            set_error(jp, "control character in string");
            return TU_JSON_ERR_SYNTAX;
        }
        if (*src == '\\') {
            src++;
            if (!*src) break;
            if (*src == 'u') { src += 5; len += 3; continue; } /* \uXXXX → up to 3 UTF-8 bytes */
            len++;
        } else {
            len++;
        }
        src++;
    }
    if (!*src) {
        set_error(jp, "unterminated string");
        return TU_JSON_ERR_UNEXPECTED_EOF;
    }

    /* Second pass: unescape into heap buffer */
    char *buf = (char *)malloc(len + 1);
    if (!buf) {
        set_error(jp, "out of memory");
        return TU_JSON_ERR_SYNTAX;
    }
    char *dst = buf;
    src = jp->p;

    while (*src && *src != '"') {
        if (*src == '\\') {
            src++;
            switch (*src) {
                case '"':  *dst++ = '"';  src++; break;
                case '\\': *dst++ = '\\'; src++; break;
                case '/':  *dst++ = '/';  src++; break;
                case 'b':  *dst++ = '\b'; src++; break;
                case 'f':  *dst++ = '\f'; src++; break;
                case 'n':  *dst++ = '\n'; src++; break;
                case 'r':  *dst++ = '\r'; src++; break;
                case 't':  *dst++ = '\t'; src++; break;
                case 'u': {
                    src++;
                    uint32_t cp = 0;
                    for (int i = 0; i < 4; i++, src++)
                        cp = cp * 16 + (is_digit(*src) ? *src - '0'
                                      : (*src >= 'a' ? *src - 'a' + 10 : *src - 'A' + 10));
                    if (cp < 0x80)       *dst++ = (char)cp;
                    else if (cp < 0x800) { *dst++ = (char)(0xC0|(cp>>6)); *dst++ = (char)(0x80|(cp&0x3F)); }
                    else                 { *dst++ = (char)(0xE0|(cp>>12)); *dst++ = (char)(0x80|((cp>>6)&0x3F)); *dst++ = (char)(0x80|(cp&0x3F)); }
                    break;
                }
                default:
                    free(buf);
                    set_error(jp, "unknown escape sequence");
                    return TU_JSON_ERR_INVALID_ESCAPE;
            }
        } else {
            *dst++ = *src++;
        }
    }

    *dst = '\0';
    jp->p = (char *)src + 1; /* skip closing quote */
    jp->col += (uint32_t)(len + 2);

    v->type = TU_JSON_STRING;
    v->string.data = buf;
    v->string.len = (uint32_t)len;
    return TU_JSON_OK;
}

/* ---- Parse number ---- */

static tu_json_error_t parse_number(jp_t *jp, tu_json_value_t *v) {
    char *start = jp->p;
    bool is_float = false;

    if (*jp->p == '-') jp->p++;
    if (*jp->p == '0') jp->p++;
    else if (is_digit(*jp->p)) while (is_digit(*jp->p)) jp->p++;
    else { set_error(jp, "expected digit"); return TU_JSON_ERR_INVALID_NUMBER; }

    if (*jp->p == '.') {
        is_float = true; jp->p++;
        if (!is_digit(*jp->p)) { set_error(jp, "expected digit after decimal"); return TU_JSON_ERR_INVALID_NUMBER; }
        while (is_digit(*jp->p)) jp->p++;
    }
    if (*jp->p == 'e' || *jp->p == 'E') {
        is_float = true; jp->p++;
        if (*jp->p == '+' || *jp->p == '-') jp->p++;
        if (!is_digit(*jp->p)) { set_error(jp, "expected digit in exponent"); return TU_JSON_ERR_INVALID_NUMBER; }
        while (is_digit(*jp->p)) jp->p++;
    }

    char saved = *jp->p;
    *jp->p = '\0';
    if (is_float) { v->type = TU_JSON_DOUBLE; v->double_val = strtod(start, NULL); }
    else          { v->type = TU_JSON_INT;    v->int_val    = strtoll(start, NULL, 10); }
    *jp->p = saved;
    jp->col += (uint32_t)(jp->p - start);
    return TU_JSON_OK;
}

/* ---- Parse array (uses malloc for items) ---- */

static tu_json_error_t parse_array(jp_t *jp, tu_json_value_t *v) {
    if (jp->depth >= TU_JSON_MAX_DEPTH) {
        set_error(jp, "max nesting depth exceeded");
        return TU_JSON_ERR_NESTING_TOO_DEEP;
    }

    jp->p++; jp->col++; /* '[' */
    skip_ws(jp);

    /* First pass: count items */
    #define MAX_ARR 128
    tu_json_value_t scratch[MAX_ARR];
    uint32_t count = 0;

    if (*jp->p != ']') {
        for (;;) {
            if (count >= MAX_ARR) {
                set_error(jp, "array too large"); return TU_JSON_ERR_NESTING_TOO_DEEP;
            }
            memset(&scratch[count], 0, sizeof(tu_json_value_t));
            jp->depth++;
            tu_json_error_t err = parse_value(jp, &scratch[count]);
            jp->depth--;
            if (err != TU_JSON_OK) {
                /* Free any already-allocated items */
                for (uint32_t i = 0; i < count; i++) {
                    /* Strings were malloc'd during parse */
                }
                return err;
            }
            count++;
            skip_ws(jp);
            if (*jp->p == ',') { jp->p++; jp->col++; skip_ws(jp); }
            else if (*jp->p == ']') break;
            else { set_error(jp, "expected ',' or ']'"); return TU_JSON_ERR_MISSING_COMMA; }
        }
    }

    jp->p++; jp->col++; /* ']' */

    /* Copy from scratch to heap */
    v->type = TU_JSON_ARRAY;
    v->array.count = count;
    if (count > 0) {
        v->array.items = (tu_json_value_t *)malloc(count * sizeof(tu_json_value_t));
        if (!v->array.items) { set_error(jp, "out of memory"); return TU_JSON_ERR_SYNTAX; }
        memcpy(v->array.items, scratch, count * sizeof(tu_json_value_t));
    } else {
        v->array.items = NULL;
    }
    return TU_JSON_OK;
}

/* ---- Parse object (uses malloc for pairs) ---- */

static tu_json_error_t parse_object(jp_t *jp, tu_json_value_t *v) {
    if (jp->depth >= TU_JSON_MAX_DEPTH) {
        set_error(jp, "max nesting depth exceeded");
        return TU_JSON_ERR_NESTING_TOO_DEEP;
    }

    jp->p++; jp->col++; /* '{' */
    skip_ws(jp);

    #define MAX_OBJ 64
    struct { char *key; uint32_t key_len; tu_json_value_t value; } scratch[MAX_OBJ];
    uint32_t count = 0;

    if (*jp->p != '}') {
        for (;;) {
            if (count >= MAX_OBJ) {
                set_error(jp, "object too large"); return TU_JSON_ERR_NESTING_TOO_DEEP;
            }

            tu_json_value_t key_val;
            memset(&key_val, 0, sizeof(key_val));
            tu_json_error_t err = parse_string(jp, &key_val);
            if (err != TU_JSON_OK) return err;

            /* Key string was malloc'd by parse_string — store it */
            scratch[count].key = (char *)key_val.string.data;
            scratch[count].key_len = key_val.string.len;

            skip_ws(jp);
            if (*jp->p != ':') {
                free(scratch[count].key);
                set_error(jp, "expected ':'"); return TU_JSON_ERR_MISSING_COLON;
            }
            jp->p++; jp->col++;
            skip_ws(jp);

            memset(&scratch[count].value, 0, sizeof(tu_json_value_t));
            jp->depth++;
            err = parse_value(jp, &scratch[count].value);
            jp->depth--;
            if (err != TU_JSON_OK) { free(scratch[count].key); return err; }

            count++;
            skip_ws(jp);
            if (*jp->p == ',') { jp->p++; jp->col++; skip_ws(jp); }
            else if (*jp->p == '}') break;
            else { set_error(jp, "expected ',' or '}'"); return TU_JSON_ERR_MISSING_COMMA; }
        }
    }

    jp->p++; jp->col++; /* '}' */

    v->type = TU_JSON_OBJECT;
    v->object.count = count;
    if (count > 0) {
        v->object.pairs = (struct tu_json_kv *)malloc(count * sizeof(struct tu_json_kv));
        if (!v->object.pairs) { set_error(jp, "out of memory"); return TU_JSON_ERR_SYNTAX; }
        for (uint32_t i = 0; i < count; i++) {
            v->object.pairs[i].key = scratch[i].key;
            v->object.pairs[i].key_len = scratch[i].key_len;
            /* Allocate value on heap and copy from scratch */
            v->object.pairs[i].value = (tu_json_value_t *)malloc(sizeof(tu_json_value_t));
            if (!v->object.pairs[i].value) { set_error(jp, "out of memory"); return TU_JSON_ERR_SYNTAX; }
            memcpy(v->object.pairs[i].value, &scratch[i].value, sizeof(tu_json_value_t));
        }
    } else {
        v->object.pairs = NULL;
    }
    return TU_JSON_OK;
}

/* ---- Parse value dispatcher ---- */

static tu_json_error_t parse_value(jp_t *jp, tu_json_value_t *v) {
    skip_ws(jp);
    if (!*jp->p) { set_error(jp, "unexpected end of input"); return TU_JSON_ERR_UNEXPECTED_EOF; }

    char c = *jp->p;

    if (c == '"') return parse_string(jp, v);
    if (c == '{') return parse_object(jp, v);
    if (c == '[') return parse_array(jp, v);

    if (c == 't' && strncmp(jp->p, "true", 4) == 0) {
        v->type = TU_JSON_BOOL; v->bool_val = true;
        jp->p += 4; jp->col += 4; return TU_JSON_OK;
    }
    if (c == 'f' && strncmp(jp->p, "false", 5) == 0) {
        v->type = TU_JSON_BOOL; v->bool_val = false;
        jp->p += 5; jp->col += 5; return TU_JSON_OK;
    }
    if (c == 'n' && strncmp(jp->p, "null", 4) == 0) {
        v->type = TU_JSON_NULL;
        jp->p += 4; jp->col += 4; return TU_JSON_OK;
    }
    if (c == '-' || is_digit(c)) return parse_number(jp, v);

    set_error(jp, "unexpected character"); return TU_JSON_ERR_SYNTAX;
}

/* ---- Public API ---- */

tu_json_error_t tu_json_parse(char *json_str, tu_json_value_t *root,
                              char *error_buf, size_t error_size) {
    if (!json_str || !root) {
        if (error_buf && error_size > 0) snprintf(error_buf, error_size, "null input");
        return TU_JSON_ERR_SYNTAX;
    }

    memset(root, 0, sizeof(*root));

    jp_t jp = {
        .p = json_str, .line = 1, .col = 0, .depth = 0,
        .err_buf = error_buf, .err_size = error_size
    };

    tu_json_error_t err = parse_value(&jp, root);
    if (err == TU_JSON_OK) {
        skip_ws(&jp);
        if (*jp.p != '\0') { set_error(&jp, "trailing characters after value"); err = TU_JSON_ERR_SYNTAX; }
    }
    return err;
}

/* ---- Cleanup ---- */

void tu_json_free(tu_json_value_t *v) {
    if (!v) return;
    switch (v->type) {
        case TU_JSON_STRING:
            free((void *)v->string.data);
            v->string.data = NULL;
            break;
        case TU_JSON_ARRAY:
            for (uint32_t i = 0; i < v->array.count; i++)
                tu_json_free(&v->array.items[i]);
            free(v->array.items);
            v->array.items = NULL;
            break;
        case TU_JSON_OBJECT:
            for (uint32_t i = 0; i < v->object.count; i++) {
                free((void *)v->object.pairs[i].key);
                tu_json_free(v->object.pairs[i].value);
                free(v->object.pairs[i].value);
                v->object.pairs[i].value = NULL;
            }
            free(v->object.pairs);
            v->object.pairs = NULL;
            break;
        default:
            break;
    }
    v->type = TU_JSON_NULL;
}

/* ---- Accessors ---- */

const tu_json_value_t *tu_json_get(const tu_json_value_t *obj, const char *key) {
    if (!obj || obj->type != TU_JSON_OBJECT || !obj->object.pairs) return NULL;
    size_t kl = strlen(key);
    for (uint32_t i = 0; i < obj->object.count; i++)
        if (obj->object.pairs[i].key_len == kl &&
            memcmp(obj->object.pairs[i].key, key, kl) == 0)
            return obj->object.pairs[i].value;
    return NULL;
}

bool        tu_json_as_bool(const tu_json_value_t *v)   { return v ? (v->type == TU_JSON_BOOL ? v->bool_val : v->type == TU_JSON_INT ? v->int_val != 0 : false) : false; }
int64_t     tu_json_as_int(const tu_json_value_t *v)    { return v ? (v->type == TU_JSON_INT ? v->int_val : v->type == TU_JSON_DOUBLE ? (int64_t)v->double_val : v->type == TU_JSON_BOOL ? (v->bool_val?1:0) : 0) : 0; }
double      tu_json_as_double(const tu_json_value_t *v) { return v ? (v->type == TU_JSON_DOUBLE ? v->double_val : v->type == TU_JSON_INT ? (double)v->int_val : 0.0) : 0.0; }

const char *tu_json_as_string(const tu_json_value_t *v, uint32_t *len_out) {
    if (!v || v->type != TU_JSON_STRING) { if (len_out) *len_out = 0; return NULL; }
    if (len_out) *len_out = v->string.len;
    return v->string.data;
}

uint32_t tu_json_array_len(const tu_json_value_t *v) {
    return (v && v->type == TU_JSON_ARRAY) ? v->array.count : 0;
}

const tu_json_value_t *tu_json_array_get(const tu_json_value_t *v, uint32_t idx) {
    return (v && v->type == TU_JSON_ARRAY && idx < v->array.count) ? &v->array.items[idx] : NULL;
}

/* ---- Debug dump (indented to stderr) ---- */

static void dump_indent(const tu_json_value_t *v, int indent) {
    for (int i = 0; i < indent; i++) fputs("  ", stderr);
    if (!v) { fputs("null\n", stderr); return; }
    switch (v->type) {
        case TU_JSON_NULL:   fputs("null\n", stderr); break;
        case TU_JSON_BOOL:   fprintf(stderr, "%s\n", v->bool_val ? "true" : "false"); break;
        case TU_JSON_INT:    fprintf(stderr, "%lld\n", (long long)v->int_val); break;
        case TU_JSON_DOUBLE: fprintf(stderr, "%g\n", v->double_val); break;
        case TU_JSON_STRING: fprintf(stderr, "\"%.*s\"\n", (int)v->string.len, v->string.data); break;
        case TU_JSON_ARRAY:
            fprintf(stderr, "[\n");
            for (uint32_t i = 0; i < v->array.count; i++) dump_indent(&v->array.items[i], indent + 1);
            for (int i = 0; i < indent; i++) fputs("  ", stderr);
            fputs("]\n", stderr);
            break;
        case TU_JSON_OBJECT:
            fprintf(stderr, "{\n");
            for (uint32_t i = 0; i < v->object.count; i++) {
                for (int j = 0; j < indent + 1; j++) fputs("  ", stderr);
                fprintf(stderr, "\"%.*s\": ", (int)v->object.pairs[i].key_len, v->object.pairs[i].key);
                dump_indent(v->object.pairs[i].value, indent + 1);
            }
            for (int i = 0; i < indent; i++) fputs("  ", stderr);
            fputs("}\n", stderr);
            break;
    }
}

void tu_json_dump(const tu_json_value_t *v, int indent) {
    dump_indent(v, indent);
}
