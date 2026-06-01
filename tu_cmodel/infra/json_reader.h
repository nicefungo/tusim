/*
 * Minimal JSON Reader — Zero-Dependency, Production-Grade
 * =======================================================
 * A1: Runtime configuration loading from JSON files.
 *
 * Design:
 *   - Recursive-descent parser, ~300 lines of C
 *   - No malloc in the parser itself (user provides buffer)
 *   - Strict JSON compliance: strings, numbers (int/double), true/false/null,
 *     arrays, objects, nested structures
 *   - Single-pass, no backtracking
 *   - Error reporting with line:column
 *
 * Usage:
 *   tu_json_value_t root;
 *   tu_json_error_t err = tu_json_parse(json_string, &root, error_buf, 256);
 *   if (err != TU_JSON_OK) { fprintf(stderr, "%s\n", error_buf); }
 *   int64_t rows = tu_json_get_int(tu_json_get(root, "pe_rows"));
 *   double bw = tu_json_get_double(tu_json_get(root, "bandwidth_gbps"));
 */

#ifndef TU_JSON_READER_H
#define TU_JSON_READER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Value Types ---- */

typedef enum {
    TU_JSON_NULL,
    TU_JSON_BOOL,
    TU_JSON_INT,
    TU_JSON_DOUBLE,
    TU_JSON_STRING,
    TU_JSON_ARRAY,
    TU_JSON_OBJECT,
} tu_json_type_t;

/* ---- Error Codes ---- */

typedef enum {
    TU_JSON_OK = 0,
    TU_JSON_ERR_SYNTAX,
    TU_JSON_ERR_UNEXPECTED_EOF,
    TU_JSON_ERR_INVALID_NUMBER,
    TU_JSON_ERR_INVALID_ESCAPE,
    TU_JSON_ERR_NESTING_TOO_DEEP,
    TU_JSON_ERR_INVALID_UTF8,
    TU_JSON_ERR_MISSING_COLON,
    TU_JSON_ERR_MISSING_COMMA,
    TU_JSON_ERR_TRAILING_COMMA,
    TU_JSON_ERR_UNMATCHED_BRACE,
} tu_json_error_t;

#define TU_JSON_MAX_DEPTH 32

/* ---- Value ---- */

typedef struct tu_json_value_t tu_json_value_t;

struct tu_json_value_t {
    tu_json_type_t  type;
    union {
        bool          bool_val;
        int64_t       int_val;
        double        double_val;
        struct {
            const char *data;
            uint32_t    len;
        } string;
        struct {
            tu_json_value_t *items;
            uint32_t          count;
        } array;
        struct {
            struct tu_json_kv {
                const char      *key;
                uint32_t         key_len;
                tu_json_value_t *value;
            } *pairs;
            uint32_t          count;
        } object;
    };
};

/* ---- Public API ---- */

/*
 * Parse a null-terminated JSON string into a value tree.
 *
 * json_str:   null-terminated JSON string (modified in-place for string unescaping)
 * root:       pointer to user-allocated root value (will be populated)
 * error_buf:  buffer for error message on failure (can be NULL)
 * error_size: size of error_buf
 *
 * Returns TU_JSON_OK on success, error code on failure.
 *
 * IMPORTANT: The json_str is SHIFTED in-place for string values
 * (unescape sequences like \n, \\, \"). The root value's string.data
 * pointers point INTO the modified json_str.
 * The caller must keep json_str alive while accessing values.
 */
tu_json_error_t tu_json_parse(char *json_str, tu_json_value_t *root,
                              char *error_buf, size_t error_size);

/*
 * Get a named field from an object value.
 * Returns a pointer to the value, or NULL if not found.
 */
const tu_json_value_t *tu_json_get(const tu_json_value_t *obj, const char *key);

/*
 * Type-safe accessors.
 * These return sensible defaults (0, 0.0, false) on type mismatch.
 */
bool        tu_json_as_bool(const tu_json_value_t *v);
int64_t     tu_json_as_int(const tu_json_value_t *v);
double      tu_json_as_double(const tu_json_value_t *v);
const char *tu_json_as_string(const tu_json_value_t *v, uint32_t *len_out);
uint32_t    tu_json_array_len(const tu_json_value_t *v);
const tu_json_value_t *tu_json_array_get(const tu_json_value_t *v, uint32_t idx);

/*
 * Free all memory allocated by tu_json_parse.
 * Must be called on the root value when done.
 */
void tu_json_free(tu_json_value_t *v);

/*
 * Debug: print a value tree to stderr (for development).
 */
void tu_json_dump(const tu_json_value_t *v, int indent);

#ifdef __cplusplus
}
#endif

#endif /* TU_JSON_READER_H */
