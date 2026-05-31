/*
 * TU Status Codes & Exception Handling — Implementation (Gap E5)
 * ===============================================================
 */

#include "tu_status.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Error info strings
 * ================================================================ */

static const char *tu_error_strings[TU_ERR_COUNT] = {
    [TU_OK]                       = "success",
    [TU_ERR_NOT_INITIALIZED]      = "not initialized",
    [TU_ERR_ALREADY_INITIALIZED]  = "already initialized",
    [TU_ERR_CONFIG_INVALID]       = "invalid config",
    [TU_ERR_INVALID_PARAM]        = "invalid parameter",
    [TU_ERR_OUT_OF_RANGE]         = "out of range",
    [TU_ERR_NULL_POINTER]         = "null pointer",
    [TU_ERR_OUT_OF_MEMORY]        = "out of memory",
    [TU_ERR_SRAM_OVERFLOW]        = "SRAM overflow",
    [TU_ERR_SRAM_UNDERFLOW]       = "SRAM underflow",
    [TU_ERR_BANK_CONFLICT]        = "bank conflict",
    [TU_ERR_DMA_OVERFLOW]         = "DMA overflow",
    [TU_ERR_DMA_INVALID_CHANNEL]  = "invalid DMA channel",
    [TU_ERR_DMA_INVALID_DESC]     = "invalid DMA descriptor",
    [TU_ERR_DMA_QUEUE_FULL]       = "DMA queue full",
    [TU_ERR_DMA_TIMEOUT]          = "DMA timeout",
    [TU_ERR_QUEUE_FULL]           = "queue full",
    [TU_ERR_QUEUE_EMPTY]          = "queue empty",
    [TU_ERR_CMD_NOT_FOUND]        = "command not found",
    [TU_ERR_CMD_TIMEOUT]          = "command timeout",
    [TU_ERR_DEPENDENCY_CYCLE]     = "dependency cycle",
    [TU_ERR_COMPUTE_INVALID_OP]   = "invalid compute op",
    [TU_ERR_COMPUTE_DIM_MISMATCH] = "dimension mismatch",
    [TU_ERR_COMPUTE_OVERFLOW]     = "compute overflow",
    [TU_ERR_COMPUTE_UNDERFLOW]    = "compute underflow",
    [TU_ERR_DTYPE_UNSUPPORTED]    = "unsupported dtype",
    [TU_ERR_DTYPE_CONVERSION]     = "dtype conversion failed",
    [TU_ERR_NAN_ENCOUNTERED]      = "NaN encountered",
    [TU_ERR_INF_ENCOUNTERED]      = "Inf encountered",
    [TU_ERR_ICC_NO_ROUTE]         = "no ICC route",
    [TU_ERR_ICC_BUFFER_FULL]      = "ICC buffer full",
    [TU_ERR_ICC_TIMEOUT]          = "ICC timeout",
    [TU_ERR_INTERNAL]             = "internal error",
    [TU_ERR_NOT_IMPLEMENTED]      = "not implemented",
    [TU_ERR_ASSERTION_FAILED]     = "assertion failed",
    [TU_ERR_VERIFY_MISMATCH]      = "verification mismatch",
    [TU_ERR_VERIFY_TOLERANCE]     = "tolerance exceeded",
};

/* ================================================================
 * Global state
 * ================================================================ */

static tu_error_mode_t g_error_mode = TU_ERR_MODE_LOG;
static tu_error_t g_last_error = {0};

/* ================================================================
 * Error injection support
 * ================================================================ */

#define MAX_INJECT_SITES 16

typedef struct {
    const char *file;
    int         line;
    tu_status_t code;
    bool        active;
} inject_site_t;

static inject_site_t g_inject_sites[MAX_INJECT_SITES];
static int g_inject_count = 0;

/* ================================================================
 * API Implementation
 * ================================================================ */

const tu_error_t *tu_get_last_error(void) {
    if (g_last_error.code == TU_OK) return NULL;
    return &g_last_error;
}

void tu_clear_error(void) {
    memset(&g_last_error, 0, sizeof(g_last_error));
}

void tu_set_error_mode(tu_error_mode_t mode) {
    g_error_mode = mode;
}

tu_error_mode_t tu_get_error_mode(void) {
    return g_error_mode;
}

const char *tu_status_str(tu_status_t code) {
    if (code < TU_ERR_COUNT && tu_error_strings[code]) {
        return tu_error_strings[code];
    }
    return "unknown error";
}

tu_status_t tu_report_error(tu_status_t code,
                             const char *file, int line, const char *func,
                             const char *msg) {
    /* Record the error */
    g_last_error.code      = code;
    g_last_error.file      = file;
    g_last_error.line      = line;
    g_last_error.function  = func;
    g_last_error.message   = msg;
    g_last_error.timestamp = 0;  /* Would use cycle counter in production */

    /* Act based on error mode */
    switch (g_error_mode) {
    case TU_ERR_MODE_LOG:
        fprintf(stderr, "[TU ERROR] %s:%d in %s(): %s (code=%d: %s)\n",
                file, line, func,
                msg ? msg : "(no message)",
                (int)code, tu_status_str(code));
        break;

    case TU_ERR_MODE_ABORT:
        fprintf(stderr, "[TU FATAL] %s:%d in %s(): %s (code=%d: %s)\n",
                file, line, func,
                msg ? msg : "(no message)",
                (int)code, tu_status_str(code));
        abort();
        break;

    case TU_ERR_MODE_SILENT:
        /* No output */
        break;
    }

    return code;
}

/* ================================================================
 * Error Injection
 * ================================================================ */

void tu_error_inject_enable(const char *file, int line, tu_status_t code) {
    if (g_inject_count >= MAX_INJECT_SITES) return;

    g_inject_sites[g_inject_count].file   = file;
    g_inject_sites[g_inject_count].line   = line;
    g_inject_sites[g_inject_count].code   = code;
    g_inject_sites[g_inject_count].active = true;
    g_inject_count++;
}

void tu_error_inject_disable_all(void) {
    g_inject_count = 0;
    memset(g_inject_sites, 0, sizeof(g_inject_sites));
}

tu_status_t tu_error_inject_check(const char *file, int line) {
    for (int i = 0; i < g_inject_count; i++) {
        if (g_inject_sites[i].active &&
            strcmp(g_inject_sites[i].file, file) == 0 &&
            g_inject_sites[i].line == line) {
            /* One-shot: deactivate after injection */
            g_inject_sites[i].active = false;
            return g_inject_sites[i].code;
        }
    }
    return TU_OK;
}
