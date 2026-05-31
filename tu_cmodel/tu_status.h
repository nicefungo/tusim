/*
 * TU Status Codes & Exception Handling Framework (Gap E5)
 * ========================================================
 *
 * Replaces scattered abort() calls with structured error codes
 * and configurable error handling behavior.
 *
 * Gap: E5 — Exception handling (P2)
 * Also addresses: graceful error propagation across all subsystems
 *
 * Design:
 *   - All TU functions that can fail return tu_status_t instead of void
 *   - Error codes are enumerated and documented
 *   - Configurable error behavior: LOG (default), ABORT (strict), SILENT (ignore)
 *   - Error context (file, line, function) captured automatically
 *   - Error injection support for testing
 */

#ifndef TU_STATUS_H
#define TU_STATUS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Status Codes
 * ================================================================ */

typedef enum {
    TU_OK                       =  0,  /* Success */

    /* Initialization errors (1-9) */
    TU_ERR_NOT_INITIALIZED      =  1,  /* Component not initialized */
    TU_ERR_ALREADY_INITIALIZED  =  2,  /* Duplicate initialization */
    TU_ERR_CONFIG_INVALID       =  3,  /* Invalid configuration parameter */

    /* Parameter errors (10-19) */
    TU_ERR_INVALID_PARAM        = 10,  /* Invalid parameter value */
    TU_ERR_OUT_OF_RANGE         = 11,  /* Parameter out of valid range */
    TU_ERR_NULL_POINTER         = 12,  /* Unexpected NULL pointer */

    /* Memory errors (20-29) */
    TU_ERR_OUT_OF_MEMORY        = 20,  /* Allocation failed */
    TU_ERR_SRAM_OVERFLOW        = 21,  /* SRAM access beyond capacity */
    TU_ERR_SRAM_UNDERFLOW       = 22,  /* SRAM access below base */
    TU_ERR_BANK_CONFLICT        = 23,  /* Unresolvable bank conflict */

    /* DMA errors (30-39) */
    TU_ERR_DMA_OVERFLOW         = 30,  /* DMA transfer exceeds buffer */
    TU_ERR_DMA_INVALID_CHANNEL  = 31,  /* Invalid DMA channel */
    TU_ERR_DMA_INVALID_DESC     = 32,  /* Malformed DMA descriptor */
    TU_ERR_DMA_QUEUE_FULL       = 33,  /* DMA descriptor queue full */
    TU_ERR_DMA_TIMEOUT          = 34,  /* DMA transfer timed out */

    /* Command queue errors (40-49) */
    TU_ERR_QUEUE_FULL           = 40,  /* Command queue full */
    TU_ERR_QUEUE_EMPTY          = 41,  /* Command queue empty */
    TU_ERR_CMD_NOT_FOUND        = 42,  /* Command ID not found */
    TU_ERR_CMD_TIMEOUT          = 43,  /* Command timed out */
    TU_ERR_DEPENDENCY_CYCLE     = 44,  /* Circular dependency detected */

    /* Compute errors (50-59) */
    TU_ERR_COMPUTE_INVALID_OP   = 50,  /* Unrecognized operation */
    TU_ERR_COMPUTE_DIM_MISMATCH = 51,  /* Tensor dimension mismatch */
    TU_ERR_COMPUTE_OVERFLOW     = 52,  /* Numeric overflow */
    TU_ERR_COMPUTE_UNDERFLOW    = 53,  /* Numeric underflow */

    /* Data type errors (60-69) */
    TU_ERR_DTYPE_UNSUPPORTED    = 60,  /* Unsupported data type */
    TU_ERR_DTYPE_CONVERSION     = 61,  /* Data type conversion failed */
    TU_ERR_NAN_ENCOUNTERED      = 62,  /* NaN value encountered */
    TU_ERR_INF_ENCOUNTERED      = 63,  /* Inf value encountered */

    /* Interconnect errors (70-79) */
    TU_ERR_ICC_NO_ROUTE         = 70,  /* No route between cores */
    TU_ERR_ICC_BUFFER_FULL      = 71,  /* ICC receive buffer full */
    TU_ERR_ICC_TIMEOUT          = 72,  /* ICC transfer timed out */

    /* Internal errors (80-89) */
    TU_ERR_INTERNAL             = 80,  /* Internal logic error */
    TU_ERR_NOT_IMPLEMENTED      = 81,  /* Feature not implemented */
    TU_ERR_ASSERTION_FAILED     = 82,  /* Internal assertion failed */

    /* Verification errors (90-99) */
    TU_ERR_VERIFY_MISMATCH      = 90,  /* Golden reference mismatch */
    TU_ERR_VERIFY_TOLERANCE     = 91,  /* Exceeded error tolerance */

    TU_ERR_COUNT                       /* Sentinel: total number of error codes */
} tu_status_t;

/* ================================================================
 * Error Behavior Modes
 * ================================================================ */

typedef enum {
    TU_ERR_MODE_LOG     = 0,  /* Log error and return status (default) */
    TU_ERR_MODE_ABORT   = 1,  /* Log error and call abort() (strict) */
    TU_ERR_MODE_SILENT  = 2,  /* Return status silently (no logging) */
} tu_error_mode_t;

/* ================================================================
 * Error Context
 * ================================================================ */

typedef struct {
    tu_status_t     code;           /* Error code */
    const char     *file;           /* Source file */
    int             line;           /* Source line */
    const char     *function;       /* Function name */
    const char     *message;        /* Human-readable message */
    uint64_t        timestamp;      /* Cycle counter at error time */
} tu_error_t;

/* ================================================================
 * Error Handler API
 * ================================================================ */

/*
 * Get the last error that occurred.
 * Thread-safe: uses thread-local storage.
 */
const tu_error_t *tu_get_last_error(void);

/*
 * Clear the last error.
 */
void tu_clear_error(void);

/*
 * Set the global error behavior mode.
 * Default: TU_ERR_MODE_LOG
 */
void tu_set_error_mode(tu_error_mode_t mode);

/*
 * Get the current error behavior mode.
 */
tu_error_mode_t tu_get_error_mode(void);

/*
 * Get a human-readable string for a status code.
 */
const char *tu_status_str(tu_status_t code);

/*
 * Check if a status code indicates success.
 */
static inline bool tu_is_ok(tu_status_t s) { return s == TU_OK; }

/*
 * Check if a status code indicates an error.
 */
static inline bool tu_is_err(tu_status_t s) { return s != TU_OK; }

/* ================================================================
 * Error Reporting Macros
 * ================================================================ */

/*
 * Report an error with the given code and message.
 * Returns the error code for chaining.
 *
 * Example:
 *   if (ptr == NULL) return TU_REPORT_ERR(TU_ERR_NULL_POINTER, "expected non-null");
 */
tu_status_t tu_report_error(tu_status_t code,
                             const char *file, int line, const char *func,
                             const char *msg);

#define TU_REPORT_ERR(code, msg) \
    tu_report_error(code, __FILE__, __LINE__, __func__, msg)

/*
 * Assert a condition. If false, report TU_ERR_ASSERTION_FAILED.
 * Returns the assertion code on failure, TU_OK on success.
 *
 * Example:
 *   TU_ASSERT(size > 0, "size must be positive");
 */
#define TU_ASSERT(cond, msg) \
    do { if (!(cond)) return TU_REPORT_ERR(TU_ERR_ASSERTION_FAILED, msg); } while(0)

/*
 * Check a condition and return an error if it fails.
 *
 * Example:
 *   TU_CHECK(ptr != NULL, TU_ERR_NULL_POINTER, "pointer is null");
 */
#define TU_CHECK(cond, err, msg) \
    do { if (!(cond)) return TU_REPORT_ERR(err, msg); } while(0)

/*
 * Forward an error from a called function.
 *
 * Example:
 *   tu_status_t s = some_function();
 *   TU_RETURN_IF_ERR(s);
 */
#define TU_RETURN_IF_ERR(s) \
    do { tu_status_t _s = (s); if (_s != TU_OK) return _s; } while(0)

/* ================================================================
 * Error Injection (for testing)
 * ================================================================ */

/*
 * Enable error injection at a specific site.
 * When enabled, the next call to tu_inject_error_at(site) at the
 * given site will return the injected status code.
 *
 * Use sparingly — primarily for testing error recovery paths.
 */
void tu_error_inject_enable(const char *file, int line, tu_status_t code);

/* Disable all error injections. */
void tu_error_inject_disable_all(void);

/* Check if an error should be injected at this call site. */
tu_status_t tu_error_inject_check(const char *file, int line);

#define TU_ERROR_INJECT() \
    do { \
        tu_status_t _inj = tu_error_inject_check(__FILE__, __LINE__); \
        if (_inj != TU_OK) return _inj; \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* TU_STATUS_H */
