/*
 * TU CModel — DPI-C Integration Wrapper (Gap I1)
 * =================================================
 *
 * SystemVerilog DPI-C compatible wrapper for the TU cmodel.
 * Enables RTL testbenches to use the cmodel as a golden reference
 * through standard DPI-C import/export calls.
 *
 * Design principles:
 *   - Scalar-only interface: no C structs cross the DPI boundary
 *   - Handle-based: opaque int handles for TU core instances
 *   - Zero external dependencies: pure C99 with no SV-specific types
 *   - Shareable: compiled as a shared library (libtudpi.so)
 *
 * DPI-C conventions:
 *   - All functions use basic C types (int, long long, void*)
 *   - No C structs in parameter lists or return values
 *   - Memory operations use byte pointers + size
 *   - Functions are declared extern for DPI import
 *
 * Usage (SystemVerilog):
 *   import "DPI-C" function int tu_dpi_init(int pe_rows, int pe_cols);
 *   import "DPI-C" function void tu_dpi_load_weights(int handle, long long src_addr, int bytes);
 *
 * Integration patterns:
 *   1. RTL co-simulation: SV testbench drives both RTL and cmodel,
 *      compares outputs cycle-by-cycle
 *   2. Golden reference: cmodel pre-computes expected results,
 *      RTL checked against golden
 *   3. Standalone verification: cmodel exercised from SV for
 *      architectural validation before RTL exists
 *
 * Reference: IEEE 1800-2017 SystemVerilog, Section 35 (DPI)
 * Reference: Synopsys C-Model Flow for verification
 */

#ifndef TU_DPI_H
#define TU_DPI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Handle Types ---- */

/* Opaque handle for a TU core instance. 0 = invalid. */
typedef int tu_dpi_handle_t;

#define TU_DPI_INVALID_HANDLE 0
#define TU_DPI_MAX_INSTANCES  16

/* ---- Status Codes ---- */

#define TU_DPI_OK           0
#define TU_DPI_ERR_INIT    -1
#define TU_DPI_ERR_HANDLE  -2
#define TU_DPI_ERR_PARAM   -3
#define TU_DPI_ERR_MEMORY  -4
#define TU_DPI_ERR_DMA     -5
#define TU_DPI_ERR_BUSY    -6

/* ---- Dataflow IDs ---- */

#define TU_DPI_DF_WS  0   /* Weight-Stationary */
#define TU_DPI_DF_OS  1   /* Output-Stationary */
#define TU_DPI_DF_RS  2   /* Row-Stationary */
#define TU_DPI_DF_NLR 3   /* No Local Reuse */

/* ---- Counter IDs ---- */

#define TU_DPI_CNT_DMA_BYTES      0
#define TU_DPI_CNT_MMA_CALLS      1
#define TU_DPI_CNT_MMA_TILES      2
#define TU_DPI_CNT_MMA_FLOPS      3
#define TU_DPI_CNT_EST_CYCLES     4
#define TU_DPI_CNT_TOTAL_CYCLES   5
#define TU_DPI_CNT_COMPUTE_ACTIVE 6
#define TU_DPI_CNT_BANK_CONFLICTS 7
#define TU_DPI_CNT_SRAM_READS     8
#define TU_DPI_CNT_SRAM_WRITES    9
#define TU_DPI_CNT_UTILIZATION    10  /* Returns scaled integer: 8530 = 85.30% */

/* ================================================================
 * Lifecycle API
 * ================================================================ */

/*
 * Initialize a TU core instance.
 *
 *   pe_rows:      number of PE rows (e.g., 16, 32, 64)
 *   pe_cols:      number of PE columns
 *   sram_kb:      total SRAM in KB (e.g., 256, 1024)
 *   dataflow:     TU_DPI_DF_WS, _OS, _RS, or _NLR
 *
 * Returns: handle (≥1) on success, negative on error.
 */
int tu_dpi_init(int pe_rows, int pe_cols, int sram_kb, int dataflow);

/*
 * Release a TU core instance.
 *
 * Returns: TU_DPI_OK on success.
 */
int tu_dpi_destroy(int handle);

/*
 * Reset a TU core (clear memories, counters, command queue).
 * Keeps the handle valid.
 *
 * Returns: TU_DPI_OK on success.
 */
int tu_dpi_reset(int handle);

/* ================================================================
 * Memory Access API
 * ================================================================ */

/*
 * Write bytes to TU SRAM.
 *
 *   handle:      TU core handle
 *   region:      0=W-buffer, 1=A-buffer, 2=O-buffer
 *   offset:      byte offset within region
 *   src:         source byte array (host memory)
 *   bytes:       number of bytes to write
 *
 * Returns: TU_DPI_OK on success, negative on error.
 */
int tu_dpi_sram_write(int handle, int region, int offset,
                       const void *src, int bytes);

/*
 * Read bytes from TU SRAM.
 *
 *   handle:      TU core handle
 *   region:      0=W, 1=A, 2=O
 *   offset:      byte offset within region
 *   dst:         destination byte array (host memory)
 *   bytes:       number of bytes to read
 *
 * Returns: TU_DPI_OK on success, negative on error.
 */
int tu_dpi_sram_read(int handle, int region, int offset,
                      void *dst, int bytes);

/*
 * Get SRAM region size in bytes.
 *
 *   handle:      TU core handle
 *   region:      0=W, 1=A, 2=O
 *
 * Returns: size in bytes, or negative on error.
 */
int tu_dpi_sram_size(int handle, int region);

/* ================================================================
 * Command Execution API
 * ================================================================ */

/*
 * Execute a GEMM operation.
 *
 *   handle:      TU core handle
 *   M, N, K:     matrix dimensions
 *   w_offset:    byte offset of W in W-buffer
 *   a_offset:    byte offset of A in A-buffer
 *   o_offset:    byte offset of O in O-buffer
 *   has_bias:    1 if O already contains bias values, 0 otherwise
 *
 * This is a BLOCKING call (synchronous execution).
 * For async, use tu_dpi_submit() + tu_dpi_wait().
 *
 * Returns: estimated cycle count, or negative on error.
 */
long long tu_dpi_gemm(int handle,
                       int M, int N, int K,
                       int w_offset, int a_offset, int o_offset,
                       int has_bias);

/*
 * Execute elementwise operation on accumulator output.
 *
 *   handle:      TU core handle
 *   op:          0=RELU, 1=GELU, 2=SILU, 3=TANH, 4=SIGMOID, 5=EXP, 6=ADD, 7=MUL
 *   o_offset:    byte offset of O in O-buffer
 *   o_rows:      number of rows
 *   o_cols:      number of cols
 *   scalar:      scalar value for ADD/MUL (FP32 bit pattern as int)
 *
 * Returns: cycle count, or negative on error.
 */
long long tu_dpi_elementwise(int handle, int op,
                              int o_offset, int o_rows, int o_cols,
                              int scalar);

/*
 * Execute Softmax on a 2D tensor.
 *
 *   handle:      TU core handle
 *   offset:      byte offset in O-buffer
 *   rows, cols:  tensor dimensions
 *
 * Returns: cycle count, or negative on error.
 */
long long tu_dpi_softmax(int handle, int offset, int rows, int cols);

/*
 * Execute LayerNorm on a 2D tensor.
 *
 *   handle:      TU core handle
 *   offset:      byte offset in O-buffer
 *   rows, cols:  tensor dimensions
 *   epsilon:     normalization epsilon (FP32 bit pattern as int)
 *
 * Returns: cycle count, or negative on error.
 */
long long tu_dpi_layernorm(int handle, int offset, int rows, int cols, int epsilon);

/* ================================================================
 * Async Command Queue API
 * ================================================================ */

/*
 * Submit a GEMM command to the async queue (non-blocking).
 *
 * Same parameters as tu_dpi_gemm().
 * Returns: command ID (≥1), or negative on error.
 */
int tu_dpi_submit_gemm(int handle,
                        int M, int N, int K,
                        int w_offset, int a_offset, int o_offset,
                        int has_bias);

/*
 * Submit a barrier (all prior commands must complete before subsequent).
 *
 * Returns: barrier ID, or negative on error.
 */
int tu_dpi_submit_barrier(int handle);

/*
 * Wait for a specific command to complete.
 *
 *   handle:      TU core handle
 *   cmd_id:      command ID from tu_dpi_submit_*()
 *   timeout_us:  max wait in microseconds, 0 = no timeout
 *
 * Returns: TU_DPI_OK if completed, TU_DPI_ERR_BUSY if timeout.
 */
int tu_dpi_wait(int handle, int cmd_id, int timeout_us);

/*
 * Wait for all outstanding commands to complete.
 *
 * Returns: TU_DPI_OK.
 */
int tu_dpi_sync(int handle);

/* ================================================================
 * Performance Counter API
 * ================================================================ */

/*
 * Read a performance counter.
 *
 *   handle:      TU core handle
 *   counter_id:  TU_DPI_CNT_* constant
 *
 * Returns: counter value (always non-negative for valid handles).
 */
long long tu_dpi_read_counter(int handle, int counter_id);

/*
 * Get a human-readable summary string.
 * Writes into buf (up to 511 chars + null).
 *
 * Returns: TU_DPI_OK.
 */
int tu_dpi_get_summary(int handle, char *buf, int buf_size);

/* ================================================================
 * Dataflow API
 * ================================================================ */

/*
 * Set the active dataflow mode.
 *
 *   handle:      TU core handle
 *   dataflow:    TU_DPI_DF_WS, _OS, _RS, or _NLR
 *
 * Returns: TU_DPI_OK on success.
 */
int tu_dpi_set_dataflow(int handle, int dataflow);

/*
 * Get the current dataflow name.
 * Writes into buf (up to 63 chars + null).
 */
int tu_dpi_get_dataflow_name(int handle, char *buf, int buf_size);

/* ================================================================
 * Configuration API
 * ================================================================ */

/*
 * Get the PE array dimensions.
 *
 *   handle: instance handle
 *   rows_out, cols_out: outputs
 *
 * Returns: TU_DPI_OK.
 */
int tu_dpi_get_pe_dims(int handle, int *rows_out, int *cols_out);

/*
 * Get the SRAM sizes for all 3 regions.
 *
 *   w_size, a_size, o_size: outputs in bytes
 *
 * Returns: TU_DPI_OK.
 */
int tu_dpi_get_sram_sizes(int handle, int *w_size, int *a_size, int *o_size);

#ifdef __cplusplus
}
#endif

#endif /* TU_DPI_H */
