/*
 * TinyTU Production Configuration — TinyTU 2.0-dev
 * Auto-generated from config/tu_config.yaml.
 * Do not edit directly.
 */

#ifndef TU_CONFIG_H
#define TU_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Compute Engine
 * ================================================================ */

#define TU_PE_ROWS              16
#define TU_PE_COLS              16
#define TU_PE_PIPELINE_DEPTH    2
#define TU_MAC_UNITS_PER_PE     1

#define TU_DATAFLOW_MODE_WS            0
#define TU_DATAFLOW_MODE_OS            1
#define TU_DATAFLOW_MODE_RS            2
#define TU_DATAFLOW_MODE_NLR           3
#define TU_DATAFLOW_MODE               TU_DATAFLOW_MODE_WS
#define TU_DATAFLOW_DISPATCH_VIA_PLUGIN 1  /* A4: use pluggable dataflow (1) or legacy inline (0) */

#define TU_PRECISION_FP16       (1 << 0)
#define TU_PRECISION_FP32       (1 << 1)
#define TU_PRECISION_BF16       (1 << 2)
#define TU_PRECISION_FP8        (1 << 3)
#define TU_PRECISION_FP8_E4M3   (1 << 3)  /* D4: Forward pass format */
#define TU_PRECISION_FP8_E5M2   (1 << 4)  /* D4: Backward pass format */
#define TU_PRECISION_INT8       (1 << 5)
#define TU_PRECISION_INT4       (1 << 6)
#define TU_PRECISION_MASK       3
#define TU_ACCUMULATOR_PRECISION_FP32  1

/* INT8 quantization: enable integer quantization path (D2) */
#define TU_INT8_ENABLED              1
#define TU_INT8_ACCUM_BITS           32
#define TU_INT8_SYMMETRIC_DEFAULT    1
/* INT4 quantization: packed UINT4 storage */
#define TU_INT4_ENABLED              1

/* ================================================================
 * Memory System
 * ================================================================ */

#define TU_SRAM_W_SIZE_KB       128
#define TU_SRAM_A_SIZE_KB       64
#define TU_SRAM_O_SIZE_KB       64

#define TU_SRAM_W_SIZE          (TU_SRAM_W_SIZE_KB * 1024)
#define TU_SRAM_A_SIZE          (TU_SRAM_A_SIZE_KB * 1024)
#define TU_SRAM_O_SIZE          (TU_SRAM_O_SIZE_KB * 1024)
#define TU_SRAM_TOTAL           (TU_SRAM_W_SIZE + TU_SRAM_A_SIZE + TU_SRAM_O_SIZE)

#define TU_SRAM_BANKS           32
#define TU_SRAM_BANK_WIDTH      4

/* ================================================================
 * Memory Hierarchy (A3)
 * ================================================================ */

/* Register File (Level 0) — per-PE */
#define TU_MEM_REGFILE_PER_PE       256      /* Bytes per PE */

/* Global Buffer (Level 2) — shared L2 */
#define TU_MEM_GBUF_SIZE            (1 * 1024 * 1024)  /* 1 MB */
#define TU_MEM_GBUF_BANKS           16
#define TU_MEM_GBUF_BANK_WIDTH      8        /* 64-bit words */

#define TU_CONFLICT_NONE        0
#define TU_CONFLICT_DETECT      1
#define TU_CONFLICT_STALL       2
#define TU_CONFLICT_MODEL       1

#define TU_LATENCY_SRAM_READ    1
#define TU_LATENCY_SRAM_WRITE   1
#define TU_LATENCY_DRAM_READ    50
#define TU_LATENCY_DRAM_WRITE   50

/* ================================================================
 * SRAM Bandwidth Model (M2)
 * ================================================================ */

/* Maximum words per bank per cycle (1 = single-ported, 2 = dual-ported) */
#define TU_SRAM_WORDS_PER_CYCLE 1

/* Arbitration policy when multiple accesses hit the same bank */
#define TU_SRAM_ARB_NONE          0  /* No arbitration — all pass (unrealistic) */
#define TU_SRAM_ARB_ROUND_ROBIN   1  /* Round-robin between contending ports */
#define TU_SRAM_ARB_PRIORITY      2  /* Fixed priority (read > write) */
#define TU_SRAM_ARB_MODE          TU_SRAM_ARB_ROUND_ROBIN

/* Bandwidth metering: refill-based budget (words per cycle window) */
#define TU_SRAM_BW_WINDOW_CYCLES  4  /* Refill window in cycles */

/* Stall penalty when bandwidth is exhausted (in cycles) */
#define TU_SRAM_BW_STALL_PENALTY  2

/* ================================================================
 * DRAM Model
 * ================================================================ */

#define TU_DRAM_IDEAL             0
#define TU_DRAM_HBM2              1
#define TU_DRAM_HBM2E             2
#define TU_DRAM_HBM3              3
#define TU_DRAM_DDR4              4
#define TU_DRAM_DDR5              5
#define TU_DRAM_LPDDR5            6
#define TU_DRAM_CUSTOM            7
#define TU_DRAM_TYPE              TU_DRAM_IDEAL
#define TU_DRAM_BANDWIDTH_GBPS    256.0
#define TU_DRAM_CHANNELS          8
#define TU_DRAM_MODEL_ROW_HIT     0

/* ================================================================
 * DMA Engine
 * ================================================================ */

#define TU_DMA_BUS_WIDTH_BITS   256
#define TU_DMA_BUS_WIDTH_BYTES  (TU_DMA_BUS_WIDTH_BITS / 8)
#define TU_DMA_MAX_BURST_BYTES  64
#define TU_DMA_CHANNELS         3
#define TU_DMA_MAX_OUTSTANDING  4
#define TU_DMA_ASYNC_MODE       0

/* ================================================================
 * ISA / Command Queue
 * ================================================================ */

#define TU_ISA_INSTR_WIDTH_BITS 96
#define TU_ISA_QUEUE_DEPTH      16
#define TU_ISA_DEP_CHECKING     0

/* ================================================================
 * Multi-Core
 * ================================================================ */

#define TU_MULTICORE_ENABLED    0
#define TU_NUM_CORES            1
#define TU_INTERCONNECT_NONE    0
#define TU_INTERCONNECT_RING    1
#define TU_INTERCONNECT_MESH    2
#define TU_INTERCONNECT_MODE    0
#define TU_CACHE_COHERENCE      0

/* ================================================================
 * Performance Model
 * ================================================================ */

#define TU_CYCLE_MODEL_FUNCTIONAL    0
#define TU_CYCLE_MODEL_ESTIMATED     1
#define TU_CYCLE_MODEL_CYCLE_ACCURATE 2
#define TU_CYCLE_MODEL               0

#define TU_COUNTERS_ENABLED           1
#define TU_COUNTERS_DETAILED_STALLS   0
#define TU_TRACE_ENABLED              0

/* ================================================================
 * Logging & Trace (Q2)
 * ================================================================ */

/* Default minimum log level: ERROR=1, WARN=2, INFO=3, DEBUG=4, TRACE=5 */
#define TU_LOG_LEVEL_DEFAULT         TU_LOG_INFO

/* Enable colored output (ANSI escape codes) */
#define TU_LOG_USE_COLOR             1

/* Show timestamps in log output [cycles] */
#define TU_LOG_SHOW_TIMESTAMPS       1

/* Show source file:line (useful for DEBUG and TRACE) */
#define TU_LOG_SHOW_FILE_LINE        0

/* Trace buffer: record execution events for VCD export */
#define TU_TRACE_MAX_EVENTS          65536
#define TU_TRACE_EXPORT_VCD          1

/* ================================================================
 * Precision Parameters
 * ================================================================ */

#define TU_FP16_ROUNDING_RNE          0
#define TU_FP16_ROUNDING_RTZ          1
#define TU_FP16_ROUNDING_STOCHASTIC   2
#define TU_FP16_ROUNDING_MODE         0
#define TU_FP16_SUBNORMAL_FLUSH       1
#define TU_FP16_SATURATE              1

#define TU_FP32_ROUNDING_MODE         0

/* ================================================================
 * Sparsity
 * ================================================================ */

#define TU_SPARSITY_ENABLED           0
#define TU_SPARSITY_2OF4              0
#define TU_SPARSITY_UNSTRUCTURED      0

/* ================================================================
 * Verification
 * ================================================================ */

#define TU_VERIFY_GOLDEN_NUMPY        0
#define TU_VERIFY_GOLDEN_PYTORCH      1
#define TU_VERIFY_GOLDEN_MODE         1
#define TU_VERIFY_RANDOM_ITERS        1000
#define TU_VERIFY_ERROR_TOLERANCE     1e-05

/* ================================================================
 * Runtime Configuration (modifiable without recompilation)
 * ================================================================ */

typedef struct {
    uint16_t pe_rows;
    uint16_t pe_cols;
    uint32_t sram_w_size;
    uint32_t sram_a_size;
    uint32_t sram_o_size;
    bool     counters_enabled;
    bool     detailed_stalls;
    bool     trace_enabled;
    char     trace_file[256];
    bool     verify_enabled;
    double   verify_tolerance;
} tu_runtime_config_t;

static inline tu_runtime_config_t tu_config_default(void) {
    return (tu_runtime_config_t){
        .pe_rows           = TU_PE_ROWS,
        .pe_cols           = TU_PE_COLS,
        .sram_w_size       = TU_SRAM_W_SIZE,
        .sram_a_size       = TU_SRAM_A_SIZE,
        .sram_o_size       = TU_SRAM_O_SIZE,
        .counters_enabled  = TU_COUNTERS_ENABLED,
        .detailed_stalls   = TU_COUNTERS_DETAILED_STALLS,
        .trace_enabled     = TU_TRACE_ENABLED,
        .trace_file        = "",
        .verify_enabled    = false,
        .verify_tolerance  = TU_VERIFY_ERROR_TOLERANCE,
    };
}

#ifdef __cplusplus
}
#endif

#endif /* TU_CONFIG_H */
