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

#define TU_DATAFLOW_WEIGHT_STATIONARY  0
#define TU_DATAFLOW_OUTPUT_STATIONARY  1
#define TU_DATAFLOW_ROW_STATIONARY     2
#define TU_DATAFLOW_MODE              0

#define TU_PRECISION_FP16       (1 << 0)
#define TU_PRECISION_FP32       (1 << 1)
#define TU_PRECISION_BF16       (1 << 2)
#define TU_PRECISION_FP8        (1 << 3)
#define TU_PRECISION_INT8       (1 << 4)
#define TU_PRECISION_INT4       (1 << 5)
#define TU_PRECISION_MASK       3
#define TU_ACCUMULATOR_PRECISION_FP32  1

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

#define TU_CONFLICT_NONE        0
#define TU_CONFLICT_DETECT      1
#define TU_CONFLICT_STALL       2
#define TU_CONFLICT_MODEL       1

#define TU_LATENCY_SRAM_READ    1
#define TU_LATENCY_SRAM_WRITE   1
#define TU_LATENCY_DRAM_READ    50
#define TU_LATENCY_DRAM_WRITE   50

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
 * Precision Parameters
 * ================================================================ */

#define TU_FP16_ROUNDING_RNE          0
#define TU_FP16_ROUNDING_RTZ          1
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
