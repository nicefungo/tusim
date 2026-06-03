/*
 * TU CModel — DPI-C Integration Wrapper Implementation
 * ======================================================
 * Gap I1: Production-grade DPI-C wrapper for SystemVerilog
 * RTL co-simulation and golden-reference comparison.
 */

#include "bindings/tu_dpi.h"
#include "tu_cmodel.h"
#include "tu_precision.h"
#include "tu_sram.h"
#include "compute/dataflow/dataflow_interface.h"
#include "compute/elementwise_pipeline.h"
#include "compute/softmax_engine.h"
#include "compute/normalization_engine.h"
#include "infra/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Internal Instance Pool ---- */

typedef struct {
    int         in_use;
    tu_state_t  state;
    int         pe_rows;
    int         pe_cols;
    int         dataflow;
} tu_dpi_instance_t;

static tu_dpi_instance_t g_instances[TU_DPI_MAX_INSTANCES];
static int g_dpi_initialized = 0;

static void tu_dpi_ensure_init(void) {
    if (!g_dpi_initialized) {
        memset(g_instances, 0, sizeof(g_instances));
        g_dpi_initialized = 1;
    }
}

static tu_dpi_instance_t* tu_dpi_get_instance(int handle) {
    if (handle < 1 || handle > TU_DPI_MAX_INSTANCES) return NULL;
    tu_dpi_instance_t *inst = &g_instances[handle - 1];
    if (!inst->in_use) return NULL;
    return inst;
}

static void tu_dpi_set_global(tu_dpi_instance_t *inst) {
    extern tu_state_t g_tu;
    memcpy(&g_tu, &inst->state, sizeof(tu_state_t));
}

static void tu_dpi_save_global(tu_dpi_instance_t *inst) {
    extern tu_state_t g_tu;
    memcpy(&inst->state, &g_tu, sizeof(tu_state_t));
}

/* Helper: get data pointer from SRAM region */
static void* tu_dpi_sram_data(tu_sram_region_t *r) {
    return tu_sram_raw_ptr(r);
}

static uint32_t tu_dpi_sram_total(tu_sram_region_t *r) {
    return r->total_size;
}

/* Helper: safe float-from-int */
static float int_to_float(int i) {
    float f;
    memcpy(&f, &i, sizeof(f));
    return f;
}

/* ================================================================
 * Lifecycle API
 * ================================================================ */

int tu_dpi_init(int pe_rows, int pe_cols, int sram_kb, int dataflow) {
    tu_dpi_ensure_init();

    if (pe_rows < 1 || pe_cols < 1 || pe_rows > 256 || pe_cols > 256)
        return TU_DPI_ERR_PARAM;
    if (sram_kb < 64 || sram_kb > 65536)
        return TU_DPI_ERR_PARAM;
    if (dataflow < 0 || dataflow > 3)
        return TU_DPI_ERR_PARAM;

    int slot = -1;
    for (int i = 0; i < TU_DPI_MAX_INSTANCES; i++) {
        if (!g_instances[i].in_use) { slot = i; break; }
    }
    if (slot < 0) return TU_DPI_ERR_INIT;

    tu_dpi_instance_t *inst = &g_instances[slot];
    memset(inst, 0, sizeof(*inst));

    extern tu_state_t g_tu;
    tu_dpi_set_global(inst);

    tu_runtime_config_t rt_cfg;
    memcpy(&rt_cfg, &g_tu.rt_cfg, sizeof(rt_cfg));
    rt_cfg.pe_rows     = (uint16_t)pe_rows;
    rt_cfg.pe_cols     = (uint16_t)pe_cols;
    rt_cfg.sram_w_size = (uint32_t)(sram_kb * 1024 / 2);
    rt_cfg.sram_a_size = (uint32_t)(sram_kb * 1024 / 4);
    rt_cfg.sram_o_size = (uint32_t)(sram_kb * 1024 / 4);

    tu_init_with_config(&rt_cfg);

    static const int df_map[] = {
        TU_DATAFLOW_WEIGHT_STATIONARY, TU_DATAFLOW_OUTPUT_STATIONARY,
        TU_DATAFLOW_ROW_STATIONARY,    TU_DATAFLOW_NO_LOCAL_REUSE,
    };
    tu_set_dataflow(df_map[dataflow]);

    inst->in_use   = 1;
    inst->pe_rows  = pe_rows;
    inst->pe_cols  = pe_cols;
    inst->dataflow = dataflow;

    tu_dpi_save_global(inst);
    return slot + 1;
}

int tu_dpi_destroy(int handle) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;
    inst->in_use = 0;
    memset(inst, 0, sizeof(*inst));
    return TU_DPI_OK;
}

int tu_dpi_reset(int handle) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    tu_init();
    static const int df_map[] = {
        TU_DATAFLOW_WEIGHT_STATIONARY, TU_DATAFLOW_OUTPUT_STATIONARY,
        TU_DATAFLOW_ROW_STATIONARY,    TU_DATAFLOW_NO_LOCAL_REUSE,
    };
    tu_set_dataflow(df_map[inst->dataflow]);
    tu_dpi_save_global(inst);

    return TU_DPI_OK;
}

/* ================================================================
 * Memory Access API
 * ================================================================ */

static tu_sram_region_t* tu_dpi_get_region(tu_state_t *state, int region) {
    switch (region) {
        case 0: return &state->sram_w;
        case 1: return &state->sram_a;
        case 2: return &state->sram_o;
        default: return NULL;
    }
}

int tu_dpi_sram_write(int handle, int region, int offset,
                       const void *src, int bytes) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst || region < 0 || region > 2 || offset < 0 || bytes < 0)
        return TU_DPI_ERR_PARAM;

    tu_dpi_set_global(inst);
    extern tu_state_t g_tu;
    tu_sram_region_t *reg = tu_dpi_get_region(&g_tu, region);
    if (!reg) { tu_dpi_save_global(inst); return TU_DPI_ERR_PARAM; }

    uint32_t total = tu_dpi_sram_total(reg);
    if ((uint32_t)(offset + bytes) > total) {
        tu_dpi_save_global(inst);
        return TU_DPI_ERR_MEMORY;
    }

    memcpy((uint8_t*)tu_dpi_sram_data(reg) + offset, src, (size_t)bytes);
    tu_dpi_save_global(inst);
    return TU_DPI_OK;
}

int tu_dpi_sram_read(int handle, int region, int offset,
                      void *dst, int bytes) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst || region < 0 || region > 2 || offset < 0 || bytes < 0 || !dst)
        return TU_DPI_ERR_PARAM;

    tu_dpi_set_global(inst);
    extern tu_state_t g_tu;
    tu_sram_region_t *reg = tu_dpi_get_region(&g_tu, region);
    if (!reg) { tu_dpi_save_global(inst); return TU_DPI_ERR_PARAM; }

    uint32_t total = tu_dpi_sram_total(reg);
    if ((uint32_t)(offset + bytes) > total) {
        tu_dpi_save_global(inst);
        return TU_DPI_ERR_MEMORY;
    }

    memcpy(dst, (uint8_t*)tu_dpi_sram_data(reg) + offset, (size_t)bytes);
    tu_dpi_save_global(inst);
    return TU_DPI_OK;
}

int tu_dpi_sram_size(int handle, int region) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst || region < 0 || region > 2) return TU_DPI_ERR_PARAM;

    extern tu_state_t g_tu;
    tu_dpi_set_global(inst);
    tu_sram_region_t *reg = tu_dpi_get_region(&g_tu, region);
    int size = reg ? (int)tu_dpi_sram_total(reg) : 0;
    tu_dpi_save_global(inst);
    return size;
}

/* ================================================================
 * Command Execution API
 * ================================================================ */

long long tu_dpi_gemm(int handle,
                       int M, int N, int K,
                       int w_offset, int a_offset, int o_offset,
                       int has_bias) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    tu_mma((uint16_t)M, (uint16_t)N, (uint16_t)K,
           (uint32_t)w_offset, (uint32_t)a_offset, (uint32_t)o_offset,
           (bool)has_bias);

    extern tu_state_t g_tu;
    long long cycles = (long long)g_tu.estimated_cycles;
    tu_dpi_save_global(inst);
    return cycles;
}

long long tu_dpi_elementwise(int handle, int op,
                              int o_offset, int o_rows, int o_cols,
                              int scalar) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    extern tu_state_t g_tu;

    /* Map DPI opcodes to internal elementwise opcodes:
     * DPI: 0=RELU, 1=GELU, 2=SILU, 3=TANH, 4=SIGMOID, 5=EXP, 6=ADD, 7=MUL
     * EW:  1=RELU, 2=GELU, 3=SILU, 5=TANH, 4=SIGMOID, 6=EXP, 11=ADD, 12=MUL
     */
    static const int op_map[] = {
        TU_EW_RELU, TU_EW_GELU, TU_EW_SILU, TU_EW_TANH,
        TU_EW_SIGMOID, TU_EW_EXP, TU_EW_ADD, TU_EW_MUL
    };
    static const int op_map_count = 8;

    if (op < 0 || op >= op_map_count) {
        tu_dpi_save_global(inst);
        return TU_DPI_ERR_PARAM;
    }

    uint32_t elem_count = (uint32_t)o_rows * (uint32_t)o_cols;
    long long cycles = 0;

    if (op == 6 || op == 7) {
        /* ADD or MUL: use scalar */
        float s = int_to_float(scalar);
        cycles = (long long)tu_ew_apply_binary_scalar(
            &g_tu.sram_o, (uint32_t)o_offset, elem_count,
            (tu_ew_opcode_t)op_map[op], s);
    } else {
        cycles = (long long)tu_ew_apply_unary(
            &g_tu.sram_o, (uint32_t)o_offset, elem_count,
            (tu_ew_opcode_t)op_map[op]);
    }

    tu_dpi_save_global(inst);
    return cycles;
}

long long tu_dpi_softmax(int handle, int offset, int rows, int cols) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    extern tu_state_t g_tu;

    long long cycles = (long long)tu_softmax_2d(
        &g_tu.sram_o, (uint32_t)offset,
        (uint32_t)rows, (uint32_t)cols,
        0.0f, true);

    tu_dpi_save_global(inst);
    return cycles;
}

long long tu_dpi_layernorm(int handle, int offset, int rows, int cols, int epsilon) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    extern tu_state_t g_tu;

    float eps = int_to_float(epsilon);
    long long cycles = (long long)tu_layernorm_2d(
        &g_tu.sram_o, (uint32_t)offset,
        (uint32_t)rows, (uint32_t)cols,
        NULL, NULL, eps);

    tu_dpi_save_global(inst);
    return cycles;
}

/* ================================================================
 * Async Command Queue API
 * ================================================================ */

int tu_dpi_submit_gemm(int handle,
                        int M, int N, int K,
                        int w_offset, int a_offset, int o_offset,
                        int has_bias) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    int cmd_id = tu_cmdq_submit_mma(
        (uint16_t)M, (uint16_t)N, (uint16_t)K,
        (uint32_t)w_offset, (uint32_t)a_offset, (uint32_t)o_offset,
        (bool)has_bias);
    tu_dpi_save_global(inst);
    return cmd_id;
}

int tu_dpi_submit_barrier(int handle) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    int id = tu_cmdq_submit_barrier();
    tu_dpi_save_global(inst);
    return id;
}

int tu_dpi_wait(int handle, int cmd_id, int timeout_us) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;
    (void)cmd_id; (void)timeout_us;

    tu_dpi_set_global(inst);
    tu_cmdq_sync_all();
    tu_dpi_save_global(inst);
    return TU_DPI_OK;
}

int tu_dpi_sync(int handle) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    tu_cmdq_sync_all();
    tu_dpi_save_global(inst);
    return TU_DPI_OK;
}

/* ================================================================
 * Performance Counter API
 * ================================================================ */

long long tu_dpi_read_counter(int handle, int counter_id) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    extern tu_state_t g_tu;
    long long val = 0;

    switch (counter_id) {
        case TU_DPI_CNT_DMA_BYTES:
            val = (long long)g_tu.total_dma_bytes; break;
        case TU_DPI_CNT_MMA_CALLS:
            val = (long long)g_tu.total_mma_calls; break;
        case TU_DPI_CNT_MMA_TILES:
            val = (long long)g_tu.total_mma_tiles; break;
        case TU_DPI_CNT_MMA_FLOPS:
            val = (long long)g_tu.total_mma_flops; break;
        case TU_DPI_CNT_EST_CYCLES:
            val = (long long)g_tu.estimated_cycles; break;
        case TU_DPI_CNT_TOTAL_CYCLES:
        case TU_DPI_CNT_COMPUTE_ACTIVE:
            val = (long long)g_tu.estimated_cycles; break;
        case TU_DPI_CNT_BANK_CONFLICTS:
            val = (long long)(g_tu.sram_w.banks.conflicts +
                              g_tu.sram_a.banks.conflicts +
                              g_tu.sram_o.banks.conflicts); break;
        case TU_DPI_CNT_SRAM_READS:
            val = (long long)(g_tu.sram_w.banks.reads +
                              g_tu.sram_a.banks.reads +
                              g_tu.sram_o.banks.reads); break;
        case TU_DPI_CNT_SRAM_WRITES:
            val = (long long)(g_tu.sram_w.banks.writes +
                              g_tu.sram_a.banks.writes +
                              g_tu.sram_o.banks.writes); break;
        case TU_DPI_CNT_UTILIZATION:
            if (g_tu.estimated_cycles > 0) {
                val = (long long)((double)g_tu.estimated_cycles /
                       g_tu.estimated_cycles * 10000.0);
            }
            break;
        default: val = 0; break;
    }

    tu_dpi_save_global(inst);
    return val;
}

int tu_dpi_get_summary(int handle, char *buf, int buf_size) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst || !buf || buf_size < 1) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    extern tu_state_t g_tu;

    snprintf(buf, (size_t)buf_size,
        "TU %dx%d | DF=%s | DMA=%llu B | MMA: %llu calls, %llu tiles, %llu FLOPS | Cyc: %llu est",
        inst->pe_rows, inst->pe_cols,
        (inst->dataflow == 0 ? "WS" : inst->dataflow == 1 ? "OS" :
         inst->dataflow == 2 ? "RS" : "NLR"),
        (unsigned long long)g_tu.total_dma_bytes,
        (unsigned long long)g_tu.total_mma_calls,
        (unsigned long long)g_tu.total_mma_tiles,
        (unsigned long long)g_tu.total_mma_flops,
        (unsigned long long)g_tu.estimated_cycles);

    tu_dpi_save_global(inst);
    return TU_DPI_OK;
}

/* ================================================================
 * Dataflow API
 * ================================================================ */

int tu_dpi_set_dataflow(int handle, int dataflow) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst || dataflow < 0 || dataflow > 3) return TU_DPI_ERR_PARAM;

    inst->dataflow = dataflow;
    tu_dpi_set_global(inst);

    static const int df_map[] = {
        TU_DATAFLOW_WEIGHT_STATIONARY, TU_DATAFLOW_OUTPUT_STATIONARY,
        TU_DATAFLOW_ROW_STATIONARY,    TU_DATAFLOW_NO_LOCAL_REUSE,
    };
    tu_set_dataflow(df_map[dataflow]);

    tu_dpi_save_global(inst);
    return TU_DPI_OK;
}

int tu_dpi_get_dataflow_name(int handle, char *buf, int buf_size) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst || !buf || buf_size < 1) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    const char *name = tu_get_dataflow_name();
    snprintf(buf, (size_t)buf_size, "%s", name ? name : "unknown");
    tu_dpi_save_global(inst);
    return TU_DPI_OK;
}

/* ================================================================
 * Configuration API
 * ================================================================ */

int tu_dpi_get_pe_dims(int handle, int *rows_out, int *cols_out) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;
    if (rows_out) *rows_out = inst->pe_rows;
    if (cols_out) *cols_out = inst->pe_cols;
    return TU_DPI_OK;
}

int tu_dpi_get_sram_sizes(int handle, int *w_size, int *a_size, int *o_size) {
    tu_dpi_instance_t *inst = tu_dpi_get_instance(handle);
    if (!inst) return TU_DPI_ERR_HANDLE;

    tu_dpi_set_global(inst);
    extern tu_state_t g_tu;
    if (w_size) *w_size = (int)tu_dpi_sram_total(&g_tu.sram_w);
    if (a_size) *a_size = (int)tu_dpi_sram_total(&g_tu.sram_a);
    if (o_size) *o_size = (int)tu_dpi_sram_total(&g_tu.sram_o);
    tu_dpi_save_global(inst);
    return TU_DPI_OK;
}
