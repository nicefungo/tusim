/*
 * TinyTU CModel — Functional Implementation
 * ==========================================
 * Parametric systolic array (TU_PE_ROWS × TU_PE_COLS).
 * FP16 multiply → FP32 accumulate. Fully configurable via tu_config.h
 * with runtime overrides via tu_runtime_config_t.
 *
 * SRAM is managed via the banked tu_sram_region_t module.
 * DMA goes through the tu_dma_engine_t module.
 */
#include "tu_cmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

tu_state_t g_tu = {0};

/* ---- Lifecycle ---- */

void tu_init(void) {
    tu_runtime_config_t cfg = tu_config_default();
    tu_init_with_config(&cfg);
}

void tu_init_with_config(const tu_runtime_config_t *cfg) {
    /* Tear down previous state if re-initializing */
    if (g_tu.initialized) {
        tu_sram_destroy(&g_tu.sram_w);
        tu_sram_destroy(&g_tu.sram_a);
        tu_sram_destroy(&g_tu.sram_o);
    }

    memset(&g_tu, 0, sizeof(g_tu));

    /* Store runtime config */
    g_tu.rt_cfg = *cfg;

    /* Allocate SRAM regions with banking */
    tu_sram_init(&g_tu.sram_w, cfg->sram_w_size, "W-buf");
    tu_sram_init(&g_tu.sram_a, cfg->sram_a_size, "A-buf");
    tu_sram_init(&g_tu.sram_o, cfg->sram_o_size, "O-buf");

    /* Initialize DMA engine */
    tu_dma_init(TU_DMA_ASYNC_MODE ? true : false);

    /* Initialize command queue */
    g_tu.cmdq = tu_cmdq_create(TU_ISA_QUEUE_DEPTH, TU_CYCLE_MODEL == TU_CYCLE_MODEL_FUNCTIONAL);

    g_tu.initialized = true;
}

void tu_print_stats(void) {
    uint16_t pe_rows = g_tu.rt_cfg.pe_rows;
    uint16_t pe_cols = g_tu.rt_cfg.pe_cols;

    fprintf(stderr,
        "\n"
        "═══════════════════════════════════════════\n"
        "  TinyTU CModel — Performance Report\n"
        "───────────────────────────────────────────\n"
        "  PE Array    : %u×%u (%u MACs)\n"
        "  DMA bytes   : %lu\n"
        "  MMA calls   : %lu\n"
        "  MMA tiles   : %lu (%u×%u×%u per tile)\n"
        "  MMA FLOPS   : %lu (FP16 MACs)\n"
        "  Est. cycles : %lu\n"
        "───────────────────────────────────────────\n",
        pe_rows, pe_cols, pe_rows * pe_cols,
        (unsigned long)g_tu.total_dma_bytes,
        (unsigned long)g_tu.total_mma_calls,
        (unsigned long)g_tu.total_mma_tiles,
        pe_rows, pe_cols, pe_cols,
        (unsigned long)g_tu.total_mma_flops,
        (unsigned long)g_tu.estimated_cycles);

    /* Print per-SRAM-region bank stats */
    tu_sram_print_stats(&g_tu.sram_w);
    tu_sram_print_stats(&g_tu.sram_a);
    tu_sram_print_stats(&g_tu.sram_o);

    /* Print DMA stats */
    tu_dma_print_stats();

    fprintf(stderr,
        "═══════════════════════════════════════════\n\n");
}

/* ---- DMA (delegates to tu_dma module) ---- */

static void check_sram_bounds(const tu_sram_region_t *r, uint32_t offset, uint32_t size_bytes) {
    if (offset + size_bytes > r->total_size) {
        fprintf(stderr, "TU ERROR: %s overflow: offset=%u size=%u max=%u\n",
                r->name, offset, size_bytes, r->total_size);
        abort();
    }
}

void tu_dma_load_w(const void *host_ptr, uint32_t tu_offset, uint32_t size_bytes) {
    check_sram_bounds(&g_tu.sram_w, tu_offset, size_bytes);
    tu_dma_load(TU_DMA_CHAN_W, &g_tu.sram_w, tu_offset, host_ptr, size_bytes);
    g_tu.total_dma_bytes += size_bytes;
    g_tu.estimated_cycles += g_tu.dma.estimated_cycles;
    g_tu.dma.estimated_cycles = 0; /* don't double-count */
}

void tu_dma_load_a(const void *host_ptr, uint32_t tu_offset, uint32_t size_bytes) {
    check_sram_bounds(&g_tu.sram_a, tu_offset, size_bytes);
    tu_dma_load(TU_DMA_CHAN_A, &g_tu.sram_a, tu_offset, host_ptr, size_bytes);
    g_tu.total_dma_bytes += size_bytes;
    g_tu.estimated_cycles += g_tu.dma.estimated_cycles;
    g_tu.dma.estimated_cycles = 0;
}

void tu_dma_store_o(void *host_ptr, uint32_t tu_offset, uint32_t size_bytes) {
    check_sram_bounds(&g_tu.sram_o, tu_offset, size_bytes);
    tu_dma_store(TU_DMA_CHAN_O, &g_tu.sram_o, tu_offset, host_ptr, size_bytes);
    g_tu.total_dma_bytes += size_bytes;
    g_tu.estimated_cycles += g_tu.dma.estimated_cycles;
    g_tu.dma.estimated_cycles = 0;
}

void tu_dma_load_o(const void *host_ptr, uint32_t tu_offset, uint32_t size_bytes) {
    check_sram_bounds(&g_tu.sram_o, tu_offset, size_bytes);
    tu_sram_write_bulk(&g_tu.sram_o, tu_offset, host_ptr, size_bytes);
    g_tu.total_dma_bytes += size_bytes;
    g_tu.estimated_cycles += (size_bytes + TU_DMA_BUS_WIDTH_BYTES - 1) / TU_DMA_BUS_WIDTH_BYTES;
}

/* ---- MMA (parameterized systolic array) ---- */

void tu_mma(uint16_t M, uint16_t N, uint16_t K,
            uint32_t w_offset, uint32_t a_offset, uint32_t o_offset,
            bool has_bias) {
    if (!g_tu.initialized) { fprintf(stderr, "TU ERROR: not initialized\n"); abort(); }

    uint16_t pe_rows = g_tu.rt_cfg.pe_rows;
    uint16_t pe_cols = g_tu.rt_cfg.pe_cols;

    g_tu.total_mma_calls++;

    uint32_t w_bytes = (uint32_t)M * K * sizeof(fp16_t);
    uint32_t a_bytes = (uint32_t)K * N * sizeof(fp16_t);
    uint32_t o_bytes = (uint32_t)M * N * sizeof(fp32_t);

    check_sram_bounds(&g_tu.sram_w, w_offset, w_bytes);
    check_sram_bounds(&g_tu.sram_a, a_offset, a_bytes);
    check_sram_bounds(&g_tu.sram_o, o_offset, o_bytes);

    uint8_t *w_raw = tu_sram_raw_ptr(&g_tu.sram_w);
    uint8_t *a_raw = tu_sram_raw_ptr(&g_tu.sram_a);
    uint8_t *o_raw = tu_sram_raw_ptr(&g_tu.sram_o);

    const fp16_t *W = (const fp16_t *)(w_raw + w_offset);
    const fp16_t *A = (const fp16_t *)(a_raw + a_offset);
    fp32_t       *O = (fp32_t       *)(o_raw + o_offset);

    /* Bias: reverse-order FP16→FP32 expansion */
    if (has_bias) {
        fp16_t *bf = (fp16_t *)O;
        for (int m = M - 1; m >= 0; m--)
            for (int n = N - 1; n >= 0; n--)
                O[m * N + n] = fp16_to_fp32(bf[m * N + n]);
    }

    /* Tiled execution using configurable PE dimensions */
    uint16_t mt = (M + pe_rows - 1) / pe_rows;
    uint16_t nt = (N + pe_cols - 1) / pe_cols;
    uint16_t kt = (K + pe_cols - 1) / pe_cols;

    for (uint16_t mi = 0; mi < mt; mi++) {
        uint16_t ms = mi * pe_rows;
        uint16_t me = ((mi + 1) * pe_rows < M ? (mi + 1) * pe_rows : M);
        uint16_t ml = me - ms;

        for (uint16_t ni = 0; ni < nt; ni++) {
            uint16_t ns = ni * pe_cols;
            uint16_t ne = ((ni + 1) * pe_cols < N ? (ni + 1) * pe_cols : N);
            uint16_t nl = ne - ns;

            for (uint16_t ki = 0; ki < kt; ki++) {
                uint16_t ks = ki * pe_cols;
                uint16_t ke = ((ki + 1) * pe_cols < K ? (ki + 1) * pe_cols : K);
                uint16_t kl = ke - ks;

                g_tu.total_mma_tiles++;
                g_tu.total_mma_flops += (uint64_t)ml * nl * kl * 2;

                /* Pipeline fill: TU_PE_PIPELINE_DEPTH cycles per column */
                g_tu.estimated_cycles += TU_PE_PIPELINE_DEPTH * pe_cols;

                /* MAC compute */
                for (uint16_t m = 0; m < ml; m++)
                    for (uint16_t n = 0; n < nl; n++) {
                        fp32_t psum = 0.0f;
                        for (uint16_t k = 0; k < kl; k++)
                            psum += fp16_to_fp32(W[(ms + m) * K + (ks + k)])
                                  * fp16_to_fp32(A[(ks + k) * N + (ns + n)]);
                        O[(ms + m) * N + (ns + n)] += psum;
                    }
                g_tu.estimated_cycles += kl;
            }
        }
    }
}

/* ---- SYNC ---- */
void tu_sync(void) {
    tu_dma_sync();
    g_tu.estimated_cycles += TU_PE_PIPELINE_DEPTH * g_tu.rt_cfg.pe_cols;
}

/* ---- Command Queue Convenience API ---- */

tu_command_queue_t *tu_get_cmdq(void) {
    return g_tu.cmdq;
}

int tu_cmdq_submit_mma(uint16_t M, uint16_t N, uint16_t K,
                       uint32_t w_offset, uint32_t a_offset, uint32_t o_offset,
                       bool has_bias) {
    tu_cmd_mma_desc_t desc = {
        .M = M, .N = N, .K = K,
        .w_offset = w_offset, .a_offset = a_offset, .o_offset = o_offset,
        .has_bias = has_bias
    };
    return tu_cmdq_submit(g_tu.cmdq, TU_CMD_MMA, &desc, 0, NULL, NULL);
}

int tu_cmdq_submit_dma_load(uint8_t channel, uint32_t sram_offset,
                            const void *host_ptr, uint32_t size_bytes) {
    tu_cmd_dma_desc_t desc = {
        .channel = channel, .is_store = false,
        .sram_offset = sram_offset, .size_bytes = size_bytes,
        .host_ptr = (void *)host_ptr
    };
    return tu_cmdq_submit(g_tu.cmdq, TU_CMD_DMA_LOAD, &desc, 0, NULL, NULL);
}

int tu_cmdq_submit_dma_store(uint8_t channel, uint32_t sram_offset,
                             void *host_ptr, uint32_t size_bytes) {
    tu_cmd_dma_desc_t desc = {
        .channel = channel, .is_store = true,
        .sram_offset = sram_offset, .size_bytes = size_bytes,
        .host_ptr = host_ptr
    };
    return tu_cmdq_submit(g_tu.cmdq, TU_CMD_DMA_STORE, &desc, 0, NULL, NULL);
}

int tu_cmdq_submit_barrier(void) {
    return tu_cmdq_barrier(g_tu.cmdq);
}

int tu_cmdq_submit_elementwise(uint8_t sram_region, uint32_t sram_offset,
                               uint32_t elem_count,
                               const uint8_t *ops, uint8_t num_ops,
                               const float *scalars, const bool *has_scalar) {
    tu_cmd_ew_desc_t desc = {0};
    desc.sram_region = sram_region;
    desc.sram_offset = sram_offset;
    desc.elem_count  = elem_count;
    desc.num_ops     = num_ops;
    if (num_ops > 8) num_ops = 8;

    uint8_t si = 0;
    for (uint8_t i = 0; i < num_ops; i++) {
        desc.ops[i] = ops[i];
        if (has_scalar && has_scalar[i] && si < 4 && scalars) {
            desc.has_scalar[si] = true;
            desc.scalars[si]    = scalars[i];
            si++;
        }
    }
    return tu_cmdq_submit(g_tu.cmdq, TU_CMD_ELEMENTWISE, &desc, 0, NULL, NULL);
}

void tu_cmdq_sync_all(void) {
    tu_cmdq_sync(g_tu.cmdq);
}
