/*
 * Softmax-After-Attention Pipeline Sweep
 * ======================================
 * Cross-engine interaction: measures the cycle-cost of running a standalone
 * softmax pass on attention output (O-buffer), simulating scenarios where
 * the attention engine's internal softmax is insufficient (e.g., cross-head
 * pooling, temperature re-scaling, or decoupled softmax for multi-pass
 * transformer variants).
 *
 * Question: "What overhead does a standalone softmax add to attention latency,
 * across PE array sizes and head dimensions?"
 *
 * Approach: Run attention on single-Q-tile workloads (contiguous O-buffer
 * output @ offset 0), then run softmax in-place on the O-buffer FP32 data.
 * The softmax engine only returns SRAM stall cycles (not total cycles), so
 * overhead is measured as stall_cycles / attention_total_cycles.
 */

#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_config.h"
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/tu_sram.h"
#include "tu_cmodel/compute/attention_engine.h"
#include "tu_cmodel/compute/softmax_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- PE Configs (SRAM scaled with PE) ---- */
typedef struct {
    const char *label;
    uint16_t pr, pc;
    uint32_t wa_sz, aa_sz, oo_sz;
} pe_t;

static pe_t pes[] = {
    {"8x8 (128K)",    8,   8,  64*1024, 32*1024, 32*1024},
    {"16x16 (256K)", 16,  16, 128*1024, 64*1024, 64*1024},
    {"32x32 (512K)", 32,  32, 256*1024, 128*1024, 128*1024},
};
#define N_PE (sizeof(pes)/sizeof(pes[0]))

/* ---- Workloads (single-Q-tile for contiguous O-buffer) ---- */
typedef struct {
    const char *label;
    uint32_t sq, skv, hd;
} wk_t;

static wk_t wks[] = {
    {"prefill-32x64",   32,  64, 64},
    {"prefill-64x64",   64,  64, 64},
    {"prefill-128x64", 128,  64, 64},
    {"prefill-32x128",  32,  64, 128},
    {"prefill-64x128",  64,  64, 128},
};
#define N_WK (sizeof(wks)/sizeof(wks[0]))

static void fill_f16(fp16_t *b, uint32_t n, unsigned s) {
    for (uint32_t i = 0; i < n; i++) {
        float v = ((float)(((i*1103515245+12345)^(s*2654435761U)) & 0x7FFFFF) / 8388608.0f) - 0.5f;
        b[i] = tu_fp32_to_fp16(v);
    }
}

int main(void) {
    printf("Softmax-After-Attention Pipeline Sweep\n");
    printf("======================================\n");
    printf("Question: What overhead does standalone softmax add to attention latency?\n\n");

    printf("%-22s %-14s %8s %10s %13s %8s\n",
           "Workload", "PE", "AttnCyc", "SMstallCy", "SMoverhead%", "SMc/e");
    printf("----------------------------------------------------------------------------------------------\n");

    for (int pi = 0; pi < (int)N_PE; pi++) {
        pe_t *p = &pes[pi];
        tu_runtime_config_t rt = tu_runtime_config_default();
        rt.pe_rows = p->pr; rt.pe_cols = p->pc;
        rt.sram_w_size = p->wa_sz; rt.sram_a_size = p->aa_sz; rt.sram_o_size = p->oo_sz;
        tu_init_with_config(&rt);

        for (int wi = 0; wi < (int)N_WK; wi++) {
            wk_t *w = &wks[wi];

            /* Check SRAM capacity for Q and KV tile */
            uint32_t q_bytes  = w->sq * w->hd * sizeof(fp16_t);
            uint32_t kv_bytes = w->skv * w->hd * sizeof(fp16_t);
            if (q_bytes > p->aa_sz || kv_bytes > p->wa_sz) {
                printf("%-22s %-14s %8s %10s %13s %8s  (SRAM skip)\n",
                       w->label, p->label, "-", "-", "-", "-");
                continue;
            }

            /* Check if output fits in O-buffer as a single tile */
            uint32_t O_bytes = w->sq * w->hd * sizeof(fp32_t);
            /* Softmax S also needs space in O-buffer */
            uint32_t S_bytes = w->sq * w->skv * sizeof(fp32_t);
            if (O_bytes + S_bytes > p->oo_sz) {
                printf("%-22s %-14s %8s %10s %13s %8s  (O-buf skip)\n",
                       w->label, p->label, "-", "-", "-", "-");
                continue;
            }

            fp16_t *Q = (fp16_t *)malloc(q_bytes);
            fp16_t *K = (fp16_t *)malloc(kv_bytes);
            fp16_t *V = (fp16_t *)malloc(kv_bytes);
            fp16_t *O = (fp16_t *)calloc(w->sq * w->hd, sizeof(fp16_t));
            if (!Q || !K || !V || !O) {
                printf("%-22s %-14s %8s (alloc fail)\n", w->label, p->label);
                free(Q); free(K); free(V); free(O);
                continue;
            }
            fill_f16(Q, w->sq * w->hd, (pi << 16) | (wi << 8) | 0);
            fill_f16(K, w->skv * w->hd, (pi << 16) | (wi << 8) | 1);
            fill_f16(V, w->skv * w->hd, (pi << 16) | (wi << 8) | 2);

            float sc = 1.0f / sqrtf((float)w->hd);

            /* Use OS dataflow (most efficient for attention) */
            tu_set_dataflow(TU_DATAFLOW_MODE_OS);

            tu_attention_desc_t d = {0};
            d.Q = Q; d.K = K; d.V = V; d.output = O;
            d.batch_size = 1; d.num_heads = 1;
            d.seq_len_q = w->sq; d.seq_len_kv = w->skv; d.head_dim = w->hd;
            d.softmax_scale = sc;
            d.mask_type = TU_ATTN_MASK_NONE;
            d.tile_m = 0; d.tile_n = 0;
            d.dataflow = TU_DATAFLOW_MODE_OS;
            tu_attention_auto_tile(&d);

            tu_attention_stats_t st = {0};
            int rc = tu_attention_execute(&d, &st);
            if (rc != 0) {
                printf("%-22s %-14s %8s (attn fail)\n", w->label, p->label);
                free(Q); free(K); free(V); free(O);
                continue;
            }

            /* Run softmax in-place on O-buffer FP32 output.
             * softmax over axis_dim=hd (last dimension), scale=1.0.
             * Returns SRAM stall cycles only (not total cycles). */
            uint64_t sm_stall = tu_softmax_2d(&g_tu.sram_o, 0,
                                               w->sq, w->hd,
                                               1.0f, true);

            uint32_t elem_count = w->sq * w->hd;
            double sm_overhead = st.total_cycles > 0 ?
                100.0 * (double)sm_stall / (double)st.total_cycles : 0.0;
            double sm_ce = elem_count > 0 ?
                (double)sm_stall / (double)elem_count : 0.0;

            printf("%-22s %-14s %8lu %10lu %12.2f%% %7.2f\n",
                   w->label, p->label,
                   (unsigned long)st.total_cycles,
                   (unsigned long)sm_stall,
                   sm_overhead,
                   sm_ce);

            free(Q); free(K); free(V); free(O);
        }

        /* Re-init between PE configs to reset internal state */
        /* (handled at top of next pi iteration) */
    }

    printf("----------------------------------------------------------------------------------------------\n");
    printf("Done. Softmax cycles measured as SRAM stall cycles.\n");
    printf("Softmax uses 2-pass algorithm: read+find_max+write per elem in pass 1,\n");
    printf("read+exp+normalize+write per elem in pass 2 = 4×elem SRAM operations.\n");
    return 0;
}
