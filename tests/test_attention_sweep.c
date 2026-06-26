/*
 * Attention Engine Sweep: Tile Size × PE Array × Head Dim
 * ========================================================
 * Answers: "What PE array size and tile dimensions optimize attention
 * throughput for typical LLM configurations?"
 *
 * Uses tu_init_with_config() to reconfigure global state, then calls
 * attention engine directly. Measures cycles via tu_attention_stats_t.
 */

#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_config.h"
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/compute/attention_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { const char *label; uint16_t pr, pc; uint32_t wa_sz, aa_sz, oo_sz; } pe_t;
static pe_t pes[] = {
    {"8×8 (128K)",      8,   8,  64*1024, 32*1024, 32*1024},
    {"16×16 (256K)",   16,  16, 128*1024, 64*1024, 64*1024},
    {"32×32 (512K)",   32,  32, 256*1024, 128*1024, 128*1024},
    {"64×4 (256K)",    64,   4, 128*1024, 64*1024, 64*1024},
    {"16×32 (256K)",   16,  32, 128*1024, 64*1024, 64*1024},
};
#define N_PE (sizeof(pes)/sizeof(pes[0]))

typedef struct { const char *label; uint32_t sq, skv, hd; bool causal; } wk_t;
static wk_t wks[] = {
    {"prefill-128×64",       128, 128, 64, false},
    {"prefill-128×128",      128, 128, 128, false},
    {"decode-1×512-64",        1, 512, 64, true},
    {"decode-1×512-128",       1, 512, 128, true},
    {"decode-1×2048-64",       1, 2048, 64, true},
    {"batch-32×256-128",      32, 256, 128, false},
    {"long-ctx-512×128",     512, 512, 128, true},
};
#define N_WK (sizeof(wks)/sizeof(wks[0]))

static float randf(unsigned *s) { *s=*s*1103515245+12345; return ((float)((*s>>16)&0x7FFF)/32768.f)*2.f-1.f; }

static void fill_f16(fp16_t *b, uint32_t n, unsigned s) {
    for(uint32_t i=0;i<n;i++) b[i]=tu_fp32_to_fp16(randf(&s));
}

int main(void) {
    printf("Attention Engine Sweep\n======================\n");
    printf("%-22s %-16s %-3s %8s %10s %10s %8s %10s\n",
           "Workload", "PE", "DF", "Cycles", "Compute", "DMA", "Util%", "MFLOPs");
    printf("----------------------------------------------------------------------------------------\n");

    int dfs[]={0,1,2}; /* WS, OS, RS */
    const char *dn[]={"WS","OS","RS"};

    for(int pi=0;pi<(int)N_PE;pi++){
        pe_t *p=&pes[pi];
        tu_runtime_config_t rt=tu_runtime_config_default();
        rt.pe_rows=p->pr; rt.pe_cols=p->pc;
        rt.sram_w_size=p->wa_sz; rt.sram_a_size=p->aa_sz; rt.sram_o_size=p->oo_sz;
        tu_init_with_config(&rt);

        for(int wi=0;wi<(int)N_WK;wi++){
            wk_t *w=&wks[wi];
            uint32_t qb=w->sq*w->hd*sizeof(fp16_t);
            uint32_t kvb=w->skv*w->hd*sizeof(fp16_t);
            /* skip if Q or KV tile can't fit in A/W SRAM */
            if(w->sq*w->hd*2 > p->aa_sz) continue;
            if(w->skv*w->hd*2 > p->wa_sz) continue;

            fp16_t *Q=(fp16_t*)malloc(qb);
            fp16_t *K=(fp16_t*)malloc(kvb);
            fp16_t *V=(fp16_t*)malloc(kvb);
            fp16_t *O=(fp16_t*)calloc(w->sq*w->hd,sizeof(fp16_t));
            fill_f16(Q,w->sq*w->hd,(pi<<16)|(wi<<8)|0);
            fill_f16(K,w->skv*w->hd,(pi<<16)|(wi<<8)|1);
            fill_f16(V,w->skv*w->hd,(pi<<16)|(wi<<8)|2);

            float sc=1.f/sqrtf((float)w->hd);

            for(int df=0;df<3;df++){
                tu_set_dataflow(dfs[df]);
                tu_attention_desc_t d={0};
                d.Q=Q; d.K=K; d.V=V; d.output=O;
                d.batch_size=1; d.num_heads=1;
                d.seq_len_q=w->sq; d.seq_len_kv=w->skv; d.head_dim=w->hd;
                d.softmax_scale=sc;
                d.mask_type=w->causal?TU_ATTN_MASK_CAUSAL:TU_ATTN_MASK_NONE;
                d.mask_fill=-1e9f;
                d.tile_m=0; d.tile_n=0;
                d.dataflow=dfs[df];
                tu_attention_auto_tile(&d);

                tu_attention_stats_t st={0};
                tu_attention_execute(&d,&st);

                float u=st.total_cycles>0?100.f*(float)st.compute_cycles/(float)st.total_cycles:0.f;
                printf("%-22s %-16s %-3s %8lu %10lu %10lu %7.1f%% %10.2f\n",
                       w->label,p->label,dn[df],
                       (unsigned long)st.total_cycles,(unsigned long)st.compute_cycles,
                       (unsigned long)st.dma_cycles,u,(double)st.mma_flops/1e6);
            }
            free(Q);free(K);free(V);free(O);
        }
    }
    printf("----------------------------------------------------------------------------------------\nSweep complete.\n");
    return 0;
}
