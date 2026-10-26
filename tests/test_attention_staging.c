/* Attention FP16 staging: public single-KV-tile execution, independent FP64
 * softmax/output oracle, exact uniform controls, and admission immutability.
 * --kv-probe is an intentionally failing diagnostic, NOT an acceptance gate.
 */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/compute/attention_engine.h"
#include "tu_cmodel/compute/dataflow/dataflow_interface.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned cases, checks, failures;
static double max_error;
static int kv_probe;
#define CHECK(c) do { checks++; if (!(c)) { \
    if (failures < 12) fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); \
    failures++; } } while (0)

static void run_case(unsigned m, unsigned n, unsigned d, unsigned tm,
                     unsigned heads, int mode, int uniform, int causal) {
    size_t qn = heads * m * d, kn = heads * n * d;
    fp16_t *q = calloc(qn, sizeof(*q)), *k = calloc(kn, sizeof(*k));
    fp16_t *v = calloc(kn, sizeof(*v)), *o = calloc(qn + 2, sizeof(*o));
    CHECK(q && k && v && o);
    if (!q || !k || !v || !o) goto done;
    tu_init();
    CHECK(tu_set_dataflow(mode) == 0);
    CHECK(g_tu.dataflow && (int)g_tu.dataflow->id == mode);
    for (size_t i = 0; i < qn; i++)
        q[i] = tu_fp32_to_fp16(uniform ? 0.0f : ((int)(i * 7 % 17) - 8) / 16.0f);
    for (size_t i = 0; i < kn; i++) {
        k[i] = tu_fp32_to_fp16(uniform ? 0.0f : ((int)(i * 11 % 19) - 9) / 16.0f);
        v[i] = tu_fp32_to_fp16(uniform ? 0.25f : ((int)(i * 13 % 23) - 11) / 16.0f);
    }
    for (size_t i = 0; i < qn + 2; i++) o[i] = 0x7e00;
    o[0] = 0x1234; o[qn + 1] = 0x5678;
    tu_attention_desc_t desc = {
        .Q=q, .K=k, .V=v, .output=o+1, .batch_size=1, .num_heads=heads,
        .seq_len_q=m, .seq_len_kv=n, .head_dim=d, .softmax_scale=0.5f,
        .mask_type=causal ? TU_ATTN_MASK_CAUSAL : TU_ATTN_MASK_NONE,
        .tile_m=tm, .tile_n=kv_probe ? n/2 : n,
    };
    tu_attention_stats_t stats = {0};
    CHECK(tu_attention_execute(&desc, &stats) == 0);
    CHECK(o[0] == 0x1234 && o[qn + 1] == 0x5678);
    CHECK(stats.dma_bytes > 0 && stats.mma_flops > 0);
    double case_error = 0;
    for (unsigned h = 0; h < heads; h++) for (unsigned r = 0; r < m; r++) {
        double scores[64], probs[64], hi = -INFINITY, sum = 0;
        for (unsigned j = 0; j < n; j++) {
            double score = 0;
            for (unsigned a = 0; a < d; a++)
                score += (double)tu_fp16_to_fp32(q[(h*m+r)*d+a]) *
                                  tu_fp16_to_fp32(k[(h*n+j)*d+a]);
            scores[j] = causal && j > r ? -1e9 : score * 0.5;
            if (scores[j] > hi) hi = scores[j];
        }
        for (unsigned j = 0; j < n; j++) sum += (probs[j] = exp(scores[j] - hi));
        for (unsigned a = 0; a < d; a++) {
            double expected = 0;
            for (unsigned j = 0; j < n; j++)
                expected += probs[j] / sum * tu_fp16_to_fp32(v[(h*n+j)*d+a]);
            float actual = tu_fp16_to_fp32(o[1+(h*m+r)*d+a]);
            double error = fabs(actual - expected);
            /* Inputs/V are bounded by 11/16, N<=33. FP16 probability plus
             * output rounding explains <=0.002 absolute error; this is not
             * a general precision-format or arbitrary-magnitude tolerance. */
            CHECK(isfinite(actual) && error <= 0.002);
            if (!isfinite(error)) case_error = INFINITY;
            else if (error > case_error) case_error = error;
            if (isfinite(error) && error > max_error) max_error = error;
            if (uniform) CHECK(o[1+(h*m+r)*d+a] == 0x3400); /* exact 0.25 */
        }
    }
    printf("ROW,%u,%u,%u,%u,%u,%d,%d,%d,%llu,%llu,%llu,%.9g\n",
        m,n,d,tm,heads,mode,uniform,causal,
        (unsigned long long)stats.dma_bytes,
        (unsigned long long)stats.compute_cycles,
        (unsigned long long)stats.dma_cycles,case_error);
    if (kv_probe) printf("KV_PROBE first_output=%.9g expected=0.25 tile_n=%u\n",
                        tu_fp16_to_fp32(o[1]),desc.tile_n);
    cases++;
done:
    tu_shutdown();
    free(q); free(k); free(v); free(o);
}

static void admission(void) {
    fp16_t x[64], out[64], before[64];
    for (unsigned i=0; i<64; i++) x[i]=0x3400;
    tu_init();
    /* 2*S + O = 2*(64*96*4) + 64*64*4 = 65536 exactly. Actual
     * tensor is small; configured scratch geometry is independently tested. */
    tu_attention_desc_t d = {
        .Q=x,.K=x,.V=x,.output=out,.batch_size=1,.num_heads=1,
        .seq_len_q=1,.seq_len_kv=1,.head_dim=64,.tile_m=64,.tile_n=96,
    };
    CHECK(tu_attention_execute(&d,NULL)==0);
    tu_shutdown();
    for (unsigned c=0; c<8; c++) {
        tu_init();
        tu_attention_desc_t bad=d;
        switch(c) {
        case 0: bad.tile_n=97; break; /* old packed-P allocation admitted this */
        case 1: bad.tile_m=UINT32_MAX; break;
        case 2: bad.tile_n=UINT32_MAX; break;
        case 3: bad.head_dim=UINT32_MAX; break;
        case 4: bad.seq_len_q=0; break;
        case 5: bad.seq_len_kv=0; break;
        case 6: bad.head_dim=0; break;
        case 7: bad.softmax_scale=-1; break;
        }
        memset(out,0xa5,sizeof(out)); memcpy(before,out,sizeof(out));
        tu_attention_stats_t stats, saved;
        memset(&stats,0x5a,sizeof(stats)); saved=stats;
        uint64_t bytes=g_tu.total_dma_bytes;
        uint64_t a=g_tu.sram_a.banks.reads+g_tu.sram_a.banks.writes;
        uint64_t w=g_tu.sram_w.banks.reads+g_tu.sram_w.banks.writes;
        uint64_t o=g_tu.sram_o.banks.reads+g_tu.sram_o.banks.writes;
        CHECK(tu_attention_execute(&bad,&stats)==-1);
        CHECK(memcmp(out,before,sizeof(out))==0);
        CHECK(memcmp(&stats,&saved,sizeof(stats))==0);
        CHECK(g_tu.total_dma_bytes==bytes);
        CHECK(g_tu.sram_a.banks.reads+g_tu.sram_a.banks.writes==a);
        CHECK(g_tu.sram_w.banks.reads+g_tu.sram_w.banks.writes==w);
        CHECK(g_tu.sram_o.banks.reads+g_tu.sram_o.banks.writes==o);
        tu_shutdown();
    }
    /* Exact W-buffer tail: K and KT use all 1024 B. No two-byte headroom. */
    tu_runtime_config_t rt=tu_runtime_config_default();
    rt.sram_w_size=1024;
    tu_init_with_config(&rt);
    fp16_t keys[256], query[16], result[16];
    for(unsigned i=0;i<256;i++) keys[i]=0x3400;
    for(unsigned i=0;i<16;i++) query[i]=0;
    d=(tu_attention_desc_t){.Q=query,.K=keys,.V=keys,.output=result,
        .batch_size=1,.num_heads=1,.seq_len_q=1,.seq_len_kv=16,.head_dim=16,
        .tile_m=1,.tile_n=16};
    CHECK(tu_attention_execute(&d,NULL)==0);
    for(unsigned i=0;i<16;i++) CHECK(result[i]==0x3400);
    tu_shutdown();
}

int main(int argc, char **argv) {
    if (argc==2 && !strcmp(argv[1],"--smoke")) {
        run_case(1,1,1,1,1,TU_DATAFLOW_MODE_WS,1,0);
    } else if (argc==2 && !strcmp(argv[1],"--kv-probe")) {
        kv_probe=1;
        run_case(1,4,1,1,1,TU_DATAFLOW_MODE_WS,1,0);
    } else if (argc==1 || (argc==2 && !strcmp(argv[1],"--matrix-only"))) {
        const unsigned ns[]={1,2,3,4,5,8,17,33};
        const unsigned ds[]={1,3,8,17};
        const int modes[]={TU_DATAFLOW_MODE_WS,TU_DATAFLOW_MODE_OS,TU_DATAFLOW_MODE_RS};
        for(unsigned f=0;f<3;f++) for(unsigned ni=0;ni<8;ni++)
            for(unsigned di=0;di<4;di++) for(unsigned u=0;u<2;u++)
                for(unsigned c=0;c<2;c++) {
                    if(u && (c || (ns[ni] & (ns[ni]-1)))) continue;
                    for(unsigned tm=1;tm<=5;tm+=2)
                        run_case(5,ns[ni],ds[di],tm,2,modes[f],u,c);
                }
        if (argc==1) admission();
    } else {
        fprintf(stderr,"usage: %s [--smoke|--matrix-only|--kv-probe]\n",argv[0]);
        return 2;
    }
    printf("attention staging: cases=%u checks=%u failures=%u max_abs_error=%.9g\n",
           cases,checks,failures,max_error);
    return failures ? 1 : 0;
}
