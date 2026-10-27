/* Runtime attention tile feasibility. Metadata uses a brute-force wide oracle,
 * not the production endpoint/budget solver; only bounded real buffers execute. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/compute/attention_engine.h"
#include "tu_cmodel/compute/dataflow/dataflow_interface.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks, failures, metadata_cases, live_cases, rejects;
#define CHECK(c) do { checks++; if (!(c)) { if (failures<20) { \
    fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c); } failures++; } } while (0)

/* Wide arithmetic permits even maximal uint32 metadata without overflow. */
static bool fits(uint32_t m,uint32_t n,uint32_t d,uint32_t full,
                 uint32_t a,uint32_t w,uint32_t o) {
    if (!m || !n || !d || !full) return false;
    __uint128_t mm=m, nn=n, dd=d;
    return 2*mm*dd<=a && 4*nn*dd<=w &&
        8*mm*nn+4*mm*dd+(n<full ? 8*mm : 0)<=o;
}
static uint64_t o_bytes(uint32_t m,uint32_t n,uint32_t d,uint32_t full) {
    return (uint64_t)m*(8ull*n+4ull*d+(n<full ? 8 : 0));
}

static void metadata(uint32_t m,uint32_t n,uint32_t d,
                     uint32_t a,uint32_t w,uint32_t o) {
    /* Deliberately metadata-only capacity snapshots: never execute with these
     * fictional large capacities or placeholder buffers. Real init below. */
    g_tu.rt_cfg.sram_a_size=a; g_tu.rt_cfg.sram_w_size=w;
    g_tu.rt_cfg.sram_o_size=o;
    tu_state_t state=g_tu;
    tu_attention_desc_t x={.seq_len_q=m,.seq_len_kv=n,.head_dim=d,
        .tile_m=0x1234,.tile_n=0x5678,.softmax_scale=.5f};
    tu_attention_desc_t original=x;
    uint32_t bm=0,bn=0;
    uint64_t score=0;
    /* Bounded N fixtures are exhaustive. Large N fixtures choose d so W
     * limits the brute-force domain to <=256; no closed-form O solver. */
    uint64_t limit=d ? w/(4ull*d) : 0;
    if (limit>n) limit=n;
    CHECK(limit<=256);
    if (limit<=256 && m && n && d) {
        for (uint32_t tm=1;tm<=m && tm<=64;tm++)
            for (uint32_t tn=1;tn<=limit;tn++)
                if (fits(tm,tn,d,n,a,w,o) && (uint64_t)tm*tn>score) {
                    score=(uint64_t)tm*tn; bm=tm; bn=tn;
                }
    }
    tu_attention_auto_tile(&x);
    CHECK(x.tile_m==bm && x.tile_n==bn);
    if (!bm) rejects++;
    CHECK(!memcmp(&g_tu,&state,sizeof(state)));
    tu_attention_desc_t again=x;
    tu_attention_auto_tile(&again);
    CHECK(!memcmp(&again,&x,sizeof(x)));
    x.tile_m=original.tile_m; x.tile_n=original.tile_n;
    CHECK(!memcmp(&x,&original,sizeof(x)));
    metadata_cases++;
}

static void metadata_matrix(void) {
    tu_shutdown();
    tu_attention_desc_t uninitialized={.seq_len_q=1,.seq_len_kv=2,.head_dim=1,
        .tile_m=1,.tile_n=2};
    tu_attention_auto_tile(&uninitialized);
    CHECK(uninitialized.tile_m==0 && uninitialized.tile_n==0);
    tu_init(); tu_state_t saved=g_tu;
    const uint32_t ms[]={0,1,3,16,65,UINT32_MAX};
    const uint32_t ns[]={0,1,2,3,8,17,33};
    const uint32_t ds[]={0,1,3,8,64,UINT32_MAX};
    const uint32_t as[]={0,1,2,16,128,65536};
    const uint32_t ws[]={0,3,4,32,128,512};
    const uint32_t os[]={0,11,12,15,16,24,63,64,1024,65536};
    for (unsigned mi=0;mi<sizeof(ms)/sizeof(*ms);mi++)
    for (unsigned ni=0;ni<sizeof(ns)/sizeof(*ns);ni++)
    for (unsigned di=0;di<sizeof(ds)/sizeof(*ds);di++)
    for (unsigned ai=0;ai<sizeof(as)/sizeof(*as);ai++)
    for (unsigned wi=0;wi<sizeof(ws)/sizeof(*ws);wi++)
    for (unsigned oi=0;oi<sizeof(os)/sizeof(*os);oi++)
        metadata(ms[mi],ns[ni],ds[di],as[ai],ws[wi],os[oi]);
    /* Wide products and N at/near uint32 max. W bounds the exhaustive domain. */
    const uint32_t large[]={1,64,0x1000000u,0x40000000u,UINT32_MAX};
    for (unsigned i=0;i<sizeof(large)/sizeof(*large);i++)
    for (unsigned j=0;j<sizeof(large)/sizeof(*large);j++)
        metadata(UINT32_MAX,large[i],large[j],UINT32_MAX,
                 large[j]>=0x1000000u ? UINT32_MAX : 512,UINT32_MAX);
    /* Full-N removes row state: N=2 and streaming N=1 both fit at O=20.
     * A solver charging state to the full endpoint would miss N=2. */
    metadata(1,2,1,2,8,20);
    /* Runtime capacity exceeds defaults, but query search remains <=64. */
    metadata(256,256,64,1048576,1048576,1048576);
    g_tu=saved;
    tu_attention_auto_tile(NULL);
    CHECK(!memcmp(&g_tu,&saved,sizeof(saved)));
    tu_shutdown();
}

static void live(unsigned m,unsigned n,unsigned d,unsigned a,unsigned w,
                 unsigned o,int mode,unsigned manual_m,unsigned manual_n,
                 int mask_type,bool simple,const char *label) {
    size_t qn=(size_t)m*d, kn=(size_t)n*d;
    fp16_t *q=calloc(qn,sizeof(*q)), *k=calloc(kn,sizeof(*k));
    fp16_t *v=calloc(kn,sizeof(*v)), *out=calloc(qn+2,sizeof(*out));
    float *mask=calloc((size_t)m*n,sizeof(*mask));
    CHECK(q && k && v && out && mask);
    if (!q || !k || !v || !out || !mask) goto done;
    for (size_t i=0;i<kn;i++) v[i]=0x3400;
    memset(out,0xa5,(qn+2)*sizeof(*out));
    tu_runtime_config_t rt=tu_runtime_config_default();
    rt.sram_a_size=a; rt.sram_w_size=w; rt.sram_o_size=o;
    rt.pe_rows=mode==TU_DATAFLOW_MODE_OS ? 8 : 16;
    rt.pe_cols=mode==TU_DATAFLOW_MODE_RS ? 4 : 16;
    tu_init_with_config(&rt);
    CHECK(tu_set_dataflow(mode)==0);
    CHECK(g_tu.dataflow && (int)g_tu.dataflow->id==mode);
    tu_attention_desc_t x={.Q=q,.K=k,.V=v,.output=out+1,.batch_size=1,
        .num_heads=1,.seq_len_q=m,.seq_len_kv=n,.head_dim=d,
        .mask_type=mask_type,.mask=mask,.mask_fill=-INFINITY};
    tu_state_t saved=g_tu;
    tu_attention_auto_tile(&x);
    CHECK(!memcmp(&g_tu,&saved,sizeof(saved)));
    if (manual_m || manual_n) { x.tile_m=manual_m; x.tile_n=manual_n; }
    /* Zero fields on execute retain compile-time PE fallback compatibility. */
    unsigned tm=x.tile_m ? x.tile_m : (m<TU_PE_ROWS ? m : TU_PE_ROWS);
    unsigned tn=x.tile_n ? x.tile_n : (n<TU_PE_COLS ? n : TU_PE_COLS);
    bool expected=fits(tm,tn,d,n,a,w,o);
    tu_attention_stats_t stats, before;
    memset(&stats,0x5a,sizeof(stats)); before=stats;
    tu_attention_desc_t descriptor_before=x;
    int status=simple ? tu_attention_simple(q,k,v,out+1,m,n,d,0,false) :
                       tu_attention_execute(&x,&stats);
    CHECK(!memcmp(&x,&descriptor_before,sizeof(x)));
    CHECK(status==(expected ? 0 : -1));
    CHECK(out[0]==0xa5a5 && out[qn+1]==0xa5a5);
    if (expected && status==0) {
        for (size_t i=0;i<qn;i++) {
            float got=tu_fp16_to_fp32(out[i+1]);
            CHECK(isfinite(got) && fabsf(got-.25f)<=.0005f);
            if (n==512 && mask_type==TU_ATTN_MASK_NONE) CHECK(out[i+1]==0x3400);
        }
        uint64_t bytes=4ull*m*d+4ull*((m+tm-1)/tm)*n*d;
        if (!simple) {
            CHECK(stats.dma_bytes==bytes);
            CHECK(stats.mma_flops==4ull*m*n*d);
        }
        printf("AUTO_ROW,%s,%d,%u,%u,%u,%u,%u,%u,%u,%u,%llu,%llu,%d\n",
            label,mode,m,n,d,a,w,o,tm,tn,(unsigned long long)o_bytes(tm,tn,d,n),
            (unsigned long long)bytes,status);
    } else if (!expected) {
        CHECK(!memcmp(&stats,&before,sizeof(stats)));
        for (size_t i=0;i<qn;i++) CHECK(out[i+1]==0xa5a5);
        CHECK(g_tu.total_dma_bytes==0 && g_tu.estimated_cycles==0);
        CHECK(g_tu.sram_o.banks.reads==0 && g_tu.sram_o.banks.writes==0);
    }
    live_cases++;
 done:
    tu_shutdown(); free(q); free(k); free(v); free(out); free(mask);
}

int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IOLBF,0);
    if (argc==2 && !strcmp(argv[1],"--metadata")) metadata_matrix();
    else {
        metadata_matrix();
        for (int f=TU_DATAFLOW_MODE_WS;f<=TU_DATAFLOW_MODE_RS;f++) {
            live(16,512,64,65536,131072,65536,f,0,0,0,false,"auto-default");
            const unsigned tm[]={1,8,15,16,16};
            const unsigned tn[]={512,512,512,128,256};
            for (unsigned i=0;i<5;i++)
                live(16,512,64,65536,131072,65536,f,tm[i],tn[i],0,false,"explicit");
            live(16,512,64,65536,131072,65536,f,0,0,0,true,"simple");
            live(5,33,8,80,96,192,f,0,0,0,false,"small-ragged");
            live(5,8,8,16,256,96,f,0,0,0,false,"a-limited");
            live(5,8,8,80,32,160,f,0,0,0,false,"w-limited");
            live(65,8,8,8192,8192,8192,f,0,0,0,false,"query-cap");
            live(65,8,8,8192,8192,8192,f,65,8,0,false,"explicit-over-cap");
            live(5,8,8,80,256,480,f,4,0,0,false,"zero-n-compat");
            live(5,8,8,80,256,480,f,0,3,0,false,"zero-m-compat");
            live(5,8,8,65536,131072,65536,f,8,16,0,false,"oversized-explicit");
            live(1,2,1,2,8,20,f,0,0,0,false,"full-endpoint");
            live(1,2,1,2,8,19,f,0,0,0,false,"no-fit-o");
            live(1,2,1,2,8,19,f,0,0,0,true,"simple-no-fit");
            live(1,2,1,1,8,20,f,0,0,0,false,"no-fit-a");
            live(1,2,1,2,3,20,f,0,0,0,false,"no-fit-w");
            live(5,8,8,80,256,480,f,0,0,1,false,"causal");
            live(5,8,8,80,256,480,f,0,0,2,false,"custom");
            live(16,512,64,131072,262144,131072,f,0,0,0,false,"large-runtime");
        }
    }
    printf("attention autotile: metadata=%u rejects=%u live=%u checks=%u failures=%u\n",
        metadata_cases,rejects,live_cases,checks,failures);
    return failures ? 1 : 0;
}
