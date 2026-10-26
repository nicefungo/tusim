/* Cross-KV global normalization, bounded independent FP64 oracle. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/compute/attention_engine.h"
#include "tu_cmodel/compute/dataflow/dataflow_interface.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks, failures, cases;
static double max_error, max_partition_error;
#ifdef TU_ATTN_TEST_ALLOC
/* Arm only after tu_init; the streaming execution path uses malloc only for
 * its two final output staging buffers. Do not fail unrelated initialization. */
static unsigned fail_malloc, malloc_count;
void *__real_malloc(size_t);
void *__wrap_malloc(size_t bytes) {
    if(fail_malloc && ++malloc_count==fail_malloc) return NULL;
    return __real_malloc(bytes);
}
#endif
#define CHECK(c) do { checks++; if (!(c)) { if (failures < 16) { \
    fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c); } failures++; } } while (0)

static void run_case(unsigned n, unsigned d, unsigned tm, unsigned tn,
                     int mode, int mask_type, int pattern) {
    const unsigned m=5, heads=4;
    size_t qn=heads*m*d, kn=heads*n*d, mn=heads*m*n;
    fp16_t *q=calloc(qn,2), *k=calloc(kn,2), *v=calloc(kn,2);
    fp16_t *o=calloc(qn+2,2), *whole=calloc(qn,2);
    float *mask=calloc(mn,sizeof(float));
    CHECK(q && k && v && o && whole && mask);
    if (!q || !k || !v || !o || !whole || !mask) goto done;
    for (size_t i=0;i<qn;i++) q[i]=tu_fp32_to_fp16(pattern==0 ? 0 :
        pattern>=2 ? 1 : ((int)(i*7%17)-8)/16.f);
    for (size_t i=0;i<kn;i++) {
        /* Monotone increasing/decreasing scores exercise changing maxima,
         * including >87 gaps and underflowing old contribution. */
        float x=pattern==0 ? 0 : pattern>=2 ?
            (pattern==2 ? (float)((i/d)%n) : -(float)((i/d)%n))*4 :
            ((int)(i*11%19)-9)/16.f;
        k[i]=tu_fp32_to_fp16(x);
        v[i]=tu_fp32_to_fp16(pattern==0 ? .25f : ((int)(i*13%23)-11)/16.f);
    }
    for (unsigned h=0;h<heads;h++) for(unsigned r=0;r<m;r++)
        for(unsigned j=0;j<n;j++) {
            mask[(h*m+r)*n+j]=((int)((h*3+r*5+j*7)%9)-4)*.25f;
            if (mask_type==TU_ATTN_MASK_CUSTOM && (j+r+h)%5==0)
                mask[(h*m+r)*n+j]=-INFINITY;
        }
    /* Custom first row is entirely masked, second starts with empty tiles. */
    if (mask_type==TU_ATTN_MASK_CUSTOM) for(unsigned h=0;h<heads;h++) {
        for(unsigned j=0;j<n;j++) mask[(h*m)*n+j]=-INFINITY;
        for(unsigned j=0;j+1<n;j++) mask[(h*m+1)*n+j]=-INFINITY;
    }
    o[0]=0x1234; o[qn+1]=0x5678;
    tu_init(); CHECK(tu_set_dataflow(mode)==0);
    CHECK(g_tu.dataflow && (int)g_tu.dataflow->id==mode);
    tu_attention_desc_t desc={.Q=q,.K=k,.V=v,.output=o+1,
        .batch_size=2,.num_heads=2,.seq_len_q=m,.seq_len_kv=n,.head_dim=d,
        .softmax_scale=.5f,.mask_type=mask_type,.mask=mask,.mask_fill=-INFINITY,
        .tile_m=tm,.tile_n=tn};
    tu_attention_stats_t stats={0};
    CHECK(tu_attention_execute(&desc,&stats)==0);
    CHECK(o[0]==0x1234 && o[qn+1]==0x5678);
    unsigned qtiles=(m+tm-1)/tm;
    CHECK(stats.dma_bytes==(uint64_t)heads*(2*m*d*2 + 2*qtiles*n*d*2));
    CHECK(stats.mma_flops==(uint64_t)4*heads*m*n*d);
    double err=0;
    for(unsigned h=0;h<heads;h++) for(unsigned r=0;r<m;r++) {
        double s[64], hi=-INFINITY, sum=0;
        for(unsigned j=0;j<n;j++) {
            double dot=0;
            for(unsigned a=0;a<d;a++) dot+=(double)tu_fp16_to_fp32(q[(h*m+r)*d+a])*
                tu_fp16_to_fp32(k[(h*n+j)*d+a]);
            s[j]=dot*.5;
            if(mask_type==TU_ATTN_MASK_CAUSAL && j>r) s[j]=-INFINITY;
            if(mask_type==TU_ATTN_MASK_CUSTOM) s[j]+=mask[(h*m+r)*n+j];
            if(s[j]>hi) hi=s[j];
        }
        for(unsigned j=0;j<n;j++) { s[j]=hi==-INFINITY ? 0 : exp(s[j]-hi); sum+=s[j]; }
        for(unsigned a=0;a<d;a++) {
            double ref=0;
            if(sum) for(unsigned j=0;j<n;j++) ref+=s[j]/sum*tu_fp16_to_fp32(v[(h*n+j)*d+a]);
            double got=tu_fp16_to_fp32(o[1+(h*m+r)*d+a]);
            double e=fabs(got-ref);
            CHECK(isfinite(got) && e<=.002);
            if(e>err) err=e;
            if(pattern==0 && mask_type==TU_ATTN_MASK_NONE)
                CHECK(o[1+(h*m+r)*d+a]==0x3400);
            if(mask_type==TU_ATTN_MASK_CUSTOM && r==0)
                CHECK(o[1+(h*m+r)*d+a]==0);
        }
    }
    if(err>max_error) max_error=err;
    /* Mathematical partition invariance uses a bounded tolerance: FP16
     * normalized P versus FP16 unnormalized exponentials need not be bit equal. */
    {
        tu_shutdown(); tu_init(); CHECK(tu_set_dataflow(mode)==0);
        desc.output=whole; desc.tile_n=n;
        CHECK(tu_attention_execute(&desc,NULL)==0);
        for(size_t i=0;i<qn;i++) {
            double e=fabs(tu_fp16_to_fp32(o[i+1])-tu_fp16_to_fp32(whole[i]));
            CHECK(isfinite(e) && e<=.002);
            if(e>max_partition_error) max_partition_error=e;
        }
    }
    printf("KV_ROW,%u,%u,%u,%u,%d,%d,%d,%llu,%llu,%llu,%.9g\n",n,d,tm,tn,
        mode,mask_type,pattern,(unsigned long long)stats.dma_bytes,
        (unsigned long long)stats.compute_cycles,(unsigned long long)stats.dma_cycles,err);
    cases++;
done:
    tu_shutdown(); free(q); free(k); free(v); free(o); free(whole); free(mask);
}

static void admission(void) {
    fp16_t q[4]={0}, kv[8], out[4], before[4];
    for(unsigned i=0;i<8;i++) kv[i]=0x3400;
    tu_attention_desc_t good={.Q=q,.K=kv,.V=kv,.output=out,.batch_size=1,
        .num_heads=1,.seq_len_q=1,.seq_len_kv=2,.head_dim=4,
        .tile_m=32,.tile_n=1};
    /* 2*S+O+8*tm = 256+512+256 = 1024. Row state is not free. */
    tu_runtime_config_t rt=tu_runtime_config_default(); rt.sram_o_size=1024;
    tu_init_with_config(&rt);
    CHECK(tu_attention_execute(&good,NULL)==0);
    for(unsigned i=0;i<4;i++) CHECK(out[i]==0x3400);
    tu_shutdown();
    for(unsigned c=0;c<9;c++) {
        rt.sram_o_size=c==0 ? 1023 : 1024;
        tu_init_with_config(&rt);
        tu_attention_desc_t bad=good;
        switch(c) {
        case 0: break;
        case 1: bad.softmax_scale=NAN; break;
        case 2: bad.softmax_scale=INFINITY; break;
        case 3: bad.mask_type=(tu_attn_mask_type_t)99; break;
        case 4: bad.mask_type=TU_ATTN_MASK_CUSTOM; break;
        case 5: bad.mask_type=TU_ATTN_MASK_CAUSAL; bad.mask_fill=NAN; break;
        case 6: bad.mask_type=TU_ATTN_MASK_CAUSAL; bad.mask_fill=INFINITY; break;
        case 7: bad.tile_m=UINT32_MAX; break;
        case 8: bad.tile_n=UINT32_MAX; break;
        }
        memset(out,0xa5,sizeof(out)); memcpy(before,out,sizeof(out));
        tu_attention_stats_t stats, saved;
        memset(&stats,0x5a,sizeof(stats)); saved=stats;
        CHECK(!tu_attention_validate_desc(&bad) || c==0 || c>=7);
        CHECK(tu_attention_execute(&bad,&stats)==-1);
        CHECK(!memcmp(out,before,sizeof(out)) && !memcmp(&stats,&saved,sizeof(stats)));
        CHECK(g_tu.total_dma_bytes==0);
        CHECK(g_tu.sram_o.banks.reads==0 && g_tu.sram_o.banks.writes==0);
        tu_shutdown();
    }
    /* Runtime nonfinite arithmetic errors are not transactional rollbacks:
     * output of this first tile/stats stays untouched, SRAM may have changed. */
    for(unsigned c=0;c<4;c++) {
        tu_init();
        tu_attention_desc_t bad=good; bad.tile_m=1;
        float mask[2]={0,0};
        bad.mask_type=TU_ATTN_MASK_CUSTOM; bad.mask=mask;
        if(c==0) mask[0]=NAN;
        if(c==1) mask[0]=INFINITY;
        if(c==2) kv[0]=0x7e00;
        if(c==3) kv[0]=0x7c00;
        memset(out,0xa5,sizeof(out)); memcpy(before,out,sizeof(out));
        tu_attention_stats_t stats, saved;
        memset(&stats,0x5a,sizeof(stats)); saved=stats;
        CHECK(tu_attention_execute(&bad,&stats)==-1);
        CHECK(!memcmp(out,before,sizeof(out)) && !memcmp(&stats,&saved,sizeof(stats)));
        CHECK(g_tu.total_dma_bytes>0);
        kv[0]=0x3400; tu_shutdown();
    }
    /* Finite FP32 accumulator can overflow only when exported to FP16.
     * Uniform P=FP16(1/27) has a slightly >1 represented row sum. */
    {
        fp16_t query=0, keys[27]={0}, values[27], result=0x1234;
        for(unsigned i=0;i<27;i++) values[i]=0x7bff;
        tu_init();
        tu_attention_desc_t d={.Q=&query,.K=keys,.V=values,.output=&result,
            .batch_size=1,.num_heads=1,.seq_len_q=1,.seq_len_kv=27,.head_dim=1,
            .tile_m=1,.tile_n=27};
        CHECK(tu_attention_execute(&d,NULL)==-1);
        CHECK(result==0x1234);
        tu_shutdown();
    }
}

/* Diagnostic only: remains RED until the separate auto-tiler audit/fix.
 * It must not be folded into passing cross-KV acceptance. */
static int auto_probe(void) {
    fp16_t *q=calloc(16*64,2), *kv=calloc(512*64,2), *out=calloc(16*64,2);
    if(!q || !kv || !out) { free(q); free(kv); free(out); return 2; }
    for(unsigned i=0;i<512*64;i++) kv[i]=0x3400;
    tu_attention_desc_t d={.Q=q,.K=kv,.V=kv,.output=out,.batch_size=1,
        .num_heads=1,.seq_len_q=16,.seq_len_kv=512,.head_dim=64};
    tu_init(); tu_attention_auto_tile(&d);
    unsigned tm=d.tile_m,tn=d.tile_n;
    int automatic=tu_attention_execute(&d,NULL);
    tu_shutdown(); tu_init(); d.tile_m=8; d.tile_n=512;
    int manual=tu_attention_execute(&d,NULL);
    printf("AUTO_PROBE chosen=%ux%u auto_status=%d manual=8x512 manual_status=%d\n",
        tm,tn,automatic,manual);
    for(unsigned i=0;i<16*64;i++) if(out[i]!=0x3400) manual=-1;
    tu_shutdown(); free(q); free(kv); free(out);
    return automatic==0 && manual==0 ? 0 : 1;
}

int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IOLBF,0); /* preserve complete rows beside stderr logs */
#ifdef TU_ATTN_TEST_ALLOC
    if(argc==2 && !strcmp(argv[1],"--alloc-probe")) {
        for(unsigned f=1;f<=2;f++) {
            fp16_t q[4]={0}, kv[8], out[4], before[4];
            for(unsigned i=0;i<8;i++) kv[i]=0x3400;
            memset(out,0xa5,sizeof(out)); memcpy(before,out,sizeof(out));
            tu_init();
            tu_attention_desc_t d={.Q=q,.K=kv,.V=kv,.output=out,.batch_size=1,
                .num_heads=1,.seq_len_q=1,.seq_len_kv=2,.head_dim=4,
                .tile_m=1,.tile_n=1};
            tu_attention_stats_t stats,saved;
            memset(&stats,0x5a,sizeof(stats)); saved=stats;
            malloc_count=0; fail_malloc=f;
            int rc=tu_attention_execute(&d,&stats);
            fail_malloc=0;
            CHECK(rc==-1 && malloc_count==2);
            CHECK(!memcmp(out,before,sizeof(out)) && !memcmp(&stats,&saved,sizeof(stats)));
            tu_shutdown();
        }
        printf("allocation-failure gates: checks=%u failures=%u\n",checks,failures);
        return failures ? 1 : 0;
    }
#endif
    if(argc==2 && !strcmp(argv[1],"--auto-probe")) return auto_probe();
    if(argc==2 && !strcmp(argv[1],"--smoke")) {
        run_case(4,1,3,2,TU_DATAFLOW_MODE_WS,TU_ATTN_MASK_NONE,0);
    } else if(argc==2 && !strcmp(argv[1],"--reuse-sweep")) {
        for(int f=TU_DATAFLOW_MODE_WS;f<=TU_DATAFLOW_MODE_RS;f++) {
            run_case(33,8,1,3,f,TU_ATTN_MASK_NONE,1);
            run_case(33,8,3,3,f,TU_ATTN_MASK_NONE,1);
        }
    } else if(argc==1) {
        const unsigned ns[]={1,4,5,8,17,33};
        const unsigned ds[]={1,3,8};
        const unsigned ts[]={1,2,3,7,33};
        const int modes[]={TU_DATAFLOW_MODE_WS,TU_DATAFLOW_MODE_OS,TU_DATAFLOW_MODE_RS};
        for(unsigned f=0;f<3;f++) for(unsigned ni=0;ni<6;ni++)
            for(unsigned di=0;di<3;di++) for(unsigned ti=0;ti<5;ti++)
                for(int mask=0;mask<3;mask++) for(int p=0;p<4;p++) {
                    if(p==0 && ((ns[ni]&(ns[ni]-1)) || mask)) continue;
                    run_case(ns[ni],ds[di],ti%2 ? 1 : 3,ts[ti],modes[f],mask,p);
                }
        admission();
    } else return 2;
    printf("attention KV: cases=%u checks=%u failures=%u max_error=%.9g max_partition_error=%.9g\n",
        cases,checks,failures,max_error,max_partition_error);
    return failures ? 1 : 0;
}
