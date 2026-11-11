/* 2D wrapper count overflow: reject before scratch/data effects. */
#include "tu_cmodel/compute/normalization_engine.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; \
    fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); } } while (0)

static uint64_t run(int mode, tu_sram_region_t *s, uint32_t rows, uint32_t cols,
                    const float *g, const float *b) {
    return mode ? tu_rmsnorm_2d(s,0,rows,cols,g,1e-5f)
                : tu_layernorm_2d(s,0,rows,cols,g,b,1e-5f);
}
int main(void) {
    const struct { uint32_t rows, cols; } bad[] = {
        {0x80000001u,2}, {0x55555556u,3}, {0x40000001u,4},
        {0xffffffffu,2}, {0x80000000u,2}
    };
    const float g[4]={1,2,3,4}, b[4]={.5f,1,1.5f,2};
    unsigned invalid=0, valid=0;
    for (int mode=0; mode<2; ++mode) for (int params=0; params<2; ++params)
    for (unsigned k=0; k<sizeof bad/sizeof bad[0]; ++k) {
        tu_sram_region_t s; tu_sram_init_runtime(&s,256,"wrapper",4,4,1,2,4);
        if (!s.banks.data || !s.banks.bw_banks) return 2;
        memset(s.banks.data,0x5a,256);
        uint8_t before[256]; memcpy(before,s.banks.data,sizeof before);
        uint64_t rc=run(mode,&s,bad[k].rows,bad[k].cols,params?g:NULL,params?b:NULL);
        printf("mode=%s params=%d rows=%u cols=%u result=%s changed=%d reads=%llu writes=%llu stalls=%llu\n",
               mode ? "rms" : "layer",params,bad[k].rows,bad[k].cols,
               rc==UINT64_MAX ? "reject" : "success",
               memcmp(before,s.banks.data,sizeof before)!=0,
               (unsigned long long)s.banks.reads,(unsigned long long)s.banks.writes,
               (unsigned long long)s.banks.stall_cycles);
        CHECK(rc==UINT64_MAX);
        CHECK(memcmp(before,s.banks.data,sizeof before)==0);
        CHECK(s.banks.reads==0 && s.banks.writes==0 && s.banks.stall_cycles==0);
        ++invalid; tu_sram_destroy(&s);
    }
    for (int mode=0; mode<2; ++mode) for (int params=0;params<2;++params)
    for (uint32_t rows=1;rows<=3;++rows) for(uint32_t cols=1;cols<=4;++cols) {
        tu_sram_region_t s; tu_sram_init_runtime(&s,256,"wrapper",4,4,1,2,4);
        if (!s.banks.data || !s.banks.bw_banks) return 2;
        float input[12]; for(uint32_t i=0;i<rows*cols;++i) input[i]=(float)(i+1);
        memcpy(s.banks.data,input,rows*cols*sizeof(float));
        uint64_t rc=run(mode,&s,rows,cols,params?g:NULL,params?b:NULL);
        CHECK(rc!=UINT64_MAX);
        float out[12]; memcpy(out,s.banks.data,rows*cols*sizeof(float));
        for(uint32_t r=0;r<rows;++r) {
            double mean=0, stat=0;
            for(uint32_t j=0;j<cols;++j) mean+=input[r*cols+j];
            mean/=cols;
            for(uint32_t j=0;j<cols;++j) { double x=input[r*cols+j]; double t=mode?x:x-mean; stat+=t*t; }
            stat/=cols;
            for(uint32_t j=0;j<cols;++j) {
                double expected=(mode?input[r*cols+j]:input[r*cols+j]-mean)/sqrt(stat+1e-5f);
                if(params) expected*=g[j];
                if(params && !mode) expected+=b[j];
                CHECK(isfinite(out[r*cols+j]) && fabs(out[r*cols+j]-expected)<1e-5);
            }
        }
        ++valid; tu_sram_destroy(&s);
    }
    printf("2D wrapper count: %u invalid cases, %u valid cases, %u checks, %u failures\n",invalid,valid,checks,failures);
    return failures ? 1 : 0;
}
