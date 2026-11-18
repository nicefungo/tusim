/* Whole-row schedule/alias gate. SRAM call tracing verifies actual execution
 * order, not merely equal output. Oracle and byte-hazard simulator are separate
 * from the production helpers. TU_TEST_OLD_SCHEDULE permits baseline execution. */
#include "tu_cmodel/compute/softmax_engine.h"
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTES 1024
#define BASE 384
#define CAP 51
static unsigned cases, checks, failures, rejected;
static uint64_t fingerprint=UINT64_C(14695981039346656037);
static bool tracing;
static unsigned events;
static uint32_t addresses[2*CAP];
static char kinds[2*CAP];
#define CHECK(c) do { ++checks; if (!(c)) { \
    if (failures<12) fprintf(stderr,"FAIL case=%u line=%d: %s\n",cases,__LINE__,#c); \
    ++failures; } } while(0)

uint64_t __real_tu_sram_read(tu_sram_region_t *,uint32_t,void *);
uint64_t __real_tu_sram_write(tu_sram_region_t *,uint32_t,const void *);
static void event(uint32_t address,char kind) {
    if (!tracing) return;
    if (events<2*CAP) { addresses[events]=address; kinds[events]=kind; }
    ++events;
}
uint64_t __wrap_tu_sram_read(tu_sram_region_t *s,uint32_t a,void *p) {
    event(a,'r'); return __real_tu_sram_read(s,a,p);
}
uint64_t __wrap_tu_sram_write(tu_sram_region_t *s,uint32_t a,const void *p) {
    event(a,'w'); return __real_tu_sram_write(s,a,p);
}
static void hash(uint64_t x) {
    for (unsigned i=0;i<8;++i) { fingerprint=(fingerprint^(x&255))*UINT64_C(1099511628211); x>>=8; }
}
static uint32_t bits(float x) { uint32_t b; memcpy(&b,&x,4); return b; }
static void select_schedule(tu_softmax_desc_t *d,int schedule) {
#ifndef TU_TEST_OLD_SCHEDULE
    d->row_schedule=(tu_softmax_row_schedule_t)schedule;
#else
    (void)d; (void)schedule;
#endif
}
/* Independent hazard oracle: a later read must not touch any earlier write.
 * A complete row is staged before its first write, so within-row alias is safe. */
static bool safe(unsigned rows,unsigned cols,unsigned in,unsigned out,bool reverse) {
    unsigned char overwritten[BYTES]={0};
    for (unsigned step=0;step<rows;++step) {
        unsigned r=reverse?rows-1-step:step;
        for (unsigned b=0;b<cols*4;++b)
            if (overwritten[in+r*cols*4+b]) return false;
        for (unsigned b=0;b<cols*4;++b) overwritten[out+r*cols*4+b]=1;
    }
    return true;
}
static void oracle(const float *x,const float *mask,unsigned kind,unsigned order,
                   unsigned cols,unsigned mode,float *y,float *maximum,float *sum_out) {
    float z[17],max=-INFINITY;
    for(unsigned c=0;c<cols;++c) {
        float a=x[c];
        if(order) a=(float)((double)a*0.5);
        if(kind==1) a=(float)((double)a+mask[c]);
        if(kind==2) a=(float)((double)a*mask[c]);
        if(!order) a=(float)((double)a*0.5);
        z[c]=a; if(a>max) max=a;
    }
    long double sum=0;
    for(unsigned c=0;c<cols;++c) sum+=expl((long double)(z[c]-max));
    for(unsigned c=0;c<cols;++c)
        y[c]=mode==TU_SOFTMAX_LOG ? (float)((long double)(z[c]-max)-logl(sum))
                                : (float)(expl((long double)(z[c]-max))/sum);
    *maximum=max; *sum_out=(float)sum;
}
static void near(float actual,float expected,unsigned cols) {
    CHECK(isfinite(actual)&&isfinite(expected));
    if(isfinite(actual)&&isfinite(expected))
        CHECK(fabsl((long double)actual-expected)<=
              (4*cols+8)*FLT_EPSILON*fmaxl(fabsl(expected),0.001L));
}
static void run(unsigned rows,unsigned cols,int displacement,int schedule,
                unsigned mode,unsigned exp_policy,unsigned order,unsigned mask_kind,
                unsigned metering,bool in_place,bool whole_row,bool matrix) {
    ++cases;
    unsigned n=rows*cols,in=BASE,out=in_place?in:(unsigned)((int)in+displacement);
    bool reverse=schedule==2 || (schedule==0 && !safe(rows,cols,in,out,false));
    bool accepted=schedule>=0 && schedule<=2 && safe(rows,cols,in,out,reverse);
    float x[CAP],mask[CAP],want[CAP],got[CAP],maxima[3],sums[3],em[3],es[3];
    for(unsigned r=0;r<rows;++r) {
        maxima[r]=sums[r]=12345;
        for(unsigned c=0;c<cols;++c) {
            unsigned k=r*cols+c;
            x[k]=(float)((int)((r*7+c*3)%17)-8)*0.25f;
            mask[k]=mask_kind==1?(float)((int)((r*3+c)%7)-3)*0.5f
                                :(float)((r+c)%5)*0.5f;
        }
        oracle(x+r*cols,mask+r*cols,mask_kind,order,cols,mode,want+r*cols,em+r,es+r);
    }
    float mask_before[CAP]; memcpy(mask_before,mask,n*4);
    tu_sram_region_t s; tu_sram_init(&s,BYTES,"row-schedule");
    CHECK(tu_sram_set_word_metering(&s,(int)metering)==0);
    unsigned char *raw=tu_sram_raw_ptr(&s),before[BYTES],after[BYTES];
    memset(raw,0xa5,BYTES); tu_sram_write_bulk(&s,in,x,n*4);
    memcpy(before,raw,BYTES);
    tu_sram_region_t state=s;
    tu_sram_bw_bank_t banks[TU_SRAM_BANKS];
    memcpy(banks,s.banks.bw_banks,sizeof(banks));
    tu_softmax_desc_t d={.mode=(tu_softmax_mode_t)mode,.data_sram=&s,
        .data_offset=in,.elem_count=n,.axis_dim=whole_row?0:cols,
        .mask=mask_kind?mask:NULL,.mask_is_additive=mask_kind==1,
        .scale=0.5f,.in_place=in_place,.out_offset=in_place?UINT32_MAX:out,
        .max_out=maxima,.sum_out=sums,.exp_policy=(tu_softmax_exp_policy_t)exp_policy,
        .preprocess_order=(tu_softmax_preprocess_order_t)order};
    select_schedule(&d,schedule);
    tu_softmax_desc_t saved=d;
    events=0; tracing=true;
    bool valid=tu_softmax_validate_desc(&d);
    uint64_t result=tu_softmax_execute(&d);
    tracing=false;
    CHECK(valid==accepted);
    CHECK(memcmp(&d,&saved,sizeof(d))==0);
    CHECK(memcmp(mask,mask_before,n*4)==0);
    if(!accepted) {
        ++rejected;
        CHECK(result==UINT64_MAX); CHECK(events==0);
        CHECK(memcmp(raw,before,BYTES)==0);
        CHECK(memcmp(&s,&state,sizeof(s))==0);
        CHECK(memcmp(banks,s.banks.bw_banks,sizeof(banks))==0);
        for(unsigned r=0;r<rows;++r) CHECK(maxima[r]==12345 && sums[r]==12345);
    } else {
        CHECK(result!=UINT64_MAX); CHECK(events==2*n);
        CHECK(s.banks.reads-state.banks.reads==n);
        CHECK(s.banks.writes-state.banks.writes==n*((metering && out%4)?2:1));
        CHECK(result==s.banks.stall_cycles-state.banks.stall_cycles);
        unsigned e=0;
        for(unsigned step=0;step<rows;++step) {
            unsigned r=reverse?rows-1-step:step;
            for(unsigned c=0;c<cols;++c,++e) {
                CHECK(addresses[e]==in+(r*cols+c)*4 && kinds[e]=='r');
            }
            for(unsigned c=0;c<cols;++c,++e) {
                CHECK(addresses[e]==out+(r*cols+c)*4 && kinds[e]=='w');
            }
        }
        memcpy(got,raw+out,n*4); memcpy(after,raw,BYTES);
        CHECK(memcmp(raw,before,out)==0);
        CHECK(memcmp(raw+out+n*4,before+out+n*4,BYTES-out-n*4)==0);
        for(unsigned r=0;r<rows;++r) {
            CHECK(maxima[r]==em[r]); near(sums[r],es[r],cols);
            hash(bits(maxima[r])); hash(bits(sums[r]));
            for(unsigned c=0;c<cols;++c) {
                near(got[r*cols+c],want[r*cols+c],cols); hash(bits(got[r*cols+c]));
            }
        }
        hash(result); hash(s.banks.reads); hash(s.banks.writes);
        for(unsigned b=0;b<s.banks.bank_count;++b) {
            const tu_sram_bw_bank_t *v=&s.banks.bw_banks[b];
            hash(v->words_available); hash(v->reads_served); hash(v->writes_served);
            hash(v->read_stalls); hash(v->write_stalls); hash(v->last_refill_cycle);
        }
        /* Optional stats, repeatability and disabled metering must not alter bytes. */
        memcpy(raw,before,BYTES); tu_sram_set_bw_modeling(&s,false);
        d.max_out=NULL; d.sum_out=NULL;
        CHECK(tu_softmax_execute(&d)==0); CHECK(memcmp(raw,after,BYTES)==0);
    }
    if(matrix) {
        unsigned delta=in>out?in-out:out-in;
        unsigned union_bytes=delta<n*4?n*4+delta:2*n*4;
        printf("%u,%u,%d,%d,%u,%s,%s,%u,%u,",rows,cols,displacement,schedule,mode,
               accepted?"accept":"reject",reverse?"reverse":"forward",union_bytes,cols*4);
        if(accepted) printf("%"PRIu64",%u,%u,%.9g,%.9g\n",result,n,n,got[0],got[n-1]);
        else printf("NA,0,0,NA,NA\n");
    }
    tu_sram_destroy(&s);
}
int main(int argc,char **argv) {
    bool matrix=argc>1 && strcmp(argv[1],"--matrix")==0;
    bool legacy=argc>1 && strcmp(argv[1],"--legacy")==0;
    if(matrix) {
        puts("rows,cols,out_minus_in,schedule,mode,status,row_order,union_bytes,row_buffer_bytes,local_penalty,reads,writes,y_first,y_last");
        const int shifts[]={-48,-4,0,4,48};
        for(unsigned m=0;m<3;++m) for(int s=0;s<3;++s) for(unsigned i=0;i<5;++i)
            run(3,4,shifts[i],s,m,0,0,1,0,false,false,true);
    } else {
        const unsigned widths[]={1,2,5,17},heights[]={1,3};
        for(unsigned h=0;h<2;++h) for(unsigned w=0;w<4;++w) {
            unsigned rows=heights[h],cols=widths[w]; int bytes=(int)(rows*cols*4);
            const int shifts[]={-bytes-1,-bytes,-bytes+1,-(int)cols*4,-1,0,1,(int)cols*4,bytes-1,bytes,bytes+1};
            for(int s=0;s<(legacy?1:3);++s) for(unsigned k=0;k<11;++k) {
                if(legacy && rows>1 && shifts[k]>0 && shifts[k]<bytes) continue;
                for(unsigned m=0;m<3;++m) for(unsigned p=0;p<3;++p)
                    for(unsigned o=0;o<2;++o) for(unsigned mask=0;mask<3;++mask)
                        for(unsigned met=0;met<2;++met)
                            run(rows,cols,shifts[k],s,m,p,o,mask,met,false,false,false);
            }
        }
        for(int s=0;s<(legacy?1:3);++s) for(unsigned m=0;m<3;++m) {
            run(3,5,0,s,m,0,0,1,0,true,false,false);
            run(1,17,1,s,m,0,1,2,1,false,true,false);
        }
        if(!legacy) {
            const int bad[]={-1,3,255,0x7fffffff};
            for(unsigned i=0;i<4;++i) for(unsigned m=0;m<3;++m)
                run(3,2,32,bad[i],m,0,0,0,0,false,false,false);
        }
    }
    fprintf(stderr,"softmax-row-schedule: cases=%u checks=%u rejected=%u failures=%u fingerprint=%016"PRIx64"\n",
            cases,checks,rejected,failures,fingerprint);
    return failures?1:0;
}
