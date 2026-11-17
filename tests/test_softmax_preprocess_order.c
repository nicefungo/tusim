/* Runtime preprocessing order: independent finite FP32 stage oracle.
 * TU_TEST_OLD_ORDER omits only the new field for baseline capability checks.
 * --legacy prints a bitwise output/statistics/meter fingerprint for comparison.
 * --matrix prints every row of the small motivating trade-off matrix. */
#include "tu_cmodel/compute/softmax_engine.h"
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTES 1024
#define CAP 51
static unsigned cases, checks, failures;
static uint64_t fingerprint = UINT64_C(14695981039346656037);
static uint64_t meter[2] = {UINT64_C(14695981039346656037), UINT64_C(14695981039346656037)};
static const char *orders[] = {"mask_then_scale", "scale_then_mask"};
#define CHECK(c) do { ++checks; if (!(c)) { \
    if (failures < 12) fprintf(stderr, "FAIL case=%u line=%d: %s\n", cases, __LINE__, #c); \
    ++failures; } } while (0)

static void hash(uint64_t *h, uint64_t v) {
    for (unsigned i=0; i<8; ++i) { *h = (*h ^ (v & 255))*UINT64_C(1099511628211); v >>= 8; }
}
static uint32_t bits(float v) { uint32_t b; memcpy(&b,&v,4); return b; }
static void select_order(tu_softmax_desc_t *d, unsigned order) {
#ifndef TU_TEST_OLD_ORDER
    d->preprocess_order = (tu_softmax_preprocess_order_t)order;
#else
    (void)d; (void)order;
#endif
}
/* Individual rounded stages, no production numerical helper. */
static float rounded_scale(float x, float scale) {
    return scale == 0 ? x : (float)((double)x * scale);
}
static float rounded_mask(float x, float m, unsigned kind, float threshold) {
    float y = kind == 1 ? (float)((double)x+m) :
              kind == 2 ? (float)((double)x*m) : x;
    return threshold != 0 && y <= threshold ? -1e9f : y;
}
static void oracle(const float *x, const float *mask, unsigned kind, float scale,
                   float threshold, unsigned order, unsigned cols, unsigned mode,
                   unsigned policy, float *out, float *maximum, float *sum_out) {
    float z[17], e[17], maximum_value = -INFINITY;
    unsigned anchor=0;
    for (unsigned i=0; i<cols; ++i) {
        z[i] = order ? rounded_mask(rounded_scale(x[i],scale),mask[i],kind,threshold)
                     : rounded_scale(rounded_mask(x[i],mask[i],kind,threshold),scale);
        if (z[i]>maximum_value) { maximum_value=z[i]; anchor=i; }
    }
    long double sum=0, residual=0;
    for (unsigned i=0; i<cols; ++i) {
        float delta=z[i]-maximum_value;
        e[i] = policy==0 && delta < -87 ? 0 : (float)expl(delta);
        if (policy==1 && e[i]>0 && e[i]<FLT_MIN) e[i]=0;
        sum+=e[i]; if (i!=anchor) residual+=e[i];
    }
    for (unsigned i=0; i<cols; ++i)
        out[i] = mode==TU_SOFTMAX_LOG ? (z[i]-maximum_value)-(float)log1pl(residual)
                                    : (float)((long double)e[i]/sum);
    *maximum=maximum_value; *sum_out=(float)sum;
}
static void near_value(float got, float want, unsigned cols, unsigned mode) {
    CHECK(isfinite(got) && isfinite(want));
    if (!isfinite(got) || !isfinite(want)) return;
    if (want==0) { CHECK(got==0); return; }
    long double tol = mode==TU_SOFTMAX_ONLINE ? (2*cols+4)*FLT_EPSILON : 8e-7L;
    CHECK(fabsl((long double)got-want) <= tol*fabsl(want));
}

static void run(unsigned cols, unsigned kind, float scale, float threshold,
                unsigned order, unsigned mode, unsigned policy, bool in_place) {
    ++cases;
    unsigned n=3*cols, in=64, out=in_place ? in : 512;
    float x[CAP], masks[CAP], masks_before[CAP], want[CAP], got[CAP];
    float maxima[3], sums[3], em[3], es[3];
    unsigned char before[BYTES], after[BYTES];
    for (unsigned r=0; r<3; ++r) {
        for (unsigned c=0; c<cols; ++c) {
            unsigned k=r*cols+c;
            /* Non-binary fractions exercise stage-rounding as well as algebra. */
            x[k]=(float)((int)((c*7+r*5)%19)-9)*0.3f;
            masks[k]=kind==1 ? (float)((int)((c*3+r)%9)-4)*0.7f
                            : (float)((int)((c+r)%7)-3)*0.6f;
        }
        oracle(x+r*cols,masks+r*cols,kind,scale,threshold,order,cols,mode,policy,
               want+r*cols,em+r,es+r);
    }
    memcpy(masks_before,masks,n*4);
    tu_sram_region_t s; tu_sram_init(&s,BYTES,"order");
    unsigned char *raw=tu_sram_raw_ptr(&s);
    memset(raw,0xa5,BYTES); tu_sram_write_bulk(&s,in,x,n*4);
    memcpy(before,raw,BYTES);
    uint64_t reads=s.banks.reads, writes=s.banks.writes, stalls=s.banks.stall_cycles;
    tu_softmax_desc_t d={.mode=(tu_softmax_mode_t)mode,.data_sram=&s,
        .data_offset=in,.elem_count=n,.axis_dim=cols,.mask=kind?masks:NULL,
        .mask_is_additive=kind==1,.mask_fill=threshold,.scale=scale,
        .in_place=in_place,.out_offset=out,.max_out=maxima,.sum_out=sums,
        .exp_policy=(tu_softmax_exp_policy_t)policy};
    select_order(&d,order);
    tu_softmax_desc_t saved=d;
    CHECK(tu_softmax_validate_desc(&d));
    uint64_t result=tu_softmax_execute(&d);
    CHECK(result!=UINT64_MAX);
    CHECK(memcmp(&d,&saved,sizeof(d))==0);
    CHECK(memcmp(masks,masks_before,n*4)==0);
    CHECK(s.banks.reads-reads==n && s.banks.writes-writes==n);
    CHECK(result==s.banks.stall_cycles-stalls);
    hash(&fingerprint,result); hash(&meter[order],result);
    hash(&meter[order],s.banks.reads); hash(&meter[order],s.banks.writes);
    for (unsigned b=0; b<s.banks.bank_count; ++b) {
        tu_sram_bw_bank_t *v=&s.banks.bw_banks[b];
        hash(&meter[order],v->words_available); hash(&meter[order],v->last_refill_cycle);
        hash(&meter[order],v->reads_served); hash(&meter[order],v->writes_served);
        hash(&meter[order],v->read_stalls); hash(&meter[order],v->write_stalls);
    }
    memcpy(got,raw+out,n*4); memcpy(after,raw,BYTES);
    CHECK(memcmp(before,after,out)==0);
    CHECK(memcmp(before+out+n*4,after+out+n*4,BYTES-out-n*4)==0);
    for (unsigned r=0; r<3; ++r) {
        CHECK(maxima[r]==em[r]); near_value(sums[r],es[r],cols,mode);
        hash(&fingerprint,bits(maxima[r])); hash(&fingerprint,bits(sums[r]));
        for (unsigned c=0; c<cols; ++c) {
            near_value(got[r*cols+c],want[r*cols+c],cols,mode);
            hash(&fingerprint,bits(got[r*cols+c]));
        }
    }
    /* Determinism, optional statistics, and disabled meter are independent. */
    memcpy(raw,before,BYTES); d.max_out=NULL; d.sum_out=NULL;
    tu_sram_set_bw_modeling(&s,false);
    CHECK(tu_softmax_execute(&d)==0);
    CHECK(memcmp(raw,after,BYTES)==0);
    tu_sram_destroy(&s);
}

static void invalid(void) {
#ifndef TU_TEST_OLD_ORDER
    const int bad[]={-1,2,255,0x7fffffff};
    for (unsigned m=0;m<3;++m) for (unsigned p=0;p<3;++p)
        for (unsigned place=0;place<2;++place) for (unsigned i=0;i<4;++i) {
            ++cases;
            tu_sram_region_t s; tu_sram_init(&s,128,"bad-order");
            memset(tu_sram_raw_ptr(&s),0x5a,128);
            unsigned char before[128]; memcpy(before,tu_sram_raw_ptr(&s),128);
            tu_sram_region_t saved=s;
            tu_sram_bw_bank_t banks[TU_SRAM_BANKS];
            memcpy(banks,s.banks.bw_banks,sizeof(banks));
            float max=7,sum=9;
            tu_softmax_desc_t d={.mode=(tu_softmax_mode_t)m,.data_sram=&s,
                .elem_count=2,.in_place=place,.out_offset=64,
                .max_out=&max,.sum_out=&sum,.exp_policy=(tu_softmax_exp_policy_t)p};
            select_order(&d,(unsigned)bad[i]);
            CHECK(!tu_softmax_validate_desc(&d));
            CHECK(tu_softmax_execute(&d)==UINT64_MAX);
            CHECK(memcmp(&saved,&s,sizeof(s))==0);
            CHECK(memcmp(banks,s.banks.bw_banks,sizeof(banks))==0);
            CHECK(memcmp(before,tu_sram_raw_ptr(&s),128)==0);
            CHECK(max==7 && sum==9);
            tu_sram_destroy(&s);
        }
#endif
}

static void wrapper(void) {
    ++cases;
    tu_sram_region_t s; tu_sram_init(&s,128,"masked-wrapper");
    float x[]={2,0,0,2}, mask[]={0,-2,-2,0}, want[4], a,b,got[4];
    for (unsigned r=0;r<2;++r)
        oracle(x+2*r,mask+2*r,1,0.5f,0,0,2,0,0,want+2*r,&a,&b);
    tu_sram_write_bulk(&s,0,x,sizeof(x));
    CHECK(tu_softmax_masked(&s,0,2,2,mask,0,0.5f)!=UINT64_MAX);
    memcpy(got,tu_sram_raw_ptr(&s),sizeof(got));
    for(unsigned i=0;i<4;++i) near_value(got[i],want[i],2,0);
    tu_sram_destroy(&s);
}

static void matrix(void) {
    const float scales[]={0,0.5f,2,-0.5f,0.5f};
    const float thresholds[]={0,0,0,0,-1};
    puts("case,order,mode,scale,threshold,y0,y1,max,sum,reads,writes,local_penalty");
    for(unsigned k=0;k<5;++k) for(unsigned order=0;order<2;++order)
        for(unsigned mode=0;mode<3;++mode) {
            ++cases;
            tu_sram_region_t s; tu_sram_init(&s,128,"order-matrix");
            float x[]={2,0}, mask[]={0,-2}, want[2], got[2], em,es,maximum,sum;
            oracle(x,mask,1,scales[k],thresholds[k],order,2,mode,0,want,&em,&es);
            tu_sram_write_bulk(&s,0,x,sizeof(x));
            uint64_t r=s.banks.reads,w=s.banks.writes;
            tu_softmax_desc_t d={.mode=(tu_softmax_mode_t)mode,.data_sram=&s,
                .elem_count=2,.in_place=true,.mask=mask,.mask_is_additive=true,
                .scale=scales[k],.mask_fill=thresholds[k],.max_out=&maximum,.sum_out=&sum};
            select_order(&d,order);
            uint64_t result=tu_softmax_execute(&d);
            memcpy(got,tu_sram_raw_ptr(&s),sizeof(got));
            CHECK(result==8); CHECK(s.banks.reads-r==2 && s.banks.writes-w==2);
            CHECK(maximum==em); near_value(sum,es,2,mode);
            near_value(got[0],want[0],2,mode); near_value(got[1],want[1],2,mode);
            printf("%u,%s,%s,%.9g,%.9g,%.12g,%.12g,%.12g,%.12g,%"PRIu64",%"PRIu64",%"PRIu64"\n",
                k,orders[order],tu_softmax_mode_name((tu_softmax_mode_t)mode),scales[k],
                thresholds[k],got[0],got[1],maximum,sum,s.banks.reads-r,s.banks.writes-w,result);
            tu_sram_destroy(&s);
        }
}
int main(int argc,char **argv) {
    bool legacy=argc==2 && strcmp(argv[1],"--legacy")==0;
    if(argc==2 && strcmp(argv[1],"--matrix")==0) matrix();
    else {
        const unsigned widths[]={1,2,5,17};
        const float scales[]={0,-0.0f,1,0.5f,2,-0.5f,0.7f};
        const float thresholds[]={0,-3,-1e9f};
        for(unsigned order=0;order<(legacy?1u:2u);++order)
            for(unsigned m=0;m<3;++m) for(unsigned p=0;p<3;++p)
            for(unsigned wi=0;wi<4;++wi) for(unsigned si=0;si<7;++si)
            for(unsigned kind=0;kind<3;++kind) for(unsigned t=0;t<3;++t)
            for(unsigned place=0;place<2;++place)
                run(widths[wi],kind,scales[si],thresholds[t],order,m,p,place);
        if(!legacy) { invalid(); wrapper(); CHECK(meter[0]==meter[1]); }
        printf("output_stats_stalls_fnv64=%016"PRIx64" meter0=%016"PRIx64" meter1=%016"PRIx64"\n",
               fingerprint,meter[0],meter[1]);
    }
    printf("preprocess-order cases=%u checks=%u failures=%u\n",cases,checks,failures);
    return failures?1:0;
}
