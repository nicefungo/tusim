/* Wide independent shape/admission oracle. Never dereference generated addresses.
 * Padding sentinel is legacy iterator metadata, NOT executable DMA zero-fill. */
#include "tu_cmodel/memory/address_generator.h"
#include <stdio.h>
#include <string.h>

static unsigned cases, rejected, enumerated;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL case %u line %d: %s\n",cases,__LINE__,#c); return 1; \
} } while (0)

static int check(tu_agen_im2col_t cfg, uint32_t base, int verbose) {
    cases++;
    __uint128_t kh = cfg.kernel_h ? (__uint128_t)(cfg.kernel_h-1)*cfg.dilation_h+1 : 0;
    __uint128_t kw = cfg.kernel_w ? (__uint128_t)(cfg.kernel_w-1)*cfg.dilation_w+1 : 0;
    __uint128_t h = (__uint128_t)cfg.input_h+2*(__uint128_t)cfg.pad_h;
    __uint128_t w = (__uint128_t)cfg.input_w+2*(__uint128_t)cfg.pad_w;
    int dims_ok = cfg.kernel_h && cfg.kernel_w && cfg.stride_h && cfg.stride_w &&
                  cfg.dilation_h && cfg.dilation_w;
    __uint128_t oh=0, ow=0;
    if (dims_ok) {
        if (cfg.input_h && h>=kh) oh=(h-kh)/cfg.stride_h+1;
        if (cfg.input_w && w>=kw) ow=(w-kw)/cfg.stride_w+1;
        dims_ok = oh<=UINT32_MAX && ow<=UINT32_MAX;
    }
    tu_agen_im2col_t dims=cfg;
    dims.output_h=dims.output_w=123;
    tu_agen_im2col_dims(&dims);
    CHECK(dims.output_h == (dims_ok ? (uint32_t)oh : 0));
    CHECK(dims.output_w == (dims_ok ? (uint32_t)ow : 0));
    uint32_t size=cfg.elem_size ? cfg.elem_size : 4;
    /* Five u32 factors can exceed even 128 bits. Saturate only after checking
     * the empty axes; the oracle needs an exact count only below the ABI cap. */
    __uint128_t n=0;
    if (dims_ok && oh && ow && cfg.input_c) {
        n=oh*ow*cfg.kernel_h*cfg.kernel_w;
        if (n>UINT32_MAX) n=(__uint128_t)UINT32_MAX+1;
        else n*=cfg.input_c;
    }
    __uint128_t end=base+(__uint128_t)cfg.input_h*cfg.input_w*cfg.input_c*size;
    int valid = dims_ok && (!n || (n<=UINT32_MAX && end<=UINT32_MAX));
    tu_agen_iterator_t it;
    memset(&it,0xa5,sizeof(it));
    cfg.output_h=777;
    cfg.output_w=999; /* Derived cached dimensions are never trusted. */
    int rc=tu_agen_iterator_init(&it,TU_AGEN_MODE_IM2COL,base,&cfg);
    if ((rc==0)!=valid) {
        fprintf(stderr,"admission rc=%d expected=%d input=%u/%u/%u kernel=%u/%u base=%u\n",
                rc,valid,cfg.input_h,cfg.input_w,cfg.input_c,cfg.kernel_h,cfg.kernel_w,base);
        return 1;
    }
    if (!valid) {
        rejected++;
        CHECK(rc == -1 && !tu_agen_has_next(&it) && !tu_agen_total(&it));
        CHECK(tu_agen_next(&it)==UINT32_MAX && it.current_idx==0);
        tu_agen_reset(&it);
        CHECK(!tu_agen_has_next(&it));
        return 0;
    }
    CHECK(it.output_h==(uint32_t)oh && it.output_w==(uint32_t)ow);
    CHECK(it.elem_size==size && tu_agen_total(&it)==(uint32_t)n);
    if (n>4096) return 0; /* Admission-only huge cases, no unbounded traversal. */
    uint32_t expected[4096], actual[4096], pos=0, padding=0;
    for (uint32_t y=0; y<(uint32_t)oh && n; y++)
        for (uint32_t x=0; x<(uint32_t)ow; x++)
            for (uint32_t ky=0; ky<cfg.kernel_h; ky++)
                for (uint32_t kx=0; kx<cfg.kernel_w; kx++)
                    for (uint32_t c=0; c<cfg.input_c; c++) {
                        __int128 iy=(__int128)y*cfg.stride_h+(__int128)ky*cfg.dilation_h-cfg.pad_h;
                        __int128 ix=(__int128)x*cfg.stride_w+(__int128)kx*cfg.dilation_w-cfg.pad_w;
                        uint32_t a=0xFFFFFF00;
                        if (iy>=0 && ix>=0 && iy<cfg.input_h && ix<cfg.input_w)
                            a=base+((uint64_t)c*cfg.input_h*cfg.input_w+
                                    (uint64_t)iy*cfg.input_w+(uint64_t)ix)*size;
                        else padding++;
                        CHECK(pos<4096 && tu_agen_has_next(&it));
                        expected[pos++]=a;
                        CHECK(tu_agen_next(&it)==a);
                    }
    CHECK(pos==(uint32_t)n);
    CHECK(!tu_agen_has_next(&it) && tu_agen_next(&it)==UINT32_MAX);
    CHECK(tu_agen_next(&it)==UINT32_MAX && it.current_idx==(uint32_t)n);
    tu_agen_reset(&it);
    CHECK(tu_agen_generate_all(&it,actual,4096)==pos);
    CHECK(!memcmp(actual,expected,pos*sizeof(*actual)));
    if (pos) {
        CHECK(tu_agen_generate_all(&it,actual,pos-1)==pos-1);
        CHECK(tu_agen_next(&it)==expected[pos-1]);
    }
    /* Range replay only for padding-free cases: tagged zero-fill is absent. */
    uint32_t runs=0;
    if (!padding) {
        tu_agen_range_t ranges[4096];
        runs=tu_agen_generate_ranges(&it,ranges,4096);
        uint32_t p=0;
        for (uint32_t r=0; r<runs; r++) {
            CHECK(ranges[r].elem_size==size && ranges[r].stride==size);
            for (uint32_t off=0; off<ranges[r].total_bytes; off+=size) {
                CHECK(p<pos && ranges[r].base_addr+off==expected[p]); p++;
            }
        }
        CHECK(p==pos);
    }
    enumerated++;
    if (verbose) printf("input=%ux%ux%u kernel=%ux%u pad=%u/%u stride=%u/%u dilation=%u/%u size=%u output=%ux%u entries=%u padding=%u runs=%u PASS\n",
        cfg.input_h,cfg.input_w,cfg.input_c,cfg.kernel_h,cfg.kernel_w,cfg.pad_h,cfg.pad_w,
        cfg.stride_h,cfg.stride_w,cfg.dilation_h,cfg.dilation_w,size,(uint32_t)oh,(uint32_t)ow,pos,padding,runs);
    return 0;
}

int main(void) {
    tu_agen_im2col_t c={.input_h=2,.input_w=2,.input_c=1,
        .kernel_h=3,.kernel_w=3,.stride_h=1,.stride_w=1,
        .dilation_h=1,.dilation_w=1,.elem_size=4};
    CHECK(!check(c,0,1)); /* Original safe RED: 1x1 instead of empty. */
    c.input_h=c.input_w=65536; c.kernel_h=c.kernel_w=1;
    CHECK(!check(c,0,0)); /* Safe RED: wrapped output count. */
    c.input_h=c.input_w=2;
    CHECK(!check(c,UINT32_MAX-7,0)); /* Safe RED: full input byte span. */
    c.input_h=2; c.input_w=1; c.pad_h=UINT32_MAX; c.stride_h=UINT32_MAX;
    CHECK(!check(c,0,1)); /* Safe RED: wrapped negative coordinate becomes input. */
    const uint32_t bases[]={0,13,UINT32_MAX-15,UINT32_MAX};
    const uint32_t sizes[]={0,1,4,UINT32_MAX};
    for (uint32_t h=0; h<=4; h++)
        for (uint32_t w=0; w<=4; w++)
            for (uint32_t channels=0; channels<=2; channels++)
                for (uint32_t k=1; k<=3; k++)
                    for (uint32_t pad=0; pad<=1; pad++)
                        for (uint32_t stride=1; stride<=2; stride++)
                            for (uint32_t dilation=1; dilation<=2; dilation++)
                                for (unsigned b=0; b<4; b++)
                                    for (unsigned s=0; s<4; s++) {
                                        tu_agen_im2col_t q={.input_h=h,.input_w=w,.input_c=channels,
                                            .kernel_h=k,.kernel_w=2,.pad_h=pad,.pad_w=pad,
                                            .stride_h=stride,.stride_w=1,.dilation_h=dilation,
                                            .dilation_w=1,.elem_size=sizes[s]};
                                        CHECK(!check(q,bases[b],0));
                                    }
    c=(tu_agen_im2col_t){.input_h=4,.input_w=5,.input_c=2,.kernel_h=2,.kernel_w=3,
        .stride_h=1,.stride_w=2,.dilation_h=1,.dilation_w=1,.elem_size=2};
    CHECK(!check(c,13,1));
    c.pad_h=c.pad_w=1; CHECK(!check(c,13,1));
    c.pad_h=c.pad_w=0; c.dilation_h=2; CHECK(!check(c,13,1));
    c.stride_h=2; CHECK(!check(c,13,1));
    for (unsigned field=0; field<6; field++) {
        tu_agen_im2col_t q=c;
        switch(field) {
            case 0:q.kernel_h=0;break; case 1:q.kernel_w=0;break;
            case 2:q.stride_h=0;break; case 3:q.stride_w=0;break;
            case 4:q.dilation_h=0;break; default:q.dilation_w=0;break;
        }
        CHECK(!check(q,0,0));
    }
    c.pad_h=UINT32_MAX; c.pad_w=UINT32_MAX; CHECK(!check(c,0,0));
    c.kernel_h=UINT32_MAX; c.dilation_h=UINT32_MAX; CHECK(!check(c,0,0));
    c.input_h=UINT32_MAX; c.input_w=UINT32_MAX; c.input_c=UINT32_MAX;
    c.kernel_h=c.kernel_w=c.dilation_h=c.dilation_w=1;
    CHECK(!check(c,0,0));
    c=(tu_agen_im2col_t){.input_h=1,.input_w=2,.input_c=1,.kernel_h=1,.kernel_w=1,
        .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,.elem_size=4};
    CHECK(!check(c,UINT32_MAX-8,0)); /* Exact exclusive end remains legal. */
    CHECK(!check(c,UINT32_MAX-7,0));
    c.input_w=UINT32_MAX; c.elem_size=1;
    CHECK(!check(c,0,0)); /* Exact maximum element count, never fully iterate. */
    tu_agen_iterator_t huge;
    CHECK(!tu_agen_iterator_init(&huge,TU_AGEN_MODE_IM2COL,0,&c));
    CHECK(tu_agen_next(&huge)==0);
    huge.current_idx=UINT32_MAX-1;
    CHECK(tu_agen_next(&huge)==UINT32_MAX-1);
    CHECK(!tu_agen_has_next(&huge) && tu_agen_next(&huge)==UINT32_MAX);
    CHECK(huge.current_idx==UINT32_MAX);
    /* Deterministic adversarial mixes keep the oracle independent of u64
     * implementation arithmetic; no invalid generated address is read. */
    const uint32_t edge[]={0,1,2,3,4,65536,0x80000000u,UINT32_MAX};
    uint32_t rng=0x73ac9e1u;
    for (unsigned i=0; i<4096; i++) {
        tu_agen_im2col_t q={0};
        uint32_t v[14];
        for (unsigned j=0; j<14; j++) {
            rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
            v[j]=edge[rng%8];
        }
        q.input_h=v[0]; q.input_w=v[1]; q.input_c=v[2];
        q.kernel_h=v[3]; q.kernel_w=v[4]; q.pad_h=v[5]; q.pad_w=v[6];
        q.stride_h=v[7]; q.stride_w=v[8]; q.dilation_h=v[9]; q.dilation_w=v[10];
        q.elem_size=v[11];
        CHECK(!check(q,v[12],0));
    }
    /* Full runtime shape trade-off matrix: retain all choices, no timing claim. */
    for (uint32_t channels=1; channels<=2; channels++)
        for (uint32_t kernel=1; kernel<=3; kernel+=2)
            for (uint32_t stride=1; stride<=2; stride++) {
                c=(tu_agen_im2col_t){.input_h=4,.input_w=5,.input_c=channels,
                    .kernel_h=kernel,.kernel_w=kernel,.stride_h=stride,.stride_w=stride,
                    .dilation_h=1,.dilation_w=1,.elem_size=2};
                CHECK(!check(c,13,1));
            }
    tu_agen_im2col_dims(NULL);
    tu_agen_iterator_t it;
    CHECK(tu_agen_iterator_init(&it,TU_AGEN_MODE_IM2COL,0,NULL)==-1);
    CHECK(!tu_agen_has_next(&it));
    printf("PASS: %u im2col cases, %u rejected, %u bounded sequence/replay cases; invalid controls/NULL pass\n",
        cases,rejected,enumerated);
    return 0;
}
