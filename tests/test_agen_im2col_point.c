/* Random-access im2col contract. All addresses are integers, never dereferenced.
 * Oracle uses signed/wide arithmetic independently of production u64 checks. */
#include "tu_cmodel/memory/address_generator.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures, valid_points, invalid_points, replay_points;
static uint32_t oracle(const tu_agen_im2col_t *c, uint32_t base,
                       uint32_t y, uint32_t x, uint32_t ky, uint32_t kx, uint32_t ch) {
    if (!c || !c->elem_size || !c->kernel_h || !c->kernel_w ||
        !c->stride_h || !c->stride_w || !c->dilation_h || !c->dilation_w)
        return UINT32_MAX;
    __int128 eh=(__int128)(c->kernel_h-1)*c->dilation_h+1;
    __int128 ew=(__int128)(c->kernel_w-1)*c->dilation_w+1;
    __int128 h=(__int128)c->input_h+2*(__int128)c->pad_h;
    __int128 w=(__int128)c->input_w+2*(__int128)c->pad_w;
    __int128 oh=c->input_h && h>=eh ? (h-eh)/c->stride_h+1 : 0;
    __int128 ow=c->input_w && w>=ew ? (w-ew)/c->stride_w+1 : 0;
    if (oh>UINT32_MAX || ow>UINT32_MAX || y>=oh || x>=ow ||
        ky>=c->kernel_h || kx>=c->kernel_w || ch>=c->input_c)
        return UINT32_MAX;
    __uint128_t end=base+(__uint128_t)c->input_h*c->input_w*c->input_c*c->elem_size;
    if (end>UINT32_MAX) return UINT32_MAX;
    __int128 iy=(__int128)y*c->stride_h+(__int128)ky*c->dilation_h-c->pad_h;
    __int128 ix=(__int128)x*c->stride_w+(__int128)kx*c->dilation_w-c->pad_w;
    if (iy<0 || ix<0 || iy>=c->input_h || ix>=c->input_w) return UINT32_MAX;
    return base+((uint64_t)ch*c->input_h*c->input_w+
                  (uint64_t)iy*c->input_w+(uint64_t)ix)*c->elem_size;
}

static void check(const char *label, const tu_agen_im2col_t *c, uint32_t base,
                  uint32_t y, uint32_t x, uint32_t ky, uint32_t kx, uint32_t ch) {
    uint32_t expected=oracle(c,base,y,x,ky,kx,ch);
    tu_agen_im2col_t before;
    if (c) before=*c;
    uint32_t actual=tu_agen_addr_im2col(c,base,y,x,ky,kx,ch);
    checks++;
    if (expected==UINT32_MAX) invalid_points++; else valid_points++;
    if (actual!=expected || (c && memcmp(c,&before,sizeof(*c)))) {
        if (failures<20) fprintf(stderr,"FAIL %s #%u expected=%u actual=%u base=%u tuple=%u/%u/%u/%u/%u\n",
            label,checks,expected,actual,base,y,x,ky,kx,ch);
        failures++;
    }
}

static void replay(tu_agen_im2col_t c, uint32_t base, int verbose) {
    tu_agen_iterator_t it;
    c.output_h=123; c.output_w=456;
    if (tu_agen_iterator_init(&it,TU_AGEN_MODE_IM2COL,base,&c)) {
        fprintf(stderr,"FAIL replay admission\n"); failures++; return;
    }
    unsigned points=0,padding=0;
    for (uint32_t y=0;y<it.output_h;y++)
        for (uint32_t x=0;x<it.output_w;x++)
            for (uint32_t ky=0;ky<c.kernel_h;ky++)
                for (uint32_t kx=0;kx<c.kernel_w;kx++)
                    for (uint32_t ch=0;ch<c.input_c;ch++) {
                        uint32_t want=oracle(&c,base,y,x,ky,kx,ch);
                        check("matrix",&c,base,y,x,ky,kx,ch);
                        uint32_t a=tu_agen_next(&it);
                        uint32_t iter_want=want==UINT32_MAX ? 0xFFFFFF00u : want;
                        if (a!=iter_want) { fprintf(stderr,"FAIL iterator agreement\n"); failures++; }
                        padding+=want==UINT32_MAX; points++; replay_points++;
                    }
    if (points!=tu_agen_total(&it) || tu_agen_next(&it)!=UINT32_MAX) failures++;
    /* Every coordinate namespace is bounded independently, not just by whether
     * an accidental source coordinate happens to land inside the image. */
    check("output-h-bound",&c,base,it.output_h,0,0,0,0);
    check("output-w-bound",&c,base,0,it.output_w,0,0,0);
    check("kernel-h-bound",&c,base,0,0,c.kernel_h,0,0);
    check("kernel-w-bound",&c,base,0,0,0,c.kernel_w,0);
    check("channel-bound",&c,base,0,0,0,0,c.input_c);
    if (verbose) printf("input=%ux%ux%u kernel=%ux%u stride=%u/%u dilation=%u/%u pad=%u/%u width=%u output=%ux%u points=%u source=%u padding=%u PASS\n",
        c.input_h,c.input_w,c.input_c,c.kernel_h,c.kernel_w,c.stride_h,c.stride_w,
        c.dilation_h,c.dilation_w,c.pad_h,c.pad_w,c.elem_size,it.output_h,it.output_w,
        points,points-padding,padding);
}

int main(int argc, char **argv) {
    tu_agen_im2col_t c={.input_h=4,.input_w=5,.input_c=1,
        .kernel_h=1,.kernel_w=1,.stride_h=1,.stride_w=1,
        .dilation_h=1,.dilation_w=1,.elem_size=4};
    check("channel-outside",&c,0,0,0,0,0,1);
    check("kernel-h-outside",&c,0,0,0,1,0,0);
    check("kernel-w-outside",&c,0,0,0,0,1,0);
    c.stride_h=0x80000000u;
    check("output-stride-wrap",&c,0,2,0,0,0,0);
    c.stride_h=1;
    check("full-element-end",&c,UINT32_MAX-3,0,0,0,0,0);
    check("source-view-end",&c,UINT32_MAX-79,0,0,0,0,0);
    c.input_h=c.input_w=65536;
    check("source-plane-wrap",&c,0,0,0,0,0,0);
    c.input_h=4; c.input_w=5; c.elem_size=0;
    check("zero-width",&c,13,0,1,0,0,0);
    c.elem_size=4; c.stride_h=0;
    check("zero-stride",&c,0,0,0,0,0,0);
    c.stride_h=1; c.kernel_h=0;
    check("zero-kernel",&c,0,0,0,0,0,0);
    c.kernel_h=1; c.dilation_w=0;
    check("zero-dilation",&c,0,0,0,0,0,0);
    if (argc==2 && !strcmp(argv[1],"--red")) {
        printf("RED probe: checks=%u failures=%u (nonzero exit expected on baseline)\n",checks,failures);
        return failures ? 1 : 0;
    }
    c.dilation_w=1;
    check("NULL",NULL,0,0,0,0,0,0);
    /* Exact full source end, then an otherwise-addressable early element in an
     * unrepresentable view. UINT32_MAX itself is never a data byte. */
    check("exact-view-end",&c,UINT32_MAX-80,0,0,0,0,0);
    check("last-source-element",&c,UINT32_MAX-80,3,4,0,0,0);
    c.input_h=c.input_w=c.input_c=1; c.elem_size=1;
    check("last-data-byte",&c,UINT32_MAX-1,0,0,0,0,0);
    check("sentinel-data",&c,UINT32_MAX,0,0,0,0,0);
    c.input_h=2; c.pad_h=UINT32_MAX; c.stride_h=UINT32_MAX;
    check("huge-pad-before",&c,0,0,0,0,0,0);
    check("huge-pad-source",&c,0,1,0,0,0,0);
    check("huge-pad-after",&c,0,2,0,0,0,0);
    c.kernel_h=2;
    check("wide-coordinate-valid",&c,13,1,0,1,0,0);
    c.kernel_h=1; c.stride_h=1;
    check("unrepresentable-output-axis",&c,0,UINT32_MAX,0,0,0,0);
    c=(tu_agen_im2col_t){.input_h=1,.input_w=UINT32_MAX,.input_c=1,
        .kernel_h=1,.kernel_w=1,.stride_h=1,.stride_w=1,
        .dilation_h=1,.dilation_w=1,.elem_size=1};
    check("max-source-last",&c,0,0,UINT32_MAX-1,0,0,0);
    check("max-source-outside",&c,0,0,UINT32_MAX,0,0,0);
    c.input_w=1; c.elem_size=UINT32_MAX;
    check("max-element-exact-end",&c,0,0,0,0,0,0);
    check("max-element-overflow",&c,1,0,0,0,0,0);
    c.elem_size=3;
    check("nonpower-width",&c,13,0,0,0,0,0);
    /* Random access need not materialize the expansion: count can exceed the
     * iterator's u32 limit while the source and individual query are valid. */
    c=(tu_agen_im2col_t){.input_h=1,.input_w=1,.input_c=1,
        .kernel_h=65536,.kernel_w=65536,.pad_h=65535,.pad_w=65535,
        .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,.elem_size=1};
    check("large-expansion-valid-point",&c,13,0,0,65535,65535,0);
    tu_agen_iterator_t it;
    if (!tu_agen_iterator_init(&it,TU_AGEN_MODE_IM2COL,13,&c)) failures++;
    /* Bounded complete shape matrix plus independent namespace edge probes. */
    for (uint32_t h=1;h<=4;h++) for (uint32_t w=1;w<=5;w++)
        for (uint32_t ch=1;ch<=2;ch++) for (uint32_t k=1;k<=3;k++)
            for (uint32_t pad=0;pad<=1;pad++) for (uint32_t s=1;s<=2;s++)
                for (uint32_t d=1;d<=2;d++) for (uint32_t size=1;size<=4;size*=2) {
                    c=(tu_agen_im2col_t){.input_h=h,.input_w=w,.input_c=ch,
                        .kernel_h=k,.kernel_w=2,.pad_h=pad,.pad_w=pad,
                        .stride_h=s,.stride_w=1,.dilation_h=d,.dilation_w=1,.elem_size=size};
                    replay(c,13,0);
                }
    /* All u32 extremes are sampled without dereferencing or unbounded loops. */
    const uint32_t edges[]={0,1,2,3,4,65536,0x80000000u,UINT32_MAX};
    uint32_t rng=0x6ab31e5u;
    for (unsigned n=0;n<10000;n++) {
        uint32_t v[20];
        for (unsigned j=0;j<20;j++) {
            rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; v[j]=edges[rng%8];
        }
        c=(tu_agen_im2col_t){.input_h=v[0],.input_w=v[1],.input_c=v[2],
            .kernel_h=v[3],.kernel_w=v[4],.pad_h=v[5],.pad_w=v[6],
            .stride_h=v[7],.stride_w=v[8],.dilation_h=v[9],.dilation_w=v[10],
            .elem_size=v[11],.output_h=v[12],.output_w=v[13]};
        check("edge-mix",&c,v[14],v[15],v[16],v[17],v[18],v[19]);
    }
    for (uint32_t ch=1;ch<=2;ch++) for (uint32_t k=1;k<=3;k+=2)
        for (uint32_t pad=0;pad<=1;pad++) {
            c=(tu_agen_im2col_t){.input_h=4,.input_w=5,.input_c=ch,
                .kernel_h=k,.kernel_w=k,.pad_h=pad,.pad_w=pad,
                .stride_h=1,.stride_w=1,.dilation_h=1,.dilation_w=1,.elem_size=2};
            replay(c,13,1);
        }
    printf("%s: %u point checks, %u valid, %u invalid/padding, %u iterator points, %u failures\n",
        failures ? "FAIL" : "PASS",checks,valid_points,invalid_points,replay_points,failures);
    return failures ? 1 : 0;
}
