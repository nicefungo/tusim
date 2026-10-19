/* Public contiguous-range constructor: admission, allocation and exact movement. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tu_cmodel/memory/address_generator.h"

static unsigned checks;
static long allocations, fail_at = -1;
void *__real_calloc(size_t, size_t);
void *__wrap_calloc(size_t n, size_t s) {
    long ordinal = allocations++;
    return ordinal == fail_at ? NULL : __real_calloc(n, s);
}
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)

static void reject(unsigned direction, tu_sram_region_t *sram,
                   const tu_agen_range_t *range, void *host) {
    long before = allocations;
    tu_dma_descriptor_t *d = tu_agen_desc_from_range(0,direction,sram,range,host);
    int absent = d == NULL;
    if (d) tu_dma_desc_destroy(d);
    CHECK(absent);
    CHECK(allocations == before);
}

int main(void) {
    unsigned char host[64], expected[64];
    for (unsigned i=0;i<sizeof(host);i++) expected[i]=(unsigned char)(i*17+3);
    tu_dma_init(false);
    tu_sram_region_t sram;
    tu_sram_init(&sram,sizeof(host),"range-admission");
    unsigned char *raw=tu_sram_raw_ptr(&sram);
    memset(raw,0xa5,sizeof(host));
    memset(host,0x5a,sizeof(host));
    tu_agen_range_t range={0};
    range.base_addr=8; range.total_bytes=16; range.elem_size=4; range.stride=4;
    /* First historical RED is safe: invalid direction was silently a store. */
    reject(TU_DMA_DIR_TU_TO_TU,&sram,&range,host);
    reject(UINT32_MAX,&sram,&range,host);
    reject(0,NULL,&range,host);
    reject(0,&sram,NULL,host);
    reject(0,&sram,&range,NULL);
    for (unsigned dir=0;dir<2;dir++) {
        tu_agen_range_t bad=range;
        bad.elem_size=0; reject(dir,&sram,&bad,host);
        bad=range; bad.total_bytes=0; reject(dir,&sram,&bad,host);
        bad=range; bad.total_bytes=15; reject(dir,&sram,&bad,host);
        bad=range; bad.elem_size=32; reject(dir,&sram,&bad,host);
    }
    CHECK(g_tu_dma.total_bytes==0);
    for (unsigned i=0;i<sizeof(host);i++) CHECK(raw[i]==0xa5 && host[i]==0x5a);
    puts("invalid direction/NULL/zero/fractional-element admission before allocation: PASS");

    /* One allocation failure in each direction; no transfer/accounting. */
    for (unsigned dir=0;dir<2;dir++) {
        fail_at=allocations;
        CHECK(tu_agen_desc_from_range(0,dir,&sram,&range,host)==NULL);
        fail_at=-1;
    }
    CHECK(g_tu_dma.total_bytes==0);
    puts("directional allocation-failure no-copy/no-accounting: PASS");

    unsigned rows=0;
    const unsigned widths[]={1,2,4,8};
    const unsigned starts[]={0,1,8,48};
    for (unsigned dir=0;dir<2;dir++)
    for (unsigned w=0;w<4;w++)
    for (unsigned a=0;a<4;a++) {
        range.base_addr=starts[a]; range.total_bytes=16;
        range.elem_size=widths[w]; range.stride=widths[w];
        memset(raw,0xa5,sizeof(host));
        memset(host,0x5a,sizeof(host));
        if (!dir) memcpy(host,expected,sizeof(host));
        else memcpy(raw,expected,sizeof(host));
        uint64_t before=g_tu_dma.total_bytes;
        tu_dma_descriptor_t *d=tu_agen_desc_from_range(0,dir,&sram,&range,host+range.base_addr);
        CHECK(d && !d->completed && d->next==NULL);
        CHECK(d->type==TU_DMA_XFER_LINEAR && d->direction==dir && d->channel==0);
        CHECK(d->total_bytes==16 && d->elem_size==widths[w]);
        CHECK(d->dims[0]==16/widths[w] && d->dims[1]==1 && d->dims[2]==1);
        CHECK(g_tu_dma.total_bytes==before);
        CHECK((dir ? d->src_base : d->dst_base)==range.base_addr);
        CHECK((dir ? d->dst_host : d->src_host)==host+range.base_addr);
        tu_dma_execute_desc(d);
        CHECK(d->completed && g_tu_dma.total_bytes==before+16);
        for (unsigned i=0;i<sizeof(host);i++) {
            int inside=i>=range.base_addr && i<range.base_addr+16;
            CHECK(raw[i]==(!dir && !inside ? 0xa5 : expected[i]));
            CHECK(host[i]==(dir && !inside ? 0x5a : expected[i]));
        }
        tu_dma_desc_destroy(d);
        printf("direction=%u width=%u base=%u bytes=16 exact_copy=PASS\n",dir,widths[w],starts[a]);
        rows++;
    }
    /* SRAM span checks remain executor-owned, construction is not submission. */
    range.base_addr=60; range.total_bytes=16; range.elem_size=4;
    for (unsigned dir=0;dir<2;dir++) {
        tu_dma_descriptor_t *d=tu_agen_desc_from_range(0,dir,&sram,&range,host);
        CHECK(d!=NULL);
        unsigned char old_host[64],old_raw[64];
        memcpy(old_host,host,64); memcpy(old_raw,raw,64);
        uint64_t before=g_tu_dma.total_bytes;
        tu_dma_execute_desc(d);
        CHECK(!d->completed && g_tu_dma.total_bytes==before);
        CHECK(memcmp(host,old_host,64)==0 && memcmp(raw,old_raw,64)==0);
        tu_dma_desc_destroy(d);
    }
    tu_sram_destroy(&sram);
    tu_dma_destroy();
    printf("PASS: %u movement rows, %u assertions; executor span rejection preserved\n",rows,checks);
    return 0;
}
