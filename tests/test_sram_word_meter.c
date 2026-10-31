/* Exact SRAM API metering oracle; not a physical port or DMA timing model. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_core.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>

static unsigned checks, failures, cases;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); } } while (0)

static void row(unsigned mode, unsigned width, unsigned banks, unsigned addr,
                unsigned size, int scalar, int write) {
    tu_sram_region_t r;
    uint8_t initial[128], in[128], out[128], expected[128];
    unsigned events[8]={0};
    for(unsigned i=0;i<128;i++) {initial[i]=(uint8_t)(i*7+3); in[i]=(uint8_t)(i*11+1);}
    memcpy(expected,initial,128); memset(out,0xa5,128);
    tu_sram_init_runtime(&r,128,"meter",banks,width,1,2,4);
    CHECK(r.banks.data && r.banks.bw_banks);
    if(!r.banks.data || !r.banks.bw_banks) return;
    memcpy(r.banks.data,initial,128);
    CHECK(tu_sram_set_word_metering(&r,(int)mode)==0);
    CHECK(tu_sram_set_word_metering(&r,2)==-1 && r.banks.word_metering==(int)mode);
    unsigned first=addr/width;
    unsigned count=size ? (mode ? ((addr+size-1)/width-first+1) : (size+width-1)/width) : 0;
    for(unsigned i=0;i<count;i++) events[(first+i)%banks]++;
    uint64_t want_stall=0; for(unsigned i=0;i<banks;i++) if(events[i]>1) want_stall+=2*(events[i]-1);
    uint64_t stall;
    if(write) {
        stall=scalar?tu_sram_write(&r,addr,in):tu_sram_write_bulk(&r,addr,in,size);
        memcpy(expected+addr,in,size); CHECK(memcmp(expected,r.banks.data,128)==0);
    } else {
        stall=scalar?tu_sram_read(&r,addr,out):tu_sram_read_bulk(&r,addr,out,size);
        CHECK(memcmp(out,initial+addr,size)==0);
        for(unsigned i=size;i<128;i++) CHECK(out[i]==0xa5);
    }
    CHECK(stall==want_stall); CHECK(r.banks.stall_cycles==want_stall);
    CHECK(r.banks.reads==(write?0:count)); CHECK(r.banks.writes==(write?count:0));
    for(unsigned i=0;i<banks;i++) {
        const tu_sram_bw_bank_t *b=&r.banks.bw_banks[i];
        CHECK(b->reads_served==(write?0:(events[i]?1:0)));
        CHECK(b->writes_served==(write?(events[i]?1:0):0));
        CHECK(b->read_stalls==(write?0:(events[i]?events[i]-1:0)));
        CHECK(b->write_stalls==(write?(events[i]?events[i]-1:0):0));
        CHECK(b->total_cycles_used==events[i]);
    }
    cases++; tu_sram_destroy(&r);
}
static void configuration(void) {
    tu_config_t c; char err[256];
    const char *names[]={"size_only","address_span","bogus", "number"};
    for(int i=0;i<4;i++) {
        char json[256];
        if(i==3) snprintf(json,sizeof json,"{\"tu\":{\"memory\":{\"banking\":{\"word_metering\":42}}}}");
        else snprintf(json,sizeof json,"{\"tu\":{\"memory\":{\"banking\":{\"word_metering\":\"%s\"}}}}",names[i]);
        tu_config_default(&c);
        int rc=tu_config_load_string(json,&c,err,sizeof err);
        CHECK((rc==0)==(i<2));
        if(i<2) {
            CHECK(c.sram_word_metering==i);
            tu_runtime_config_t rt=tu_config_to_runtime(&c);
            CHECK(rt.sram_word_metering==i);
            CHECK(tu_init_from_config(&c)==0);
            CHECK(g_tu.sram_w.banks.word_metering==i && g_tu.sram_a.banks.word_metering==i && g_tu.sram_o.banks.word_metering==i);
            uint8_t x[4]={0}; tu_sram_write(&g_tu.sram_a,1,x);
            CHECK(g_tu.sram_a.banks.writes==(uint64_t)(i?2:1));
            rt.sram_word_metering=2;
            uint8_t *p=g_tu.sram_a.banks.data;
            tu_init_with_config(&rt); CHECK(g_tu.sram_a.banks.data==p);
            CHECK(tu_core_create(&rt)==NULL);
            tu_shutdown();
        }
    }
    tu_config_default(&c); CHECK(c.sram_word_metering==0);
    CHECK(tu_config_load("config/tu_config.json",&c,err,sizeof err)==0 && c.sram_word_metering==0);
    tu_runtime_config_t rt=tu_runtime_config_default(); CHECK(rt.sram_word_metering==0);
    tu_runtime_config_t zero={0}; CHECK(zero.sram_word_metering==0);
    rt.sram_word_metering=1; tu_core_t *core=tu_core_create(&rt);
    CHECK(core!=NULL);
    if(core) { CHECK(core->state.sram_a.banks.word_metering==1); tu_core_destroy(core); }
}
static void matrix(void) {
    const unsigned addresses[]={0,1,3,4,15}, lengths[]={0,1,2,4,5,16,17};
    puts("mode addr bytes bank0 bank1 bank2 bank3 stall");
    for(int mode=0;mode<2;mode++) for(unsigned a=0;a<5;a++) for(unsigned n=0;n<7;n++) {
        tu_sram_region_t r; uint8_t out[32];
        tu_sram_init_runtime(&r,64,"matrix",4,4,1,2,4);
        CHECK(tu_sram_set_word_metering(&r,mode)==0);
        uint64_t stall=tu_sram_read_bulk(&r,addresses[a],out,lengths[n]);
        printf("%s %u %u",mode?"span":"size",addresses[a],lengths[n]);
        for(unsigned b=0;b<4;b++) printf(" %" PRIu64,r.banks.bw_banks[b].reads_served+r.banks.bw_banks[b].read_stalls);
        printf(" %" PRIu64 "\n",stall);
        tu_sram_destroy(&r);
    }
}
static void boundaries(void) {
    for(int mode=0;mode<2;mode++) for(int wr=0;wr<2;wr++) {
        tu_sram_region_t r; uint8_t x[8]={0};
        tu_sram_init_runtime(&r,16,"bounds",4,4,1,2,4);
        CHECK(tu_sram_set_word_metering(&r,mode)==0);
        CHECK(tu_sram_set_word_metering(NULL,mode)==-1);
        uint64_t stall = wr ? tu_sram_write_bulk(&r,16,x,0) : tu_sram_read_bulk(&r,16,x,0);
        CHECK(stall==0 && r.banks.reads==0 && r.banks.writes==0);
        stall = wr ? tu_sram_write_bulk(&r,15,x,2) : tu_sram_read_bulk(&r,15,x,2);
        CHECK(stall==0 && r.banks.reads==0 && r.banks.writes==0);
        for(unsigned b=0;b<4;b++) CHECK(r.banks.bw_banks[b].total_cycles_used==0);
        tu_sram_destroy(&r);
    }
}
/* Descriptor DMA uses its own compatibility meter, not the SRAM API meter. */
static void dma_boundary(void) {
    for(int async=0;async<2;async++) for(int mode=0;mode<2;mode++) {
        tu_sram_region_t r; uint8_t src[16]; memset(src,0x73,sizeof src);
        tu_sram_init_runtime(&r,64,"dma",4,4,1,2,4);
        CHECK(tu_sram_set_word_metering(&r,mode)==0);
        tu_dma_init_full(async!=0,1,2);
        tu_dma_descriptor_t *d=tu_dma_desc_create_linear(0,TU_DMA_DIR_HOST_TO_TU,&r,1,src,1,sizeof src);
        CHECK(d!=NULL);
        if(!d) { tu_dma_destroy(); tu_sram_destroy(&r); continue; }
        CHECK(tu_dma_submit_desc(d)>0);
        if(async) {
            for(unsigned i=0;i<1000 && !g_tu_dma.channels[0].total_completed;i++) tu_dma_tick();
            CHECK(g_tu_dma.channels[0].total_completed==1);
        }
        CHECK(d->completed); CHECK(memcmp(r.banks.data+1,src,sizeof src)==0);
        for(unsigned b=0;b<4;b++) CHECK(r.banks.bw_banks[b].writes_served+r.banks.bw_banks[b].write_stalls==1);
        printf("dma scope mode=%d async=%d bank-events=1,1,1,1 cycles=%" PRIu64 "\n",mode,async,d->cycles_completed);
        d->next=NULL; tu_dma_destroy(); tu_dma_desc_destroy(d); tu_sram_destroy(&r);
    }
}
int main(void) {
    const unsigned widths[]={1,2,4,8}, bank_counts[]={1,3,4,8};
    for(unsigned m=0;m<2;m++) for(unsigned w=0;w<4;w++) for(unsigned b=0;b<4;b++)
    for(unsigned addr=0;addr<24;addr++) for(unsigned n=0;n<=24;n++) for(int wr=0;wr<2;wr++)
        row(m,widths[w],bank_counts[b],addr,n,0,wr);
    for(unsigned m=0;m<2;m++) for(unsigned w=0;w<4;w++) for(unsigned b=0;b<4;b++)
    for(unsigned addr=0;addr<24;addr++) for(int wr=0;wr<2;wr++)
        row(m,widths[w],bank_counts[b],addr,widths[w],1,wr);
    configuration();
    boundaries();
    dma_boundary();
    matrix();
    printf("SRAM word meter: cases=%u checks=%u failures=%u\n",cases,checks,failures);
    return failures?1:0;
}
