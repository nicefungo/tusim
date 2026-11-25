/* Exact SRAM word/penalty accounting, independent arithmetic oracle and guards. */
#include "compute/softmax_engine.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { ROWS=2, COLS=3, N=ROWS*COLS, CAP=96, BANKS=2 };
static const float x[N] = {2, 1, -1, -2, 0, 3};
static unsigned cases, failures;

/* Enumerate charged byte addresses, independently of the production ceil math. */
static void request(unsigned addr, unsigned bytes, unsigned width, int span,
                    unsigned hits[BANKS]) {
    for (unsigned i=0; i<bytes; ++i) {
        if (i == 0 || (span ? (addr+i)%width : i%width) == 0)
            hits[((addr+i)/width)%BANKS]++;
    }
}

static void run(int access, unsigned width, int span, unsigned in, unsigned out,
                int mode, bool report) {
    tu_sram_region_t r;
    tu_sram_init_runtime(&r, CAP, "softmax-io", BANKS, width, 1, 2, 4);
    uint8_t *data=tu_sram_raw_ptr(&r), before[CAP];
    memset(data, 0xa5, CAP);
    memcpy(data+in, x, sizeof(x));
    memcpy(before, data, CAP);
    float mx[ROWS]={-55,-55}, sum[ROWS]={-55,-55};
    tu_softmax_desc_t d = {.data_sram=&r, .elem_count=N, .axis_dim=COLS,
        .mode=mode, .sram_access=access, .data_offset=in, .out_offset=out,
        .max_out=mx, .sum_out=sum};
    bool ok=tu_sram_set_word_metering(&r, span)==0;
    uint64_t cost=UINT64_MAX;
    if (!tu_softmax_validate_desc(&d)) { ok=false; goto done; }
    cost=tu_softmax_execute(&d);
    unsigned rd[BANKS]={0}, wr[BANKS]={0};
    unsigned chunk=access==TU_SOFTMAX_SRAM_ROW ? COLS*sizeof(float) : sizeof(float);
    for (unsigned i=0;i<sizeof(x);i+=chunk) {
        request(in+i,chunk,width,span,rd); request(out+i,chunk,width,span,wr);
    }
    unsigned penalty=0;
    for (unsigned b=0;b<BANKS;b++) {
        tu_sram_bw_bank_t *p=&r.banks.bw_banks[b];
        ok &= p->reads_served+p->read_stalls==rd[b] && p->writes_served+p->write_stalls==wr[b];
        if (rd[b]+wr[b]) penalty+=2*(rd[b]+wr[b]-1);
    }
    ok &= cost==penalty && cost==r.banks.stall_cycles && r.banks.reads==rd[0]+rd[1] &&
          r.banks.writes==wr[0]+wr[1];
    for (unsigned i=0;i<CAP;i++) if (i<out || i>=out+sizeof(x)) ok &= data[i]==before[i];
    for (unsigned j=0;j<ROWS;j++) {
        const float *row=x+j*COLS;
        double m=fmax(row[0],fmax(row[1],row[2])), s=0;
        for (unsigned k=0;k<COLS;k++) s+=exp(row[k]-m);
        ok &= mx[j]==m && isfinite(sum[j]) && fabs(sum[j]-s)<3e-7;
        for (unsigned k=0;k<COLS;k++) {
            float got; memcpy(&got,data+out+(j*COLS+k)*sizeof(got),sizeof(got));
            double want=mode==TU_SOFTMAX_LOG ? row[k]-m-log(s) : exp(row[k]-m)/s;
            ok &= isfinite(got) && fabs(got-want)<4e-7;
        }
    }
done:
    if (report) printf("%d,%u,%d,%u,%s,%"PRIu64",%"PRIu64",%"PRIu64"\n",
        access,width,span,in%8,"accepted",r.banks.reads,r.banks.writes,r.banks.stall_cycles);
    ++cases;
    if (!ok) { ++failures; fprintf(stderr,"FAIL access=%d width=%u meter=%d in=%u out=%u mode=%d\n",access,width,span,in,out,mode); }
    tu_sram_destroy(&r);
}

int main(void) {
    puts("access,word_bytes,span_meter,input_mod8,status,read_words,write_words,local_penalties");
    for (int a=0;a<=2;a++) for (unsigned w=2;w<=8;w*=2) for (int s=0;s<=1;s++)
        for (unsigned mis=0;mis<=1;mis++) {
            if (a==TU_SOFTMAX_SRAM_NATIVE && w!=sizeof(float)) continue;
            run(a,w,s,8+mis,56+mis,TU_SOFTMAX_STANDARD,true);
            for (int m=0;m<=2;m++) {
                run(a,w,s,8+mis,9+mis,m,false); /* higher overlap: reverse */
                run(a,w,s,9+mis,8+mis,m,false); /* lower overlap: forward */
                run(a,w,s,CAP-sizeof(x),CAP-sizeof(x),m,false); /* exact end/alias */
            }
        }
    fprintf(stderr,"softmax SRAM access: %u cases, %u failures\n",cases,failures);
    return failures!=0;
}
