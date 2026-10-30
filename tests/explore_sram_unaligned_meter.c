/* Live SRAM metering versus two explicitly different word-count contracts.
 * This is a diagnostic of the compatibility path, not a hardware timing model. */
#include "tu_cmodel/tu_sram.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

static unsigned failures, cases;
static void run(unsigned addr, unsigned bytes, int scalar, int write) {
    tu_sram_region_t r;
    uint8_t input[64], before[64], out[64];
    unsigned legacy[4] = {0}, span[4] = {0};
    for (unsigned i=0;i<64;i++) input[i]=(uint8_t)(i+1);
    memset(out, 0xa5, sizeof out);
    tu_sram_init_runtime(&r,64,"unaligned",4,4,1,2,4);
    if (!r.banks.data || !r.banks.bw_banks) { failures++; return; }
    memcpy(r.banks.data,input,64); memcpy(before,input,64);
    for (unsigned i=0;i<(bytes+3)/4;i++) legacy[((addr+4*i)/4)%4]++;
    if (bytes) for (unsigned word=addr/4;word<=(addr+bytes-1)/4;word++) span[word%4]++;
    uint64_t expected_stall=0, actual_stall;
    for (unsigned i=0;i<4;i++) if (legacy[i]>1) expected_stall+=2*(legacy[i]-1);
    if (write) {
        memset(input,0x69,sizeof input);
        actual_stall=scalar?tu_sram_write(&r,addr,input):tu_sram_write_bulk(&r,addr,input,bytes);
        memcpy(before+addr,input,bytes);
        if (memcmp(before,r.banks.data,64)) failures++;
    } else {
        actual_stall=scalar?tu_sram_read(&r,addr,out):tu_sram_read_bulk(&r,addr,out,bytes);
        if (memcmp(before+addr,out,bytes)) failures++;
        for (unsigned i=bytes;i<64;i++) if (out[i]!=0xa5) {failures++; break;}
    }
    if (actual_stall!=expected_stall || r.banks.reads!=(write?0:(bytes+3)/4) || r.banks.writes!=(write?(bytes+3)/4:0)) failures++;
    unsigned sum=0; for(unsigned i=0;i<4;i++) {
        tu_sram_bw_bank_t *b=&r.banks.bw_banks[i];
        unsigned observed=(unsigned)(b->reads_served+b->writes_served+b->read_stalls+b->write_stalls);
        if(observed!=legacy[i]) failures++;
        sum+=span[i];
    }
    printf("%s %s %2u %2u %u%u%u%u %u%u%u%u %2" PRIu64 " %u\n",scalar?"scalar":"bulk  ",write?"write":"read ",addr,bytes,legacy[0],legacy[1],legacy[2],legacy[3],span[0],span[1],span[2],span[3],actual_stall,sum);
    cases++; tu_sram_destroy(&r);
}
int main(void) {
    const unsigned offsets[]={0,1,3,4,15};
    const unsigned sizes[]={0,1,2,4,5,16,17};
    puts("api direction addr bytes legacy-bank-counts span-bank-counts legacy-stalls span-words");
    for(unsigned a=0;a<sizeof offsets/sizeof offsets[0];a++) for(int w=0;w<2;w++) {
        run(offsets[a],4,1,w);
        for(unsigned n=0;n<sizeof sizes/sizeof sizes[0];n++) run(offsets[a],sizes[n],0,w);
    }
    printf("cases=%u failures=%u\n",cases,failures);
    return failures!=0;
}
