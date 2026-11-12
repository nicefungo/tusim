/* Normalization local SRAM penalties: independent serial-event oracle. */
#include "tu_cmodel/compute/normalization_engine.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); } } while (0)

/* No time advance: each address consumes a bank grant or incurs penalty 2. */
typedef struct { unsigned used[3][4], banks, grant; uint64_t penalty[3]; } oracle_t;
static void event(oracle_t *o, unsigned region, unsigned byte_offset) {
    unsigned bank = (byte_offset / 4) % o->banks;
    if (o->used[region][bank]++ >= o->grant) o->penalty[region] += 2;
}
int main(void) {
    const float input[] = {1,3,5,7}, gamma[] = {2,3}, beta[] = {.5f,-.5f};
    unsigned cases = 0;
    printf("mode,regions,params,inplace,banks,grant,returned,expected,data_penalty,gamma_penalty,beta_penalty\n");
    for (int mode=0; mode<2; ++mode) for (int shared=0; shared<2; ++shared)
    for (int params=0; params<4; ++params) for (int inplace=0; inplace<2; ++inplace)
    for (unsigned banks=1; banks<=4; banks*=4) for (unsigned grant=1; grant<=4; grant*=2) {
        tu_sram_region_t s[3];
        for (unsigned i=0;i<3;++i) {
            tu_sram_init_runtime(&s[i],256,"norm",banks,4,(uint8_t)grant,2,4);
            if (!s[i].banks.data || !s[i].banks.bw_banks) return 2;
        }
        unsigned g = shared ? 0 : 1, b = shared ? 0 : 2;
        memcpy(s[0].banks.data,input,sizeof input);
        memcpy(s[g].banks.data+128,gamma,sizeof gamma);
        memcpy(s[b].banks.data+192,beta,sizeof beta);
        tu_norm_desc_t d = {.mode=(tu_norm_mode_t)mode,.data_sram=&s[0],
            .elem_count=4,.norm_axis_dim=2,.epsilon=1e-5f,
            .in_place=(bool)inplace,.out_offset=64,
            .gamma_sram=(params&1)?&s[g]:NULL,.gamma_offset=128,
            .beta_sram=(params&2)?&s[b]:NULL,.beta_offset=192};
        oracle_t o = {.banks=banks,.grant=grant};
        if(params&1) for(unsigned i=0;i<2;++i) event(&o,g,128+4*i);
        if((params&2) && !mode) for(unsigned i=0;i<2;++i) event(&o,b,192+4*i);
        for(unsigned row=0;row<2;++row) {
            for(unsigned i=0;i<2;++i) event(&o,0,4*(2*row+i));
            for(unsigned i=0;i<2;++i) event(&o,0,(inplace?0:64)+4*(2*row+i));
        }
        uint64_t result=tu_norm_execute(&d);
        uint64_t expected=o.penalty[0]+o.penalty[1]+o.penalty[2];
        printf("%s,%s,%d,%d,%u,%u,%llu,%llu,%llu,%llu,%llu\n",
            mode?"rms":"layer",shared?"shared":"separate",params,inplace,banks,grant,
            (unsigned long long)result,(unsigned long long)expected,
            (unsigned long long)s[0].banks.stall_cycles,(unsigned long long)s[1].banks.stall_cycles,
            (unsigned long long)s[2].banks.stall_cycles);
        CHECK(result==expected);
        for(unsigned i=0;i<3;++i) CHECK(s[i].banks.stall_cycles==o.penalty[i]);
        float out[4]; memcpy(out,s[0].banks.data+(inplace?0:64),sizeof out);
        for(unsigned row=0;row<2;++row) {
            double mean=(input[2*row]+input[2*row+1])/2.0;
            double rms2=(input[2*row]*input[2*row]+input[2*row+1]*input[2*row+1])/2.0;
            for(unsigned col=0;col<2;++col) {
                double v=mode?input[2*row+col]/sqrt(rms2+1e-5f)
                             :(input[2*row+col]-mean)/sqrt(1.0+1e-5f);
                if(params&1) v*=gamma[col];
                if((params&2) && !mode) v+=beta[col];
                CHECK(isfinite(out[2*row+col]) && fabs(out[2*row+col]-v)<1e-5);
            }
        }
        for(unsigned i=0;i<3;++i) tu_sram_destroy(&s[i]);
        ++cases;
    }
    /* A nonzero historical meter must not be returned; disabled metering
     * remains a zero-penalty compatibility control. */
    for (int enabled=0; enabled<2; ++enabled) {
        tu_sram_region_t s;
        tu_sram_init_runtime(&s,256,"epoch",1,4,1,2,4);
        if (!s.banks.data || !s.banks.bw_banks) return 2;
        memcpy(s.banks.data,input,sizeof input);
        float tmp;
        tu_sram_read(&s,0,&tmp);
        tu_sram_read(&s,0,&tmp);
        uint64_t before=s.banks.stall_cycles;
        if (!enabled) tu_sram_set_bw_modeling(&s,false);
        tu_norm_desc_t d={.mode=TU_NORM_LAYER_NORM,.data_sram=&s,
            .elem_count=4,.norm_axis_dim=2,.epsilon=1e-5f,.in_place=true};
        uint64_t result=tu_norm_execute(&d);
        CHECK(before==2);
        CHECK(result==(enabled?16u:0u));
        CHECK(s.banks.stall_cycles-before==result);
        tu_sram_destroy(&s);
    }
    printf("norm stall accounting: %u cases, 2 epoch controls, %u checks, %u failures\n",cases,checks,failures);
    return failures?1:0;
}
