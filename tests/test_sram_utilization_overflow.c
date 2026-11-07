/* Nominal legacy-utilization arithmetic, including overflowing intermediates.
 * Artificial counter injection checks arithmetic only, not attainable histories. */
#include "tu_cmodel/tu_sram.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static float expected(__uint128_t served, __uint128_t capacity) {
    if (!capacity) return 0;
    if (served <= UINT64_MAX && capacity <= UINT64_MAX) {
        float x = (float)(uint64_t)served / (float)(uint64_t)capacity;
        return x > 1 ? 1 : x;
    }
    long double x = (long double)served / (long double)capacity;
    return x > 1 ? 1 : (float)x;
}

int main(void) {
    const unsigned bs[] = {1,2,4,8}, gs[] = {1,2,3,255};
    const uint64_t windows[] = {1,4,17};
    const uint64_t cycles[] = {0,1,10,UINT64_C(1)<<32,
        (UINT64_C(1)<<63)-1,UINT64_C(1)<<63,(UINT64_C(1)<<63)+1,UINT64_MAX};
    const uint64_t vals[] = {0,1,100,UINT64_MAX/2,UINT64_MAX-1,UINT64_MAX};
    unsigned cases=0, failures=0;
    for (unsigned bi=0;bi<4;bi++) for (unsigned gi=0;gi<4;gi++)
    for (unsigned ci=0;ci<8;ci++) for (unsigned vi=0;vi<6;vi++)
    for (unsigned wi=0;wi<3;wi++) {
        tu_sram_region_t r;
        tu_sram_init_runtime(&r, 32, "overflow-gate", bs[bi], 4, gs[gi], 2, windows[wi]);
        r.banks.current_cycle=cycles[ci];
        __uint128_t all=0;
        for (unsigned b=0;b<bs[bi];b++) {
            uint64_t rd=vals[(vi+b)%6], wr=vals[(vi+2*b+1)%6];
            r.banks.bw_banks[b].reads_served=rd;
            r.banks.bw_banks[b].writes_served=wr;
            all+=(__uint128_t)rd+wr;
        }
        uint64_t periods=cycles[ci] / windows[wi];
        if (!periods) periods=1;
        float want=cycles[ci] ? expected(all, (__uint128_t)bs[bi]*gs[gi]*periods) : 0;
        float got=tu_sram_get_bandwidth_utilization(&r);
        if (!isfinite(got) || got!=want) failures++;
        tu_sram_set_bw_modeling(&r, false);
        if (tu_sram_get_bandwidth_utilization(&r)!=0) failures++;
        tu_sram_set_bw_modeling(&r, true);
        for (unsigned b=0;b<bs[bi];b++) {
            float bank=-1;
            __uint128_t sum=(__uint128_t)r.banks.bw_banks[b].reads_served+r.banks.bw_banks[b].writes_served;
            float bank_want=expected(sum, (__uint128_t)gs[gi]*periods);
            tu_sram_get_bank_bw_stats(&r,b,NULL,NULL,NULL,NULL,&bank);
            if (!isfinite(bank) || bank!=bank_want) failures++;
        }
        tu_sram_destroy(&r);
        cases++;
    }
    printf("utilization-overflow cases=%u failures=%u\n", cases, failures);
    return failures ? 1 : 0;
}
