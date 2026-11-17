#include "tu_cmodel/compute/softmax_engine.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    tu_sram_region_t s; tu_sram_init(&s, 128, "order-tracer");
    float x[2] = {2, 0}, mask[2] = {0, -2}, out[2];
    tu_sram_write_bulk(&s, 0, x, sizeof(x));
    tu_softmax_desc_t d = {.data_sram=&s, .elem_count=2, .in_place=true,
        .mask=mask, .mask_is_additive=true, .scale=0.5f};
#ifdef TU_TEST_NEW_ORDER
    d.preprocess_order = TU_SOFTMAX_SCALE_THEN_MASK;
#endif
    uint64_t stall = tu_softmax_execute(&d);
    memcpy(out, tu_sram_raw_ptr(&s), sizeof(out));
    double want = 1.0/(1.0+exp(-3.0));
    int fail = stall == UINT64_MAX || !isfinite(out[0]) || fabs(out[0]-want)>2e-7;
    printf("scale-before-mask: got=%.12g expected=%.12g local_penalty=%llu result=%s\n",
           out[0],want,(unsigned long long)stall,fail?"FAIL":"PASS");
    tu_sram_destroy(&s); return fail;
}
