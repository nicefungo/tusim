/* Diagnostic: compare live DMA descriptor bank events with a per-row
 * touched-word hypothesis. This does not implement a physical SRAM schedule. */
#include "tu_cmodel/dma_descriptor.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int run(unsigned stride, unsigned base, int store, int meter) {
    tu_sram_region_t s;
    uint8_t host[16], expected[16], before[128], after[128];
    unsigned model[4] = {0}, touched[4] = {0};
    unsigned live[4] = {0};
    int fail = 0;
    tu_dma_init_full(false, 3, 8);
    tu_sram_init_runtime(&s, sizeof(before), "geometry", 4, 4, 1, 2, 4);
    if (tu_sram_set_word_metering(&s, meter) != 0) return 1;
    for (unsigned i = 0; i < sizeof(before); ++i) before[i] = (uint8_t)(i + 1);
    for (unsigned i = 0; i < sizeof(host); ++i) host[i] = (uint8_t)(0xe0 + i);
    memcpy(tu_sram_raw_ptr(&s), before, sizeof(before));
    memcpy(after, before, sizeof(after));
    for (unsigned row = 0; row < 4; ++row) {
        unsigned a = base + row * stride;
        if (store) memcpy(expected + row * 4, before + a, 4);
        else memcpy(after + a, host + row * 4, 4);
        /* One word event per intersected word per logical row; repeated words
           remain distinct events, assuming no inter-row coalescing. */
        for (unsigned w = a / 4; w <= (a + 3) / 4; ++w) ++touched[w % 4];
    }
    for (unsigned i = 0; i < 4; ++i) ++model[((base + i * 4) / 4) % 4];
    tu_dma_descriptor_t *d = tu_dma_desc_create_strided_2d(
        0, store ? TU_DMA_DIR_TU_TO_HOST : TU_DMA_DIR_HOST_TO_TU,
        &s, base, host, stride, 4, 1, 4, 4);
    if (!d) { tu_dma_destroy(); tu_sram_destroy(&s); return 1; }
    tu_dma_execute_desc(d);
    if (!d->completed) fail++;
    if (store && memcmp(host, expected, sizeof(host))) fail++;
    if (memcmp(tu_sram_raw_ptr(&s), after, sizeof(after))) fail++;
    if (!store) {
        for (unsigned i = 0; i < sizeof(host); ++i)
            if (host[i] != (uint8_t)(0xe0 + i)) fail++;
    }
    for (unsigned i = 0; i < 4; ++i) {
        tu_sram_bw_bank_t *b = &s.banks.bw_banks[i];
        live[i] = store ? (unsigned)(b->reads_served + b->read_stalls) :
                          (unsigned)(b->writes_served + b->write_stalls);
        if (live[i] != model[i]) fail++;
    }
    printf("%s %s %2u %2u %u/%u/%u/%u %u/%u/%u/%u %llu %s\n",
           store ? "store" : "load ", meter ? "span" : "size", base, stride,
           live[0], live[1], live[2], live[3],
           touched[0], touched[1], touched[2], touched[3],
           (unsigned long long)d->cycles_completed, fail ? "FAIL" : "OK");
    tu_dma_desc_destroy(d);
    tu_dma_destroy();
    tu_sram_destroy(&s);
    return fail;
}
int main(void) {
    int failures = 0;
    const unsigned strides[] = {4, 8, 16, 20};
    puts("dir   mode base stride live-bank-events row-span-hypothesis completion status");
    for (int store = 0; store < 2; ++store)
      for (int mode = 0; mode < 2; ++mode)
        for (unsigned base = 0; base <= 1; ++base)
          for (unsigned i = 0; i < 4; ++i)
            failures += run(strides[i], base, store, mode);
    printf("cases=32 failures=%d\n", failures);
    return failures != 0;
}
