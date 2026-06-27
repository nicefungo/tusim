/*
 * Pooling Engine Sweep: Kernel Size × Stride × Pool Type
 * =======================================================
 * Answers: "How do kernel size and stride affect MaxPool vs AvgPool
 * throughput on a ResNet-style feature map?"
 *
 * Uses tu_pool_execute() directly — no tu_init_with_config() needed
 * since the pooling engine doesn't depend on global tu state.
 */

#include "tu_cmodel/compute/pooling_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint32_t kh, kw;
    const char *label;
} kernel_t;

static kernel_t kernels[] = {
    {2, 2, "2×2"},
    {3, 3, "3×3"},
    {5, 5, "5×5"},
    {7, 7, "7×7"},
};
#define N_KERN (sizeof(kernels)/sizeof(kernels[0]))

int main(void) {
    /* Fixed input: 56×56 × 64 channels (ResNet-50 stage 2 style) */
    const uint32_t batch = 1, channels = 64;
    const uint32_t ih = 56, iw = 56;

    /* Allocate input data in NCHW layout */
    uint32_t total_in = batch * channels * ih * iw;

    float *input = (float *)malloc(total_in * sizeof(float));
    if (!input) { fprintf(stderr, "malloc input failed\n"); return 1; }
    for (uint32_t i = 0; i < total_in; i++)
        input[i] = (float)((i * 1103515245 + 12345) & 0x7FFFFF) / 8388608.0f;

    printf("Pooling Engine Sweep: Kernel × Stride × Type\n");
    printf("Input: %ux%u, %u channels, %u spatial elements/ch\n\n", ih, iw, channels, ih*iw);
    printf("%-5s %-6s %-8s %8s %8s %12s %10s %8s\n",
           "Type", "Kernel", "Stride", "Out H×W", "OutElem", "Cycles", "Elem/cyc", "MOPs/s");
    printf("--------------------------------------------------------------------------------\n");

    int strides[] = {1, 2};
    const char *type_names[] = {"Max", "Avg"};

    for (int t = 0; t < 2; t++) {
        for (int s = 0; s < 2; s++) {
            for (int k = 0; k < (int)N_KERN; k++) {
                kernel_t *kn = &kernels[k];
                int stride = strides[s];

                /* Skip invalid: kernel > input (no padding) */
                if (kn->kh > ih || kn->kw > iw) continue;

                tu_sram_region_t src_region, dst_region;
                uint32_t max_out = total_in * sizeof(float);
                tu_sram_init(&src_region, total_in * sizeof(float), "pool_src");
                tu_sram_init(&dst_region, max_out, "pool_dst");
                memcpy(tu_sram_raw_ptr(&src_region), input, total_in * sizeof(float));

                tu_pool_desc_t desc = {0};
                desc.pool_type  = (t == 0) ? TU_POOL_MAX : TU_POOL_AVG;
                desc.batch      = batch;
                desc.channels   = channels;
                desc.ih = ih; desc.iw = iw;
                desc.kh = kn->kh; desc.kw = kn->kw;
                desc.sh = stride; desc.sw = stride;
                desc.ph = 0; desc.pw = 0;
                desc.elem_size  = sizeof(float);
                desc.is_float   = true;
                desc.src_region = &src_region;
                desc.src_offset = 0;
                desc.dst_region = &dst_region;
                desc.dst_offset = 0;

                if (tu_pool_compute_dims(&desc) != 0) {
                    printf("%-5s %-6s %-8s %8s\n",
                           type_names[t], kn->label, "SKIP", "invalid dims");
                    tu_sram_destroy(&src_region);
                    tu_sram_destroy(&dst_region);
                    continue;
                }

                int64_t cycles = tu_pool_execute(&desc);
                if (cycles <= 0) {
                    printf("%-5s %-6s stride=%-2d %s\n",
                           type_names[t], kn->label, stride, "exec failed");
                    tu_sram_destroy(&src_region);
                    tu_sram_destroy(&dst_region);
                    continue;
                }

                uint32_t out_elems = batch * channels * desc.oh * desc.ow;
                double elem_per_cyc = (double)out_elems / (double)cycles;

                /* Pooling "ops": kernel_h * kernel_w operations per output element.
                 * MaxPool: (kh*kw - 1) comparisons; AvgPool: kh*kw additions + 1 division.
                 * We report element throughput for a fair cross-type comparison. */
                printf("%-5s %-6s %-8d %4ux%-4u %8u %12ld %10.4f %8.1f\n",
                       type_names[t], kn->label, stride,
                       desc.oh, desc.ow, out_elems, (long)cycles,
                       elem_per_cyc, (double)out_elems * 1e6 / (double)cycles);

                tu_sram_destroy(&src_region);
                tu_sram_destroy(&dst_region);
            }
        }
    }

    free(input);
    printf("\nDone.\n");
    return 0;
}
