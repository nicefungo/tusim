/*
 * Conv+Pool Cascade Sweep
 * ========================
 * Answers: "What fraction of a conv+pool vision block does pooling add,
 * across conv kernel sizes, PE array dimensions, and pool configurations?"
 *
 * Uses tu_conv_estimate_cycles() for conv (analytical) and
 * tu_pool_execute() for pool (functional SRAM-backed cycle counting).
 * Workload: ResNet mid-layer — 56×56 input, 64→128 channels, stride-2 pool.
 */

#include "tu_cmodel/compute/convolution_engine.h"
#include "tu_cmodel/compute/pooling_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    const char *label;
    uint32_t    kh, kw;
    uint32_t    pad;
} conv_kernel_t;

typedef struct {
    const char *label;
    uint16_t    pr, pc;
} pe_cfg_t;

typedef struct {
    const char *label;
    uint32_t    kh, kw;
    uint32_t    stride;
} pool_cfg_t;

int main(void) {
    /* --- Conv kernels: ResNet-50 style --- */
    conv_kernel_t conv_kernels[] = {
        {"1×1",        1, 1, 0},
        {"3×3",        3, 3, 1},
        {"5×5",        5, 5, 2},
    };
    int n_conv_k = sizeof(conv_kernels) / sizeof(conv_kernels[0]);

    /* --- PE arrays --- */
    pe_cfg_t pes[] = {
        {"8×8",    8,  8},
        {"16×16", 16, 16},
        {"32×32", 32, 32},
    };
    int n_pes = sizeof(pes) / sizeof(pes[0]);

    /* --- Pool configs: stride-2 downsampling, typical ResNet --- */
    pool_cfg_t pools[] = {
        {"Max 2×2s2", 2, 2, 2},
        {"Avg 2×2s2", 2, 2, 2},
        {"Max 3×3s2", 3, 3, 2},
        {"Avg 3×3s2", 3, 3, 2},
    };
    int n_pools = sizeof(pools) / sizeof(pools[0]);

    printf("\nConv+Pool Cascade Sweep\n");
    printf("========================\n");
    printf("Workload: 56×56 input, 64→128 channels, stride-2 pool\n");
    printf("Analytical conv cycles + functional pool cycles\n\n");

    printf("%-8s %-8s %-12s %10s %10s %10s %8s %10s\n",
           "ConvK", "PE", "Pool", "ConvCyc", "PoolCyc", "TotalCyc", "Pool%", "GOPS");
    printf("--------------------------------------------------------------------------------\n");

    for (int ck = 0; ck < n_conv_k; ck++) {
        conv_kernel_t *ckn = &conv_kernels[ck];

        /* Set up conv descriptor */
        tu_conv_desc_t conv_desc = {
            .batch        = 1,
            .in_channels  = 64,
            .in_height    = 56,
            .in_width     = 56,
            .out_channels = 128,
            .kernel_h     = ckn->kh,
            .kernel_w     = ckn->kw,
            .stride_h     = 1,
            .stride_w     = 1,
            .pad_t = ckn->pad, .pad_b = ckn->pad,
            .pad_l = ckn->pad, .pad_r = ckn->pad,
            .dilation_h   = 1,
            .dilation_w   = 1,
            .groups       = 1,
            .input_format = TU_CONV_FORMAT_NHWC,
            .activation   = TU_CONV_ACTIVATION_NONE,
            .has_bias     = false,
        };
        tu_conv_compute_dims(&conv_desc);

        uint32_t conv_oh = conv_desc.out_height; /* pool input height */
        uint32_t conv_ow = conv_desc.out_width;  /* pool input width */
        uint32_t conv_oc = conv_desc.out_channels; /* pool input channels */

        for (int p = 0; p < n_pes; p++) {
            pe_cfg_t *pe = &pes[p];
            uint64_t conv_cycles = tu_conv_estimate_cycles(&conv_desc, pe->pr, pe->pc);

            for (int pi = 0; pi < n_pools; pi++) {
                pool_cfg_t *pool = &pools[pi];

                /* Skip if pool kernel > conv output dimensions */
                if (pool->kh > conv_oh || pool->kw > conv_ow) continue;

                /* Set up SRAM regions for pool input (conv output sized) */
                uint32_t pool_in_elem = conv_oc * conv_oh * conv_ow;
                size_t   pool_in_bytes = pool_in_elem * sizeof(float);

                tu_sram_region_t src_region, dst_region;
                uint32_t max_out_bytes = pool_in_bytes; /* pool output ≤ input */
                tu_sram_init(&src_region, pool_in_bytes, "pool_src");
                tu_sram_init(&dst_region, max_out_bytes, "pool_dst");

                /* Fill with dummy data — pool cycle count is data-independent */
                memset(tu_sram_raw_ptr(&src_region), 0xAB, pool_in_bytes);

                tu_pool_desc_t pool_desc = {0};
                pool_desc.pool_type  = (pool->kh == 2 && pool->kw == 2 && pool->label[0] == 'M')
                                       ? TU_POOL_MAX : TU_POOL_AVG;
                /* Fix pool type from label */
                if (strncmp(pool->label, "Max", 3) == 0)
                    pool_desc.pool_type = TU_POOL_MAX;
                else
                    pool_desc.pool_type = TU_POOL_AVG;
                pool_desc.batch      = 1;
                pool_desc.channels   = conv_oc;
                pool_desc.ih         = conv_oh;
                pool_desc.iw         = conv_ow;
                pool_desc.kh         = pool->kh;
                pool_desc.kw         = pool->kw;
                pool_desc.sh         = pool->stride;
                pool_desc.sw         = pool->stride;
                pool_desc.ph         = 0;
                pool_desc.pw         = 0;
                pool_desc.elem_size  = sizeof(float);
                pool_desc.is_float   = true;
                pool_desc.count_include_pad = false;
                pool_desc.src_region = &src_region;
                pool_desc.src_offset = 0;
                pool_desc.dst_region = &dst_region;
                pool_desc.dst_offset = 0;

                int64_t pool_cycles = tu_pool_execute(&pool_desc);
                if (pool_cycles < 0) {
                    fprintf(stderr, "pool_execute failed for %s %s %s\n",
                            ckn->label, pe->label, pool->label);
                    continue;
                }

                uint64_t total_cycles = conv_cycles + (uint64_t)pool_cycles;
                double pool_pct = (total_cycles > 0)
                    ? 100.0 * (double)pool_cycles / (double)total_cycles
                    : 0.0;

                /* Compute GOPS: total MACs = in_c * k_h * k_w * out_c * oh * ow * 2 (multiply-add)
                 * For 1×1 s1: 64 * 1 * 1 * 128 * 56 * 56 = 25,690,112 MACs
                 */
                uint64_t total_macs = (uint64_t)conv_desc.in_channels
                                    * ckn->kh * ckn->kw
                                    * conv_desc.out_channels
                                    * conv_oh * conv_ow * 2ULL;
                double gops = (total_cycles > 0)
                    ? (double)total_macs / (double)total_cycles / 1e9
                    : 0.0;

                printf("%-8s %-8s %-12s %10lu %10ld %10lu %7.1f%% %9.3f\n",
                       ckn->label, pe->label, pool->label,
                       (unsigned long)conv_cycles,
                       (long)pool_cycles,
                       (unsigned long)total_cycles,
                       pool_pct, gops);

                tu_sram_destroy(&src_region);
                tu_sram_destroy(&dst_region);
            }
        }
    }

    printf("\nDone.\n");
    return 0;
}
