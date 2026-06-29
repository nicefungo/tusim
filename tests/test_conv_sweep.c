/*
 * Convolution Engine Sweep: Kernel Size × Stride × PE Array
 * ==========================================================
 * Answers: "How does convolution cycle count scale with kernel size
 * and stride across different PE array dimensions?"
 *
 * Workload: ResNet mid-layer (56×56 feature map, 128→128 channels)
 * Sweeps: kernel 1×1, 3×3, 5×5, 7×7 × stride 1, 2 × PE 8×8, 16×16, 32×32
 *
 * Uses tu_conv_estimate_cycles() — analytical, no data needed.
 */

#include "tu_cmodel/compute/convolution_engine.h"
#include <stdio.h>

typedef struct {
    const char *label;
    uint32_t k_h, k_w;
    uint32_t s_h, s_w;
    uint32_t pad;
} kernel_cfg_t;

typedef struct {
    const char *label;
    uint16_t pr, pc;
} pe_cfg_t;

int main(void) {
    kernel_cfg_t kernels[] = {
        {"1×1, s=1",        1, 1, 1, 1, 0},
        {"3×3, s=1",        3, 3, 1, 1, 1},
        {"5×5, s=1",        5, 5, 1, 1, 2},
        {"7×7, s=1",        7, 7, 1, 1, 3},
        {"3×3, s=2",        3, 3, 2, 2, 1},
        {"5×5, s=2",        5, 5, 2, 2, 2},
        {"7×7, s=2",        7, 7, 2, 2, 3},
    };
    int n_kernels = sizeof(kernels) / sizeof(kernels[0]);

    pe_cfg_t pes[] = {
        {"8×8",    8,  8},
        {"16×16", 16, 16},
        {"32×32", 32, 32},
    };
    int n_pes = sizeof(pes) / sizeof(pes[0]);

    printf("\nConvolution Kernel×Stride×PE Sweep\n");
    printf("====================================\n");
    printf("Workload: 56×56 input, 128→128 ch, pad=same\n\n");
    printf("%-14s %-8s %8s %8s %10s %10s %10s %10s\n",
           "Kernel", "PE", "out_HW", "im2colK", "im2col_Cyc", "GEMM_Cyc",
           "Total_Cyc", "GOPS");
    printf("--------------------------------------------------------------------------------\n");

    /* Baseline config: ResNet mid-layer, 128→128 channels on 56×56 */
    tu_conv_desc_t desc = {
        .batch = 1,
        .in_channels = 128,
        .in_height = 56,
        .in_width = 56,
        .out_channels = 128,
        .groups = 1,
    };

    for (int ki = 0; ki < n_kernels; ki++) {
        kernel_cfg_t *k = &kernels[ki];
        desc.kernel_h = k->k_h;
        desc.kernel_w = k->k_w;
        desc.stride_h = k->s_h;
        desc.stride_w = k->s_w;
        /* "same" padding: pad = kernel/2 */
        desc.pad_t = desc.pad_b = k->pad;
        desc.pad_l = desc.pad_r = k->pad;
        desc.dilation_h = desc.dilation_w = 1;

        tu_conv_compute_dims(&desc);
        uint32_t oh = desc.out_height;
        uint32_t ow = desc.out_width;
        uint32_t im2col_kdim = desc.im2col_rows;

        for (int pi = 0; pi < n_pes; pi++) {
            pe_cfg_t *p = &pes[pi];
            uint64_t cycles = tu_conv_estimate_cycles(&desc, p->pr, p->pc);

            /* Operations: 2 × K × C × R × S × OH × OW (MACs count as 2 ops) */
            uint64_t total_macs = 2ULL * desc.out_channels * desc.in_channels
                                * k->k_h * k->k_w * oh * ow;
            /* Estimate frequency at 1 GHz for GOPS */
            double gops = (double)total_macs / (double)cycles; /* GOPS = Giga-ops/sec @ 1 GHz */

            uint64_t im2col_cyc = (uint64_t)desc.in_channels * desc.in_height
                                 * desc.in_width * 4 / 16; /* FP32 / 16B bus */
            uint64_t gemm_cyc = cycles - im2col_cyc;

            printf("%-14s %-8s %4u×%-4u %8u %10lu %10lu %10lu %10.1f\n",
                   k->label, p->label, oh, ow, im2col_kdim,
                   (unsigned long)im2col_cyc, (unsigned long)gemm_cyc,
                   (unsigned long)cycles, gops);
        }
        if (ki < n_kernels - 1) printf("---\n");
    }

    printf("\nKey Finding:\n");
    printf("  Larger kernels (7×7 vs 1×1) increase im2col K dimension from C to C×R×S,\n");
    printf("  causing GEMM tiles to grow by R×S×. With 128 channels: 1×1→K=128,\n");
    printf("  3×3→K=1152, 5×5→K=3200, 7×7→K=6272.\n");
    printf("  Stride=2 halves output spatial dims → 4× fewer GEMM N tiles.\n");

    return 0;
}
