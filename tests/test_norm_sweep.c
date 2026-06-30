/*
 * Normalization Engine Sweep: LayerNorm vs RMSNorm
 * ==================================================
 * Answers: "How do LayerNorm and RMSNorm cycle counts compare across
 * varying element counts? RMSNorm should be cheaper (1 statistic vs 2)."
 */

#include "tu_cmodel/tu_config.h"
#include "tu_cmodel/tu_sram.h"
#include "tu_cmodel/compute/normalization_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Helper: write FP32 element to SRAM */
static void sram_put_f32(tu_sram_region_t *sram, uint32_t byte_offset, float val) {
    tu_sram_write(sram, byte_offset, &val);
}

int main(void) {
    /* Sweep over realistic element counts: 768 (hidden dim), 1536, 3072, 4096, 8192 */
    uint32_t elem_counts[] = {256, 512, 768, 1024, 2048, 4096, 8192};
    const char *labels[] = {"256", "512", "768", "1024", "2048", "4096", "8192"};
    int n_sizes = sizeof(elem_counts) / sizeof(elem_counts[0]);

    const char *mode_names[] = {"LayerNorm", "RMSNorm"};

    printf("Normalization Engine Sweep: LayerNorm vs RMSNorm\n");
    printf("Question: What is the cycle-cost difference between LayerNorm and RMSNorm?\n\n");
    printf("%-9s %-12s %10s %10s %14s\n",
           "ElemCount", "Mode", "StallCycles", "Cycles/Elem", "RelCost(vsLN)");
    printf("-------------------------------------------------------------------\n");

    for (int s = 0; s < n_sizes; s++) {
        uint32_t n = elem_counts[s];
        uint64_t ln_cycles = 0, rms_cycles = 0;

        /* We need enough SRAM for the largest test */
        uint32_t sram_bytes = n * sizeof(float) + 256;
        if (sram_bytes < 4096) sram_bytes = 4096;

        /* --- LayerNorm --- */
        {
            tu_sram_region_t sram;
            tu_sram_init(&sram, sram_bytes, "norm_ln");

            /* Fill with random-ish FP32 values */
            for (uint32_t i = 0; i < n; i++) {
                float v = (float)((i * 1103515245 + 12345) & 0x7FFFFF) / 8388608.0f;
                sram_put_f32(&sram, i * sizeof(float), v);
            }

            ln_cycles = tu_layernorm(&sram, 0, n, NULL, NULL, 1e-5f, true);
            tu_sram_destroy(&sram);
        }

        /* --- RMSNorm --- */
        {
            tu_sram_region_t sram;
            tu_sram_init(&sram, sram_bytes, "norm_rms");

            for (uint32_t i = 0; i < n; i++) {
                float v = (float)((i * 1103515245 + 12345) & 0x7FFFFF) / 8388608.0f;
                sram_put_f32(&sram, i * sizeof(float), v);
            }

            rms_cycles = tu_rmsnorm(&sram, 0, n, NULL, 1e-5f, true);
            tu_sram_destroy(&sram);
        }

        if (ln_cycles == 0 && rms_cycles == 0) {
            printf("%-9s %-12s %10s %10s %14s\n",
                   labels[s], "BOTH", "FAILED", "-", "-");
            continue;
        }

        double ln_per_elem = (double)ln_cycles / (double)n;
        double rms_per_elem = (double)rms_cycles / (double)n;

        printf("%-9s %-12s %10lu %10.3f %14s\n",
               labels[s], mode_names[0],
               (unsigned long)ln_cycles, ln_per_elem, "1.00×");

        /* Relative cost: RMSNorm cycles / LayerNorm cycles */
        double rel = (ln_cycles > 0) ? (double)rms_cycles / (double)ln_cycles : 0.0;
        printf("%-9s %-12s %10lu %10.3f %13.2f×\n",
               "", mode_names[1],
               (unsigned long)rms_cycles, rms_per_elem, rel);

        printf("-------------------------------------------------------------------\n");
    }

    printf("\nDone.\n");
    return 0;
}
