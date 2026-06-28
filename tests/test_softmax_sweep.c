/*
 * Softmax Mode Sweep: Standard vs Log vs Online
 * ===============================================
 * Answers: "How do SRAM stall cycles differ across softmax modes
 * for attention-like workloads of varying row count and dimension?"
 *
 * Uses tu_softmax_execute() directly with standalone SRAM regions.
 * Captures per-mode SRAM stall cycles.
 */

#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/tu_sram.h"
#include "../tu_cmodel/compute/softmax_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    const char *label;
} workload_t;

static workload_t workloads[] = {
    {   1,   64, "1×64   (single row, small head)"  },
    {   1,  128, "1×128  (typical attention head)"  },
    {   1,  256, "1×256  (large head_dim)"          },
    {   1,  512, "1×512  (very large head_dim)"     },
    {  16,   64, "16×64  (batched small)"           },
    {  16,  128, "16×128 (batched typical)"         },
    {  64,   64, "64×64  (square attention score)"  },
    { 128,  128, "128×128 (large square)"           },
    { 256,  128, "256×128 (seq_len×head_dim)"       },
};
#define N_WORKLOADS (sizeof(workloads)/sizeof(workloads[0]))

int main(void) {
    printf("Softmax Mode Sweep: Standard vs Log vs Online\n");
    printf("Metrics: SRAM stall cycles (lower = less bandwidth contention)\n\n");
    printf("%-25s %6s %8s %8s %8s %8s %9s\n",
           "Workload", "Elems", "Std", "Log", "Online", "Best", "Std=Baseline");
    printf("-------------------------------------------------------------------------------\n");

    for (int w = 0; w < (int)N_WORKLOADS; w++) {
        workload_t *wl = &workloads[w];
        uint32_t total_elems = wl->rows * wl->cols;
        uint32_t sram_bytes  = total_elems * sizeof(float) + 1024;  /* headroom */

        /* Allocate input data on host */
        float *input = (float *)malloc(total_elems * sizeof(float));
        if (!input) { fprintf(stderr, "malloc failed\\n"); return 1; }

        /* Fill with deterministic pseudo-random values spread over [-5, 5] */
        for (uint32_t i = 0; i < total_elems; i++)
            input[i] = ((float)((i * 1103515245 + 12345) & 0x7FFFFF) / 104857.6f) - 5.0f;

        uint64_t cycles[3] = {0, 0, 0};
        const char *mode_names[] = {"Standard", "Log", "Online"};
        tu_softmax_mode_t modes[] = {TU_SOFTMAX_STANDARD, TU_SOFTMAX_LOG, TU_SOFTMAX_ONLINE};

        for (int m = 0; m < 3; m++) {
            tu_sram_region_t sram;
            tu_sram_init(&sram, sram_bytes, "sweep");

            /* Write input data to SRAM */
            for (uint32_t i = 0; i < total_elems; i++)
                tu_sram_write(&sram, i * sizeof(float), &input[i]);

            tu_softmax_desc_t desc = {
                .mode        = modes[m],
                .data_sram   = &sram,
                .data_offset = 0,
                .elem_count  = total_elems,
                .axis_dim    = (wl->rows > 1) ? wl->cols : 0,
                .scale       = 0.0f,
                .in_place    = true,
                .out_offset  = 0,
                .mask        = NULL,
            };

            cycles[m] = tu_softmax_execute(&desc);

            if (cycles[m] == UINT64_MAX) {
                printf("  ERROR: %s mode failed on %s\\n", mode_names[m], wl->label);
            }

            tu_sram_destroy(&sram);
        }

        /* Find best (lowest stall cycles) */
        int best_idx = 0;
        for (int m = 1; m < 3; m++)
            if (cycles[m] < cycles[best_idx]) best_idx = m;

        printf("%-25s %6u %8lu %8lu %8lu %8s %9.1f%%\n",
               wl->label, total_elems,
               (unsigned long)cycles[0], (unsigned long)cycles[1], (unsigned long)cycles[2],
               mode_names[best_idx],
               cycles[0] > 0 ? (double)cycles[best_idx] / (double)cycles[0] * 100.0 : 100.0);

        free(input);
    }

    printf("\nDone.\n");
    return 0;
}
