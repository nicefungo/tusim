/* Dense vs 2:4 structured-sparsity architecture exploration.
 * Uses the cmodel's canonical runtime config and linked cycle estimator. */
#include "tu_cmodel/sparsity/structured_2of4.h"
#include "tu_cmodel/infra/config.h"
#include <inttypes.h>
#include <stdio.h>

typedef struct { const char *name; uint32_t m, n, k; } workload_t;

static const workload_t workloads[] = {
    {"small projection", 64, 64, 64},
    {"square GEMM", 128, 128, 128},
    {"narrow-N", 512, 16, 512},
    {"wide-N", 64, 512, 512},
    {"large square", 512, 512, 512},
};

int main(void) {
    static const uint32_t decode_rates[] = {1, 4, 16};
    tu_config_t cfg;
    tu_config_default(&cfg);
    cfg.sparsity_enabled = true;
    cfg.sparsity_2of4 = true;

    printf("Dense vs 2:4 Structured Sparsity Sweep\n");
    printf("Model: FP16 W/A, FP32 O, %ux%u PE, %u-bit DMA; DMA serialized; decode overlaps compute\n\n",
           cfg.pe_rows, cfg.pe_cols, cfg.dma_bus_width_bits);
    printf("%-17s %6s %12s %12s %9s %10s %10s %10s\n",
           "workload", "dec/gc", "dense cyc", "2:4 cyc", "speedup",
           "W save", "compute", "decode");

    for (size_t w = 0; w < sizeof(workloads) / sizeof(workloads[0]); ++w) {
        for (size_t r = 0; r < sizeof(decode_rates) / sizeof(decode_rates[0]); ++r) {
            tu_sparsity_2of4_cycle_stats_t s;
            cfg.sparsity_decoder_groups_per_cycle = decode_rates[r];
            if (!tu_sparsity_2of4_estimate_cycles(
                    &cfg, workloads[w].m, workloads[w].n, workloads[w].k, &s)) {
                fprintf(stderr, "estimate failed for %s\n", workloads[w].name);
                return 1;
            }
            double speedup = (double)s.dense_total_cycles / s.sparse_total_cycles;
            double weight_save = 100.0 * (1.0 -
                (double)s.sparse_weight_bytes / s.dense_weight_bytes);
            printf("%-17s %6u %12" PRIu64 " %12" PRIu64 " %8.3fx %9.1f%% %10" PRIu64 " %10" PRIu64 "\n",
                   workloads[w].name, decode_rates[r], s.dense_total_cycles,
                   s.sparse_total_cycles, speedup, weight_save,
                   s.sparse_compute_cycles, s.sparse_decode_cycles);
        }
    }

    printf("\nFidelity: analytical upper bound; no metadata fetch alignment, sparse-lane imbalance,\n");
    printf("pruning accuracy, decoder area/power, or compiler packing overhead is quantified.\n");
    return 0;
}
