/* TU DRAM core-clock and base-latency-domain sweep. */
#include "tu_cmodel/memory/dram_model.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>

static int failures;

static void run_case(tu_dram_latency_domain_t domain, double ghz) {
    const tu_dram_params_t p = {
        .clock_ghz = 1.0, .bandwidth_gbps = 64.0,
        .read_latency_cycles = 50, .write_latency_cycles = 40,
        .bus_width_bytes = 8, .burst_length = 64,
        .channels = 1, .banks_per_channel = 4,
        .row_buffer_size = 256, .model_row_conflicts = false
    };
    tu_dram_model_t *dram = tu_dram_create_custom(&p, "latency-domain-sweep");
    if (!dram || !tu_dram_configure_core_clock(dram, ghz) ||
        !tu_dram_set_latency_domain(dram, domain, 50.0, 40.0) ||
        !tu_dram_set_refresh(dram, TU_DRAM_REFRESH_ALL_BANK,
                             TU_DRAM_REFRESH_SCHEDULING_FIXED,
                             1, 1000, 100, 40, 500)) {
        fprintf(stderr, "setup failed for domain=%d %.1f GHz\n", domain, ghz);
        failures++;
        tu_dram_destroy(dram);
        return;
    }

    uint64_t latency = (domain == TU_DRAM_LATENCY_PHYSICAL_NS)
                           ? (uint64_t)ceil(50.0 * ghz) : 50;
    uint64_t expected_dma = latency + (uint64_t)ceil(4096.0 * ghz / 64.0);
    uint64_t dma = tu_dram_estimate_transfer(dram, 4096, true);
    uint64_t expected_trefi = (uint64_t)ceil(1000.0 * ghz);
    uint64_t expected_trfc = (uint64_t)ceil(100.0 * ghz);
    double bytes_per_cycle = 64.0 / ghz;
    double dma_ns = (double)dma / ghz;
    const char *name = domain == TU_DRAM_LATENCY_PHYSICAL_NS
                           ? "physical_ns" : "core_cycles";

    printf("%s\t%.1f\t%.1f\t%u\t%" PRIu64 "\t%.1f\t%" PRIu64 "\t%" PRIu64 "\n",
           name, ghz, bytes_per_cycle, dram->params.read_latency_cycles,
           dma, dma_ns, dram->refresh_trefi_cycles, dram->refresh_trfc_cycles);

    if (dram->params.read_latency_cycles != latency || dma != expected_dma ||
        dram->refresh_trefi_cycles != expected_trefi ||
        dram->refresh_trfc_cycles != expected_trfc) {
        fprintf(stderr, "gate failed for %s %.1f GHz\n", name, ghz);
        failures++;
    }
    tu_dram_destroy(dram);
}

int main(void) {
    puts("domain\tclock_GHz\tBW_B/cyc\tbase_cycles\tDMA_cycles\tDMA_ns\ttREFI_cycles\ttRFC_cycles");
    for (int domain = TU_DRAM_LATENCY_CORE_CYCLES;
         domain <= TU_DRAM_LATENCY_PHYSICAL_NS; ++domain) {
        run_case((tu_dram_latency_domain_t)domain, 0.5);
        run_case((tu_dram_latency_domain_t)domain, 1.0);
        run_case((tu_dram_latency_domain_t)domain, 2.0);
    }
    if (failures) return 1;
    puts("PASS: 6/6 clock/latency-domain rows matched exact conversions");
    return 0;
}
