/*
 * Exploration: explicit process-node and clock assumptions.
 * This is an activity-driven first-order sweep, not a calibrated RTL power run.
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include "infra/config.h"
#include "perf/power_model.h"

static int run_case(const char *label, int node, double mhz,
                    uint64_t macs, uint64_t cycles,
                    uint64_t input_bytes, uint64_t output_bytes) {
    tu_config_t cfg;
    tu_power_model_t pm;
    tu_config_default(&cfg);
    cfg.pe_rows = 16;
    cfg.pe_cols = 16;
    cfg.sram_w_size_kb = 128;
    cfg.gbuf_size_kb = 1024;
    cfg.dram_bandwidth_gbps = 256.0;
    cfg.counters_enabled = true;
    cfg.power_tech_node = node + TU_POWER_CONFIG_TECH_45NM;
    cfg.power_clock_freq_mhz = mhz;

    if (tu_config_validate(&cfg, NULL, 0) != 0) return 1;
    tu_power_model_from_config(&pm, &cfg);

    /* Explicit activity contract for one FP16 GEMM-like kernel. */
    tu_power_record_mac(&pm, macs, 0);
    tu_power_record_regfile_access(&pm, false, 2 * macs);
    tu_power_record_regfile_access(&pm, true, macs);
    tu_power_record_spad_access(&pm, false, input_bytes / 4);
    tu_power_record_spad_access(&pm, true, output_bytes / 4);
    tu_power_record_dram_access(&pm, false, input_bytes, true);
    tu_power_record_dram_access(&pm, true, output_bytes, true);
    tu_power_record_dma(&pm, input_bytes + output_bytes);
    tu_power_tick(&pm, cycles);
    tu_power_compute_total(&pm);

    const double latency_us = (double)cycles / pm.clock_freq_mhz;
    const double energy_uj = pm.energy_total_pj / 1.0e6;
    const double power_mw = tu_power_get_avg_power_mw(&pm);
    const tu_power_breakdown_t bd = tu_power_get_breakdown(&pm);
    if (!(latency_us > 0.0 && energy_uj > 0.0 && power_mw > 0.0)) return 1;

    printf("%-16s %-5s %7.0f %10.3f %10.6f %10.3f %9.3f %7.2f %7.2f %7.2f\n",
           label, tu_power_tech_node_name(pm.tech_node), pm.clock_freq_mhz,
           latency_us, energy_uj, power_mw, pm.estimated_area_mm2,
           100.0 * bd.fraction_mac, 100.0 * bd.fraction_dram,
           100.0 * bd.fraction_leakage);
    return 0;
}

int main(void) {
    const uint64_t macs = 64ULL * 64ULL * 256ULL;
    const uint64_t cycles = 8192;
    const uint64_t input_bytes = (64ULL * 256ULL + 256ULL * 64ULL) * 2ULL;
    const uint64_t output_bytes = 64ULL * 64ULL * 4ULL;
    int failures = 0;

    printf("Explicit process/clock power-assumption sweep\n");
    printf("Activity: MACs=%" PRIu64 ", cycles=%" PRIu64
           ", input=%" PRIu64 " B, output=%" PRIu64 " B\n\n",
           macs, cycles, input_bytes, output_bytes);
    printf("%-16s %-5s %7s %10s %10s %10s %9s %7s %7s %7s\n",
           "case", "node", "MHz", "lat_us", "energy_uJ", "power_mW",
           "area_mm2", "MAC%", "DRAM%", "leak%");

    failures += run_case("node_nominal", TU_TECH_NODE_45NM, 800.0,
                         macs, cycles, input_bytes, output_bytes);
    failures += run_case("node_nominal", TU_TECH_NODE_28NM, 1200.0,
                         macs, cycles, input_bytes, output_bytes);
    failures += run_case("node_nominal", TU_TECH_NODE_16NM, 1500.0,
                         macs, cycles, input_bytes, output_bytes);
    failures += run_case("node_nominal", TU_TECH_NODE_7NM, 2000.0,
                         macs, cycles, input_bytes, output_bytes);
    failures += run_case("node_nominal", TU_TECH_NODE_5NM, 2500.0,
                         macs, cycles, input_bytes, output_bytes);
    failures += run_case("node_nominal", TU_TECH_NODE_3NM, 3000.0,
                         macs, cycles, input_bytes, output_bytes);

    failures += run_case("7nm_clock", TU_TECH_NODE_7NM, 750.0,
                         macs, cycles, input_bytes, output_bytes);
    failures += run_case("7nm_clock", TU_TECH_NODE_7NM, 1500.0,
                         macs, cycles, input_bytes, output_bytes);
    failures += run_case("7nm_clock", TU_TECH_NODE_7NM, 2500.0,
                         macs, cycles, input_bytes, output_bytes);

    if (failures) {
        fprintf(stderr, "FAIL: %d invalid rows\n", failures);
        return 1;
    }
    puts("\nPASS: all explicit configurations produced finite positive metrics");
    return 0;
}
