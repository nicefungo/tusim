/*
 * TU CModel — Configurable Power/Energy Model Implementation (Gap E4)
 * ===================================================================
 *
 * Energy parameters are calibrated against published silicon data and
 * CACTI 7.0 models. Key reference points:
 *
 *   - Horowitz (ISSCC 2014): 45nm energy for 32b FP add (0.9 pJ), 
 *     32b SRAM read (5 pJ), 32b DRAM (640 pJ)
 *   - NVIDIA A100 (7nm): ~0.4 pJ/FP16 MAC (extrapolated from 312 TFLOPS @ 400W)
 *   - NVIDIA H100 (4nm/5nm): ~0.25 pJ/FP16 MAC
 *   - TPUv4 (7nm): ~275 TFLOPS @ ~170W → ~0.31 pJ/bf16 MAC
 *   - Stillmaker & Baas (2017): Scaling trends across nodes
 *   - CACTI 7.0: SRAM energy at 7nm, 16nm, 45nm
 *
 * Scaling assumptions (per-node factor vs 45nm baseline):
 *   45nm: 1.00x   (baseline)
 *   28nm: 0.65x
 *   16nm: 0.40x   (FinFET transition)
 *   7nm:  0.20x
 *   5nm:  0.14x
 *   3nm:  0.10x
 *
 * These are approximate — real energy depends on circuit design,
 * voltage scaling, temperature, and workload. The model provides
 * a reasonable first-order estimate for architectural exploration.
 */

#include "power_model.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <strings.h>

/* ================================================================
 * CACTI-Derived Energy Tables per Technology Node
 * ================================================================
 *
 * Format:
 *   pj_per_*_mac    — energy per MAC in picojoules
 *   pj_per_read/write — energy per 32-bit word access in pJ
 *   pj_per_dram_*   — energy per 64B DRAM transaction in pJ
 *   pj_per_dma_byte — energy per byte on DMA bus
 *   pj_per_noc_hop  — energy per NoC router hop per word
 *   pj_per_clock_tree — energy per cycle for clock tree
 *   static_power_mw_per_mm2 — leakage power density
 *   nominal_voltage_v — typical supply voltage
 *   frequency_ghz — typical clock frequency at this node
 *   area_um2_per_mac — MAC cell area
 *   area_um2_per_kb_sram — SRAM cell area per KB
 */

static const tu_tech_node_energy_t tech_node_table[] = {
    /* ---- 45nm (baseline, conservative) ---- */
    [TU_TECH_NODE_45NM] = {
        .node = TU_TECH_NODE_45NM,
        .name = "45nm",
        .pj_per_fp16_mac    = 1.00,
        .pj_per_fp8_mac     = 0.25,
        .pj_per_int8_mac    = 0.20,
        .pj_per_int4_mac    = 0.12,
        .memory = {
            .regfile =    { .pj_per_read = 0.02, .pj_per_write = 0.02, .pj_leakage_per_byte_per_cycle = 0.000002, .name = "RegFile(45nm)" },
            .spad =       { .pj_per_read = 0.50, .pj_per_write = 0.50, .pj_leakage_per_byte_per_cycle = 0.000010, .name = "SPAD(45nm)" },
            .global_buf = { .pj_per_read = 1.20, .pj_per_write = 1.20, .pj_leakage_per_byte_per_cycle = 0.000015, .name = "GBuf(45nm)" },
        },
        .pj_per_dram_read       = 640.0,
        .pj_per_dram_write      = 600.0,
        .pj_per_dram_activate   = 1200.0,
        .dram_idle_power_mw     = 100.0,
        .pj_per_dma_byte        = 0.05,
        .pj_per_noc_hop         = 0.10,
        .pj_per_clock_tree      = 0.05,
        .pj_per_pe_regfile_access = 0.01,
        .static_power_mw_per_mm2 = 15.0,
        .nominal_voltage_v      = 1.0,
        .frequency_ghz          = 0.8,
        .area_um2_per_mac       = 800.0,
        .area_um2_per_kb_sram   = 12000.0,
    },

    /* ---- 28nm ---- */
    [TU_TECH_NODE_28NM] = {
        .node = TU_TECH_NODE_28NM,
        .name = "28nm",
        .pj_per_fp16_mac    = 0.65,
        .pj_per_fp8_mac     = 0.16,
        .pj_per_int8_mac    = 0.13,
        .pj_per_int4_mac    = 0.08,
        .memory = {
            .regfile =    { .pj_per_read = 0.013, .pj_per_write = 0.013, .pj_leakage_per_byte_per_cycle = 0.000001, .name = "RegFile(28nm)" },
            .spad =       { .pj_per_read = 0.33,  .pj_per_write = 0.33,  .pj_leakage_per_byte_per_cycle = 0.000006, .name = "SPAD(28nm)" },
            .global_buf = { .pj_per_read = 0.78,  .pj_per_write = 0.78,  .pj_leakage_per_byte_per_cycle = 0.000009, .name = "GBuf(28nm)" },
        },
        .pj_per_dram_read       = 420.0,
        .pj_per_dram_write      = 390.0,
        .pj_per_dram_activate   = 780.0,
        .dram_idle_power_mw     = 65.0,
        .pj_per_dma_byte        = 0.033,
        .pj_per_noc_hop         = 0.065,
        .pj_per_clock_tree      = 0.033,
        .pj_per_pe_regfile_access = 0.007,
        .static_power_mw_per_mm2 = 10.0,
        .nominal_voltage_v      = 0.95,
        .frequency_ghz          = 1.2,
        .area_um2_per_mac       = 520.0,
        .area_um2_per_kb_sram   = 7800.0,
    },

    /* ---- 16nm/14nm (FinFET) ---- */
    [TU_TECH_NODE_16NM] = {
        .node = TU_TECH_NODE_16NM,
        .name = "16nm",
        .pj_per_fp16_mac    = 0.40,
        .pj_per_fp8_mac     = 0.10,
        .pj_per_int8_mac    = 0.08,
        .pj_per_int4_mac    = 0.05,
        .memory = {
            .regfile =    { .pj_per_read = 0.008, .pj_per_write = 0.008, .pj_leakage_per_byte_per_cycle = 0.0000008, .name = "RegFile(16nm)" },
            .spad =       { .pj_per_read = 0.20,  .pj_per_write = 0.20,  .pj_leakage_per_byte_per_cycle = 0.000004,  .name = "SPAD(16nm)" },
            .global_buf = { .pj_per_read = 0.48,  .pj_per_write = 0.48,  .pj_leakage_per_byte_per_cycle = 0.000006,  .name = "GBuf(16nm)" },
        },
        .pj_per_dram_read       = 260.0,
        .pj_per_dram_write      = 240.0,
        .pj_per_dram_activate   = 480.0,
        .dram_idle_power_mw     = 40.0,
        .pj_per_dma_byte        = 0.020,
        .pj_per_noc_hop         = 0.040,
        .pj_per_clock_tree      = 0.020,
        .pj_per_pe_regfile_access = 0.004,
        .static_power_mw_per_mm2 = 8.0,
        .nominal_voltage_v      = 0.85,
        .frequency_ghz          = 1.5,
        .area_um2_per_mac       = 320.0,
        .area_um2_per_kb_sram   = 4800.0,
    },

    /* ---- 7nm (datacenter: TPUv4, A100) ---- */
    [TU_TECH_NODE_7NM] = {
        .node = TU_TECH_NODE_7NM,
        .name = "7nm",
        .pj_per_fp16_mac    = 0.20,
        .pj_per_fp8_mac     = 0.05,
        .pj_per_int8_mac    = 0.04,
        .pj_per_int4_mac    = 0.025,
        .memory = {
            .regfile =    { .pj_per_read = 0.004, .pj_per_write = 0.004, .pj_leakage_per_byte_per_cycle = 0.0000004, .name = "RegFile(7nm)" },
            .spad =       { .pj_per_read = 0.10,  .pj_per_write = 0.10,  .pj_leakage_per_byte_per_cycle = 0.000002,  .name = "SPAD(7nm)" },
            .global_buf = { .pj_per_read = 0.24,  .pj_per_write = 0.24,  .pj_leakage_per_byte_per_cycle = 0.000003,  .name = "GBuf(7nm)" },
        },
        .pj_per_dram_read       = 130.0,
        .pj_per_dram_write      = 120.0,
        .pj_per_dram_activate   = 240.0,
        .dram_idle_power_mw     = 20.0,
        .pj_per_dma_byte        = 0.010,
        .pj_per_noc_hop         = 0.020,
        .pj_per_clock_tree      = 0.010,
        .pj_per_pe_regfile_access = 0.002,
        .static_power_mw_per_mm2 = 5.0,
        .nominal_voltage_v      = 0.75,
        .frequency_ghz          = 2.0,
        .area_um2_per_mac       = 160.0,
        .area_um2_per_kb_sram   = 2400.0,
    },

    /* ---- 5nm/4nm (datacenter: H100, TPUv5) ---- */
    [TU_TECH_NODE_5NM] = {
        .node = TU_TECH_NODE_5NM,
        .name = "5nm",
        .pj_per_fp16_mac    = 0.14,
        .pj_per_fp8_mac     = 0.035,
        .pj_per_int8_mac    = 0.028,
        .pj_per_int4_mac    = 0.017,
        .memory = {
            .regfile =    { .pj_per_read = 0.003, .pj_per_write = 0.003, .pj_leakage_per_byte_per_cycle = 0.0000002, .name = "RegFile(5nm)" },
            .spad =       { .pj_per_read = 0.07,  .pj_per_write = 0.07,  .pj_leakage_per_byte_per_cycle = 0.000001,  .name = "SPAD(5nm)" },
            .global_buf = { .pj_per_read = 0.17,  .pj_per_write = 0.17,  .pj_leakage_per_byte_per_cycle = 0.000002,  .name = "GBuf(5nm)" },
        },
        .pj_per_dram_read       = 90.0,
        .pj_per_dram_write      = 84.0,
        .pj_per_dram_activate   = 168.0,
        .dram_idle_power_mw     = 15.0,
        .pj_per_dma_byte        = 0.007,
        .pj_per_noc_hop         = 0.014,
        .pj_per_clock_tree      = 0.007,
        .pj_per_pe_regfile_access = 0.001,
        .static_power_mw_per_mm2 = 3.5,
        .nominal_voltage_v      = 0.70,
        .frequency_ghz          = 2.5,
        .area_um2_per_mac       = 112.0,
        .area_um2_per_kb_sram   = 1680.0,
    },

    /* ---- 3nm (next-gen) ---- */
    [TU_TECH_NODE_3NM] = {
        .node = TU_TECH_NODE_3NM,
        .name = "3nm",
        .pj_per_fp16_mac    = 0.10,
        .pj_per_fp8_mac     = 0.025,
        .pj_per_int8_mac    = 0.020,
        .pj_per_int4_mac    = 0.012,
        .memory = {
            .regfile =    { .pj_per_read = 0.002, .pj_per_write = 0.002, .pj_leakage_per_byte_per_cycle = 0.0000001, .name = "RegFile(3nm)" },
            .spad =       { .pj_per_read = 0.05,  .pj_per_write = 0.05,  .pj_leakage_per_byte_per_cycle = 0.0000006, .name = "SPAD(3nm)" },
            .global_buf = { .pj_per_read = 0.12,  .pj_per_write = 0.12,  .pj_leakage_per_byte_per_cycle = 0.000001,   .name = "GBuf(3nm)" },
        },
        .pj_per_dram_read       = 64.0,
        .pj_per_dram_write      = 60.0,
        .pj_per_dram_activate   = 120.0,
        .dram_idle_power_mw     = 10.0,
        .pj_per_dma_byte        = 0.005,
        .pj_per_noc_hop         = 0.010,
        .pj_per_clock_tree      = 0.005,
        .pj_per_pe_regfile_access = 0.001,
        .static_power_mw_per_mm2 = 2.5,
        .nominal_voltage_v      = 0.65,
        .frequency_ghz          = 3.0,
        .area_um2_per_mac       = 80.0,
        .area_um2_per_kb_sram   = 1200.0,
    },
};

/* ================================================================
 * Technology Node Lookup
 * ================================================================ */

const tu_tech_node_energy_t* tu_power_get_tech_node(tu_tech_node_t node) {
    if (node >= TU_TECH_NODE_COUNT) return NULL;
    return &tech_node_table[node];
}

tu_tech_node_t tu_power_tech_node_from_string(const char *name) {
    if (!name) return TU_TECH_NODE_7NM; /* Default */
    for (int i = 0; i < TU_TECH_NODE_COUNT; i++) {
        if (strcasecmp(name, tech_node_table[i].name) == 0) {
            return (tu_tech_node_t)i;
        }
    }
    /* Try numeric: "7" -> 7nm */
    if (strcmp(name, "7") == 0 || strcmp(name, "7nm") == 0) return TU_TECH_NODE_7NM;
    if (strcmp(name, "5") == 0 || strcmp(name, "5nm") == 0) return TU_TECH_NODE_5NM;
    if (strcmp(name, "3") == 0 || strcmp(name, "3nm") == 0) return TU_TECH_NODE_3NM;
    if (strcmp(name, "16") == 0 || strcmp(name, "16nm") == 0) return TU_TECH_NODE_16NM;
    if (strcmp(name, "28") == 0 || strcmp(name, "28nm") == 0) return TU_TECH_NODE_28NM;
    if (strcmp(name, "45") == 0 || strcmp(name, "45nm") == 0) return TU_TECH_NODE_45NM;
    return TU_TECH_NODE_7NM; /* Default fallback */
}

const char* tu_power_tech_node_name(tu_tech_node_t node) {
    if (node >= TU_TECH_NODE_COUNT) return "unknown";
    return tech_node_table[node].name;
}

/* ================================================================
 * Power Model Lifecycle
 * ================================================================ */

void tu_power_model_init(tu_power_model_t *pm,
                          tu_tech_node_t tech_node,
                          double clock_freq_mhz) {
    memset(pm, 0, sizeof(*pm));
    pm->tech_node = tech_node;
    pm->enabled = true;
    pm->clock_freq_mhz = clock_freq_mhz > 0.0 ? clock_freq_mhz : 1000.0;

    /* Load energy parameters from tech node table */
    const tu_tech_node_energy_t *t = tu_power_get_tech_node(tech_node);
    if (t) {
        pm->params = *t;
    } else {
        /* Fallback to 7nm */
        pm->params = tech_node_table[TU_TECH_NODE_7NM];
        pm->tech_node = TU_TECH_NODE_7NM;
    }
}

void tu_power_model_reset(tu_power_model_t *pm) {
    tu_tech_node_t node = pm->tech_node;
    double freq = pm->clock_freq_mhz;
    bool en = pm->enabled;
    tu_power_model_init(pm, node, freq);
    pm->enabled = en;
}

void tu_power_model_set_tech_node(tu_power_model_t *pm,
                                   tu_tech_node_t tech_node) {
    pm->tech_node = tech_node;
    const tu_tech_node_energy_t *t = tu_power_get_tech_node(tech_node);
    if (t) {
        pm->params = *t;
    }
}

void tu_power_model_set_enabled(tu_power_model_t *pm, bool enabled) {
    pm->enabled = enabled;
}

/* ================================================================
 * Energy Recording
 * ================================================================ */

void tu_power_record_mac(tu_power_model_t *pm,
                          uint64_t count, uint8_t precision_type) {
    if (!pm->enabled) return;
    pm->total_macs += count;

    double pj_per_op;
    switch (precision_type) {
    case 0: /* FP16 */ pj_per_op = pm->params.pj_per_fp16_mac; break;
    case 1: /* BF16 */ pj_per_op = pm->params.pj_per_fp16_mac; break; /* Same as FP16 */
    case 2: /* INT8 */ pj_per_op = pm->params.pj_per_int8_mac; break;
    case 3: /* INT4 */ pj_per_op = pm->params.pj_per_int4_mac; break;
    case 4: /* FP8  */ pj_per_op = pm->params.pj_per_fp8_mac;  break;
    default:           pj_per_op = pm->params.pj_per_fp16_mac; break;
    }
    pm->energy_mac_pj += pj_per_op * (double)count;
}

void tu_power_record_regfile_access(tu_power_model_t *pm,
                                     bool is_write, uint64_t count) {
    if (!pm->enabled) return;
    if (is_write) {
        pm->regfile_writes += count;
        pm->energy_regfile_write_pj += pm->params.memory.regfile.pj_per_write * (double)count;
    } else {
        pm->regfile_reads += count;
        pm->energy_regfile_read_pj += pm->params.memory.regfile.pj_per_read * (double)count;
    }
}

void tu_power_record_spad_access(tu_power_model_t *pm,
                                  bool is_write, uint64_t count) {
    if (!pm->enabled) return;
    if (is_write) {
        pm->spad_writes += count;
        pm->energy_spad_write_pj += pm->params.memory.spad.pj_per_write * (double)count;
    } else {
        pm->spad_reads += count;
        pm->energy_spad_read_pj += pm->params.memory.spad.pj_per_read * (double)count;
    }
}

void tu_power_record_global_buf_access(tu_power_model_t *pm,
                                        bool is_write, uint64_t count) {
    if (!pm->enabled) return;
    if (is_write) {
        pm->global_buf_writes += count;
        pm->energy_global_buf_write_pj += pm->params.memory.global_buf.pj_per_write * (double)count;
    } else {
        pm->global_buf_reads += count;
        pm->energy_global_buf_read_pj += pm->params.memory.global_buf.pj_per_read * (double)count;
    }
}

void tu_power_record_dram_access(tu_power_model_t *pm,
                                  bool is_write, uint64_t bytes, bool page_hit) {
    if (!pm->enabled) return;

    /* Count DRAM transactions in 64B granularity */
    uint64_t transactions = (bytes + 63) / 64;
    if (is_write) {
        pm->dram_writes += transactions;
        pm->energy_dram_write_pj += pm->params.pj_per_dram_write * (double)transactions;
    } else {
        pm->dram_reads += transactions;
        pm->energy_dram_read_pj += pm->params.pj_per_dram_read * (double)transactions;
    }

    /* Page miss = row activation penalty */
    if (!page_hit) {
        pm->dram_activates++;
        pm->energy_dram_activate_pj += pm->params.pj_per_dram_activate;
    }
}

void tu_power_record_dma(tu_power_model_t *pm, uint64_t bytes) {
    if (!pm->enabled) return;
    pm->dma_bytes += bytes;
    pm->energy_dma_pj += pm->params.pj_per_dma_byte * (double)bytes;
}

void tu_power_tick(tu_power_model_t *pm, uint64_t cycles) {
    if (!pm->enabled) return;
    pm->total_cycles += cycles;

    /* Clock tree distribution energy */
    pm->energy_clock_pj += pm->params.pj_per_clock_tree * (double)cycles;

    /* Leakage energy = static_power_density * area * time */
    if (pm->estimated_area_mm2 > 0.0) {
        double cycles_sec = (double)cycles / (pm->clock_freq_mhz * 1e6);
        double leakage_joules = (pm->params.static_power_mw_per_mm2 * 1e-3)
                                * pm->estimated_area_mm2 * cycles_sec;
        pm->energy_leakage_pj += leakage_joules * 1e12;
    }
}

/* ================================================================
 * Derived Metrics
 * ================================================================ */

void tu_power_compute_total(tu_power_model_t *pm) {
    pm->energy_total_pj =
        pm->energy_mac_pj
        + pm->energy_regfile_read_pj + pm->energy_regfile_write_pj
        + pm->energy_spad_read_pj + pm->energy_spad_write_pj
        + pm->energy_global_buf_read_pj + pm->energy_global_buf_write_pj
        + pm->energy_dram_read_pj + pm->energy_dram_write_pj
        + pm->energy_dram_activate_pj
        + pm->energy_dma_pj
        + pm->energy_clock_pj
        + pm->energy_leakage_pj;
}

double tu_power_get_avg_power_mw(const tu_power_model_t *pm) {
    if (pm->total_cycles == 0 || pm->clock_freq_mhz == 0.0) return 0.0;
    double seconds = (double)pm->total_cycles / (pm->clock_freq_mhz * 1e6);
    if (seconds == 0.0) return 0.0;
    /* Use total energy; compute if zero (backward compat) */
    double total = pm->energy_total_pj;
    if (total == 0.0) {
        /* Compute on the fly */
        double e_mac    = pm->energy_mac_pj;
        double e_rf     = pm->energy_regfile_read_pj + pm->energy_regfile_write_pj;
        double e_spad   = pm->energy_spad_read_pj + pm->energy_spad_write_pj;
        double e_gbuf   = pm->energy_global_buf_read_pj + pm->energy_global_buf_write_pj;
        double e_dram   = pm->energy_dram_read_pj + pm->energy_dram_write_pj + pm->energy_dram_activate_pj;
        double e_dma    = pm->energy_dma_pj;
        double e_clk    = pm->energy_clock_pj;
        double e_leak   = pm->energy_leakage_pj;
        total = e_mac + e_rf + e_spad + e_gbuf + e_dram + e_dma + e_clk + e_leak;
    }
    /* pJ / s = pW; pW / 1e9 = mW */
    return (total / seconds) / 1e9;
}

double tu_power_get_energy_per_mac(const tu_power_model_t *pm) {
    if (pm->total_macs == 0) return INFINITY;
    tu_power_model_t *mut = (tu_power_model_t *)pm;
    if (mut->energy_total_pj == 0.0) tu_power_compute_total(mut);
    return pm->energy_total_pj / (double)pm->total_macs;
}

tu_power_breakdown_t tu_power_get_breakdown(const tu_power_model_t *pm) {
    tu_power_breakdown_t bd;
    memset(&bd, 0, sizeof(bd));

    tu_power_model_t total_model;
    memcpy(&total_model, pm, sizeof(total_model));
    tu_power_compute_total(&total_model);
    double total = total_model.energy_total_pj;
    if (total <= 0.0) return bd;

    bd.fraction_mac        = pm->energy_mac_pj / total;
    bd.fraction_regfile    = (pm->energy_regfile_read_pj + pm->energy_regfile_write_pj) / total;
    bd.fraction_spad       = (pm->energy_spad_read_pj + pm->energy_spad_write_pj) / total;
    bd.fraction_global_buf = (pm->energy_global_buf_read_pj + pm->energy_global_buf_write_pj) / total;
    bd.fraction_dram       = (pm->energy_dram_read_pj + pm->energy_dram_write_pj + pm->energy_dram_activate_pj) / total;
    bd.fraction_dma        = pm->energy_dma_pj / total;
    bd.fraction_clock      = pm->energy_clock_pj / total;
    bd.fraction_leakage    = pm->energy_leakage_pj / total;
    return bd;
}

double tu_power_estimate_area(const tu_power_model_t *pm,
                               uint32_t pe_rows, uint32_t pe_cols,
                               uint32_t spad_bytes, uint32_t gbuf_bytes) {
    double mac_area = pm->params.area_um2_per_mac * (double)(pe_rows * pe_cols);
    double spad_area = pm->params.area_um2_per_kb_sram * ((double)spad_bytes / 1024.0);
    double gbuf_area = pm->params.area_um2_per_kb_sram * ((double)gbuf_bytes / 1024.0);
    /* 30% overhead for control logic, NoC, DMA, etc. */
    double total_um2 = (mac_area + spad_area + gbuf_area) * 1.30;
    double mm2 = total_um2 / 1e6;

    /* Store area estimate for leakage calculation */
    ((tu_power_model_t *)pm)->estimated_area_mm2 = mm2;
    return mm2;
}

/* ================================================================
 * Reporting
 * ================================================================ */

void tu_power_print_report(const tu_power_model_t *pm) {
    tu_power_compute_total((tu_power_model_t *)pm);

    double avg_power = tu_power_get_avg_power_mw(pm);
    double pj_per_mac = tu_power_get_energy_per_mac(pm);
    tu_power_breakdown_t bd = tu_power_get_breakdown(pm);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              TU Power & Energy Report                        ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Technology Node:    %-6s                                   ║\n", pm->params.name);
    printf("║  Frequency:          %-8.1f MHz                             ║\n", pm->clock_freq_mhz);
    printf("║  Nominal Vdd:        %-8.2f V                               ║\n", pm->params.nominal_voltage_v);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Energy Breakdown:                                           ║\n");
    printf("║    MAC compute:      %12.1f pJ  (%5.1f%%)                    ║\n",
           pm->energy_mac_pj, bd.fraction_mac * 100.0);
    printf("║    RegFile:          %12.1f pJ  (%5.1f%%)                    ║\n",
           pm->energy_regfile_read_pj + pm->energy_regfile_write_pj,
           bd.fraction_regfile * 100.0);
    printf("║    Scratchpad:       %12.1f pJ  (%5.1f%%)                    ║\n",
           pm->energy_spad_read_pj + pm->energy_spad_write_pj,
           bd.fraction_spad * 100.0);
    printf("║    Global Buffer:    %12.1f pJ  (%5.1f%%)                    ║\n",
           pm->energy_global_buf_read_pj + pm->energy_global_buf_write_pj,
           bd.fraction_global_buf * 100.0);
    printf("║    DRAM:             %12.1f pJ  (%5.1f%%)                    ║\n",
           pm->energy_dram_read_pj + pm->energy_dram_write_pj + pm->energy_dram_activate_pj,
           bd.fraction_dram * 100.0);
    printf("║    DMA / bus:        %12.1f pJ  (%5.1f%%)                    ║\n",
           pm->energy_dma_pj, bd.fraction_dma * 100.0);
    printf("║    Clock tree:       %12.1f pJ  (%5.1f%%)                    ║\n",
           pm->energy_clock_pj, bd.fraction_clock * 100.0);
    printf("║    Leakage:          %12.1f pJ  (%5.1f%%)                    ║\n",
           pm->energy_leakage_pj, bd.fraction_leakage * 100.0);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  TOTAL ENERGY:       %12.1f pJ                               ║\n",
           pm->energy_total_pj);
    printf("║  Average Power:      %12.3f mW                               ║\n", avg_power);
    printf("║  Energy per MAC:     %12.3f pJ                               ║\n", pj_per_mac);
    printf("║  Total MACs:         %12lu                                   ║\n",
           (unsigned long)pm->total_macs);
    printf("║  Total Cycles:       %12lu                                   ║\n",
           (unsigned long)pm->total_cycles);
    if (pm->estimated_area_mm2 > 0.0) {
        printf("║  Est. Chip Area:     %12.3f mm²                              ║\n",
               pm->estimated_area_mm2);
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

void tu_power_print_summary(const tu_power_model_t *pm) {
    tu_power_compute_total((tu_power_model_t *)pm);
    double avg_power = tu_power_get_avg_power_mw(pm);
    printf("[power] %s %.1f MHz | %.1f pJ total | %.3f mW avg | %.3f pJ/MAC | %lu MACs\n",
           pm->params.name, pm->clock_freq_mhz, pm->energy_total_pj,
           avg_power, tu_power_get_energy_per_mac(pm),
           (unsigned long)pm->total_macs);
}

/* ================================================================
 * Snapshot & Diff
 * ================================================================ */

tu_power_snapshot_t tu_power_snapshot(const tu_power_model_t *pm) {
    tu_power_snapshot_t snap;
    memcpy(&snap.state, pm, sizeof(*pm));
    snap.snapshot_cycle = pm->total_cycles;
    return snap;
}

tu_power_model_t tu_power_diff(const tu_power_snapshot_t *before,
                                const tu_power_snapshot_t *after) {
    tu_power_model_t diff;
    memset(&diff, 0, sizeof(diff));

    if (!before || !after) return diff;

    const tu_power_model_t *a = &after->state;
    const tu_power_model_t *b = &before->state;

    /* Preserve config */
    diff.tech_node = a->tech_node;
    diff.enabled = a->enabled;
    diff.params = a->params;
    diff.clock_freq_mhz = a->clock_freq_mhz;
    diff.estimated_area_mm2 = a->estimated_area_mm2;

    /* Diff energy counters */
    diff.energy_mac_pj                = a->energy_mac_pj - b->energy_mac_pj;
    diff.energy_regfile_read_pj       = a->energy_regfile_read_pj - b->energy_regfile_read_pj;
    diff.energy_regfile_write_pj      = a->energy_regfile_write_pj - b->energy_regfile_write_pj;
    diff.energy_spad_read_pj          = a->energy_spad_read_pj - b->energy_spad_read_pj;
    diff.energy_spad_write_pj         = a->energy_spad_write_pj - b->energy_spad_write_pj;
    diff.energy_global_buf_read_pj    = a->energy_global_buf_read_pj - b->energy_global_buf_read_pj;
    diff.energy_global_buf_write_pj   = a->energy_global_buf_write_pj - b->energy_global_buf_write_pj;
    diff.energy_dram_read_pj          = a->energy_dram_read_pj - b->energy_dram_read_pj;
    diff.energy_dram_write_pj         = a->energy_dram_write_pj - b->energy_dram_write_pj;
    diff.energy_dram_activate_pj      = a->energy_dram_activate_pj - b->energy_dram_activate_pj;
    diff.energy_dma_pj                = a->energy_dma_pj - b->energy_dma_pj;
    diff.energy_clock_pj              = a->energy_clock_pj - b->energy_clock_pj;
    diff.energy_leakage_pj            = a->energy_leakage_pj - b->energy_leakage_pj;
    diff.energy_total_pj              = a->energy_total_pj - b->energy_total_pj;

    /* Diff activity counters */
    diff.total_macs      = a->total_macs - b->total_macs;
    diff.regfile_reads   = a->regfile_reads - b->regfile_reads;
    diff.regfile_writes  = a->regfile_writes - b->regfile_writes;
    diff.spad_reads      = a->spad_reads - b->spad_reads;
    diff.spad_writes     = a->spad_writes - b->spad_writes;
    diff.global_buf_reads  = a->global_buf_reads - b->global_buf_reads;
    diff.global_buf_writes = a->global_buf_writes - b->global_buf_writes;
    diff.dram_reads      = a->dram_reads - b->dram_reads;
    diff.dram_writes     = a->dram_writes - b->dram_writes;
    diff.dram_activates  = a->dram_activates - b->dram_activates;
    diff.dma_bytes       = a->dma_bytes - b->dma_bytes;
    diff.total_cycles    = a->total_cycles - b->total_cycles;

    return diff;
}

/* ================================================================
 * Integration with tu_config_t
 * ================================================================ */

void tu_power_model_from_config(tu_power_model_t *pm,
                                 const tu_config_t *config) {
    if (!pm || !config) return;

    /* Explicit selection is preferred for architecture studies; AUTO
     * preserves the historical array-size heuristic. */
    tu_tech_node_t node = TU_TECH_NODE_7NM; /* Default */

    if (config->power_tech_node > 0 && config->power_tech_node <= TU_TECH_NODE_COUNT) {
        node = (tu_tech_node_t)(config->power_tech_node - 1);
    } else {
        if (config->pe_rows <= 16 && config->pe_cols <= 16) {
            node = TU_TECH_NODE_7NM;
        } else if (config->pe_rows >= 128 && config->pe_cols >= 128) {
            node = TU_TECH_NODE_5NM;
        }
    }

    double freq = config->power_clock_freq_mhz;
    if (freq == 0.0) {
        freq = 1000.0;
        if (config->dram_bandwidth_gbps > 500.0)
            freq = 2000.0;
    }

    tu_power_model_init(pm, node, freq);
    pm->enabled = config->counters_enabled;

    /* Estimate chip area */
    uint32_t spad_bytes = config->sram_w_size_kb * 1024;
    uint32_t gbuf_bytes = config->gbuf_size_kb * 1024;
    tu_power_estimate_area(pm, config->pe_rows, config->pe_cols,
                            spad_bytes, gbuf_bytes);
}
