/*
 * TU CModel — Configurable Power/Energy Model (Gap E4)
 * =====================================================
 *
 * Production-grade energy estimation with per-technology-node
 * energy tables derived from CACTI and published silicon data.
 *
 * Architecture:
 *   Technology node presets (45nm, 28nm, 16nm, 7nm, 5nm, 3nm)
 *   Per-component energy: MAC, RegFile, SPAD, GlobalBuffer, DRAM, DMA
 *   Configurable from tu_config.yaml (perf.power_model)
 *   Integrates with tu_perf_counters_t for live energy tracking
 *
 * Gap: E4 (Power/Energy Model)
 * Dependencies: tu_config.h
 *
 * Reference data sources:
 *   - CACTI 7.0 for SRAM energy at each technology node
 *   - NVIDIA/TPU published energy breakdowns
 *   - Horowitz (2014) "1.1 Computing's energy problem"
 *   - ISSCC/JSSC MAC energy scaling trends
 */

#ifndef TU_POWER_MODEL_H
#define TU_POWER_MODEL_H

#include "../infra/config.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Technology Node Enumeration
 * ================================================================ */

typedef enum {
    TU_TECH_NODE_45NM = 0,   /* 45nm — conservative baseline */
    TU_TECH_NODE_28NM = 1,   /* 28nm — IoT/edge */
    TU_TECH_NODE_16NM = 2,   /* 16nm/14nm — mobile SoC */
    TU_TECH_NODE_7NM  = 3,   /* 7nm — datacenter (TPUv4, A100) */
    TU_TECH_NODE_5NM  = 4,   /* 5nm/4nm — datacenter (H100, TPUv5) */
    TU_TECH_NODE_3NM  = 5,   /* 3nm — next-gen */
    TU_TECH_NODE_COUNT
} tu_tech_node_t;

/* ================================================================
 * Per-Memory-Level Energy Parameters
 * ================================================================
 *
 * SRAM energy varies with size, ports, and banking. We model
 * three memory levels with distinct energy costs:
 *
 *   RegFile:      Tiny (256B-2KB), single-ported, lowest energy
 *   SPAD (L1):    Medium (32KB-256KB), multi-banked, moderate energy
 *   GlobalBuf(L2): Large (512KB-4MB), multi-banked, higher energy
 */

typedef struct {
    double pj_per_read;       /* Energy per 32-bit word read */
    double pj_per_write;      /* Energy per 32-bit word write */
    double pj_leakage_per_byte_per_cycle; /* Leakage per byte per cycle */
    const char *name;
} tu_mem_energy_t;

typedef struct {
    tu_mem_energy_t regfile;     /* Per-PE register file */
    tu_mem_energy_t spad;        /* Local scratchpad (L1) */
    tu_mem_energy_t global_buf;  /* Global buffer (L2) */
} tu_mem_hierarchy_energy_t;

/* ================================================================
 * Full Technology Node Energy Table
 * ================================================================ */

typedef struct {
    tu_tech_node_t node;
    const char    *name;           /* Human-readable name */

    /* Compute energy */
    double pj_per_fp16_mac;        /* pJ per FP16 MAC (FP16×FP16→FP32) */
    double pj_per_fp8_mac;         /* pJ per FP8 MAC */
    double pj_per_int8_mac;        /* pJ per INT8 MAC */
    double pj_per_int4_mac;        /* pJ per INT4 MAC */

    /* Memory hierarchy energy */
    tu_mem_hierarchy_energy_t memory;

    /* DRAM energy */
    double pj_per_dram_read;       /* pJ per DRAM read access (64B) */
    double pj_per_dram_write;      /* pJ per DRAM write access (64B) */
    double pj_per_dram_activate;   /* pJ per row activation (page miss) */
    double dram_idle_power_mw;     /* DRAM idle power (mW) */

    /* DMA / interconnect energy */
    double pj_per_dma_byte;        /* pJ per byte on DMA bus */
    double pj_per_noc_hop;         /* pJ per NoC hop (multicast routing) */

    /* Clock distribution */
    double pj_per_clock_tree;      /* pJ per cycle for clock distribution */
    double pj_per_pe_regfile_access; /* pJ per register file access in PE */

    /* Leakage */
    double static_power_mw_per_mm2; /* Static (leakage) power per mm² */

    /* Chip-level */
    double nominal_voltage_v;      /* Nominal Vdd */
    double frequency_ghz;          /* Typical clock frequency */
    double area_um2_per_mac;       /* MAC unit area */
    double area_um2_per_kb_sram;   /* SRAM area per KB */

} tu_tech_node_energy_t;

/* ================================================================
 * Power Model Instance
 * ================================================================ */

typedef struct {
    /* Selected technology node */
    tu_tech_node_t          tech_node;

    /* Whether power modeling is enabled */
    bool                    enabled;

    /* Energy parameters (loaded from tech node table) */
    tu_tech_node_energy_t   params;

    /* Live energy counters (picojoules) */
    double  energy_mac_pj;
    double  energy_regfile_read_pj;
    double  energy_regfile_write_pj;
    double  energy_spad_read_pj;
    double  energy_spad_write_pj;
    double  energy_global_buf_read_pj;
    double  energy_global_buf_write_pj;
    double  energy_dram_read_pj;
    double  energy_dram_write_pj;
    double  energy_dram_activate_pj;
    double  energy_dma_pj;
    double  energy_clock_pj;
    double  energy_leakage_pj;
    double  energy_total_pj;

    /* Activity counters (for energy computation) */
    uint64_t total_macs;
    uint64_t regfile_reads;
    uint64_t regfile_writes;
    uint64_t spad_reads;
    uint64_t spad_writes;
    uint64_t global_buf_reads;
    uint64_t global_buf_writes;
    uint64_t dram_reads;
    uint64_t dram_writes;
    uint64_t dram_activates;
    uint64_t dma_bytes;
    uint64_t total_cycles;

    /* Clock frequency for power calculation */
    double   clock_freq_mhz;

    /* Chip area estimate (mm²) */
    double   estimated_area_mm2;

} tu_power_model_t;

/* ================================================================
 * API: Technology Node Lookup
 * ================================================================ */

/*
 * Get the energy table for a specific technology node.
 * Returns NULL for invalid nodes.
 */
const tu_tech_node_energy_t* tu_power_get_tech_node(tu_tech_node_t node);

/*
 * Get technology node by name string.
 * Supports: "45nm", "28nm", "16nm", "7nm", "5nm", "3nm"
 * Returns TU_TECH_NODE_7NM on unrecognized name.
 */
tu_tech_node_t tu_power_tech_node_from_string(const char *name);

/*
 * Get the name string for a technology node.
 */
const char* tu_power_tech_node_name(tu_tech_node_t node);

/* ================================================================
 * API: Power Model Lifecycle
 * ================================================================ */

/*
 * Initialize a power model with a specific technology node and clock frequency.
 */
void tu_power_model_init(tu_power_model_t *pm,
                          tu_tech_node_t tech_node,
                          double clock_freq_mhz);

/*
 * Reset all energy counters to zero (preserves tech node config).
 */
void tu_power_model_reset(tu_power_model_t *pm);

/*
 * Change the technology node (preserves counters, updates params).
 */
void tu_power_model_set_tech_node(tu_power_model_t *pm,
                                   tu_tech_node_t tech_node);

/*
 * Enable/disable power tracking.
 */
void tu_power_model_set_enabled(tu_power_model_t *pm, bool enabled);

/* ================================================================
 * API: Energy Recording (per-component, cycle-accurate)
 * ================================================================ */

/*
 * Record a MAC operation. Automatically selects the correct
 * energy based on precision type.
 *
 * precision_type: 0=FP16, 1=BF16, 2=INT8, 3=INT4, 4=FP8
 */
void tu_power_record_mac(tu_power_model_t *pm,
                          uint64_t count, uint8_t precision_type);

/*
 * Record memory accesses at each level of the hierarchy.
 * Each call records one word (32-bit) access.
 */
void tu_power_record_regfile_access(tu_power_model_t *pm,
                                     bool is_write, uint64_t count);
void tu_power_record_spad_access(tu_power_model_t *pm,
                                  bool is_write, uint64_t count);
void tu_power_record_global_buf_access(tu_power_model_t *pm,
                                        bool is_write, uint64_t count);

/*
 * Record DRAM access with page hit/miss distinction.
 * bytes: amount transferred (typically aligned to 64B cache line)
 * page_hit: true if row buffer hit, false for activate+read
 */
void tu_power_record_dram_access(tu_power_model_t *pm,
                                  bool is_write, uint64_t bytes, bool page_hit);

/*
 * Record DMA bus activity.
 */
void tu_power_record_dma(tu_power_model_t *pm, uint64_t bytes);

/*
 * Advance the clock (for leakage and clock distribution energy).
 */
void tu_power_tick(tu_power_model_t *pm, uint64_t cycles);

/* ================================================================
 * API: Derived Metrics
 * ================================================================ */

/*
 * Compute total energy from component energies.
 */
void tu_power_compute_total(tu_power_model_t *pm);

/*
 * Get average power in milliwatts.
 */
double tu_power_get_avg_power_mw(const tu_power_model_t *pm);

/*
 * Get energy efficiency (pJ per effective MAC, accounting for all overhead).
 */
double tu_power_get_energy_per_mac(const tu_power_model_t *pm);

/*
 * Get energy breakdown as fractions of total.
 */
typedef struct {
    double fraction_mac;
    double fraction_regfile;
    double fraction_spad;
    double fraction_global_buf;
    double fraction_dram;
    double fraction_dma;
    double fraction_clock;
    double fraction_leakage;
} tu_power_breakdown_t;

tu_power_breakdown_t tu_power_get_breakdown(const tu_power_model_t *pm);

/*
 * Estimate chip area based on PE count, SRAM sizes, and tech node.
 */
double tu_power_estimate_area(const tu_power_model_t *pm,
                               uint32_t pe_rows, uint32_t pe_cols,
                               uint32_t spad_bytes, uint32_t gbuf_bytes);

/* ================================================================
 * API: Reporting
 * ================================================================ */

/*
 * Print a formatted power/energy report.
 */
void tu_power_print_report(const tu_power_model_t *pm);

/*
 * Print a compact one-line summary.
 */
void tu_power_print_summary(const tu_power_model_t *pm);

/* ================================================================
 * API: Snapshot / Diff (for interval profiling)
 * ================================================================ */

/*
 * Take a snapshot of current power counters.
 */
typedef struct {
    tu_power_model_t state;
    uint64_t         snapshot_cycle;
} tu_power_snapshot_t;

tu_power_snapshot_t tu_power_snapshot(const tu_power_model_t *pm);

/*
 * Compute energy consumed between two snapshots.
 */
tu_power_model_t tu_power_diff(const tu_power_snapshot_t *before,
                                const tu_power_snapshot_t *after);

/* ================================================================
 * API: Integration with tu_config_t
 * ================================================================ */

/*
 * Configure the power model from tu_config_t.
 * Called automatically during tu_core_create().
 */
void tu_power_model_from_config(tu_power_model_t *pm,
                                 const tu_config_t *config);

#ifdef __cplusplus
}
#endif

#endif /* TU_POWER_MODEL_H */
