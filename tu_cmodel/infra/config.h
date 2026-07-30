/*
 * TU Configuration Loader — JSON-Driven Runtime Configuration (Gap A1)
 * =====================================================================
 *
 * Loads a JSON configuration file and populates a tu_config_t struct
 * which drives all cmodel behavior. This replaces the compile-time
 * #define constants with runtime configuration.
 *
 * Usage:
 *   tu_config_t cfg;
 *   int err = tu_config_load("config/tu_config.json", &cfg, error_buf, 256);
 *   if (err) { fprintf(stderr, "%s\n", error_buf); return 1; }
 *   tu_core_t *core = tu_core_create_from_config(&cfg);
 */

#ifndef TU_CONFIG_LOADER_H
#define TU_CONFIG_LOADER_H

#include "../tu_config.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Canonical power-process selection. Zero is AUTO so legacy zero-initialized
 * configuration structs retain the historical heuristic. */
typedef enum {
    TU_POWER_CONFIG_TECH_AUTO = 0,
    TU_POWER_CONFIG_TECH_45NM = 1,
    TU_POWER_CONFIG_TECH_28NM = 2,
    TU_POWER_CONFIG_TECH_16NM = 3,
    TU_POWER_CONFIG_TECH_7NM  = 4,
    TU_POWER_CONFIG_TECH_5NM  = 5,
    TU_POWER_CONFIG_TECH_3NM  = 6
} tu_power_config_tech_t;

/* Zero preserves the historical boolean model for legacy callers. */
typedef enum {
    TU_DRAM_CONFIG_ROW_LEGACY = 0,
    TU_DRAM_CONFIG_ROW_OPEN_PAGE = 1,
    TU_DRAM_CONFIG_ROW_CLOSED_PAGE = 2
} tu_dram_row_policy_t;

/* Zero preserves the historical burst-granularity channel striping. */
typedef enum {
    TU_DRAM_CONFIG_ADDR_BURST_INTERLEAVED = 0,
    TU_DRAM_CONFIG_ADDR_ROW_INTERLEAVED = 1
} tu_dram_address_mapping_t;

/* ================================================================
 * Full Configuration Struct
 * ================================================================
 *
 * This is the canonical configuration for a TU instance.
 * It extends tu_runtime_config_t with all tunable parameters.
 */

typedef struct tu_config_t {
    /* ---- Compute Engine ---- */
    uint16_t pe_rows;
    uint16_t pe_cols;
    uint16_t pe_pipeline_depth;
    uint16_t mac_units_per_pe;
    int      dataflow_mode;         /* TU_DATAFLOW_MODE_WS/OS/RS/NLR */
    bool     dataflow_via_plugin;

    /* ---- Precision ---- */
    bool     fp16_enabled;
    bool     fp32_enabled;
    bool     bf16_enabled;
    bool     fp8_e4m3_enabled;
    bool     fp8_e5m2_enabled;
    bool     int8_enabled;
    bool     int4_enabled;
    int      rounding_mode;         /* RNE=0, RTZ=1, STOCHASTIC=2 */
    bool     subnormal_flush;       /* true=FTZ, false=full support */
    bool     saturate;

    /* ---- Memory: SRAM ---- */
    uint32_t sram_w_size_kb;
    uint32_t sram_a_size_kb;
    uint32_t sram_o_size_kb;
    uint32_t sram_num_banks;
    uint32_t sram_bank_width;       /* bytes per bank word */
    uint32_t sram_words_per_cycle;
    int      sram_arb_mode;         /* NONE=0, RR=1, PRIORITY=2 */
    int      sram_conflict_mode;    /* NONE=0, DETECT=1, STALL=2 */
    uint8_t  sram_stall_penalty;
    uint64_t sram_bw_window_cycles;
    bool     sram_bw_modeling;

    /* ---- Memory: Global Buffer ---- */
    uint32_t gbuf_size_kb;
    uint32_t gbuf_banks;
    uint32_t gbuf_bank_width;

    /* ---- Memory: DRAM ---- */
    int      dram_type;             /* IDEAL=0, HBM2=1, ... LPDDR5=6 */
    double   dram_bandwidth_gbps;
    uint32_t dram_channels;
    bool     dram_model_row_conflicts;
    int      dram_row_policy;       /* legacy=0, open-page=1, closed-page=2 */
    int      dram_address_mapping;  /* burst-interleaved=0, row-interleaved=1 */
    uint32_t dram_row_miss_penalty_cycles;
    double   dram_latency_read;
    double   dram_latency_write;

    /* ---- DMA ---- */
    uint32_t dma_bus_width_bits;
    uint32_t dma_max_burst_bytes;
    uint32_t dma_num_channels;
    uint32_t dma_max_outstanding;
    bool     dma_async_mode;
    bool     dma_multicast_enabled;  /* DM4: multicast/broadcast DMA */

    /* ---- Weight Compression ---- */
    bool     compression_enabled;
    int      compression_type;        /* 0=none, 1=RLE, 2=adaptive RLE, 3=bitmap, 4=adaptive all */
    double   compression_rle_epsilon; /* 0=lossless exact runs */
    bool     compression_decoder_enabled; /* Include decompressor throughput in estimates */
    bool     compression_decoder_overlap_dma; /* Stream decode concurrently with payload DMA */
    uint32_t compression_decoder_elements_per_cycle; /* Dense FP16 output lanes */
    uint32_t compression_rle_runs_per_cycle; /* RLE metadata/run issue width */
    uint32_t compression_bitmap_elements_per_cycle; /* Bitmap scan width */

    /* ---- ISA / Command Queue ---- */
    uint32_t isa_instr_width_bits;
    uint32_t isa_queue_depth;
    bool     isa_dep_checking;

    /* ---- Multi-Core ---- */
    bool     multicore_enabled;
    uint32_t num_cores;
    int      interconnect_mode;     /* NONE=0, RING=1, MESH=2 */
    int      icc_switching_mode;    /* legacy=0, cut-through=1, store-forward=2 */
    int      icc_contention_mode;   /* ideal parallel=0, shared-link bound=1 */
    int      icc_mesh_routing_mode; /* deterministic XY=0, deterministic YX=1 */
    uint32_t icc_link_bytes_per_cycle;
    uint32_t icc_router_latency_cycles;

    /* ---- Performance ---- */
    int      cycle_model;           /* FUNCTIONAL=0, ESTIMATED=1, CYCLE_ACCURATE=2 */
    bool     counters_enabled;
    bool     detailed_stalls;
    bool     trace_enabled;
    char     trace_file[256];
    uint32_t trace_max_events;

    /* ---- Power / Physical Assumptions ---- */
    int      power_tech_node;       /* tu_power_config_tech_t */
    double   power_clock_freq_mhz;  /* 0=auto heuristic, otherwise explicit modeled clock */

    /* ---- Sparsity ---- */
    bool     sparsity_enabled;
    bool     sparsity_2of4;
    bool     sparsity_unstructured;
    int      sparsity_metadata_format;  /* 0=bitmask, 1=csf, 2=coordinate_list */
    uint32_t sparsity_decoder_groups_per_cycle; /* 2:4 groups decoded per cycle */

    /* ---- Verification ---- */
    int      golden_reference;      /* 0=numpy, 1=pytorch */
    uint32_t random_test_iters;
    double   error_tolerance;

    /* ---- Logging ---- */
    int      log_level;             /* TU_LOG_NONE=0 ... TU_LOG_TRACE=5 */

} tu_config_t;

/* ================================================================
 * API
 * ================================================================ */

/*
 * Load a JSON configuration file and populate a tu_config_t.
 *
 * path:       path to JSON config file
 * cfg:        output — filled with parsed values
 * error_buf:  buffer for error message on failure (can be NULL)
 * error_size: size of error_buf
 *
 * Returns 0 on success, non-zero on failure.
 *
 * Missing keys default to TU defaults (matches tu_config.h).
 * Unknown keys are silently ignored (forward compatibility).
 */
int tu_config_load(const char *path, tu_config_t *cfg,
                   char *error_buf, size_t error_size);

/*
 * Load configuration from an in-memory JSON string.
 * Same semantics as tu_config_load() but reads from a buffer.
 */
int tu_config_load_string(const char *json_str, tu_config_t *cfg,
                          char *error_buf, size_t error_size);

/*
 * Populate with safe defaults (matches tu_config.h constants).
 */
void tu_config_default(tu_config_t *cfg);

/*
 * Convert a tu_config_t to the legacy tu_runtime_config_t
 * (for backward compatibility with existing APIs).
 */
tu_runtime_config_t tu_config_to_runtime(const tu_config_t *cfg);

/*
 * Validate configuration constraints.
 * Returns 0 if valid, non-zero with error message in error_buf.
 */
int tu_config_validate(const tu_config_t *cfg,
                       char *error_buf, size_t error_size);

/*
 * Print configuration to stderr (for debugging).
 */
void tu_config_dump(const tu_config_t *cfg);

/*
 * Generate a markdown configuration reference from a tu_config_t.
 * Writes a self-contained markdown document describing every config
 * field with its current value, type, and description.
 *
 * cfg:  the config to document (NULL = use defaults)
 * out:  output stream
 */
void tu_config_emit_docs(const tu_config_t *cfg, FILE *out);

#ifdef __cplusplus
}
#endif

#endif /* TU_CONFIG_LOADER_H */
