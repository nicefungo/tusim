/*
 * TU Configuration Loader — Implementation (Gap A1)
 * ==================================================
 * JSON-driven runtime configuration for all TU cmodel parameters.
 */

#define _POSIX_C_SOURCE 200809L  /* for strdup */
#include "config.h"
#include "json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Helper: read entire file ---- */

static char *read_file(const char *path, char *error_buf, size_t error_size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "cannot open config file: %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0 || sz > 10 * 1024 * 1024) {
        fclose(f);
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "config file too large: %s", path);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

/* ---- JSON value helpers ---- */

static bool parse_opt_string(const tu_json_value_t *obj, const char *key,
                             char *out, size_t out_size) {
    const tu_json_value_t *v = tu_json_get(obj, key);
    if (!v || v->type != TU_JSON_STRING) return false;
    uint32_t len; const char *s = tu_json_as_string(v, &len);
    if (!s) return false;
    size_t copy = len < out_size - 1 ? len : out_size - 1;
    memcpy(out, s, copy); out[copy] = '\0';
    return true;
}

static bool parse_opt_int64(const tu_json_value_t *obj, const char *key, int64_t *out) {
    const tu_json_value_t *v = tu_json_get(obj, key);
    if (!v) return false;
    if (v->type == TU_JSON_INT || v->type == TU_JSON_DOUBLE || v->type == TU_JSON_BOOL) {
        *out = tu_json_as_int(v); return true;
    }
    return false;
}

static bool parse_opt_uint(const tu_json_value_t *obj, const char *key,
                           uint32_t *out, uint32_t dflt) {
    const tu_json_value_t *v = tu_json_get(obj, key);
    if (!v) return false;
    if (v->type == TU_JSON_INT || v->type == TU_JSON_DOUBLE) {
        int64_t iv = tu_json_as_int(v);
        *out = iv < 0 ? dflt : (uint32_t)(iv & 0xFFFFFFFF);
        return true;
    }
    return false;
}

static bool parse_opt_uint16(const tu_json_value_t *obj, const char *key,
                             uint16_t *out, uint16_t dflt) {
    const tu_json_value_t *v = tu_json_get(obj, key);
    if (!v) return false;
    if (v->type == TU_JSON_INT || v->type == TU_JSON_DOUBLE) {
        int64_t iv = tu_json_as_int(v);
        *out = (iv < 1 || iv > 65535) ? dflt : (uint16_t)iv;
        return true;
    }
    return false;
}

static bool parse_opt_double(const tu_json_value_t *obj, const char *key, double *out) {
    const tu_json_value_t *v = tu_json_get(obj, key);
    if (!v) return false;
    if (v->type == TU_JSON_DOUBLE || v->type == TU_JSON_INT) {
        *out = tu_json_as_double(v); return true;
    }
    return false;
}

static bool parse_opt_bool(const tu_json_value_t *obj, const char *key, bool *out) {
    const tu_json_value_t *v = tu_json_get(obj, key);
    if (!v) return false;
    *out = tu_json_as_bool(v); return true;
}

static int parse_dataflow_str(const char *s) {
    if (!s) return -1;
    if (strcmp(s, "weight_stationary") == 0) return 0;
    if (strcmp(s, "output_stationary") == 0) return 1;
    if (strcmp(s, "row_stationary") == 0) return 2;
    /* NLR is reserved in the public enum but has no executable plugin. */
    return -1;
}

static int parse_dram_type_str(const char *s) {
    if (!s) return 0;
    if (strcmp(s, "ideal") == 0) return 0;
    if (strcmp(s, "hbm2") == 0) return 1;
    if (strcmp(s, "hbm2e") == 0) return 2;
    if (strcmp(s, "hbm3") == 0) return 3;
    if (strcmp(s, "ddr4") == 0) return 4;
    if (strcmp(s, "ddr5") == 0) return 5;
    if (strcmp(s, "lpddr5") == 0) return 6;
    return 0;
}

static int parse_cycle_model_str(const char *s) {
    if (!s) return 2;
    if (strcmp(s, "functional") == 0) return 0;
    if (strcmp(s, "estimated") == 0) return 1;
    if (strcmp(s, "cycle_accurate") == 0) return 2;
    return 2;
}

static int parse_rounding_str(const char *s) {
    if (!s) return 0;
    if (strcmp(s, "round_nearest_even") == 0) return 0;
    if (strcmp(s, "round_toward_zero") == 0) return 1;
    if (strcmp(s, "stochastic") == 0) return 2;
    return 0;
}

static int parse_conflict_str(const char *s) {
    if (!s) return 1;
    if (strcmp(s, "none") == 0) return 0;
    if (strcmp(s, "detect") == 0) return 1;
    if (strcmp(s, "stall_cycle") == 0) return 2;
    return 1;
}

static int parse_dma_bus_mode_str(const char *s) {
    if (!s || strcmp(s, "independent") == 0)
        return TU_DMA_CONFIG_BUS_INDEPENDENT;
    if (strcmp(s, "shared_serial") == 0)
        return TU_DMA_CONFIG_BUS_SHARED_SERIAL;
    return -1;
}

static int parse_dma_arb_policy_str(const char *s) {
    if (!s || strcmp(s, "round_robin") == 0)
        return TU_DMA_CONFIG_ARB_ROUND_ROBIN;
    if (strcmp(s, "strict_priority") == 0)
        return TU_DMA_CONFIG_ARB_STRICT_PRIORITY;
    return -1;
}

static int parse_dma_binding_policy_str(const char *s) {
    if (!s || strcmp(s, "explicit") == 0)
        return TU_DMA_CONFIG_BIND_EXPLICIT;
    if (strcmp(s, "round_robin") == 0)
        return TU_DMA_CONFIG_BIND_ROUND_ROBIN;
    if (strcmp(s, "least_outstanding") == 0)
        return TU_DMA_CONFIG_BIND_LEAST_OUTSTANDING;
    if (strcmp(s, "least_bytes") == 0)
        return TU_DMA_CONFIG_BIND_LEAST_BYTES;
    return -1;
}

static int parse_power_tech_node_str(const char *s) {
    if (!s || strcmp(s, "auto") == 0) return 0;
    if (strcmp(s, "45nm") == 0) return 1;
    if (strcmp(s, "28nm") == 0) return 2;
    if (strcmp(s, "16nm") == 0) return 3;
    if (strcmp(s, "7nm") == 0) return 4;
    if (strcmp(s, "5nm") == 0) return 5;
    if (strcmp(s, "3nm") == 0) return 6;
    return -1;
}

static int parse_dram_row_policy_str(const char *s) {
    if (!s || strcmp(s, "legacy") == 0) return TU_DRAM_CONFIG_ROW_LEGACY;
    if (strcmp(s, "open_page") == 0) return TU_DRAM_CONFIG_ROW_OPEN_PAGE;
    if (strcmp(s, "closed_page") == 0) return TU_DRAM_CONFIG_ROW_CLOSED_PAGE;
    if (strcmp(s, "adaptive_timeout") == 0) return TU_DRAM_CONFIG_ROW_ADAPTIVE_TIMEOUT;
    return -1;
}

static int parse_dram_address_mapping_str(const char *s) {
    if (!s || strcmp(s, "burst_interleaved") == 0)
        return TU_DRAM_CONFIG_ADDR_BURST_INTERLEAVED;
    if (strcmp(s, "row_interleaved") == 0)
        return TU_DRAM_CONFIG_ADDR_ROW_INTERLEAVED;
    if (strcmp(s, "xor_interleaved") == 0)
        return TU_DRAM_CONFIG_ADDR_XOR_INTERLEAVED;
    return -1;
}

static int parse_dram_latency_domain_str(const char *s) {
    if (!s || strcmp(s, "core_cycles") == 0)
        return TU_DRAM_CONFIG_LATENCY_CORE_CYCLES;
    if (strcmp(s, "physical_ns") == 0)
        return TU_DRAM_CONFIG_LATENCY_PHYSICAL_NS;
    return -1;
}

static int parse_dram_row_timeout_domain_str(const char *s) {
    if (!s || strcmp(s, "core_cycles") == 0)
        return TU_DRAM_CONFIG_ROW_TIMEOUT_CORE_CYCLES;
    if (strcmp(s, "physical_ns") == 0)
        return TU_DRAM_CONFIG_ROW_TIMEOUT_PHYSICAL_NS;
    return -1;
}

static int parse_dram_turnaround_mode_str(const char *s) {
    if (!s || strcmp(s, "none") == 0) return TU_DRAM_CONFIG_TURNAROUND_NONE;
    if (strcmp(s, "fixed") == 0) return TU_DRAM_CONFIG_TURNAROUND_FIXED;
    if (strcmp(s, "idle_credit") == 0) return TU_DRAM_CONFIG_TURNAROUND_IDLE_CREDIT;
    if (strcmp(s, "burst_credit") == 0) return TU_DRAM_CONFIG_TURNAROUND_BURST_CREDIT;
    if (strcmp(s, "burst_round_credit") == 0)
        return TU_DRAM_CONFIG_TURNAROUND_BURST_ROUND_CREDIT;
    if (strcmp(s, "burst_span_credit") == 0)
        return TU_DRAM_CONFIG_TURNAROUND_BURST_SPAN_CREDIT;
    return -1;
}

static int parse_dram_turnaround_domain_str(const char *s) {
    if (!s || strcmp(s, "core_cycles") == 0)
        return TU_DRAM_CONFIG_TURNAROUND_CORE_CYCLES;
    if (strcmp(s, "physical_ns") == 0)
        return TU_DRAM_CONFIG_TURNAROUND_PHYSICAL_NS;
    return -1;
}

static int parse_dram_refresh_mode_str(const char *s) {
    if (!s || strcmp(s, "none") == 0) return TU_DRAM_CONFIG_REFRESH_NONE;
    if (strcmp(s, "all_bank") == 0) return TU_DRAM_CONFIG_REFRESH_ALL_BANK;
    if (strcmp(s, "per_bank") == 0) return TU_DRAM_CONFIG_REFRESH_PER_BANK;
    return -1;
}

static int parse_dram_refresh_scheduling_str(const char *s) {
    if (!s || strcmp(s, "fixed") == 0) return TU_DRAM_CONFIG_REFRESH_SCHED_FIXED;
    if (strcmp(s, "deferred") == 0) return TU_DRAM_CONFIG_REFRESH_SCHED_DEFERRED;
    return -1;
}

/* ---- Default configuration ---- */

void tu_config_default(struct tu_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->pe_rows             = 16;
    cfg->pe_cols             = 16;
    cfg->pe_pipeline_depth   = 2;
    cfg->mac_units_per_pe    = 1;
    cfg->dataflow_mode       = 0;
    cfg->dataflow_via_plugin = true;

    cfg->fp16_enabled        = true;
    cfg->fp32_enabled        = true;
    cfg->bf16_enabled        = false;
    cfg->fp8_e4m3_enabled    = false;
    cfg->fp8_e5m2_enabled    = false;
    cfg->int8_enabled        = true;
    cfg->int4_enabled        = true;
    cfg->rounding_mode       = 0;
    cfg->subnormal_flush     = true;
    cfg->saturate            = true;

    cfg->sram_w_size_kb      = 128;
    cfg->sram_a_size_kb      = 64;
    cfg->sram_o_size_kb      = 64;
    cfg->sram_num_banks      = 32;
    cfg->sram_bank_width     = 4;
    cfg->sram_words_per_cycle = 1;
    cfg->sram_arb_mode       = 1;
    cfg->sram_conflict_mode  = 1;
    cfg->sram_stall_penalty  = 2;
    cfg->sram_bw_window_cycles = 4;
    cfg->sram_bw_modeling    = true;

    cfg->gbuf_size_kb        = 1024;
    cfg->gbuf_banks          = 16;
    cfg->gbuf_bank_width     = 8;

    cfg->dram_type           = 0;
    cfg->dram_bandwidth_gbps = 256.0;
    cfg->dram_channels       = 8;
    cfg->dram_model_row_conflicts = false;
    cfg->dram_row_policy      = TU_DRAM_CONFIG_ROW_LEGACY;
    cfg->dram_address_mapping = TU_DRAM_CONFIG_ADDR_BURST_INTERLEAVED;
    cfg->dram_row_miss_penalty_cycles = 10;
    cfg->dram_row_conflict_penalty_cycles = 0; /* inherit miss cost: compatibility */
    cfg->dram_row_open_timeout_cycles = 100;
    cfg->dram_row_open_timeout_ns = 100.0;
    cfg->dram_row_timeout_domain = TU_DRAM_CONFIG_ROW_TIMEOUT_CORE_CYCLES;
    cfg->dram_latency_read   = 50;
    cfg->dram_latency_write  = 50;
    cfg->dram_latency_domain = TU_DRAM_CONFIG_LATENCY_CORE_CYCLES;
    cfg->dram_core_clock_ghz = 1.0;
    cfg->dram_turnaround_mode = TU_DRAM_CONFIG_TURNAROUND_NONE;
    cfg->dram_turnaround_domain = TU_DRAM_CONFIG_TURNAROUND_CORE_CYCLES;
    cfg->dram_read_to_write_turnaround = 0.0;
    cfg->dram_write_to_read_turnaround = 0.0;
    cfg->dram_read_burst_bytes = TU_DRAM_READ_BURST_BYTES;
    cfg->dram_write_burst_bytes = TU_DRAM_WRITE_BURST_BYTES;
    cfg->dram_refresh_mode       = TU_DRAM_CONFIG_REFRESH_NONE;
    cfg->dram_refresh_scheduling = TU_DRAM_CONFIG_REFRESH_SCHED_FIXED;
    cfg->dram_refresh_rate       = 1;
    cfg->dram_trefi_ns           = 7800;
    cfg->dram_trfc_ns            = 350;
    cfg->dram_trfc_pb_ns         = 90;
    cfg->dram_refresh_max_deferral_ns = 7800;

    cfg->dma_bus_width_bits  = 256;
    cfg->dma_max_burst_bytes = 64;
    cfg->dma_num_channels    = 3;
    cfg->dma_bus_mode        = TU_DMA_CONFIG_BUS_INDEPENDENT;
    cfg->dma_arb_policy      = TU_DMA_CONFIG_ARB_ROUND_ROBIN;
    cfg->dma_binding_policy  = TU_DMA_CONFIG_BIND_EXPLICIT;
    cfg->dma_max_outstanding = 4;
    cfg->dma_async_mode      = false;
    cfg->dma_multicast_enabled = false;

    /* Conservative default preserves legacy uncompressed DMA traffic. */
    cfg->compression_enabled = false;
    cfg->compression_type = 0;
    cfg->compression_rle_epsilon = 0.0;
    cfg->compression_decoder_enabled = false;
    cfg->compression_decoder_overlap_dma = true;
    cfg->compression_decoder_elements_per_cycle = 1;
    cfg->compression_rle_runs_per_cycle = 1;
    cfg->compression_bitmap_elements_per_cycle = 1;

    cfg->isa_instr_width_bits = 96;
    cfg->isa_queue_depth     = 16;
    cfg->isa_dep_checking    = false;

    cfg->multicore_enabled   = false;
    cfg->num_cores           = 1;
    cfg->interconnect_mode   = 0;
    cfg->icc_switching_mode  = TU_ICC_SWITCH_LEGACY_HOP_ONLY;
    cfg->icc_contention_mode = TU_ICC_CONTENTION_IDEAL_PARALLEL;
    cfg->icc_mesh_routing_mode = TU_ICC_MESH_ROUTE_XY;
    cfg->icc_link_bytes_per_cycle = TU_ICC_LINK_BYTES_PER_CYCLE;
    cfg->icc_router_latency_cycles = TU_ICC_ROUTER_LATENCY_CYCLES;

    cfg->cycle_model         = 2;
    cfg->counters_enabled    = true;
    cfg->detailed_stalls     = false;
    cfg->trace_enabled       = false;
    cfg->trace_file[0]       = '\0';
    cfg->trace_max_events    = 65536;

    /* AUTO preserves the historical process/frequency heuristic. */
    cfg->power_tech_node      = 0;
    cfg->power_clock_freq_mhz = 0.0;

    cfg->sparsity_enabled    = false;
    cfg->sparsity_2of4       = false;
    cfg->sparsity_unstructured = false;
    cfg->sparsity_metadata_format = 0;
    cfg->sparsity_decoder_groups_per_cycle = 1;

    cfg->golden_reference    = 1;
    cfg->random_test_iters   = 1000;
    cfg->error_tolerance     = 1e-5;

    cfg->log_level           = 3;
}

/* ---- Convert to legacy ---- */

tu_runtime_config_t tu_config_to_runtime(const struct tu_config_t *cfg) {
    tu_runtime_config_t rt; memset(&rt, 0, sizeof(rt));
    rt.pe_rows          = cfg->pe_rows;
    rt.pe_cols          = cfg->pe_cols;
    rt.pe_pipeline_depth = cfg->pe_pipeline_depth;
    rt.dataflow_mode    = cfg->dataflow_mode;
    rt.sram_w_size      = cfg->sram_w_size_kb * 1024;
    rt.sram_a_size      = cfg->sram_a_size_kb * 1024;
    rt.sram_o_size      = cfg->sram_o_size_kb * 1024;
    rt.sram_num_banks   = cfg->sram_num_banks;
    rt.sram_bank_width  = cfg->sram_bank_width;
    rt.sram_words_per_cycle = (uint8_t)cfg->sram_words_per_cycle;
    rt.sram_stall_penalty = cfg->sram_stall_penalty;
    rt.sram_bw_window_cycles = cfg->sram_bw_window_cycles;
    rt.dma_num_channels = cfg->dma_num_channels;
    rt.dma_bus_mode = cfg->dma_bus_mode;
    rt.dma_arb_policy = cfg->dma_arb_policy;
    rt.dma_binding_policy = cfg->dma_binding_policy;
    rt.dma_max_outstanding = cfg->dma_max_outstanding;
    rt.dma_async_mode = cfg->dma_async_mode;
    rt.counters_enabled = cfg->counters_enabled;
    rt.detailed_stalls  = cfg->detailed_stalls;
    rt.trace_enabled    = cfg->trace_enabled;
    memcpy(rt.trace_file, cfg->trace_file, sizeof(rt.trace_file));
    rt.verify_enabled   = cfg->golden_reference >= 0;
    rt.verify_tolerance = cfg->error_tolerance;
    rt.icc_switching_mode = cfg->icc_switching_mode;
    rt.icc_contention_mode = cfg->icc_contention_mode;
    rt.icc_mesh_routing_mode = cfg->icc_mesh_routing_mode;
    rt.icc_link_bytes_per_cycle = cfg->icc_link_bytes_per_cycle;
    rt.icc_router_latency_cycles = cfg->icc_router_latency_cycles;
    return rt;
}

/* ---- Load from JSON string ---- */

int tu_config_load_string(const char *json_str, struct tu_config_t *cfg,
                          char *error_buf, size_t error_size) {
    if (!json_str || !cfg) {
        if (error_buf && error_size > 0) snprintf(error_buf, error_size, "null input");
        return -1;
    }
    tu_config_default(cfg);

    char *copy = strdup(json_str);
    if (!copy) return -1;

    tu_json_value_t root;
    tu_json_error_t err = tu_json_parse(copy, &root, error_buf, error_size);
    if (err != TU_JSON_OK) { free(copy); return -1; }

    const tu_json_value_t *tu = tu_json_get(&root, "tu");
    if (!tu) tu = &root;

    /* Compute */
    const tu_json_value_t *c = tu_json_get(tu, "compute");
    if (c) {
        const tu_json_value_t *pe = tu_json_get(c, "pe_array");
        if (pe) {
            parse_opt_uint16(pe, "rows", &cfg->pe_rows, 16);
            parse_opt_uint16(pe, "cols", &cfg->pe_cols, 16);
            const tu_json_value_t *df = tu_json_get(pe, "dataflow");
            if (df && df->type == TU_JSON_STRING)
                cfg->dataflow_mode = parse_dataflow_str(tu_json_as_string(df, NULL));
            int64_t pipeline_depth;
            if (parse_opt_int64(pe, "pipeline_depth", &pipeline_depth))
                cfg->pe_pipeline_depth =
                    (pipeline_depth >= 1 && pipeline_depth <= 16) ?
                    (uint16_t)pipeline_depth : 0;
        }
        parse_opt_uint16(c, "mac_units_per_pe", &cfg->mac_units_per_pe, 16);
        const tu_json_value_t *sp = tu_json_get(c, "supported_precisions");
        if (sp && sp->type == TU_JSON_ARRAY) {
            cfg->fp16_enabled = false; cfg->fp32_enabled = false;
            cfg->bf16_enabled = false; cfg->fp8_e4m3_enabled = false; cfg->fp8_e5m2_enabled = false;
            for (uint32_t i = 0; i < sp->array.count; i++) {
                const char *s = tu_json_as_string(&sp->array.items[i], NULL);
                if (!s) continue;
                if (strcmp(s, "fp16") == 0) cfg->fp16_enabled = true;
                else if (strcmp(s, "fp32") == 0) cfg->fp32_enabled = true;
                else if (strcmp(s, "bf16") == 0) cfg->bf16_enabled = true;
                else if (strcmp(s, "fp8_e4m3") == 0) cfg->fp8_e4m3_enabled = true;
                else if (strcmp(s, "fp8_e5m2") == 0) cfg->fp8_e5m2_enabled = true;
            }
        }
    }

    /* Memory */
    const tu_json_value_t *m = tu_json_get(tu, "memory");
    if (m) {
        const tu_json_value_t *sram = tu_json_get(m, "sram");
        if (sram) {
            int64_t iv;
            if (parse_opt_int64(sram, "w_buffer_kb", &iv)) cfg->sram_w_size_kb = (uint32_t)iv;
            if (parse_opt_int64(sram, "a_buffer_kb", &iv)) cfg->sram_a_size_kb = (uint32_t)iv;
            if (parse_opt_int64(sram, "o_buffer_kb", &iv)) cfg->sram_o_size_kb = (uint32_t)iv;
        }
        const tu_json_value_t *bank = tu_json_get(m, "banking");
        if (bank) {
            int64_t iv;
            if (parse_opt_int64(bank, "banks", &iv)) cfg->sram_num_banks = (uint32_t)iv;
            if (parse_opt_int64(bank, "bank_width_bytes", &iv)) cfg->sram_bank_width = (uint32_t)iv;
            if (parse_opt_int64(bank, "words_per_refill", &iv))
                cfg->sram_words_per_cycle =
                    (iv >= 1 && iv <= UINT8_MAX) ? (uint32_t)iv : 0;
            if (parse_opt_int64(bank, "refill_window_cycles", &iv))
                cfg->sram_bw_window_cycles =
                    (iv >= 1 && iv <= 1000000) ? (uint64_t)iv : 0;
            if (parse_opt_int64(bank, "stall_penalty_cycles", &iv))
                cfg->sram_stall_penalty =
                    (iv >= 1 && iv <= UINT8_MAX) ? (uint8_t)iv : 0;
            const tu_json_value_t *cm = tu_json_get(bank, "conflict_model");
            if (cm && cm->type == TU_JSON_STRING)
                cfg->sram_conflict_mode = parse_conflict_str(tu_json_as_string(cm, NULL));
        }
        const tu_json_value_t *lat = tu_json_get(m, "latency");
        if (lat) {
            parse_opt_double(lat, "dram_read", &cfg->dram_latency_read);
            parse_opt_double(lat, "dram_write", &cfg->dram_latency_write);
        }
        const tu_json_value_t *dram = tu_json_get(m, "dram");
        if (dram) {
            const tu_json_value_t *dt = tu_json_get(dram, "type");
            if (dt && dt->type == TU_JSON_STRING)
                cfg->dram_type = parse_dram_type_str(tu_json_as_string(dt, NULL));
            parse_opt_double(dram, "bandwidth_gbps", &cfg->dram_bandwidth_gbps);
            double core_clock_ghz;
            if (parse_opt_double(dram, "core_clock_ghz", &core_clock_ghz))
                cfg->dram_core_clock_ghz =
                    (core_clock_ghz > 0.0 && core_clock_ghz <= 10.0)
                        ? core_clock_ghz : -1.0;
            int64_t dram_channels;
            if (parse_opt_int64(dram, "channels", &dram_channels))
                cfg->dram_channels = (dram_channels > 0 && dram_channels <= 1024)
                    ? (uint32_t)dram_channels : 0;
            parse_opt_bool(dram, "model_row_conflicts", &cfg->dram_model_row_conflicts);
            const tu_json_value_t *rp = tu_json_get(dram, "row_policy");
            if (rp && rp->type == TU_JSON_STRING)
                cfg->dram_row_policy = parse_dram_row_policy_str(tu_json_as_string(rp, NULL));
            const tu_json_value_t *am = tu_json_get(dram, "address_mapping");
            if (am && am->type == TU_JSON_STRING)
                cfg->dram_address_mapping =
                    parse_dram_address_mapping_str(tu_json_as_string(am, NULL));
            int64_t row_penalty;
            if (parse_opt_int64(dram, "row_miss_penalty_cycles", &row_penalty))
                cfg->dram_row_miss_penalty_cycles =
                    (row_penalty >= 0 && row_penalty <= 1000000) ? (uint32_t)row_penalty : UINT32_MAX;
            if (parse_opt_int64(dram, "row_conflict_penalty_cycles", &row_penalty))
                cfg->dram_row_conflict_penalty_cycles =
                    (row_penalty >= 0 && row_penalty <= 1000000) ? (uint32_t)row_penalty : UINT32_MAX;
            if (parse_opt_int64(dram, "row_open_timeout_cycles", &row_penalty))
                cfg->dram_row_open_timeout_cycles =
                    (row_penalty >= 0 && row_penalty <= 1000000) ? (uint32_t)row_penalty : UINT32_MAX;
            parse_opt_double(dram, "row_open_timeout_ns", &cfg->dram_row_open_timeout_ns);
            const tu_json_value_t *td = tu_json_get(dram, "row_timeout_domain");
            if (td && td->type == TU_JSON_STRING)
                cfg->dram_row_timeout_domain =
                    parse_dram_row_timeout_domain_str(tu_json_as_string(td, NULL));
            const tu_json_value_t *ld = tu_json_get(dram, "latency_domain");
            if (ld && ld->type == TU_JSON_STRING)
                cfg->dram_latency_domain =
                    parse_dram_latency_domain_str(tu_json_as_string(ld, NULL));
            const tu_json_value_t *tm = tu_json_get(dram, "turnaround_mode");
            if (tm && tm->type == TU_JSON_STRING)
                cfg->dram_turnaround_mode =
                    parse_dram_turnaround_mode_str(tu_json_as_string(tm, NULL));
            const tu_json_value_t *tad = tu_json_get(dram, "turnaround_domain");
            if (tad && tad->type == TU_JSON_STRING)
                cfg->dram_turnaround_domain =
                    parse_dram_turnaround_domain_str(tu_json_as_string(tad, NULL));
            parse_opt_double(dram, "read_to_write_turnaround",
                             &cfg->dram_read_to_write_turnaround);
            parse_opt_double(dram, "write_to_read_turnaround",
                             &cfg->dram_write_to_read_turnaround);
            int64_t burst_bytes;
            if (parse_opt_int64(dram, "read_burst_bytes", &burst_bytes))
                cfg->dram_read_burst_bytes = (burst_bytes >= 0 && burst_bytes <= 1048576)
                    ? (uint32_t)burst_bytes : UINT32_MAX;
            if (parse_opt_int64(dram, "write_burst_bytes", &burst_bytes))
                cfg->dram_write_burst_bytes = (burst_bytes >= 0 && burst_bytes <= 1048576)
                    ? (uint32_t)burst_bytes : UINT32_MAX;
            const tu_json_value_t *rf = tu_json_get(dram, "refresh");
            if (rf && rf->type == TU_JSON_OBJECT) {
                const tu_json_value_t *rm = tu_json_get(rf, "mode");
                if (rm && rm->type == TU_JSON_STRING)
                    cfg->dram_refresh_mode =
                        parse_dram_refresh_mode_str(tu_json_as_string(rm, NULL));
                const tu_json_value_t *rs = tu_json_get(rf, "scheduling");
                if (rs && rs->type == TU_JSON_STRING)
                    cfg->dram_refresh_scheduling =
                        parse_dram_refresh_scheduling_str(tu_json_as_string(rs, NULL));
                int64_t rv;
                if (parse_opt_int64(rf, "rate", &rv))
                    cfg->dram_refresh_rate =
                        (rv >= 0 && rv <= 4) ? (uint32_t)rv : UINT32_MAX;
                if (parse_opt_int64(rf, "trefi_ns", &rv))
                    cfg->dram_trefi_ns =
                        (rv >= 0 && rv <= 100000000) ? (uint32_t)rv : UINT32_MAX;
                if (parse_opt_int64(rf, "trfc_ns", &rv))
                    cfg->dram_trfc_ns =
                        (rv >= 0 && rv <= 100000000) ? (uint32_t)rv : UINT32_MAX;
                if (parse_opt_int64(rf, "trfc_pb_ns", &rv))
                    cfg->dram_trfc_pb_ns =
                        (rv >= 0 && rv <= 100000000) ? (uint32_t)rv : UINT32_MAX;
                if (parse_opt_int64(rf, "max_deferral_ns", &rv))
                    cfg->dram_refresh_max_deferral_ns =
                        (rv >= 0 && rv <= 100000000) ? (uint32_t)rv : UINT32_MAX;
            }
        }
    }

    /* DMA */
    const tu_json_value_t *d = tu_json_get(tu, "dma");
    if (d) {
        int64_t iv;
        if (parse_opt_int64(d, "bus_width_bits", &iv)) cfg->dma_bus_width_bits = (uint32_t)iv;
        if (parse_opt_int64(d, "max_burst_bytes", &iv)) cfg->dma_max_burst_bytes = (uint32_t)iv;
        if (parse_opt_int64(d, "channels", &iv)) cfg->dma_num_channels = (uint32_t)iv;
        const tu_json_value_t *bus_mode = tu_json_get(d, "bus_topology");
        if (bus_mode && bus_mode->type == TU_JSON_STRING)
            cfg->dma_bus_mode = parse_dma_bus_mode_str(tu_json_as_string(bus_mode, NULL));
        const tu_json_value_t *arb = tu_json_get(d, "arbitration");
        if (arb && arb->type == TU_JSON_STRING)
            cfg->dma_arb_policy = parse_dma_arb_policy_str(tu_json_as_string(arb, NULL));
        const tu_json_value_t *binding = tu_json_get(d, "channel_binding");
        if (binding && binding->type == TU_JSON_STRING)
            cfg->dma_binding_policy = parse_dma_binding_policy_str(
                tu_json_as_string(binding, NULL));
        if (parse_opt_int64(d, "max_outstanding", &iv)) cfg->dma_max_outstanding = (uint32_t)iv;
        parse_opt_bool(d, "async_mode", &cfg->dma_async_mode);
        parse_opt_bool(d, "multicast_enabled", &cfg->dma_multicast_enabled);
    }

    /* Codec placement is architecture-dependent, so compression is modeled
     * as its own runtime block rather than an implicit DMA behavior. */
    const tu_json_value_t *wc = tu_json_get(tu, "weight_compression");
    if (wc) {
        parse_opt_bool(wc, "enabled", &cfg->compression_enabled);
        const tu_json_value_t *ct = tu_json_get(wc, "type");
        if (ct && ct->type == TU_JSON_STRING) {
            const char *s = tu_json_as_string(ct, NULL);
            if (strcmp(s, "none") == 0) cfg->compression_type = 0;
            else if (strcmp(s, "rle") == 0) cfg->compression_type = 1;
            else if (strcmp(s, "adaptive_rle") == 0) cfg->compression_type = 2;
            else if (strcmp(s, "bitmap") == 0) cfg->compression_type = 3;
            else if (strcmp(s, "adaptive") == 0) cfg->compression_type = 4;
            else cfg->compression_type = -1;
        }
        parse_opt_double(wc, "rle_epsilon", &cfg->compression_rle_epsilon);
        parse_opt_bool(wc, "decoder_enabled", &cfg->compression_decoder_enabled);
        parse_opt_bool(wc, "decoder_overlap_dma", &cfg->compression_decoder_overlap_dma);
        int64_t iv;
        if (parse_opt_int64(wc, "decoder_elements_per_cycle", &iv))
            cfg->compression_decoder_elements_per_cycle =
                (iv > 0 && iv <= 1048576) ? (uint32_t)iv : 0;
        if (parse_opt_int64(wc, "rle_runs_per_cycle", &iv))
            cfg->compression_rle_runs_per_cycle =
                (iv > 0 && iv <= 1048576) ? (uint32_t)iv : 0;
        if (parse_opt_int64(wc, "bitmap_elements_per_cycle", &iv))
            cfg->compression_bitmap_elements_per_cycle =
                (iv > 0 && iv <= 1048576) ? (uint32_t)iv : 0;
    }

    /* ISA */
    const tu_json_value_t *isa = tu_json_get(tu, "isa");
    if (isa) {
        int64_t iv;
        if (parse_opt_int64(isa, "instruction_width_bits", &iv)) cfg->isa_instr_width_bits = (uint32_t)iv;
        if (parse_opt_int64(isa, "queue_depth", &iv)) cfg->isa_queue_depth = (uint32_t)iv;
        parse_opt_bool(isa, "dependency_checking", &cfg->isa_dep_checking);
    }

    /* Multicore */
    const tu_json_value_t *mc = tu_json_get(tu, "multicore");
    if (mc) {
        parse_opt_bool(mc, "enabled", &cfg->multicore_enabled);
        int64_t iv;
        if (parse_opt_int64(mc, "num_cores", &iv)) cfg->num_cores = (uint32_t)iv;
        const tu_json_value_t *ic = tu_json_get(mc, "interconnect");
        if (ic && ic->type == TU_JSON_STRING) {
            const char *s = tu_json_as_string(ic, NULL);
            if (strcmp(s, "none") == 0) cfg->interconnect_mode = 0;
            else if (strcmp(s, "ring") == 0) cfg->interconnect_mode = 1;
            else if (strcmp(s, "mesh") == 0) cfg->interconnect_mode = 2;
            else cfg->interconnect_mode = -1;
        }
        const tu_json_value_t *sw = tu_json_get(mc, "switching");
        if (sw && sw->type == TU_JSON_STRING) {
            const char *s = tu_json_as_string(sw, NULL);
            if (strcmp(s, "legacy_hop_only") == 0) cfg->icc_switching_mode = TU_ICC_SWITCH_LEGACY_HOP_ONLY;
            else if (strcmp(s, "cut_through") == 0) cfg->icc_switching_mode = TU_ICC_SWITCH_CUT_THROUGH;
            else if (strcmp(s, "store_and_forward") == 0) cfg->icc_switching_mode = TU_ICC_SWITCH_STORE_FORWARD;
            else cfg->icc_switching_mode = -1;
        }
        const tu_json_value_t *cm = tu_json_get(mc, "contention");
        if (cm && cm->type == TU_JSON_STRING) {
            const char *s = tu_json_as_string(cm, NULL);
            if (strcmp(s, "ideal_parallel") == 0)
                cfg->icc_contention_mode = TU_ICC_CONTENTION_IDEAL_PARALLEL;
            else if (strcmp(s, "shared_link") == 0)
                cfg->icc_contention_mode = TU_ICC_CONTENTION_SHARED_LINK;
            else
                cfg->icc_contention_mode = -1;
        }
        const tu_json_value_t *mr = tu_json_get(mc, "mesh_routing");
        if (mr && mr->type == TU_JSON_STRING) {
            const char *s = tu_json_as_string(mr, NULL);
            if (strcmp(s, "xy") == 0)
                cfg->icc_mesh_routing_mode = TU_ICC_MESH_ROUTE_XY;
            else if (strcmp(s, "yx") == 0)
                cfg->icc_mesh_routing_mode = TU_ICC_MESH_ROUTE_YX;
            else
                cfg->icc_mesh_routing_mode = -1;
        }
        if (parse_opt_int64(mc, "link_bytes_per_cycle", &iv))
            cfg->icc_link_bytes_per_cycle = (iv > 0 && iv <= 1048576) ? (uint32_t)iv : 0;
        if (parse_opt_int64(mc, "router_latency_cycles", &iv))
            cfg->icc_router_latency_cycles = (iv >= 0 && iv <= 1048576) ? (uint32_t)iv : UINT32_MAX;
    }

    /* Performance */
    const tu_json_value_t *p = tu_json_get(tu, "performance");
    if (p) {
        const tu_json_value_t *cm = tu_json_get(p, "cycle_model");
        if (cm && cm->type == TU_JSON_STRING)
            cfg->cycle_model = parse_cycle_model_str(tu_json_as_string(cm, NULL));
        const tu_json_value_t *cnt = tu_json_get(p, "counters");
        if (cnt) {
            parse_opt_bool(cnt, "enabled", &cfg->counters_enabled);
            parse_opt_bool(cnt, "detailed_stalls", &cfg->detailed_stalls);
        }
        const tu_json_value_t *trc = tu_json_get(p, "tracing");
        if (trc) {
            parse_opt_bool(trc, "enabled", &cfg->trace_enabled);
            parse_opt_string(trc, "output_file", cfg->trace_file, sizeof(cfg->trace_file));
        }
    }

    /* Power-model process and clock are explicit physical assumptions. */
    const tu_json_value_t *power = tu_json_get(tu, "power");
    if (power) {
        const tu_json_value_t *tn = tu_json_get(power, "tech_node");
        if (tn && tn->type == TU_JSON_STRING)
            cfg->power_tech_node = parse_power_tech_node_str(tu_json_as_string(tn, NULL));
        parse_opt_double(power, "clock_freq_mhz", &cfg->power_clock_freq_mhz);
    }

    /* Precision */
    const tu_json_value_t *pr = tu_json_get(tu, "precision");
    if (pr) {
        const tu_json_value_t *fp = tu_json_get(pr, "fp16");
        if (fp) {
            const tu_json_value_t *rm = tu_json_get(fp, "rounding");
            if (rm && rm->type == TU_JSON_STRING)
                cfg->rounding_mode = parse_rounding_str(tu_json_as_string(rm, NULL));
            const tu_json_value_t *sn = tu_json_get(fp, "subnormal");
            if (sn && sn->type == TU_JSON_STRING)
                cfg->subnormal_flush = (strcmp(tu_json_as_string(sn, NULL), "flush_to_zero") == 0);
            parse_opt_bool(fp, "saturate", &cfg->saturate);
        }
    }

    /* Sparsity */
    const tu_json_value_t *sp = tu_json_get(tu, "sparsity");
    if (sp) {
        parse_opt_bool(sp, "enabled", &cfg->sparsity_enabled);
        parse_opt_bool(sp, "structured_2of4", &cfg->sparsity_2of4);
        parse_opt_bool(sp, "unstructured", &cfg->sparsity_unstructured);
        int64_t iv;
        if (parse_opt_int64(sp, "decoder_groups_per_cycle", &iv))
            cfg->sparsity_decoder_groups_per_cycle =
                (iv > 0 && iv <= 1048576) ? (uint32_t)iv : 0;
    }

    /* Verification */
    const tu_json_value_t *v = tu_json_get(tu, "verification");
    if (v) {
        const tu_json_value_t *gr = tu_json_get(v, "golden_reference");
        if (gr && gr->type == TU_JSON_STRING) {
            const char *s = tu_json_as_string(gr, NULL);
            if (strcmp(s, "numpy") == 0) cfg->golden_reference = 0;
            else if (strcmp(s, "pytorch") == 0) cfg->golden_reference = 1;
        }
        int64_t iv;
        if (parse_opt_int64(v, "random_test_iterations", &iv)) cfg->random_test_iters = (uint32_t)iv;
        parse_opt_double(v, "error_tolerance", &cfg->error_tolerance);
    }

    tu_json_free(&root);
    free(copy);
    return tu_config_validate(cfg, error_buf, error_size);
}

/* ---- Load from file ---- */

int tu_config_load(const char *path, struct tu_config_t *cfg,
                   char *error_buf, size_t error_size) {
    char *json_str = read_file(path, error_buf, error_size);
    if (!json_str) return -1;
    int r = tu_config_load_string(json_str, cfg, error_buf, error_size);
    free(json_str);
    return r;
}

/* ---- Validation ---- */

int tu_config_validate(const struct tu_config_t *cfg, char *error_buf, size_t error_size) {
    if (!cfg) return -1;
    if (cfg->pe_rows < 1 || cfg->pe_rows > 1024) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "pe_rows must be [1,1024], got %u", cfg->pe_rows);
        return -1;
    }
    if (cfg->pe_cols < 1 || cfg->pe_cols > 1024) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "pe_cols must be [1,1024], got %u", cfg->pe_cols);
        return -1;
    }
    if (cfg->pe_pipeline_depth < 1 || cfg->pe_pipeline_depth > 16) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "pe pipeline_depth must be [1,16], got %u",
                     cfg->pe_pipeline_depth);
        return -1;
    }
    if (cfg->dataflow_mode < 0 || cfg->dataflow_mode > 2) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "dataflow must be weight_stationary, output_stationary, or row_stationary");
        return -1;
    }
    if (cfg->sram_w_size_kb == 0 || cfg->sram_a_size_kb == 0 || cfg->sram_o_size_kb == 0) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "SRAM buffers must be > 0 KB");
        return -1;
    }
    if (cfg->sram_bank_width != 1 && cfg->sram_bank_width != 2 &&
        cfg->sram_bank_width != 4 && cfg->sram_bank_width != 8) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "bank_width must be 1/2/4/8, got %u", cfg->sram_bank_width);
        return -1;
    }
    if (cfg->sram_num_banks < 1 || cfg->sram_num_banks > 1024) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "num_banks must be [1,1024], got %u", cfg->sram_num_banks);
        return -1;
    }
    if (cfg->sram_words_per_cycle < 1 || cfg->sram_words_per_cycle > UINT8_MAX) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "SRAM words_per_refill must be [1,255]");
        return -1;
    }
    if (cfg->sram_bw_window_cycles < 1 ||
        cfg->sram_bw_window_cycles > 1000000) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "SRAM refill_window_cycles must be [1,1000000]");
        return -1;
    }
    if (cfg->sram_stall_penalty < 1) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "SRAM stall_penalty_cycles must be [1,255]");
        return -1;
    }
    uint32_t bw = cfg->dma_bus_width_bits;
    if (bw < 32 || bw > 1024 || (bw & (bw - 1)) != 0) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "dma_bus_width must be power of 2 [32,1024], got %u", bw);
        return -1;
    }
    if (cfg->dma_num_channels < 1 ||
        cfg->dma_num_channels > TU_DMA_ENGINE_MAX_CHANNELS) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "DMA channels must be [1,%u], got %u",
                     TU_DMA_ENGINE_MAX_CHANNELS, cfg->dma_num_channels);
        return -1;
    }
    if (cfg->dma_max_outstanding < 1 || cfg->dma_max_outstanding > 65535) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DMA max_outstanding must be [1,65535], got %u",
                     cfg->dma_max_outstanding);
        return -1;
    }
    if (cfg->dma_bus_mode < TU_DMA_CONFIG_BUS_INDEPENDENT ||
        cfg->dma_bus_mode > TU_DMA_CONFIG_BUS_SHARED_SERIAL) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DMA bus_topology must be independent or shared_serial");
        return -1;
    }
    if (cfg->dma_arb_policy < TU_DMA_CONFIG_ARB_ROUND_ROBIN ||
        cfg->dma_arb_policy > TU_DMA_CONFIG_ARB_STRICT_PRIORITY) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DMA arbitration must be round_robin or strict_priority");
        return -1;
    }
    if (cfg->dma_binding_policy < TU_DMA_CONFIG_BIND_EXPLICIT ||
        cfg->dma_binding_policy > TU_DMA_CONFIG_BIND_LEAST_BYTES) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DMA channel_binding must be explicit, round_robin, least_outstanding, or least_bytes");
        return -1;
    }
    if (cfg->isa_queue_depth == 0) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "queue_depth must be > 0");
        return -1;
    }
    if (cfg->dram_row_policy < TU_DRAM_CONFIG_ROW_LEGACY ||
        cfg->dram_row_policy > TU_DRAM_CONFIG_ROW_ADAPTIVE_TIMEOUT) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM row_policy must be legacy, open_page, closed_page, or adaptive_timeout");
        return -1;
    }
    if (cfg->dram_channels == 0 || cfg->dram_channels > 1024) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "DRAM channels must be in [1,1024]");
        return -1;
    }
    if (cfg->dram_address_mapping < TU_DRAM_CONFIG_ADDR_BURST_INTERLEAVED ||
        cfg->dram_address_mapping > TU_DRAM_CONFIG_ADDR_XOR_INTERLEAVED) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM address_mapping must be burst_interleaved, row_interleaved, or xor_interleaved");
        return -1;
    }
    if (cfg->dram_address_mapping == TU_DRAM_CONFIG_ADDR_XOR_INTERLEAVED &&
        (cfg->dram_channels == 0 ||
         (cfg->dram_channels & (cfg->dram_channels - 1)) != 0)) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM xor_interleaved mapping requires a power-of-two channel count");
        return -1;
    }
    if (cfg->dram_row_miss_penalty_cycles == UINT32_MAX ||
        cfg->dram_row_conflict_penalty_cycles == UINT32_MAX ||
        cfg->dram_row_open_timeout_cycles == UINT32_MAX ||
        (cfg->dram_row_policy == TU_DRAM_CONFIG_ROW_ADAPTIVE_TIMEOUT &&
         ((cfg->dram_row_timeout_domain == TU_DRAM_CONFIG_ROW_TIMEOUT_CORE_CYCLES &&
           cfg->dram_row_open_timeout_cycles == 0) ||
          (cfg->dram_row_timeout_domain == TU_DRAM_CONFIG_ROW_TIMEOUT_PHYSICAL_NS &&
           !(cfg->dram_row_open_timeout_ns > 0.0))))) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "DRAM row timing/timeout value is out of range");
        return -1;
    }
    if (cfg->dram_row_timeout_domain < TU_DRAM_CONFIG_ROW_TIMEOUT_CORE_CYCLES ||
        cfg->dram_row_timeout_domain > TU_DRAM_CONFIG_ROW_TIMEOUT_PHYSICAL_NS) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM row_timeout_domain must be core_cycles or physical_ns");
        return -1;
    }
    if (!isfinite(cfg->dram_row_open_timeout_ns) ||
        cfg->dram_row_open_timeout_ns < 0.0 ||
        cfg->dram_row_open_timeout_ns > 100000000.0) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "DRAM row timeout ns value is out of range");
        return -1;
    }
    if (cfg->dram_latency_domain < TU_DRAM_CONFIG_LATENCY_CORE_CYCLES ||
        cfg->dram_latency_domain > TU_DRAM_CONFIG_LATENCY_PHYSICAL_NS) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM latency_domain must be core_cycles or physical_ns");
        return -1;
    }
    if (!(cfg->dram_latency_read >= 0.0 && cfg->dram_latency_read <= 100000000.0) ||
        !(cfg->dram_latency_write >= 0.0 && cfg->dram_latency_write <= 100000000.0)) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "DRAM latency value is out of range");
        return -1;
    }
    if (cfg->dram_core_clock_ghz != 0.0 &&
        !(cfg->dram_core_clock_ghz > 0.0 && cfg->dram_core_clock_ghz <= 10.0)) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM core_clock_ghz must be 0 (compatibility 1 GHz) or in (0,10]");
        return -1;
    }
    if (cfg->dram_turnaround_mode < TU_DRAM_CONFIG_TURNAROUND_NONE ||
        cfg->dram_turnaround_mode > TU_DRAM_CONFIG_TURNAROUND_BURST_SPAN_CREDIT) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM turnaround_mode must be none, fixed, idle_credit, burst_credit, burst_round_credit, or burst_span_credit");
        return -1;
    }
    if (cfg->dram_turnaround_domain < TU_DRAM_CONFIG_TURNAROUND_CORE_CYCLES ||
        cfg->dram_turnaround_domain > TU_DRAM_CONFIG_TURNAROUND_PHYSICAL_NS) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM turnaround_domain must be core_cycles or physical_ns");
        return -1;
    }
    if (!isfinite(cfg->dram_read_to_write_turnaround) ||
        !isfinite(cfg->dram_write_to_read_turnaround) ||
        cfg->dram_read_to_write_turnaround < 0.0 ||
        cfg->dram_write_to_read_turnaround < 0.0 ||
        cfg->dram_read_to_write_turnaround > 100000000.0 ||
        cfg->dram_write_to_read_turnaround > 100000000.0 ||
        (cfg->dram_turnaround_mode != TU_DRAM_CONFIG_TURNAROUND_NONE &&
         cfg->dram_read_to_write_turnaround == 0.0 &&
         cfg->dram_write_to_read_turnaround == 0.0)) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM turnaround values must be finite, in range, and nonzero when enabled");
        return -1;
    }
    if (cfg->dram_turnaround_mode != TU_DRAM_CONFIG_TURNAROUND_NONE &&
        cfg->dram_turnaround_domain == TU_DRAM_CONFIG_TURNAROUND_CORE_CYCLES &&
        cfg->dram_read_to_write_turnaround < 1.0 &&
        cfg->dram_write_to_read_turnaround < 1.0) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "fixed core-cycle DRAM turnaround requires at least one value >= 1");
        return -1;
    }
    if (cfg->dram_read_burst_bytes == UINT32_MAX ||
        cfg->dram_write_burst_bytes == UINT32_MAX ||
        (cfg->dram_read_burst_bytes != 0 &&
         (cfg->dram_read_burst_bytes & (cfg->dram_read_burst_bytes - 1)) != 0) ||
        (cfg->dram_write_burst_bytes != 0 &&
         (cfg->dram_write_burst_bytes & (cfg->dram_write_burst_bytes - 1)) != 0)) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM read/write burst bytes must be 0 (preset) or powers of two up to 1048576");
        return -1;
    }
    if (cfg->dram_refresh_mode < TU_DRAM_CONFIG_REFRESH_NONE ||
        cfg->dram_refresh_mode > TU_DRAM_CONFIG_REFRESH_PER_BANK) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM refresh mode must be none, all_bank, or per_bank");
        return -1;
    }
    if (cfg->dram_refresh_scheduling < TU_DRAM_CONFIG_REFRESH_SCHED_FIXED ||
        cfg->dram_refresh_scheduling > TU_DRAM_CONFIG_REFRESH_SCHED_DEFERRED) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM refresh scheduling must be fixed or deferred");
        return -1;
    }
    if (cfg->dram_refresh_rate != 0 && cfg->dram_refresh_rate != 1 &&
        cfg->dram_refresh_rate != 2 && cfg->dram_refresh_rate != 4) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "DRAM refresh rate must be 0 (default), 1x, 2x, or 4x");
        return -1;
    }
    if (cfg->dram_trefi_ns == UINT32_MAX || cfg->dram_trfc_ns == UINT32_MAX ||
        cfg->dram_trfc_pb_ns == UINT32_MAX ||
        cfg->dram_refresh_max_deferral_ns == UINT32_MAX) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "DRAM refresh timing value is out of range");
        return -1;
    }
    {
        /* Zero fields mean "use defaults", so resolve them before comparing. */
        uint64_t trefi = cfg->dram_trefi_ns ? cfg->dram_trefi_ns : 7800;
        uint64_t mdef = cfg->dram_refresh_max_deferral_ns
                            ? cfg->dram_refresh_max_deferral_ns : trefi;
        if (cfg->dram_refresh_mode != TU_DRAM_CONFIG_REFRESH_NONE &&
            mdef > trefi) {
            if (error_buf && error_size > 0)
                snprintf(error_buf, error_size,
                         "DRAM refresh max_deferral_ns must not exceed trefi_ns");
            return -1;
        }
    }
    if (cfg->interconnect_mode < 0 || cfg->interconnect_mode > 2) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "interconnect mode must be none, ring, or mesh");
        return -1;
    }
    if (cfg->icc_switching_mode < TU_ICC_SWITCH_LEGACY_HOP_ONLY ||
        cfg->icc_switching_mode > TU_ICC_SWITCH_STORE_FORWARD) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "interconnect switching must be legacy_hop_only, cut_through, or store_and_forward");
        return -1;
    }
    if (cfg->icc_contention_mode < TU_ICC_CONTENTION_IDEAL_PARALLEL ||
        cfg->icc_contention_mode > TU_ICC_CONTENTION_SHARED_LINK) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "interconnect contention must be ideal_parallel or shared_link");
        return -1;
    }
    if (cfg->icc_mesh_routing_mode < TU_ICC_MESH_ROUTE_XY ||
        cfg->icc_mesh_routing_mode > TU_ICC_MESH_ROUTE_YX) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "interconnect mesh_routing must be xy or yx");
        return -1;
    }
    if (cfg->icc_link_bytes_per_cycle == 0) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "interconnect link_bytes_per_cycle must be > 0");
        return -1;
    }
    if (cfg->icc_router_latency_cycles == UINT32_MAX) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "interconnect router_latency_cycles is out of range");
        return -1;
    }
    if (cfg->power_tech_node < 0 || cfg->power_tech_node > 6) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "power tech_node must be auto, 45nm, 28nm, 16nm, 7nm, 5nm, or 3nm");
        return -1;
    }
    if (!(cfg->power_clock_freq_mhz >= 0.0 && cfg->power_clock_freq_mhz <= 10000.0)) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "power clock_freq_mhz must be 0 (auto) or in (0,10000]");
        return -1;
    }
    if (cfg->compression_type < 0 || cfg->compression_type > 4) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "weight compression type must be none, rle, adaptive_rle, bitmap, or adaptive");
        return -1;
    }
    if (cfg->compression_rle_epsilon < 0.0) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "compression rle_epsilon must be >= 0");
        return -1;
    }
    if (cfg->compression_decoder_elements_per_cycle == 0 ||
        cfg->compression_rle_runs_per_cycle == 0 ||
        cfg->compression_bitmap_elements_per_cycle == 0) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "compression decoder throughput values must be > 0");
        return -1;
    }
    if (cfg->sparsity_unstructured) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size, "unstructured sparsity is not implemented");
        return -1;
    }
    if (cfg->sparsity_enabled && !cfg->sparsity_2of4) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "enabled sparsity requires structured_2of4=true");
        return -1;
    }
    if (cfg->sparsity_2of4 && !cfg->sparsity_enabled) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "structured_2of4 requires sparsity enabled");
        return -1;
    }
    if (cfg->sparsity_decoder_groups_per_cycle == 0) {
        if (error_buf && error_size > 0)
            snprintf(error_buf, error_size,
                     "sparsity decoder_groups_per_cycle must be > 0");
        return -1;
    }
    return 0;
}

/* ---- Dump ---- */

void tu_config_dump(const struct tu_config_t *cfg) {
    if (!cfg) return;
    fprintf(stderr,
        "════ TU Config ════\n"
        "  PE: %u×%u, depth=%u, dataflow=%d, plugin=%s\n"
        "  Prec: FP16=%s BF16=%s FP8e4=%s FP8e5=%s INT8=%s INT4=%s\n"
        "  SRAM: W=%uK A=%uK O=%uK, banks=%u×%uB\n"
        "  DRAM: type=%d BW=%.1f GB/s\n"
        "  DMA: bus=%ub, ch=%u, async=%s\n"
        "  Compression: %s type=%d epsilon=%g decoder=%s (%u elem/cyc, %u run/cyc, %u bitmap elem/cyc)\n"
        "  ISA: instr=%ub, qdepth=%u\n"
        "  Multi-core: %s, cores=%u\n"
        "  Cycle: model=%d, counters=%s, trace=%s\n"
        "  Sparsity: %s 2:4=%s decoder=%u groups/cycle\n"
        "══════════════════════\n",
        cfg->pe_rows, cfg->pe_cols, cfg->pe_pipeline_depth, cfg->dataflow_mode,
        cfg->dataflow_via_plugin ? "yes" : "no",
        cfg->fp16_enabled ? "on" : "off", cfg->bf16_enabled ? "on" : "off",
        cfg->fp8_e4m3_enabled ? "on" : "off", cfg->fp8_e5m2_enabled ? "on" : "off",
        cfg->int8_enabled ? "on" : "off", cfg->int4_enabled ? "on" : "off",
        cfg->sram_w_size_kb, cfg->sram_a_size_kb, cfg->sram_o_size_kb,
        cfg->sram_num_banks, cfg->sram_bank_width,
        cfg->dram_type, cfg->dram_bandwidth_gbps,
        cfg->dma_bus_width_bits, cfg->dma_num_channels,
        cfg->dma_async_mode ? "yes" : "no",
        cfg->compression_enabled ? "on" : "off", cfg->compression_type,
        cfg->compression_rle_epsilon,
        cfg->compression_decoder_enabled ? "on" : "off",
        cfg->compression_decoder_elements_per_cycle,
        cfg->compression_rle_runs_per_cycle,
        cfg->compression_bitmap_elements_per_cycle,
        cfg->isa_instr_width_bits, cfg->isa_queue_depth,
        cfg->multicore_enabled ? "on" : "off", cfg->num_cores,
        cfg->cycle_model, cfg->counters_enabled ? "on" : "off",
        cfg->trace_enabled ? "on" : "off",
        cfg->sparsity_enabled ? "on" : "off",
        cfg->sparsity_2of4 ? "on" : "off",
        cfg->sparsity_decoder_groups_per_cycle);
}

/* ---- Markdown documentation generator (Gap Q4) ---- */

static const char *fmt_bool(bool v) { return v ? "`true`" : "`false`"; }

void tu_config_emit_docs(const tu_config_t *cfg, FILE *out) {
    tu_config_t def;
    if (!cfg) {
        tu_config_default(&def);
        cfg = &def;
    }

    fprintf(out,
        "# TU CModel Configuration Reference\n\n"
        "> Auto-generated from current configuration.\n"
        "> Each field shows its **value**, type, and description.\n\n"
        "---\n\n");

    /* ---- Compute Engine ---- */
    fprintf(out, "## 1. Compute Engine\n\n");
    fprintf(out, "| Field | Value | Type | Description |\n");
    fprintf(out, "|-------|-------|------|-------------|\n");
    fprintf(out, "| `pe_rows` | %u | uint16 | PE array height (rows) |\n",
            cfg->pe_rows);
    fprintf(out, "| `pe_cols` | %u | uint16 | PE array width (columns) |\n",
            cfg->pe_cols);
    fprintf(out, "| `pe_pipeline_depth` | %u | uint16 | Pipeline stages per MAC |\n",
            cfg->pe_pipeline_depth);
    fprintf(out, "| `mac_units_per_pe` | %u | uint16 | MAC units per PE |\n",
            cfg->mac_units_per_pe);
    fprintf(out, "| `dataflow_mode` | %d | int | 0=WS, 1=OS, 2=RS (NLR reserved, not executable) |\n",
            cfg->dataflow_mode);
    fprintf(out, "| `dataflow_via_plugin` | %s | bool | Use pluggable dataflow dispatcher |\n\n",
            fmt_bool(cfg->dataflow_via_plugin));

    /* ---- Precision ---- */
    fprintf(out, "## 2. Precision & Data Types\n\n");
    fprintf(out, "| Field | Value | Type | Description |\n");
    fprintf(out, "|-------|-------|------|-------------|\n");
    fprintf(out, "| `fp16_enabled` | %s | bool | IEEE 754 half-precision (1-5-10) |\n",
            fmt_bool(cfg->fp16_enabled));
    fprintf(out, "| `fp32_enabled` | %s | bool | IEEE 754 single-precision (accumulator) |\n",
            fmt_bool(cfg->fp32_enabled));
    fprintf(out, "| `bf16_enabled` | %s | bool | Brain Float 16 (1-8-7) |\n",
            fmt_bool(cfg->bf16_enabled));
    fprintf(out, "| `fp8_e4m3_enabled` | %s | bool | FP8 E4M3 (OCP, forward pass) |\n",
            fmt_bool(cfg->fp8_e4m3_enabled));
    fprintf(out, "| `fp8_e5m2_enabled` | %s | bool | FP8 E5M2 (OCP, backward pass) |\n",
            fmt_bool(cfg->fp8_e5m2_enabled));
    fprintf(out, "| `int8_enabled` | %s | bool | INT8 symmetric quantization |\n",
            fmt_bool(cfg->int8_enabled));
    fprintf(out, "| `int4_enabled` | %s | bool | INT4 packed quantization |\n",
            fmt_bool(cfg->int4_enabled));
    fprintf(out, "| `rounding_mode` | %d | int | 0=RNE, 1=RTZ, 2=Stochastic |\n",
            cfg->rounding_mode);
    fprintf(out, "| `subnormal_flush` | %s | bool | Flush-to-zero (FTZ) for subnormals |\n",
            fmt_bool(cfg->subnormal_flush));
    fprintf(out, "| `saturate` | %s | bool | Saturate on overflow |\n\n",
            fmt_bool(cfg->saturate));

    /* ---- Memory ---- */
    fprintf(out, "## 3. Memory System\n\n");
    fprintf(out, "| Field | Value | Type | Description |\n");
    fprintf(out, "|-------|-------|------|-------------|\n");
    fprintf(out, "| `sram_w_size_kb` | %u | uint32 | Weight buffer size (KB) |\n",
            cfg->sram_w_size_kb);
    fprintf(out, "| `sram_a_size_kb` | %u | uint32 | Activation buffer size (KB) |\n",
            cfg->sram_a_size_kb);
    fprintf(out, "| `sram_o_size_kb` | %u | uint32 | Output/accumulator buffer size (KB) |\n",
            cfg->sram_o_size_kb);
    fprintf(out, "| `sram_num_banks` | %u | uint32 | Number of SRAM banks |\n",
            cfg->sram_num_banks);
    fprintf(out, "| `sram_bank_width` | %u | uint32 | Bytes per bank word |\n",
            cfg->sram_bank_width);
    fprintf(out, "| `sram_words_per_cycle` | %u | uint32 | Words granted per bank per refill window; per-cycle only when window=1 |\n",
            cfg->sram_words_per_cycle);
    fprintf(out, "| `sram_arb_mode` | %d | int | Arbitration: 0=None, 1=RR, 2=Priority |\n",
            cfg->sram_arb_mode);
    fprintf(out, "| `sram_conflict_mode` | %d | int | Conflict: 0=None, 1=Detect, 2=Stall |\n",
            cfg->sram_conflict_mode);
    fprintf(out, "| `sram_stall_penalty` | %u | uint8 | Stall penalty cycles |\n",
            cfg->sram_stall_penalty);
    fprintf(out, "| `sram_bw_window_cycles` | %lu | uint64 | Bandwidth refill window |\n",
            (unsigned long)cfg->sram_bw_window_cycles);
    fprintf(out, "| `sram_bw_modeling` | %s | bool | Enable bandwidth modeling |\n",
            fmt_bool(cfg->sram_bw_modeling));
    fprintf(out, "| `gbuf_size_kb` | %u | uint32 | Global buffer size (KB), 0=disabled |\n",
            cfg->gbuf_size_kb);
    fprintf(out, "| `gbuf_banks` | %u | uint32 | Global buffer banks |\n",
            cfg->gbuf_banks);
    fprintf(out, "| `dram_type` | %d | int | 0=Ideal, 1=HBM2, 2=HBM2e, 3=HBM3, 4=DDR4, 5=DDR5, 6=LPDDR5 |\n",
            cfg->dram_type);
    fprintf(out, "| `dram_bandwidth_gbps` | %.1f | double | DRAM bandwidth (GB/s) |\n",
            cfg->dram_bandwidth_gbps);
    fprintf(out, "| `dram_channels` | %u | uint32 | DRAM channel count |\n",
            cfg->dram_channels);
    fprintf(out, "| `dram_model_row_conflicts` | %s | bool | Model row buffer hit/miss |\n\n",
            fmt_bool(cfg->dram_model_row_conflicts));
    fprintf(out, "| `dram_row_policy` | %d | int | 0=Legacy, 1=Open-page, 2=Closed-page, 3=Adaptive-timeout |\n",
            cfg->dram_row_policy);
    fprintf(out, "| `dram_address_mapping` | %d | int | 0=Burst-interleaved, 1=Row-interleaved, 2=XOR-interleaved |\n",
            cfg->dram_address_mapping);
    fprintf(out, "| `dram_row_miss_penalty_cycles` | %u | uint32 | Added activate/precharge penalty per modeled miss |\n",
            cfg->dram_row_miss_penalty_cycles);
    fprintf(out, "| `dram_row_conflict_penalty_cycles` | %u | uint32 | Open-row replacement penalty; 0 inherits row_miss_penalty_cycles |\n",
            cfg->dram_row_conflict_penalty_cycles);
    fprintf(out, "| `dram_row_open_timeout_cycles` | %u | uint32 | Idle cycles before adaptive-timeout lazily precharges a row |\n",
            cfg->dram_row_open_timeout_cycles);
    fprintf(out, "| `dram_row_open_timeout_ns` | %.3f | double | Physical-ns adaptive timeout source |\n",
            cfg->dram_row_open_timeout_ns);
    fprintf(out, "| `dram_row_timeout_domain` | %d | int | 0=Fixed TU/core cycles (compat), 1=Physical ns converted at core clock |\n",
            cfg->dram_row_timeout_domain);
    fprintf(out, "| `dram_latency_domain` | %d | int | 0=Fixed TU/core cycles (compat), 1=Physical ns converted at core clock |\n",
            cfg->dram_latency_domain);
    fprintf(out, "| `dram_latency_read/write` | %.3f / %.3f | double | Base read/write latency in selected domain |\n",
            cfg->dram_latency_read, cfg->dram_latency_write);
    fprintf(out, "| `dram_core_clock_ghz` | %.3f | double | TU/core clock used for GB/s-to-bytes/cycle and ns-to-cycle conversion |\n",
            cfg->dram_core_clock_ghz);
    fprintf(out, "| `dram_turnaround_mode` | %d | int | 0=None, 1=Fixed, 2=Idle credit, 3=Exact-byte credit, 4=Payload-size-rounded bursts, 5=Address-span-rounded bursts |\n",
            cfg->dram_turnaround_mode);
    fprintf(out, "| `dram_turnaround_domain` | %d | int | 0=Fixed TU/core cycles, 1=Physical ns converted at core clock |\n",
            cfg->dram_turnaround_domain);
    fprintf(out, "| `dram_read_to_write_turnaround` | %.3f | double | Read-to-write bus turnaround in selected domain |\n",
            cfg->dram_read_to_write_turnaround);
    fprintf(out, "| `dram_write_to_read_turnaround` | %.3f | double | Write-to-read bus turnaround in selected domain |\n",
            cfg->dram_write_to_read_turnaround);
    fprintf(out, "| `dram_read_burst_bytes` | %u | uint32 | Rounded-mode read occupancy granule; 0 inherits the DRAM preset |\n",
            cfg->dram_read_burst_bytes);
    fprintf(out, "| `dram_write_burst_bytes` | %u | uint32 | Rounded-mode write occupancy granule; 0 inherits the DRAM preset |\n",
            cfg->dram_write_burst_bytes);
    fprintf(out, "| `dram_refresh_mode` | %d | int | 0=None (compat), 1=All-bank, 2=Per-bank (JEDEC tREFI/tRFC) |\n",
            cfg->dram_refresh_mode);
    fprintf(out, "| `dram_refresh_scheduling` | %d | int | 0=Fixed periodic, 1=Deferred (bounded postponement) |\n",
            cfg->dram_refresh_scheduling);
    fprintf(out, "| `dram_refresh_rate` | %u | uint32 | 1x/2x/4x refresh-rate multiplier (high-temp retention) |\n",
            cfg->dram_refresh_rate);
    fprintf(out, "| `dram_trefi_ns` | %u | uint32 | JEDEC tREFI per-bank refresh interval (ns) |\n",
            cfg->dram_trefi_ns);
    fprintf(out, "| `dram_trfc_ns` | %u | uint32 | All-bank refresh lockout tRFC (ns) |\n",
            cfg->dram_trfc_ns);
    fprintf(out, "| `dram_trfc_pb_ns` | %u | uint32 | Per-bank refresh lockout tRFCpb (ns) |\n",
            cfg->dram_trfc_pb_ns);
    fprintf(out, "| `dram_refresh_max_deferral_ns` | %u | uint32 | Deferred hard deadline (ns, ≤ tREFI) |\n\n",
            cfg->dram_refresh_max_deferral_ns);

    /* ---- DMA ---- */
    fprintf(out, "## 4. DMA Engine\n\n");
    fprintf(out, "| Field | Value | Type | Description |\n");
    fprintf(out, "|-------|-------|------|-------------|\n");
    fprintf(out, "| `dma_bus_width_bits` | %u | uint32 | AXI bus width |\n",
            cfg->dma_bus_width_bits);
    fprintf(out, "| `dma_max_burst_bytes` | %u | uint32 | Max burst size |\n",
            cfg->dma_max_burst_bytes);
    fprintf(out, "| `dma_num_channels` | %u | uint32 | DMA channel count |\n",
            cfg->dma_num_channels);
    fprintf(out, "| `dma_bus_topology` | %s | enum | Channel data paths: independent or shared_serial |\n",
            cfg->dma_bus_mode == TU_DMA_CONFIG_BUS_SHARED_SERIAL ? "shared_serial" : "independent");
    fprintf(out, "| `dma_arbitration` | %s | enum | Shared-serial selection: round_robin or strict_priority |\n",
            cfg->dma_arb_policy == TU_DMA_CONFIG_ARB_STRICT_PRIORITY ? "strict_priority" : "round_robin");
    const char *binding_name = cfg->dma_binding_policy == TU_DMA_CONFIG_BIND_ROUND_ROBIN ?
                               "round_robin" :
                               (cfg->dma_binding_policy == TU_DMA_CONFIG_BIND_LEAST_OUTSTANDING ?
                                "least_outstanding" :
                                (cfg->dma_binding_policy == TU_DMA_CONFIG_BIND_LEAST_BYTES ?
                                 "least_bytes" : "explicit"));
    fprintf(out, "| `dma_channel_binding` | %s | enum | Descriptor queue binding: explicit, round_robin, least_outstanding, or least_bytes |\n",
            binding_name);
    fprintf(out, "| `dma_max_outstanding` | %u | uint32 | Max outstanding descriptors |\n",
            cfg->dma_max_outstanding);
    fprintf(out, "| `dma_async_mode` | %s | bool | Async DMA with descriptor queues |\n",
            fmt_bool(cfg->dma_async_mode));
    fprintf(out, "| `dma_multicast_enabled` | %s | bool | Multicast/broadcast DMA |\n",
            fmt_bool(cfg->dma_multicast_enabled));
    fprintf(out, "| `compression_enabled` | %s | bool | Enable weight-stream compression |\n",
            fmt_bool(cfg->compression_enabled));
    fprintf(out, "| `compression_type` | %d | int | 0=None, 1=RLE, 2=Adaptive RLE, 3=Bitmap, 4=Adaptive all |\n",
            cfg->compression_type);
    fprintf(out, "| `compression_rle_epsilon` | %.6g | double | Merge tolerance; 0 is lossless |\n",
            cfg->compression_rle_epsilon);
    fprintf(out, "| `compression_decoder_enabled` | %s | bool | Include decompressor throughput in stream-cycle estimates |\n",
            fmt_bool(cfg->compression_decoder_enabled));
    fprintf(out, "| `compression_decoder_overlap_dma` | %s | bool | Pipeline payload DMA and decode; false serializes them |\n",
            fmt_bool(cfg->compression_decoder_overlap_dma));
    fprintf(out, "| `compression_decoder_elements_per_cycle` | %u | uint32 | Dense FP16 outputs reconstructed per cycle |\n",
            cfg->compression_decoder_elements_per_cycle);
    fprintf(out, "| `compression_rle_runs_per_cycle` | %u | uint32 | RLE run descriptors issued per cycle |\n",
            cfg->compression_rle_runs_per_cycle);
    fprintf(out, "| `compression_bitmap_elements_per_cycle` | %u | uint32 | Bitmap positions scanned per cycle |\n\n",
            cfg->compression_bitmap_elements_per_cycle);

    /* ---- ISA ---- */
    fprintf(out, "## 5. ISA & Command Queue\n\n");
    fprintf(out, "| Field | Value | Type | Description |\n");
    fprintf(out, "|-------|-------|------|-------------|\n");
    fprintf(out, "| `isa_instr_width_bits` | %u | uint32 | Instruction encoding width |\n",
            cfg->isa_instr_width_bits);
    fprintf(out, "| `isa_queue_depth` | %u | uint32 | Command queue depth |\n",
            cfg->isa_queue_depth);
    fprintf(out, "| `isa_dep_checking` | %s | bool | Dependency checking |\n\n",
            fmt_bool(cfg->isa_dep_checking));

    /* ---- Multi-Core ---- */
    fprintf(out, "## 6. Multi-Core\n\n");
    fprintf(out, "| Field | Value | Type | Description |\n");
    fprintf(out, "|-------|-------|------|-------------|\n");
    fprintf(out, "| `multicore_enabled` | %s | bool | Multi-core TU cluster |\n",
            fmt_bool(cfg->multicore_enabled));
    fprintf(out, "| `num_cores` | %u | uint32 | Core count |\n",
            cfg->num_cores);
    fprintf(out, "| `interconnect_mode` | %d | int | 0=None, 1=Ring, 2=Mesh |\n",
            cfg->interconnect_mode);
    fprintf(out, "| `icc_switching_mode` | %d | int | 0=Legacy hop-only, 1=Cut-through, 2=Store-and-forward |\n",
            cfg->icc_switching_mode);
    fprintf(out, "| `icc_contention_mode` | %d | int | 0=Ideal parallel links, 1=Shared-link lower bound |\n",
            cfg->icc_contention_mode);
    fprintf(out, "| `icc_mesh_routing_mode` | %d | int | 0=Deterministic XY, 1=Deterministic YX |\n",
            cfg->icc_mesh_routing_mode);
    fprintf(out, "| `icc_link_bytes_per_cycle` | %u | uint32 | Physical link payload width |\n",
            cfg->icc_link_bytes_per_cycle);
    fprintf(out, "| `icc_router_latency_cycles` | %u | uint32 | Per-hop router/link latency |\n\n",
            cfg->icc_router_latency_cycles);

    /* ---- Performance ---- */
    fprintf(out, "## 7. Performance Model\n\n");
    fprintf(out, "| Field | Value | Type | Description |\n");
    fprintf(out, "|-------|-------|------|-------------|\n");
    fprintf(out, "| `cycle_model` | %d | int | 0=Functional, 1=Estimated, 2=Cycle-Accurate |\n",
            cfg->cycle_model);
    fprintf(out, "| `counters_enabled` | %s | bool | Performance counters |\n",
            fmt_bool(cfg->counters_enabled));
    fprintf(out, "| `detailed_stalls` | %s | bool | Detailed stall breakdown |\n",
            fmt_bool(cfg->detailed_stalls));
    fprintf(out, "| `trace_enabled` | %s | bool | VCD execution trace |\n",
            fmt_bool(cfg->trace_enabled));
    fprintf(out, "| `trace_max_events` | %u | uint32 | Max trace events |\n",
            cfg->trace_max_events);
    fprintf(out, "| `power_tech_node` | %d | int | 0=Auto, 1=45nm, 2=28nm, 3=16nm, 4=7nm, 5=5nm, 6=3nm |\n",
            cfg->power_tech_node);
    fprintf(out, "| `power_clock_freq_mhz` | %.1f | double | 0=Auto heuristic; otherwise explicit modeled clock MHz |\n\n",
            cfg->power_clock_freq_mhz);

    /* ---- Sparsity ---- */
    fprintf(out, "## 8. Sparsity\n\n");
    fprintf(out, "| Field | Value | Type | Description |\n");
    fprintf(out, "|-------|-------|------|-------------|\n");
    fprintf(out, "| `sparsity_enabled` | %s | bool | Sparsity support |\n",
            fmt_bool(cfg->sparsity_enabled));
    fprintf(out, "| `sparsity_2of4` | %s | bool | 2:4 structured sparsity |\n",
            fmt_bool(cfg->sparsity_2of4));
    fprintf(out, "| `sparsity_unstructured` | %s | bool | Unstructured sparsity |\n",
            fmt_bool(cfg->sparsity_unstructured));
    fprintf(out, "| `sparsity_decoder_groups_per_cycle` | %u | uint32 | 2:4 metadata groups decoded per cycle |\n",
            cfg->sparsity_decoder_groups_per_cycle);
    fprintf(out, "| `sparsity_metadata_format` | %d | int | 0=Bitmask, 1=CSR, 2=Coord |\n\n",
            cfg->sparsity_metadata_format);

    /* ---- Verification ---- */
    fprintf(out, "## 9. Verification\n\n");
    fprintf(out, "| Field | Value | Type | Description |\n");
    fprintf(out, "|-------|-------|------|-------------|\n");
    fprintf(out, "| `golden_reference` | %d | int | 0=NumPy, 1=PyTorch |\n",
            cfg->golden_reference);
    fprintf(out, "| `random_test_iters` | %u | uint32 | Random test iterations |\n",
            cfg->random_test_iters);
    fprintf(out, "| `error_tolerance` | %e | double | Golden comparison tolerance |\n\n",
            cfg->error_tolerance);

    /* ---- Derived Values ---- */
    fprintf(out, "## 10. Derived Values\n\n");
    fprintf(out, "| Metric | Value | Formula |\n");
    fprintf(out, "|--------|-------|--------|\n");
    uint32_t total_sram = cfg->sram_w_size_kb + cfg->sram_a_size_kb +
                          cfg->sram_o_size_kb;
    fprintf(out, "| Total SRAM | %u KB | W + A + O |\n", total_sram);
    uint32_t total_macs = cfg->pe_rows * cfg->pe_cols * cfg->mac_units_per_pe;
    fprintf(out, "| Total MACs | %u | rows × cols × mac_units_per_pe |\n",
            total_macs);
    uint64_t peak_ops = (uint64_t)total_macs * 2; /* multiply + add */
    fprintf(out, "| Peak Ops/cycle | %lu | MACs × 2 |\n",
            (unsigned long)peak_ops);
    fprintf(out, "| DMA bandwidth | %.1f GB/s | bus_width / 8 × frequency |\n",
            cfg->dma_bus_width_bits / 8.0);

    fprintf(out, "\n---\n\n"
            "*Generated by `tu_config_emit_docs()`. "
            "Regenerate with `make config-docs` to reflect current settings.*\n");
}
