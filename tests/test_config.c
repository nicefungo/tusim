/*
 * Test: JSON Config Loader (Gap A1)
 * ==================================
 * Tests:
 *   1. JSON parser correctness (primitives, arrays, objects, nested)
 *   2. Config defaults
 *   3. Config loading from string
 *   4. Config loading from file
 *   5. Config validation errors
 *   6. Config-to-runtime conversion
 *   7. TU init from config + MMA verification
 */

#include "tu_cmodel.h"
#include "infra/config.h"
#include "infra/json_reader.h"
#include "test_framework.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

tu_test_stats_t g_test_stats = {0};

#define CHECK(cond, msg) do { if (!(cond)) { FAIL("%s", msg); return; } } while(0)

int main(void) {
    test_stats_init();

    /* ── JSON Parser: Primitives ── */

    TEST("JSON: integer 42");
    {
        char json[] = "42";
        tu_json_value_t root;
        tu_json_error_t err = tu_json_parse(json, &root, NULL, 0);
        CHECK(err == TU_JSON_OK && root.type == TU_JSON_INT && root.int_val == 42, "int parse");
        tu_json_free(&root);
        PASS();
    }

    TEST("JSON: double 3.14");
    {
        char json[] = "3.14159";
        tu_json_value_t root;
        CHECK(tu_json_parse(json, &root, NULL, 0) == TU_JSON_OK, "parse");
        CHECK(root.type == TU_JSON_DOUBLE && root.double_val > 3.14 && root.double_val < 3.15, "double");
        tu_json_free(&root);
        PASS();
    }

    TEST("JSON: string hello");
    {
        char json[] = "\"hello, world\"";
        tu_json_value_t root;
        CHECK(tu_json_parse(json, &root, NULL, 0) == TU_JSON_OK, "parse");
        uint32_t len; const char *s = tu_json_as_string(&root, &len);
        CHECK(s && len == 12 && strncmp(s, "hello, world", 12) == 0, "string");
        tu_json_free(&root);
        PASS();
    }

    TEST("JSON: string escapes");
    {
        char json[] = "\"hello\\nworld\\t!\"";
        tu_json_value_t root;
        CHECK(tu_json_parse(json, &root, NULL, 0) == TU_JSON_OK, "parse");
        uint32_t len; const char *s = tu_json_as_string(&root, &len);
        CHECK(s && len == 13 && strncmp(s, "hello\nworld\t!", 13) == 0, "escapes");
        tu_json_free(&root);
        PASS();
    }

    TEST("JSON: bool/null");
    {
        tu_json_value_t r;
        CHECK(tu_json_parse("true", &r, NULL, 0) == TU_JSON_OK && r.type == TU_JSON_BOOL && r.bool_val, "true");
        tu_json_free(&r);
        CHECK(tu_json_parse("false", &r, NULL, 0) == TU_JSON_OK && r.type == TU_JSON_BOOL && !r.bool_val, "false");
        tu_json_free(&r);
        CHECK(tu_json_parse("null", &r, NULL, 0) == TU_JSON_OK && r.type == TU_JSON_NULL, "null");
        tu_json_free(&r);
        PASS();
    }

    TEST("JSON: array");
    {
        char json[] = "[1, 2, 3, \"four\"]";
        tu_json_value_t root;
        CHECK(tu_json_parse(json, &root, NULL, 0) == TU_JSON_OK && root.type == TU_JSON_ARRAY, "parse");
        CHECK(root.array.count == 4, "count");
        CHECK(root.array.items[0].int_val == 1, "idx0");
        CHECK(root.array.items[1].int_val == 2, "idx1");
        CHECK(root.array.items[2].int_val == 3, "idx2");
        CHECK(tu_json_as_string(&root.array.items[3], NULL) != NULL, "idx3");
        tu_json_free(&root);
        PASS();
    }

    TEST("JSON: object + get");
    {
        char json[] = "{\"name\": \"TinyTU\", \"rows\": 32, \"cols\": 32}";
        tu_json_value_t root;
        CHECK(tu_json_parse(json, &root, NULL, 0) == TU_JSON_OK, "parse");
        CHECK(root.object.count == 3, "count");

        const tu_json_value_t *v = tu_json_get(&root, "rows");
        CHECK(v && v->int_val == 32, "rows");

        v = tu_json_get(&root, "cols");
        CHECK(v && v->int_val == 32, "cols");

        v = tu_json_get(&root, "name");
        uint32_t len; const char *s = tu_json_as_string(v, &len);
        CHECK(s && len == 6 && strncmp(s, "TinyTU", 6) == 0, "name");

        v = tu_json_get(&root, "nonexistent");
        CHECK(v == NULL, "missing");

        tu_json_free(&root);
        PASS();
    }

    TEST("JSON: nested objects");
    {
        char json[] = "{\"a\": {\"b\": {\"c\": 99}}}";
        tu_json_value_t root;
        CHECK(tu_json_parse(json, &root, NULL, 0) == TU_JSON_OK, "parse");
        const tu_json_value_t *a = tu_json_get(&root, "a");
        const tu_json_value_t *b = tu_json_get(a, "b");
        const tu_json_value_t *c = tu_json_get(b, "c");
        CHECK(c && c->int_val == 99, "nested");
        tu_json_free(&root);
        PASS();
    }

    TEST("JSON: trailing characters error");
    {
        char json[] = "42 extra";
        tu_json_value_t root;
        CHECK(tu_json_parse(json, &root, NULL, 0) != TU_JSON_OK, "should fail");
        tu_json_free(&root);
        PASS();
    }

    /* ── Config System ── */

    TEST("Config: defaults");
    {
        tu_config_t cfg;
        tu_config_default(&cfg);
        CHECK(cfg.pe_rows == 16, "pe_rows");
        CHECK(cfg.pe_cols == 16, "pe_cols");
        CHECK(cfg.sram_w_size_kb == 128, "w_size");
        CHECK(cfg.sram_a_size_kb == 64, "a_size");
        CHECK(cfg.sram_o_size_kb == 64, "o_size");
        CHECK(cfg.dma_bus_width_bits == 256, "bus");
        CHECK(cfg.cycle_model == 2, "cycle");
        CHECK(cfg.counters_enabled, "counters");
        CHECK(cfg.dataflow_mode == 0, "dataflow");
        PASS();
    }

    TEST("Config: load from string");
    {
        const char *json =
            "{\"tu\": {"
            "  \"compute\": {"
            "    \"pe_array\": {"
            "      \"rows\": 32,"
            "      \"cols\": 32,"
            "      \"dataflow\": \"output_stationary\""
            "    }"
            "  },"
            "  \"memory\": {"
            "    \"sram\": {"
            "      \"w_buffer_kb\": 256"
            "    },"
            "    \"banking\": {"
            "      \"banks\": 64,"
            "      \"bank_width_bytes\": 8"
            "    }"
            "  },"
            "  \"dma\": {"
            "    \"bus_width_bits\": 512,"
            "    \"async_mode\": true"
            "  }"
            "}}";

        tu_config_t cfg;
        CHECK(tu_config_load_string(json, &cfg, NULL, 0) == 0, "load");
        CHECK(cfg.pe_rows == 32, "rows");
        CHECK(cfg.pe_cols == 32, "cols");
        CHECK(cfg.dataflow_mode == 1, "dataflow=OS");
        CHECK(cfg.sram_w_size_kb == 256, "w_size");
        CHECK(cfg.sram_num_banks == 64, "banks");
        CHECK(cfg.sram_bank_width == 8, "bank_width");
        CHECK(cfg.dma_bus_width_bits == 512, "dma_bus");
        CHECK(cfg.dma_async_mode, "async");
        PASS();
    }

    TEST("Config: load from file");
    {
        tu_config_t cfg;
        CHECK(tu_config_load("config/tu_config.json", &cfg, NULL, 0) == 0, "load_file");
        CHECK(cfg.pe_rows == 16, "rows");
        CHECK(cfg.pe_cols == 16, "cols");
        CHECK(cfg.dma_bus_width_bits == 256, "bus");
        PASS();
    }

    TEST("Config: validation pass");
    {
        tu_config_t cfg;
        tu_config_default(&cfg);
        CHECK(tu_config_validate(&cfg, NULL, 0) == 0, "valid default");
        PASS();
    }

    TEST("Config: validation fail (bad rows)");
    {
        tu_config_t cfg;
        tu_config_default(&cfg);
        cfg.pe_rows = 0;
        CHECK(tu_config_validate(&cfg, NULL, 0) != 0, "should reject pe_rows=0");
        PASS();
    }

    TEST("Config: validation fail (bad bank width)");
    {
        tu_config_t cfg;
        tu_config_default(&cfg);
        cfg.sram_bank_width = 3;
        CHECK(tu_config_validate(&cfg, NULL, 0) != 0, "should reject bank_width=3");
        PASS();
    }

    TEST("Config: validation fail (bad bus width)");
    {
        tu_config_t cfg;
        tu_config_default(&cfg);
        cfg.dma_bus_width_bits = 100;
        CHECK(tu_config_validate(&cfg, NULL, 0) != 0, "should reject bus=100");
        PASS();
    }

    TEST("Config: interconnect switching parse + validation");
    {
        const char *json =
            "{\"tu\":{\"multicore\":{\"enabled\":true,\"num_cores\":8,"
            "\"interconnect\":\"mesh\",\"switching\":\"cut_through\","
            "\"link_bytes_per_cycle\":32,\"router_latency_cycles\":3}}}";
        tu_config_t cfg;
        CHECK(tu_config_load_string(json, &cfg, NULL, 0) == 0, "ICC config parse");
        CHECK(cfg.icc_switching_mode == TU_ICC_SWITCH_CUT_THROUGH, "switching mode");
        CHECK(cfg.icc_link_bytes_per_cycle == 32, "link width");
        CHECK(cfg.icc_router_latency_cycles == 3, "router latency");
        tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
        CHECK(rt.icc_switching_mode == TU_ICC_SWITCH_CUT_THROUGH, "runtime switching");
        CHECK(rt.icc_link_bytes_per_cycle == 32, "runtime link width");
        PASS();
    }

    TEST("Config: reject unsupported interconnect switching");
    {
        tu_config_t cfg;
        char err[160];
        CHECK(tu_config_load_string(
            "{\"tu\":{\"multicore\":{\"switching\":\"teleport\"}}}",
            &cfg, err, sizeof(err)) != 0, "unsupported switch accepted");
        CHECK(strstr(err, "switching") != NULL, "wrong switch validation error");
        CHECK(tu_config_load_string(
            "{\"tu\":{\"multicore\":{\"switching\":\"cut_through\","
            "\"link_bytes_per_cycle\":0}}}",
            &cfg, err, sizeof(err)) != 0, "zero link width accepted");
        PASS();
    }

    TEST("Config: to runtime conversion");
    {
        tu_config_t cfg;
        tu_config_default(&cfg);
        cfg.pe_rows = 32; cfg.pe_cols = 64;
        cfg.sram_w_size_kb = 256;

        tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
        CHECK(rt.pe_rows == 32, "rt_rows");
        CHECK(rt.pe_cols == 64, "rt_cols");
        CHECK(rt.sram_w_size == 256 * 1024, "rt_w_size");
        PASS();
    }

    TEST("Config: TU init from config + MMA");
    {
        tu_config_t cfg;
        tu_config_default(&cfg);
        cfg.pe_rows = 8; cfg.pe_cols = 8;

        int err = tu_init_from_config(&cfg);
        CHECK(err == 0, "init");
        CHECK(g_tu.rt_cfg.pe_rows == 8, "rt_pe_rows");
        CHECK(g_tu.rt_cfg.pe_cols == 8, "rt_pe_cols");

        /* Small MMA: 8×8×8, W=all 1.0, A=all 2.0 → O[m][n] = 8*2 = 16.0 */
        fp16_t w[64], a[64];
        float o_exp[64];
        for (int i = 0; i < 64; i++) {
            w[i] = fp32_to_fp16(1.0f);
            a[i] = fp32_to_fp16(2.0f);
            o_exp[i] = 16.0f;
        }

        tu_dma_load_w(w, 0, sizeof(w));
        tu_dma_load_a(a, 0, sizeof(a));
        /* Output buf initially zero (no bias) */
        float o_zero[64] = {0};
        tu_dma_load_o(o_zero, 0, sizeof(o_zero));
        tu_mma(8, 8, 8, 0, 0, 0, false);

        float o_result[64];
        tu_dma_store_o(o_result, 0, sizeof(o_result));

        float max_err = max_abs_error(o_exp, o_result, 64);
        CHECK(max_err < 0.1f, "mma result close to 16.0");

        PASS();
    }

    return test_exit();
}
