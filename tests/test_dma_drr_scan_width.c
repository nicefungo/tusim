/* Grouped DRR scans: deterministic aggregate control timing, not RTL. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/infra/config.h"
#include "tu_cmodel/dma_descriptor.h"
#include <stdio.h>
#include <string.h>

static int run_case(unsigned width, unsigned mode, unsigned cursor,
                    unsigned target, unsigned round_cost, unsigned select_cost,
                    bool store, uint64_t expected) {
    char json[512], error[256];
    const char *names[] = {"fixed", "per_channel", "per_visited_channel"};
    snprintf(json, sizeof(json),
        "{\"tu\":{\"dma\":{\"channels\":8,\"async_mode\":true,"
        "\"bus_topology\":\"shared_serial\",\"arbitration\":\"deficit_round_robin\","
        "\"drr_quantum_bytes\":64,\"drr_round_cost_mode\":\"%s\","
        "\"drr_round_issue_cycles\":%u,\"drr_select_issue_cycles\":%u,"
        "\"drr_scan_width\":%u}}}", names[mode], round_cost, select_cost, width);
    tu_config_t cfg;
    tu_config_default(&cfg);
    if (tu_config_load_string(json, &cfg, error, sizeof(error)) != 0) {
        fprintf(stderr, "parse failed: %s\n", error);
        return 1;
    }
    unsigned effective_width = width ? width : 1u;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (cfg.dma_drr_scan_width != width || rt.dma_drr_scan_width != effective_width)
        return 1;
    tu_init_with_config(&rt);
    if (g_tu_dma.drr_scan_width != effective_width ||
        g_tu_dma.bus_mode != TU_DMA_BUS_MODE_SHARED_SERIAL ||
        g_tu_dma.arb_policy != TU_DMA_ARB_DEFICIT_ROUND_ROBIN) return 1;
    g_tu.sram_w.banks.bw_modeling = false;
    g_tu_dma.next_shared_channel = cursor;
    unsigned char source[256], host[256];
    memset(source, 0x97, sizeof(source));
    memset(host, 0, sizeof(host));
    memset(tu_sram_raw_ptr(&g_tu.sram_w), store ? 0x97 : 0, sizeof(source));
    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        target, store ? TU_DMA_DIR_TU_TO_HOST : TU_DMA_DIR_HOST_TO_TU,
        &g_tu.sram_w, 0, store ? host : source, 1, sizeof(source));
    int failed = !desc || !tu_dma_submit_desc(desc);
    while (!failed && g_tu_dma.channels[target].total_completed < 1 &&
           g_tu_dma.current_cycle < 1000)
        tu_dma_tick();
    if (!failed) {
        const void *result = store ? (void *)host : tu_sram_raw_ptr(&g_tu.sram_w);
        failed = desc->cycles_completed != expected ||
                 g_tu_dma.current_cycle != expected ||
                 memcmp(result, source, sizeof(source)) != 0 ||
                 g_tu_dma.total_bytes != sizeof(source) ||
                 g_tu_dma.total_occupied_bytes != sizeof(source);
        printf("%u %s %u %u %u %u %s %llu %llu %s\n", width, names[mode],
               cursor, target, round_cost, select_cost, store ? "store" : "load",
               (unsigned long long)desc->cycles_completed,
               (unsigned long long)expected, failed ? "FAIL" : "PASS");
    }
    tu_shutdown();
    if (desc) { desc->next = NULL; tu_dma_desc_destroy(desc); }
    return failed;
}

static int controls(void) {
    tu_config_t cfg;
    char json[128], error[256];
    const char *invalid[] = {"-1", "9", "4294967296", "1.5", "true", "null", "\"2\""};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        snprintf(json, sizeof(json), "{\"tu\":{\"dma\":{\"drr_scan_width\":%s}}}", invalid[i]);
        if (tu_config_load_string(json, &cfg, error, sizeof(error)) == 0) {
            fprintf(stderr, "invalid scan width accepted: %s\n", invalid[i]);
            return 1;
        }
    }
    tu_config_default(&cfg);
    if (cfg.dma_drr_scan_width != 1) return 1;
    if (tu_config_load("config/tu_config.json", &cfg, error, sizeof(error)) != 0 ||
        cfg.dma_drr_scan_width != 1) return 1;
    memset(&cfg, 0, sizeof(cfg));
    if (tu_config_to_runtime(&cfg).dma_drr_scan_width != 1) return 1;
    tu_runtime_config_t rt = tu_runtime_config_default();
    if (rt.dma_drr_scan_width != 1) return 1;
    rt.dma_drr_scan_width = 0;
    tu_init_with_config(&rt);
    if (g_tu_dma.drr_scan_width != 1) return 1;
    g_tu_dma.channels[0].drr_deficit_credits = 17;
    g_tu_dma.drr_continue_visit = true;
    g_tu_dma.drr_continue_channel = 0;
    g_tu_dma.next_shared_channel = 2;
    if (!tu_dma_set_drr_scan_width(3) || tu_dma_set_drr_scan_width(9) ||
        tu_dma_set_drr_scan_width(UINT32_MAX) || g_tu_dma.drr_scan_width != 3 ||
        g_tu_dma.channels[0].drr_deficit_credits != 17 ||
        !g_tu_dma.drr_continue_visit || g_tu_dma.next_shared_channel != 2)
        return 1;
    if (!tu_dma_set_drr_scan_width(0) || g_tu_dma.drr_scan_width != 1) return 1;
    tu_shutdown();
    /* Old initializer surfaces must retain one-wide scanning. */
    tu_dma_init(true);
    int failed = g_tu_dma.drr_scan_width != 1;
    tu_dma_destroy();
    return failed;
}

static int queued_case(unsigned width) {
    tu_runtime_config_t rt = tu_runtime_config_default();
    rt.dma_num_channels = 8;
    rt.dma_async_mode = true;
    rt.dma_bus_mode = TU_DMA_BUS_MODE_SHARED_SERIAL;
    rt.dma_arb_policy = TU_DMA_ARB_DEFICIT_ROUND_ROBIN;
    rt.dma_drr_quantum_bytes = 64;
    rt.dma_drr_round_cost_mode = TU_DMA_DRR_ROUND_COST_PER_VISITED_CHANNEL;
    rt.dma_drr_round_issue_cycles = rt.dma_drr_select_issue_cycles = 1;
    rt.dma_drr_scan_width = width;
    tu_init_with_config(&rt);
    g_tu.sram_w.banks.bw_modeling = false;
    g_tu_dma.next_shared_channel = 6;
    unsigned char src[2][64];
    memset(src[0], 0x36, 64); memset(src[1], 0x8a, 64);
    tu_dma_descriptor_t *a = tu_dma_desc_create_linear(7, TU_DMA_DIR_HOST_TO_TU,
        &g_tu.sram_w, 0, src[0], 1, 64);
    tu_dma_descriptor_t *b = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU,
        &g_tu.sram_w, 64, src[1], 1, 64);
    int failed = !a || !b || !tu_dma_submit_desc(a) || !tu_dma_submit_desc(b);
    while (!failed && g_tu_dma.channels[0].total_completed < 1 &&
           g_tu_dma.current_cycle < 1000) tu_dma_tick();
    uint64_t first = 53 + (2 + width - 1) / width;
    uint64_t second = first + 53;
    if (!failed) failed = a->cycles_completed != first || b->cycles_completed != second ||
        g_tu_dma.current_cycle != second || g_tu_dma.total_bytes != 128 ||
        memcmp(tu_sram_raw_ptr(&g_tu.sram_w), src, 128) != 0;
    tu_shutdown();
    if (a) { a->next = NULL; tu_dma_desc_destroy(a); }
    if (b) { b->next = NULL; tu_dma_desc_destroy(b); }
    return failed;
}

int main(void) {
    if (controls()) return 1;
    puts("width mode cursor target round_cost select_cost direction completion expected status");
    /* Tracer bullet: three failed rounds plus one full-distance selection. */
    if (run_case(2, 2, 0, 7, 1, 1, false, 75)) return 1;
    unsigned rows = 1;
    for (unsigned w = 0; w <= 8; ++w) {
        unsigned width = w ? w : 1;
        for (unsigned mode = 0; mode < 3; ++mode)
            for (unsigned cursor = 0; cursor <= 6; cursor += 6)
                for (unsigned target = 0; target < 8; ++target)
                    for (unsigned store = 0; store < 2; ++store) {
                        unsigned probes = (target + 8 - cursor) % 8 + 1;
                        unsigned round_groups = mode == 0 ? 1 : (8 + width - 1) / width;
                        unsigned selected_probes = mode == 1 ? 8 : probes;
                        unsigned select_groups = mode == 0 ? 1 :
                            (selected_probes + width - 1) / width;
                        uint64_t expected = 59 + 3 * round_groups + select_groups;
                        if (run_case(w, mode, cursor, target, 1, 1, store, expected)) return 1;
                        ++rows;
                    }
        if (run_case(w, 2, 6, 5, 0, 0, false, 59)) return 1;
        ++rows;
        if (queued_case(width)) return 1;
    }
    printf("PASS: %u timing rows; 9 queued-order gates; parse-to-live, defaults, zero inheritance, invalid rejection, state preservation, both directions, exact bytes\n", rows);
    return 0;
}
