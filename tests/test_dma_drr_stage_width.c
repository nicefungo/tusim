/* Independent DRR scan-stage provisioning. Aggregate timing, not RTL. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/infra/config.h"
#include "tu_cmodel/dma_descriptor.h"
#include <stdio.h>
#include <string.h>

static unsigned timing_rows, queue_rows;
static const char *mode_names[] = {"fixed", "per_channel", "per_visited_channel"};

static unsigned groups(unsigned n, unsigned w) { return n / w + (n % w != 0); }

static int run_case(unsigned channels, unsigned common, unsigned round_width,
                    unsigned select_width, unsigned mode, unsigned cursor,
                    unsigned target, unsigned bytes, bool store,
                    unsigned round_cost, unsigned select_cost) {
    char json[768], error[256];
    snprintf(json, sizeof(json),
        "{\"tu\":{\"dma\":{\"channels\":%u,\"async_mode\":true,"
        "\"bus_topology\":\"shared_serial\",\"arbitration\":\"deficit_round_robin\","
        "\"drr_quantum_bytes\":64,\"drr_round_cost_mode\":\"%s\","
        "\"drr_round_issue_cycles\":%u,\"drr_select_issue_cycles\":%u,"
        "\"drr_scan_width\":%u,\"drr_round_scan_width\":%u,"
        "\"drr_select_scan_width\":%u}}}", channels, mode_names[mode], round_cost,
        select_cost, common, round_width, select_width);
    tu_config_t cfg;
    if (tu_config_load_string(json, &cfg, error, sizeof(error))) {
        fprintf(stderr, "parse failed: %s\n", error);
        return 1;
    }
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    unsigned cw = common ? common : 1;
    if (cfg.dma_drr_round_scan_width != round_width ||
        cfg.dma_drr_select_scan_width != select_width ||
        rt.dma_drr_scan_width != cw ||
        rt.dma_drr_round_scan_width != round_width ||
        rt.dma_drr_select_scan_width != select_width) return 1;
    tu_init_with_config(&rt);
    if (g_tu_dma.drr_scan_width != cw ||
        g_tu_dma.drr_round_scan_width != round_width ||
        g_tu_dma.drr_select_scan_width != select_width ||
        g_tu_dma.bus_mode != TU_DMA_BUS_MODE_SHARED_SERIAL ||
        g_tu_dma.arb_policy != TU_DMA_ARB_DEFICIT_ROUND_ROBIN) return 1;
    g_tu.sram_w.banks.bw_modeling = false;
    g_tu_dma.next_shared_channel = cursor;
    unsigned char source[256], host[256];
    memset(source, 0x97, sizeof(source));
    memset(host, 0, sizeof(host));
    memset(tu_sram_raw_ptr(&g_tu.sram_w), store ? 0x97 : 0, bytes);
    tu_dma_descriptor_t *d = tu_dma_desc_create_linear(target,
        store ? TU_DMA_DIR_TU_TO_HOST : TU_DMA_DIR_HOST_TO_TU,
        &g_tu.sram_w, 0, store ? host : source, 1, bytes);
    int fail = !d || !tu_dma_submit_desc(d);
    while (!fail && !g_tu_dma.channels[target].total_completed &&
           g_tu_dma.current_cycle < 10000) tu_dma_tick();
    unsigned rw = round_width ? round_width : cw;
    unsigned sw = select_width ? select_width : cw;
    unsigned probes = (target + channels - cursor) % channels + 1;
    unsigned round_groups = mode == 0 ? 1 : groups(channels, rw);
    unsigned select_groups = mode == 0 ? 1 : groups(mode == 1 ? channels : probes, sw);
    unsigned rounds = (bytes + 63) / 64 - 1;
    uint64_t expected = 1 + 50 + (bytes + 31) / 32 +
        rounds * round_cost * round_groups + select_cost * select_groups;
    if (!fail) {
        const void *result = store ? (void *)host : tu_sram_raw_ptr(&g_tu.sram_w);
        fail = d->cycles_completed != expected ||
            g_tu_dma.current_cycle != expected || !d->completed ||
            memcmp(result, source, bytes) || g_tu_dma.total_bytes != bytes ||
            g_tu_dma.total_occupied_bytes != bytes || g_tu_dma.channels[target].active;
        printf("%u %u %u %u %s %u %u %u %u %u %s %llu %llu %s\n",
            channels, common, round_width, select_width, mode_names[mode], cursor,
            target, bytes, round_cost, select_cost, store ? "store" : "load",
            (unsigned long long)d->cycles_completed,
            (unsigned long long)expected, fail ? "FAIL" : "PASS");
    }
    tu_shutdown();
    if (d) { d->next = NULL; tu_dma_desc_destroy(d); }
    ++timing_rows;
    return fail;
}

static int controls(void) {
    tu_config_t cfg;
    char json[256], error[256];
    const char *keys[] = {"drr_round_scan_width", "drr_select_scan_width"};
    const char *invalid[] = {"-1", "9", "4294967296", "1.5", "true", "null", "\"2\""};
    for (unsigned s = 0; s < 2; ++s) {
        for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
            snprintf(json, sizeof(json), "{\"tu\":{\"dma\":{\"%s\":%s}}}", keys[s], invalid[i]);
            if (!tu_config_load_string(json, &cfg, error, sizeof(error))) {
                fprintf(stderr, "invalid %s accepted: %s\n", keys[s], invalid[i]);
                return 1;
            }
        }
        tu_config_default(&cfg);
        if (s) cfg.dma_drr_select_scan_width = UINT32_MAX;
        else cfg.dma_drr_round_scan_width = UINT32_MAX;
        if (!tu_config_validate(&cfg, error, sizeof(error))) return 1;
    }
    tu_config_default(&cfg);
    if (cfg.dma_drr_scan_width != 1 || cfg.dma_drr_round_scan_width ||
        cfg.dma_drr_select_scan_width) return 1;
    if (tu_config_load("config/tu_config.json", &cfg, error, sizeof(error)) ||
        cfg.dma_drr_scan_width != 1 || cfg.dma_drr_round_scan_width ||
        cfg.dma_drr_select_scan_width) return 1;
    memset(&cfg, 0, sizeof(cfg));
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (rt.dma_drr_scan_width != 1 || rt.dma_drr_round_scan_width ||
        rt.dma_drr_select_scan_width) return 1;
    rt = tu_runtime_config_default();
    if (rt.dma_drr_scan_width != 1 || rt.dma_drr_round_scan_width ||
        rt.dma_drr_select_scan_width) return 1;
    rt.dma_drr_scan_width = 0;
    tu_init_with_config(&rt);
    if (g_tu_dma.drr_scan_width != 1 || g_tu_dma.drr_round_scan_width ||
        g_tu_dma.drr_select_scan_width) return 1;
    g_tu_dma.channels[0].drr_deficit_credits = 17;
    g_tu_dma.drr_continue_visit = true;
    g_tu_dma.drr_continue_channel = 0;
    g_tu_dma.next_shared_channel = 2;
    if (!tu_dma_set_drr_stage_scan_widths(3, 7)) return 1;
    tu_dma_engine_t saved;
    memcpy(&saved, &g_tu_dma, sizeof(saved));
    if (tu_dma_set_drr_stage_scan_widths(2, 9) || memcmp(&saved, &g_tu_dma, sizeof(saved)) ||
        tu_dma_set_drr_stage_scan_widths(UINT32_MAX, 2) ||
        memcmp(&saved, &g_tu_dma, sizeof(saved))) return 1;
    if (!tu_dma_set_drr_stage_scan_widths(0, 4)) return 1;
    saved.drr_round_scan_width = 0;
    saved.drr_select_scan_width = 4;
    if (memcmp(&saved, &g_tu_dma, sizeof(saved))) return 1;
    if (!tu_dma_set_drr_scan_width(5)) return 1;
    saved.drr_scan_width = 5;
    if (memcmp(&saved, &g_tu_dma, sizeof(saved))) return 1;
    tu_shutdown();
    tu_dma_init(true);
    int fail = g_tu_dma.drr_scan_width != 1 || g_tu_dma.drr_round_scan_width ||
        g_tu_dma.drr_select_scan_width;
    tu_dma_destroy();
    return fail;
}

/* Both requests are ready before the first decision; no speculative grants to
 * later positions in the winning scan group. Reconfigure with an active request
 * and a queued peer, proving no credit/cursor/ownership reset or active retime. */
static int queued_case(unsigned round_width, unsigned select_width, bool store) {
    tu_runtime_config_t rt = tu_runtime_config_default();
    rt.dma_num_channels = 8;
    rt.dma_async_mode = true;
    rt.dma_bus_mode = TU_DMA_BUS_MODE_SHARED_SERIAL;
    rt.dma_arb_policy = TU_DMA_ARB_DEFICIT_ROUND_ROBIN;
    rt.dma_drr_quantum_bytes = 64;
    rt.dma_drr_round_cost_mode = TU_DMA_DRR_ROUND_COST_PER_VISITED_CHANNEL;
    rt.dma_drr_round_issue_cycles = rt.dma_drr_select_issue_cycles = 1;
    rt.dma_drr_round_scan_width = round_width;
    rt.dma_drr_select_scan_width = select_width;
    tu_init_with_config(&rt);
    g_tu.sram_w.banks.bw_modeling = false;
    g_tu_dma.next_shared_channel = 6;
    unsigned char src[2][256], host[2][256];
    memset(src[0], 0x36, 256); memset(src[1], 0x8a, 256); memset(host, 0, sizeof(host));
    memcpy(tu_sram_raw_ptr(&g_tu.sram_w), store ? (void *)src : (void *)host, sizeof(src));
    tu_dma_direction_t direction = store ? TU_DMA_DIR_TU_TO_HOST : TU_DMA_DIR_HOST_TO_TU;
    tu_dma_descriptor_t *a = tu_dma_desc_create_linear(7, direction,
        &g_tu.sram_w, 0, store ? host[0] : src[0], 1, 256);
    tu_dma_descriptor_t *b = tu_dma_desc_create_linear(0, direction,
        &g_tu.sram_w, 256, store ? host[1] : src[1], 1, 256);
    int fail = !a || !b || !tu_dma_submit_desc(a) || !tu_dma_submit_desc(b);
    if (!fail) tu_dma_tick();
    unsigned rw = round_width ? round_width : 1;
    unsigned sw = select_width ? select_width : 1;
    uint64_t first = 59 + 3 * groups(8, rw) + groups(2, sw);
    if (!fail) {
        fail = a->cycles_completed != first || b->completed ||
            g_tu_dma.channels[0].drr_deficit_credits != 192 ||
            g_tu_dma.channels[7].active != a || g_tu_dma.channels[0].head != b;
        tu_dma_engine_t saved;
        memcpy(&saved, &g_tu_dma, sizeof(saved));
        if (!tu_dma_set_drr_stage_scan_widths(0, 4) || !tu_dma_set_drr_scan_width(3)) fail = 1;
        saved.drr_scan_width = 3;
        saved.drr_round_scan_width = 0;
        saved.drr_select_scan_width = 4;
        if (memcmp(&saved, &g_tu_dma, sizeof(saved)) || a->cycles_completed != first) fail = 1;
    }
    while (!fail && !g_tu_dma.channels[0].total_completed &&
           g_tu_dma.current_cycle < 10000) tu_dma_tick();
    uint64_t second = first + 59;
    const void *result = store ? (void *)host : tu_sram_raw_ptr(&g_tu.sram_w);
    if (!fail) fail = b->cycles_completed != second || g_tu_dma.current_cycle != second ||
        g_tu_dma.total_bytes != 512 || g_tu_dma.total_occupied_bytes != 512 ||
        memcmp(result, src, sizeof(src));
    tu_shutdown();
    if (a) { a->next = NULL; tu_dma_desc_destroy(a); }
    if (b) { b->next = NULL; tu_dma_desc_destroy(b); }
    ++queue_rows;
    return fail;
}

/* Zero overrides follow subsequent common-width updates at execution time. */
static int live_inheritance(void) {
    for (unsigned s = 0; s < 3; ++s) {
        tu_runtime_config_t rt = tu_runtime_config_default();
        rt.dma_num_channels = 8;
        rt.dma_async_mode = true;
        rt.dma_bus_mode = TU_DMA_BUS_MODE_SHARED_SERIAL;
        rt.dma_arb_policy = TU_DMA_ARB_DEFICIT_ROUND_ROBIN;
        rt.dma_drr_quantum_bytes = 64;
        rt.dma_drr_round_cost_mode = TU_DMA_DRR_ROUND_COST_PER_VISITED_CHANNEL;
        rt.dma_drr_round_issue_cycles = rt.dma_drr_select_issue_cycles = 1;
        tu_init_with_config(&rt);
        g_tu.sram_w.banks.bw_modeling = false;
        unsigned char src[256]; memset(src, 0xa5, sizeof(src));
        tu_dma_descriptor_t *d = tu_dma_desc_create_linear(7, TU_DMA_DIR_HOST_TO_TU,
            &g_tu.sram_w, 0, src, 1, sizeof(src));
        int fail = !d || !tu_dma_submit_desc(d);
        if (!tu_dma_set_drr_stage_scan_widths(s == 1 ? 1 : 0, s == 0 ? 1 : 0) ||
            !tu_dma_set_drr_scan_width(s == 2 ? 3 : 8)) fail = 1;
        while (!fail && !g_tu_dma.channels[7].total_completed &&
               g_tu_dma.current_cycle < 1000) tu_dma_tick();
        const uint64_t expected[] = {70, 84, 71};
        if (!fail) fail = d->cycles_completed != expected[s] ||
            g_tu_dma.current_cycle != expected[s] ||
            memcmp(tu_sram_raw_ptr(&g_tu.sram_w), src, sizeof(src));
        tu_shutdown();
        if (d) { d->next = NULL; tu_dma_desc_destroy(d); }
        if (fail) return 1;
    }
    return 0;
}

static int unaffected_paths(void) {
    for (unsigned path = 0; path < 3; ++path) {
        tu_runtime_config_t rt = tu_runtime_config_default();
        rt.dma_async_mode = path != 2;
        rt.dma_bus_mode = path == 0 ? TU_DMA_BUS_MODE_INDEPENDENT : TU_DMA_BUS_MODE_SHARED_SERIAL;
        rt.dma_arb_policy = path == 1 ? TU_DMA_ARB_ROUND_ROBIN : TU_DMA_ARB_DEFICIT_ROUND_ROBIN;
        rt.dma_drr_round_cost_mode = TU_DMA_DRR_ROUND_COST_PER_VISITED_CHANNEL;
        rt.dma_drr_round_issue_cycles = rt.dma_drr_select_issue_cycles = 1024;
        rt.dma_drr_round_scan_width = 8;
        rt.dma_drr_select_scan_width = 3;
        tu_init_with_config(&rt);
        g_tu.sram_w.banks.bw_modeling = false;
        unsigned char src[64]; memset(src, 0x3e, sizeof(src));
        tu_dma_descriptor_t *d = tu_dma_desc_create_linear(0, TU_DMA_DIR_HOST_TO_TU,
            &g_tu.sram_w, 0, src, 1, sizeof(src));
        int fail = !d || !tu_dma_submit_desc(d);
        while (!fail && !g_tu_dma.channels[0].total_completed &&
               g_tu_dma.current_cycle < 1000) tu_dma_tick();
        if (!fail) fail = d->cycles_completed != (path == 2 ? 52u : 53u) ||
            memcmp(tu_sram_raw_ptr(&g_tu.sram_w), src, sizeof(src));
        tu_shutdown();
        if (d) { d->next = NULL; tu_dma_desc_destroy(d); }
        if (fail) return 1;
    }
    return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (controls()) { fputs("FAIL: controls\n", stderr); return 1; }
    puts("channels common round_width select_width mode cursor target bytes round_cost select_cost direction completion expected status");
    if (run_case(8, 1, 8, 1, 2, 0, 7, 256, false, 1, 1)) return 1;
    for (unsigned rw = 0; rw <= 8; ++rw)
        for (unsigned sw = 0; sw <= 8; ++sw) {
            for (unsigned mode = 0; mode < 3; ++mode)
                for (unsigned cursor = 0; cursor <= 7; cursor += 7)
                    for (unsigned target = 0; target < 8; ++target)
                        for (unsigned store = 0; store < 2; ++store)
                            if (run_case(8, 3, rw, sw, mode, cursor, target,
                                         256, store, 1, 1)) return 1;
            for (unsigned store = 0; store < 2; ++store) {
                if (run_case(8, 3, rw, sw, 2, 0, 7, 32, store, 1, 1) ||
                    run_case(3, 0, rw, sw, 2, 2, 1, 32, store, 2, 3) ||
                    run_case(3, 3, rw, sw, 2, 2, 1, 256, store, 2, 3) ||
                    run_case(8, 3, rw, sw, 2, 7, 6, 256, store, 0, 0) ||
                    queued_case(rw, sw, store)) return 1;
            }
        }
    if (live_inheritance()) { fputs("FAIL: live inheritance\n", stderr); return 1; }
    if (unaffected_paths()) { fputs("FAIL: unaffected paths\n", stderr); return 1; }
    printf("PASS: %u timing rows; %u simultaneous-queue/reconfiguration gates; defaults, inheritance, parse-to-live, rejection, ordered credit, bytes, unaffected paths\n",
           timing_rows, queue_rows);
    return 0;
}
