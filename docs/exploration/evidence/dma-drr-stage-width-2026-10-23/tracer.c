/* Independent DRR round/selection widths: aggregate timing, not RTL. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/infra/config.h"
#include "tu_cmodel/dma_descriptor.h"
#include <stdio.h>
#include <string.h>

static int tracer(void) {
    const char *json = "{\"tu\":{\"dma\":{\"channels\":8,\"async_mode\":true,"
        "\"bus_topology\":\"shared_serial\",\"arbitration\":\"deficit_round_robin\","
        "\"drr_quantum_bytes\":64,\"drr_round_cost_mode\":\"per_visited_channel\","
        "\"drr_round_issue_cycles\":1,\"drr_select_issue_cycles\":1,"
        "\"drr_scan_width\":1,\"drr_round_scan_width\":8,\"drr_select_scan_width\":1}}}";
    tu_config_t cfg;
    char error[256];
    if (tu_config_load_string(json, &cfg, error, sizeof(error))) return 1;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    tu_init_with_config(&rt);
    g_tu.sram_w.banks.bw_modeling = false;
    unsigned char source[256];
    memset(source, 0x97, sizeof(source));
    tu_dma_descriptor_t *d = tu_dma_desc_create_linear(7, TU_DMA_DIR_HOST_TO_TU,
        &g_tu.sram_w, 0, source, 1, sizeof(source));
    int fail = !d || !tu_dma_submit_desc(d);
    while (!fail && !g_tu_dma.channels[7].total_completed &&
           g_tu_dma.current_cycle < 1000) tu_dma_tick();
    if (!fail) {
        printf("round_width=8 select_width=1 completion=%llu expected=70\n",
               (unsigned long long)d->cycles_completed);
        fail = d->cycles_completed != 70 || g_tu_dma.current_cycle != 70 ||
            memcmp(tu_sram_raw_ptr(&g_tu.sram_w), source, sizeof(source)) != 0;
    }
    tu_shutdown();
    if (d) { d->next = NULL; tu_dma_desc_destroy(d); }
    return fail;
}

int main(void) { return tracer(); }
