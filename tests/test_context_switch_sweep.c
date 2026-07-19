/* Context retention trade-off sweep: execute the real manager cost model. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_core.h"
#include "tu_cmodel/infra/tu_context.h"
#include <stdio.h>

static const char *scope_name(tu_ctx_save_scope_t s)
{
    return s == TU_CTX_SAVE_FULL_SRAM ? "full" :
           s == TU_CTX_SAVE_LIVE_SRAM ? "live25" : "control";
}

int main(void)
{
    const uint32_t totals_kib[] = {128, 256, 512};
    const uint32_t bandwidths[] = {16, 32, 64};

    printf("Context save-scope sweep (fixed=100 cycles)\n");
    printf("SRAM_KiB scope    state_B switch_cycles\n");
    for (size_t si = 0; si < 3; si++) {
        tu_runtime_config_t rt = tu_runtime_config_default();
        rt.sram_w_size = totals_kib[si] * 512;
        rt.sram_a_size = totals_kib[si] * 256;
        rt.sram_o_size = totals_kib[si] * 256;
        tu_core_t *core = tu_core_create(&rt);
        if (!core) return 1;

        for (int mode = TU_CTX_SAVE_FULL_SRAM;
             mode <= TU_CTX_SAVE_CONTROL_ONLY; mode++) {
            tu_ctx_manager_config_t cfg = {
                .max_contexts = 2,
                .sched_policy = TU_CTX_SCHED_ROUND_ROBIN,
                .switch_overhead = 100,
                .save_scope = (tu_ctx_save_scope_t)mode,
                .live_w_bytes = rt.sram_w_size / 4,
                .live_a_bytes = rt.sram_a_size / 4,
                .live_o_bytes = rt.sram_o_size / 4,
                .state_bytes_per_cycle = 32,
            };
            tu_ctx_manager_t *mgr = tu_ctx_manager_create(core, &cfg);
            if (!mgr || tu_ctx_alloc(mgr) < 0 || tu_ctx_alloc(mgr) < 0 ||
                tu_ctx_switch(mgr, 1) != 0) return 2;
            printf("%8u %-8s %7llu %13llu\n", totals_kib[si],
                   scope_name((tu_ctx_save_scope_t)mode),
                   (unsigned long long)tu_ctx_get(mgr, 1)->saved_sram_bytes,
                   (unsigned long long)tu_ctx_get_switch_overhead(mgr));
            tu_ctx_manager_destroy(mgr);
        }
        tu_core_destroy(core);
    }

    printf("\nFull-save bandwidth sensitivity (256 KiB SRAM)\n");
    printf("BW_Bpc switch_cycles\n");
    for (size_t bi = 0; bi < 3; bi++) {
        tu_runtime_config_t rt = tu_runtime_config_default();
        tu_core_t *core = tu_core_create(&rt);
        tu_ctx_manager_config_t cfg = {
            .max_contexts = 2, .sched_policy = TU_CTX_SCHED_ROUND_ROBIN,
            .switch_overhead = 100, .save_scope = TU_CTX_SAVE_FULL_SRAM,
            .state_bytes_per_cycle = bandwidths[bi],
        };
        tu_ctx_manager_t *mgr = tu_ctx_manager_create(core, &cfg);
        if (!mgr || tu_ctx_alloc(mgr) < 0 || tu_ctx_alloc(mgr) < 0 ||
            tu_ctx_switch(mgr, 1) != 0) return 3;
        printf("%6u %13llu\n", bandwidths[bi],
               (unsigned long long)tu_ctx_get_switch_overhead(mgr));
        tu_ctx_manager_destroy(mgr);
        tu_core_destroy(core);
    }
    return 0;
}
