/* Diagnostic snapshot of existing context/meter interaction, not a correctness gate.
 * An expected change in behavior should update this probe rather than preserve the defect. */
#include "tu_cmodel/tu_core.h"
#include "tu_cmodel/infra/tu_context.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int run(int write, int switching) {
    int ok = 1;
    tu_runtime_config_t cfg = tu_runtime_config_default();
    cfg.sram_num_banks = 1; cfg.sram_words_per_cycle = 1;
    cfg.sram_bw_window_cycles = 4; cfg.sram_stall_penalty = 2;
    tu_core_t *core = tu_core_create(&cfg);
    if (!core) return 1;
    tu_ctx_manager_config_t cc = { .max_contexts = 2, .save_scope = TU_CTX_SAVE_FULL_SRAM };
    tu_ctx_manager_t *mgr = tu_ctx_manager_create(core, &cc);
    if (!mgr) { tu_core_destroy(core); return 1; }
    if (tu_ctx_alloc(mgr) != 0 || tu_ctx_alloc(mgr) != 1) { ok = 0; goto done; }
    tu_sram_region_t *r = tu_core_get_sram_a(core);
    uint32_t value = 0x78563412u, out = 0;
    memcpy(r->banks.data, &value, sizeof value);
    uint64_t first = write ? tu_sram_write(r, 0, &value) : tu_sram_read(r, 0, &out);
    if (first != 0 || (!write && out != value)) ok = 0;
    if (switching) {
        if (tu_ctx_switch(mgr, 1) != 0) { ok = 0; goto done; }
        tu_sram_advance_cycle(r, 5);
        if (tu_ctx_switch(mgr, 0) != 0) { ok = 0; goto done; }
    }
    uint64_t before = r->banks.current_cycle;
    uint64_t last = r->banks.bw_banks[0].last_refill_cycle;
    int available_before = r->banks.bw_banks[0].words_available;
    tu_sram_advance_cycle(r, 0);
    uint64_t last_after = r->banks.bw_banks[0].last_refill_cycle;
    int available_after = r->banks.bw_banks[0].words_available;
    uint64_t second = write ? tu_sram_write(r, 0, &value) : tu_sram_read(r, 0, &out);
    if (before != 0 || (switching ? last != 5 : last != 0) || last_after != 0 ||
        available_before != (switching ? 1 : 0) || available_after != available_before ||
        second != (switching ? 0u : 2u) ||
        memcmp(r->banks.data, &value, sizeof value) != 0 || (!write && out != value)) ok = 0;
    printf("%s,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d,%d,%" PRIu64 ",%s\n",
           write ? "write" : "read", switching ? "switch" : "control",
           before, last, last_after, available_before, available_after, second, ok ? "OBSERVED" : "MISMATCH");
done:
    tu_ctx_manager_destroy(mgr);
    tu_core_destroy(core);
    return !ok;
}
int main(void) {
    int failures = 0;
    puts("direction,path,restored_cycle,last_before,last_after,available_before,available_after,second_stall,result");
    for (int w = 0; w < 2; ++w) for (int s = 0; s < 2; ++s) failures += run(w, s);
    printf("snapshot_mismatches=%d\n", failures);
    return failures != 0;
}
