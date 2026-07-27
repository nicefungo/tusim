/*
 * Per-core heterogeneous dataflow sweep.
 *
 * Executes identical GEMMs concurrently representable on three independent
 * core snapshots configured as WS, OS, and RS. Fails closed if the retained
 * mode changes, numerical outputs differ, or unsupported NLR is accepted.
 */
#include "tu_cmodel/tu_core.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *label;
    uint16_t m, n, k;
} workload_t;

static int run_workload(const workload_t *wl) {
    const tu_dataflow_id_t modes[3] = {
        TU_DATAFLOW_WEIGHT_STATIONARY,
        TU_DATAFLOW_OUTPUT_STATIONARY,
        TU_DATAFLOW_ROW_STATIONARY,
    };
    tu_runtime_config_t cfg = tu_runtime_config_default();
    cfg.pe_pipeline_depth = 2;
    tu_core_t *cores[3] = {NULL, NULL, NULL};
    fp16_t *w = NULL, *a = NULL;
    fp32_t *out[3] = {NULL, NULL, NULL};
    uint64_t cycles[3] = {0, 0, 0};
    int failed = 0;

    size_t w_count = (size_t)wl->m * wl->k;
    size_t a_count = (size_t)wl->k * wl->n;
    size_t o_count = (size_t)wl->m * wl->n;
    w = malloc(w_count * sizeof(*w));
    a = malloc(a_count * sizeof(*a));
    for (int i = 0; i < 3; ++i) out[i] = malloc(o_count * sizeof(*out[i]));
    if (!w || !a || !out[0] || !out[1] || !out[2]) {
        failed = 1;
        goto cleanup;
    }
    for (size_t i = 0; i < w_count; ++i)
        w[i] = tu_fp32_to_fp16((float)((int)(i % 11) - 5) * 0.125f);
    for (size_t i = 0; i < a_count; ++i)
        a[i] = tu_fp32_to_fp16((float)((int)(i % 7) - 3) * 0.25f);

    for (int i = 0; i < 3; ++i) {
        cores[i] = tu_core_create_with_id((uint32_t)i, &cfg);
        if (!cores[i] || tu_core_set_dataflow(cores[i], modes[i]) != 0 ||
            tu_core_get_dataflow(cores[i]) != modes[i]) {
            failed = 1;
            goto cleanup;
        }
        tu_core_dma_load_w(cores[i], w, 0, (uint32_t)(w_count * sizeof(*w)));
        tu_core_dma_load_a(cores[i], a, 0, (uint32_t)(a_count * sizeof(*a)));
        uint64_t before = cores[i]->state.estimated_cycles;
        tu_core_mma(cores[i], wl->m, wl->n, wl->k, 0, 0, 0, false);
        cycles[i] = cores[i]->state.estimated_cycles - before;
        if (tu_core_get_dataflow(cores[i]) != modes[i]) {
            failed = 1;
            goto cleanup;
        }
        tu_core_dma_store_o(cores[i], out[i], 0,
                            (uint32_t)(o_count * sizeof(*out[i])));
    }

    if (memcmp(out[0], out[1], o_count * sizeof(*out[0])) != 0 ||
        memcmp(out[0], out[2], o_count * sizeof(*out[0])) != 0) {
        failed = 1;
        goto cleanup;
    }
    if (tu_core_set_dataflow(cores[1], TU_DATAFLOW_NO_LOCAL_REUSE) == 0 ||
        tu_core_get_dataflow(cores[1]) != TU_DATAFLOW_OUTPUT_STATIONARY) {
        failed = 1;
        goto cleanup;
    }

    printf("%-22s %3ux%-3ux%-3u %10llu %10llu %10llu  exact\n",
           wl->label, wl->m, wl->n, wl->k,
           (unsigned long long)cycles[0],
           (unsigned long long)cycles[1],
           (unsigned long long)cycles[2]);

cleanup:
    for (int i = 0; i < 3; ++i) {
        tu_core_destroy(cores[i]);
        free(out[i]);
    }
    free(w);
    free(a);
    return failed;
}

int main(void) {
    const workload_t workloads[] = {
        {"edge + multi-K", 31, 19, 17},
        {"square", 64, 64, 64},
        {"wide small-K", 32, 128, 16},
        {"tall small-K", 128, 32, 16},
    };
    int failures = 0;

    printf("Per-core heterogeneous dataflow sweep (16x16 PE, depth=2)\n");
    printf("Workload                 MxNxK           WS         OS         RS  outputs\n");
    for (size_t i = 0; i < sizeof(workloads) / sizeof(workloads[0]); ++i)
        failures += run_workload(&workloads[i]);

    if (failures) {
        fprintf(stderr, "FAIL: %d workload(s) violated selection/output gates\n", failures);
        return 1;
    }
    printf("PASS: all core-local modes retained; all outputs byte-identical\n");
    return 0;
}
