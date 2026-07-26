/*
 * Executable WS / OS / RS dataflow evidence sweep.
 *
 * This is a fail-closed gate, not merely a report: every row asserts that the
 * requested plugin is active in the exact global state that executes, checks
 * FP32 output bits against an independent canonical-conversion oracle, and
 * checks the live dispatcher's per-K-tile cycle accounting.
 */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/compute/dataflow/dataflow_interface.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint16_t m, n, k;
    const char *name;
} workload_t;

static const workload_t workloads[] = {
    {31, 19, 17, "edge-multik"},
    {64, 64, 64, "square"},
    {32, 128, 16, "wide-smallk"},
    {128, 32, 16, "tall-smallk"},
};

static const char *df_names[] = {
    "weight_stationary", "output_stationary", "row_stationary"
};

static uint32_t fbits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void fill_inputs(fp16_t *w, fp16_t *a, uint16_t m, uint16_t n, uint16_t k) {
    for (uint32_t i = 0; i < (uint32_t)m * k; ++i) {
        int value = (int)((i * 17u + 3u) % 19u) - 9;
        w[i] = tu_fp32_to_fp16((float)value / 8.0f);
    }
    for (uint32_t i = 0; i < (uint32_t)k * n; ++i) {
        int value = (int)((i * 11u + 5u) % 23u) - 11;
        a[i] = tu_fp32_to_fp16((float)value / 16.0f);
    }
}

/* Match the dispatcher's FP32 accumulation grouping: one partial sum per K tile,
 * then add that partial sum to O. Conversion itself uses the canonical module. */
static void oracle(float *out, const fp16_t *w, const fp16_t *a,
                   uint16_t m_dim, uint16_t n_dim, uint16_t k_dim,
                   uint16_t tile_m, uint16_t tile_n, uint16_t tile_k) {
    memset(out, 0, (size_t)m_dim * n_dim * sizeof(*out));
    for (uint16_t ms = 0; ms < m_dim; ms += tile_m) {
        uint16_t mc = ms + tile_m <= m_dim ? tile_m : m_dim - ms;
        for (uint16_t ns = 0; ns < n_dim; ns += tile_n) {
            uint16_t nc = ns + tile_n <= n_dim ? tile_n : n_dim - ns;
            for (uint16_t ks = 0; ks < k_dim; ks += tile_k) {
                uint16_t kc = ks + tile_k <= k_dim ? tile_k : k_dim - ks;
                for (uint16_t m = 0; m < mc; ++m) {
                    for (uint16_t n = 0; n < nc; ++n) {
                        float partial = 0.0f;
                        for (uint16_t k = 0; k < kc; ++k) {
                            partial += tu_fp16_to_fp32(w[(ms + m) * k_dim + ks + k]) *
                                       tu_fp16_to_fp32(a[(ks + k) * n_dim + ns + n]);
                        }
                        out[(ms + m) * n_dim + ns + n] += partial;
                    }
                }
            }
        }
    }
}

static uint64_t expected_cycles(int mode, const workload_t *wl,
                                uint16_t rows, uint16_t cols, uint16_t pd) {
    uint64_t cycles = 0;
    for (uint16_t ms = 0; ms < wl->m; ms += rows) {
        uint16_t mc = ms + rows <= wl->m ? rows : wl->m - ms;
        for (uint16_t ns = 0; ns < wl->n; ns += cols) {
            uint16_t nc = ns + cols <= wl->n ? cols : wl->n - ns;
            for (uint16_t ks = 0; ks < wl->k; ks += cols) {
                uint16_t kc = ks + cols <= wl->k ? cols : wl->k - ks;
                if (mode == TU_DATAFLOW_WEIGHT_STATIONARY)
                    cycles += (uint64_t)pd * nc + kc + (uint64_t)pd * mc;
                else if (mode == TU_DATAFLOW_OUTPUT_STATIONARY)
                    cycles += kc + (kc + 3u) / 4u;
                else if (pd <= 1)
                    cycles += kc;
                else
                    cycles += (uint64_t)(pd - 1u) * nc + 1u + kc +
                              (uint64_t)(pd - 1u) * mc;
            }
        }
    }
    return cycles;
}

static int run_row(const workload_t *wl, int mode, uint16_t pd,
                   uint64_t *cycles_out) {
    size_t w_count = (size_t)wl->m * wl->k;
    size_t a_count = (size_t)wl->k * wl->n;
    size_t o_count = (size_t)wl->m * wl->n;
    fp16_t *w = malloc(w_count * sizeof(*w));
    fp16_t *a = malloc(a_count * sizeof(*a));
    float *got = malloc(o_count * sizeof(*got));
    float *want = malloc(o_count * sizeof(*want));
    if (!w || !a || !got || !want) {
        fprintf(stderr, "allocation failed\n");
        free(w); free(a); free(got); free(want);
        return 1;
    }

    fill_inputs(w, a, wl->m, wl->n, wl->k);
    tu_runtime_config_t cfg = tu_runtime_config_default();
    cfg.dataflow_mode = mode;
    cfg.pe_pipeline_depth = pd;
    tu_init_with_config(&cfg);

    if (!g_tu.dataflow || (int)g_tu.dataflow->id != mode ||
        strcmp(tu_get_dataflow_name(), df_names[mode]) != 0) {
        fprintf(stderr, "%s pd=%u: requested mode %d but active=%s\n",
                wl->name, pd, mode, tu_get_dataflow_name());
        free(w); free(a); free(got); free(want);
        return 1;
    }

    memcpy(tu_sram_raw_ptr(&g_tu.sram_w), w, w_count * sizeof(*w));
    memcpy(tu_sram_raw_ptr(&g_tu.sram_a), a, a_count * sizeof(*a));
    memset(tu_sram_raw_ptr(&g_tu.sram_o), 0, o_count * sizeof(*got));
    uint64_t before = g_tu.estimated_cycles;
    tu_mma(wl->m, wl->n, wl->k, 0, 0, 0, false);
    uint64_t cycles = g_tu.estimated_cycles - before;
    memcpy(got, tu_sram_raw_ptr(&g_tu.sram_o), o_count * sizeof(*got));
    oracle(want, w, a, wl->m, wl->n, wl->k,
           cfg.pe_rows, cfg.pe_cols, cfg.pe_cols);

    int failed = 0;
    for (size_t i = 0; i < o_count; ++i) {
        if (fbits(got[i]) != fbits(want[i])) {
            fprintf(stderr, "%s/%s: output[%zu] bits %08x != oracle %08x\n",
                    wl->name, df_names[mode], i, fbits(got[i]), fbits(want[i]));
            failed = 1;
            break;
        }
    }
    uint64_t expected = expected_cycles(mode, wl, cfg.pe_rows, cfg.pe_cols, pd);
    if (cycles != expected) {
        fprintf(stderr, "%s/%s pd=%u: cycles=%" PRIu64 " expected=%" PRIu64 "\n",
                wl->name, df_names[mode], pd, cycles, expected);
        failed = 1;
    }
    *cycles_out = cycles;
    free(w); free(a); free(got); free(want);
    return failed;
}

int main(void) {
    int failures = 0;
    printf("dataflow,workload,M,N,K,pipeline_depth,live_mma_cycles,status\n");
    for (size_t wi = 0; wi < sizeof(workloads) / sizeof(workloads[0]); ++wi) {
        for (int mode = TU_DATAFLOW_WEIGHT_STATIONARY;
             mode <= TU_DATAFLOW_ROW_STATIONARY; ++mode) {
            uint64_t cycles = 0;
            int failed = run_row(&workloads[wi], mode, 2, &cycles);
            failures += failed;
            printf("%s,%s,%u,%u,%u,2,%" PRIu64 ",%s\n",
                   df_names[mode], workloads[wi].name,
                   workloads[wi].m, workloads[wi].n, workloads[wi].k,
                   cycles, failed ? "FAIL" : "PASS");
        }
    }

    /* A nondefault-depth behavioral gate proves parse/runtime storage is not a
     * documented no-op. OS is expected to be depth-independent; WS/RS are not. */
    for (uint16_t pd = 1; pd <= 4; pd = (uint16_t)(pd * 2)) {
        for (int mode = TU_DATAFLOW_WEIGHT_STATIONARY;
             mode <= TU_DATAFLOW_ROW_STATIONARY; ++mode) {
            uint64_t cycles = 0;
            int failed = run_row(&workloads[0], mode, pd, &cycles);
            failures += failed;
            printf("%s,%s,%u,%u,%u,%u,%" PRIu64 ",%s\n",
                   df_names[mode], "pipeline-sensitivity",
                   workloads[0].m, workloads[0].n, workloads[0].k,
                   pd, cycles, failed ? "FAIL" : "PASS");
        }
    }

    if (failures) {
        fprintf(stderr, "dataflow sweep failed: %d row(s)\n", failures);
        return 1;
    }
    printf("# all rows passed active-plugin, raw-bit oracle, and live-cycle gates\n");
    return 0;
}
