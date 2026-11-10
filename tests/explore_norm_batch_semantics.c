/* Pinned semantic exploration, not a BatchNorm conformance test.
 * Tests the current documented compatibility behavior against independent
 * row-normalization and two-channel training-BatchNorm calculations. */
#include "tu_cmodel/compute/normalization_engine.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const float x[4] = {1.0f, 10.0f, 3.0f, 14.0f};
static const float gamma4[4] = {2.0f, 0.5f, 2.0f, 0.5f};
static const float beta4[4] = {0.25f, -0.5f, 0.25f, -0.5f};
static const float eps = 0.01f;

static void expected_row(float out[4], unsigned axis, int mode, int beta) {
    unsigned width = axis ? axis : 4;
    for (unsigned r = 0; r < 4; r += width) {
        double mean = 0.0, var = 0.0;
        for (unsigned j = 0; j < width; ++j) mean += x[r+j];
        mean /= width;
        for (unsigned j = 0; j < width; ++j) {
            double v = mode == TU_NORM_RMS_NORM ? x[r+j] : x[r+j] - mean;
            var += v*v;
        }
        var /= width;
        for (unsigned j = 0; j < width; ++j) {
            double v = mode == TU_NORM_RMS_NORM ? x[r+j] : x[r+j] - mean;
            out[r+j] = (float)(v / sqrt(var + (double)eps) * gamma4[j] +
                ((beta && mode == TU_NORM_LAYER_NORM) ? beta4[j] : 0.0));
        }
    }
}

static void expected_batch(float out[4], int beta) {
    /* N=2, C=2, row-major NC; per-channel population variance. */
    for (unsigned c = 0; c < 2; ++c) {
        double m = ((double)x[c] + x[2+c]) / 2.0;
        double d0 = x[c] - m, d1 = x[2+c] - m;
        double var = (d0*d0 + d1*d1) / 2.0;
        out[c] = (float)(d0 / sqrt(var + (double)eps) * gamma4[c] + (beta ? beta4[c] : 0.0));
        out[2+c] = (float)(d1 / sqrt(var + (double)eps) * gamma4[c] + (beta ? beta4[c] : 0.0));
    }
}

static double maxdiff(const float a[4], const float b[4]) {
    double m = 0.0;
    for (unsigned i = 0; i < 4; ++i) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return INFINITY;
        double d = fabs((double)a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    int failures = 0, rows = 0;
    puts("mode axis placement beta stalls max_row_error max_batch_error output[0..3]");
    for (int mode = 0; mode < 3; ++mode) {
        for (unsigned axis_i = 0; axis_i < 2; ++axis_i) {
            unsigned axis = axis_i ? 2 : 0;
            for (int inplace = 0; inplace < 2; ++inplace) {
                for (int beta = 0; beta < 2; ++beta) {
                    tu_sram_region_t s;
                    tu_sram_init(&s, 256, "norm_semantic_probe");
                    tu_sram_write_bulk(&s, 0, x, sizeof(x));
                    tu_sram_write_bulk(&s, 192, gamma4, sizeof(gamma4));
                    tu_sram_write_bulk(&s, 208, beta4, sizeof(beta4));
                    tu_norm_desc_t d = { .mode = (tu_norm_mode_t)mode,
                        .data_sram = &s, .data_offset = 0, .elem_count = 4,
                        .gamma_sram = &s, .gamma_offset = 192,
                        .beta_sram = beta ? &s : NULL, .beta_offset = 208,
                        .epsilon = eps, .norm_axis_dim = axis,
                        .in_place = inplace != 0, .out_offset = 128 };
                    uint64_t cycles = tu_norm_execute(&d);
                    float got[4], row[4], batch[4];
                    memcpy(got, (const unsigned char *)tu_sram_raw_ptr(&s) + (inplace ? 0 : 128), sizeof(got));
                    expected_row(row, axis, mode, beta);
                    expected_batch(batch, beta);
                    double er = maxdiff(got, row), eb = maxdiff(got, batch);
                    printf("%s %u %s %d %llu %.9g %.9g %.7g %.7g %.7g %.7g\n",
                        tu_norm_mode_name((tu_norm_mode_t)mode), axis,
                        inplace ? "in" : "out", beta, (unsigned long long)cycles,
                        er, eb, got[0], got[1], got[2], got[3]);
                    if (cycles == UINT64_MAX || er > 1e-6 ||
                        (mode == TU_NORM_BATCH_NORM && eb < 0.1)) ++failures;
                    tu_sram_destroy(&s);
                    ++rows;
                }
            }
        }
    }
    printf("snapshot_rows=%d failures=%d\n", rows, failures);
    return failures ? 1 : 0;
}
