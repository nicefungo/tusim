/* Finite normal-exponential LOG gate. Long-double residual oracle, relative
 * error (no absolute tolerance near zero), and explicit cutoff limitations. */
#include "tu_cmodel/compute/softmax_engine.h"
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAP 771
#define SRAM_BYTES 8192
static unsigned checks, failures, cases;
static long double max_relative;
#define CHECK(c) do { ++checks; if (!(c)) { \
    if (failures < 12) fprintf(stderr, "FAIL case=%u line=%d: %s\n", cases, __LINE__, #c); \
    ++failures; } } while (0)

static void near_value(float got, long double want) {
    CHECK(isfinite(got) && isfinite(want));
    if (!isfinite(got) || !isfinite(want)) return;
    if (want == 0) { CHECK(got == 0); return; }
    long double error = fabsl(((long double)got - want) / want);
    if (error > max_relative) max_relative = error;
    CHECK(error <= 4e-7L);
    CHECK(got < 0);
}

/* Only preprocessing follows the FP32 API. The oracle uses expl/log1pl,
 * excludes one maximum before adding tiny tails, and never calls a TU helper.
 * log(sum(exp)) in even long double would lose the gap-80 control. */
static void oracle(const float *x, const float *mask, unsigned kind, float scale,
                   unsigned rows, unsigned cols, long double *out,
                   long double *maxima, long double *sums) {
    for (unsigned r = 0; r < rows; ++r) {
        long double v[CAP], maximum = -INFINITY, residual = 0;
        unsigned anchor = 0;
        for (unsigned i = 0; i < cols; ++i) {
            unsigned k = r * cols + i;
            float a = x[k];
            if (kind == 1) a += mask[k];
            if (kind == 2) a *= mask[k];
            if (scale != 0) a *= scale;
            v[i] = a;
            if (v[i] > maximum) { maximum = v[i]; anchor = i; }
        }
        for (unsigned i = 0; i < cols; ++i)
            if (i != anchor) residual += expl(v[i] - maximum);
        long double normalizer = log1pl(residual);
        for (unsigned i = 0; i < cols; ++i)
            out[r * cols + i] = (v[i] - maximum) - normalizer;
        maxima[r] = maximum;
        sums[r] = 1 + residual;
    }
}

static void descriptor_case(unsigned cols, float gap, float scale,
                            unsigned kind, bool in_place, bool tied) {
    const unsigned rows = 3, n = rows * cols, in = 128;
    const unsigned out = in_place ? in : 4096;
    float x[CAP], mask[CAP], maxima[3], sums[3];
    long double want[CAP], em[3], es[3];
    unsigned char before[SRAM_BYTES], after[SRAM_BYTES];
    tu_sram_region_t s;
    ++cases;
    for (unsigned r = 0; r < rows; ++r) {
        unsigned anchor = r == 0 ? 0 : (r == 1 ? cols / 2 : cols - 1);
        for (unsigned i = 0; i < cols; ++i) {
            unsigned k = r * cols + i;
            float target = i == anchor || (tied && i == (anchor + 1) % cols)
                ? 0 : -gap;
            mask[k] = kind == 1 ? (i % 2 ? -0.5f : 0.5f) : 0.5f;
            float a = scale == 0 ? target : target / scale;
            x[k] = kind == 1 ? a - mask[k] : (kind == 2 ? a / mask[k] : a);
        }
    }
    oracle(x, mask, kind, scale, rows, cols, want, em, es);
    tu_sram_init(&s, SRAM_BYTES, "log-nearzero");
    unsigned char *raw = tu_sram_raw_ptr(&s);
    memset(raw, 0x5a, SRAM_BYTES);
    tu_sram_write_bulk(&s, in, x, n * sizeof(float));
    memcpy(before, raw, SRAM_BYTES);
    uint64_t reads = s.banks.reads, writes = s.banks.writes;
    uint64_t stalls = s.banks.stall_cycles;
    tu_softmax_desc_t d = {.mode = TU_SOFTMAX_LOG, .data_sram = &s,
        .data_offset = in, .elem_count = n, .axis_dim = cols,
        .mask = kind ? mask : NULL, .mask_is_additive = kind == 1,
        .scale = scale, .in_place = in_place, .out_offset = out,
        .max_out = maxima, .sum_out = sums};
    uint64_t result = tu_softmax_execute(&d);
    CHECK(result != UINT64_MAX);
    CHECK(s.banks.reads - reads == n);
    CHECK(s.banks.writes - writes == n);
    CHECK(result == s.banks.stall_cycles - stalls);
    memcpy(after, raw, SRAM_BYTES);
    CHECK(memcmp(before, after, out) == 0);
    CHECK(memcmp(before + out + n * 4, after + out + n * 4,
                 SRAM_BYTES - out - n * 4) == 0);
    for (unsigned r = 0; r < rows; ++r) {
        CHECK(isfinite(maxima[r]) && maxima[r] == em[r]);
        CHECK(isfinite(sums[r]) && fabsl(sums[r] - es[r]) <= 2e-7L * es[r]);
        for (unsigned i = 0; i < cols; ++i) {
            float got;
            memcpy(&got, raw + out + (r * cols + i) * 4, 4);
            near_value(got, want[r * cols + i]);
        }
    }
    if (gap == 20 && kind == 0) {
        memcpy(raw, before, SRAM_BYTES);
        d.max_out = NULL; d.sum_out = NULL;
        tu_sram_set_bw_modeling(&s, false);
        CHECK(tu_softmax_execute(&d) == 0);
        CHECK(memcmp(raw + out, after + out, n * 4) == 0);
    }
    tu_sram_destroy(&s);
}

static void public_paths(void) {
    const float gaps[] = {0, 1, 16, 17, 20, 40, 80, 87};
    const float scales[] = {0, 1, 0.5f, 2, -1};
    for (unsigned gi = 0; gi < sizeof(gaps) / sizeof(*gaps); ++gi)
        for (unsigned si = 0; si < sizeof(scales) / sizeof(*scales); ++si) {
            float x[6], got[6];
            long double want[6], maxima[3], sums[3];
            for (unsigned r = 0; r < 3; ++r) {
                x[r * 2 + r % 2] = 0;
                x[r * 2 + 1 - r % 2] = scales[si] == 0
                    ? -gaps[gi] : -gaps[gi] / scales[si];
            }
            oracle(x, NULL, 0, scales[si], 3, 2, want, maxima, sums);
            ++cases;
            memcpy(got, x, sizeof(got));
            for (unsigned r = 0; r < 3; ++r)
                tu_log_softmax_host(got + r * 2, 2, scales[si]);
            for (unsigned i = 0; i < 6; ++i) near_value(got[i], want[i]);
            for (unsigned batch = 0; batch < 2; ++batch)
                for (unsigned place = 0; place < 2; ++place) {
                    ++cases;
                    tu_sram_region_t s;
                    tu_sram_init(&s, 128, "log-nearzero-wrapper");
                    unsigned char *raw = tu_sram_raw_ptr(&s);
                    memset(raw, 0x6a, 128);
                    tu_sram_write_bulk(&s, 16, x, sizeof(x));
                    if (batch) CHECK(tu_log_softmax_2d(&s, 16, 3, 2,
                                                     scales[si], place) != UINT64_MAX);
                    else for (unsigned r = 0; r < 3; ++r)
                        CHECK(tu_log_softmax(&s, 16 + r * 8, 2,
                                             scales[si], place) != UINT64_MAX);
                    memcpy(got, raw + 16, sizeof(got));
                    for (unsigned i = 0; i < 6; ++i) near_value(got[i], want[i]);
                    for (unsigned i = 0; i < 128; ++i)
                        if (i < 16 || i >= 16 + sizeof(x)) CHECK(raw[i] == 0x6a);
                    tu_sram_destroy(&s);
                }
        }
    float guard = 42;
    tu_log_softmax_host(&guard, 0, 1);
    CHECK(guard == 42);
    tu_log_softmax_host(NULL, 0, 1);
}

static void matrix(void) {
    const float gaps[] = {0, 12, 16, 17, 20, 40, 80, 87, 88, 104};
    puts("mode,gap,output_max,output_tail,reference_log_max,log_relative_error,exp_sum,reads,writes,local_penalty");
    for (unsigned mode = 0; mode < 3; ++mode)
        for (unsigned gi = 0; gi < sizeof(gaps) / sizeof(*gaps); ++gi) {
            float x[] = {0, -gaps[gi]}, got[2], maximum, sum;
            tu_sram_region_t s;
            tu_sram_init(&s, 128, "log-nearzero-matrix");
            tu_sram_write_bulk(&s, 0, x, sizeof(x));
            uint64_t reads = s.banks.reads, writes = s.banks.writes;
            tu_softmax_desc_t d = {.mode = (tu_softmax_mode_t)mode, .data_sram = &s,
                .elem_count = 2, .in_place = true, .max_out = &maximum, .sum_out = &sum};
            uint64_t penalty = tu_softmax_execute(&d);
            memcpy(got, tu_sram_raw_ptr(&s), sizeof(got));
            long double ref = -log1pl(expl(-(long double)gaps[gi]));
            long double rel = mode == TU_SOFTMAX_LOG
                ? fabsl(((long double)got[0] - ref) / ref) : 0;
            printf("%s,%g,%.12g,%.12g,%.12Lg,%.12Lg,%.12g,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                   tu_softmax_mode_name((tu_softmax_mode_t)mode), gaps[gi], got[0], got[1],
                   ref, rel, sum, s.banks.reads - reads, s.banks.writes - writes, penalty);
            CHECK(penalty == 8);
            CHECK(s.banks.reads - reads == 2 && s.banks.writes - writes == 2);
            CHECK(isfinite(got[0]) && isfinite(got[1]));
            if (mode == TU_SOFTMAX_LOG && gaps[gi] <= 87) near_value(got[0], ref);
            /* Existing cutoff is intentionally NOT a full relative guarantee. */
            if (mode == TU_SOFTMAX_LOG && gaps[gi] > 87) CHECK(got[0] == 0);
            tu_sram_destroy(&s);
        }
}

int main(int argc, char **argv) {
    CHECK(LDBL_MANT_DIG > FLT_MANT_DIG);
    if (argc == 2 && strcmp(argv[1], "--matrix") == 0) matrix();
    else if (argc != 1) { fprintf(stderr, "usage: %s [--matrix]\n", argv[0]); return 2; }
    else {
        const unsigned widths[] = {1, 2, 3, 8, 65, 257};
        const float gaps[] = {0, 0.25f, 1, 12, 16, nextafterf(17, 0),
            17, nextafterf(17, INFINITY), 20, 40, 80, 86.5f, 87};
        const float scales[] = {0, 1, 0.5f, 2, -1};
        for (unsigned wi = 0; wi < sizeof(widths) / sizeof(*widths); ++wi)
            for (unsigned gi = 0; gi < sizeof(gaps) / sizeof(*gaps); ++gi)
                for (unsigned si = 0; si < sizeof(scales) / sizeof(*scales); ++si)
                    for (unsigned mask = 0; mask < 3; ++mask)
                        for (unsigned place = 0; place < 2; ++place)
                            for (unsigned tied = 0; tied < 2; ++tied)
                                descriptor_case(widths[wi], gaps[gi], scales[si], mask, place, tied);
        public_paths();
    }
    printf("log-nearzero cases=%u checks=%u failures=%u max_relative_error=%.12Lg\n",
           cases, checks, failures, max_relative);
    return failures ? 1 : 0;
}
