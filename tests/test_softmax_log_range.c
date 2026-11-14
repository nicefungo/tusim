/* Finite dynamic-range gate. Independent double oracle, not the shared
 * production host helper. Not an all-domain IEEE or relative-near-zero gate. */
#include "tu_cmodel/compute/softmax_engine.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAP 192
#define SRAM_BYTES 8192
static unsigned checks, failures, cases;
static double max_abs[3];
static uint64_t fingerprint[3] = {14695981039346656037ULL,
    14695981039346656037ULL, 14695981039346656037ULL};
static uint64_t accounting[3] = {14695981039346656037ULL,
    14695981039346656037ULL, 14695981039346656037ULL};
#define CHECK(c) do { ++checks; if (!(c)) { \
    if (failures < 12) fprintf(stderr, "FAIL case=%u line=%d: %s\n", cases, __LINE__, #c); \
    ++failures; } } while (0)

static void mix(uint64_t *hash, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; ++i) {
        *hash ^= b[i];
        *hash *= 1099511628211ULL;
    }
}

static void near_value(float got, double want, unsigned mode) {
    double error = fabs((double)got - want);
    if (isfinite(error) && error > max_abs[mode]) max_abs[mode] = error;
    /* Allow FP32 shift/output rounding at large magnitudes and accumulated
     * online FP32 normalizer error. Unexpected NaN/Inf always fail. */
    CHECK(isfinite(got) && isfinite(want) &&
          error <= 8e-6 + 3e-7 * fabs(want));
}

/* Input masking and scale follow the existing FP32 preprocessing contract;
 * everything after preprocessing is independently evaluated in double. */
static void oracle(const float *x, const float *mask, unsigned mask_kind,
                   float scale, unsigned rows, unsigned cols, unsigned mode,
                   double *out, double *maxima, double *sums) {
    for (unsigned r = 0; r < rows; ++r) {
        double v[CAP], m = -INFINITY, sum = 0;
        for (unsigned i = 0; i < cols; ++i) {
            unsigned k = r * cols + i;
            float a = x[k];
            if (mask_kind == 1) a += mask[k];
            if (mask_kind == 2) a *= mask[k];
            if (scale != 0) a *= scale;
            v[i] = a;
            if (v[i] > m) m = v[i];
        }
        for (unsigned i = 0; i < cols; ++i) sum += exp(v[i] - m);
        for (unsigned i = 0; i < cols; ++i)
            out[r * cols + i] = mode == TU_SOFTMAX_LOG
                ? (v[i] - m) - log(sum) : exp(v[i] - m) / sum;
        maxima[r] = m;
        sums[r] = sum;
    }
}

static float pattern(unsigned p, unsigned i, unsigned row, unsigned cols) {
    unsigned j = row == 1 ? cols - 1 - i : (i + row) % cols;
    switch (p) {
    case 0: return 5.0f;
    case 1: return -(float)j;
    case 2: {
        const float boundary[] = {0, -86, nextafterf(-87.0f, INFINITY),
            -87, nextafterf(-87.0f, -INFINITY), -88, -104, -200};
        return boundary[j % 8];
    }
    case 3: return -1000.0f * j;
    case 4: return -10000.0f - 100.0f * j;
    case 5: return 10000.0f + 0.125f * j;
    case 6: return j < 2 ? 0.0f : -200.0f;
    default: return (float)((int)((j * 37 + row * 19) % 101) - 50) * 0.25f;
    }
}

static void descriptor_case(unsigned p, unsigned cols, float scale,
                            unsigned mask_kind, bool in_place, unsigned mode) {
    const unsigned rows = 3, n = rows * cols, in = 128;
    unsigned out = in_place ? in : 4096;
    float x[CAP], mask[CAP], maxima[3], sums[3];
    double expected[CAP], em[3], es[3];
    unsigned char before[SRAM_BYTES], after[SRAM_BYTES];
    tu_sram_region_t s;
    ++cases;
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned i = 0; i < cols; ++i) {
            unsigned k = r * cols + i;
            x[k] = pattern(p, i, r, cols);
            mask[k] = mask_kind == 1 ? (i % 3 == 0 ? -250.0f : 0.25f)
                                      : (i % 3 == 0 ? 0.0f : 0.5f);
        }
    oracle(x, mask, mask_kind, scale, rows, cols, mode, expected, em, es);
    tu_sram_init(&s, SRAM_BYTES, "log-range");
    unsigned char *raw = tu_sram_raw_ptr(&s);
    memset(raw, 0x5a, SRAM_BYTES);
    tu_sram_write_bulk(&s, in, x, n * sizeof(float));
    memcpy(before, raw, SRAM_BYTES);
    uint64_t reads = s.banks.reads, writes = s.banks.writes;
    uint64_t stalls = s.banks.stall_cycles;
    tu_softmax_desc_t d = {.mode = (tu_softmax_mode_t)mode, .data_sram = &s,
        .data_offset = in, .elem_count = n, .axis_dim = cols,
        .mask = mask_kind ? mask : NULL, .mask_is_additive = mask_kind == 1,
        .scale = scale, .in_place = in_place, .out_offset = out,
        .max_out = maxima, .sum_out = sums};
    uint64_t returned = tu_softmax_execute(&d);
    CHECK(returned != UINT64_MAX);
    CHECK(s.banks.reads - reads == n);
    CHECK(s.banks.writes - writes == n);
    CHECK(returned == s.banks.stall_cycles - stalls);
    memcpy(after, raw, SRAM_BYTES);
    CHECK(memcmp(before, after, out) == 0);
    CHECK(memcmp(before + out + n * 4, after + out + n * 4,
                 SRAM_BYTES - out - n * 4) == 0);
    mix(&fingerprint[mode], after + out, n * 4);
    mix(&fingerprint[mode], maxima, sizeof(maxima));
    mix(&fingerprint[mode], sums, sizeof(sums));
    mix(&fingerprint[mode], &returned, sizeof(returned));
    mix(&accounting[mode], maxima, sizeof(maxima));
    mix(&accounting[mode], sums, sizeof(sums));
    mix(&accounting[mode], &returned, sizeof(returned));
    mix(&accounting[mode], &s.banks.reads, sizeof(s.banks.reads));
    mix(&accounting[mode], &s.banks.writes, sizeof(s.banks.writes));
    for (unsigned r = 0; r < rows; ++r) {
        CHECK(isfinite(maxima[r]) && (double)maxima[r] == em[r]);
        CHECK(isfinite(sums[r]) && fabs((double)sums[r] - es[r]) <= 1e-5 * es[r]);
        double total = 0;
        for (unsigned i = 0; i < cols; ++i) {
            unsigned k = r * cols + i;
            float got;
            memcpy(&got, after + out + k * 4, 4);
            near_value(got, expected[k], mode);
            total += mode == TU_SOFTMAX_LOG ? exp((double)got) : (double)got;
        }
        CHECK(isfinite(total) && fabs(total - 1.0) < 2e-5);
    }
    /* NULL optional statistics and disabled meter: same outputs, zero stalls. */
    if (p == 3 && mask_kind == 0) {
        memcpy(raw, before, SRAM_BYTES);
        tu_sram_set_bw_modeling(&s, false);
        d.max_out = NULL; d.sum_out = NULL;
        CHECK(tu_softmax_execute(&d) == 0);
        CHECK(memcmp(raw + out, after + out, n * 4) == 0);
    }
    tu_sram_destroy(&s);
}

static void host_and_wrappers(void) {
    const float scales[] = {0, 1, 0.5f, -1, 2};
    for (unsigned p = 0; p < 8; ++p)
        for (unsigned si = 0; si < 5; ++si) {
            float x[24], got[24];
            double want[24], m[3], sum[3];
            for (unsigned i = 0; i < 24; ++i) x[i] = pattern(p, i % 8, i / 8, 8);
            oracle(x, NULL, 0, scales[si], 3, 8, TU_SOFTMAX_LOG, want, m, sum);
            memcpy(got, x, sizeof(x));
            ++cases;
            for (unsigned r = 0; r < 3; ++r) tu_log_softmax_host(got + r * 8, 8, scales[si]);
            for (unsigned i = 0; i < 24; ++i) near_value(got[i], want[i], TU_SOFTMAX_LOG);
            for (unsigned batched = 0; batched < 2; ++batched)
                for (unsigned placement = 0; placement < 2; ++placement) {
                    tu_sram_region_t s;
                    ++cases;
                    tu_sram_init(&s, 256, "wrapper-range");
                    tu_sram_write_bulk(&s, 16, x, sizeof(x));
                    uint64_t stalls = 0;
                    if (batched) stalls = tu_log_softmax_2d(&s, 16, 3, 8, scales[si], placement);
                    else for (unsigned r = 0; r < 3; ++r) {
                        uint64_t v = tu_log_softmax(&s, 16 + r * 32, 8, scales[si], placement);
                        CHECK(v != UINT64_MAX); stalls += v;
                    }
                    CHECK(stalls != UINT64_MAX);
                    /* Existing wrappers use the same offset even when false. */
                    memcpy(got, (unsigned char *)tu_sram_raw_ptr(&s) + 16, sizeof(got));
                    for (unsigned i = 0; i < 24; ++i) near_value(got[i], want[i], TU_SOFTMAX_LOG);
                    tu_sram_destroy(&s);
                }
        }
    float untouched = 123;
    tu_log_softmax_host(&untouched, 0, 1);
    CHECK(untouched == 123);
}

/* Full compact mode/gap matrix; outputs and local SRAM penalties are live. */
static void print_matrix(void) {
    const float gaps[] = {1, 80, 87, 88, 100, 200, 1000};
    puts("mode,gap,tail,oracle_tail,abs_error,local_stall_cycles,reads,writes");
    for (unsigned mode = 0; mode < 3; ++mode)
        for (unsigned j = 0; j < sizeof(gaps)/sizeof(gaps[0]); ++j) {
            tu_sram_region_t s;
            float x[] = {0, -gaps[j]}, y[2];
            double expected[2], m[1], sum[1];
            ++cases;
            tu_sram_init(&s, 128, "range-matrix");
            tu_sram_write_bulk(&s, 0, x, sizeof(x));
            uint64_t reads = s.banks.reads, writes = s.banks.writes;
            tu_softmax_desc_t d = {.mode = (tu_softmax_mode_t)mode,
                .data_sram = &s, .elem_count = 2, .in_place = true};
            uint64_t stalls = tu_softmax_execute(&d);
            memcpy(y, tu_sram_raw_ptr(&s), sizeof(y));
            oracle(x, NULL, 0, 0, 1, 2, mode, expected, m, sum);
            CHECK(stalls != UINT64_MAX);
            for (unsigned i = 0; i < 2; ++i) near_value(y[i], expected[i], mode);
            printf("%s,%.0f,%.12g,%.12g,%.12g,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                tu_softmax_mode_name((tu_softmax_mode_t)mode), (double)gaps[j],
                (double)y[1], expected[1], fabs((double)y[1] - expected[1]),
                stalls, s.banks.reads - reads, s.banks.writes - writes);
            tu_sram_destroy(&s);
        }
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--matrix") == 0) {
        print_matrix();
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--matrix]\n", argv[0]);
        return 2;
    } else {
        const unsigned widths[] = {1, 2, 3, 8, 31, 64};
        const float scales[] = {0, 1, 0.5f, -1, 2};
        for (unsigned p = 0; p < 8; ++p)
            for (unsigned w = 0; w < 6; ++w)
                for (unsigned sc = 0; sc < 5; ++sc)
                    for (unsigned mask = 0; mask < 3; ++mask)
                        for (unsigned place = 0; place < 2; ++place)
                            for (unsigned mode = 0; mode < 3; ++mode)
                                descriptor_case(p, widths[w], scales[sc], mask, place, mode);
        host_and_wrappers();
        for (unsigned mode = 0; mode < 3; ++mode)
            printf("mode=%s output_stats_stalls_fnv64=%016" PRIx64
                " accounting_fnv64=%016" PRIx64 " max_abs_error=%.12g\n",
                tu_softmax_mode_name((tu_softmax_mode_t)mode), fingerprint[mode],
                accounting[mode], max_abs[mode]);
    }
    printf("log-range cases=%u checks=%u failures=%u\n", cases, checks, failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
