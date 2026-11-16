/* Descriptor-level exponential policies. Independent expl/log1pl oracle;
 * finite FP32 preprocessing and rounded FP32 exponentials are the contract.
 * TU_TEST_LEGACY_SOURCE allows a safe old-source capability comparison.
 * TU_TEST_EXP_INTERNAL additionally gates the private exponential contract
 * because tiny ONLINE denominator changes can round away at public FP32 output. */
#include "tu_cmodel/compute/softmax_engine.h"
#ifdef TU_TEST_EXP_INTERNAL
#include "tu_cmodel/compute/softmax_engine.c"
#endif
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAP 195
#define BYTES 2048
static unsigned cases, checks, failures;
static unsigned max_subnormal_ulp;
static long double normal_tolerance = 5e-7L;
static uint64_t meter_hash[3][3];
static void hash_meter(uint64_t *hash, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        *hash = (*hash ^ (value & 255)) * UINT64_C(1099511628211);
        value >>= 8;
    }
}
static const char *names[] = {"legacy", "ftz", "gradual"};
#define CHECK(c) do { ++checks; if (!(c)) { \
    if (failures < 16) fprintf(stderr,"FAIL case=%u line=%d: %s\n",cases,__LINE__,#c); \
    ++failures; } } while (0)

static void select_policy(tu_softmax_desc_t *d, unsigned p) {
#ifndef TU_TEST_LEGACY_SOURCE
    d->exp_policy = (tu_softmax_exp_policy_t)p;
#else
    (void)d; (void)p;
#endif
}

static float reference_exp(float x, unsigned p) {
    if (p == 0 && x < -87) return 0;
    float value = (float)expl((long double)x);
    if (p == 1 && value > 0 && value < FLT_MIN) return 0;
    return value;
}

static uint32_t bits(float x) {
    uint32_t b; memcpy(&b, &x, sizeof(b)); return b;
}

static void near_value(float got, float want) {
    CHECK(isfinite(got) && isfinite(want));
    if (!isfinite(got) || !isfinite(want)) return;
    if (want == 0) { CHECK(got == 0); return; }
    CHECK(got != 0);
    CHECK(signbit(got) == signbit(want));
    if (fabsf(want) < FLT_MIN) {
        uint32_t a = bits(fabsf(got)), b = bits(fabsf(want));
        unsigned ulp = a > b ? a - b : b - a;
        if (ulp > max_subnormal_ulp) max_subnormal_ulp = ulp;
        CHECK(ulp <= 2);
    } else {
        CHECK(fabsl((long double)got - want) <= normal_tolerance * fabsl(want));
    }
}

static void reference(const float *input, const float *mask, unsigned kind,
                      float scale, unsigned cols, unsigned mode, unsigned policy,
                      float *out, float *maximum, float *sum_out) {
    float x[65], e[65], m = -INFINITY;
    unsigned anchor = 0;
    for (unsigned i = 0; i < cols; ++i) {
        x[i] = input[i];
        if (kind == 1) x[i] += mask[i];
        if (kind == 2) x[i] *= mask[i];
        if (scale != 0) x[i] *= scale;
        if (x[i] > m) { m = x[i]; anchor = i; }
    }
    long double sum = 0, residual = 0;
    for (unsigned i = 0; i < cols; ++i) {
        float shifted = x[i] - m;
        e[i] = reference_exp(shifted, policy);
        sum += e[i];
        if (i != anchor) residual += e[i];
    }
    for (unsigned i = 0; i < cols; ++i) {
        if (mode == TU_SOFTMAX_LOG)
            out[i] = (x[i] - m) - (float)log1pl(residual);
        else out[i] = (float)((long double)e[i] / sum);
    }
    *maximum = m; *sum_out = (float)sum;
}

static void run_case(unsigned cols, float gap, float scale, unsigned kind,
                     bool in_place, bool tied, unsigned mode, unsigned policy) {
    const unsigned rows = 3, n = rows * cols, in = 64;
    const unsigned out = in_place ? in : 1024;
    float x[CAP], mask[CAP], want[CAP], got[CAP], maxima[3], sums[3], em[3], es[3];
    unsigned char before[BYTES], after[BYTES];
    tu_sram_region_t s;
    ++cases;
    /* ONLINE sums in FP32: two rounded operations per input plus normalization.
     * This length-aware normal-value bound is separate from the <=2 ULP
     * subnormal contract and exact zero/classification checks. */
    normal_tolerance = mode == TU_SOFTMAX_ONLINE
        ? (2 * cols + 4) * (long double)FLT_EPSILON : 5e-7L;
    for (unsigned r = 0; r < rows; ++r) {
        unsigned anchor = r == 0 ? 0 : r == 1 ? cols / 2 : cols - 1;
        for (unsigned i = 0; i < cols; ++i) {
            unsigned k = r * cols + i;
            float target = i == anchor || (tied && i == (anchor + 1) % cols) ? 0 : -gap;
            mask[k] = kind == 1 ? (i % 2 ? -0.5f : 0.5f) : 0.5f;
            float a = scale == 0 ? target : target / scale;
            x[k] = kind == 1 ? a - mask[k] : kind == 2 ? a / mask[k] : a;
        }
        reference(x + r * cols, mask + r * cols, kind, scale, cols, mode,
                  policy, want + r * cols, em + r, es + r);
    }
    tu_sram_init(&s, BYTES, "exp-policy");
    unsigned char *raw = tu_sram_raw_ptr(&s);
    memset(raw, 0x5a, BYTES);
    tu_sram_write_bulk(&s, in, x, n * sizeof(float));
    memcpy(before, raw, BYTES);
    uint64_t reads = s.banks.reads, writes = s.banks.writes, stalls = s.banks.stall_cycles;
    tu_softmax_desc_t d = {.mode = (tu_softmax_mode_t)mode, .data_sram = &s,
        .data_offset = in, .elem_count = n, .axis_dim = cols,
        .mask = kind ? mask : NULL, .mask_is_additive = kind == 1,
        .scale = scale, .in_place = in_place, .out_offset = out,
        .max_out = maxima, .sum_out = sums};
    select_policy(&d, policy);
    CHECK(tu_softmax_validate_desc(&d));
    uint64_t result = tu_softmax_execute(&d);
    CHECK(result != UINT64_MAX);
    CHECK(s.banks.reads - reads == n && s.banks.writes - writes == n);
    CHECK(result == s.banks.stall_cycles - stalls);
    uint64_t *hash = &meter_hash[policy][mode];
    hash_meter(hash, result);
    hash_meter(hash, s.banks.reads);
    hash_meter(hash, s.banks.writes);
    for (unsigned b = 0; b < s.banks.bank_count; ++b) {
        const tu_sram_bw_bank_t *bank = &s.banks.bw_banks[b];
        hash_meter(hash, (uint64_t)bank->words_available);
        hash_meter(hash, bank->last_refill_cycle);
        hash_meter(hash, bank->reads_served);
        hash_meter(hash, bank->writes_served);
        hash_meter(hash, bank->read_stalls);
        hash_meter(hash, bank->write_stalls);
    }
    memcpy(after, raw, BYTES);
    CHECK(memcmp(before, after, out) == 0);
    CHECK(memcmp(before + out + n * 4, after + out + n * 4, BYTES - out - n * 4) == 0);
    memcpy(got, raw + out, n * 4);
    for (unsigned r = 0; r < rows; ++r) {
        CHECK(maxima[r] == em[r]);
        near_value(sums[r], es[r]);
        for (unsigned i = 0; i < cols; ++i) near_value(got[r * cols + i], want[r * cols + i]);
    }
    /* No hidden arithmetic/global policy state; optional stats and disabled
     * meter must preserve exactly the same output bits. */
    memcpy(raw, before, BYTES);
    d.max_out = NULL; d.sum_out = NULL;
    tu_sram_set_bw_modeling(&s, false);
    CHECK(tu_softmax_execute(&d) == 0);
    CHECK(memcmp(raw, after, BYTES) == 0);
    tu_sram_destroy(&s);
}

static void invalid(void) {
#ifndef TU_TEST_LEGACY_SOURCE
    const int invalids[] = {-1, 3, 255, 0x7fffffff};
    for (unsigned mode = 0; mode < 3; ++mode)
        for (unsigned place = 0; place < 2; ++place)
            for (unsigned i = 0; i < sizeof(invalids)/sizeof(*invalids); ++i) {
                ++cases;
                tu_sram_region_t s;
                tu_sram_init(&s, 128, "invalid-policy");
                unsigned char before[128];
                memset(tu_sram_raw_ptr(&s), 0x5a, 128);
                memcpy(before, tu_sram_raw_ptr(&s), 128);
                tu_sram_region_t state = s;
                tu_sram_bw_bank_t banks[TU_SRAM_BANKS];
                memcpy(banks, s.banks.bw_banks, sizeof(banks));
                float m = 23, sum = 45;
                tu_softmax_desc_t d = {.mode = (tu_softmax_mode_t)mode,
                    .data_sram = &s, .elem_count = 2, .in_place = place,
                    .out_offset = 64, .max_out = &m, .sum_out = &sum,
                    .exp_policy = (tu_softmax_exp_policy_t)invalids[i]};
                tu_softmax_desc_t saved = d;
                CHECK(!tu_softmax_validate_desc(&d));
                CHECK(tu_softmax_execute(&d) == UINT64_MAX);
                CHECK(memcmp(&saved, &d, sizeof(d)) == 0);
                CHECK(memcmp(&state, &s, sizeof(s)) == 0);
                CHECK(memcmp(banks, s.banks.bw_banks, sizeof(banks)) == 0);
                CHECK(memcmp(before, tu_sram_raw_ptr(&s), 128) == 0);
                CHECK(m == 23 && sum == 45);
                tu_sram_destroy(&s);
            }
#endif
}

static void wrappers(void) {
    for (unsigned which = 0; which < 7; ++which) {
        ++cases;
        float x[] = {0, -88, 0, -88};
        tu_sram_region_t s;
        tu_sram_init(&s, 128, "legacy-wrappers");
        tu_sram_write_bulk(&s, 0, x, sizeof(x));
        uint64_t status = 0;
        switch (which) {
        case 0: tu_softmax_host(x, 2, 1); break;
        case 1: tu_log_softmax_host(x, 2, 1); break;
        case 2: status = tu_softmax(&s, 0, 2, 1, true); break;
        case 3: status = tu_softmax_2d(&s, 0, 2, 2, 1, true); break;
        case 4: status = tu_softmax_masked(&s, 0, 2, 2, NULL, 0, 1); break;
        case 5: status = tu_log_softmax(&s, 0, 2, 1, true); break;
        case 6: status = tu_log_softmax_2d(&s, 0, 2, 2, 1, true); break;
        }
        if (which >= 2) memcpy(x, tu_sram_raw_ptr(&s), sizeof(x));
        CHECK(status != UINT64_MAX);
        if (which == 1 || which >= 5) CHECK(x[0] == 0 && x[1] == -88);
        else CHECK(x[0] == 1 && x[1] == 0);
        tu_sram_destroy(&s);
    }
}

static void matrix(void) {
    normal_tolerance = 5e-7L;
    const float gaps[] = {0, 20, 87, 87.25f, 87.5f, 88, 100, 103.25f, 104};
    puts("policy,mode,gap,output_max,output_tail,mathematical_log_max,exp_sum,reads,writes,local_penalty");
    for (unsigned p = 0; p < 3; ++p)
        for (unsigned m = 0; m < 3; ++m)
            for (unsigned gi = 0; gi < sizeof(gaps)/sizeof(*gaps); ++gi) {
                ++cases;
                float x[] = {0, -gaps[gi]}, got[2], maximum, sum;
                tu_sram_region_t s;
                tu_sram_init(&s, 128, "exp-matrix");
                tu_sram_write_bulk(&s, 0, x, sizeof(x));
                uint64_t reads = s.banks.reads, writes = s.banks.writes;
                tu_softmax_desc_t d = {.mode = (tu_softmax_mode_t)m, .data_sram = &s,
                    .elem_count = 2, .in_place = true, .max_out = &maximum, .sum_out = &sum};
                select_policy(&d, p);
                uint64_t penalty = tu_softmax_execute(&d);
                memcpy(got, tu_sram_raw_ptr(&s), sizeof(got));
                float want[2], em, es;
                reference(x, NULL, 0, 0, 2, m, p, want, &em, &es);
                near_value(got[0], want[0]); near_value(got[1], want[1]);
                printf("%s,%s,%g,%.12g,%.12g,%.12Lg,%.12g,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                    names[p], tu_softmax_mode_name((tu_softmax_mode_t)m), gaps[gi], got[0], got[1],
                    -log1pl(expl(-(long double)gaps[gi])), sum,
                    s.banks.reads-reads, s.banks.writes-writes, penalty);
                CHECK(penalty == 8 && s.banks.reads-reads == 2 && s.banks.writes-writes == 2);
                tu_sram_destroy(&s);
            }
}

int main(int argc, char **argv) {
    CHECK(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && sizeof(float) == 4);
    CHECK(LDBL_MANT_DIG > FLT_MANT_DIG);
    for (unsigned p = 0; p < 3; ++p)
        for (unsigned m = 0; m < 3; ++m)
            meter_hash[p][m] = UINT64_C(14695981039346656037);
    volatile float tiny = FLT_TRUE_MIN;
    CHECK(tiny > 0 && tiny * 2 == (float)ldexpl(1.L, -148));
    float normal = (float)-logl((long double)FLT_MIN);
    float minimum = (float)-logl((long double)FLT_TRUE_MIN);
    float half = (float)-logl((long double)FLT_TRUE_MIN / 2);
    const float gaps[] = {0, 0.25f, 1, 20, nextafterf(87, 0), 87, nextafterf(87, INFINITY),
        87.25f, nextafterf(normal, 0), normal, nextafterf(normal, INFINITY),
        88, 100, nextafterf(minimum, 0), minimum, nextafterf(minimum, INFINITY),
        nextafterf(half, 0), half, nextafterf(half, INFINITY), 104};
    if (argc == 2 && strcmp(argv[1], "--matrix") == 0) matrix();
    else if (argc != 1) { fprintf(stderr, "usage: %s [--matrix]\n", argv[0]); return 2; }
    else {
        const unsigned widths[] = {1, 2, 3, 8, 65};
        const float scales[] = {0, 0.5f, 2, -1};
        for (unsigned p = 0; p < 3; ++p)
            for (unsigned m = 0; m < 3; ++m)
                for (unsigned w = 0; w < sizeof(widths)/sizeof(*widths); ++w)
                    for (unsigned g = 0; g < sizeof(gaps)/sizeof(*gaps); ++g)
                        for (unsigned sc = 0; sc < sizeof(scales)/sizeof(*scales); ++sc)
                            for (unsigned mask = 0; mask < 3; ++mask)
                                for (unsigned place = 0; place < 2; ++place)
                                    for (unsigned tied = 0; tied < 2; ++tied)
                                        run_case(widths[w], gaps[g], scales[sc], mask, place, tied, m, p);
        for (unsigned p = 0; p < 3; ++p)
            for (unsigned m = 0; m < 3; ++m) {
                CHECK(meter_hash[p][m] == meter_hash[0][0]);
                printf("policy=%s mode=%s accounting_fnv64=%016" PRIx64 "\n",
                    names[p], tu_softmax_mode_name((tu_softmax_mode_t)m), meter_hash[p][m]);
            }
        invalid(); wrappers();
#ifdef TU_TEST_EXP_INTERNAL
        /* Public FP32 sum >=1 cannot reveal every tiny rescale term. Gate the
         * actual private helper in both call contexts, not a copied helper. */
        normal_tolerance = 5e-7L;
        for (unsigned p = 0; p < 3; ++p)
            for (unsigned online = 0; online < 2; ++online)
                for (unsigned g = 0; g < sizeof(gaps)/sizeof(*gaps); ++g) {
                    float want = reference_exp(-gaps[g], p == 0 && online ? 2 : p);
                    near_value(softmax_exp(-gaps[g], (tu_softmax_exp_policy_t)p, online), want);
                }
#endif
    }
    printf("exp-policy cases=%u checks=%u failures=%u max_subnormal_ulp=%u\n",
           cases, checks, failures, max_subnormal_ulp);
    return failures ? 1 : 0;
}
