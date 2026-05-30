/*
 * TinyTU Random Tensor Generator — Reusable testing utilities
 * =============================================================
 * Gap V6: Random/differential testing infrastructure.
 *
 * Provides deterministic PRNG, random tensor generation with
 * configurable distributions, edge-case tensors, and golden
 * reference computation helpers.
 *
 * Thread-safe: global state per file; use per-test seeding.
 * Reproducible: same seed → same tensors across platforms.
 */

#ifndef TINYTU_RANDOM_TENSOR_H
#define TINYTU_RANDOM_TENSOR_H

#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/tu_config.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── PRNG: xorshift128+ (deterministic, fast, good distribution) ── */

typedef struct {
    uint64_t s[2];
} tu_random_state_t;

/* Initialize PRNG with a seed */
static inline void tu_random_seed(tu_random_state_t *rng, uint64_t seed) {
    /* Splitmix64 to initialize state from single seed */
    uint64_t z = (seed + 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    rng->s[0] = z;

    z = (z + 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    rng->s[1] = z;
}

/* Generate next random uint64_t */
static inline uint64_t tu_random_u64(tu_random_state_t *rng) {
    uint64_t s1 = rng->s[0];
    const uint64_t s0 = rng->s[1];
    rng->s[0] = s0;
    s1 ^= s1 << 23;
    rng->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return rng->s[1] + s0;
}

/* Random float in [0, 1) using 53-bit mantissa precision */
static inline float tu_random_f32(tu_random_state_t *rng) {
    uint64_t r = tu_random_u64(rng);
    /* Use top 24 bits for float precision */
    return (float)(r >> 40) / (float)(1ULL << 24);
}

/* Random float in [-range, range] */
static inline float tu_random_f32_range(tu_random_state_t *rng, float range) {
    return (tu_random_f32(rng) * 2.0f - 1.0f) * range;
}

/* Random int in [min, max] */
static inline int32_t tu_random_i32_range(tu_random_state_t *rng,
                                           int32_t min, int32_t max) {
    uint64_t r = tu_random_u64(rng);
    uint32_t span = (uint32_t)(max - min + 1);
    return min + (int32_t)(r % span);
}

/* ── Tensor Generation ────────────────────────────────────────── */

typedef enum {
    TU_DIST_UNIFORM,       /* Uniform [-range, range] */
    TU_DIST_NORMAL,        /* Approx normal (Box-Muller) */
    TU_DIST_LOG_UNIFORM,   /* Log-uniform [10^-range, 10^range] */
    TU_DIST_ZERO,          /* All zeros */
    TU_DIST_ONES,          /* All ones */
    TU_DIST_EXTREME,       /* Mix of 0, ±1, ±max */
    TU_DIST_SPARSE,        /* Mostly zeros, occasional spikes */
} tu_distribution_t;

/* Fill a FP32 tensor with given distribution */
static inline void tu_tensor_fill_fp32(
    fp32_t *data, uint32_t count,
    tu_distribution_t dist, float param,
    tu_random_state_t *rng)
{
    switch (dist) {
    case TU_DIST_UNIFORM:
        for (uint32_t i = 0; i < count; i++)
            data[i] = tu_random_f32_range(rng, param);
        break;

    case TU_DIST_NORMAL:
        /* Box-Muller transform */
        for (uint32_t i = 0; i < count; i += 2) {
            float u1 = tu_random_f32(rng);
            float u2 = tu_random_f32(rng);
            if (u1 < 1e-10f) u1 = 1e-10f;
            float r = sqrtf(-2.0f * logf(u1));
            float theta = 2.0f * 3.141592653589793f * u2;
            data[i] = r * cosf(theta) * param;
            if (i + 1 < count)
                data[i + 1] = r * sinf(theta) * param;
        }
        break;

    case TU_DIST_LOG_UNIFORM:
        for (uint32_t i = 0; i < count; i++) {
            float sign = (tu_random_u64(rng) & 1) ? 1.0f : -1.0f;
            float mag = powf(10.0f, tu_random_f32_range(rng, param));
            data[i] = sign * mag;
        }
        break;

    case TU_DIST_ZERO:
        memset(data, 0, count * sizeof(fp32_t));
        break;

    case TU_DIST_ONES:
        for (uint32_t i = 0; i < count; i++) data[i] = 1.0f;
        break;

    case TU_DIST_EXTREME:
        for (uint32_t i = 0; i < count; i++) {
            uint64_t r = tu_random_u64(rng);
            switch (r & 7) {
            case 0: data[i] = 0.0f; break;
            case 1: data[i] = 1.0f; break;
            case 2: data[i] = -1.0f; break;
            case 3: data[i] = 65504.0f; break;      /* max FP16 */
            case 4: data[i] = -65504.0f; break;
            case 5: data[i] = 1e-8f; break;          /* subnormal */
            case 6: data[i] = -1e-8f; break;
            case 7: data[i] = tu_random_f32_range(rng, param); break;
            }
        }
        break;

    case TU_DIST_SPARSE:
        memset(data, 0, count * sizeof(fp32_t));
        for (uint32_t i = 0; i < count; i++) {
            /* param controls sparsity: ~param% non-zero */
            if (tu_random_f32(rng) < param) {
                data[i] = tu_random_f32_range(rng, 10.0f);
            }
        }
        break;
    }
}

/* Fill a FP16 tensor (from FP32 distribution, then quantize) */
static inline void tu_tensor_fill_fp16(
    fp16_t *data, uint32_t count,
    tu_distribution_t dist, float param,
    tu_random_state_t *rng)
{
    fp32_t *tmp = malloc(count * sizeof(fp32_t));
    if (!tmp) return;

    tu_tensor_fill_fp32(tmp, count, dist, param, rng);
    tu_fp32_to_fp16_buffer(tmp, data, count);
    free(tmp);
}

/* Fill with known values (identity matrix, sequential, etc.) */
typedef enum {
    TU_PATTERN_IDENTITY,     /* Identity matrix: diag=1, rest=0 */
    TU_PATTERN_SEQUENTIAL,   /* 0, 1, 2, 3, ... */
    TU_PATTERN_CHECKERBOARD, /* Alternating ±1 */
    TU_PATTERN_RAMP,         /* Linear ramp from -scale to +scale */
} tu_pattern_t;

static inline void tu_tensor_fill_pattern(
    fp32_t *data, uint32_t rows, uint32_t cols,
    tu_pattern_t pattern, float scale)
{
    uint32_t count = rows * cols;

    switch (pattern) {
    case TU_PATTERN_IDENTITY:
        memset(data, 0, count * sizeof(fp32_t));
        for (uint32_t i = 0; i < rows && i < cols; i++)
            data[i * cols + i] = scale;
        break;

    case TU_PATTERN_SEQUENTIAL:
        for (uint32_t i = 0; i < count; i++)
            data[i] = (float)i * scale;
        break;

    case TU_PATTERN_CHECKERBOARD:
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t c = 0; c < cols; c++)
                data[r * cols + c] = ((r + c) & 1) ? scale : -scale;
        break;

    case TU_PATTERN_RAMP:
        if (count <= 1) {
            data[0] = 0.0f;
        } else {
            for (uint32_t i = 0; i < count; i++)
                data[i] = -scale + (2.0f * scale * (float)i / (float)(count - 1));
        }
        break;
    }
}

/* ── Golden Reference: FP32 GEMM ─────────────────────────────── */

/*
 * Compute O_ref = W @ A in pure FP32.
 * W: [M, K], A: [K, N] — both FP32.
 * O: [M, N] — output, FP32.
 * row-major layout: index(i,j) = i * cols + j
 */
static inline void tu_golden_gemm_fp32(
    const fp32_t *W, const fp32_t *A,
    fp32_t *O,
    uint32_t M, uint32_t N, uint32_t K,
    bool transpose_w, bool transpose_a)
{
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                float w = transpose_w ? W[k * M + m] : W[m * K + k];
                float a = transpose_a ? A[n * K + k] : A[k * N + n];
                sum += w * a;
            }
            O[m * N + n] = sum;
        }
    }
}

/* Golden elementwise: ReLU */
static inline void tu_golden_relu(const fp32_t *input, fp32_t *output,
                                   uint32_t count) {
    for (uint32_t i = 0; i < count; i++)
        output[i] = fmaxf(0.0f, input[i]);
}

/* Golden elementwise: GELU (approximation) */
static inline void tu_golden_gelu(const fp32_t *input, fp32_t *output,
                                   uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        float x = input[i];
        /* tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))) */
        float c = 0.7978845608f;  /* sqrt(2/pi) */
        float inner = c * (x + 0.044715f * x * x * x);
        float tanh_val = tanhf(inner);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

/* Golden softmax (online, numerically stable) */
static inline void tu_golden_softmax(const fp32_t *input, fp32_t *output,
                                      uint32_t count) {
    /* Find max for numerical stability */
    float max_val = input[0];
    for (uint32_t i = 1; i < count; i++)
        if (input[i] > max_val) max_val = input[i];

    /* Exp and sum */
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    /* Normalize */
    if (sum > 0.0f) {
        for (uint32_t i = 0; i < count; i++)
            output[i] /= sum;
    }
}

/* ── Progress Reporting ──────────────────────────────────────── */

typedef void (*tu_progress_cb_t)(int done, int total, void *userdata);

static inline void tu_progress_default(int done, int total, void *userdata) {
    (void)userdata;
    if (done % 500 == 0 || done == total) {
        printf("    Progress: %d/%d tests\n", done, total);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* TINYTU_RANDOM_TENSOR_H */
