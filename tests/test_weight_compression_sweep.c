/* Weight-compression exploration: RLE sensitivity to zero placement. */
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/memory/weight_compress.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4096u
#define BUS_BYTES 32u

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }

static void measure(const char *name, fp16_t *w)
{
    uint32_t cap = tu_compress_max_size(N), bytes = 0, runs = 0;
    uint8_t *encoded = malloc(cap);
    if (!encoded || tu_compress_rle(w, N, 0.0f, encoded, cap, &bytes) != 0) exit(2);
    if (bytes >= 8) runs = ((uint32_t *)encoded)[1];
    const uint32_t raw = N * (uint32_t)sizeof(fp16_t);
    printf("%-22s %6u %6u %6u %8.3f %8u %8u\n", name, runs, raw, bytes,
           (double)raw / bytes, (raw + BUS_BYTES - 1) / BUS_BYTES,
           (bytes + BUS_BYTES - 1) / BUS_BYTES);
    free(encoded);
}

static void fill_random_sparse(fp16_t *w, unsigned zero_pct)
{
    uint32_t seed = 7;
    for (uint32_t i = 0; i < N; i++)
        w[i] = (lcg(&seed) % 100 < zero_pct) ? 0 :
               tu_fp32_to_fp16((float)((lcg(&seed) % 31) + 1));
}

static void fill_clustered_sparse(fp16_t *w, unsigned zero_pct)
{
    uint32_t zeros = N * zero_pct / 100;
    for (uint32_t i = 0; i < N; i++)
        w[i] = i < zeros ? 0 : tu_fp32_to_fp16((float)(1 + ((i - zeros) / 16) % 16));
}

int main(void)
{
    fp16_t *w = malloc(N * sizeof(*w));
    if (!w) return 2;
    printf("workload                 runs    raw    rle    ratio raw_cyc rle_cyc\n");
    printf("---------------------- ------ ------ ------ -------- -------- --------\n");

    for (uint32_t i = 0; i < N; i++) w[i] = (fp16_t)(i & 1u);
    measure("alternating", w);
    for (unsigned p = 50; p <= 90; p += 20) {
        char name[32];
        fill_random_sparse(w, p); snprintf(name, sizeof(name), "random-zero-%u%%", p); measure(name, w);
        fill_clustered_sparse(w, p); snprintf(name, sizeof(name), "cluster-zero-%u%%", p); measure(name, w);
    }
    for (uint32_t i = 0; i < N; i++) w[i] = 0;
    measure("all-zero", w);
    free(w);
    return 0;
}
