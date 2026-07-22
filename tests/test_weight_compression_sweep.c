/* Weight-compression exploration: payload plus configurable decoder throughput. */
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/memory/weight_compress.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4096u
#define BUS_BITS 256u
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static const char *codec_name(tu_weight_payload_codec_t c)
{
    return c == TU_WEIGHT_PAYLOAD_RLE ? "RLE" :
           c == TU_WEIGHT_PAYLOAD_BITMAP ? "BMP" : "RAW";
}

typedef struct {
    const char *name;
    uint32_t output_width;
    uint32_t rle_width;
    uint32_t bitmap_width;
} decoder_profile_t;

static const decoder_profile_t profiles[] = {
    {"serial",   1,  1,  1},
    {"balanced", 8,  4,  8},
    {"wide",    16,  8, 16},
    {"xwide",   32, 16, 32},
};

static uint64_t estimate(const uint8_t *stream, uint32_t bytes,
                         tu_compress_type_t type, const decoder_profile_t *p,
                         tu_compress_cycle_stats_t *stats)
{
    tu_compress_config_t cfg = tu_compress_config_default;
    cfg.enabled = type != TU_COMPRESS_NONE;
    cfg.type = type;
    cfg.decoder_enabled = true;
    cfg.decoder_overlap_dma = true;
    cfg.decoder_elements_per_cycle = p->output_width;
    cfg.rle_runs_per_cycle = p->rle_width;
    cfg.bitmap_elements_per_cycle = p->bitmap_width;
    if (tu_compress_estimate_cycles(stream, bytes, &cfg, BUS_BITS, stats) != 0) exit(2);
    return stats->total_cycles;
}

static void measure(const char *name, fp16_t *w)
{
    uint32_t rle_cap = tu_compress_max_size(N), rle_bytes = 0;
    uint32_t bmp_cap = tu_compress_bitmap_max_size(N), bmp_bytes = 0;
    uint32_t ad_cap = tu_compress_adaptive_max_size(N), ad_bytes = 0, runs = 0, nnz = 0;
    uint8_t *rle = malloc(rle_cap), *bitmap = malloc(bmp_cap), *adaptive = malloc(ad_cap);
    tu_weight_payload_codec_t selected;
    if (!rle || !bitmap || !adaptive ||
        tu_compress_rle(w, N, 0.0f, rle, rle_cap, &rle_bytes) ||
        tu_compress_bitmap(w, N, bitmap, bmp_cap, &bmp_bytes) ||
        tu_compress_adaptive(w, N, 0.0f, adaptive, ad_cap, &ad_bytes, &selected)) exit(2);
    __builtin_memcpy(&runs, rle + 4, sizeof(runs));
    __builtin_memcpy(&nnz, bitmap + 4, sizeof(nnz));
    for (size_t i = 0; i < sizeof(profiles) / sizeof(profiles[0]); i++) {
        const decoder_profile_t *p = &profiles[i];
        tu_compress_cycle_stats_t raw_s, rle_s, bmp_s, ad_s;
        uint64_t raw = estimate((const uint8_t *)w, N * sizeof(*w), TU_COMPRESS_NONE, p, &raw_s);
        uint64_t rc = estimate(rle, rle_bytes, TU_COMPRESS_RLE, p, &rle_s);
        uint64_t bc = estimate(bitmap, bmp_bytes, TU_COMPRESS_BITMAP, p, &bmp_s);
        uint64_t ac = estimate(adaptive, ad_bytes, TU_COMPRESS_ADAPTIVE, p, &ad_s);
        printf("%-20s %-8s %4u %4u %4u %5u %5u %-3s %5llu %5llu %5llu %5llu %s\n",
               name, p->name, p->output_width, p->rle_width, p->bitmap_width,
               nnz, runs, codec_name(selected),
               (unsigned long long)raw, (unsigned long long)rc,
               (unsigned long long)bc, (unsigned long long)ac,
               ad_s.decoder_bound ? "DEC" : "DMA");
    }
    free(rle); free(bitmap); free(adaptive);
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
    printf("workload             profile  outW rleW bmpW   nnz  runs fmt   raw   rle   bmp adapt bound\n");
    printf("-------------------- -------- ---- ---- ---- ----- ----- --- ----- ----- ----- ----- -----\n");
    for (uint32_t i = 0; i < N; i++) w[i] = (fp16_t)(i & 1u);
    measure("alternating", w);
    for (unsigned p = 10; p <= 90; p += 20) {
        char name[32];
        fill_random_sparse(w, p); snprintf(name, sizeof(name), "random-zero-%u%%", p); measure(name, w);
        fill_clustered_sparse(w, p); snprintf(name, sizeof(name), "cluster-zero-%u%%", p); measure(name, w);
    }
    for (uint32_t i = 0; i < N; i++) w[i] = 0;
    measure("all-zero", w);
    free(w);
    return 0;
}
