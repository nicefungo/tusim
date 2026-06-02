/*
 * TU CModel — Comparative Benchmark Suite
 * =========================================
 * MLPerf Tiny workloads + ResNet-50 + Transformer benchmarks.
 * Measures FLOPs, cycles, utilization, bandwidth, and compares
 * against theoretical peak. Uses the perf counter infrastructure
 * for comprehensive metrics.
 *
 * All dimensions are sized to fit within the default 16×16 PE /
 * 256 KB SRAM configuration. O-buffer limit: 64KB = 32K FP16 elements.
 *
 * Gap: P2.9 (Comparative Benchmarking), V5 (Benchmark Suite)
 */

#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/tu_core.h"
#include "tu_cmodel/tu_config.h"
#include "tu_cmodel/tu_precision.h"
#include "tu_cmodel/perf/performance_counters.h"
#include "tu_cmodel/dma_descriptor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Benchmark metadata ───────────────────────────────────────── */

typedef struct {
    const char *name;
    const char *category;
    uint64_t    total_flops;
    uint64_t    total_cycles;
    uint64_t    total_bytes;
    uint64_t    peak_flops_per_cycle;
    float       utilization;
    float       tops;
    float       bandwidth_gbps;
    float       mac_efficiency;
} tu_bench_result_t;

static tu_bench_result_t g_results[32];
static int g_result_count = 0;

/* ── Utility ──────────────────────────────────────────────────── */

static tu_core_t *bench_init(void) {
    tu_runtime_config_t cfg = tu_runtime_config_default();
    tu_core_t *core = tu_core_create(&cfg);
    if (!core) {
        fprintf(stderr, "BENCH: failed to create TU core\n");
        return NULL;
    }
    tu_core_init(core);
    return core;
}

static void fill_random_fp16(fp16_t *data, uint32_t count, uint32_t seed) {
    srand(seed);
    for (uint32_t i = 0; i < count; i++) {
        float v = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        data[i] = tu_fp32_to_fp16(v);
    }
}

/* ── Benchmark runner ─────────────────────────────────────────── */

typedef struct {
    tu_core_t *core;
    tu_perf_counters_t counters;
    uint32_t pe_rows;
    uint32_t pe_cols;
} bench_ctx_t;

static void bench_start(bench_ctx_t *ctx, tu_core_t *core) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->core = core;
    ctx->pe_rows = core->state.rt_cfg.pe_rows;
    ctx->pe_cols = core->state.rt_cfg.pe_cols;
    tu_perf_init(&ctx->counters, 1000.0);
}

static void bench_stop(bench_ctx_t *ctx, tu_bench_result_t *result) {
    tu_perf_metrics_t m = tu_perf_compute_metrics(&ctx->counters);
    result->total_flops   = ctx->counters.compute.total_flops;
    result->total_cycles  = ctx->counters.total_cycles;
    result->total_bytes   = ctx->counters.dma.dma_read_bytes
                          + ctx->counters.dma.dma_write_bytes;
    result->utilization   = m.compute_utilization;
    result->tops          = m.mac_throughput_tops;
    result->bandwidth_gbps = m.dma_bandwidth_gbps;
    result->mac_efficiency = m.mac_efficiency;
    result->peak_flops_per_cycle = (uint64_t)ctx->pe_rows * ctx->pe_cols * 2;
}

/* ── MMA with perf tracking + tiled DMA ─────────────────────── */

/*
 * Execute a GEMM M×N×K using the TU cmodel.
 * Weights, activations, and output are DMA'd in/out respecting SRAM limits.
 * The TU MMA tiles internally; we handle multi-tile output DMA.
 */
static void bench_mma(bench_ctx_t *ctx,
                       uint32_t M, uint32_t N, uint32_t K,
                       fp16_t *w_host, fp16_t *a_host, fp16_t *o_host)
{
    tu_core_t *core = ctx->core;
    uint32_t o_max_elems = core->state.rt_cfg.sram_o_size / sizeof(fp16_t);
    uint32_t pe_rows = ctx->pe_rows;
    uint32_t pe_cols = ctx->pe_cols;

    /* DMA weights in (entire weight matrix fits in W-buffer) */
    uint32_t w_bytes = M * K * sizeof(fp16_t);
    tu_core_dma_load_w(core, w_host, 0, w_bytes);
    tu_perf_dma_record_read(&ctx->counters, w_bytes, 10, 0, TU_DMA_CHAN_W, 0);

    /* DMA activations in (entire activation matrix) */
    uint32_t a_bytes = K * N * sizeof(fp16_t);
    tu_core_dma_load_a(core, a_host, 0, a_bytes);
    tu_perf_dma_record_read(&ctx->counters, a_bytes, 10, 0, TU_DMA_CHAN_A, 0);

    /* Compute total FLOPs */
    uint64_t flops = (uint64_t)M * N * K * 2;
    uint64_t tiles_M = (M + pe_rows - 1) / pe_rows;
    uint64_t tiles_N = (N + pe_cols - 1) / pe_cols;
    uint64_t tile_K = 16;
    uint64_t k_tiles = (K + tile_K - 1) / tile_K;
    uint64_t total_tiles = tiles_M * tiles_N * k_tiles;
    uint64_t fill_per_tile = pe_rows + pe_cols;
    uint64_t compute_per_tile = tile_K;
    uint64_t cycles = total_tiles * (fill_per_tile + compute_per_tile) + 10;

    /* MMA: the cmodel internally tiles and writes output to O-buffer.
     * Since O-buffer may be smaller than M×N, we need tiled output DMA.
     * The cmodel's tu_mma() handles the compute tiling, but we must
     * DMA output in tiles that fit in O-buffer. */
    uint32_t tile_m = pe_rows;
    while (o_max_elems / tile_m < N && tile_m > 1) {
        tile_m /= 2;
    }
    if (tile_m < 1) tile_m = 1;
    uint32_t tile_elems = tile_m * N;
    if (tile_elems > o_max_elems) tile_elems = o_max_elems;

    /* Simplified: use the cmodel's built-in MMA which handles tiling */
    tu_core_mma(core, (uint16_t)M, (uint16_t)N, (uint16_t)K, 0, 0, 0, false);

    tu_perf_compute_record_mma(&ctx->counters, flops / 2, M, N, K,
                                total_tiles, 0, cycles, 0, 0, 0);
    tu_perf_tick(&ctx->counters, cycles);

    /* DMA output: tile if needed */
    for (uint32_t m_off = 0; m_off < M; m_off += tile_m) {
        uint32_t m_chunk = (m_off + tile_m > M) ? (M - m_off) : tile_m;
        uint32_t chunk_bytes = m_chunk * N * sizeof(fp16_t);
        tu_core_dma_store_o(core, o_host + m_off * N, 0, chunk_bytes);
        tu_perf_dma_record_write(&ctx->counters, chunk_bytes, 10, 0, TU_DMA_CHAN_O);
    }
}

/* ── bench_mma with on-the-fly allocation ────────────────────── */

static void bench_mma_alloc(bench_ctx_t *ctx,
                             uint32_t M, uint32_t N, uint32_t K)
{
    fp16_t *w = malloc(M * K * sizeof(fp16_t));
    fp16_t *a = malloc(K * N * sizeof(fp16_t));
    fp16_t *o = malloc(M * N * sizeof(fp16_t));

    if (!w || !a || !o) {
        fprintf(stderr, "BENCH: allocation failed for %u×%u×%u\n", M, N, K);
        free(w); free(a); free(o);
        return;
    }

    fill_random_fp16(w, M * K, (uint32_t)(M * 1000 + N));
    fill_random_fp16(a, K * N, (uint32_t)(N * 1000 + K));

    bench_mma(ctx, M, N, K, w, a, o);

    free(w); free(a); free(o);
}

/* ═══════════════════════════════════════════════════════════════
 * MLPerf Tiny Benchmarks
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Keyword Spotting (KWS): Small 3-layer MLP
 * Scaled for 16×16 PE / 64KB O-buffer (max ~32K elements per output tile)
 *   Layer 1: M=128, N=64, K=128   → 8,192 output elements (fits)
 *   Layer 2: M=64, N=32, K=64     → 2,048 output elements
 *   Layer 3: M=32, N=12, K=32     →   384 output elements
 */
static void bench_kws(bench_ctx_t *ctx) {
    bench_mma_alloc(ctx, 128, 64, 128);
    bench_mma_alloc(ctx, 64, 32, 64);
    bench_mma_alloc(ctx, 32, 12, 32);
}

/*
 * Visual Wake Words (VWW): Depthwise-separable conv approximated as GEMMs
 *   Conv1-like:  M=256, N=16, K=64
 *   Conv2-like:  M=128, N=32, K=128
 */
static void bench_vww(bench_ctx_t *ctx) {
    bench_mma_alloc(ctx, 256, 16, 64);
    bench_mma_alloc(ctx, 128, 32, 128);
}

/*
 * Anomaly Detection: Autoencoder
 *   Encoder: 256→64→16
 *   Decoder: 16→64→256
 */
static void bench_anomaly(bench_ctx_t *ctx) {
    bench_mma_alloc(ctx, 256, 64, 1);   /* Encoder compress */
    bench_mma_alloc(ctx, 64, 16, 1);    /* Bottleneck */
    bench_mma_alloc(ctx, 16, 64, 1);    /* Decoder expand */
    bench_mma_alloc(ctx, 64, 256, 1);   /* Decoder output */
}

/*
 * Image Classification: ConvNet-style blocks (GEMM-im2col equivalent)
 *   3 blocks of increasing channel depth
 */
static void bench_ic(bench_ctx_t *ctx) {
    /* Block 1: M=128, N=64, K=128 */
    bench_mma_alloc(ctx, 128, 64, 128);
    /* Block 2: M=64, N=128, K=128 */
    bench_mma_alloc(ctx, 64, 128, 128);
    /* Block 3: M=128, N=16, K=128 */
    bench_mma_alloc(ctx, 128, 16, 128);
}

/* ═══════════════════════════════════════════════════════════════
 * ResNet-50 Bottleneck Block (scaled)
 * ═══════════════════════════════════════════════════════════════ */

/*
 * ResNet-50 Bottleneck (scaled):
 *   1×1 reduce:  M=128, N=64, K=128
 *   3×3 spatial: M=128, N=64, K=128
 *   1×1 expand:  M=128, N=128, K=64
 *   Skip 1×1:    M=128, N=128, K=128
 */
static void bench_resnet50_block(bench_ctx_t *ctx) {
    bench_mma_alloc(ctx, 128, 64, 128);   /* 1×1 reduce */
    bench_mma_alloc(ctx, 128, 64, 128);   /* 3×3 spatial */
    bench_mma_alloc(ctx, 128, 128, 64);   /* 1×1 expand */
    bench_mma_alloc(ctx, 128, 128, 128);  /* Skip projection */
}

/* ═══════════════════════════════════════════════════════════════
 * Transformer Benchmarks
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Transformer Encoder Block (BERT-style, scaled)
 *   QKV:  M=64, N=192, K=64
 *   Out:  M=64, N=64, K=64
 *   FFN1: M=64, N=256, K=64
 *   FFN2: M=64, N=64, K=256
 */
static void bench_transformer_encoder(bench_ctx_t *ctx) {
    bench_mma_alloc(ctx, 64, 192, 64);    /* QKV projection */
    bench_mma_alloc(ctx, 64, 64, 64);     /* Attention output */
    bench_mma_alloc(ctx, 64, 256, 64);    /* FFN up */
    bench_mma_alloc(ctx, 64, 64, 256);    /* FFN down */
}

/*
 * Transformer Decoder Block (GPT-style, scaled)
 *   Self-QKV:   M=32, N=96, K=32
 *   Self-Out:   M=32, N=32, K=32
 *   Cross-Q:    M=32, N=32, K=32
 *   Cross-KV:   M=32, N=64, K=32
 *   Cross-Out:  M=32, N=32, K=32
 *   FFN-Up:     M=32, N=128, K=32
 *   FFN-Down:   M=32, N=32, K=128
 */
static void bench_transformer_decoder(bench_ctx_t *ctx) {
    bench_mma_alloc(ctx, 32, 96, 32);     /* Self-QKV */
    bench_mma_alloc(ctx, 32, 32, 32);     /* Self-Out */
    bench_mma_alloc(ctx, 32, 32, 32);     /* Cross-Q */
    bench_mma_alloc(ctx, 32, 64, 32);     /* Cross-KV */
    bench_mma_alloc(ctx, 32, 32, 32);     /* Cross-Out */
    bench_mma_alloc(ctx, 32, 128, 32);    /* FFN-Up */
    bench_mma_alloc(ctx, 32, 32, 128);    /* FFN-Down */
}

/* ── Report ───────────────────────────────────────────────────── */

static void print_separator(void) {
    printf("├──────────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤\n");
}

static void print_result(const tu_bench_result_t *r) {
    double peak_tops = (double)r->peak_flops_per_cycle * 1.0 / 1000.0;
    printf("│ %-24s │ %8.2f │ %8.2f │ %8.2f │ %8.1f │ %8.2f │ %8.2f │\n",
           r->name,
           (double)r->total_flops / 1e6,            /* MFLOPs */
           (double)r->total_cycles / 1e3,            /* kCycles */
           r->tops * 1000.0,                          /* mTOPS (from GFLOPS) */
           r->utilization * 100.0,
           peak_tops,
           r->bandwidth_gbps);
}

static void print_summary_header(uint32_t pe_rows, uint32_t pe_cols) {
    double peak_tops = (double)pe_rows * pe_cols * 2.0 / 1000.0;
    printf("\n╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║     TU CModel Comparative Benchmark Results                         ║\n");
    printf("║     Hardware: %u×%u PE array, FP16, 1 GHz, Peak %.2f TOPS           ║\n",
           pe_rows, pe_cols, peak_tops);
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Workload                │ MFLOPs   │ kCycles  │ mTOPS    │ Util %%   │ Peak TOPS│ BW GB/s  ║\n");
    print_separator();
}

static void print_summary_footer(void) {
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(void) {
    tu_core_t *core = bench_init();
    if (!core) return 1;

    bench_ctx_t ctx;
    uint32_t pe_rows = core->state.rt_cfg.pe_rows;
    uint32_t pe_cols = core->state.rt_cfg.pe_cols;

    printf("\nTU CModel Comparative Benchmark Suite\n");
    printf("=======================================\n");
    printf("Array: %u×%u PEs, FP16 precision, 1 GHz clock\n", pe_rows, pe_cols);
    printf("Peak: %u MACs/cycle = %.2f TOPS\n",
           pe_rows * pe_cols,
           (double)(pe_rows * pe_cols * 2) / 1000.0);
    printf("Note: Dimensions scaled to fit 256 KB SRAM (64 KB O-buffer)\n\n");

    /* ── MLPerf Tiny ───────────────────── */
    printf("─── MLPerf Tiny ─────────────────────────────\n");

    bench_start(&ctx, core);
    bench_kws(&ctx);
    bench_stop(&ctx, &g_results[g_result_count]);
    g_results[g_result_count].name = "KWS (Keyword Spotting)";
    g_results[g_result_count].category = "MLPerf Tiny";
    g_result_count++;

    bench_start(&ctx, core);
    bench_vww(&ctx);
    bench_stop(&ctx, &g_results[g_result_count]);
    g_results[g_result_count].name = "VWW (Visual Wake Words)";
    g_results[g_result_count].category = "MLPerf Tiny";
    g_result_count++;

    bench_start(&ctx, core);
    bench_anomaly(&ctx);
    bench_stop(&ctx, &g_results[g_result_count]);
    g_results[g_result_count].name = "AD (Anomaly Detection)";
    g_results[g_result_count].category = "MLPerf Tiny";
    g_result_count++;

    bench_start(&ctx, core);
    bench_ic(&ctx);
    bench_stop(&ctx, &g_results[g_result_count]);
    g_results[g_result_count].name = "IC (Image Classification)";
    g_results[g_result_count].category = "MLPerf Tiny";
    g_result_count++;

    /* ── ResNet-50 ─────────────────────── */
    printf("─── ResNet-50 ──────────────────────────────\n");

    bench_start(&ctx, core);
    bench_resnet50_block(&ctx);
    bench_stop(&ctx, &g_results[g_result_count]);
    g_results[g_result_count].name = "ResNet-50 Bottleneck";
    g_results[g_result_count].category = "ResNet";
    g_result_count++;

    /* ── Transformer ────────────────────── */
    printf("─── Transformer ────────────────────────────\n");

    bench_start(&ctx, core);
    bench_transformer_encoder(&ctx);
    bench_stop(&ctx, &g_results[g_result_count]);
    g_results[g_result_count].name = "Transformer Encoder";
    g_results[g_result_count].category = "Transformer";
    g_result_count++;

    bench_start(&ctx, core);
    bench_transformer_decoder(&ctx);
    bench_stop(&ctx, &g_results[g_result_count]);
    g_results[g_result_count].name = "Transformer Decoder";
    g_results[g_result_count].category = "Transformer";
    g_result_count++;

    /* ── Summary ───────────────────────── */
    print_summary_header(pe_rows, pe_cols);

    for (int i = 0; i < g_result_count; i++) {
        print_result(&g_results[i]);
    }
    print_separator();
    print_summary_footer();

    /* Peak comparison */
    double peak_tops = (double)pe_rows * pe_cols * 2.0 / 1000.0;
    printf("\n  Peak TOPS: %.2f (at 1 GHz, %u×%u array)\n",
           peak_tops, pe_rows, pe_cols);
    printf("  Average utilization: ");
    double avg_util = 0;
    for (int i = 0; i < g_result_count; i++) {
        avg_util += g_results[i].utilization;
    }
    avg_util /= g_result_count;
    printf("%.1f%%\n", avg_util * 100.0);

    printf("  Average MAC efficiency: ");
    double avg_eff = 0;
    for (int i = 0; i < g_result_count; i++) {
        avg_eff += g_results[i].mac_efficiency;
    }
    avg_eff /= g_result_count;
    printf("%.1f%%\n", avg_eff * 100.0);

    /* Category breakdown */
    printf("\n  ── Per-Category Average Utilization ──\n");
    const char *cats[] = {"MLPerf Tiny", "ResNet", "Transformer"};
    for (int c = 0; c < 3; c++) {
        double cat_util = 0;
        int cat_count = 0;
        for (int i = 0; i < g_result_count; i++) {
            if (strcmp(g_results[i].category, cats[c]) == 0) {
                cat_util += g_results[i].utilization;
                cat_count++;
            }
        }
        if (cat_count > 0) {
            printf("    %-20s: %5.1f%% (%d workloads)\n",
                   cats[c], cat_util / cat_count * 100.0, cat_count);
        }
    }

    tu_core_destroy(core);
    printf("\nBenchmark complete.\n");
    return 0;
}
