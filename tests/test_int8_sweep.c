/*
 * INT8 Quantization Throughput Sweep (Analytical)
 * ================================================
 * Answers: "How much effective throughput does INT8 quantization
 * deliver over FP16 for GEMM workloads, and how does the gain
 * vary with matrix dimensions?"
 *
 * Model: fill + compute + drain + DMA (weight-stationary)
 * FP16: 2 B/element, accumulator = FP32 (4 B)
 * INT8: 1 B/element, accumulator = INT32 (4 B)
 *
 * No cmodel dependency — pure analytical cycle model.
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ================================================================
 * Architecture parameters (matching tu_config.h defaults)
 * ================================================================ */
#define PDEPTH          2       /* pipeline depth */
#define BUS_WIDTH       32      /* DMA bus width (bytes/cycle) */

/* PE array configurations to sweep */
typedef struct { const char *label; uint16_t pr, pc; } pe_cfg_t;
static pe_cfg_t pes[] = {
    {"8x8",      8,   8},
    {"16x16",   16,  16},
    {"32x32",   32,  32},
    {"64x64",   64,  64},
    {"128x16", 128,  16},
    {"16x128",  16, 128},
};
#define N_PE (sizeof(pes)/sizeof(pes[0]))

/* Workload configs */
#define WK(m,n,k) {#m "x" #n "x" #k, m, n, k}
typedef struct {
    const char *label;
    uint16_t M, N, K;
} wk_cfg_t;

static wk_cfg_t wks[] = {
    WK( 32,  32,  64),
    WK( 64,  64,  64),
    WK( 64,  64, 128),
    WK( 64,  64, 256),
    WK(128, 128,  64),
    WK(128, 128, 128),
    WK(256, 256,  64),
    WK(256, 256, 128),
    WK(512, 512,  64),
    WK( 64, 512,  64),   /* wide-N: decoder projection */
    WK(512,  64,  64),   /* wide-M: encoder projection */
};
#define N_WK (sizeof(wks)/sizeof(wks[0]))

/* ================================================================
 * Cycle model for a single GEMM (M×K × K×N → M×N)
 * ================================================================ */
typedef struct {
    uint64_t dma_cycles;
    uint64_t compute_cycles;
    uint64_t total_cycles;
    uint64_t total_macs;
    double   effective_tops;   /* at 1 GHz clock */
    double   utilization_pct;
} gemm_result_t;

static gemm_result_t model_gemm(uint16_t M, uint16_t N, uint16_t K,
                                 uint16_t pr, uint16_t pc,
                                 uint16_t elem_bytes,  /* 1=INT8, 2=FP16 */
                                 uint16_t accum_bytes) /* 4=FP32/INT32 */
{
    gemm_result_t r = {0};
    uint64_t total_macs = (uint64_t)M * N * K;

    /* ---- Tiling ---- */
    uint16_t mt = (M + pr - 1) / pr;
    uint16_t nt = (N + pc - 1) / pc;
    uint16_t kt = (K + pr - 1) / pr;  /* K tile matches PE rows for WS */
    uint64_t n_tiles = (uint64_t)mt * nt * kt;

    /* ---- DMA cycles ---- */
    /* W buffer: K×N weights, loaded in tiles of pr×pc */
    uint64_t w_total_bytes = (uint64_t)K * N * elem_bytes;
    /* A buffer: M×K activations, loaded in tiles of pr×pr (K tile) */
    uint64_t a_total_bytes = (uint64_t)M * K * elem_bytes;
    /* O buffer: M×N accumulators, stored in tiles of pr×pc */
    uint64_t o_total_bytes = (uint64_t)M * N * accum_bytes;

    uint64_t total_dma_bytes = w_total_bytes + a_total_bytes + o_total_bytes;
    r.dma_cycles = (total_dma_bytes + BUS_WIDTH - 1) / BUS_WIDTH;

    /* ---- Compute cycles (WS dataflow) ---- */
    /* Fill pipeline: pdepth cycles for first tile */
    uint64_t fill = PDEPTH;
    /* Per-tile compute: kt * pc cycles per tile (kt dot products, pc-wide) */
    uint64_t per_tile_compute = (uint64_t)kt * pc;
    /* Total compute */
    r.compute_cycles = fill + n_tiles * per_tile_compute + (PDEPTH - 1); /* drain */

    /* ---- Total ---- */
    r.total_cycles = r.dma_cycles + r.compute_cycles;
    r.total_macs = total_macs;

    /* Effective TOPS at 1 GHz: MACs / total_cycles / 1e12 * 1e9 = MACs / total_cycles / 1000 */
    /* Actually: (2 ops per MAC) * total_macs / total_cycles * 1GHz = 2*total_macs/total_cycles GOPS */
    /* TOPS = (2 * total_macs / total_cycles) * 1e9 / 1e12 = (2*total_macs/total_cycles) / 1000 */
    double gops = (2.0 * total_macs) / (double)r.total_cycles;
    r.effective_tops = gops / 1000.0;

    /* Peak TOPS at 1 GHz: 2 * pr * pc / 1000 */
    double peak_tops = (2.0 * pr * pc) / 1000.0;
    r.utilization_pct = (peak_tops > 0) ? (r.effective_tops / peak_tops * 100.0) : 0.0;

    return r;
}

/* ================================================================
 * Main sweep
 * ================================================================ */
int main(void) {
    printf("\nINT8 vs FP16 Quantization Throughput Sweep\n");
    printf("==========================================\n");
    printf("Model: Weight-stationary, pipeline depth=%d, DMA bus=%d B/cycle\n", PDEPTH, BUS_WIDTH);
    printf("FP16: 2 B/elem, FP32 accum (4 B)\n");
    printf("INT8: 1 B/elem, INT32 accum (4 B)\n");
    printf("\n");

    /* ---- Summary: INT8 vs FP16 speedup by PE config ---- */
    printf("=== SPEEDUP SUMMARY (GEMM 128×128×128) ===\n");
    printf("%-10s  %12s  %12s  %12s  %10s\n",
           "PE Array", "FP16 Cycles", "INT8 Cycles", "Speedup", "INT8 TOPS");
    printf("----------  ------------  ------------  ------------  ----------\n");
    for (int pi = 0; pi < N_PE; pi++) {
        gemm_result_t fp16 = model_gemm(128, 128, 128, pes[pi].pr, pes[pi].pc, 2, 4);
        gemm_result_t int8 = model_gemm(128, 128, 128, pes[pi].pr, pes[pi].pc, 1, 4);
        double speedup = (double)fp16.total_cycles / (double)int8.total_cycles;
        printf("%-10s  %12lu  %12lu  %11.2fx  %9.3f\n",
               pes[pi].label,
               (unsigned long)fp16.total_cycles,
               (unsigned long)int8.total_cycles,
               speedup,
               int8.effective_tops);
    }

    /* ---- Detailed: varied workloads on 32×32 PE ---- */
    printf("\n=== WORKLOAD SWEEP (32×32 PE, weight-stationary) ===\n");
    printf("%-14s  %12s  %12s  %12s  %10s  %10s  %10s\n",
           "GEMM M×N×K", "FP16 Cycles", "INT8 Cycles", "Speedup",
           "FP16 Util%", "INT8 Util%", "DMA Save%");
    printf("--------------  ------------  ------------  ------------  ----------  ----------  ----------\n");
    for (int wi = 0; wi < N_WK; wi++) {
        gemm_result_t fp16 = model_gemm(wks[wi].M, wks[wi].N, wks[wi].K, 32, 32, 2, 4);
        gemm_result_t int8 = model_gemm(wks[wi].M, wks[wi].N, wks[wi].K, 32, 32, 1, 4);
        double speedup = (double)fp16.total_cycles / (double)int8.total_cycles;
        double dma_save_pct = (1.0 - (double)int8.dma_cycles / (double)fp16.dma_cycles) * 100.0;
        printf("%-14s  %12lu  %12lu  %11.2fx  %9.1f%%  %9.1f%%  %9.1f%%\n",
               wks[wi].label,
               (unsigned long)fp16.total_cycles,
               (unsigned long)int8.total_cycles,
               speedup,
               fp16.utilization_pct,
               int8.utilization_pct,
               dma_save_pct);
    }

    /* ---- DMA-vs-compute breakdown for key workload ---- */
    printf("\n=== DMA VS COMPUTE BREAKDOWN (GEMM 256×256×128, 32×32 PE) ===\n");
    gemm_result_t fp16_b = model_gemm(256, 256, 128, 32, 32, 2, 4);
    gemm_result_t int8_b = model_gemm(256, 256, 128, 32, 32, 1, 4);
    printf("%-10s  DMA cycles: %lu (%.1f%%)  Compute: %lu (%.1f%%)  Total: %lu\n",
           "FP16",
           (unsigned long)fp16_b.dma_cycles,
           100.0 * fp16_b.dma_cycles / (double)fp16_b.total_cycles,
           (unsigned long)fp16_b.compute_cycles,
           100.0 * fp16_b.compute_cycles / (double)fp16_b.total_cycles,
           (unsigned long)fp16_b.total_cycles);
    printf("%-10s  DMA cycles: %lu (%.1f%%)  Compute: %lu (%.1f%%)  Total: %lu\n",
           "INT8",
           (unsigned long)int8_b.dma_cycles,
           100.0 * int8_b.dma_cycles / (double)int8_b.total_cycles,
           (unsigned long)int8_b.compute_cycles,
           100.0 * int8_b.compute_cycles / (double)int8_b.total_cycles,
           (unsigned long)int8_b.total_cycles);
    printf("DMA cycle reduction: %.1f%%\n",
           100.0 * (1.0 - (double)int8_b.dma_cycles / (double)fp16_b.dma_cycles));

    /* ---- Key finding ---- */
    /* For small-K workloads, DMA dominates and INT8 wins big.
     * For large-K workloads, compute dominates and INT8 speedup approaches 1.0
     * (same compute, less DMA, but DMA is small fraction of total). */
    gemm_result_t small_k = model_gemm(128, 128, 64, 32, 32, 2, 4);
    gemm_result_t small_k_i8 = model_gemm(128, 128, 64, 32, 32, 1, 4);
    gemm_result_t large_k = model_gemm(128, 128, 512, 32, 32, 2, 4);
    gemm_result_t large_k_i8 = model_gemm(128, 128, 512, 32, 32, 1, 4);

    printf("\n=== K-SENSITIVITY (128×128, 32×32 PE) ===\n");
    printf("K=64:  FP16=%lu cyc  INT8=%lu cyc  speedup=%.2fx  DMA=%.0f%%→%.0f%%\n",
           (unsigned long)small_k.total_cycles, (unsigned long)small_k_i8.total_cycles,
           (double)small_k.total_cycles / small_k_i8.total_cycles,
           100.0*small_k.dma_cycles/(double)small_k.total_cycles,
           100.0*small_k_i8.dma_cycles/(double)small_k_i8.total_cycles);
    printf("K=512: FP16=%lu cyc  INT8=%lu cyc  speedup=%.2fx  DMA=%.0f%%→%.0f%%\n",
           (unsigned long)large_k.total_cycles, (unsigned long)large_k_i8.total_cycles,
           (double)large_k.total_cycles / large_k_i8.total_cycles,
           100.0*large_k.dma_cycles/(double)large_k.total_cycles,
           100.0*large_k_i8.dma_cycles/(double)large_k_i8.total_cycles);
    printf("\nFinding: INT8 speedup is largest for DMA-bound (small-K) workloads.\n");
    printf("For compute-bound (large-K) workloads, INT8 gains diminish — same\n");
    printf("MAC count but half the data movement. The crossover where DMA\n");
    printf("drops below 10%% of total cycles marks diminishing INT8 returns.\n");

    return 0;
}
