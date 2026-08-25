/*
 * TinyTU DMA Descriptor Engine — Test Suite
 * ===========================================
 * Verifies:
 *   1. Linear transfer (host→SRAM, SRAM→host)
 *   2. Strided 2D transfer
 *   3. Strided 3D transfer
 *   4. Descriptor chaining
 *   5. Completion signaling
 *   6. Async mode (queue + tick + flush)
 *   7. Multiple channels
 *   8. Queue overflow
 *   9. Legacy API backward compat
 */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0, tests_pass = 0;

#define TEST(name) do { tests_run++; printf("  %-52s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_pass++; } while(0)
#define FAIL(fmt, ...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); } while(0)

/* ================================================================
 * Test 1: Linear transfer host→SRAM
 * ================================================================ */
static void test_dma_linear_host_to_sram(void) {
    TEST("Linear transfer host→SRAM (128 bytes)");

    tu_dma_init_full(false, 3, 8);

    tu_sram_region_t sram;
    tu_sram_init(&sram, 256, "testbuf");

    /* Prepare source data */
    uint8_t src[128];
    for (int i = 0; i < 128; i++) src[i] = (uint8_t)(i & 0xFF);

    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU,
        &sram, 0, src, 1, 128);

    uint32_t id = tu_dma_submit_desc(desc);
    if (id == 0) { FAIL("submit failed"); tu_sram_destroy(&sram); return; }

    /* Verify SRAM contents (sync mode = immediate) */
    uint8_t *sram_data = tu_sram_raw_ptr(&sram);
    int ok = 1;
    for (int i = 0; i < 128 && ok; i++) {
        if (sram_data[i] != (uint8_t)i) {
            FAIL("sram[%d] = %d, expected %d", i, sram_data[i], i);
            ok = 0;
        }
    }
    if (ok) PASS();

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 2: Linear transfer SRAM→host
 * ================================================================ */
static void test_dma_linear_sram_to_host(void) {
    TEST("Linear transfer SRAM→host (256 bytes)");

    tu_dma_init_full(false, 3, 8);

    tu_sram_region_t sram;
    tu_sram_init(&sram, 512, "testbuf");

    /* Write known data to SRAM */
    uint8_t *raw = tu_sram_raw_ptr(&sram);
    for (int i = 0; i < 256; i++) raw[i] = (uint8_t)(255 - i);

    uint8_t dst[256];
    memset(dst, 0, sizeof(dst));

    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        2, TU_DMA_DIR_TU_TO_HOST,
        &sram, 0, dst, 1, 256);

    uint32_t id = tu_dma_submit_desc(desc);
    if (id == 0) { FAIL("submit failed"); tu_sram_destroy(&sram); return; }

    /* Verify */
    int ok = 1;
    for (int i = 0; i < 256 && ok; i++) {
        if (dst[i] != (uint8_t)(255 - i)) {
            FAIL("dst[%d] = %d, expected %d", i, dst[i], 255 - i);
            ok = 0;
        }
    }
    if (ok) PASS();

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 3: Strided 2D transfer (matrix column extraction)
 * ================================================================ */
static void test_dma_strided_2d(void) {
    TEST("Strided 2D — extract 4×4 col from 4×8 matrix");

    tu_dma_init_full(false, 3, 8);

    /* SRAM has a 4×8 FP32 matrix (row-major, tight packing):
     *   row0:  0  1  2  3  4  5  6  7
     *   row1:  8  9 10 11 12 13 14 15
     *   row2: 16 17 18 19 20 21 22 23
     *   row3: 24 25 26 27 28 29 30 31
     *
     * Extract columns 2-5 (cols 2,3,4,5) into a 4×4 dense host buffer.
     * SRAM row stride = 8 elems * 4 bytes = 32 bytes
     * Host row stride = 4 elems * 4 bytes = 16 bytes
     * Element size = 4 bytes
     * Start at SRAM offset (col=2): 2 * 4 = 8 bytes into each row
     */
    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "matrix");

    fp32_t *src = (fp32_t *)tu_sram_raw_ptr(&sram);
    for (int i = 0; i < 4 * 8; i++) src[i] = (fp32_t)i;

    fp32_t dst[4 * 4];
    memset(dst, 0, sizeof(dst));

    /* SRAM row stride: 8 elems * 4 bytes. Host row stride: 4 elems * 4 bytes. */
    tu_dma_descriptor_t *desc = tu_dma_desc_create_strided_2d(
        2, TU_DMA_DIR_TU_TO_HOST,
        &sram, 2 * 4, dst,  /* SRAM base at column 2 */
        8 * 4,   /* SRAM row stride = 32 bytes */
        4 * 4,   /* Host row stride = 16 bytes */
        4,       /* elem_size = sizeof(fp32_t) */
        4, 4);   /* rows=4, cols=4 */

    uint32_t id = tu_dma_submit_desc(desc);
    if (id == 0) { FAIL("submit failed"); tu_sram_destroy(&sram); return; }

    /* Expected: each row in dst has cols 2,3,4,5 from the corresponding source row */
    int ok = 1;
    float expected[4][4] = {
        {2, 3, 4, 5},
        {10, 11, 12, 13},
        {18, 19, 20, 21},
        {26, 27, 28, 29}
    };
    for (int r = 0; r < 4 && ok; r++) {
        for (int c = 0; c < 4 && ok; c++) {
            if (dst[r * 4 + c] != expected[r][c]) {
                FAIL("dst[%d][%d] = %f, expected %f",
                     r, c, dst[r * 4 + c], expected[r][c]);
                ok = 0;
            }
        }
    }
    if (ok) PASS();

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 4: Strided 3D transfer
 * ================================================================ */
static void test_dma_strided_3d(void) {
    TEST("Strided 3D — 2×2×2 cube from 4×4×4 volume");

    tu_dma_init_full(false, 3, 8);

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "volume");

    /* SRAM has a 4×4×4 FP32 volume (depth-major):
     * Fill with sequential values 0..63
     */
    fp32_t *src = (fp32_t *)tu_sram_raw_ptr(&sram);
    for (int i = 0; i < 4 * 4 * 4; i++) src[i] = (fp32_t)i;

    /* Extract a 2×2×2 sub-volume starting at depth=1, row=1, col=1.
     *
     * SRAM layout (depth-major, row-major within each slice):
     *   offset(d,r,c) = d*16 + r*4 + c  (in elements)
     *   offset in bytes = offset(d,r,c) * 4
     *
     * Start: d=1, r=1, c=1 → offset = 1*16 + 1*4 + 1 = 21 elements = 84 bytes
     * Strides:
     *   row_stride    = 4 * 4 = 16 bytes (1 row in the 4×4 slice)
     *   depth_stride  = 16 * 4 = 64 bytes (1 slice)
     *
     * Destination: 2×2×2 dense buffer
     *   depth_stride  = 2*2 * 4 = 16 bytes
     *   row_stride    = 2 * 4 = 8 bytes
     */
    fp32_t dst[2 * 2 * 2];
    memset(dst, 0, sizeof(dst));

    tu_dma_descriptor_t *desc = tu_dma_desc_create_strided_3d(
        2, TU_DMA_DIR_TU_TO_HOST,
        &sram, 21 * 4, dst,
        16,     /* SRAM row stride */
        64,     /* SRAM depth stride */
        8,      /* Host row stride */
        16,     /* Host depth stride */
        4,      /* elem_size */
        2, 2, 2);  /* depth=2, rows=2, cols=2 */

    uint32_t id = tu_dma_submit_desc(desc);
    if (id == 0) { FAIL("submit failed"); tu_sram_destroy(&sram); return; }

    /* Expected: elements from SRAM at positions [d][r][c]:
     * d=1,r=1,c=1 → 21,  d=1,r=1,c=2 → 22,  d=1,r=2,c=1 → 25, d=1,r=2,c=2 → 26
     * d=2,r=1,c=1 → 37,  d=2,r=1,c=2 → 38,  d=2,r=2,c=1 → 41, d=2,r=2,c=2 → 42
     */
    float expected[] = {21, 22, 25, 26, 37, 38, 41, 42};
    int ok = 1;
    for (int i = 0; i < 8 && ok; i++) {
        if (dst[i] != expected[i]) {
            FAIL("dst[%d] = %f, expected %f", i, dst[i], expected[i]);
            ok = 0;
        }
    }
    if (ok) PASS();

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 5: Descriptor chaining
 * ================================================================ */
static void test_dma_chain(void) {
    TEST("Descriptor chaining (3 linked xfers)");

    tu_dma_init_full(false, 3, 8);

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "chainbuf");

    /* Chain: load 64 bytes at offset 0, then 64 at offset 128, then 64 at offset 256 */
    uint8_t src0[64], src1[64], src2[64];
    for (int i = 0; i < 64; i++) { src0[i] = 0xAA; src1[i] = 0xBB; src2[i] = 0xCC; }

    tu_dma_descriptor_t *d0 = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, src0, 1, 64);
    tu_dma_descriptor_t *d1 = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 128, src1, 1, 64);
    tu_dma_descriptor_t *d2 = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 256, src2, 1, 64);

    tu_dma_desc_chain(d0, d1);
    tu_dma_desc_chain(d0, d2);

    uint32_t id = tu_dma_submit_desc(d0);
    if (id == 0) { FAIL("submit failed"); tu_sram_destroy(&sram); return; }

    uint8_t *raw = tu_sram_raw_ptr(&sram);

    /* Check offset 0: all 0xAA */
    int ok = 1;
    for (int i = 0; i < 64 && ok; i++) {
        if (raw[i] != 0xAA) { FAIL("offset 0[%d] = %02x", i, raw[i]); ok = 0; }
    }
    /* Check offset 128: all 0xBB */
    for (int i = 0; i < 64 && ok; i++) {
        if (raw[128 + i] != 0xBB) { FAIL("offset 128[%d] = %02x", i, raw[128+i]); ok = 0; }
    }
    /* Check offset 256: all 0xCC */
    for (int i = 0; i < 64 && ok; i++) {
        if (raw[256 + i] != 0xCC) { FAIL("offset 256[%d] = %02x", i, raw[256+i]); ok = 0; }
    }
    /* Check gap at offset 64-127: should be zeros */
    for (int i = 64; i < 128 && ok; i++) {
        if (raw[i] != 0) { FAIL("gap[%d] = %02x, expected 0", i, raw[i]); ok = 0; }
    }
    if (ok) PASS();

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 6: Async mode — queue, tick, flush
 * ================================================================ */
static void test_dma_async(void) {
    TEST("Async mode: queue + tick + flush");

    tu_dma_init_full(true, 3, 8);  /* async=true */

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "asyncbuf");

    uint8_t src[128];
    for (int i = 0; i < 128; i++) src[i] = (uint8_t)i;

    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, src, 1, 128);

    /* Submit in async mode — should NOT execute immediately */
    uint32_t id = tu_dma_submit_desc(desc);
    if (id == 0) { FAIL("submit failed"); tu_sram_destroy(&sram); return; }

    /* Verify SRAM is still zero (not yet executed) */
    uint8_t *raw = tu_sram_raw_ptr(&sram);
    int all_zero = 1;
    for (int i = 0; i < 128; i++) {
        if (raw[i] != 0) { all_zero = 0; break; }
    }
    if (!all_zero) { FAIL("data appeared before tick"); tu_sram_destroy(&sram); return; }

    /* Tick to execute */
    for (int i = 0; i < 200; i++) tu_dma_tick();

    /* Now verify */
    int ok = 1;
    for (int i = 0; i < 128 && ok; i++) {
        if (raw[i] != (uint8_t)i) {
            FAIL("sram[%d] = %d after tick", i, raw[i]);
            ok = 0;
        }
    }
    if (ok) PASS();

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 7: Large element transfer (FP32 matrix)
 * ================================================================ */
static void test_dma_fp32_transfer(void) {
    TEST("FP32 matrix transfer (16×16, 1024 bytes)");

    tu_dma_init_full(false, 3, 8);

    tu_sram_region_t sram;
    tu_sram_init(&sram, 4096, "mat");

    fp32_t src[16 * 16];
    for (int i = 0; i < 16 * 16; i++) src[i] = (fp32_t)(i * 0.5f);

    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU,
        &sram, 0, src, sizeof(fp32_t), 16 * 16);

    uint32_t id = tu_dma_submit_desc(desc);
    if (id == 0) { FAIL("submit failed"); tu_sram_destroy(&sram); return; }

    /* Verify */
    fp32_t *dst = (fp32_t *)tu_sram_raw_ptr(&sram);
    int ok = 1;
    for (int i = 0; i < 16 * 16 && ok; i++) {
        if (dst[i] != src[i]) {
            FAIL("dst[%d] = %f, expected %f", i, dst[i], src[i]);
            ok = 0;
        }
    }
    if (ok) PASS();

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 8: Legacy API backward compat
 * ================================================================ */
static void test_dma_legacy_api(void) {
    TEST("Legacy tu_dma_load + tu_dma_store");

    tu_dma_init(false);

    tu_sram_region_t sram;
    tu_sram_init(&sram, 1024, "legacy");

    uint8_t src[64];
    for (int i = 0; i < 64; i++) src[i] = (uint8_t)(i + 100);

    tu_dma_load(TU_DMA_CHAN_W, &sram, 0, src, 64);

    uint8_t dst[64] = {0};
    tu_dma_store(TU_DMA_CHAN_O, &sram, 0, dst, 64);

    int ok = 1;
    for (int i = 0; i < 64 && ok; i++) {
        if (dst[i] != (uint8_t)(i + 100)) {
            FAIL("dst[%d] = %d", i, dst[i]);
            ok = 0;
        }
    }
    if (ok) PASS();

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 9: Completion signaling
 * ================================================================ */
static void test_dma_completion_signal(void) {
    TEST("Completion signal assignment");

    tu_dma_init_full(false, 3, 8);

    tu_sram_region_t sram;
    tu_sram_init(&sram, 256, "sigbuf");

    uint8_t src[64];
    memset(src, 0xDE, 64);

    tu_dma_descriptor_t *desc = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, src, 1, 64);
    desc->signal_id = 42;

    uint32_t id = tu_dma_submit_desc(desc);
    if (id == 0) { FAIL("submit failed"); tu_sram_destroy(&sram); return; }

    /* In sync mode, descriptor should be completed */
    if (desc->completed) PASS();
    else FAIL("descriptor not completed");

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 10: Verifies transfer stats
 * ================================================================ */
static void test_dma_stats(void) {
    TEST("DMA transfer statistics");

    tu_dma_init_full(false, 3, 8);

    tu_sram_region_t sram;
    tu_sram_init(&sram, 1024, "statsbuf");

    uint8_t src[512];
    memset(src, 0x42, 512);

    tu_dma_descriptor_t *d1 = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, src, 1, 256);
    tu_dma_descriptor_t *d2 = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 256, src, 1, 256);

    tu_dma_submit_desc(d1);
    tu_dma_submit_desc(d2);

    if (g_tu_dma.total_bytes == 512 && g_tu_dma.total_transfers == 2) PASS();
    else FAIL("bytes=%lu transfers=%lu", g_tu_dma.total_bytes, g_tu_dma.total_transfers);

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

/* ================================================================
 * Test 11: Runtime channel capacity must fail closed
 * ================================================================ */
static void test_dma_channel_capacity(void) {
    TEST("Runtime channels/topology fail closed");

    tu_dma_init_config(true, TU_DMA_ENGINE_MAX_CHANNELS, 4,
                       TU_DMA_BUS_MODE_SHARED_SERIAL);
    if (g_tu_dma.num_channels != TU_DMA_ENGINE_MAX_CHANNELS ||
        g_tu_dma.bus_mode != TU_DMA_BUS_MODE_SHARED_SERIAL) {
        FAIL("max channel count/topology did not initialize");
        return;
    }
    tu_dma_destroy();

    tu_dma_init_full(true, TU_DMA_ENGINE_MAX_CHANNELS + 1, 4);
    if (g_tu_dma.num_channels != 0) {
        FAIL("unsupported count silently became %u", g_tu_dma.num_channels);
        tu_dma_destroy();
        return;
    }
    tu_dma_destroy();

    tu_dma_init_config(true, 3, 4, 2);
    if (g_tu_dma.num_channels == 0) PASS();
    else FAIL("unsupported topology initialized %u channels", g_tu_dma.num_channels);
    tu_dma_destroy();
}

/* ================================================================
 * Test 12: max_outstanding includes the active descriptor
 * ================================================================ */
static void test_dma_outstanding_includes_active(void) {
    TEST("Max outstanding counts active + queued descriptors");

    tu_dma_init_full(true, 1, 1);
    tu_sram_region_t sram;
    tu_sram_init(&sram, 256, "outstanding");
    static uint8_t src0[64], src1[64];
    memset(src0, 0x11, sizeof(src0));
    memset(src1, 0x22, sizeof(src1));

    tu_dma_descriptor_t *d0 = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 0, src0, 1, sizeof(src0));
    uint32_t id0 = tu_dma_submit_desc(d0);
    tu_dma_tick(); /* d0 becomes active; pending depth returns to zero */

    tu_dma_descriptor_t *d1 = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &sram, 64, src1, 1, sizeof(src1));
    uint32_t id1 = tu_dma_submit_desc(d1);

    for (int i = 0; i < 200; i++) tu_dma_tick();
    uint8_t *raw = tu_sram_raw_ptr(&sram);
    int ok = id0 > 0 && id1 == 0 && g_tu_dma.channels[0].total_submitted == 1;
    for (size_t i = 0; i < sizeof(src0) && ok; i++) ok = raw[i] == 0x11;
    for (size_t i = 0; i < sizeof(src1) && ok; i++) ok = raw[64 + i] == 0;

    if (ok) PASS();
    else FAIL("id0=%u id1=%u submitted=%lu", id0, id1,
              (unsigned long)g_tu_dma.channels[0].total_submitted);

    tu_sram_destroy(&sram);
    tu_dma_destroy();
}

static void test_dma_shared_arbitration(void) {
    TEST("Shared bus round-robin / strict-priority arbitration");
    tu_sram_region_t sram;
    static uint8_t src[3][64];
    int ok = 1;
    tu_sram_init(&sram, 3 * 64, "arbitration");

    tu_dma_init_config_policy(true, 3, 2, TU_DMA_BUS_MODE_SHARED_SERIAL,
                              TU_DMA_ARB_ROUND_ROBIN);
    for (uint8_t i = 0; i < 3; i++) {
        tu_dma_descriptor_t *d = tu_dma_desc_create_linear(
            i, TU_DMA_DIR_HOST_TO_TU, &sram, i * 64, src[i], 1, 64);
        d->priority = (uint8_t)(i == 1 ? 10 : i == 2 ? 5 : 0);
        ok = ok && tu_dma_submit_desc(d) > 0;
    }
    tu_dma_tick();
    ok = ok && g_tu_dma.channels[0].active != NULL &&
         g_tu_dma.channels[1].active == NULL;
    tu_dma_destroy();

    tu_dma_init_config_policy(true, 3, 2, TU_DMA_BUS_MODE_SHARED_SERIAL,
                              TU_DMA_ARB_STRICT_PRIORITY);
    for (uint8_t i = 0; i < 3; i++) {
        tu_dma_descriptor_t *d = tu_dma_desc_create_linear(
            i, TU_DMA_DIR_HOST_TO_TU, &sram, i * 64, src[i], 1, 64);
        d->priority = (uint8_t)(i == 1 ? 10 : i == 2 ? 5 : 0);
        ok = ok && tu_dma_submit_desc(d) > 0;
    }
    tu_dma_tick();
    ok = ok && g_tu_dma.channels[1].active != NULL &&
         g_tu_dma.channels[0].active == NULL;
    tu_dma_destroy();

    /* Equal priority falls back to rotating tie-break, not channel 0 forever. */
    tu_dma_init_config_policy(true, 3, 2, TU_DMA_BUS_MODE_SHARED_SERIAL,
                              TU_DMA_ARB_STRICT_PRIORITY);
    for (uint8_t i = 0; i < 3; i++) {
        tu_dma_descriptor_t *d = tu_dma_desc_create_linear(
            i, TU_DMA_DIR_HOST_TO_TU, &sram, i * 64, src[i], 1, 64);
        d->priority = 7;
        ok = ok && tu_dma_submit_desc(d) > 0;
    }
    tu_dma_tick();
    ok = ok && g_tu_dma.channels[0].active != NULL;
    for (int i = 0; i < 200 && g_tu_dma.channels[0].total_completed == 0; i++)
        tu_dma_tick();
    ok = ok && g_tu_dma.channels[1].active != NULL;
    tu_dma_destroy();

    tu_dma_init_config_policy(true, 3, 2, TU_DMA_BUS_MODE_SHARED_SERIAL, 2);
    ok = ok && g_tu_dma.num_channels == 0;
    tu_dma_destroy();
    tu_sram_destroy(&sram);
    if (ok) PASS(); else FAIL("policy selection or rejection failed");
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
    printf("TinyTU DMA Descriptor Engine Tests\n");
    printf("==================================\n\n");

    test_dma_linear_host_to_sram();
    test_dma_linear_sram_to_host();
    test_dma_strided_2d();
    test_dma_strided_3d();
    test_dma_chain();
    test_dma_async();
    test_dma_fp32_transfer();
    test_dma_legacy_api();
    test_dma_completion_signal();
    test_dma_stats();
    test_dma_channel_capacity();
    test_dma_outstanding_includes_active();
    test_dma_shared_arbitration();

    printf("\n═══════════════════════════════════════════\n");
    printf("  %d/%d tests passed\n", tests_pass, tests_run);
    printf("═══════════════════════════════════════════\n");
    return tests_pass == tests_run ? 0 : 1;
}
