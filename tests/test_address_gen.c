/*
 * TU CModel — Address Generator Tests (Gap M3)
 * ==============================================
 * Tests for all addressing modes: linear, strided 2D/3D,
 * tiled 2D/3D, im2col, block interleaved, transposed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/tu_sram.h"
#include "../tu_cmodel/dma_descriptor.h"
#include "../tu_cmodel/memory/address_generator.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  TEST %s ... ", name); \
    fflush(stdout); \
} while(0)

#define PASS() do { \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
    tests_failed++; \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    long _a = (long)(a); \
    long _b = (long)(b); \
    if (_a != _b) { FAIL(msg); printf("  expected %ld, got %ld\n", _b, _a); return; } \
} while(0)

/* ---- Test 1: Linear addressing ---- */
static void test_linear(void) {
    TEST("linear addressing");
    tu_agen_iterator_t it;
    uint32_t config[2] = { 16, 4 }; /* 16 elements, 4 bytes each */
    int rc = tu_agen_iterator_init(&it, TU_AGEN_MODE_LINEAR, 0x1000, config);
    ASSERT_EQ(rc, 0, "init should succeed");
    ASSERT_EQ((long)tu_agen_total(&it), 16L, "16 elements total");

    for (uint32_t i = 0; i < 16; i++) {
        uint32_t addr = tu_agen_next(&it);
        ASSERT_EQ((long)addr, (long)(0x1000 + i * 4), "correct linear address");
    }
    ASSERT_EQ((long)tu_agen_next(&it), (long)UINT32_MAX, "should return UINT32_MAX at end");
    ASSERT_TRUE(!tu_agen_has_next(&it), "should have no more elements");

    /* Reset */
    tu_agen_reset(&it);
    ASSERT_TRUE(tu_agen_has_next(&it), "should have elements after reset");
    PASS();
}

/* ---- Test 2: Strided 2D addressing ---- */
static void test_strided_2d(void) {
    TEST("strided 2D addressing");
    tu_agen_iterator_t it;
    tu_agen_2d_config_t cfg = { 4, 8, 16 }; /* 4 rows, 8 cols, row_stride=16 */
    int rc = tu_agen_iterator_init(&it, TU_AGEN_MODE_STRIDED_2D, 0x100, &cfg);
    ASSERT_EQ(rc, 0, "init succeeded");
    ASSERT_EQ((long)tu_agen_total(&it), 32L, "4×8=32 elements");

    /* First element: row=0, col=0 → base + 0*16+0 = 0x100 */
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x100, "first element");

    /* Element at row=0, col=7 → base + (0*16+7)*4 = 0x100+28 = 0x11C */
    for (int c = 0; c < 6; c++) tu_agen_next(&it); /* skip to col 7 */
    ASSERT_EQ((long)tu_agen_next(&it), (long)(0x100 + 7*4), "row 0, col 7");

    /* Element at row=1, col=0 → base + (1*16+0)*4 = 0x100+64 = 0x140 */
    ASSERT_EQ((long)tu_agen_next(&it), (long)(0x100 + 16*4), "row 1, col 0");

    PASS();
}

/* ---- Test 3: Strided 3D addressing ---- */
static void test_strided_3d(void) {
    TEST("strided 3D addressing");
    tu_agen_iterator_t it;
    /* depth=2, rows=3, cols=4, depth_stride=20, row_stride=8, elem_size=4 */
    uint32_t config[6] = { 2, 3, 4, 20, 8, 4 };
    int rc = tu_agen_iterator_init(&it, TU_AGEN_MODE_STRIDED_3D, 0x200, config);
    ASSERT_EQ(rc, 0, "init succeeded");
    ASSERT_EQ((long)tu_agen_total(&it), 24L, "2×3×4=24 elements");

    /* First: depth=0, row=0, col=0 → 0x200 + (0*20+0*8+0)*4 = 0x200 */
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x200, "depth=0 row=0 col=0");

    /* depth=0, row=1, col=0 → 0x200 + (0*20+1*8+0)*4 = 0x200+32 = 0x220 */
    /* Skip remaining cols in row 0: cols 1-3 = 3 elements */
    for (int c = 0; c < 3; c++) tu_agen_next(&it);
    ASSERT_EQ((long)tu_agen_next(&it), (long)(0x200 + 8*4), "depth=0 row=1 col=0");

    /* depth=1, row=0, col=0 → 0x200 + (1*20+0*8+0)*4 = 0x200+80 = 0x250 */
    /* Skip remaining: row 1 cols 1-3 (3) + row 2 cols 0-3 (4) = 7 elements */
    for (int c = 0; c < 7; c++) tu_agen_next(&it);
    ASSERT_EQ((long)tu_agen_next(&it), (long)(0x200 + 20*4), "depth=1 row=0 col=0");

    PASS();
}

/* ---- Test 4: Tiled 2D addressing ---- */
static void test_tiled_2d(void) {
    TEST("tiled 2D addressing");
    tu_agen_iterator_t it;
    /* 6×6 matrix, tiles of 2×3, base=0 */
    tu_agen_tiled2d_config_t cfg = { 6, 6, 2, 3, 3, 2 }; /* total: 6×6, tile: 2×3, 3×2 grid */
    int rc = tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_2D, 0x0, &cfg);
    ASSERT_EQ(rc, 0, "init succeeded");
    ASSERT_EQ((long)tu_agen_total(&it), 36L, "6×6=36 elements");

    /* First tile (0,0): rows 0-1, cols 0-2. First element: (0,0) → 0 */
    ASSERT_EQ((long)tu_agen_next(&it), 0L, "tile(0,0) elem(0,0)");

    /* Skip to tile(0,0) elem(1,2): row 1, col 2 → 1*6+2 = 8, addr=8*4=32 */
    /* Elements in tile(0,0): (0,0)(0,1)(0,2)(1,0)(1,1)(1,2) = 6 elements */
    for (int c = 0; c < 4; c++) tu_agen_next(&it); /* skip 4 */
    ASSERT_EQ((long)tu_agen_next(&it), (long)((1*6+2)*4), "tile(0,0) row 1, col 2");

    PASS();
}

/* ---- Test 5: im2col addressing ---- */
static void test_im2col(void) {
    TEST("im2col addressing");
    tu_agen_iterator_t it;
    tu_agen_im2col_t cfg = {
        .input_h = 4, .input_w = 4, .input_c = 1,
        .kernel_h = 2, .kernel_w = 2,
        .pad_h = 0, .pad_w = 0,
        .stride_h = 1, .stride_w = 1,
        .dilation_h = 1, .dilation_w = 1,
        .elem_size = 4
    };
    /* output: (4-2)/1+1 = 3×3, total elements = 3*3*2*2*1 = 36 */

    int rc = tu_agen_iterator_init(&it, TU_AGEN_MODE_IM2COL, 0x0, &cfg);
    ASSERT_EQ(rc, 0, "init succeeded");
    ASSERT_EQ((long)tu_agen_total(&it), 36L, "3×3×2×2=36 elements");

    /* First: oh=0, ow=0, kh=0, kw=0, ic=0 → h_in=0, w_in=0 → addr=0 */
    ASSERT_EQ((long)tu_agen_next(&it), 0L, "oh=0 ow=0 kh=0 kw=0");

    /* oh=0, ow=0, kh=0, kw=1, ic=0 → h_in=0, w_in=1 → addr=4 */
    ASSERT_EQ((long)tu_agen_next(&it), (long)4, "oh=0 ow=0 kh=0 kw=1");

    /* oh=0, ow=0, kh=1, kw=0, ic=0 → h_in=1, w_in=0 → addr=1*4*4+0 = 16 */
    /* Need to skip: (kh=0,kw=0,ic=0) already done. Next in ic is still 0.
       Pattern: for oh=0,ow=0: ic=0 then iterate (kh,kw): (0,0)(0,1)(1,0)(1,1) */
    /* We already consumed (0,0) and (0,1). Next is (1,0) */
    ASSERT_EQ((long)tu_agen_next(&it), (long)(16), "oh=0 ow=0 kh=1 kw=0");

    PASS();
}

/* ---- Test 6: Block interleaved addressing ---- */
static void test_block_interleaved(void) {
    TEST("block interleaved addressing");
    tu_agen_iterator_t it;
    /* 8 elements, 4 banks, block_width=2, elem_size=4 */
    uint32_t config[4] = { 8, 4, 2, 4 };
    int rc = tu_agen_iterator_init(&it, TU_AGEN_MODE_BLOCK_INTERLEAVED, 0x100, config);
    ASSERT_EQ(rc, 0, "init succeeded");
    ASSERT_EQ((long)tu_agen_total(&it), 8L, "8 elements");

    /* Block 0: banks 0,1 → (0*2+0)*4=0, (0*2+1)*4=4 */
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x100, "elem 0 bank 0");
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x104, "elem 0 bank 1");

    /* Block 1: banks 0,1 → (1*2+0)*4=8, (1*2+1)*4=12 */
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x108, "elem 1 bank 0");
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x10C, "elem 1 bank 1");

    PASS();
}

/* ---- Test 7: Transposed addressing ---- */
static void test_transposed(void) {
    TEST("transposed (column-major) addressing");
    tu_agen_iterator_t it;
    uint32_t config[3] = { 3, 4, 4 }; /* 3 rows, 4 cols, elem_size=4 */
    int rc = tu_agen_iterator_init(&it, TU_AGEN_MODE_TRANSPOSED, 0x100, config);
    ASSERT_EQ(rc, 0, "init succeeded");
    ASSERT_EQ((long)tu_agen_total(&it), 12L, "3×4=12 elements");

    /* Column-major order over row-major storage:
     * (0,0) → base+0*4 = 0x100 ✓
     * (1,0) → base+(1*4+0)*4 = base+16 = 0x110 ✓
     * (2,0) → base+(2*4+0)*4 = base+32 = 0x120 ✓
     * (0,1) → base+(0*4+1)*4 = base+4 = 0x104 ✓
     */
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x100, "col 0, row 0");
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x110, "col 0, row 1");
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x120, "col 0, row 2");
    ASSERT_EQ((long)tu_agen_next(&it), (long)0x104, "col 1, row 0");

    PASS();
}

/* ---- Test 8: Bulk address generation ---- */
static void test_bulk_generation(void) {
    TEST("bulk address generation");
    tu_agen_iterator_t it;
    uint32_t config[2] = { 8, 4 };
    tu_agen_iterator_init(&it, TU_AGEN_MODE_LINEAR, 0x0, config);

    uint32_t addrs[8];
    uint32_t count = tu_agen_generate_all(&it, addrs, 8);
    ASSERT_EQ((long)count, 8L, "generated 8 addresses");
    for (uint32_t i = 0; i < 8; i++) {
        ASSERT_EQ((long)addrs[i], (long)(i * 4), "correct bulk address");
    }
    PASS();
}

/* ---- Test 9: Range generation ---- */
static void test_range_generation(void) {
    TEST("contiguous range generation");
    tu_agen_iterator_t it;
    uint32_t config[2] = { 10, 4 };
    tu_agen_iterator_init(&it, TU_AGEN_MODE_LINEAR, 0x100, config);

    tu_agen_range_t ranges[16];
    uint32_t n = tu_agen_generate_ranges(&it, ranges, 16);
    ASSERT_EQ((long)n, 1L, "linear pattern → 1 contiguous range");
    ASSERT_EQ((long)ranges[0].base_addr, (long)0x100, "range start");
    ASSERT_EQ((long)ranges[0].total_bytes, 40L, "10×4=40 bytes");

    PASS();
}

/* ---- Test 10: im2col dimension computation ---- */
static void test_im2col_dims(void) {
    TEST("im2col dimension computation");
    tu_agen_im2col_t im2col = {
        .input_h = 28, .input_w = 28, .input_c = 3,
        .kernel_h = 3, .kernel_w = 3,
        .pad_h = 1, .pad_w = 1,
        .stride_h = 2, .stride_w = 2,
        .dilation_h = 1, .dilation_w = 1,
        .elem_size = 4
    };
    tu_agen_im2col_dims(&im2col);
    /* output = (28 + 2*1 - 3)/2 + 1 = (27)/2 + 1 = 14 */
    ASSERT_EQ((long)im2col.output_h, 14L, "output height = 14");
    ASSERT_EQ((long)im2col.output_w, 14L, "output width = 14");

    PASS();
}

/* ---- Test 11: Tiling computation ---- */
static void test_tiling_computation(void) {
    TEST("tiling computation");
    tu_agen_tiling_t tiling = {
        .tile_dims = { 1, 16, 16 },
        .total_dims = { 1, 64, 64 },
        .elem_size = 4
    };
    tu_agen_compute_tiling(&tiling);
    ASSERT_EQ((long)tiling.tiles_per_dim[0], 1L, "1 depth tile");
    ASSERT_EQ((long)tiling.tiles_per_dim[1], 4L, "4 row tiles (64/16)");
    ASSERT_EQ((long)tiling.tiles_per_dim[2], 4L, "4 col tiles (64/16)");
    ASSERT_EQ((long)tiling.num_tiles, 16L, "1×4×4=16 tiles");

    PASS();
}

/* ---- Test 12: Inline address helpers ---- */
static void test_inline_helpers(void) {
    TEST("inline address helpers");
    /* tu_agen_addr_2d */
    ASSERT_EQ((long)tu_agen_addr_2d(0, 10, 2, 3, 4), (long)((2*10+3)*4), "addr_2d");
    /* tu_agen_addr_3d */
    ASSERT_EQ((long)tu_agen_addr_3d(0, 30, 10, 1, 2, 3, 4),
              (long)((1*30+2*10+3)*4), "addr_3d");
    /* tu_agen_addr_banked */
    ASSERT_EQ((long)tu_agen_addr_banked(0x100, 4, 2, 5, 4),
              (long)(0x100+(5*4+2)*4), "addr_banked");
    /* tu_agen_addr_tile_start */
    ASSERT_EQ((long)tu_agen_addr_tile_start(0, 8, 1, 2, 3, 4, 4),
              (long)((1*3*8+2*4)*4), "tile_start");

    PASS();
}

/* ---- Test 13: im2col single address ---- */
static void test_im2col_addr(void) {
    TEST("im2col single address computation");
    tu_agen_im2col_t cfg = {
        .input_h = 5, .input_w = 5, .input_c = 1,
        .kernel_h = 3, .kernel_w = 3,
        .pad_h = 0, .pad_w = 0,
        .stride_h = 1, .stride_w = 1,
        .dilation_h = 1, .dilation_w = 1,
        .elem_size = 4
    };

    /* oh=1, ow=1, kh=1, kw=1 → h_in=1*1+1*1-0=2, w_in=1*1+1*1-0=2
     * addr = 0 + (0*25 + 2*5 + 2)*4 = 12*4 = 48 */
    uint32_t addr = tu_agen_addr_im2col(&cfg, 0, 1, 1, 1, 1, 0);
    ASSERT_EQ((long)addr, (long)((2*5+2)*4), "im2col single addr");

    PASS();
}

/* ---- Main ---- */

int main(void) {
    printf("\n===========================================\n");
    printf("  Address Generator Tests (Gap M3)\n");
    printf("===========================================\n\n");

    test_linear();
    test_strided_2d();
    test_strided_3d();
    test_tiled_2d();
    test_im2col();
    test_block_interleaved();
    test_transposed();
    test_bulk_generation();
    test_range_generation();
    test_im2col_dims();
    test_tiling_computation();
    test_inline_helpers();
    test_im2col_addr();

    printf("\n-------------------------------------------\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(", %d FAILED\n", tests_failed);
        printf("-------------------------------------------\n");
        return 1;
    }
    printf("\n-------------------------------------------\n");
    return 0;
}
