/*
 * Test: Pooling Engine (O6)
 * ==========================
 * Validates MaxPool2D, AvgPool2D, dimension computation,
 * stride, padding, and edge cases.
 *
 * Gap: O6 — Pooling operations (MaxPool, AvgPool)
 */

#include "test_framework.h"
#include "tu_cmodel/compute/pooling_engine.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

tu_test_stats_t g_test_stats;

#define CHECK(cond, msg) do { \
    if (!(cond)) { FAIL("%s", msg); return; } \
} while(0)

/* ---- Test: Dimension Computation ---- */
static void test_pool_dims(void) {
    tu_pool_desc_t desc = {0};
    desc.batch = 1; desc.channels = 1;  /* required by validation */
    desc.ih = 32; desc.iw = 32;
    desc.kh = 2;  desc.kw = 2;
    desc.sh = 2;  desc.sw = 2;
    desc.ph = 0;  desc.pw = 0;

    CHECK(tu_pool_compute_dims(&desc) == 0, "Dim computation should succeed");
    CHECK(desc.oh == 16, "OH should be (32-2)/2+1 = 16");
    CHECK(desc.ow == 16, "OW should be (32-2)/2+1 = 16");

    /* With padding */
    desc.ih = 32; desc.iw = 32;
    desc.kh = 3; desc.kw = 3;
    desc.sh = 2; desc.sw = 2;
    desc.ph = 1; desc.pw = 1;
    CHECK(tu_pool_compute_dims(&desc) == 0, "Dim comp with padding");
    CHECK(desc.oh == 16, "OH with padding: (32+2-3)/2+1 = 16");
    CHECK(desc.ow == 16, "OW with padding: (32+2-3)/2+1 = 16");

    /* Edge: kernel > input + padding */
    desc.ih = 2; desc.iw = 2;
    desc.kh = 5; desc.kw = 5;
    desc.sh = 1; desc.sw = 1;
    desc.ph = 0; desc.pw = 0;
    CHECK(tu_pool_compute_dims(&desc) != 0, "Should fail when kernel > input + padding");
    PASS();
}

/* ---- Test: MaxPool2D basic ---- */
static void test_maxpool_basic(void) {
    /* 4x4 input, 2x2 kernel, stride 2, no padding → 2x2 output */
    float src[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10,11,12,
        13,14,15,16
    };
    float dst[4] = {0};

    tu_pool_max_2d(src, dst, 4, 4, 2, 2, 2, 2, 2, 2, 0, 0, -INFINITY);

    CHECK(fabsf(dst[0] - 6.0f) < 1e-6f, "MaxPool[0,0] = max(1,2,5,6) = 6");
    CHECK(fabsf(dst[1] - 8.0f) < 1e-6f, "MaxPool[0,1] = max(3,4,7,8) = 8");
    CHECK(fabsf(dst[2] - 14.0f) < 1e-6f, "MaxPool[1,0] = max(9,10,13,14) = 14");
    CHECK(fabsf(dst[3] - 16.0f) < 1e-6f, "MaxPool[1,1] = max(11,12,15,16) = 16");
    PASS();
}

/* ---- Test: MaxPool2D with stride 1 ---- */
static void test_maxpool_stride1(void) {
    /* 3x3 input, 2x2 kernel, stride 1, no padding → 2x2 output */
    float src[9] = {
        1, 3, 2,
        4, 6, 5,
        8, 9, 7
    };
    float dst[4] = {0};

    tu_pool_max_2d(src, dst, 3, 3, 2, 2, 2, 2, 1, 1, 0, 0, -INFINITY);

    CHECK(fabsf(dst[0] - 6.0f) < 1e-6f, "MaxPool s1 [0,0] = max(1,3,4,6) = 6");
    CHECK(fabsf(dst[1] - 6.0f) < 1e-6f, "MaxPool s1 [0,1] = max(3,2,6,5) = 6");
    CHECK(fabsf(dst[2] - 9.0f) < 1e-6f, "MaxPool s1 [1,0] = max(4,6,8,9) = 9");
    CHECK(fabsf(dst[3] - 9.0f) < 1e-6f, "MaxPool s1 [1,1] = max(6,5,9,7) = 9");
    PASS();
}

/* ---- Test: MaxPool with negative values ---- */
static void test_maxpool_negative(void) {
    float src[16] = {
        -1, -2, -3, -4,
        -5, -6, -7, -8,
        -9,-10,-11,-12,
       -13,-14,-15,-16
    };
    float dst[4] = {0};

    tu_pool_max_2d(src, dst, 4, 4, 2, 2, 2, 2, 2, 2, 0, 0, -INFINITY);

    CHECK(fabsf(dst[0] - (-1.0f)) < 1e-6f, "Negative max pool [0,0] = -1");
    CHECK(fabsf(dst[3] - (-11.0f)) < 1e-6f, "Negative max pool [1,1] = -11");
    PASS();
}

/* ---- Test: AvgPool2D basic ---- */
static void test_avgpool_basic(void) {
    /* 4x4 input, 2x2 kernel, stride 2 → 2x2 output */
    float src[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10,11,12,
        13,14,15,16
    };
    float dst[4] = {0};

    tu_pool_avg_2d(src, dst, 4, 4, 2, 2, 2, 2, 2, 2, 0, 0, true);

    CHECK(fabsf(dst[0] - 3.5f) < 1e-6f, "AvgPool[0,0] = mean(1,2,5,6) = 3.5");
    CHECK(fabsf(dst[1] - 5.5f) < 1e-6f, "AvgPool[0,1] = mean(3,4,7,8) = 5.5");
    CHECK(fabsf(dst[2] - 11.5f) < 1e-6f, "AvgPool[1,0] = mean(9,10,13,14) = 11.5");
    CHECK(fabsf(dst[3] - 13.5f) < 1e-6f, "AvgPool[1,1] = mean(11,12,15,16) = 13.5");
    PASS();
}

/* ---- Test: AvgPool with count_exclude_pad ---- */
static void test_avgpool_exclude_pad(void) {
    /* 3x3 → 2x2 with stride 2, kernel 2 → edge windows have 4, 2, 2, 1 elements */
    float src[9] = {1,2,3, 4,5,6, 7,8,9};
    float dst[4] = {0};

    tu_pool_avg_2d(src, dst, 3, 3, 2, 2, 2, 2, 2, 2, 0, 0, false);

    /* Window [0,0]: elements 1,2,4,5 / 4 = 3 */
    CHECK(fabsf(dst[0] - 3.0f) < 1e-6f, "AvgPool exclude pad [0,0] = 3");
    /* Window [0,1]: elements 3,6 / 2 = 4.5 */
    CHECK(fabsf(dst[1] - 4.5f) < 1e-6f, "AvgPool exclude pad [0,1] = 4.5");
    /* Window [1,0]: elements 7,8 /2 = 7.5 */
    CHECK(fabsf(dst[2] - 7.5f) < 1e-6f, "AvgPool exclude pad [1,0] = 7.5");
    /* Window [1,1]: element 9/1 = 9 */
    CHECK(fabsf(dst[3] - 9.0f) < 1e-6f, "AvgPool exclude pad [1,1] = 9");
    PASS();
}

/* ---- Test: MaxPool with padding ---- */
static void test_maxpool_with_padding(void) {
    /* 2x2 input, 3x3 kernel, stride 1, padding 1 → 2x2 output */
    float src[4] = {1, 2, 3, 4};
    float dst[4] = {0};

    tu_pool_max_2d(src, dst, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, -INFINITY);

    /* Window centered on (0,0): covers pad, pad, pad, 1,2, pad,3,4, pad */
    /* Valid elements: 1,2,3,4 — max is 4 */
    CHECK(fabsf(dst[0] - 4.0f) < 1e-6f, "MaxPool w/ pad [0,0] = max(1,2,3,4) = 4");
    /* All windows should be 4 since everything is covered by the full input */
    CHECK(fabsf(dst[3] - 4.0f) < 1e-6f, "MaxPool w/ pad [1,1] = 4");
    PASS();
}

/* ---- Test: AvgPool with padding (include and exclude) ---- */
static void test_avgpool_padding(void) {
    /* 2x2 input, 3x3 kernel, stride 1, padding 1 → 2x2 output */
    float src[4] = {1, 2, 3, 4};
    float dst_include[4] = {0};
    float dst_exclude[4] = {0};

    tu_pool_avg_2d(src, dst_include, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, true);
    tu_pool_avg_2d(src, dst_exclude, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, false);

    /* Include pad: always / 9 */
    /* [0,0] covers: 1,2,3,4 (4 values from input, 5 pad zeros) = 10/9 = 1.111... */
    CHECK(fabsf(dst_include[0] - 10.0f/9.0f) < 1e-5f, "AvgPool inc pad [0,0] = 10/9");

    /* Exclude pad: divide by actual count */
    /* [0,0] covers 4 elements: (1+2+3+4)/4 = 2.5 */
    CHECK(fabsf(dst_exclude[0] - 2.5f) < 1e-6f, "AvgPool exc pad [0,0] = 2.5");

    /* With 2x2 input and 3x3 kernel, all windows cover all 4 input elements */
    CHECK(fabsf(dst_exclude[1] - 2.5f) < 1e-6f, "AvgPool exc pad [0,1] = 2.5");
    CHECK(fabsf(dst_exclude[2] - 2.5f) < 1e-6f, "AvgPool exc pad [1,0] = 2.5");
    CHECK(fabsf(dst_exclude[3] - 2.5f) < 1e-6f, "AvgPool exc pad [1,1] = 2.5");
    PASS();
}

/* ---- Test: All-equal input ---- */
static void test_pool_all_equal(void) {
    float src[9] = {5,5,5, 5,5,5, 5,5,5};
    float dst_max[4], dst_avg[4];

    tu_pool_max_2d(src, dst_max, 3, 3, 2, 2, 2, 2, 1, 1, 0, 0, -INFINITY);
    tu_pool_avg_2d(src, dst_avg, 3, 3, 2, 2, 2, 2, 1, 1, 0, 0, true);

    /* All should be 5 */
    CHECK(fabsf(dst_max[0] - 5.0f) < 1e-6f, "All-equal max = 5");
    CHECK(fabsf(dst_avg[0] - 5.0f) < 1e-6f, "All-equal avg = 5");
    PASS();
}

/* ---- Test: Single element input ---- */
static void test_pool_single_element(void) {
    float src[1] = {42};
    float dst[1] = {0};

    tu_pool_max_2d(src, dst, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -INFINITY);
    CHECK(fabsf(dst[0] - 42.0f) < 1e-6f, "1x1 max pool = 42");

    tu_pool_avg_2d(src, dst, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, true);
    CHECK(fabsf(dst[0] - 42.0f) < 1e-6f, "1x1 avg pool = 42");
    PASS();
}

/* ---- Test: Non-square kernel (KH≠KW) ---- */
static void test_pool_rectangular_kernel(void) {
    /* 4x4 input, 2x3 kernel, stride 2 → 2x1 output */
    float src[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10,11,12,
        13,14,15,16
    };
    float dst[2] = {0};

    tu_pool_max_2d(src, dst, 4, 4, 2, 1, 2, 3, 2, 2, 0, 0, -INFINITY);

    /* [0,0]: max of rows 0-1, cols 0-2 = max(1,2,3,5,6,7) = 7 */
    CHECK(fabsf(dst[0] - 7.0f) < 1e-6f, "2x3 kernel [0,0] = 7");
    /* [1,0]: max of rows 2-3, cols 0-2 = max(9,10,11,13,14,15) = 15 */
    CHECK(fabsf(dst[1] - 15.0f) < 1e-6f, "2x3 kernel [1,0] = 15");
    PASS();
}

/* ---- Test: Full execution with SRAM ---- */
static void test_pool_full_execution(void) {
    tu_sram_region_t src_region, dst_region;
    tu_sram_init(&src_region, 1024, "pool_src");
    tu_sram_init(&dst_region, 1024, "pool_dst");

    /* Write 4x4 float data to SRAM */
    float input[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10,11,12,
        13,14,15,16
    };
    memcpy(tu_sram_raw_ptr(&src_region), input, 64);

    tu_pool_desc_t desc = {0};
    desc.pool_type  = TU_POOL_MAX;
    desc.batch      = 1;
    desc.channels   = 1;
    desc.ih = 4; desc.iw = 4;
    desc.kh = 2; desc.kw = 2;
    desc.sh = 2; desc.sw = 2;
    desc.ph = 0; desc.pw = 0;
    desc.elem_size  = 4;
    desc.is_float   = true;
    desc.src_region = &src_region;
    desc.src_offset = 0;
    desc.dst_region = &dst_region;
    desc.dst_offset = 0;

    int64_t cycles = tu_pool_execute(&desc);
    CHECK(cycles > 0, "Execution should return positive cycles");

    float *output = (float *)tu_sram_raw_ptr(&dst_region);
    CHECK(fabsf(output[0] - 6.0f) < 1e-6f, "Full exec [0,0] = 6");
    CHECK(fabsf(output[1] - 8.0f) < 1e-6f, "Full exec [0,1] = 8");
    CHECK(fabsf(output[2] - 14.0f) < 1e-6f, "Full exec [1,0] = 14");
    CHECK(fabsf(output[3] - 16.0f) < 1e-6f, "Full exec [1,1] = 16");

    tu_sram_destroy(&src_region);
    tu_sram_destroy(&dst_region);
    PASS();
}

/* ---- Test: Multi-channel pooling ---- */
static void test_pool_multichannel(void) {
    tu_sram_region_t src_region, dst_region;
    tu_sram_init(&src_region, 1024, "pool_src_mc");
    tu_sram_init(&dst_region, 1024, "pool_dst_mc");

    /* 2 channels, 4x4 each */
    float input[32] = {
        1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,     /* ch0 */
        10,20,30,40, 50,60,70,80, 90,100,110,120, 130,140,150,160  /* ch1 */
    };
    memcpy(tu_sram_raw_ptr(&src_region), input, 128);

    tu_pool_desc_t desc = {0};
    desc.pool_type  = TU_POOL_MAX;
    desc.batch      = 1;
    desc.channels   = 2;
    desc.ih = 4; desc.iw = 4;
    desc.kh = 2; desc.kw = 2;
    desc.sh = 2; desc.sw = 2;
    desc.ph = 0; desc.pw = 0;
    desc.elem_size  = 4;
    desc.is_float   = true;
    desc.src_region = &src_region;
    desc.src_offset = 0;
    desc.dst_region = &dst_region;
    desc.dst_offset = 0;

    int64_t cycles = tu_pool_execute(&desc);
    CHECK(cycles > 0, "Multi-channel execution should return positive cycles");

    float *output = (float *)tu_sram_raw_ptr(&dst_region);
    /* Channel 0: same as basic test */
    CHECK(fabsf(output[0] - 6.0f) < 1e-6f, "MC ch0 [0,0] = 6");
    CHECK(fabsf(output[3] - 16.0f) < 1e-6f, "MC ch0 [1,1] = 16");
    /* Channel 1: values are 10x channel 0 */
    CHECK(fabsf(output[4] - 60.0f) < 1e-6f, "MC ch1 [0,0] = 60");
    CHECK(fabsf(output[7] - 160.0f) < 1e-6f, "MC ch1 [1,1] = 160");

    tu_sram_destroy(&src_region);
    tu_sram_destroy(&dst_region);
    PASS();
}

/* ---- Test: Validation ---- */
static void test_pool_validation(void) {
    tu_sram_region_t src, dst;
    tu_sram_init(&src, 256, "vsrc");
    tu_sram_init(&dst, 256, "vdst");

    tu_pool_desc_t desc = {0};
    desc.pool_type  = TU_POOL_MAX;
    desc.batch = 1; desc.channels = 1;
    desc.ih = 4; desc.iw = 4;
    desc.kh = 2; desc.kw = 2;
    desc.sh = 2; desc.sw = 2;
    desc.elem_size = 4;
    desc.is_float = true;
    desc.src_region = &src;
    desc.src_offset = 0;
    desc.dst_region = &dst;
    desc.dst_offset = 0;
    tu_pool_compute_dims(&desc);

    CHECK(tu_pool_validate(&desc) == 0, "Valid desc should pass");

    /* Invalid: offset overflow */
    desc.src_offset = 300;
    CHECK(tu_pool_validate(&desc) != 0, "Overflow src_offset should fail");
    desc.src_offset = 0;

    /* Invalid: null region */
    desc.dst_region = NULL;
    CHECK(tu_pool_validate(&desc) != 0, "Null dst_region should fail");

    tu_sram_destroy(&src);
    tu_sram_destroy(&dst);
    PASS();
}

/* ---- Test Runner ---- */
int main(void) {
    test_stats_init();

    TEST("pool_dim_computation");     test_pool_dims();
    TEST("maxpool_basic");            test_maxpool_basic();
    TEST("maxpool_stride1");          test_maxpool_stride1();
    TEST("maxpool_negative");         test_maxpool_negative();
    TEST("avgpool_basic");            test_avgpool_basic();
    TEST("avgpool_exclude_pad");      test_avgpool_exclude_pad();
    TEST("maxpool_with_padding");     test_maxpool_with_padding();
    TEST("avgpool_padding");          test_avgpool_padding();
    TEST("pool_all_equal");           test_pool_all_equal();
    TEST("pool_single_element");      test_pool_single_element();
    TEST("pool_rectangular_kernel");  test_pool_rectangular_kernel();
    TEST("pool_full_execution");      test_pool_full_execution();
    TEST("pool_multichannel");        test_pool_multichannel();
    TEST("pool_validation");          test_pool_validation();

    return test_exit();
}
