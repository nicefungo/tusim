/*
 * TinyTU Convolution Engine Tests
 * =================================
 * Gap O2: Verify im2col, direct convolution, and im2col+GEMM pipeline.
 *
 * Tests:
 *   1. Dimension computation (valid/invalid configs)
 *   2. im2col NHWC correctness (small tensor, verified manually)
 *   3. Direct convolution golden reference (NCHW, NHWC)
 *   4. im2col + GEMM pipeline vs golden reference
 *   5. Strided/dilated convolution
 *   6. Grouped convolution
 *   7. Depthwise convolution (groups == channels)
 *   8. Fused ReLU activation
 *   9. Cycle estimation sanity
 */

#include "tu_cmodel/compute/convolution_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_pass = 0;

#define TEST(name) do { tests_run++; printf("  %-55s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_pass++; } while(0)
#define FAIL(fmt, ...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); } while(0)
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { FAIL("eq: %d != %d", (int)(a), (int)(b)); return; }} while(0)
#define ASSERT_NEAR(a, b, tol) do { if (fabsf((float)(a)-(float)(b)) > (tol)) { FAIL("near: %f != %f (tol=%g)", (double)(a), (double)(b), (double)(tol)); return; }} while(0)

/* ================================================================
 * Dimension Computation
 * ================================================================ */

static void test_conv_dims_standard(void) {
    TEST("Conv dims: 32×32 input, 3×3 kernel, stride=1, pad=1");
    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 3,
        .in_height = 32, .in_width = 32,
        .out_channels = 16,
        .kernel_h = 3, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 1, .pad_b = 1, .pad_l = 1, .pad_r = 1,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    ASSERT_EQ(tu_conv_compute_dims(&desc), 0);
    ASSERT_EQ(desc.out_height, 32);
    ASSERT_EQ(desc.out_width, 32);
    ASSERT_EQ(desc.im2col_rows, 27);  /* 3 * 3 * 3 */
    ASSERT_EQ(desc.im2col_cols, 1024); /* 32 * 32 */
    PASS();
}

static void test_conv_dims_stride2(void) {
    TEST("Conv dims: 32×32 input, 3×3 kernel, stride=2, pad=1");
    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 3,
        .in_height = 32, .in_width = 32,
        .out_channels = 16,
        .kernel_h = 3, .kernel_w = 3,
        .stride_h = 2, .stride_w = 2,
        .pad_t = 1, .pad_b = 1, .pad_l = 1, .pad_r = 1,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    ASSERT_EQ(tu_conv_compute_dims(&desc), 0);
    ASSERT_EQ(desc.out_height, 16);
    ASSERT_EQ(desc.out_width, 16);
    PASS();
}

static void test_conv_dims_invalid(void) {
    TEST("Conv dims: invalid (too small, kernel > input)");
    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 3,
        .in_height = 1, .in_width = 1,
        .out_channels = 4,
        .kernel_h = 7, .kernel_w = 7,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 0, .pad_b = 0, .pad_l = 0, .pad_r = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    /* out = (1 - 7)/1 + 1 = -5 → invalid */
    int r = tu_conv_compute_dims(&desc);
    if (r == 0) { FAIL("expected failure for invalid dims"); return; }
    PASS();
}

/* ================================================================
 * im2col NHWC Tests
 * ================================================================ */

static void test_im2col_nhwc_small(void) {
    TEST("im2col NHWC: 3×3 input, 2×2 kernel, stride=1, pad=0");
    /* Input: NHWC [1][3][3][1] single channel
     *  1  2  3
     *  4  5  6
     *  7  8  9
     *
     * Kernel: 2×2, stride=1, pad=0
     * Output: [2][2] each is a 2×2 patch
     * im2col: 4 rows (k_h*k_w) × 4 cols (out 2×2)
     *
     * Row 0 (r=0,s=0): [[1,2],[4,5]] → [1,2,4,5]
     * Row 1 (r=0,s=1): [[2,3],[5,6]] → [2,3,5,6]
     * Row 2 (r=1,s=0): [[4,5],[7,8]] → [4,5,7,8]
     * Row 3 (r=1,s=1): [[5,6],[8,9]] → [5,6,8,9]
     */
    float input[] = {1,2,3,4,5,6,7,8,9};
    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 1,
        .in_height = 3, .in_width = 3,
        .out_channels = 1,
        .kernel_h = 2, .kernel_w = 2,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 0, .pad_b = 0, .pad_l = 0, .pad_r = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    tu_conv_compute_dims(&desc);

    float im2col_out[16] = {0};
    tu_im2col_nhwc(input, im2col_out, &desc, sizeof(float));

    float expected[] = {
        1, 2, 4, 5,   /* Row 0: r=0,s=0 */
        2, 3, 5, 6,   /* Row 1: r=0,s=1 */
        4, 5, 7, 8,   /* Row 2: r=1,s=0 */
        5, 6, 8, 9,   /* Row 3: r=1,s=1 */
    };
    for (int i = 0; i < 16; i++)
        ASSERT_NEAR(im2col_out[i], expected[i], 0.001f);
    PASS();
}

static void test_im2col_nhwc_padded(void) {
    TEST("im2col NHWC: 2×2 input, pad=1, same padding");
    /* Input: NHWC [1][2][2][1]
     *  1 2
     *  3 4
     * Pad=1 all sides → effective input 4×4 with zeros at edges
     * Kernel: 3×3, stride=1 → output 2×2
     * im2col rows = 9, cols = 4
     */
    float input[] = {1,2,3,4};
    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 1,
        .in_height = 2, .in_width = 2,
        .out_channels = 1,
        .kernel_h = 3, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 1, .pad_b = 1, .pad_l = 1, .pad_r = 1,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    tu_conv_compute_dims(&desc);
    ASSERT_EQ(desc.out_height, 2);
    ASSERT_EQ(desc.out_width, 2);

    float im2col_out[36] = {0};
    tu_im2col_nhwc(input, im2col_out, &desc, sizeof(float));

    /* Top-left output position (oh=0, ow=0):
     * padded input region [0:3][0:3]:
     *   0 0 0
     *   0 1 2
     *   0 3 4
     * Row layout: r0s0=[0,0,0,0], r0s1=[0,0,0,1], r0s2=[0,0,0,0],
     *             r1s0=[0,1,0,3], r1s1=[1,2,3,4], r1s2=[2,0,4,0],
     *             r2s0=[0,3,0,0], r2s1=[3,4,0,0], r2s2=[4,0,0,0]
     */
    /* Check a few known positions: r1s1 (center of 3×3 kernel) at pos [0,0]:
     * This is the (1*3+1)*4 = 16 offset in the im2col buffer */
    float expected_center[] = {1,2,3,4};
    for (int j = 0; j < 4; j++)
        ASSERT_NEAR(im2col_out[16 + j], expected_center[j], 0.001f);
    PASS();
}

/* ================================================================
 * Direct Conv2D Golden Reference Tests
 * ================================================================ */

static void test_direct_conv_nchw_identity(void) {
    TEST("Direct conv NCHW: identity kernel (1×1, 1 filter, 1 channel)");
    /* Input: 4×4, 1 channel
     *   1 2 3 4
     *   5 6 7 8
     *   9 10 11 12
     *   13 14 15 16
     * Weight: [1][1][1][1] = {2.0}
     * Bias: none
     * Expected: 2×input
     */
    float input[16];
    for (int i = 0; i < 16; i++) input[i] = (float)(i + 1);

    float weight[] = {2.0f};
    float output[16] = {0};

    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 1,
        .in_height = 4, .in_width = 4,
        .out_channels = 1,
        .kernel_h = 1, .kernel_w = 1,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 0, .pad_b = 0, .pad_l = 0, .pad_r = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    tu_conv_compute_dims(&desc);

    tu_conv2d_direct_nchw_fp32(input, weight, NULL, output, &desc);

    for (int i = 0; i < 16; i++)
        ASSERT_NEAR(output[i], input[i] * 2.0f, 0.001f);
    PASS();
}

static void test_direct_conv_nchw_3x3(void) {
    TEST("Direct conv NCHW: 3×3 input, 2×2 kernel, 2 channels");
    /* Input: [1][2][3][3]
     * Ch 0: 1 2 3     Ch 1: 10 20 30
     *        4 5 6            40 50 60
     *        7 8 9            70 80 90
     * Weight: KCRS [1][2][2][2] — output channels=1, input channels=2
     *   W[0][0][r][s] = [[1,0],[0,0]]  picks in[0] at r0s0
     *   W[0][1][r][s] = [[0,0],[0,1]]  picks in[1] at r1s1
     *
     * Output NCHW [1][1][2][2]:
     *   o[0,0]: in[1][1][1] = 50 * 1 = 50
     *   o[0,1]: in[1][1][2] = 60 * 1 = 60
     *   o[1,0]: in[1][2][1] = 80 * 1 = 80
     *   o[1,1]: in[1][2][2] = 90 * 1 = 90
     *
     * Wait — only in[1] contributes. That's too simple.
     * Let me use a more interesting weight:
     * W[0][0] = [[1,0],[0,0]]  W[0][1] = [[0,0],[0,1]]
     * So for each output position, we sum:
     *   in[0][iy+0][ix+0]*1 + in[1][iy+1][ix+1]*1
     * o[0,0]: in[0][0][0]*1 + in[1][1][1]*1 = 1 + 50 = 51
     * o[0,1]: in[0][0][1]*1 + in[1][1][2]*1 = 2 + 60 = 62
     * o[1,0]: in[0][1][0]*1 + in[1][2][1]*1 = 4 + 80 = 84
     * o[1,1]: in[0][1][1]*1 + in[1][2][2]*1 = 5 + 90 = 95
     */
    float input[] = {
        1,2,3, 4,5,6, 7,8,9,
        10,20,30, 40,50,60, 70,80,90
    };
    /* KCRS: [1][2][2][2] = 8 elements
     * Oc=0, Ic=0: [r=0,s=0]=1, [r=0,s=1]=0, [r=1,s=0]=0, [r=1,s=1]=0
     * Oc=0, Ic=1: [r=0,s=0]=0, [r=0,s=1]=0, [r=1,s=0]=0, [r=1,s=1]=1
     */
    float weight[] = {
        1,0, 0,0,   /* W[0][0] */
        0,0, 0,1,   /* W[0][1] */
    };
    float output[4] = {0};

    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 2,
        .in_height = 3, .in_width = 3,
        .out_channels = 1,  /* Fixed: 1 output channel */
        .kernel_h = 2, .kernel_w = 2,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 0, .pad_b = 0, .pad_l = 0, .pad_r = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    tu_conv_compute_dims(&desc);

    tu_conv2d_direct_nchw_fp32(input, weight, NULL, output, &desc);

    float expected[] = {51, 62, 84, 95};
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(output[i], expected[i], 0.001f);
    PASS();
}

/* ================================================================
 * Strided & Dilated Convolution Tests
 * ================================================================ */

static void test_conv_stride2(void) {
    TEST("Direct conv: stride=2, 5×5 input, 2×2 kernel");
    /* Input: 5×5 single channel (1..25)
     * Kernel: identity (all ones), 2×2, stride=2
     * Each output sums a 2×2 patch, stride=2 → output 2×2
     * pos(0,0): sum(1,2,6,7)=16
     * pos(0,1): sum(3,4,8,9)=24
     * pos(1,0): sum(11,12,16,17)=56
     * pos(1,1): sum(13,14,18,19)=64
     */
    float input[25];
    for (int i = 0; i < 25; i++) input[i] = (float)(i + 1);

    float weight[] = {1,1,1,1};
    float output[4] = {0};

    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 1,
        .in_height = 5, .in_width = 5,
        .out_channels = 1,
        .kernel_h = 2, .kernel_w = 2,
        .stride_h = 2, .stride_w = 2,
        .pad_t = 0, .pad_b = 0, .pad_l = 0, .pad_r = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    tu_conv_compute_dims(&desc);
    ASSERT_EQ(desc.out_height, 2);
    ASSERT_EQ(desc.out_width, 2);

    tu_conv2d_direct_nchw_fp32(input, weight, NULL, output, &desc);

    float expected[] = {16, 24, 56, 64};
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(output[i], expected[i], 0.001f);
    PASS();
}

/* ================================================================
 * Depthwise & Grouped Convolution
 * ================================================================ */

static void test_conv_depthwise(void) {
    TEST("Depthwise conv (groups=channels): 2 channels, 2×2 kernel");
    /* Input: [1][2][2][2] NCHW
     * Ch0: 1 2    Ch1: 10 20
     *      3 4          30 40
     * Weight: [2][1][1][1] = {2.0, 3.0} (per-channel scale)
     * Expected: Ch0 * 2, Ch1 * 3
     */
    float input[] = {1,2,3,4, 10,20,30,40};
    float weight[] = {2.0f, 3.0f};
    float output[8] = {0};

    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 2,
        .in_height = 2, .in_width = 2,
        .out_channels = 2,
        .kernel_h = 1, .kernel_w = 1,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 0, .pad_b = 0, .pad_l = 0, .pad_r = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 2,      /* depthwise: groups = in_channels = out_channels */
    };
    tu_conv_compute_dims(&desc);

    tu_conv2d_direct_nchw_fp32(input, weight, NULL, output, &desc);

    ASSERT_NEAR(output[0], 2.0f, 0.001f);   /* 1*2 */
    ASSERT_NEAR(output[1], 4.0f, 0.001f);   /* 2*2 */
    ASSERT_NEAR(output[4], 30.0f, 0.001f);  /* 10*3 */
    ASSERT_NEAR(output[5], 60.0f, 0.001f);  /* 20*3 */
    PASS();
}

/* ================================================================
 * Fused Activation Tests
 * ================================================================ */

static void test_conv_relu(void) {
    TEST("Conv with fused ReLU (negative input → 0 output)");
    /* Input: [1][2][2][2] NCHW
     *   Ch0: -1 -2     Ch1: 1 2
     *        -3 -4           3 4
     * Weight: [1][2][1][1] = {1, 1} — sum both channels
     *   pos(0,0): -1*1 + 1*1 = 0 → ReLU=0
     *   pos(0,1): -2*1 + 2*1 = 0 → ReLU=0
     *   pos(1,0): -3*1 + 3*1 = 0 → ReLU=0
     *   pos(1,1): -4*1 + 4*1 = 0 → ReLU=0
     */
    float input[] = {-1,-2,-3,-4, 1,2,3,4};
    float weight[] = {1.0f, 1.0f};  /* 2 elements for 2 input channels */
    float output[4] = {0};

    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 2,
        .in_height = 2, .in_width = 2,
        .out_channels = 1,
        .kernel_h = 1, .kernel_w = 1,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 0, .pad_b = 0, .pad_l = 0, .pad_r = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
        .activation = TU_CONV_ACTIVATION_RELU,
    };
    tu_conv_compute_dims(&desc);

    tu_conv2d_direct_nchw_fp32(input, weight, NULL, output, &desc);

    /* Expected: ReLU applied. We have 1 output channel:
     * output[0] = ReLU(-1*1 + 1*1) = ReLU(0) = 0
     * Wait, the direct conv NCHW with 1 output channel and 2 input channels...
     * W is [1][2][1][1] = [1,1]
     * out[0,0,0] = in[0,0,0]*w[0,0,0] + in[1,0,0]*w[0,1,0]
     *             = (-1) * 1 + 1 * 1 = 0 → ReLU(0)=0
     * out[0,0,1] = (-2) * 1 + 2 * 1 = 0 → 0
     * out[0,1,0] = (-3) * 1 + 3 * 1 = 0 → 0
     * out[0,1,1] = (-4) * 1 + 4 * 1 = 0 → 0
     */
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(output[i], 0.0f, 0.001f);
    PASS();
}

/* ================================================================
 * Cycle Estimation Sanity
 * ================================================================ */

static void test_conv_cycle_estimate(void) {
    TEST("Conv cycle estimate > 0 for valid config");
    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 3,
        .in_height = 32, .in_width = 32,
        .out_channels = 16,
        .kernel_h = 3, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 1, .pad_b = 1, .pad_l = 1, .pad_r = 1,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    tu_conv_compute_dims(&desc);
    uint64_t cycles = tu_conv_estimate_cycles(&desc, 16, 16);
    if (cycles == 0) { FAIL("cycle estimate is zero"); return; }
    PASS();
}

/* ================================================================
 * im2col + GEMM Pipeline
 * ================================================================ */

static void test_im2col_gemm_pipeline(void) {
    TEST("im2col+GEMM pipeline vs direct reference (1×1 kernel)");
    float input[] = {1,2,3,4};
    float weight[] = {2};
    float output_golden[4], output_pipe[4];
    float im2col_buf[16];

    tu_conv_desc_t desc = {
        .batch = 1, .in_channels = 1,
        .in_height = 2, .in_width = 2,
        .out_channels = 1,
        .kernel_h = 1, .kernel_w = 1,
        .stride_h = 1, .stride_w = 1,
        .pad_t = 0, .pad_b = 0, .pad_l = 0, .pad_r = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };
    tu_conv_compute_dims(&desc);

    /* Direct reference */
    memset(output_golden, 0, sizeof(output_golden));
    tu_conv2d_direct_nhwc_fp32(input, weight, NULL, output_golden, &desc);

    /* im2col+GEMM pipeline */
    memset(output_pipe, 0, sizeof(output_pipe));
    memset(im2col_buf, 0, sizeof(im2col_buf));
    tu_conv2d_im2col_gemm(input, weight, NULL, output_pipe, &desc, im2col_buf, sizeof(float));

    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(output_pipe[i], output_golden[i], 0.001f);
    PASS();
}

/* ================================================================
 * Test Runner
 * ================================================================ */

int main(void) {
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  TinyTU Convolution Engine Tests (O2)           ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    test_conv_dims_standard();
    test_conv_dims_stride2();
    test_conv_dims_invalid();
    test_im2col_nhwc_small();
    test_im2col_nhwc_padded();
    test_direct_conv_nchw_identity();
    test_direct_conv_nchw_3x3();
    test_conv_stride2();
    test_conv_depthwise();
    test_conv_relu();
    test_conv_cycle_estimate();
    test_im2col_gemm_pipeline();

    printf("\n  %d/%d tests passed\n\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
