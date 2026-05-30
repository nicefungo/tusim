/*
 * TinyTU INT8/INT4 Quantization Tests
 * =====================================
 * Gap D2: Verify INT8 and INT4 quantization correctness.
 *
 * Tests:
 *   1. INT8 round-trip conversion (quantize + dequantize = original within tolerance)
 *   2. INT8 symmetric calibration
 *   3. INT8 asymmetric calibration
 *   4. UINT4 packed storage
 *   5. INT8 dot product and MMA tile correctness
 *   6. Precision registry entries
 */

#include "tu_cmodel/tu_int_quant.h"
#include "tu_cmodel/tu_precision.h"
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
 * INT8 Round-Trip Tests
 * ================================================================ */

static void test_int8_roundtrip_zero(void) {
    TEST("INT8 round-trip (0.0)");
    tu_quant_params_t qp;
    tu_quant_params_init_int8(&qp);
    int8_t_t q = tu_fp32_to_int8(0.0f, &qp);
    float r = tu_int8_to_fp32(q, &qp);
    ASSERT_NEAR(r, 0.0f, qp.scale * 0.6f);
    PASS();
}

static void test_int8_roundtrip_positive(void) {
    TEST("INT8 round-trip (6.35, symmetric)");
    tu_quant_params_t qp;
    tu_quant_params_init_int8(&qp);
    /* With default scale ~0.00787, 6.35/0.00787 ≈ 806 → clamped to 127 */
    qp.scale = 0.05f;
    int8_t_t q = tu_fp32_to_int8(6.35f, &qp);
    float r = tu_int8_to_fp32(q, &qp);
    ASSERT_NEAR(r, 6.35f, qp.scale * 0.6f);
    PASS();
}

static void test_int8_roundtrip_negative(void) {
    TEST("INT8 round-trip (-3.5, symmetric)");
    tu_quant_params_t qp;
    tu_quant_params_init_int8(&qp);
    qp.scale = 0.05f;
    int8_t_t q = tu_fp32_to_int8(-3.5f, &qp);
    float r = tu_int8_to_fp32(q, &qp);
    ASSERT_NEAR(r, -3.5f, qp.scale * 0.6f);
    PASS();
}

static void test_int8_batch_conversion(void) {
    TEST("INT8 batch conversion (16 elements)");
    tu_quant_params_t qp;
    tu_quant_params_init_int8(&qp);
    qp.scale = 0.1f;

    float src[16], dst[16];
    int8_t_t qbuf[16];
    for (int i = 0; i < 16; i++) src[i] = (float)(i - 8) * 0.5f;

    tu_fp32_to_int8_buffer(src, qbuf, 16, &qp);
    tu_int8_to_fp32_buffer(qbuf, dst, 16, &qp);

    for (int i = 0; i < 16; i++)
        ASSERT_NEAR(dst[i], src[i], qp.scale * 0.6f);
    PASS();
}

/* ================================================================
 * INT8 Calibration Tests
 * ================================================================ */

static void test_int8_symmetric_cal(void) {
    TEST("INT8 symmetric calibration (range [-5, 5])");
    float data[] = {-5.0f, 0.0f, 3.0f, 5.0f, -2.0f};
    tu_quant_params_t qp;
    tu_quant_params_calibrate_int8_symmetric(data, 5, &qp);
    ASSERT_EQ(qp.zero_point, 0);
    ASSERT_NEAR(qp.scale, 5.0f / 127.0f, 0.001f);
    PASS();
}

static void test_int8_asymmetric_cal(void) {
    TEST("INT8 asymmetric calibration (range [0, 10])");
    float data[] = {0.0f, 2.0f, 5.0f, 10.0f, 7.0f};
    tu_quant_params_t qp;
    tu_quant_params_calibrate_int8_asymmetric(data, 5, &qp);
    /* scale = (10-0)/255 ≈ 0.0392 */
    ASSERT_NEAR(qp.scale, 10.0f / 255.0f, 0.001f);
    /* zero_point = clamp(round(-128 - 0/scale), -128, 127) = -128 since 0/scale ≈ 0 */
    PASS();
}

/* ================================================================
 * UINT4 Packed Storage Tests
 * ================================================================ */

static void test_uint4_pack_unpack(void) {
    TEST("UINT4 pack/unpack round-trip (16 elements)");
    uint8_t packed[8] = {0};
    for (int i = 0; i < 16; i++)
        tu_uint4_pack(packed, i, (uint8_t)(i % 16));

    for (int i = 0; i < 16; i++) {
        uint8_t v = tu_uint4_unpack(packed, i);
        if (v != (uint8_t)(i % 16)) {
            FAIL("UINT4 pack/unpack mismatch at idx %d: got %d, expected %d", i, (int)v, i % 16);
            return;
        }
    }
    PASS();
}

static void test_uint4_quant_roundtrip(void) {
    TEST("UINT4 quantize/dequantize round-trip");
    tu_quant_params_t qp;
    tu_quant_params_init_uint4(&qp);
    qp.scale = 0.1f;  /* Small enough that 7*0.3 = 2.1 fits in [0, 1.5] range */
    qp.zero_point = 0;  /* Use zero-centered for full dynamic range */

    float src[8], dst[8];
    uint8_t packed[4] = {0};
    for (int i = 0; i < 8; i++) src[i] = (float)i * 0.2f;  /* 0.0, 0.2, 0.4, ..., 1.4 */

    tu_fp32_to_uint4_buffer(src, packed, 8, &qp);
    tu_uint4_to_fp32_buffer(packed, dst, 8, &qp);

    for (int i = 0; i < 8; i++)
        ASSERT_NEAR(dst[i], src[i], qp.scale * 0.6f);
    PASS();
}

/* ================================================================
 * INT8 MAC Tests
 * ================================================================ */

static void test_int8_dot_product(void) {
    TEST("INT8 dot product [1,2,3] · [4,5,6] = 32");
    int8_t_t a[] = {1, 2, 3};
    int8_t_t b[] = {4, 5, 6};
    int32_t sum = tu_int8_dot_product(a, b, 3);
    ASSERT_EQ(sum, 32);  /* 1*4 + 2*5 + 3*6 = 4+10+18 = 32 */
    PASS();
}

static void test_int8_dot_product_large(void) {
    TEST("INT8 dot product [127,127] · [127,127] = 32258");
    int8_t_t a[] = {127, 127};
    int8_t_t b[] = {127, 127};
    int32_t sum = tu_int8_dot_product(a, b, 2);
    ASSERT_EQ(sum, 32258);  /* 127*127 + 127*127 = 16129 + 16129 */
    PASS();
}

static void test_int8_mma_tile(void) {
    TEST("INT8 MMA tile (2x3x2)");
    /* W = [[1, 2],
     *      [3, 4]]  — 2×2
     * A = [[1, 2, 3],
     *      [4, 5, 6]] — 2×3
     * O = W × A (accumulate into zeros)
     *   = [[1*1+2*4, 1*2+2*5, 1*3+2*6],
     *      [3*1+4*4, 3*2+4*5, 3*3+4*6]]
     *   = [[9, 12, 15],
     *      [19, 26, 33]]
     */
    int8_t_t W[] = {1, 2, 3, 4};
    int8_t_t A[] = {1, 2, 3, 4, 5, 6};
    int32_t O[] = {0, 0, 0, 0, 0, 0};

    tu_int8_mma_tile(W, A, O, 2, 3, 2);

    int32_t expected[] = {9, 12, 15, 19, 26, 33};
    for (int i = 0; i < 6; i++)
        ASSERT_EQ(O[i], expected[i]);
    PASS();
}

static void test_int8_mma_accumulate(void) {
    TEST("INT8 MMA accumulate (adds to existing O)");
    int8_t_t W[] = {1, 0, 0, 1};
    int8_t_t A[] = {5, 0, 0, 5};
    int32_t O[] = {10, 0, 0, 10};

    tu_int8_mma_tile(W, A, O, 2, 2, 2);
    /* W × A = [[5,0],[0,5]], O += result → [[15,0],[0,15]] */
    ASSERT_EQ(O[0], 15);
    ASSERT_EQ(O[3], 15);
    PASS();
}

/* ================================================================
 * Precision Registry Tests
 * ================================================================ */

static void test_precision_registry_int8(void) {
    TEST("Precision registry: INT8 entry exists");
    const tu_precision_desc_t *pd = tu_precision_get(TU_PREC_INT8);
    if (!pd) { FAIL("INT8 precision not registered"); return; }
    ASSERT_EQ(pd->type, TU_PREC_INT8);
    ASSERT_EQ(pd->elem_bytes, 1);
    PASS();
}

static void test_precision_registry_int4(void) {
    TEST("Precision registry: INT4 entry exists");
    const tu_precision_desc_t *pd = tu_precision_get(TU_PREC_INT4);
    if (!pd) { FAIL("INT4 precision not registered"); return; }
    ASSERT_EQ(pd->type, TU_PREC_INT4);
    PASS();
}

/* ================================================================
 * Test Runner
 * ================================================================ */

int main(void) {
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  TinyTU INT8/INT4 Quantization Tests (D2)       ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    test_int8_roundtrip_zero();
    test_int8_roundtrip_positive();
    test_int8_roundtrip_negative();
    test_int8_batch_conversion();
    test_int8_symmetric_cal();
    test_int8_asymmetric_cal();
    test_uint4_pack_unpack();
    test_uint4_quant_roundtrip();
    test_int8_dot_product();
    test_int8_dot_product_large();
    test_int8_mma_tile();
    test_int8_mma_accumulate();
    test_precision_registry_int8();
    test_precision_registry_int4();

    printf("\n  %d/%d tests passed\n\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
