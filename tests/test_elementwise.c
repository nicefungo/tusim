/*
 * TinyTU Elementwise Pipeline Tests
 * ==================================
 * Tests for fused elementwise ops in the accumulator path (gap O4).
 */

#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/compute/elementwise_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s", name); \
} while(0)

#define PASS() do { \
    tests_passed++; \
    printf(" PASS\n"); \
} while(0)

#define FAIL(msg) do { \
    tests_failed++; \
    printf(" FAIL (%s)\n", msg); \
} while(0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        char buf[128]; \
        snprintf(buf, sizeof(buf), "%s: %f != %f", msg, (double)(a), (double)(b)); \
        FAIL(buf); return; \
    } \
} while(0)

/* ---- Test helpers ---- */

static void fill_fp32(tu_sram_region_t *sram, uint32_t offset,
                      uint32_t count, const float *vals) {
    memcpy(tu_sram_raw_ptr(sram) + offset, vals, count * sizeof(float));
}

static void read_fp32(tu_sram_region_t *sram, uint32_t offset,
                      uint32_t count, float *out) {
    memcpy(out, tu_sram_raw_ptr(sram) + offset, count * sizeof(float));
}

/* ---- Test 1: ReLU ---- */
static void test_relu(void) {
    TEST("ReLU in-place (4 elements)");
    tu_init();

    float input[] = { -1.0f, 0.0f, 1.0f, 2.5f };
    float expected[] = { 0.0f, 0.0f, 1.0f, 2.5f };

    fill_fp32(&g_tu.sram_o, 0, 4, input);
    tu_ew_apply_unary(&g_tu.sram_o, 0, 4, TU_EW_RELU);

    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f, "ReLU mismatch");
    }
    PASS();
}

/* ---- Test 2: GELU ---- */
static void test_gelu(void) {
    TEST("GELU (4 elements)");
    tu_init();

    /* GELU(0) = 0, GELU(inf) ~ inf, GELU(-inf) ~ 0 */
    float input[] = { -3.0f, 0.0f, 1.0f, 3.0f };
    float expected[4];

    fill_fp32(&g_tu.sram_o, 0, 4, input);

    /* Compute expected using the same tanh approx */
    for (int i = 0; i < 4; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        expected[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    tu_ew_apply_unary(&g_tu.sram_o, 0, 4, TU_EW_GELU);

    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-5f, "GELU mismatch");
    }
    PASS();
}

/* ---- Test 3: SiLU ---- */
static void test_silu(void) {
    TEST("SiLU (4 elements)");
    tu_init();

    float input[] = { -2.0f, 0.0f, 1.0f, 2.0f };
    float expected[4];

    fill_fp32(&g_tu.sram_o, 0, 4, input);

    for (int i = 0; i < 4; i++) {
        float x = input[i];
        expected[i] = x / (1.0f + expf(-x));
    }

    tu_ew_apply_unary(&g_tu.sram_o, 0, 4, TU_EW_SILU);

    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-5f, "SiLU mismatch");
    }
    PASS();
}

/* ---- Test 4: Sigmoid ---- */
static void test_sigmoid(void) {
    TEST("Sigmoid (4 elements)");
    tu_init();

    float input[] = { -5.0f, 0.0f, 2.0f, 5.0f };
    float expected[4];

    fill_fp32(&g_tu.sram_o, 0, 4, input);

    for (int i = 0; i < 4; i++) {
        expected[i] = 1.0f / (1.0f + expf(-input[i]));
    }

    tu_ew_apply_unary(&g_tu.sram_o, 0, 4, TU_EW_SIGMOID);

    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f, "Sigmoid mismatch");
    }
    PASS();
}

/* ---- Test 5: Tanh ---- */
static void test_tanh(void) {
    TEST("Tanh (4 elements)");
    tu_init();

    float input[] = { -2.0f, 0.0f, 0.5f, 2.0f };
    float expected[4];

    fill_fp32(&g_tu.sram_o, 0, 4, input);

    for (int i = 0; i < 4; i++) {
        expected[i] = tanhf(input[i]);
    }

    tu_ew_apply_unary(&g_tu.sram_o, 0, 4, TU_EW_TANH);

    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f, "Tanh mismatch");
    }
    PASS();
}

/* ---- Test 6: Neg and Abs ---- */
static void test_neg_abs(void) {
    TEST("Neg + Abs (4 elements)");
    tu_init();

    float input[] = { -3.0f, -1.0f, 0.0f, 2.0f };
    float expected_neg[] = { 3.0f, 1.0f, 0.0f, -2.0f };
    float expected_abs[] = { 3.0f, 1.0f, 0.0f, 2.0f };

    /* Neg */
    fill_fp32(&g_tu.sram_o, 0, 4, input);
    tu_ew_apply_unary(&g_tu.sram_o, 0, 4, TU_EW_NEG);
    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);
    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected_neg[i], 1e-6f, "Neg mismatch");
    }

    /* Abs */
    fill_fp32(&g_tu.sram_o, 0, 4, input);
    tu_ew_apply_unary(&g_tu.sram_o, 0, 4, TU_EW_ABS);
    read_fp32(&g_tu.sram_o, 0, 4, result);
    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected_abs[i], 1e-6f, "Abs mismatch");
    }

    PASS();
}

/* ---- Test 7: Binary Add Scalar ---- */
static void test_add_scalar(void) {
    TEST("Add scalar (4 elements)");
    tu_init();

    float input[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float expected[] = { 3.5f, 4.5f, 5.5f, 6.5f };

    fill_fp32(&g_tu.sram_o, 0, 4, input);
    tu_ew_apply_binary_scalar(&g_tu.sram_o, 0, 4, TU_EW_ADD, 2.5f);

    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f, "Add scalar mismatch");
    }
    PASS();
}

/* ---- Test 8: Binary Mul Scalar ---- */
static void test_mul_scalar(void) {
    TEST("Mul scalar (4 elements)");
    tu_init();

    float input[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float expected[] = { 3.0f, 6.0f, 9.0f, 12.0f };

    fill_fp32(&g_tu.sram_o, 0, 4, input);
    tu_ew_apply_binary_scalar(&g_tu.sram_o, 0, 4, TU_EW_MUL, 3.0f);

    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f, "Mul scalar mismatch");
    }
    PASS();
}

/* ---- Test 9: Fused Add + ReLU ---- */
static void test_fused_add_relu(void) {
    TEST("Fused Add(1.0) + ReLU (8 elements)");
    tu_init();

    float input[] = { -3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 5.0f };
    float expected[8];

    fill_fp32(&g_tu.sram_o, 0, 8, input);

    for (int i = 0; i < 8; i++) {
        float x = input[i] + 1.0f;  /* Add 1.0 */
        expected[i] = x > 0.0f ? x : 0.0f;  /* ReLU */
    }

    tu_ew_op_t ops[2];
    ops[0].opcode = TU_EW_ADD;
    ops[0].has_scalar = true;
    ops[0].scalar = 1.0f;
    ops[1].opcode = TU_EW_RELU;
    ops[1].has_scalar = false;

    tu_ew_apply_fused(&g_tu.sram_o, 0, 8, ops, 2);

    float result[8];
    read_fp32(&g_tu.sram_o, 0, 8, result);

    for (int i = 0; i < 8; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f, "Fused Add+ReLU mismatch");
    }
    PASS();
}

/* ---- Test 10: Fused Mul + GELU ---- */
static void test_fused_mul_gelu(void) {
    TEST("Fused Mul(0.5) + GELU (4 elements)");
    tu_init();

    float input[] = { -3.0f, 0.0f, 1.0f, 3.0f };
    float expected[4];

    fill_fp32(&g_tu.sram_o, 0, 4, input);

    for (int i = 0; i < 4; i++) {
        float x = input[i] * 0.5f;
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        expected[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    tu_ew_op_t ops[2];
    ops[0].opcode = TU_EW_MUL;
    ops[0].has_scalar = true;
    ops[0].scalar = 0.5f;
    ops[1].opcode = TU_EW_GELU;
    ops[1].has_scalar = false;

    tu_ew_apply_fused(&g_tu.sram_o, 0, 4, ops, 2);

    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);

    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-5f, "Fused Mul+GELU mismatch");
    }
    PASS();
}

/* ---- Test 11: Exp and Sqrt ---- */
static void test_exp_sqrt(void) {
    TEST("Exp + Sqrt (4 elements)");
    tu_init();

    float input_exp[] = { 0.0f, 1.0f, 2.0f, -1.0f };
    float e = expf(1.0f);
    float expected_exp[] = { 1.0f, e, e * e, 1.0f / e };

    fill_fp32(&g_tu.sram_o, 0, 4, input_exp);
    tu_ew_apply_unary(&g_tu.sram_o, 0, 4, TU_EW_EXP);

    float result[4];
    read_fp32(&g_tu.sram_o, 0, 4, result);
    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected_exp[i], 1e-5f, "Exp mismatch");
    }

    /* Sqrt */
    float input_sqrt[] = { 0.0f, 1.0f, 4.0f, 9.0f };
    float expected_sqrt[] = { 0.0f, 1.0f, 2.0f, 3.0f };

    fill_fp32(&g_tu.sram_o, 0, 4, input_sqrt);
    tu_ew_apply_unary(&g_tu.sram_o, 0, 4, TU_EW_SQRT);

    read_fp32(&g_tu.sram_o, 0, 4, result);
    for (int i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(result[i], expected_sqrt[i], 1e-5f, "Sqrt mismatch");
    }

    PASS();
}

/* ---- Test 12: Large tensor (256 elements) ---- */
static void test_large_tensor(void) {
    TEST("ReLU large tensor (256 elements)");
    tu_init();

    float input[256];
    float expected[256];
    for (int i = 0; i < 256; i++) {
        input[i] = (float)(i - 128) * 0.1f;
        expected[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

    fill_fp32(&g_tu.sram_o, 0, 256, input);
    tu_ew_apply_unary(&g_tu.sram_o, 0, 256, TU_EW_RELU);

    float result[256];
    read_fp32(&g_tu.sram_o, 0, 256, result);

    for (int i = 0; i < 256; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f, "Large ReLU mismatch");
    }
    PASS();
}

/* ---- Test 13: Command Queue Elementwise ---- */
static void test_cmdq_elementwise(void) {
    TEST("CMDQ elementwise ReLU (16 elements)");
    tu_init();

    float input[16];
    float expected[16];
    for (int i = 0; i < 16; i++) {
        input[i] = (float)(i - 8);
        expected[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

    /* Load data into O-buffer */
    tu_dma_load_o(input, 0, 16 * sizeof(float));

    /* Submit elementwise ReLU via command queue */
    uint8_t ops[1] = { TU_EW_RELU };
    int cmd_id = tu_cmdq_submit_elementwise(0, 0, 16, ops, 1, NULL, NULL);
    ASSERT(cmd_id > 0, "elementwise cmdq submit failed");

    tu_cmdq_sync_all();

    /* Read back results */
    float result[16];
    tu_dma_store_o(result, 0, 16 * sizeof(float));

    for (int i = 0; i < 16; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-6f, "CMDQ EW ReLU mismatch");
    }
    PASS();
}

/* ---- Test 14: MMA + Elementwise Pipeline ---- */
static void test_mma_relu_pipeline(void) {
    TEST("MMA + ReLU pipeline (16x16)");
    tu_init();

    /* Simple identity weight: W = I (16x16) */
    fp16_t W[16 * 16];
    fp16_t A[16 * 16];
    fp32_t expected[16 * 16];

    for (int i = 0; i < 256; i++) {
        W[i] = fp32_to_fp16((i % 17 == 0) ? 1.0f : 0.0f);  /* Identity */
        A[i] = fp32_to_fp16((float)((i % 16) - 8));  /* values from -8 to 7 */
        expected[i] = ((i % 16) - 8) > 0 ? ((float)(i % 16) - 8) : 0.0f;  /* ReLU expected */
    }

    /* Load W and A */
    tu_dma_load_w(W, 0, sizeof(W));
    tu_dma_load_a(A, 0, sizeof(A));

    /* MMA: O = W × A */
    tu_mma(16, 16, 16, 0, 0, 0, false);

    /* Apply ReLU in-place on O-buffer */
    tu_ew_apply_unary(&g_tu.sram_o, 0, 256, TU_EW_RELU);

    /* Read result and verify */
    fp32_t result[256];
    memcpy(result, tu_sram_raw_ptr(&g_tu.sram_o), sizeof(result));

    for (int i = 0; i < 256; i++) {
        ASSERT_FLOAT_EQ(result[i], expected[i], 1e-4f, "MMA+ReLU mismatch");
    }
    PASS();
}

/* ---- Test 15: Opcode name utility ---- */
static void test_opcode_names(void) {
    TEST("Opcode name strings");
    ASSERT(strcmp(tu_ew_opcode_name(TU_EW_RELU), "ReLU") == 0, "RELU name");
    ASSERT(strcmp(tu_ew_opcode_name(TU_EW_GELU), "GELU") == 0, "GELU name");
    ASSERT(strcmp(tu_ew_opcode_name(TU_EW_ADD), "Add") == 0, "ADD name");
    ASSERT(strcmp(tu_ew_opcode_name(TU_EW_MUL), "Mul") == 0, "MUL name");
    PASS();
}

/* ---- Test 16: Validate descriptor ---- */
static void test_validate_desc(void) {
    TEST("Descriptor validation");

    tu_ew_desc_t desc;
    memset(&desc, 0, sizeof(desc));

    /* NULL region should fail */
    ASSERT(!tu_ew_validate_desc(NULL), "null desc should be invalid");

    /* Missing region */
    ASSERT(!tu_ew_validate_desc(&desc), "missing region invalid");

    /* Valid minimal */
    tu_init();
    desc.sram_region = &g_tu.sram_o;
    desc.elem_count = 4;
    desc.num_ops = 1;
    desc.sram_offset = 0;
    desc.in_place = true;
    desc.ops[0].opcode = TU_EW_RELU;
    ASSERT(tu_ew_validate_desc(&desc), "valid desc should pass");

    /* Out of bounds */
    desc.sram_offset = g_tu.sram_o.total_size + 1;
    ASSERT(!tu_ew_validate_desc(&desc), "OOB offset should fail");

    /* Bad opcode */
    desc.sram_offset = 0;
    desc.ops[0].opcode = (tu_ew_opcode_t)255;
    ASSERT(!tu_ew_validate_desc(&desc), "bad opcode should fail");

    PASS();
}

/* ---- Main ---- */
int main(void) {
    printf("\nTinyTU Elementwise Pipeline Tests\n");
    printf("==================================\n\n");

    test_relu();
    test_gelu();
    test_silu();
    test_sigmoid();
    test_tanh();
    test_neg_abs();
    test_add_scalar();
    test_mul_scalar();
    test_fused_add_relu();
    test_fused_mul_gelu();
    test_exp_sqrt();
    test_large_tensor();
    test_cmdq_elementwise();
    test_mma_relu_pipeline();
    test_opcode_names();
    test_validate_desc();

    printf("\n");
    printf("═══════════════════════════════════════════\n");
    printf("  %d/%d tests passed", tests_passed, tests_run);
    if (tests_failed > 0) printf(", %d FAILED", tests_failed);
    printf("\n");
    printf("═══════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
