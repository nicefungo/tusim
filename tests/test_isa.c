/*
 * TU CModel — ISA Test Suite
 * ============================
 * Tests: opcode enumeration, instruction encoding size, flag
 * decoding, category classification, query functions, backward
 * compatibility with command queue opcodes.
 */

#include "tu_cmodel/isa/tu_isa.h"
#include "tu_cmodel/command_queue.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [%d] %s ... ", tests_run, name); \
} while(0)

#define PASS() do { \
    tests_passed++; \
    printf("PASS\n"); \
} while(0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
} while(0)

#define CHECK(cond, msg) do { \
    if (!(cond)) { FAIL(msg); goto done; } \
} while(0)

/* ---- Test 1: Instruction size is exactly 96 bits ---- */
static void test_instruction_size(void) {
    TEST("Instruction size = 96 bits (12 bytes)");
    CHECK(sizeof(tu_instruction_t) == 12, "wrong instruction size");
    CHECK(sizeof(tu_instruction_t) * 8 == 96, "not 96 bits");
    PASS();
done:;
}

/* ---- Test 2: All defined opcodes have names ---- */
static void test_all_opcodes_named(void) {
    TEST("All defined opcodes have names (reserved slots may be UNKNOWN)");
    int named = 0, unknown = 0;
    for (int i = 0; i < TU_ISA_OPCODE_COUNT; i++) {
        const char *name = tu_isa_opcode_name((tu_isa_opcode_t)i);
        if (strcmp(name, "UNKNOWN") == 0) {
            unknown++;
        } else {
            named++;
        }
    }
    CHECK(named >= 50, "too few named opcodes (expected 50+)");
    /* Reserved gaps are OK — they return UNKNOWN */
    printf("(%d named, %d reserved) ", named, unknown);
    PASS();
done:;
}

/* ---- Test 3: Opcode names are correct ---- */
static void test_opcode_names_correct(void) {
    TEST("Key opcode names are correct");
    CHECK(strcmp(tu_isa_opcode_name(TU_ISA_NOP), "NOP") == 0, "NOP");
    CHECK(strcmp(tu_isa_opcode_name(TU_ISA_MMA), "MMA") == 0, "MMA");
    CHECK(strcmp(tu_isa_opcode_name(TU_ISA_CONV2D), "CONV2D") == 0, "CONV2D");
    CHECK(strcmp(tu_isa_opcode_name(TU_ISA_ATTENTION), "ATTENTION") == 0, "ATTENTION");
    CHECK(strcmp(tu_isa_opcode_name(TU_ISA_SOFTMAX), "SOFTMAX") == 0, "SOFTMAX");
    CHECK(strcmp(tu_isa_opcode_name(TU_ISA_LAYER_NORM), "LAYER_NORM") == 0, "LAYER_NORM");
    CHECK(strcmp(tu_isa_opcode_name(TU_ISA_DMA_LOAD), "DMA.LOAD") == 0, "DMA.LOAD");
    CHECK(strcmp(tu_isa_opcode_name(TU_ISA_RELU), "RELU") == 0, "RELU");
    PASS();
done:;
}

/* ---- Test 4: Category classification ---- */
static void test_category_classification(void) {
    TEST("Category classification correct");
    CHECK(tu_isa_opcode_category(TU_ISA_NOP) == TU_ISA_CAT_CONTROL, "NOP not control");
    CHECK(tu_isa_opcode_category(TU_ISA_BARRIER) == TU_ISA_CAT_CONTROL, "BARRIER not control");
    CHECK(tu_isa_opcode_category(TU_ISA_MMA) == TU_ISA_CAT_MATRIX, "MMA not matrix");
    CHECK(tu_isa_opcode_category(TU_ISA_CONV2D) == TU_ISA_CAT_MATRIX, "CONV2D not matrix");
    CHECK(tu_isa_opcode_category(TU_ISA_ATTENTION) == TU_ISA_CAT_MATRIX, "ATTN not matrix");
    CHECK(tu_isa_opcode_category(TU_ISA_RELU) == TU_ISA_CAT_ELEMENTWISE, "RELU not ew");
    CHECK(tu_isa_opcode_category(TU_ISA_SOFTMAX) == TU_ISA_CAT_NORM_REDUCE, "SOFTMAX not norm");
    CHECK(tu_isa_opcode_category(TU_ISA_LAYER_NORM) == TU_ISA_CAT_NORM_REDUCE, "LN not norm");
    CHECK(tu_isa_opcode_category(TU_ISA_POOL_MAX) == TU_ISA_CAT_POOLING, "POOL not pool");
    CHECK(tu_isa_opcode_category(TU_ISA_DMA_LOAD) == TU_ISA_CAT_DATA_MOVEMENT, "DMA not dm");
    CHECK(tu_isa_opcode_category(TU_ISA_TRANSPOSE) == TU_ISA_CAT_DATA_LAYOUT, "TRANS not layout");
    CHECK(tu_isa_opcode_category(TU_ISA_SPARSE_MMA) == TU_ISA_CAT_SPARSITY, "SPARSE not sparsity");
    PASS();
done:;
}

/* ---- Test 5: Query functions ---- */
static void test_query_functions(void) {
    TEST("Query functions correct");

    /* SRAM operands */
    CHECK(tu_isa_has_sram_operands(TU_ISA_NOP) == false, "NOP has SRAM");
    CHECK(tu_isa_has_sram_operands(TU_ISA_BARRIER) == false, "BARRIER has SRAM");
    CHECK(tu_isa_has_sram_operands(TU_ISA_MMA) == true, "MMA no SRAM");
    CHECK(tu_isa_has_sram_operands(TU_ISA_CONV2D) == true, "CONV2D no SRAM");
    CHECK(tu_isa_has_sram_operands(TU_ISA_DMA_LOAD) == true, "DMA no SRAM");

    /* Compute ops */
    CHECK(tu_isa_is_compute_op(TU_ISA_NOP) == false, "NOP is compute");
    CHECK(tu_isa_is_compute_op(TU_ISA_DMA_LOAD) == false, "DMA is compute");
    CHECK(tu_isa_is_compute_op(TU_ISA_MMA) == true, "MMA not compute");
    CHECK(tu_isa_is_compute_op(TU_ISA_RELU) == true, "RELU not compute");
    CHECK(tu_isa_is_compute_op(TU_ISA_SOFTMAX) == true, "SOFTMAX not compute");
    CHECK(tu_isa_is_compute_op(TU_ISA_SPARSE_MMA) == true, "SPARSE not compute");

    /* DMA ops */
    CHECK(tu_isa_is_dma_op(TU_ISA_MMA) == false, "MMA is dma");
    CHECK(tu_isa_is_dma_op(TU_ISA_DMA_LOAD) == true, "DMA.LOAD not dma");
    CHECK(tu_isa_is_dma_op(TU_ISA_DMA_STORE) == true, "DMA.STORE not dma");
    CHECK(tu_isa_is_dma_op(TU_ISA_DMA_SCATTER) == true, "DMA.SCATTER not dma");
    CHECK(tu_isa_is_dma_op(TU_ISA_DMA_BROADCAST) == true, "DMA.BCAST not dma");

    PASS();
done:;
}

/* ---- Test 6: Flag decoding ---- */
static void test_flag_decoding(void) {
    TEST("Flag decoding");
    uint8_t precision, transpose, activation;
    bool has_bias;

    /* No flags set */
    tu_isa_decode_flags(0x00, &precision, &transpose, &activation, &has_bias);
    CHECK(precision == 0, "precision != 0");
    CHECK(transpose == 0, "transpose != 0");
    CHECK(activation == 0, "activation != 0");
    CHECK(has_bias == false, "has_bias != false");

    /* FP16 + transpose A + RELU + bias */
    uint8_t flags = TU_FLAG_PREC_FP16 | TU_FLAG_TRANSPOSE_A |
                    TU_FLAG_ACT_RELU | TU_FLAG_BIAS;
    tu_isa_decode_flags(flags, &precision, &transpose, &activation, &has_bias);
    CHECK(precision == 0, "precision wrong (fp16)");
    CHECK(transpose == 1, "transpose wrong (A)");
    CHECK(activation == 1, "activation wrong (relu)");
    CHECK(has_bias == true, "bias wrong");

    /* BF16 + transpose B */
    flags = TU_FLAG_PREC_BF16 | TU_FLAG_TRANSPOSE_B;
    tu_isa_decode_flags(flags, &precision, &transpose, NULL, NULL);
    CHECK(precision == 2, "precision wrong (bf16)");
    CHECK(transpose == 2, "transpose wrong (B)");

    /* INT8 + GELU + no bias */
    flags = TU_FLAG_PREC_INT8 | TU_FLAG_ACT_GELU;
    tu_isa_decode_flags(flags, NULL, NULL, &activation, &has_bias);
    CHECK(activation == 2, "activation wrong (gelu)");
    CHECK(has_bias == false, "bias wrong (false)");

    PASS();
done:;
}

/* ---- Test 7: Backward-compat opcode mapping ---- */
static void test_backward_compat_opcodes(void) {
    TEST("Backward-compat CMD opcodes map to ISA");
    CHECK(TU_CMD_NOP == TU_ISA_NOP, "NOP mismatch");
    CHECK(TU_CMD_DMA_LOAD == TU_ISA_DMA_LOAD, "DMA_LOAD mismatch");
    CHECK(TU_CMD_DMA_STORE == TU_ISA_DMA_STORE, "DMA_STORE mismatch");
    CHECK(TU_CMD_MMA == TU_ISA_MMA, "MMA mismatch");
    CHECK(TU_CMD_SYNC == TU_ISA_SYNC, "SYNC mismatch");
    CHECK(TU_CMD_BARRIER == TU_ISA_BARRIER, "BARRIER mismatch");
    CHECK(TU_CMD_HALT == TU_ISA_HALT, "HALT mismatch");
    PASS();
done:;
}

/* ---- Test 8: All opcodes are distinct ---- */
static void test_opcodes_distinct(void) {
    TEST("All opcodes are distinct (excluding reserved UNKNOWNs)");
    for (int i = 0; i < TU_ISA_OPCODE_COUNT; i++) {
        const char *ni = tu_isa_opcode_name((tu_isa_opcode_t)i);
        if (!ni || strcmp(ni, "UNKNOWN") == 0) continue;
        for (int j = i + 1; j < TU_ISA_OPCODE_COUNT; j++) {
            const char *nj = tu_isa_opcode_name((tu_isa_opcode_t)j);
            if (!nj || strcmp(nj, "UNKNOWN") == 0) continue;
            if (strcmp(ni, nj) == 0) {
                printf("\n  Duplicate name: %s at %d and %d\n", ni, i, j);
                FAIL("duplicate opcode name");
                goto done;
            }
        }
    }
    PASS();
done:;
}

/* ---- Test 9: Descriptor struct sizes are reasonable ---- */
static void test_descriptor_sizes(void) {
    TEST("Descriptor struct sizes");
    CHECK(sizeof(tu_mma_op_desc_t) <= 64, "MMA desc too large");
    CHECK(sizeof(tu_conv_op_desc_t) <= 128, "Conv desc too large");
    CHECK(sizeof(tu_attention_op_desc_t) <= 96, "Attention desc too large");
    CHECK(sizeof(tu_ew_op_desc_t) <= 64, "EW desc too large");
    CHECK(sizeof(tu_norm_op_desc_t) <= 64, "Norm desc too large");
    CHECK(sizeof(tu_pool_op_desc_t) <= 64, "Pool desc too large");
    CHECK(sizeof(tu_dma_op_desc_t) <= 64, "DMA desc too large");
    CHECK(sizeof(tu_op_descriptor_t) <= 256, "Unified desc too large");
    PASS();
done:;
}

/* ---- Main ---- */
int main(void) {
    printf("\n=== TU ISA Tests ===\n\n");

    test_instruction_size();
    test_all_opcodes_named();
    test_opcode_names_correct();
    test_category_classification();
    test_query_functions();
    test_flag_decoding();
    test_backward_compat_opcodes();
    test_opcodes_distinct();
    test_descriptor_sizes();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
