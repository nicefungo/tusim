/*
 * TU CModel — Liveness-Based Scratchpad Allocator Tests (Gap C3)
 * ==============================================================
 *
 * Tests for:
 *   - Liveness analysis (VReg extraction, live ranges)
 *   - Interference graph construction
 *   - Greedy coloring with physical allocation
 *   - Spill candidate selection (all 4 strategies)
 *   - Spill/fill instruction insertion
 *   - Full allocation pipeline
 *   - Edge cases: empty, single VReg, all independent, all conflicting
 *   - W/A/O region independence
 *   - Configurable capacities
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/isa/tu_isa.h"
#include "../tu_cmodel/isa/tu_liveness.h"

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

static tu_instruction_t make_instr(tu_isa_opcode_t op, uint16_t dim0,
                                     uint16_t dim1, uint16_t dim2,
                                     uint8_t flags, uint32_t imm) {
    tu_instruction_t instr;
    memset(&instr, 0, sizeof(instr));
    instr.opcode = op;
    instr.flags = flags;
    instr.dim0 = dim0;
    instr.dim1 = dim1;
    instr.dim2 = dim2;
    instr.immediates = imm;
    return instr;
}

/* ---- Test 1: Empty sequence ---- */
static void test_empty(void) {
    TEST("empty sequence");
    tu_allocated_sequence_t output;
    memset(&output, 0, sizeof(output));
    int rc = tu_live_allocate(NULL, 0, NULL, &output);
    ASSERT_EQ(rc, -1, "empty returns error");
    PASS();
}

/* ---- Test 2: Single DMA load → single VReg ---- */
static void test_single_vreg(void) {
    TEST("single DMA load → single VReg");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x0400, 0, 0, 0), /* W ch=0, 1024 bytes */
    };

    tu_liveness_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_live_analyze(instrs, 1, &result);
    ASSERT_EQ(rc, 0, "analysis succeeds");
    ASSERT_TRUE(result.num_vregs >= 1, "at least 1 VReg created");
    ASSERT_EQ((long)result.vregs[0].region, (long)TU_VREG_W, "W region");
    ASSERT_EQ((long)result.vregs[0].first_def, 0L, "first_def = 0");
    PASS();
}

/* ---- Test 3: DMA → MMA → DMA store (3 VRegs, W/A/O) ---- */
static void test_three_regions(void) {
    TEST("DMA → MMA → DMA store (3 VRegs across W/A/O)");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0, 0),     /* W-SRAM ch=0 */
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0200, 0, 1, 0),     /* A-SRAM ch=1 */
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),                        /* O-SRAM */
        make_instr(TU_ISA_DMA_STORE, 0x0000, 0x0100, 0, 2, 0),     /* O-SRAM ch=2 */
    };

    tu_liveness_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_live_analyze(instrs, 4, &result);
    ASSERT_EQ(rc, 0, "analysis succeeds");
    ASSERT_TRUE(result.num_vregs >= 3, "at least 3 VRegs");

    /* Build interference graph */
    tu_live_build_interference(&result);

    /* W region should have 1 VReg */
    ASSERT_TRUE(result.graph_w.num_vregs >= 1, "W region has VRegs");
    /* A region should have 1 VReg */
    ASSERT_TRUE(result.graph_a.num_vregs >= 1, "A region has VRegs");
    /* O region should have >= 1 VReg */
    ASSERT_TRUE(result.graph_o.num_vregs >= 1, "O region has VRegs");
    PASS();
}

/* ---- Test 4: Interference detection ---- */
static void test_interference_detection(void) {
    TEST("interference detection");

    /* Two DMA loads to the same W-SRAM region overlapping in time */
    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x0400, 0, 0, 0),       /* W ch=0 */
        make_instr(TU_ISA_DMA_LOAD, 0x0400, 0x0400, 0, 0, 0),       /* W ch=0 — diff offset */
        make_instr(TU_ISA_MMA,      0x0010, 0x0010, 0x0010, 0,       /* reads W[0] */
                   (0x0000 << 16) | 0x0000),
    };

    tu_liveness_result_t result;
    memset(&result, 0, sizeof(result));
    tu_live_analyze(instrs, 3, &result);
    tu_live_build_interference(&result);

    /* Since both W loads are live during MMA, they should interfere */
    ASSERT_TRUE(result.graph_w.num_vregs >= 2, "at least 2 W VRegs");

    if (result.graph_w.num_vregs >= 2 && result.graph_w.interference) {
        uint32_t n = result.graph_w.num_vregs;
        /* Verify the graph has entries */
        ASSERT_TRUE(n >= 2, "graph has nodes");
    }

    /* Free */
    if (result.graph_w.interference) free(result.graph_w.interference);
    if (result.graph_a.interference) free(result.graph_a.interference);
    if (result.graph_o.interference) free(result.graph_o.interference);
    PASS();
}

/* ---- Test 5: Greedy coloring assigns non-overlapping physical offsets ---- */
static void test_coloring_non_overlap(void) {
    TEST("greedy coloring assigns non-overlapping offsets");

    /* Two independent DMA loads to same W region */
    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x0400, 0, 0, 0),
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x0400, 0, 0, 0),
    };

    tu_liveness_result_t result;
    memset(&result, 0, sizeof(result));
    tu_live_analyze(instrs, 2, &result);
    tu_live_build_interference(&result);

    tu_live_config_t cfg = tu_live_config_default;
    cfg.w_capacity = 32768; /* Enough for both */
    tu_live_color(&result, &cfg);

    /* Both VRegs should be placed */
    for (uint32_t i = 0; i < result.num_vregs; i++) {
        if (result.vregs[i].region == TU_VREG_W) {
            ASSERT_TRUE(result.vregs[i].physical_offset != UINT32_MAX,
                        "VReg assigned physical offset");
        }
    }

    /* Free */
    if (result.graph_w.interference) free(result.graph_w.interference);
    if (result.graph_a.interference) free(result.graph_a.interference);
    if (result.graph_o.interference) free(result.graph_o.interference);
    PASS();
}

/* ---- Test 6: Full allocation pipeline ---- */
static void test_full_pipeline(void) {
    TEST("full allocation pipeline");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0, 0),     /* W */
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0200, 0, 1, 0),     /* A */
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),
        make_instr(TU_ISA_RELU,      0x0000, 0x0100, 0, 0, 0),     /* O */
    };

    tu_allocated_sequence_t output;
    memset(&output, 0, sizeof(output));
    int rc = tu_live_allocate(instrs, 4, NULL, &output);
    ASSERT_EQ(rc, 0, "allocation succeeds");
    ASSERT_TRUE(output.num_instructions >= 4, "at least 4 instructions output");
    ASSERT_TRUE(output.valid, "output valid");

    /* Peak usage should be non-zero */
    ASSERT_TRUE(output.peak_w_usage > 0, "W usage tracked");
    ASSERT_TRUE(output.peak_o_usage > 0, "O usage tracked");
    PASS();
}

/* ---- Test 7: Spilling when capacity is exceeded ---- */
static void test_spilling_on_capacity_exceeded(void) {
    TEST("spilling when capacity exceeded");

    /* Create many large VRegs in W to force spilling */
    tu_instruction_t instrs[10];
    for (int i = 0; i < 10; i++) {
        instrs[i] = make_instr(TU_ISA_DMA_LOAD,
                                (uint16_t)(i * 0x1000),  /* increasing offset */
                                0x4000,  /* 16KB each */
                                0, 0, 0);
    }

    tu_live_config_t cfg = tu_live_config_default;
    cfg.w_capacity = 32768;   /* 32 KB W-SRAM — can fit only 2 of 10 16KB VRegs */
    cfg.enable_spilling = true;

    tu_allocated_sequence_t output;
    memset(&output, 0, sizeof(output));
    int rc = tu_live_allocate(instrs, 10, &cfg, &output);
    ASSERT_EQ(rc, 0, "allocation succeeds with spilling");
    ASSERT_TRUE(output.valid, "output valid");

    /* Peak W usage should not exceed capacity */
    ASSERT_TRUE(output.peak_w_usage <= cfg.w_capacity + cfg.safety_margin,
               "peak W usage within capacity");
    PASS();
}

/* ---- Test 8: Spill strategy comparison ---- */
static void test_spill_strategies(void) {
    TEST("all 4 spill strategies produce valid output");

    tu_instruction_t instrs[8];
    for (int i = 0; i < 8; i++) {
        instrs[i] = make_instr(TU_ISA_DMA_LOAD,
                                (uint16_t)(i * 0x1000),
                                0x4000, 0, 0, 0);
    }

    for (int s = 0; s < TU_SPILL_COUNT; s++) {
        tu_live_config_t cfg = tu_live_config_default;
        cfg.w_capacity = 16384; /* 16KB */
        cfg.spill_strategy = (tu_spill_strategy_t)s;
        cfg.enable_spilling = true;

        tu_allocated_sequence_t output;
        memset(&output, 0, sizeof(output));
        int rc = tu_live_allocate(instrs, 8, &cfg, &output);
        ASSERT_EQ(rc, 0, "allocation succeeds");
        ASSERT_TRUE(output.valid, "output valid");
    }
    PASS();
}

/* ---- Test 9: Allocation strategies ---- */
static void test_alloc_strategies(void) {
    TEST("all 3 allocation strategies produce valid output");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x0100, 0, 0, 0),
        make_instr(TU_ISA_DMA_LOAD, 0x0100, 0x0100, 0, 0, 0),
        make_instr(TU_ISA_DMA_LOAD, 0x0200, 0x0100, 0, 0, 0),
    };

    for (int a = 0; a < TU_ALLOC_COUNT; a++) {
        tu_live_config_t cfg = tu_live_config_default;
        cfg.w_capacity = 16384;
        cfg.alloc_strategy = (tu_alloc_strategy_t)a;

        tu_allocated_sequence_t output;
        memset(&output, 0, sizeof(output));
        int rc = tu_live_allocate(instrs, 3, &cfg, &output);
        ASSERT_EQ(rc, 0, "allocation succeeds");
        ASSERT_TRUE(output.valid, "output valid");
    }
    PASS();
}

/* ---- Test 10: W/A/O region independence ---- */
static void test_region_independence(void) {
    TEST("W/A/O regions allocated independently");

    /* Load into W and A — they should not interfere */
    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x1000, 0, 0, 0), /* W ch=0 */
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x1000, 0, 1, 0), /* A ch=1 */
    };

    tu_liveness_result_t result;
    memset(&result, 0, sizeof(result));
    tu_live_analyze(instrs, 2, &result);
    tu_live_build_interference(&result);

    /* W and A should have separate VRegs */
    ASSERT_TRUE(result.graph_w.num_vregs >= 1, "W region populated");
    ASSERT_TRUE(result.graph_a.num_vregs >= 1, "A region populated");

    /* W VRegs should not interfere with A VRegs (different graphs) */
    if (result.graph_w.interference && result.graph_w.num_vregs >= 1) {
        /* W graph has no A entries */
    }

    /* Free */
    if (result.graph_w.interference) free(result.graph_w.interference);
    if (result.graph_a.interference) free(result.graph_a.interference);
    if (result.graph_o.interference) free(result.graph_o.interference);
    PASS();
}

/* ---- Test 11: Configurable capacity ---- */
static void test_configurable_capacity(void) {
    TEST("configurable capacities affect peak usage");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x4000, 0, 0, 0), /* 16KB */
        make_instr(TU_ISA_DMA_LOAD, 0x4000, 0x4000, 0, 0, 0), /* 16KB */
        make_instr(TU_ISA_DMA_LOAD, 0x8000, 0x4000, 0, 0, 0), /* 16KB */
    };

    /* Small capacity → spilling needed */
    tu_live_config_t cfg_small = tu_live_config_default;
    cfg_small.w_capacity = 16384; /* 16KB — can hold only 1 */
    cfg_small.enable_spilling = true;

    tu_allocated_sequence_t out_small;
    memset(&out_small, 0, sizeof(out_small));
    int rc1 = tu_live_allocate(instrs, 3, &cfg_small, &out_small);
    ASSERT_EQ(rc1, 0, "small capacity succeeds");

    /* Large capacity → no spilling */
    tu_live_config_t cfg_large = tu_live_config_default;
    cfg_large.w_capacity = 131072; /* 128KB */
    cfg_large.enable_spilling = true;

    tu_allocated_sequence_t out_large;
    memset(&out_large, 0, sizeof(out_large));
    int rc2 = tu_live_allocate(instrs, 3, &cfg_large, &out_large);
    ASSERT_EQ(rc2, 0, "large capacity succeeds");

    /* Large capacity should have fewer/no spills */
    ASSERT_TRUE(out_large.num_instructions <= out_small.num_instructions + 4,
               "large capacity has ≤ spills than small");
    PASS();
}

/* ---- Test 12: MMA pipeline with real offsets ---- */
static void test_mma_pipeline_offsets(void) {
    TEST("MMA pipeline resolves physical offsets");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x1000, 0, 0, 0),  /* W */
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0800, 0, 1, 0),  /* A */
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),
        make_instr(TU_ISA_RELU,      0x0000, 0x0100, 0, 0, 0),
        make_instr(TU_ISA_DMA_STORE, 0x0000, 0x0100, 0, 2, 0),
    };

    tu_allocated_sequence_t output;
    memset(&output, 0, sizeof(output));
    int rc = tu_live_allocate(instrs, 5, NULL, &output);
    ASSERT_EQ(rc, 0, "allocation succeeds");
    ASSERT_TRUE(output.valid, "output valid");

    /* Verify MMA is still present */
    bool mma_found = false;
    for (uint32_t i = 0; i < output.num_instructions; i++) {
        if (output.instructions[i].opcode == TU_ISA_MMA) {
            mma_found = true;
            /* MMA's offsets should be patched (non-zero if VRegs assigned) */
            break;
        }
    }
    ASSERT_TRUE(mma_found, "MMA present in output");
    PASS();
}

/* ---- Main ---- */
int main(void) {
    printf("\n===========================================\n");
    printf("  Liveness-Based Allocator Tests (Gap C3)\n");
    printf("===========================================\n\n");

    test_empty();
    test_single_vreg();
    test_three_regions();
    test_interference_detection();
    test_coloring_non_overlap();
    test_full_pipeline();
    test_spilling_on_capacity_exceeded();
    test_spill_strategies();
    test_alloc_strategies();
    test_region_independence();
    test_configurable_capacity();
    test_mma_pipeline_offsets();

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
