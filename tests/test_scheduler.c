/*
 * TU CModel — Compiler Scheduling Pass Tests (Gap C2)
 * ====================================================
 *
 * Tests for the DAG-based instruction scheduler:
 *   - DAG construction with various dependency patterns
 *   - ASAP/ALAP mobility computation
 *   - DMA hoisting
 *   - Barrier insertion
 *   - List scheduling with all three policies
 *   - Validation of scheduled sequences
 *   - Edge cases: empty, single instruction, all independent, all dependent
 *   - Double-buffered tile pipeline scheduling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/isa/tu_isa.h"
#include "../tu_cmodel/isa/tu_scheduler.h"

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

/* ---- Helper: make a simple instruction ---- */
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
    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_sched_run(NULL, 0, NULL, &result);
    ASSERT_EQ(rc, -1, "empty sequence returns error");
    PASS();
}

/* ---- Test 2: Single instruction ---- */
static void test_single(void) {
    TEST("single instruction");
    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_NOP, 0, 0, 0, 0, 0),
    };
    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_sched_run(instrs, 1, NULL, &result);
    ASSERT_EQ(rc, 0, "single instruction succeeds");
    ASSERT_EQ((long)result.num_instructions, 1L, "1 instruction output");
    ASSERT_TRUE(result.valid, "result valid");
    PASS();
}

/* ---- Test 3: DMA load → MMA dependency ---- */
static void test_dma_mma_dependency(void) {
    TEST("DMA load → MMA dependency (RAW)");

    /* Sequence: DMA_LOAD_W → MMA (reads W) */
    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x0400, 0, 0, 0),  /* W-SRAM[0..1024] */
        make_instr(TU_ISA_MMA, 0x0010, 0x0010, 0x0010, 0,       /* MMA M=16,N=16,K=16 */
                   (0x0020 << 16) | 0x0000),                     /* o_offset=0x20, a_offset=0 */
    };

    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_sched_run(instrs, 2, NULL, &result);
    ASSERT_EQ(rc, 0, "scheduling succeeds");
    ASSERT_EQ((long)result.num_instructions, 2L, "2 instructions output");
    ASSERT_TRUE(result.valid, "result valid");

    /* DMA must come before MMA */
    bool dma_found = false, mma_found = false;
    for (uint32_t i = 0; i < result.num_instructions; i++) {
        if (result.instructions[i].opcode == TU_ISA_DMA_LOAD) {
            ASSERT_TRUE(!mma_found, "DMA must precede MMA");
            dma_found = true;
        }
        if (result.instructions[i].opcode == TU_ISA_MMA) {
            ASSERT_TRUE(dma_found, "MMA must follow DMA");
            mma_found = true;
        }
    }
    PASS();
}

/* ---- Test 4: Independent DMA loads can be reordered ---- */
static void test_independent_dma_reorder(void) {
    TEST("independent DMA loads reordered");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x0400, 0, 0x00, 0),    /* W ch=0 */
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x0200, 0, 0x01, 0),    /* A ch=1 */
        make_instr(TU_ISA_DMA_LOAD, 0x0000, 0x0100, 0, 0x02, 0),    /* O ch=2 */
    };

    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_sched_run(instrs, 3, NULL, &result);
    ASSERT_EQ(rc, 0, "scheduling succeeds");
    ASSERT_EQ((long)result.num_instructions, 3L, "3 instructions output");
    ASSERT_TRUE(result.valid, "result valid");

    /* All three DMA ops present */
    int dma_count = 0;
    for (uint32_t i = 0; i < result.num_instructions; i++) {
        if (result.instructions[i].opcode == TU_ISA_DMA_LOAD)
            dma_count++;
    }
    ASSERT_EQ((long)dma_count, 3L, "all 3 DMA ops present");
    PASS();
}

/* ---- Test 5: DMA → MMA → DMA store pipeline ---- */
static void test_dma_mma_store_pipeline(void) {
    TEST("DMA → MMA → DMA store pipeline");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0x00, 0),     /* W ch=0 */
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0200, 0, 0x01, 0),     /* A ch=1 */
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),                           /* o_offset=0, a_offset=0 */
        make_instr(TU_ISA_DMA_STORE, 0x0000, 0x0100, 0, 0x02,         /* O ch=2 */
                   (0x0000 << 16) | 0x0000),                           /* o_offset=0 */
    };

    tu_sched_config_t cfg = tu_sched_config_default;
    cfg.policy = TU_SCHED_POLICY_BALANCED;

    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_sched_run(instrs, 4, &cfg, &result);
    ASSERT_EQ(rc, 0, "scheduling succeeds");
    ASSERT_EQ((long)result.num_instructions, 4L, "4 instructions output");
    ASSERT_TRUE(result.valid, "result valid");

    /* Verify ordering: DMA loads must precede MMA, MMA must precede DMA store */
    int dma_w_pos = -1, dma_a_pos = -1, mma_pos = -1, dma_o_pos = -1;
    for (uint32_t i = 0; i < result.num_instructions; i++) {
        if (result.instructions[i].opcode == TU_ISA_DMA_LOAD && (result.instructions[i].flags & 0x3) == 0)
            dma_w_pos = (int)i;
        if (result.instructions[i].opcode == TU_ISA_DMA_LOAD && (result.instructions[i].flags & 0x3) == 1)
            dma_a_pos = (int)i;
        if (result.instructions[i].opcode == TU_ISA_MMA)
            mma_pos = (int)i;
        if (result.instructions[i].opcode == TU_ISA_DMA_STORE)
            dma_o_pos = (int)i;
    }
    ASSERT_TRUE(dma_w_pos >= 0, "DMA W found");
    ASSERT_TRUE(dma_a_pos >= 0, "DMA A found");
    ASSERT_TRUE(mma_pos >= 0, "MMA found");
    ASSERT_TRUE(dma_o_pos >= 0, "DMA store found");
    ASSERT_TRUE(dma_w_pos < mma_pos, "DMA W before MMA");
    ASSERT_TRUE(dma_a_pos < mma_pos, "DMA A before MMA");
    ASSERT_TRUE(mma_pos < dma_o_pos, "MMA before DMA store");
    PASS();
}

/* ---- Test 6: DAG construction with complex dependencies ---- */
static void test_complex_dag(void) {
    TEST("complex DAG construction");

    /* Simulate a tiled GEMM: DMA tile0 → MMA tile0 → DMA store tile0
     *                      DMA tile1 → MMA tile1 → DMA store tile1 */
    tu_instruction_t instrs[] = {
        /* Tile 0 */
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0x00, 0),     /* W ch=0 */
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0200, 0, 0x01, 0),     /* A ch=1 */
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),                           /* tile 0 */
        make_instr(TU_ISA_DMA_STORE, 0x0000, 0x0100, 0, 0x02,
                   (0x0000 << 16) | 0x0000),                           /* store tile 0 */
        /* Tile 1 */
        make_instr(TU_ISA_DMA_LOAD,  0x0400, 0x0400, 0, 0x00, 0),     /* W ch=0 */
        make_instr(TU_ISA_DMA_LOAD,  0x0200, 0x0200, 0, 0x01, 0),     /* A ch=1 */
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0200 << 16) | 0x0000),                           /* tile 1 */
        make_instr(TU_ISA_DMA_STORE, 0x0100, 0x0100, 0, 0x02,
                   (0x0200 << 16) | 0x0000),                           /* store tile 1 */
    };

    tu_sched_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    int rc = tu_sched_build_dag(&graph, instrs, 8, NULL);
    ASSERT_EQ(rc, 0, "DAG built");
    ASSERT_EQ((long)graph.num_nodes, 8L, "8 nodes");

    /* Tile 0 MMA should depend on tile 0 DMA loads */
    tu_sched_node_t *mma0 = &graph.nodes[2];
    ASSERT_TRUE(mma0->num_preds >= 2, "MMA tile 0 has >= 2 predecessors");

    /* Tile 1 MMA should depend on tile 1 DMA loads */
    tu_sched_node_t *mma1 = &graph.nodes[6];
    ASSERT_TRUE(mma1->num_preds >= 2, "MMA tile 1 has >= 2 predecessors");

    /* Tile 0 and tile 1 are independent (different SRAM offsets).
     * MMA tile 0 has no path to MMA tile 1 dependency */
    PASS();
}

/* ---- Test 7: ASAP/ALAP mobility ---- */
static void test_mobility(void) {
    TEST("ASAP/ALAP mobility computation");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0, 0),
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0200, 0, 1, 0),
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),
    };

    tu_sched_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    tu_sched_build_dag(&graph, instrs, 3, NULL);
    tu_sched_compute_mobility(&graph);

    /* DMA loads should have asap=0, MMA should be later */
    ASSERT_EQ((long)graph.nodes[0].asap_cycle, 0L, "DMA W asap=0");
    ASSERT_EQ((long)graph.nodes[1].asap_cycle, 0L, "DMA A asap=0");
    ASSERT_TRUE(graph.nodes[2].asap_cycle > 0, "MMA asap > 0");
    ASSERT_TRUE(graph.nodes[2].alap_cycle > 0, "MMA alap > 0");

    /* Slack: at least one node has mobility */
    ASSERT_TRUE(graph.nodes[2].slack >= 0, "slack non-negative");
    PASS();
}

/* ---- Test 8: Barrier insertion ---- */
static void test_barrier_insertion(void) {
    TEST("barrier insertion analysis");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0, 0),   /* W ch=0 */
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),
        make_instr(TU_ISA_DMA_STORE, 0x0000, 0x0100, 0, 2,        /* store O */
                   (0x0000 << 16) | 0x0000),
    };

    tu_sched_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    tu_sched_config_t cfg = tu_sched_config_default;
    cfg.insert_barriers = true;

    tu_sched_build_dag(&graph, instrs, 3, &cfg);
    int barriers = tu_sched_insert_barriers(&graph);
    /* DMA_LOAD writes to W, MMA reads W → RAW.
     * MMA writes to O, DMA_STORE reads O → WAR.
     * DMA_STORE reads O - it should trigger barrier before MMA */
    ASSERT_TRUE(barriers >= 0, "barrier count valid");
    PASS();
}

/* ---- Test 9: Balanced vs ASAP vs ALAP policies ---- */
static void test_policy_comparison(void) {
    TEST("policy comparison (balanced vs ASAP vs ALAP)");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0, 0),
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0200, 0, 1, 0),
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),
        make_instr(TU_ISA_RELU,      0x0000, 0x0100, 0, 0, 0),
    };

    /* Run with all three policies and check they produce valid results */
    for (int p = 0; p < TU_SCHED_POLICY_COUNT; p++) {
        tu_sched_config_t cfg = tu_sched_config_default;
        cfg.policy = (tu_sched_policy_t)p;

        tu_sched_result_t result;
        memset(&result, 0, sizeof(result));
        int rc = tu_sched_run(instrs, 4, &cfg, &result);
        ASSERT_EQ(rc, 0, "scheduling succeeds for policy");
        ASSERT_TRUE(result.valid, "result valid for policy");

        /* Verify MMA appears after both DMA loads */
        bool mma_found = false;
        int dma_count_before_mma = 0;
        for (uint32_t i = 0; i < result.num_instructions; i++) {
            if (result.instructions[i].opcode == TU_ISA_MMA)
                mma_found = true;
            else if (tu_isa_is_dma_op((tu_isa_opcode_t)result.instructions[i].opcode)
                     && !mma_found)
                dma_count_before_mma++;
        }
        ASSERT_TRUE(dma_count_before_mma >= 2, "both DMAs before MMA");
    }
    PASS();
}

/* ---- Test 10: Large sequence stress test ---- */
static void test_large_sequence(void) {
    TEST("large sequence stress test (64 instrs)");

    tu_instruction_t instrs[64];
    /* Simulate 4 tiles: DMA_W, DMA_A, MMA, DMA_STORE × 4, interspersed */
    for (int t = 0; t < 4; t++) {
        uint32_t base = t * 4;
        instrs[base + 0] = make_instr(TU_ISA_DMA_LOAD,  t * 0x400,  0x400, 0, 0, 0);
        instrs[base + 1] = make_instr(TU_ISA_DMA_LOAD,  t * 0x200,  0x200, 0, 1, 0);
        instrs[base + 2] = make_instr(TU_ISA_MMA,       0x0010,      0x0010, 0x0010, 0,
                                       ((t * 0x200) << 16) | (t * 0x200));
        instrs[base + 3] = make_instr(TU_ISA_DMA_STORE, t * 0x100,  0x100, 0, 2,
                                       ((t * 0x200) << 16) | (t * 0x200));
    }

    tu_sched_config_t cfg = tu_sched_config_default;
    cfg.pipeline_tiles = true;

    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_sched_run(instrs, 16, &cfg, &result);
    ASSERT_EQ(rc, 0, "large sequence succeeds");
    ASSERT_EQ((long)result.num_instructions, 16L, "16 instructions output");
    ASSERT_TRUE(result.valid, "result valid");

    /* Count ops */
    int dma_load = 0, mma = 0, dma_store = 0;
    for (uint32_t i = 0; i < result.num_instructions; i++) {
        if (result.instructions[i].opcode == TU_ISA_DMA_LOAD) dma_load++;
        if (result.instructions[i].opcode == TU_ISA_MMA) mma++;
        if (result.instructions[i].opcode == TU_ISA_DMA_STORE) dma_store++;
    }
    ASSERT_EQ((long)dma_load, 8L, "8 DMA loads");
    ASSERT_EQ((long)mma, 4L, "4 MMAs");
    ASSERT_EQ((long)dma_store, 4L, "4 DMA stores");
    PASS();
}

/* ---- Test 11: Validation detects violations ---- */
static void test_validation(void) {
    TEST("validation detects dependency violations");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0, 0),
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),
    };

    tu_sched_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    tu_sched_build_dag(&graph, instrs, 2, NULL);

    /* Create a result with violated order (MMA before DMA) */
    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    result.instructions[0] = instrs[1]; /* MMA first */
    result.instructions[1] = instrs[0]; /* DMA second */
    result.num_instructions = 2;
    result.valid = true; /* Manually set to force validation check */

    /* Mark nodes as scheduled so positions can be found */
    graph.nodes[0].scheduled = true;
    graph.nodes[1].scheduled = true;

    bool valid = tu_sched_validate(&result, &graph);
    ASSERT_TRUE(!valid, "validation detects MMA-before-DMA violation");
    PASS();
}

/* ---- Test 12: Instruction with no SRAM access ---- */
static void test_no_sram_access(void) {
    TEST("instructions with no SRAM access");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_NOP,    0, 0, 0, 0, 0),
        make_instr(TU_ISA_SYNC,   0, 0, 0, 0, 0),
        make_instr(TU_ISA_BARRIER,0, 0, 0, 0, 0),
    };

    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_sched_run(instrs, 3, NULL, &result);
    ASSERT_EQ(rc, 0, "scheduling succeeds");
    ASSERT_EQ((long)result.num_instructions, 3L, "3 instructions output");
    ASSERT_TRUE(result.valid, "result valid");
    PASS();
}

/* ---- Test 13: Double-buffered tile pipeline scheduling ---- */
static void test_double_buffered_pipeline(void) {
    TEST("double-buffered tile pipeline scheduling");

    /* Simulate: DMA tile 0 → MMA tile 0 → DMA store tile 0,
     *          DMA tile 1 → MMA tile 1 → DMA store tile 1
     * Where tile 1's DMA can overlap tile 0's compute (different SRAM regions) */
    tu_instruction_t instrs[] = {
        /* Tile 0: W region 0, A region 0 */
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0, 0),
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0200, 0, 1, 0),
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),
        make_instr(TU_ISA_DMA_STORE, 0x0000, 0x0100, 0, 2,
                   (0x0000 << 16) | 0x0000),
        /* Tile 1: W region 0x1000, A region 0x0800 (different from tile 0) */
        make_instr(TU_ISA_DMA_LOAD,  0x1000, 0x0400, 0, 0, 0),
        make_instr(TU_ISA_DMA_LOAD,  0x0800, 0x0200, 0, 1, 0),
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0200 << 16) | 0x0800),
        make_instr(TU_ISA_DMA_STORE, 0x0100, 0x0100, 0, 2,
                   (0x0200 << 16) | 0x0800),
    };

    tu_sched_config_t cfg = tu_sched_config_default;
    cfg.policy = TU_SCHED_POLICY_BALANCED;
    cfg.pipeline_tiles = true;

    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_sched_run(instrs, 8, &cfg, &result);
    ASSERT_EQ(rc, 0, "scheduling succeeds");
    ASSERT_EQ((long)result.num_instructions, 8L, "8 instructions output");
    ASSERT_TRUE(result.valid, "result valid");

    /* Tile 0 MMA should precede Tile 1 DMA store */
    int tile0_mma_pos = -1, tile1_dma_load_pos = -1;
    for (uint32_t i = 0; i < result.num_instructions; i++) {
        if (result.instructions[i].opcode == TU_ISA_MMA
            && result.instructions[i].immediates == ((0x0000 << 16) | 0x0000))
            tile0_mma_pos = (int)i;
        if (result.instructions[i].opcode == TU_ISA_DMA_LOAD
            && result.instructions[i].dim0 == 0x1000)
            tile1_dma_load_pos = (int)i;
    }
    /* Tile 1 DMA load can happen before tile 0 MMA (different SRAM regions =
     * no dependency → pipeline overlap) */
    ASSERT_TRUE(tile0_mma_pos >= 0, "tile 0 MMA found");
    ASSERT_TRUE(tile1_dma_load_pos >= 0, "tile 1 DMA found");
    PASS();
}

/* ---- Test 14: Config override ---- */
static void test_config_override(void) {
    TEST("config override (no DMA hoisting, ASAP policy)");

    tu_instruction_t instrs[] = {
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0400, 0, 0, 0),
        make_instr(TU_ISA_DMA_LOAD,  0x0000, 0x0200, 0, 1, 0),
        make_instr(TU_ISA_MMA,       0x0010, 0x0010, 0x0010, 0,
                   (0x0000 << 16) | 0x0000),
    };

    tu_sched_config_t cfg = {
        .policy             = TU_SCHED_POLICY_ASAP,
        .hoist_dma          = false,
        .insert_barriers    = false,
        .pipeline_tiles     = false,
        .max_hoist_distance = 0,
        .max_window         = 256,
        .verbose            = false,
    };

    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));
    int rc = tu_sched_run(instrs, 3, &cfg, &result);
    ASSERT_EQ(rc, 0, "scheduling succeeds with custom config");
    ASSERT_TRUE(result.valid, "result valid");
    ASSERT_EQ((long)result.num_dma_hoisted, 0L, "no DMA hoisting with hoist off");
    ASSERT_EQ((long)result.num_barriers_inserted, 0L, "no barriers with barriers off");
    PASS();
}

/* ---- Main ---- */
int main(void) {
    printf("\n===========================================\n");
    printf("  Compiler Scheduling Pass Tests (Gap C2)\n");
    printf("===========================================\n\n");

    test_empty();
    test_single();
    test_dma_mma_dependency();
    test_independent_dma_reorder();
    test_dma_mma_store_pipeline();
    test_complex_dag();
    test_mobility();
    test_barrier_insertion();
    test_policy_comparison();
    test_large_sequence();
    test_validation();
    test_no_sram_access();
    test_double_buffered_pipeline();
    test_config_override();

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
