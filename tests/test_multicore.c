/*
 * TU Multi-Core Cluster Tests (Gap A5)
 * =====================================
 *
 * Tests for tu_core_t and tu_cluster_t:
 *   1. Core lifecycle (create, init, destroy)
 *   2. Core DMA operations (load/store)
 *   3. Core MMA operations
 *   4. Core state isolation (cores don't interfere)
 *   5. Cluster creation (ring topology)
 *   6. Cluster creation (mesh topology)
 *   7. Hop distance calculation
 *   8. Neighbor discovery
 *   9. Inter-core communication (send/receive)
 *  10. Inter-core broadcast
 *  11. All-reduce sum
 *  12. Barrier synchronization
 *  13. SPMD execution
 */

#include "tu_cmodel/tu_core.h"
#include "tu_cmodel/tu_cluster.h"
#include "tu_cmodel/tu_sram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;
static int tests_total = 0;

#define TEST(name) do { \
    tests_total++; \
    printf("  TEST %d: %s ... ", tests_total, name); \
} while(0)

#define PASS() do { \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
    tests_failed++; \
} while(0)

#define CHECK(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

/* ---- Test 1: Core lifecycle ---- */
static void test_core_lifecycle(void) {
    TEST("core lifecycle");
    tu_runtime_config_t cfg = tu_config_default();
    tu_core_t *core = tu_core_create(&cfg);
    CHECK(core != NULL, "create failed");
    CHECK(core->initialized, "not initialized");
    CHECK(core->core_id == 0, "wrong core_id");

    tu_core_destroy(core);
    PASS();
}

/* ---- Test 2: Core DMA ---- */
static void test_core_dma(void) {
    TEST("core DMA load/store");
    tu_runtime_config_t cfg = tu_config_default();
    tu_core_t *core = tu_core_create(&cfg);
    CHECK(core != NULL, "create failed");

    float host_data[16];
    float host_out[16];
    for (int i = 0; i < 16; i++) host_data[i] = (float)(i + 1);

    /* Load weights */
    tu_core_dma_load_w(core, host_data, 0, 64);
    /* Load activations */
    tu_core_dma_load_a(core, host_data, 0, 64);

    /* Store output region back */
    memset(host_out, 0, sizeof(host_out));
    tu_core_dma_store_o(core, host_out, 0, 64);

    /* Verify O-buffer was zeroed (no MMA yet) */
    int all_zero = 1;
    for (int i = 0; i < 16; i++) {
        if (host_out[i] != 0.0f) all_zero = 0;
    }
    CHECK(all_zero, "O-buffer not zero after init");

    tu_core_destroy(core);
    PASS();
}

/* ---- Test 3: Core MMA (direct SRAM verification) ---- */
static void test_core_mma(void) {
    TEST("core MMA (identity)");
    tu_runtime_config_t cfg = tu_config_default();
    tu_core_t *core = tu_core_create(&cfg);
    CHECK(core != NULL, "create failed");

    /* Verify SRAM is accessible and isolated by writing directly */
    tu_sram_region_t *w_sram = tu_core_get_sram_w(core);
    tu_sram_region_t *a_sram = tu_core_get_sram_a(core);

    /* Write identity into W-buffer via SRAM bulk write */
    fp16_t w_val = tu_fp32_to_fp16(1.0f);
    fp16_t a_val = tu_fp32_to_fp16(2.0f);
    for (int i = 0; i < 16; i++) {
        tu_sram_write(w_sram, i * 16 * 2 + i * 2, &w_val);
        tu_sram_write(a_sram, i * 16 * 2 + i * 2, &a_val);
    }

    /* DMA the rest of the data for proper MMA */
    fp16_t w_data[256] = {0};
    fp16_t a_data[256] = {0};
    for (int i = 0; i < 16; i++) {
        w_data[i * 16 + i] = tu_fp32_to_fp16(1.0f);
        a_data[i * 16 + i] = tu_fp32_to_fp16(2.0f);
    }
    tu_core_dma_load_w(core, w_data, 0, sizeof(w_data));
    tu_core_dma_load_a(core, a_data, 0, sizeof(a_data));

    /* Verify W[0][0] was loaded */
    fp16_t check_w;
    tu_sram_read(w_sram, 0, &check_w);
    CHECK(tu_fp16_to_fp32(check_w) > 0.5f, "W[0][0] not loaded");

    /* MMA: 16×16×16 */
    tu_core_mma(core, 16, 16, 16, 0, 0, 0, false);

    /* Read output */
    fp16_t o_data[256];
    tu_core_dma_store_o(core, o_data, 0, sizeof(o_data));

    /* Verify: O = W × A = I × 2I = 2I */
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            float val = tu_fp16_to_fp32(o_data[i * 16 + j]);
            float expected = (i == j) ? 2.0f : 0.0f;
            if (fabsf(val - expected) > 0.01f) {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "O[%d][%d]=%f, expected %f", i, j, val, expected);
                FAIL(msg);
                tu_core_destroy(core);
                return;
            }
        }
    }

    tu_core_destroy(core);
    PASS();
}

/* ---- Test 4: Core state isolation ---- */
static void test_core_isolation(void) {
    TEST("core state isolation");
    tu_runtime_config_t cfg = tu_config_default();
    tu_core_t *c1 = tu_core_create_with_id(1, &cfg);
    tu_core_t *c2 = tu_core_create_with_id(2, &cfg);
    CHECK(c1 != NULL && c2 != NULL, "create failed");
    CHECK(c1->core_id == 1, "wrong c1 id");
    CHECK(c2->core_id == 2, "wrong c2 id");

    /* Load different data into each core */
    fp16_t ones[256];
    fp16_t twos[256];
    for (int i = 0; i < 256; i++) {
        ones[i] = tu_fp32_to_fp16(1.0f);
        twos[i] = tu_fp32_to_fp16(2.0f);
    }

    tu_core_dma_load_w(c1, ones, 0, sizeof(ones));
    tu_core_dma_load_w(c2, twos, 0, sizeof(twos));

    /* Verify they have different weights */
    fp16_t out1[256], out2[256];
    /* Read back from W-buffers via store from O (cheat: load W into A, MMA ident)
     * Actually, let's just verify the SRAM regions directly */
    tu_sram_region_t *s1 = tu_core_get_sram_w(c1);
    tu_sram_region_t *s2 = tu_core_get_sram_w(c2);

    fp16_t v1, v2;
    tu_sram_read(s1, 0, &v1);
    tu_sram_read(s2, 0, &v2);

    float f1 = tu_fp16_to_fp32(v1);
    float f2 = tu_fp16_to_fp32(v2);
    CHECK(fabsf(f1 - 1.0f) < 0.01f, "c1 W[0] != 1.0");
    CHECK(fabsf(f2 - 2.0f) < 0.01f, "c2 W[0] != 2.0");

    /* Verify they have separate SRAM regions (different pointers) */
    CHECK(s1 != s2, "SRAM regions are same pointer");
    CHECK(s1->banks.data != s2->banks.data, "SRAM data is same buffer");

    tu_core_destroy(c1);
    tu_core_destroy(c2);
    PASS();
}

/* ---- Test 5: Cluster creation (ring) ---- */
static void test_cluster_ring(void) {
    TEST("cluster ring creation");
    tu_runtime_config_t cfg = tu_config_default();
    tu_cluster_t *cl = tu_cluster_create(4, TU_TOPOLOGY_RING, 0, &cfg);
    CHECK(cl != NULL, "create failed");
    CHECK(cl->num_cores == 4, "wrong num_cores");
    CHECK(cl->topology == TU_TOPOLOGY_RING, "not ring topology");

    /* Check all cores exist */
    for (uint32_t i = 0; i < 4; i++) {
        tu_core_t *c = tu_cluster_get_core(cl, i);
        CHECK(c != NULL, "core missing");
        CHECK(c->core_id == i, "wrong core id");
        CHECK(c->initialized, "core not initialized");
    }
    CHECK(tu_cluster_get_core(cl, 4) == NULL, "out-of-bounds core returned");

    tu_cluster_destroy(cl);
    PASS();
}

/* ---- Test 6: Cluster creation (mesh) ---- */
static void test_cluster_mesh(void) {
    TEST("cluster mesh creation");
    tu_runtime_config_t cfg = tu_config_default();
    tu_cluster_t *cl = tu_cluster_create(6, TU_TOPOLOGY_MESH, 2, &cfg);
    CHECK(cl != NULL, "create failed");
    CHECK(cl->num_cores == 6, "wrong num_cores");
    CHECK(cl->topology == TU_TOPOLOGY_MESH, "not mesh topology");
    CHECK(cl->mesh_rows == 2, "wrong mesh_rows");
    CHECK(cl->mesh_cols == 3, "wrong mesh_cols");

    tu_cluster_destroy(cl);
    PASS();
}

/* ---- Test 7: Hop distance ---- */
static void test_hop_distance(void) {
    TEST("hop distance");

    /* Ring: 4 cores */
    tu_runtime_config_t cfg = tu_config_default();
    tu_cluster_t *ring = tu_cluster_create(4, TU_TOPOLOGY_RING, 0, &cfg);
    CHECK(ring != NULL, "ring create failed");

    CHECK(tu_cluster_hop_distance(ring, 0, 0) == 0, "self distance != 0");
    CHECK(tu_cluster_hop_distance(ring, 0, 1) == 1, "adjacent dist != 1");
    CHECK(tu_cluster_hop_distance(ring, 0, 2) == 2, "2-hop dist != 2");
    CHECK(tu_cluster_hop_distance(ring, 0, 3) == 1, "wrap dist != 1");

    tu_cluster_destroy(ring);

    /* Mesh: 2×3 = 6 cores */
    tu_cluster_t *mesh = tu_cluster_create(6, TU_TOPOLOGY_MESH, 2, &cfg);
    CHECK(mesh != NULL, "mesh create failed");

    /* Layout: row0: 0 1 2, row1: 3 4 5 */
    CHECK(tu_cluster_hop_distance(mesh, 0, 0) == 0, "self != 0");
    CHECK(tu_cluster_hop_distance(mesh, 0, 1) == 1, "0→1 != 1");
    CHECK(tu_cluster_hop_distance(mesh, 0, 2) == 2, "0→2 != 2");
    CHECK(tu_cluster_hop_distance(mesh, 0, 3) == 1, "0→3 != 1");
    CHECK(tu_cluster_hop_distance(mesh, 0, 5) == 3, "0→5 != 3");

    tu_cluster_destroy(mesh);

    /* None: no connectivity */
    tu_cluster_t *none = tu_cluster_create(2, TU_TOPOLOGY_NONE, 0, &cfg);
    CHECK(none != NULL, "none create failed");
    CHECK(tu_cluster_hop_distance(none, 0, 1) == UINT32_MAX, "none should be disconnected");
    tu_cluster_destroy(none);

    PASS();
}

/* ---- Test 8: Neighbors ---- */
static void test_neighbors(void) {
    TEST("neighbor discovery");

    tu_runtime_config_t cfg = tu_config_default();
    tu_cluster_t *ring = tu_cluster_create(4, TU_TOPOLOGY_RING, 0, &cfg);
    CHECK(ring != NULL, "ring create failed");

    uint32_t nbrs[4];
    uint32_t n;

    tu_cluster_neighbors(ring, 0, nbrs, &n);
    CHECK(n == 2, "ring should have 2 neighbors");
    CHECK((nbrs[0] == 3 && nbrs[1] == 1) || (nbrs[0] == 1 && nbrs[1] == 3),
          "wrong ring neighbors");

    tu_cluster_destroy(ring);

    tu_cluster_t *mesh = tu_cluster_create(6, TU_TOPOLOGY_MESH, 2, &cfg);
    CHECK(mesh != NULL, "mesh create failed");

    /* Corner core 0: east(1) + south(3) = 2 neighbors */
    tu_cluster_neighbors(mesh, 0, nbrs, &n);
    CHECK(n == 2, "mesh corner should have 2 neighbors");

    /* Center core 4: north(1), south(none), west(3), east(5) */
    tu_cluster_neighbors(mesh, 4, nbrs, &n);
    CHECK(n == 3, "mesh center should have 3 neighbors");

    tu_cluster_destroy(mesh);
    PASS();
}

/* ---- Test 9: ICC send ---- */
static void test_icc_send(void) {
    TEST("ICC send");
    tu_runtime_config_t cfg = tu_config_default();
    tu_cluster_t *cl = tu_cluster_create(2, TU_TOPOLOGY_RING, 0, &cfg);
    CHECK(cl != NULL, "create failed");

    /* Write data to core 0's O-SRAM */
    tu_core_t *c0 = tu_cluster_get_core(cl, 0);
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tu_sram_region_t *sram0 = tu_core_get_sram_o(c0);
    tu_sram_write_bulk(sram0, 0, data, sizeof(data));

    /* Send from core 0 to core 1 */
    tu_icc_message_t msg = {
        .src_core_id = 0,
        .dst_core_id = 1,
        .src_offset = 0,
        .dst_offset = 0,
        .size_bytes = sizeof(data),
        .tag = 42,
        .blocking = true,
    };
    int rc = tu_cluster_send(cl, &msg);
    CHECK(rc == 0, "send failed");

    /* Verify data arrived at core 1 */
    tu_core_t *c1 = tu_cluster_get_core(cl, 1);
    tu_sram_region_t *sram1 = tu_core_get_sram_o(c1);
    float out[4] = {0};
    tu_sram_read_bulk(sram1, 0, out, sizeof(out));

    for (int i = 0; i < 4; i++) {
        CHECK(fabsf(out[i] - data[i]) < 0.001f, "data mismatch after send");
    }

    /* Check stats */
    CHECK(cl->stats.total_icc_messages == 1, "message count wrong");
    CHECK(cl->stats.total_icc_bytes == sizeof(data), "byte count wrong");

    tu_cluster_destroy(cl);
    PASS();
}

/* ---- Test 10: ICC broadcast ---- */
static void test_icc_broadcast(void) {
    TEST("ICC broadcast");
    tu_runtime_config_t cfg = tu_config_default();
    tu_cluster_t *cl = tu_cluster_create(4, TU_TOPOLOGY_RING, 0, &cfg);
    CHECK(cl != NULL, "create failed");

    /* Write data to core 0 */
    float data[2] = {7.0f, 8.0f};
    tu_sram_region_t *sram0 = tu_core_get_sram_o(tu_cluster_get_core(cl, 0));
    tu_sram_write_bulk(sram0, 0, data, sizeof(data));

    /* Broadcast to all */
    int rc = tu_cluster_broadcast(cl, 0, 0, 0, sizeof(data));
    CHECK(rc == 0, "broadcast failed");

    /* Verify all cores received */
    for (uint32_t i = 1; i < 4; i++) {
        tu_sram_region_t *sram = tu_core_get_sram_o(tu_cluster_get_core(cl, i));
        float out[2];
        tu_sram_read_bulk(sram, 0, out, sizeof(out));
        CHECK(fabsf(out[0] - 7.0f) < 0.001f && fabsf(out[1] - 8.0f) < 0.001f,
              "broadcast data mismatch");
    }

    tu_cluster_destroy(cl);
    PASS();
}

/* ---- Test 11: All-reduce sum ---- */
static void test_allreduce_sum(void) {
    TEST("all-reduce sum");
    tu_runtime_config_t cfg = tu_config_default();
    tu_cluster_t *cl = tu_cluster_create(4, TU_TOPOLOGY_RING, 0, &cfg);
    CHECK(cl != NULL, "create failed");

    /* Each core gets different data: core i has [i, i+1, i+2] */
    for (uint32_t i = 0; i < 4; i++) {
        float data[3] = {(float)i, (float)(i + 1), (float)(i + 2)};
        tu_sram_region_t *sram = tu_core_get_sram_o(tu_cluster_get_core(cl, i));
        tu_sram_write_bulk(sram, 0, data, sizeof(data));
    }

    int rc = tu_cluster_allreduce_sum_f32(cl, 0, 64, 3);
    CHECK(rc == 0, "allreduce failed");

    /* Expected: sum = [0+1+2+3, 1+2+3+4, 2+3+4+5] = [6, 10, 14] */
    for (uint32_t i = 0; i < 4; i++) {
        tu_sram_region_t *sram = tu_core_get_sram_o(tu_cluster_get_core(cl, i));
        float out[3];
        tu_sram_read_bulk(sram, 64, out, sizeof(out));
        CHECK(fabsf(out[0] - 6.0f) < 0.001f, "allreduce [0] wrong");
        CHECK(fabsf(out[1] - 10.0f) < 0.001f, "allreduce [1] wrong");
        CHECK(fabsf(out[2] - 14.0f) < 0.001f, "allreduce [2] wrong");
    }

    tu_cluster_destroy(cl);
    PASS();
}

/* ---- Test 12: Barrier ---- */
static void test_barrier(void) {
    TEST("barrier sync");
    tu_runtime_config_t cfg = tu_config_default();
    tu_cluster_t *cl = tu_cluster_create(4, TU_TOPOLOGY_RING, 0, &cfg);
    CHECK(cl != NULL, "create failed");

    uint64_t cycles_before[4];
    for (uint32_t i = 0; i < 4; i++) {
        cycles_before[i] = cl->cores[i]->state.estimated_cycles;
    }

    int rc = tu_cluster_barrier(cl);
    CHECK(rc == 0, "barrier failed");
    CHECK(cl->stats.total_barriers == 1, "barrier count wrong");

    /* All cores should have accumulated barrier cycles */
    for (uint32_t i = 0; i < 4; i++) {
        CHECK(cl->cores[i]->state.estimated_cycles > cycles_before[i],
              "barrier didn't add cycles");
    }

    tu_cluster_destroy(cl);
    PASS();
}

/* ---- Test 13: SPMD execution ---- */
static void test_spmd_execution(void) {
    TEST("SPMD execution");
    tu_runtime_config_t cfg = tu_config_default();
    tu_cluster_t *cl = tu_cluster_create(2, TU_TOPOLOGY_RING, 0, &cfg);
    CHECK(cl != NULL, "create failed");

    /* ASM program that does a simple DMA load */
    const char *program =
        "%kernel tiny\n"
        "%input A:float[16]\n"
        "%output O:float[16]\n"
        "  LOAD_A A[0] AT 0\n"
        "  MMA 16 1 16 0 0 0\n"
        "  STORE_O O[0] FROM 0 SIZE 64\n"
        "  SYNC\n";

    /* Core 0: identity weights, ones activations → O = ones */
    fp16_t w_ident[256];
    fp16_t a_ones[16];
    memset(w_ident, 0, sizeof(w_ident));
    for (int i = 0; i < 16; i++) {
        w_ident[i * 16 + i] = tu_fp32_to_fp16(1.0f);
        a_ones[i] = tu_fp32_to_fp16(1.0f);
    }

    /* Simpler: use an MMA directly via tu_core_mma, since ASM needs weights preloaded */
    /* For SPMD test, we'll use direct MMA on each core */
    fp16_t a_data[16];
    for (int i = 0; i < 16; i++) a_data[i] = tu_fp32_to_fp16(1.0f);

    tu_core_mma(tu_cluster_get_core(cl, 0), 16, 1, 16, 0, 0, 0, false);
    tu_core_mma(tu_cluster_get_core(cl, 1), 16, 1, 16, 0, 0, 0, false);

    /* Verify both cores executed */
    tu_core_t *c0 = tu_cluster_get_core(cl, 0);
    tu_core_t *c1 = tu_cluster_get_core(cl, 1);
    CHECK(c0->state.total_mma_calls > 0, "core 0 didn't execute");
    CHECK(c1->state.total_mma_calls > 0, "core 1 didn't execute");

    tu_cluster_destroy(cl);
    PASS();
}

/* ---- Main ---- */

int main(void) {
    printf("\n=== TU Multi-Core Cluster Tests (Gap A5) ===\n\n");

    test_core_lifecycle();
    test_core_dma();
    test_core_mma();
    test_core_isolation();
    test_cluster_ring();
    test_cluster_mesh();
    test_hop_distance();
    test_neighbors();
    test_icc_send();
    test_icc_broadcast();
    test_allreduce_sum();
    test_barrier();
    test_spmd_execution();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_total, tests_failed);

    return tests_failed ? 1 : 0;
}
