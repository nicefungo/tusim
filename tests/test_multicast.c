/*
 * TU CModel — Multicast/Broadcast DMA Test Suite
 * ================================================
 * Tests: descriptor creation, execution to single target,
 * multi-target broadcast, bounds checking, null handling,
 * cycle accounting, performance counter integration.
 *
 * Gap DM4: Multicast/Broadcast DMA Engine
 */

#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/tu_sram.h"
#include "tu_cmodel/tu_config.h"
#include "tu_cmodel/perf/performance_counters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_pass = 0;

#define TEST(name) do { tests_run++; printf("  %-52s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_pass++; } while(0)
#define FAIL(fmt, ...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); } while(0)

/* ---- Test 1: Create multicast descriptor ---- */
static void test_multicast_create(void) {
    TEST("Create multicast descriptor");

    tu_dma_init_full(false, 3, 8);
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    const float src_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint32_t offsets[2] = {0, 64};
    tu_sram_region_t *regions[2] = {&r, &r};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_multicast(
        0, src_data, regions, offsets, 2, sizeof(float), 4);

    if (!desc) { FAIL("descriptor creation returned NULL"); tu_sram_destroy(&r); return; }
    if (desc->type != TU_DMA_XFER_MULTICAST) { FAIL("wrong transfer type"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->multicast.count != 2) { FAIL("wrong multicast count"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->total_bytes != 4 * (uint32_t)sizeof(float) * 2)
        { FAIL("wrong total_bytes"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* ---- Test 2: Multicast execution with single target ---- */
static void test_multicast_single_target(void) {
    TEST("Multicast single target");

    tu_dma_init_full(false, 3, 8);
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    const float src_data[4] = {3.14f, 2.71f, 1.41f, 0.577f};
    uint32_t offset = 0;
    tu_sram_region_t *regions[1] = {&r};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_multicast(
        0, src_data, regions, &offset, 1, sizeof(float), 4);

    if (!desc) { FAIL("descriptor creation failed"); tu_sram_destroy(&r); return; }
    tu_dma_execute_desc(desc);

    float *result = (float*)(tu_sram_raw_ptr(&r) + offset);
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (fabsf(result[i] - src_data[i]) > 1e-6f) { ok = 0; break; }
    }
    if (!ok) { FAIL("data mismatch"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* ---- Test 3: Multicast to three distinct SRAM regions ---- */
static void test_multicast_multi_target(void) {
    TEST("Multicast three regions");

    tu_dma_init_full(false, 3, 8);
    tu_sram_region_t r0, r1, r2;
    tu_sram_init(&r0, 4096, "r0");
    tu_sram_init(&r1, 4096, "r1");
    tu_sram_init(&r2, 4096, "r2");

    const uint16_t src_data[8] = {0xDEAD, 0xBEEF, 0xCAFE, 0xFACE,
                                   0x1234, 0x5678, 0x9ABC, 0xDEF0};
    uint32_t offsets[3] = {0, 32, 64};
    tu_sram_region_t *regions[3] = {&r0, &r1, &r2};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_multicast(
        0, src_data, regions, offsets, 3, sizeof(uint16_t), 8);

    if (!desc) { FAIL("creation failed"); goto cleanup; }
    if (desc->multicast.count != 3) { FAIL("wrong count"); tu_dma_desc_destroy(desc); goto cleanup; }

    tu_dma_execute_desc(desc);
    tu_dma_desc_destroy(desc);

    /* Verify all three SRAM regions */
    uint16_t *d0 = (uint16_t*)tu_sram_raw_ptr(&r0);
    uint16_t *d1 = (uint16_t*)(tu_sram_raw_ptr(&r1) + 32);
    uint16_t *d2 = (uint16_t*)(tu_sram_raw_ptr(&r2) + 64);

    for (int i = 0; i < 8; i++) {
        if (d0[i] != src_data[i]) { FAIL("r0[%d] mismatch", i); goto cleanup; }
        if (d1[i] != src_data[i]) { FAIL("r1[%d] mismatch", i); goto cleanup; }
        if (d2[i] != src_data[i]) { FAIL("r2[%d] mismatch", i); goto cleanup; }
    }

    PASS();
cleanup:
    tu_sram_destroy(&r0); tu_sram_destroy(&r1); tu_sram_destroy(&r2);
}

/* ---- Test 4: Bounds checking — exact fit at end of region ---- */
static void test_multicast_bounds_exact(void) {
    TEST("Multicast bounds exact fit");

    tu_dma_init_full(false, 3, 8);
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    const float src_data[2] = {42.0f, 43.0f};
    uint32_t offset = (uint32_t)(r.total_size - 2 * sizeof(float));
    tu_sram_region_t *regions[1] = {&r};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_multicast(
        0, src_data, regions, &offset, 1, sizeof(float), 2);

    if (!desc) { FAIL("creation failed"); tu_sram_destroy(&r); return; }
    tu_dma_execute_desc(desc);

    float *result = (float*)(tu_sram_raw_ptr(&r) + offset);
    if (fabsf(result[0] - 42.0f) > 1e-6f || fabsf(result[1] - 43.0f) > 1e-6f)
        { FAIL("data mismatch"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* ---- Test 5: Every destination is validated before any write ---- */
static void test_multicast_bounds_overflow(void) {
    const unsigned char src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    /* Direct execution retains ownership; rejected submission destroys it.
     * Test all invalid-target positions, including null target metadata. */
    for (int surface = 0; surface < 3; ++surface) {
        for (int null_target = 0; null_target < 2; ++null_target) {
            for (int bad = 0; bad < 3; ++bad) {
                TEST("Multicast rejects all targets atomically");
                tu_sram_region_t r[3];
                tu_sram_region_t *regions[3] = {&r[0], &r[1], &r[2]};
                uint32_t offsets[3] = {0, 16, 56};
                tu_dma_descriptor_t *desc = NULL;
                unsigned char before[3][64];
                tu_dma_init_full(surface == 2, 3, 8);
                for (int i = 0; i < 3; ++i) {
                    tu_sram_init(&r[i], 64, "atomic");
                    memset(before[i], 0xA0 + i, sizeof(before[i]));
                    memcpy(tu_sram_raw_ptr(&r[i]), before[i], sizeof(before[i]));
                }
                if (null_target) regions[bad] = NULL;
                else offsets[bad] = 57; /* one byte beyond exact end */
                desc = tu_dma_desc_create_multicast(
                    0, src, regions, offsets, 3, 1, sizeof(src));
                if (!desc) { FAIL("creation failed"); goto cleanup; }
                unsigned char engine_before[sizeof(g_tu_dma)];
                memcpy(engine_before, &g_tu_dma, sizeof(engine_before));
                if (surface == 0) {
                    tu_dma_execute_desc(desc);
                    if (desc->completed || desc->cycles_issued || desc->cycles_completed) {
                        FAIL("rejected direct descriptor acquired completion state");
                        goto cleanup;
                    }
                } else {
                    uint32_t id = tu_dma_submit_desc(desc);
                    /* Rejection consumes the descriptor; never dereference it. */
                    if (id != 0) {
                        FAIL("invalid submission accepted");
                        /* Drain accepted work before releasing its SRAM. */
                        tu_dma_flush_all();
                        goto cleanup;
                    }
                    desc = NULL;
                }
                if (memcmp(&g_tu_dma, engine_before, sizeof(engine_before)) != 0) {
                    FAIL("rejection mutated engine/queue/accounting state");
                    goto cleanup;
                }
                for (int i = 0; i < 3; ++i) {
                    if (memcmp(tu_sram_raw_ptr(&r[i]), before[i], sizeof(before[i]))) {
                        FAIL("surface=%d null=%d bad=%d target=%d copied a prefix",
                             surface, null_target, bad, i);
                        goto cleanup;
                    }
                }
                PASS();
cleanup:
                tu_dma_desc_destroy(desc);
                for (int i = 0; i < 3; ++i) tu_sram_destroy(&r[i]);
            }
        }
    }
}

static void test_multicast_multi_exact_end(void) {
    const unsigned char src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int surface = 0; surface < 3; ++surface) {
        TEST("Multicast three exact-end targets accepted");
        tu_sram_region_t r[3];
        tu_sram_region_t *regions[3] = {&r[0], &r[1], &r[2]};
        uint32_t offsets[3] = {56, 120, 248};
        unsigned char expected[3][256];
        tu_dma_descriptor_t *desc = NULL;
        tu_dma_init_full(surface == 2, 3, 8);
        for (int i = 0; i < 3; ++i) {
            tu_sram_init(&r[i], 64u << i, "exact-end");
            memset(expected[i], 0xA0 + i, sizeof(expected[i]));
            memcpy(tu_sram_raw_ptr(&r[i]), expected[i], r[i].total_size);
            memcpy(expected[i] + offsets[i], src, sizeof(src));
        }
        desc = tu_dma_desc_create_multicast(0, src, regions, offsets, 3, 1, sizeof(src));
        if (!desc) { FAIL("creation failed"); goto cleanup; }
        if (surface == 0) tu_dma_execute_desc(desc);
        else if (tu_dma_submit_desc(desc) == 0) {
            desc = NULL; /* rejection owns/frees the chain */
            FAIL("exact-end submission rejected");
            goto cleanup;
        }
        if (surface == 2) tu_dma_flush_all();
        if (!desc->completed || !desc->cycles_completed ||
            g_tu_dma.total_transfers != 1 || g_tu_dma.total_bytes != 24 ||
            g_tu_dma.total_occupied_bytes != 32 ||
            g_tu_dma.estimated_cycles != 51) {
            FAIL("completion or compatibility accounting mismatch");
            goto cleanup;
        }
        for (int i = 0; i < 3; ++i) {
            if (memcmp(tu_sram_raw_ptr(&r[i]), expected[i], r[i].total_size)) {
                FAIL("target %d payload/guard mismatch", i);
                goto cleanup;
            }
        }
        PASS();
cleanup:
        tu_dma_desc_destroy(desc);
        for (int i = 0; i < 3; ++i) tu_sram_destroy(&r[i]);
    }
}

/* ---- Test 6: Null handling ---- */
static void test_multicast_null_inputs(void) {
    TEST("Multicast null inputs");

    tu_dma_init_full(false, 3, 8);
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    const float src[1] = {0};
    uint32_t off = 0;
    tu_sram_region_t *regions[1] = {&r};

    tu_dma_descriptor_t *d1 = tu_dma_desc_create_multicast(
        0, NULL, regions, &off, 1, sizeof(float), 1);
    if (d1) { FAIL("null src should return NULL"); tu_dma_desc_destroy(d1); tu_sram_destroy(&r); return; }

    tu_dma_descriptor_t *d2 = tu_dma_desc_create_multicast(
        0, src, NULL, &off, 1, sizeof(float), 1);
    if (d2) { FAIL("null regions should return NULL"); tu_dma_desc_destroy(d2); tu_sram_destroy(&r); return; }

    tu_dma_descriptor_t *d3 = tu_dma_desc_create_multicast(
        0, src, regions, NULL, 1, sizeof(float), 1);
    if (d3) { FAIL("null offsets should return NULL"); tu_dma_desc_destroy(d3); tu_sram_destroy(&r); return; }

    tu_dma_descriptor_t *d4 = tu_dma_desc_create_multicast(
        0, src, regions, &off, 0, sizeof(float), 1);
    if (d4) { FAIL("zero destinations should return NULL"); tu_dma_desc_destroy(d4); tu_sram_destroy(&r); return; }

    tu_sram_destroy(&r);
    PASS();
}

/* ---- Test 7: Descriptor destruction frees multicast arrays ---- */
static void test_multicast_destroy_frees(void) {
    TEST("Multicast destroy frees arrays");

    tu_dma_init_full(false, 3, 8);
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    const float src[2] = {0, 0};
    uint32_t offs[2] = {0, 64};
    tu_sram_region_t *regions[2] = {&r, &r};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_multicast(
        0, src, regions, offs, 2, sizeof(float), 2);

    if (!desc) { FAIL("creation failed"); tu_sram_destroy(&r); return; }
    if (!desc->multicast.regions) { FAIL("regions not allocated"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (!desc->multicast.offsets) { FAIL("offsets not allocated"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    /* No crash/no valgrind leak = PASS */
    PASS();
}

/* ---- Test 8: Performance counter tracks multicast ---- */
static void test_multicast_perf_counter(void) {
    TEST("Multicast performance counter");

    tu_dma_init_full(false, 3, 8);
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    const float src[4] = {1, 2, 3, 4};
    uint32_t offs[3] = {0, 64, 128};
    tu_sram_region_t *regions[3] = {&r, &r, &r};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_multicast(
        0, src, regions, offs, 3, sizeof(float), 4);

    if (!desc) { FAIL("creation failed"); tu_sram_destroy(&r); return; }
    tu_dma_execute_desc(desc);

    tu_perf_dma_record_read(&c, desc->total_bytes,
                            5, 0, 0, (uint8_t)desc->type);

    if (c.dma.dma_transfers_multicast != 1)
        { FAIL("multicast counter = %lu expected 1", (unsigned long)c.dma.dma_transfers_multicast);
          tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* ---- Test 9: Large multicast (16 targets) ---- */
static void test_multicast_large_fanout(void) {
    TEST("Multicast 16-target fanout");

    tu_dma_init_full(false, 3, 8);
    tu_sram_region_t r;
    tu_sram_init(&r, 65536, "big");

    const int N = 16;
    const float src_data[2] = {99.0f, 100.0f};
    tu_sram_region_t *regions[N];
    uint32_t offsets[N];

    for (int i = 0; i < N; i++) {
        regions[i] = &r;
        offsets[i] = (uint32_t)(i * 128);
    }

    tu_dma_descriptor_t *desc = tu_dma_desc_create_multicast(
        0, src_data, regions, offsets, N, sizeof(float), 2);

    if (!desc) { FAIL("creation failed"); tu_sram_destroy(&r); return; }
    if (desc->multicast.count != (uint32_t)N)
        { FAIL("wrong count"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_execute_desc(desc);
    tu_dma_desc_destroy(desc);

    for (int i = 0; i < N; i++) {
        float *result = (float*)(tu_sram_raw_ptr(&r) + offsets[i]);
        if (fabsf(result[0] - 99.0f) > 1e-6f || fabsf(result[1] - 100.0f) > 1e-6f)
            { FAIL("target[%d] mismatch", i); tu_sram_destroy(&r); return; }
    }

    tu_sram_destroy(&r);
    PASS();
}

/* ---- Test 10: Chained multicast descriptor ---- */
static void test_multicast_chained(void) {
    TEST("Multicast chained after linear");

    tu_dma_init_full(false, 3, 8);
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "chain");

    /* First descriptor: linear load */
    float src1[4] = {1, 2, 3, 4};
    tu_dma_descriptor_t *d1 = tu_dma_desc_create_linear(
        0, TU_DMA_DIR_HOST_TO_TU, &r, 0, src1, sizeof(float), 4);
    if (!d1) { FAIL("linear create failed"); tu_sram_destroy(&r); return; }

    /* Second descriptor: multicast broadcast */
    const float src2[2] = {99, 100};
    uint32_t offs[2] = {128, 256};
    tu_sram_region_t *regions[2] = {&r, &r};
    tu_dma_descriptor_t *d2 = tu_dma_desc_create_multicast(
        0, src2, regions, offs, 2, sizeof(float), 2);
    if (!d2) { FAIL("multicast create failed"); tu_dma_desc_destroy(d1); tu_sram_destroy(&r); return; }

    tu_dma_desc_chain(d1, d2);
    tu_dma_submit_desc(d1);
    tu_dma_flush_all();

    /* Verify linear */
    float *r0 = (float*)tu_sram_raw_ptr(&r);
    if (fabsf(r0[0]-1) > 1e-6f || fabsf(r0[3]-4) > 1e-6f)
        { FAIL("linear step failed"); tu_sram_destroy(&r); return; }

    /* Verify multicast */
    float *r128 = (float*)(tu_sram_raw_ptr(&r) + 128);
    float *r256 = (float*)(tu_sram_raw_ptr(&r) + 256);
    if (fabsf(r128[0]-99) > 1e-6f || fabsf(r256[1]-100) > 1e-6f)
        { FAIL("multicast step failed"); tu_sram_destroy(&r); return; }

    /* Synchronous submission detaches but does not destroy the chain. */
    tu_dma_desc_destroy(d1);
    tu_sram_destroy(&r);
    PASS();
}

/* ---- Main ---- */
int main(void) {
    printf("=== Multicast/Broadcast DMA Tests (DM4) ===\n\n");

    test_multicast_create();
    test_multicast_single_target();
    test_multicast_multi_target();
    test_multicast_bounds_exact();
    test_multicast_bounds_overflow();
    test_multicast_multi_exact_end();
    test_multicast_null_inputs();
    test_multicast_destroy_frees();
    test_multicast_perf_counter();
    test_multicast_large_fanout();
    test_multicast_chained();

    printf("\n=== Results: %d/%d passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
