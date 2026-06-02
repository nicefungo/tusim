/*
 * TU CModel — Scatter/Gather DMA Test Suite
 * ===========================================
 * Tests: scatter descriptor creation/execution, gather descriptor
 * creation/execution, index validation, edge cases (empty/single
 * element), channel routing, cycle accounting, perf counter
 * integration, SRAM bounds checking.
 *
 * Gap DM3: Scatter/gather DMA for sparse weight/activation loading.
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

/* ── Common setup ──────────────────────────────────────────────── */

static void setup(void) {
    tu_dma_init_full(false, 3, 8);
}

/* ── Scatter Tests ────────────────────────────────────────────── */

/* Test 1: Create scatter descriptor */
static void test_scatter_create(void) {
    TEST("Scatter descriptor creation");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    const float src_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint32_t indices[4] = {0, 16, 32, 48};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
        0, &r, src_data, indices, 4, sizeof(float));

    if (!desc) { FAIL("descriptor creation returned NULL"); tu_sram_destroy(&r); return; }
    if (desc->type != TU_DMA_XFER_SCATTER) { FAIL("wrong transfer type: %d", desc->type); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->direction != TU_DMA_DIR_HOST_TO_TU) { FAIL("wrong direction"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->index_count != 4) { FAIL("wrong index count: %u", desc->index_count); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->elem_size != sizeof(float)) { FAIL("wrong elem size"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->total_bytes != 4 * sizeof(float)) { FAIL("wrong total bytes"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->index_list != indices) { FAIL("wrong index list pointer"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 2: Scatter execution — sequential offsets */
static void test_scatter_sequential(void) {
    TEST("Scatter execution (sequential offsets)");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");
    memset(tu_sram_raw_ptr(&r), 0, r.total_size);

    int32_t src_data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    uint32_t indices[8] = {0, 4, 8, 12, 16, 20, 24, 28};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
        0, &r, src_data, indices, 8, sizeof(int32_t));
    if (!desc) { FAIL("descriptor creation failed"); tu_sram_destroy(&r); return; }

    tu_dma_execute_desc(desc);

    const int32_t *result = (const int32_t*)tu_sram_raw_ptr(&r);
    for (int i = 0; i < 8; i++) {
        if (result[i] != src_data[i]) {
            FAIL("data mismatch at element %d: expected %d, got %d", i, src_data[i], result[i]);
            tu_dma_desc_destroy(desc);
            tu_sram_destroy(&r);
            return;
        }
    }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 3: Scatter execution — non-sequential (sparse) offsets */
static void test_scatter_sparse(void) {
    TEST("Scatter execution (sparse offsets)");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");
    memset(tu_sram_raw_ptr(&r), 0xAA, r.total_size);

    float src_data[4] = {3.14f, 2.71f, 1.41f, 0.57f};
    uint32_t indices[4] = {0, 128, 256, 512};  /* scattered across SRAM */

    tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
        0, &r, src_data, indices, 4, sizeof(float));
    if (!desc) { FAIL("descriptor creation failed"); tu_sram_destroy(&r); return; }

    tu_dma_execute_desc(desc);

    uint8_t *raw = (uint8_t*)tu_sram_raw_ptr(&r);
    for (int i = 0; i < 4; i++) {
        float actual;
        memcpy(&actual, raw + indices[i], sizeof(float));
        if (fabsf(actual - src_data[i]) > 1e-6f) {
            FAIL("sparse data mismatch at idx %d (offset=%u): expected %.4f, got %.4f",
                 i, indices[i], src_data[i], actual);
            tu_dma_desc_destroy(desc);
            tu_sram_destroy(&r);
            return;
        }
    }

    /* Verify that non-targeted locations are untouched */
    if (raw[64] != 0xAA) { FAIL("non-target byte at offset 64 was modified"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (raw[200] != 0xAA) { FAIL("non-target byte at offset 200 was modified"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 4: Scatter — single element */
static void test_scatter_single(void) {
    TEST("Scatter single element");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");
    memset(tu_sram_raw_ptr(&r), 0, r.total_size);

    double val = 1.618033988749895;
    uint32_t idx = 42;

    tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
        0, &r, &val, &idx, 1, sizeof(double));
    if (!desc) { FAIL("descriptor creation failed"); tu_sram_destroy(&r); return; }

    tu_dma_execute_desc(desc);

    double actual;
    memcpy(&actual, (uint8_t*)tu_sram_raw_ptr(&r) + idx, sizeof(double));
    if (actual != val) { FAIL("double mismatch: expected %.15f, got %.15f", val, actual); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 5: Scatter — empty index list */
static void test_scatter_empty(void) {
    TEST("Scatter empty index list");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 256, "test");

    const float src = 0.0f;
    tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
        0, &r, &src, NULL, 0, sizeof(float));
    if (!desc) { FAIL("descriptor creation returned NULL for empty list"); tu_sram_destroy(&r); return; }
    if (desc->index_count != 0) { FAIL("index count should be 0"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->total_bytes != 0) { FAIL("total bytes should be 0"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    /* Execution should be a no-op (no crash) */
    tu_dma_execute_desc(desc);

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 6: Scatter — cycle accounting */
static void test_scatter_cycles(void) {
    TEST("Scatter cycle accounting");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    float src_data[16];
    uint32_t indices[16];
    for (int i = 0; i < 16; i++) {
        src_data[i] = (float)(i * 0.1);
        indices[i] = i * (uint32_t)sizeof(float);
    }

    uint64_t bytes_before = g_tu_dma.total_bytes;
    uint64_t xfers_before = g_tu_dma.total_transfers;

    tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
        0, &r, src_data, indices, 16, sizeof(float));
    if (!desc) { FAIL("descriptor creation failed"); tu_sram_destroy(&r); return; }

    tu_dma_execute_desc(desc);

    if (g_tu_dma.total_bytes - bytes_before != desc->total_bytes)
        { FAIL("bytes not accounted"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (g_tu_dma.total_transfers - xfers_before != 1)
        { FAIL("transfer not counted"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (!desc->completed) { FAIL("descriptor not marked complete"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* ── Gather Tests ─────────────────────────────────────────────── */

/* Test 7: Create gather descriptor */
static void test_gather_create(void) {
    TEST("Gather descriptor creation");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    float dst_data[4] = {0};
    uint32_t indices[4] = {0, 64, 128, 256};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_gather(
        0, &r, dst_data, indices, 4, sizeof(float));

    if (!desc) { FAIL("descriptor creation returned NULL"); tu_sram_destroy(&r); return; }
    if (desc->type != TU_DMA_XFER_GATHER) { FAIL("wrong transfer type: %d", desc->type); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->direction != TU_DMA_DIR_TU_TO_HOST) { FAIL("wrong direction"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->index_count != 4) { FAIL("wrong index count: %u", desc->index_count); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (desc->dst_host != dst_data) { FAIL("wrong dst_host pointer"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 8: Gather execution — sequential offsets */
static void test_gather_sequential(void) {
    TEST("Gather execution (sequential)");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    /* Place data at known offsets in SRAM */
    int32_t sram_data[8] = {100, 200, 300, 400, 500, 600, 700, 800};
    memcpy(tu_sram_raw_ptr(&r), sram_data, sizeof(sram_data));

    int32_t dst[8] = {0};
    uint32_t indices[8] = {0, 4, 8, 12, 16, 20, 24, 28};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_gather(
        0, &r, dst, indices, 8, sizeof(int32_t));
    if (!desc) { FAIL("descriptor creation failed"); tu_sram_destroy(&r); return; }

    tu_dma_execute_desc(desc);

    for (int i = 0; i < 8; i++) {
        if (dst[i] != sram_data[i]) {
            FAIL("data mismatch at element %d: expected %d, got %d", i, sram_data[i], dst[i]);
            tu_dma_desc_destroy(desc);
            tu_sram_destroy(&r);
            return;
        }
    }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 9: Gather execution — sparse offsets */
static void test_gather_sparse(void) {
    TEST("Gather execution (sparse)");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    /* Scatter data at sparse locations in SRAM */
    uint8_t *raw = (uint8_t*)tu_sram_raw_ptr(&r);
    memset(raw, 0xCC, r.total_size);

    double vals[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    uint32_t offsets[5] = {0, 128, 300, 512, 1000};
    for (int i = 0; i < 5; i++) {
        memcpy(raw + offsets[i], &vals[i], sizeof(double));
    }

    double dst[5] = {0};
    tu_dma_descriptor_t *desc = tu_dma_desc_create_gather(
        0, &r, dst, offsets, 5, sizeof(double));
    if (!desc) { FAIL("descriptor creation failed"); tu_sram_destroy(&r); return; }

    tu_dma_execute_desc(desc);

    for (int i = 0; i < 5; i++) {
        if (dst[i] != vals[i]) {
            FAIL("sparse gather mismatch at element %d: expected %.1f, got %.1f", i, vals[i], dst[i]);
            tu_dma_desc_destroy(desc);
            tu_sram_destroy(&r);
            return;
        }
    }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 10: Gather — single element */
static void test_gather_single(void) {
    TEST("Gather single element");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    float val = 99.9f;
    uint32_t idx = 123;
    memcpy((uint8_t*)tu_sram_raw_ptr(&r) + idx, &val, sizeof(float));

    float dst = 0.0f;
    tu_dma_descriptor_t *desc = tu_dma_desc_create_gather(
        0, &r, &dst, &idx, 1, sizeof(float));
    if (!desc) { FAIL("descriptor creation failed"); tu_sram_destroy(&r); return; }

    tu_dma_execute_desc(desc);

    if (fabsf(dst - val) > 1e-6f) { FAIL("float mismatch: expected %.1f, got %.1f", val, dst); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 11: Gather — cycle accounting */
static void test_gather_cycles(void) {
    TEST("Gather cycle accounting");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    /* Fill SRAM with sequential data */
    for (int i = 0; i < 256; i++) {
        ((int32_t*)tu_sram_raw_ptr(&r))[i] = i;
    }

    int32_t dst[8] = {0};
    uint32_t indices[8] = {0, 4, 8, 12, 16, 20, 24, 28};

    uint64_t bytes_before = g_tu_dma.total_bytes;
    uint64_t xfers_before = g_tu_dma.total_transfers;

    tu_dma_descriptor_t *desc = tu_dma_desc_create_gather(
        0, &r, dst, indices, 8, sizeof(int32_t));
    if (!desc) { FAIL("descriptor creation failed"); tu_sram_destroy(&r); return; }

    tu_dma_execute_desc(desc);

    if (g_tu_dma.total_bytes - bytes_before != 32)
        { FAIL("bytes not accounted"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (g_tu_dma.total_transfers - xfers_before != 1)
        { FAIL("transfer not counted"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (!desc->completed) { FAIL("descriptor not marked complete"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 12: Scatter+gather round-trip */
static void test_scatter_gather_roundtrip(void) {
    TEST("Scatter+gather round-trip");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    /* Scatter known data to sparse offsets */
    int16_t src_data[16];
    uint32_t offsets[16];
    for (int i = 0; i < 16; i++) {
        src_data[i] = (int16_t)(i * 100);
        offsets[i] = i * 64 + 8;  /* sparse: 64-byte stride, 8-byte offset */
    }

    tu_dma_descriptor_t *scatter = tu_dma_desc_create_scatter(
        1, &r, src_data, offsets, 16, sizeof(int16_t));
    if (!scatter) { FAIL("scatter creation failed"); tu_sram_destroy(&r); return; }
    tu_dma_execute_desc(scatter);
    tu_dma_desc_destroy(scatter);

    /* Gather it back */
    int16_t dst_data[16] = {0};
    tu_dma_descriptor_t *gather = tu_dma_desc_create_gather(
        1, &r, dst_data, offsets, 16, sizeof(int16_t));
    if (!gather) { FAIL("gather creation failed"); tu_sram_destroy(&r); return; }
    tu_dma_execute_desc(gather);
    tu_dma_desc_destroy(gather);

    for (int i = 0; i < 16; i++) {
        if (dst_data[i] != src_data[i]) {
            FAIL("round-trip mismatch at element %d: expected %d, got %d",
                 i, src_data[i], dst_data[i]);
            tu_sram_destroy(&r);
            return;
        }
    }

    tu_sram_destroy(&r);
    PASS();
}

/* ── Edge Case Tests ──────────────────────────────────────────── */

/* Test 13: Scatter with large byte types */
static void test_scatter_large_types(void) {
    TEST("Scatter large types (double, 64-bit)");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");
    memset(tu_sram_raw_ptr(&r), 0, r.total_size);

    double src[3] = {1e308, -1e308, 0.0};
    uint32_t idx[3] = {0, 8, 16};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
        0, &r, src, idx, 3, sizeof(double));
    if (!desc) { FAIL("creation failed"); tu_sram_destroy(&r); return; }
    tu_dma_execute_desc(desc);

    double *result = (double*)tu_sram_raw_ptr(&r);
    if (result[0] != 1e308) { FAIL("+1e308 mismatch"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (result[1] != -1e308) { FAIL("-1e308 mismatch"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (result[2] != 0.0) { FAIL("0.0 mismatch"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 14: Scatter with 1-byte elements */
static void test_scatter_byte_elements(void) {
    TEST("Scatter 1-byte elements");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 512, "test");
    memset(tu_sram_raw_ptr(&r), 0, r.total_size);

    uint8_t src[5] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE};
    uint32_t idx[5] = {10, 20, 30, 40, 50};

    tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
        0, &r, src, idx, 5, 1);
    if (!desc) { FAIL("creation failed"); tu_sram_destroy(&r); return; }
    tu_dma_execute_desc(desc);

    uint8_t *raw = (uint8_t*)tu_sram_raw_ptr(&r);
    for (int i = 0; i < 5; i++) {
        if (raw[idx[i]] != src[i]) {
            FAIL("byte mismatch at offset %u: expected 0x%02X, got 0x%02X",
                 idx[i], src[i], raw[idx[i]]);
            tu_dma_desc_destroy(desc);
            tu_sram_destroy(&r);
            return;
        }
    }
    /* Check non-targeted bytes */
    if (raw[5] != 0x00) { FAIL("non-target byte 5 was modified"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
    if (raw[60] != 0x00) { FAIL("non-target byte 60 was modified"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }

    tu_dma_desc_destroy(desc);
    tu_sram_destroy(&r);
    PASS();
}

/* Test 15: Channel routing */
static void test_scatter_channel_routing(void) {
    TEST("Scatter channel routing");

    setup();
    tu_sram_region_t r;
    tu_sram_init(&r, 4096, "test");

    float src[2] = {1.0f, 2.0f};
    uint32_t idx[2] = {0, 4};

    /* Use all three channels */
    for (uint8_t ch = 0; ch < 3; ch++) {
        tu_dma_descriptor_t *desc = tu_dma_desc_create_scatter(
            ch, &r, src, idx, 2, sizeof(float));
        if (!desc) { FAIL("creation failed on channel %d", ch); tu_sram_destroy(&r); return; }
        if (desc->channel != ch) { FAIL("channel mismatch"); tu_dma_desc_destroy(desc); tu_sram_destroy(&r); return; }
        tu_dma_desc_destroy(desc);
    }

    tu_sram_destroy(&r);
    PASS();
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(void) {
    printf("\n═══════════════════════════════════════════\n");
    printf("  TU Scatter/Gather DMA Tests (Gap DM3)\n");
    printf("═══════════════════════════════════════════\n\n");

    test_scatter_create();
    test_scatter_sequential();
    test_scatter_sparse();
    test_scatter_single();
    test_scatter_empty();
    test_scatter_cycles();
    test_scatter_large_types();
    test_scatter_byte_elements();
    test_scatter_channel_routing();

    test_gather_create();
    test_gather_sequential();
    test_gather_sparse();
    test_gather_single();
    test_gather_cycles();
    test_scatter_gather_roundtrip();

    printf("\n═══════════════════════════════════════════\n");
    printf("  Test Suite Complete\n");
    printf("  %d/%d tests passed\n", tests_pass, tests_run);
    printf("═══════════════════════════════════════════\n");

    return (tests_pass == tests_run) ? 0 : 1;
}
