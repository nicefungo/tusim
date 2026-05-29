/*
 * Memory Hierarchy Unit Tests
 * =============================
 * Tests for: initialization, level-aware access, global buffer,
 * RegFile model, statistics, and DRAM delegation.
 */

#include "tu_cmodel/tu_config.h"
#include "tu_cmodel/tu_sram.h"
#include "tu_cmodel/memory/memory_hierarchy.h"
#include "tu_cmodel/memory/dram_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

static int passed = 0, failed = 0;

#define TEST(name) do { \
    printf("  %-55s", name); \
    fflush(stdout); \
} while(0)

#define PASS() do { printf("PASS\n"); passed++; } while(0)
#define FAIL(msg) do { printf("FAIL — %s\n", msg); failed++; } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ================================================================
 * Test 1: Hierarchy initialization & destruction
 * ================================================================ */
static void test_init_destroy(void) {
    TEST("Hierarchy init + destroy");
    tu_memory_hierarchy_t h;
    tu_mem_hierarchy_init(&h);
    CHECK(h.initialized, "not initialized");
    CHECK(h.dram != NULL, "DRAM model not created");
    CHECK(h.gbuf.sram.total_size == TU_MEM_GBUF_SIZE, "GBUF wrong size");
    CHECK(h.regfile.size_per_pe == TU_MEM_REGFILE_PER_PE, "RegFile wrong size");
    tu_mem_hierarchy_destroy(&h);
    CHECK(!h.initialized, "not destroyed");
    PASS();
}

/* ================================================================
 * Test 2: Level name lookup
 * ================================================================ */
static void test_level_names(void) {
    TEST("Level name lookup");
    CHECK(strcmp(tu_mem_level_name(TU_MEM_REGFILE), "RegFile (L0)") == 0, "RegFile name mismatch");
    CHECK(strcmp(tu_mem_level_name(TU_MEM_LOCAL_SPAD), "LocalSPAD (L1)") == 0, "LocalSPAD name mismatch");
    CHECK(strcmp(tu_mem_level_name(TU_MEM_GLOBAL_BUF), "GlobalBuf (L2)") == 0, "GlobalBuf name mismatch");
    CHECK(strcmp(tu_mem_level_name(TU_MEM_DRAM), "DRAM (L3)") == 0, "DRAM name mismatch");
    PASS();
}

/* ================================================================
 * Test 3: Local SPAD read/write through hierarchy
 * ================================================================ */
static void test_local_spad_access(void) {
    TEST("Local SPAD read/write via hierarchy");

    tu_memory_hierarchy_t h;
    tu_mem_hierarchy_init(&h);

    tu_sram_region_t spad;
    tu_sram_init(&spad, 4096, "test_spad");

    /* Write pattern */
    float write_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t stall = 0;
    int rc = tu_mem_hierarchy_write(&h, TU_MEM_LOCAL_SPAD, &spad,
                                     0, write_vals, sizeof(write_vals), &stall);
    CHECK(rc == 0, "write failed");

    /* Read back */
    float read_vals[4] = {0};
    rc = tu_mem_hierarchy_read(&h, TU_MEM_LOCAL_SPAD, &spad,
                                0, read_vals, sizeof(read_vals), &stall);
    CHECK(rc == 0, "read failed");
    CHECK(read_vals[0] == 1.0f, "read[0] mismatch");
    CHECK(read_vals[3] == 4.0f, "read[3] mismatch");

    /* Check level counters */
    CHECK(h.level_writes[TU_MEM_LOCAL_SPAD] >= 1, "SPAD write counter not incremented");
    CHECK(h.level_reads[TU_MEM_LOCAL_SPAD] >= 1, "SPAD read counter not incremented");

    tu_sram_destroy(&spad);
    tu_mem_hierarchy_destroy(&h);
    PASS();
}

/* ================================================================
 * Test 4: Global Buffer read/write
 * ================================================================ */
static void test_global_buffer_access(void) {
    TEST("Global Buffer read/write");

    tu_memory_hierarchy_t h;
    tu_mem_hierarchy_init(&h);

    /* Write to global buffer */
    float data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    uint64_t stall = 0;
    int rc = tu_mem_hierarchy_write(&h, TU_MEM_GLOBAL_BUF, NULL,
                                     0, data, sizeof(data), &stall);
    CHECK(rc == 0, "GBUF write failed");

    /* Read back */
    float readback[8] = {0};
    rc = tu_mem_hierarchy_read(&h, TU_MEM_GLOBAL_BUF, NULL,
                                0, readback, sizeof(readback), &stall);
    CHECK(rc == 0, "GBUF read failed");
    CHECK(readback[0] == 10.0f, "readback[0] mismatch");
    CHECK(readback[7] == 80.0f, "readback[7] mismatch");

    /* Check GBUF hit counters */
    CHECK(h.gbuf.total_hits >= 1, "GBUF hits not counted");

    /* Check level counters */
    CHECK(h.level_writes[TU_MEM_GLOBAL_BUF] >= 1, "GBUF write counter");
    CHECK(h.level_reads[TU_MEM_GLOBAL_BUF] >= 1, "GBUF read counter");

    tu_mem_hierarchy_destroy(&h);
    PASS();
}

/* ================================================================
 * Test 5: RegFile access (zero-latency)
 * ================================================================ */
static void test_regfile_access(void) {
    TEST("RegFile access tracking");

    tu_memory_hierarchy_t h;
    tu_mem_hierarchy_init(&h);

    /* RegFile reads are tracked but no data moved */
    float buf[4];
    uint64_t stall = 0;
    int rc = tu_mem_hierarchy_read(&h, TU_MEM_REGFILE, NULL,
                                    0, buf, sizeof(buf), &stall);
    CHECK(rc == 0, "RegFile read failed");
    CHECK(stall == 0, "RegFile should have zero stall");

    /* RegFile writes */
    float wdata[4] = {1, 2, 3, 4};
    rc = tu_mem_hierarchy_write(&h, TU_MEM_REGFILE, NULL,
                                 0, wdata, sizeof(wdata), &stall);
    CHECK(rc == 0, "RegFile write failed");

    CHECK(h.regfile.total_reads >= 1, "RegFile reads not counted");
    CHECK(h.regfile.total_writes >= 1, "RegFile writes not counted");

    tu_mem_hierarchy_destroy(&h);
    PASS();
}

/* ================================================================
 * Test 6: DRAM access delegation
 * ================================================================ */
static void test_dram_access(void) {
    TEST("DRAM access delegation");

    tu_memory_hierarchy_t h;
    tu_mem_hierarchy_init(&h);

    float ddata[4] = {99, 98, 97, 96};
    uint64_t stall = 0;
    int rc = tu_mem_hierarchy_write(&h, TU_MEM_DRAM, NULL,
                                     0, ddata, sizeof(ddata), &stall);
    CHECK(rc == 0, "DRAM write failed");

    /* DRAM writes increment counter */
    CHECK(h.level_writes[TU_MEM_DRAM] >= 1, "DRAM write counter not incremented");

    tu_mem_hierarchy_destroy(&h);
    PASS();
}

/* ================================================================
 * Test 7: On-chip total size calculation
 * ================================================================ */
static void test_onchip_total(void) {
    TEST("On-chip total size");

    tu_memory_hierarchy_t h;
    tu_mem_hierarchy_init(&h);

    uint64_t onchip = tu_mem_hierarchy_get_onchip_total(&h);
    uint64_t expected = (uint64_t)TU_MEM_REGFILE_PER_PE * TU_PE_ROWS * TU_PE_COLS
                        + TU_SRAM_TOTAL + TU_MEM_GBUF_SIZE;
    CHECK(onchip == expected, "on-chip total mismatch");

    tu_mem_hierarchy_destroy(&h);
    PASS();
}

/* ================================================================
 * Test 8: Reset preserves config, zeros stats
 * ================================================================ */
static void test_reset(void) {
    TEST("Reset zeros stats, keeps config");

    tu_memory_hierarchy_t h;
    tu_mem_hierarchy_init(&h);

    /* Do some accesses */
    float buf[4];
    uint64_t stall = 0;
    tu_mem_hierarchy_read(&h, TU_MEM_LOCAL_SPAD,
                           tu_gbuf_get_sram(&h.gbuf), /* use GBUF as region for test */
                           0, buf, 4, &stall);

    /* Verify non-zero stats */
    CHECK(h.level_reads[TU_MEM_LOCAL_SPAD] > 0, "should have non-zero reads before reset");

    /* Reset */
    tu_mem_hierarchy_reset(&h);

    /* Verify zeroed */
    CHECK(h.level_reads[TU_MEM_LOCAL_SPAD] == 0, "reads not zeroed by reset");
    CHECK(h.current_cycle == 0, "cycle not zeroed");
    CHECK(h.initialized, "initialized flag lost");

    tu_mem_hierarchy_destroy(&h);
    PASS();
}

/* ================================================================
 * Test 9: Cycle advancement
 * ================================================================ */
static void test_cycle_advance(void) {
    TEST("Hierarchy cycle advancement");

    tu_memory_hierarchy_t h;
    tu_mem_hierarchy_init(&h);

    CHECK(h.current_cycle == 0, "initial cycle not zero");
    tu_mem_hierarchy_tick(&h, 100);
    CHECK(tu_mem_hierarchy_get_cycle(&h) == 100, "cycle not advanced");
    tu_mem_hierarchy_tick(&h, 50);
    CHECK(tu_mem_hierarchy_get_cycle(&h) == 150, "cycle not accumulated");

    tu_mem_hierarchy_destroy(&h);
    PASS();
}

/* ================================================================
 * Test 10: Statistics print (smoke test — no crash)
 * ================================================================ */
static void test_stats_print(void) {
    TEST("Statistics print (smoke)");

    tu_memory_hierarchy_t h;
    tu_mem_hierarchy_init(&h);

    /* Do a variety of accesses */
    float buf[8];
    uint64_t stall;
    tu_sram_region_t spad;
    tu_sram_init(&spad, 4096, "test");

    tu_mem_hierarchy_write(&h, TU_MEM_LOCAL_SPAD, &spad, 0, buf, 32, &stall);
    tu_mem_hierarchy_read(&h, TU_MEM_LOCAL_SPAD, &spad, 0, buf, 32, &stall);
    tu_mem_hierarchy_write(&h, TU_MEM_GLOBAL_BUF, NULL, 0, buf, 32, &stall);
    tu_mem_hierarchy_read(&h, TU_MEM_GLOBAL_BUF, NULL, 0, buf, 32, &stall);
    tu_mem_hierarchy_write(&h, TU_MEM_DRAM, NULL, 0, buf, 32, &stall);

    /* Print to /dev/null to verify no crash */
    FILE *null_out = fopen("/dev/null", "w");
    CHECK(null_out != NULL, "cannot open /dev/null");
    tu_mem_hierarchy_print_stats(&h, null_out);
    fclose(null_out);

    tu_sram_destroy(&spad);
    tu_mem_hierarchy_destroy(&h);
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
    printf("TinyTU Memory Hierarchy — Unit Tests\n");
    printf("=====================================\n");
    printf("Config: RegFile=%u B/PE, SPAD=%u KB (%u banks), GBUF=%u MB (%u banks)\n\n",
           TU_MEM_REGFILE_PER_PE,
           TU_SRAM_TOTAL / 1024, TU_SRAM_BANKS,
           TU_MEM_GBUF_SIZE / (1024*1024), TU_MEM_GBUF_BANKS);

    test_init_destroy();
    test_level_names();
    test_local_spad_access();
    test_global_buffer_access();
    test_regfile_access();
    test_dram_access();
    test_onchip_total();
    test_reset();
    test_cycle_advance();
    test_stats_print();

    printf("\n═══════════════════════════════════════════\n");
    printf("  %d/%d tests passed\n", passed, passed + failed);
    printf("═══════════════════════════════════════════\n");

    return failed > 0 ? 1 : 0;
}
