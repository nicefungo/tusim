/*
 * Test: Performance Counter Infrastructure
 * ==========================================
 * Validates counter initialization, recording, snapshot/diff/merge,
 * reporting, and energy accounting.
 *
 * Gap: E4 (Power/Energy Model), P2.5 (Cycle-Accurate Model) foundation
 */

#include "test_framework.h"
#include "tu_cmodel/perf/performance_counters.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

tu_test_stats_t g_test_stats;

/* ---- Helper: verify condition, call FAIL if false ---- */
#define CHECK(cond, msg) do { \
    if (!(cond)) { FAIL("%s", msg); return; } \
} while(0)

/* ---- Test: Initialization ---- */
static void test_perf_init(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    CHECK(c.enabled, "Counters should be enabled by default");
    CHECK(c.clock_freq_mhz == 1000.0, "Clock should be 1000 MHz");
    CHECK(c.total_cycles == 0, "Total cycles should start at 0");
    CHECK(c.power.power_modeling_enabled, "Power modeling should be enabled");
    CHECK(c.power.pj_per_mac > 0.0, "MAC energy param should be positive");
    CHECK(c.power.pj_per_sram_read > 0.0, "SRAM read energy param should be positive");
    PASS();
}

/* ---- Test: Enable/Disable ---- */
static void test_perf_enable_disable(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 500.0);

    /* Record with counters enabled */
    tu_perf_dma_record_read(&c, 1024, 10, 2, 0, 0);
    CHECK(c.dma.dma_read_bytes == 1024, "Should record DMA read when enabled");
    CHECK(c.total_cycles > 0, "Cycles should advance when enabled");

    uint64_t cycles_after = c.total_cycles;

    /* Disable and record */
    tu_perf_set_enabled(&c, false);
    tu_perf_dma_record_read(&c, 2048, 20, 5, 1, 1);
    CHECK(c.dma.dma_read_bytes == 1024, "Should NOT increment when disabled");
    CHECK(c.total_cycles == cycles_after, "Cycles should NOT advance when disabled");

    /* Re-enable */
    tu_perf_set_enabled(&c, true);
    tu_perf_dma_record_read(&c, 512, 5, 0, 2, 2);
    CHECK(c.dma.dma_read_bytes == 1536, "Should record after re-enabling");
    PASS();
}

/* ---- Test: DMA Recording ---- */
static void test_perf_dma_recording(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    /* Record a read (linear) */
    tu_perf_dma_record_read(&c, 256, 5, 1, 0, 0);
    CHECK(c.dma.dma_read_bytes == 256, "Read bytes should accumulate");
    CHECK(c.dma.dma_transfers_linear == 1, "Linear transfer count should increment");

    /* Record a write (strided 2D) */
    tu_perf_dma_record_write(&c, 512, 8, 3, 1);
    CHECK(c.dma.dma_write_bytes == 512, "Write bytes should accumulate");
    CHECK(c.dma.dma_stall_cycles == 4, "Stall cycles should accumulate (1+3)");

    /* Record scatter/gather (DM3) */
    tu_perf_dma_record_read(&c, 128, 3, 0, 2, 3);  /* scatter */
    tu_perf_dma_record_read(&c, 64, 2, 0, 2, 4);   /* gather */
    CHECK(c.dma.dma_transfers_scatter == 1, "Scatter count should increment");
    CHECK(c.dma.dma_transfers_gather == 1, "Gather count should increment");

    /* Channel tracking */
    CHECK(c.dma.dma_channel_bytes[0] == 256, "Channel 0 should have 256 bytes");
    CHECK(c.dma.dma_channel_bytes[1] == 512, "Channel 1 should have 512 bytes");
    PASS();
}

/* ---- Test: Compute Recording ---- */
static void test_perf_compute_recording(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    /* Record FP16 MMA */
    tu_perf_compute_record_mma(&c, 4096, 16, 16, 16, 1, 0, 100, 10, 1, 0);
    CHECK(c.compute.total_macs == 4096, "Total MACs should be 4096");
    CHECK(c.compute.total_flops == 8192, "Total FLOPS = 2x MACs");
    CHECK(c.compute.op_mma_fp16 == 1, "FP16 MMA count should be 1");
    CHECK(c.compute.compute_active_cycles == 100, "Active cycles should be 100");
    CHECK(c.compute.compute_stall_cycles == 10, "Stall cycles should be 10");

    /* Record BF16 MMA (different dataflow) */
    tu_perf_compute_record_mma(&c, 16384, 32, 32, 32, 4, 1, 400, 50, 2, 1);
    CHECK(c.compute.op_mma_bf16 == 1, "BF16 MMA count should be 1");
    CHECK(c.compute.total_tiles == 5, "Total tiles = 1 + 4");
    CHECK(c.compute.edge_tiles == 1, "Edge tiles = 0 + 1");

    /* Record per-op codes */
    tu_perf_compute_record_op(&c, 2, 50, 5, 1000);  /* Conv2D */
    tu_perf_compute_record_op(&c, 3, 200, 20, 2000); /* Attention */
    tu_perf_compute_record_op(&c, 9, 30, 2, 100);    /* PoolMax (O6) */
    tu_perf_compute_record_op(&c, 10, 30, 2, 100);   /* PoolAvg (O6) */

    CHECK(c.compute.op_conv2d == 1, "Conv2D count should be 1");
    CHECK(c.compute.op_attention == 1, "Attention count should be 1");
    CHECK(c.compute.op_pool_max == 1, "PoolMax count should be 1");
    CHECK(c.compute.op_pool_avg == 1, "PoolAvg count should be 1");
    CHECK(c.compute.total_flops >= 11292, "Total FLOPS should include all ops");
    PASS();
}

/* ---- Test: Memory Recording ---- */
static void test_perf_memory_recording(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    tu_perf_mem_record_spad_access(&c, false, 32, 0, 0);
    CHECK(c.memory.mem_spad_reads == 32, "SPAD reads should be 32");

    tu_perf_mem_record_spad_access(&c, true, 16, 3, 5);
    CHECK(c.memory.mem_spad_writes == 16, "SPAD writes should be 16");
    CHECK(c.memory.mem_spad_bank_conflicts == 3, "Bank conflicts should be 3");
    CHECK(c.memory.mem_spad_stall_cycles == 5, "SPAD stall cycles should be 5");

    tu_perf_mem_record_dram_access(&c, false, 64, true, 0);
    CHECK(c.memory.mem_dram_reads == 1, "DRAM reads should be 1");
    CHECK(c.memory.mem_dram_row_hits == 1, "Row hits should be 1");
    CHECK(c.memory.mem_dram_bytes_read == 64, "DRAM read bytes should be 64");

    tu_perf_mem_record_dram_access(&c, true, 128, false, 10);
    CHECK(c.memory.mem_dram_writes == 1, "DRAM writes should be 1");
    CHECK(c.memory.mem_dram_row_misses == 1, "Row misses should be 1");

    tu_perf_mem_record_reqfile_access(&c, false, 4);
    tu_perf_mem_record_reqfile_access(&c, true, 2);
    CHECK(c.memory.mem_reqfile_reads == 4, "RegFile reads should be 4");
    CHECK(c.memory.mem_reqfile_writes == 2, "RegFile writes should be 2");
    PASS();
}

/* ---- Test: Snapshot, Diff, Merge ---- */
static void test_perf_snapshot_diff_merge(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    tu_perf_dma_record_read(&c, 1024, 10, 2, 0, 0);
    tu_perf_compute_record_mma(&c, 4096, 16, 16, 16, 1, 0, 100, 10, 1, 0);
    tu_perf_snapshot_t snap1 = tu_perf_snapshot(&c);

    tu_perf_dma_record_write(&c, 512, 5, 1, 1);
    tu_perf_compute_record_mma(&c, 16384, 32, 32, 32, 4, 0, 400, 50, 1, 0);
    tu_perf_snapshot_t snap2 = tu_perf_snapshot(&c);

    tu_perf_counters_t diff = tu_perf_diff(&snap1, &snap2);
    CHECK(diff.dma.dma_write_bytes == 512, "Diff DMA write bytes should be 512");
    CHECK(diff.compute.total_macs == 16384, "Diff MACs should be 16384");
    CHECK(diff.dma.dma_read_bytes == 0, "Diff DMA read bytes should be 0");

    /* Merge two independent counter sets */
    tu_perf_counters_t merged;
    tu_perf_init(&merged, 1000.0);
    tu_perf_merge(&merged, &snap1.counters);
    tu_perf_merge(&merged, &diff);
    CHECK(merged.dma.dma_read_bytes == 1024, "Merged read bytes should match");
    CHECK(merged.compute.total_macs == 20480, "Merged MACs should be sum");
    PASS();
}

/* ---- Test: Power Energy Accounting ---- */
static void test_perf_power_accounting(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 500.0);

    tu_perf_power_config(&c, 1.5, 0.3, 0.3, 10.0, 0.02, 0.005);

    tu_perf_compute_record_mma(&c, 1000, 16, 16, 16, 1, 0, 100, 0, 1, 0);
    tu_perf_mem_record_spad_access(&c, false, 50, 0, 0);

    CHECK(c.power.energy_mac_pj > 0.0, "MAC energy should be positive");
    CHECK(c.power.energy_sram_read_pj > 0.0, "SRAM read energy should be positive");
    CHECK(c.power.energy_leakage_pj > 0.0, "Leakage energy should be positive");

    tu_perf_power_set_enabled(&c, false);
    double mac_before = c.power.energy_mac_pj;
    tu_perf_compute_record_mma(&c, 500, 8, 8, 8, 1, 0, 50, 0, 1, 0);
    CHECK(c.power.energy_mac_pj == mac_before, "Energy should NOT increase when disabled");
    PASS();
}

/* ---- Test: Metrics Computation ---- */
static void test_perf_metrics(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    /* Simulate realistic workload: 16x16 PE array, 4 tiles of 16³ = 16K MACs,
     * 200 active compute cycles + 20 stall, plus DMA overhead */
    tu_perf_compute_record_mma(&c, 16384, 64, 64, 16, 4, 0, 200, 20, 1, 0);
    tu_perf_dma_record_read(&c, 64 * 16 * 2, 50, 10, 0, 0);
    tu_perf_dma_record_write(&c, 64 * 16 * 2, 50, 5, 1);

    tu_perf_metrics_t m = tu_perf_compute_metrics(&c);

    CHECK(m.compute_utilization >= 0.0f && m.compute_utilization <= 1.0f,
          "Utilization should be in [0,1]");
    CHECK(m.dma_bandwidth_gbps >= 0.0f, "DMA bandwidth should be non-negative");
    CHECK(m.mac_throughput_tops >= 0.0f, "MAC throughput should be non-negative");
    CHECK(m.spad_hit_rate >= 0.0f && m.spad_hit_rate <= 1.0f,
          "SPAD hit rate should be in [0,1]");
    CHECK(m.energy_per_mac_pj >= 0.0f, "Energy per MAC should be non-negative");
    CHECK(m.power_mw >= 0.0f, "Power should be non-negative");
    PASS();
}

/* ---- Test: Reporting (smoke test) ---- */
static void test_perf_reporting(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    tu_perf_dma_record_read(&c, 1024, 10, 2, 0, 0);
    tu_perf_compute_record_mma(&c, 4096, 16, 16, 16, 1, 0, 100, 10, 1, 0);

    tu_perf_print_summary(&c);
    tu_perf_print_report(&c);

    /* If we got here without crashing, test passes */
    PASS();
}

/* ---- Test: Reset ---- */
static void test_perf_reset(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    tu_perf_dma_record_read(&c, 1024, 10, 2, 0, 0);
    tu_perf_compute_record_mma(&c, 4096, 16, 16, 16, 1, 0, 100, 10, 1, 0);

    double pj_mac = c.power.pj_per_mac;
    double pj_sram = c.power.pj_per_sram_read;

    tu_perf_reset(&c);

    CHECK(c.total_cycles == 0, "Cycles should reset to 0");
    CHECK(c.dma.dma_read_bytes == 0, "DMA bytes should reset to 0");
    CHECK(c.compute.total_macs == 0, "MACs should reset to 0");
    CHECK(c.power.pj_per_mac == pj_mac, "Energy params should be preserved");
    CHECK(c.power.pj_per_sram_read == pj_sram, "SRAM energy params should be preserved");
    PASS();
}

/* ---- Test: Idle/Pipeline Bubble Tracking ---- */
static void test_perf_idle_and_bubbles(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    tu_perf_compute_record_idle(&c, 50);
    CHECK(c.compute.compute_idle_cycles == 50, "Idle cycles should be 50");

    tu_perf_compute_record_pipeline_bubble(&c, 3);
    CHECK(c.compute.compute_pipeline_bubbles == 3, "Pipeline bubbles should be 3");
    PASS();
}

/* ---- Test: DMA descriptor integration helper ---- */
static void test_perf_from_dma_descriptor(void) {
    tu_perf_counters_t c;
    tu_perf_init(&c, 1000.0);

    tu_perf_from_dma_descriptor(&c, 1024, 0, 0, true, 10, 2, 1);

    CHECK(c.dma.dma_read_bytes == 1024, "Should record DMA read");
    CHECK(c.dma.dma_transfers_linear == 1, "Should be linear transfer");
    CHECK(c.dma.dma_channel_bytes[0] == 1024, "Channel 0 bytes should match");
    CHECK(c.memory.mem_spad_stall_cycles == 1, "SRAM stall should be recorded");
    PASS();
}

/* ---- Test Runner ---- */
int main(void) {
    test_stats_init();

    TEST("perf_init");               test_perf_init();
    TEST("perf_enable_disable");     test_perf_enable_disable();
    TEST("perf_dma_recording");      test_perf_dma_recording();
    TEST("perf_compute_recording");  test_perf_compute_recording();
    TEST("perf_memory_recording");   test_perf_memory_recording();
    TEST("perf_snapshot_diff_merge");test_perf_snapshot_diff_merge();
    TEST("perf_power_accounting");   test_perf_power_accounting();
    TEST("perf_metrics");            test_perf_metrics();
    TEST("perf_reporting");          test_perf_reporting();
    TEST("perf_reset");              test_perf_reset();
    TEST("perf_idle_and_bubbles");   test_perf_idle_and_bubbles();
    TEST("perf_from_dma_descriptor");test_perf_from_dma_descriptor();

    return test_exit();
}
