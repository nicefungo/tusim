/* Direction-aware DMA base-latency propagation and cycle sweep. */
#include "tu_cmodel/tu_cmodel.h"
#include "tu_cmodel/dma_descriptor.h"
#include "tu_cmodel/infra/config.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BYTES 4096u
#define WIDTH_BITS 256u
static uint8_t host_src[BYTES];
static uint8_t host_dst[BYTES];

typedef struct {
    const char *name;
    uint32_t read_cycles;
    uint32_t write_cycles;
} timing_case_t;

static int propagation_gate(void) {
    static const char json[] =
        "{\"tu\":{\"memory\":{\"latency\":{\"dram_read\":15.25,\"dram_write\":35.5},"
        "\"dram\":{\"latency_domain\":\"physical_ns\",\"core_clock_ghz\":2.0}},"
        "\"dma\":{\"bus_width_bits\":256}}}";
    tu_config_t cfg;
    char err[160] = {0};
    if (tu_config_load_string(json, &cfg, err, sizeof(err)) != 0) return -1;
    if (tu_config_validate(&cfg, err, sizeof(err)) != 0) return -2;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    if (!rt.dma_latency_configured || rt.dma_read_latency_cycles != 31u ||
        rt.dma_write_latency_cycles != 71u) return -3;
    tu_init_with_config(&rt);
    if (g_tu_dma.read_latency_cycles != 31u ||
        g_tu_dma.write_latency_cycles != 71u) return -4;

    rt.dma_latency_configured = false;
    rt.dma_read_latency_cycles = 0;
    rt.dma_write_latency_cycles = 0;
    tu_init_with_config(&rt);
    if (g_tu_dma.read_latency_cycles != TU_LATENCY_DRAM_READ ||
        g_tu_dma.write_latency_cycles != TU_LATENCY_DRAM_WRITE) return -5;
    return 0;
}

static int run_direction(tu_dma_direction_t dir, uint32_t read_cycles,
                         uint32_t write_cycles, uint64_t *done_out) {
    tu_sram_region_t sram;
    tu_sram_init(&sram, BYTES, "dma-directional-latency");
    sram.banks.bw_modeling = false;
    memset(host_src, 0x5a, sizeof(host_src));
    memset(host_dst, 0, sizeof(host_dst));
    if (dir == TU_DMA_DIR_TU_TO_HOST)
        memcpy(tu_sram_raw_ptr(&sram), host_src, BYTES);

    tu_dma_init_config_timing(true, 1, 2, TU_DMA_BUS_MODE_INDEPENDENT,
                              TU_DMA_ARB_ROUND_ROBIN, TU_DMA_BIND_EXPLICIT,
                              WIDTH_BITS, read_cycles, write_cycles);
    void *host = dir == TU_DMA_DIR_HOST_TO_TU ? (void *)host_src : (void *)host_dst;
    tu_dma_descriptor_t *d = tu_dma_desc_create_linear(
        0, dir, &sram, 0, host, 1, BYTES);
    if (!d || tu_dma_submit_desc(d) == 0) return -1;
    while (g_tu_dma.channels[0].total_completed == 0 &&
           g_tu_dma.current_cycle < 1000u)
        tu_dma_tick();

    uint32_t base = dir == TU_DMA_DIR_TU_TO_HOST ? write_cycles : read_cycles;
    uint64_t expected = 1u + base + BYTES / (WIDTH_BITS / 8u);
    *done_out = d->cycles_completed;
    if (*done_out != expected || g_tu_dma.current_cycle != expected) return -2;
    const uint8_t *got = dir == TU_DMA_DIR_HOST_TO_TU ?
                         (const uint8_t *)tu_sram_raw_ptr(&sram) : host_dst;
    if (memcmp(got, host_src, BYTES) != 0) return -3;

    tu_dma_destroy();
    d->next = NULL;
    tu_dma_desc_destroy(d);
    tu_sram_destroy(&sram);
    return 0;
}

int main(void) {
    if (propagation_gate() != 0) {
        fprintf(stderr, "FAIL: latency parse/runtime/default propagation\n");
        return 1;
    }
    const timing_case_t cases[] = {
        {"symmetric", 50u, 50u},
        {"read_fast", 30u, 70u},
        {"write_fast", 70u, 30u},
    };
    printf("DMA directional base-latency sweep (4096 B, 256 bit, SRAM meter disabled)\n");
    printf("mode read_base write_base load_complete store_complete\n");
    for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint64_t load_done = 0, store_done = 0;
        int rc = run_direction(TU_DMA_DIR_HOST_TO_TU, cases[i].read_cycles,
                               cases[i].write_cycles, &load_done);
        if (rc == 0)
            rc = run_direction(TU_DMA_DIR_TU_TO_HOST, cases[i].read_cycles,
                               cases[i].write_cycles, &store_done);
        if (rc != 0) {
            fprintf(stderr, "FAIL: %s rc=%d\n", cases[i].name, rc);
            return 2;
        }
        printf("%-10s %9u %10u %13lu %14lu\n", cases[i].name,
               cases[i].read_cycles, cases[i].write_cycles,
               (unsigned long)load_done, (unsigned long)store_done);
    }
    printf("PASS: directional service, physical-ns conversion, defaults, exact bytes/cycles\n");
    return 0;
}
