/* Bounds regression: 32-byte allocation with 16-byte logical capacity keeps
 * old-code safe probes inside host memory. --safe excludes huge addresses. */
#include "tu_cmodel/tu_sram.h"
#include "tu_cmodel/tu_status.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); } } while (0)

static void run_case(int bulk, int write, uint32_t offset, uint32_t length,
                     int valid) {
    tu_sram_region_t r;
    uint8_t before[32], input[8], output[8], output_before[8];
    memset(input, 0x37, sizeof(input));
    memset(output, 0xa5, sizeof(output));
    memcpy(output_before, output, sizeof(output));
    tu_sram_init_runtime(&r, 32, "bounds-test", 4, 4, 1, 2, 4);
    if (!r.banks.data || !r.banks.bw_banks) { ++failures; return; }
    memset(r.banks.data, 0x5a, 32);
    memcpy(before, r.banks.data, 32);
    r.total_size = 16;
    tu_clear_error();
    uint64_t stall;
    if (write) stall = bulk ? tu_sram_write_bulk(&r, offset, input, length)
                            : tu_sram_write(&r, offset, input);
    else stall = bulk ? tu_sram_read_bulk(&r, offset, output, length)
                      : tu_sram_read(&r, offset, output);
    const tu_error_t *e = tu_get_last_error();
    CHECK(valid ? e == NULL : (e && e->code == TU_ERR_SRAM_OVERFLOW));
    CHECK(stall == 0);
    if (!valid || length == 0) {
        CHECK(memcmp(before, r.banks.data, 32) == 0);
        CHECK(memcmp(output_before, output, sizeof(output)) == 0);
        CHECK(r.banks.reads == 0 && r.banks.writes == 0);
        CHECK(r.banks.stall_cycles == 0);
        for (unsigned i=0; i<4; ++i) {
            CHECK(r.banks.bw_banks[i].words_available == 1);
            CHECK(r.banks.bw_banks[i].total_cycles_used == 0);
        }
    } else {
        uint32_t n = bulk ? length : 4;
        if (write) {
            memcpy(before + offset, input, n);
            CHECK(memcmp(before, r.banks.data, 32) == 0);
            CHECK(r.banks.writes == 1 && r.banks.reads == 0);
        } else {
            CHECK(memcmp(output, before + offset, n) == 0);
            CHECK(memcmp(output+n, output_before+n, sizeof(output)-n) == 0);
            CHECK(r.banks.reads == 1 && r.banks.writes == 0);
        }
    }
    tu_sram_destroy(&r);
}

int main(int argc, char **argv) {
    int safe = argc > 1 && strcmp(argv[1], "--safe") == 0;
    tu_set_error_mode(TU_ERR_MODE_SILENT);
    for (int bulk=0; bulk<2; ++bulk) for (int write=0; write<2; ++write) {
        run_case(bulk,write,12,4,1);
        run_case(bulk,write,14,4,0);
        run_case(bulk,write,16,4,0);
        if (!safe) run_case(bulk,write,UINT32_MAX-1,4,0);
        if (bulk) {
            run_case(bulk,write,16,0,1);
            run_case(bulk,write,17,0,0);
            if (!safe) run_case(bulk,write,0,UINT32_MAX,0);
            run_case(bulk,write,15,1,1);
        }
    }
    printf("SRAM bounds %s: %u checks, %u failures\n", safe ? "safe" : "full", checks, failures);
    return failures != 0;
}
