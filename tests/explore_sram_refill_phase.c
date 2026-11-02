/* Observe how SRAM refill phase depends on advance-call boundaries.
 * This is an exploration of the existing API, not a physical timing model. */
#include "tu_cmodel/tu_sram.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int run(unsigned window, int write, int intermediate) {
    tu_sram_region_t r;
    uint32_t value = 0x12345678u, out = 0;
    uint64_t stall[3];
    tu_sram_init_runtime(&r, 32, "phase", 1, 4, 1, 2, window);
    if (!r.banks.data || !r.banks.bw_banks) return 1;
#define ACCESS(i) do { \
    if (write) stall[i] = tu_sram_write(&r, 0, &value); \
    else stall[i] = tu_sram_read(&r, 0, &out); \
} while (0)
    ACCESS(0);
    if (intermediate) {
        tu_sram_advance_cycle(&r, window);
        tu_sram_advance_cycle(&r, 1);
    } else {
        tu_sram_advance_cycle(&r, window + 1);
    }
    ACCESS(1);
    tu_sram_advance_cycle(&r, window - 1);
    ACCESS(2);
    uint64_t served = write ? r.banks.bw_banks[0].writes_served : r.banks.bw_banks[0].reads_served;
    uint64_t blocked = write ? r.banks.bw_banks[0].write_stalls : r.banks.bw_banks[0].read_stalls;
    int ok = r.banks.current_cycle == 2u * window && stall[0] == 0 && stall[1] == 0 &&
             stall[2] == (intermediate ? 0u : 2u) && served == (intermediate ? 3u : 2u) &&
             blocked == (intermediate ? 0u : 1u) &&
             r.banks.stall_cycles == (intermediate ? 0u : 2u) &&
             (write ? memcmp(r.banks.data, &value, sizeof value) == 0 : out == 0);
    printf("%u,%s,%s,0;%u;%u,0;0;%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s\n",
           window, write ? "write" : "read", intermediate ? "intermediate" : "batched",
           window + 1, 2 * window, stall[2], served, blocked,
           r.banks.bw_banks[0].last_refill_cycle, ok ? "PASS" : "FAIL");
    tu_sram_destroy(&r);
    return !ok;
#undef ACCESS
}

int main(void) {
    int failures = 0;
    puts("window,direction,advance_calls,access_cycles,stalls,served,stalled,last_refill,result");
    for (unsigned window = 4; window <= 8; window += 4)
        for (int write = 0; write < 2; ++write)
            for (int intermediate = 0; intermediate < 2; ++intermediate)
                failures += run(window, write, intermediate);
    printf("failures=%d\n", failures);
    return failures != 0;
}
