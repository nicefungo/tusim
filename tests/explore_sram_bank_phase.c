/* Live global-observation refill versus a hypothetical per-bank first-use phase.
 * The alternative is an independent accounting calculation, not a cmodel mode. */
#include "tu_cmodel/tu_sram.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    static const uint64_t when[] = {0, 2, 4, 4, 6, 6};
    static const unsigned bank[] = {0, 1, 0, 1, 0, 1};
    static const uint64_t live_expected[] = {0, 0, 0, 0, 2, 2};
    static const uint64_t alt_expected[] = {0, 0, 0, 2, 2, 0};
    unsigned failures = 0, rows = 0;
    puts("direction,step,cycle,bank,live_penalty,first_use_hypothesis,live_last0,live_last1,data_ok");
    for (unsigned write = 0; write != 2; ++write) {
        tu_sram_region_t r;
        uint64_t last[2] = {0,0};
        int initialized[2] = {0,0}, remaining[2] = {1,1};
        uint32_t value[2] = {0x12345678u, 0x98abcdefu}, out = 0;
        tu_sram_init_runtime(&r, 8, "phase", 2, 4, 1, 2, 4);
        memcpy(r.banks.data, value, sizeof value);
        uint64_t prev = 0;
        for (unsigned i = 0; i != 6; ++i) {
            unsigned b = bank[i];
            tu_sram_advance_cycle(&r, when[i] - prev);
            prev = when[i];
            if (!initialized[b]) { initialized[b] = 1; last[b] = when[i]; }
            else if (when[i] - last[b] >= 4) { remaining[b] = 1; last[b] = when[i]; }
            uint64_t alt = remaining[b] ? 0 : 2;
            if (remaining[b]) --remaining[b];
            uint64_t live = write ? tu_sram_write(&r, b*4, &value[b]) : tu_sram_read(&r, b*4, &out);
            uint32_t observed; memcpy(&observed, r.banks.data + b*4, 4);
            int ok = observed == value[b] && (write || out == value[b]);
            printf("%s,%u,%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d\n",
                   write ? "write" : "read", i, when[i], b, live, alt,
                   r.banks.bw_banks[0].last_refill_cycle, r.banks.bw_banks[1].last_refill_cycle, ok);
            if (live != live_expected[i] || alt != alt_expected[i] || !ok) ++failures;
            ++rows;
        }
        tu_sram_destroy(&r);
    }
    fprintf(stderr, "rows=%u failures=%u\n", rows, failures);
    return failures ? 1 : 0;
}
