/* Observational gate, not an endorsement of legacy utilization semantics. */
#include "tu_cmodel/tu_sram.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int close_enough(float a, float b) { return fabsf(a - b) < 0.00001f; }

static int run(unsigned window, unsigned banks, int write) {
    tu_sram_region_t r;
    const uint32_t input = 0x12345678u;
    uint32_t output = input;
    uint64_t now = 0, reads, writes, rs, ws;
    const unsigned times[] = {0, 1, window-1, window, 2*window-1, 2*window, 2*window+1};
    tu_sram_init_runtime(&r, 32, "util_probe", banks, 4, 1, 2, window);
    memcpy(r.banks.data, &input, sizeof(input));
    uint64_t penalty = write ? tu_sram_write(&r, 0, &input) : tu_sram_read(&r, 0, &output);
    if (penalty || output != input) { tu_sram_destroy(&r); return 1; }
    for (unsigned i = 0; i < sizeof(times)/sizeof(times[0]); ++i) {
        tu_sram_advance_cycle(&r, times[i] - now);
        now = times[i];
        float bank_util = -1;
        tu_sram_get_bank_bw_stats(&r, 0, &reads, &writes, &rs, &ws, &bank_util);
        float global = tu_sram_get_bandwidth_utilization(&r);
        uint64_t periods = now / window;
        if (!periods) periods = 1;
        float expected_bank = 1.0f / (float)periods;
        float expected_global = now ? 1.0f / (float)(banks * periods) : 0.0f;
        float inclusive_bank = 1.0f / (float)(1 + now / window);
        float inclusive_global = inclusive_bank / banks;
        if (!close_enough(global, expected_global) || !close_enough(bank_util, expected_bank) ||
            reads != (uint64_t)!write || writes != (uint64_t)write || rs || ws ||
            memcmp(r.banks.data, &input, sizeof(input))) {
            fprintf(stderr, "FAIL window=%u banks=%u write=%d time=%llu\n", window, banks, write, (unsigned long long)now);
            tu_sram_destroy(&r); return 1;
        }
        printf("%u,%u,%s,%llu,%.6f,%.6f,%.6f,%.6f,%llu,%llu\n", window, banks,
               write ? "write" : "read", (unsigned long long)now, global, bank_util,
               inclusive_global, inclusive_bank, (unsigned long long)reads, (unsigned long long)writes);
    }
    tu_sram_destroy(&r); return 0;
}
int main(void) {
    puts("window,banks,direction,cycle,legacy_global,legacy_bank,inclusive_global,inclusive_bank,reads,writes");
    for (unsigned window = 4; window <= 8; window *= 2)
        for (unsigned banks = 1; banks <= 2; ++banks)
            for (int write = 0; write <= 1; ++write)
                if (run(window, banks, write)) return 1;
    fputs("PASS: 56 pinned observations, bytes and counters\n", stderr);
    return 0;
}
