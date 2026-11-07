/* Exercise legacy utilization denominators with a wide independent oracle.
 * Synthetic large cycle advances test integer representation, not performance. */
#include "tu_cmodel/tu_sram.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    int baseline = argc == 2 && strcmp(argv[1], "--baseline") == 0;
    if (argc > 2 || (argc == 2 && !baseline)) return 2;
    static const uint64_t times[] = {10, UINT64_C(1) << 63, (UINT64_C(1) << 63) + 1};
    static const float legacy_global[] = {0.05f, 0.0f, 0.5f};
    static const float legacy_one_bank[] = {0.05f, 0.0f, 0.5f};
    unsigned failures = 0, rows = 0;
    puts("layout,cycle,served,reported_global,reported_bank0,wide_global,wide_bank0,data_ok");
    for (unsigned layout = 0; layout < 2; ++layout) {
        for (unsigned i = 0; i < sizeof times / sizeof times[0]; ++i) {
            tu_sram_region_t r;
            uint32_t src = 0x12345678u, out = 0;
            unsigned banks = layout ? 1 : 2;
            unsigned grants = layout ? 2 : 1;
            tu_sram_init_runtime(&r, 8, "util-overflow", banks, 4, grants, 2, 1);
            memcpy(r.banks.data, &src, sizeof src);
            uint64_t stall = tu_sram_read(&r, 0, &out);
            tu_sram_advance_cycle(&r, times[i]);
            float global = tu_sram_get_bandwidth_utilization(&r), bank = -1;
            uint64_t served = 0;
            tu_sram_get_bank_bw_stats(&r, 0, &served, NULL, NULL, NULL, &bank);
            long double global_ref = 1.0L / ((long double)banks * grants * times[i]);
            long double bank_ref = 1.0L / ((long double)grants * times[i]);
            float global_expected = baseline ? legacy_global[i] : (float)global_ref;
            float bank_expected = baseline && layout ? legacy_one_bank[i] : (float)bank_ref;
            int data_ok = out == src && stall == 0 && served == 1;
            printf("%s,%" PRIu64 ",%" PRIu64 ",%.9g,%.9g,%.9Lg,%.9Lg,%d\n",
                   layout ? "one-bank-two-grants" : "two-banks-one-grant", times[i], served,
                   global, bank, global_ref, bank_ref, data_ok);
            if (!data_ok || !isfinite(global) || !isfinite(bank) ||
                global != global_expected || bank != bank_expected) ++failures;
            ++rows;
            tu_sram_destroy(&r);
        }
    }
    fprintf(stderr, "rows=%u failures=%u\n", rows, failures);
    return failures ? 1 : 0;
}
