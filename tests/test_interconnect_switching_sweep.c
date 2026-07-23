/* Analytical sweep for physically plausible on-chip switching modes.
 * Mirrors tu_cluster_estimate_transfer_cycles() without cmodel state. */
#include <stdint.h>
#include <stdio.h>

static uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
static uint64_t cut_through(uint32_t hops, uint32_t bytes,
                            uint32_t width, uint32_t router) {
    return (uint64_t)hops * router + ceil_div(bytes, width);
}
static uint64_t store_forward(uint32_t hops, uint32_t bytes,
                              uint32_t width, uint32_t router) {
    return (uint64_t)hops * (router + ceil_div(bytes, width));
}

int main(void) {
    const uint32_t hops[] = {1, 3, 7};
    const uint32_t bytes[] = {64, 1024, 65536};
    const uint32_t width = 16, router = 5;

    puts("Interconnect switching sweep (16 B/cycle, 5 cycles/router)");
    puts("bytes  hops  cut_through  store_forward  sf/ct");
    for (size_t b = 0; b < sizeof(bytes) / sizeof(bytes[0]); ++b) {
        for (size_t h = 0; h < sizeof(hops) / sizeof(hops[0]); ++h) {
            uint64_t ct = cut_through(hops[h], bytes[b], width, router);
            uint64_t sf = store_forward(hops[h], bytes[b], width, router);
            printf("%-6u %-5u %-12llu %-14llu %.3f\n", bytes[b], hops[h],
                   (unsigned long long)ct, (unsigned long long)sf,
                   (double)sf / (double)ct);
        }
    }

    puts("\nWidth sensitivity (65536 B, 3 hops, 5 cycles/router)");
    puts("B/cycle  cut_through  store_forward  sf/ct");
    const uint32_t widths[] = {8, 16, 32};
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i) {
        uint64_t ct = cut_through(3, 65536, widths[i], router);
        uint64_t sf = store_forward(3, 65536, widths[i], router);
        printf("%-8u %-12llu %-14llu %.3f\n", widths[i],
               (unsigned long long)ct, (unsigned long long)sf,
               (double)sf / (double)ct);
    }
    return 0;
}
