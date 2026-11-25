/* All input rows must be consumed before the first result is written. */
#include "tu_cluster.h"
#include <stdio.h>
#include <string.h>

enum { CAP = 64, CORES = 3 };
static int check(unsigned src, unsigned dst, unsigned count) {
    tu_runtime_config_t cfg = tu_runtime_config_default();
    cfg.sram_o_size = CAP;
    tu_cluster_t *cl = tu_cluster_create(count, TU_TOPOLOGY_RING, 0, &cfg);
    if (!cl) return 1;
    unsigned char expected[CORES][CAP];
    float sum[2] = {0, 0};
    int failed = 0;
    for (unsigned i = 0; i < count; ++i) {
        tu_sram_region_t *r = tu_core_get_sram_o(cl->cores[i]);
        if (r->total_size != CAP) { tu_cluster_destroy(cl); return 1; }
        float row[2] = {i + 1, 2 * (i + 1)};
        memset(expected[i], 0x35, CAP);
        memcpy(expected[i] + src, row, sizeof(row));
        memcpy(tu_sram_raw_ptr(r), expected[i], CAP);
        sum[0] += row[0]; sum[1] += row[1];
    }
    failed |= tu_cluster_allreduce_sum_f32(cl, src, dst, 2) != 0;
    for (unsigned i = 0; i < count; ++i) {
        tu_sram_region_t *r = tu_core_get_sram_o(cl->cores[i]);
        memcpy(expected[i] + dst, sum, sizeof(sum));
        failed |= memcmp(expected[i], tu_sram_raw_ptr(r), CAP) != 0;
        failed |= r->banks.reads != 2 || r->banks.writes != 2;
    }
    failed |= cl->stats.total_icc_messages != count - 1 ||
              cl->stats.total_icc_bytes != sizeof(sum) * (count - 1) ||
              cl->stats.total_icc_cycles != 0;
    printf("cores=%u src=%u dst=%u: %s\n", count, src, dst, failed ? "FAIL" : "PASS");
    tu_cluster_destroy(cl);
    return failed;
}

int main(void) {
    static const unsigned layouts[][2] = {{0, 56}, {56, 0}, {56, 56}, {4, 8}};
    int failed = 0;
    for (unsigned count = 1; count <= CORES; count += 2)
        for (unsigned i = 0; i < sizeof(layouts) / sizeof(*layouts); ++i)
            failed |= check(layouts[i][0], layouts[i][1], count);
    return failed;
}
