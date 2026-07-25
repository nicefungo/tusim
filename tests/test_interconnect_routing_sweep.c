/* Cmodel-linked deterministic XY/YX mesh-routing sweep. */
#include "tu_cmodel/tu_cluster.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TOP_ROW_TO_LEFT_COL,
    LEFT_COL_TO_BOTTOM_ROW,
    ALL_TO_ALL
} traffic_kind_t;

static uint32_t build_traffic(tu_icc_message_t *messages, uint32_t side,
                              uint32_t bytes, traffic_kind_t kind) {
    uint32_t count = 0;
    uint32_t n = side * side;
    if (kind == TOP_ROW_TO_LEFT_COL) {
        for (uint32_t src_col = 1; src_col < side; ++src_col)
            for (uint32_t dst_row = 1; dst_row < side; ++dst_row)
                messages[count++] = (tu_icc_message_t){
                    .src_core_id = src_col,
                    .dst_core_id = dst_row * side,
                    .size_bytes = bytes};
    } else if (kind == LEFT_COL_TO_BOTTOM_ROW) {
        for (uint32_t src_row = 0; src_row + 1 < side; ++src_row)
            for (uint32_t dst_col = 1; dst_col < side; ++dst_col)
                messages[count++] = (tu_icc_message_t){
                    .src_core_id = src_row * side,
                    .dst_core_id = (side - 1) * side + dst_col,
                    .size_bytes = bytes};
    } else {
        for (uint32_t src = 0; src < n; ++src)
            for (uint32_t dst = 0; dst < n; ++dst)
                if (src != dst)
                    messages[count++] = (tu_icc_message_t){
                        .src_core_id = src, .dst_core_id = dst,
                        .size_bytes = bytes};
    }
    return count;
}

static int run_case(uint32_t side, const char *traffic_name,
                    traffic_kind_t kind, uint32_t bytes) {
    uint32_t n = side * side;
    tu_icc_message_t *messages = calloc((size_t)n * n, sizeof(*messages));
    if (!messages) return -1;
    uint32_t count = build_traffic(messages, side, bytes, kind);

    tu_cluster_t cluster;
    memset(&cluster, 0, sizeof(cluster));
    cluster.num_cores = n;
    cluster.topology = TU_TOPOLOGY_MESH;
    cluster.mesh_rows = side;
    cluster.mesh_cols = side;
    cluster.hop_latency = 5;
    cluster.switching_mode = TU_ICC_SWITCH_CUT_THROUGH;
    cluster.contention_mode = TU_ICC_CONTENTION_SHARED_LINK;
    cluster.link_bytes_per_cycle = 16;

    for (int mode = TU_ICC_MESH_ROUTE_XY; mode <= TU_ICC_MESH_ROUTE_YX; ++mode) {
        tu_icc_traffic_stats_t stats;
        cluster.mesh_routing_mode = mode;
        if (tu_cluster_estimate_traffic_cycles(&cluster, messages, count, &stats) != 0) {
            free(messages);
            return -1;
        }
        printf("%-4ux%-4u %-18s %-2s %-5u %-7llu %-7llu %u->%u\n",
               side, side, traffic_name, mode == TU_ICC_MESH_ROUTE_XY ? "XY" : "YX",
               count, (unsigned long long)stats.estimated_cycles,
               (unsigned long long)stats.bottleneck_link_cycles,
               stats.bottleneck_src, stats.bottleneck_dst);
    }
    free(messages);
    return 0;
}

int main(void) {
    const uint32_t bytes = 4096;
    puts("Mesh routing sweep: cut-through, shared links, 16 B/cycle, 5 cycles/router, 4096 B/message");
    puts("mesh      traffic            rt msgs  cycles  bottlnk link");
    for (uint32_t side = 3; side <= 4; ++side) {
        if (run_case(side, "top-row->left-col", TOP_ROW_TO_LEFT_COL, bytes)) return 1;
        if (run_case(side, "left-col->bottom", LEFT_COL_TO_BOTTOM_ROW, bytes)) return 1;
        if (run_case(side, "all-to-all", ALL_TO_ALL, bytes)) return 1;
    }
    return 0;
}
