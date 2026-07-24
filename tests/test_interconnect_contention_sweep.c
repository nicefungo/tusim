/* Cmodel-linked traffic-matrix sweep for shared directed-link contention. */
#include "tu_cmodel/tu_cluster.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t build_neighbor(tu_icc_message_t *m, uint32_t n, uint32_t bytes) {
    for (uint32_t i = 0; i < n; ++i) {
        m[i] = (tu_icc_message_t){.src_core_id = i,
                                  .dst_core_id = (i + 1) % n,
                                  .size_bytes = bytes};
    }
    return n;
}

static uint32_t build_hotspot(tu_icc_message_t *m, uint32_t n, uint32_t bytes) {
    uint32_t count = 0;
    for (uint32_t i = 1; i < n; ++i)
        m[count++] = (tu_icc_message_t){.src_core_id = i,
                                        .dst_core_id = 0,
                                        .size_bytes = bytes};
    return count;
}

static uint32_t build_all_to_all(tu_icc_message_t *m, uint32_t n, uint32_t bytes) {
    uint32_t count = 0;
    for (uint32_t src = 0; src < n; ++src)
        for (uint32_t dst = 0; dst < n; ++dst)
            if (src != dst)
                m[count++] = (tu_icc_message_t){.src_core_id = src,
                                                .dst_core_id = dst,
                                                .size_bytes = bytes};
    return count;
}

static int run_case(uint32_t n, tu_topology_t topology, const char *topology_name,
                    const char *traffic_name, tu_icc_message_t *messages,
                    uint32_t count) {
    tu_cluster_t cluster;
    memset(&cluster, 0, sizeof(cluster));
    cluster.num_cores = n;
    cluster.topology = topology;
    cluster.mesh_rows = n == 8 ? 2 : 4;
    cluster.mesh_cols = (n + cluster.mesh_rows - 1) / cluster.mesh_rows;
    cluster.hop_latency = 5;
    cluster.switching_mode = TU_ICC_SWITCH_CUT_THROUGH;
    cluster.link_bytes_per_cycle = 16;

    tu_icc_traffic_stats_t ideal, shared;
    cluster.contention_mode = TU_ICC_CONTENTION_IDEAL_PARALLEL;
    if (tu_cluster_estimate_traffic_cycles(&cluster, messages, count, &ideal) != 0)
        return -1;
    cluster.contention_mode = TU_ICC_CONTENTION_SHARED_LINK;
    if (tu_cluster_estimate_traffic_cycles(&cluster, messages, count, &shared) != 0)
        return -1;

    printf("%-5u %-5s %-10s %-5u %-7llu %-7llu %-7.2f %-7llu %u->%u\n",
           n, topology_name, traffic_name, count,
           (unsigned long long)ideal.estimated_cycles,
           (unsigned long long)shared.estimated_cycles,
           ideal.estimated_cycles ?
               (double)shared.estimated_cycles / (double)ideal.estimated_cycles : 0.0,
           (unsigned long long)shared.bottleneck_link_cycles,
           shared.bottleneck_src, shared.bottleneck_dst);
    return 0;
}

int main(void) {
    const uint32_t payload = 4096;
    puts("Interconnect contention sweep: cut-through, 16 B/cycle, 5 cycles/router, 4096 B/message");
    puts("cores topo  traffic    msgs  ideal   shared  penalty bottlnk link");
    for (uint32_t n = 8; n <= 16; n *= 2) {
        tu_icc_message_t *messages = calloc((size_t)n * n, sizeof(*messages));
        if (!messages) return 1;
        const tu_topology_t topologies[] = {TU_TOPOLOGY_RING, TU_TOPOLOGY_MESH};
        const char *names[] = {"RING", "MESH"};
        for (size_t t = 0; t < 2; ++t) {
            uint32_t count = build_neighbor(messages, n, payload);
            if (run_case(n, topologies[t], names[t], "neighbor", messages, count)) return 1;
            count = build_hotspot(messages, n, payload);
            if (run_case(n, topologies[t], names[t], "hotspot", messages, count)) return 1;
            count = build_all_to_all(messages, n, payload);
            if (run_case(n, topologies[t], names[t], "all-to-all", messages, count)) return 1;
        }
        free(messages);
    }
    return 0;
}
