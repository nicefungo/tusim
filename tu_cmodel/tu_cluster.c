/*
 * TU Cluster — Multi-Core Implementation (Gap A5)
 * ================================================
 */

#include "tu_core.h"
#include "tu_cluster.h"
#include "tu_sram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Helpers ---- */

static uint32_t tu_abs_diff(uint32_t a, uint32_t b) {
    return a > b ? a - b : b - a;
}

/* ---- Lifecycle ---- */

tu_cluster_t *tu_cluster_create(uint32_t num_cores,
                                 tu_topology_t topology,
                                 uint32_t mesh_rows,
                                 const tu_runtime_config_t *base_config) {
    if (num_cores == 0 || num_cores > 256) {
        fprintf(stderr, "tu_cluster_create: invalid num_cores=%u (max 256)\n", num_cores);
        return NULL;
    }

    if (topology == TU_TOPOLOGY_MESH) {
        if (mesh_rows == 0) {
            fprintf(stderr, "tu_cluster_create: mesh_rows must be > 0 for MESH topology\n");
            return NULL;
        }
    }

    tu_cluster_t *cluster = calloc(1, sizeof(tu_cluster_t));
    if (!cluster) return NULL;

    cluster->num_cores = num_cores;
    cluster->topology = topology;
    cluster->hop_latency = base_config ? base_config->icc_router_latency_cycles :
                                         TU_ICC_ROUTER_LATENCY_CYCLES;
    cluster->switching_mode = base_config ? base_config->icc_switching_mode :
                                            TU_ICC_SWITCHING_MODE;
    cluster->contention_mode = base_config ? base_config->icc_contention_mode :
                                            TU_ICC_CONTENTION_MODE;
    cluster->mesh_routing_mode = base_config ? base_config->icc_mesh_routing_mode :
                                               TU_ICC_MESH_ROUTING_MODE;
    cluster->link_bytes_per_cycle = base_config ? base_config->icc_link_bytes_per_cycle :
                                                  TU_ICC_LINK_BYTES_PER_CYCLE;

    if (topology == TU_TOPOLOGY_MESH) {
        cluster->mesh_rows = mesh_rows;
        cluster->mesh_cols = (num_cores + mesh_rows - 1) / mesh_rows;
    }

    /* Allocate core array */
    cluster->cores = calloc(num_cores, sizeof(tu_core_t *));
    if (!cluster->cores) {
        free(cluster);
        return NULL;
    }

    /* Create cores */
    tu_runtime_config_t cfg;
    if (base_config) {
        cfg = *base_config;
    } else {
        cfg = tu_runtime_config_default();
    }

    for (uint32_t i = 0; i < num_cores; i++) {
        cluster->cores[i] = tu_core_create_with_id(i, &cfg);
        if (!cluster->cores[i]) {
            /* Cleanup on failure */
            for (uint32_t j = 0; j < i; j++) {
                tu_core_destroy(cluster->cores[j]);
            }
            free(cluster->cores);
            free(cluster);
            fprintf(stderr, "tu_cluster_create: failed to create core %u\n", i);
            return NULL;
        }
    }

    cluster->initialized = true;

    TU_LOG_INFO(TU_COMP_CORE, "tu_cluster_t created: %u cores, topology=%s",
                num_cores,
                topology == TU_TOPOLOGY_RING ? "ring" :
                topology == TU_TOPOLOGY_MESH ? "mesh" : "none");

    return cluster;
}

void tu_cluster_destroy(tu_cluster_t *cluster) {
    if (!cluster) return;

    for (uint32_t i = 0; i < cluster->num_cores; i++) {
        tu_core_destroy(cluster->cores[i]);
    }

    free(cluster->cores);
    free(cluster);

    TU_LOG_INFO(TU_COMP_CORE, "tu_cluster_t destroyed");
}

tu_core_t *tu_cluster_get_core(tu_cluster_t *cluster, uint32_t core_id) {
    if (!cluster || core_id >= cluster->num_cores) return NULL;
    return cluster->cores[core_id];
}

/* ---- Topology Helpers ---- */

uint32_t tu_cluster_hop_distance(const tu_cluster_t *cluster,
                                  uint32_t src, uint32_t dst) {
    if (!cluster || src >= cluster->num_cores || dst >= cluster->num_cores) {
        return UINT32_MAX;
    }
    if (src == dst) return 0;

    switch (cluster->topology) {
    case TU_TOPOLOGY_NONE:
        return UINT32_MAX;

    case TU_TOPOLOGY_RING: {
        /* Min of forward and backward distance */
        uint32_t forward = (dst + cluster->num_cores - src) % cluster->num_cores;
        uint32_t backward = (src + cluster->num_cores - dst) % cluster->num_cores;
        return forward < backward ? forward : backward;
    }

    case TU_TOPOLOGY_MESH: {
        uint32_t src_row = src / cluster->mesh_cols;
        uint32_t src_col = src % cluster->mesh_cols;
        uint32_t dst_row = dst / cluster->mesh_cols;
        uint32_t dst_col = dst % cluster->mesh_cols;

        if (src_row >= cluster->mesh_rows || dst_row >= cluster->mesh_rows) {
            return UINT32_MAX;
        }

        return tu_abs_diff(src_row, dst_row) + tu_abs_diff(src_col, dst_col);
    }

    default:
        return UINT32_MAX;
    }
}

uint64_t tu_cluster_estimate_transfer_cycles(const tu_cluster_t *cluster,
                                              uint32_t src, uint32_t dst,
                                              uint32_t size_bytes) {
    if (!cluster) return UINT64_MAX;
    uint32_t hops = tu_cluster_hop_distance(cluster, src, dst);
    if (hops == UINT32_MAX) return UINT64_MAX;
    if (hops == 0 || size_bytes == 0) return 0;

    uint64_t route_cycles = (uint64_t)hops * cluster->hop_latency;
    if (cluster->switching_mode == TU_ICC_SWITCH_LEGACY_HOP_ONLY)
        return route_cycles;
    if (cluster->link_bytes_per_cycle == 0) return UINT64_MAX;

    uint64_t serialization =
        ((uint64_t)size_bytes + cluster->link_bytes_per_cycle - 1) /
        cluster->link_bytes_per_cycle;
    if (cluster->switching_mode == TU_ICC_SWITCH_CUT_THROUGH)
        return route_cycles + serialization;
    if (cluster->switching_mode == TU_ICC_SWITCH_STORE_FORWARD)
        return (uint64_t)hops * (cluster->hop_latency + serialization);
    return UINT64_MAX;
}

static void add_link_service(uint64_t *links, uint32_t n,
                             uint32_t src, uint32_t dst, uint64_t service) {
    links[(uint64_t)src * n + dst] += service;
}

static int add_route_service(const tu_cluster_t *cluster, uint32_t src,
                             uint32_t dst, uint64_t service, uint64_t *links) {
    uint32_t n = cluster->num_cores;
    if (cluster->topology == TU_TOPOLOGY_RING) {
        uint32_t forward = (dst + n - src) % n;
        uint32_t backward = (src + n - dst) % n;
        bool clockwise = forward <= backward;
        uint32_t cur = src;
        while (cur != dst) {
            uint32_t next = clockwise ? (cur + 1) % n : (cur + n - 1) % n;
            add_link_service(links, n, cur, next, service);
            cur = next;
        }
        return 0;
    }
    if (cluster->topology == TU_TOPOLOGY_MESH) {
        uint32_t cols = cluster->mesh_cols;
        uint32_t cur = src;
        uint32_t dst_row = dst / cols, dst_col = dst % cols;
        bool x_first = cluster->mesh_routing_mode == TU_ICC_MESH_ROUTE_XY;
        if (!x_first && cluster->mesh_routing_mode != TU_ICC_MESH_ROUTE_YX)
            return -1;
        for (int dimension = 0; dimension < 2; ++dimension) {
            bool route_x = dimension == 0 ? x_first : !x_first;
            if (route_x) {
                while (cur % cols != dst_col) {
                    uint32_t next = (cur % cols < dst_col) ? cur + 1 : cur - 1;
                    if (next >= n || next / cols != cur / cols) return -1;
                    add_link_service(links, n, cur, next, service);
                    cur = next;
                }
            } else {
                while (cur / cols != dst_row) {
                    uint32_t next = (cur / cols < dst_row) ? cur + cols : cur - cols;
                    if (next >= n) return -1;
                    add_link_service(links, n, cur, next, service);
                    cur = next;
                }
            }
        }
        return 0;
    }
    return src == dst ? 0 : -1;
}

int tu_cluster_estimate_traffic_cycles(const tu_cluster_t *cluster,
                                       const tu_icc_message_t *messages,
                                       uint32_t message_count,
                                       tu_icc_traffic_stats_t *stats) {
    if (!cluster || !stats || (message_count && !messages)) return -1;
    memset(stats, 0, sizeof(*stats));
    stats->bottleneck_src = UINT32_MAX;
    stats->bottleneck_dst = UINT32_MAX;
    if (message_count == 0) return 0;

    uint32_t n = cluster->num_cores;
    uint64_t *links = calloc((uint64_t)n * n, sizeof(*links));
    if (!links) return -1;
    uint64_t max_route_cycles = 0;

    for (uint32_t i = 0; i < message_count; ++i) {
        const tu_icc_message_t *m = &messages[i];
        uint64_t isolated = tu_cluster_estimate_transfer_cycles(
            cluster, m->src_core_id, m->dst_core_id, m->size_bytes);
        if (isolated == UINT64_MAX) { free(links); return -1; }
        if (isolated > stats->isolated_cycles) stats->isolated_cycles = isolated;
        uint32_t hops = tu_cluster_hop_distance(cluster, m->src_core_id,
                                                 m->dst_core_id);
        uint64_t route = (uint64_t)hops * cluster->hop_latency;
        if (route > max_route_cycles) max_route_cycles = route;
        uint64_t service = 0;
        if (cluster->switching_mode != TU_ICC_SWITCH_LEGACY_HOP_ONLY &&
            m->size_bytes != 0) {
            service = ((uint64_t)m->size_bytes + cluster->link_bytes_per_cycle - 1) /
                      cluster->link_bytes_per_cycle;
        }
        if (add_route_service(cluster, m->src_core_id, m->dst_core_id,
                              service, links) != 0) {
            free(links);
            return -1;
        }
    }

    for (uint32_t src = 0; src < n; ++src) {
        for (uint32_t dst = 0; dst < n; ++dst) {
            uint64_t load = links[(uint64_t)src * n + dst];
            if (load > stats->bottleneck_link_cycles) {
                stats->bottleneck_link_cycles = load;
                stats->bottleneck_src = src;
                stats->bottleneck_dst = dst;
            }
        }
    }
    free(links);

    stats->estimated_cycles = stats->isolated_cycles;
    if (cluster->contention_mode == TU_ICC_CONTENTION_SHARED_LINK) {
        uint64_t shared_bound = stats->bottleneck_link_cycles + max_route_cycles;
        if (shared_bound > stats->estimated_cycles)
            stats->estimated_cycles = shared_bound;
    } else if (cluster->contention_mode != TU_ICC_CONTENTION_IDEAL_PARALLEL) {
        return -1;
    }
    return 0;
}

void tu_cluster_neighbors(const tu_cluster_t *cluster,
                           uint32_t core_id,
                           uint32_t *neighbors,
                           uint32_t *num_neighbors) {
    if (!cluster || !neighbors || !num_neighbors) return;

    *num_neighbors = 0;
    for (int i = 0; i < 4; i++) neighbors[i] = UINT32_MAX;

    if (core_id >= cluster->num_cores) return;

    switch (cluster->topology) {
    case TU_TOPOLOGY_NONE:
        break;

    case TU_TOPOLOGY_RING:
        neighbors[0] = (core_id + cluster->num_cores - 1) % cluster->num_cores;
        neighbors[1] = (core_id + 1) % cluster->num_cores;
        if (neighbors[0] == core_id) neighbors[0] = UINT32_MAX;
        if (neighbors[1] == core_id) neighbors[1] = UINT32_MAX;
        *num_neighbors = (neighbors[0] != UINT32_MAX ? 1 : 0) +
                         (neighbors[1] != UINT32_MAX ? 1 : 0);
        break;

    case TU_TOPOLOGY_MESH: {
        uint32_t row = core_id / cluster->mesh_cols;
        uint32_t col = core_id % cluster->mesh_cols;
        uint32_t idx = 0;

        /* North */
        if (row > 0) neighbors[idx++] = core_id - cluster->mesh_cols;
        /* South */
        if (row + 1 < cluster->mesh_rows) {
            uint32_t s = core_id + cluster->mesh_cols;
            if (s < cluster->num_cores) neighbors[idx++] = s;
        }
        /* West */
        if (col > 0) neighbors[idx++] = core_id - 1;
        /* East */
        if (col + 1 < cluster->mesh_cols) {
            uint32_t e = core_id + 1;
            if (e < cluster->num_cores && (e / cluster->mesh_cols) == row) {
                neighbors[idx++] = e;
            }
        }
        *num_neighbors = idx;
        break;
    }
    }
}

/* ---- Inter-Core Communication ---- */

int tu_cluster_send(tu_cluster_t *cluster,
                     const tu_icc_message_t *msg) {
    if (!cluster || !msg) return -1;
    if (msg->src_core_id >= cluster->num_cores ||
        msg->dst_core_id >= cluster->num_cores) return -1;
    if (msg->size_bytes == 0) return 0;

    tu_core_t *src_core = cluster->cores[msg->src_core_id];
    tu_core_t *dst_core = cluster->cores[msg->dst_core_id];

    if (!src_core || !dst_core) return -1;
    if (!src_core->initialized || !dst_core->initialized) return -1;

    tu_sram_region_t *src_sram = tu_core_get_sram_o(src_core);
    tu_sram_region_t *dst_sram = tu_core_get_sram_o(dst_core);

    /* Validate offsets using total_size */
    if (msg->src_offset + msg->size_bytes > src_sram->total_size) return -1;
    if (msg->dst_offset + msg->size_bytes > dst_sram->total_size) return -1;

    /* Compute route and serialization latency for the configured switch. */
    uint64_t latency = tu_cluster_estimate_transfer_cycles(
        cluster, msg->src_core_id, msg->dst_core_id, msg->size_bytes);
    if (latency == UINT64_MAX) return -1;

    /* Perform the transfer using bulk SRAM read/write */
    /* Allocate temporary buffer */
    void *tmp = malloc(msg->size_bytes);
    if (!tmp) return -1;

    /* Read from source SRAM */
    tu_sram_read_bulk(src_sram, msg->src_offset, tmp, msg->size_bytes);

    /* Write to destination SRAM */
    tu_sram_write_bulk(dst_sram, msg->dst_offset, tmp, msg->size_bytes);

    free(tmp);

    /* Update stats */
    cluster->stats.total_icc_messages++;
    cluster->stats.total_icc_bytes += msg->size_bytes;
    cluster->stats.total_icc_cycles += latency;

    /* Update cycle counters on the destination core */
    dst_core->state.estimated_cycles += latency;

    return 0;
}

int tu_cluster_broadcast(tu_cluster_t *cluster,
                          uint32_t src_core_id,
                          uint32_t src_offset,
                          uint32_t dst_offset,
                          uint32_t size_bytes) {
    if (!cluster || src_core_id >= cluster->num_cores) return -1;
    if (size_bytes == 0) return 0;

    for (uint32_t dst = 0; dst < cluster->num_cores; dst++) {
        if (dst == src_core_id) continue;

        tu_icc_message_t msg = {
            .src_core_id = src_core_id,
            .dst_core_id = dst,
            .src_offset = src_offset,
            .dst_offset = dst_offset,
            .size_bytes = size_bytes,
            .tag = 0,
            .blocking = true,
            .latency_cycles = 0,
        };

        int rc = tu_cluster_send(cluster, &msg);
        if (rc != 0) return rc;
    }

    return 0;
}

int tu_cluster_allreduce_sum_f32(tu_cluster_t *cluster,
                                  uint32_t src_offset,
                                  uint32_t dst_offset,
                                  uint32_t num_elements) {
    if (!cluster || num_elements == 0) return -1;

    uint32_t size_bytes = num_elements * sizeof(float);

    /* Step 1: Gather all data to core 0 */
    float *accumulator = calloc(num_elements, sizeof(float));
    if (!accumulator) return -1;

    /* Read from core 0 first */
    {
        tu_core_t *core = cluster->cores[0];
        tu_sram_region_t *sram = tu_core_get_sram_o(core);
        tu_sram_read_bulk(sram, src_offset, accumulator, size_bytes);
    }

    /* Add data from remaining cores */
    for (uint32_t c = 1; c < cluster->num_cores; c++) {
        tu_core_t *core = cluster->cores[c];
        if (!core || !core->initialized) {
            free(accumulator);
            return -1;
        }

        tu_sram_region_t *sram = tu_core_get_sram_o(core);
        float *tmp = malloc(size_bytes);
        if (!tmp) { free(accumulator); return -1; }

        tu_sram_read_bulk(sram, src_offset, tmp, size_bytes);
        for (uint32_t i = 0; i < num_elements; i++) {
            accumulator[i] += tmp[i];
        }
        free(tmp);
    }

    /* Step 2: Write result back to all cores */
    for (uint32_t c = 0; c < cluster->num_cores; c++) {
        tu_core_t *core = cluster->cores[c];
        tu_sram_region_t *sram = tu_core_get_sram_o(core);
        tu_sram_write_bulk(sram, dst_offset, accumulator, size_bytes);
    }

    cluster->stats.total_icc_messages += cluster->num_cores - 1;
    cluster->stats.total_icc_bytes += (uint64_t)size_bytes * (cluster->num_cores - 1);

    free(accumulator);
    return 0;
}

/* ---- Synchronization ---- */

int tu_cluster_barrier(tu_cluster_t *cluster) {
    if (!cluster) return -1;
    cluster->stats.total_barriers++;
    /* In a functional model, barrier is a no-op for correctness.
     * The cycle counter accounts for synchronization latency. */
    uint64_t barrier_cycles = cluster->hop_latency * 2;  /* Round-trip sync cost */
    for (uint32_t i = 0; i < cluster->num_cores; i++) {
        if (cluster->cores[i]) {
            cluster->cores[i]->state.estimated_cycles += barrier_cycles;
        }
    }
    return 0;
}

/* ---- SPMD Execution ---- */

int tu_cluster_spmd_execute(tu_cluster_t *cluster,
                             const char **programs,
                             const tu_host_buffer_t **buffers,
                             const int *n_buffers_per) {
    if (!cluster || !programs || !buffers || !n_buffers_per) return -1;

    int overall_rc = 0;

    for (uint32_t i = 0; i < cluster->num_cores; i++) {
        int rc = tu_core_execute_asm_text(cluster->cores[i],
                                           programs[i],
                                           buffers[i],
                                           n_buffers_per[i]);
        if (rc != 0) {
            fprintf(stderr, "tu_cluster_spmd_execute: core %u failed with rc=%d\n",
                    i, rc);
            overall_rc = rc;
        }
    }

    return overall_rc;
}

/* ---- Statistics ---- */

void tu_cluster_print_stats(const tu_cluster_t *cluster) {
    if (!cluster) return;

    printf("===== TU Cluster Stats =====\n");
    printf("  Cores:              %u\n", cluster->num_cores);
    printf("  Topology:           %s\n",
           cluster->topology == TU_TOPOLOGY_RING ? "ring" :
           cluster->topology == TU_TOPOLOGY_MESH ? "mesh" : "none");
    if (cluster->topology == TU_TOPOLOGY_MESH) {
        printf("  Mesh dims:          %u × %u\n",
               cluster->mesh_rows, cluster->mesh_cols);
    }
    printf("  Hop latency:        %u cycles\n", cluster->hop_latency);
    printf("  Switching mode:     %d (0=legacy, 1=cut-through, 2=store-forward)\n",
           cluster->switching_mode);
    printf("  Contention mode:    %d (0=ideal-parallel, 1=shared-link bound)\n",
           cluster->contention_mode);
    printf("  Mesh routing:       %s\n",
           cluster->mesh_routing_mode == TU_ICC_MESH_ROUTE_YX ? "YX" : "XY");
    printf("  Link width:         %u bytes/cycle\n", cluster->link_bytes_per_cycle);
    printf("  ICC messages:       %lu\n",
           (unsigned long)cluster->stats.total_icc_messages);
    printf("  ICC bytes:          %lu\n",
           (unsigned long)cluster->stats.total_icc_bytes);
    printf("  ICC cycles:         %lu\n",
           (unsigned long)cluster->stats.total_icc_cycles);
    printf("  Barriers:           %lu\n",
           (unsigned long)cluster->stats.total_barriers);
    if (cluster->stats.total_icc_cycles > 0) {
        double bw = (cluster->stats.total_icc_bytes / 1e9) /
                    (cluster->stats.total_icc_cycles * 1e-9);  /* rough */
        printf("  ICC bandwidth:      %.2f GB/s (effective)\n", bw);
    }

    /* Per-core summary */
    printf("  Per-core cycles:\n");
    for (uint32_t i = 0; i < cluster->num_cores; i++) {
        if (cluster->cores[i]) {
            printf("    Core %u: %lu cycles, %lu MACs\n", i,
                   (unsigned long)cluster->cores[i]->state.estimated_cycles,
                   (unsigned long)cluster->cores[i]->state.total_mma_flops / 2);
        }
    }
    printf("=============================\n");
}
