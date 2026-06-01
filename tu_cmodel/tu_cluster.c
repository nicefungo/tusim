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
    cluster->hop_latency = 5;  /* Default: 5 cycles per hop */

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

    /* Compute hop distance for cycle accounting */
    uint32_t hops = tu_cluster_hop_distance(cluster, msg->src_core_id, msg->dst_core_id);
    if (hops == UINT32_MAX) return -1;

    /* Simulated latency */
    uint64_t latency = (uint64_t)hops * cluster->hop_latency;

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
