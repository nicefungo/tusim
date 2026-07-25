/*
 * TU Cluster — Multi-Core SPMD Orchestrator (Gap A5)
 * ====================================================
 *
 * Manages an array of tu_core_t instances with configurable
 * inter-core communication topology and SPMD execution model.
 *
 * Supported topologies:
 *   - RING: 1D ring, each core communicates with neighbors
 *   - MESH: 2D mesh, cores communicate with N/S/E/W neighbors
 *   - NONE: isolated cores (no inter-core communication)
 *
 * Inter-core communication:
 *   - Message-passing API with source/destination addressing
 *   - Buffer-based data transfer between core local SRAMs
 *   - Barrier synchronization across all cores
 *   - Broadcast and reduction primitives
 *
 * Gap: A5 — Multi-instance / multi-core (P1)
 * Dependencies: tu_core.h, tu_config.h
 */

#ifndef TU_CLUSTER_H
#define TU_CLUSTER_H

#include "tu_core.h"
#include "tu_config.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Topology types ---- */
typedef enum {
    TU_TOPOLOGY_NONE = 0,    /* Independent cores */
    TU_TOPOLOGY_RING = 1,    /* 1D ring: core[i] ↔ core[(i+1)%N] */
    TU_TOPOLOGY_MESH = 2,    /* 2D mesh: cores arranged in rows×cols grid */
} tu_topology_t;

/* ---- Inter-core message descriptor ---- */
typedef struct {
    uint32_t    src_core_id;        /* Source core */
    uint32_t    dst_core_id;        /* Destination core */
    uint32_t    src_offset;         /* Byte offset in source core's O-SRAM */
    uint32_t    dst_offset;         /* Byte offset in destination core's O-SRAM */
    uint32_t    size_bytes;         /* Transfer size */
    uint32_t    tag;                /* User-defined message tag */
    bool        blocking;           /* true = wait for completion */
    uint64_t    latency_cycles;     /* Simulated interconnect latency */
} tu_icc_message_t;

/* Simultaneous traffic-matrix estimate. shared_link is a deterministic-routing
 * lower bound: it captures aggregate directed-link serialization but not queue
 * ordering, finite buffers, virtual channels, or head-of-line blocking. */
typedef struct {
    uint64_t isolated_cycles;
    uint64_t estimated_cycles;
    uint64_t bottleneck_link_cycles;
    uint32_t bottleneck_src;
    uint32_t bottleneck_dst;
} tu_icc_traffic_stats_t;

/* ---- Cluster statistics ---- */
typedef struct {
    uint64_t    total_icc_messages;
    uint64_t    total_icc_bytes;
    uint64_t    total_icc_cycles;
    uint64_t    total_barriers;
    double      icc_bandwidth_gbps;
} tu_cluster_stats_t;

/* ---- Cluster state ---- */
typedef struct tu_cluster_t {
    /* Identity */
    uint32_t        num_cores;
    uint32_t        cluster_id;

    /* Topology */
    tu_topology_t   topology;
    uint32_t        mesh_rows;          /* For MESH topology */
    uint32_t        mesh_cols;          /* For MESH topology */

    /* Core array */
    tu_core_t     **cores;              /* Array of core pointers */

    /* Interconnect latency model (cycles per hop) */
    uint32_t        hop_latency;
    int             switching_mode;
    int             contention_mode;
    int             mesh_routing_mode;
    uint32_t        link_bytes_per_cycle;

    /* Statistics */
    tu_cluster_stats_t stats;

    /* Lifecycle */
    bool            initialized;

    /* Barrier counter */
    uint32_t        barrier_counter;
} tu_cluster_t;

/* ---- Lifecycle ---- */

/*
 * Create a cluster with N cores.
 *
 * num_cores:     number of TU cores in the cluster
 * topology:      interconnect topology (NONE, RING, MESH)
 * mesh_rows:     for MESH topology, number of rows (ignored for RING/NONE)
 * base_config:   runtime configuration applied to all cores
 *
 * Returns NULL on failure (OOM, invalid parameters).
 */
tu_cluster_t *tu_cluster_create(uint32_t num_cores,
                                 tu_topology_t topology,
                                 uint32_t mesh_rows,
                                 const tu_runtime_config_t *base_config);

/*
 * Destroy the cluster and all its cores.
 */
void tu_cluster_destroy(tu_cluster_t *cluster);

/*
 * Get a specific core by ID.
 * Returns NULL if core_id is out of range.
 */
tu_core_t *tu_cluster_get_core(tu_cluster_t *cluster, uint32_t core_id);

/* ---- Inter-Core Communication ---- */

/*
 * Send data from one core's O-SRAM to another core's O-SRAM.
 *
 * Models interconnect latency based on topology distance.
 * Returns 0 on success, -1 on invalid parameters.
 */
int tu_cluster_send(tu_cluster_t *cluster,
                     const tu_icc_message_t *msg);

/*
 * Broadcast data from one core to all other cores.
 *
 * src_core_id:   source core
 * src_offset:    byte offset in source core's O-SRAM
 * dst_offset:    byte offset in destination cores' O-SRAM
 * size_bytes:    transfer size
 *
 * Returns 0 on success, -1 on failure.
 */
int tu_cluster_broadcast(tu_cluster_t *cluster,
                          uint32_t src_core_id,
                          uint32_t src_offset,
                          uint32_t dst_offset,
                          uint32_t size_bytes);

/*
 * All-reduce: reduce data across all cores, result in each core.
 *
 * Currently supports element-wise FP32 sum reduction.
 * Data is read from src_offset in each core's O-SRAM,
 * reduced, and written back to dst_offset in each core's O-SRAM.
 *
 * num_elements:  number of FP32 elements to reduce
 */
int tu_cluster_allreduce_sum_f32(tu_cluster_t *cluster,
                                  uint32_t src_offset,
                                  uint32_t dst_offset,
                                  uint32_t num_elements);

/* ---- Synchronization ---- */

/*
 * Barrier: all cores must reach this point before any proceeds.
 * Returns after all cores have synchronized.
 * Returns 0 on success, -1 on failure.
 */
int tu_cluster_barrier(tu_cluster_t *cluster);

/* ---- SPMD Execution ---- */

/*
 * Execute the same ASM program on all cores concurrently.
 * Each core receives its own set of host buffers.
 *
 * programs:      array of ASM programs (one per core)
 * buffers:       array of buffer arrays (one per core)
 * n_buffers_per: array of buffer counts (one per core)
 *
 * Returns 0 if all cores succeed, non-zero if any core fails.
 */
int tu_cluster_spmd_execute(tu_cluster_t *cluster,
                             const char **programs,
                             const tu_host_buffer_t **buffers,
                             const int *n_buffers_per);

/* ---- Topology Helpers ---- */

/*
 * Compute the minimum hop distance between two cores.
 * Returns UINT32_MAX if cores cannot communicate.
 */
uint32_t tu_cluster_hop_distance(const tu_cluster_t *cluster,
                                  uint32_t src, uint32_t dst);

/* Estimate one point-to-point transfer. Legacy mode preserves the original
 * hop-only model; cut-through serializes once; store-forward serializes at
 * every hop. Returns UINT64_MAX for invalid/unreachable routes. */
uint64_t tu_cluster_estimate_transfer_cycles(const tu_cluster_t *cluster,
                                              uint32_t src, uint32_t dst,
                                              uint32_t size_bytes);

/* Estimate messages injected simultaneously. RING uses shortest-path routing
 * (clockwise on ties); MESH uses configured deterministic XY or YX routing. */
int tu_cluster_estimate_traffic_cycles(const tu_cluster_t *cluster,
                                       const tu_icc_message_t *messages,
                                       uint32_t message_count,
                                       tu_icc_traffic_stats_t *stats);

/*
 * Get neighbor core IDs for a given core.
 * For RING: prev and next in the ring.
 * For MESH: north, south, east, west (UINT32_MAX if edge).
 * Neighbors array must have at least 4 entries.
 */
void tu_cluster_neighbors(const tu_cluster_t *cluster,
                           uint32_t core_id,
                           uint32_t *neighbors,
                           uint32_t *num_neighbors);

/* ---- Statistics ---- */

void tu_cluster_print_stats(const tu_cluster_t *cluster);

#ifdef __cplusplus
}
#endif

#endif /* TU_CLUSTER_H */
