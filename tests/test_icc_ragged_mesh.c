/* Verify admitted paths against a coordinate-step oracle, not Manhattan distance alone. */
#include "tu_cluster.h"
#include "infra/config.h"
#include <stdio.h>
#include <string.h>

static const char *const modes[] = {"xy", "yx", "xy_else_yx", "yx_else_xy"};

/* Count the exact occupied directed links independently of the estimator. */
static int walk(unsigned n, unsigned cols, unsigned src, unsigned dst,
                int xy, unsigned loads[6][6]) {
    int x = src % cols, y = src / cols;
    int dx = dst % cols, dy = dst / cols;
    unsigned edges[6][2], count = 0;
    while (x != dx || y != dy) {
        unsigned prev = y * cols + x;
        if (x != dx && (xy || y == dy)) x += x < dx ? 1 : -1;
        else y += y < dy ? 1 : -1;
        unsigned next = y * cols + x;
        if (next >= n) return 0;
        edges[count][0] = prev; edges[count++][1] = next;
    }
    for (unsigned i = 0; i < count; i++) loads[edges[i][0]][edges[i][1]]++;
    return 1;
}

static int run(unsigned n, unsigned mode) {
    char json[192];
    snprintf(json,sizeof(json),"{\"tu\":{\"multicore\":{\"mesh_routing\":\"%s\",\"switching\":\"cut_through\",\"link_bytes_per_cycle\":16,\"router_latency_cycles\":5}}}",modes[mode]);
    tu_config_t cfg;
    if (tu_config_load_string(json,&cfg,NULL,0)) return 1;
    tu_runtime_config_t rt = tu_config_to_runtime(&cfg);
    tu_cluster_t *c = tu_cluster_create(n,TU_TOPOLOGY_MESH,2,&rt);
    if (!c) return 1;
    unsigned ok = 0, bad = 0, failures = c->mesh_routing_mode != (int)mode;
    for (unsigned s=0; s<n; s++) for (unsigned d=0; d<n; d++) {
        unsigned loads[6][6] = {{0}};
        int xy = !(mode % 2);
        int valid = walk(n,3,s,d,xy,loads);
        if (!valid && mode >= 2) valid = walk(n,3,s,d,!xy,loads);
        unsigned hops = 0, first_s = UINT32_MAX, first_d = UINT32_MAX;
        for (unsigned a=0; a<n; a++) for (unsigned b=0; b<n; b++) if (loads[a][b]) {
            if (!hops) { first_s=a; first_d=b; }
            hops++;
        }
        tu_icc_message_t m = {.src_core_id=s,.dst_core_id=d,.src_offset=0,.dst_offset=16,.size_bytes=4};
        tu_icc_traffic_stats_t snap;
        int planned = tu_cluster_estimate_traffic_cycles(c,&m,1,&snap)==0;
        uint64_t t = tu_cluster_estimate_transfer_cycles(c,s,d,4);
        uint32_t input=0x10203040u+s, guard=0xabcdef01u, output=0;
        tu_sram_region_t *source=tu_core_get_sram_o(c->cores[s]), *dest=tu_core_get_sram_o(c->cores[d]);
        tu_sram_write_bulk(source,0,&input,4); tu_sram_write_bulk(dest,16,&guard,4);
        tu_cluster_stats_t before=c->stats;
        uint64_t core_before=c->cores[d]->state.estimated_cycles;
        int sent=tu_cluster_send(c,&m)==0;
        tu_sram_read_bulk(dest,16,&output,4);
        uint64_t cycles=hops ? hops*5+1 : 0;
        if (planned!=valid || sent!=valid || t!=(valid ? cycles : UINT64_MAX) ||
            output!=(valid ? input : guard) || c->stats.total_icc_messages-before.total_icc_messages!=(unsigned)valid ||
            c->stats.total_icc_bytes-before.total_icc_bytes!=(valid ? 4u : 0u) ||
            c->stats.total_icc_cycles-before.total_icc_cycles!=(valid ? cycles : 0) ||
            c->cores[d]->state.estimated_cycles-core_before!=(valid ? cycles : 0) ||
            (planned && (snap.isolated_cycles!=cycles || snap.bottleneck_link_cycles!=(hops!=0) ||
                         snap.bottleneck_src!=first_s || snap.bottleneck_dst!=first_d))) {
            fprintf(stderr,"FAIL n=%u %s %u->%u expected=%d batch=%d send=%d\n",n,modes[mode],s,d,valid,planned,sent);
            failures++;
        }
        if (valid) ok++; else bad++;
        /* A second packet on a known directed edge doubles its load iff
         * the first packet traverses it. Check the whole route, not just a
         * lexicographic tie winner. Never probe nonexistent neighbors. */
        if (valid) for (unsigned a=0; a<n; a++) for (unsigned b=0; b<n; b++) {
            if (tu_cluster_hop_distance(c,a,b)!=1) continue;
            tu_icc_message_t pair[]={m,{.src_core_id=a,.dst_core_id=b,.size_bytes=4}};
            if (tu_cluster_estimate_traffic_cycles(c,pair,2,&snap) ||
                snap.bottleneck_link_cycles!=1+loads[a][b]) failures++;
        }
    }
    printf("%u,%s,%u,%u\n",n,modes[mode],ok,bad);
    tu_cluster_destroy(c);
    return failures!=0;
}

int main(void) {
    unsigned failures=0;
    puts("cores,routing,admitted_pairs,rejected_pairs");
    for (unsigned n=5; n<=6; n++) for (unsigned mode=0; mode<4; mode++) failures+=run(n,mode);
    tu_config_t cfg;
    if (!tu_config_load_string("{\"tu\":{\"multicore\":{\"mesh_routing\":3}}}",&cfg,NULL,0) ||
        !tu_config_load_string("{\"tu\":{\"multicore\":{\"mesh_routing\":\"unknown\"}}}",&cfg,NULL,0)) failures++;
    tu_runtime_config_t rt=tu_runtime_config_default();
    if (rt.icc_mesh_routing_mode!=0) failures++;
    /* A tall, lightly populated mesh still has a single column. */
    tu_cluster_t *tall=tu_cluster_create(2,TU_TOPOLOGY_MESH,3,&rt);
    if (!tall || tall->mesh_cols!=1 || tu_cluster_hop_distance(tall,0,1)!=1) failures++;
    tu_cluster_destroy(tall);
    rt.icc_mesh_routing_mode=4;
    tu_cluster_t *invalid=tu_cluster_create(1,TU_TOPOLOGY_MESH,1,&rt);
    if (invalid) { failures++; tu_cluster_destroy(invalid); }
    return failures!=0;
}
