/* Snapshot scores, not a router simulation. Each row uses the public config path. */
#include "tu_cluster.h"
#include "infra/config.h"
#include <inttypes.h>
#include <stdio.h>

static const tu_icc_message_t flows[] = {
    {.src_core_id=0, .dst_core_id=1, .size_bytes=1024},
    {.src_core_id=0, .dst_core_id=1, .size_bytes=1024},
    {.src_core_id=4, .dst_core_id=7, .size_bytes=16},
    {.src_core_id=4, .dst_core_id=7, .size_bytes=0}
};

static int check(tu_cluster_t *c, const char *sw, const char *mode, const char *shape,
                 const tu_icc_message_t *m, unsigned n, unsigned iso, unsigned link, unsigned score) {
    tu_icc_traffic_stats_t s;
    if (tu_cluster_estimate_traffic_cycles(c,m,n,&s)) return 1;
    printf("%s,%s,%s,%"PRIu64",%"PRIu64",%"PRIu64"\n",sw,mode,shape,s.isolated_cycles,s.bottleneck_link_cycles,s.estimated_cycles);
    if (s.isolated_cycles!=iso || s.bottleneck_link_cycles!=link || s.estimated_cycles!=score ||
        s.bottleneck_src!=(link ? 0u : UINT32_MAX) || s.bottleneck_dst!=(link ? 1u : UINT32_MAX)) {
        fprintf(stderr,"FAIL %s/%s/%s: expected %u,%u,%u\n",sw,mode,shape,iso,link,score);
        return 1;
    }
    return 0;
}

int main(void) {
    const char *modes[]={"ideal_parallel","shared_link","shared_link_floor"};
    const char *switches[]={"legacy_hop_only","cut_through","store_and_forward"};
    unsigned failures=0;
    puts("switching,contention,shape,isolated,busiest_link,estimate");
    for (unsigned sw=0;sw<3;sw++) for (unsigned i=0;i<3;i++) {
        char json[256]; tu_config_t cfg; tu_config_default(&cfg);
        snprintf(json,sizeof(json),"{\"tu\":{\"multicore\":{\"switching\":\"%s\",\"contention\":\"%s\",\"router_latency_cycles\":5,\"link_bytes_per_cycle\":16}}}",switches[sw],modes[i]);
        if (tu_config_load_string(json,&cfg,NULL,0)) { fprintf(stderr,"FAIL parsing %s\n",modes[i]); failures++; continue; }
        tu_runtime_config_t rt=tu_config_to_runtime(&cfg);
        tu_cluster_t *c=tu_cluster_create(8,TU_TOPOLOGY_MESH,2,&rt);
        if (!c) return 1;
        if (c->contention_mode!=(int)i || c->switching_mode!=(int)sw) failures++;
        const unsigned iso=sw ? 69:15, link=sw ? 128:0;
        const unsigned score=sw ? (i==0 ? 69 : i==1 ? 143 : 128) : 15;
        failures+=check(c,switches[sw],modes[i],"disjoint-maxima",flows,3,iso,link,score);
        failures+=check(c,switches[sw],modes[i],"zero-only",flows+3,1,0,0,0);
        failures+=check(c,switches[sw],modes[i],"empty",NULL,0,0,0,0);
        tu_icc_message_t mixed[]={flows[0],flows[3]};
        failures+=check(c,switches[sw],modes[i],"mixed-zero",mixed,2,sw?69:5,sw?64:0,sw?69:5);
        mixed[1].dst_core_id=8; tu_icc_traffic_stats_t s;
        if (!tu_cluster_estimate_traffic_cycles(c,mixed,2,&s)) failures++;
        tu_cluster_destroy(c);
    }
    tu_config_t cfg; tu_config_default(&cfg);
    if (!tu_config_load_string("{\"tu\":{\"multicore\":{\"contention\":2}}}",&cfg,NULL,0)) failures++;
    tu_runtime_config_t rt=tu_runtime_config_default();
    if (rt.icc_contention_mode!=TU_ICC_CONTENTION_IDEAL_PARALLEL) failures++;
    rt.icc_contention_mode=3;
    tu_cluster_t *bad=tu_cluster_create(1,TU_TOPOLOGY_RING,0,&rt);
    if (bad) { failures++; tu_cluster_destroy(bad); }
    fprintf(stderr,"ICC traffic floor: %u failures\n",failures);
    return failures!=0;
}
