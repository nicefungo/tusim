/* Finite-width service on one lane per direction, or one shared lane per edge. */
#include "tu_cluster.h"
#include "infra/config.h"
#include <stdio.h>

struct traffic { const char *name; unsigned a,b,c,d,p,q, peak[2],iso; };
/* One-hop routes on both 4-ring and 2x2 mesh. Packets round independently. */
static const struct traffic cases[] = {
    {"opposed",0,1,1,0,256,256,{16,32},18},
    {"aligned",0,1,0,1,256,256,{32,32},18},
    {"disjoint",0,1,2,3,256,256,{16,16},18},
    {"reverse-tail",1,0,0,1,33,1,{3,4},5},
    {"self-empty",1,1,0,1,256,0,{0,0},0},
};
int main(void) {
    const char *duplex[]={"full","half"};
    unsigned failures=0;
    puts("topology,duplex,traffic,isolated_cycles,peak_service_cycles,floor_cycles");
    for (int t=1;t<=2;t++) for (int h=0;h<2;h++) {
        char json[240]; tu_config_t cfg;
        snprintf(json,sizeof json,"{\"tu\":{\"multicore\":{\"link_duplex\":\"%s\",\"switching\":\"cut_through\",\"contention\":\"shared_link_floor\",\"link_bytes_per_cycle\":16,\"router_latency_cycles\":2}}}",duplex[h]);
        if (tu_config_load_string(json,&cfg,NULL,0)) return 1;
        tu_runtime_config_t rt=tu_config_to_runtime(&cfg);
        tu_cluster_t *cl=tu_cluster_create(4,t,2,&rt);
        if (!cl) return 1;
        if (cfg.icc_link_duplex!=h || rt.icc_link_duplex!=h || cl->link_duplex!=h) failures++;
        for (size_t i=0;i<sizeof cases/sizeof *cases;i++) {
            const struct traffic *f=cases+i;
            tu_icc_message_t msg[]={{.src_core_id=f->a,.dst_core_id=f->b,.size_bytes=f->p},
                                    {.src_core_id=f->c,.dst_core_id=f->d,.size_bytes=f->q}};
            tu_icc_traffic_stats_t s;
            if (tu_cluster_estimate_traffic_cycles(cl,msg,2,&s)) return 1;
            unsigned floor=f->peak[h]>f->iso ? f->peak[h] : f->iso;
            unsigned src=i==4 ? UINT32_MAX : (i==3 && !h ? 1u : 0u);
            unsigned dst=i==4 ? UINT32_MAX : 1u-src;
            if (s.bottleneck_link_cycles!=f->peak[h] || s.isolated_cycles!=f->iso || s.estimated_cycles!=floor || s.bottleneck_src!=src || s.bottleneck_dst!=dst) {
                fprintf(stderr,"mismatch: %s %s peak %llu expected %u\n",duplex[h],f->name,(unsigned long long)s.bottleneck_link_cycles,f->peak[h]); failures++;
            }
            printf("%s,%s,%s,%llu,%llu,%llu\n",t==1?"ring":"mesh",duplex[h],f->name,(unsigned long long)s.isolated_cycles,(unsigned long long)s.bottleneck_link_cycles,(unsigned long long)s.estimated_cycles);
        }
        /* Real send: duplex must not change the isolated cost or endpoint bytes. */
        uint32_t sent=0xcafed00d, received=0;
        tu_icc_message_t m={.src_core_id=0,.dst_core_id=1,.size_bytes=4};
        tu_sram_write_bulk(tu_core_get_sram_o(cl->cores[0]),0,&sent,4);
        if (tu_cluster_send(cl,&m)) failures++;
        tu_sram_read_bulk(tu_core_get_sram_o(cl->cores[1]),0,&received,4);
        if (received!=sent || cl->stats.total_icc_bytes!=4 || cl->stats.total_icc_cycles!=3) failures++;
        tu_cluster_destroy(cl);
    }
    tu_config_t cfg;
    if (tu_config_load("config/tu_config.json",&cfg,NULL,0) || cfg.icc_link_duplex!=0 ||
        tu_config_load_string("{\"tu\":{}}",&cfg,NULL,0) || cfg.icc_link_duplex!=0 ||
        tu_runtime_config_default().icc_link_duplex!=0) return 1;
    tu_runtime_config_t invalid=tu_runtime_config_default();
    invalid.icc_link_duplex=2;
    if (tu_cluster_create(1,TU_TOPOLOGY_RING,0,&invalid)) return 1;
    /* Two two-hop flows share just the middle edge in opposite directions:
     * 0->1->2 and 2->1->4 on a 2x3 XY mesh. */
    for (int h=0;h<2;h++) for (int sw=0;sw<3;sw++) for (int cm=0;cm<3;cm++) {
        tu_runtime_config_t r=tu_runtime_config_default();
        r.icc_link_duplex=h; r.icc_switching_mode=sw; r.icc_contention_mode=cm;
        r.icc_router_latency_cycles=2; r.icc_link_bytes_per_cycle=16;
        tu_cluster_t *cl=tu_cluster_create(6,TU_TOPOLOGY_MESH,2,&r);
        if (!cl) return 1;
        tu_icc_message_t msg[]={{.src_core_id=0,.dst_core_id=2,.size_bytes=256},
                                {.src_core_id=2,.dst_core_id=4,.size_bytes=256}};
        tu_icc_traffic_stats_t s;
        unsigned iso=sw==0 ? 4 : sw==1 ? 20 : 36;
        unsigned peak=sw==0 ? 0 : h ? 32 : 16;
        unsigned competing=cm==0 ? 0 : peak+(cm==1 ? 4 : 0);
        if (tu_cluster_estimate_traffic_cycles(cl,msg,2,&s) || s.isolated_cycles!=iso ||
            s.bottleneck_link_cycles!=peak || s.estimated_cycles!=(iso>competing?iso:competing) ||
            (sw && h && (s.bottleneck_src!=1 || s.bottleneck_dst!=2))) failures++;
        tu_cluster_destroy(cl);
    }
    if (!tu_config_load_string("{\"tu\":{\"multicore\":{\"link_duplex\":\"auto\"}}}",&cfg,NULL,0) ||
        !tu_config_load_string("{\"tu\":{\"multicore\":{\"link_duplex\":0}}}",&cfg,NULL,0)) failures++;
    return failures!=0;
}
