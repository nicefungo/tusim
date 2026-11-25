/* A rejected broadcast is observable through both SRAM and accounting. */
#include "tu_cluster.h"
#include "infra/config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run(int atomic, int fallback) {
    const char *mode = atomic ? "preflight" : "sequential";
    const char *route = fallback ? "xy_else_yx" : "xy";
    char json[256];
    snprintf(json,sizeof(json),"{\"tu\":{\"multicore\":{\"broadcast_admission\":\"%s\",\"mesh_routing\":\"%s\",\"switching\":\"cut_through\",\"link_bytes_per_cycle\":16,\"router_latency_cycles\":5}}}",mode,route);
    tu_config_t cfg;
    if (tu_config_load_string(json,&cfg,NULL,0)) return 1;
    tu_runtime_config_t rt=tu_config_to_runtime(&cfg);
    tu_cluster_t *c=tu_cluster_create(5,TU_TOPOLOGY_MESH,2,&rt);
    if (!c) return 1;
    int failed=cfg.icc_broadcast_admission!=atomic || rt.icc_broadcast_admission!=atomic || c->broadcast_admission!=atomic;
    const unsigned src=3, srcword=0xab1212ef, marker=0xcc55cc55;
    tu_sram_region_t *regions[5];
    uint64_t before[5];
    for (unsigned i=0;i<5;i++) {
        before[i]=c->cores[i]->state.estimated_cycles;
        regions[i]=tu_core_get_sram_o(c->cores[i]);
        tu_sram_write_bulk(regions[i],16,&marker,4);
    }
    tu_sram_write_bulk(regions[src],0,&srcword,4);
    unsigned cap=regions[4]->total_size;
    unsigned char *image=calloc(5,cap);
    if (!image) { tu_cluster_destroy(c); return 1; }
    uint64_t reads[5],writes[5];
    for (unsigned i=0;i<5;i++) {
        if (regions[i]->total_size!=cap) { free(image); tu_cluster_destroy(c); return 1; }
        memcpy(image+(size_t)i*cap,tu_sram_raw_ptr(regions[i]),cap);
        reads[i]=regions[i]->banks.reads; writes[i]=regions[i]->banks.writes;
    }
    int rc=tu_cluster_broadcast(c,src,0,16,4);
    const int reject=!fallback;
    const unsigned want=fallback ? 23 : atomic ? 0 : 3;
    const unsigned hops[]={1,2,3,0,1};
    unsigned delivered=0,count=0,cycles=0;
    failed |= rc!=(reject ? -1 : 0);
    for (unsigned i=0;i<5;i++) {
        unsigned value=0;
        memcpy(&value,(unsigned char *)tu_sram_raw_ptr(regions[i])+16,4);
        if (value==srcword) delivered |= 1u<<i;
        unsigned changed=(want>>i)&1u;
        if (changed) memcpy(image+(size_t)i*cap+16,&srcword,4);
        failed |= memcmp(image+(size_t)i*cap,tu_sram_raw_ptr(regions[i]),cap)!=0;
        failed |= regions[i]->banks.writes-writes[i]!=changed;
        unsigned cost=changed ? hops[i]*5+1 : 0;
        count+=changed; cycles+=cost;
        failed |= value!=(changed ? srcword : marker) || c->cores[i]->state.estimated_cycles-before[i]!=cost;
    }
    failed |= c->stats.total_icc_messages!=count || c->stats.total_icc_bytes!=4*count || c->stats.total_icc_cycles!=cycles;
    for (unsigned i=0;i<5;i++) failed |= regions[i]->banks.reads-reads[i]!=(i==src ? count : 0);
    free(image);
    printf("%s,%s,%d,0x%x,%llu,%llu,%llu\n",mode,route,rc,delivered,(unsigned long long)c->stats.total_icc_messages,(unsigned long long)c->stats.total_icc_bytes,(unsigned long long)c->stats.total_icc_cycles);
    if (failed) fprintf(stderr,"FAIL %s/%s: expected mask 0x%x, got 0x%x\n",mode,route,want,delivered);
    tu_cluster_destroy(c);
    return failed;
}
int main(void) {
    int failures=0;
    tu_config_t cfg;
    if (tu_config_load("config/tu_config.json",&cfg,NULL,0) || cfg.icc_broadcast_admission ||
        tu_config_load_string("{\"tu\":{}}",&cfg,NULL,0) || cfg.icc_broadcast_admission ||
        !tu_config_load_string("{\"tu\":{\"multicore\":{\"broadcast_admission\":\"auto\"}}}",&cfg,NULL,0) ||
        !tu_config_load_string("{\"tu\":{\"multicore\":{\"broadcast_admission\":1}}}",&cfg,NULL,0)) return 1;
    tu_runtime_config_t rt=tu_runtime_config_default();
    if (rt.icc_broadcast_admission!=0) return 1;
    rt.icc_broadcast_admission=2;
    if (tu_cluster_create(1,TU_TOPOLOGY_NONE,0,&rt)) return 1;
    for (int a=0;a<2;a++) {
        rt.icc_broadcast_admission=a;
        tu_cluster_t *c=tu_cluster_create(1,TU_TOPOLOGY_NONE,0,&rt);
        if (!c) return 1;
        failures += tu_cluster_broadcast(c,0,0,0,4)!=0 || tu_cluster_broadcast(c,0,0,0,0)!=0 ||
                    tu_cluster_broadcast(c,1,0,0,0)!=-1 || c->stats.total_icc_messages!=0;
        tu_cluster_destroy(c);
    }
    puts("admission,routing,status,changed_mask,messages,bytes,charged_cycles");
    for (int a=0;a<2;a++) for (int r=0;r<2;r++) failures+=run(a,r);
    return failures!=0;
}
