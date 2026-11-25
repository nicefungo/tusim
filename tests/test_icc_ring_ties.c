/* Parse-to-route sweep: each message consumes 16 directed-link service cycles. */
#include "tu_cluster.h"
#include "infra/config.h"
#include <stdio.h>
#include <string.h>

static const char *const policy[] = {"clockwise", "counterclockwise", "even_cw", "odd_cw"};
struct fixture {
    const char *name;
    unsigned n, src, dst, other_src, other_dst, hops;
    unsigned peak[4]; /* expected packets sharing the busiest link */
};
static const struct fixture cases[] = {
    {"even+cw", 8, 0,4, 0,1, 4, {2,1,2,1}},
    {"even+ccw",8, 0,4, 0,7, 4, {1,2,1,2}},
    {"odd+cw",  8, 1,5, 1,2, 4, {2,1,1,2}},
    {"odd+ccw", 8, 1,5, 1,0, 4, {1,2,2,1}},
    {"two ties",8, 0,4, 1,5, 4, {2,2,1,1}},
    {"no tie",  7, 0,3, 0,1, 3, {2,2,2,2}},
    {"wrap",    7, 0,4, 0,6, 3, {2,2,2,2}},
    {"two nodes",2,0,1, 1,0, 1, {1,1,1,1}},
};

int main(void) {
    unsigned failures = 0;
    tu_config_t base;
    char error[160];
    if (tu_config_load("config/tu_config.json", &base, error, sizeof error) || base.icc_ring_tie_mode != 0 ||
        tu_config_load_string("{\"tu\":{}}", &base, error, sizeof error) || base.icc_ring_tie_mode != 0 ||
        tu_runtime_config_default().icc_ring_tie_mode != 0) return 1;
    const char *invalid[] = {"3", "null", "\"adaptive\""};
    for (unsigned i=0; i<3; ++i) {
        char json[96];
        snprintf(json,sizeof json,"{\"tu\":{\"multicore\":{\"ring_tie\":%s}}}",invalid[i]);
        if (!tu_config_load_string(json,&base,error,sizeof error) || !strstr(error,"ring_tie")) return 1;
    }
    tu_runtime_config_t bad=tu_runtime_config_default();
    bad.icc_ring_tie_mode=4;
    if (tu_cluster_create(2,TU_TOPOLOGY_RING,0,&bad)) return 1;
    puts("case,policy,max_isolated_cycles,busiest_link_cycles,shared_floor_cycles");
    for (unsigned p=0; p<4; ++p) {
        char json[256];
        snprintf(json,sizeof json,"{\"tu\":{\"multicore\":{\"ring_tie\":\"%s\",\"switching\":\"cut_through\",\"contention\":\"shared_link_floor\",\"link_bytes_per_cycle\":16,\"router_latency_cycles\":2}}}",policy[p]);
        tu_config_t cfg;
        if (tu_config_load_string(json,&cfg,NULL,0)) return 1;
        tu_runtime_config_t rt=tu_config_to_runtime(&cfg);
        for (unsigned i=0; i<sizeof cases/sizeof cases[0]; ++i) {
            const struct fixture *f=&cases[i];
            tu_cluster_t *c=tu_cluster_create(f->n,TU_TOPOLOGY_RING,0,&rt);
            if (!c) return 1;
            if (cfg.icc_ring_tie_mode!=(int)p || rt.icc_ring_tie_mode!=(int)p || c->ring_tie_mode!=(int)p) ++failures;
            tu_icc_message_t m[]={{.src_core_id=f->src,.dst_core_id=f->dst,.size_bytes=256},
                                  {.src_core_id=f->other_src,.dst_core_id=f->other_dst,.size_bytes=256}};
            tu_icc_traffic_stats_t s;
            if (tu_cluster_estimate_traffic_cycles(c,m,2,&s)) return 1;
            unsigned iso=2*f->hops+16, peak=16*f->peak[p];
            tu_icc_message_t reversed[]={m[1],m[0]};
            tu_icc_traffic_stats_t replay;
            if (tu_cluster_estimate_traffic_cycles(c,reversed,2,&replay) ||
                replay.bottleneck_src!=s.bottleneck_src || replay.bottleneck_dst!=s.bottleneck_dst ||
                replay.estimated_cycles!=s.estimated_cycles || replay.bottleneck_link_cycles!=s.bottleneck_link_cycles) ++failures;
            unsigned char input[256], output[256];
            for (unsigned b=0; b<sizeof input; ++b) input[b]=(unsigned char)(b+f->src);
            tu_sram_write_bulk(tu_core_get_sram_o(c->cores[f->src]),0,input,sizeof input);
            if (tu_cluster_send(c,&m[0]) || c->stats.total_icc_bytes!=sizeof input || c->stats.total_icc_cycles!=iso) ++failures;
            tu_sram_read_bulk(tu_core_get_sram_o(c->cores[f->dst]),0,output,sizeof output);
            if (memcmp(input,output,sizeof input)) ++failures;
            if (s.isolated_cycles!=iso || s.bottleneck_link_cycles!=peak ||
                s.estimated_cycles!=(peak>iso ? peak : iso)) {
                fprintf(stderr,"FAIL %s %s expected peak %u got %llu\n",f->name,policy[p],peak,(unsigned long long)s.bottleneck_link_cycles);
                ++failures;
            }
            printf("%s,%s,%llu,%llu,%llu\n",f->name,policy[p],(unsigned long long)s.isolated_cycles,
                   (unsigned long long)s.bottleneck_link_cycles,(unsigned long long)s.estimated_cycles);
            tu_cluster_destroy(c);
        }
    }
    return failures!=0;
}
