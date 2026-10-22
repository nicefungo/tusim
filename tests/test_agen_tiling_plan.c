/* Metadata-only planner oracle. Huge dimensions are never allocated/walked.
 * Independent 128-bit ceil/product avoids sharing production's checked-u32
 * implementation. Bounded cases additionally enumerate clipped tile origins. */
#include "tu_cmodel/memory/address_generator.h"
#include <stdio.h>
#include <string.h>

static unsigned cases, failures, empty_cases, overflow_cases, valid_cases;
static unsigned enumerated_cases, iterator_cases;

static void fail(const char *label, const tu_agen_tiling_t *t) {
    if (failures < 16)
        fprintf(stderr,"FAIL %s dims=%u/%u/%u tile=%u/%u/%u grid=%u/%u/%u count=%u\n",
            label,t->total_dims[0],t->total_dims[1],t->total_dims[2],
            t->tile_dims[0],t->tile_dims[1],t->tile_dims[2],
            t->tiles_per_dim[0],t->tiles_per_dim[1],t->tiles_per_dim[2],t->num_tiles);
    failures++;
}

static void check(tu_agen_tiling_t t, const char *label, int verbose) {
    tu_agen_tiling_t before=t;
    uint32_t widths[3], axes[3];
    __uint128_t product=1;
    for (unsigned i=0;i<3;i++) {
        widths[i]=t.tile_dims[i] ? t.tile_dims[i] : 1;
        axes[i]=(uint32_t)(((__uint128_t)t.total_dims[i]+widths[i]-1)/widths[i]);
        product*=axes[i];
    }
    uint32_t count=product>UINT32_MAX ? 0 : (uint32_t)product;
    tu_agen_compute_tiling(&t);
    cases++;
    if (!product) empty_cases++;
    else if (product>UINT32_MAX) overflow_cases++;
    else valid_cases++;
    if (t.num_tiles!=count || t.elem_size!=before.elem_size ||
        memcmp(t.total_dims,before.total_dims,sizeof(t.total_dims)) ||
        memcmp(t.tile_dims,widths,sizeof(widths)) ||
        memcmp(t.tiles_per_dim,axes,sizeof(axes))) fail(label,&t);
    tu_agen_tiling_t once=t;
    tu_agen_compute_tiling(&t);
    if (memcmp(&t,&once,sizeof(t))) fail("idempotence",&t);

    if (t.total_dims[0]<=7 && t.total_dims[1]<=7 && t.total_dims[2]<=7) {
        unsigned origins=0,volume=0;
        /* uint64 cursors make UINT32_MAX tile steps safe in this tiny view. */
        for (uint64_t z=0;z<t.total_dims[0];z+=widths[0])
            for (uint64_t y=0;y<t.total_dims[1];y+=widths[1])
                for (uint64_t x=0;x<t.total_dims[2];x+=widths[2]) {
                    uint64_t dz=t.total_dims[0]-z,dy=t.total_dims[1]-y,dx=t.total_dims[2]-x;
                    if (dz>widths[0]) dz=widths[0];
                    if (dy>widths[1]) dy=widths[1];
                    if (dx>widths[2]) dx=widths[2];
                    origins++; volume+=(unsigned)(dz*dy*dx);
                }
        enumerated_cases++;
        if (origins!=count || volume!=(uint64_t)t.total_dims[0]*t.total_dims[1]*t.total_dims[2])
            fail("clipped-origin-enumeration",&t);
        uint32_t cfg[]={t.total_dims[0],t.total_dims[1],t.total_dims[2],
                        widths[0],widths[1],widths[2],4};
        tu_agen_iterator_t it;
        if (tu_agen_iterator_init(&it,TU_AGEN_MODE_TILED_3D,0,cfg) ||
            memcmp(it.dims,axes,sizeof(axes)) || tu_agen_total(&it)!=volume)
            fail("bounded-iterator-grid",&t);
        iterator_cases++;
    }
    if (verbose)
        printf("%s dims=%u/%u/%u tile=%u/%u/%u grid=%u/%u/%u count=%u status=%s\n",
            label,t.total_dims[0],t.total_dims[1],t.total_dims[2],
            t.tile_dims[0],t.tile_dims[1],t.tile_dims[2],
            t.tiles_per_dim[0],t.tiles_per_dim[1],t.tiles_per_dim[2],t.num_tiles,
            !product ? "empty" : product>UINT32_MAX ? "unrepresentable" : "representable");
}

static uint32_t state=0x813b7465u;
static uint32_t random_u32(void) {
    state^=state<<13; state^=state>>17; state^=state<<5; return state;
}

int main(int argc, char **argv) {
    tu_agen_compute_tiling(NULL);
    const tu_agen_tiling_t probes[]={
        {.total_dims={1,1,UINT32_MAX},.tile_dims={1,1,2},.elem_size=4},
        {.total_dims={65537,65537,1},.tile_dims={1,1,1},.elem_size=4},
        {.total_dims={UINT32_MAX,UINT32_MAX,UINT32_MAX},.tile_dims={1,1,1}},
        {.total_dims={UINT32_MAX,UINT32_MAX,0},.tile_dims={2,2,UINT32_MAX}},
        {.total_dims={0,0,0},.tile_dims={UINT32_MAX,UINT32_MAX,UINT32_MAX}},
        {.total_dims={1,1,UINT32_MAX},.tile_dims={1,1,1}},
        {.total_dims={1,65536,65536},.tile_dims={1,1,1}},
        /* Tile count is not tensor-byte or iterator-count admission. */
        {.total_dims={UINT32_MAX,UINT32_MAX,UINT32_MAX},
         .tile_dims={UINT32_MAX,UINT32_MAX,UINT32_MAX},.elem_size=UINT32_MAX}
    };
    for (unsigned i=0;i<sizeof(probes)/sizeof(probes[0]);i++) check(probes[i],"boundary",1);
    if (argc>1 && !strcmp(argv[1],"probes")) goto report;
    const uint32_t choices[]={0,1,2,3,65535,65536,65537,0x80000000u,UINT32_MAX-1,UINT32_MAX};
    const unsigned n=sizeof(choices)/sizeof(choices[0]);
    for (unsigned a=0;a<n*n;a++)
        for (unsigned b=0;b<n*n;b++)
            for (unsigned c=0;c<n*n;c++) {
                tu_agen_tiling_t t={.num_tiles=0xabababab,
                    .tiles_per_dim={123,456,789},
                    .total_dims={choices[a/n],choices[b/n],choices[c/n]},
                    .tile_dims={choices[a%n],choices[b%n],choices[c%n]},
                    .elem_size=(a+b+c)%3==0 ? 0 : (a+b+c)%3==1 ? 4 : UINT32_MAX};
                check(t,"edge-grid",0);
            }
    for (unsigned i=0;i<30000;i++) {
        tu_agen_tiling_t t={0};
        for (unsigned d=0;d<3;d++) {
            t.total_dims[d]=random_u32(); t.tile_dims[d]=random_u32();
        }
        t.elem_size=random_u32(); check(t,"seeded",0);
    }
    const uint32_t tiles[][3]={{1,1,1},{1,3,7},{2,3,4},{3,5,7},{8,8,8},{0,0,0}};
    for (unsigned i=0;i<sizeof(tiles)/sizeof(tiles[0]);i++) {
        tu_agen_tiling_t t={.total_dims={3,5,7},.elem_size=4};
        memcpy(t.tile_dims,tiles[i],sizeof(t.tile_dims));
        check(t,"clipped-workload",1);
    }
report:
    printf("tiling-plan cases=%u representable=%u empty=%u overflow=%u enumerated=%u iterator=%u failures=%u\n",
        cases,valid_cases,empty_cases,overflow_cases,enumerated_cases,iterator_cases,failures);
    return failures ? 1 : 0;
}
