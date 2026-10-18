/* Independent wide-arithmetic admission oracle plus nested address enumeration.
 * No invalid address is dereferenced and huge accepted views are not traversed. */
#include "tu_cmodel/memory/address_generator.h"
#include <stdio.h>
#include <string.h>

static unsigned cases, rejected;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL case %u line %d: %s\n", cases, __LINE__, #c); return 1; \
} } while (0)

static int check(tu_agen_mode_t mode, uint32_t base, uint32_t d, uint32_t r,
                 uint32_t c, uint32_t ds, uint32_t rs, uint32_t size, int verbose) {
    uint32_t cfg[6] = {d,r,c,ds,rs,size};
    uint32_t linear[2] = {c,size}, transposed[3] = {r,c,size};
    tu_agen_2d_config_t cfg2 = {r,c,rs};
    const void *config = cfg;
    uint32_t width = size;
    if (mode == TU_AGEN_MODE_LINEAR) { d=r=1; ds=rs=0; config=linear; }
    if (mode == TU_AGEN_MODE_STRIDED_2D) { d=1; ds=0; width=4; config=&cfg2; }
    if (mode == TU_AGEN_MODE_TRANSPOSED) { d=1; ds=0; rs=c; config=transposed; }
    if ((mode == TU_AGEN_MODE_TRANSPOSED || mode == TU_AGEN_MODE_STRIDED_3D) && !width)
        width=4;
    /* Test-only 128-bit oracle deliberately differs from subtractive budget
     * validation. The API's entire product/sum fits this wider domain. */
    __uint128_t count = (__uint128_t)d*r*c;
    __uint128_t end = 0;
    if (count) end = base + (((__uint128_t)(d-1)*ds +
                              (__uint128_t)(r-1)*rs + c) * width);
    int valid = !count || (width && count <= UINT32_MAX && end <= UINT32_MAX);
    tu_agen_iterator_t it;
    memset(&it, 0xa5, sizeof(it));
    cases++;
    int rc = tu_agen_iterator_init(&it, mode, base, config);
    if ((rc == 0) != valid) {
        fprintf(stderr, "mode=%u base=%u D/R/C=%u/%u/%u strides=%u/%u size=%u rc=%d valid=%d\n",
                mode,base,d,r,c,ds,rs,width,rc,valid);
        return 1;
    }
    if (!valid) {
        rejected++;
        CHECK(rc == -1 && tu_agen_total(&it) == 0 && !tu_agen_has_next(&it));
        CHECK(tu_agen_next(&it) == UINT32_MAX && it.current_idx == 0);
        tu_agen_reset(&it);
        CHECK(!tu_agen_has_next(&it));
        return 0;
    }
    CHECK(tu_agen_total(&it) == (uint32_t)count);
    uint32_t n = (uint32_t)count;
    if (n > 256) {
        CHECK(tu_agen_next(&it) == base);
        it.current_idx = n-1; /* Sample last element; never loop over huge views. */
        uint32_t last = base + (uint64_t)(d-1)*ds*width +
                        (uint64_t)(r-1)*rs*width + (uint64_t)(c-1)*width;
        CHECK(tu_agen_next(&it) == last);
        CHECK(!tu_agen_has_next(&it) && tu_agen_next(&it) == UINT32_MAX);
        CHECK(it.current_idx == n);
        return 0;
    }
    uint32_t expected[256], actual[256], pos=0;
    if (mode == TU_AGEN_MODE_TRANSPOSED) {
        for (uint32_t x=0; x<c; x++)
            for (uint32_t y=0; y<r; y++) expected[pos++]=base+(y*c+x)*width;
    } else {
        for (uint32_t z=0; z<d; z++)
            for (uint32_t y=0; y<r; y++)
                for (uint32_t x=0; x<c; x++)
                    expected[pos++]=base+((uint64_t)z*ds+(uint64_t)y*rs+x)*width;
    }
    CHECK(pos == n);
    for (uint32_t i=0; i<n; i++) {
        CHECK(tu_agen_has_next(&it));
        CHECK(tu_agen_next(&it) == expected[i]);
    }
    CHECK(tu_agen_next(&it) == UINT32_MAX && it.current_idx == n);
    CHECK(tu_agen_next(&it) == UINT32_MAX && it.current_idx == n);
    tu_agen_reset(&it);
    CHECK(tu_agen_generate_all(&it, actual, 256) == n);
    CHECK(memcmp(expected, actual, n*sizeof(*actual)) == 0);
    CHECK(tu_agen_generate_all(&it, actual, n ? n-1 : 0) == (n ? n-1 : 0));
    if (n) CHECK(tu_agen_next(&it) == expected[n-1]);
    tu_agen_range_t ranges[256];
    uint32_t nr=tu_agen_generate_ranges(&it,ranges,256);
    pos=0;
    for (uint32_t i=0; i<nr; i++) {
        CHECK(ranges[i].elem_size == width && ranges[i].stride == width);
        for (uint32_t off=0; off<ranges[i].total_bytes; off+=width) {
            CHECK(pos<n && ranges[i].base_addr+off == expected[pos]);
            pos++;
        }
    }
    CHECK(pos == n);
    if (verbose) printf("mode=%u D/R/C=%u/%u/%u strides=%u/%u width=%u elements=%u logical_runs=%u PASS\n",
                        mode,d,r,c,ds,rs,width,n,nr);
    return 0;
}

int main(void) {
    const tu_agen_mode_t modes[] = {TU_AGEN_MODE_LINEAR,TU_AGEN_MODE_STRIDED_2D,
        TU_AGEN_MODE_STRIDED_3D,TU_AGEN_MODE_TRANSPOSED};
    /* Historical four safe RED admission probes. */
    CHECK(!check(modes[0],UINT32_MAX-7,1,1,2,0,0,4,0));
    CHECK(!check(modes[1],0,1,2,1,0,UINT32_MAX,4,0));
    CHECK(!check(modes[2],0,2,1,1,UINT32_MAX,1,4,0));
    CHECK(!check(modes[3],0,1,65536,65536,0,65536,4,0));
    const uint32_t bases[] = {0,13,UINT32_MAX-1,UINT32_MAX-7,UINT32_MAX-8,UINT32_MAX};
    const uint32_t sizes[] = {0,1,2,4,8,UINT32_MAX};
    const uint32_t strides[] = {0,1,3,8,UINT32_MAX};
    for (unsigned m=0; m<4; m++)
        for (uint32_t r=0; r<=4; r++)
            for (uint32_t c=0; c<=4; c++)
                for (unsigned b=0; b<6; b++)
                    for (unsigned e=0; e<6; e++)
                        for (unsigned s=0; s<5; s++)
                            CHECK(!check(modes[m],bases[b],m==2?2:1,r,c,
                                         strides[s],strides[s],sizes[e],0));
    const uint32_t large[][6] = {
        {65536,65536,1,0,0,1}, {1,65536,65536,0,0,1},
        {UINT32_MAX,1,1,0,0,4}, {UINT32_MAX,1,1,UINT32_MAX,0,UINT32_MAX},
        {1,65535,65537,0,65537,1}, {1,1,UINT32_MAX,0,0,1},
        {0,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX},
        {2,2,1,UINT32_MAX,UINT32_MAX,UINT32_MAX}
    };
    for (unsigned m=0; m<4; m++)
        for (unsigned l=0; l<sizeof(large)/sizeof(large[0]); l++)
            CHECK(!check(modes[m],0,large[l][0],large[l][1],large[l][2],
                         large[l][3],large[l][4],large[l][5],0));
    tu_agen_iterator_t it;
    CHECK(tu_agen_iterator_init(NULL,TU_AGEN_MODE_LINEAR,0,NULL) == -1);
    CHECK(tu_agen_iterator_init(&it,TU_AGEN_MODE_LINEAR,UINT32_MAX,NULL) == 0);
    CHECK(!tu_agen_has_next(&it) && it.elem_size == 1);
    for (unsigned m=1; m<4; m++)
        CHECK(tu_agen_iterator_init(&it,modes[m],0,NULL) == -1);
    CHECK(tu_agen_iterator_init(&it,(tu_agen_mode_t)99,0,NULL) == -1);
    CHECK(!tu_agen_has_next(&it));
    /* Layout alternatives: dense, padded, overlapping, broadcast and transpose.
     * Report logical runs only; no command, device-cycle or traffic inference. */
    CHECK(!check(TU_AGEN_MODE_STRIDED_2D,13,1,3,4,0,4,4,1));
    CHECK(!check(TU_AGEN_MODE_STRIDED_2D,13,1,3,4,0,8,4,1));
    CHECK(!check(TU_AGEN_MODE_STRIDED_2D,13,1,3,4,0,2,4,1));
    CHECK(!check(TU_AGEN_MODE_STRIDED_2D,13,1,3,4,0,0,4,1));
    CHECK(!check(TU_AGEN_MODE_TRANSPOSED,13,1,3,4,0,0,4,1));
    CHECK(!check(TU_AGEN_MODE_STRIDED_3D,13,2,3,4,32,8,4,1));
    printf("PASS: %u affine admission/sequence cases, %u rejected; NULL/unknown/default gates pass\n",
           cases,rejected);
    return 0;
}
