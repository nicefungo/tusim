/* Independent nested-loop oracle for compact, tile-major row-major storage. */
#include "tu_cmodel/memory/address_generator.h"
#include <stdio.h>
#include <string.h>

#define CAP 512
static unsigned cases, failures;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); return 1; \
} } while (0)

static int check_case(uint32_t d, uint32_t r, uint32_t c,
                      uint32_t td, uint32_t tr, uint32_t tc,
                      uint32_t size, int is_2d, int verbose) {
    const uint32_t base = 0x123;
    uint32_t width = is_2d ? 4 : (size ? size : 4);
    uint32_t expected[CAP], actual[CAP];
    unsigned char seen[CAP] = {0};
    uint32_t n = 0;
    /* Six nested loops intentionally differ from the implementation's
     * ordinal decomposition. Only in-bounds elements belong to a tile. */
    for (uint32_t db = 0; db < d; db += td)
        for (uint32_t rb = 0; rb < r; rb += tr)
            for (uint32_t cb = 0; cb < c; cb += tc)
                for (uint32_t dz = 0; dz < td && db + dz < d; dz++)
                    for (uint32_t ry = 0; ry < tr && rb + ry < r; ry++)
                        for (uint32_t cx = 0; cx < tc && cb + cx < c; cx++) {
                            CHECK(n < CAP);
                            expected[n++] = base +
                                (((db + dz) * r + rb + ry) * c + cb + cx) * width;
                        }
    tu_agen_iterator_t it;
    uint32_t cfg3[] = {d, r, c, td, tr, tc, size};
    tu_agen_tiled2d_config_t cfg2 = {r, c, tr, tc,
        r / tr + (r % tr != 0), c / tc + (c % tc != 0)};
    CHECK(tu_agen_iterator_init(&it,
          is_2d ? TU_AGEN_MODE_TILED_2D : TU_AGEN_MODE_TILED_3D,
          base, is_2d ? (const void *)&cfg2 : (const void *)cfg3) == 0);
    CHECK(tu_agen_total(&it) == n);
    uint32_t ranges_count = 0, prev = 0;
    for (uint32_t i = 0; i < n; i++) {
        CHECK(tu_agen_has_next(&it));
        uint32_t addr = tu_agen_next(&it);
        if (addr != expected[i]) {
            fprintf(stderr, "shape=%u,%u,%u tile=%u,%u,%u size=%u rank=%d i=%u expected=%u got=%u\n",
                    d,r,c,td,tr,tc,width,is_2d?2:3,i,expected[i],addr);
            return 1;
        }
        CHECK(addr >= base && addr < base + n * width);
        CHECK((addr - base) % width == 0);
        uint32_t index = (addr - base) / width;
        CHECK(!seen[index]);
        seen[index] = 1;
        if (i == 0 || addr != prev + width) ranges_count++;
        prev = addr;
    }
    for (uint32_t i = 0; i < n; i++) CHECK(seen[i]);
    CHECK(!tu_agen_has_next(&it));
    CHECK(tu_agen_next(&it) == UINT32_MAX);
    CHECK(tu_agen_next(&it) == UINT32_MAX);
    CHECK(it.current_idx == n);
    tu_agen_reset(&it);
    for (uint32_t i = 0; i < n; i++) CHECK(tu_agen_next(&it) == expected[i]);
    memset(actual, 0, sizeof(actual));
    CHECK(tu_agen_generate_all(&it, actual, CAP) == n);
    CHECK(memcmp(actual, expected, n * sizeof(uint32_t)) == 0);
    CHECK(tu_agen_generate_all(&it, actual, n ? n - 1 : 0) == (n ? n - 1 : 0));
    if (n) CHECK(tu_agen_next(&it) == expected[n - 1]);
    tu_agen_range_t ranges[CAP];
    CHECK(tu_agen_generate_ranges(&it, ranges, CAP) == ranges_count);
    uint32_t pos = 0;
    for (uint32_t k = 0; k < ranges_count; k++) {
        CHECK(ranges[k].elem_size == width);
        for (uint32_t off = 0; off < ranges[k].total_bytes; off += width) {
            CHECK(pos < n);
            CHECK(ranges[k].base_addr + off == expected[pos++]);
        }
    }
    CHECK(pos == n);
    if (verbose)
        printf("%d %u,%u,%u %u,%u,%u %u %u %u PASS\n",
               is_2d?2:3,d,r,c,td,tr,tc,width,n,ranges_count);
    cases++;
    return 0;
}

static int invalid_and_boundary(void) {
    tu_agen_iterator_t it;
    uint32_t cfg[] = {2,3,4,1,2,2,4};
    for (unsigned axis = 3; axis < 6; axis++) {
        uint32_t old = cfg[axis]; cfg[axis] = 0;
        CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, 0, cfg) == -1);
        CHECK(!tu_agen_has_next(&it)); cfg[axis] = old;
    }
    tu_agen_tiled2d_config_t cfg2 = {3,5,0,2,0,0};
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_2D, 0, &cfg2) == -1);
    cfg2.tile_rows=2; cfg2.tile_cols=0;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_2D, 0, &cfg2) == -1);
    cfg2.tile_cols=2; cfg2.num_tile_rows=1;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_2D, 0, &cfg2) == -1);
    cfg2.num_tile_rows=0; cfg2.num_tile_cols=0;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_2D, 0, &cfg2) == 0);
    CHECK(tu_agen_total(&it) == 15);
    cfg2.num_tile_cols=1;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_2D, 0, &cfg2) == -1);
    CHECK(!tu_agen_has_next(&it));
    cfg2.num_tile_cols=0; cfg2.total_rows=65536; cfg2.total_cols=65536;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_2D, 0, &cfg2) == -1);
    cfg2.total_rows=1; cfg2.total_cols=2;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_2D, UINT32_MAX-7, &cfg2) == -1);
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_2D, UINT32_MAX-8, &cfg2) == 0);
    CHECK(tu_agen_next(&it) == UINT32_MAX-8);
    CHECK(tu_agen_next(&it) == UINT32_MAX-4);
    cfg[0]=65536; cfg[1]=65536;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, 0, cfg) == -1);
    cfg[0]=1; cfg[1]=1; cfg[2]=2;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, UINT32_MAX-7, cfg) == -1);
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, UINT32_MAX-8, cfg) == 0);
    CHECK(tu_agen_next(&it) == UINT32_MAX-8);
    CHECK(tu_agen_next(&it) == UINT32_MAX-4);
    CHECK(tu_agen_next(&it) == UINT32_MAX);
    /* Isolate the final count multiply, pure byte-span overflow, and a
     * nonzero element size at the uint32 boundary. No memory is accessed. */
    cfg[0]=65536; cfg[1]=1; cfg[2]=65536;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, 0, cfg) == -1);
    cfg[0]=1; cfg[2]=UINT32_C(0x40000000);
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, 0, cfg) == -1);
    cfg[2]=2; cfg[6]=UINT32_MAX;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, 0, cfg) == -1);
    cfg[2]=1;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, 0, cfg) == 0);
    CHECK(tu_agen_next(&it) == 0);
    CHECK(tu_agen_next(&it) == UINT32_MAX);
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, 1, cfg) == -1);
    cfg[2]=2; cfg[6]=4;
    /* Oversized tiles are valid; their volume need not fit uint32_t. */
    cfg[3]=cfg[4]=cfg[5]=UINT32_MAX;
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TILED_3D, 0, cfg) == 0);
    CHECK(tu_agen_next(&it) == 0);
    CHECK(tu_agen_next(&it) == 4);
    return 0;
}

/* Safe capacity probe: never call the currently unsafe descriptor-chain
 * consumer with this truncated buffer. It iterates the returned required
 * count even though its private array stores only 128 ranges. */
static int range_capacity_probe(void) {
    tu_agen_iterator_t it;
    uint32_t cfg[] = {16,16,4};
    struct {
        tu_agen_range_t ranges[128];
        uint32_t guard;
    } output = {.guard = 0x12345678};
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TRANSPOSED, 0, cfg) == 0);
    uint32_t required = tu_agen_generate_ranges(&it, output.ranges, 128);
    CHECK(required == 256);
    CHECK(output.guard == 0x12345678);
    for (uint32_t i=0; i<128; i++) {
        CHECK(output.ranges[i].base_addr == ((i % 16) * 16 + i / 16) * 4);
        CHECK(output.ranges[i].total_bytes == 4);
    }
    printf("RANGE_CAPACITY: required=%u stored=128 guard=unchanged; unsafe chain consumer not invoked\n", required);
    return 0;
}

int main(void) {
    printf("rank shape(D,R,C) tile(D,R,C) element_bytes elements contiguous_ranges result\n");
    failures += check_case(1,3,5,1,2,3,4,1,1);
    failures += check_case(2,3,8,1,1,2,4,0,1);
    if (failures) return 1; /* safe RED: do not reach invalid zero-tile probes */
    const uint32_t shapes[][3] = {
        {1,6,6},{1,3,5},{1,5,3},{2,3,8},{3,5,4},{5,3,7},
        {1,1,7},{7,1,1},{1,7,1},{0,3,5},{3,0,5},{3,5,0}
    };
    const uint32_t sizes[] = {0,1,2,4,8};
    for (unsigned s=0; s<sizeof(shapes)/sizeof(shapes[0]); s++)
        for (uint32_t td=1; td<=6; td++)
            for (uint32_t tr=1; tr<=6; tr++)
                for (uint32_t tc=1; tc<=6; tc++)
                    for (unsigned e=0; e<sizeof(sizes)/sizeof(sizes[0]); e++) {
                        if (check_case(shapes[s][0],shapes[s][1],shapes[s][2],
                                       td,tr,tc,sizes[e],0,0)) return 1;
                    }
    for (uint32_t r=0; r<=7; r++)
        for (uint32_t c=0; c<=7; c++)
            for (uint32_t tr=1; tr<=8; tr++)
                for (uint32_t tc=1; tc<=8; tc++)
                    if (check_case(1,r,c,1,tr,tc,4,1,0)) return 1;
    if (invalid_and_boundary()) return 1;
    if (range_capacity_probe()) return 1;
    /* Compact and ragged/oversized workload matrix: report traffic order,
     * not cycle performance or physical DRAM command counts. */
    const uint32_t matrix[][6] = {
        {1,6,6,1,2,3},{1,3,5,1,2,3},{1,3,5,1,8,8},
        {2,3,8,1,1,2},{3,5,4,2,3,3},{5,3,7,2,2,4},
        {1,3,5,1,1,1},{1,3,5,1,3,5},
        {3,5,4,1,5,4},{3,5,4,8,8,8},{3,5,4,1,2,2}
    };
    for (unsigned i=0; i<sizeof(matrix)/sizeof(matrix[0]); i++)
        if (check_case(matrix[i][0],matrix[i][1],matrix[i][2],
                       matrix[i][3],matrix[i][4],matrix[i][5],4,matrix[i][0]==1,1)) return 1;
    printf("PASS: %u tiled permutation cases; invalid geometry, count/address overflow, exact-end and oversized-tile gates\n", cases);
    return 0;
}
