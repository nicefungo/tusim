/* BLOCK_INTERLEAVED compatibility is a linear byte sequence, not a modeled
 * bank scheduler. Test full-byte admission independently of its metadata.
 * Invalid/huge views are never dereferenced or exhaustively traversed. */
#include "tu_cmodel/memory/address_generator.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static uint64_t cases, rejected, bounded, checks, failures;
#define CHECK(c) do { checks++; if (!(c)) { \
    if (failures < 12) fprintf(stderr, "FAIL case=%" PRIu64 " line=%d: %s\n", \
                              cases, __LINE__, #c); \
    failures++; return; \
} } while (0)

static void check_case(uint32_t base, uint32_t count, uint32_t banks,
                       uint32_t block, uint32_t size, int verbose) {
    uint32_t cfg[4] = {count, banks, block, size}, saved[4];
    memcpy(saved, cfg, sizeof(cfg));
    uint32_t width = size ? size : 4;
    /* Wider oracle: no shared validator, quotient, or index decoder. */
    __uint128_t end = (__uint128_t)base + (__uint128_t)count * width;
    int valid = !count || end <= UINT32_MAX;
    tu_agen_iterator_t it;
    memset(&it, 0xa5, sizeof(it));
    cases++;
    int rc = tu_agen_iterator_init(&it, TU_AGEN_MODE_BLOCK_INTERLEAVED, base, cfg);
    CHECK(memcmp(cfg, saved, sizeof(cfg)) == 0);
    if ((rc == 0) != valid && failures < 12)
        fprintf(stderr, "base=%u count=%u banks=%u block=%u size=%u rc=%d expected=%s\n",
                base, count, banks, block, size, rc, valid ? "accept" : "reject");
    CHECK((rc == 0) == valid);
    if (!valid) {
        rejected++;
        CHECK(rc == -1 && tu_agen_total(&it) == 0 && !tu_agen_has_next(&it));
        CHECK(tu_agen_next(&it) == UINT32_MAX && it.current_idx == 0);
        tu_agen_reset(&it);
        CHECK(!tu_agen_has_next(&it) && tu_agen_next(&it) == UINT32_MAX);
        uint32_t out = 0x12345678;
        tu_agen_range_t range;
        memset(&range, 0xa5, sizeof(range));
        tu_agen_range_t before = range;
        CHECK(tu_agen_generate_all(&it, &out, 1) == 0 && out == 0x12345678);
        CHECK(tu_agen_generate_ranges(&it, &range, 1) == 0);
        CHECK(memcmp(&range, &before, sizeof(range)) == 0);
        return;
    }
    CHECK(rc == 0 && tu_agen_total(&it) == count);
    CHECK(it.elem_size == width && it.num_banks == (banks ? banks : 1));
    CHECK(it.bank_width == (block ? block : 1));
    CHECK(it.base_addr == base && it.current_idx == 0);
    if (count > 256) {
        CHECK(tu_agen_next(&it) == base);
        it.current_idx = count - 1; /* Bounded sample of a huge accepted view. */
        CHECK(tu_agen_next(&it) == (uint32_t)(end - width));
        CHECK(!tu_agen_has_next(&it));
        CHECK(tu_agen_next(&it) == UINT32_MAX && it.current_idx == count);
        CHECK(tu_agen_next(&it) == UINT32_MAX && it.current_idx == count);
        tu_agen_reset(&it);
        CHECK(it.current_idx == 0 && tu_agen_next(&it) == base);
        return;
    }
    bounded++;
    uint32_t expected[256], actual[256];
    uint64_t address = base;
    for (uint32_t i = 0; i < count; i++, address += width) {
        expected[i] = (uint32_t)address;
        CHECK(tu_agen_has_next(&it));
        CHECK(tu_agen_next(&it) == expected[i]);
        CHECK(address + width <= UINT32_MAX);
    }
    CHECK(!tu_agen_has_next(&it));
    CHECK(tu_agen_next(&it) == UINT32_MAX && it.current_idx == count);
    CHECK(tu_agen_next(&it) == UINT32_MAX && it.current_idx == count);
    tu_agen_reset(&it);
    CHECK(it.current_idx == 0 && tu_agen_has_next(&it) == (count != 0));
    CHECK(tu_agen_generate_all(&it, actual, 256) == count);
    CHECK(memcmp(actual, expected, count * sizeof(*actual)) == 0);
    CHECK(tu_agen_generate_all(&it, actual, count ? count - 1 : 0) ==
          (count ? count - 1 : 0));
    if (count) CHECK(tu_agen_next(&it) == expected[count - 1]);
    /* Range replay, including the legacy im2col sentinel split if encountered:
     * a split does not change these ordinary addresses or complete coverage. */
    tu_agen_range_t ranges[256];
    uint32_t nr = tu_agen_generate_ranges(&it, ranges, 256), pos = 0;
    CHECK(nr <= 256);
    for (uint32_t r = 0; r < nr; r++) {
        CHECK(ranges[r].elem_size == width && ranges[r].stride == width);
        CHECK(ranges[r].total_bytes > 0 && ranges[r].total_bytes % width == 0);
        for (uint64_t off = 0; off < ranges[r].total_bytes; off += width) {
            CHECK(pos < count && (uint64_t)ranges[r].base_addr + off == expected[pos]);
            pos++;
        }
    }
    CHECK(pos == count && it.current_idx == count);
    if (verbose)
        printf("base=%u count=%u banks=%u block=%u width=%u logical_bytes=%" PRIu64
               " logical_runs=%u PASS\n", base, count, it.num_banks, it.bank_width,
               width, (uint64_t)count * width, nr);
}

static uint32_t rng = 0x7c136fa9;
static uint32_t random_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

/* Valid layouts exercise the existing range -> descriptor -> direct-copy path.
 * Each descriptor remains unsubmitted/caller-owned and is destroyed once. */
static void movement_case(uint32_t banks, uint32_t block, uint32_t direction) {
    unsigned char host[128], expected_host[128], expected_sram[128];
    uint32_t cfg[4] = {17, banks, block, 4};
    const uint32_t base = 13, bytes = 68;
    tu_agen_iterator_t it;
    tu_sram_region_t sram;
    tu_dma_descriptor_t *head = NULL;
    int okay = 1;
    for (unsigned i = 0; i < sizeof(host); i++) host[i] = (unsigned char)(i * 37 + 11);
    memset(expected_sram, 0xa5, sizeof(expected_sram));
    tu_sram_init(&sram, sizeof(host), "block-compatibility");
    unsigned char *raw = tu_sram_raw_ptr(&sram);
    memset(raw, 0xa5, sizeof(host));
    if (direction == TU_DMA_DIR_HOST_TO_TU)
        memcpy(expected_sram + base, host + base, bytes);
    else {
        memcpy(raw + base, host + base, bytes);
        memcpy(expected_sram, raw, sizeof(host));
        memset(host + base, 0x5a, bytes);
    }
    memcpy(expected_host, host, sizeof(host));
    if (direction == TU_DMA_DIR_TU_TO_HOST)
        memcpy(expected_host + base, raw + base, bytes);
    checks++;
    if (tu_agen_iterator_init(&it, TU_AGEN_MODE_BLOCK_INTERLEAVED, base, cfg)) okay = 0;
    uint64_t before = g_tu_dma.total_bytes;
    if (okay) head = tu_agen_desc_chain_from_iterator(&it, 0, direction, &sram, host);
    checks++;
    if (!head || head->next || head->total_bytes != bytes) okay = 0;
    if (okay) tu_dma_execute_desc(head);
    checks += 4;
    if (!head || !head->completed || g_tu_dma.total_bytes - before != bytes ||
        memcmp(host, expected_host, sizeof(host)) ||
        memcmp(raw, expected_sram, sizeof(host))) okay = 0;
    tu_dma_desc_destroy(head);
    tu_sram_destroy(&sram);
    if (!okay) {
        failures++;
        fprintf(stderr, "FAIL movement banks=%u block=%u direction=%u\n", banks, block, direction);
    } else printf("movement banks=%u block=%u direction=%u bytes=%u full_region_unchanged_outside=PASS\n",
                  banks, block, direction, bytes);
}

static void api_guards(void) {
    tu_agen_iterator_t it;
    uint32_t cfg[4] = {1, 0, 0, 0};
    CHECK(tu_agen_iterator_init(NULL, TU_AGEN_MODE_BLOCK_INTERLEAVED, 0, cfg) == -1);
    memset(&it, 0xa5, sizeof(it));
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_BLOCK_INTERLEAVED, 0, NULL) == -1);
    CHECK(!tu_agen_has_next(&it) && tu_agen_next(&it) == UINT32_MAX);
    CHECK(it.current_idx == 0 && it.total_elements == 0);
}

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--probes"))) return 2;
    /* Admission-only RED probes: no memcpy, DMA or SRAM dereference. */
    check_case(UINT32_MAX - 7, 2, 4, 2, 4, 0); /* includes reserved byte */
    check_case(UINT32_MAX - 7, 3, 4, 2, 4, 0); /* wraps address */
    check_case(0, UINT32_MAX / 4 + 1, 4, 2, 4, 0); /* byte product wraps */
    check_case(UINT32_MAX, 1, 0, 0, 1, 0); /* first address is exhaustion */
    check_case(1, 1, 3, 7, UINT32_MAX, 0); /* final element overruns */
    check_case(UINT32_MAX - 8, 2, 4, 2, 4, 1); /* exact exclusive-end control */
    check_case(UINT32_MAX, 0, UINT32_MAX, UINT32_MAX, UINT32_MAX, 1); /* empty */
    if (argc == 1) {
        const uint32_t counts[] = {0,1,2,3,7,8,9,31,255,256,257,65535,65536,
                                  UINT32_MAX/4,UINT32_MAX/4+1,UINT32_MAX};
        const uint32_t sizes[] = {0,1,2,3,4,7,8,65536,UINT32_MAX};
        const uint32_t bases[] = {0,13,UINT32_MAX-1,UINT32_MAX-7,UINT32_MAX-8,
                                  UINT32_MAX-255,0xFFFFFEFC,UINT32_MAX};
        const uint32_t banks[] = {0,1,3,32,UINT32_MAX};
        const uint32_t blocks[] = {0,1,2,7,256,UINT32_MAX};
        for (unsigned n=0; n<sizeof(counts)/sizeof(*counts); n++)
            for (unsigned s=0; s<sizeof(sizes)/sizeof(*sizes); s++)
                for (unsigned a=0; a<sizeof(bases)/sizeof(*bases); a++)
                    for (unsigned b=0; b<sizeof(banks)/sizeof(*banks); b++)
                        for (unsigned w=0; w<sizeof(blocks)/sizeof(*blocks); w++)
                            check_case(bases[a],counts[n],banks[b],blocks[w],sizes[s],0);
        /* Target exact exclusive-end boundary and one byte beyond, then random
         * full-domain input. Fixed seed; do not normalize invalid source tokens. */
        for (unsigned i=0; i<4096; i++) {
            uint32_t base = random_u32(), size = (random_u32() % 65536) + 1;
            uint32_t count = (UINT32_MAX - base) / size;
            uint32_t banks_i = random_u32(), block_i = random_u32();
            check_case(base,count,banks_i,block_i,size,0);
            check_case(base,count+1,banks_i,block_i,size,0);
            check_case(random_u32(),random_u32(),banks_i,block_i,random_u32(),0);
        }
        api_guards();
        /* Complete logical layout matrix: changing metadata does not prove
         * banking, bank conflicts, parallelism, or a traffic/cycle reduction. */
        const uint32_t matrix[][2] = {{1,1},{4,1},{4,2},{4,7},{32,4}};
        tu_dma_init(false);
        for (unsigned i=0; i<sizeof(matrix)/sizeof(*matrix); i++) {
            check_case(256,17,matrix[i][0],matrix[i][1],4,1);
            movement_case(matrix[i][0],matrix[i][1],TU_DMA_DIR_HOST_TO_TU);
            movement_case(matrix[i][0],matrix[i][1],TU_DMA_DIR_TU_TO_HOST);
        }
        tu_dma_destroy();
    }
    printf("%s: cases=%" PRIu64 " rejected=%" PRIu64 " bounded_sequences=%" PRIu64
           " checks=%" PRIu64 " failures=%" PRIu64 "\n", failures ? "FAIL" : "PASS",
           cases,rejected,bounded,checks,failures);
    return failures ? 1 : 0;
}
