/* Iterator-to-DMA capacity, byte movement, and allocation atomicity gates. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tu_cmodel/memory/address_generator.h"

static long fail_at = -1, allocation_count;
void *__real_calloc(size_t n, size_t size);
void *__wrap_calloc(size_t n, size_t size) {
    if (fail_at >= 0 && allocation_count++ == fail_at) return NULL;
    return __real_calloc(n, size);
}
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static void capacity_case(unsigned rows, unsigned cols, int transposed, unsigned direction) {
    const unsigned base = 12, bytes = rows * cols * 4, size = base + bytes + 16;
    unsigned char *host = malloc(size), *expected = malloc(size);
    CHECK(host && expected);
    for (unsigned i = 0; i < size; ++i) host[i] = (unsigned char)(i * 37 + 11);
    memcpy(expected, host, size);
    tu_sram_region_t sram;
    tu_sram_init(&sram, size, "chain");
    unsigned char *raw = tu_sram_raw_ptr(&sram);
    memset(raw, 0xa5, size);
    if (direction == TU_DMA_DIR_TU_TO_HOST) {
        memcpy(raw + base, host + base, bytes);
        memset(host + base, 0x5a, bytes);
    }
    tu_agen_iterator_t it;
    uint32_t cfg[3] = {rows, cols, 4};
    uint32_t linear[2] = {rows * cols, 4};
    CHECK(tu_agen_iterator_init(&it, transposed ? TU_AGEN_MODE_TRANSPOSED : TU_AGEN_MODE_LINEAR,
                                base, transposed ? cfg : linear) == 0);
    tu_agen_range_t prefix[128];
    unsigned required = tu_agen_generate_ranges(&it, prefix, 128);
    tu_dma_descriptor_t *head = tu_agen_desc_chain_from_iterator(&it, 0, direction, &sram, host);
    CHECK((head != NULL) == (bytes != 0));
    unsigned count = 0, total = 0;
    for (tu_dma_descriptor_t *d = head; d; d = d->next) {
        CHECK(++count <= required);
        unsigned addr = direction == TU_DMA_DIR_HOST_TO_TU ? d->dst_base : d->src_base;
        unsigned expected_addr = transposed ? base + (((count - 1) % rows) * cols + (count - 1) / rows) * 4 : base;
        CHECK(addr == expected_addr);
        CHECK(addr >= base && addr + d->total_bytes <= base + bytes);
        CHECK((direction == TU_DMA_DIR_HOST_TO_TU ? d->src_host : d->dst_host) == host + addr);
        tu_dma_execute_desc(d);
        CHECK(d->completed);
        total += d->total_bytes;
    }
    CHECK(count == required && total == bytes);
    CHECK(memcmp(host, expected, size) == 0);
    CHECK(memcmp(raw + base, expected + base, bytes) == 0);
    for (unsigned i = 0; i < size; ++i)
        if (i < base || i >= base + bytes) CHECK(raw[i] == 0xa5);
    tu_dma_desc_destroy(head);
    tu_sram_destroy(&sram);
    free(host); free(expected);
    printf("capacity rows=%u cols=%u mode=%s direction=%u ranges=%u bytes=%u PASS\n",
           rows, cols, transposed ? "transposed" : "linear", direction, count, total);
}

static void invalid_cases(void) {
    tu_sram_region_t sram;
    unsigned char host[16] = {0};
    uint32_t cfg[2] = {1, 4};
    tu_agen_iterator_t it;
    tu_sram_init(&sram, sizeof(host), "invalid");
    CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_LINEAR, 0, cfg) == 0);
    tu_agen_iterator_t before = it;
    allocation_count = 0; fail_at = 0;
    CHECK(tu_agen_desc_chain_from_iterator(NULL, 0, 0, &sram, host) == NULL);
    CHECK(tu_agen_desc_chain_from_iterator(&it, 0, 0, NULL, host) == NULL);
    CHECK(tu_agen_desc_chain_from_iterator(&it, 0, 0, &sram, NULL) == NULL);
    CHECK(tu_agen_desc_chain_from_iterator(&it, 0, TU_DMA_DIR_TU_TO_TU, &sram, host) == NULL);
    CHECK(tu_agen_desc_chain_from_iterator(&it, 0, UINT32_MAX, &sram, host) == NULL);
    CHECK(memcmp(&it, &before, sizeof(it)) == 0);
    it.elem_size = 0;
    CHECK(tu_agen_desc_chain_from_iterator(&it, 0, 0, &sram, host) == NULL);
    CHECK(allocation_count == 0);
    fail_at = -1;
    tu_sram_destroy(&sram);
    puts("six invalid-input guards before allocation/traversal: PASS");
}

int main(int argc, char **argv) {
    tu_dma_init(false);
    if (argc > 1 && strcmp(argv[1], "oom") == 0) {
        tu_sram_region_t sram;
        tu_sram_init(&sram, 2048, "oom");
        memset(tu_sram_raw_ptr(&sram), 0xa5, 2048);
        unsigned char host[2048] = {0};
        uint32_t cfg[3] = {16, 16, 4};
        /* Range storage plus every descriptor allocation must fail atomically. */
        for (unsigned direction = 0; direction < 2; ++direction)
        for (long i = 0; i <= 256; ++i) {
            tu_agen_iterator_t it;
            CHECK(tu_agen_iterator_init(&it, TU_AGEN_MODE_TRANSPOSED, 0, cfg) == 0);
            allocation_count = 0; fail_at = i;
            tu_dma_descriptor_t *head = tu_agen_desc_chain_from_iterator(&it, 0, direction, &sram, host);
            fail_at = -1;
            CHECK(head == NULL);
            CHECK(g_tu_dma.total_bytes == 0);
            for (unsigned j = 0; j < sizeof(host); ++j) {
                CHECK(host[j] == 0);
                CHECK(((unsigned char *)tu_sram_raw_ptr(&sram))[j] == 0xa5);
            }
        }
        tu_sram_destroy(&sram);
        puts("514 allocation-failure direction/position cases: PASS");
    } else {
        for (unsigned dir = 0; dir < 2; ++dir) {
            capacity_case(0, 0, 0, dir);
            capacity_case(1, 4, 0, dir);
            capacity_case(2, 2, 1, dir);
            capacity_case(2, 64, 1, dir);
            capacity_case(16, 16, 1, dir);
            capacity_case(3, 129, 1, dir);
        }
        invalid_cases();
    }
    tu_dma_destroy();
    return 0;
}
