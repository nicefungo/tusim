/*
 * TinyTU SRAM Module — Implementation
 */

#include "tu_sram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void tu_sram_init(tu_sram_region_t *r, uint32_t size_bytes, const char *name) {
    memset(r, 0, sizeof(*r));
    r->total_size = size_bytes;
    r->name = name;
    r->banks.data = (uint8_t *)calloc(1, size_bytes);
    r->banks.size = size_bytes;
    r->banks.bank_count = TU_SRAM_BANKS;
    r->banks.bank_width = TU_SRAM_BANK_WIDTH;
}

void tu_sram_destroy(tu_sram_region_t *r) {
    free(r->banks.data);
    memset(r, 0, sizeof(*r));
}

void tu_sram_read(const tu_sram_region_t *r, uint32_t addr, void *out) {
    if (addr + r->banks.bank_width > r->total_size) {
        fprintf(stderr, "SRAM %s read overflow: addr=%u size=%u\n", r->name, addr, r->total_size);
        abort();
    }
    uint32_t bank = (addr / r->banks.bank_width) % r->banks.bank_count;
    (void)bank; /* tracked for conflict detection at higher level */
    memcpy(out, r->banks.data + addr, r->banks.bank_width);
    ((tu_sram_region_t *)r)->banks.reads++;
}

void tu_sram_write(tu_sram_region_t *r, uint32_t addr, const void *data) {
    if (addr + r->banks.bank_width > r->total_size) {
        fprintf(stderr, "SRAM %s write overflow: addr=%u size=%u\n", r->name, addr, r->total_size);
        abort();
    }
    memcpy(r->banks.data + addr, data, r->banks.bank_width);
    r->banks.writes++;
}

void tu_sram_read_bulk(const tu_sram_region_t *r, uint32_t addr, void *out, uint32_t bytes) {
    if (addr + bytes > r->total_size) { fprintf(stderr, "SRAM bulk read overflow\n"); abort(); }
    memcpy(out, r->banks.data + addr, bytes);
    ((tu_sram_region_t *)r)->banks.reads += (bytes + r->banks.bank_width - 1) / r->banks.bank_width;
}

void tu_sram_write_bulk(tu_sram_region_t *r, uint32_t addr, const void *data, uint32_t bytes) {
    if (addr + bytes > r->total_size) { fprintf(stderr, "SRAM bulk write overflow\n"); abort(); }
    memcpy(r->banks.data + addr, data, bytes);
    r->banks.writes += (bytes + r->banks.bank_width - 1) / r->banks.bank_width;
}

void *tu_sram_raw_ptr(tu_sram_region_t *r) {
    return r->banks.data;
}

void tu_sram_print_stats(const tu_sram_region_t *r) {
    fprintf(stderr, "  SRAM %-8s: reads=%lu writes=%lu conflicts=%lu stalls=%lu\n",
            r->name, r->banks.reads, r->banks.writes, r->banks.conflicts, r->banks.stall_cycles);
}
