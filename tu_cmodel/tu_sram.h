/*
 * TinyTU SRAM Module
 * ===================
 * Banked scratchpad SRAM with configurable banking, conflict detection,
 * and latency accounting.
 */

#ifndef TU_SRAM_H
#define TU_SRAM_H

#include "tu_config.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SRAM bank descriptor */
typedef struct {
    uint8_t *data;
    uint32_t size;       /* bytes */
    uint32_t bank_count;
    uint32_t bank_width; /* bytes per bank (word size) */
    uint64_t reads;
    uint64_t writes;
    uint64_t conflicts;  /* bank conflicts detected */
    uint64_t stall_cycles;
} tu_sram_bank_t;

/* SRAM region (W-buffer, A-buffer, O-buffer) */
typedef struct {
    tu_sram_bank_t banks;
    uint32_t       total_size;
    const char    *name;
} tu_sram_region_t;

/* Initialize a banked SRAM region */
void tu_sram_init(tu_sram_region_t *r, uint32_t size_bytes, const char *name);

/* Destroy SRAM region */
void tu_sram_destroy(tu_sram_region_t *r);

/*
 * Access functions.
 * addr: byte offset within the region.
 * All accesses increment counters and check for bank conflicts.
 */

/* Read a single word (bank_width bytes) */
void tu_sram_read(const tu_sram_region_t *r, uint32_t addr, void *out);

/* Write a single word (bank_width bytes) */
void tu_sram_write(tu_sram_region_t *r, uint32_t addr, const void *data);

/* Bulk read (sequential, single bank at a time for contiguous access) */
void tu_sram_read_bulk(const tu_sram_region_t *r, uint32_t addr, void *out, uint32_t bytes);

/* Bulk write */
void tu_sram_write_bulk(tu_sram_region_t *r, uint32_t addr, const void *data, uint32_t bytes);

/* Compute bank index for a given byte address */
static inline uint32_t tu_sram_bank_index(const tu_sram_region_t *r, uint32_t addr) {
    return (addr / r->banks.bank_width) % r->banks.bank_count;
}

/* Get raw data pointer (for direct memcpy — bypasses bank modeling) */
void *tu_sram_raw_ptr(tu_sram_region_t *r);

/* Print bank statistics */
void tu_sram_print_stats(const tu_sram_region_t *r);

#ifdef __cplusplus
}
#endif

#endif /* TU_SRAM_H */
