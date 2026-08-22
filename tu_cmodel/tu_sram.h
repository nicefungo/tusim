/*
 * TinyTU SRAM Module
 * ===================
 * Banked scratchpad SRAM with configurable banking, conflict detection,
 * bandwidth modeling, arbitration, and latency accounting.
 *
 * M2: Bandwidth Modeling
 * =======================
 * Each bank has a per-refill-window bandwidth grant
 * (TU_SRAM_WORDS_PER_CYCLE; historical identifier retained).
 * When multiple accesses target the same bank in one refill window, the
 * budget determines which accesses return a stall penalty. The arb_mode field
 * is reserved metadata; scalar calls do not implement RR/priority ordering.
 */

#ifndef TU_SRAM_H
#define TU_SRAM_H

#include "tu_config.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Bandwidth state per bank ---- */
typedef struct {
    int32_t   words_available;   /* Remaining word budget this cycle window */
    uint64_t  last_refill_cycle; /* Cycle when budget was last refilled */
    uint64_t  reads_served;      /* Total reads served by this bank */
    uint64_t  writes_served;     /* Total writes served by this bank */
    uint64_t  read_stalls;       /* Read accesses that stalled on this bank */
    uint64_t  write_stalls;      /* Write accesses that stalled on this bank */
    uint64_t  total_cycles_used; /* Cycles where any access occurred */
} tu_sram_bw_bank_t;

/* SRAM bank descriptor */
typedef struct {
    uint8_t   *data;
    uint32_t   size;             /* bytes */
    uint32_t   bank_count;
    uint32_t   bank_width;       /* bytes per bank (word size) */
    uint64_t   reads;
    uint64_t   writes;
    uint64_t   conflicts;        /* bank conflicts detected */
    uint64_t   stall_cycles;      /* total stall cycles from bandwidth contention */

    /* M2: per-bank bandwidth state */
    tu_sram_bw_bank_t *bw_banks; /* Array of bandwidth meters (one per bank) */
    uint64_t   bw_refill_window; /* Cycles between refills */
    uint64_t   current_cycle;    /* Monotonically increasing cycle counter */
    uint8_t    words_per_cycle;  /* Words granted per bank/refill window */
    uint8_t    arb_mode;         /* Arbitration policy */
    uint8_t    stall_penalty;    /* Cycles to add per bandwidth stall */
    bool       bw_modeling;      /* Whether bandwidth modeling is active */
} tu_sram_bank_t;

/* SRAM region (W-buffer, A-buffer, O-buffer) */
typedef struct tu_double_buffer_t tu_double_buffer_t;

typedef struct {
    tu_sram_bank_t      banks;
    uint32_t            total_size;
    const char         *name;
    tu_double_buffer_t *db;       /* Double-buffer state (NULL = disabled) */
} tu_sram_region_t;

/* ---- Lifecycle ---- */

/* Initialize a banked SRAM region */
void tu_sram_init(tu_sram_region_t *r, uint32_t size_bytes, const char *name);

/* Initialize with full bandwidth modeling parameters */
void tu_sram_init_bw(tu_sram_region_t *r, uint32_t size_bytes, const char *name,
                     uint8_t words_per_cycle, uint8_t arb_mode,
                     uint8_t stall_penalty, uint64_t refill_window);

/* Runtime-geometry variant. Zero-valued fields inherit checked-in defaults.
 * words_per_cycle is a grant per refill window; it is literally a per-cycle
 * issue rate only when refill_window is one cycle. */
void tu_sram_init_runtime(tu_sram_region_t *r, uint32_t size_bytes,
                          const char *name, uint32_t bank_count,
                          uint32_t bank_width, uint8_t words_per_cycle,
                          uint8_t stall_penalty, uint64_t refill_window);

/* Destroy SRAM region (frees all memory) */
void tu_sram_destroy(tu_sram_region_t *r);

/* ---- Access Functions ---- */

/*
 * All accesses now go through bandwidth arbitration.
 * addr: byte offset within the region.
 * Returns the number of stall cycles incurred (0 = no stall).
 */

/* Read a single word (bank_width bytes). Returns stall cycles. */
uint64_t tu_sram_read(tu_sram_region_t *r, uint32_t addr, void *out);

/* Write a single word (bank_width bytes). Returns stall cycles. */
uint64_t tu_sram_write(tu_sram_region_t *r, uint32_t addr, const void *data);

/* Bulk read (sequential, models per-word bandwidth). Returns total stall cycles. */
uint64_t tu_sram_read_bulk(tu_sram_region_t *r, uint32_t addr, void *out, uint32_t bytes);

/* Bulk write. Returns total stall cycles. */
uint64_t tu_sram_write_bulk(tu_sram_region_t *r, uint32_t addr, const void *data, uint32_t bytes);

/* ---- Bandwidth Model Control ---- */

/* Advance the cycle counter. Call periodically to trigger bandwidth refill. */
void tu_sram_advance_cycle(tu_sram_region_t *r, uint64_t cycles);

/* Refill all bank bandwidth budgets (called internally by advance_cycle) */
void tu_sram_refill_bw(tu_sram_region_t *r);

/* Enable/disable bandwidth modeling at runtime */
void tu_sram_set_bw_modeling(tu_sram_region_t *r, bool enabled);

/* ---- Access Helpers ---- */

/* Compute bank index for a given byte address */
static inline uint32_t tu_sram_bank_index(const tu_sram_region_t *r, uint32_t addr) {
    return (addr / r->banks.bank_width) % r->banks.bank_count;
}

/* Check if two addresses target the same bank (conflict) */
static inline bool tu_sram_is_bank_conflict(const tu_sram_region_t *r,
                                              uint32_t addr1, uint32_t addr2) {
    return tu_sram_bank_index(r, addr1) == tu_sram_bank_index(r, addr2);
}

/* Get raw data pointer (for direct memcpy — bypasses bank modeling) */
void *tu_sram_raw_ptr(tu_sram_region_t *r);

/* ---- Statistics ---- */

/* Print bank statistics including bandwidth utilization */
void tu_sram_print_stats(const tu_sram_region_t *r);

/* Get comprehensive bandwidth utilization as a float [0.0, 1.0] */
float tu_sram_get_bandwidth_utilization(const tu_sram_region_t *r);

/* Get per-bank bandwidth stats for a specific bank */
void tu_sram_get_bank_bw_stats(const tu_sram_region_t *r, uint32_t bank_idx,
                               uint64_t *reads, uint64_t *writes,
                               uint64_t *read_stalls, uint64_t *write_stalls,
                               float *utilization);

/* Current cycle counter */
uint64_t tu_sram_get_cycle(const tu_sram_region_t *r);

#ifdef __cplusplus
}
#endif

#endif /* TU_SRAM_H */
