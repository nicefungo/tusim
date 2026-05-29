/*
 * TinyTU SRAM Module — Bandwidth-Modeled Implementation
 * ======================================================
 * M2: Per-bank bandwidth metering with refill-based budget,
 *     arbitration modes, and stall cycle accounting.
 */

#include "tu_sram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Internal: bandwidth arbitration for a single access ---- */

/*
 * Try to consume one word from a bank's bandwidth budget.
 * Returns 0 on success, or stall_penalty cycles if budget is exhausted.
 */
static uint64_t sram_bw_consume(tu_sram_bank_t *b, uint32_t bank_idx, bool is_write) {
    if (!b->bw_modeling || !b->bw_banks)
        return 0;

    tu_sram_bw_bank_t *bw = &b->bw_banks[bank_idx];

    /* Track activity */
    bw->total_cycles_used++;

    if (bw->words_available > 0) {
        bw->words_available--;
        if (is_write) bw->writes_served++;
        else          bw->reads_served++;
        return 0;  /* No stall */
    }

    /* Budget exhausted — stall */
    if (is_write) {
        bw->write_stalls++;
        b->stall_cycles += b->stall_penalty;
    } else {
        bw->read_stalls++;
        b->stall_cycles += b->stall_penalty;
    }
    return b->stall_penalty;
}

/* ---- Lifecycle ---- */

void tu_sram_init(tu_sram_region_t *r, uint32_t size_bytes, const char *name) {
    tu_sram_init_bw(r, size_bytes, name,
                    TU_SRAM_WORDS_PER_CYCLE,
                    TU_SRAM_ARB_MODE,
                    TU_SRAM_BW_STALL_PENALTY,
                    TU_SRAM_BW_WINDOW_CYCLES);
}

void tu_sram_init_bw(tu_sram_region_t *r, uint32_t size_bytes, const char *name,
                     uint8_t words_per_cycle, uint8_t arb_mode,
                     uint8_t stall_penalty, uint64_t refill_window) {
    memset(r, 0, sizeof(*r));
    r->total_size = size_bytes;
    r->name = name;

    tu_sram_bank_t *b = &r->banks;
    b->data = (uint8_t *)calloc(1, size_bytes);
    b->size = size_bytes;
    b->bank_count = TU_SRAM_BANKS;
    b->bank_width = TU_SRAM_BANK_WIDTH;
    b->words_per_cycle = words_per_cycle;
    b->arb_mode = arb_mode;
    b->stall_penalty = stall_penalty;
    b->bw_refill_window = refill_window;
    b->bw_modeling = true;
    b->current_cycle = 0;

    /* Allocate per-bank bandwidth meters */
    b->bw_banks = (tu_sram_bw_bank_t *)calloc(b->bank_count, sizeof(tu_sram_bw_bank_t));
    for (uint32_t i = 0; i < b->bank_count; i++) {
        b->bw_banks[i].words_available = words_per_cycle;
    }
}

void tu_sram_destroy(tu_sram_region_t *r) {
    free(r->banks.data);
    free(r->banks.bw_banks);
    memset(r, 0, sizeof(*r));
}

/* ---- Cycle Management ---- */

void tu_sram_advance_cycle(tu_sram_region_t *r, uint64_t cycles) {
    tu_sram_bank_t *b = &r->banks;
    b->current_cycle += cycles;
    tu_sram_refill_bw(r);
}

void tu_sram_refill_bw(tu_sram_region_t *r) {
    tu_sram_bank_t *b = &r->banks;
    if (!b->bw_modeling || !b->bw_banks)
        return;

    uint64_t window = (b->bw_refill_window > 0) ? b->bw_refill_window : 1;

    for (uint32_t i = 0; i < b->bank_count; i++) {
        tu_sram_bw_bank_t *bw = &b->bw_banks[i];

        /* Check if refill window has elapsed */
        if (b->current_cycle - bw->last_refill_cycle >= window) {
            /* Refill: reset to full bandwidth budget */
            bw->words_available = b->words_per_cycle;
            bw->last_refill_cycle = b->current_cycle;
        }
    }
}

void tu_sram_set_bw_modeling(tu_sram_region_t *r, bool enabled) {
    r->banks.bw_modeling = enabled;
}

/* ---- Access Functions ---- */

static void bounds_check(const tu_sram_region_t *r, uint32_t addr, uint32_t size) {
    if (addr + size > r->total_size) {
        fprintf(stderr, "SRAM %s overflow: addr=%u size=%u max=%u\n",
                r->name, addr, size, r->total_size);
        abort();
    }
}

uint64_t tu_sram_read(tu_sram_region_t *r, uint32_t addr, void *out) {
    bounds_check(r, addr, r->banks.bank_width);
    uint32_t bank = tu_sram_bank_index(r, addr);

    /* Bandwidth arbitration */
    uint64_t stall = sram_bw_consume(&r->banks, bank, false);

    memcpy(out, r->banks.data + addr, r->banks.bank_width);
    r->banks.reads++;
    return stall;
}

uint64_t tu_sram_write(tu_sram_region_t *r, uint32_t addr, const void *data) {
    bounds_check(r, addr, r->banks.bank_width);
    uint32_t bank = tu_sram_bank_index(r, addr);

    /* Bandwidth arbitration */
    uint64_t stall = sram_bw_consume(&r->banks, bank, true);

    memcpy(r->banks.data + addr, data, r->banks.bank_width);
    r->banks.writes++;
    return stall;
}

uint64_t tu_sram_read_bulk(tu_sram_region_t *r, uint32_t addr, void *out, uint32_t bytes) {
    bounds_check(r, addr, bytes);
    uint32_t bw = r->banks.bank_width;
    uint64_t total_stall = 0;
    uint32_t words = (bytes + bw - 1) / bw;

    for (uint32_t i = 0; i < words; i++) {
        uint32_t off = addr + i * bw;
        uint32_t bank = tu_sram_bank_index(r, off);
        total_stall += sram_bw_consume(&r->banks, bank, false);
    }

    memcpy(out, r->banks.data + addr, bytes);
    r->banks.reads += words;
    return total_stall;
}

uint64_t tu_sram_write_bulk(tu_sram_region_t *r, uint32_t addr, const void *data, uint32_t bytes) {
    bounds_check(r, addr, bytes);
    uint32_t bw = r->banks.bank_width;
    uint64_t total_stall = 0;
    uint32_t words = (bytes + bw - 1) / bw;

    for (uint32_t i = 0; i < words; i++) {
        uint32_t off = addr + i * bw;
        uint32_t bank = tu_sram_bank_index(r, off);
        total_stall += sram_bw_consume(&r->banks, bank, true);
    }

    memcpy(r->banks.data + addr, data, bytes);
    r->banks.writes += words;
    return total_stall;
}

void *tu_sram_raw_ptr(tu_sram_region_t *r) {
    return r->banks.data;
}

/* ---- Statistics ---- */

float tu_sram_get_bandwidth_utilization(const tu_sram_region_t *r) {
    const tu_sram_bank_t *b = &r->banks;
    if (!b->bw_modeling || !b->bw_banks || b->current_cycle == 0)
        return 0.0f;

    /* Max possible words served = bank_count × words_per_cycle × refill_periods */
    uint64_t refill_periods = b->current_cycle / b->bw_refill_window;
    if (refill_periods == 0) refill_periods = 1;
    uint64_t max_possible = (uint64_t)b->bank_count * b->words_per_cycle * refill_periods;

    uint64_t total_served = 0;
    for (uint32_t i = 0; i < b->bank_count; i++) {
        total_served += b->bw_banks[i].reads_served + b->bw_banks[i].writes_served;
    }

    if (max_possible == 0) return 0.0f;
    float util = (float)total_served / (float)max_possible;
    return util > 1.0f ? 1.0f : util;
}

void tu_sram_get_bank_bw_stats(const tu_sram_region_t *r, uint32_t bank_idx,
                               uint64_t *reads_out, uint64_t *writes_out,
                               uint64_t *read_stalls_out, uint64_t *write_stalls_out,
                               float *utilization_out) {
    const tu_sram_bank_t *b = &r->banks;
    if (bank_idx >= b->bank_count || !b->bw_banks) {
        if (reads_out)       *reads_out = 0;
        if (writes_out)      *writes_out = 0;
        if (read_stalls_out) *read_stalls_out = 0;
        if (write_stalls_out)*write_stalls_out = 0;
        if (utilization_out) *utilization_out = 0.0f;
        return;
    }

    const tu_sram_bw_bank_t *bw = &b->bw_banks[bank_idx];
    if (reads_out)       *reads_out = bw->reads_served;
    if (writes_out)      *writes_out = bw->writes_served;
    if (read_stalls_out) *read_stalls_out = bw->read_stalls;
    if (write_stalls_out)*write_stalls_out = bw->write_stalls;

    if (utilization_out) {
        uint64_t refill_periods = b->current_cycle / b->bw_refill_window;
        if (refill_periods == 0) refill_periods = 1;
        uint64_t max_bank = (uint64_t)b->words_per_cycle * refill_periods;
        uint64_t served = bw->reads_served + bw->writes_served;
        float u = max_bank > 0 ? (float)served / (float)max_bank : 0.0f;
        *utilization_out = u > 1.0f ? 1.0f : u;
    }
}

uint64_t tu_sram_get_cycle(const tu_sram_region_t *r) {
    return r->banks.current_cycle;
}

void tu_sram_print_stats(const tu_sram_region_t *r) {
    const tu_sram_bank_t *b = &r->banks;
    float util = tu_sram_get_bandwidth_utilization(r);

    fprintf(stderr,
        "  SRAM %-8s: reads=%lu writes=%lu conflicts=%lu stalls=%lu "
        "bw_util=%.1f%%\n",
        r->name,
        (unsigned long)b->reads,
        (unsigned long)b->writes,
        (unsigned long)b->conflicts,
        (unsigned long)b->stall_cycles,
        util * 100.0f);

    /* Top 3 hottest banks by stall count */
    if (b->bw_banks && b->bank_count > 0) {
        uint32_t hottest[3] = {0};
        uint64_t hottest_stalls[3] = {0};

        for (uint32_t i = 0; i < b->bank_count; i++) {
            uint64_t s = b->bw_banks[i].read_stalls + b->bw_banks[i].write_stalls;
            if (s > hottest_stalls[0]) {
                hottest_stalls[2] = hottest_stalls[1]; hottest[2] = hottest[1];
                hottest_stalls[1] = hottest_stalls[0]; hottest[1] = hottest[0];
                hottest_stalls[0] = s; hottest[0] = i;
            } else if (s > hottest_stalls[1]) {
                hottest_stalls[2] = hottest_stalls[1]; hottest[2] = hottest[1];
                hottest_stalls[1] = s; hottest[1] = i;
            } else if (s > hottest_stalls[2]) {
                hottest_stalls[2] = s; hottest[2] = i;
            }
        }

        if (hottest_stalls[0] > 0) {
            fprintf(stderr, "    hottest banks (stalls):");
            for (int h = 0; h < 3 && hottest_stalls[h] > 0; h++) {
                fprintf(stderr, " b%u=%lu", hottest[h], (unsigned long)hottest_stalls[h]);
            }
            fprintf(stderr, "\n");
        }
    }
}
