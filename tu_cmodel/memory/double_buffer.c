/*
 * TU CModel — Double Buffering Implementation
 * =============================================
 * Gap A7: Ping-pong buffer management for DMA/compute overlap.
 *
 * All double-buffer state lives in tu_double_buffer_t, embedded
 * as a heap-allocated pointer in tu_sram_region_t->db.
 */
#include "double_buffer.h"
#include "../tu_sram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Public API ---- */

int tu_sram_enable_double_buffer(tu_sram_region_t *r) {
    if (!r || r->total_size == 0)
        return -1;

    if (r->db && r->db->enabled)
        return 0;  /* Already enabled */

    /* Allocate double-buffer state */
    tu_double_buffer_t *db = (tu_double_buffer_t *)calloc(1, sizeof(tu_double_buffer_t));
    if (!db) {
        fprintf(stderr, "tu_db: malloc(double_buffer_t) failed\n");
        return -1;
    }

    /* Allocate shadow buffer (same size as primary) */
    db->shadow_data = (uint8_t *)calloc(1, r->total_size);
    if (!db->shadow_data) {
        fprintf(stderr, "tu_db: malloc(shadow %u bytes) failed\n", r->total_size);
        free(db);
        return -1;
    }

    db->enabled     = true;
    db->buffer_size = r->total_size;
    db->active_idx  = 0;        /* Start with primary buffer active */
    db->swap_count  = 0;
    db->shadow_dirty = false;
    db->overlapped_cycles = 0;

    r->db = db;

    fprintf(stderr, "tu_db: %s double-buffered (%u bytes ×2 = %u total)\n",
            r->name, r->total_size, r->total_size * 2);
    return 0;
}

void tu_sram_disable_double_buffer(tu_sram_region_t *r) {
    if (!r || !r->db) return;

    /* If shadow is active, copy shadow back to primary */
    if (r->db->active_idx == 1) {
        memcpy(r->banks.data, r->db->shadow_data, r->total_size);
    }

    free(r->db->shadow_data);
    free(r->db);
    r->db = NULL;
}

bool tu_sram_is_double_buffered(const tu_sram_region_t *r) {
    return r && r->db && r->db->enabled;
}

uint64_t tu_sram_swap_buffers(tu_sram_region_t *r) {
    if (!r || !r->db || !r->db->enabled)
        return 0;

    /* Toggle active buffer */
    r->db->active_idx = (r->db->active_idx == 0) ? 1 : 0;
    r->db->swap_count++;
    r->db->shadow_dirty = false;  /* New shadow is clean (whatever was active) */

    return r->db->swap_count;
}

uint8_t *tu_sram_get_active_ptr(tu_sram_region_t *r) {
    if (!r) return NULL;
    if (r->db && r->db->enabled) {
        return (r->db->active_idx == 0) ? r->banks.data : r->db->shadow_data;
    }
    return r->banks.data;
}

uint8_t *tu_sram_get_shadow_ptr(tu_sram_region_t *r) {
    if (!r || !r->db || !r->db->enabled)
        return NULL;
    /* Shadow is the OPPOSITE of active */
    return (r->db->active_idx == 0) ? r->db->shadow_data : r->banks.data;
}

void tu_sram_notify_shadow_write(tu_sram_region_t *r,
                                  uint32_t bytes, uint64_t cycles) {
    if (!r || !r->db || !r->db->enabled) return;

    r->db->dma_to_shadow_bytes  += bytes;
    r->db->dma_to_shadow_cycles += cycles;
    r->db->shadow_dirty          = true;
}

bool tu_sram_is_shadow_dirty(const tu_sram_region_t *r) {
    return r && r->db && r->db->enabled && r->db->shadow_dirty;
}

void tu_sram_get_db_stats(const tu_sram_region_t *r, tu_db_stats_t *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));

    if (!r || !r->db) {
        stats->enabled = false;
        return;
    }

    stats->enabled              = r->db->enabled;
    stats->buffer_size          = r->db->buffer_size;
    stats->swap_count           = r->db->swap_count;
    stats->dma_to_shadow_bytes  = r->db->dma_to_shadow_bytes;
    stats->dma_to_shadow_cycles = r->db->dma_to_shadow_cycles;
    stats->overlapped_cycles    = r->db->overlapped_cycles;
    stats->shadow_dirty         = r->db->shadow_dirty;
}

void tu_sram_print_db_stats(const tu_sram_region_t *r) {
    if (!r || !r->db || !r->db->enabled) {
        fprintf(stderr, "  SRAM %-8s: double-buffering DISABLED\n", r ? r->name : "?");
        return;
    }

    tu_double_buffer_t *db = r->db;
    fprintf(stderr,
        "  SRAM %-8s: DB 2×%uB  swaps=%lu  shadow_writes=%luB/%lucyc  "
        "overlapped=%lucyc  dirty=%s\n",
        r->name,
        db->buffer_size,
        (unsigned long)db->swap_count,
        (unsigned long)db->dma_to_shadow_bytes,
        (unsigned long)db->dma_to_shadow_cycles,
        (unsigned long)db->overlapped_cycles,
        db->shadow_dirty ? "yes" : "no");
}

void tu_sram_record_overlapped_cycles(tu_sram_region_t *r, uint64_t cycles) {
    if (r && r->db && r->db->enabled)
        r->db->overlapped_cycles += cycles;
}

uint64_t tu_sram_get_overlapped_cycles(const tu_sram_region_t *r) {
    if (r && r->db && r->db->enabled)
        return r->db->overlapped_cycles;
    return 0;
}
