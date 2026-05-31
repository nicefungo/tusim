/*
 * TU CModel — Software Pipelining Controller Implementation (Gap E2)
 *
 * Tile-level DMA/compute overlap orchestration.
 * See pipeline_controller.h for architecture documentation.
 */

#include "pipeline_controller.h"
#include "../memory/double_buffer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

/* ---- Global instance (singleton, matching existing architecture) ---- */
tu_pipeline_controller_t g_tu_pipeline = {0};

/* ---- Configuration defaults ---- */
tu_pipeline_config_t tu_pipeline_config_default(void) {
    return (tu_pipeline_config_t){
        .max_depth            = 2,
        .enable_load_overlap  = true,
        .enable_store_overlap = true,
        .enable_triple_overlap = false,
        .tile_timeout_cycles  = 1000000,
        .model_stalls         = true,
    };
}

/* ---- Initialization ---- */
void tu_pipeline_init(uint32_t depth, const tu_pipeline_config_t *config) {
    if (g_tu_pipeline.initialized) {
        tu_pipeline_destroy();
    }

    if (config) {
        g_tu_pipeline.config = *config;
    } else {
        g_tu_pipeline.config = tu_pipeline_config_default();
    }

    /* Clamp depth */
    if (depth < 1) depth = 1;
    if (depth > 8) depth = 8;
    g_tu_pipeline.config.max_depth = depth;
    g_tu_pipeline.depth = depth;

    /* Allocate pipeline tile slots */
    g_tu_pipeline.slots = calloc(depth, sizeof(tu_pipeline_tile_t));
    if (!g_tu_pipeline.slots) {
        fprintf(stderr, "tu_pipeline_init: allocation failed for %u slots\n", depth);
        g_tu_pipeline.depth = 0;
        return;
    }

    g_tu_pipeline.active_count = 0;
    g_tu_pipeline.next_tile_id = 0;
    g_tu_pipeline.current_cycle = 0;
    g_tu_pipeline.initialized = true;
    g_tu_pipeline.total_tiles_processed = 0;
    g_tu_pipeline.total_stalls = 0;

    /* Zero out statistics */
    g_tu_pipeline.total_compute_cycles = 0;
    g_tu_pipeline.total_load_cycles = 0;
    g_tu_pipeline.total_store_cycles = 0;
    g_tu_pipeline.overlapped_load_cycles = 0;
    g_tu_pipeline.overlapped_store_cycles = 0;
    g_tu_pipeline.stall_cycles = 0;
    g_tu_pipeline.sequential_total = 0;
}

void tu_pipeline_destroy(void) {
    if (g_tu_pipeline.slots) {
        /* Flush all DMA to ensure no in-flight descriptors reference pipeline state.
         * DMA engine owns submitted descriptors; do NOT free them here. */
        tu_dma_flush_all();

        /* Clear slot pointers — DMA engine freed the descriptors */
        for (uint32_t i = 0; i < g_tu_pipeline.depth; i++) {
            g_tu_pipeline.slots[i].load_desc = NULL;
            g_tu_pipeline.slots[i].store_desc = NULL;
        }

        free(g_tu_pipeline.slots);
        g_tu_pipeline.slots = NULL;
    }
    memset(&g_tu_pipeline, 0, sizeof(g_tu_pipeline));
}

void tu_pipeline_reset(void) {
    uint32_t depth = g_tu_pipeline.depth;
    tu_pipeline_config_t config = g_tu_pipeline.config;
    tu_pipeline_destroy();
    g_tu_pipeline.depth = depth;
    g_tu_pipeline.config = config;
    /* Re-allocate; init will zero counters since it calls destroy first */
}

/* ---- Pipeline slot management ---- */

/* Find the next free slot in the pipeline */
static int find_free_slot(tu_pipeline_controller_t *p) {
    for (uint32_t i = 0; i < p->depth; i++) {
        if (p->slots[i].stage == TU_PIPE_STAGE_IDLE ||
            p->slots[i].stage == TU_PIPE_STAGE_DONE) {
            /* Recycle the slot */
            p->slots[i].stage = TU_PIPE_STAGE_IDLE;
            p->slots[i].load_desc = NULL;
            p->slots[i].store_desc = NULL;
            return (int)i;
        }
    }
    return -1;
}

/* ---- Tile submission ---- */

int tu_pipeline_submit_tile(tu_dma_descriptor_t *load_desc,
                             tu_dma_descriptor_t *store_desc,
                             uint64_t compute_cycles,
                             uint32_t cmd_id,
                             tu_sram_region_t *buffer_region) {
    if (!g_tu_pipeline.initialized) {
        /* Auto-initialize with depth=1 (no pipelining) */
        tu_pipeline_config_t cfg = tu_pipeline_config_default();
        cfg.max_depth = 1;
        tu_pipeline_init(1, &cfg);
    }

    tu_pipeline_controller_t *p = &g_tu_pipeline;

    /* Calculate sequential baseline: load + compute + store */
    uint64_t load_cycles = load_desc ? load_desc->total_bytes / TU_DMA_BUS_WIDTH_BYTES : 1;
    uint64_t store_cycles = store_desc ? store_desc->total_bytes / TU_DMA_BUS_WIDTH_BYTES : 1;

    /* Sequential cost (what it would be without overlap) */
    p->sequential_total += load_cycles + compute_cycles + store_cycles;

    /* Find a free slot. If none, tick DMA/advance until one frees up
     * or DMA makes no progress for several ticks (backpressure). */
    int slot_idx = find_free_slot(p);
    if (slot_idx < 0) {
        /* Attempt to advance the pipeline to free a slot.
         * Keep advancing until a slot frees up or we hit the timeout. */
        for (int retry = 0; retry < 1000000 && slot_idx < 0; retry++) {
            p->current_cycle++;
            tu_pipeline_advance();
            slot_idx = find_free_slot(p);
        }
    }
    if (slot_idx < 0) {
        /* Pipeline full — this is backpressure */
        p->total_stalls++;
        p->stall_cycles += compute_cycles;  /* Conservative: count compute as stalled */
        return -1;
    }

    tu_pipeline_tile_t *tile = &p->slots[slot_idx];
    memset(tile, 0, sizeof(*tile));

    tile->tile_id = p->next_tile_id++;
    tile->buffer_region = buffer_region;

    /* Determine initial stage based on pipeline depth and config */
    if (p->depth == 1 || !p->config.enable_load_overlap) {
        /* No pipelining: execute DMA load → compute → DMA store sequentially */
        tile->stage = TU_PIPE_STAGE_DMA_PRELOAD;
    } else {
        /* With pipelining: load goes to shadow buffer, compute runs on active */
        /* First tile may skip preload if data is already in active buffer */
        if (p->total_tiles_processed == 0 && !load_desc) {
            /* First tile: data is already in place (no DMA load needed) */
            tile->stage = TU_PIPE_STAGE_COMPUTE;
            tile->swapped = true;  /* No swap needed */
        } else {
            tile->stage = TU_PIPE_STAGE_DMA_PRELOAD;
        }

        /* If a load is happening and we have double buffering, target shadow buffer */
        if (load_desc && buffer_region && tu_sram_is_double_buffered(buffer_region)) {
            uint8_t *shadow = tu_sram_get_shadow_ptr(buffer_region);
            if (shadow) {
                /* Redirect DMA to shadow buffer */
                load_desc->dst_region = buffer_region;
                if (load_desc->direction == TU_DMA_DIR_HOST_TO_TU) {
                    load_desc->dst_host = shadow;
                }
            }
        }
    }

    tile->load_desc = load_desc;
    tile->store_desc = store_desc;
    tile->compute_cycles = compute_cycles;
    tile->cmd_id = cmd_id;
    tile->cycle_entered = p->current_cycle;
    tile->cycle_expected = p->current_cycle +
        (tile->stage == TU_PIPE_STAGE_COMPUTE ? compute_cycles :
         (tile->stage == TU_PIPE_STAGE_DMA_PRELOAD ? load_cycles : 0));

    /* Submit the load DMA if present */
    if (load_desc) {
        tile->load_signal_id = tu_dma_submit_desc(load_desc);
    }

    /* Track load cycles for overlap accounting */
    if (load_desc) {
        p->total_load_cycles += load_cycles;
    }

    p->active_count++;

    return (int)tile->tile_id;
}

/* ---- Pipeline advancement ---- */

int tu_pipeline_advance(void) {
    if (!g_tu_pipeline.initialized) return 0;

    tu_pipeline_controller_t *p = &g_tu_pipeline;

    /* Tick the DMA engine to process in-flight transfers */
    tu_dma_tick();

    int advances = 0;

    for (uint32_t i = 0; i < p->depth; i++) {
        tu_pipeline_tile_t *tile = &p->slots[i];

        switch (tile->stage) {
        case TU_PIPE_STAGE_IDLE:
        case TU_PIPE_STAGE_DONE:
            break;

        case TU_PIPE_STAGE_DMA_PRELOAD: {
            /* Check if load DMA has completed */
            bool load_done = true;
            if (tile->load_desc) {
                load_done = tile->load_desc->completed;
            }

            if (!load_done) break;

            /* If double-buffered, swap buffers now */
            if (tile->buffer_region && tu_sram_is_double_buffered(tile->buffer_region)) {
                if (p->config.model_stalls && p->active_count > 1) {
                    /* Compute on previous tile may still be running.
                     * Record overlap: load cycles that happened during compute */
                }
                tu_sram_swap_buffers(tile->buffer_region);
                tile->swapped = true;

                /* Notify the shadow buffer tracking */
                if (tile->load_desc) {
                    uint64_t load_cycles = tile->load_desc->total_bytes / TU_DMA_BUS_WIDTH_BYTES;
                    tu_sram_notify_shadow_write(tile->buffer_region,
                                                 tile->load_desc->total_bytes,
                                                 load_cycles);
                }
            }

            /* Advance to compute stage */
            tile->stage = TU_PIPE_STAGE_COMPUTE;
            tile->cycle_entered = p->current_cycle;
            tile->cycle_expected = p->current_cycle + tile->compute_cycles;
            p->total_compute_cycles += tile->compute_cycles;

            /* Model overlap: DMA load was happening while previous tile computed */
            if (p->active_count >= 2 && p->config.enable_load_overlap) {
                uint64_t load_cycles = 0;
                if (tile->load_desc) {
                    load_cycles = tile->load_desc->total_bytes / TU_DMA_BUS_WIDTH_BYTES;
                }
                p->overlapped_load_cycles += load_cycles;
            }

            advances++;
            break;
        }

        case TU_PIPE_STAGE_COMPUTE: {
            /* Compute is considered complete based on cycle model */
            bool compute_done = (p->current_cycle >= tile->cycle_expected);

            if (!compute_done) break;

            /* Advance to store stage */
            tile->stage = TU_PIPE_STAGE_DMA_STORE;
            tile->cycle_entered = p->current_cycle;

            /* Submit the store DMA if present */
            if (tile->store_desc) {
                tile->store_signal_id = tu_dma_submit_desc(tile->store_desc);
            }

            /* Record store cycles for overlap */
            if (tile->store_desc) {
                p->total_store_cycles += tile->store_desc->total_bytes / TU_DMA_BUS_WIDTH_BYTES;
            }

            /* Model overlap: compute was happening while next tile's load may have run */
            advances++;
            break;
        }

        case TU_PIPE_STAGE_DMA_STORE: {
            /* Check if store DMA has completed */
            bool store_done = true;
            if (tile->store_desc) {
                store_done = tile->store_desc->completed;
            }

            if (!store_done) break;

            /* Mark tile as done */
            tile->stage = TU_PIPE_STAGE_DONE;
            p->active_count--;
            p->total_tiles_processed++;

            /* Model overlap: store was happening during next tile's compute */
            if (p->active_count >= 1 && p->config.enable_store_overlap) {
                uint64_t store_cycles = 0;
                if (tile->store_desc) {
                    store_cycles = tile->store_desc->total_bytes / TU_DMA_BUS_WIDTH_BYTES;
                }
                p->overlapped_store_cycles += store_cycles;
            }

            /* Update overlap stats with double-buffer tracking */
            if (tile->buffer_region && tu_sram_is_double_buffered(tile->buffer_region)) {
                uint64_t db_overlap = tu_sram_get_overlapped_cycles(tile->buffer_region);
                tile->cycles_saved = db_overlap;
            }

            /* Clean up descriptors */
            if (tile->load_desc) {
                /* Don't destroy here — DMA engine owns the descriptor */
                tile->load_desc = NULL;
            }
            if (tile->store_desc) {
                tile->store_desc = NULL;
            }

            advances++;
            break;
        }
        }
    }

    return advances;
}

/* ---- Queries ---- */

int tu_pipeline_free_slots(void) {
    if (!g_tu_pipeline.initialized) return 1;
    tu_pipeline_controller_t *p = &g_tu_pipeline;
    int free = 0;
    for (uint32_t i = 0; i < p->depth; i++) {
        if (p->slots[i].stage == TU_PIPE_STAGE_IDLE ||
            p->slots[i].stage == TU_PIPE_STAGE_DONE) {
            free++;
        }
    }
    return free;
}

void tu_pipeline_sync(void) {
    if (!g_tu_pipeline.initialized) return;
    tu_pipeline_controller_t *p = &g_tu_pipeline;

    /* In functional mode, all tiles complete immediately.
     * In estimated/cycle-accurate modes, we wait. */
    uint32_t max_wait = 1000000; /* Safety limit */
    while (p->active_count > 0 && max_wait > 0) {
        p->current_cycle++;
        tu_pipeline_advance();
        max_wait--;
    }
}

bool tu_pipeline_is_idle(void) {
    if (!g_tu_pipeline.initialized) return true;
    return g_tu_pipeline.active_count == 0;
}

/* ---- Statistics ---- */

void tu_pipeline_get_stats(tu_pipeline_stats_t *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));

    if (!g_tu_pipeline.initialized) return;

    tu_pipeline_controller_t *p = &g_tu_pipeline;

    stats->depth                  = p->depth;
    stats->total_tiles            = p->total_tiles_processed;
    stats->total_compute_cycles   = p->total_compute_cycles;
    stats->total_load_cycles      = p->total_load_cycles;
    stats->total_store_cycles     = p->total_store_cycles;
    stats->overlapped_load_cycles = p->overlapped_load_cycles;
    stats->overlapped_store_cycles= p->overlapped_store_cycles;
    stats->stall_cycles           = p->stall_cycles;
    stats->sequential_total       = p->sequential_total;
    stats->total_stalls           = p->total_stalls;

    /* Compute pipelined total: max of non-overlapped paths */
    uint64_t non_overlap_load    = p->total_load_cycles - p->overlapped_load_cycles;
    uint64_t non_overlap_store   = p->total_store_cycles - p->overlapped_store_cycles;

    /* Pipelined execution time ≈ max(compute, load, store) across all tiles
     * plus any non-overlapped components */
    stats->pipelined_total = p->total_compute_cycles + non_overlap_load + non_overlap_store;

    if (stats->sequential_total > 0) {
        stats->speedup = (double)stats->sequential_total / (double)stats->pipelined_total;
    } else {
        stats->speedup = 1.0;
    }

    if (p->total_load_cycles > 0) {
        stats->load_overlap_pct = 100.0 * (double)p->overlapped_load_cycles / (double)p->total_load_cycles;
    } else {
        stats->load_overlap_pct = 0.0;
    }

    if (p->total_store_cycles > 0) {
        stats->store_overlap_pct = 100.0 * (double)p->overlapped_store_cycles / (double)p->total_store_cycles;
    } else {
        stats->store_overlap_pct = 0.0;
    }
}

uint64_t tu_pipeline_get_saved_cycles(void) {
    if (!g_tu_pipeline.initialized) return 0;
    return g_tu_pipeline.overlapped_load_cycles + g_tu_pipeline.overlapped_store_cycles;
}

void tu_pipeline_print_stats(void) {
    tu_pipeline_stats_t s;
    tu_pipeline_get_stats(&s);

    fprintf(stderr, "\n═══════════════════════════════════════════\n");
    fprintf(stderr, "  Pipeline Controller Statistics\n");
    fprintf(stderr, "═══════════════════════════════════════════\n");
    fprintf(stderr, "  Depth:                  %u\n", s.depth);
    fprintf(stderr, "  Total tiles:            %u\n", s.total_tiles);
    fprintf(stderr, "  Total stalls:           %u\n", s.total_stalls);
    fprintf(stderr, "  ───────────────────────────────────────\n");
    fprintf(stderr, "  DMA load cycles:        %lu\n", (unsigned long)s.total_load_cycles);
    fprintf(stderr, "  Compute cycles:         %lu\n", (unsigned long)s.total_compute_cycles);
    fprintf(stderr, "  DMA store cycles:       %lu\n", (unsigned long)s.total_store_cycles);
    fprintf(stderr, "  ───────────────────────────────────────\n");
    fprintf(stderr, "  Overlapped load:        %lu (%.1f%%)\n",
            (unsigned long)s.overlapped_load_cycles, s.load_overlap_pct);
    fprintf(stderr, "  Overlapped store:       %lu (%.1f%%)\n",
            (unsigned long)s.overlapped_store_cycles, s.store_overlap_pct);
    fprintf(stderr, "  Stall cycles:           %lu\n", (unsigned long)s.stall_cycles);
    fprintf(stderr, "  ───────────────────────────────────────\n");
    fprintf(stderr, "  Sequential total:       %lu\n", (unsigned long)s.sequential_total);
    fprintf(stderr, "  Pipelined total:        %lu\n", (unsigned long)s.pipelined_total);
    fprintf(stderr, "  Speedup:                %.2fx\n", s.speedup);
    fprintf(stderr, "  Cycles saved:           %lu\n",
            (unsigned long)(s.sequential_total - s.pipelined_total));
    fprintf(stderr, "═══════════════════════════════════════════\n\n");
}
