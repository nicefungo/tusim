/*
 * TU CModel — Cycle-Accurate Timing Model Implementation (Gap P2.5)
 * ===================================================================
 *
 * Production-grade cycle-level simulator.
 *
 * Models:
 *   1. Systolic pipeline with hazard detection
 *   2. Multi-bank SRAM with bandwidth constraints and conflict detection
 *   3. DRAM with row buffer hit/miss (open-page policy)
 *   4. DMA bus arbitration between channels
 *
 * The model is configurable at three fidelity levels:
 *   FUNCTIONAL:      No cycle accounting (returns 0)
 *   ESTIMATED:       Simple fill + compute + drain model
 *   CYCLE_ACCURATE:  Full pipeline hazards + bank conflicts + DRAM row buffer
 */

#include "cycle_model.h"
#include "../tu_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Pipeline Tracker
 * ================================================================ */

void tu_cycle_pipeline_init(tu_pipeline_tracker_t *pt, uint32_t max_in_flight) {
    memset(pt, 0, sizeof(*pt));
    pt->num_entries = max_in_flight > 0 ? max_in_flight : TU_PE_PIPELINE_DEPTH;
    pt->entries = calloc(pt->num_entries, sizeof(tu_pipeline_entry_t));
}

uint64_t tu_cycle_pipeline_issue(tu_pipeline_tracker_t *pt,
                            uint16_t m_start, uint16_t m_count,
                            uint16_t n_start, uint16_t n_count,
                            uint16_t k_start, uint16_t k_count,
                            const uint32_t *src_regs, uint32_t num_src,
                            const uint32_t *dst_regs, uint32_t num_dst,
                            uint64_t current_cycle)
{
    if (!pt || !pt->entries) return 0;

    uint64_t stall_cycles = 0;

    /* Check for RAW hazards: does any in-flight entry write a register
     * that this tile needs to read? */
    for (uint32_t i = 0; i < pt->num_entries; i++) {
        tu_pipeline_entry_t *ent = &pt->entries[i];
        if (!ent->active) continue;

        for (uint32_t s = 0; s < num_src && s < 4; s++) {
            for (uint32_t d = 0; d < num_dst && d < 4; d++) {
                if (src_regs[s] == ent->reg_outputs[d] && src_regs[s] != 0) {
                    /* RAW: need to stall until ent completes */
                    uint64_t remaining = 0;
                    if (ent->complete_cycle > current_cycle) {
                        remaining = ent->complete_cycle - current_cycle;
                    } else if (ent->complete_cycle == 0) {
                        /* Tile not yet completed — assume worst-case stall */
                        remaining = TU_PE_PIPELINE_DEPTH * 16 + 64;  /* fill + compute */
                    }
                    if (remaining > stall_cycles) stall_cycles = remaining;
                }
            }
        }
    }

    /* Check for WAW hazards: does any in-flight entry also write
     * a register this tile wants to write? */
    for (uint32_t i = 0; i < pt->num_entries; i++) {
        tu_pipeline_entry_t *ent = &pt->entries[i];
        if (!ent->active) continue;

        for (uint32_t d1 = 0; d1 < num_dst && d1 < 4; d1++) {
            for (uint32_t d2 = 0; d2 < 4; d2++) {
                if (dst_regs[d1] == ent->reg_outputs[d2] && dst_regs[d1] != 0) {
                    uint64_t remaining = 0;
                    if (ent->complete_cycle > current_cycle) {
                        remaining = ent->complete_cycle - current_cycle;
                    } else if (ent->complete_cycle == 0) {
                        remaining = TU_PE_PIPELINE_DEPTH * 16 + 64;
                    }
                    if (remaining > stall_cycles) stall_cycles = remaining;
                }
            }
        }
    }

    /* Advance cycle counter by stall cycles */
    uint64_t issue_cycle = current_cycle + stall_cycles;

    /* Allocate pipeline entry */
    tu_pipeline_entry_t *ent = &pt->entries[pt->head];
    ent->issue_cycle    = issue_cycle;
    ent->complete_cycle = 0;
    ent->stage          = 0;
    ent->m_start = m_start; ent->m_count = m_count;
    ent->n_start = n_start; ent->n_count = n_count;
    ent->k_start = k_start; ent->k_count = k_count;
    ent->active         = true;

    for (uint32_t i = 0; i < num_src && i < 4; i++) ent->reg_deps[i] = src_regs[i];
    for (uint32_t i = 0; i < num_dst && i < 4; i++) ent->reg_outputs[i] = dst_regs[i];

    pt->head = (pt->head + 1) % pt->num_entries;
    pt->total_issues++;
    pt->total_stall_cycles += stall_cycles;

    return stall_cycles;
}

uint64_t tu_cycle_pipeline_complete(tu_pipeline_tracker_t *pt, uint64_t current_cycle) {
    if (!pt || !pt->entries) return 0;

    tu_pipeline_entry_t *ent = &pt->entries[pt->tail];
    if (!ent->active) return 0;

    uint64_t elapsed = current_cycle - ent->issue_cycle;
    ent->complete_cycle = current_cycle;
    ent->active = false;

    pt->tail = (pt->tail + 1) % pt->num_entries;
    pt->total_completions++;
    return elapsed;
}

float tu_cycle_pipeline_utilization(const tu_pipeline_tracker_t *pt) {
    if (!pt || pt->num_entries == 0) return 0.0f;
    int active = 0;
    for (uint32_t i = 0; i < pt->num_entries; i++) {
        if (pt->entries[i].active) active++;
    }
    return (float)active / (float)pt->num_entries;
}

void tu_cycle_pipeline_destroy(tu_pipeline_tracker_t *pt) {
    if (!pt) return;
    free(pt->entries);
    memset(pt, 0, sizeof(*pt));
}

/* ================================================================
 * Bank Conflict Model
 * ================================================================ */

void tu_bank_model_init(tu_bank_model_t *bm, uint32_t num_banks,
                         uint32_t bank_width_bytes, uint32_t refill_window,
                         uint32_t stall_penalty, uint32_t max_accesses)
{
    memset(bm, 0, sizeof(*bm));
    bm->num_banks = num_banks;
    bm->bank_width_bytes = bank_width_bytes;
    bm->refill_window_cycles = refill_window;
    bm->stall_penalty = stall_penalty;
    bm->max_accesses_per_cycle = max_accesses;

    bm->banks = calloc(num_banks, sizeof(tu_bank_state_t));
    for (uint32_t i = 0; i < num_banks; i++) {
        bm->banks[i].bank_id = i;
        bm->banks[i].max_words_per_cycle = max_accesses;
        bm->banks[i].words_available = max_accesses;
        bm->banks[i].refill_cycle = refill_window;
    }
}

uint32_t tu_bank_model_access(tu_bank_model_t *bm, uint32_t bank_id,
                               bool is_write, uint32_t word_count,
                               uint64_t cycle)
{
    if (!bm || bank_id >= bm->num_banks) return 0;

    tu_bank_state_t *bank = &bm->banks[bank_id];

    /* Check if refill is needed */
    if (cycle >= bank->refill_cycle) {
        bank->words_available = bank->max_words_per_cycle;
        bank->refill_cycle = cycle + bm->refill_window_cycles;
    }

    uint32_t stall = 0;
    if (word_count > bank->words_available) {
        /* Not enough bandwidth — stall */
        uint32_t shortfall = word_count - bank->words_available;
        stall = shortfall * bm->stall_penalty;

        if (is_write) bank->write_stalls += shortfall;
        else          bank->read_stalls += shortfall;

        /* Consume what's available and stall for rest */
        bank->words_available = 0;
    } else {
        bank->words_available -= word_count;
    }

    if (is_write) bank->total_writes += word_count;
    else          bank->total_reads += word_count;

    /* Track concurrent access for conflict detection */
    if (word_count > 0 && bank->words_available < bm->max_accesses_per_cycle) {
        bank->conflict_count++;
    }

    return stall;
}

void tu_bank_model_tick(tu_bank_model_t *bm, uint64_t cycle) {
    if (!bm) return;
    bm->current_cycle = cycle;
    for (uint32_t i = 0; i < bm->num_banks; i++) {
        if (cycle >= bm->banks[i].refill_cycle) {
            bm->banks[i].words_available = bm->banks[i].max_words_per_cycle;
            bm->banks[i].refill_cycle = cycle + bm->refill_window_cycles;
        }
    }
}

void tu_bank_model_get_stats(const tu_bank_model_t *bm,
                              uint64_t *total_reads, uint64_t *total_writes,
                              uint64_t *total_stalls, uint64_t *total_conflicts,
                              double *avg_utilization)
{
    if (!bm) return;
    uint64_t r = 0, w = 0, s = 0, c = 0;
    for (uint32_t i = 0; i < bm->num_banks; i++) {
        r += bm->banks[i].total_reads;
        w += bm->banks[i].total_writes;
        s += bm->banks[i].read_stalls + bm->banks[i].write_stalls;
        c += bm->banks[i].conflict_count;
    }
    if (total_reads)  *total_reads  = r;
    if (total_writes) *total_writes = w;
    if (total_stalls) *total_stalls = s;
    if (total_conflicts) *total_conflicts = c;
    if (avg_utilization) {
        uint64_t total_cap = (uint64_t)bm->num_banks * bm->max_accesses_per_cycle;
        *avg_utilization = total_cap > 0 ? (double)(r + w) / (double)total_cap : 0.0;
    }
}

void tu_bank_model_destroy(tu_bank_model_t *bm) {
    if (!bm) return;
    free(bm->banks);
    memset(bm, 0, sizeof(*bm));
}

/* ================================================================
 * DRAM Row Buffer Model
 * ================================================================ */

/* Standard JEDEC timing presets */
static void set_ideal_timing(tu_dram_timing_t *t, uint32_t bw) {
    memset(t, 0, sizeof(*t));
    t->bus_width_bytes = bw;
    t->tCL = t->tCWL = 1;
    t->tBL = bw;
    t->num_banks = 1;
    t->num_bank_groups = 1;
}

static void set_hbm2_timing(tu_dram_timing_t *t, double mhz, uint32_t bw) {
    memset(t, 0, sizeof(*t));
    double ns_per_cycle = 1000.0 / mhz;
    t->tRCD    = (uint32_t)(14.0 / ns_per_cycle);  /* ~14ns */
    t->tRP     = (uint32_t)(14.0 / ns_per_cycle);
    t->tRAS    = (uint32_t)(34.0 / ns_per_cycle);
    t->tCL     = (uint32_t)(14.0 / ns_per_cycle);
    t->tCWL    = (uint32_t)(7.0 / ns_per_cycle);
    t->tWR     = (uint32_t)(12.0 / ns_per_cycle);
    t->tRTP    = (uint32_t)(5.0 / ns_per_cycle);
    t->tBL     = 4;
    t->bus_width_bytes = bw;
    t->num_banks = 8;
    t->num_bank_groups = 1;
    t->rows_per_bank = 32768;
    t->columns_per_row = 256;
    t->freq_mhz = mhz;
}

void tu_dram_channel_init(tu_dram_channel_t *ch, uint32_t dram_type,
                           double freq_mhz, uint32_t bus_width_bytes)
{
    memset(ch, 0, sizeof(*ch));
    tu_dram_timing_preset(&ch->timing, dram_type, freq_mhz, bus_width_bytes);
    ch->banks = calloc(ch->timing.num_banks, sizeof(tu_dram_bank_state_t));
    for (uint32_t i = 0; i < ch->timing.num_banks; i++) {
        ch->banks[i].open_row = UINT32_MAX;  /* No row open */
        ch->banks[i].num_rows = ch->timing.rows_per_bank;
    }
}

void tu_dram_timing_preset(tu_dram_timing_t *t, uint32_t dram_type,
                            double freq_mhz, uint32_t bus_width_bytes)
{
    switch (dram_type) {
    case TU_DRAM_IDEAL:  set_ideal_timing(t, bus_width_bytes); break;
    case TU_DRAM_HBM2:   set_hbm2_timing(t, freq_mhz, bus_width_bytes); break;
    case TU_DRAM_HBM2E:  set_hbm2_timing(t, freq_mhz, bus_width_bytes); t->tCL=(uint32_t)(12.0/(1000.0/freq_mhz)); break;
    case TU_DRAM_HBM3:   set_hbm2_timing(t, freq_mhz, bus_width_bytes); t->num_banks=16; t->tCL=(uint32_t)(10.0/(1000.0/freq_mhz)); break;
    case TU_DRAM_DDR4:
        set_hbm2_timing(t, freq_mhz, bus_width_bytes);
        t->num_banks = 16;
        t->num_bank_groups = 4;
        t->tCCD = 4;
        t->tBL = 8;
        break;
    case TU_DRAM_DDR5:
    case TU_DRAM_LPDDR5:
        set_hbm2_timing(t, freq_mhz, bus_width_bytes);
        t->num_banks = 32;
        t->num_bank_groups = 8;
        t->tBL = 16;
        break;
    default:
        set_ideal_timing(t, bus_width_bytes);
        break;
    }
}

void tu_dram_decode_address(const tu_dram_channel_t *ch, uint64_t address,
                             uint32_t *bank, uint32_t *row, uint32_t *column)
{
    if (!ch) { if(bank)*bank=0; if(row)*row=0; if(column)*column=0; return; }

    /* Simple interleaving: low bits = column, mid bits = bank, high bits = row */
    const tu_dram_timing_t *t = &ch->timing;
    uint32_t col_bits = 10;  /* 1024 columns per row */
    uint32_t bank_bits = 0;
    uint32_t nb = t->num_banks;
    while (nb >>= 1) bank_bits++;

    uint32_t col_mask = (1u << col_bits) - 1;
    uint32_t bank_mask = ((1u << bank_bits) - 1) << col_bits;

    if (column) *column = (uint32_t)(address & col_mask);
    if (bank)   *bank   = (uint32_t)((address & bank_mask) >> col_bits);
    if (row)    *row    = (uint32_t)(address >> (col_bits + bank_bits));
}

uint64_t tu_dram_access(tu_dram_channel_t *ch, uint64_t address,
                         bool is_write, uint32_t bytes, uint64_t cycle)
{
    if (!ch) return 0;

    uint32_t bank_id, row, col;
    tu_dram_decode_address(ch, address, &bank_id, &row, &col);
    if (bank_id >= ch->timing.num_banks) bank_id = 0;

    tu_dram_bank_state_t *bank = &ch->banks[bank_id];
    const tu_dram_timing_t *t = &ch->timing;
    uint64_t latency = 0;

    /* Determine row buffer state */
    if (bank->open_row == row) {
        /* Row hit — just CAS latency + data transfer */
        latency = (is_write ? t->tCWL : t->tCL);
        ch->total_row_hits++;
        bank->row_hits++;
    } else if (bank->open_row == UINT32_MAX) {
        /* No row open — activate + CAS */
        latency = t->tRCD + (is_write ? t->tCWL : t->tCL);
        ch->total_row_misses++;
        bank->row_misses++;
        bank->open_row = row;
    } else {
        /* Row miss — precharge + activate + CAS */
        latency = t->tRP + t->tRCD + (is_write ? t->tCWL : t->tCL);
        ch->total_row_misses++;
        bank->row_misses++;
        bank->open_row = row;
    }

    /* Data transfer cycles: bytes / (bus_width * tBL) bursts */
    uint32_t burst_bytes = t->bus_width_bytes * t->tBL;
    uint32_t bursts = (bytes + burst_bytes - 1) / burst_bytes;
    latency += bursts * t->tCCD;

    ch->total_accesses++;
    ch->total_bytes_transferred += bytes;
    ch->total_cycles_stalled += latency;
    bank->access_count++;

    /* Advance cycle */
    ch->current_cycle = cycle + latency;

    return latency;
}

void tu_cycle_dram_tick(tu_dram_channel_t *ch, uint64_t cycle) {
    if (!ch) return;
    ch->current_cycle = cycle;
    for (uint32_t i = 0; i < ch->timing.num_banks; i++) {
        if (ch->banks[i].open_row != UINT32_MAX) {
            ch->banks[i].total_cycles_active++;
        } else {
            ch->banks[i].total_cycles_idle++;
        }
    }
}

void tu_cycle_dram_get_stats(const tu_dram_channel_t *ch,
                        uint64_t *accesses, uint64_t *row_hits,
                        uint64_t *row_misses, double *hit_rate,
                        double *effective_bw_gbps, uint64_t *stall_cycles)
{
    if (!ch) return;
    if (accesses)  *accesses  = ch->total_accesses;
    if (row_hits)  *row_hits  = ch->total_row_hits;
    if (row_misses) *row_misses = ch->total_row_misses;
    if (hit_rate && ch->total_accesses > 0)
        *hit_rate = (double)ch->total_row_hits / (double)ch->total_accesses;
    if (effective_bw_gbps && ch->total_cycles_stalled > 0)
        *effective_bw_gbps = (double)ch->total_bytes_transferred /
                             (double)ch->total_cycles_stalled * ch->timing.freq_mhz / 1000.0;
    if (stall_cycles) *stall_cycles = ch->total_cycles_stalled;
}

void tu_cycle_dram_destroy(tu_dram_channel_t *ch) {
    if (!ch) return;
    free(ch->banks);
    memset(ch, 0, sizeof(*ch));
}

/* ================================================================
 * DRAM Presets
 * ================================================================ */

void tu_dram_preset_ideal(tu_dram_timing_t *t, uint32_t bw)  { set_ideal_timing(t, bw); }
void tu_dram_preset_hbm2(tu_dram_timing_t *t, uint32_t bw)   { set_hbm2_timing(t, 1000.0, bw); }
void tu_dram_preset_hbm2e(tu_dram_timing_t *t, uint32_t bw)  { set_hbm2_timing(t, 1800.0, bw); }
void tu_dram_preset_hbm3(tu_dram_timing_t *t, uint32_t bw)   { set_hbm2_timing(t, 3200.0, bw); t->num_banks = 16; }
void tu_dram_preset_ddr4(tu_dram_timing_t *t, uint32_t bw) {
    set_hbm2_timing(t, 1600.0, bw);
    t->num_banks = 16; t->num_bank_groups = 4; t->tCCD = 4; t->tBL = 8;
}
void tu_dram_preset_ddr5(tu_dram_timing_t *t, uint32_t bw) {
    set_hbm2_timing(t, 3200.0, bw);
    t->num_banks = 32; t->num_bank_groups = 8; t->tBL = 16;
}
void tu_dram_preset_lpddr5(tu_dram_timing_t *t, uint32_t bw) {
    set_hbm2_timing(t, 3200.0, bw);
    t->num_banks = 16; t->num_bank_groups = 4; t->tBL = 8;
}

/* ================================================================
 * Cycle Model Integration
 * ================================================================ */

tu_cycle_model_t *tu_cycle_model_create(uint32_t mode, tu_perf_counters_t *perf) {
    tu_cycle_model_t *cm = calloc(1, sizeof(*cm));
    if (!cm) return NULL;

    cm->mode = mode;
    cm->perf = perf;

    if (mode >= TU_CYCLE_MODEL_CYCLE_ACCURATE) {
        /* Create pipeline tracker */
        cm->pipeline = calloc(1, sizeof(tu_pipeline_tracker_t));
        tu_cycle_pipeline_init(cm->pipeline, TU_PE_PIPELINE_DEPTH * 4);

        /* Create bank model */
        cm->bank_model = calloc(1, sizeof(tu_bank_model_t));
        tu_bank_model_init(cm->bank_model,
            TU_SRAM_BANKS, TU_SRAM_BANK_WIDTH,
            TU_SRAM_BW_WINDOW_CYCLES, TU_SRAM_BW_STALL_PENALTY,
            TU_SRAM_WORDS_PER_CYCLE);

        /* Create DRAM channel */
        cm->dram_channel = calloc(1, sizeof(tu_dram_channel_t));
        uint32_t type = TU_DRAM_TYPE;
        double freq_mhz = type == TU_DRAM_IDEAL ? 1000.0 :
                          type == TU_DRAM_HBM2 ? 1000.0 :
                          type == TU_DRAM_HBM2E ? 1800.0 :
                          type == TU_DRAM_HBM3 ? 3200.0 : 1600.0;
        tu_dram_channel_init(cm->dram_channel, type, freq_mhz, TU_DMA_BUS_WIDTH_BYTES);
    }

    cm->dma_channels = TU_DMA_CHANNELS;
    return cm;
}

void tu_cycle_model_reset(tu_cycle_model_t *cm) {
    if (!cm) return;
    cm->current_cycle = 0;
    memset(cm->dma_bus_cycles, 0, sizeof(cm->dma_bus_cycles));
    cm->dma_bus_stall_cycles = 0;

    if (cm->pipeline) {
        tu_cycle_pipeline_destroy(cm->pipeline);
        tu_cycle_pipeline_init(cm->pipeline, TU_PE_PIPELINE_DEPTH * 4);
    }
    if (cm->bank_model) {
        tu_bank_model_destroy(cm->bank_model);
        tu_bank_model_init(cm->bank_model,
            TU_SRAM_BANKS, TU_SRAM_BANK_WIDTH,
            TU_SRAM_BW_WINDOW_CYCLES, TU_SRAM_BW_STALL_PENALTY,
            TU_SRAM_WORDS_PER_CYCLE);
    }
    if (cm->dram_channel) {
        tu_cycle_dram_destroy(cm->dram_channel);
        uint32_t type = TU_DRAM_TYPE;
        double mhz = type == TU_DRAM_IDEAL ? 1000.0 : 1000.0;
        tu_dram_channel_init(cm->dram_channel, type, mhz, TU_DMA_BUS_WIDTH_BYTES);
    }
}

uint64_t tu_cycle_model_execute_tile(
    tu_cycle_model_t *cm,
    uint16_t m_start, uint16_t m_count,
    uint16_t n_start, uint16_t n_count,
    uint16_t k_start, uint16_t k_count,
    uint32_t w_sram_addr, uint32_t a_sram_addr, uint32_t o_sram_addr)
{
    if (!cm) return 0;

    switch (cm->mode) {
    case TU_CYCLE_MODEL_FUNCTIONAL:
        return 0;  /* No cycle accounting */

    case TU_CYCLE_MODEL_ESTIMATED: {
        /* Simple model: pipeline fill + compute + drain */
        uint64_t fill = TU_PE_PIPELINE_DEPTH * (uint64_t)n_count;
        uint64_t compute = (uint64_t)k_count;
        uint64_t drain = TU_PE_PIPELINE_DEPTH * (uint64_t)m_count;
        uint64_t total = fill + compute + drain;
        cm->current_cycle += total;
        return total;
    }

    case TU_CYCLE_MODEL_CYCLE_ACCURATE: {
        uint64_t start_cycle = cm->current_cycle;
        uint64_t total_stall = 0;

        /* 1. Instruction decode (1 cycle) */
        cm->current_cycle += 1;

        /* 2. Check pipeline hazards */
        uint32_t src_regs[] = {w_sram_addr, a_sram_addr, 0, 0};
        uint32_t dst_regs[] = {o_sram_addr, 0, 0, 0};
        if (cm->pipeline) {
            uint64_t hazard_stall = tu_cycle_pipeline_issue(
                cm->pipeline, m_start, m_count, n_start, n_count,
                k_start, k_count,
                src_regs, 2, dst_regs, 1,
                cm->current_cycle);
            cm->current_cycle += hazard_stall;
            total_stall += hazard_stall;
        }

        /* 3. SRAM read: weight and activation banks */
        uint32_t w_bank = w_sram_addr % cm->bank_model->num_banks;
        uint32_t a_bank = a_sram_addr % cm->bank_model->num_banks;

        /* Weight read */
        uint32_t w_words = (m_count * k_count * 2 + cm->bank_model->bank_width_bytes - 1)
                           / cm->bank_model->bank_width_bytes;
        uint32_t w_stall = tu_bank_model_access(cm->bank_model, w_bank, false,
                                                  w_words, cm->current_cycle);
        cm->current_cycle += w_stall;
        total_stall += w_stall;

        /* Activation read */
        uint32_t a_words = (k_count * n_count * 2 + cm->bank_model->bank_width_bytes - 1)
                           / cm->bank_model->bank_width_bytes;
        uint32_t a_stall = tu_bank_model_access(cm->bank_model, a_bank, false,
                                                  a_words, cm->current_cycle);
        cm->current_cycle += a_stall;
        total_stall += a_stall;

        /* 4. MAC computation: k_count cycles (1 MAC per cycle per PE) */
        cm->current_cycle += k_count;

        /* 5. Writeback: accumulate results */
        uint32_t o_bank = o_sram_addr % cm->bank_model->num_banks;
        uint32_t o_words = (m_count * n_count * 4 + cm->bank_model->bank_width_bytes - 1)
                           / cm->bank_model->bank_width_bytes;
        uint32_t o_stall = tu_bank_model_access(cm->bank_model, o_bank, true,
                                                  o_words, cm->current_cycle);
        cm->current_cycle += o_stall;
        total_stall += o_stall;

        /* 6. Complete pipeline entry */
        if (cm->pipeline) {
            tu_cycle_pipeline_complete(cm->pipeline, cm->current_cycle);
        }

        /* 7. Advance bank model refill */
        tu_bank_model_tick(cm->bank_model, cm->current_cycle);

        /* 8. Record in performance counters */
        if (cm->perf) {
            uint64_t elapsed = cm->current_cycle - start_cycle;
            uint64_t active = elapsed - total_stall;
            uint64_t macs = (uint64_t)m_count * n_count * k_count;
            tu_perf_compute_record_mma(cm->perf, macs, m_count, n_count, k_count,
                                        1, (m_count < TU_PE_ROWS) ? 1 : 0,
                                        active, total_stall, 0, 0);
        }

        return cm->current_cycle - start_cycle;
    }

    default:
        return 0;
    }
}

uint64_t tu_cycle_model_dma_transfer(
    tu_cycle_model_t *cm,
    uint8_t channel, uint32_t bytes,
    bool is_read, uint64_t dram_addr, uint32_t sram_bank)
{
    if (!cm || bytes == 0) return 0;

    switch (cm->mode) {
    case TU_CYCLE_MODEL_FUNCTIONAL:
        return 0;

    case TU_CYCLE_MODEL_ESTIMATED: {
        uint64_t cycles = (bytes + TU_DMA_BUS_WIDTH_BYTES - 1) / TU_DMA_BUS_WIDTH_BYTES;
        cycles += TU_LATENCY_DRAM_READ;
        cm->current_cycle += cycles;
        return cycles;
    }

    case TU_CYCLE_MODEL_CYCLE_ACCURATE: {
        uint64_t start = cm->current_cycle;

        /* 1. Bus arbitration */
        uint64_t arb_stall = tu_cycle_model_dma_arbitrate(cm, channel, 1);
        cm->current_cycle += arb_stall;

        /* 2. DRAM access */
        if (cm->dram_channel && dram_addr != 0) {
            uint64_t dram_lat = tu_dram_access(cm->dram_channel, dram_addr,
                                                is_read, bytes, cm->current_cycle);
            cm->current_cycle += dram_lat;
        } else {
            /* Direct SRAM access */
            uint64_t xfer_cycles = (bytes + TU_DMA_BUS_WIDTH_BYTES - 1) / TU_DMA_BUS_WIDTH_BYTES;
            cm->current_cycle += xfer_cycles;
        }

        /* 3. SRAM bank access */
        if (cm->bank_model && sram_bank < cm->bank_model->num_banks) {
            uint32_t words = (bytes + cm->bank_model->bank_width_bytes - 1)
                             / cm->bank_model->bank_width_bytes;
            uint32_t sram_stall = tu_bank_model_access(cm->bank_model, sram_bank,
                                                         !is_read, words,
                                                         cm->current_cycle);
            cm->current_cycle += sram_stall;
        }

        /* 4. Record DMA bus usage */
        if (channel < 8) {
            cm->dma_bus_cycles[channel] += cm->current_cycle - start;
        }

        /* 5. Update perf counters */
        if (cm->perf) {
            tu_perf_dma_record_read(cm->perf, bytes,
                                     cm->current_cycle - start, arb_stall,
                                     channel, 0);
        }

        return cm->current_cycle - start;
    }

    default:
        return 0;
    }
}

uint64_t tu_cycle_model_dma_arbitrate(
    tu_cycle_model_t *cm, uint8_t channel, uint64_t transfer_cycles)
{
    (void)channel;
    (void)transfer_cycles;
    if (!cm || cm->mode < TU_CYCLE_MODEL_CYCLE_ACCURATE) return 0;
    if (cm->dma_channels <= 1) return 0;

    /* Simple round-robin: count active channels at current cycle */
    uint32_t active = 0;
    for (uint32_t i = 0; i < cm->dma_channels; i++) {
        if (cm->dma_bus_cycles[i] > cm->current_cycle) active++;
    }

    if (active > 1) {
        /* Bus contention: add arbitration delay */
        uint64_t arbitration_cycles = active;  /* 1 cycle per active channel */
        cm->dma_bus_stall_cycles += arbitration_cycles;
        return arbitration_cycles;
    }
    return 0;
}

void tu_cycle_model_advance(tu_cycle_model_t *cm, uint64_t cycles) {
    if (!cm) return;
    cm->current_cycle += cycles;
    if (cm->perf) tu_perf_tick(cm->perf, cycles);
}

void tu_cycle_model_report(const tu_cycle_model_t *cm) {
    if (!cm) return;

    printf("\n=== TU Cycle Model Report ===\n");
    printf("Model fidelity: %s\n",
           cm->mode == TU_CYCLE_MODEL_FUNCTIONAL ? "FUNCTIONAL" :
           cm->mode == TU_CYCLE_MODEL_ESTIMATED ? "ESTIMATED" :
           cm->mode == TU_CYCLE_MODEL_CYCLE_ACCURATE ? "CYCLE_ACCURATE" : "UNKNOWN");
    printf("Total cycles: %lu\n", (unsigned long)cm->current_cycle);

    if (cm->pipeline) {
        printf("\n-- Pipeline --\n");
        printf("  Issues:           %lu\n", (unsigned long)cm->pipeline->total_issues);
        printf("  Completions:      %lu\n", (unsigned long)cm->pipeline->total_completions);
        printf("  Hazard stalls:    %lu cycles\n", (unsigned long)cm->pipeline->total_stall_cycles);
        printf("  Utilization:      %.1f%%\n", tu_cycle_pipeline_utilization(cm->pipeline) * 100.0f);
    }

    if (cm->bank_model) {
        printf("\n-- SRAM Banks --\n");
        uint64_t reads, writes, stalls, conf;
        double util;
        tu_bank_model_get_stats(cm->bank_model, &reads, &writes, &stalls, &conf, &util);
        printf("  Reads:            %lu\n", (unsigned long)reads);
        printf("  Writes:           %lu\n", (unsigned long)writes);
        printf("  Bank stalls:      %lu cycles\n", (unsigned long)stalls);
        printf("  Bank conflicts:   %lu\n", (unsigned long)conf);
        printf("  Avg utilization:  %.1f%%\n", util * 100.0);
    }

    if (cm->dram_channel) {
        printf("\n-- DRAM --\n");
        uint64_t acc, hits, misses, stall;
        double hr, bw;
        tu_cycle_dram_get_stats(cm->dram_channel, &acc, &hits, &misses, &hr, &bw, &stall);
        printf("  Accesses:         %lu\n", (unsigned long)acc);
        printf("  Row hits:         %lu (%.1f%%)\n", (unsigned long)hits, hr * 100.0);
        printf("  Row misses:       %lu\n", (unsigned long)misses);
        printf("  Stall cycles:     %lu\n", (unsigned long)stall);
        printf("  Data transferred: %lu bytes\n",
               (unsigned long)cm->dram_channel->total_bytes_transferred);
        if (bw > 0) printf("  Eff. bandwidth:   %.1f GB/s\n", bw);
    }

    printf("\n-- DMA Bus --\n");
    printf("  Bus stalls:       %lu cycles\n", (unsigned long)cm->dma_bus_stall_cycles);
    for (uint32_t i = 0; i < cm->dma_channels; i++) {
        if (cm->dma_bus_cycles[i] > 0)
            printf("  Channel %u:       %lu cycles\n", i,
                   (unsigned long)cm->dma_bus_cycles[i]);
    }
    printf("===============================\n");
}

void tu_cycle_model_destroy(tu_cycle_model_t *cm) {
    if (!cm) return;
    tu_cycle_pipeline_destroy(cm->pipeline);
    free(cm->pipeline);
    tu_bank_model_destroy(cm->bank_model);
    free(cm->bank_model);
    tu_cycle_dram_destroy(cm->dram_channel);
    free(cm->dram_channel);
    free(cm);
}
