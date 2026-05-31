/*
 * TU CModel — Performance Counter Implementation
 * ===============================================
 *
 * Comprehensive performance monitoring with cycle-accurate counting,
 * utilization tracking, stall analysis, and energy estimation.
 *
 * Gap: E4 (Power/Energy Model), P2.5 (Cycle-Accurate Model) foundation
 */
#include "performance_counters.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ================================================================
 * Lifecycle
 * ================================================================ */

void tu_perf_init(tu_perf_counters_t *c, double clock_freq_mhz) {
    memset(c, 0, sizeof(*c));
    c->enabled          = true;
    c->clock_freq_mhz   = clock_freq_mhz > 0.0 ? clock_freq_mhz : 1000.0;

    /* Default energy parameters (45nm-like, conservative) */
    c->power.pj_per_mac          = 1.0;   /* 1 pJ per MAC */
    c->power.pj_per_sram_read    = 0.5;   /* 0.5 pJ per SRAM read */
    c->power.pj_per_sram_write   = 0.5;   /* 0.5 pJ per SRAM write */
    c->power.pj_per_dram_access  = 20.0;  /* 20 pJ per DRAM access */
    c->power.pj_per_dma_byte     = 0.05;  /* 0.05 pJ per DMA byte */
    c->power.pj_leakage_per_cycle= 0.001; /* 1 fJ leakage per cycle */
    c->power.power_modeling_enabled = true;
}

void tu_perf_reset(tu_perf_counters_t *c) {
    double freq = c->clock_freq_mhz;
    tu_power_counters_t saved_power = c->power;
    tu_perf_init(c, freq);
    c->power = saved_power;  /* Preserve energy parameters */
}

void tu_perf_set_enabled(tu_perf_counters_t *c, bool enabled) {
    c->enabled = enabled;
}

/* ================================================================
 * Cycle Management
 * ================================================================ */

void tu_perf_tick(tu_perf_counters_t *c, uint64_t cycles) {
    if (!c->enabled) return;
    c->total_cycles += cycles;
    c->wall_clock_ns = (uint64_t)((double)c->total_cycles / c->clock_freq_mhz * 1000.0);

    /* Update leakage energy */
    if (c->power.power_modeling_enabled) {
        c->power.energy_leakage_pj += c->power.pj_leakage_per_cycle * (double)cycles;
    }
}

uint64_t tu_perf_get_cycle(const tu_perf_counters_t *c) {
    return c->total_cycles;
}

/* ================================================================
 * DMA Counter API
 * ================================================================ */

void tu_perf_dma_record_read(tu_perf_counters_t *c, uint32_t bytes,
                              uint64_t active_cycles, uint64_t stall_cycles,
                              uint8_t channel, uint8_t transfer_type) {
    if (!c->enabled) return;
    c->dma.dma_read_bytes += bytes;
    c->dma.dma_read_cycles += active_cycles;
    c->dma.dma_stall_cycles += stall_cycles;

    if (channel < 8) {
        c->dma.dma_channel_stalls[channel] += stall_cycles;
        c->dma.dma_channel_bytes[channel] += bytes;
    }

    switch (transfer_type) {
    case 0: c->dma.dma_transfers_linear++; break;
    case 1: c->dma.dma_transfers_strided_2d++; break;
    case 2: c->dma.dma_transfers_strided_3d++; break;
    case 3: c->dma.dma_transfers_scatter++; break;
    case 4: c->dma.dma_transfers_gather++; break;
    }

    tu_perf_tick(c, active_cycles + stall_cycles);

    /* Energy accounting */
    if (c->power.power_modeling_enabled) {
        c->power.energy_dma_pj += c->power.pj_per_dma_byte * (double)bytes;
        c->power.energy_dram_pj += c->power.pj_per_dram_access * (double)(active_cycles);
    }
}

void tu_perf_dma_record_write(tu_perf_counters_t *c, uint32_t bytes,
                               uint64_t active_cycles, uint64_t stall_cycles,
                               uint8_t channel) {
    if (!c->enabled) return;
    c->dma.dma_write_bytes += bytes;
    c->dma.dma_write_cycles += active_cycles;
    c->dma.dma_stall_cycles += stall_cycles;

    if (channel < 8) {
        c->dma.dma_channel_stalls[channel] += stall_cycles;
        c->dma.dma_channel_bytes[channel] += bytes;
    }

    tu_perf_tick(c, active_cycles + stall_cycles);

    if (c->power.power_modeling_enabled) {
        c->power.energy_dma_pj += c->power.pj_per_dma_byte * (double)bytes;
    }
}

void tu_perf_dma_record_internal(tu_perf_counters_t *c, uint32_t bytes,
                                  uint64_t active_cycles, uint8_t channel) {
    if (!c->enabled) return;
    c->dma.dma_internal_bytes += bytes;
    c->dma.dma_read_cycles += active_cycles;

    if (channel < 8) {
        c->dma.dma_channel_bytes[channel] += bytes;
    }

    tu_perf_tick(c, active_cycles);
}

/* ================================================================
 * Compute Counter API
 * ================================================================ */

void tu_perf_compute_record_mma(tu_perf_counters_t *c,
                                 uint64_t macs, uint32_t m, uint32_t n, uint32_t k,
                                 uint32_t tiles, uint32_t edge_tiles,
                                 uint64_t active_cycles, uint64_t stall_cycles,
                                 uint8_t precision_type, uint8_t dataflow_mode) {
    if (!c->enabled) return;
    (void)m; (void)n; (void)k; /* Reserved for future dim-specific analysis */

    c->compute.total_macs += macs;
    c->compute.total_flops += macs * 2;
    c->compute.total_tiles += tiles;
    c->compute.edge_tiles += edge_tiles;
    c->compute.full_tiles += (tiles - edge_tiles);
    c->compute.compute_active_cycles += active_cycles;
    c->compute.compute_stall_cycles += stall_cycles;
    c->compute.compute_total_cycles += active_cycles + stall_cycles;

    /* Track by dataflow */
    if (dataflow_mode == 0) c->compute.df_ws_cycles += active_cycles;
    else                     c->compute.df_os_cycles += active_cycles;

    /* Track by precision */
    switch (precision_type) {
    case 1: c->compute.op_mma_fp16++; break;   /* FP16 */
    case 2: c->compute.op_mma_bf16++; break;   /* BF16 */
    case 3: c->compute.op_mma_int8++; break;    /* INT8 */
    case 4: c->compute.op_mma_fp8++;  break;    /* FP8 */
    default: c->compute.op_mma_fp16++; break;
    }

    /* Update utilization */
    if (c->compute.compute_total_cycles > 0) {
        c->compute.compute_utilization =
            (float)c->compute.compute_active_cycles / (float)c->compute.compute_total_cycles;
    }

    tu_perf_tick(c, active_cycles + stall_cycles);

    /* Energy accounting */
    if (c->power.power_modeling_enabled) {
        c->power.energy_mac_pj += c->power.pj_per_mac * (double)macs;
    }
}

void tu_perf_compute_record_op(tu_perf_counters_t *c, uint8_t op_code,
                                uint64_t active_cycles, uint64_t stall_cycles,
                                uint64_t flops) {
    if (!c->enabled) return;

    c->compute.total_flops += flops;
    c->compute.compute_active_cycles += active_cycles;
    c->compute.compute_stall_cycles += stall_cycles;
    c->compute.compute_total_cycles += active_cycles + stall_cycles;

    /* Per-opcode tracking */
    switch (op_code) {
    case 1:  c->compute.op_mma_fp16++;     break;
    case 2:  c->compute.op_conv2d++;       break;
    case 3:  c->compute.op_attention++;    break;
    case 4:  c->compute.op_elementwise++;  break;
    case 6:  c->compute.op_softmax++;      break;
    case 7:  c->compute.op_layernorm++;    break;
    case 8:  c->compute.op_rmsnorm++;      break;
    case 9:  c->compute.op_pool_max++;     break;
    case 10: c->compute.op_pool_avg++;     break;
    default: c->compute.op_other++;        break;
    }

    if (c->compute.compute_total_cycles > 0) {
        c->compute.compute_utilization =
            (float)c->compute.compute_active_cycles / (float)c->compute.compute_total_cycles;
    }

    tu_perf_tick(c, active_cycles + stall_cycles);
}

void tu_perf_compute_record_idle(tu_perf_counters_t *c, uint64_t cycles) {
    if (!c->enabled) return;
    c->compute.compute_idle_cycles += cycles;
    c->compute.compute_total_cycles += cycles;

    if (c->compute.compute_total_cycles > 0) {
        c->compute.compute_utilization =
            (float)c->compute.compute_active_cycles / (float)c->compute.compute_total_cycles;
    }

    tu_perf_tick(c, cycles);
}

void tu_perf_compute_record_pipeline_bubble(tu_perf_counters_t *c, uint64_t count) {
    if (!c->enabled) return;
    c->compute.compute_pipeline_bubbles += count;
}

/* ================================================================
 * Memory Counter API
 * ================================================================ */

void tu_perf_mem_record_spad_access(tu_perf_counters_t *c,
                                     bool is_write, uint32_t words,
                                     uint32_t bank_conflicts, uint64_t stall_cycles) {
    if (!c->enabled) return;
    if (is_write) {
        c->memory.mem_spad_writes += words;
    } else {
        c->memory.mem_spad_reads += words;
    }
    c->memory.mem_spad_bank_conflicts += bank_conflicts;
    c->memory.mem_spad_stall_cycles += stall_cycles;

    tu_perf_tick(c, stall_cycles);

    if (c->power.power_modeling_enabled) {
        if (is_write) {
            c->power.energy_sram_write_pj += c->power.pj_per_sram_write * (double)words;
        } else {
            c->power.energy_sram_read_pj += c->power.pj_per_sram_read * (double)words;
        }
    }
}

void tu_perf_mem_record_gbuf_access(tu_perf_counters_t *c,
                                     bool is_write, uint32_t words,
                                     uint32_t bank_conflicts) {
    if (!c->enabled) return;
    if (is_write) {
        c->memory.mem_gbuf_writes += words;
    } else {
        c->memory.mem_gbuf_reads += words;
    }
    c->memory.mem_gbuf_bank_conflicts += bank_conflicts;
}

void tu_perf_mem_record_dram_access(tu_perf_counters_t *c,
                                     bool is_write, uint32_t bytes,
                                     bool row_hit, uint64_t stall_cycles) {
    if (!c->enabled) return;
    if (is_write) {
        c->memory.mem_dram_writes++;
        c->memory.mem_dram_bytes_written += bytes;
    } else {
        c->memory.mem_dram_reads++;
        c->memory.mem_dram_bytes_read += bytes;
    }
    if (row_hit) {
        c->memory.mem_dram_row_hits++;
    } else {
        c->memory.mem_dram_row_misses++;
    }
    c->memory.mem_dram_stall_cycles += stall_cycles;

    tu_perf_tick(c, stall_cycles);

    if (c->power.power_modeling_enabled) {
        c->power.energy_dram_pj += c->power.pj_per_dram_access;
    }
}

void tu_perf_mem_record_reqfile_access(tu_perf_counters_t *c,
                                        bool is_write, uint32_t words) {
    if (!c->enabled) return;
    if (is_write) {
        c->memory.mem_reqfile_writes += words;
    } else {
        c->memory.mem_reqfile_reads += words;
    }
}

/* ================================================================
 * Power Counter API
 * ================================================================ */

void tu_perf_power_config(tu_perf_counters_t *c,
                           double pj_mac, double pj_sram_r, double pj_sram_w,
                           double pj_dram, double pj_dma_byte, double pj_leakage) {
    c->power.pj_per_mac          = pj_mac;
    c->power.pj_per_sram_read    = pj_sram_r;
    c->power.pj_per_sram_write   = pj_sram_w;
    c->power.pj_per_dram_access  = pj_dram;
    c->power.pj_per_dma_byte     = pj_dma_byte;
    c->power.pj_leakage_per_cycle= pj_leakage;
}

void tu_perf_power_set_enabled(tu_perf_counters_t *c, bool enabled) {
    c->power.power_modeling_enabled = enabled;
}

/* ================================================================
 * Snapshot & Merge
 * ================================================================ */

tu_perf_snapshot_t tu_perf_snapshot(const tu_perf_counters_t *c) {
    tu_perf_snapshot_t snap;
    memcpy(&snap.counters, c, sizeof(*c));
    snap.snapshot_cycle = c->total_cycles;
    return snap;
}

/* Helper: subtract two uint64s, clamp to 0 */
static inline uint64_t sub_nonneg(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : 0;
}

static inline float subf(float a, float b) {
    return (a >= b) ? (a - b) : 0.0f;
}

tu_perf_counters_t tu_perf_diff(const tu_perf_snapshot_t *before,
                                 const tu_perf_snapshot_t *after) {
    tu_perf_counters_t diff;
    memset(&diff, 0, sizeof(diff));

    if (!before || !after) return diff;

    const tu_perf_counters_t *a = &after->counters;
    const tu_perf_counters_t *b = &before->counters;

    /* DMA */
    diff.dma.dma_read_bytes          = sub_nonneg(a->dma.dma_read_bytes, b->dma.dma_read_bytes);
    diff.dma.dma_write_bytes         = sub_nonneg(a->dma.dma_write_bytes, b->dma.dma_write_bytes);
    diff.dma.dma_internal_bytes      = sub_nonneg(a->dma.dma_internal_bytes, b->dma.dma_internal_bytes);
    diff.dma.dma_read_cycles         = sub_nonneg(a->dma.dma_read_cycles, b->dma.dma_read_cycles);
    diff.dma.dma_write_cycles        = sub_nonneg(a->dma.dma_write_cycles, b->dma.dma_write_cycles);
    diff.dma.dma_stall_cycles        = sub_nonneg(a->dma.dma_stall_cycles, b->dma.dma_stall_cycles);
    diff.dma.dma_transfers_linear    = sub_nonneg(a->dma.dma_transfers_linear, b->dma.dma_transfers_linear);
    diff.dma.dma_transfers_strided_2d= sub_nonneg(a->dma.dma_transfers_strided_2d, b->dma.dma_transfers_strided_2d);
    diff.dma.dma_transfers_strided_3d= sub_nonneg(a->dma.dma_transfers_strided_3d, b->dma.dma_transfers_strided_3d);
    diff.dma.dma_transfers_scatter   = sub_nonneg(a->dma.dma_transfers_scatter, b->dma.dma_transfers_scatter);
    diff.dma.dma_transfers_gather    = sub_nonneg(a->dma.dma_transfers_gather, b->dma.dma_transfers_gather);
    for (int i = 0; i < 8; i++) {
        diff.dma.dma_channel_stalls[i] = sub_nonneg(a->dma.dma_channel_stalls[i], b->dma.dma_channel_stalls[i]);
        diff.dma.dma_channel_bytes[i]  = sub_nonneg(a->dma.dma_channel_bytes[i], b->dma.dma_channel_bytes[i]);
    }

    /* Compute */
    diff.compute.compute_total_cycles    = sub_nonneg(a->compute.compute_total_cycles, b->compute.compute_total_cycles);
    diff.compute.compute_active_cycles   = sub_nonneg(a->compute.compute_active_cycles, b->compute.compute_active_cycles);
    diff.compute.compute_stall_cycles    = sub_nonneg(a->compute.compute_stall_cycles, b->compute.compute_stall_cycles);
    diff.compute.compute_idle_cycles     = sub_nonneg(a->compute.compute_idle_cycles, b->compute.compute_idle_cycles);
    diff.compute.compute_pipeline_bubbles= sub_nonneg(a->compute.compute_pipeline_bubbles, b->compute.compute_pipeline_bubbles);
    diff.compute.total_macs              = sub_nonneg(a->compute.total_macs, b->compute.total_macs);
    diff.compute.total_flops             = sub_nonneg(a->compute.total_flops, b->compute.total_flops);
    diff.compute.total_tiles             = sub_nonneg(a->compute.total_tiles, b->compute.total_tiles);
    diff.compute.edge_tiles              = sub_nonneg(a->compute.edge_tiles, b->compute.edge_tiles);
    diff.compute.full_tiles              = sub_nonneg(a->compute.full_tiles, b->compute.full_tiles);
    diff.compute.op_mma_fp16             = sub_nonneg(a->compute.op_mma_fp16, b->compute.op_mma_fp16);
    diff.compute.op_mma_bf16             = sub_nonneg(a->compute.op_mma_bf16, b->compute.op_mma_bf16);
    diff.compute.op_mma_int8             = sub_nonneg(a->compute.op_mma_int8, b->compute.op_mma_int8);
    diff.compute.op_mma_fp8              = sub_nonneg(a->compute.op_mma_fp8, b->compute.op_mma_fp8);
    diff.compute.op_conv2d               = sub_nonneg(a->compute.op_conv2d, b->compute.op_conv2d);
    diff.compute.op_attention            = sub_nonneg(a->compute.op_attention, b->compute.op_attention);
    diff.compute.op_elementwise          = sub_nonneg(a->compute.op_elementwise, b->compute.op_elementwise);
    diff.compute.op_softmax              = sub_nonneg(a->compute.op_softmax, b->compute.op_softmax);
    diff.compute.op_layernorm            = sub_nonneg(a->compute.op_layernorm, b->compute.op_layernorm);
    diff.compute.op_rmsnorm              = sub_nonneg(a->compute.op_rmsnorm, b->compute.op_rmsnorm);
    diff.compute.op_pool_max             = sub_nonneg(a->compute.op_pool_max, b->compute.op_pool_max);
    diff.compute.op_pool_avg             = sub_nonneg(a->compute.op_pool_avg, b->compute.op_pool_avg);
    diff.compute.op_other                = sub_nonneg(a->compute.op_other, b->compute.op_other);

    if (diff.compute.compute_total_cycles > 0) {
        diff.compute.compute_utilization =
            (float)diff.compute.compute_active_cycles / (float)diff.compute.compute_total_cycles;
    }

    /* Memory */
    diff.memory.mem_reqfile_reads       = sub_nonneg(a->memory.mem_reqfile_reads, b->memory.mem_reqfile_reads);
    diff.memory.mem_reqfile_writes      = sub_nonneg(a->memory.mem_reqfile_writes, b->memory.mem_reqfile_writes);
    diff.memory.mem_spad_reads          = sub_nonneg(a->memory.mem_spad_reads, b->memory.mem_spad_reads);
    diff.memory.mem_spad_writes         = sub_nonneg(a->memory.mem_spad_writes, b->memory.mem_spad_writes);
    diff.memory.mem_spad_bank_conflicts = sub_nonneg(a->memory.mem_spad_bank_conflicts, b->memory.mem_spad_bank_conflicts);
    diff.memory.mem_spad_stall_cycles   = sub_nonneg(a->memory.mem_spad_stall_cycles, b->memory.mem_spad_stall_cycles);
    diff.memory.mem_gbuf_reads          = sub_nonneg(a->memory.mem_gbuf_reads, b->memory.mem_gbuf_reads);
    diff.memory.mem_gbuf_writes         = sub_nonneg(a->memory.mem_gbuf_writes, b->memory.mem_gbuf_writes);
    diff.memory.mem_dram_reads          = sub_nonneg(a->memory.mem_dram_reads, b->memory.mem_dram_reads);
    diff.memory.mem_dram_writes         = sub_nonneg(a->memory.mem_dram_writes, b->memory.mem_dram_writes);
    diff.memory.mem_dram_stall_cycles   = sub_nonneg(a->memory.mem_dram_stall_cycles, b->memory.mem_dram_stall_cycles);
    diff.memory.mem_dram_bytes_read     = sub_nonneg(a->memory.mem_dram_bytes_read, b->memory.mem_dram_bytes_read);
    diff.memory.mem_dram_bytes_written  = sub_nonneg(a->memory.mem_dram_bytes_written, b->memory.mem_dram_bytes_written);

    /* Power */
    diff.power.energy_mac_pj           = a->power.energy_mac_pj - b->power.energy_mac_pj;
    diff.power.energy_sram_read_pj     = a->power.energy_sram_read_pj - b->power.energy_sram_read_pj;
    diff.power.energy_sram_write_pj    = a->power.energy_sram_write_pj - b->power.energy_sram_write_pj;
    diff.power.energy_dram_pj          = a->power.energy_dram_pj - b->power.energy_dram_pj;
    diff.power.energy_dma_pj           = a->power.energy_dma_pj - b->power.energy_dma_pj;
    diff.power.energy_leakage_pj       = a->power.energy_leakage_pj - b->power.energy_leakage_pj;
    diff.power.energy_total_pj         = a->power.energy_total_pj - b->power.energy_total_pj;

    /* Global */
    diff.total_cycles = sub_nonneg(a->total_cycles, b->total_cycles);
    diff.enabled = a->enabled;
    diff.clock_freq_mhz = a->clock_freq_mhz;

    return diff;
}

void tu_perf_merge(tu_perf_counters_t *dst, const tu_perf_counters_t *src) {
    if (!dst || !src) return;

    /* DMA */
    dst->dma.dma_read_bytes           += src->dma.dma_read_bytes;
    dst->dma.dma_write_bytes          += src->dma.dma_write_bytes;
    dst->dma.dma_internal_bytes       += src->dma.dma_internal_bytes;
    dst->dma.dma_read_cycles          += src->dma.dma_read_cycles;
    dst->dma.dma_write_cycles         += src->dma.dma_write_cycles;
    dst->dma.dma_stall_cycles         += src->dma.dma_stall_cycles;
    dst->dma.dma_transfers_linear     += src->dma.dma_transfers_linear;
    dst->dma.dma_transfers_strided_2d += src->dma.dma_transfers_strided_2d;
    dst->dma.dma_transfers_strided_3d += src->dma.dma_transfers_strided_3d;
    dst->dma.dma_transfers_scatter    += src->dma.dma_transfers_scatter;
    dst->dma.dma_transfers_gather     += src->dma.dma_transfers_gather;
    for (int i = 0; i < 8; i++) {
        dst->dma.dma_channel_stalls[i] += src->dma.dma_channel_stalls[i];
        dst->dma.dma_channel_bytes[i]  += src->dma.dma_channel_bytes[i];
    }

    /* Compute */
    dst->compute.compute_total_cycles     += src->compute.compute_total_cycles;
    dst->compute.compute_active_cycles    += src->compute.compute_active_cycles;
    dst->compute.compute_stall_cycles     += src->compute.compute_stall_cycles;
    dst->compute.compute_idle_cycles      += src->compute.compute_idle_cycles;
    dst->compute.compute_pipeline_bubbles += src->compute.compute_pipeline_bubbles;
    dst->compute.total_macs               += src->compute.total_macs;
    dst->compute.total_flops              += src->compute.total_flops;
    dst->compute.total_tiles              += src->compute.total_tiles;
    dst->compute.edge_tiles               += src->compute.edge_tiles;
    dst->compute.full_tiles               += src->compute.full_tiles;
    dst->compute.op_mma_fp16              += src->compute.op_mma_fp16;
    dst->compute.op_mma_bf16              += src->compute.op_mma_bf16;
    dst->compute.op_mma_int8              += src->compute.op_mma_int8;
    dst->compute.op_mma_fp8               += src->compute.op_mma_fp8;
    dst->compute.op_conv2d                += src->compute.op_conv2d;
    dst->compute.op_attention             += src->compute.op_attention;
    dst->compute.op_elementwise           += src->compute.op_elementwise;
    dst->compute.op_softmax               += src->compute.op_softmax;
    dst->compute.op_layernorm             += src->compute.op_layernorm;
    dst->compute.op_rmsnorm               += src->compute.op_rmsnorm;
    dst->compute.op_pool_max              += src->compute.op_pool_max;
    dst->compute.op_pool_avg              += src->compute.op_pool_avg;
    dst->compute.op_other                 += src->compute.op_other;

    if (dst->compute.compute_total_cycles > 0) {
        dst->compute.compute_utilization =
            (float)dst->compute.compute_active_cycles / (float)dst->compute.compute_total_cycles;
    }

    /* Memory */
    dst->memory.mem_reqfile_reads        += src->memory.mem_reqfile_reads;
    dst->memory.mem_reqfile_writes       += src->memory.mem_reqfile_writes;
    dst->memory.mem_spad_reads           += src->memory.mem_spad_reads;
    dst->memory.mem_spad_writes          += src->memory.mem_spad_writes;
    dst->memory.mem_spad_bank_conflicts  += src->memory.mem_spad_bank_conflicts;
    dst->memory.mem_spad_stall_cycles    += src->memory.mem_spad_stall_cycles;
    dst->memory.mem_gbuf_reads           += src->memory.mem_gbuf_reads;
    dst->memory.mem_gbuf_writes          += src->memory.mem_gbuf_writes;
    dst->memory.mem_dram_reads           += src->memory.mem_dram_reads;
    dst->memory.mem_dram_writes          += src->memory.mem_dram_writes;
    dst->memory.mem_dram_stall_cycles    += src->memory.mem_dram_stall_cycles;
    dst->memory.mem_dram_bytes_read      += src->memory.mem_dram_bytes_read;
    dst->memory.mem_dram_bytes_written   += src->memory.mem_dram_bytes_written;

    /* Power */
    dst->power.energy_mac_pj        += src->power.energy_mac_pj;
    dst->power.energy_sram_read_pj  += src->power.energy_sram_read_pj;
    dst->power.energy_sram_write_pj += src->power.energy_sram_write_pj;
    dst->power.energy_dram_pj       += src->power.energy_dram_pj;
    dst->power.energy_dma_pj        += src->power.energy_dma_pj;
    dst->power.energy_leakage_pj    += src->power.energy_leakage_pj;

    /* Global */
    dst->total_cycles += src->total_cycles;
}

void tu_perf_from_dma_descriptor(tu_perf_counters_t *c,
                                  uint32_t bytes, uint8_t channel,
                                  uint8_t transfer_type, bool is_read,
                                  uint64_t active_cycles, uint64_t stall_cycles,
                                  uint64_t sram_stall_cycles) {
    if (!c->enabled) return;

    if (is_read) {
        tu_perf_dma_record_read(c, bytes, active_cycles, stall_cycles,
                                 channel, transfer_type);
    } else {
        tu_perf_dma_record_write(c, bytes, active_cycles, stall_cycles, channel);
    }

    /* SRAM stall accounting */
    if (sram_stall_cycles > 0) {
        uint32_t words = (bytes + 3) / 4; /* approximate word count */
        tu_perf_mem_record_spad_access(c, !is_read, words, 0, sram_stall_cycles);
    }
}

/* ================================================================
 * Metric Computation
 * ================================================================ */

tu_perf_metrics_t tu_perf_compute_metrics(const tu_perf_counters_t *c) {
    tu_perf_metrics_t m;
    memset(&m, 0, sizeof(m));

    if (c->total_cycles == 0) return m;

    double seconds = (double)c->total_cycles / (c->clock_freq_mhz * 1e6);

    /* Utilization */
    m.compute_utilization = c->compute.compute_utilization;

    /* DMA bandwidth (GB/s) */
    double dma_bytes = (double)(c->dma.dma_read_bytes + c->dma.dma_write_bytes);
    m.dma_bandwidth_gbps = (seconds > 0) ? (float)(dma_bytes / seconds / 1e9) : 0.0f;

    /* DRAM bandwidth (GB/s) */
    double dram_bytes = (double)(c->memory.mem_dram_bytes_read + c->memory.mem_dram_bytes_written);
    m.dram_bandwidth_gbps = (seconds > 0) ? (float)(dram_bytes / seconds / 1e9) : 0.0f;

    /* MAC throughput (TOPS — FP16-equivalent) */
    m.mac_throughput_tops = (seconds > 0) ? (float)((double)c->compute.total_macs / seconds / 1e12) : 0.0f;

    /* MAC efficiency: effective / peak */
    double peak_macs_per_cycle = (double)(TU_PE_ROWS * TU_PE_COLS);
    double peak_macs = peak_macs_per_cycle * (double)c->total_cycles;
    m.mac_efficiency = (peak_macs > 0) ? (float)((double)c->compute.total_macs / peak_macs) : 0.0f;

    /* Scratchpad hit rate */
    uint64_t spad_total = c->memory.mem_spad_reads + c->memory.mem_spad_writes;
    m.spad_hit_rate = (spad_total > 0)
        ? 1.0f - (float)c->memory.mem_spad_bank_conflicts / (float)spad_total
        : 1.0f;

    /* Energy per MAC */
    m.energy_per_mac_pj = (c->compute.total_macs > 0)
        ? (float)((c->power.energy_mac_pj + c->power.energy_sram_read_pj +
                   c->power.energy_sram_write_pj + c->power.energy_dram_pj +
                   c->power.energy_dma_pj + c->power.energy_leakage_pj) /
                  (double)c->compute.total_macs)
        : 0.0f;

    /* Average power (mW) */
    m.power_mw = (seconds > 0)
        ? (float)((c->power.energy_mac_pj + c->power.energy_sram_read_pj +
                   c->power.energy_sram_write_pj + c->power.energy_dram_pj +
                   c->power.energy_dma_pj + c->power.energy_leakage_pj) / 1e9 / seconds * 1e3)
        : 0.0f;

    return m;
}

/* ================================================================
 * Reporting
 * ================================================================ */

void tu_perf_print_report(const tu_perf_counters_t *c) {
    tu_perf_metrics_t m = tu_perf_compute_metrics(c);
    double seconds = (double)c->total_cycles / (c->clock_freq_mhz * 1e6);

    printf("\n");
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│          TU CModel — Performance Counter Report             │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│ Global                                                       │\n");
    printf("│   Total cycles:     %12lu                                │\n", (unsigned long)c->total_cycles);
    printf("│   Simulated time:   %12.3f µs                            │\n", seconds * 1e6);
    printf("│   Clock frequency:  %12.0f MHz                           │\n", c->clock_freq_mhz);
    printf("├──────────────────────────────────┬───────────────────────────┤\n");
    printf("│ DMA Engine                       │                           │\n");
    printf("│   Read bytes:    %15lu │  Write bytes: %15lu │\n",
           (unsigned long)c->dma.dma_read_bytes, (unsigned long)c->dma.dma_write_bytes);
    printf("│   Internal:      %15lu │  Stall cycles:%14lu │\n",
           (unsigned long)c->dma.dma_internal_bytes, (unsigned long)c->dma.dma_stall_cycles);
    printf("│   Linear xfers:  %15lu │  Strided 2D:  %14lu │\n",
           (unsigned long)c->dma.dma_transfers_linear, (unsigned long)c->dma.dma_transfers_strided_2d);
    printf("│   Strided 3D:    %15lu │  Scatter:     %14lu │\n",
           (unsigned long)c->dma.dma_transfers_strided_3d, (unsigned long)c->dma.dma_transfers_scatter);
    printf("│   Gather:        %15lu │  BW:        %10.3f GB/s │\n",
           (unsigned long)c->dma.dma_transfers_gather, m.dma_bandwidth_gbps);
    printf("├──────────────────────────────────┼───────────────────────────┤\n");
    printf("│ Compute Engine                   │                           │\n");
    printf("│   Total MACs:    %15lu │  Utilization:%12.1f %%   │\n",
           (unsigned long)c->compute.total_macs, m.compute_utilization * 100.0f);
    printf("│   Total tiles:   %15lu │  Edge tiles: %14lu │\n",
           (unsigned long)c->compute.total_tiles, (unsigned long)c->compute.edge_tiles);
    printf("│   Active cycles: %15lu │  Stall:       %14lu │\n",
           (unsigned long)c->compute.compute_active_cycles,
           (unsigned long)c->compute.compute_stall_cycles);
    printf("│   Idle cycles:   %15lu │  Bubbles:     %14lu │\n",
           (unsigned long)c->compute.compute_idle_cycles,
           (unsigned long)c->compute.compute_pipeline_bubbles);
    printf("│   Throughput:    %12.3f TOPS │  Efficiency:  %12.1f %%   │\n",
           m.mac_throughput_tops, m.mac_efficiency * 100.0f);
    printf("├──────────────────────────────────┼───────────────────────────┤\n");
    printf("│ Memory Hierarchy                 │                           │\n");
    printf("│   SPAD reads:    %15lu │  SPAD writes: %14lu │\n",
           (unsigned long)c->memory.mem_spad_reads, (unsigned long)c->memory.mem_spad_writes);
    printf("│   Bank conflicts:%15lu │  SPAD stalls: %14lu │\n",
           (unsigned long)c->memory.mem_spad_bank_conflicts,
           (unsigned long)c->memory.mem_spad_stall_cycles);
    printf("│   DRAM reads:    %15lu │  DRAM writes: %14lu │\n",
           (unsigned long)c->memory.mem_dram_reads, (unsigned long)c->memory.mem_dram_writes);
    printf("│   Row hits:      %15lu │  Row misses:  %14lu │\n",
           (unsigned long)c->memory.mem_dram_row_hits,
           (unsigned long)c->memory.mem_dram_row_misses);
    printf("│   DRAM BW:       %12.3f GB/s │  SPAD hit rate:%12.1f %%   │\n",
           m.dram_bandwidth_gbps, m.spad_hit_rate * 100.0f);
    printf("├──────────────────────────────────┼───────────────────────────┤\n");
    printf("│ Power / Energy                   │                           │\n");
    printf("│   MAC energy:    %15.1f pJ │  SRAM read:   %13.1f pJ │\n",
           c->power.energy_mac_pj, c->power.energy_sram_read_pj);
    printf("│   SRAM write:    %15.1f pJ │  DRAM:        %13.1f pJ │\n",
           c->power.energy_sram_write_pj, c->power.energy_dram_pj);
    printf("│   DMA energy:    %15.1f pJ │  Leakage:     %13.1f pJ │\n",
           c->power.energy_dma_pj, c->power.energy_leakage_pj);
    printf("│   Total energy:  %15.1f pJ │  Avg power:   %12.3f mW │\n",
           c->power.energy_mac_pj + c->power.energy_sram_read_pj +
           c->power.energy_sram_write_pj + c->power.energy_dram_pj +
           c->power.energy_dma_pj + c->power.energy_leakage_pj,
           m.power_mw);
    printf("│   Energy/MAC:    %12.3f pJ/MAC │                          │\n",
           m.energy_per_mac_pj);
    printf("├──────────────────────────────────┴───────────────────────────┤\n");
    printf("│ Per-Operation Counts                                         │\n");
    printf("│   FP16 MMA: %10lu  BF16 MMA: %10lu  INT8 MMA:  %10lu │\n",
           (unsigned long)c->compute.op_mma_fp16, (unsigned long)c->compute.op_mma_bf16,
           (unsigned long)c->compute.op_mma_int8);
    printf("│   FP8  MMA: %10lu  Conv2D:   %10lu  Attention:  %10lu │\n",
           (unsigned long)c->compute.op_mma_fp8, (unsigned long)c->compute.op_conv2d,
           (unsigned long)c->compute.op_attention);
    printf("│   Elemwise: %10lu  Softmax:  %10lu  LayerNorm:  %10lu │\n",
           (unsigned long)c->compute.op_elementwise, (unsigned long)c->compute.op_softmax,
           (unsigned long)c->compute.op_layernorm);
    printf("│   RMSNorm:  %10lu  PoolMax:  %10lu  PoolAvg:   %10lu │\n",
           (unsigned long)c->compute.op_rmsnorm, (unsigned long)c->compute.op_pool_max,
           (unsigned long)c->compute.op_pool_avg);
    printf("│   Other:    %10lu                                      │\n",
           (unsigned long)c->compute.op_other);
    printf("└──────────────────────────────────────────────────────────────┘\n");
}

void tu_perf_print_summary(const tu_perf_counters_t *c) {
    tu_perf_metrics_t m = tu_perf_compute_metrics(c);
    printf("[perf] %lu cyc | %.1f%% util | %.3f TOPS | %.1f mW | %lu MACs\n",
           (unsigned long)c->total_cycles,
           m.compute_utilization * 100.0f,
           m.mac_throughput_tops,
           m.power_mw,
           (unsigned long)c->compute.total_macs);
}
