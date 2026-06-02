/*
 * TU CModel — Liveness-Based Scratchpad Allocator Implementation (Gap C3)
 *
 * See tu_liveness.h for the architecture and API documentation.
 *
 * Implementation notes:
 *   - Three independent scratchpad memories (W, A, O) are allocated separately.
 *   - Virtual registers are extracted from instruction operands: each write to
 *     an SRAM region creates a VReg; subsequent reads extend its live range.
 *   - Interference: two VRegs conflict if their live ranges overlap AND they
 *     are in the same SRAM region. Cross-region VRegs never conflict.
 *   - Coloring: greedy first-fit placement. Walk VRegs in order of increasing
 *     start time. For each, find the lowest physical offset where it doesn't
 *     overlap any already-placed interfering VReg in physical space.
 *   - Spilling: when no physical slot fits, select a victim using the configured
 *     spill strategy. Insert DMA_STORE (spill) before the victim's last use and
 *     DMA_LOAD (fill) after.
 *   - This allocator uses the simplified functional model (all instructions take
 *     ~equal time). Cycle-accurate liveness integration is future work (P2.5).
 */

#include "tu_liveness.h"
#include "tu_scheduler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Default config ---- */
const tu_live_config_t tu_live_config_default = {
    .w_capacity      = 131072,    /* 128 KB W-SRAM (default TinyTU) */
    .a_capacity      = 65536,     /* 64 KB A-SRAM */
    .o_capacity      = 65536,     /* 64 KB O-SRAM */
    .alloc_strategy  = TU_ALLOC_FIRST_FIT,
    .spill_strategy  = TU_SPILL_LRU,
    .safety_margin   = 4096,      /* 4 KB for spill descriptors */
    .enable_spilling = true,
    .verbose         = false,
};

/* ---- Helpers ---- */

static uint32_t region_capacity(const tu_live_config_t *cfg, tu_vreg_region_t region) {
    switch (region) {
        case TU_VREG_W: return cfg->w_capacity;
        case TU_VREG_A: return cfg->a_capacity;
        case TU_VREG_O: return cfg->o_capacity;
        default: return 0;
    }
}

static tu_vreg_region_t vreg_region_from_sram(int sram_region) {
    return (tu_vreg_region_t)sram_region; /* 0=W, 1=A, 2=O */
}

static bool is_load_op(tu_isa_opcode_t op) {
    return (op == TU_ISA_DMA_LOAD || op == TU_ISA_DMA_LOAD_STRIDED
         || op == TU_ISA_DMA_SCATTER || op == TU_ISA_DMA_BROADCAST);
}

static bool is_store_op(tu_isa_opcode_t op) {
    return (op == TU_ISA_DMA_STORE || op == TU_ISA_DMA_STORE_STRIDED
         || op == TU_ISA_DMA_GATHER);
}

/*
 * Extract the SRAM byte range that an instruction writes (defines).
 * For DMA_LOAD: the destination SRAM region gets a definition.
 * For MMA/Compute: the O-SRAM (or A-SRAM for certain ops) gets a definition.
 * Returns region index + [start, end), or -1 if no definition.
 */
static int extract_vreg_def(const tu_instruction_t *instr,
                             tu_vreg_region_t *region_out,
                             uint32_t *start_out, uint32_t *end_out) {
    if (is_load_op((tu_isa_opcode_t)instr->opcode)) {
        /* DMA load: writes to SRAM. Channel in flags bits [1:0] */
        uint8_t ch = instr->flags & 0x3;
        *region_out = vreg_region_from_sram(ch < 3 ? ch : 0);
        *start_out  = instr->dim0;
        *end_out    = instr->dim0 + (instr->dim1 ? instr->dim1 : 1);
        return 0;
    }

    if (instr->opcode == TU_ISA_MMA || instr->opcode == TU_ISA_MMA_BIAS
        || instr->opcode == TU_ISA_MMA_FUSED || instr->opcode == TU_ISA_SPARSE_MMA) {
        /* MMA writes to O-SRAM */
        *region_out = TU_VREG_O;
        *start_out  = (instr->immediates >> 16) & 0xFFFF;
        *end_out    = ((instr->immediates >> 16) & 0xFFFF) + instr->dim0 * instr->dim1 * 4;
        return 0;
    }

    if (instr->opcode >= TU_ISA_ADD && instr->opcode <= TU_ISA_EXP) {
        /* Elementwise writes O */
        *region_out = TU_VREG_O;
        *start_out  = instr->dim0;
        *end_out    = instr->dim0 + instr->dim1 * 4;
        return 0;
    }

    if (instr->opcode == TU_ISA_SOFTMAX || instr->opcode == TU_ISA_LOG_SOFTMAX
        || instr->opcode >= TU_ISA_LAYER_NORM) {
        *region_out = TU_VREG_O;
        *start_out  = (instr->immediates >> 16) & 0xFFFF;
        *end_out    = ((instr->immediates >> 16) & 0xFFFF) + instr->dim1 * 4;
        return 0;
    }

    if (instr->opcode == TU_ISA_CONV2D || instr->opcode == TU_ISA_CONV3D
        || instr->opcode == TU_ISA_DEPTHWISE_CONV) {
        *region_out = TU_VREG_O;
        /* Conservative: full O-SRAM */
        *start_out = 0;
        *end_out   = UINT32_MAX;
        return 0;
    }

    if (instr->opcode == TU_ISA_ATTENTION || instr->opcode == TU_ISA_ATTN_QK
        || instr->opcode == TU_ISA_ATTN_PV) {
        *region_out = TU_VREG_O;
        *start_out = (instr->immediates >> 16) & 0xFFFF;
        *end_out   = ((instr->immediates >> 16) & 0xFFFF) + 65536;
        return 0;
    }

    if (instr->opcode == TU_ISA_POOL_MAX || instr->opcode == TU_ISA_POOL_AVG
        || instr->opcode == TU_ISA_POOL_GLOBAL_AVG) {
        *region_out = TU_VREG_O;
        *start_out = (instr->immediates >> 16) & 0xFFFF;
        *end_out   = ((instr->immediates >> 16) & 0xFFFF) + 65536;
        return 0;
    }

    return -1; /* No definition */
}

/*
 * Extract which SRAM byte ranges an instruction reads (uses).
 */
static int extract_vreg_uses(const tu_instruction_t *instr,
                              tu_vreg_region_t *regions_out,
                              uint32_t *starts_out,
                              uint32_t *ends_out,
                              int max_uses) {
    int count = 0;

    if (is_store_op((tu_isa_opcode_t)instr->opcode)) {
        /* DMA store: reads from SRAM */
        uint8_t ch = instr->flags & 0x3;
        if (count < max_uses) {
            regions_out[count] = vreg_region_from_sram(ch < 3 ? ch : 0);
            starts_out[count]  = instr->dim0;
            ends_out[count]    = instr->dim0 + (instr->dim1 ? instr->dim1 : 1);
            count++;
        }
        return count;
    }

    if (instr->opcode == TU_ISA_MMA || instr->opcode == TU_ISA_MMA_BIAS
        || instr->opcode == TU_ISA_MMA_FUSED || instr->opcode == TU_ISA_SPARSE_MMA) {
        /* Reads W-SRAM and A-SRAM */
        if (count < max_uses) {
            regions_out[count] = TU_VREG_W;
            starts_out[count]  = instr->dim0;
            ends_out[count]    = instr->dim0 + instr->dim2 * instr->dim1 * 2;
            count++;
        }
        if (count < max_uses) {
            regions_out[count] = TU_VREG_A;
            starts_out[count]  = instr->immediates & 0xFFFF;
            ends_out[count]    = (instr->immediates & 0xFFFF) + instr->dim2 * instr->dim1 * 2;
            count++;
        }
        return count;
    }

    if (instr->opcode >= TU_ISA_RELU && instr->opcode <= TU_ISA_EXP) {
        if (count < max_uses) {
            regions_out[count] = TU_VREG_O;
            starts_out[count]  = instr->dim0;
            ends_out[count]    = instr->dim0 + instr->dim1 * 4;
            count++;
        }
        return count;
    }

    if (instr->opcode == TU_ISA_SOFTMAX || instr->opcode == TU_ISA_LOG_SOFTMAX
        || instr->opcode >= TU_ISA_LAYER_NORM) {
        if (count < max_uses) {
            regions_out[count] = TU_VREG_A;
            starts_out[count]  = instr->dim0;
            ends_out[count]    = instr->dim0 + instr->dim1 * 4;
            count++;
        }
        return count;
    }

    return count;
}

/*
 * Find or create a virtual register that covers [start, end) in the given region.
 * DEFINITIONS (is_def=true): always create a new VReg (each write is a new value).
 * USES (is_def=false): extend the live range of the most recently defined VReg
 * whose virtual range overlaps.
 */
static tu_vreg_t *find_or_create_vreg(tu_liveness_result_t *result,
                                       tu_vreg_region_t region,
                                       uint32_t start, uint32_t end,
                                       int32_t instr_idx, bool is_def) {
    (void)start; (void)end;

    if (is_def) {
        /* Each definition creates a new virtual register */
        if (result->num_vregs >= TU_LIVE_MAX_VREGS) return NULL;
        tu_vreg_t *v = &result->vregs[result->num_vregs++];
        memset(v, 0, sizeof(*v));
        v->id             = result->num_vregs;
        v->region          = region;
        v->size_bytes      = end - start;
        v->first_def       = instr_idx;
        v->last_use        = instr_idx;
        v->spilled         = false;
        v->spill_slot      = 0;
        v->access_count    = 1;
        v->physical_offset = UINT32_MAX;
        return v;
    }

    /* For uses: extend the most recently defined VReg in this region */
    for (int32_t i = (int32_t)result->num_vregs - 1; i >= 0; i--) {
        tu_vreg_t *v = &result->vregs[i];
        if (v->region != region) continue;
        if (v->first_def < 0) continue;
        /* Found the most recent definition in this region — extend its live range */
        if (instr_idx > v->last_use) v->last_use = instr_idx;
        v->access_count++;
        return v;
    }

    /* No prior definition found — create one (implicit def) */
    if (result->num_vregs >= TU_LIVE_MAX_VREGS) return NULL;
    tu_vreg_t *v = &result->vregs[result->num_vregs++];
    memset(v, 0, sizeof(*v));
    v->id             = result->num_vregs;
    v->region          = region;
    v->size_bytes      = end - start;
    v->first_def       = -1;  /* Implicit definition */
    v->last_use        = instr_idx;
    v->spilled         = false;
    v->access_count    = 1;
    v->physical_offset = UINT32_MAX;
    return v;
}

/* ================================================================
 * Liveness Analysis
 * ================================================================ */

int tu_live_analyze(const tu_instruction_t *instrs,
                     uint32_t n_instrs,
                     tu_liveness_result_t *result) {
    if (!instrs || !result || n_instrs == 0) return -1;

    memset(result, 0, sizeof(*result));

    for (uint32_t i = 0; i < n_instrs; i++) {
        const tu_instruction_t *instr = &instrs[i];

        /* Extract definition (write) */
        tu_vreg_region_t def_region;
        uint32_t def_start, def_end;
        if (extract_vreg_def(instr, &def_region, &def_start, &def_end) == 0) {
            find_or_create_vreg(result, def_region, def_start, def_end,
                                (int32_t)i, true);
        }

        /* Extract uses (reads) */
        tu_vreg_region_t use_regions[8];
        uint32_t use_starts[8], use_ends[8];
        int n_uses = extract_vreg_uses(instr, use_regions, use_starts, use_ends, 8);
        for (int u = 0; u < n_uses; u++) {
            find_or_create_vreg(result, use_regions[u], use_starts[u], use_ends[u],
                                (int32_t)i, false);
        }
    }

    return 0;
}

/* ================================================================
 * Interference Graph Construction
 * ================================================================ */

static tu_interference_graph_t *region_graph(tu_liveness_result_t *result,
                                               tu_vreg_region_t region) {
    switch (region) {
        case TU_VREG_W: return &result->graph_w;
        case TU_VREG_A: return &result->graph_a;
        case TU_VREG_O: return &result->graph_o;
        default: return NULL;
    }
}

void tu_live_build_interference(tu_liveness_result_t *result) {
    /* Build per-region VReg lists */
    for (uint32_t i = 0; i < result->num_vregs; i++) {
        tu_vreg_t *v = &result->vregs[i];
        tu_interference_graph_t *g = region_graph(result, v->region);
        if (g && g->num_vregs < TU_LIVE_MAX_VREGS) {
            g->vregs[g->num_vregs++] = v;
        }
    }

    /* Allocate adjacency matrices */
    for (int r = 0; r < 3; r++) {
        tu_interference_graph_t *g = region_graph(result, (tu_vreg_region_t)r);
        if (!g || g->num_vregs == 0) continue;

        uint32_t n = g->num_vregs;
        g->interference = calloc(n * n, sizeof(bool));
        g->colored = false;
        g->num_physical_slots = 0;

        /* Populate interference: two VRegs interfere if live ranges overlap */
        for (uint32_t i = 0; i < n; i++) {
            tu_vreg_t *vi = g->vregs[i];
            for (uint32_t j = i + 1; j < n; j++) {
                tu_vreg_t *vj = g->vregs[j];
                /* Interfere if vi's live range [first_def, last_use]
                 * overlaps vj's live range [first_def, last_use] */
                if (vi->first_def >= 0 && vj->first_def >= 0) {
                    bool overlap = (vi->first_def <= vj->last_use
                                 && vj->first_def <= vi->last_use);
                    if (overlap) {
                        g->interference[i * n + j] = true;
                        g->interference[j * n + i] = true;
                    }
                }
            }
        }
    }
}

/* ================================================================
 * Greedy Coloring
 * ================================================================ */

/*
 * Check if VReg v at physical_offset conflicts with any already-placed
 * interfering VReg.
 */
static bool physical_conflict(const tu_interference_graph_t *g,
                               uint32_t v_idx, uint32_t phys_offset) {
    tu_vreg_t *v = g->vregs[v_idx];
    uint32_t n = g->num_vregs;

    for (uint32_t j = 0; j < n; j++) {
        if (j == v_idx) continue;
        if (!g->interference[v_idx * n + j]) continue;
        tu_vreg_t *other = g->vregs[j];
        if (other->physical_offset == UINT32_MAX) continue; /* Not placed yet */

        /* Check physical overlap */
        uint32_t v_end   = phys_offset + v->size_bytes;
        uint32_t o_start = other->physical_offset;
        uint32_t o_end   = o_start + other->size_bytes;

        if (phys_offset < o_end && o_start < v_end) {
            return true; /* Conflict */
        }
    }
    return false;
}

/*
 * Select a victim VReg to spill based on spill strategy.
 */
static uint32_t select_spill_victim(tu_interference_graph_t *g,
                                      tu_spill_strategy_t strategy) {
    uint32_t n = g->num_vregs;
    uint32_t victim = 0;
    uint32_t best_score = 0;

    for (uint32_t i = 0; i < n; i++) {
        tu_vreg_t *v = g->vregs[i];
        if (v->spilled || v->physical_offset != UINT32_MAX) continue;

        uint32_t score = 0;
        switch (strategy) {
            case TU_SPILL_FIFO:
                score = UINT32_MAX - v->id; /* Oldest = lowest ID */
                break;
            case TU_SPILL_LRU:
                score = UINT32_MAX - v->last_use; /* Furthest next use */
                break;
            case TU_SPILL_LARGEST:
                score = v->size_bytes; /* Free most space */
                break;
            case TU_SPILL_LEAST_ACCESSED:
                score = UINT32_MAX - v->access_count;
                break;
            default:
                break;
        }
        if (score > best_score) {
            best_score = score;
            victim = i;
        }
    }
    return victim;
}

void tu_live_color(tu_liveness_result_t *result,
                    const tu_live_config_t *config) {
    if (!config) config = &tu_live_config_default;

    for (int r = 0; r < 3; r++) {
        tu_interference_graph_t *g = region_graph(result, (tu_vreg_region_t)r);
        if (!g || g->num_vregs == 0) continue;

        uint32_t n = g->num_vregs;
        uint32_t capacity = region_capacity(config, (tu_vreg_region_t)r)
                          - config->safety_margin;

        /* Sort VRegs by first_def (earlier = placed first) */
        /* Simple insertion sort */
        for (uint32_t i = 1; i < n; i++) {
            tu_vreg_t *key = g->vregs[i];
            int32_t j = (int32_t)i - 1;
            while (j >= 0 && g->vregs[j]->first_def > key->first_def) {
                g->vregs[j + 1] = g->vregs[j];
                j--;
            }
            g->vregs[j + 1] = key;
        }

        /* Greedy coloring */
        int spilled = 0;
        for (uint32_t i = 0; i < n; i++) {
            tu_vreg_t *v = g->vregs[i];
            if (v->first_def < 0) {
                v->physical_offset = 0; /* Never defined → no allocation needed */
                continue;
            }

            /* Try each possible offset */
            bool placed = false;
            uint32_t step = (config->alloc_strategy == TU_ALLOC_BEST_FIT) ? 4 : 16;
            for (uint32_t off = 0; off + v->size_bytes <= capacity; off += step) {
                if (!physical_conflict(g, i, off)) {
                    v->physical_offset = off;
                    placed = true;
                    break;
                }
            }

            if (!placed && config->enable_spilling) {
                /* Try lower granularity for best-fit */
                if (config->alloc_strategy == TU_ALLOC_BEST_FIT) {
                    for (uint32_t off = 0; off + v->size_bytes <= capacity; off += 1) {
                        if (!physical_conflict(g, i, off)) {
                            v->physical_offset = off;
                            placed = true;
                            break;
                        }
                    }
                }
                if (!placed) {
                    /* Use spill victim selection to decide which VReg to spill */
                    uint32_t victim = select_spill_victim(g, config->spill_strategy);
                    if (victim < n && !g->vregs[victim]->spilled) {
                        g->vregs[victim]->spilled = true;
                        result->num_spills++;
                        result->spill_bytes += g->vregs[victim]->size_bytes;
                        /* Re-try placing current VReg */
                        for (uint32_t off = 0; off + v->size_bytes <= capacity; off += step) {
                            if (!physical_conflict(g, i, off)) {
                                v->physical_offset = off;
                                placed = true;
                                break;
                            }
                        }
                    }
                    if (!placed) {
                        v->spilled = true;
                        result->num_spills++;
                        result->spill_bytes += v->size_bytes;
                        spilled++;
                    }
                }
            } else if (!placed) {
                /* Without spilling, force placement (may cause functional errors
                 * in cycle-accurate mode but correct in functional mode) */
                v->physical_offset = 0;
            }
        }

        g->colored = (spilled == 0);
        g->num_physical_slots = n - spilled;
    }
}

/* ================================================================
 * Apply Allocation to Instruction Sequence
 * ================================================================ */

/*
 * Find the VReg corresponding to a write at a given instruction index.
 */
static tu_vreg_t *find_def_vreg(tu_liveness_result_t *result,
                                  int32_t instr_idx) {
    for (uint32_t i = 0; i < result->num_vregs; i++) {
        if (result->vregs[i].first_def == instr_idx) {
            return &result->vregs[i];
        }
    }
    return NULL;
}

/*
 * Apply physical offset to an instruction's operands.
 * Patches the instruction in-place.
 */
static void patch_instruction(tu_instruction_t *instr,
                               tu_liveness_result_t *result,
                               int32_t instr_idx) {
    /* Find the VReg defined by this instruction and patch the write offset */
    tu_vreg_t *def = find_def_vreg(result, instr_idx);
    if (!def || def->physical_offset == UINT32_MAX) return;

    if (def->region == TU_VREG_O) {
        /* Patch O-sram offset (in immediates high 16 bits for MMA, dim0 for elementwise) */
        if (instr->opcode == TU_ISA_MMA || instr->opcode == TU_ISA_MMA_BIAS
            || instr->opcode == TU_ISA_MMA_FUSED) {
            instr->immediates = (instr->immediates & 0xFFFF)
                              | ((def->physical_offset & 0xFFFF) << 16);
        } else if (instr->opcode >= TU_ISA_RELU && instr->opcode <= TU_ISA_EXP) {
            instr->dim0 = def->physical_offset;
        }
    }

    /* Patch W-SRAM offsets for DMA loads */
    if (def->region == TU_VREG_W) {
        if (is_load_op((tu_isa_opcode_t)instr->opcode)) {
            instr->dim0 = def->physical_offset;
        }
    }

    if (def->region == TU_VREG_A) {
        if (is_load_op((tu_isa_opcode_t)instr->opcode)) {
            instr->dim0 = def->physical_offset;
        }
    }

    /* For MMA, patch A-SRAM and W-SRAM read offsets */
    if (instr->opcode == TU_ISA_MMA || instr->opcode == TU_ISA_MMA_BIAS
        || instr->opcode == TU_ISA_MMA_FUSED) {
        for (uint32_t i = 0; i < result->num_vregs; i++) {
            tu_vreg_t *v = &result->vregs[i];
            if (v->first_def < 0) continue;
            /* Patch if this VReg's live range includes this instruction */
            if (v->first_def <= instr_idx && instr_idx <= v->last_use) {
                if (v->region == TU_VREG_W) {
                    instr->dim0 = v->physical_offset;
                }
                if (v->region == TU_VREG_A) {
                    instr->immediates = (instr->immediates & 0xFFFF0000)
                                      | (v->physical_offset & 0xFFFF);
                }
            }
        }
    }
}

/*
 * Insert a spill DMA instruction for a spilled VReg.
 */
static tu_instruction_t make_spill_instr(uint32_t sram_offset, uint32_t size,
                                          tu_vreg_region_t region) {
    tu_instruction_t instr;
    memset(&instr, 0, sizeof(instr));
    instr.opcode   = TU_ISA_DMA_STORE;
    instr.flags    = (uint8_t)region;     /* Channel */
    instr.dim0     = (uint16_t)(sram_offset & 0xFFFF);
    instr.dim1     = (uint16_t)(size & 0xFFFF);
    return instr;
}

/*
 * Insert a fill DMA instruction for a previously spilled VReg.
 */
static tu_instruction_t make_fill_instr(uint32_t sram_offset, uint32_t size,
                                         tu_vreg_region_t region) {
    tu_instruction_t instr;
    memset(&instr, 0, sizeof(instr));
    instr.opcode   = TU_ISA_DMA_LOAD;
    instr.flags    = (uint8_t)region;
    instr.dim0     = (uint16_t)(sram_offset & 0xFFFF);
    instr.dim1     = (uint16_t)(size & 0xFFFF);
    return instr;
}

int tu_live_apply(tu_liveness_result_t *result,
                   const tu_instruction_t *input_instrs,
                   uint32_t n_input,
                   const tu_live_config_t *config,
                   tu_allocated_sequence_t *output) {
    if (!result || !input_instrs || !output) return -1;
    if (!config) config = &tu_live_config_default;

    memset(output, 0, sizeof(*output));
    output->valid = false;

    uint32_t out_idx = 0;
    uint32_t peak_w = 0, peak_a = 0, peak_o = 0;

    /* Track current SRAM usage */
    uint32_t w_usage = 0, a_usage = 0, o_usage = 0;

    for (uint32_t i = 0; i < n_input && out_idx < TU_SCHED_MAX_INSTRS * 2; i++) {
        tu_instruction_t instr = input_instrs[i];

        /* Check if any VReg needs filling before this instruction */
        for (uint32_t v = 0; v < result->num_vregs; v++) {
            tu_vreg_t *vr = &result->vregs[v];
            if (!vr->spilled) continue;
            if (vr->first_def == (int32_t)i) {
                /* Need to fill before first definition? No — the def IS the fill.
                 * For a spilled VReg, the "definition" is actually a DMA_LOAD fill. */
            }
            /* If this instruction reads a spilled VReg, insert fill */
            if (vr->first_def < (int32_t)i && i <= (uint32_t)vr->last_use) {
                /* Insert fill DMA */
                if (out_idx < TU_SCHED_MAX_INSTRS * 2 - 1) {
                    output->instructions[out_idx++] =
                        make_fill_instr(vr->physical_offset, vr->size_bytes, vr->region);
                }
            }
        }

        /* Patch and emit the instruction */
        patch_instruction(&instr, result, (int32_t)i);
        output->instructions[out_idx++] = instr;

        /* Update SRAM usage tracking */
        tu_vreg_t *def = find_def_vreg(result, (int32_t)i);
        if (def && def->physical_offset != UINT32_MAX) {
            switch (def->region) {
                case TU_VREG_W:
                    if (def->physical_offset + def->size_bytes > w_usage)
                        w_usage = def->physical_offset + def->size_bytes;
                    break;
                case TU_VREG_A:
                    if (def->physical_offset + def->size_bytes > a_usage)
                        a_usage = def->physical_offset + def->size_bytes;
                    break;
                case TU_VREG_O:
                    if (def->physical_offset + def->size_bytes > o_usage)
                        o_usage = def->physical_offset + def->size_bytes;
                    break;
            }
        }

        /* Track peaks */
        if (w_usage > peak_w) peak_w = w_usage;
        if (a_usage > peak_a) peak_a = a_usage;
        if (o_usage > peak_o) peak_o = o_usage;

        /* Check if any VReg dies at this instruction (last_use == i) */
        for (uint32_t v = 0; v < result->num_vregs; v++) {
            tu_vreg_t *vr = &result->vregs[v];
            if (vr->spilled) continue;
            if (vr->last_use == (int32_t)i) {
                /* Free its physical space */
                switch (vr->region) {
                    case TU_VREG_W:
                        /* Simplification: don't reclaim W (weights persist) */
                        break;
                    case TU_VREG_A:
                        a_usage = 0; /* Simplification */
                        break;
                    case TU_VREG_O:
                        o_usage = 0; /* Simplification */
                        break;
                }
            }
        }

        /* If a VReg is spilled, insert spill DMA before it dies */
        for (uint32_t v = 0; v < result->num_vregs; v++) {
            tu_vreg_t *vr = &result->vregs[v];
            if (!vr->spilled) continue;
            if (vr->last_use == (int32_t)i && out_idx < TU_SCHED_MAX_INSTRS * 2 - 1) {
                output->instructions[out_idx++] =
                    make_spill_instr(vr->physical_offset, vr->size_bytes, vr->region);
            }
        }
    }

    output->num_instructions = out_idx;
    output->peak_w_usage = peak_w;
    output->peak_a_usage = peak_a;
    output->peak_o_usage = peak_o;
    output->valid = true;

    return 0;
}

/* ================================================================
 * Full Allocation Pass
 * ================================================================ */

int tu_live_allocate(const tu_instruction_t *instrs,
                      uint32_t n_instrs,
                      const tu_live_config_t *config,
                      tu_allocated_sequence_t *output) {
    tu_liveness_result_t result;

    /* Step 1: Liveness analysis */
    if (tu_live_analyze(instrs, n_instrs, &result) != 0)
        return -1;

    /* Step 2: Build interference graph */
    tu_live_build_interference(&result);

    /* Step 3: Color */
    tu_live_color(&result, config);

    /* Step 4: Apply */
    int rc = tu_live_apply(&result, instrs, n_instrs, config, output);

    /* Free interference matrices */
    for (int r = 0; r < 3; r++) {
        tu_interference_graph_t *g = region_graph(&result, (tu_vreg_region_t)r);
        if (g && g->interference) {
            free(g->interference);
            g->interference = NULL;
        }
    }

    return rc;
}

/* ================================================================
 * Debug Output
 * ================================================================ */

void tu_live_print_result(const tu_liveness_result_t *result) {
    printf("=== Liveness Analysis: %u virtual registers ===\n", result->num_vregs);
    for (uint32_t i = 0; i < result->num_vregs; i++) {
        const tu_vreg_t *v = &result->vregs[i];
        const char *region_names[] = {"W", "A", "O"};
        printf("  V%03u: %s-SRAM [%u..%u) (%u bytes) live [%d..%d] phys=0x%04x %s\n",
               v->id,
               region_names[v->region],
               0U, v->size_bytes, v->size_bytes,
               v->first_def, v->last_use,
               v->physical_offset,
               v->spilled ? "(SPILLED)" : "");
    }
    printf("  Spills: %u, Spill bytes: %u\n", result->num_spills, result->spill_bytes);
    printf("=== End Liveness ===\n");
}

void tu_live_print_interference(const tu_interference_graph_t *graph) {
    if (!graph || graph->num_vregs == 0) {
        printf("  (empty)\n");
        return;
    }
    uint32_t n = graph->num_vregs;
    printf("  Interference graph (%u VRegs):\n", n);
    for (uint32_t i = 0; i < n; i++) {
        printf("  V%03u interferes with: ", graph->vregs[i]->id);
        bool any = false;
        for (uint32_t j = 0; j < n; j++) {
            if (i != j && graph->interference[i * n + j]) {
                printf("V%03u ", graph->vregs[j]->id);
                any = true;
            }
        }
        if (!any) printf("(none)");
        printf("\n");
    }
}
