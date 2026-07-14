/*
 * Scheduler Policy Sweep: ASAP vs ALAP vs BALANCED
 * =================================================
 * Compares the three scheduling policies across 5 workload topologies.
 * DMA channel: flags & 0x3 → 0=W, 1=A, 2=O
 * MMA: reads W+A, writes O. dim0=w_off, dim1=N, dim2=K.
 *
 * Metrics: estimated_cycles, barriers_inserted, DMA_hoisted, schedule_length
 */

#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/isa/tu_isa.h"
#include "../tu_cmodel/isa/tu_scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Helper: build an instruction ---- */
static tu_instruction_t I(tu_isa_opcode_t op, uint16_t dim0,
                           uint16_t dim1, uint16_t dim2,
                           uint8_t flags, uint32_t imm) {
    tu_instruction_t instr;
    memset(&instr, 0, sizeof(instr));
    instr.opcode = op;
    instr.flags = flags;
    instr.dim0 = dim0;
    instr.dim1 = dim1;
    instr.dim2 = dim2;
    instr.immediates = imm;
    return instr;
}

static const char *policy_name(tu_sched_policy_t p) {
    switch (p) {
        case TU_SCHED_POLICY_ASAP:     return "ASAP";
        case TU_SCHED_POLICY_ALAP:     return "ALAP";
        case TU_SCHED_POLICY_BALANCED: return "BALANCED";
        default: return "???";
    }
}

static int run_one(const char *label, tu_instruction_t *instrs, int n,
                    tu_sched_policy_t pol,
                    uint32_t *cycles, uint32_t *barriers, uint32_t *hoisted,
                    uint32_t *sched_len) {
    tu_sched_config_t cfg = tu_sched_config_default;
    cfg.policy = pol;

    tu_sched_result_t result;
    memset(&result, 0, sizeof(result));

    int rc = tu_sched_run(instrs, n, &cfg, &result);
    if (rc != 0) {
        printf("  %-20s %-8s: ERROR rc=%d\n", label, policy_name(pol), rc);
        return -1;
    }

    *cycles   = result.estimated_cycles;
    *barriers = result.num_barriers_inserted;
    *hoisted  = result.num_dma_hoisted;
    *sched_len = result.num_instructions;
    return 0;
}

static void print_row(const char *label, const char *pol,
                      uint32_t cycles, uint32_t barriers,
                      uint32_t hoisted, uint32_t sched_len) {
    printf("  %-20s %-8s  %9u  %7u  %6u  %6u\n",
           label, pol, cycles, barriers, hoisted, sched_len);
}

int main(void) {
    printf("\n=== TU Scheduler Policy Sweep ===\n\n");
    printf("Comparing ASAP, ALAP, BALANCED across 5 workload topologies.\n");
    printf("DMA channel: flags&3 -> 0=W,1=A,2=O.  MMA: reads W+A, writes O.\n\n");

    printf("%-20s %-8s  %9s  %7s  %6s  %6s\n",
           "Topology", "Policy", "Cycles", "Barrier", "Hoist", "Len");
    printf("%-20s %-8s  %9s  %7s  %6s  %6s\n",
           "--------", "------", "------", "-------", "-----", "---");

    uint32_t cy, ba, ho, sl;

    /* --- Workload 1: All-Independent (4 NOPs, no SRAM deps) --- */
    {
        tu_instruction_t ins[] = {
            I(TU_ISA_NOP, 0,0,0, 0,0),
            I(TU_ISA_NOP, 0,0,0, 0,0),
            I(TU_ISA_NOP, 0,0,0, 0,0),
            I(TU_ISA_NOP, 0,0,0, 0,0),
        };
        int n = 4;
        run_one("All-Independent", ins, n, TU_SCHED_POLICY_ASAP, &cy,&ba,&ho,&sl);
        print_row("All-Independent", "ASAP", cy,ba,ho,sl);
        run_one("All-Independent", ins, n, TU_SCHED_POLICY_ALAP, &cy,&ba,&ho,&sl);
        print_row("All-Independent", "ALAP", cy,ba,ho,sl);
        run_one("All-Independent", ins, n, TU_SCHED_POLICY_BALANCED, &cy,&ba,&ho,&sl);
        print_row("All-Independent", "BALANCED", cy,ba,ho,sl);
    }

    /* --- Workload 2: Serial Chain ---
     * DMA_LOAD(W, off=0) → MMA(w_off=0, N=16, K=16) → DMA_STORE(O, off=0)
     * RAW: DMA_W→MMA (W), MMA→DMA_O (O)
     */
    {
        tu_instruction_t ins[] = {
            I(TU_ISA_DMA_LOAD,   0,512,0,   0, 0),   /* 0: load W@0, 512 bytes, ch=0 */
            I(TU_ISA_MMA,        0,16, 16,  0, 0),   /* 1: MMA w_off=0, N=16, K=16 */
            I(TU_ISA_DMA_STORE,  0,256,0,   2, 0),   /* 2: store O@0, 256 bytes, ch=2 */
            I(TU_ISA_HALT,       0,0,0,     0, 0),
        };
        int n = 4;
        run_one("Serial-Chain", ins, n, TU_SCHED_POLICY_ASAP, &cy,&ba,&ho,&sl);
        print_row("Serial-Chain", "ASAP", cy,ba,ho,sl);
        run_one("Serial-Chain", ins, n, TU_SCHED_POLICY_ALAP, &cy,&ba,&ho,&sl);
        print_row("Serial-Chain", "ALAP", cy,ba,ho,sl);
        run_one("Serial-Chain", ins, n, TU_SCHED_POLICY_BALANCED, &cy,&ba,&ho,&sl);
        print_row("Serial-Chain", "BALANCED", cy,ba,ho,sl);
    }

    /* --- Workload 3: Fan-Out ---
     * One DMA_LOAD(W) loads a large W tile. 4 MMA ops all read the same W
     * but write to DIFFERENT O offsets → MMAs independent of each other.
     * Only RAW dep: DMA→each MMA.
     * ASAP should hoist DMA early; ALAP should delay non-critical ops.
     */
    {
        tu_instruction_t ins[] = {
            I(TU_ISA_DMA_LOAD,  0,2048,0,   0, 0),     /* 0: load W@0, 2KB, ch=0 */
            I(TU_ISA_MMA,       0,16, 16,   0, 0),     /* 1: MMA #1, o_off=0 */
            I(TU_ISA_MMA,       0,16, 16,   0, (16*16*4) << 16),  /* 2: MMA #2, o_off=1024 */
            I(TU_ISA_MMA,       0,16, 16,   0, (32*16*4) << 16),  /* 3: MMA #3, o_off=2048 */
            I(TU_ISA_MMA,       0,16, 16,   0, (48*16*4) << 16),  /* 4: MMA #4, o_off=3072 */
            I(TU_ISA_HALT,      0,0,0,     0, 0),
        };
        int n = 6;
        run_one("Fan-Out", ins, n, TU_SCHED_POLICY_ASAP, &cy,&ba,&ho,&sl);
        print_row("Fan-Out", "ASAP", cy,ba,ho,sl);
        run_one("Fan-Out", ins, n, TU_SCHED_POLICY_ALAP, &cy,&ba,&ho,&sl);
        print_row("Fan-Out", "ALAP", cy,ba,ho,sl);
        run_one("Fan-Out", ins, n, TU_SCHED_POLICY_BALANCED, &cy,&ba,&ho,&sl);
        print_row("Fan-Out", "BALANCED", cy,ba,ho,sl);
    }

    /* --- Workload 4: Fan-In ---
     * 4 DMA_LOAD ops (2 to W, 2 to A) converge to 1 MMA.
     * DMA use different offsets to avoid WAW conflicts.
     * All 4 DMA → MMA RAW dependency.
     * Heavy barrier insertion expected.
     */
    {
        tu_instruction_t ins[] = {
            I(TU_ISA_DMA_LOAD,  0,1024,0,   0, 0),   /* 0: load W@0, ch=0 */
            I(TU_ISA_DMA_LOAD,  1024,512,0,  1, 0),   /* 1: load A@1024, ch=1 */
            I(TU_ISA_DMA_LOAD,  2048,1024,0, 0, 0),   /* 2: load W@2048, ch=0 */
            I(TU_ISA_DMA_LOAD,  3072,512,0,  1, 0),   /* 3: load A@3072, ch=1 */
            I(TU_ISA_MMA,       0,16, 16,   0, 0),    /* 4: MMA w_off=0 */
            I(TU_ISA_HALT,      0,0,0,     0, 0),
        };
        int n = 6;
        run_one("Fan-In", ins, n, TU_SCHED_POLICY_ASAP, &cy,&ba,&ho,&sl);
        print_row("Fan-In", "ASAP", cy,ba,ho,sl);
        run_one("Fan-In", ins, n, TU_SCHED_POLICY_ALAP, &cy,&ba,&ho,&sl);
        print_row("Fan-In", "ALAP", cy,ba,ho,sl);
        run_one("Fan-In", ins, n, TU_SCHED_POLICY_BALANCED, &cy,&ba,&ho,&sl);
        print_row("Fan-In", "BALANCED", cy,ba,ho,sl);
    }

    /* --- Workload 5: Pipeline Tiles (4 DMA+MMA pairs) ---
     * Tile pipeline: DMA_tileN → MMA_tileN → DMA_tileN+1 → MMA_tileN+1 ...
     * Each tile uses distinct W/A/O offsets.
     * BALANCED should interleave tileN+1 DMA with tileN MMA.
     */
    {
        tu_instruction_t ins[] = {
            /* Tile 0 */
            I(TU_ISA_DMA_LOAD,  0,512,0,   0, 0),   /* 0: W@0 */
            I(TU_ISA_DMA_LOAD,  4096,512,0, 1, 0),   /* 1: A@4096 */
            I(TU_ISA_MMA,       0,16,16,   0, 100<<16),  /* 2: MMA O@100 */
            /* Tile 1 */
            I(TU_ISA_DMA_LOAD,  1024,512,0, 0, 0),   /* 3: W@1024 */
            I(TU_ISA_DMA_LOAD,  4608,512,0, 1, 0),   /* 4: A@4608 */
            I(TU_ISA_MMA,       1024,16,16, 0, 200<<16), /* 5: MMA O@200 */
            /* Tile 2 */
            I(TU_ISA_DMA_LOAD,  2048,512,0, 0, 0),   /* 6: W@2048 */
            I(TU_ISA_DMA_LOAD,  5120,512,0, 1, 0),   /* 7: A@5120 */
            I(TU_ISA_MMA,       2048,16,16, 0, 300<<16), /* 8: MMA O@300 */
            /* Tile 3 */
            I(TU_ISA_DMA_LOAD,  3072,512,0, 0, 0),   /* 9: W@3072 */
            I(TU_ISA_DMA_LOAD,  5632,512,0, 1, 0),   /* 10: A@5632 */
            I(TU_ISA_MMA,       3072,16,16, 0, 400<<16), /* 11: MMA O@400 */
            I(TU_ISA_HALT,      0,0,0,    0, 0),
        };
        int n = 13;
        run_one("Pipeline-Tiles", ins, n, TU_SCHED_POLICY_ASAP, &cy,&ba,&ho,&sl);
        print_row("Pipeline-Tiles", "ASAP", cy,ba,ho,sl);
        run_one("Pipeline-Tiles", ins, n, TU_SCHED_POLICY_ALAP, &cy,&ba,&ho,&sl);
        print_row("Pipeline-Tiles", "ALAP", cy,ba,ho,sl);
        run_one("Pipeline-Tiles", ins, n, TU_SCHED_POLICY_BALANCED, &cy,&ba,&ho,&sl);
        print_row("Pipeline-Tiles", "BALANCED", cy,ba,ho,sl);
    }

    printf("\n=== Sweep Complete ===\n\n");
    return 0;
}
