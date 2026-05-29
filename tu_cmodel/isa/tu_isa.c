/*
 * TU CModel — ISA Helper Functions
 * ==================================
 * Opcode metadata, flag decoding, category queries.
 */

#include "tu_isa.h"
#include <stdio.h>

/* ---- Opcode name table ---- */
static const char *opcode_names[] = {
    [TU_ISA_NOP]          = "NOP",
    [TU_ISA_HALT]         = "HALT",
    [TU_ISA_SYNC]         = "SYNC",
    [TU_ISA_BARRIER]      = "BARRIER",
    [TU_ISA_FENCE]        = "FENCE",
    [TU_ISA_WAIT]         = "WAIT",
    [TU_ISA_SIGNAL]       = "SIGNAL",
    [7]                   = "RESERVED_7",
    [8]                   = "RESERVED_8",
    [9]                   = "RESERVED_9",
    [10]                  = "RESERVED_10",
    [11]                  = "RESERVED_11",
    [12]                  = "RESERVED_12",
    [13]                  = "RESERVED_13",
    [14]                  = "RESERVED_14",
    [15]                  = "RESERVED_15",
    [TU_ISA_MMA]          = "MMA",
    [TU_ISA_MMA_BIAS]     = "MMA.BIAS",
    [TU_ISA_MMA_FUSED]    = "MMA.FUSED",
    [TU_ISA_CONV2D]       = "CONV2D",
    [TU_ISA_CONV3D]       = "CONV3D",
    [TU_ISA_DEPTHWISE_CONV] = "CONV.DEPTHWISE",
    [TU_ISA_TRANSPOSED_CONV] = "CONV.TRANSPOSED",
    [TU_ISA_ATTENTION]    = "ATTENTION",
    [TU_ISA_ATTN_QK]      = "ATTN.QK",
    [TU_ISA_ATTN_PV]      = "ATTN.PV",
    [TU_ISA_ELEMENTWISE]  = "ELEMENTWISE",
    [TU_ISA_ADD]          = "ADD",
    [TU_ISA_MUL]          = "MUL",
    [TU_ISA_RELU]         = "RELU",
    [TU_ISA_GELU]         = "GELU",
    [TU_ISA_SILU]         = "SILU",
    [TU_ISA_TANH]         = "TANH",
    [TU_ISA_SIGMOID]      = "SIGMOID",
    [TU_ISA_EXP]          = "EXP",
    [TU_ISA_SCALE]        = "SCALE",
    [TU_ISA_REDUCE_SUM]   = "REDUCE.SUM",
    [TU_ISA_REDUCE_MAX]   = "REDUCE.MAX",
    [TU_ISA_REDUCE_MEAN]  = "REDUCE.MEAN",
    [TU_ISA_SOFTMAX]      = "SOFTMAX",
    [TU_ISA_LOG_SOFTMAX]  = "LOG_SOFTMAX",
    [TU_ISA_LAYER_NORM]   = "LAYER_NORM",
    [TU_ISA_RMS_NORM]     = "RMS_NORM",
    [TU_ISA_BATCH_NORM]   = "BATCH_NORM",
    [TU_ISA_GROUP_NORM]   = "GROUP_NORM",
    [TU_ISA_POOL_MAX]     = "POOL.MAX",
    [TU_ISA_POOL_AVG]     = "POOL.AVG",
    [TU_ISA_POOL_GLOBAL_AVG] = "POOL.GLOBAL_AVG",
    [TU_ISA_DMA_LOAD]     = "DMA.LOAD",
    [TU_ISA_DMA_STORE]    = "DMA.STORE",
    [TU_ISA_DMA_CHAIN]    = "DMA.CHAIN",
    [TU_ISA_DMA_LOAD_STRIDED]  = "DMA.LOAD.STRIDED",
    [TU_ISA_DMA_STORE_STRIDED] = "DMA.STORE.STRIDED",
    [TU_ISA_DMA_SCATTER]  = "DMA.SCATTER",
    [TU_ISA_DMA_GATHER]   = "DMA.GATHER",
    [TU_ISA_DMA_BROADCAST]= "DMA.BROADCAST",
    [TU_ISA_TRANSPOSE]    = "TRANSPOSE",
    [TU_ISA_PERMUTE]      = "PERMUTE",
    [TU_ISA_RESHAPE]      = "RESHAPE",
    [TU_ISA_SLICE]        = "SLICE",
    [TU_ISA_CONCAT]       = "CONCAT",
    [TU_ISA_PAD]          = "PAD",
    [TU_ISA_TILE]         = "TILE",
    [TU_ISA_SPARSE_MMA]   = "SPARSE_MMA",
    [TU_ISA_DECOMPRESS]   = "DECOMPRESS",
    [TU_ISA_COMPRESS]     = "COMPRESS",
    [TU_ISA_SET_CONFIG]   = "SET_CONFIG",
    [TU_ISA_GET_CONFIG]   = "GET_CONFIG",
};

const char *tu_isa_opcode_name(tu_isa_opcode_t opcode) {
    if (opcode >= TU_ISA_OPCODE_COUNT || !opcode_names[opcode]) {
        return "UNKNOWN";
    }
    return opcode_names[opcode];
}

/* ---- Category mapping ---- */
tu_isa_category_t tu_isa_opcode_category(tu_isa_opcode_t opcode) {
    if (opcode <= TU_ISA_SIGNAL)
        return TU_ISA_CAT_CONTROL;
    if (opcode >= TU_ISA_MMA && opcode <= TU_ISA_ATTN_PV)
        return TU_ISA_CAT_MATRIX;
    if (opcode >= TU_ISA_ELEMENTWISE && opcode <= TU_ISA_SCALE)
        return TU_ISA_CAT_ELEMENTWISE;
    if (opcode >= TU_ISA_REDUCE_SUM && opcode <= TU_ISA_GROUP_NORM)
        return TU_ISA_CAT_NORM_REDUCE;
    if (opcode >= TU_ISA_POOL_MAX && opcode <= TU_ISA_POOL_GLOBAL_AVG)
        return TU_ISA_CAT_POOLING;
    if (opcode >= TU_ISA_DMA_LOAD && opcode <= TU_ISA_DMA_BROADCAST)
        return TU_ISA_CAT_DATA_MOVEMENT;
    if (opcode >= TU_ISA_TRANSPOSE && opcode <= TU_ISA_TILE)
        return TU_ISA_CAT_DATA_LAYOUT;
    if (opcode >= TU_ISA_SPARSE_MMA && opcode <= TU_ISA_COMPRESS)
        return TU_ISA_CAT_SPARSITY;
    if (opcode >= TU_ISA_SET_CONFIG && opcode <= TU_ISA_GET_CONFIG)
        return TU_ISA_CAT_CONFIG;
    return TU_ISA_CAT_UNKNOWN;
}

/* ---- Operation queries ---- */
bool tu_isa_has_sram_operands(tu_isa_opcode_t opcode) {
    switch (opcode) {
        case TU_ISA_NOP:
        case TU_ISA_HALT:
        case TU_ISA_SYNC:
        case TU_ISA_BARRIER:
        case TU_ISA_FENCE:
        case TU_ISA_WAIT:
        case TU_ISA_SIGNAL:
        case TU_ISA_SET_CONFIG:
        case TU_ISA_GET_CONFIG:
            return false;
        default:
            return true;
    }
}

bool tu_isa_is_compute_op(tu_isa_opcode_t opcode) {
    tu_isa_category_t cat = tu_isa_opcode_category(opcode);
    return cat == TU_ISA_CAT_MATRIX ||
           cat == TU_ISA_CAT_ELEMENTWISE ||
           cat == TU_ISA_CAT_NORM_REDUCE ||
           cat == TU_ISA_CAT_POOLING ||
           cat == TU_ISA_CAT_SPARSITY;
}

bool tu_isa_is_dma_op(tu_isa_opcode_t opcode) {
    return tu_isa_opcode_category(opcode) == TU_ISA_CAT_DATA_MOVEMENT;
}

/* ---- Flag decoding ---- */
void tu_isa_decode_flags(uint8_t flags,
                          uint8_t *precision_out,
                          uint8_t *transpose_out,
                          uint8_t *activation_out,
                          bool *has_bias_out) {
    if (precision_out)  *precision_out  = (flags & TU_FLAG_PREC_MASK);
    if (transpose_out)  *transpose_out  = (flags & TU_FLAG_TRANSPOSE_MASK) >> 3;
    if (activation_out) *activation_out = (flags & TU_FLAG_ACT_MASK) >> 5;
    if (has_bias_out)   *has_bias_out   = (flags & TU_FLAG_BIAS) != 0;
}
