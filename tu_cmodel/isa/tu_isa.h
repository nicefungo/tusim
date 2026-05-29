/*
 * TU CModel — Expanded ISA Definitions
 * =====================================
 * Full instruction set with 30+ opcodes, fixed-width binary encoding,
 * operation descriptors, and text format extensions.
 *
 * Gap C1: 6 instructions → 30+ instruction production ISA.
 *
 * Architecture:
 *   The TU ISA is a fixed-width 96-bit instruction set designed for
 *   systolic/vector accelerators. It covers matrix operations (MMA,
 *   convolution, attention), elementwise/activation, normalization,
 *   data movement (DMA load/store, scatter/gather), synchronization
 *   (barrier, fence, sync), and configuration.
 *
 * Encoding:
 *   [7:0]   opcode       — operation code
 *   [15:8]  flags        — precision, transpose, activation, bias flags
 *   [31:16] dim0         — context-dependent (M, in_channels, seq_len, etc.)
 *   [47:32] dim1         — context-dependent (N, out_channels, head_dim, etc.)
 *   [63:48] dim2         — context-dependent (K, kernel_size, num_heads, etc.)
 *   [95:64] immediates   — addresses, offsets, strides, scales
 *
 *   This is the canonical encoding. Text format (.tuasm) is a human-readable
 *   surface syntax; binary is the machine format.
 */

#ifndef TU_ISA_H
#define TU_ISA_H

#include "../tu_config.h"
#include "../tu_precision.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Opcode Catalog (30+ instructions)
 * ================================================================ */

typedef enum {
    /* ── Control (0x00–0x0F) ── */
    TU_ISA_NOP        = 0x00,  /* No operation */
    TU_ISA_HALT       = 0x01,  /* Stop execution */
    TU_ISA_SYNC       = 0x02,  /* Drain pipeline, sync all units */
    TU_ISA_BARRIER    = 0x03,  /* Ordering barrier */
    TU_ISA_FENCE      = 0x04,  /* Memory fence (all prior DMA complete) */
    TU_ISA_WAIT       = 0x05,  /* Wait for signal/event */
    TU_ISA_SIGNAL     = 0x06,  /* Fire signal/event */

    /* ── Matrix Operations (0x10–0x1F) ── */
    TU_ISA_MMA        = 0x10,  /* Matrix multiply-accumulate: O += W × A */
    TU_ISA_MMA_BIAS   = 0x11,  /* MMA with bias: O = bias + W × A */
    TU_ISA_MMA_FUSED  = 0x12,  /* MMA + fused activation */
    TU_ISA_CONV2D     = 0x13,  /* 2D convolution */
    TU_ISA_CONV3D     = 0x14,  /* 3D convolution */
    TU_ISA_DEPTHWISE_CONV = 0x15,  /* Depthwise convolution */
    TU_ISA_TRANSPOSED_CONV = 0x16,  /* Transposed convolution */
    TU_ISA_ATTENTION  = 0x17,  /* Q×K^T + Softmax + ×V (fused) */
    TU_ISA_ATTN_QK    = 0x18,  /* Attention: Q×K^T only */
    TU_ISA_ATTN_PV    = 0x19,  /* Attention: P×V (after softmax) */

    /* ── Elementwise & Activation (0x20–0x2F) ── */
    TU_ISA_ELEMENTWISE = 0x20,  /* Generic elementwise (opcode in flags) */
    TU_ISA_ADD        = 0x21,  /* Elementwise add: C = A + B */
    TU_ISA_MUL        = 0x22,  /* Elementwise multiply: C = A × B */
    TU_ISA_RELU       = 0x23,  /* ReLU: C = max(0, A) */
    TU_ISA_GELU       = 0x24,  /* GELU activation */
    TU_ISA_SILU       = 0x25,  /* SiLU / Swish activation */
    TU_ISA_TANH       = 0x26,  /* Hyperbolic tangent */
    TU_ISA_SIGMOID    = 0x27,  /* Sigmoid */
    TU_ISA_EXP        = 0x28,  /* Exponential */
    TU_ISA_SCALE      = 0x29,  /* C = A × scalar */

    /* ── Reduction & Normalization (0x30–0x3F) ── */
    TU_ISA_REDUCE_SUM = 0x30,  /* Sum reduction along axis */
    TU_ISA_REDUCE_MAX = 0x31,  /* Max reduction along axis */
    TU_ISA_REDUCE_MEAN= 0x32,  /* Mean reduction along axis */
    TU_ISA_SOFTMAX    = 0x33,  /* Online softmax */
    TU_ISA_LOG_SOFTMAX= 0x34,  /* Log-softmax */
    TU_ISA_LAYER_NORM = 0x35,  /* Layer normalization */
    TU_ISA_RMS_NORM   = 0x36,  /* RMS normalization */
    TU_ISA_BATCH_NORM = 0x37,  /* Batch normalization */
    TU_ISA_GROUP_NORM = 0x38,  /* Group normalization */

    /* ── Pooling (0x40–0x4F) ── */
    TU_ISA_POOL_MAX   = 0x40,  /* Max pooling */
    TU_ISA_POOL_AVG   = 0x41,  /* Average pooling */
    TU_ISA_POOL_GLOBAL_AVG = 0x42,  /* Global average pooling */

    /* ── Data Movement (0x50–0x5F) ── */
    TU_ISA_DMA_LOAD   = 0x50,  /* DRAM → SRAM load */
    TU_ISA_DMA_STORE  = 0x51,  /* SRAM → DRAM store */
    TU_ISA_DMA_CHAIN  = 0x52,  /* Execute descriptor chain */
    TU_ISA_DMA_LOAD_STRIDED = 0x53,  /* 2D/3D strided load */
    TU_ISA_DMA_STORE_STRIDED = 0x54, /* 2D/3D strided store */
    TU_ISA_DMA_SCATTER = 0x55, /* Scatter via index list */
    TU_ISA_DMA_GATHER  = 0x56, /* Gather via index list */
    TU_ISA_DMA_BROADCAST = 0x57, /* 1-to-N broadcast */

    /* ── Data Layout (0x60–0x6F) ── */
    TU_ISA_TRANSPOSE  = 0x60,  /* 2D transpose */
    TU_ISA_PERMUTE    = 0x61,  /* N-D permute */
    TU_ISA_RESHAPE    = 0x62,  /* Logical reshape (no data movement) */
    TU_ISA_SLICE      = 0x63,  /* Extract slice */
    TU_ISA_CONCAT     = 0x64,  /* Concatenate along axis */
    TU_ISA_PAD        = 0x65,  /* Pad tensor */
    TU_ISA_TILE       = 0x66,  /* Tile/repeat tensor */

    /* ── Sparsity (0x70–0x7F) ── */
    TU_ISA_SPARSE_MMA = 0x70,  /* Sparse MMA (2:4 structured) */
    TU_ISA_DECOMPRESS = 0x71,  /* Decompress sparse weights */
    TU_ISA_COMPRESS   = 0x72,  /* Compress weights */

    /* ── Configuration (0x7E–0x7F) ── */
    TU_ISA_SET_CONFIG = 0x7E,  /* Runtime configuration update */
    TU_ISA_GET_CONFIG = 0x7F,  /* Read configuration register */

    TU_ISA_OPCODE_COUNT
} tu_isa_opcode_t;

/* ================================================================
 * Flag Encoding (bits [15:8] of instruction word)
 * ================================================================ */

/* Precision flags (bits [2:0]) */
#define TU_FLAG_PREC_FP16     0x0
#define TU_FLAG_PREC_FP32     0x1
#define TU_FLAG_PREC_BF16     0x2
#define TU_FLAG_PREC_FP8_E4M3 0x3
#define TU_FLAG_PREC_FP8_E5M2 0x4
#define TU_FLAG_PREC_INT8     0x5
#define TU_FLAG_PREC_INT4     0x6
#define TU_FLAG_PREC_MASK     0x7

/* Transpose flags (bits [4:3]) */
#define TU_FLAG_TRANSPOSE_NONE  (0x0 << 3)
#define TU_FLAG_TRANSPOSE_A     (0x1 << 3)
#define TU_FLAG_TRANSPOSE_B     (0x2 << 3)
#define TU_FLAG_TRANSPOSE_BOTH  (0x3 << 3)
#define TU_FLAG_TRANSPOSE_MASK  (0x3 << 3)

/* Activation flags (bits [6:5]) */
#define TU_FLAG_ACT_NONE     (0x0 << 5)
#define TU_FLAG_ACT_RELU     (0x1 << 5)
#define TU_FLAG_ACT_GELU     (0x2 << 5)
#define TU_FLAG_ACT_SILU     (0x3 << 5)
#define TU_FLAG_ACT_MASK     (0x3 << 5)

/* Bias flag (bit [7]) */
#define TU_FLAG_BIAS         (1 << 7)

/* ================================================================
 * Instruction Encoding (96-bit fixed-width)
 * ================================================================ */

typedef struct __attribute__((packed)) {
    uint8_t   opcode;          /* [7:0]   — operation code */
    uint8_t   flags;           /* [15:8]  — precision, transpose, activation, bias */
    uint16_t  dim0;            /* [31:16] — dimension 0 (context-dependent) */
    uint16_t  dim1;            /* [47:32] — dimension 1 */
    uint16_t  dim2;            /* [63:48] — dimension 2 */
    uint32_t  immediates;      /* [95:64] — addresses, offsets, scales */
} tu_instruction_t;

/* Verify size */
_Static_assert(sizeof(tu_instruction_t) == 12, "tu_instruction_t must be 12 bytes (96 bits)");

/* ================================================================
 * Operation Descriptors (decoded forms for execution)
 * ================================================================ */

/* ---- MMA Descriptor ---- */
typedef struct {
    uint32_t  w_offset;        /* Weight buffer offset */
    uint32_t  a_offset;        /* Activation buffer offset */
    uint32_t  o_offset;        /* Output buffer offset */
    uint32_t  bias_offset;     /* Bias buffer offset (0 = no bias) */
    uint16_t  M, N, K;         /* Matrix dimensions */
    uint8_t   precision;       /* Input precision */
    bool      transpose_a;
    bool      transpose_b;
    bool      has_bias;
    bool      has_activation;
    uint8_t   activation;      /* Activation type (relu, gelu, etc.) */
    float     alpha;           /* Scaling factor (for A*alpha*B) */
} tu_mma_op_desc_t;

/* ---- Convolution Descriptor ---- */
typedef struct {
    uint32_t  input_offset;    /* Input tensor offset (NCHW) */
    uint32_t  weight_offset;   /* Weight tensor offset (KCRS) */
    uint32_t  output_offset;   /* Output tensor offset */
    uint32_t  bias_offset;
    uint16_t  N;               /* Batch size */
    uint16_t  C_in;            /* Input channels */
    uint16_t  C_out;           /* Output channels */
    uint16_t  H, W;            /* Input spatial dims */
    uint16_t  R, S;            /* Kernel spatial dims */
    uint16_t  stride_h, stride_w;
    uint16_t  pad_t, pad_b, pad_l, pad_r;
    uint16_t  dilation_h, dilation_w;
    uint16_t  groups;
    bool      has_bias;
    bool      has_activation;
    uint8_t   activation;
} tu_conv_op_desc_t;

/* ---- Attention Descriptor ---- */
typedef struct {
    uint32_t  q_offset;        /* Query tensor */
    uint32_t  k_offset;        /* Key tensor */
    uint32_t  v_offset;        /* Value tensor */
    uint32_t  o_offset;        /* Output tensor */
    uint32_t  mask_offset;     /* Attention mask (0 = none) */
    uint16_t  batch_size;
    uint16_t  num_heads;
    uint16_t  seq_len_q;
    uint16_t  seq_len_kv;
    uint16_t  head_dim;
    float     softmax_scale;
    bool      causal;
    bool      has_mask;
} tu_attention_op_desc_t;

/* ---- Elementwise Descriptor ---- */
typedef enum {
    TU_EW_OP_ADD = 0,
    TU_EW_OP_SUB,
    TU_EW_OP_MUL,
    TU_EW_OP_DIV,
    TU_EW_OP_MIN,
    TU_EW_OP_MAX,
} tu_ew_binary_op_t;

typedef struct {
    uint32_t  a_offset;
    uint32_t  b_offset;        /* 0 = unary operation */
    uint32_t  o_offset;
    uint32_t  num_elements;
    bool      is_unary;
    union {
        tu_ew_binary_op_t binary_op;
        uint8_t           activation; /* relu, gelu, silu, tanh, sigmoid, exp */
    };
    float     scalar;         /* For scale operations */
} tu_ew_op_desc_t;

/* ---- Normalization Descriptor ---- */
typedef struct {
    uint32_t  input_offset;
    uint32_t  weight_offset;   /* Gamma (0 = none) */
    uint32_t  bias_offset;     /* Beta (0 = none) */
    uint32_t  output_offset;
    uint32_t  num_elements;
    uint32_t  normalized_shape[4];
    uint8_t   num_axes;
    float     epsilon;
} tu_norm_op_desc_t;

/* ---- Softmax Descriptor ---- */
typedef struct {
    uint32_t  input_offset;
    uint32_t  output_offset;
    uint32_t  num_elements;
    uint32_t  axis_size;       /* Size of dimension to normalize over */
    bool      log_softmax;
} tu_softmax_op_desc_t;

/* ---- Pooling Descriptor ---- */
typedef struct {
    uint32_t  input_offset;
    uint32_t  output_offset;
    uint16_t  H, W;            /* Input spatial */
    uint16_t  C;               /* Channels */
    uint16_t  kernel_h, kernel_w;
    uint16_t  stride_h, stride_w;
    uint16_t  pad_t, pad_b, pad_l, pad_r;
    bool      is_avg;
    bool      is_global;
} tu_pool_op_desc_t;

/* ---- DMA Descriptor ---- */
typedef struct {
    uint32_t  host_addr_lo;    /* Host address (low 32 bits) */
    uint32_t  host_addr_hi;    /* Host address (high 32 bits) */
    uint32_t  sram_offset;     /* SRAM byte offset */
    uint32_t  size_bytes;      /* Transfer size */
    uint8_t   channel;         /* DMA channel */
    bool      is_store;        /* true = SRAM→host, false = host→SRAM */
    uint8_t   transfer_type;   /* linear, strided_2d, strided_3d, scatter, gather */
    uint16_t  stride_rows;     /* Row stride for strided xfers */
    uint16_t  stride_depth;    /* Depth stride for 3D xfers */
    uint16_t  rows, cols, depth; /* Transfer geometry */
} tu_dma_op_desc_t;

/* ---- Reduce Descriptor ---- */
typedef struct {
    uint32_t  input_offset;
    uint32_t  output_offset;
    uint32_t  num_elements;
    uint16_t  axis;            /* Reduction axis */
    bool      keep_dims;
    uint8_t   reduce_op;       /* sum, max, mean */
} tu_reduce_op_desc_t;

/* ---- Sparse MMA Descriptor ---- */
typedef struct {
    uint32_t  w_offset;        /* Compressed weight buffer offset */
    uint32_t  w_meta_offset;   /* Weight metadata (sparsity mask) */
    uint32_t  a_offset;
    uint32_t  o_offset;
    uint16_t  M, N, K;
    uint8_t   sparsity;        /* 2:4, unstructured, block-sparse */
    uint8_t   precision;
} tu_sparse_mma_op_desc_t;

/* ================================================================
 * Unified Operation Descriptor (union for dispatch)
 * ================================================================ */

typedef struct {
    tu_isa_opcode_t opcode;
    uint16_t        flags;      /* Decoded flags from instruction */
    union {
        tu_mma_op_desc_t       mma;
        tu_conv_op_desc_t      conv;
        tu_attention_op_desc_t attention;
        tu_ew_op_desc_t        elementwise;
        tu_norm_op_desc_t      norm;
        tu_softmax_op_desc_t   softmax;
        tu_pool_op_desc_t      pool;
        tu_dma_op_desc_t       dma;
        tu_reduce_op_desc_t    reduce;
        tu_sparse_mma_op_desc_t sparse_mma;
    };
} tu_op_descriptor_t;

/* ================================================================
 * ISA Helper Functions
 * ================================================================ */

/* Get human-readable opcode name */
const char *tu_isa_opcode_name(tu_isa_opcode_t opcode);

/* Get opcode category (control, compute, memory, etc.) */
typedef enum {
    TU_ISA_CAT_CONTROL,
    TU_ISA_CAT_MATRIX,
    TU_ISA_CAT_ELEMENTWISE,
    TU_ISA_CAT_NORM_REDUCE,
    TU_ISA_CAT_POOLING,
    TU_ISA_CAT_DATA_MOVEMENT,
    TU_ISA_CAT_DATA_LAYOUT,
    TU_ISA_CAT_SPARSITY,
    TU_ISA_CAT_CONFIG,
    TU_ISA_CAT_UNKNOWN,
} tu_isa_category_t;

tu_isa_category_t tu_isa_opcode_category(tu_isa_opcode_t opcode);

/* Check if opcode requires SRAM operands (has memory offsets) */
bool tu_isa_has_sram_operands(tu_isa_opcode_t opcode);

/* Check if opcode is a compute operation (consumes MAC cycles) */
bool tu_isa_is_compute_op(tu_isa_opcode_t opcode);

/* Check if opcode is a DMA/data-movement operation */
bool tu_isa_is_dma_op(tu_isa_opcode_t opcode);

/* Decode flags from instruction word */
void tu_isa_decode_flags(uint8_t flags,
                          uint8_t *precision_out,
                          uint8_t *transpose_out,
                          uint8_t *activation_out,
                          bool *has_bias_out);

#ifdef __cplusplus
}
#endif

#endif /* TU_ISA_H */
