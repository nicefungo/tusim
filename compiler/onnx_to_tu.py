#!/usr/bin/env python3
"""
onnx_to_tu.py — Compile ONNX model → RISC-V host + TinyTU accelerator code.

TU MMA semantics:
    O[N][M] += W[N][K] × A[K][M]      (weight-stationary systolic array)

ONNX Gemm mapping:
    Y[M][N] = X[M][K] × B[K][N]
    → W = B^T [N][K] in W-buffer   (weights, stationary)
    → A = X^T [K][M] in A-buffer   (activations, streamed)
    → O = Y^T [N][M] in O-buffer   (output, transposed)

When weights exceed W-buffer (128 KB), K-dimension tiling splits the
weight along K into tiles of [N][K_tile].

Usage:
  python3 onnx_to_tu.py <model.onnx> [--output out.c] [--name model_name]
"""

import sys
import os
import struct
import argparse
from collections import OrderedDict
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# ================================================================
# Constants
# ================================================================

TU_PE_ROWS = 16
TU_PE_COLS = 16
SRAM_W_SIZE = 128 * 1024   # weight buffer (holds B^T for stationary)
SRAM_A_SIZE =  64 * 1024   # activation buffer (holds X^T)
SRAM_O_SIZE =  64 * 1024   # output buffer (FP32 accumulators)
FP16_BYTES  = 2
FP32_BYTES  = 4

ONNX_FLOAT   = 1
ONNX_FLOAT16 = 10

# ================================================================
# FP16 helpers
# ================================================================

def fp32_to_fp16_bytes(arr_fp32):
    """Convert numpy float32 array to bytes of float16."""
    arr = np.asarray(arr_fp32, dtype=np.float32).ravel()
    result = bytearray()
    for v in arr:
        result.extend(struct.pack('<H', _float_to_half(v)))
    return bytes(result)

def _float_to_half(f):
    """IEEE 754 float32 → float16."""
    bits = struct.unpack('<I', struct.pack('<f', f))[0]
    sign = (bits >> 31) & 1
    exp  = ((bits >> 23) & 0xFF) - 127
    mant = bits & 0x7FFFFF

    if exp > 15:
        return (sign << 15) | 0x7C00
    if exp < -25:
        return sign << 15

    if exp <= -15:
        shift = -14 - exp
        m = (mant | 0x800000) >> shift
        if shift > 0 and ((mant >> (shift - 1)) & 1):
            r = (mant >> (shift - 1)) & 1
            s = 0 if shift <= 1 else (mant & ((1 << (shift - 1)) - 1)) != 0
            if r and (s or (m & 1)):
                m += 1
        return (sign << 15) | (m & 0x3FF)
    else:
        e = exp + 15
        r = (mant >> 12) & 1
        s = (mant & 0xFFF) != 0
        m = mant >> 13
        if r and (s or (m & 1)):
            m += 1
        if m >= 0x400:
            m = 0
            e += 1
        if e > 31:
            return (sign << 15) | 0x7C00
        return (sign << 15) | (e << 10) | (m & 0x3FF)


# ================================================================
# SRAM Bump Allocator
# ================================================================

class SRAMAllocator:
    def __init__(self, total_size, name):
        self.total = total_size
        self.name = name
        self.offset = 0

    def alloc(self, size_bytes, align=16):
        off = (self.offset + align - 1) & ~(align - 1)
        if off + size_bytes > self.total:
            raise RuntimeError(
                f"TU {self.name}-buffer overflow: need {off + size_bytes}, have {self.total}"
            )
        self.offset = off + size_bytes
        return off

    def used(self):
        return self.offset


# ================================================================
# ONNX Graph Analysis
# ================================================================

def analyze_onnx_model(model):
    graph = model.graph
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    shape_map = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        shape = []
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_param:
                shape.append(d.dim_param)
            elif d.dim_value:
                shape.append(d.dim_value)
            else:
                shape.append(-1)
        shape_map[vi.name] = shape

    for name, arr in initializers.items():
        shape_map[name] = list(arr.shape)

    node_list = list(graph.node)
    producer_map = {}
    for node in node_list:
        for out in node.output:
            producer_map[out] = None if out in producer_map else node.name

    in_degree = {n.name: 0 for n in node_list}
    edges = {}
    for node in node_list:
        consumers = set()
        for inp in node.input:
            prod = producer_map.get(inp)
            if prod and prod != node.name:
                consumers.add(prod)
        edges[node.name] = consumers
        for c in consumers:
            if c in in_degree:
                in_degree[c] += 1

    queue = [n.name for n in node_list if in_degree[n.name] == 0]
    sorted_names = []
    while queue:
        name = queue.pop(0)
        sorted_names.append(name)
        for consumer in edges.get(name, set()):
            in_degree[consumer] -= 1
            if in_degree[consumer] == 0:
                queue.append(consumer)

    node_map = {n.name: n for n in node_list}
    ordered = []
    for name in sorted_names:
        n = node_map[name]
        ordered.append({
            'op': n.op_type,
            'name': n.name,
            'inputs': list(n.input),
            'outputs': list(n.output),
            'attrs': {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute},
        })

    input_names = [i.name for i in graph.input if i.name not in initializers]
    output_names = [o.name for o in graph.output]

    # ---- Shape inference pass ----
    # Topo sort is reverse (outputs first); reverse for forward propagation
    _infer_shapes(list(reversed(ordered)), initializers, shape_map)

    return ordered, initializers, shape_map, input_names, output_names


def _infer_shapes(ordered_nodes, initializers, shape_map):
    """Forward shape propagation for common ops."""
    for node in ordered_nodes:
        op = node['op']
        inputs = node['inputs']
        outputs = node['outputs']
        attrs = node.get('attrs', {})

        # Skip if output already known
        if all(o in shape_map for o in outputs):
            continue

        in_shapes = [shape_map.get(i, [-1]) for i in inputs]

        try:
            if op == 'MatMul':
                a_s, b_s = in_shapes[0], in_shapes[1]
                if -1 not in a_s and -1 not in b_s:
                    # ND MatMul: A[...,K] @ B[K,N] → A[...] + [N]
                    if len(a_s) > 2:
                        batch_dims = a_s[:-1]
                        shape_map[outputs[0]] = list(batch_dims) + [b_s[1]]
                    else:
                        shape_map[outputs[0]] = [a_s[0], b_s[1]]

            elif op == 'Gemm':
                a_s, b_s = in_shapes[0], in_shapes[1]
                transA = attrs.get('transA', 0)
                M = a_s[0] if not transA else a_s[1]
                N = b_s[0] if attrs.get('transB', 0) else b_s[1]
                shape_map[outputs[0]] = [M, N]

            elif op in ('Add', 'Sub', 'Mul', 'Div', 'Relu', 'Sigmoid', 'Tanh',
                        'Softmax', 'Erf', 'Cast', 'Identity', 'Dropout',
                        'LayerNormalization', 'Gelu'):
                # Pick the operand with the most dims (activations, not bias)
                best = None
                for s in in_shapes:
                    if isinstance(s, list) and -1 not in s:
                        if best is None or len(s) > len(best):
                            best = s
                if best:
                    shape_map[outputs[0]] = list(best)

            elif op == 'Reshape':
                # Second input is the target shape as a constant initializer
                shape_tensor = initializers.get(inputs[1]) if len(inputs) > 1 else None
                if shape_tensor is not None:
                    new_shape = [int(s) for s in np.asarray(shape_tensor).ravel()]
                    # Replace 0 with original dim, -1 with inferred
                    orig = in_shapes[0]
                    for i, s in enumerate(new_shape):
                        if s == 0 and i < len(orig):
                            new_shape[i] = orig[i]
                    shape_map[outputs[0]] = new_shape

            elif op == 'Transpose':
                perm = attrs.get('perm', None)
                if -1 not in in_shapes[0] and perm:
                    s = in_shapes[0]
                    shape_map[outputs[0]] = [s[p] for p in perm]

            elif op == 'ReduceMean':
                axes = attrs.get('axes', None)
                keepdims = attrs.get('keepdims', 1)
                if -1 not in in_shapes[0] and axes:
                    s = list(in_shapes[0])
                    if keepdims:
                        for a in axes:
                            s[a] = 1
                    else:
                        for a in sorted(axes, reverse=True):
                            s.pop(a)
                    shape_map[outputs[0]] = s

            elif op == 'Concat':
                axis = attrs.get('axis', 0)
                if all(-1 not in s for s in in_shapes):
                    concat_dim = sum(s[axis] for s in in_shapes)
                    out = list(in_shapes[0])
                    out[axis] = concat_dim
                    shape_map[outputs[0]] = out

            elif op == 'Split':
                axis = attrs.get('axis', 0)
                split_sizes = attrs.get('split', None)
                if -1 not in in_shapes[0]:
                    s = list(in_shapes[0])
                    if split_sizes:
                        offset = 0
                        for i, size in enumerate(split_sizes):
                            if i < len(outputs):
                                o = list(s)
                                o[axis] = size
                                shape_map[outputs[i]] = o

            elif op == 'Unsqueeze':
                axes = attrs.get('axes', [0])
                if -1 not in in_shapes[0]:
                    s = list(in_shapes[0])
                    for a in sorted(axes):
                        s.insert(a, 1)
                    shape_map[outputs[0]] = s

            elif op == 'Constant':
                # Should already be in shape_map via initializers
                pass

            elif op == 'Pow' or op == 'Sqrt':
                if -1 not in in_shapes[0]:
                    shape_map[outputs[0]] = list(in_shapes[0])

        except (IndexError, TypeError, ValueError):
            pass  # shape inference failed, leave unknown


# ================================================================
# Code Generator
# ================================================================

class CodeGenerator:
    def __init__(self, ordered_nodes, initializers, shape_map,
                 input_names, output_names, model_name="model"):
        self.nodes = ordered_nodes
        self.initializers = initializers
        self.shape_map = shape_map
        self.input_names = input_names
        self.output_names = output_names
        self.model_name = model_name

        self.w_alloc = SRAMAllocator(SRAM_W_SIZE, 'W')
        self.a_alloc = SRAMAllocator(SRAM_A_SIZE, 'A')
        self.o_alloc = SRAMAllocator(SRAM_O_SIZE, 'O')

        self.tensor_map = {}
        self.weight_blobs = []
        self.init_code = []
        self.infer_code = []
        self.output_code = []
        self.tu_nodes = 0
        self.host_nodes = 0

    def get_shape(self, name):
        s = self.shape_map.get(name, [-1])
        return [d if isinstance(d, int) else -1 for d in s]

    def resolve_initializer(self, name):
        if name in self.initializers:
            return np.asarray(self.initializers[name], dtype=np.float32)
        return None

    def generate(self):
        self._preload_weights()
        self._generate_inference_graph()
        self._generate_outputs()
        return self._emit_c_program()

    # ================================================================
    # Weight preloading
    # ================================================================

    def _preload_weights(self):
        """Embed all GEMM weights as C arrays. Preload only those that fit in W-buffer."""
        for node in self.nodes:
            op = node['op']
            if op not in ('Gemm', 'MatMul'):
                continue

            weight_input = node['inputs'][1]  # B in ONNX Gemm
            if weight_input not in self.initializers:
                continue
            if weight_input in self.tensor_map:
                continue

            arr = self.resolve_initializer(weight_input)
            if arr is None:
                continue

            attrs = node.get('attrs', {})
            transB = attrs.get('transB', 0)

            # Build W = effective_B^T [N][K] for TU MMA
            # If transB=0: B is [K,N] → W = B^T [N,K]
            # If transB=1: B is [N,K] → W = B [N,K]
            if transB:
                # B is already [N,K] — embed as-is
                W_arr = arr
            else:
                # B is [K,N] — transpose to get W [N,K]
                W_arr = arr.T.copy()

            W_shape = list(W_arr.shape)  # [N, K]
            fp16_bytes = fp32_to_fp16_bytes(W_arr)
            var_name = f"w_{self._sanitize(weight_input)}"
            self.weight_blobs.append((var_name, fp16_bytes))

            if len(fp16_bytes) <= (SRAM_W_SIZE // 2) - self.w_alloc.offset:
                # Fits in lower half of W-buffer → preload at init
                offset = self.w_alloc.alloc(len(fp16_bytes))
                self.init_code.append(
                    f"  /* {weight_input}: {W_shape} (transposed) */"
                )
                self.init_code.append(
                    f"  tu_dma_load_w({var_name}, {offset}, {len(fp16_bytes)});"
                )
                self.tensor_map[weight_input] = {
                    'tiled': False,
                    'offset': offset,
                    'shape': W_shape,
                    'var': var_name,
                    'bytes': len(fp16_bytes),
                }
            else:
                # Too large — embed but don't preload
                self.tensor_map[weight_input] = {
                    'tiled': True,
                    'shape': W_shape,
                    'var': var_name,
                    'bytes': len(fp16_bytes),
                    'fp16_bytes': fp16_bytes,
                }

    # ================================================================
    # Inference graph generation
    # ================================================================

    def _generate_inference_graph(self):
        for node in self.nodes:
            op = node['op']
            inputs = node['inputs']
            outputs = node['outputs']
            name = node['name']
            handler = getattr(self, f'_emit_{op.lower()}', None)
            if handler:
                handler(name, inputs, outputs, node.get('attrs', {}))
            else:
                self._emit_fallback(op, name, inputs, outputs, node.get('attrs', {}))

    def _generate_outputs(self):
        for out_name in self.output_names:
            info = self.tensor_map.get(out_name)
            if info and info.get('loc') == 'tu_o':
                var = self._sanitize(out_name) + '_host'
                shape = info['shape']
                nelem = int(np.prod(shape)) if all(isinstance(d, int) and d > 0 for d in shape) else (256 * 256)
                # O-buffer stores FP32 accumulators
                self.output_code.append(f"  fp32_t {var}[{nelem}];")
                self.output_code.append(
                    f"  tu_dma_store_o({var}, {info['offset']}, {nelem * FP32_BYTES});"
                )
            elif info and info.get('loc') == 'host':
                pass
            else:
                self.output_code.append(f"  /* WARNING: output {out_name} not tracked */")

    # ================================================================
    # Op handlers
    # ================================================================

    def _emit_gemm(self, name, inputs, outputs, attrs):
        """
        ONNX Gemm Y[M][N] = X[M][K] × B[K][N] + C

        TU MMA: O[N'][M'] += W[N'][K'] × A[K'][M']
        With transposition: W = B^T[N][K], A = X^T[K][M], O = Y^T[N][M]
        """
        A_name = inputs[0]   # X (activations)
        B_name = inputs[1]   # B (weights)
        C_name = inputs[2] if len(inputs) > 2 else None
        Y_name = outputs[0]

        transA = attrs.get('transA', 0)
        transB = attrs.get('transB', 0)
        alpha = attrs.get('alpha', 1.0)
        beta  = attrs.get('beta', 1.0)

        A_shape = self.get_shape(A_name)
        B_shape = self.get_shape(B_name)
        B_info = self.tensor_map.get(B_name, {})

        if any(d == -1 for d in A_shape) or any(d == -1 for d in B_shape):
            self._emit_fallback('Gemm', name, inputs, outputs, attrs)
            return

        # Flatten batch dims: [B,T,K] → [B*T, K]
        A_batch_dims = A_shape[:-1] if len(A_shape) > 2 else []
        M_flat = int(np.prod(A_batch_dims)) if A_batch_dims else A_shape[0]
        K_a = A_shape[-1]
        K_b = B_shape[1] if transB else B_shape[0]
        N = B_shape[0] if transB else B_shape[1]

        if K_a != K_b:
            self._emit_fallback('Gemm', name, inputs, outputs, attrs)
            return

        K = K_a
        M = M_flat

        # Output in ONNX space: Y[...][M][N]
        # Output in TU space: Y^T[N][M_flat] (flattened batch)
        tu_N = N
        tu_M = M

        self.tu_nodes += 1

        # Allocate output in O-buffer (FP32 accumulators, N×M elements)
        o_nelem = tu_N * tu_M
        o_size = o_nelem * FP32_BYTES
        o_off = self.o_alloc.alloc(o_size)

        # Load activations X^T[K][M] into A-buffer (flatten batch dims)
        # Activation variable: for inputs it's input_<name>, otherwise <name>_fp16
        if A_name in self.input_names:
            a_var = f"input_{self._sanitize(A_name)}"
        else:
            a_var = self._sanitize(A_name) + '_fp16'
        a_size = K * M * FP16_BYTES
        a_off = self.a_alloc.alloc(a_size)

        self.infer_code.append(
            f"  /* {name}: A{A_shape} @ B{B_shape}, "
            f"TU: W{N}×{K} × A{K}×{M} → O{N}×{M} */"
        )
        self.infer_code.append(
            f"  tu_transpose_fp16((const fp16_t*){a_var}, scratch_T, {M}, {K});"
        )
        self.infer_code.append(
            f"  tu_dma_load_a(scratch_T, {a_off}, {a_size});"
        )

        # Bias handling: load expanded bias [N][M] into O-buffer
        has_bias = (C_name is not None and beta != 0.0)
        if has_bias:
            c_arr = self.resolve_initializer(C_name)
            if c_arr is not None:
                bias_var = f"c_{self._sanitize(C_name)}"
                # Bias is [N] → expand to [N][M] by replicating each element M times
                c_flat = np.asarray(c_arr, dtype=np.float32).ravel()
                c_expanded = np.repeat(c_flat, M)  # [N*M]
                c_fp16 = fp32_to_fp16_bytes(c_expanded)
                self.weight_blobs.append((bias_var, c_fp16))
                self.infer_code.append(
                    f"  tu_dma_load_o({bias_var}, {o_off}, {len(c_fp16)});"
                )
            else:
                has_bias = False

        # Weight: use preloaded or tile
        if B_info.get('tiled'):
            self._emit_tiled_gemm(M, N, K, tu_N, tu_M,
                                  B_info, a_off, o_off, has_bias, name)
        else:
            w_off = B_info.get('offset', 0)
            self.infer_code.append(
                f"  tu_mma({tu_N}, {tu_M}, {K}, "
                f"{w_off}, {a_off}, {o_off}, "
                f"{'true' if has_bias else 'false'});"
            )

        self.tensor_map[Y_name] = {
            'loc': 'tu_o',
            'offset': o_off,
            'shape': [N, M],  # transposed in TU space
            'bytes': o_size,
        }

    def _emit_tiled_gemm(self, M, N, K, tu_N, tu_M,
                         B_info, a_off, o_off, has_bias, name):
        """Emit K-dimension tiling for weight matrices that don't fit in W-buffer."""
        w_workspace = SRAM_W_SIZE // 2  # upper half reserved for tile workspace
        max_w_elems = (SRAM_W_SIZE - w_workspace) // FP16_BYTES
        K_tile = max_w_elems // tu_N  # W tile is [N][K_tile]
        if K_tile < 16:
            self._emit_fallback('Gemm', name, [], [], {})
            return

        n_tiles = (K + K_tile - 1) // K_tile
        w_var = B_info['var']

        self.infer_code.append(
            f"  /* K-tiling: {n_tiles} tiles, K_tile={K_tile}, "
            f"W workspace @ {w_workspace} */"
        )

        for tile_i in range(n_tiles):
            ks = tile_i * K_tile
            ke = min(ks + K_tile, K)
            k_len = ke - ks
            tile_bytes = tu_N * k_len * FP16_BYTES

            # Weight tile: W[:, ks:ke] stored in W_var at byte offset
            # W[N][K] row-major → W[:, ks:ke] starts at byte ks*FP16_BYTES per row
            w_byte_offset = ks * FP16_BYTES

            self.infer_code.append(
                f"  tu_dma_load_w((const uint8_t*){w_var} + {w_byte_offset}, "
                f"{w_workspace}, {tile_bytes});  /* W[:,{ks}:{ke}] */"
            )

            # A-slice: A[ks:ke, :] in A-buffer
            # A is [K][M] row-major → A[ks:ke, :] starts at offset ks*M*FP16_BYTES
            a_slice_off = a_off + ks * tu_M * FP16_BYTES

            tile_has_bias = (tile_i == 0 and has_bias)
            self.infer_code.append(
                f"  tu_mma({tu_N}, {tu_M}, {k_len}, "
                f"{w_workspace}, {a_slice_off}, {o_off}, "
                f"{'true' if tile_has_bias else 'false'});"
            )

    def _emit_matmul(self, name, inputs, outputs, attrs):
        fake_attrs = {'alpha': 1.0, 'beta': 0.0, 'transA': 0, 'transB': 0}
        all_inputs = list(inputs) + ['']
        self._emit_gemm(name, all_inputs, outputs, fake_attrs)

    def _emit_fallback(self, op, name, inputs, outputs, attrs):
        self.host_nodes += 1
        in_str = ', '.join(inputs)
        out_str = ', '.join(outputs)
        self.infer_code.append(
            f"  /* [HOST] {name}: {op}({in_str}) → {out_str} */"
        )
        self.infer_code.append(
            f"  host_{op.lower()}(/* TODO: wire up */);"
        )
        for o in outputs:
            if o not in self.tensor_map:
                s = self.get_shape(o)
                self.tensor_map[o] = {'loc': 'host', 'offset': 0, 'shape': s, 'bytes': 0}

    # ================================================================
    # C program emission
    # ================================================================

    @staticmethod
    def _sanitize(name):
        return name.replace('/', '_').replace('.', '_').replace('-', '_').replace(':', '_')

    def _emit_c_program(self):
        lines = []
        L = lines.append

        L(f'/* Auto-generated by onnx_to_tu.py — {self.model_name} */')
        L('/* Target: RISC-V host + TinyTU accelerator */')
        L('/* MMA semantics: O[N][M] += W[N][K] × A[K][M] */')
        L('')
        L('#include "tu_cmodel/tu_cmodel.h"')
        L('#include <stdio.h>')
        L('#include <stdlib.h>')
        L('#include <string.h>')
        L('#include <math.h>')
        L('')

        for var_name, blob in self.weight_blobs:
            L(f'/* {var_name}: {len(blob)} bytes */')
            L(f'static const uint8_t {var_name}[] = {{')
            for i in range(0, len(blob), 16):
                chunk = blob[i:i+16]
                hex_str = ', '.join(f'0x{b:02X}' for b in chunk)
                if i + 16 < len(blob):
                    hex_str += ','
                L(f'  {hex_str}')
            L('};')
            L('')

        L('/* Host-side buffers */')
        for inp in self.input_names:
            s = self.get_shape(inp)
            nelem = int(np.prod(s)) if all(isinstance(d, int) and d > 0 for d in s) else (256 * 256)
            L(f'static fp16_t input_{self._sanitize(inp)}[{nelem}];')
        for out in self.output_names:
            s = self.get_shape(out)
            nelem = int(np.prod(s)) if all(isinstance(d, int) and d > 0 for d in s) else (256 * 256)
            L(f'static fp32_t output_{self._sanitize(out)}[{nelem}];  /* FP32 from O-buffer */')
        L('')

        L('/* ---- Host fallback stubs (RISC-V scalar) ---- */')
        L('')
        L('/* FP16 matrix transpose: src[rows][cols] → dst[cols][rows] */')
        L('static void tu_transpose_fp16(const fp16_t *src, fp16_t *dst,')
        L('                              int rows, int cols) {')
        L('  for (int r = 0; r < rows; r++)')
        L('    for (int c = 0; c < cols; c++)')
        L('      dst[c * rows + r] = src[r * cols + c];')
        L('}')
        L('')
        fallback_ops = set(n['op'] for n in self.nodes if n['op'] not in ('Gemm', 'MatMul'))
        for op in sorted(fallback_ops):
            L(f'static void host_{op.lower()}(void) {{')
            L(f'  fprintf(stderr, "HOST WARNING: {op} not implemented\\\\n");')
            L(f'}}')
            L('')

        L('')
        L('/* ================================================================')
        L(f' * Inference: {self.model_name}')
        L(' * ================================================================ */')
        L(f'void {self._sanitize(self.model_name)}_infer(void) {{')
        L('')
        L('  /* Scratch buffer for activation transpose (max K*M FP16) */')
        # Find max K*M across all Gemm nodes
        max_scratch = 256 * 256  # default
        for node in self.nodes:
            if node['op'] in ('Gemm', 'MatMul'):
                A_shape = self.get_shape(node['inputs'][0])
                if all(isinstance(d, int) and d > 0 for d in A_shape):
                    M = A_shape[0]
                    K = A_shape[1]
                    max_scratch = max(max_scratch, M * K)
        L(f'  fp16_t scratch_T[{max_scratch}];')
        L('')
        for line in self.init_code:
            L(line)
        L('')
        for line in self.infer_code:
            L(line)
        L('')
        for line in self.output_code:
            L(line)
        L('}')
        L('')

        L('int main(void) {')
        L('  tu_init();')
        L(f'  {self._sanitize(self.model_name)}_infer();')
        L('  tu_sync();')
        L('  tu_print_stats();')
        L('  return 0;')
        L('}')
        L('')

        return '\n'.join(lines)


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='ONNX → TinyTU compiler')
    parser.add_argument('model', help='Path to ONNX model (.onnx)')
    parser.add_argument('--output', '-o', default=None, help='Output C file')
    parser.add_argument('--name', '-n', default=None, help='Model name')
    parser.add_argument('--shape', '-s', action='append', default=[],
                        help='Concrete dims: B=2,T=1,T_past=0 (repeatable)')
    args = parser.parse_args()

    if not HAS_ONNX:
        print("ERROR: 'onnx' package not installed. Run: pip install onnx", file=sys.stderr)
        sys.exit(1)

    # Parse shape overrides
    shape_overrides = {}
    for s in args.shape:
        for pair in s.split(','):
            k, v = pair.split('=')
            shape_overrides[k.strip()] = int(v.strip())

    model_path = args.model
    model_name = args.name or os.path.splitext(os.path.basename(model_path))[0]

    print(f"// Loading {model_path} ...", file=sys.stderr)
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    print(f"// Analyzing graph ...", file=sys.stderr)
    ordered, initializers, shape_map, input_names, output_names = analyze_onnx_model(model)

    # Resolve dynamic dimensions
    if shape_overrides:
        print(f"// Shape overrides: {shape_overrides}", file=sys.stderr)
        for name in shape_map:
            shape_map[name] = [
                shape_overrides.get(d, d) if isinstance(d, str) else d
                for d in shape_map[name]
            ]

    print(f"// Nodes: {len(ordered)}, Inputs: {input_names}, Outputs: {output_names}",
          file=sys.stderr)

    gen = CodeGenerator(ordered, initializers, shape_map,
                        input_names, output_names, model_name)
    code = gen.generate()

    print(f"// TU ops: {gen.tu_nodes}, Host ops: {gen.host_nodes}", file=sys.stderr)
    print(f"// W-buffer: {gen.w_alloc.used()}/{SRAM_W_SIZE} bytes "
          f"({100.0*gen.w_alloc.used()/SRAM_W_SIZE:.1f}%)", file=sys.stderr)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(code)
        print(f"// Wrote: {args.output}", file=sys.stderr)
    else:
        print(code)


if __name__ == '__main__':
    main()
