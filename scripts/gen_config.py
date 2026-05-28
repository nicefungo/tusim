#!/usr/bin/env python3
"""
gen_config.py — Generate tu_config.h from tu_config.yaml

Usage:
  python3 scripts/gen_config.py [config/tu_config.yaml] [-o tu_cmodel/tu_config.h]
"""

import sys
import os
import re
import argparse

def load_yaml_simple(path):
    """Minimal YAML parser — handles the subset we use (no pyyaml dependency)."""
    with open(path) as f:
        lines = f.readlines()

    root = {}
    stack = [(root, -1)]

    for line in lines:
        stripped = line.rstrip()
        if not stripped or stripped.startswith('#'):
            continue

        indent = len(line) - len(line.lstrip())

        # Handle list items (no colon)
        if stripped.lstrip().startswith('- '):
            val = stripped.lstrip()[2:].strip()
            val = val.split('#')[0].strip().strip('"').strip("'")
            while stack and stack[-1][1] >= indent:
                stack.pop()
            parent = stack[-1][0]
            # Find the placeholder dict (empty, just created) and convert to list
            for k, v in list(parent.items()):
                if isinstance(v, dict) and len(v) == 0:
                    parent[k] = []
                    parent[k].append(val)
                    break
            else:
                # Existing list
                for k, v in parent.items():
                    if isinstance(v, list):
                        v.append(val)
                        break
            continue

        key_val = stripped.split(':', 1)
        if len(key_val) != 2:
            continue

        key = key_val[0].strip()
        val = key_val[1].strip()

        # Pop stack to correct indentation
        while stack and stack[-1][1] >= indent:
            stack.pop()

        parent = stack[-1][0]

        # Preprocess value: strip inline comment and whitespace
        val_stripped = val.split('#')[0].strip().strip('"').strip("'")

        # Handle bracketed lists: [a, b, c]
        if val_stripped.startswith('[') and val_stripped.endswith(']'):
            items = [x.strip().strip('"').strip("'") for x in val_stripped[1:-1].split(',')]
            parent[key] = items
            continue

        if val_stripped == '' or val_stripped == '{}':
            # Could be dict or list — create placeholder, list items will convert it
            parent[key] = {}  # placeholder
            stack.append((parent[key], indent))
        elif val_stripped.startswith('- '):
            # List item
            v = val_stripped[2:].strip().strip('"').strip("'")
            if key not in parent:
                parent[key] = []
            parent[key].append(v)
        elif val_stripped in ('true', 'false'):
            parent[key] = (val_stripped == 'true')
        elif val_stripped.isdigit() or (val_stripped.startswith('-') and val_stripped[1:].isdigit()):
            parent[key] = int(val_stripped)
        elif re.match(r'^-?\d+\.?\d*([eE][+-]?\d+)?$', val_stripped):
            parent[key] = float(val_stripped)
        else:
            parent[key] = val_stripped

    return root


def generate_header(config, output_path):
    """Generate tu_config.h from parsed config dict."""
    c = config['tu']
    comp = c['compute']
    mem = c['memory']
    dma = c['dma']
    isa = c['isa']
    mc = c['multicore']
    perf = c['performance']
    prec = c['precision']
    sparsity = c['sparsity']
    verify = c['verification']

    lines = []
    L = lines.append

    L('/*')
    L(f' * TinyTU Production Configuration — {c["name"]} {c["version"]}')
    L(' * Auto-generated from config/tu_config.yaml.')
    L(' * Do not edit directly.')
    L(' */')
    L('')
    L('#ifndef TU_CONFIG_H')
    L('#define TU_CONFIG_H')
    L('')
    L('#include <stdint.h>')
    L('#include <stdbool.h>')
    L('')
    L('#ifdef __cplusplus')
    L('extern "C" {')
    L('#endif')
    L('')

    # Compute engine
    L('/* ================================================================')
    L(' * Compute Engine')
    L(' * ================================================================ */')
    L('')
    pe = comp['pe_array']
    L(f'#define TU_PE_ROWS              {pe["rows"]}')
    L(f'#define TU_PE_COLS              {pe["cols"]}')
    L(f'#define TU_PE_PIPELINE_DEPTH    {pe["pipeline_depth"]}')
    L(f'#define TU_MAC_UNITS_PER_PE     {comp["mac_units_per_pe"]}')
    L('')

    df_map = {'weight_stationary': 0, 'output_stationary': 1, 'row_stationary': 2}
    L(f'#define TU_DATAFLOW_WEIGHT_STATIONARY  0')
    L(f'#define TU_DATAFLOW_OUTPUT_STATIONARY  1')
    L(f'#define TU_DATAFLOW_ROW_STATIONARY     2')
    L(f'#define TU_DATAFLOW_MODE              {df_map[pe["dataflow"]]}')
    L('')

    prec_mask = 0
    for p in comp['supported_precisions']:
        if p == 'fp16': prec_mask |= 1
        elif p == 'fp32': prec_mask |= 2
        elif p == 'bf16': prec_mask |= 4
        elif p == 'fp8': prec_mask |= 8
        elif p == 'int8': prec_mask |= 16
        elif p == 'int4': prec_mask |= 32
    L(f'#define TU_PRECISION_FP16       (1 << 0)')
    L(f'#define TU_PRECISION_FP32       (1 << 1)')
    L(f'#define TU_PRECISION_BF16       (1 << 2)')
    L(f'#define TU_PRECISION_FP8        (1 << 3)')
    L(f'#define TU_PRECISION_INT8       (1 << 4)')
    L(f'#define TU_PRECISION_INT4       (1 << 5)')
    L(f'#define TU_PRECISION_MASK       {prec_mask}')
    L(f'#define TU_ACCUMULATOR_PRECISION_{comp["accumulator_precision"].upper()}  1')
    L('')

    # Memory
    L('/* ================================================================')
    L(' * Memory System')
    L(' * ================================================================ */')
    L('')
    sram = mem['sram']
    L(f'#define TU_SRAM_W_SIZE_KB       {sram["w_buffer_kb"]}')
    L(f'#define TU_SRAM_A_SIZE_KB       {sram["a_buffer_kb"]}')
    L(f'#define TU_SRAM_O_SIZE_KB       {sram["o_buffer_kb"]}')
    L('')
    L('#define TU_SRAM_W_SIZE          (TU_SRAM_W_SIZE_KB * 1024)')
    L('#define TU_SRAM_A_SIZE          (TU_SRAM_A_SIZE_KB * 1024)')
    L('#define TU_SRAM_O_SIZE          (TU_SRAM_O_SIZE_KB * 1024)')
    L('#define TU_SRAM_TOTAL           (TU_SRAM_W_SIZE + TU_SRAM_A_SIZE + TU_SRAM_O_SIZE)')
    L('')

    bank = mem['banking']
    L(f'#define TU_SRAM_BANKS           {bank["banks"]}')
    L(f'#define TU_SRAM_BANK_WIDTH      {bank["bank_width_bytes"]}')
    L('')
    conflict_map = {'none': 0, 'detect': 1, 'stall_cycle': 2}
    L(f'#define TU_CONFLICT_NONE        0')
    L(f'#define TU_CONFLICT_DETECT      1')
    L(f'#define TU_CONFLICT_STALL       2')
    L(f'#define TU_CONFLICT_MODEL       {conflict_map[bank["conflict_model"]]}')
    L('')

    lat = mem['latency']
    L(f'#define TU_LATENCY_SRAM_READ    {lat["sram_read"]}')
    L(f'#define TU_LATENCY_SRAM_WRITE   {lat["sram_write"]}')
    L(f'#define TU_LATENCY_DRAM_READ    {lat["dram_read"]}')
    L(f'#define TU_LATENCY_DRAM_WRITE   {lat["dram_write"]}')
    L('')

    # DMA
    L('/* ================================================================')
    L(' * DMA Engine')
    L(' * ================================================================ */')
    L('')
    L(f'#define TU_DMA_BUS_WIDTH_BITS   {dma["bus_width_bits"]}')
    L('#define TU_DMA_BUS_WIDTH_BYTES  (TU_DMA_BUS_WIDTH_BITS / 8)')
    L(f'#define TU_DMA_MAX_BURST_BYTES  {dma["max_burst_bytes"]}')
    L(f'#define TU_DMA_CHANNELS         {dma["channels"]}')
    L(f'#define TU_DMA_MAX_OUTSTANDING  {dma["max_outstanding"]}')
    L(f'#define TU_DMA_ASYNC_MODE       {1 if dma["async_mode"] else 0}')
    L('')

    # ISA
    L('/* ================================================================')
    L(' * ISA / Command Queue')
    L(' * ================================================================ */')
    L('')
    L(f'#define TU_ISA_INSTR_WIDTH_BITS {isa["instruction_width_bits"]}')
    L(f'#define TU_ISA_QUEUE_DEPTH      {isa["queue_depth"]}')
    L(f'#define TU_ISA_DEP_CHECKING     {1 if isa["dependency_checking"] else 0}')
    L('')

    # Multi-core
    L('/* ================================================================')
    L(' * Multi-Core')
    L(' * ================================================================ */')
    L('')
    L(f'#define TU_MULTICORE_ENABLED    {1 if mc["enabled"] else 0}')
    L(f'#define TU_NUM_CORES            {mc["num_cores"]}')
    ic_map = {'none': 0, 'ring': 1, 'mesh': 2, 'crossbar': 3}
    L(f'#define TU_INTERCONNECT_NONE    0')
    L(f'#define TU_INTERCONNECT_RING    1')
    L(f'#define TU_INTERCONNECT_MESH    2')
    L(f'#define TU_INTERCONNECT_MODE    {ic_map[mc["interconnect"]]}')
    L(f'#define TU_CACHE_COHERENCE      {1 if mc["cache_coherence"] else 0}')
    L('')

    # Performance
    L('/* ================================================================')
    L(' * Performance Model')
    L(' * ================================================================ */')
    L('')
    cmap = {'functional': 0, 'estimated': 1, 'cycle_accurate': 2}
    L(f'#define TU_CYCLE_MODEL_FUNCTIONAL    0')
    L(f'#define TU_CYCLE_MODEL_ESTIMATED     1')
    L(f'#define TU_CYCLE_MODEL_CYCLE_ACCURATE 2')
    L(f'#define TU_CYCLE_MODEL               {cmap[perf["cycle_model"]]}')
    L('')
    ctr = perf['counters']
    tr = perf['tracing']
    L(f'#define TU_COUNTERS_ENABLED           {1 if ctr["enabled"] else 0}')
    L(f'#define TU_COUNTERS_DETAILED_STALLS   {1 if ctr["detailed_stalls"] else 0}')
    L(f'#define TU_TRACE_ENABLED              {1 if tr["enabled"] else 0}')
    L('')

    # Precision
    L('/* ================================================================')
    L(' * Precision Parameters')
    L(' * ================================================================ */')
    L('')
    fp16 = prec['fp16']
    rmap = {'round_nearest_even': 0, 'round_toward_zero': 1}
    L(f'#define TU_FP16_ROUNDING_RNE          0')
    L(f'#define TU_FP16_ROUNDING_RTZ          1')
    L(f'#define TU_FP16_ROUNDING_MODE         {rmap[fp16["rounding"]]}')
    L(f'#define TU_FP16_SUBNORMAL_FLUSH       {1 if fp16["subnormal"] == "flush_to_zero" else 0}')
    L(f'#define TU_FP16_SATURATE              {1 if fp16["saturate"] else 0}')
    L('')
    L(f'#define TU_FP32_ROUNDING_MODE         {rmap[prec["fp32"]["rounding"]]}')
    L('')

    # Sparsity
    L('/* ================================================================')
    L(' * Sparsity')
    L(' * ================================================================ */')
    L('')
    L(f'#define TU_SPARSITY_ENABLED           {1 if sparsity["enabled"] else 0}')
    L(f'#define TU_SPARSITY_2OF4              {1 if sparsity["structured_2of4"] else 0}')
    L(f'#define TU_SPARSITY_UNSTRUCTURED      {1 if sparsity["unstructured"] else 0}')
    L('')

    # Verification
    L('/* ================================================================')
    L(' * Verification')
    L(' * ================================================================ */')
    L('')
    vmap = {'numpy': 0, 'pytorch': 1, 'onnxruntime': 2}
    L(f'#define TU_VERIFY_GOLDEN_NUMPY        0')
    L(f'#define TU_VERIFY_GOLDEN_PYTORCH      1')
    L(f'#define TU_VERIFY_GOLDEN_MODE         {vmap[verify["golden_reference"]]}')
    L(f'#define TU_VERIFY_RANDOM_ITERS        {verify["random_test_iterations"]}')
    L(f'#define TU_VERIFY_ERROR_TOLERANCE     {verify["error_tolerance"]}')
    L('')

    # Runtime config
    L('/* ================================================================')
    L(' * Runtime Configuration (modifiable without recompilation)')
    L(' * ================================================================ */')
    L('')
    L('typedef struct {')
    L('    uint16_t pe_rows;')
    L('    uint16_t pe_cols;')
    L('    uint32_t sram_w_size;')
    L('    uint32_t sram_a_size;')
    L('    uint32_t sram_o_size;')
    L('    bool     counters_enabled;')
    L('    bool     detailed_stalls;')
    L('    bool     trace_enabled;')
    L('    char     trace_file[256];')
    L('    bool     verify_enabled;')
    L('    double   verify_tolerance;')
    L('} tu_runtime_config_t;')
    L('')
    L('static inline tu_runtime_config_t tu_config_default(void) {')
    L('    return (tu_runtime_config_t){')
    L(f'        .pe_rows           = TU_PE_ROWS,')
    L(f'        .pe_cols           = TU_PE_COLS,')
    L(f'        .sram_w_size       = TU_SRAM_W_SIZE,')
    L(f'        .sram_a_size       = TU_SRAM_A_SIZE,')
    L(f'        .sram_o_size       = TU_SRAM_O_SIZE,')
    L(f'        .counters_enabled  = TU_COUNTERS_ENABLED,')
    L(f'        .detailed_stalls   = TU_COUNTERS_DETAILED_STALLS,')
    L(f'        .trace_enabled     = TU_TRACE_ENABLED,')
    L(f'        .trace_file        = "",')
    L(f'        .verify_enabled    = false,')
    L(f'        .verify_tolerance  = TU_VERIFY_ERROR_TOLERANCE,')
    L('    };')
    L('}')
    L('')
    L('#ifdef __cplusplus')
    L('}')
    L('#endif')
    L('')
    L('#endif /* TU_CONFIG_H */')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate tu_config.h from YAML')
    parser.add_argument('yaml', nargs='?', default='config/tu_config.yaml')
    parser.add_argument('-o', '--output', default='tu_cmodel/tu_config.h')
    args = parser.parse_args()

    config = load_yaml_simple(args.yaml)
    generate_header(config, args.output)


if __name__ == '__main__':
    main()
