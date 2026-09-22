#!/usr/bin/env python3
"""Fail when the checked-in TU header drifts from its YAML generator."""

import difflib
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "tu_cmodel" / "tu_config.h"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        generated = tmp / "tu_config.h"
        subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "gen_config.py"),
                str(ROOT / "config" / "tu_config.yaml"),
                "-o",
                str(generated),
            ],
            check=True,
        )
        expected_text = EXPECTED.read_text()
        generated_text = generated.read_text()
        if expected_text != generated_text:
            print("checked-in tu_config.h is stale:")
            print(
                "".join(
                    difflib.unified_diff(
                        expected_text.splitlines(keepends=True),
                        generated_text.splitlines(keepends=True),
                        fromfile=str(EXPECTED),
                        tofile="generated/tu_config.h",
                    )
                ),
                end="",
            )
            return 1

        yaml_text = (ROOT / "config" / "tu_config.yaml").read_text()
        alternatives = {
            'dataflow: "weight_stationary"': 'dataflow: "output_stationary"',
            "gbuf_size_kb: 1024": "gbuf_size_kb: 2048",
            'type: "ideal"': 'type: "hbm3"',
            "bandwidth_gbps: 256.0": "bandwidth_gbps: 128.0",
            "channels: 8          # power": "channels: 4          # power",
            "level: 3                   #": "level: 4                   #",
            'issue_payload_mode: "serialized"':
                'issue_payload_mode: "overlapped"',
            'burst_boundary_mode: "size_only"':
                'burst_boundary_mode: "both_address"',
            'arbitration: "round_robin"':
                'arbitration: "aging_priority"',
            'aging_scope: "submission"':
                'aging_scope: "queue_head"',
            'payload_scope: "descriptor"':
                'payload_scope: "burst_commands"',
            '    fp16:\n      rounding: "round_nearest_even"':
                '    fp16:\n      rounding: "stochastic"',
        }
        for old, new in alternatives.items():
            if yaml_text.count(old) != 1:
                raise RuntimeError(f"non-unique YAML test fixture: {old!r}")
            yaml_text = yaml_text.replace(old, new)
        alternative_yaml = tmp / "alternative.yaml"
        alternative_yaml.write_text(yaml_text)
        subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "gen_config.py"),
                str(alternative_yaml),
                "-o",
                str(generated),
            ],
            check=True,
        )
        probe = tmp / "probe.c"
        probe.write_text(
            '#include "tu_config.h"\n'
            '#include "tu_cmodel/tu_cmodel.h"\n'
            '_Static_assert(TU_DATAFLOW_MODE == TU_DATAFLOW_MODE_OS, "dataflow");\n'
            '_Static_assert(TU_MEM_GBUF_SIZE == 2048 * 1024, "gbuf");\n'
            '_Static_assert(TU_DRAM_TYPE == TU_DRAM_HBM3, "dram type");\n'
            '_Static_assert(TU_DRAM_CHANNELS == 4, "dram channels");\n'
            '_Static_assert(TU_LOG_LEVEL_DEFAULT == 4, "log level");\n'
            '_Static_assert(TU_DMA_ISSUE_PAYLOAD_MODE == TU_DMA_ISSUE_PAYLOAD_DEFAULT_OVERLAPPED, "DMA issue/payload overlap");\n'
            '_Static_assert(TU_DMA_PAYLOAD_SCOPE == TU_DMA_PAYLOAD_SCOPE_BURST_COMMANDS, "DMA burst payload scope");\n'
            '_Static_assert(TU_DMA_ARB_POLICY == TU_DMA_ARB_DEFAULT_AGING_PRIORITY, "DMA aging arbitration");\n'
            '_Static_assert(TU_DMA_AGING_SCOPE == TU_DMA_AGING_SCOPE_QUEUE_HEAD, "DMA aging scope");\n'
            '_Static_assert(TU_FP16_ROUNDING_MODE == TU_FP16_ROUNDING_STOCHASTIC, "rounding");\n'
            'int main(void) { tu_runtime_config_t c = tu_runtime_config_default(); '
            'return c.dataflow_mode != TU_DATAFLOW_MODE_OS || '
            'c.dma_issue_payload_mode != TU_DMA_ISSUE_PAYLOAD_DEFAULT_OVERLAPPED || '
            'c.dma_burst_boundary_mode != TU_DMA_BURST_BOUNDARY_BOTH_ADDRESS || '
            'c.dma_arb_policy != TU_DMA_ARB_DEFAULT_AGING_PRIORITY || '
            'c.dma_aging_scope != TU_DMA_AGING_SCOPE_QUEUE_HEAD || '
            'c.dma_payload_scope != TU_DMA_PAYLOAD_SCOPE_BURST_COMMANDS; }\n'
        )
        subprocess.run(
            [
                "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
                "-I", str(tmp), "-I", str(ROOT), "-I", str(ROOT / "tu_cmodel"),
                str(probe),
            ],
            check=True,
        )

        bad_yaml = tmp / "bad.yaml"
        bad_yaml.write_text(
            yaml_text.replace('dataflow: "output_stationary"',
                              'dataflow: "unsupported_dataflow"')
        )
        bad = subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "gen_config.py"),
                str(bad_yaml),
                "-o",
                str(generated),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if bad.returncode == 0:
            print("unsupported generated configuration was accepted")
            return 1

    print("generated configuration identity, alternatives, compile, rejection: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
