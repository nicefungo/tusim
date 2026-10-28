#!/usr/bin/env python3
"""Safe admission mutants: no invalid address is ever dereferenced.
Run from a rebuilt checkout. Each mutant must compile and exit 1 with the
harness's own nonzero failure summary, not a crash or a tool failure.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "tu_cmodel/memory/address_generator.c"
TEST = ROOT / "tests/test_agen_block_bounds.c"
ARCHIVE = ROOT / "libtucmodel.a"
ORIGINAL = """        uint32_t size = cfg[3] ? cfg[3] : 4;
        if (!affine_geometry_valid(base_addr, 1, 1, cfg[0], 0, 0, size))
            return -1;
        it->total_elements = cfg[0];"""
MUTANTS = {
    "ignore_base": ORIGINAL.replace("affine_geometry_valid(base_addr,", "affine_geometry_valid(0,"),
    "starting_byte_only": ORIGINAL.replace(
        "!affine_geometry_valid(base_addr, 1, 1, cfg[0], 0, 0, size)",
        "cfg[0] && (uint64_t)base_addr + (uint64_t)(cfg[0] - 1) * size >= UINT32_MAX"),
    "reject_exact_end": ORIGINAL.replace(
        "!affine_geometry_valid(base_addr, 1, 1, cfg[0], 0, 0, size)",
        "cfg[0] && (uint64_t)base_addr + (uint64_t)cfg[0] * size >= UINT32_MAX"),
    "wrong_zero_width_default": ORIGINAL.replace("cfg[3] : 4", "cfg[3] : 1"),
}


def run(command):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=90)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}\n{result.stdout}")
    return result.stdout


def main():
    source = SOURCE.read_text()
    assert source.count(ORIGINAL) == 1, "mutation anchor changed; review before running"
    assert ARCHIVE.is_file(), "build libtucmodel.a first"
    with tempfile.TemporaryDirectory(prefix="tusim-block-mutants-", dir=os.environ["TMPDIR"]) as temp:
        directory = Path(temp)
        # Rebuild and run an unmodified control from exactly the same source path.
        for name, changed in [("control", ORIGINAL), *MUTANTS.items()]:
            cfile, obj, exe = (directory / f"{name}{suffix}" for suffix in (".c", ".o", ""))
            cfile.write_text(source.replace(ORIGINAL, changed))
            flags = ["cc", "-O2", "-Wall", "-Wextra", "-std=c11", "-I.", "-Itu_cmodel"]
            run([*flags, "-Itu_cmodel/memory", "-c", str(cfile), "-o", str(obj)])
            run([*flags, str(TEST), str(obj), str(ARCHIVE), "-lm", "-o", str(exe)])
            result = subprocess.run([str(exe)], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=90)
            summary = re.search(r"^(PASS|FAIL): cases=(\d+) rejected=(\d+) bounded_sequences=(\d+) "
                                r"checks=(\d+) failures=(\d+)$", result.stdout, re.M)
            assert summary, f"{name}: missing harness completion, exit={result.returncode}"
            if name == "control":
                assert result.returncode == 0 and summary[1] == "PASS" and int(summary[6]) == 0
            else:
                assert result.returncode == 1 and summary[1] == "FAIL" and int(summary[6]) > 0
            print(f"{name}: exit={result.returncode} {summary[0]}")
    print(f"PASS: unmodified control and {len(MUTANTS)} independently rejected safe mutants")


if __name__ == "__main__":
    main()
