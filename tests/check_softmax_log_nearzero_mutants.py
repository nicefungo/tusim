#!/usr/bin/env python3
"""Safe numerical mutation controls for the near-zero gate (no tree edits)."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "tu_cmodel/compute/softmax_engine.c").read_text()
mutations = {
    "log-of-double-sum": (
        "float log_sum = (float)log1p(log_residual);",
        "float log_sum = (float)log(sum);",
    ),
    "exclude-all-tied-maxima": (
        "mode == TU_SOFTMAX_LOG && i != max_index",
        "mode == TU_SOFTMAX_LOG && shifted != 0.0f",
    ),
    "wrong-anchor-position": ("max_index = i;", "max_index = 0;"),
}
# Every candidate provides all softmax symbols before the static archive, so
# the archive's softmax object is not selected. Other modules remain identical.
with tempfile.TemporaryDirectory(prefix="softmax-nearzero-mutants-",
                                 dir=os.environ["TMPDIR"]) as tmp:
    for name, (old, new) in mutations.items():
        if source.count(old) != 1:
            raise SystemExit(f"mutation anchor not unique: {name}")
        path = Path(tmp) / f"{name}.c"
        exe = Path(tmp) / name
        path.write_text(source.replace(old, new))
        subprocess.run([
            "cc", "-O2", "-Wall", "-Wextra", "-std=c11", "-I", str(ROOT),
            "-I", str(ROOT / "tu_cmodel"),
            "-I", str(ROOT / "tu_cmodel/compute"),
            str(ROOT / "tests/test_softmax_log_nearzero.c"), str(path),
            str(ROOT / "libtucmodel.a"), "-lm", "-o", str(exe),
        ], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True)
        summary = re.search(r"log-nearzero cases=\d+ checks=\d+ failures=(\d+).*",
                            result.stdout)
        if result.returncode != 1 or not summary or int(summary[1]) == 0:
            raise SystemExit(f"mutant not rejected by numerical assertions: {name}\n"
                             f"exit={result.returncode}\n{result.stdout}\n{result.stderr}")
        print(f"{name}: exit={result.returncode} {summary[0]}")
print("All three safe numerical mutants rejected.")
