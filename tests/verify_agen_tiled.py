#!/usr/bin/env python3
"""Reproduce tiled RED/GREEN and instrument the entire exercised address path."""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    print("$ " + shlex.join(map(str, command)), flush=True)
    kwargs.setdefault("timeout", 300)
    result = subprocess.run(list(map(str, command)), cwd=ROOT, **kwargs)
    print(f"exit={result.returncode}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="8b3e3cb")
    args = parser.parse_args()
    source = ROOT / "tu_cmodel/memory/address_generator.c"
    harness = ROOT / "tests/test_agen_tiled_sweep.c"
    for path in (source, harness):
        print(f"sha256 {path.relative_to(ROOT)} {hashlib.sha256(path.read_bytes()).hexdigest()}")
    if run(["make", "libtucmodel.a"]).returncode:
        return 1
    scratch = Path(os.environ.get("TMPDIR", ROOT / "build/probes"))
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agen-tiled-", dir=scratch) as tmp:
        tmp = Path(tmp)
        compiler = shlex.split(os.environ.get("CC", "cc"))
        common = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-g", "-O1",
                  "-I" + str(ROOT), "-I" + str(ROOT / "tu_cmodel"),
                  "-I" + str(ROOT / "tu_cmodel/memory")]
        old = subprocess.check_output(
            ["git", "show", args.baseline_ref + ":tu_cmodel/memory/address_generator.c"], cwd=ROOT, timeout=30)
        old_source = tmp / "baseline.c"
        old_source.write_bytes(old)
        print(f"baseline={args.baseline_ref} source_sha256={hashlib.sha256(old).hexdigest()}", flush=True)
        red = tmp / "red"
        if run(compiler + common + [harness, old_source, ROOT / "libtucmodel.a", "-lm", "-o", red]).returncode:
            return 1
        result = run([red], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout, end="", flush=True)
        if (result.returncode != 1 or "rank=2 i=8" not in result.stdout
                or "rank=3 i=4" not in result.stdout):
            raise RuntimeError("baseline must reject both discriminating traversal cases")
        green = tmp / "sanitized"
        flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        if run(compiler + common + flags + [harness, source, ROOT / "libtucmodel.a", "-lm", "-o", green]).returncode:
            return 1
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        if run([green], env=env).returncode:
            return 1
    print("PASS: original traversal rejected; current complete iterator/range path ASan/UBSan/LSan clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
