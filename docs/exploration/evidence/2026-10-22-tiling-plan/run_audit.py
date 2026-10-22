#!/usr/bin/env python3
"""Reproduce tiling metadata gates from a pinned base plus explicit overlays.

Usage: python3 <this-file> REPO BASE_HEAD OUTPUT_DIR
Run from the corrected source checkout. Scratch builds respect TMPDIR.
"""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

repo = Path(sys.argv[1]).resolve()
base = sys.argv[2]
out = Path(sys.argv[3]).resolve()
out.mkdir(parents=True, exist_ok=False)
work = Path(tempfile.mkdtemp(prefix="tiling-audit-", dir=os.environ["TMPDIR"]))
overlays = [".gitignore", "Makefile", "tu_cmodel/memory/address_generator.c",
            "tu_cmodel/memory/address_generator.h", "tests/test_agen_tiling_plan.c"]
records = []


def run(label, args, cwd, expected=0, extra=None):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(cwd)
    if extra:
        env.update(extra)
    p = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=360)
    (out / (label + ".log.gz")).write_bytes(gzip.compress(p.stdout, mtime=0))
    records.append({"label": label, "cwd": str(cwd), "command": args,
                    "LD_LIBRARY_PATH": str(cwd), "extra_env": extra or {},
                    "returncode": p.returncode, "expected": expected})
    (out / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
    print(label, "exit=", p.returncode, flush=True)
    if expected is not None and p.returncode != expected:
        raise RuntimeError(f"{label}: {p.stdout[-4000:].decode(errors='replace')}")
    return p.stdout


def extract(target):
    target.mkdir()
    archive = subprocess.check_output(["git", "archive", base], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        # Existing external symlink fixtures are unused by these C-only tests.
        members = [m for m in tf.getmembers() if m.isdir() or m.isfile()]
        tf.extractall(target, members=members, filter="data")


def overlay(target):
    for path in overlays:
        shutil.copy2(repo / path, target / path)


try:
    resolved = subprocess.check_output(["git", "rev-parse", base], cwd=repo).decode().strip()
    (out / "inputs.json").write_text(json.dumps({
        "baseline_head": resolved, "overlay_sha256": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in overlays},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scratch": str(work), "symlinks_omitted": True}, indent=2) + "\n")
    baseline, final, sanitized = (work / n for n in ("baseline", "final", "sanitized"))
    extract(baseline)
    run("baseline-build", ["make", "-j4"], baseline)
    run("baseline-quick", ["make", "test-quick"], baseline)
    shutil.copy2(repo / overlays[-1], baseline / overlays[-1])
    run("baseline-new-gate-build", ["cc", "-O2", "-Wall", "-Wextra", "-std=c11",
        "-I.", "-Itu_cmodel", overlays[-1], "./libtucmodel.a", "-lm", "-o", "tiling-probe"], baseline)
    run("baseline-probes-red", ["./tiling-probe", "probes"], baseline, expected=1)
    run("baseline-full-gate-red", ["./tiling-probe"], baseline, expected=1)
    run("baseline-broad", ["make", "test"], baseline, expected=None)
    extract(final)
    overlay(final)
    run("final-build", ["make", "-j4"], final)
    run("final-quick", ["make", "test-quick"], final)
    run("final-focused", ["make", "test-agen-tiling-plan", "test-config", "test-conv",
                          "test-scatter-gather"], final)
    run("final-broad", ["make", "test"], final, expected=None)
    extract(sanitized)
    overlay(sanitized)
    flags = "-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer"
    run("sanitizer-build", ["make", "-j4", "libtucmodel.a", "CFLAGS=" + flags], sanitized)
    run("sanitizer-gates", ["make", "test-agen-tiling-plan", "test-agen", "test-agen-tiled-sweep",
        "test-agen-affine-bounds", "test-agen-im2col-bounds", "test-agen-im2col-point",
        "test-agen-dma-chain", "test-agen-range-admission", "CFLAGS=" + flags,
        "LDFLAGS=-lm -fsanitize=address,undefined"], sanitized,
        extra={"ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
               "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"})
finally:
    # Preserve raw output bytes; never whitespace-normalize logs.
    (out / "SHA256SUMS").write_text("".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
        for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"))
