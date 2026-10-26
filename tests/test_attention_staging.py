#!/usr/bin/env python3
"""Reproduce all-object sanitizer and isolated attention staging mutation gates.

The output must be a new external directory. No checkout build is modified.
The former --kv-probe counterexample must now pass global normalization.
"""
import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    "Makefile", ".gitignore", "tu_cmodel/compute/attention_engine.c",
    "tu_cmodel/compute/attention_engine.h", "tests/test_attention.c",
    "tests/test_attention_staging.c", "tests/test_attention_staging.py", "tests/test_framework.h",
    "tests/test_attention_kv.c",
]
FLAGS = ["-O1", "-g", "-Wall", "-Wextra", "-std=c11", "-fPIC",
         "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT or ROOT in out.parents or out.exists():
        parser.error("output must be a new directory outside the checkout")
    out.mkdir(parents=True)
    source = out / "source"
    source.mkdir()
    # Only build inputs; unrelated example symlinks may point outside the tree.
    archive = subprocess.check_output(["git", "archive", "HEAD", "--", "Makefile", "tu_cmodel"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    hashes = {}
    for name in INPUTS:
        data = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(data)
    (out / "inputs.json").write_text(json.dumps({
        "base_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "overlaid_inputs_sha256": hashes,
    }, indent=2) + "\n")
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    records = []

    def run(label, cmd, expected=0, diagnostic=None):
        result = subprocess.run(cmd, cwd=source, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=240)
        log = result.stdout.decode(errors="replace")
        (out / f"{label}.log").write_bytes(result.stdout)
        records.append({"label": label, "command": cmd, "cwd": str(source),
                        "returncode": result.returncode, "expected": expected,
                        "log_sha256": hashlib.sha256(result.stdout).hexdigest()})
        (out / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode != expected:
            raise RuntimeError(f"{label}: expected {expected}, got {result.returncode}")
        if diagnostic:
            if diagnostic not in log:
                raise RuntimeError(f"{label}: required diagnostic missing: {diagnostic}")
        elif "ERROR: AddressSanitizer" in log or "runtime error:" in log:
            raise RuntimeError(f"{label}: unexpected sanitizer diagnostic")
        return log

    run("sanitized-build", ["make", "-j4", "CFLAGS=" + " ".join(FLAGS)])
    cc = shutil.which("cc")
    if not cc:
        raise RuntimeError("cc is required")
    binary = out / "staging-asan"
    run("sanitized-link", [cc, *FLAGS, "-I.", "-Itu_cmodel",
                           "tests/test_attention_staging.c", "./libtucmodel.a", "-lm", "-o", str(binary)])
    log = run("sanitized-matrix", [str(binary)])
    if "cases=720 checks=67034 failures=0" not in log:
        raise RuntimeError("complete matrix summary missing")
    legacy = out / "attention-asan"
    run("sanitized-legacy-link", [cc, *FLAGS, "-I.", "-Itu_cmodel",
                                  "tests/test_attention.c", "./libtucmodel.a", "-lm", "-o", str(legacy)])
    run("sanitized-legacy", [str(legacy)])
    kv = run("kv-regression", [str(binary), "--kv-probe"])
    if "first_output=0.25 expected=0.25" not in kv:
        raise RuntimeError("former KV counterexample did not return 0.25")
    kv_binary = out / "kv-asan"
    run("sanitized-kv-link", [cc, *FLAGS, "-Werror", "-I.", "-Itu_cmodel",
                              "tests/test_attention_kv.c", "./libtucmodel.a", "-lm", "-o", str(kv_binary)])
    kv_log = run("sanitized-kv-matrix", [str(kv_binary)])
    if "attention KV: cases=2565 checks=457309 failures=0" not in kv_log:
        raise RuntimeError("complete cross-KV gate missing")
    rows = re.findall(r"^KV_ROW,(?:[0-9]+,){10}[0-9.e+-]+$", kv_log, re.MULTILINE)
    if len(rows) != 2565 or len(set(rows)) != 2565:
        raise RuntimeError("truncated/duplicated/interleaved KV matrix rows")
    reuse = run("sanitized-query-reuse-sweep", [str(kv_binary), "--reuse-sweep"])
    if len(re.findall(r"^KV_ROW,", reuse, re.MULTILINE)) != 6:
        raise RuntimeError("query reuse matrix incomplete")
    auto = run("auto-tiler-known-defect", [str(kv_binary), "--auto-probe"], expected=1)
    if "chosen=16x512 auto_status=-1 manual=8x512 manual_status=0" not in auto:
        raise RuntimeError("auto-tiler counterexample changed; investigate")
    alloc_binary = out / "kv-alloc-asan"
    run("sanitized-alloc-link", [cc, *FLAGS, "-Werror", "-DTU_ATTN_TEST_ALLOC", "-I.", "-Itu_cmodel",
                                 "tests/test_attention_kv.c", "./libtucmodel.a", "-lm",
                                 "-Wl,--wrap=malloc", "-o", str(alloc_binary)])
    run("sanitized-alloc-failures", [str(alloc_binary), "--alloc-probe"])

    production = source / "tu_cmodel/compute/attention_engine.c"
    current = production.read_text()
    mutations = {
        "word-width": [
            ("tu_sram_read_bulk(sram, src_addr, &val, sizeof(val))", "tu_sram_read(sram, src_addr, &val)"),
            ("tu_sram_write_bulk(sram, dst_addr, &val, sizeof(val))", "tu_sram_write(sram, dst_addr, &val)"),
            ("tu_sram_write_bulk(sram, dst_addr, &h, sizeof(h))", "tu_sram_write(sram, dst_addr, &h)"),
        ],
        "reverse-pack": [
            ("for (uint32_t idx = 0; idx < elem_count; idx++) {",
             "for (uint32_t i = 1; i <= elem_count; i++) { uint32_t idx = elem_count - i;"),
        ],
        "packed-p-layout": [
            ("uint32_t O_off_aligned = s_tile_bytes;",
             "uint32_t O_off_aligned = ((s_tile_bytes / 2 + 3) / 4) * 4;"),
        ],
    }
    for name, substitutions in mutations.items():
        changed = current
        for old, new in substitutions:
            if changed.count(old) != 1:
                raise RuntimeError(f"mutation anchor drift: {name}: {old}")
            changed = changed.replace(old, new)
        # Keep relative includes and avoid rebuilding/mutating the good archive.
        production.write_text(changed)
        directory = out / name
        directory.mkdir()
        obj = directory / "attention_engine.o"
        lib = directory / "libtucmodel.a"
        shutil.copyfile(source / "libtucmodel.a", lib)
        run(name + "-compile", [cc, *FLAGS, "-Itu_cmodel", "-c", str(production), "-o", str(obj)])
        run(name + "-archive", ["ar", "rcs", str(lib), str(obj)])
        mutant = directory / "gate"
        run(name + "-link", [cc, *FLAGS, "-I.", "-Itu_cmodel",
                             "tests/test_attention_staging.c", str(lib), "-lm", "-o", str(mutant)])
        if name == "word-width":
            run(name + "-rejection", [str(mutant), "--smoke"], expected=1,
                diagnostic="ERROR: AddressSanitizer: stack-buffer-overflow")
        else:
            rejected = run(name + "-rejection", [str(mutant)], expected=1)
            if "FAIL line" not in rejected:
                raise RuntimeError(f"{name}: no numerical/admission assertion caught mutant")
        production.write_text(current)
    streaming_mutations = {
        "lost-old-normalizer": ("alpha * old_sum + (float)tile_sum", "(float)tile_sum"),
        "missing-output-rescale": ("out *= alpha;", "out *= 1.0f;"),
        "missing-final-division": ("out = sum == 0.0f ? 0.0f : out / sum;", "out = out;"),
        "custom-mask-pitch": ("((size_t)h * M + qi + r) * N + kvi + j",
                              "((size_t)h * M + qi) * N + kvi + r * kv_count + j"),
        "missing-state-footprint": ("o_end + s_tile_bytes + state_bytes", "o_end + s_tile_bytes"),
    }
    for name, (old, new) in streaming_mutations.items():
        if current.count(old) != 1:
            raise RuntimeError(f"mutation anchor drift: {name}")
        production.write_text(current.replace(old, new))
        directory = out / name
        directory.mkdir()
        obj = directory / "attention_engine.o"
        lib = directory / "libtucmodel.a"
        shutil.copyfile(source / "libtucmodel.a", lib)
        run(name + "-compile", [cc, *FLAGS, "-Itu_cmodel", "-c", str(production), "-o", str(obj)])
        run(name + "-archive", ["ar", "rcs", str(lib), str(obj)])
        mutant = directory / "gate"
        run(name + "-link", [cc, *FLAGS, "-I.", "-Itu_cmodel",
                             "tests/test_attention_kv.c", str(lib), "-lm", "-o", str(mutant)])
        rejected = run(name + "-rejection", [str(mutant)], expected=1)
        if "FAIL " not in rejected:
            raise RuntimeError(f"{name}: no assertion caught mutant")
        production.write_text(current)
    summary = {"status": "PASS", "cases": 720, "checks": 67034,
               "kv_cases": 2565, "kv_checks": 457309,
               "mutants_rejected": list(mutations),
               "kv_mutants_rejected": list(streaming_mutations),
               "former_kv_defect": "0.25 exactly; corrected",
               "auto_tiler_known_defect": "auto 16x512 rejects; explicit 8x512 succeeds",
               "commands": len(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
