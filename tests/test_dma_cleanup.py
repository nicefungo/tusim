#!/usr/bin/env python3
"""Fresh all-object sanitizer gate for DMA fixture ownership and failure paths.

Usage: python3 tests/test_dma_cleanup.py --output /absolute/new/output-directory
Writes a source snapshot, full logs, commands/statuses, and SHA-256 manifests.
Only the selected output directory is written; the checkout build is untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FLAGS = '-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer'
FAULT_KEYS = ('DMA_CLEANUP_FAIL_ALLOC', 'DMA_CLEANUP_REJECT_SUBMIT',
              'DMA_CLEANUP_EARLY_DATA', 'DMA_CLEANUP_HOLD_ACTIVE')
SAN_ERROR = re.compile(r'AddressSanitizer:|LeakSanitizer:|runtime error:')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('output must be outside the checkout')
    output.mkdir(parents=True, exist_ok=False)
    build = output / 'source'
    build.mkdir()
    shutil.copy2(ROOT / 'Makefile', build / 'Makefile')
    for name in ('tu_cmodel', 'tests', 'config', 'scripts'):
        shutil.copytree(ROOT / name, build / name,
                        ignore=shutil.ignore_patterns('*.o', '*.a', '*.so', '*.pyc', '__pycache__'))
    inputs = {str(p.relative_to(build)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(build.rglob('*')) if p.is_file()}
    (output / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    records = []
    env = dict(os.environ, LD_LIBRARY_PATH=str(build),
               ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    for key in FAULT_KEYS:
        env.pop(key, None)
    env.pop('LSAN_OPTIONS', None)  # Do not inherit suppression files.

    def run(name, command, extra=None, expected=None, marker=None):
        result = subprocess.run(command, cwd=build, env=dict(env, **(extra or {})),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=600, text=True)
        (output / (name + '.log')).write_text(result.stdout)
        record = dict(name=name, command=command, environment=extra or {},
                      exit=result.returncode)
        records.append(record)
        (output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        if SAN_ERROR.search(result.stdout):
            raise AssertionError(f'{name}: sanitizer diagnostic; inspect {output / (name + ".log")}')
        if expected is not None and result.returncode not in expected:
            raise AssertionError(f'{name}: exit {result.returncode}, expected {expected}')
        if marker and marker not in result.stdout:
            raise AssertionError(f'{name}: missing injection marker {marker!r}')
        return result

    make_flags = ['CFLAGS=' + FLAGS, 'LDFLAGS=-lm -fsanitize=address,undefined']
    run('build', ['make', *make_flags], expected=(0,))
    run('normal', ['make', 'test-dma', 'test-dma-drr-scan-width',
                   'test-dma-drr-stage-width', *make_flags], expected=(0,))
    cc = shlex.split(os.environ.get('CC', 'cc'))
    run('fault-build', [*cc, *shlex.split(FLAGS), '-I.', '-Itu_cmodel',
        'tests/test_dma.c', 'tests/dma_cleanup_faults.c', './libtucmodel.a',
        '-Wl,--wrap=calloc', '-Wl,--wrap=tu_dma_submit_desc',
        '-Wl,--wrap=tu_dma_tick', '-lm', '-fsanitize=address,undefined',
        '-o', 'dma-cleanup-faults'], expected=(0,))
    control = run('fault-control', ['./dma-cleanup-faults'], expected=(0,))
    assert '15/15 tests passed' in control.stdout
    count = re.search(r'CLEANUP_COUNTS allocations=(\d+) submissions=(\d+) ticks=(\d+)',
                      control.stdout)
    assert count, 'missing wrapper coverage counts'
    allocs, submits, _ = map(int, count.groups())
    assert allocs > 0 and submits > 0
    for index in range(1, allocs + 1):
        result = run(f'oom-{index:03}', ['./dma-cleanup-faults'],
                     {'DMA_CLEANUP_FAIL_ALLOC': str(index)}, expected=(1,),
                     marker=f'INJECT allocation {index}\n')
        assert re.search(r'\d+/15 tests passed', result.stdout), 'suite did not finish'
    rejection_detected = rejection_already_expected = 0
    for index in range(1, submits + 1):
        result = run(f'reject-{index:03}', ['./dma-cleanup-faults'],
                     {'DMA_CLEANUP_REJECT_SUBMIT': str(index)}, expected=(0, 1),
                     marker=f'INJECT rejection {index}\n')
        assert re.search(r'\d+/15 tests passed', result.stdout), 'suite did not finish'
        rejection_detected += result.returncode == 1
        rejection_already_expected += result.returncode == 0
    run('early-data', ['./dma-cleanup-faults'], {'DMA_CLEANUP_EARLY_DATA': '1'},
        expected=(1,), marker='INJECT pre-tick data ')
    run('hold-active', ['./dma-cleanup-faults'], {'DMA_CLEANUP_HOLD_ACTIVE': '1'},
        expected=(1,), marker='INJECT active hold\n')
    summary = dict(status='PASS', normal_tests=15, descriptor_oom_cases=allocs,
                   rejection_cases=submits, rejection_detected=rejection_detected,
                   rejection_already_expected=rejection_already_expected,
                   additional_failure_controls=2,
                   source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                       cwd=ROOT, text=True).strip(),
                   sanitizer_flags=FLAGS,
                   environment={k: env[k] for k in ('ASAN_OPTIONS', 'UBSAN_OPTIONS')})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.glob('*.log'))}
    (output / 'log_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
