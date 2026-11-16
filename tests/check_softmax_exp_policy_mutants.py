#!/usr/bin/env python3
"""Reject numerical policy mutants and uncentralized exponential call sites."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'tu_cmodel/compute/softmax_engine.c').read_text()
test = (ROOT / 'tests/test_softmax_exp_policy.c').read_text()


def centralized(text):
    code = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    return (len(re.findall(r'\bexpf\s*\(', code)) == 1
            and 'softmax_exp(max_val - x, exp_policy, true)' in code
            and 'softmax_exp(x - max_val, exp_policy, true)' in code
            and 'softmax_exp(shifted, exp_policy, false)' in code)


if not centralized(source):
    raise SystemExit('exponential call-site centralization failed')
for old in ('softmax_exp(max_val - x, exp_policy, true)',
            'softmax_exp(x - max_val, exp_policy, true)'):
    if source.count(old) != 1 or centralized(source.replace(old, 'expf(x - max_val)')):
        raise SystemExit('online call-site validator failed negative control')
print('centralized exponential producer and both ONLINE call sites: pass; two bypass controls rejected')

mutations = {
    'ignore-descriptor-policy': ('desc->scale, mode, desc->exp_policy,',
                                 'desc->scale, mode, TU_SOFTMAX_EXP_LEGACY,'),
    'ftz-is-legacy-cutoff': ('policy == TU_SOFTMAX_EXP_LEGACY && !online_accumulator',
                            '(policy == TU_SOFTMAX_EXP_LEGACY || policy == TU_SOFTMAX_EXP_FTZ) && !online_accumulator'),
    'omit-result-ftz': ('policy == TU_SOFTMAX_EXP_FTZ && fpclassify(value) == FP_SUBNORMAL',
                        'false && fpclassify(value) == FP_SUBNORMAL'),
    'flush-gradual-results': ('policy == TU_SOFTMAX_EXP_FTZ && fpclassify(value) == FP_SUBNORMAL',
                              'policy != TU_SOFTMAX_EXP_LEGACY && fpclassify(value) == FP_SUBNORMAL'),
    'break-legacy-online-sum': ('&& !online_accumulator && x < -87.0f', '&& x < -87.0f'),
    'flush-final-probability': ('row[i] = softmax_exp(shifted, exp_policy, false) * inv_sum;',
                               'row[i] = softmax_exp(shifted, exp_policy, false) * inv_sum;\n'
                               '            if (exp_policy == TU_SOFTMAX_EXP_FTZ && fpclassify(row[i]) == FP_SUBNORMAL) row[i] = 0;'),
}
with tempfile.TemporaryDirectory(prefix='softmax-exp-mutants-', dir=os.environ['TMPDIR']) as tmp:
    for name, (old, new) in mutations.items():
        if source.count(old) != 1:
            raise SystemExit(f'nonunique anchor: {name}')
        path = Path(tmp) / f'{name}.c'
        harness = Path(tmp) / f'{name}-test.c'
        exe = Path(tmp) / name
        path.write_text(source.replace(old, new))
        harness.write_text(test.replace('#include "tu_cmodel/compute/softmax_engine.c"',
                                         f'#include "{path}"'))
        subprocess.run(['cc', '-O2', '-Wall', '-Wextra', '-std=c11', '-DTU_TEST_EXP_INTERNAL',
                        '-I', str(ROOT), '-I', str(ROOT / 'tu_cmodel'),
                        '-I', str(ROOT / 'tu_cmodel/compute'), str(harness),
                        str(ROOT / 'libtucmodel.a'), '-lm', '-o', str(exe)], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True)
        summary = re.search(r'exp-policy cases=\d+ checks=\d+ failures=(\d+).*', result.stdout)
        if result.returncode != 1 or not summary or int(summary[1]) == 0:
            raise SystemExit(f'mutant not rejected numerically: {name}\n{result.stdout}\n{result.stderr}')
        print(f'{name}: exit={result.returncode} {summary[0]}')
print('All six safe numerical mutants rejected.')
