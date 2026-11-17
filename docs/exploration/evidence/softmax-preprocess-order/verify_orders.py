"""Reproducible local baseline, mutation and sanitizer evidence; no Git writes."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(os.environ.get('TUSIM_ROOT', os.getcwd())).resolve()
OUT = Path(os.environ.get('TUSIM_EVIDENCE', '/home/zxy/.hermes/cache/scratch/tusim-20261117')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
BASE = '04d3c02310ceaf0cf549ed78dd1b2d1470e9e78e'
INPUTS = ['tu_cmodel/compute/softmax_engine.c','tu_cmodel/compute/softmax_engine.h',
          'tests/test_softmax_preprocess_order.c','Makefile','.gitignore']
def run(args, cwd, log, expected=0, env=None):
    with (OUT/log).open('wb') as f:
        f.write(('COMMAND: '+repr(args)+'\nCWD: '+str(cwd)+'\n').encode()); f.flush()
        p = subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=env)
        f.write(('\nEXIT: '+str(p.returncode)+'\n').encode())
    if p.returncode != expected: raise RuntimeError((args,p.returncode,log))
def extract(name, overlay=False):
    dest=OUT/name; dest.mkdir(exist_ok=True)
    archive=subprocess.run(['git','archive',BASE],cwd=ROOT,capture_output=True,check=True).stdout
    subprocess.run(['tar','-x','-C',str(dest)],input=archive,check=True)
    if overlay:
        for path in INPUTS: shutil.copy2(ROOT/path,dest/path)
    return dest
cc=['cc','-O2','-Wall','-Wextra','-std=c11','-I.','-Itu_cmodel']
if sys.argv[1]=='baseline':
    dest=extract('old')
    shutil.copy2(ROOT/INPUTS[2],dest/INPUTS[2])
    run(['make'],dest,'old-build.log')
    run(cc+['-DTU_TEST_OLD_ORDER',INPUTS[2],'./libtucmodel.a','-lm','-o','order-old'],dest,'old-compile.log')
    run(['./order-old'],dest,'old-red.log',expected=1)
    text=(OUT/'old-red.log').read_text()
    assert 'FAIL case=' in text and 'failures=0' not in text
    for args,name in [(['./order-old','--legacy'],'old-legacy-run.log'),
                      (['make','test-softmax-exp-policy','test-softmax-log-nearzero','test-softmax-log-range','test-softmax-sweep','test-softmax-attention-sweep'],'old-focused.log')]:
        run(args,dest,name)
    for args,name in [(['./order-old','--legacy'],'old-legacy.txt'),
                      (['./test-softmax-exp-policy','--matrix'],'old-exp-matrix.csv'),
                      (['./test-softmax-sweep'],'old-mode-sweep.txt'),
                      (['./test-softmax-attention-sweep'],'old-attention-sweep.txt')]:
        with (OUT/name).open('wb') as f: subprocess.run(args,cwd=dest,stdout=f,check=True)
    assert (OUT/'old-legacy.txt').read_bytes()==(OUT/'new-legacy.txt').read_bytes()
    print('old-source RED and legacy bitwise fingerprint comparison PASS')
elif sys.argv[1]=='sanitizer':
    dest=extract('sanitized',True)
    flags='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer'
    link='-lm -fsanitize=address,undefined'
    run(['make',f'CFLAGS={flags}',f'LDFLAGS={link}'],dest,'sanitizer-build.log')
    env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
    run(['make',f'CFLAGS={flags}',f'LDFLAGS={link}','test-softmax-preprocess-order','test-softmax-exp-policy','test-softmax-exp-policy-internal','test-softmax-log-nearzero','test-softmax-log-range','test-softmax','test-softmax-overflow'],dest,'sanitizer.log',env=env)
    run(['./test-softmax-preprocess-order','--matrix'],dest,'sanitizer-matrix.log',env=env)
    print('all-object ASan/UBSan/LSan focused and matrix PASS')
elif sys.argv[1]=='mutants':
    source=(ROOT/INPUTS[0]).read_text()
    mutants={
        'ignore-selection':('if (preprocess_order == TU_SOFTMAX_SCALE_THEN_MASK) {','if (false) {'),
        'change-default':('if (preprocess_order == TU_SOFTMAX_SCALE_THEN_MASK) {','if (true) {'),
        'double-scale':('        scale = 0.0f;','        /* mutant: preserve scale for second application */'),
        'zero-is-multiply':('        if (scale != 0.0f)\n            for','        if (true)\n            for'),
        'ignore-threshold':('    if (mask_fill != 0.0f) {','    if (mask_fill != 0.0f && preprocess_order == TU_SOFTMAX_MASK_THEN_SCALE) {'),
    }
    for name,(old,new) in mutants.items():
        assert source.count(old)==1,(name,source.count(old))
        path=OUT/(name+'.c'); path.write_text(source.replace(old,new))
        binary=OUT/name
        run(cc+['-Itu_cmodel/compute',str(path),INPUTS[2],'./libtucmodel.a','-lm','-o',str(binary)],ROOT,name+'-compile.log')
        run([str(binary)],ROOT,name+'.log',expected=1)
        assert 'FAIL case=' in (OUT/(name+'.log')).read_text()
        print(name,'rejected')
else: raise ValueError(sys.argv[1])
