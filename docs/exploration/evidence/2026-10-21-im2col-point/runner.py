#!/usr/bin/env python3
"""Retain raw baseline, final, archive and all-object sanitizer observations.
Run from repository root. Existing logs are never overwritten.
"""
import gzip, hashlib, io, json, os, pathlib, subprocess, tarfile, tempfile
R=pathlib.Path.cwd()
O=R/'docs/exploration/evidence/2026-10-21-im2col-point'
O.mkdir(parents=True,exist_ok=True)
commands=[]
HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
CHANGED=['.gitignore','Makefile','tu_cmodel/memory/address_generator.c',
         'tu_cmodel/memory/address_generator.h','tests/test_agen_im2col_point.c']

def run(label,args,cwd=R,expected=0,env=None):
    dest=O/(label+'.log.gz')
    if dest.exists(): raise RuntimeError('Refusing to overwrite '+str(dest))
    p=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                     env=env,timeout=480)
    dest.write_bytes(gzip.compress(p.stdout,mtime=0))
    commands.append(dict(label=label,args=args,cwd=str(cwd),returncode=p.returncode))
    (O/'commands.json').write_text(json.dumps(dict(input_head=HEAD,commands=commands),indent=2)+'\n')
    print(label,p.returncode,flush=True)
    if expected is not None and p.returncode!=expected:
        raise RuntimeError(p.stdout.decode(errors='replace')[-8000:])
    return p

def archive(dest,overlay=False):
    dest.mkdir()
    data=subprocess.check_output(['git','archive',HEAD])
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        # Skip external symlink fixtures: these C-only gates do not use them.
        members=[m for m in t.getmembers() if m.isfile() or m.isdir()]
        t.extractall(dest,members=members,filter='data')
    if overlay:
        for name in CHANGED:
            p=dest/name; p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes((R/name).read_bytes())

with tempfile.TemporaryDirectory(prefix='tusim-point-',dir=os.environ['TMPDIR']) as scratch:
    S=pathlib.Path(scratch); B=S/'baseline'; F=S/'final'; A=S/'asan'
    archive(B)
    run('baseline-archive-build',['make'],B)
    run('baseline-archive-quick',['make','test-quick'],B)
    # New test, old complete archive, static linkage; no unsafe address reads.
    (B/'tests/test_agen_im2col_point.c').write_bytes((R/'tests/test_agen_im2col_point.c').read_bytes())
    run('baseline-probe-build',['cc','-O2','-Wall','-Wextra','-std=c11','-I.','-Itu_cmodel',
          '-o','point-probe','tests/test_agen_im2col_point.c','./libtucmodel.a','-lm'],B)
    p=run('baseline-point-red',['./point-probe','--red'],B,expected=1)
    assert b'checks=11 failures=11' in p.stdout
    run('baseline-point-full',['./point-probe'],B,expected=1)
    run('baseline-full',['make','test'],B,expected=None)
    for n in range(1,4): run('baseline-attention-repeat-'+str(n),['./test-attention'],B,expected=None)
    archive(F,True)
    hashes=[]
    for p in sorted((F/'tu_cmodel').rglob('*')):
        if p.is_file() and p.suffix in ('.c','.h'): hashes.append((str(p.relative_to(F)),hashlib.sha256(p.read_bytes()).hexdigest()))
    for name in CHANGED+['config/tu_config.yaml','scripts/gen_config.py']:
        hashes.append((name,hashlib.sha256((F/name).read_bytes()).hexdigest()))
    (O/'INPUT_SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(set(hashes))))
    run('final-archive-build',['make'],F)
    run('final-archive-quick',['make','test-quick'],F)
    focused=['make','test-agen-im2col-point','test-agen-im2col-bounds','test-config','test-conv','test-scatter-gather']
    run('final-archive-focused',focused,F)
    run('final-archive-full',['make','test'],F,expected=None)
    run('live-clean',['make','clean'])
    run('live-build',['make'])
    run('live-quick',['make','test-quick'])
    run('live-focused',focused)
    archive(A,True)
    cf='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer'
    ld='-lm -fsanitize=address,undefined'
    run('sanitizer-build',['make','CFLAGS='+cf,'LDFLAGS='+ld,'libtucmodel.a'],A)
    e=os.environ.copy(); e['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1'; e['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1'
    run('sanitizer-focused',['make','CFLAGS='+cf,'LDFLAGS='+ld,'test-agen-im2col-point',
          'test-agen-im2col-bounds','test-agen-affine-bounds','test-agen-tiled-sweep',
          'test-agen-dma-chain','test-agen-range-admission'],A,env=e)
print('Audit finished; full-suite returncodes remain observations, not acceptance.',flush=True)
