from pathlib import Path
import subprocess, tempfile, tarfile, io, gzip, shutil, os, json
root=Path('/home/zxy/Workplace/projects/onnx-compiler')
e=Path(__file__).resolve().parent
baseline_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
commands=[]
def run(args,cwd,log,env=None):
    p=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    (e/(log+'.log')).write_bytes(p.stdout)
    commands.append(dict(cwd=str(cwd),argv=args,exit=p.returncode,log=log+'.log'))
    (e/'commands.json').write_text(json.dumps(dict(baseline_head=baseline_head,commands=commands),indent=2)+'\n')
    print(log, p.returncode,flush=True)
    return p.returncode

def snapshot(prefix,overlay=False):
    d=Path(tempfile.mkdtemp(prefix=prefix,dir=e))
    blob=subprocess.check_output(['git','archive',baseline_head],cwd=root)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(d,members=(m for m in tar.getmembers() if m.isfile() or m.isdir()),filter='data')
    if overlay:
        for name in ['Makefile','tu_cmodel/memory/address_generator.c','tu_cmodel/memory/address_generator.h','tests/test_agen_im2col_bounds.c']:
            shutil.copy2(root/name,d/name)
    return d
baseline=snapshot('baseline-')
assert run(['make'],baseline,'baseline-archive-build')==0
assert run(['make','test'],baseline,'baseline-archive-full')!=0
san=snapshot('san-',True)
flags='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer'
assert run(['make','libtucmodel.a','CFLAGS='+flags],san,'san-build')==0
sanenv=os.environ.copy(); sanenv.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
assert run(['make','test-agen-im2col-bounds','test-agen-affine-bounds','test-agen-tiled-sweep','test-agen-dma-chain','test-agen-range-admission','CFLAGS='+flags,'LDFLAGS=-lm -fsanitize=address,undefined'],san,'san-tests',sanenv)==0
print('baseline archive and exercised sanitizers verified',flush=True)
