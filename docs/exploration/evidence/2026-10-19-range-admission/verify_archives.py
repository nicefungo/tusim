from pathlib import Path
import subprocess, tempfile, tarfile, io, gzip, shutil, os
root=Path('/home/zxy/Workplace/projects/onnx-compiler')
e=root/'docs/exploration/evidence/2026-10-19-range-admission'
commands=[]
def run(args,cwd,log,env=None):
    with (e/(log+'.gz')).open('wb') as dest:
        p=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
        dest.write(gzip.compress(p.stdout,mtime=0))
    commands.append(f'{log}: cwd={cwd}; argv={args!r}; exit={p.returncode}')
    print(commands[-1],flush=True)
    return p.returncode

def snapshot(prefix,overlay=False):
    d=Path(tempfile.mkdtemp(prefix=prefix))
    blob=subprocess.check_output(['git','archive','HEAD'],cwd=root)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(d, members=(m for m in tar.getmembers() if not m.issym() and not m.islnk()))
    if overlay:
        for name in ['Makefile','tu_cmodel/memory/address_generator.c','tu_cmodel/memory/address_generator.h','tests/test_agen_range_admission.c']:
            shutil.copy2(root/name,d/name)
    return d
baseline=snapshot('tusim-range-baseline-')
assert run(['make'],baseline,'baseline-archive-build')==0
assert run(['make','test'],baseline,'baseline-archive-full')!=0
assert run(['make','test'],root,'final-full')!=0
san=snapshot('tusim-range-san-',True)
flags='-O1 -g -Wall -Wextra -std=c11 -fPIC -fsanitize=address,undefined -fno-omit-frame-pointer'
assert run(['make','libtucmodel.a','CFLAGS='+flags],san,'san-build')==0
env=os.environ.copy(); env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
assert run(['make','test-agen-range-admission','test-agen-dma-chain','test-agen-affine-bounds','test-agen-tiled-sweep','CFLAGS='+flags,'LDFLAGS=-lm -fsanitize=address,undefined'],san,'san-tests',env)==0
(e/'commands.txt').write_text('\n'.join(commands)+'\nASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 for san-tests\n')
for n in ('final-build.log','final-quick.log','final-focused.log'):
    (e/(n+'.gz')).write_bytes(gzip.compress((Path(os.environ['TMPDIR'])/('tusim-'+n)).read_bytes(),mtime=0))
print('archive/full/sanitizer verification complete')
