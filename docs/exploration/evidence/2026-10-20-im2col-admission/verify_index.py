from pathlib import Path
import subprocess,tempfile,tarfile,io,json,os
root=Path('/home/zxy/Workplace/projects/onnx-compiler')
p=Path(__file__).resolve().parent
commands=[]
def run(argv,cwd,name,expected=0):
    r=subprocess.run(argv,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (p/(name+'.log')).write_bytes(r.stdout)
    commands.append(dict(argv=argv,cwd=str(cwd),exit=r.returncode,log=name+'.log'))
    (p/'index-commands.json').write_text(json.dumps(dict(tree=tree,commands=commands),indent=2)+'\n')
    print(name,r.returncode,flush=True)
    assert r.returncode==expected,(name,r.returncode)

os.environ.pop('GIT_INDEX_FILE',None)
tree=subprocess.check_output(['git','write-tree'],cwd=root,text=True).strip()
names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).splitlines()
base={'Makefile','.gitignore','tu_cmodel/memory/address_generator.c','tu_cmodel/memory/address_generator.h',
      'tests/test_agen_im2col_bounds.c','docs/exploration/im2col-geometry-admission.md','docs/exploration/IMPLEMENTATION_BACKLOG.md'}
assert all(n in base or n.startswith('docs/exploration/evidence/2026-10-20-im2col-admission/') for n in names)
assert base.issubset(names)
assert not any('HANDOFF' in n.upper() or n.endswith('/TERMS.md') for n in names)
d=Path(tempfile.mkdtemp(prefix='index-',dir=p))
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',tree],cwd=root))) as tar:
    tar.extractall(d,members=(m for m in tar.getmembers() if m.isfile() or m.isdir()),filter='data')
run(['make','clean'],d,'index-clean')
run(['make'],d,'index-build')
run(['make','test-quick'],d,'index-quick')
run(['make','test-config','test-conv','test-scatter-gather'],d,'index-focused')
baseline=Path(json.loads((p/'commands.json').read_text())['commands'][0]['cwd'])
red=p/'final-regression-against-baseline'
run(['cc','-O2','-Wall','-Wextra','-std=c11','-I.','-Itu_cmodel',str(root/'tests/test_agen_im2col_bounds.c'),str(baseline/'libtucmodel.a'),'-lm','-o',str(red)],baseline,'baseline-regression-build')
run([str(red)],baseline,'baseline-regression-red',expected=1)
print('Exact staged source snapshot verified; final gate rejects baseline',flush=True)
