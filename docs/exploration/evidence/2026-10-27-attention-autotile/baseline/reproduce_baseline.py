#!/usr/bin/env python3
"""Rebuild the recorded pre-change HEAD and run its safe auto-tiler diagnostic."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/home/zxy/Workplace/projects/onnx-compiler')
out = Path('/home/zxy/.hermes/cache/scratch/tusim-20261027/baseline-archive')
out.mkdir()
source = out / 'source'
source.mkdir()
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
archive = subprocess.check_output(['git','archive',head,'--','Makefile','tu_cmodel','tests/test_attention_kv.c'],cwd=root)
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    tar.extractall(source,filter='data')
records=[]
for label,command,expected in [
    ('build',['make','-j4'],0),
    ('link',['cc','-O2','-Wall','-Wextra','-Werror','-std=c11','-I.','-Itu_cmodel','tests/test_attention_kv.c','./libtucmodel.a','-lm','-o','auto-baseline'],0),
    ('probe',['./auto-baseline','--auto-probe'],1),
]:
    result=subprocess.run(command,cwd=source,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (out/(label+'.log')).write_bytes(result.stdout)
    records.append({'label':label,'command':command,'expected':expected,'returncode':result.returncode,'sha256':hashlib.sha256(result.stdout).hexdigest()})
    if result.returncode!=expected:
        raise RuntimeError(records[-1])
text=(out/'probe.log').read_text()
assert 'chosen=16x512 auto_status=-1 manual=8x512 manual_status=0' in text
(out/'commands.json').write_text(json.dumps({'input_head':head,'commands':records},indent=2)+'\n')
print(text)
