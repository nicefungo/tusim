import gzip,hashlib,io,json,os,pathlib,subprocess,tarfile,tempfile
R=pathlib.Path.cwd(); O=R/'docs/exploration/evidence/2026-10-21-im2col-point'
tree=subprocess.check_output(['git','write-tree'],text=True).strip()
records=[]
with tempfile.TemporaryDirectory(prefix='tusim-point-index-',dir=os.environ['TMPDIR']) as td:
    D=pathlib.Path(td)
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',tree]))) as t:
        t.extractall(D,members=[m for m in t.getmembers() if m.isfile() or m.isdir()],filter='data')
    for line in (O/'INPUT_SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1)
        assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h,n
    for label,cmd in [('index-build',['make']),('index-quick',['make','test-quick']),
                      ('index-focused',['make','test-agen-im2col-point','test-agen-im2col-bounds','test-config','test-conv','test-scatter-gather'])]:
        p=subprocess.run(cmd,cwd=D,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=480)
        out=O/(label+'.log.gz'); assert not out.exists()
        out.write_bytes(gzip.compress(p.stdout,mtime=0))
        records.append(dict(label=label,args=cmd,returncode=p.returncode))
        (O/'index-commands.json').write_text(json.dumps(dict(staged_tree=tree,inputs_verified=True,commands=records),indent=2)+'\n')
        print(label,p.returncode,flush=True)
        assert p.returncode==0,p.stdout.decode(errors='replace')[-4000:]
# Exact raw artifact manifest; never normalize log content.
(O/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(O.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
