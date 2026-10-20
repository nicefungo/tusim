from pathlib import Path
import subprocess,json
p=Path(__file__).resolve().parent
context=json.loads((p/'commands.json').read_text())
baseline=Path(context['commands'][0]['cwd'])
summary=[]
for i in range(1,9):
    r=subprocess.run(['./test-attention'],cwd=baseline,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    name=f'baseline-attention-repeat-{i}.log'
    (p/name).write_bytes(r.stdout)
    lines=[x for x in r.stdout.decode(errors='replace').splitlines() if 'FAIL:' in x or '/9 tests passed' in x]
    summary.append(dict(attempt=i,argv=['./test-attention'],cwd=str(baseline),exit=r.returncode,log=name,observations=lines))
    print(i,r.returncode,lines)
(p/'attention-repeats.json').write_text(json.dumps(dict(baseline_head=context['baseline_head'],runs=summary),indent=2)+'\n')
