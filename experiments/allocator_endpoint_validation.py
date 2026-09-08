"""Profile and verify a corrected full frontier against retained allocation."""
import csv
import hashlib
import json
import os
from pathlib import Path
import pstats
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request

root = Path('/mnt/shared/tessera-measurements/first-model-20260907/allocation-preparation-01')
inv = json.loads((root/'frontier-invocation-01.json').read_text())
reference = root/'validation-pair-01/after/frontier'
out = root/'endpoint-validation-01'
out.mkdir(exist_ok=False)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

assert sha(root/'joint-allocation-cost.pkl') == inv['input_sha256']
stop = threading.Event()
errors = []
query = urllib.parse.urlencode({'format':'json','filter':'system.cpu system.ram system.io system.load nvidia_smi.*_power_draw'})
def sample(host,address):
    with (out/(host+'-netdata.jsonl')).open('w') as stream:
        while not stop.is_set():
            row={'host':host,'time':time.time()}
            try:
                with urllib.request.urlopen('http://'+address+':19999/api/v1/allmetrics?'+query,timeout=3) as response:
                    raw=response.read(131073)
                assert len(raw)<=131072
                row['metrics']=json.loads(raw)
                assert 'system.cpu' in row['metrics']
            except Exception as exc:
                row['error']=repr(exc);errors.append(row)
            stream.write(json.dumps(row)+'\n');stream.flush();stop.wait(5)
threads=[threading.Thread(target=sample,args=host) for host in [
    ('dl380g10','127.0.0.1'),('sparky','192.168.1.180'),('sparklina','192.168.1.110')]]
with tempfile.TemporaryDirectory(prefix='endpoint-profile-') as tmp:
    profile=Path(tmp)/'allocator.pstats'
    argv=[x.replace(str(root/'frontier-01'),str(out/'frontier')) for x in inv['command']]
    argv=argv[:1]+['-m','cProfile','-o',str(profile)]+argv[1:]
    (out/'invocation.json').write_text(json.dumps({'argv':argv,'env':inv['env'],'input_sha256':inv['input_sha256']},indent=2))
    for thread in threads:thread.start()
    start=time.time()
    try:
        with (out/'allocator.log').open('w') as log:
            result=subprocess.run(argv,env={**os.environ,**inv['env']},stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.time()-start
    finally:
        stop.set()
        for thread in threads:thread.join()
    shutil.copy2(profile,out/'allocator.pstats')
    with (out/'profile-top.txt').open('w') as stream:
        pstats.Stats(str(profile),stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
    assert result.returncode==0
new=out/'frontier'
with (reference/'pareto.csv').open() as a, (new/'pareto.csv').open() as b:
    before=list(csv.DictReader(a));after=list(csv.DictReader(b))
assert len(before)==len(after)==114
changed=[]
for a,b in zip(before,after):
    assert a['target_bits']==b['target_bits']
    if a!=b:
        assert float(b['target_bits'])==16.0
        changed.append(b['target_bits'])
        assert float(b['predicted_dloss'])==0.0 and float(b['achieved_bits'])==16.0
assert changed==['16.0']
terminal=json.loads((new/'terminal-bf16-control.json').read_text())
units={k:v for k,v in terminal.items() if k!='__prismaquant__'}
assert len(units)==2142
assert all(v['data_type']=='float' and v['bits']==16 for v in units.values())
# Compare every retained lower-budget assignment payload; only the 16-bpp
# endpoint's content-addressed name may be replaced by a new assignment.
a=json.loads((reference/'assignments/manifest.json').read_text())
b=json.loads((new/'assignments/manifest.json').read_text())
old={float(x['target_bits']):x for x in a['candidates']}
current={float(x['target_bits']):x for x in b['candidates']}
assert old.keys()==current.keys()
unchanged=[]
for target in old:
    if target==16.0:continue
    left=Path(old[target]['path']).read_bytes();right=Path(current[target]['path']).read_bytes()
    assert left==right,target
    unchanged.append(target)
assert not errors
assert sha(root/'joint-allocation-cost.pkl')==inv['input_sha256']
report={'returncode':result.returncode,'elapsed_s':elapsed,'started_unix':start,
        'targets':114,'changed_targets':changed,'unchanged_assignment_targets':unchanged,
        'bf16_units':len(units),'netdata_errors':errors,
        'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
(out/'result.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='artifacts'}),flush=True)
