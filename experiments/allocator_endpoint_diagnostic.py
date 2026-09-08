"""Inspect one real allocator solve from an existing probe; no GPU or reprobe."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

root = Path('/mnt/shared/tessera-measurements/first-model-20260907/allocation-preparation-01')
inv = json.loads((root / 'frontier-invocation-01.json').read_text())
with (root / 'joint-allocation-cost.pkl').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == inv['input_sha256']
out = root / 'endpoint-diagnostic-01'
out.mkdir(exist_ok=False)
os.environ.update(inv['env'])
from prismaquant import allocator_solver as solver
original = solver.solve_allocation

def inspect(stats, candidates, target_bits, bit_precision=0.001):
    result = original(stats, candidates, target_bits, bit_precision)
    total = sum(stats[n]['n_params'] for n in candidates)
    baseline = {n: min(cs, key=lambda c: c.bits_per_param) for n, cs in candidates.items()}
    minimum = sum(baseline[n].bits_per_param * stats[n]['n_params'] for n in candidates) / total
    best = {n: min(cs, key=lambda c: (c.predicted_dloss, c.bits_per_param)) for n, cs in candidates.items()}
    bf16 = {n: next((c for c in cs if c.fmt == 'BF16'), None) for n, cs in candidates.items()}
    assert all(c is not None for c in bf16.values())
    charges = {n: solver._charged_bins((c.bits_per_param-baseline[n].bits_per_param)*stats[n]['n_params']/total, bit_precision) for n,c in bf16.items()}
    payload = {'target_bits':target_bits, 'bit_precision':bit_precision, 'total_params':total,
      'min_bits':minimum, 'capacity_bins':round((target_bits-minimum)/bit_precision)+1,
      'all_bf16_bins':sum(charges.values()),
      'all_bf16_exact_bpp':sum(c.bits_per_param*stats[n]['n_params'] for n,c in bf16.items())/total,
      'all_bf16_loss':sum(c.predicted_dloss for c in bf16.values()),
      'selected_loss':sum(c.predicted_dloss for c in result[1].values()),
      'selected_non_bf16':{n:c.fmt for n,c in result[1].items() if c.fmt!='BF16'},
      'max_baseline_loss':max(c.predicted_dloss for c in baseline.values()),
      'stats':{n:{'n_params':stats[n]['n_params']} for n in candidates},
      'candidates':{n:[dataclasses.asdict(c) for c in cs] for n,cs in candidates.items()},
      'assignment':result[0], 'bf16_charges':charges, 'input_sha256':inv['input_sha256']}
    (out/'solver-inputs.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps({k:v for k,v in payload.items() if k not in ('stats','candidates','assignment','bf16_charges')}),flush=True)
    # Diagnostic intentionally stops after the first real solver invocation.
    raise SystemExit(0)

solver.solve_allocation = inspect
argv = inv['command'][3:]
for option in ('--pareto-output-dir','--pareto-csv','--layer-config'):
    i=argv.index(option)
    argv[i+1]=str(out/Path(argv[i+1]).name)
i=argv.index('--pareto-targets');argv[i+1]='16'
sys.argv=['prismaquant.allocator',*argv]
runpy.run_module('prismaquant.allocator',run_name='__main__')
