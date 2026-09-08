"""Profile menu construction for a real GLM expert shape; no tensor/GPU work."""
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import cProfile
import time

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
from prismaquant.tessera_campaign import expand_menus_for_targets
from prismaquant.tessera_menu import PARALLEL_NONE
name = 'model.language_model.layers.4.mlp.experts.0.gate_proj'
weights = {name: SimpleNamespace(shape=(2048, 4096))}
profile = cProfile.Profile()
begin_unix = time.time()
begin = time.perf_counter()
with profile:
    menus = expand_menus_for_targets(weights, [name], mode='readable',
        tp_degree=1, parallel_kind=PARALLEL_NONE)
seconds = time.perf_counter() - begin
profile.dump_stats(str(args.out/'menu.pstats'))
def json_default(value):
    if isinstance(value, Fraction):
        return dict(numerator=value.numerator, denominator=value.denominator)
    raise TypeError(f'unsupported menu field: {type(value).__name__}')

raw = json.dumps([asdict(row) for row in menus[name]], sort_keys=True,
                 separators=(',', ':'), default=json_default).encode()
(args.out/'menu.json').write_bytes(raw)
result = dict(scope='one_actual_GLM_expert_shape_menu_only_not_full_census',
    shape=[2048,4096], mode='readable', tp_degree=1,
    begin_unix=begin_unix, end_unix=time.time(), seconds=seconds, rows=len(menus[name]), menu_sha256=hashlib.sha256(raw).hexdigest())
(args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
