"""Profile old/new shared expert packing on one real profile-selected MoE layer via PB."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

import torch

BASELINE_COMMIT = '1c7492333'


def tensor_digest(value):
    digest = hashlib.sha256()
    # Bound CPU staging to one expert/projection slab; hash original BF16 bytes.
    for slab in value if value.ndim == 3 else value.unsqueeze(0):
        raw = slab.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--baseline-file', type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    from prismaquant.layer_streaming import _read_layer_to_device, _pack_per_expert_into_packed
    from prismaquant.model_profiles import detect_profile
    profile = detect_profile(str(args.model))
    baseline_bytes = args.baseline_file.read_bytes()
    assert hashlib.sha256(baseline_bytes).hexdigest() == args.baseline_sha256
    source = baseline_bytes.decode()
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_pack_per_expert_into_packed')
    namespace = {'torch':torch, 'defaultdict':defaultdict}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<recorded baseline packer>', 'exec'), namespace)
    old_pack = namespace['_pack_per_expert_into_packed']
    old_code = ast.get_source_segment(source, fn)
    (args.out/'baseline-packer.py').write_text(old_code+'\n')
    prefix = f'{profile.body_layer_prefix()}.{args.layer}.'
    from prismaquant.artifact_completeness import read_artifact_header
    header = read_artifact_header(args.model)
    index_path = args.model/'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text())['weight_map']
    else:
        assert (args.model/'model.safetensors').is_file()
        index = dict.fromkeys(header, 'model.safetensors')
    pattern = re.compile(profile.per_expert_moe_regex().removeprefix('re:'))
    def is_expert(name):
        return bool(pattern.match(name) or pattern.match(profile.to_vllm_internal_name(name)))
    model_to_shard = {name:str(args.model/shard) for name, shard in index.items()
                      if name.startswith(prefix) and is_expert(name.removesuffix('.weight'))}
    model_to_ckpt = {name:name for name in model_to_shard}
    assert model_to_shard, 'selected source layer has no per-expert tensors'
    groups = defaultdict(lambda: defaultdict(dict))
    for name in model_to_shard:
        owner, projection = name.removesuffix('.weight').rsplit('.', 1)
        experts_path, expert = owner.rsplit('.', 1)
        parent = profile.packed_expert_parent_for_projection(projection)
        assert parent is not None
        groups[experts_path+'.'+parent][int(expert)][projection] = tuple(header[name]['shape'])
    shapes = {}
    for name, experts in groups.items():
        order = profile.packed_expert_projection_names(name.rsplit('.', 1)[1])
        assert set(experts) == set(range(len(experts)))
        assert all(projections == experts[0] for projections in experts.values())
        assert set(experts[0]) == set(order)
        shapes[name] = (len(experts), sum(experts[0][p][0] for p in order), experts[0][order[0]][1])
    callbacks = dict(is_per_expert=is_expert, parent_for_projection=profile.packed_expert_parent_for_projection,
        projection_names_for=profile.packed_expert_projection_names, live_param_shape=shapes.get)
    stopped = threading.Event()
    errors = []
    def monitor():
        with (args.out/'netdata.jsonl').open('w') as out:
            while not stopped.is_set():
                for host in ('sparky','sparklina'):
                    try:
                        with urllib.request.urlopen(f'http://{host}:19999/api/v1/allmetrics?format=json', timeout=5) as response:
                            metrics = json.load(response)
                        out.write(json.dumps(dict(host=host,time=time.time(),metrics=metrics))+'\n');out.flush()
                    except Exception as exc:
                        errors.append(dict(host=host,error=repr(exc)))
                stopped.wait(1)
    thread = threading.Thread(target=monitor, daemon=True);thread.start()
    result = dict(schema='prismaquant.expert_packer_ab.v1', model=str(args.model), profile=profile.name, layer=args.layer,
        baseline_commit=BASELINE_COMMIT, baseline_function_sha256=hashlib.sha256(old_code.encode()).hexdigest(),
        shapes=shapes, source_tensors=len(model_to_shard), device=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,cpu_affinity=sorted(os.sched_getaffinity(0)),
        arms=[],telemetry_errors=errors)
    expected_inputs = expected_outputs = None
    try:
        # ABBA closes order/cache effects. Source reads and byte audits are
        # outside the profiled pack interval; this measures packing only.
        for iteration, label in enumerate(('before','after','after','before')):
            torch.cuda.empty_cache()
            raw = _read_layer_to_device(prefix,model_to_shard,model_to_ckpt,torch.bfloat16,torch.device('cuda'))
            inputs = {name:tensor_digest(t) for name,t in sorted(raw.items())}
            if expected_inputs is None: expected_inputs = inputs
            assert inputs == expected_inputs, 'source bytes or dtype changed between arms'
            torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            initial_allocated, initial_reserved = torch.cuda.memory_allocated(),torch.cuda.memory_reserved()
            begin = time.time()
            pack = old_pack if label=='before' else _pack_per_expert_into_packed
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                produced = pack(raw, **callbacks)
            torch.cuda.synchronize()
            end = time.time()
            name = f'{iteration}-{label}'
            prof.export_chrome_trace(str(args.out/(name+'.trace.json')))
            (args.out/(name+'.profile.txt')).write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
            record = dict(iteration=iteration, arm=label,begin=begin,end=end,seconds=end-begin,
                initial_allocated=initial_allocated,initial_reserved=initial_reserved,
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                final_allocated=torch.cuda.memory_allocated(), final_reserved=torch.cuda.memory_reserved(),
                produced=produced)
            outputs = {name:tensor_digest(t) for name,t in sorted(raw.items())}
            assert produced==len(shapes) and set(raw)==set(shapes)
            if expected_outputs is None:expected_outputs=outputs
            assert outputs==expected_outputs,'packed bytes, shape or dtype differ'
            result['arms'].append(record)
            print(json.dumps(record),flush=True)
            del raw
        result['input_manifest_sha256']=hashlib.sha256(json.dumps(expected_inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        result['outputs']=expected_outputs
        result['exact_source_and_output_bytes_dtype']=True
    finally:
        stopped.set();thread.join(timeout=12)
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not errors,errors
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
