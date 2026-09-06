#!/usr/bin/env python3
"""Opt-in full-body LFM trellis-family sanity preparation; never an exporter.

One resident model capture calibrates six dense W4A4 inputs. The explicit
weights-only plan and exact producer projection are sealed for a PB-owned
export campaign. This neither selects an optimum nor qualifies a serving lane.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TS_COMMIT = '6faa5ce314cadeee8a190cbeadcf6cde3a333efb'
IMAGE = 'sha256:47dd0e9aaa4e7a6575d21cfc661d96a47c0e35e87c64e850631e210bdf04ebc0'
SERVING_IMAGE = 'eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c'
SAMPLES, SEQLEN, SEED = 32, 512, 0
DENSE4 = {f'model.layers.{layer}.feed_forward.w{role}.weight'
          for layer in (0, 1) for role in (1, 2, 3)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify_source(root, manifest_path, expected_sha):
    require(sha(manifest_path) == expected_sha, 'Tessera manifest changed')
    manifest = read(manifest_path)
    require(set(manifest) == {'schema', 'commit', 'files'} and
            manifest['schema'] == 'prismaquant.lfm-mixed-tessera-source.v1' and
            manifest['commit'] == TS_COMMIT, 'wrong Tessera source seal')
    actual = {}
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if '.git' in rel.parts:
            continue
        require(not path.is_symlink(), f'unsealed source symlink: {rel}')
        if path.is_file():
            actual[str(rel)] = sha(path)
    require(actual and actual == manifest['files'], 'Tessera full source roster changed')
    return {'commit': TS_COMMIT, 'manifest_sha256': expected_sha, 'files': len(actual)}


def mixed_plan(producer, dense, packed, routed, config):
    """Derive classification, fusion and projection from the actual producer."""
    require(config.get('architectures') == ['Lfm2MoeForCausalLM'] and
            config.get('num_hidden_layers') == 24 and config.get('num_experts') == 32,
            'this opt-in layout requires the full 24-layer, 32-expert LFM source')
    routers = {n for n in dense if producer.MOE_ROUTER.fullmatch(n)}
    ordinary = set(dense) - routers
    require(len(ordinary) == 66 and len(routers) == 22 and DENSE4 <= ordinary,
            'dense/router population differs from the frozen full-body layout')
    unpacked = producer.expert_stacks(routed)
    packed_stacks = producer.packed_expert_stacks(packed)
    require(not packed_stacks and not packed, 'this source contract requires unpacked checkpoint experts')
    expected_stacks = {f'model.layers.{i}.feed_forward.experts' for i in range(2, 24)}
    require(set(unpacked) == expected_stacks, 'incomplete or unexpected routed stack roster')
    stack_plan = {n: {'grid': 'E4M3', 'q256': 1024, 'source_layout': producer.MOE_SOURCE_UNPACKED}
                  for n in sorted(unpacked)}
    projection = producer.project_expert_plan({**dense, **routed}, config, stack_plan)
    units = [u for s in projection['stacks'].values() for u in s['units']]
    require(len(units) == 2112 and len({u['tensor'] for u in units}) == 2112 and
            {u['tensor'] for u in units} == set(routed), 'producer projection lost or duplicated source units')
    plan = {n: {'grid': 'E2M1x2', 'q256': 896} if n in DENSE4 else
            {'grid': 'BF16', 'q256': 1792} for n in sorted(ordinary)}
    owners = {}
    for name in sorted(ordinary):
        fused = producer.fused_module(name)
        owner, members = fused if fused is not None else (producer.module_of(name), [name])
        require(set(members) <= ordinary, f'incomplete fused group: {owner}')
        require(all(plan[m] == plan[name] for m in members), f'mixed-family fused group: {owner}')
        for member in members:
            rows, cols = dense[member]
            arity = 2 if plan[member]['grid'] == 'E2M1x2' else 1
            require(rows % (arity * 32) == 0 and cols % 16 == 0,
                    f'explicit dense geometry cannot be encoded: {member}')
        record = {'members': list(members), **plan[name]}
        require(owner not in owners or owners[owner] == record, f'ambiguous fused owner: {owner}')
        owners[owner] = record
    require(len(owners) == 52 and sum(r['grid'] == 'E2M1x2' for r in owners.values()) == 4,
            'producer fusion differs from the expected 52 dense owners')
    plan.update(stack_plan)
    return plan, projection, {'schema': 'prismaquant.lfm-mixed-population.v1',
        'dense_owners': owners, 'dense_tensors': sorted(ordinary),
        'routed_stacks': sorted(unpacked), 'routed_tensors': sorted(routed),
        'retained_router_tensors': sorted(routers), 'quantized_body_matrices': 2178,
        'served_owners': 74, 'weights_only': True, 'hessian': None,
        'quality_promotion_claimed': False, 'allocator_optimum_claimed': False}


def dense_scale_profile(population):
    """Adapt the producer's fusion roster to PQ's existing scale-policy API."""
    owners = {member.removesuffix('.weight'): owner
              for owner, record in population['dense_owners'].items()
              for member in record['members']}
    return SimpleNamespace(fused_sibling_group=owners.get)


def scale_record(max_abs, counts, scales, policy, population, *, calibration):
    """Bind actual complete dense observations and fused scales to one corpus."""
    names = {n.removesuffix('.weight') for n in DENSE4}
    require(set(max_abs) == set(counts) == set(scales) == names, 'static scale roster is not exactly six dense inputs')
    require(calibration['nsamples'] == SAMPLES and calibration['seqlen'] == SEQLEN and
            calibration['seed'] == SEED and re.fullmatch('[0-9a-f]{64}', calibration['token_sha256']) and
            re.fullmatch('[0-9a-f]{64}', calibration['corpus_sha256']), 'calibration identity differs')
    for name in names:
        require(counts[name] == SAMPLES * SEQLEN, f'incomplete dense calibration rows: {name}')
        require(math.isfinite(max_abs[name]) and max_abs[name] > 0 and
                math.isfinite(scales[name]) and scales[name] > 0, f'invalid calibrated scale: {name}')
    for record in population['dense_owners'].values():
        if record['grid'] == 'E2M1x2':
            values = {scales[n.removesuffix('.weight')] for n in record['members']}
            require(len(values) == 1, 'fused siblings did not receive one calibrated scale')
    return {'schema': 'prismaquant.lfm-mixed-static-scales.v1', 'calibration': calibration,
            'policy': policy, 'max_abs': max_abs, 'rows': counts, 'input_global_scales': scales,
            'source': 'actual_dense_forward_inputs', 'hessian_captured': False,
            'scoring_rows_retained': 0, 'expert_rows_claimed': False}


def producer(args):
    authority = read(args.out / 'host-input.json')
    require(authority['schema'] == 'prismaquant.lfm-mixed-host-input.v1' and
            re.fullmatch('[0-9a-f]{40}', authority['pq_snapshot']) and
            authority['producer_image_id'] == IMAGE and
            authority['tessera_source']['manifest_sha256'] == args.tessera_source_manifest_sha256 and
            authority['mode'] == args.mode, 'invalid admitted host source/image authority')
    verify_source(args.tessera_repo, args.tessera_source_manifest, args.tessera_source_manifest_sha256)
    import torch
    import safetensors.torch
    from transformers import AutoModelForCausalLM
    from prismaquant.tessera_campaign import _calibration_tokens, _collect_activations, _static_input_scales
    from prismaquant.nvfp4_activation_contract import input_global_scale_tensor
    from tessera.serving_parts import source_identity
    module = importlib.import_module('export_tessera_serving')
    require(Path(module.__file__).resolve() == args.tessera_repo / 'experiments/export_tessera_serving.py',
            'producer imported from another source tree')
    source = source_identity(args.model)
    _, dense, packed, routed = module.quantizable(args.model)
    plan, projection, population = mixed_plan(module, dense, packed, routed, read(args.model / 'config.json'))
    population['retained_source_tensors'] = sorted(set(source['tensors']) -
        set(population['dense_tensors']) - set(population['routed_tensors']))
    require(set(population['retained_router_tensors']) <= set(population['retained_source_tensors']),
            'router retention disappeared')
    projection['source'] = source
    write(args.out / 'plan.json', plan)
    write(args.out / 'producer-projection.json', projection)
    write(args.out / 'population.json', population)
    write(args.out / 'source-identity.json', source)
    if args.mode == 'calibrate':
        require(torch.cuda.is_available(), 'calibration requires the admitted GPU')
        tokens, text = _calibration_tokens(str(args.model), SAMPLES, SEQLEN, SEED)
        require(len(tokens) == SAMPLES and all(tuple(t.shape) == (1, SEQLEN) for t in tokens),
                'calibration draw shape differs')
        identity = {'dataset': 'wikitext/wikitext-2-raw-v1', 'split': 'train',
                    'nsamples': SAMPLES, 'seqlen': SEQLEN, 'seed': SEED,
                    'corpus_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'token_sha256': hashlib.sha256(torch.cat(tokens).numpy().tobytes()).hexdigest(),
                    'tokenizer_json_sha256': source['auxiliary_sha256']['tokenizer.json']}
        model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=torch.bfloat16).to('cuda').eval()
        require(all(p.device.type == 'cuda' for p in model.parameters()), 'model is not fully GPU resident')
        targets = sorted(n.removesuffix('.weight') for n in DENSE4)
        profiler = cProfile.Profile()
        profiler.enable()
        _, hessians, counts, maxima = _collect_activations(model, targets, tokens, 0, 'cuda', want_hessian=False)
        profiler.disable()
        profiler.dump_stats(str(args.out / 'calibration.prof'))
        require(not hessians, 'weights-only preparation unexpectedly captured Hessians')
        scales, policy = _static_input_scales(maxima, profile=dense_scale_profile(population))
        record = scale_record(maxima, counts, scales, policy, population, calibration=identity)
        record['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['versions'] = {n: metadata.version(n) for n in
                              ('torch', 'transformers', 'numpy', 'safetensors', 'datasets')}
        safetensors.torch.save_file({n + '.input_global_scale': input_global_scale_tensor(v)
            for n, v in scales.items()}, str(args.out / 'input-scales.safetensors'),
            metadata={'input_global_scale_policy': policy})
        os.chmod(args.out / 'input-scales.safetensors', 0o644)
        write(args.out / 'calibration.json', record)
        del model
        torch.cuda.empty_cache()
    else:
        require(not torch.cuda.is_available(), 'CPU preflight unexpectedly sees a GPU')
    require(source_identity(args.model) == source, 'source model changed during preparation')
    verify_source(args.tessera_repo, args.tessera_source_manifest, args.tessera_source_manifest_sha256)
    names = ['host-input.json', 'plan.json', 'producer-projection.json', 'population.json', 'source-identity.json']
    if args.mode == 'calibrate':
        names += ['calibration.json', 'calibration.prof', 'input-scales.safetensors']
    write(args.out / 'preparation-seal.json', {'schema': 'prismaquant.lfm-mixed-preparation.v1',
        'mode': args.mode, 'weights_only_export': True, 'hessian': None,
        'producer_image_id': IMAGE, 'serving_target_image': SERVING_IMAGE,
        'files': {n: {'bytes': (args.out / n).stat().st_size, 'sha256': sha(args.out / n)} for n in names}})
    print(json.dumps({'status': 'prepared', 'mode': args.mode,
                      'seal_sha256': sha(args.out / 'preparation-seal.json')}), flush=True)


def host(args):
    from experiments.pq183_lfm_bound import verify_producer_image
    from experiments.pq87_paired_validation import require_container_name_available
    from experiments.pq87_physical_ab import _http, cleanup_container, dump
    owner = os.environ.get('PRISMABUILD_CONTAINER_OWNER')
    require(owner and len(args.netdata_url) == 2 and len(set(args.netdata_url)) == 2,
            'host requires PB admission and both distinct telemetry hosts')
    require(args.producer_image == IMAGE, 'wrong qualified producer image')
    require(not args.out.exists(), 'output exists; a fresh attempt is required')
    args.out.mkdir(parents=True)
    name = 'lfm-mixed-preparation-' + uuid.uuid4().hex[:12]
    require_container_name_available(name)
    inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE], text=True))[0]
    verify_producer_image(inspected, IMAGE)
    source = verify_source(args.tessera_repo, args.tessera_source_manifest, args.tessera_source_manifest_sha256)
    memory_gib = 64 if args.mode == 'calibrate' else 4
    mask = sorted(os.sched_getaffinity(0))
    require(len(mask) <= 4, 'preparation requires at most four admitted CPUs')
    authority = {'schema': 'prismaquant.lfm-mixed-host-input.v1', 'mode': args.mode,
        'pq_snapshot': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'tessera_source': source, 'producer_image_id': IMAGE, 'serving_target_image': SERVING_IMAGE,
        'producer_config': inspected['Config'], 'producer_rootfs': inspected['RootFS'],
        'cpu_affinity': mask, 'memory_cap_gib': memory_gib, 'native_threads': 1}
    command = ['docker', 'run', '--rm', '--pull=never', '--name', name,
        '--cidfile', str(args.out / 'container.cid'), f'--memory={memory_gib}g', f'--memory-swap={memory_gib}g',
        '--cpus=4', '--ipc=host', '--network=none']
    if args.mode == 'calibrate':
        command += ['--gpus', 'all']
    for p in (REPO, args.tessera_repo, args.model, args.tessera_source_manifest):
        command += ['-v', f'{p}:{p}:ro']
    command += ['-v', f'{args.out}:{args.out}', '-w', str(REPO)]
    for value in ['OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                  'NUMEXPR_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONNOUSERSITE=1',
                  'HF_HOME=/opt/pq183-hf-cache', 'HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1',
                  f'TESSERA_GIT={TS_COMMIT}', f'TESSERA_REPO={args.tessera_repo}',
                  f'TRITON_CACHE_DIR={args.out}/triton', f'TORCH_EXTENSIONS_DIR={args.out}/ext']:
        command += ['-e', value]
    command += ['--entrypoint', 'python3', IMAGE, str(Path(__file__).resolve()), 'producer',
                '--mode', args.mode, '--model', str(args.model), '--out', str(args.out),
                '--tessera-repo', str(args.tessera_repo), '--tessera-source-manifest', str(args.tessera_source_manifest),
                '--tessera-source-manifest-sha256', args.tessera_source_manifest_sha256]
    authority['producer_command'] = command
    write(args.out / 'host-input.json', authority)
    status = {'schema': 'prismaquant.lfm-mixed-preparation-host.v1', 'status': 'inconclusive',
              'started_unix': time.time(), 'container_name': name, 'owner': owner,
              'telemetry_success': {u: False for u in args.netdata_url}, 'monitor_errors': []}
    stop = threading.Event()

    def observe():
        got = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=10)
        if got.returncode:
            require(any(f'no such {kind}: {name}' in (got.stdout + got.stderr).lower()
                        for kind in ('object', 'container')), 'cannot verify owned container absence')
            return None
        obj = json.loads(got.stdout)[0]
        require(obj['Name'] == '/' + name and obj['Config']['Labels'].get('prismabuild.action') == owner,
                'owned container identity mismatch')
        cpus = set()
        for part in obj['HostConfig']['CpusetCpus'].split(','):
            if '-' in part:
                a, b = map(int, part.split('-')); cpus.update(range(a, b + 1))
            elif part:
                cpus.add(int(part))
        require(cpus and cpus <= set(mask) and obj['HostConfig']['Memory'] == memory_gib * 2**30,
                'container widened CPU/memory envelope')
        status['container_id'] = obj['Id']
        status['container_cpu_set'] = sorted(cpus)
        return obj

    def monitor():
        with (args.out / 'telemetry.jsonl').open('x') as log:
            while not stop.is_set():
                row = {'time': time.time(), 'meminfo': Path('/proc/meminfo').read_text(),
                       'cpu_stat': Path('/proc/stat').read_text()}
                try:
                    observe()
                    row['gpu_power_w'] = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw',
                        '--format=csv,noheader,nounits'], text=True, timeout=3).strip()
                    for url in args.netdata_url:
                        row[url] = _http(url.rstrip('/') + '/api/v1/allmetrics?format=json', timeout=2)
                        require(bool(row[url]), 'empty Netdata response')
                        status['telemetry_success'][url] = True
                except Exception as exc:
                    row['error'] = repr(exc); status['monitor_errors'].append(repr(exc))
                log.write(json.dumps(row) + '\n'); log.flush()
                stop.wait(5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    def interrupted(_sig, _frame):
        raise TimeoutError('preparation action interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (args.out / 'producer.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds)
        status['returncode'] = result.returncode
        require(result.returncode == 0, 'producer preparation failed')
        seal = read(args.out / 'preparation-seal.json')
        for rel, item in seal['files'].items():
            require(sha(args.out / rel) == item['sha256'], f'prepared input changed: {rel}')
        require(all(status['telemetry_success'].values()) and not status['monitor_errors'],
                'incomplete telemetry observation')
        status['seal_sha256'] = sha(args.out / 'preparation-seal.json')
        status['status'] = 'prepared'
    except BaseException as exc:
        status['error'] = repr(exc)
    finally:
        stop.set(); thread.join(timeout=30)
        try:
            obj = observe()
            cidpath = args.out / 'container.cid'
            cid = cidpath.read_text().strip() if cidpath.exists() else status.get('container_id')
            require(cid and re.fullmatch('[0-9a-f]{64}', cid), 'missing exact owned container ID')
            require(obj is None or obj['Id'] == cid, 'cleanup container ID changed')
            status['cleanup'] = cleanup_container(cid) if obj else {'safe': True, 'already_absent': True, 'container_id': cid}
        except Exception as exc:
            status['cleanup'] = {'safe': False, 'error': repr(exc)}
        if not status['cleanup']['safe']:
            status['status'] = 'inconclusive'
        status['finished_unix'] = time.time()
        dump(args.out / 'host-status.json', status)
    print(json.dumps({'status': status['status'], 'out': str(args.out),
                      'seal_sha256': status.get('seal_sha256')}), flush=True)
    return 0 if status['status'] == 'prepared' else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['host', 'producer'])
    ap.add_argument('--mode', choices=['preflight', 'calibrate'], required=True)
    for field in ['model', 'out', 'tessera-repo', 'tessera-source-manifest']:
        ap.add_argument('--' + field, type=Path, required=True)
    ap.add_argument('--tessera-source-manifest-sha256', required=True)
    ap.add_argument('--producer-image', default=IMAGE)
    ap.add_argument('--seconds', type=int, default=600)
    ap.add_argument('--netdata-url', action='append', default=[])
    args = ap.parse_args(argv)
    require(all(p.is_absolute() for p in [args.model, args.out, args.tessera_repo, args.tessera_source_manifest]),
            'all input/output paths must be absolute')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.tessera_repo / 'experiments'), str(args.tessera_repo / 'src')]
    return globals()[args.stage](args)


if __name__ == '__main__':
    raise SystemExit(main())
