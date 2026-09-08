"""Measure one real GLM layer with original B1 source or synthetic boundaries.

This is a PrismaBuild-admitted workspace experiment, never a calibration
census/capture or source-forward quality qualification. Source weights, cache,
prefetch, profile adapters and activation collection use their existing paths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import torch

from experiments.workspace_netdata import (
    CHART_PATTERNS, MAX_CHARTS, MAX_RECORD_BYTES, MAX_RESPONSE_BYTES,
    MAX_TOTAL_BYTES, SCHEMA as NETDATA_SCHEMA, SCOPE as NETDATA_SCOPE,
    NetdataWriter, sample_netdata,
)


def digest_tensor(tensor):
    digest = hashlib.sha256()
    if tensor.device.type == 'cpu':
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
        return digest.hexdigest()
    for slab in tensor if tensor.ndim > 1 else tensor.unsqueeze(0):
        raw = slab.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def proc_fields(path):
    return {key.rstrip(':'): int(value) for key, value, *_ in
            (line.split() for line in Path(path).read_text().splitlines())}


def unique_storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device),storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def collector_row_counts(seen, targets, *, status, scope):
    """Snapshot integer observations; absent/malformed data stays unknown."""
    if (not isinstance(seen, dict) or set(seen) != set(targets)
            or any(type(value) is not int or value < 0 for value in seen.values())):
        return dict(status='unknown', scope=scope, expected_qnames=sorted(targets))
    rows = {name: seen[name] for name in sorted(targets)}
    return dict(status=status, scope=scope, rows=rows,
                positive_row_qnames=[name for name, count in rows.items() if count],
                zero_row_qnames=[name for name, count in rows.items() if not count])


def audit_collector_rows(counts, targets, result, check_guard):
    """Retain returned observations before guard or coverage can refuse them."""
    record = collector_row_counts(counts, targets, status='collector_returned',
                                  scope='workspace_observations_not_census_or_admission')
    result['returned_target_row_observations'] = record
    check_guard('after_collection')
    if record['status'] == 'unknown':
        raise RuntimeError('workspace collector returned malformed row observations')
    if record['zero_row_qnames']:
        raise RuntimeError('workspace collector targets have no observed rows: '
                           + ', '.join(record['zero_row_qnames']))


def failed_collector_row_counts(exc, collector, targets):
    """Copy only integer counters from this collector's retained failure frame.

    These are rows observed before failure, not captured Hessians, a census or
    successful coverage. Missing/malformed state remains explicitly unknown.
    """
    tb = exc.__traceback__
    seen = None
    while tb is not None:
        if tb.tb_frame.f_code is collector.__code__:
            seen = tb.tb_frame.f_locals.get('seen')
            break
        tb = tb.tb_next
    return collector_row_counts(seen, targets, status='observed_before_failure',
                                scope='partial_failed_collection_not_census')


def record_materialization_checkpoint(result, label, observation):
    """Record foreground group transitions before a latched guard can raise."""
    if label.startswith('before_output_group:'):
        progress = result.setdefault('output_materialization_progress', dict(
            scope='observed_shared_group_materialization_not_completed_capture', groups_started=0,
            groups_completed=0, qnames_completed=0, current_group=None,
            completed_groups=[], group_boundaries=[]))
        if progress['current_group'] is not None:
            raise RuntimeError('collector began a group before completing its predecessor')
        progress['current_group'] = label.split(':', 1)[1]
        progress['groups_started'] += 1
    progress = result.get('output_materialization_progress')
    if progress is None:
        return
    if label.startswith('after_output_clone:'):
        progress['qnames_completed'] += 1
    if label.startswith('after_output_group:'):
        name = label.split(':', 1)[1]
        if progress['current_group'] != name:
            raise RuntimeError('collector completed a different output group')
        progress['completed_groups'].append(name)
        progress['groups_completed'] += 1
        progress['current_group'] = None
    progress['last_checkpoint'] = label
    progress['last_observation'] = dict(observation)
    if label.startswith(('before_output_group:', 'after_output_group:')):
        progress['group_boundaries'].append(dict(checkpoint=label,
            groups_started=progress['groups_started'],
            groups_completed=progress['groups_completed'],
            qnames_completed=progress['qnames_completed'], **observation))


def watch_memory_samples(stopped, observe, write_record, on_error, interval=0.1):
    """Keep observing through guard detection and actual execution unwind."""
    while not stopped.is_set():
        try:
            write_record(observe())
        except Exception as exc:
            on_error(exc)
        stopped.wait(interval)


def finalize_workspace(shutdown, write_evidence, result, active_error):
    """Finalize evidence even if shutdown fails; preserve an existing failure."""
    try:
        shutdown()
    except BaseException as exc:
        result['cleanup_failure'] = dict(type=type(exc).__name__, message=str(exc))
        result['status'] = 'failed'
        if active_error is None:
            raise
    finally:
        write_evidence()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--nsamples', type=int, default=512)
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--max-act-rows', type=int, default=512)
    parser.add_argument('--fixture-seed', type=int, default=17092026)
    parser.add_argument('--physical-cap-gb', type=int, default=104)
    parser.add_argument('--boundary-mode', choices=('synthetic', 'source-prefix'), default='synthetic')
    parser.add_argument('--input-binding', type=Path)
    parser.add_argument('--input-binding-sha256')
    parser.add_argument('--shared-packed-inputs', action='store_true',
                        help='experimental shared-input collector; full fit remains unqualified')
    parser.add_argument('--release-completed-source', action='store_true',
                        help='opt-in current-source release before shared collector CPU return')
    parser.add_argument('--qualify-shared-inputs', action='store_true',
                        help='bounded exact real-input collector replay, not a full capture')
    parser.add_argument('--qualification-expert-ids', type=int, nargs='+')
    parser.add_argument('--qualification-fixture-mib', type=int, default=2048)
    args = parser.parse_args()
    if args.qualify_shared_inputs and (args.shared_packed_inputs or
            args.boundary_mode != 'source-prefix' or not args.qualification_expert_ids):
        parser.error('replay requires source-prefix and explicit expert IDs, without --shared-packed-inputs')
    if args.qualification_fixture_mib <= 0:
        parser.error('qualification fixture budget must be positive')
    if args.release_completed_source and (not args.shared_packed_inputs or
            args.boundary_mode != 'source-prefix' or args.qualify_shared_inputs):
        parser.error('source release requires shared-packed-inputs and source-prefix full collection')
    args.out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    assert 1 < args.physical_cap_gb <= 104, 'measurement must fit the declared worker capacity'
    from prismaquant.cost_streaming import build_streamed_causal_lm, StreamedForwardBoundaries
    from prismaquant.model_profiles import detect_profile
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    from prismaquant.tessera_campaign import _collect_activations
    from prismaquant.autoscale import streamed_calibration_resources

    profile = detect_profile(str(args.model))
    assert profile.name == 'glm5_next'
    os.environ['PRISMAQUANT_TMPDIR'] = str(args.out/'staging')
    binding = None
    def verify_binding():
        from experiments.glm_workspace_inputs import fingerprint
        assert binding is not None
        for item in binding['source_files']:
            expected = {k:v for k,v in item.items() if k != 'sha256'}
            assert fingerprint(Path(item['path'])) == expected, f"bound source changed: {item['path']}"

    if args.boundary_mode == 'source-prefix':
        assert args.input_binding is not None and args.input_binding_sha256
        raw_binding = args.input_binding.read_bytes()
        assert hashlib.sha256(raw_binding).hexdigest() == args.input_binding_sha256
        binding = json.loads(raw_binding)
        assert binding['schema'] == 'prismaquant.glm_workspace_inputs.v1'
        assert binding['model'] == str(args.model) and binding['stop_after_layer'] == args.layer
        assert (args.nsamples,args.seqlen,binding['nsamples'],binding['seqlen'],binding['seed']) == (512,512,512,512,0)
        verify_binding()
    result = dict(schema='prismaquant.glm_layer_workspace.v1',
        scope=('bounded_real_source_prefix_workspace_only_unfrozen_production_draw'
               if binding else 'synthetic_boundary_real_source_layer_workspace_only'),
        status='running', source=str(args.model), layer=args.layer,
        config_sha256=hashlib.sha256((args.model/'config.json').read_bytes()).hexdigest(),
        fixture_seed=args.fixture_seed, nsamples=args.nsamples, seqlen=args.seqlen,
        original_microbatch=1, max_act_rows=args.max_act_rows, cache_slots=2,
        prefetch_workers=1, declared_source_headroom_gb=24,
        physical_measurement_cap_gb=args.physical_cap_gb, guard_margin_gb=2,
        torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        device_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', '')),
        source_page_release_optin=os.environ.get('PRISMAQUANT_RELEASE_SOURCE_PAGES') == '1',
        shared_packed_inputs=args.shared_packed_inputs,
        release_completed_source=args.release_completed_source,
        collector_replay_qualification=args.qualify_shared_inputs,
        host_kernel_release=os.uname().release,
        phases=[], telemetry_errors=[])
    for field, module_path in (
            ('nfsv4_module_build_id_note_hex', 'notes/.note.gnu.build-id'),
            ('nfsv4_module_srcversion', 'srcversion'),
            ('nfsv4_delegation_watermark', 'parameters/delegation_watermark')):
        try:
            raw = (Path('/sys/module/nfsv4') / module_path).read_bytes()
            result[field] = raw.hex() if module_path.startswith('notes/') else raw.decode().strip()
        except OSError:
            result[field] = None
    if binding:
        result['input_binding_sha256'] = args.input_binding_sha256
        result['source_files_sha256'] = {item['path']:item['sha256'] for item in binding['source_files']}
        result['calibration_seed'] = binding['seed']
        result['corpus_sha256'] = binding['corpus_sha256']
        result['tokens_sha256'] = binding['tokens_sha256']
    result['netdata_contract'] = dict(schema=NETDATA_SCHEMA, scope=NETDATA_SCOPE,
        chart_patterns=CHART_PATTERNS, maximum_charts=MAX_CHARTS,
        response_byte_cap=MAX_RESPONSE_BYTES, record_byte_cap=MAX_RECORD_BYTES,
        total_byte_cap=MAX_TOTAL_BYTES, interval_seconds=1, hosts=['sparky', 'sparklina'])
    telemetry_failed = threading.Event()
    stopped = threading.Event()
    guard_tripped = threading.Event()
    sampling_enabled = threading.Event()
    main_thread_id = threading.get_ident()
    started = time.time()
    cgroot = Path('/sys/fs/cgroup')

    def physical_memory_record():
        # GB10 CUDA allocations were absent from the packer action's cgroup
        # charge (14.61 GiB cgroup peak versus 36 GiB CUDA allocated). Add the
        # entire CUDA reservation conservatively; retain the separate planes
        # so overlapping charges cannot masquerade as measured physical use.
        current = int((cgroot/'memory.current').read_text())
        reserved = torch.cuda.memory_reserved()
        host = proc_fields('/proc/meminfo')
        return dict(cgroup_current_bytes=current, cuda_reserved_bytes=reserved,
            conservative_cgroup_plus_cuda_reserved_bytes=current+reserved,
            host_mem_available_bytes=host['MemAvailable']*1024,
            host_mem_total_bytes=host['MemTotal']*1024,
            host_memory_bytes={name:host[name]*1024 for name in
                ('MemFree','Cached','Buffers','Shmem','SReclaimable','SUnreclaim')})

    def check_guard(label, *, raise_on_trip=True):
        if raise_on_trip:
            checkpoint = dict(time=time.time(), label=label)
            if label.startswith(('before_output_', 'after_output_', 'before_rows_',
                                 'after_rows_', 'before_hessian_', 'after_hessian_')):
                allocator = torch.cuda.memory_stats()
                checkpoint.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    inactive_split_bytes=allocator.get('inactive_split_bytes.all.current'),
                    allocation_retries=allocator.get('num_alloc_retries'), ooms=allocator.get('num_ooms'))
            result['last_foreground_checkpoint'] = checkpoint
            record_materialization_checkpoint(result, label, checkpoint)
        if raise_on_trip and telemetry_failed.is_set():
            result['status'] = 'failed'
            raise RuntimeError('required workspace host telemetry is incomplete')
        path = cgroot/'memory.current'
        if not path.is_file():
            raise RuntimeError('workspace guard requires container cgroup memory observations')
        observed = physical_memory_record()
        if (observed['conservative_cgroup_plus_cuda_reserved_bytes'] >
                (args.physical_cap_gb-2)*1024**3 or
                observed['host_mem_available_bytes'] < 8*1024**3):
            result.setdefault('memory_guard_failure', dict(label=label,time=time.time(),
                **observed, requested_cap_bytes=args.physical_cap_gb*1024**3,
                stats=proc_fields(cgroot/'memory.stat'),
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                minimum_host_available_bytes=8*1024**3))
            guard_tripped.set()
        if guard_tripped.is_set():
            result['status'] = 'refused_by_memory_guard'
            if raise_on_trip:
                result['memory_guard_refusal_checkpoint'] = dict(checkpoint)
                raise RuntimeError('workspace measurement reached its physical memory guard; '
                                   'no full-capture admission is established')

    def watch_memory():
        def observe():
            check_guard('continuous_cgroup_sample', raise_on_trip=False)
            return dict(time=time.time(), **physical_memory_record(),
                        cuda_allocated_bytes=torch.cuda.memory_allocated(),
                        guard_tripped=guard_tripped.is_set(),
                        stats=proc_fields(cgroot/'memory.stat'))

        def on_error(exc):
            failure = dict(error=repr(exc), time=time.time())
            result.setdefault('memory_guard_failure', failure)
            result.setdefault('memory_observer_error', failure)
            result['status'] = 'refused_by_memory_guard'
            guard_tripped.set()

        with (args.out/'cgroup-memory.jsonl').open('w') as out:
            def write_record(record):
                out.write(json.dumps(record)+'\n')
                out.flush()
            watch_memory_samples(stopped, observe, write_record, on_error)

    def memory_record():
        record = dict(time=time.time(), allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(), io=proc_fields('/proc/self/io'),
            **physical_memory_record())
        path = cgroot/'memory.max'
        if path.is_file():
            record['memory.max'] = path.read_text().strip()
        for name in ('memory.current', 'memory.peak'):
            path = cgroot/name
            if path.is_file():
                record[name] = int(path.read_text())
        path = cgroot/'memory.stat'
        if path.is_file():
            record['memory.stat'] = proc_fields(path)
        stats = torch.cuda.memory_stats()
        record['inactive_split_bytes'] = stats.get('inactive_split_bytes.all.current')
        record['allocation_retries'] = stats.get('num_alloc_retries')
        record['ooms'] = stats.get('num_ooms')
        return record

    def mark(label, **extra):
        torch.cuda.synchronize()
        record = dict(label=label, **memory_record(), **extra)
        if runner is not None:
            cache = runner.context.layer_cache
            cached = cache._cache.copy()
            storages = {}
            tensors = [*runner.model.parameters(), *runner.model.buffers(),
                       *(t for values in cached.values() for t in values.values())]
            for tensor in tensors:
                if tensor.is_meta or tensor.device.type != 'cuda':
                    continue
                storage = tensor.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
            record['source_storage_bytes_installed_and_cached_unique'] = sum(storages.values())
            record['source_cache_layers'] = sorted(cached)
            record['source_cache_reported_bytes'] = cache.total_bytes
            record['source_prefetch_memory_skips'] = runner.context.prefetch_memory_skips
            record['source_prefetch_delivered_unretained'] = runner.context.prefetch_delivered_unretained
        result['phases'].append(record)
        print(json.dumps(record), flush=True)
        check_guard(label)

    def monitor():
        with (args.out/'netdata.jsonl').open('w') as out:
            writer = NetdataWriter(out)
            while not stopped.is_set():
                for host in ('sparky', 'sparklina'):
                    try:
                        writer.write(sample_netdata(host))
                    except Exception as exc:
                        result['telemetry_errors'].append(dict(host=host,error=repr(exc)))
                        result['status'] = 'failed'
                        telemetry_failed.set()
                        return
                    finally:
                        result['netdata_recorded_bytes'] = writer.bytes_written
                stopped.wait(1)

    def sample_python():
        with (args.out/'python-main-thread-stacks.jsonl').open('w') as out:
            while not stopped.is_set():
                if sampling_enabled.is_set():
                    frame = sys._current_frames().get(main_thread_id)
                    stack = traceback.extract_stack(frame)
                    del frame
                    out.write(json.dumps(dict(time=time.time(), scope='python_main_thread_only',
                        frames=[dict(file=x.filename,line=x.lineno,function=x.name) for x in stack]))+'\n')
                    out.flush()
                stopped.wait(0.1)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    memory_thread = threading.Thread(target=watch_memory, daemon=True)
    sampler_thread = threading.Thread(target=sample_python, daemon=True)
    monitor_thread.start()
    memory_thread.start()
    sampler_thread.start()
    runner = None
    try:
        mark('process_baseline')
        check_guard('before_source_construction')
        runner = build_streamed_causal_lm(str(args.model), device=torch.device('cuda'),
            dtype=torch.bfloat16, offload_folder=str(args.out/'offload'), profile=profile,
            max_cache_slots=2, prefetch_workers=1, cache_headroom_gb=24,
            prefetch_min_available_gb=24, prefetch_lookahead=1,
            require_prefetched_residency=True, attn_implementation='eager')
        assert 0 <= args.layer < runner.num_layers-1
        config = runner.base_model.config
        mark('resident_head_and_skeleton')
        layer_prefix = f'{runner.layers_prefix}{args.layer}.'
        dense = [name for name, module in runner.model.named_modules()
                 if name.startswith(layer_prefix) and isinstance(module, torch.nn.Linear)
                 and not profile.is_pinned_name(name)]
        import re
        excluded = profile.probe_linear_exclude_extra()
        dense = [name for name in dense if not excluded or not re.search(excluded, name)]
        members = [member for member in profile_declared_packed_expert_projections(runner.model, profile)
                   if member.qname.startswith(layer_prefix)]
        targets = [*dense, *(member.qname for member in members)]
        modules = dict(runner.model.named_modules())
        shapes = {name: list(modules[name].weight.shape) for name in dense}
        shapes.update({member.qname:list(member.weight.shape) for member in members})
        result['resource_upper_bound'] = streamed_calibration_resources(str(args.model),
            unit_shapes=shapes, counts=dict.fromkeys(targets,args.nsamples*args.seqlen),
            nsamples=args.nsamples, seqlen=args.seqlen, max_act_rows=args.max_act_rows,
            cache_slots=2, prefetch_workers=1, headroom_gb=24)
        result['units'] = len(targets)
        if binding:
            token_bytes = Path(binding['tokens_path']).read_bytes()
            assert hashlib.sha256(token_bytes).hexdigest() == binding['tokens_sha256']
            tokens = [torch.tensor(ids, dtype=torch.int64) for ids in json.loads(token_bytes)]
            assert len(tokens) == args.nsamples and all(tuple(t.shape) == (1,args.seqlen) for t in tokens)
        else:
            # Explicit synthetic residual streams, never a source-model draw.
            generator = torch.Generator(device='cuda').manual_seed(args.fixture_seed)
            shape = (1,args.seqlen,int(config.hc_mult),int(config.hidden_size))
            states = [torch.randn(shape, generator=generator, device='cuda', dtype=torch.bfloat16)
                      for _ in range(args.nsamples)]
            tokens = [torch.arange(args.seqlen).remainder(int(config.vocab_size)-2).add(2).reshape(1,-1)
                      for _ in states]
            result['boundary_input_sha256'] = [digest_tensor(state) for state in states]
            mark('synthetic_current_boundaries_resident',
                 boundary_bytes=sum(x.numel()*x.element_size() for x in states))

        def collect_and_audit(actual_forward):
            check_guard('before_activation_collection')
            if args.qualify_shared_inputs:
                from experiments.glm_shared_input_replay import qualify_collector
                expert_ids = set(args.qualification_expert_ids)
                selected = [member for member in members if member.expert_id in expert_ids]
                if {member.expert_id for member in selected} != expert_ids:
                    raise RuntimeError('qualification expert roster is not present in the real source')
                next_source_batch = 0

                def capture_forward(input_ids):
                    nonlocal next_source_batch
                    check_guard('before_replay_source_batch')
                    actual_forward(input_ids)
                    next_source_batch += 1
                    check_guard('after_replay_source_batch')

                try:
                    result['collector_replay'] = qualify_collector(
                        runner.model, selected, tokens, args.max_act_rows, runner.device,
                        profile=profile, actual_forward=capture_forward, out=args.out,
                        check_guard=check_guard, mark=mark,
                        max_fixture_bytes=args.qualification_fixture_mib * 1024**2)
                finally:
                    result['target_source_batches_completed'] = next_source_batch
                result['scope'] = 'bounded_real_routed_collector_replay_not_full_capture'
                return
            next_batch = 0
            traces_written = []
            with (args.out/'batch-memory.jsonl').open('w') as samples:
                def forward_batch(input_ids):
                    nonlocal next_batch
                    check_guard('before_batch')
                    actual_forward(input_ids)
                    next_batch += 1
                    check_guard('after_batch')
                    samples.write(json.dumps(dict(sample=next_batch, **memory_record()))+'\n')
                    samples.flush()
                    prof.step()

                def trace_ready(prof):
                    label = ('forward-first-two-batches' if not traces_written
                             else 'forward-batches-32-33')
                    prof.export_chrome_trace(str(args.out/(label+'.trace.json')))
                    (args.out/(label+'.profile.txt')).write_text(
                        prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
                    traces_written.append(label)

                def profiler_schedule(step):
                    if step in (1,32):
                        return torch.profiler.ProfilerAction.RECORD_AND_SAVE
                    if step in (0,31):
                        return torch.profiler.ProfilerAction.RECORD
                    return torch.profiler.ProfilerAction.NONE

                def release_completed_source():
                    if next_batch != args.nsamples:
                        raise RuntimeError('cannot release source before all original forwards')
                    layers = range(args.layer, min(args.layer + 2, runner.num_layers))
                    before = runner.context.source_residency_snapshot(layers)
                    mark('before_completed_source_release', source_owners=before)
                    runner.context.release_completed_layer(args.layer)
                    after = runner.context.source_residency_snapshot(layers)
                    mark('after_completed_source_release', source_owners=after)

                # Observe cold and later windows in the original traversal;
                # no additional forwards or changed microbatches are needed.
                sampling_enabled.set()
                try:
                    with torch.no_grad(), torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                            schedule=profiler_schedule, record_shapes=True,profile_memory=True,
                            on_trace_ready=trace_ready) as prof:
                        acts,hessians,counts,maxima = _collect_activations(runner.model,targets,tokens,
                            args.max_act_rows,runner.device,want_hessian=True,profile=profile,
                            forward_batch=forward_batch, resource_check=check_guard,
                            shared_packed_inputs=args.shared_packed_inputs,
                            on_forwards_complete=(release_completed_source
                                if args.release_completed_source else None))
                        audit_collector_rows(counts, targets, result, check_guard)
                except BaseException as exc:
                    result['partial_target_row_observations'] = failed_collector_row_counts(
                        exc, _collect_activations, targets)
                    raise
                finally:
                    sampling_enabled.clear()
                    result['profile_windows'] = traces_written
                    result['target_batches_completed'] = next_batch
            assert next_batch == args.nsamples
            mark('collector_return_full_h_and_prefix_x_on_cpu',
                 hessian_bytes=sum(h.numel()*h.element_size() for h in hessians.values()),
                 prefix_bytes=sum(x.numel()*x.element_size() for x in acts.values()),
                 hessian_unique_storage_bytes=unique_storage_bytes(hessians.values()),
                 prefix_unique_storage_bytes=unique_storage_bytes(acts.values()),
                 minimum_observed_rows=min(counts.values()))
            _next, source = runner.context.ensure_loaded(args.layer+1,require_prefetched=True)
            assert all(t.device.type == 'cuda' for t in _next.values())
            del _next
            mark('next_source_resident_after_collection', delivery=source)
            result['unit_results'] = {}
            for name in targets:
                x,h = acts[name],hessians[name]
                assert x.dtype == h.dtype == torch.float32
                assert torch.isfinite(x).all() and torch.isfinite(h).all()
                result['unit_results'][name] = dict(count=counts[name],max_abs=maxima[name],
                    x_shape=list(x.shape),h_shape=list(h.shape),
                    x_sha256=digest_tensor(x),h_sha256=digest_tensor(h))
            mark('output_bytes_audited')
            del x,h,acts,hessians
            mark('layer_capture_drained')

        if binding:
            class PrefixMeasurementComplete(Exception):
                pass

            def visit(layer, forward_batch):
                mark('source_prefix_layer_installed', source_layer=layer)
                if layer == args.layer:
                    collect_and_audit(forward_batch)
                    # The existing visitor's finally unloads this source layer.
                    # No later layer or full-source initialization is claimed.
                    raise PrefixMeasurementComplete()
                assert layer < args.layer
                for token in tokens:
                    check_guard('before_prefix_batch')
                    forward_batch(token.to(runner.device))
                    check_guard('after_prefix_batch')
                mark('source_prefix_layer_forward_complete', source_layer=layer,
                     samples=args.nsamples)
                # Match _run_streamed_calibration.visit's existing boundary
                # cleanup even though these prefix layers collect no H/X.
                torch.cuda.empty_cache()
                mark('source_prefix_layer_allocator_released', source_layer=layer)

            try:
                runner.visit_layer_batches(tokens, visit)
            except PrefixMeasurementComplete:
                pass
            else:
                raise RuntimeError('bounded prefix did not stop after its requested layer')
            verify_binding()
            result['source_binding_unchanged_after_prefix'] = True
            mark('source_prefix_driver_drained_and_current_layer_unloaded')
        else:
            runner.context.schedule_prefetch(args.layer)
            runner.context.schedule_prefetch(args.layer+1)
            runner.context.install(args.layer, require_prefetched=True)
            mark('current_layer_installed_next_prefetch_inflight')
            synthetic_batch = 0
            def synthetic_forward(input_ids):
                nonlocal synthetic_batch
                assert synthetic_batch < len(states)
                assert torch.equal(input_ids.cpu(), tokens[synthetic_batch])
                ids, positions, initial, embeddings, mask = runner._prepare(input_ids)
                del initial
                batch = StreamedForwardBoundaries(ids, positions, embeddings, mask, [], None)
                states[synthetic_batch] = runner.isolated_layer(batch,args.layer,states[synthetic_batch],
                    pass_state=profile.new_forward_pass_state())
                synthetic_batch += 1
            collect_and_audit(synthetic_forward)
            runner.context.unload(args.layer)
            mark('source_layer_unloaded')
        torch.cuda.empty_cache()
        mark('explicit_allocator_release_after_layer_drain')
        result['status'] = 'complete'
    except BaseException as exc:
        if result['status'] == 'running':
            result['status'] = 'failed'
        result['failure'] = dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        active_error = sys.exc_info()[1]

        def write_evidence():
            if binding:
                try:
                    verify_binding()
                    result['source_binding_unchanged_at_exit'] = True
                except Exception as exc:
                    result['source_binding_unchanged_at_exit'] = False
                    result['source_binding_failure'] = repr(exc)
                    result['status'] = 'failed'
            stopped.set()
            monitor_thread.join(timeout=12)
            memory_thread.join(timeout=2)
            sampler_thread.join(timeout=2)
            if 'collector_replay' in result:
                from experiments.glm_shared_input_replay import summarize_device_energy
                try:
                    with (args.out/'netdata.jsonl').open() as samples:
                        energy = summarize_device_energy(
                            (json.loads(line) for line in samples),
                            result['collector_replay']['arms'], result['device_uuid'])
                    result['collector_replay']['energy'] = energy
                    (args.out/'collector-replay.json').write_text(
                        json.dumps(result['collector_replay'], indent=2)+'\n')
                except Exception as exc:
                    result['telemetry_errors'].append(f'collector energy summary: {exc!r}')
            if 'cleanup_failure' in result or result['telemetry_errors']:
                result['status'] = 'failed'
            if guard_tripped.is_set() and result['status'] == 'complete':
                result['status'] = 'refused_by_memory_guard'
            result['begin'] = started
            result['end'] = time.time()
            (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        finalize_workspace(runner.shutdown if runner is not None else lambda: None,
                           write_evidence, result, active_error)
    assert not result['telemetry_errors'], result['telemetry_errors']
    assert result['status'] == 'complete', result.get('source_binding_failure')


if __name__ == '__main__':
    main()
