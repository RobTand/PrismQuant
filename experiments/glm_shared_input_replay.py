"""Bounded collector A/B on identical real routed inputs, never a capture cache.

The workspace driver freezes explicitly selected actual derivation outputs once.
Replay substitutes only input delivery; both arms run the real campaign
collector, including FP32 Gram/add/fmax, prefix storage and full CPU return.
Its timings exclude source forwards and routing/SwiGLU derivation. This evidence
fixture is private to one qualification action and is released before exit.
"""
from contextlib import contextmanager
import hashlib
import io
import time
from unittest.mock import patch

import torch

INPUT_KIND = {'gate_up_proj': 'gate_up', 'down_proj': 'down'}


def tensor_bytes_digest(value):
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def serialized_unit(capture, name):
    stream = io.BytesIO()
    torch.save({'X': capture[0][name], 'H': capture[1][name],
                'count': capture[2][name], 'amax': capture[3][name]}, stream)
    return stream


def compare_captures(reference, candidate):
    left = right = None
    try:
        if any(set(reference[i]) != set(candidate[i]) for i in range(4)):
            raise RuntimeError('collector replay output roster differs')
        for name in reference[0]:
            for index in (0, 1):
                left, right = reference[index][name], candidate[index][name]
                if (left is None or right is None or left.dtype != right.dtype
                        or left.shape != right.shape
                        or not torch.equal(left.contiguous().view(torch.uint8),
                                           right.contiguous().view(torch.uint8))):
                    raise RuntimeError(f'collector replay tensor bytes differ: {name}/{index}')
            if reference[2][name] != candidate[2][name] or reference[3][name] != candidate[3][name]:
                raise RuntimeError(f'collector replay counts/maxima differ: {name}')
            with serialized_unit(reference, name) as left, serialized_unit(candidate, name) as right:
                if left.getbuffer() != right.getbuffer():
                    raise RuntimeError(f'collector replay serialized tensor payload differs: {name}')
    finally:
        left = right = None


def audit_cpu_storage(capture):
    """Require independent compact CPU allocations, including across H and X."""
    seen = set()
    total = 0
    value = None
    try:
        for output in capture[:2]:
            for name, value in output.items():
                if value is None or value.device.type != 'cpu':
                    raise RuntimeError(f'replay output is not a materialized CPU tensor: {name}')
                storage = value.untyped_storage()
                identity = storage.data_ptr()
                if identity in seen or storage.nbytes() != value.numel() * value.element_size():
                    raise RuntimeError(f'replay output aliases or retains excess storage: {name}')
                seen.add(identity)
                total += storage.nbytes()
        return total
    finally:
        value = storage = None


class RoutedInputFixture:
    """Finite, explicitly budgeted evidence from the existing derivation path."""
    def __init__(self, members, *, max_batches, max_bytes, resource_check):
        self.members = tuple(members)
        modules = {member.module_qname for member in members}
        if len(modules) != 1 or not members:
            raise ValueError('bounded replay requires one explicitly selected packed module')
        self.module_qname = next(iter(modules))
        self.module = members[0].module
        self.expert_ids = sorted({member.expert_id for member in members})
        self.max_batches, self.max_bytes = int(max_batches), int(max_bytes)
        self.check = resource_check
        self.records = []
        self.bytes = 0

    @contextmanager
    def capture(self):
        from prismaquant import measure_quant_cost
        original = measure_quant_cost.derive_per_expert_activations

        def derive(module, *args, **kwargs):
            result = original(module, *args, **kwargs)
            if module is not self.module:
                return result
            if len(self.records) >= self.max_batches:
                raise RuntimeError('routed replay fixture exceeded its batch bound')
            record = {kind: [None] * len(result[kind]) for kind in ('gate_up', 'down')}
            record['row_counts'] = list(result['row_counts'])
            source = None
            try:
                for kind in ('gate_up', 'down'):
                    for expert in self.expert_ids:
                        source = result[kind][expert]
                        required = source.numel() * source.element_size()
                        if self.bytes + required > self.max_bytes:
                            raise RuntimeError('routed replay fixture exceeded its byte bound')
                        self.check(f'before_routed_fixture_copy:{len(self.records)}:{kind}:{expert}')
                        record[kind][expert] = source.detach().clone()
                        self.bytes += required
                        self.check(f'after_routed_fixture_copy:{len(self.records)}:{kind}:{expert}')
            except BaseException:
                for kind in ('gate_up', 'down'):
                    record[kind].clear()
                record.clear()
                result.clear()
                raise
            finally:
                source = None
            self.records.append(record)
            return result

        with patch.object(measure_quant_cost, 'derive_per_expert_activations', derive):
            yield

    def manifest(self, max_rows):
        cumulative = dict.fromkeys(self.expert_ids, 0)
        warm_start = None
        records = []
        for batch, record in enumerate(self.records):
            payloads = {}
            for kind in ('gate_up', 'down'):
                payloads[kind] = {}
                for expert in self.expert_ids:
                    value = record[kind][expert]
                    payloads[kind][str(expert)] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                                     sha256=tensor_bytes_digest(value))
            for expert in self.expert_ids:
                cumulative[expert] += record['row_counts'][expert]
            if warm_start is None and min(cumulative.values()) >= max_rows:
                warm_start = batch + 1
            records.append(dict(batch=batch, inputs=payloads))
        value = record = None
        if (len(self.records) != self.max_batches or warm_start is None
                or warm_start < 2 or warm_start + 1 >= len(self.records)):
            raise RuntimeError('routed fixture lacks complete batches or a row-filled profile window')
        return dict(schema='prismaquant.routed_collector_replay_inputs.v1',
                    scope='selected_real_routed_inputs_original_batch_order',
                    module=self.module_qname, expert_ids=self.expert_ids,
                    bytes=self.bytes, maximum_bytes=self.max_bytes,
                    batches=len(records), row_counts=cumulative, warm_start=warm_start,
                    qnames=[m.qname for m in self.members], records=records)

    def replay(self, model, targets, tokens, max_rows, device, *, shared, profile,
               after_batch, resource_check):
        from prismaquant import measure_quant_cost, production_weight_cache
        from prismaquant.tessera_campaign import _collect_activations
        original_collector = production_weight_cache._PackedExpertActivationCollector
        installed = []
        next_batch = 0

        class InputDelivery(original_collector):
            def install(delivery):
                installed.append(delivery)

            def remove(delivery):
                installed.remove(delivery)

        def derive(module, _x, _parent, **kwargs):
            if module is not self.module or not kwargs.get('capture_down') or kwargs.get('max_rows_per_expert') is not None:
                raise RuntimeError('replay changed the original packed derivation contract')
            return self.records[next_batch]

        def forward(_tokens):
            nonlocal next_batch
            if len(installed) != 1 or next_batch >= len(self.records):
                raise RuntimeError('replay delivery order differs')
            # row_consumer is the existing collector input boundary. Its
            # derivation call receives the exact frozen result above; no
            # source module forward, router or gate arithmetic is timed here.
            installed[0].row_consumer(self.module_qname, None)
            next_batch += 1
            after_batch(next_batch)

        with patch.object(production_weight_cache, '_PackedExpertActivationCollector', InputDelivery), \
                patch.object(measure_quant_cost, 'derive_per_expert_activations', derive):
            result = _collect_activations(model, targets, tokens, max_rows, device,
                want_hessian=True, profile=profile, forward_batch=forward,
                resource_check=resource_check, shared_packed_inputs=shared)
        if next_batch != len(self.records):
            raise RuntimeError('replay omitted original routed batches')
        return result

    def clear(self):
        for record in self.records:
            for kind in ('gate_up', 'down'):
                record[kind].clear()
            record.clear()
        self.records.clear()
        self.bytes = 0


def qualify_collector(model, members, tokens, max_rows, device, *, profile,
                      actual_forward, out, check_guard, mark, max_fixture_bytes):
    """Capture once; interleave legacy/shared/shared/legacy in one admitted action."""
    import json
    from prismaquant.tessera_campaign import _collect_activations

    targets = [member.qname for member in members]
    fixture = RoutedInputFixture(members, max_batches=len(tokens), max_bytes=max_fixture_bytes,
                                 resource_check=check_guard)
    reference = candidate = current = capture = None
    report = dict(schema='prismaquant.shared_collector_qualification.v1', status='running',
                  scope='isolated_collector_only_source_forward_and_derivation_excluded',
                  arm_order=['legacy', 'shared', 'shared', 'legacy'], arms=[])
    try:
        with fixture.capture():
            capture = _collect_activations(model, targets, tokens, 0, device,
                want_hessian=False, profile=profile, forward_batch=actual_forward,
                resource_check=check_guard)
        if any(value is not None for value in capture[0].values()) or capture[1]:
            raise RuntimeError('input qualification unexpectedly retained scoring/H outputs')
        source_counts = dict(capture[2])
        capture = None
        mark('routed_fixture_source_forward_complete', fixture_bytes=fixture.bytes)
        manifest = fixture.manifest(max_rows)
        raw = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        (out/'routed-inputs.json').write_bytes(raw)
        report['input_manifest_sha256'] = hashlib.sha256(raw).hexdigest()
        report['input_manifest'] = str(out/'routed-inputs.json')
        report['targets'] = targets
        report['source_observed_rows'] = source_counts
        report['memory_comparison'] = ('First legacy arm has no retained CPU reference. '
            'Later arms retain that same reference; compare those matched contexts for memory deltas.')
        warm_start = manifest['warm_start']
        for ordinal, arm in enumerate(report['arm_order']):
            prefix = f'collector-{ordinal}-{arm}'
            retained_reference_bytes = 0 if reference is None else audit_cpu_storage(reference)
            trace_labels = iter(('cold', 'row-filled', 'cpu-return'))
            traces = []
            batch_times = []
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            check_guard(f'before_{prefix}')

            def trace_ready(prof):
                label = next(trace_labels)
                path = out/f'{prefix}-{label}.trace.json'
                prof.export_chrome_trace(str(path))
                traces.append(str(path))

            def schedule(step):
                if step in (1, warm_start + 1):
                    return torch.profiler.ProfilerAction.RECORD_AND_SAVE
                if step in (0, warm_start, len(tokens)):
                    return torch.profiler.ProfilerAction.RECORD
                return torch.profiler.ProfilerAction.NONE

            def after_batch(batch):
                check_guard(f'{prefix}_batch:{batch}')
                batch_times.append(dict(batch=batch, time=time.time()))
                prof.step()

            mark(f'{prefix}_begin')
            begin = time.time()
            with torch.no_grad(), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=schedule, record_shapes=True, profile_memory=True,
                    on_trace_ready=trace_ready) as prof:
                candidate = fixture.replay(model, targets, tokens, max_rows, device,
                    shared=arm == 'shared', profile=profile, after_batch=after_batch,
                    resource_check=check_guard)
            torch.cuda.synchronize()
            end = time.time()
            mark(f'{prefix}_cpu_return_complete', hessian_bytes=sum(v.numel()*v.element_size() for v in candidate[1].values()))
            if len(traces) != 3:
                raise RuntimeError('replay did not retain cold, row-filled and CPU-return profiles')
            if any(not (out/path.split('/')[-1]).stat().st_size for path in traces):
                raise RuntimeError('replay profile trace is empty')
            output_cpu_bytes = audit_cpu_storage(candidate)
            if candidate[2] != source_counts:
                raise RuntimeError('replayed row counts differ from the original source collector')
            if reference is None:
                reference, candidate = candidate, None
                current = reference
            else:
                compare_captures(reference, candidate)
                current = candidate
            payloads = {}
            for name in targets:
                with serialized_unit(current, name) as stream:
                    payloads[name] = hashlib.sha256(stream.getbuffer()).hexdigest()
            report['arms'].append(dict(arm=arm, begin=begin, end=end, seconds=end-begin,
                completed_batches=len(batch_times), original_routed_rows=sum(current[2].values()),
                batches=batch_times, traces=traces, serialized_unit_sha256=payloads,
                independent_output_cpu_bytes=output_cpu_bytes,
                reference_cpu_bytes_retained_at_begin=retained_reference_bytes,
                cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved=torch.cuda.max_memory_reserved()))
            current = candidate = None
        report['status'] = 'exact_for_selected_real_inputs'
        return report
    except BaseException as error:
        report['status'] = 'failed'
        report['failure'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        # Clear maps as well as local aliases: a failed comparison may retain
        # its capture argument through the exception's traceback.
        for output in (reference, candidate, current, capture):
            if output is not None:
                for part in output:
                    part.clear()
        reference = candidate = current = capture = output = part = None
        fixture.clear()
        (out/'collector-replay.json').write_text(json.dumps(report, indent=2)+'\n')


def summarize_device_energy(records, arms, device_uuid):
    """Gross device energy from distinct Netdata power updates, never utilization.

    Hold each reported power value until its next update. Require at least two
    updates within an arm and no gap/stale boundary over 15 seconds. This is a
    coarse telemetry estimate including profiler/observer and external load.
    """
    by_host = {}
    uuid = str(device_uuid).lower().removeprefix('gpu-')
    for record in records:
        for name, chart in record['metrics'].items():
            if not uuid or uuid not in name.lower() or not name.endswith('_power_draw'):
                continue
            value = chart['dimensions']['power_draw']['value']
            if value is not None:
                by_host.setdefault(record['host'], {})[float(chart['last_updated'])] = float(value)
    if len(by_host) != 1:
        return dict(status='unresolved', reason='device UUID does not identify exactly one observed host')
    host, updates = next(iter(by_host.items()))
    points = sorted(updates.items())
    summaries = []
    for arm in arms:
        begin, end = arm['begin'], arm['end']
        before = [(t, p) for t, p in points if t <= begin]
        inside = [(t, p) for t, p in points if begin < t < end]
        summary = dict(arm=arm['arm'], begin=begin, end=end,
                       distinct_updates_inside=len(inside), gross_device_joules=None,
                       completed_collector_batches_per_joule=None)
        if not before or len(inside) < 2:
            summary['reason'] = 'insufficient distinct power updates at this arm duration'
        else:
            selected = [before[-1], *inside]
            gaps = [right[0]-left[0] for left, right in zip(selected, selected[1:])]
            if begin-selected[0][0] > 15 or end-selected[-1][0] > 15 or any(gap > 15 for gap in gaps):
                summary['reason'] = 'power updates are too sparse or stale for this arm'
            else:
                joules = 0.0
                for index, (timestamp, watts) in enumerate(selected):
                    stop = selected[index+1][0] if index+1 < len(selected) else end
                    joules += (stop-max(begin, timestamp))*watts
                if joules > 0:
                    summary['gross_device_joules'] = joules
                    summary['completed_collector_batches_per_joule'] = arm['completed_batches']/joules
                    summary['mean_power_watts'] = joules/(end-begin)
                    summary['mean_power_envelope_fraction_140w'] = joules/(end-begin)/140
                else:
                    summary['reason'] = 'nonpositive observed energy'
        summaries.append(summary)
    return dict(status='coarse_gross_estimates', host=host, device_uuid=str(device_uuid),
                method='hold reported power between distinct updates; includes profiling and observer load',
                workload='completed bounded collector batches with full H/X CPU return; excludes source forwards',
                arms=summaries)
