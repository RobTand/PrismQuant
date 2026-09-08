"""Read retained workspace evidence; execute batch analysis through PrismaBuild."""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(root):
    result = json.loads((root / 'result.json').read_text())
    samples = list(read_jsonl(root / 'cgroup-memory.jsonl'))
    traces = {}
    for path in sorted(root.glob('forward-*.trace.json')):
        events = json.loads(path.read_text())['traceEvents']
        categories = collections.Counter()
        calls = collections.Counter()
        kernels = collections.Counter()
        for event in events:
            category = event.get('cat', '')
            if event.get('ph') == 'X':
                categories[category] += event.get('dur', 0)
                calls[category] += 1
                if category == 'kernel':
                    kernels[event['name']] += event.get('dur', 0)
        traces[path.name] = dict(sha256=digest(path),
            category_duration_us=dict(categories), category_calls=dict(calls),
            top_kernel_duration_us=kernels.most_common(8))
    hosts = collections.defaultdict(lambda: dict(samples=0, errors=0, series=collections.defaultdict(list)))
    for record in read_jsonl(root / 'netdata.jsonl'):
        host = hosts[record['host']]
        host['samples'] += 1
        if record.get('error'):
            host['errors'] += 1
        for chart, values in record.get('metrics', {}).items():
            if chart == 'system.cpu' or 'power' in chart:
                for name, value in values.get('dimensions', {}).items():
                    number = value.get('value')
                    if isinstance(number, (int, float)):
                        host['series'][chart + '/' + name].append(number)
    telemetry = {}
    for name, host in hosts.items():
        telemetry[name] = dict(samples=host['samples'], errors=host['errors'],
            metrics={key: dict(n=len(values), mean=statistics.mean(values),
                              min=min(values), max=max(values))
                     for key, values in host['series'].items()})
    fields = ['cgroup_current_bytes', 'cuda_reserved_bytes',
              'conservative_cgroup_plus_cuda_reserved_bytes', 'cuda_allocated_bytes']
    memory = {key: max(record[key] for record in samples) for key in fields}
    memory['minimum_host_available_bytes'] = min(r['host_mem_available_bytes'] for r in samples)
    memory['guard_tripped_samples'] = sum(bool(r.get('guard_tripped')) for r in samples)
    phases = [{k: v for k, v in row.items() if k in {
        'label', 'time', 'allocated', 'reserved', 'inactive_split_bytes',
        'source_layer', 'source_cache_layers', 'source_prefetch_memory_skips',
        'source_prefetch_delivered_unretained', 'hessian_bytes', 'prefix_bytes',
        'minimum_observed_rows', 'delivery'}} for row in result['phases']]
    selected = {key: result.get(key) for key in [
        'status', 'target_batches_completed', 'input_binding_sha256',
        'source_binding_unchanged_at_exit', 'telemetry_errors', 'torch', 'cuda',
        'device_uuid', 'host_kernel_release', 'nfsv4_module_build_id_note_hex']}
    selected.update(units=result['units'], wall_seconds=result['end']-result['begin'], memory=memory,
        phases=phases, traces=traces, telemetry=telemetry,
        materialization={k:v for k,v in result.get('output_materialization_progress',{}).items()
                         if k not in ['completed_groups','group_boundaries','last_observation']},
        unit_results_count=len(result.get('unit_results',{})),
        artifact_sha256={p.name:digest(p) for p in sorted(root.iterdir()) if p.is_file()})
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema='prismaquant.glm_expandable_analysis.v1',
                  before=summarize(args.before), after=summarize(args.after),
                  scope='bounded_workspace_memory_and_profile_evidence_not_full_model_or_served_quality')
    before = json.loads((args.before/'result.json').read_text())
    after = json.loads((args.after/'result.json').read_text())
    left = before.get('partial_target_row_observations', {}).get('rows')
    right = after.get('returned_target_row_observations', {}).get('rows')
    report['all_observed_row_counts_equal'] = left == right if left is not None and right is not None else None
    old = {r['checkpoint']:r for r in before['output_materialization_progress']['group_boundaries']}
    new = {r['checkpoint']:r for r in after.get('output_materialization_progress',{}).get('group_boundaries',[])}
    common = [k for k in old if k in new]
    report['matched_group_boundaries'] = len(common)
    report['last_matched_group_boundary'] = ({'checkpoint': common[-1],
        'before':old[common[-1]], 'after':new[common[-1]]} if common else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps({k:v for k,v in report.items() if k not in ['before','after','last_matched_group_boundary']}))


if __name__ == '__main__':
    main()
