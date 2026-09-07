"""PB CPU comparison of retained canonical and actual routed input rows."""
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import torch

from experiments.qdq_constant_residency import ROOT, NAME, sha, tensor_sha


def main():
    torch.set_num_threads(1)
    source_root = ROOT / 'qdq-residency/source-capture-01'
    receipt = json.loads((source_root / 'receipt.json').read_text())
    assert sha(receipt['payload']['path']) == receipt['payload']['sha256']
    payload = torch.load(receipt['payload']['path'], map_location='cpu', weights_only=True)
    manifest_path = ROOT / 'capture-reuse/canonical/capture_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    record = manifest['entries'][NAME]
    path = manifest_path.parent / record['path']
    assert sha(path) == record['sha256']
    canonical = torch.load(path, map_location='cpu', weights_only=True)['inputs']
    actual_records = [entry for entry in payload['records'] if entry['probe'] == 0]
    actual = torch.cat([entry['x'].float() for entry in actual_records])
    original = canonical[:len(actual)]
    delta = actual - original
    first = (delta != 0).nonzero()[0].tolist()
    actual_hashes = [tensor_sha(row) for row in actual]
    original_hashes = [tensor_sha(row) for row in original]
    indices = defaultdict(deque)
    for i, digest in enumerate(original_hashes):
        indices[digest].append(i)
    order = [indices[digest].popleft() if indices[digest] else None for digest in actual_hashes]
    result = {'source_receipt_sha256': sha(source_root / 'receipt.json'),
              'canonical_sha256': record['sha256'], 'shapes': [list(actual.shape), list(canonical.shape)],
              'first_difference': {'index': first, 'actual': actual[tuple(first)].item(), 'canonical': original[tuple(first)].item()},
              'ordered_max_abs': delta.abs().max().item(), 'ordered_relative_l2': (delta.norm()/original.norm()).item(),
              'exact_multiset_equal': Counter(actual_hashes) == Counter(original_hashes),
              'ordered_equal': torch.equal(actual, original),
              'actual_to_canonical_row': order, 'invocation_checks': []}
    start = 0
    for entry in actual_records:
        end = start + len(entry['x'])
        result['invocation_checks'].append({'source_row': entry['row'], 'rows': end-start,
            'equal_multiset': Counter(actual_hashes[start:end]) == Counter(original_hashes[start:end]),
            'same_order': actual_hashes[start:end] == original_hashes[start:end]})
        start = end
    if not result['exact_multiset_equal']:
        distances = torch.cdist(actual, canonical)
        best = distances.argmin(dim=1)
        diff = actual - canonical[best]
        result['nearest_canonical_rows'] = best.tolist()
        result['nearest_max_abs'] = diff.abs().max().item()
        result['nearest_relative_l2'] = (diff.norm()/actual.norm()).item()
    out = ROOT / 'qdq-residency/capture-order-diagnosis.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'actual_to_canonical_row'}))


if __name__ == '__main__':
    main()
