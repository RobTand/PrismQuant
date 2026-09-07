"""The measurement observer must see real torch serialization I/O."""
import hashlib
import torch

from experiments.file_read_audit import FileReadAudit


def test_observer_counts_path_hash_and_torch_load_reads(tmp_path):
    path = tmp_path / 'tensor.pt'
    tensor = torch.arange(8192, dtype=torch.bfloat16).reshape(128, 64)
    torch.save(tensor, path)
    raw = path.read_bytes()
    with FileReadAudit({path: 'render'}) as audit:
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == hashlib.sha256(raw).hexdigest()
        loaded = torch.load(path, weights_only=True)
        assert torch.equal(loaded, tensor)
    observed = audit.summary()['by_kind']['render']
    assert observed['opens'] == 2
    assert observed['read_bytes'] >= len(raw) + tensor.numel() * tensor.element_size()
    assert observed['read_calls'] >= 3
