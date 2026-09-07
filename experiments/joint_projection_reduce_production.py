"""Qualify actual source X/g through the real prewarmed production lease API."""
import inspect
from pathlib import Path
import shutil
import sys
import textwrap

import torch

from experiments.qdq_constant_residency import main as replay, ROOT
from prismaquant import joint_aura
from prismaquant.joint_projection_backend import prewarm_projection_backend

BINARY = ROOT / 'joint-fused-projection/actual-ab-01/pq_joint_projection_reduce_17d100e93d552b85.so'
BINARY_SHA = '9305c183c5214dc5ff1f73382963f8275eb1b6197cb8d173c6a93bffd700c115'


class LeasePair:
    def __init__(self, reference, fused):
        self.leases = {'before': reference, 'after': fused}
        self.current = 'before'
        self.entered = False

    def __enter__(self):
        self.leases[self.current].__enter__()
        self.entered = True
        return self

    def switch(self, label):
        if label == self.current:
            return
        if self.entered:
            self.leases[self.current].__exit__(None, None, None)
        self.current = label
        if self.entered:
            self.leases[self.current].__enter__()

    def __exit__(self, *args):
        self.entered = False
        self.leases[self.current].__exit__(*args)

    def begin_probe(self):
        self.leases[self.current].begin_probe()

    def finish_probe(self):
        return self.leases[self.current].finish_probe()

    @property
    def telemetry(self):
        return {name: sum(lease.telemetry[name] for lease in self.leases.values())
                for name in self.leases['before'].telemetry}


class ProductionCandidate:
    schema = 'pq.production_joint_projection_qualification.v1'
    before, after = 'before', 'after'

    def __init__(self):
        self.reference = prewarm_projection_backend({'name': 'torch'}, device='cuda')
        self.fused = prewarm_projection_backend({'name': 'fused_fp32_v1',
            'binary': {'path': str(BINARY), 'sha256': BINARY_SHA}}, device='cuda')
        self.identity = self.fused.identity
        self.baseline_path = Path(joint_aura.__file__)
        self.before_source = self.after_source = textwrap.dedent(inspect.getsource(joint_aura.SignedJointProjectionLease._observe))
        self.checking = False
        self.reduction_checks = []
        self.original_product = self.fused.product_sum
        self.fused.product_sum = self.product

    @property
    def checking(self):
        return self._checking

    @checking.setter
    def checking(self, enabled):
        self._checking = bool(enabled)
        if hasattr(self, 'pair'):
            # Instrument individual bits only during qualification. Profiles
            # call the original admitted production backend directly.
            self.pair.leases['after']._projection_product_sum = (self.product if enabled
                                                                else self.original_product)

    def product(self, left, right):
        actual = self.original_product(left, right)
        if self.checking:
            expected = (left * right).sum()
            bits, reference = actual.view(torch.int32).item(), expected.view(torch.int32).item()
            row = {'shape': list(left.shape), 'left_stride': list(left.stride()),
                   'right_stride': list(right.stride()), 'candidate_bits': bits, 'reference_bits': reference,
                   'equal': bits == reference}
            self.reduction_checks.append(row)
            assert row['equal'], f'production fused reduction changed FP32 bits: {row}'
        return actual

    def make_lease(self, *args, **kwargs):
        self.pair = LeasePair(
            joint_aura.SignedJointProjectionLease(*args, **kwargs, projection_backend=self.reference),
            joint_aura.SignedJointProjectionLease(*args, **kwargs, projection_backend=self.fused))
        return self.pair

    def activate(self, label):
        self.pair.switch(label)

    def persist_binary(self, output):
        shutil.copy2(BINARY, output / BINARY.name)
        shutil.copy2(BINARY.with_name('build.ninja'), output / 'build.ninja')


def main():
    candidate = ProductionCandidate()
    if '--qualification-only' not in sys.argv:
        sys.argv.append('--qualification-only')
    replay(variant_controller=candidate)
    assert len(candidate.reduction_checks) == 208


if __name__ == '__main__':
    main()
