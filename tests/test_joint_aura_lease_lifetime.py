"""The reverse loop releases a completed lease before loading another layer."""
import weakref

import torch

import prismaquant.aura_cost as aura
import prismaquant.joint_aura as joint
from test_joint_aura_streamed import _fixture
from test_streamed_cost_checkpoints import _model_identity


def test_completed_joint_lease_does_not_retain_previous_layer_deltas(monkeypatch):
    _, context, runner, cache = _fixture()
    previous_deltas = []
    released_layers = []
    original_lease = joint.SignedJointProjectionLease
    class ObservedLease(original_lease):
        def __exit__(self, *args):
            previous_deltas[:] = [weakref.ref(tensor) for tensor in self.deltas.values()]
            return super().__exit__(*args)
    monkeypatch.setattr(joint, 'SignedJointProjectionLease', ObservedLease)
    original_install = context.install
    def install(layer, **kwargs):
        if previous_deltas:
            assert all(reference() is None for reference in previous_deltas), \
                'completed joint lease retained deltas into the next layer install'
            released_layers.append(layer)
        return original_install(layer, **kwargs)
    context.install = install
    aura.compute_aura_cost_streamed(runner, torch.tensor([[1, 2, 3, 4]]),
        ['FP8_DYNAMIC', 'BF16'], n_probes=2, min_free_gib=0,
        production_cache=cache, joint_activation=True,
        model_identity=_model_identity('joint-lifetime'))
    assert released_layers == [0]
