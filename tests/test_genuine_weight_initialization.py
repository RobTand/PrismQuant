"""`prismaquant` no-ops weight init process-wide; this is the documented undo.

`prismaquant/__init__.py::_polyfill_transformers` replaces
`PreTrainedModel._initialize_weights` with a no-op at import time. That is
sound for PrismaQuant's own loaders, which build a `from_config` skeleton and
then overwrite every parameter from the checkpoint. It is unsound for any
caller that *uses* a from-config model: every tensor a modeling file allocates
as a bare `nn.Parameter(torch.empty(...))` then keeps whatever the allocator
last left in that page -- not reproducible, and not guaranteed finite.

That is not hypothetical. It reached CI as a NaN in
`test_glm5_next_streamed_forward_parity.py::test_long_sequence_parity_per_layer_type[dsa_only]`
(run 33284249771: py3.11 failed, py3.12 passed, same commit and same torch and
transformers), where the uninitialized tensors were the routed-expert
`mlp.experts.gate_up_proj` / `down_proj`.

So the polyfill must keep the real method reachable, and the undo must be
exception-safe -- leaking the real `_initialize_weights` back into the process
would silently re-cost every subsequent PrismaQuant model load.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.modeling_utils import PreTrainedModel  # noqa: E402

from prismaquant import genuine_weight_initialization  # noqa: E402


def _polyfilled() -> bool:
    return bool(getattr(PreTrainedModel, "_prismaquant_init_noop", False))


def test_the_polyfill_keeps_the_real_initializer_reachable():
    if not _polyfilled():
        pytest.skip("the transformers polyfill did not apply in this build")
    assert getattr(
        PreTrainedModel, "_prismaquant_real_initialize_weights", None
    ) is not None, (
        "the polyfill overwrote _initialize_weights without keeping the real "
        "method, so nothing in the process can build an initialized "
        "from-config model any more"
    )


def test_the_context_manager_swaps_the_method_and_puts_it_back():
    if not _polyfilled():
        pytest.skip("the transformers polyfill did not apply in this build")
    real = PreTrainedModel._prismaquant_real_initialize_weights
    noop = PreTrainedModel._initialize_weights
    assert noop is not real

    with genuine_weight_initialization():
        assert PreTrainedModel._initialize_weights is real
    assert PreTrainedModel._initialize_weights is noop


def test_the_context_manager_restores_on_an_exception():
    if not _polyfilled():
        pytest.skip("the transformers polyfill did not apply in this build")
    noop = PreTrainedModel._initialize_weights
    with pytest.raises(RuntimeError, match="construction failed"):
        with genuine_weight_initialization():
            raise RuntimeError("construction failed")
    assert PreTrainedModel._initialize_weights is noop


def test_weight_init_actually_runs_inside_the_context_manager():
    """The contract is behavioural: `_init_weights` must be reached."""
    from transformers import PretrainedConfig

    class _Config(PretrainedConfig):
        model_type = "prismaquant_init_probe"

    class _Model(PreTrainedModel):
        config_class = _Config
        base_model_prefix = "probe"

        def __init__(self, config):
            super().__init__(config)
            self.linear = torch.nn.Linear(2, 2)
            self.initialized = []
            self.post_init()

        def _init_weights(self, module):
            self.initialized.append(type(module).__name__)

    outside = _Model(_Config())
    with genuine_weight_initialization():
        inside = _Model(_Config())

    assert inside.initialized, (
        "`_init_weights` never ran inside genuine_weight_initialization(); "
        "the from-config model would carry uninitialized parameters"
    )
    if _polyfilled():
        assert not outside.initialized, (
            "`_init_weights` ran outside the context manager -- the polyfill "
            "under test is not actually in effect, so this test proves nothing"
        )
