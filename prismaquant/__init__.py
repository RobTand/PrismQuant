"""PrismaQuant: mixed-native quantization policy engine for LLMs.

Current production path:
    1. incremental_probe.py              stream Fisher and activation shards
    2. incremental_measure_quant_cost.py build per-Linear format costs
    3. allocator.py                      choose Pareto layer-format assignments
    4. validate_assignments_kl.py        measure held-out KL for candidates
    5. export_native_compressed.py       write compressed-tensors artifacts
    6. validate_native_export.py         run vLLM load/generation smoke
    7. validate_quantized_model.py       run downstream quality checks

Older cross-layer allocators are archived under archive/cross_layer_2026-05-09
for artifact replay and comparison.
"""
import contextlib as _contextlib
from contextvars import ContextVar as _ContextVar

_checkpoint_initializing = _ContextVar("prismaquant_checkpoint_initializing", default=False)
_INITIALIZATION_CONTRACT_ATTRIBUTE = "_prismaquant_pretrained_initialization_contract"

from .format_registry import FormatSpec, REGISTRY, register_format

# Resolved from installed metadata rather than duplicated here, so
# pyproject.toml stays the single source of truth (the release pipeline asserts
# the git tag matches the built version). A source checkout that was never
# installed has no metadata; report that honestly instead of guessing a number.
try:  # pragma: no cover - trivial metadata lookup
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("prismaquant")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"
del _pkg_version, PackageNotFoundError


def _ensure_triton_cache_writable() -> None:
    """Keep Triton-backed model code from falling back to CPU.

    Some of the Qwen linear-attention dependencies import FLA, which asks
    Triton for the active CUDA target at import time.  If Triton cannot write
    its cache, FLA catches the exception, decides the platform is CPU-only,
    and later crashes in the CUDA forward path.  Respect an explicit
    TRITON_CACHE_DIR; otherwise redirect only when the default cache is not
    writable by this user.
    """
    import os
    import tempfile
    from pathlib import Path

    def _is_writable_dir(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".prismaquant-write-",
                dir=path,
                delete=True,
            ):
                pass
            return True
        except Exception:
            return False

    configured = os.environ.get("TRITON_CACHE_DIR")
    if configured:
        try:
            Path(configured).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return

    default = Path.home() / ".triton" / "cache"
    if _is_writable_dir(default):
        return

    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    fallback = base / "prismaquant" / "triton"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(fallback)


# Transformers compatibility polyfill.
# Some remote modeling files (e.g. MiniMax-M2/M2.7's modeling_minimax_m2.py)
# import `OutputRecorder` from `transformers.utils.generic`. In
# transformers 5.x that symbol moved to `transformers.modeling_utils`.
# Re-expose the 4.x import path so remote code that matches the checkpoint
# tensor naming still loads. Idempotent — no-op if the symbol is already
# there (4.x or future 5.x that re-exports it).
def _polyfill_transformers() -> None:
    try:
        # OutputRecorder: transformers.utils.generic → transformers.modeling_utils
        import transformers.utils.generic as _gen
        if not hasattr(_gen, "OutputRecorder"):
            import transformers.modeling_utils as _mu
            if hasattr(_mu, "OutputRecorder"):
                _gen.OutputRecorder = _mu.OutputRecorder
    except Exception:
        pass
    try:
        # From-config meta skeletons overwrite checkpoint parameters later,
        # so they retain the existing no-init policy. Ordinary checkpoint
        # loads must still initialize missing state, including nonpersistent
        # buffers that Transformers rematerializes with empty_like.
        import transformers.modeling_utils as _mu
        if hasattr(_mu, "PreTrainedModel") and \
                not getattr(_mu.PreTrainedModel, "_prismaquant_init_noop", False):
            model_class = _mu.PreTrainedModel
            real_initialize = model_class._initialize_weights
            real_missing = model_class._initialize_missing_keys
            model_class._prismaquant_real_initialize_weights = real_initialize

            def _initialize_for_checkpoint(self, *args, **kwargs):
                if _checkpoint_initializing.get():
                    return real_initialize(self, *args, **kwargs)
                return None

            def _initialize_missing_checkpoint_state(self, *args, **kwargs):
                import transformers
                # An unsuccessful repeated finalization must not retain an
                # earlier completion descriptor.
                self.__dict__.pop(_INITIALIZATION_CONTRACT_ATTRIBUTE, None)
                token = _checkpoint_initializing.set(True)
                try:
                    result = real_missing(self, *args, **kwargs)
                finally:
                    _checkpoint_initializing.reset(token)
                setattr(self, _INITIALIZATION_CONTRACT_ATTRIBUTE, {
                    "schema": "prismaquant.pretrained_initialization.v1",
                    "scope": "checkpoint_missing_state",
                    "status": "completed",
                    "transformers_version": transformers.__version__,
                })
                return result

            model_class._initialize_weights = _initialize_for_checkpoint
            model_class._initialize_missing_keys = _initialize_missing_checkpoint_state
            model_class._prismaquant_init_noop = True
    except Exception:
        pass
    try:
        # Transformers 5.x's fine-grained-FP8 quantizer's
        # `validate_environment` has an operator-precedence bug —
        # the `"disk" in device_map.values()` check is not gated on
        # `pre_quantized`, so it rejects disk-device-map loads even
        # for pre-quantized checkpoints. Override with a bypass:
        # prismaquant's streaming loader (for probe/cost) REQUIRES
        # disk-device-map to offload body layers, and our caller
        # guarantees the checkpoint is pre-quantized (it's the
        # source on disk) so this rejection is a false positive.
        from transformers.quantizers.quantizer_finegrained_fp8 import (
            FineGrainedFP8HfQuantizer as _FP8Q,
        )
        if not getattr(_FP8Q, "_prismaquant_validator_bypass", False):
            def _bypass_validate(self, *a, **kw):
                return None
            _FP8Q.validate_environment = _bypass_validate
            _FP8Q._prismaquant_validator_bypass = True
    except Exception:
        pass
    try:
        # ROPE_INIT_FUNCTIONS['default'] was removed in transformers 5.x
        # (renamed to 'linear', which takes a 'factor' kwarg the old
        # default never needed). Remote modeling files from older
        # checkpoints still look up 'default'. Re-register the old
        # implementation verbatim — a ~6-line function computing the
        # standard rotary inv_freq schedule with no scaling.
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            import torch as _torch
            def _compute_default_rope_parameters(config=None, device=None, **_):
                base = config.rope_theta
                partial = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(
                    config, "head_dim",
                    config.hidden_size // config.num_attention_heads)
                dim = int(head_dim * partial)
                inv_freq = 1.0 / (base ** (
                    _torch.arange(0, dim, 2, dtype=_torch.int64)
                        .to(dtype=_torch.float32, device=device) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    except Exception:
        pass
    try:
        # Qwen3 upstream recomputes rotary cos/sin with a per-forward
        # FP32 BMM. Under deterministic cuBLAS plus PrismaQuant imports,
        # that BMM can produce NaNs after model.cuda(). Route Qwen3
        # AutoModel loads to the vendored cached-RoPE copy.
        from .vendored import register_qwen3 as _register_qwen3
        _register_qwen3()
    except Exception:
        pass


def validate_pretrained_initialization_contract(value):
    """Validate and copy a checkpoint initialization provenance descriptor."""
    if not isinstance(value, dict) or set(value) != {
        "schema", "scope", "status", "transformers_version"
    } or value.get("schema") != "prismaquant.pretrained_initialization.v1" \
            or value.get("scope") != "checkpoint_missing_state" \
            or value.get("status") != "completed" \
            or not isinstance(value.get("transformers_version"), str) \
            or not value["transformers_version"].strip():
        raise ValueError("Missing or invalid pretrained initialization contract")
    return dict(value)


def pretrained_initialization_contract(model):
    """Require evidence that this model completed checkpoint initialization.

    From-config skeletons have no such descriptor. This records the load
    phase, not a general certification of subsequent model mutations.
    """
    return validate_pretrained_initialization_contract(
        getattr(model, _INITIALIZATION_CONTRACT_ATTRIBUTE, None))


def validate_source_initialization_contract(value):
    """Read either qualified source-loading route without conflating them."""
    if isinstance(value, dict) and value.get("schema") == "prismaquant.streaming_initialization.v1":
        from .streaming_model import validate_streaming_initialization_contract
        return validate_streaming_initialization_contract(value)
    return validate_pretrained_initialization_contract(value)


@_contextlib.contextmanager
def genuine_weight_initialization():
    """Build a model the way transformers would, inside a PrismaQuant process.

    `_polyfill_transformers` suppresses `PreTrainedModel._initialize_weights`
    outside checkpoint missing-state finalization, because streaming loads
    build a `from_config` skeleton and overwrite checkpoint parameters. That makes `from_config` construction
    silently return **uninitialized** parameters for any tensor the
    modeling file allocates as a bare `nn.Parameter(torch.empty(...))` --
    routed-expert weights and hyper-connection tensors are the common
    cases -- and heap contents are neither reproducible nor guaranteed
    finite, so a forward on such a model can be fine on one run and `nan`
    on the next.

    Wrap a from-config construction in this context manager when the model
    it returns is used as-is:

        with genuine_weight_initialization():
            model = SomeForCausalLM(config)

    A no-op when the polyfill never applied (older transformers, or an
    import that raised), so callers need not test for it.
    """
    import transformers.modeling_utils as _mu

    real = getattr(
        _mu.PreTrainedModel, "_prismaquant_real_initialize_weights", None)
    if real is None:
        yield
        return
    patched = _mu.PreTrainedModel._initialize_weights
    _mu.PreTrainedModel._initialize_weights = real
    try:
        yield
    finally:
        _mu.PreTrainedModel._initialize_weights = patched


_ensure_triton_cache_writable()
_polyfill_transformers()
del _ensure_triton_cache_writable
del _polyfill_transformers
