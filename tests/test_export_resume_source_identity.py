"""The standalone export resume cache must bind the source checkpoint.

`materialize_tensors_streaming(..., export_cache_dir=...)` saves each
quantized layer to `layer_NNN.pt` and, on a restart, replays those payloads
instead of re-reading the source. Admission was decided by
`_render_lever_provenance()` plus `assignment_hash` alone, so a cache dir
reused against a DIFFERENT checkpoint replayed the first checkpoint's
quantized bytes: the render levers and the recipe both still matched, and
nothing in the manifest named the source (#340).

These tests drive the real exporter on a tiny synthetic checkpoint. They
assert on the admission decision (was a `layer_*.pt` read at all?) and on the
emitted bytes, so a refusal that still replays, or a replay that silently
re-renders, both fail.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from prismaquant import export_native_compressed as export  # noqa: E402
from prismaquant import sensitivity_probe, streaming_model  # noqa: E402

# Two FP32 values inside one BF16 ulp at 0.5, as in test_export_buffer_*: a
# buffer edit the parameter dtype cannot represent, so a replayed payload is
# distinguishable from a re-rendered one.
_BIAS_A = torch.tensor([0.5, 0.5 + 2.0 ** -12], dtype=torch.float32)
_BIAS_B = torch.tensor([0.5, 0.5 + 3.0 * 2.0 ** -12], dtype=torch.float32)
_WEIGHT_A = torch.eye(2, dtype=torch.bfloat16)
_WEIGHT_B = torch.tensor([[1.0, 0.0], [0.0, 0.5]], dtype=torch.bfloat16)


def _write_source(source: Path, *, weight, bias) -> None:
    source.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "model.layers.0.expert_bias": bias.clone(),
            "model.layers.0.proj.weight": weight.clone(),
        },
        str(source / "model.safetensors"),
    )


def _exporter(tmp_path, monkeypatch):
    """Replace architecture construction only; source reads, the cache and
    emission are the real code paths."""

    class Skeleton:
        @staticmethod
        def _from_config(config):
            model = nn.Module()
            model.model = nn.Module()
            layer = nn.Module()
            layer.proj = nn.Linear(2, 2, bias=False, device="meta")
            layer.register_buffer(
                "expert_bias",
                torch.empty(2, device="meta", dtype=torch.float32),
            )
            model.model.layers = nn.ModuleList([layer])
            return model

    profile = SimpleNamespace(
        requires_multimodal_skeleton=lambda: False,
        concat_merge_groups=lambda: (),
        live_to_recipe_name=lambda name: name,
        packed_expert_param_names=lambda: (),
    )
    from transformers import AutoConfig

    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(sensitivity_probe, "stage_text_only", lambda p: p)
    monkeypatch.setattr(
        streaming_model, "_skeleton_config_and_class",
        lambda config, **kw: (config, Skeleton))
    monkeypatch.setattr(streaming_model, "_init_rotary_inplace", lambda *a: None)
    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", None)

    source = tmp_path / "source"

    def run(cache: Path | None, *, replay_log: list | None = None):
        emitted: dict = {}

        def sink(tensors):
            emitted.update(tensors)

        if replay_log is None:
            export.materialize_tensors_streaming(
                str(source), {}, profile=profile, bf16_passthrough=set(),
                device=torch.device("cpu"), tensor_sink=sink,
                export_cache_dir=None if cache is None else str(cache))
            return emitted

        real_load = torch.load

        def spy(path, *args, **kwargs):
            if Path(str(path)).name.startswith("layer_"):
                replay_log.append(str(path))
            return real_load(path, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(torch, "load", spy)
            export.materialize_tensors_streaming(
                str(source), {}, profile=profile, bf16_passthrough=set(),
                device=torch.device("cpu"), tensor_sink=sink,
                export_cache_dir=None if cache is None else str(cache))
        return emitted

    return run, source


def _fill_cache(run, source, cache, *, weight, bias):
    _write_source(source, weight=weight, bias=bias)
    emitted = run(cache)
    assert list(cache.glob("layer_*.pt")), "fixture wrote no layer cache"
    return emitted


# --------------------------------------------------------------------------
# Admission: a changed source must refuse replay.
# --------------------------------------------------------------------------

def test_changed_source_parameter_refuses_replay(tmp_path, monkeypatch):
    """Same assignment, same levers, different weight bytes of the same size
    and shape. The cached layer must not be replayed, and the export must
    carry the NEW weight."""
    run, source = _exporter(tmp_path, monkeypatch)
    cache = tmp_path / "resume"
    _fill_cache(run, source, cache, weight=_WEIGHT_A, bias=_BIAS_A)

    _write_source(source, weight=_WEIGHT_B, bias=_BIAS_A)
    replays: list = []
    emitted = run(cache, replay_log=replays)

    assert replays == [], f"replayed a payload from the previous source: {replays}"
    assert torch.equal(emitted["model.layers.0.proj.weight"], _WEIGHT_B)


def test_changed_persistent_buffer_refuses_replay(tmp_path, monkeypatch):
    """A persistent FP32 buffer edit is a source change too. Replay would
    ship the old buffer bytes verbatim."""
    run, source = _exporter(tmp_path, monkeypatch)
    cache = tmp_path / "resume"
    _fill_cache(run, source, cache, weight=_WEIGHT_A, bias=_BIAS_A)

    _write_source(source, weight=_WEIGHT_A, bias=_BIAS_B)
    replays: list = []
    emitted = run(cache, replay_log=replays)

    assert replays == [], f"replayed a payload from the previous source: {replays}"
    got = emitted["model.layers.0.expert_bias"]
    assert torch.equal(got.view(torch.uint8), _BIAS_B.view(torch.uint8))


def test_manifest_without_source_identity_is_refused(tmp_path, monkeypatch):
    """Fail closed on a pre-fix cache: a manifest that names no source cannot
    authorize replay even when nothing else changed."""
    run, source = _exporter(tmp_path, monkeypatch)
    cache = tmp_path / "resume"
    _fill_cache(run, source, cache, weight=_WEIGHT_A, bias=_BIAS_A)

    manifest = cache / "manifest.json"
    legacy = json.loads(manifest.read_text())
    assert "source_identity" in legacy, (
        "the resume manifest binds no source identity")
    legacy.pop("source_identity")
    manifest.write_text(json.dumps(legacy))

    replays: list = []
    run(cache, replay_log=replays)
    assert replays == [], "a manifest with no source identity admitted replay"
    assert "source_identity" in json.loads(manifest.read_text())


# --------------------------------------------------------------------------
# The identical-source resume must keep working, byte for byte.
# --------------------------------------------------------------------------

def test_identical_source_still_replays_byte_identically(tmp_path, monkeypatch):
    run, source = _exporter(tmp_path, monkeypatch)
    cache = tmp_path / "resume"
    first = _fill_cache(run, source, cache, weight=_WEIGHT_A, bias=_BIAS_A)
    first = {k: v.clone() for k, v in first.items()}

    replays: list = []
    second = run(cache, replay_log=replays)

    assert replays, "an unchanged source stopped resuming"
    assert set(first) == set(second)
    for name, value in first.items():
        assert value.dtype == second[name].dtype, name
        assert torch.equal(
            value.view(torch.uint8), second[name].view(torch.uint8)), name


# --------------------------------------------------------------------------
# The fingerprint itself: dtype and the declared buffer-dtype map are bound.
# --------------------------------------------------------------------------

def _fingerprint(source: Path, **overrides):
    kwargs = dict(
        assignment={},
        model_path=str(source),
        dtype=torch.bfloat16,
        declared_buffer_dtypes={"model.layers.0.expert_bias": torch.float32},
    )
    kwargs.update(overrides)
    return export._export_resume_fingerprint(**kwargs)


def test_fingerprint_binds_requested_dtype(tmp_path):
    source = tmp_path / "source"
    _write_source(source, weight=_WEIGHT_A, bias=_BIAS_A)
    assert (_fingerprint(source)["requested_dtype"]
            != _fingerprint(source, dtype=torch.float16)["requested_dtype"])


def test_fingerprint_binds_declared_buffer_dtypes(tmp_path):
    source = tmp_path / "source"
    _write_source(source, weight=_WEIGHT_A, bias=_BIAS_A)
    narrowed = _fingerprint(
        source,
        declared_buffer_dtypes={"model.layers.0.expert_bias": torch.bfloat16},
    )
    assert (_fingerprint(source)["declared_buffer_dtypes"]
            != narrowed["declared_buffer_dtypes"])


def test_fingerprint_source_identity_tracks_content_not_path(tmp_path):
    """Content identity: the same bytes under a different directory are the
    same source; a same-size value edit is a different source."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    _write_source(first, weight=_WEIGHT_A, bias=_BIAS_A)
    _write_source(second, weight=_WEIGHT_A, bias=_BIAS_A)
    assert (_fingerprint(first)["source_identity"]["content_sha256"]
            == _fingerprint(second)["source_identity"]["content_sha256"])

    edited = tmp_path / "c"
    _write_source(edited, weight=_WEIGHT_A, bias=_BIAS_B)
    assert (_fingerprint(first)["source_identity"]["content_sha256"]
            != _fingerprint(edited)["source_identity"]["content_sha256"])


def test_source_identity_never_degrades_to_none(tmp_path):
    """`None == None` would admit. A source that cannot be identified must
    raise, not stamp a null the next run can match."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception):
        _fingerprint(empty)


# --------------------------------------------------------------------------
# The shared checkpoint-identity helper.
# --------------------------------------------------------------------------

def test_digest_cache_rehashes_a_same_size_rewrite(tmp_path):
    """The digest cache keys on the full stat fingerprint (ctime included),
    so restoring mtime after an in-place same-size rewrite still re-hashes."""
    import os

    from prismaquant.cost_streaming import build_source_checkpoint_identity

    source = tmp_path / "source"
    _write_source(source, weight=_WEIGHT_A, bias=_BIAS_A)
    shard = source / "model.safetensors"
    stat = shard.stat()
    cache = tmp_path / "identity_cache.json"

    before = build_source_checkpoint_identity(
        str(source), digest_cache_path=cache)
    assert cache.is_file()

    _write_source(source, weight=_WEIGHT_A, bias=_BIAS_B)
    assert shard.stat().st_size == stat.st_size
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    after = build_source_checkpoint_identity(
        str(source), digest_cache_path=cache)
    assert before["content_sha256"] != after["content_sha256"]


def test_identity_cache_hit_returns_the_same_digest(tmp_path):
    from prismaquant.cost_streaming import build_source_checkpoint_identity

    source = tmp_path / "source"
    _write_source(source, weight=_WEIGHT_A, bias=_BIAS_A)
    cache = tmp_path / "identity_cache.json"
    first = build_source_checkpoint_identity(str(source), digest_cache_path=cache)
    second = build_source_checkpoint_identity(str(source), digest_cache_path=cache)
    assert first == second
