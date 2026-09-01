from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest

from prismaquant.export_native_compressed import (
    _resolve_gptq_fixed_damp,
    gptq_damp_sweep_enabled,
)
from prismaquant.production_weight_cache import _resolve_production_render_levers
from prismaquant.build_production_cache import (
    validate_render_assignment_cache_coverage,
)
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.production_weight_cache import (
    canonical_cb_col_weights_sha256,
)
from prismaquant.production_weight_cache import fill_packed_expert_cache_entries
from prismaquant.production_weight_cache import _format_supports_render_mechanism
from prismaquant.production_weight_cache import fill_production_weight_cache
from prismaquant.production_recache import (
    _load_assignment,
    activation_max_abs_delta_summary,
    assignment_digest,
    preload_production_cache_for_assignment,
    production_cache_keys_for_assignment,
    recache_production_weight_cache,
)


def _normalized_production_cache_levers(value: str | None) -> dict[str, object]:
    """The `--production-cache-levers` string -> provenance dict contract.

    Lived on `kl_sensitivity_probe` until 2026-07-30, when the L3 probe was
    walled (`archive/l3_propagated_2026-07-30/`, re-vet R4). The probe's copy
    was a one-line delegation to `_resolve_production_render_levers` after D5
    was fixed earlier that day, so this shim is that delegation verbatim — the
    assertions below still pin the ONE production render-lever contract
    (notably: sweep OFF by default since 2026-06-12, and sweep-off renders must
    record which fixed damp they used).
    """
    enabled = {
        part.strip()
        for part in str(value or "").split(",")
        if part.strip()
    }
    return dict(sorted(
        _resolve_production_render_levers({name: True for name in enabled}).items()
    ))


class _CBIdentityTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(32, 32, bias=False)

    def forward(self, input_ids, use_cache=False):
        batch, seqlen = input_ids.shape
        x = torch.ones((batch, seqlen, 32), dtype=torch.float32)
        return self.l1(x)



def test_prefetch_loads_disk_entries_and_respects_lru(tmp_path):
    weights = {}
    tensor_nbytes = 0
    for idx, name in enumerate(("a", "b", "c")):
        tensor = torch.full((2, 2), float(idx), dtype=torch.float32)
        tensor_nbytes = tensor.numel() * tensor.element_size()
        path = tmp_path / f"{name}.pt"
        torch.save(tensor, path)
        weights[(name, "NVFP4")] = path.name

    cache = ProductionWeightCache(
        weights=weights,
        levers={"gptq": True},
        cache_dir=str(tmp_path),
    )
    cache.enable_lru(2 * tensor_nbytes)

    assert cache.prefetch(max_workers=2) == 3
    resident = sum(isinstance(value, torch.Tensor) for value in cache.weights.values())

    assert resident <= 2
    assert cache._lru_bytes <= 2 * tensor_nbytes
    assert torch.equal(cache.get("a", "NVFP4"), torch.zeros((2, 2)))

    assert cache.compact_for_pickle() >= 1
    assert all(not isinstance(value, torch.Tensor) for value in cache.weights.values())
    assert cache._lru_bytes == 0


def test_release_resident_tensors_frees_memory_without_losing_data(tmp_path):
    """Releasing after a one-way install must be lossless, not destructive.

    `validate_assignments_kl` drops the cache's resident copies once
    `_materialize_assignment_inplace` has copied the render set into the live
    model, because the hooks then run under
    PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT=1 and never read the cache again.
    On a 27B assignment that is tens of GiB of dead bytes inside a shared
    121.6 GB pool. The guarantee that makes it safe is the one asserted here:
    every key stays resolvable and reloads bit-identically.
    """
    weights = {}
    expected = {}
    for idx, name in enumerate(("a", "b", "c")):
        tensor = torch.full((4, 4), float(idx), dtype=torch.float32)
        torch.save(tensor, tmp_path / f"{name}.pt")
        weights[(name, "NVFP4")] = f"{name}.pt"
        expected[name] = tensor

    cache = ProductionWeightCache(
        weights=weights, levers={"gptq": True}, cache_dir=str(tmp_path))
    cache.enable_lru(64 * 1024 * 1024)
    assert cache.prefetch(max_workers=2) == 3
    assert any(isinstance(v, torch.Tensor) for v in cache.weights.values())

    released = cache.release_resident_tensors()

    assert released >= 1
    assert all(not isinstance(v, torch.Tensor) for v in cache.weights.values())
    assert cache._lru_bytes == 0
    # Every key still resolves, and to the SAME bytes -- a release is a
    # residency decision, never a data one.
    for name, want in expected.items():
        assert torch.equal(cache.get(name, "NVFP4"), want)


def test_spilled_ref_log_probs_round_trip_and_hold_nothing_resident(tmp_path):
    """Reference log-probs go to disk, and come back bit-identical.

    They must all exist before the first measurement (the in-place install
    destroys the model that produces them), and at 27B full-vocab fp32 that is
    ~8 GiB per repeat. Relocating them to host memory would be pointless on a
    unified-memory box -- one pool -- so they go to NVMe. This asserts the
    property that makes that safe: what comes back equals what went in.
    """
    from prismaquant.validate_assignments_kl import _SpilledRefLogProbs

    vocab, seqlen, n_seq = 7, 3, 4

    class _StubModel:
        def __call__(self, batch):
            # Deterministic, distinguishable per sequence.
            base = float(batch[0, 0].item())
            logits = torch.arange(
                seqlen * vocab, dtype=torch.float32
            ).reshape(1, seqlen, vocab) + base
            return SimpleNamespace(logits=logits)

    calib_ids = torch.arange(n_seq * seqlen).reshape(n_seq, seqlen)
    spilled = _SpilledRefLogProbs(
        _StubModel(), calib_ids, torch.device("cpu"),
        kl_scope="full_sequence", out_dir=tmp_path / "refs")

    assert len(spilled) == n_seq
    assert spilled.nbytes == n_seq * seqlen * vocab * 4

    import torch.nn.functional as F
    for i in range(n_seq):
        want = F.log_softmax(
            _StubModel()(calib_ids[i:i + 1]).logits.float(), dim=-1)
        assert torch.equal(spilled[i], want)

    # Nothing cached in the object: each read comes off disk, so N repeats cost
    # one sequence of memory, not N x the whole set.
    assert not any(
        isinstance(v, torch.Tensor) for v in vars(spilled).values())


def test_inplace_measure_can_skip_a_redundant_materialization():
    """Repeats vary the calibration draw, never the weights.

    The in-place install is one-way and nothing reverts it, so re-installing on
    every repeat re-copies identical bytes into the parameters that already
    hold them -- a numeric no-op that re-reads the whole render set and refills
    the LRU. At 27B/repeats=4 that is what drove the pool to its ceiling.
    """
    import inspect

    from prismaquant.validate_assignments_kl import (
        _measure_inplace_assignment_kl,
    )

    sig = inspect.signature(_measure_inplace_assignment_kl)
    assert sig.parameters["materialize"].default is True, (
        "materializing must stay the default; only a caller that KNOWS the "
        "model already holds this assignment may skip it")

    src = inspect.getsource(_measure_inplace_assignment_kl)
    assert "release_resident_tensors" in src, (
        "the one-way install must release the cache's now-unreadable resident "
        "tensors")


def test_production_cache_resolves_format_aliases_and_sizes(tmp_path):
    tensor = torch.ones((2, 2), dtype=torch.float32)
    path = tmp_path / "layer.pt"
    torch.save(tensor, path)
    cache = ProductionWeightCache(
        weights={("layer", "MXFP8_E4M3"): path.name},
        levers={"gptq": True},
        cache_dir=str(tmp_path),
    )

    assert cache.resolve_key("layer", "MXFP8") == ("layer", "MXFP8_E4M3")
    assert ("layer", "MXFP8") in cache
    assert cache.estimate_nbytes([("layer", "MXFP8_E4M3")]) == path.stat().st_size
    assert torch.equal(cache.get("layer", "MXFP8"), tensor)

def test_recache_preload_respects_resident_budget(tmp_path):
    for name in ("a", "b"):
        torch.save(torch.ones((2, 2)), tmp_path / f"{name}.pt")
    cache = ProductionWeightCache(
        weights={
            ("a", "NVFP4"): "a.pt",
            ("b", "MXFP8_E4M3"): "b.pt",
        },
        levers={"gptq": True},
        cache_dir=str(tmp_path),
    )
    assignment = {"a": "NVFP4", "b": "MXFP8", "c": "BF16"}
    keys, missing = production_cache_keys_for_assignment(cache, assignment)
    method_keys, method_missing = cache.assignment_keys(assignment)

    assert keys == [("a", "NVFP4"), ("b", "MXFP8_E4M3")]
    assert missing == []
    assert method_keys == keys
    assert method_missing == []

    stats = cache.prefetch_assignment(
        assignment,
        max_resident_bytes=1,
        require=False,
        progress=False,
    )
    assert stats["skipped"] is True
    assert stats["loaded"] == 0

    stats = preload_production_cache_for_assignment(
        cache,
        assignment,
        max_resident_bytes=10_000_000,
        require=True,
        progress=False,
    )
    assert stats["loaded"] == 2
    assert all(isinstance(cache.weights[k], torch.Tensor) for k in keys)


def test_assignment_keys_ignore_uncached_packed_expert_entries(tmp_path):
    torch.save(torch.ones((2, 2)), tmp_path / "dense.pt")
    cache = ProductionWeightCache(
        weights={("dense", "NVFP4"): "dense.pt"},
        levers={},
        cache_dir=str(tmp_path),
    )

    keys, missing = cache.assignment_keys(
        {
            "dense": "NVFP4",
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.mlp.experts.down_proj": "MXFP8_E4M3",
            "missing_dense": "NVFP4",
        }
    )

    assert keys == [("dense", "NVFP4")]
    assert missing == [("missing_dense", "NVFP4")]


def test_assignment_keys_can_require_packed_expert_entries(tmp_path):
    cache = ProductionWeightCache(weights={}, levers={}, cache_dir=str(tmp_path))

    keys, missing = cache.assignment_keys(
        {"model.layers.0.mlp.experts.gate_up_proj": "NVFP4"},
        include_packed_experts=True,
    )

    assert keys == []
    assert missing == [(
        "model.layers.0.mlp.experts.gate_up_proj",
        "NVFP4",
    )]


def test_packed_expert_cache_docstring_describes_batched_recipe():
    doc = fill_packed_expert_cache_entries.__doc__ or ""

    assert "fixed-damp batched GPTQ" in doc
    assert "without dense JSO/act-order/damp-sweep" in doc
    assert "identical GPTQ + JSO + damp-sweep" not in doc


def test_build_render_assignment_coverage_does_not_skip_packed_experts():
    cache = ProductionWeightCache(
        weights={("dense", "NVFP4"): torch.ones((2, 2))},
        levers={},
    )
    assignment = {
        "dense": "NVFP4",
        "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
        "bf16": "BF16",
    }

    with pytest.raises(RuntimeError, match="assignment coverage failure"):
        validate_render_assignment_cache_coverage(cache, assignment)

    cache.weights[(
        "model.layers.0.mlp.experts.gate_up_proj",
        "NVFP4",
    )] = torch.ones((2, 2, 2))
    validate_render_assignment_cache_coverage(cache, assignment)


def test_cb_col_weights_digest_is_canonical_over_order_dtype_and_strides():
    base = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    noncontiguous = base[:, :, 1]
    assert not noncontiguous.is_contiguous()
    same = noncontiguous.to(torch.float32).contiguous()

    first = canonical_cb_col_weights_sha256(
        {"z": torch.tensor([3.0, 4.0]), "a": noncontiguous},
        ["z", "a"],
    )
    second = canonical_cb_col_weights_sha256(
        {"a": same, "z": torch.tensor([3, 4], dtype=torch.int64)},
        ["a", "z"],
    )
    changed = canonical_cb_col_weights_sha256(
        {"a": same, "z": torch.tensor([3.0, 5.0])},
        ["a", "z"],
    )

    assert first == second
    assert first != changed


def test_cb_source_weight_chunked_digest_matches_contiguous_fp32_reference():
    import hashlib

    from prismaquant.production_weight_cache import (
        _source_weight_value_identity,
    )

    # Exceed the implementation's 4M-element chunk ceiling so this exercises
    # the bounded-memory path rather than only checking the small-tensor case.
    trailing = 1024 * 1024 + 1
    source = torch.arange(2 * 2 * trailing, dtype=torch.int32).reshape(
        2, 2, trailing
    )
    source = source.transpose(0, 1)
    assert not source.is_contiguous()
    shape, observed = _source_weight_value_identity(source)
    reference = source.to(torch.float32).contiguous().numpy().astype(
        "<f4", copy=False
    ).tobytes(order="C")

    assert shape == list(source.shape)
    assert observed == hashlib.sha256(reference).hexdigest()


def test_fresh_cb_cache_records_and_validates_render_identity(monkeypatch):
    import prismaquant.production_weight_cache as pwc
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    model = _CBIdentityTiny()
    context = CBSerializationContext.production()
    col_weights = {"l1": torch.linspace(0.1, 1.0, model.l1.in_features)}
    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda weight, fmt, **_kwargs: weight.detach().to(torch.float32),
    )

    cache = fill_production_weight_cache(
        model,
        torch.tensor([[0, 1]], dtype=torch.long),
        qnames=["l1"],
        formats=["NVFP4_CB_K16"],
        levers={"gptq": False},
        col_weights=col_weights,
        cb_serialization_context=context,
        progress=False,
    )

    identity = cache.metadata["cb_render_identity"]
    assert identity["schema"].endswith("cb_render_identity.v2")
    assert identity["cb_serialized_payload"]["layout_version"] == 2
    assert identity["col_weights_qnames"] == ["l1"]
    assert identity["col_weights_shapes"] == {
        "l1": [model.l1.in_features]
    }
    assert identity["col_weights_sha256"] == canonical_cb_col_weights_sha256(
        col_weights,
        ["l1"],
    )
    assert cache.validate_cb_render_identity(
        expected_context=context,
        col_weights=col_weights,
        require_for_formats=["NVFP4_CB_K16"],
    ) == context

    changed_col_weights = {"l1": col_weights["l1"].clone()}
    changed_col_weights["l1"][0] += 1.0
    with pytest.raises(ValueError, match="col_weights identity differs"):
        cache.validate_cb_render_identity(
            col_weights=changed_col_weights,
            require_for_formats=["NVFP4_CB_K16"],
        )

    from prismaquant.production_weight_cache import (
        validate_cb_render_source_weight,
    )

    validate_cb_render_source_weight(
        identity,
        "l1",
        model.l1.weight.detach(),
        where="test source",
    )
    with pytest.raises(ValueError, match="source-weight value differs"):
        validate_cb_render_source_weight(
            identity,
            "l1",
            model.l1.weight.detach() + 1.0,
            where="test source",
        )

    wrong = CBSerializationContext.legacy_v1()
    with pytest.raises(ValueError, match="differs from allocator recipe"):
        cache.validate_cb_render_identity(expected_context=wrong)


def test_cb_render_identity_projection_recomputes_assignment_scope_hashes():
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext
    from prismaquant.production_weight_cache import (
        bind_cb_render_identity_source_weights,
        build_production_cache_cb_render_identity,
        project_cb_render_identity,
        validate_cb_render_identity_metadata,
    )

    fmt = "NVFP4_CB_K16"
    col_weights = {
        "a": torch.arange(256, dtype=torch.float32),
        "b": torch.arange(256, dtype=torch.float32) + 1,
    }
    source = {
        "a": torch.randn(2, 256),
        "b": torch.randn(3, 256),
    }
    identity = build_production_cache_cb_render_identity(
        {"a": [fmt], "b": [fmt]},
        cb_serialization_context=CBSerializationContext.production(),
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    identity = bind_cb_render_identity_source_weights(identity, source)

    projected = project_cb_render_identity(
        identity,
        {"b": fmt},
        col_weights=col_weights,
        where="test projection",
    )

    assert projected["col_weights_qnames"] == ["b"]
    assert set(projected["source_weights_content_sha256"]) == {"b"}
    assert projected["col_weights_sha256"] != identity["col_weights_sha256"]
    validate_cb_render_identity_metadata(
        projected,
        col_weights=col_weights,
        source_weights={"b": source["b"]},
        where="test projection",
    )


def test_cb_render_source_collector_hashes_each_source_exactly_once(monkeypatch):
    import prismaquant.production_weight_cache as pwc
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    fmt = "NVFP4_CB_K16"
    col_weights = {
        "a": torch.arange(256, dtype=torch.float32),
        "b": torch.arange(256, dtype=torch.float32) + 1,
    }
    source = {
        "a": torch.randn(2, 256),
        "b": torch.randn(3, 256),
    }
    seed = pwc.build_production_cache_cb_render_identity(
        {"a": [fmt], "b": [fmt]},
        cb_serialization_context=CBSerializationContext.production(),
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    expected = pwc.bind_cb_render_identity_source_weights(seed, source)
    calls: list[int] = []
    real_identity = pwc._source_weight_value_identity

    def counted(weight):
        calls.append(id(weight))
        return real_identity(weight)

    monkeypatch.setattr(pwc, "_source_weight_value_identity", counted)
    collector = pwc.CBRenderSourceIdentityCollector(seed, where="test collector")
    collector.observe("a", source["a"])
    collector.observe("b", source["b"])
    completed = collector.finalize()

    assert calls == [id(source["a"]), id(source["b"])]
    assert completed == expected
    assert seed["source_weights_complete"] is False
    assert seed["source_weights_content_sha256"] == {}
    with pytest.raises(ValueError, match="already finalized"):
        collector.finalize()
    with pytest.raises(ValueError, match="already finalized"):
        collector.observe("a", source["a"])


def test_cb_render_source_collector_refuses_duplicate_or_mutated_source():
    import prismaquant.production_weight_cache as pwc
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    col_weights = {"a": torch.arange(256, dtype=torch.float32)}
    seed = pwc.build_production_cache_cb_render_identity(
        {"a": ["NVFP4_CB_K16"]},
        cb_serialization_context=CBSerializationContext.production(),
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    weight = torch.randn(2, 256)
    collector = pwc.CBRenderSourceIdentityCollector(seed, where="test collector")
    collector.observe("a", weight)
    with pytest.raises(ValueError, match="observed more than once"):
        collector.observe("a", weight.clone())

    mutated = pwc.CBRenderSourceIdentityCollector(seed, where="test collector")
    mutated.observe("a", weight)
    weight.add_(1)
    with pytest.raises(ValueError, match="source weight changed"):
        mutated.observe("a", weight)


def test_cb_render_source_collector_finalize_fails_closed_on_partial_scope():
    import prismaquant.production_weight_cache as pwc
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    col_weights = {
        "a": torch.arange(256, dtype=torch.float32),
        "b": torch.arange(256, dtype=torch.float32) + 1,
    }
    seed = pwc.build_production_cache_cb_render_identity(
        {"a": ["NVFP4_CB_K16"], "b": ["NVFP4_CB_K16"]},
        cb_serialization_context=CBSerializationContext.production(),
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    collector = pwc.CBRenderSourceIdentityCollector(seed, where="test collector")
    collector.observe("a", torch.randn(2, 256))
    with pytest.raises(ValueError, match=r"coverage differs: missing=\['b'\]"):
        collector.finalize()
    assert collector.observed_qnames == frozenset({"a"})


def test_dense_cb_resume_rejects_preexisting_shard(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    model = _CBIdentityTiny()
    shard = tmp_path / pwc._cache_weight_filename("l1", "NVFP4_CB_K16")
    torch.save(model.l1.weight.detach(), shard)
    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda weight, fmt, **_kwargs: weight.detach().to(torch.float32),
    )

    with pytest.raises(RuntimeError, match="CB cache resume is disabled"):
        fill_production_weight_cache(
            model,
            torch.tensor([[0, 1]], dtype=torch.long),
            qnames=["l1"],
            formats=["NVFP4_CB_K16"],
            cache_dir=tmp_path,
            levers={"gptq": False},
            col_weights={"l1": torch.ones(model.l1.in_features)},
            cb_serialization_context=CBSerializationContext.production(),
            progress=False,
        )


def test_fresh_cb_render_drops_stale_score_sidecar_entry(tmp_path, monkeypatch):
    import json

    import prismaquant.production_weight_cache as pwc
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    fmt = "NVFP4_CB_K16"
    stale_key = f"l1|{fmt}"
    (tmp_path / "render_scores.json").write_text(json.dumps({
        "schema": "prismaquant.production_render_scores.v1",
        "records": {
            stale_key: {
                "qname": "l1",
                "format": fmt,
                "metric": "output_mse",
                "score": 123.0,
                "score_sum": 123.0,
            },
        },
    }))
    model = _CBIdentityTiny()

    def fail_fresh_render(*_args, **_kwargs):
        raise RuntimeError("fresh render failed")

    monkeypatch.setattr(pwc, "render_production_weight", fail_fresh_render)
    cache = fill_production_weight_cache(
        model,
        torch.tensor([[0, 1]], dtype=torch.long),
        qnames=["l1"],
        formats=[fmt],
        cache_dir=tmp_path,
        levers={"gptq": False},
        col_weights={"l1": torch.ones(model.l1.in_features)},
        cb_serialization_context=CBSerializationContext.production(),
        progress=False,
    )

    assert ("l1", fmt) in cache.failed
    assert stale_key not in cache.metadata["render_scores"]["records"]
    sidecar = json.loads((tmp_path / "render_scores.json").read_text())
    assert stale_key not in sidecar["records"]


def test_legacy_cb_cache_cannot_be_consumed_without_render_identity():
    cache = ProductionWeightCache(
        weights={("l1", "NVFP4_CB_K16"): torch.ones(2, 256)},
        levers={},
        metadata={},
    )

    with pytest.raises(ValueError, match="legacy or partially resumed"):
        cache.get("l1", "NVFP4_CB_K16")


def test_production_cache_file_page_prefetch_does_not_load_tensors(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    for name in ("a", "b"):
        torch.save(torch.ones((2, 2)), tmp_path / f"{name}.pt")
    cache = ProductionWeightCache(
        weights={
            ("a", "NVFP4"): "a.pt",
            ("b", "MXFP8_E4M3"): "b.pt",
        },
        levers={},
        cache_dir=str(tmp_path),
    )
    seen_paths = []

    def fake_prefetch(paths, **kwargs):
        seen_paths.extend(Path(path).name for path in paths)
        return {
            "mode": kwargs["mode"],
            "files": len(paths),
            "bytes": 123,
            "prefetched_bytes": 123,
            "skipped": False,
        }

    monkeypatch.setattr(pwc, "prefetch_files_to_page_cache", fake_prefetch)

    stats = cache.prefetch_assignment_file_pages(
        {"a": "NVFP4", "b": "MXFP8", "c": "BF16"},
        mode="require",
        progress=False,
    )

    assert sorted(seen_paths) == ["a.pt", "b.pt"]
    assert stats["keys"] == 2
    assert stats["missing"] == 0
    assert all(not isinstance(value, torch.Tensor) for value in cache.weights.values())


def test_production_cache_records_damp_sweep_lever(monkeypatch):
    model = nn.Linear(1, 1, bias=False)
    calib_ids = torch.empty((0, 1), dtype=torch.long)

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "0")
    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=[],
        levers={"gptq": True},
        progress=False,
    )

    assert cache.levers["gptq_damp_sweep"] is False
    off = _normalized_production_cache_levers("gptq,scale_sweep")
    assert off["gptq_damp_sweep"] is False
    # Sweep-off renders must record WHICH fixed damp they used, or two
    # fixed-damp caches are metadata-indistinguishable.
    assert off["gptq_fixed_damp"] == _resolve_gptq_fixed_damp()

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1")
    on = _normalized_production_cache_levers("gptq,scale_sweep")
    assert on["gptq_damp_sweep"] is True
    assert "gptq_fixed_damp" not in on

    # The probe's provenance must follow the production policy default
    # (sweep OFF since 2026-06-12, f2363e2) — it kept a stale sweep-ON
    # fork of this defaulting until 2026-07-30.
    monkeypatch.delenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", raising=False)
    assert _normalized_production_cache_levers(
        "gptq,scale_sweep"
    )["gptq_damp_sweep"] is gptq_damp_sweep_enabled() is False


def test_production_cache_none_lever_disables_defaults(monkeypatch):
    model = nn.Linear(1, 1, bias=False)
    calib_ids = torch.empty((0, 1), dtype=torch.long)

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "1")
    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=[],
        levers={"none": True},
        progress=False,
    )

    assert cache.levers["gptq"] is False
    assert cache.levers["gptq_damp_sweep"] is False
    assert cache.levers["scale_sweep"] is False
    normalized = _normalized_production_cache_levers("none")
    assert normalized["gptq"] is False
    assert normalized["gptq_damp_sweep"] is False
    assert normalized["scale_sweep"] is False
    assert normalized["static_act_order"] is False
    assert normalized["joint_scale_opt"] is False


def test_production_cache_records_lift_gptq_levers(monkeypatch):
    model = nn.Linear(1, 1, bias=False)
    calib_ids = torch.empty((0, 1), dtype=torch.long)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=[],
        levers={"gptq": True, "static_act_order": True, "joint_scale_opt": True},
        progress=False,
    )
    normalized = _normalized_production_cache_levers(
        "gptq,static_act_order,joint_scale_opt"
    )

    assert cache.levers["static_act_order"] is True
    assert cache.levers["joint_scale_opt"] is True
    assert cache.levers["nvfp4_scale_rule"] == "joint_mse"
    assert normalized["static_act_order"] is True
    assert normalized["joint_scale_opt"] is True
    assert normalized["nvfp4_scale_rule"] == "joint_mse"
    from prismaquant.render_score import resolve_render_mechanism_order

    names = resolve_render_mechanism_order(
        ("gptq", "static_act_order", "joint_scale_opt")
    ).names()
    assert "static_act_order" in names
    assert "joint_scale_opt" in names


def test_production_cache_records_nvfp4_scale_rule(monkeypatch):
    model = nn.Linear(1, 1, bias=False)
    calib_ids = torch.empty((0, 1), dtype=torch.long)

    monkeypatch.setenv("PRISMAQUANT_NVFP4_SCALE_RULE", "four_over_six_mse")
    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=[],
        levers={"gptq": False, "scale_sweep": False},
        progress=False,
    )

    assert cache.levers["nvfp4_scale_rule"] == "four_over_six_mse"


def test_render_score_clips_nvfp4_but_not_dynamic_mxfp8(monkeypatch):
    from prismaquant.production_weight_cache import _render_score_record

    monkeypatch.setenv("PRISMAQUANT_DISABLE_RTN_COMPILE", "1")
    reference = torch.eye(2, 32, dtype=torch.float32)
    rendered = reference.clone()
    activations = torch.linspace(-3.7, 5.9, 128, dtype=torch.float32).reshape(4, 32)

    nvfp4 = _render_score_record(
        qname="layer",
        fmt="NVFP4",
        render_format="NVFP4",
        reference_weight=reference,
        rendered_weight=rendered,
        activations=activations,
        activation_max_abs=0.5,
    )
    mxfp8 = _render_score_record(
        qname="layer",
        fmt="MXFP8_E4M3",
        render_format="MXFP8_E4M3",
        reference_weight=reference,
        rendered_weight=rendered,
        activations=activations,
        activation_max_abs=0.5,
    )

    assert nvfp4["activation_clipped"] is True
    assert nvfp4["activation_max_abs"] == 0.5
    assert mxfp8["activation_clipped"] is False
    assert mxfp8["activation_max_abs"] is None
    assert mxfp8["activation_quantized"] is True


def test_production_cache_unifies_activation_max_abs_for_profile_fused_siblings(
    monkeypatch,
):
    import prismaquant.export_native_compressed as enc
    import prismaquant.production_weight_cache as pwc

    class FusedActivationTiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(32, 32, bias=False)
            self.k = nn.Linear(32, 32, bias=False)

        def forward(self, input_ids, use_cache=False):
            batch, seqlen = input_ids.shape
            q_input = torch.ones((batch, seqlen, 32), dtype=torch.float32)
            k_input = torch.full((batch, seqlen, 32), 7.0, dtype=torch.float32)
            return self.q(q_input) + self.k(k_input)

    class Profile:
        def fused_sibling_group(self, qname):
            if qname in {"q", "k"}:
                return "qk_fused"
            return None

    def fake_render_production_weight(weight, fmt, **_kwargs):
        return weight.detach().to(torch.float32)

    monkeypatch.setattr(
        enc,
        "_compute_nvfp4_joint_global",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)

    cache = fill_production_weight_cache(
        FusedActivationTiny(),
        torch.tensor([[0, 1]], dtype=torch.long),
        qnames=["q", "k"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False},
        max_act_rows=8,
        recache_profile=Profile(),
        progress=False,
    )

    assert cache.activation_max_abs["q"] == pytest.approx(7.0)
    assert cache.activation_max_abs["k"] == pytest.approx(7.0)


def test_recache_pass_forwards_detected_profile(monkeypatch):
    import prismaquant.export_native_compressed as enc
    import prismaquant.model_profiles as model_profiles
    import prismaquant.production_recache as production_recache
    import prismaquant.production_weight_cache as pwc

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(32, 32, bias=False)

        def forward(self, input_ids, use_cache=False):
            batch, seqlen = input_ids.shape
            x = torch.ones((batch, seqlen, 32), dtype=torch.float32)
            return self.q(x)

    class Profile:
        pass

    detected_profile = Profile()
    observed = {}

    def fake_render_production_weight(weight, fmt, **_kwargs):
        return weight.detach().to(torch.float32)

    def fake_recache(
        model,
        calib_ids,
        assignment,
        cache,
        *,
        profile=None,
        **_kwargs,
    ):
        observed["profile"] = profile
        return {}

    monkeypatch.setattr(
        model_profiles,
        "profile_from_model",
        lambda _model: detected_profile,
    )
    monkeypatch.setattr(
        enc,
        "_compute_nvfp4_joint_global",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)
    monkeypatch.setattr(
        production_recache,
        "recache_production_weight_cache",
        fake_recache,
    )

    fill_production_weight_cache(
        TinyModel(),
        torch.tensor([[0, 1]], dtype=torch.long),
        qnames=["q"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False},
        recache_pass=True,
        recache_assignment={"q": "NVFP4"},
        progress=False,
    )

    assert observed["profile"] is detected_profile


def test_production_cache_gates_four_over_six_as_first_class_plugin(monkeypatch):
    model = _TinyChain()
    with torch.no_grad():
        model.l1.weight.zero_()
        model.l1.weight[:, 0] = 1.0
        model.l1.weight[:, 1] = 0.75
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)

    monkeypatch.setenv("PRISMAQUANT_NVFP4_SCALE_RULE", "four_over_six_mse")
    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False},
        max_act_rows=8,
        progress=False,
    )

    f6 = cache.metadata["four_over_six"]
    assert f6["accepted"] >= 1
    assert cache.metadata["render_gates"]["mechanisms"]["four_over_six"][
        "accepted"
    ] >= 1
    trace = cache.metadata["render_gates"]["records"][0]["trace"]
    assert any(
        step.get("mechanism") == "four_over_six" and step.get("accepted")
        for step in trace
    )

def test_production_cache_passes_fisher_row_weights(monkeypatch, tmp_path):
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    torch.save({"g2_per_token": torch.tensor([1.0, 2.0, 3.0])}, tmp_path / "l1.pt")
    seen: list[torch.Tensor | None] = []

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        fisher_row_weights=None,
        **_kwargs,
    ):
        seen.append(fisher_row_weights)
        return weight.detach().to(torch.float32)

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False, "fisher_gptq": True},
        h_detail_dir=tmp_path,
        max_act_rows=8,
        progress=False,
    )

    assert seen and torch.equal(seen[0], torch.tensor([1.0, 2.0, 3.0]))
    assert cache.metadata["fisher_weighted_gptq"]["loaded"] == 1


def test_production_cache_fisher_rows_resolve_fused_qkv_and_gate_up(
    monkeypatch,
    tmp_path,
):
    import prismaquant.production_weight_cache as pwc

    class FusedTiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(4, 32)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            layer = self.model.layers[0]
            layer.self_attn = nn.Module()
            layer.self_attn.qkv_proj = nn.Linear(32, 96, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.gate_up_proj = nn.Linear(32, 64, bias=False)

        def forward(self, input_ids, use_cache=False):
            x = self.embed(input_ids)
            layer = self.model.layers[0]
            return layer.self_attn.qkv_proj(x) + layer.mlp.gate_up_proj(x).sum() * 0

    def save_detail(name: str, values: list[float]) -> None:
        safe = pwc._FisherRowWeightCache._FNAME_SUB.sub("__", name)
        torch.save({"g2_per_token": torch.tensor(values)}, tmp_path / f"{safe}.pt")

    save_detail("model.layers.0.self_attn.q_proj", [1.0, 2.0, 3.0])
    save_detail("model.layers.0.self_attn.k_proj", [3.0, 4.0, 5.0])
    save_detail("model.layers.0.self_attn.v_proj", [5.0, 6.0, 7.0])
    save_detail("model.layers.0.mlp.gate_proj", [2.0, 4.0, 6.0])
    save_detail("model.layers.0.mlp.up_proj", [4.0, 6.0, 8.0])

    seen: dict[str, torch.Tensor | None] = {}

    def fake_render_production_weight(
        weight,
        fmt,
        *,
        qname=None,
        fisher_row_weights=None,
        **_kwargs,
    ):
        seen[str(qname)] = fisher_row_weights
        return weight.detach().to(torch.float32)

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)

    cache = fill_production_weight_cache(
        FusedTiny(),
        torch.tensor([[0, 1]], dtype=torch.long),
        qnames=[
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.mlp.gate_up_proj",
        ],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False, "fisher_gptq": True},
        h_detail_dir=tmp_path,
        max_act_rows=8,
        progress=False,
    )

    assert torch.equal(
        seen["model.layers.0.self_attn.qkv_proj"],
        torch.tensor([3.0, 4.0, 5.0]),
    )
    assert torch.equal(
        seen["model.layers.0.mlp.gate_up_proj"],
        torch.tensor([3.0, 5.0, 7.0]),
    )
    assert cache.metadata["fisher_weighted_gptq"]["loaded"] == 2
    assert cache.metadata["fisher_weighted_gptq"]["misses"] == 0


def test_fisher_row_cache_uses_configured_fused_sibling_mapping(tmp_path):
    import prismaquant.production_weight_cache as pwc

    def save_detail(name: str, values: list[float]) -> None:
        safe = pwc._FisherRowWeightCache._FNAME_SUB.sub("__", name)
        torch.save({"g2_per_token": torch.tensor(values)}, tmp_path / f"{safe}.pt")

    save_detail("model.layers.0.custom.left", [1.0, 3.0])
    save_detail("model.layers.0.custom.right", [5.0, 7.0])

    cache = pwc._FisherRowWeightCache(
        tmp_path,
        fused_sibling_mapping={"combo": ("left", "right")},
    )

    assert torch.equal(
        cache.get("model.layers.0.custom.combo"),
        torch.tensor([3.0, 5.0]),
    )


def test_production_cache_mxfp8_uses_activation_scale_sweep(monkeypatch):
    import prismaquant.export_native_compressed as enc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    calls = []

    def fake_mxfp8_scale_sweep(weight, activations, **_kwargs):
        calls.append(activations.shape)
        rows, cols = weight.shape
        return (
            torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
            torch.zeros((rows, cols // 32), dtype=torch.uint8),
            weight.detach().to(torch.float32) + 2.0,
        )

    monkeypatch.setattr(enc, "_mxfp8_scale_sweep_quantize", fake_mxfp8_scale_sweep)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["MXFP8_E4M3"],
        levers={"gptq": False, "scale_sweep": True},
        max_act_rows=8,
        progress=False,
    )

    assert calls
    # The fake candidate is intentionally bad; progressive gates should keep
    # the MXFP8 baseline instead of blindly accepting the scale-sweep output.
    assert not torch.allclose(
        cache.get("l1", "MXFP8_E4M3").to(torch.float32),
        model.l1.weight.detach().to(torch.float32) + 2.0,
    )
    scale_meta = cache.metadata["render_gates"]["mechanisms"]["scale_sweep"]
    assert scale_meta["rejected"] == 1
    assert scale_meta["reasons"]["regressed_or_tied"] == 1


def test_production_cache_fp8_uses_activation_scale_sweep(monkeypatch):
    import prismaquant.export_native_compressed as enc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    calls = []

    def fake_fp8_scale_sweep(weight, activations, **_kwargs):
        calls.append(activations.shape)
        rows, _cols = weight.shape
        return (
            torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
            torch.ones((rows, 1), dtype=torch.float32),
            weight.detach().to(torch.float32) + 2.0,
        )

    monkeypatch.setattr(enc, "_fp8_dynamic_scale_sweep_quantize", fake_fp8_scale_sweep)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["FP8_E4M3"],
        levers={"gptq": False, "scale_sweep": True},
        max_act_rows=8,
        progress=False,
    )

    assert calls
    assert not torch.allclose(
        cache.get("l1", "FP8_E4M3").to(torch.float32),
        model.l1.weight.detach().to(torch.float32) + 2.0,
    )
    scale_meta = cache.metadata["render_gates"]["mechanisms"]["scale_sweep"]
    assert scale_meta["rejected"] == 1
    assert scale_meta["reasons"]["regressed_or_tied"] == 1


def test_production_cache_fp8_e4m3_uses_gptq(monkeypatch):
    import prismaquant.export_native_compressed as enc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    calls = []

    def fake_fp8_gptq(weight, activations, **kwargs):
        calls.append((
            activations.shape,
            kwargs["fmt"],
            kwargs["joint_scale_opt"],
            kwargs["static_act_order"],
        ))
        rows, _cols = weight.shape
        return (
            torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
            torch.ones((rows, 1), dtype=torch.float32),
            weight.detach().to(torch.float32),
        )

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "0")
    monkeypatch.setattr(enc, "_gptq_obs_rounding_fp8_like", fake_fp8_gptq)

    fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["FP8_E4M3"],
        levers={"gptq": True, "static_act_order": True, "scale_sweep": False},
        max_act_rows=8,
        progress=False,
    )

    assert calls == [(torch.Size([2, 32]), "FP8_E4M3", False, False)]


def test_format_gate_disables_joint_scale_opt_for_mxfp8():
    assert _format_supports_render_mechanism("NVFP4", "joint_scale_opt")
    assert _format_supports_render_mechanism("NVFP4", "static_act_order")
    assert _format_supports_render_mechanism("MXFP4", "gptq")
    assert _format_supports_render_mechanism("MXFP4", "static_act_order")
    assert not _format_supports_render_mechanism("MXFP4", "joint_scale_opt")
    assert not _format_supports_render_mechanism("FP8_E4M3", "static_act_order")
    assert not _format_supports_render_mechanism("FP8_E5M2", "static_act_order")
    assert _format_supports_render_mechanism("MXFP8_E4M3", "static_act_order")
    assert _format_supports_render_mechanism("MXFP8_E5M2", "static_act_order")
    assert not _format_supports_render_mechanism("FP8_E4M3", "joint_scale_opt")
    assert not _format_supports_render_mechanism("MXFP8_E4M3", "joint_scale_opt")
    assert not _format_supports_render_mechanism("MXFP8_E5M2", "joint_scale_opt")


def test_production_cache_non_nv_scale_sweep_refines_current_render(monkeypatch):
    import prismaquant.export_native_compressed as enc
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    scale_inputs = []

    def fake_fp8_gptq(weight, activations, **kwargs):
        rows, _cols = weight.shape
        del activations, kwargs
        return (
            torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
            torch.ones((rows, 1), dtype=torch.float32),
            weight.detach().to(torch.float32) + 1.0,
        )

    def fake_fp8_scale_sweep(weight, activations, **kwargs):
        rows, _cols = weight.shape
        del activations, kwargs
        scale_inputs.append(weight.detach().clone())
        return (
            torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
            torch.ones((rows, 1), dtype=torch.float32),
            weight.detach().to(torch.float32),
        )

    def fake_gate_score(reference_weight, rendered_weight, activations):
        del activations
        target = reference_weight.to(torch.float32) + 1.0
        if torch.allclose(rendered_weight.to(torch.float32), target):
            return 0.0, "output_mse"
        return 1.0, "output_mse"

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "0")
    monkeypatch.setattr(enc, "_gptq_obs_rounding_fp8_like", fake_fp8_gptq)
    monkeypatch.setattr(enc, "_fp8_dynamic_scale_sweep_quantize", fake_fp8_scale_sweep)
    monkeypatch.setattr(pwc, "_render_score_for_gate", fake_gate_score)

    fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["FP8_E4M3"],
        levers={"gptq": True, "scale_sweep": True},
        max_act_rows=8,
        progress=False,
    )

    assert scale_inputs
    assert torch.allclose(
        scale_inputs[0],
        model.l1.weight.detach().to(torch.float32) + 1.0,
    )


def test_production_cache_mxfp8_e5m2_uses_gptq_without_joint_scale(monkeypatch):
    import prismaquant.export_native_compressed as enc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    calls = []

    def fake_mxfp8_gptq(weight, activations, **kwargs):
        calls.append((
            activations.shape,
            kwargs["fmt"],
            kwargs["joint_scale_opt"],
            kwargs["static_act_order"],
        ))
        rows, cols = weight.shape
        return (
            torch.zeros_like(weight, dtype=torch.float8_e5m2),
            torch.zeros((rows, cols // 32), dtype=torch.uint8),
            weight.detach().to(torch.float32),
        )

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "0")
    monkeypatch.setattr(enc, "_gptq_obs_rounding_fp8_like", fake_mxfp8_gptq)

    fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["MXFP8_E5M2"],
        levers={
            "gptq": True,
            "joint_scale_opt": True,
            "static_act_order": True,
            "scale_sweep": False,
        },
        max_act_rows=8,
        progress=False,
    )

    assert calls == [
        (torch.Size([2, 32]), "MXFP8_E5M2", False, False),
        (torch.Size([2, 32]), "MXFP8_E5M2", False, True),
    ]


def test_production_cache_mxfp8_static_act_order_do_no_harm(monkeypatch):
    import prismaquant.export_native_compressed as enc
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    calls = []

    def fake_mxfp8_gptq(weight, activations, **kwargs):
        del activations
        use_static = bool(kwargs["static_act_order"])
        calls.append(use_static)
        rows, cols = weight.shape
        candidate = (
            weight.detach().to(torch.float32) + (2.0 if use_static else 1.0)
        )
        return (
            torch.zeros_like(weight, dtype=torch.float8_e4m3fn),
            torch.zeros((rows, cols // 32), dtype=torch.uint8),
            candidate,
        )

    def fake_gate_score(reference_weight, rendered_weight, activations):
        del activations
        target = reference_weight.to(torch.float32) + 1.0
        return float((rendered_weight.to(torch.float32) - target).pow(2).sum()), "output_mse"

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "0")
    monkeypatch.setattr(enc, "_gptq_obs_rounding_fp8_like", fake_mxfp8_gptq)
    monkeypatch.setattr(pwc, "_render_score_for_gate", fake_gate_score)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["MXFP8_E4M3"],
        levers={"gptq": True, "static_act_order": True, "scale_sweep": False},
        max_act_rows=8,
        progress=False,
    )

    assert calls == [False, True]
    assert torch.allclose(
        cache.get("l1", "MXFP8_E4M3").to(torch.float32),
        model.l1.weight.detach().to(torch.float32) + 1.0,
    )
    mechanisms = cache.metadata["render_gates"]["mechanisms"]
    assert mechanisms["gptq"]["accepted"] == 1
    assert mechanisms.get("static_act_order", {}).get("package_accepted", 0) == 0


def test_production_cache_mxfp4_static_act_order_do_no_harm(monkeypatch):
    import prismaquant.export_native_compressed as enc
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    calls = []

    def fake_mxfp4_gptq(weight, activations, **kwargs):
        del activations
        use_static = bool(kwargs["static_act_order"])
        calls.append(use_static)
        rows, cols = weight.shape
        candidate = (
            weight.detach().to(torch.float32) + (2.0 if use_static else 1.0)
        )
        return (
            torch.zeros((rows, cols // 2), dtype=torch.uint8),
            torch.zeros((rows, cols // 32), dtype=torch.uint8),
            candidate,
        )

    def fake_gate_score(reference_weight, rendered_weight, activations):
        del activations
        target = reference_weight.to(torch.float32) + 1.0
        return float((rendered_weight.to(torch.float32) - target).pow(2).sum()), "output_mse"

    monkeypatch.setenv("PRISMAQUANT_GPTQ_DAMP_SWEEP", "0")
    monkeypatch.setattr(enc, "_gptq_obs_rounding_mxfp4", fake_mxfp4_gptq)
    monkeypatch.setattr(pwc, "_render_score_for_gate", fake_gate_score)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["MXFP4"],
        levers={"gptq": True, "static_act_order": True, "scale_sweep": False},
        max_act_rows=8,
        progress=False,
    )

    assert calls == [False, True]
    assert torch.allclose(
        cache.get("l1", "MXFP4").to(torch.float32),
        model.l1.weight.detach().to(torch.float32) + 1.0,
    )
    mechanisms = cache.metadata["render_gates"]["mechanisms"]
    assert mechanisms["gptq"]["accepted"] == 1
    assert mechanisms.get("static_act_order", {}).get("package_accepted", 0) == 0


def test_fill_production_cache_assignment_scope_only_renders_selected_formats(
    monkeypatch,
):
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    seen: list[tuple[str | None, str]] = []

    def fake_render_production_weight(weight, fmt, *, qname=None, **_kwargs):
        seen.append((qname, fmt.upper()))
        return weight.detach().to(torch.float32)

    monkeypatch.setattr(pwc, "render_production_weight", fake_render_production_weight)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1", "l2"],
        formats=["NVFP4", "MXFP8_E4M3"],
        render_assignment={"l1": "NVFP4", "l2": "BF16"},
        levers={"gptq": False, "scale_sweep": False},
        max_act_rows=8,
        progress=False,
    )

    assert seen == [("l1", "NVFP4")]
    assert ("l1", "NVFP4") in cache
    assert ("l1", "MXFP8_E4M3") not in cache
    assert ("l2", "NVFP4") not in cache
    assert cache.metadata["render_scope"] == "assignment"
    assert cache.metadata["requested_entries"] == 1


def test_production_cache_records_render_scores(monkeypatch, tmp_path):
    import prismaquant.export_native_compressed as enc
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    atomic_paths: list[Path] = []
    real_atomic_write = pwc.atomic_write_bytes

    def record_atomic_write(path, payload):
        atomic_paths.append(Path(path))
        real_atomic_write(path, payload)

    monkeypatch.setattr(
        enc,
        "_compute_nvfp4_joint_global",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda weight, fmt, **_kwargs: weight.detach().to(torch.float32),
    )
    monkeypatch.setattr(pwc, "atomic_write_bytes", record_atomic_write)

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False},
        cache_dir=tmp_path,
        max_act_rows=8,
        progress=False,
    )

    scores = cache.metadata["render_scores"]["records"]
    record = scores["l1|NVFP4"]
    assert record["metric"] == "output_mse"
    assert record["activation_rows"] == 2
    assert record["normalizer"] == 64.0
    assert record["score"] == pytest.approx(0.0)
    assert (tmp_path / "render_scores.json").is_file()
    assert tmp_path / "activation_max_abs.json" in atomic_paths


def test_resume_collects_activations_when_render_scores_missing(tmp_path):
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    torch.save(
        model.l1.weight.detach().to(torch.float32),
        tmp_path / pwc._cache_weight_filename("l1", "NVFP4"),
    )

    cache = fill_production_weight_cache(
        model,
        calib_ids,
        qnames=["l1"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False},
        cache_dir=tmp_path,
        max_act_rows=8,
        progress=False,
    )

    record = cache.metadata["render_scores"]["records"]["l1|NVFP4"]
    assert record["activation_rows"] == 2
    assert record["metric"] == "output_mse"
    assert (tmp_path / "render_scores.json").is_file()


def test_render_score_record_raises_on_activation_score_failure(monkeypatch):
    import prismaquant.format_registry as fr
    import prismaquant.production_weight_cache as pwc

    def bad_activation_quantizer(_x):
        raise RuntimeError("quantizer failed")

    monkeypatch.setattr(
        fr,
        "get_format",
        lambda _fmt: SimpleNamespace(
            activation_quantize_dequantize=bad_activation_quantizer,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="activation-aware render scoring failed for l1 @ FP8_E4M3",
    ):
        pwc._render_score_record(
            qname="l1",
            fmt="FP8_E4M3",
            render_format="FP8_E4M3",
            reference_weight=torch.eye(4),
            rendered_weight=torch.eye(4),
            activations=torch.randn(2, 4),
            activation_max_abs=None,
        )


class _TinyChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 32)
        self.l1 = nn.Linear(32, 32, bias=False)
        self.l2 = nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[0, 0] = 1.0
            self.embed.weight[1, 1] = 2.0
            self.l1.weight.copy_(torch.eye(32))
            self.l2.weight.fill_(1.0)

    def forward(self, input_ids, use_cache=False):
        x = self.embed(input_ids)
        x = self.l1(x)
        return self.l2(x)


def test_resume_refuses_truncated_activation_max_abs_sidecar(tmp_path):
    import json
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    torch.save(
        model.l1.weight.detach().to(torch.float32),
        tmp_path / pwc._cache_weight_filename("l1", "NVFP4"),
    )
    (tmp_path / "render_scores.json").write_text(json.dumps({
        "schema": "prismaquant.production_render_scores.v1",
        "records": {"l1|NVFP4": {}},
    }))
    sidecar = tmp_path / "activation_max_abs.json"
    sidecar.write_text('{"l1":')

    with pytest.raises(RuntimeError) as exc_info:
        fill_production_weight_cache(
            model,
            torch.tensor([[0, 1]], dtype=torch.long),
            qnames=["l1"],
            formats=["NVFP4"],
            levers={"gptq": False, "scale_sweep": False},
            cache_dir=tmp_path,
            max_act_rows=8,
            progress=False,
        )

    assert str(sidecar) in str(exc_info.value)


def test_resume_loads_existing_activation_max_abs_sidecar(tmp_path):
    import json
    import prismaquant.production_weight_cache as pwc

    model = _TinyChain()
    torch.save(
        model.l1.weight.detach().to(torch.float32),
        tmp_path / pwc._cache_weight_filename("l1", "NVFP4"),
    )
    (tmp_path / "render_scores.json").write_text(json.dumps({
        "schema": "prismaquant.production_render_scores.v1",
        "records": {"l1|NVFP4": {}},
    }))
    sidecar = tmp_path / "activation_max_abs.json"
    sidecar.write_text(json.dumps({"l1": 2.5}, indent=2))

    cache = fill_production_weight_cache(
        model,
        torch.tensor([[0, 1]], dtype=torch.long),
        qnames=["l1"],
        formats=["NVFP4"],
        levers={"gptq": False, "scale_sweep": False},
        cache_dir=tmp_path,
        max_act_rows=8,
        progress=False,
    )

    assert cache.activation_max_abs == {"l1": 2.5}


def test_resume_refuses_truncated_render_score_sidecar(tmp_path):
    sidecar = tmp_path / "render_scores.json"
    sidecar.write_text('{"records":')

    with pytest.raises(RuntimeError) as exc_info:
        fill_production_weight_cache(
            _TinyChain(),
            torch.tensor([[0, 1]], dtype=torch.long),
            qnames=["l1"],
            formats=["NVFP4"],
            levers={"gptq": False, "scale_sweep": False},
            cache_dir=tmp_path,
            max_act_rows=8,
            progress=False,
        )

    assert str(sidecar) in str(exc_info.value)


def test_production_recache_measures_quantized_upstream_activation_range():
    model = _TinyChain()
    cache = ProductionWeightCache(
        weights={("l1", "NVFP4"): 3.0 * torch.eye(32)},
        levers={"gptq": True, "scale_sweep": True},
        activation_max_abs={"l1": 2.0, "l2": 2.0},
    )
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)

    max_abs = recache_production_weight_cache(
        model,
        calib_ids,
        {"l1": "NVFP4", "l2": "BF16"},
        cache,
        include_activation_quant=False,
        progress=False,
    )

    assert max_abs["l1"] == pytest.approx(2.0)
    assert max_abs["l2"] == pytest.approx(6.0)
    assert cache.activation_max_abs["l2"] == pytest.approx(6.0)
    assert cache.metadata["activation_recache"]["status"] == "applied"
    assert cache.metadata["activation_recache"]["assignment_entries"] == 2
    assert cache.metadata["activation_recache"]["assignment_sha256"] == assignment_digest(
        {"l2": "BF16", "l1": "NVFP4"}
    )
    delta = cache.metadata["activation_recache"]["activation_max_abs_delta"]
    assert delta["n_common"] == 2
    assert delta["changed_gt_5pct"] == 1
    assert delta["ratio_p50"] == pytest.approx(2.0)


def test_production_recache_atomically_publishes_activation_sidecar(
    monkeypatch,
    tmp_path,
):
    import json
    import prismaquant.production_recache as production_recache

    cache = ProductionWeightCache(
        weights={},
        levers={},
        cache_dir=str(tmp_path),
        activation_max_abs={"l1": 1.0},
    )
    monkeypatch.setattr(
        production_recache,
        "measure_production_activation_max_abs",
        lambda *_args, **_kwargs: {"l1": 2.0},
    )
    atomic_paths: list[Path] = []
    real_atomic_write = production_recache.atomic_write_bytes

    def record_atomic_write(path, payload):
        atomic_paths.append(Path(path))
        real_atomic_write(path, payload)

    monkeypatch.setattr(
        production_recache,
        "atomic_write_bytes",
        record_atomic_write,
    )

    recache_production_weight_cache(
        nn.Linear(1, 1),
        torch.ones((1, 1), dtype=torch.long),
        {"l1": "BF16"},
        cache,
        progress=False,
    )

    sidecar = tmp_path / "activation_max_abs.json"
    assert atomic_paths == [sidecar]
    assert json.loads(sidecar.read_text()) == {"l1": 2.0}


def test_production_recache_tempdir_uses_requested_parent(monkeypatch, tmp_path):
    import prismaquant.production_recache as production_recache

    seen_dirs: list[Path] = []

    class FakeTemporaryDirectory:
        def __init__(self, *, prefix, dir):
            del prefix
            self.path = Path(dir) / "fake_recache"
            seen_dirs.append(Path(dir))

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeActivationCache:
        def __init__(self, *args, **kwargs):
            self.max_abs = {"q": 1.0}

        def install(self):
            pass

        def remove(self):
            pass

    monkeypatch.setattr(
        production_recache.tempfile,
        "TemporaryDirectory",
        FakeTemporaryDirectory,
    )
    monkeypatch.setattr(
        production_recache,
        "PerturbedActivationCache",
        FakeActivationCache,
    )
    monkeypatch.setattr(
        production_recache,
        "iter_calibration_forwards",
        lambda *args, **kwargs: [],
    )

    production_recache.measure_production_activation_max_abs(
        nn.Linear(1, 1),
        torch.ones(1, 1, dtype=torch.long),
        {"q": "BF16"},
        ProductionWeightCache(weights={}, levers={}),
        progress=False,
        temp_parent=tmp_path,
    )

    assert seen_dirs == [tmp_path]


def test_activation_max_abs_delta_summary_reports_ratio_quantiles():
    summary = activation_max_abs_delta_summary(
        {"a": 1.0, "b": 2.0, "c": 4.0, "missing": 3.0},
        {"a": 1.0, "b": 3.0, "c": 2.0},
    )

    assert summary["n_common"] == 3
    assert summary["n_before"] == 4
    assert summary["n_after"] == 3
    assert summary["ratio_min"] == pytest.approx(0.5)
    assert summary["ratio_p50"] == pytest.approx(1.0)
    assert summary["ratio_max"] == pytest.approx(1.5)
    assert summary["changed_gt_1pct"] == 2
    assert summary["changed_gt_5pct"] == 2


def test_fill_production_cache_recache_requires_concrete_assignment():
    model = nn.Linear(1, 1, bias=False)
    calib_ids = torch.empty((0, 1), dtype=torch.long)

    with pytest.raises(ValueError, match="recache_assignment"):
        fill_production_weight_cache(
            model,
            calib_ids,
            qnames=[],
            progress=False,
            recache_pass=True,
        )


def test_recache_assignment_loader_is_exporter_independent(monkeypatch, tmp_path):
    path = tmp_path / "layer_config.json"
    path.write_text(
        '{"layer.weight": {"bits": 4, "group_size": 16, "data_type": "nv_fp", '
        '"act_bits": 4, "act_group_size": 16, "act_data_type": "nv_fp"}}'
    )

    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name in {"accelerate", "prismaquant.export_native_compressed"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    assert _load_assignment(path) == {"layer": "NVFP4"}


def test_render_gate_scores_on_gptq_clipped_activations():
    """Audit 2026-07-02 §3.9: the cache-fill progressive gate must score
    candidates on the SAME clipped activation matrix the GPTQ loop optimized
    under (export's shared-matrix contract), not the raw outlier-carrying
    capture — otherwise accept-vs-RTN decisions near ties are biased by the
    outlier rows."""
    import prismaquant.production_weight_cache as pwc
    from prismaquant import export_native_compressed as enc
    from prismaquant.render_score import score_render_error

    torch.manual_seed(0)
    weight = torch.randn(8, 32)
    acts = torch.randn(64, 32) * 0.3
    acts[5] = 40.0    # outlier tokens: clipped away by the GPTQ matrix
    acts[17] = -55.0
    clip = 1.0

    trace: list[dict] = []
    rendered = pwc.render_production_weight(
        weight,
        "NVFP4",
        qname="lin",
        activations={"lin": acts},
        levers={"gptq": True},
        act_clip_threshold=clip,
        gate_trace=trace,
    )

    X = enc._activation_matrix_for_gptq(
        acts.float(), 32, clip_threshold=clip, clip_rescale="none",
    )

    # Baseline candidate = static_6 RTN; its recorded gate score must be the
    # shared scorer evaluated on the CLIPPED matrix, not on raw acts.
    with pwc._temporary_nvfp4_scale_rule(enc.NVFP4_SCALE_RULE_STATIC_6):
        baseline = enc._rtn_dequant_nvfp4(weight.float(), group_size=16)
    expected_clipped = score_render_error(weight.float(), baseline, X)
    raw_score = score_render_error(weight.float(), baseline, acts.float())
    assert trace[0]["mechanism"] == "baseline"
    assert trace[0]["metric"] == "output_mse"
    assert trace[0]["score"] == pytest.approx(expected_clipped, rel=1e-5)
    # Sanity: on this outlier set the two matrices give very different
    # scores, so the assertion above actually discriminates.
    assert abs(raw_score - expected_clipped) > 10.0 * expected_clipped

    # The selected candidate's recorded score is also clip-consistent with
    # the returned weight.
    final_clipped = score_render_error(weight.float(), rendered.float(), X)
    step = trace[1]
    assert step["mechanism"] in ("gptq", "fisher_gptq")
    selected_score = (
        step["candidate_score"] if step["accepted"] else step["baseline_score"]
    )
    assert selected_score == pytest.approx(final_clipped, rel=1e-5)


def test_non_nv_render_gate_scores_on_gptq_clipped_activations():
    """Same clip-consistency contract for the non-NVFP4 (FP8) gate path."""
    import prismaquant.production_weight_cache as pwc
    from prismaquant import export_native_compressed as enc
    from prismaquant import format_registry as fr
    from prismaquant.render_score import score_render_error

    torch.manual_seed(1)
    weight = torch.randn(8, 32)
    acts = torch.randn(64, 32) * 0.3
    acts[9] = 35.0
    clip = 1.0

    trace: list[dict] = []
    rendered = pwc.render_production_weight(
        weight,
        "FP8_E4M3",
        qname="lin",
        activations={"lin": acts},
        levers={"gptq": True},
        act_clip_threshold=clip,
        gate_trace=trace,
    )

    X = enc._activation_matrix_for_gptq(
        acts.float(), 32, clip_threshold=clip, clip_rescale="none",
    )
    baseline = fr.get_format("FP8_E4M3").quantize_dequantize(
        weight.detach().clone()
    )
    expected_clipped = score_render_error(weight.float(), baseline.float(), X)
    assert trace[0]["mechanism"] == "baseline"
    assert trace[0]["score"] == pytest.approx(expected_clipped, rel=1e-5)
    final_clipped = score_render_error(weight.float(), rendered.float(), X)
    step = trace[1]
    selected_score = (
        step["candidate_score"] if step["accepted"] else step["baseline_score"]
    )
    assert selected_score == pytest.approx(final_clipped, rel=1e-5)
