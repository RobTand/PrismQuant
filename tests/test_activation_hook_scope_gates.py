"""Gates that read ``metadata["activation_hook_scope"]`` (#147).

#130 stamped the hooked enumeration's sha256, its size, the rendered count
and a ``render_narrowed`` flag because ``render_scope`` alone cannot tell a
stripe from a whole build. The stamp was written and never read -- provenance
nothing consumes is not a gate. These tests pin the three consumers:

1. ``union_production_cache`` refuses shards whose hook digests disagree
   (equal digests is the rule: each stripe hooks the whole enumeration and
   renders its slice, so binding per-shard digests into the identity would
   refuse every striped union);
2. the cost/export pair is compared, not merely both recorded: the render-cost
   payload carries the hook scope it was priced from and synthesis refuses a
   baseline priced from a different rendering;
3. the export fingerprint records which enumeration the shipped bytes were
   rendered against.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
from prismaquant.production_weight_cache import (
    ACTIVATION_HOOK_SCOPE_SCHEMA,
    ProductionWeightCache,
)
from prismaquant import union_production_cache as union


CODE_IDENTITY = {
    "git_commit": "1" * 40,
    "producer_source_sha256": "2" * 64,
}

HOOK_A = "aa" * 32
HOOK_B = "bb" * 32


def _hook_scope(digest: str, *, hooked: int = 2, rendered: int = 1) -> dict:
    return {
        "schema": ACTIVATION_HOOK_SCOPE_SCHEMA,
        "hooked_qnames_sha256": digest,
        "hooked_qnames": hooked,
        "rendered_qnames": rendered,
        "render_narrowed": rendered != hooked,
    }


def _source_identity() -> dict:
    value_bearing = {
        "config": {"model_type": "test", "hidden_size": 4},
        "weight_map": {
            "model.a.weight": "model.safetensors",
            "model.b.weight": "model.safetensors",
        },
        "shards": [{
            "path": "/source/model.safetensors",
            "size": 128,
            "sha256": "3" * 64,
        }],
        "checkpoint_weight_map": {
            "model.a.weight": "model.safetensors",
            "model.b.weight": "model.safetensors",
        },
    }
    return {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "source": "/source",
        "resolved_commit": "source-commit",
        "content_sha256": canonical_json_sha256(
            value_bearing, where="test source identity"
        ),
        **value_bearing,
    }


def _metadata(entries, assignment, *, extra=None) -> tuple[dict, dict]:
    levers = {"gptq": True, "joint_scale_opt": True}
    records = {
        f"{qname}|{fmt}": {"schema": "test.render_score.v1", "score": 1.0}
        for (qname, fmt) in sorted(entries)
    }
    metadata = {
        "render_scope": "assignment",
        "render_retention": "materialized",
        "requested_formats": sorted({
            fmt for fmt in assignment.values() if fmt != "BF16"
        }),
        "requested_entries": len(entries),
        "streaming": True,
        "calib_hash": "4" * 32,
        "format_plan_identity_sha256": "5" * 64,
        "render_mechanism_order": [{
            "name": "gptq",
            "operation": "gptq",
            "scope": "linear",
            "gate_metric": "output_mse",
        }],
        "render_scores": {
            "schema": "prismaquant.production_render_scores.v1",
            "entries": len(records),
            "records": records,
        },
    }
    metadata.update(extra or {})
    return metadata, levers


def _make_shard(
    root: Path,
    *,
    shard_id: str,
    entries: dict,
    assignment: dict,
    hook_scope: dict,
) -> Path:
    bundle = root / shard_id
    weights = bundle / "weights"
    weights.mkdir(parents=True)
    cache_weights = {}
    activation_max_abs = {}
    for index, ((qname, fmt), tensor) in enumerate(sorted(entries.items())):
        filename = f"render-{index}.pt"
        torch.save(tensor, weights / filename)
        cache_weights[(qname, fmt)] = filename
        activation_max_abs[qname] = float(index + 2)
    metadata, levers = _metadata(
        entries, assignment, extra={"activation_hook_scope": hook_scope}
    )
    cache = ProductionWeightCache(
        weights=cache_weights,
        levers=levers,
        activation_max_abs=activation_max_abs,
        failed={},
        cache_dir="weights",
        metadata=metadata,
    )
    cache_path = bundle / "cache.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = bundle / "shard_manifest.json"
    union.create_shard_manifest(
        cache_path=cache_path,
        cache_dir=weights,
        manifest_path=manifest,
        shard_id=shard_id,
        source_model_identity=_source_identity(),
        settings={"schema": "test.settings.v1", "max_act_rows": 512},
        code_identity=CODE_IDENTITY,
        assignment=assignment,
    )
    return manifest


@pytest.fixture(autouse=True)
def _stable_current_code(monkeypatch):
    monkeypatch.setattr(
        union, "_current_code_identity", lambda: dict(CODE_IDENTITY)
    )


ASSIGNMENT = {"model.a": "NVFP4", "model.b": "FP8_E4M3"}


def test_union_refuses_shards_with_disagreeing_hook_digests(tmp_path):
    """Consumer 1: two renderings must not union into one bundle."""
    shard_a = _make_shard(
        tmp_path,
        shard_id="a",
        entries={("model.a", "NVFP4"): torch.arange(8, dtype=torch.bfloat16)},
        assignment=ASSIGNMENT,
        hook_scope=_hook_scope(HOOK_A),
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="b",
        entries={("model.b", "FP8_E4M3"): torch.arange(6, dtype=torch.bfloat16)},
        assignment=ASSIGNMENT,
        hook_scope=_hook_scope(HOOK_B),
    )
    with pytest.raises(ValueError, match="hook digests differ"):
        union.union_shard_manifests(
            [shard_a, shard_b],
            assignment=ASSIGNMENT,
            output_bundle=tmp_path / "union",
        )


def test_union_accepts_stripes_sharing_one_hook_digest(tmp_path):
    """Equal digests are the pass condition; the merged scope sums the slices.

    Per-shard digests must NOT be bound into the render identity: that would
    refuse every striped union and amount to fixing #130 by disabling
    striping. The shards below render different slice sizes (1 and 2 qnames),
    so exact-equality of the whole scope dict would refuse them; the merged
    scope keeps the shared digest and reports the union's rendered count, so
    a full-coverage union reads ``render_narrowed=False`` exactly like the
    unstriped build it reproduces.
    """
    assignment = {
        "model.a": "NVFP4",
        "model.b": "FP8_E4M3",
        "model.c": "NVFP4",
    }
    shard_a = _make_shard(
        tmp_path,
        shard_id="a",
        entries={("model.a", "NVFP4"): torch.arange(8, dtype=torch.bfloat16)},
        assignment=assignment,
        hook_scope=_hook_scope(HOOK_A, hooked=3, rendered=1),
    )
    shard_b = _make_shard(
        tmp_path,
        shard_id="b",
        entries={
            ("model.b", "FP8_E4M3"): torch.arange(6, dtype=torch.bfloat16),
            ("model.c", "NVFP4"): torch.arange(6, dtype=torch.bfloat16),
        },
        assignment=assignment,
        hook_scope=_hook_scope(HOOK_A, hooked=3, rendered=2),
    )
    output = tmp_path / "union"
    union.union_shard_manifests(
        [shard_a, shard_b], assignment=assignment, output_bundle=output
    )
    _, cache = union.verify_union_manifest(
        output / "union_manifest.json", assignment=assignment
    )
    scope = cache.metadata.get("activation_hook_scope")
    assert scope is not None, "union dropped the activation hook scope"
    assert scope["schema"] == ACTIVATION_HOOK_SCOPE_SCHEMA
    assert scope["hooked_qnames_sha256"] == HOOK_A
    assert scope["hooked_qnames"] == 3
    assert scope["rendered_qnames"] == 3, (
        f"merged scope reports rendered={scope['rendered_qnames']} for a "
        "3-qname union; a shard's slice count leaked into the bundle"
    )
    assert scope["render_narrowed"] is False


def _cost_cache(*, digest: str):
    return ProductionWeightCache(
        weights={},
        levers={"gptq": True},
        metadata={"activation_hook_scope": _hook_scope(digest, rendered=2)},
    )


def _baseline_payload(*, digest: str | None) -> dict:
    provenance: dict = {}
    if digest is not None:
        provenance["activation_hook_scope"] = _hook_scope(digest, rendered=2)
    return {
        "schema": "test.baseline_cost.v1",
        "costs": {
            "model.a": {"NVFP4": {
                "predicted_dloss": 0.5,
                "output_mse_measured": False,
            }},
        },
        "formats": ["NVFP4"],
        "provenance": provenance,
        "meta": {},
    }


def test_render_cost_payload_carries_the_hook_scope_it_was_priced_from():
    """Consumer 2a: the priced contract records its rendering."""
    from prismaquant.production_render_cost import (
        synthesize_production_render_cost_payload,
    )

    payload = synthesize_production_render_cost_payload(
        _cost_cache(digest=HOOK_A),
        _baseline_payload(digest=None),
    )
    scope = payload["provenance"].get("activation_hook_scope")
    assert scope is not None, "cost payload dropped the hook scope"
    assert scope["hooked_qnames_sha256"] == HOOK_A


def test_render_cost_synthesis_refuses_a_differently_rendered_baseline():
    """Consumer 2b: the priced and served contracts are compared, not filed."""
    from prismaquant.production_render_cost import (
        synthesize_production_render_cost_payload,
    )

    with pytest.raises(ValueError, match="hook digests differ"):
        synthesize_production_render_cost_payload(
            _cost_cache(digest=HOOK_A),
            _baseline_payload(digest=HOOK_B),
        )


def test_export_fingerprint_records_which_enumeration_it_ships():
    """Consumer 3: a shipped artifact names the enumeration its bytes saw."""
    from prismaquant.export_native_compressed import (
        _production_cache_fingerprint,
    )

    cache = ProductionWeightCache(
        weights={("model.a", "NVFP4"): torch.arange(4, dtype=torch.float32)},
        levers={"gptq": True},
        metadata={"activation_hook_scope": _hook_scope(HOOK_A, rendered=1)},
    )
    fingerprint = _production_cache_fingerprint(
        cache, [("model.a", "NVFP4")]
    )
    scope = fingerprint.get("activation_hook_scope")
    assert scope is not None, "export fingerprint dropped the hook scope"
    assert scope["hooked_qnames_sha256"] == HOOK_A
