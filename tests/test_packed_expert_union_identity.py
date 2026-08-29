"""A packed-MoE expert append must leave the cache's render identity exact.

``fill_packed_expert_cache_entries`` pushes 3-D ``(experts_qname.pn, fmt)``
keys straight into ``cache.weights``.  ``union_production_cache`` treats three
counters as invariants of a materialized cache and refuses the exact union at
its first subcommand when any of them goes stale:

  * ``requested_entries`` must equal the materialized key count
    (``union_production_cache.py`` ``_render_identity``);
  * one ``render_scores`` record per cache key (``_validate_render_scores``);
  * one ``render_gates`` record per non-MTP cache key (``_merge_render_gates``
    plus the merged-coverage check in ``union_shard_manifests``).

Every packed expert is scored honestly (per-expert render scores summed into
the one record the cache key owns) and every packed entry carries a truthful
gate record: an EMPTY trace, because no progressive-gate mechanism runs on the
packed path, plus the per-expert GPTQ-vs-RTN do-no-harm counts the render did
measure.

Both directions are asserted.  A test that only proves the union PASSES says
nothing about whether the gate can still fire, so each invariant is also
broken on purpose and the specific refusal is pinned.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from prismaquant import union_production_cache as union
from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    _cache_weight_filename,
    fill_packed_expert_cache_entries,
)

from test_packed_expert_cross_domain_gate import ASSIGNMENT, TinyLM
from test_union_production_cache import CODE_IDENTITY, _source_identity


DENSE_QNAME = "model.dense"
DENSE_FMT = "NVFP4"
CALIB_HASH = "4" * 32
FULL_ASSIGNMENT = {DENSE_QNAME: DENSE_FMT, **ASSIGNMENT}
SETTINGS = {
    "schema": "test.union_settings.v1",
    "max_act_rows": 512,
    "dtype": "bf16",
}


@pytest.fixture(autouse=True)
def _stable_current_code(monkeypatch):
    monkeypatch.setattr(
        union, "_current_code_identity", lambda: dict(CODE_IDENTITY)
    )


def _base_metadata() -> dict:
    """Metadata as the non-streaming dense fill leaves it for a shard with no
    dense keys: an empty-but-present ``render_gates`` scope (the union demands
    the key on every shard once any shard has it) and zeroed counters."""
    return {
        "render_scope": "assignment",
        "render_retention": "materialized",
        "requested_formats": [DENSE_FMT],
        "requested_entries": 0,
        "streaming": False,
        "calib_hash": CALIB_HASH,
        "format_plan_identity_sha256": "5" * 64,
        "render_mechanism_order": [{
            "name": "gptq",
            "operation": "gptq",
            "scope": "linear",
            "gate_metric": "output_mse",
        }],
        "render_scores": {
            "schema": "prismaquant.production_render_scores.v1",
            "entries": 0,
            "records": {},
        },
        "render_gates": {
            "enabled": True,
            "entries": 0,
            "mechanisms": {},
            "records": [],
        },
    }


def _packed_only_cache(cache_dir: Path) -> ProductionWeightCache:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ProductionWeightCache(
        weights={},
        levers={"gptq": True},
        activation_max_abs={},
        failed={},
        cache_dir=str(cache_dir),
        metadata=_base_metadata(),
    )


def _dense_cache(cache_dir: Path) -> ProductionWeightCache:
    """A one-dense-entry cache shaped exactly as the non-streaming fill leaves
    it: score record, gate record with a real mechanism trace, and the three
    counters agreeing with a single cache key."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = _cache_weight_filename(DENSE_QNAME, DENSE_FMT)
    torch.save(torch.arange(8, dtype=torch.bfloat16), cache_dir / fname)
    metadata = {
        "render_scope": "assignment",
        "render_retention": "materialized",
        "requested_formats": [DENSE_FMT],
        "requested_entries": 1,
        "streaming": False,
        "calib_hash": CALIB_HASH,
        "format_plan_identity_sha256": "5" * 64,
        "render_mechanism_order": [{
            "name": "gptq",
            "operation": "gptq",
            "scope": "linear",
            "gate_metric": "output_mse",
        }],
        "render_scores": {
            "schema": "prismaquant.production_render_scores.v1",
            "entries": 1,
            "records": {
                f"{DENSE_QNAME}|{DENSE_FMT}": {
                    "qname": DENSE_QNAME,
                    "format": DENSE_FMT,
                    "metric": "output_mse",
                    "score": 1.0,
                },
            },
        },
        "render_gates": {
            "enabled": True,
            "entries": 1,
            "mechanisms": {
                "gptq": {
                    "accepted": 1,
                    "rejected": 0,
                    "reasons": {"improved": 1},
                    "package_accepted": 0,
                },
            },
            "records": [{
                "qname": DENSE_QNAME,
                "format": DENSE_FMT,
                "render_format": DENSE_FMT,
                "trace": [{
                    "mechanism": "gptq",
                    "accepted": True,
                    "reason": "improved",
                }],
            }],
        },
    }
    return ProductionWeightCache(
        weights={(DENSE_QNAME, DENSE_FMT): fname},
        levers={"gptq": True},
        activation_max_abs={DENSE_QNAME: 3.0},
        failed={},
        cache_dir=str(cache_dir),
        metadata=metadata,
    )


def _fill_packed(cache: ProductionWeightCache, cache_dir: Path, seed: int = 11):
    torch.manual_seed(seed)
    model = TinyLM().eval()
    calib = torch.randint(0, 32, (2, 64))
    return fill_packed_expert_cache_entries(
        cache,
        model,
        calib,
        render_assignment=ASSIGNMENT,
        levers={"gptq": True},
        profile=None,
        module_token_budget=4096,
        eval_rows_per_expert=8,
        cache_dir=cache_dir,
        progress=False,
    )


def _write_manifest(
    bundle: Path,
    cache: ProductionWeightCache,
    cache_dir: Path,
    *,
    shard_id: str = "sparky",
) -> Path:
    cache_path = bundle / "cache.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path = bundle / "shard_manifest.json"
    union.create_shard_manifest(
        cache_path=cache_path,
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        shard_id=shard_id,
        source_model_identity=_source_identity(),
        settings=SETTINGS,
        code_identity=CODE_IDENTITY,
        assignment=FULL_ASSIGNMENT,
    )
    return manifest_path


def _packed_shard(
    tmp_path: Path,
    *,
    fills: int = 1,
    name: str = "shard",
    dense: bool = True,
) -> tuple[Path, Path, ProductionWeightCache]:
    bundle = tmp_path / name
    cache_dir = bundle / "weights"
    cache = _dense_cache(cache_dir) if dense else _packed_only_cache(cache_dir)
    for _ in range(fills):
        _fill_packed(cache, cache_dir)
    return bundle, cache_dir, cache


def _dense_shard(tmp_path: Path, *, name: str = "dense-shard") -> Path:
    """The sibling stripe of an exact union: dense keys only."""
    bundle = tmp_path / name
    cache_dir = bundle / "weights"
    cache = _dense_cache(cache_dir)
    return _write_manifest(bundle, cache, cache_dir, shard_id="sparky")


# ---------------------------------------------------------------------------
# PASS direction
# ---------------------------------------------------------------------------

def test_packed_append_keeps_the_exact_union_manifest_valid(tmp_path):
    bundle, cache_dir, cache = _packed_shard(tmp_path)

    assert len(cache) == 1 + len(ASSIGNMENT)
    metadata = cache.metadata
    assert metadata["requested_entries"] == len(cache)
    assert metadata["render_scores"]["entries"] == len(cache)
    assert metadata["render_gates"]["entries"] == len(cache)
    # The dense records survive the packed append untouched.
    assert (
        metadata["render_scores"]["records"][f"{DENSE_QNAME}|{DENSE_FMT}"][
            "score"
        ] == 1.0
    )
    assert metadata["render_gates"]["mechanisms"]["gptq"]["accepted"] == 1
    # Every packed entry is scored, and the gate record is the truthful
    # "nothing progressive ran" form rather than a synthesized trace step.
    for qname in ASSIGNMENT:
        record = metadata["render_scores"]["records"][f"{qname}|NVFP4"]
        assert record["packed_experts"] == 2
        assert record["experts_scored_with_activations"] == 2
        assert record["metric"] == "output_mse"
        assert record["score"] > 0.0
    packed_gates = [
        row for row in metadata["render_gates"]["records"]
        if row["qname"] in ASSIGNMENT
    ]
    assert len(packed_gates) == len(ASSIGNMENT)
    for row in packed_gates:
        assert row["trace"] == []
        assert row["progressive_gate"] == "not-run"
        assert row["mechanism"] == "batched_gptq_fixed_damp"
        assert row["expert_do_no_harm"]["n_experts"] == 2
    assert set(metadata["packed_expert_coverage"]) == set(ASSIGNMENT)

    _write_manifest(bundle, cache, cache_dir)


def test_packed_shard_unions_with_a_dense_sibling(tmp_path):
    bundle, cache_dir, cache = _packed_shard(
        tmp_path, name="packed-shard", dense=False,
    )
    assert len(cache) == len(ASSIGNMENT)
    packed_manifest = _write_manifest(
        bundle, cache, cache_dir, shard_id="sparklina",
    )
    dense_manifest = _dense_shard(tmp_path)

    output = tmp_path / "union"
    union.union_shard_manifests(
        [dense_manifest, packed_manifest],
        assignment=FULL_ASSIGNMENT,
        output_bundle=output,
    )
    payload, merged = union.verify_union_manifest(
        output / "union_manifest.json", assignment=FULL_ASSIGNMENT
    )
    expected = 1 + len(ASSIGNMENT)
    assert payload["entries"] == expected
    assert merged.metadata["requested_entries"] == expected
    assert merged.metadata["render_scores"]["entries"] == expected
    assert merged.metadata["render_gates"]["entries"] == expected
    assert set(merged.metadata["packed_expert_coverage"]) == set(ASSIGNMENT)


def test_repeated_packed_fill_stays_exact(tmp_path):
    """The M4 lazy gap-fill calls this in a loop over frontier points with
    overlapping assignments. Records are replaced by key, never appended, and
    the counters are recomputed rather than incremented."""
    bundle, cache_dir, cache = _packed_shard(tmp_path, fills=3)

    metadata = cache.metadata
    assert len(cache) == 1 + len(ASSIGNMENT)
    assert metadata["requested_entries"] == len(cache)
    assert metadata["render_scores"]["entries"] == len(cache)
    assert metadata["render_gates"]["entries"] == len(cache)
    pairs = [
        (row["qname"], row["format"])
        for row in metadata["render_gates"]["records"]
    ]
    assert len(pairs) == len(set(pairs))

    _write_manifest(bundle, cache, cache_dir)


# ---------------------------------------------------------------------------
# REFUSE direction — the same gates must still fire when the metadata lies
# ---------------------------------------------------------------------------

def test_manifest_refuses_a_dropped_packed_render_score(tmp_path):
    bundle, cache_dir, cache = _packed_shard(tmp_path)
    victim = f"{sorted(ASSIGNMENT)[0]}|NVFP4"
    records = dict(cache.metadata["render_scores"]["records"])
    assert records.pop(victim, None) is not None
    cache.metadata["render_scores"] = {
        **cache.metadata["render_scores"],
        "entries": len(records),
        "records": records,
    }

    with pytest.raises(ValueError, match="render-score records"):
        _write_manifest(bundle, cache, cache_dir)


def test_manifest_refuses_a_stale_requested_entries_counter(tmp_path):
    bundle, cache_dir, cache = _packed_shard(tmp_path)
    cache.metadata["requested_entries"] = 1  # the pre-packed dense count

    with pytest.raises(
        ValueError, match="requested_entries must equal its materialized"
    ):
        _write_manifest(bundle, cache, cache_dir)


def test_union_refuses_a_dropped_packed_render_gate(tmp_path):
    bundle, cache_dir, cache = _packed_shard(
        tmp_path, name="packed-shard", dense=False,
    )
    dense_manifest = _dense_shard(tmp_path)

    gates = dict(cache.metadata["render_gates"])
    records = [
        row for row in gates["records"] if row["qname"] not in ASSIGNMENT
    ]
    assert len(records) < len(gates["records"])
    gates["records"] = records
    gates["entries"] = len(records)
    cache.metadata["render_gates"] = gates
    tampered = _write_manifest(
        bundle, cache, cache_dir, shard_id="sparklina",
    )

    with pytest.raises(ValueError, match="render_gates coverage differs"):
        union.union_shard_manifests(
            [dense_manifest, tampered],
            assignment=FULL_ASSIGNMENT,
            output_bundle=tmp_path / "union-bad",
        )
