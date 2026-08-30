"""Unit sharding: partition, env parsing, merge gate, sentinel compare."""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismaquant import unit_sharding as us  # noqa: E402
from prismaquant.production_weight_cache import ProductionWeightCache  # noqa: E402


def _load_tool(name: str):
    path = REPO_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge_tool = _load_tool("merge_unit_shards")
sentinel_tool = _load_tool("cluster_render_sentinel")


LEAVES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def make_units(n_layers: int = 8, base: int = 1_000_000):
    units = [("model.embed_projection", 4096)]
    for layer in range(n_layers):
        for leaf in LEAVES:
            units.append((f"model.layers.{layer}.{leaf}", base + layer * 1000))
    return units


# --------------------------------------------------------------------------
# env parsing
# --------------------------------------------------------------------------
def test_parse_shard_spec_accepts_well_formed():
    spec = us.parse_shard_spec("0/2")
    assert (spec.index, spec.count, spec.label) == (0, 2, "0/2")
    assert us.parse_shard_spec(" 3/4 ").index == 3
    assert us.parse_shard_spec("0/1").count == 1


def test_parse_shard_spec_unset_is_no_shard():
    assert us.parse_shard_spec(None) is None
    assert us.parse_shard_spec("") is None
    assert us.parse_shard_spec("   ") is None


@pytest.mark.parametrize(
    "value",
    ["2/2", "3/2", "1", "a/b", "1/0", "-1/2", "1/2/3", "1 / 2", "0/-2", "0.5/2"],
)
def test_parse_shard_spec_rejects_malformed(value):
    with pytest.raises(ValueError):
        us.parse_shard_spec(value)


def test_resolve_shard_spec_reads_the_env(monkeypatch):
    monkeypatch.delenv(us.SHARD_ENV, raising=False)
    assert us.resolve_shard_spec() is None
    monkeypatch.setenv(us.SHARD_ENV, "1/3")
    assert us.resolve_shard_spec().label == "1/3"


# --------------------------------------------------------------------------
# partition
# --------------------------------------------------------------------------
def test_partition_is_total_and_disjoint():
    units = make_units()
    partition = us.partition_units(units, 3)
    flat = [name for shard in partition.shards for name in shard]
    assert sorted(flat) == sorted(name for name, _ in units)
    assert len(flat) == len(set(flat))


def test_partition_is_deterministic_and_pure():
    units = make_units()
    first = us.partition_units(units, 4)
    second = us.partition_units(list(units), 4)
    assert first.shards == second.shards
    assert first.partition_hash == second.partition_hash


def test_partition_hash_tracks_inputs():
    units = make_units()
    baseline = us.partition_units(units, 2).partition_hash
    assert us.partition_units(units, 3).partition_hash != baseline
    perturbed = list(units)
    perturbed[5] = (perturbed[5][0], perturbed[5][1] + 1)
    assert us.partition_units(perturbed, 2).partition_hash != baseline


def test_partition_keeps_layers_contiguous_and_whole():
    units = make_units(n_layers=9)
    partition = us.partition_units(units, 3)
    for shard in partition.shards:
        atoms = [us.atom_key(name) for name in shard]
        # each atom appears as one contiguous run
        assert atoms == sorted(atoms, key=lambda a: atoms.index(a))
    seen: dict[str, int] = {}
    for index, shard in enumerate(partition.shards):
        for name in shard:
            key = us.atom_key(name)
            assert seen.setdefault(key, index) == index, (
                f"atom {key} straddles shards"
            )


def test_partition_balances_by_exact_bytes():
    units = make_units(n_layers=8)
    partition = us.partition_units(units, 2)
    total = sum(nbytes for _, nbytes in units)
    assert sum(partition.shard_bytes) == total
    # Layer atoms are equal-ish, so a two-way split lands within one atom.
    atom_bytes = max(
        sum(n for name, n in units if us.atom_key(name) == key)
        for key in {us.atom_key(name) for name, _ in units}
    )
    assert abs(partition.shard_bytes[0] - partition.shard_bytes[1]) <= atom_bytes


def test_partition_minimizes_the_heaviest_shard():
    # Deliberately lopsided: an exact DP must not just cut in the middle.
    units = [
        ("model.layers.0.mlp.down_proj", 10),
        ("model.layers.1.mlp.down_proj", 10),
        ("model.layers.2.mlp.down_proj", 100),
        ("model.layers.3.mlp.down_proj", 10),
    ]
    partition = us.partition_units(units, 2)
    assert max(partition.shard_bytes) == 110
    assert partition.shards[0] == (
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
    )


def test_partition_allows_more_shards_than_atoms():
    units = [("model.layers.0.mlp.down_proj", 10)]
    partition = us.partition_units(units, 3)
    assert sum(len(shard) for shard in partition.shards) == 1


def test_partition_rejects_non_contiguous_enumeration():
    units = [
        ("model.layers.0.mlp.down_proj", 10),
        ("model.layers.1.mlp.down_proj", 10),
        ("model.layers.0.mlp.up_proj", 10),
    ]
    with pytest.raises(ValueError, match="not layer-contiguous"):
        us.partition_units(units, 2)


def test_partition_rejects_duplicate_units():
    units = [("model.layers.0.mlp.down_proj", 10)] * 2
    with pytest.raises(ValueError, match="duplicate unit"):
        us.partition_units(units, 2)


def test_partition_rejects_bad_count():
    with pytest.raises(ValueError):
        us.partition_units(make_units(), 0)


def test_atom_key_groups_experts_into_their_layer():
    assert us.atom_key("model.layers.3.mlp.experts.5.w1") == "model.layers.3"
    assert us.atom_key("model.language_model.layers.2.mlp.up_proj") == (
        "model.language_model.layers.2"
    )
    assert us.atom_key("lm_head") == "lm_head"


def test_assert_groups_within_atoms_accepts_fused_siblings():
    units = [name for name, _ in make_units(n_layers=3)]
    us.assert_groups_within_atoms(
        units,
        lambda name: name.rsplit(".", 1)[0],
        where="test",
    )


def test_assert_groups_within_atoms_refuses_a_straddling_group():
    units = [name for name, _ in make_units(n_layers=3)]
    with pytest.raises(ValueError, match="straddle layer atoms"):
        us.assert_groups_within_atoms(
            units,
            lambda name: name.rsplit(".", 1)[-1],  # groups q_proj across layers
            where="test",
        )


def test_shard_stamp_round_trips_through_partition_from_stamp():
    units = make_units()
    partition = us.partition_units(units, 3)
    stamp = us.shard_stamp(partition, us.ShardSpec(1, 3))
    assert stamp["shard"] == "1/3"
    assert list(stamp["unit_names"]) == list(partition.shards[1])
    rebuilt = us.partition_from_stamp(stamp)
    assert rebuilt.shards == partition.shards
    assert rebuilt.partition_hash == partition.partition_hash


def test_partition_from_stamp_refuses_a_tampered_stamp():
    partition = us.partition_units(make_units(), 2)
    stamp = us.shard_stamp(partition, us.ShardSpec(0, 2))
    stamp["all_units"][3][1] += 1
    with pytest.raises(ValueError, match="partition_hash does not match"):
        us.partition_from_stamp(stamp)


# --------------------------------------------------------------------------
# merge completeness gate (A3)
# --------------------------------------------------------------------------
def _shard_cache(tmp_path, partition, index, *, formats=("NVFP4",), drop=(),
                 extra=(), duplicate_of=None, stamp_owed=True, gates=None):
    spec = us.ShardSpec(index, partition.count)
    stamp = {**us.shard_stamp(partition, spec), "host": {"hostname": "test"}}
    if stamp_owed:
        # A real shard stamps the debt it set out to render, before the loop
        # runs — so `drop` below leaves the stamp intact and the gate sees it.
        stamp.update(us.owed_pairs_stamp(
            (name, fmt)
            for name in partition.shards[index]
            for fmt in formats
        ))
    names = list(partition.shards[index])
    if duplicate_of is not None:
        names = names + list(partition.shards[duplicate_of])
    weights = {}
    scores = {}
    for name in names:
        if name in drop:
            continue
        for fmt in formats:
            weights[(name, fmt)] = torch.zeros(2, 2, dtype=torch.bfloat16)
            scores[f"{name}|{fmt}"] = {"qname": name, "format": fmt}
    for name in extra:
        for fmt in formats:
            weights[(name, fmt)] = torch.zeros(2, 2, dtype=torch.bfloat16)
    cache = ProductionWeightCache(
        weights=weights,
        levers={"gptq": True},
        activation_max_abs={name: 1.0 for name, _ in partition.units},
        failed={},
        cache_dir=None,
        metadata={
            "unit_shard": stamp,
            "render_scope": "format-menu",
            "requested_formats": list(formats),
            "render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "entries": len(scores),
                "records": scores,
            },
            **({"render_gates": gates} if gates is not None else {}),
        },
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"shard{index}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(cache, fh)
    return path


def _merge(tmp_path, shard_paths):
    return merge_tool.main([
        "merge",
        *[arg for path in shard_paths for arg in ("--shard", str(path))],
        "--output", str(tmp_path / "merged.pkl"),
    ])


def test_merge_accepts_a_complete_partition(tmp_path):
    partition = us.partition_units(make_units(n_layers=4), 2)
    paths = [_shard_cache(tmp_path, partition, i) for i in range(2)]
    assert _merge(tmp_path, paths) == 0
    with open(tmp_path / "merged.pkl", "rb") as fh:
        merged = pickle.load(fh)
    assert len(merged.weights) == len(partition.units)
    stamp = merged.metadata["unit_shard_merge"]
    assert stamp["shard_count"] == 2
    assert [entry["shard"] for entry in stamp["shards"]] == ["0/2", "1/2"]
    assert "unit_shard" not in merged.metadata
    # canonical order: full-enumeration unit order, not shard-append order
    assert [key[0] for key in merged.weights] == [
        name for name, _ in partition.units
    ]


def test_merge_refuses_a_missing_unit(tmp_path):
    partition = us.partition_units(make_units(n_layers=4), 2)
    victim = partition.shards[1][2]
    paths = [
        _shard_cache(tmp_path, partition, 0),
        _shard_cache(tmp_path, partition, 1, drop=(victim,)),
    ]
    with pytest.raises(SystemExit) as excinfo:
        _merge(tmp_path, paths)
    assert "missing entries" in str(excinfo.value)
    assert victim in str(excinfo.value)


def test_merge_refuses_a_duplicate_unit(tmp_path):
    partition = us.partition_units(make_units(n_layers=4), 2)
    paths = [
        _shard_cache(tmp_path, partition, 0),
        _shard_cache(tmp_path, partition, 1, duplicate_of=0),
    ]
    with pytest.raises(SystemExit) as excinfo:
        _merge(tmp_path, paths)
    assert "outside their shard" in str(excinfo.value)


def test_merge_refuses_an_extra_unit(tmp_path):
    partition = us.partition_units(make_units(n_layers=4), 2)
    paths = [
        _shard_cache(tmp_path, partition, 0, extra=("model.layers.99.mlp.up_proj",)),
        _shard_cache(tmp_path, partition, 1),
    ]
    with pytest.raises(SystemExit) as excinfo:
        _merge(tmp_path, paths)
    assert "outside their shard" in str(excinfo.value)


def test_merge_refuses_a_missing_shard(tmp_path):
    partition = us.partition_units(make_units(n_layers=4), 3)
    paths = [_shard_cache(tmp_path, partition, i) for i in (0, 1)]
    with pytest.raises(SystemExit) as excinfo:
        _merge(tmp_path, paths)
    assert "declares 3 shards" in str(excinfo.value)


def test_merge_refuses_shards_of_different_partitions(tmp_path):
    partition_a = us.partition_units(make_units(n_layers=4), 2)
    partition_b = us.partition_units(make_units(n_layers=5), 2)
    paths = [
        _shard_cache(tmp_path, partition_a, 0),
        _shard_cache(tmp_path / "b", partition_b, 1),
    ]
    with pytest.raises(SystemExit) as excinfo:
        _merge(tmp_path, paths)
    assert "partition_hash differs" in str(excinfo.value)


def test_merge_refuses_activation_max_abs_disagreement(tmp_path):
    partition = us.partition_units(make_units(n_layers=4), 2)
    paths = [_shard_cache(tmp_path, partition, i) for i in range(2)]
    with open(paths[1], "rb") as fh:
        cache = pickle.load(fh)
    victim = partition.units[0][0]
    cache.activation_max_abs[victim] = 2.0
    with open(paths[1], "wb") as fh:
        pickle.dump(cache, fh)
    with pytest.raises(SystemExit) as excinfo:
        _merge(tmp_path, paths)
    assert "activation_max_abs disagreements" in str(excinfo.value)


def test_merge_refuses_an_unsharded_cache(tmp_path):
    cache = ProductionWeightCache(weights={}, levers={}, metadata={})
    path = tmp_path / "plain.pkl"
    with open(path, "wb") as fh:
        pickle.dump(cache, fh)
    with pytest.raises(SystemExit) as excinfo:
        _merge(tmp_path, [path, path])
    assert "no 'unit_shard' stamp" in str(excinfo.value)


# --------------------------------------------------------------------------
# the shard states its own debt
# --------------------------------------------------------------------------
def test_owed_pairs_stamp_round_trips_and_is_order_insensitive():
    pairs = [("a", "NVFP4"), ("b", "fp8_e4m3"), ("a", "FP8_E4M3")]
    stamp = us.owed_pairs_stamp(pairs)
    assert stamp["owed_pair_count"] == 3
    assert us.owed_pairs_from_stamp(stamp) == {
        ("a", "NVFP4"), ("a", "FP8_E4M3"), ("b", "FP8_E4M3"),
    }
    assert us.owed_pairs_stamp(reversed(pairs)) == stamp


def test_owed_pairs_from_stamp_refuses_a_tampered_stamp():
    stamp = us.owed_pairs_stamp([("a", "NVFP4"), ("b", "NVFP4")])
    stamp["owed_pairs"] = [["a", "NVFP4"]]
    with pytest.raises(ValueError, match="owed_pairs_sha256"):
        us.owed_pairs_from_stamp(stamp)


def test_owed_pairs_from_stamp_is_none_when_the_field_is_absent():
    assert us.owed_pairs_from_stamp({"shard": "0/2"}) is None


def test_merge_gate_trusts_the_stamped_debt_over_an_operator_config(tmp_path):
    """A layer config that under-declares must not excuse a dropped unit.

    The operator supplies `--render-layer-config`; the shard supplies its own
    owed list. If the config wins, a config that calls the dropped unit BF16
    makes a silently incomplete merge pass — the exact failure the gate exists
    to prevent.
    """
    partition = us.partition_units(make_units(n_layers=4), 2)
    victim = partition.shards[1][2]
    paths = [
        _shard_cache(tmp_path, partition, 0),
        _shard_cache(tmp_path, partition, 1, drop=(victim,)),
    ]
    config = tmp_path / "layer_config.json"
    config.write_text(json.dumps(
        {name: ("BF16" if name == victim else "NVFP4")
         for name, _ in partition.units}
    ))
    with pytest.raises(SystemExit) as excinfo:
        merge_tool.main([
            "merge",
            *[arg for path in paths for arg in ("--shard", str(path))],
            "--output", str(tmp_path / "merged.pkl"),
            "--render-layer-config", str(config),
        ])
    assert "missing entries" in str(excinfo.value)
    assert victim in str(excinfo.value)


def test_merge_falls_back_to_reconstruction_for_a_legacy_stamp(tmp_path):
    partition = us.partition_units(make_units(n_layers=4), 2)
    victim = partition.shards[0][1]
    paths = [
        _shard_cache(tmp_path, partition, 0, drop=(victim,), stamp_owed=False),
        _shard_cache(tmp_path, partition, 1, stamp_owed=False),
    ]
    with pytest.raises(SystemExit) as excinfo:
        _merge(tmp_path, paths)
    assert "missing entries" in str(excinfo.value)
    assert victim in str(excinfo.value)


def test_merge_orders_keys_like_an_unsharded_run(tmp_path):
    """Unit order, then REQUESTED-format order — not alphabetical."""
    partition = us.partition_units(make_units(n_layers=4), 2)
    formats = ("NVFP4", "FP8_E4M3")
    paths = [
        _shard_cache(tmp_path, partition, i, formats=formats)
        for i in range(2)
    ]
    assert _merge(tmp_path, paths) == 0
    with open(tmp_path / "merged.pkl", "rb") as fh:
        merged = pickle.load(fh)
    expected = [
        (name, fmt) for name, _ in partition.units for fmt in formats
    ]
    assert list(merged.weights) == expected


def test_merged_sidecars_are_readable_by_the_production_loader(tmp_path):
    """A merged cache_dir must resume exactly like an unsharded one."""
    from prismaquant.production_weight_cache import _load_render_score_sidecar

    partition = us.partition_units(make_units(n_layers=4), 2)
    paths = [_shard_cache(tmp_path, partition, i) for i in range(2)]
    out_dir = tmp_path / "merged_cache_dir"
    assert merge_tool.main([
        "merge",
        *[arg for path in paths for arg in ("--shard", str(path))],
        "--output", str(tmp_path / "merged.pkl"),
        "--output-cache-dir", str(out_dir),
    ]) == 0
    records = _load_render_score_sidecar(out_dir / "render_scores.json")
    assert len(records) == len(partition.units)
    assert all(
        f"{name}|NVFP4" in records for name, _ in partition.units
    )
    max_abs = json.loads((out_dir / "activation_max_abs.json").read_text())
    # enumeration order, as the unsharded stage writes it
    assert list(max_abs) == [name for name, _ in partition.units]


def test_merge_unions_the_render_gate_summary(tmp_path):
    """Merged counters cover every shard, not shard 0's tally relabelled."""
    def gates_for(index):
        records = [
            {
                "qname": name,
                "format": "NVFP4",
                "trace": [{
                    "mechanism": "gptq",
                    "accepted": True,
                    "reason": "improved",
                    "package": ["gptq"],
                }],
            }
            for name in partition.shards[index]
        ]
        return {"enabled": True, "entries": len(records),
                "mechanisms": {"gptq": {
                    "accepted": len(records), "rejected": 0,
                    "reasons": {"improved": len(records)},
                    "package_accepted": len(records),
                }},
                "records": records}

    partition = us.partition_units(make_units(n_layers=4), 2)
    paths = [
        _shard_cache(tmp_path, partition, i, gates=gates_for(i))
        for i in range(2)
    ]
    assert _merge(tmp_path, paths) == 0
    with open(tmp_path / "merged.pkl", "rb") as fh:
        merged = pickle.load(fh)
    total = len(partition.units)
    gates = merged.metadata["render_gates"]
    assert gates["entries"] == total
    assert len(gates["records"]) == total
    assert gates["mechanisms"]["gptq"]["accepted"] == total
    assert gates["mechanisms"]["gptq"]["reasons"]["improved"] == total
    # records land in canonical enumeration order, not shard-append order
    assert [rec["qname"] for rec in gates["records"]] == [
        name for name, _ in partition.units
    ]


# --------------------------------------------------------------------------
# sentinel
# --------------------------------------------------------------------------
def test_select_sentinel_units_is_deterministic_and_spread():
    names = [f"model.layers.{i}.mlp.down_proj" for i in range(40)]
    picked = sentinel_tool.select_sentinel_units(names, 5)
    assert picked == sentinel_tool.select_sentinel_units(names, 5)
    assert len(picked) == 5
    assert picked[0] == names[0]
    assert picked[-1] != names[-1]  # evenly spaced, not the tail
    assert len(set(picked)) == len(picked)


def test_select_sentinel_units_clamps_to_the_enumeration():
    names = ["a", "b"]
    assert sentinel_tool.select_sentinel_units(names, 10) == names


def _manifest(entries, deterministic="1", host="boxA"):
    return {
        "schema": sentinel_tool.SCHEMA,
        "formats": ["NVFP4"],
        "levers": ["gptq"],
        "units": sorted({key.split("|")[0] for key in entries}),
        "k": 2,
        "calibration": {"calib_hash": "abc"},
        "deterministic_mode": deterministic,
        "host": {"hostname": host},
        "entries": entries,
    }


def test_sentinel_compare_accepts_matching_manifests(tmp_path):
    entries = {"a|NVFP4": {"sha256": "aa", "dtype": "torch.bfloat16",
                           "shape": [2, 2]}}
    paths = []
    for name, host in (("a.json", "boxA"), ("b.json", "boxB")):
        path = tmp_path / name
        path.write_text(json.dumps(_manifest(entries, host=host)))
        paths.append(str(path))
    assert sentinel_tool.main(
        ["compare", "--manifest", paths[0], "--manifest", paths[1]]
    ) == 0


def test_sentinel_compare_flags_a_differing_unit(tmp_path):
    a = _manifest({"a|NVFP4": {"sha256": "aa"}})
    b = _manifest({"a|NVFP4": {"sha256": "bb"}}, host="boxB")
    problems = sentinel_tool.compare_manifests([a, b])
    assert any("rendered units differ" in p for p in problems)


def test_sentinel_compare_flags_a_missing_unit():
    a = _manifest({"a|NVFP4": {"sha256": "aa"}, "b|NVFP4": {"sha256": "bb"}})
    b = _manifest({"a|NVFP4": {"sha256": "aa"}}, host="boxB")
    problems = sentinel_tool.compare_manifests([a, b])
    assert any("only in the first manifest" in p for p in problems)


def test_sentinel_compare_flags_an_incomplete_manifest():
    # A box that could not render a format at all (2026-08-23: no python3-dev
    # on sparklina → Triton build failure → every NVFP4 entry absent) must be
    # named as incomplete, not just "missing units".
    complete = _manifest({"a|NVFP4": {"sha256": "aa"}, "b|NVFP4": {"sha256": "bb"}})
    partial = dict(complete, host={"hostname": "boxB"},
                   entries={"a|NVFP4": {"sha256": "aa"}})
    assert sentinel_tool.missing_entries(partial) == ["b|NVFP4"]
    assert sentinel_tool.missing_entries(complete) == []
    # The CLI alias is promised; the cache keys by the canonical name.
    aliased = dict(complete, formats=["FP8_DYNAMIC"],
                   entries={"a|FP8_E4M3": {"sha256": "aa"},
                            "b|FP8_E4M3": {"sha256": "bb"}})
    assert sentinel_tool.missing_entries(aliased) == []
    problems = sentinel_tool.compare_manifests([complete, partial])
    assert any("boxB is incomplete" in p and "b|NVFP4" in p for p in problems)


def test_sentinel_compare_refuses_a_nondeterministic_match(tmp_path):
    entries = {"a|NVFP4": {"sha256": "aa"}}
    paths = []
    for name, host in (("a.json", "boxA"), ("b.json", "boxB")):
        path = tmp_path / name
        path.write_text(
            json.dumps(_manifest(entries, deterministic="0", host=host))
        )
        paths.append(str(path))
    assert sentinel_tool.main(
        ["compare", "--manifest", paths[0], "--manifest", paths[1]]
    ) == 1


def test_sentinel_compare_flags_a_deterministic_mode_disagreement():
    a = _manifest({"a|NVFP4": {"sha256": "aa"}}, deterministic="1")
    b = _manifest({"a|NVFP4": {"sha256": "aa"}}, deterministic="0")
    problems = sentinel_tool.compare_manifests([a, b])
    assert any("PRISMAQUANT_DETERMINISTIC differs" in p for p in problems)


# --------------------------------------------------------------------------
# streaming render path: shard filter at the dense-module enumeration seam
# --------------------------------------------------------------------------
class _StreamAttn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, h):
        return self.o_proj(self.q_proj(h) + self.k_proj(h) + self.v_proj(h))


class _StreamLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = _StreamAttn(hidden)

    def forward(self, h):
        self.self_attn(h)
        return h


class _StreamInner(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.layers = nn.ModuleList([_StreamLayer(hidden) for _ in range(layers)])

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


class _StreamModel(nn.Module):
    def __init__(self, hidden=16, layers=4):
        super().__init__()
        self.model = _StreamInner(hidden, layers)

    def forward(self, h, **kw):
        return self.model(h)


def _stream_render(tmp_path, model, act_dir, tag, unit_shard=None):
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.streaming_production_cache import run_streaming_render

    out_dir = tmp_path / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    dense = [
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
    ]
    assignment = {name: "NVFP4" for name in dense}
    return run_streaming_render(
        model,
        layers_prefix="model.layers.",
        num_layers=4,
        render_assignment=assignment,
        act_index=ActivationIndex(act_dir, []),
        formats=["NVFP4"],
        levers={"gptq": False, "static_act_order": False,
                "joint_scale_opt": False},
        cache_dir_path=out_dir,
        profile=None,
        skip_tokens=[],
        device=torch.device("cpu"),
        expert_render_mode="batched",
        progress=False,
        unit_shard=unit_shard,
    ), out_dir


def test_streaming_shards_merge_to_the_unsharded_render(tmp_path):
    import re

    torch.manual_seed(0)
    model = _StreamModel().to(dtype=torch.bfloat16).eval()
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    rows = torch.randn(32, 16, dtype=torch.bfloat16)
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        fname = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
        torch.save({"inputs": rows.float(), "name": name}, act_dir / fname)

    full, full_dir = _stream_render(tmp_path, model, act_dir, "full")
    shard0, dir0 = _stream_render(
        tmp_path, model, act_dir, "s0", us.ShardSpec(0, 2)
    )
    shard1, dir1 = _stream_render(
        tmp_path, model, act_dir, "s1", us.ShardSpec(1, 2)
    )

    assert set(shard0.weights) | set(shard1.weights) == set(full.weights)
    assert not (set(shard0.weights) & set(shard1.weights))
    assert shard0.metadata["unit_shard"]["shard"] == "0/2"
    assert (
        shard0.metadata["unit_shard"]["partition_hash"]
        == shard1.metadata["unit_shard"]["partition_hash"]
    )
    # layer atomicity: no layer appears in both shards
    atoms0 = {us.atom_key(q) for q, _ in shard0.weights}
    atoms1 = {us.atom_key(q) for q, _ in shard1.weights}
    assert not (atoms0 & atoms1)

    from prismaquant.production_weight_cache import _cache_weight_filename

    for cache, cache_dir in ((shard0, dir0), (shard1, dir1)):
        for key in cache.weights:
            got = torch.load(
                cache_dir / _cache_weight_filename(*key),
                map_location="cpu", weights_only=True,
            )
            want = torch.load(
                full_dir / _cache_weight_filename(*key),
                map_location="cpu", weights_only=True,
            )
            assert torch.equal(got, want), f"{key} differs from the full render"


# --------------------------------------------------------------------------
# the reservoir coupling a unit shard depends on
# --------------------------------------------------------------------------
class _ResAttn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class _ResModel(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.layers = nn.ModuleList([_ResAttn(hidden) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_stored_subset_keeps_the_full_runs_rows():
    """A run that STORES a subset must sample the same rows as the full run.

    One `torch.Generator` feeds every hooked Linear's priority reservoir, so
    drawing only for stored Linears makes the surviving rows a function of
    which other Linears the run happened to store. That is exactly what a
    unit shard changes, and the rendered bytes follow the rows.
    """
    from prismaquant.production_weight_cache import _LinearActivationCollector

    torch.manual_seed(0)
    model = _ResModel().eval()
    names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]
    assert len(names) == 8
    subset = set(names[4:])
    batches = [torch.randn(1, 40, 64) for _ in range(3)]  # 120 rows > max_rows

    def collect(store):
        collector = _LinearActivationCollector(
            model, set(names), max_rows=16, store_qnames=store
        )
        collector.install()
        try:
            with torch.no_grad():
                for batch in batches:
                    model(batch)
        finally:
            collector.remove()
        return collector.collected()

    full = collect(set(names))
    partial = collect(subset)
    assert set(partial) == subset
    for name in sorted(subset):
        assert torch.equal(full[name], partial[name]), (
            f"{name} sampled different rows when only a subset was stored"
        )
