"""Fanning the anchor campaign out, and putting it back together.

The campaign is sharded per **fused anchor group**, because that is the
smallest scope whose measured values do not depend on what else the run priced.
Three properties make the split honest, and these tests pin them:

* a selection narrows the checkpoint identity to exactly the units it names, so
  two invocations over disjoint selections can never contend for one journal;
* the scope-wide answers a shard may not re-derive -- the calibration row
  counts, the activation maxima, the draw itself -- are read from a census and
  **checked** against what the shard saw, never adopted on trust;
* the merge reproduces a whole-scope table, and refuses a set of rows that does
  not describe one campaign: a mixed Hessian identity, a gap in the coverage,
  or a unit priced twice.
"""
import copy
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))


# ---------------------------------------------------------------------------
# The selection
# ---------------------------------------------------------------------------

def _selection(*groups):
    return {
        "schema": "prismaquant.tessera_campaign_units.v1",
        "groups": [{"key": key, "members": list(members)} for key, members in groups],
    }


def test_selection_is_checked_against_the_run_s_own_grouping():
    from prismaquant.tessera_campaign import select_anchor_groups

    resolved = {"g:attn": ["q", "k", "v"], "u:down": ["down"]}
    assert select_anchor_groups(
        _selection(("g:attn", ["k", "q", "v"])), resolved, where="w") == ["g:attn"]

    with pytest.raises(RuntimeError, match="does not contain"):
        select_anchor_groups(_selection(("g:mlp", ["w1"])), resolved, where="w")
    # A group whose membership differs here is not the group that was planned:
    # a shard that quietly measured the smaller set would leave the merged
    # table short of rows nothing reported.
    with pytest.raises(RuntimeError, match="has members"):
        select_anchor_groups(_selection(("g:attn", ["q", "k"])), resolved, where="w")
    with pytest.raises(RuntimeError, match="selected twice"):
        select_anchor_groups(
            _selection(("u:down", ["down"]), ("u:down", ["down"])), resolved, where="w")


def test_group_key_puts_fused_siblings_and_a_whole_stack_together():
    from prismaquant.tessera_campaign import anchor_group_key, resolve_anchor_groups

    class Profile:
        def fused_sibling_group(self, name):
            return "layer0.qkv" if name.endswith(("q_proj", "k_proj")) else None

    class Member:
        module_qname = "layers.0.experts"

    members = {"layers.0.experts.3.w1": Member()}
    assert anchor_group_key("layers.0.self_attn.q_proj", profile=Profile(),
                            expert_members={}) == "g:layer0.qkv"
    assert anchor_group_key("layers.0.mlp.down_proj", profile=Profile(),
                            expert_members={}) == "u:layers.0.mlp.down_proj"
    assert anchor_group_key("layers.0.experts.3.w1", profile=Profile(),
                            expert_members=members) == "s:layers.0.experts"

    groups = resolve_anchor_groups(
        ["layers.0.self_attn.k_proj", "layers.0.self_attn.q_proj",
         "layers.0.experts.3.w1"],
        profile=Profile(), expert_members=members)
    assert groups == {
        "g:layer0.qkv": ["layers.0.self_attn.k_proj", "layers.0.self_attn.q_proj"],
        "s:layers.0.experts": ["layers.0.experts.3.w1"],
    }


def test_the_checkpoint_identity_covers_only_the_selected_units(monkeypatch):
    """Disjoint selections are disjoint identities, so they cannot contend."""
    import argparse

    from prismaquant import tessera_campaign as tc

    class Api:
        @staticmethod
        def encoder_source_sha256():
            return "encoder"

        @staticmethod
        def tensor_identity(tensor):
            return {"id": tensor}

    monkeypatch.setattr(tc, "_checkpoint_identity_api", lambda: Api)
    monkeypatch.setattr(tc.th, "encoder_recipe", lambda: {"recipe": 1})
    monkeypatch.setattr(
        "prismaquant.production_weight_cache._production_cache_source_sha256",
        lambda: "package")

    weights = {"a": "wa", "b": "wb", "c": "wc"}
    common = dict(
        acts={n: None for n in weights}, hessians={n: None for n in weights},
        menus={n: [] for n in weights},
        args=argparse.Namespace(model="m", layer_stride=1, out="o", cache_dir="c",
                                checkpoint="k", deadline_seconds=0.0, units="path",
                                calibration_census=None, census_out=None,
                                seed_checkpoint=None, seed_wire_dir=None),
        calibration_identity={"text_sha256": "t"}, serving_scope=None,
        static_scales={}, static_scale_policy="policy")

    whole = tc._campaign_checkpoint_identity(weights=weights, **common)
    left = tc._campaign_checkpoint_identity(
        weights={"a": "wa"}, **{**common, "acts": {"a": None},
                                "hessians": {"a": None}, "menus": {"a": []}})
    right = tc._campaign_checkpoint_identity(
        weights={"b": "wb", "c": "wc"},
        **{**common, "acts": {"b": None, "c": None},
           "hessians": {"b": None, "c": None}, "menus": {"b": [], "c": []}})

    assert set(whole["units"]) == {"a", "b", "c"}
    assert set(left["units"]) == {"a"}
    assert set(right["units"]) == {"b", "c"}
    assert not set(left["units"]) & set(right["units"])
    # The selection file's PATH is not bound: it is a location, and its content
    # is already bound by the units map above.
    assert "units" not in left["settings"]
    assert left["settings"] == right["settings"] == whole["settings"]


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

def _census(counts, maxima=None):
    maxima = maxima or {name: 1.0 for name in counts}
    return {"counts": dict(counts), "max_abs": dict(maxima)}


def test_census_supplies_the_scope_s_counts_and_verifies_the_run_s_own():
    from prismaquant.tessera_campaign import census_max_abs, census_token_counts

    census = _census({"a": 16384, "b": 16384, "e": 512})
    # Without a census a run reports what it saw; with one it reports the
    # scope's, which is what makes a shard's fit_tokens the monolith's.
    assert census_token_counts(None, {"a": 16384, "b": 16384}) == (16384, 16384)
    assert census_token_counts(census, {"a": 16384}) == (16384, 512)
    assert census_max_abs(census, {"a": 1.0}) == {"a": 1.0, "b": 1.0, "e": 1.0}

    with pytest.raises(RuntimeError, match="does not cover"):
        census_token_counts(census, {"z": 1})
    with pytest.raises(RuntimeError, match="disagrees"):
        census_token_counts(census, {"a": 8192})
    with pytest.raises(RuntimeError, match="disagrees"):
        census_max_abs(census, {"a": 2.0})


def test_a_census_of_another_draw_is_refused():
    from prismaquant.tessera_campaign import require_census_draw

    identity = {"text_sha256": "corpus", "fit_ids_sha256": "ids"}
    require_census_draw({"text_sha256": "corpus", "fit_ids_sha256": "ids"},
                        identity, where="w")
    with pytest.raises(RuntimeError, match="different draws"):
        require_census_draw({"text_sha256": "corpus", "fit_ids_sha256": "other"},
                            identity, where="w")


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------

HESSIAN = {
    "supplied": True, "text_sha": "ids", "token_count": 16384,
    "text_sha256": "corpus", "fit_ids_sha256": "ids", "fit_tokens": 16384,
    "capture_sha256": "shard-digest", "kwarg": ["hessian"], "applied": True,
}

SCOPE = {
    "dense_targets": ["a", "b"], "expert_targets": [], "dense_all": ["a", "b"],
    "pinned": [], "declared_stacks": {}, "packed_in_scope": {},
    "packed_outside_layer_stride": {},
    "anchor_groups": {"u:a": ["a"], "u:b": ["b"]},
    "calibration_census": {"counts": {"a": 16384, "b": 16384},
                           "token_count": 16384, "token_count_min": 16384},
}


def _shard(row_key, unit):
    return {
        "schema": "prismaquant.tessera_campaign_cost.v1",
        "currency": "output_mse_under_route_activation_contract",
        "costs": {unit: {"E4M3_R512": {"output_mse": 1.5, "output_mse_measured": True,
                                       "hessian_identity": dict(HESSIAN)}}},
        "formats": ["E4M3_R512"],
        "leave_one_anchor_out": {unit: {"e4m3": {"max_abs_log2_error": 0.1}}},
        "non_interpolable": [],
        "menu_sizes": {unit: 4},
        "anchor_counts": {unit: {"e4m3": 3}},
        "provenance": {
            "menu_mode": "research", "tp_degree": 1, "model": "m", "nsamples": 32,
            "seqlen": 512, "max_act_rows": 512, "layer_stride": 1000,
            "anchors_round_one": 3, "max_rounds": 0, "anchor_budget": 12,
            "loo_gate": 0.25, "max_artifact_bpp": 8.0, "cost_mode": "render-score",
            "rounds_run": 2, "stopped_early": False, "wall_seconds": 10.0,
            "surfaces": {unit: {"e4m3": {"anchors": 3}}},
            "anchor_groups": {row_key: [unit]},
            "seed_checkpoint": None,
            "unit_selection": {"schema": "prismaquant.tessera_campaign_units.v1",
                               "selected": True,
                               "groups": [{"key": row_key, "members": [unit]}]},
            "campaign_scope": copy.deepcopy(SCOPE),
            "activation_static_scales": {"policy": "p", "path": None,
                                         "units": {"a": 1.0, "b": 2.0}},
            "hessian": {"supplied": True, "calibration_identity": {"text_sha256": "corpus"},
                        "capture_sha256": "shard-digest"},
        },
    }


def _payloads():
    return {"row-0000": _shard("u:a", "a"), "row-0001": _shard("u:b", "b")}


def test_merge_reproduces_a_whole_scope_table():
    import dispatch_tessera_campaign as dispatch

    merged = dispatch.merge_payloads(
        _payloads(), census={"counts": {"a": 16384, "b": 16384}},
        capture_sha256="merged-digest")

    assert sorted(merged["costs"]) == ["a", "b"]
    assert merged["formats"] == ["E4M3_R512"]
    assert sorted(merged["leave_one_anchor_out"]) == ["a", "b"]
    assert sorted(merged["provenance"]["surfaces"]) == ["a", "b"]
    # One Hessian identity, and it is the merged capture's -- the digest of the
    # union, not either shard's digest of its own half.
    digests = {row["hessian_identity"]["capture_sha256"]
               for rows in merged["costs"].values() for row in rows.values()}
    assert digests == {"merged-digest"}
    assert merged["provenance"]["hessian"]["capture_sha256"] == "merged-digest"
    # The merged table claims the whole scope, not a shard's selection.
    assert merged["provenance"]["unit_selection"]["selected"] is False
    assert merged["provenance"]["tessera_population"]["priced"]["dense"] == ["a", "b"]
    assert merged["provenance"]["campaign_fanout"]["rows"] == {
        "row-0000": ["u:a"], "row-0001": ["u:b"]}


def test_merge_refuses_a_mixed_hessian_identity():
    import dispatch_tessera_campaign as dispatch

    payloads = _payloads()
    row = payloads["row-0001"]["costs"]["b"]["E4M3_R512"]
    row["hessian_identity"] = {**row["hessian_identity"], "fit_ids_sha256": "other"}
    with pytest.raises(dispatch.MergeRefused, match="fit_ids_sha256"):
        dispatch.merge_payloads(payloads, census={"counts": {}},
                                capture_sha256="merged-digest")


def test_merge_refuses_a_gap_and_a_double_price():
    import dispatch_tessera_campaign as dispatch

    payloads = _payloads()
    del payloads["row-0001"]
    with pytest.raises(dispatch.MergeRefused, match="do not cover"):
        dispatch.merge_payloads(payloads, census={"counts": {}},
                                capture_sha256="merged-digest")

    payloads = _payloads()
    payloads["row-0001"]["provenance"]["unit_selection"]["groups"] = [
        {"key": "u:a", "members": ["a"]}]
    with pytest.raises(dispatch.MergeRefused, match="priced by both"):
        dispatch.merge_payloads(payloads, census={"counts": {}},
                                capture_sha256="merged-digest")


def test_merge_refuses_rows_from_two_calibrations():
    import dispatch_tessera_campaign as dispatch

    payloads = _payloads()
    payloads["row-0001"]["provenance"]["nsamples"] = 64
    with pytest.raises(dispatch.MergeRefused, match="provenance.nsamples"):
        dispatch.merge_payloads(payloads, census={"counts": {}},
                                capture_sha256="merged-digest")


def test_a_row_may_not_carry_the_round_loop_s_own_deadline(tmp_path):
    import dispatch_tessera_campaign as dispatch

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "model": "m", "cwd": "/tmp", "python": "py", "env": {},
        "campaign_argv": ["--menu-mode", "research", "--deadline-seconds", "100"]}))
    with pytest.raises(RuntimeError, match="deadline"):
        dispatch.load_spec(spec)
