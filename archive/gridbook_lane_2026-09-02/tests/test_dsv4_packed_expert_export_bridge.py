"""DSv4-Flash per-expert source -> gridbook packed-stack export bridge.

Three seams, one artifact:

* NAMING — the DSv4 rename rules anchor on a following ``.``, so mapping a bare
  module qname and appending ``.weight`` afterwards skipped every shared- and
  routed-expert rule. Pinned against the REAL checkpoint index where one is
  mounted, and against the shape of the rules everywhere else.
* COLLAPSE — the allocator writes its layer_config expanded per tensor even
  though it decided each expert group atomically; gridbook names only stacks.
* NAMESPACE — the export base's skeleton-existence fallback silently demoted a
  packed stack (whose parent name never appears on disk) to the LIVE spelling,
  which gridbook resolves as no-scheme rather than rejecting.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant.export_nvfp4_cb import (
    _export_base_name,
    _source_module_name,
    _source_tensor_key,
)
from prismaquant.export_nvfp4_cb_streaming import (
    _collapse_per_expert_assignment,
    _packed_expert_col_weights,
    _plan_expert_stacks,
    export_nvfp4_cb_streaming as _export_streaming,
)
from prismaquant.model_profiles import detect_profile

REAL_SOURCE = Path("/home/rob/dq-runs/dsv4-flash-0731/source")
REAL_INDEX = REAL_SOURCE / "model.safetensors.index.json"
needs_real_source = pytest.mark.skipif(
    not REAL_INDEX.exists(),
    reason="DSv4-Flash checkpoint index is not mounted on this box",
)

# This module builds synthetic CB bodies on CPU and never serves them.
# Gridbook 0.9.1's v12 table names no CB cell on sm_121, so the route gate
# refuses these exports unless the artifact declares what it is.  See
# tests/cb_synthetic_target.py; the real sm_121 refusal stays asserted in
# tests/test_cb_route_status_gate.py.
pytestmark = pytest.mark.usefixtures("synthetic_cb_target")




def _dsv4_profile():
    return detect_profile(str(REAL_SOURCE))


def _sampled_recipe_qnames() -> list[str]:
    """One qname per DSv4 leaf CLASS the probe inventory contains."""
    names = [f"model.layers.0.self_attn.{leaf}"
             for leaf in ("wq_a", "wq_b", "wkv", "wo_b")]
    names += [f"model.layers.0.mlp.shared_experts.{leaf}"
              for leaf in ("gate_proj", "up_proj", "down_proj")]
    names += [f"model.layers.0.mlp.experts.{e}.{leaf}"
              for e in (0, 1, 255)
              for leaf in ("gate_proj", "up_proj", "down_proj")]
    return names


# ---------------------------------------------------------------------------
# NAMING — against the real checkpoint index
# ---------------------------------------------------------------------------

@needs_real_source
def test_every_sampled_recipe_leaf_resolves_in_the_real_checkpoint():
    keys = set(json.loads(REAL_INDEX.read_text())["weight_map"])
    profile = _dsv4_profile()
    sample = _sampled_recipe_qnames()
    unresolved = [q for q in sample
                  if _source_tensor_key(q, profile, ".weight") not in keys]
    assert unresolved == [], (
        f"{len(sample) - len(unresolved)}/{len(sample)} resolved; the naming "
        f"bridge still misses {unresolved}")


@needs_real_source
def test_full_probe_inventory_resolves_in_the_real_checkpoint():
    """33,325 selectable Linears, every one of them, or the export cannot run.

    33,024 routed-expert projections + 43 x (4 attention + 3 shared-expert).
    """
    col_weights_path = Path(
        "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6/artifacts/"
        "cb_col_weights.pkl")
    if not col_weights_path.exists():
        pytest.skip("production col-weights inventory is not mounted")
    import pickle

    with col_weights_path.open("rb") as fh:
        inventory = sorted(pickle.load(fh))
    keys = set(json.loads(REAL_INDEX.read_text())["weight_map"])
    profile = _dsv4_profile()
    missing = [q for q in inventory
               if _source_tensor_key(q, profile, ".weight") not in keys]
    assert len(inventory) == 33_325
    assert missing == [], (
        f"{len(inventory) - len(missing)}/{len(inventory)} resolvable; "
        f"missing e.g. {missing[:5]}")


@needs_real_source
def test_routed_and_shared_expert_renames_are_the_mixtral_convention():
    profile = _dsv4_profile()
    assert _source_module_name(
        "model.layers.0.mlp.shared_experts.gate_proj", profile
    ) == "layers.0.ffn.shared_experts.w1"
    assert _source_module_name(
        "model.layers.0.mlp.shared_experts.up_proj", profile
    ) == "layers.0.ffn.shared_experts.w3"
    assert _source_module_name(
        "model.layers.0.mlp.shared_experts.down_proj", profile
    ) == "layers.0.ffn.shared_experts.w2"
    assert _source_module_name(
        "model.layers.7.mlp.experts.42.gate_proj", profile
    ) == "layers.7.ffn.experts.42.w1"
    assert _source_module_name(
        "model.layers.7.mlp.experts.42.up_proj", profile
    ) == "layers.7.ffn.experts.42.w3"
    assert _source_module_name(
        "model.layers.7.mlp.experts.42.down_proj", profile
    ) == "layers.7.ffn.experts.42.w2"
    # The `.weight$`-anchored exact rules fire through the same convention.
    assert _source_module_name("lm_head", profile) == "head"


@needs_real_source
def test_packed_stack_export_base_keeps_the_ffn_checkpoint_namespace():
    """The packed parent never appears on disk, so the existence fallback has
    to be told the target IS resolved — else the manifest ships two
    namespaces and gridbook silently resolves the `mlp` half to no-scheme."""
    skeleton = set(json.loads(REAL_INDEX.read_text())["weight_map"])
    profile = _dsv4_profile()
    for leaf in ("gate_up_proj", "down_proj"):
        qname = f"model.layers.0.mlp.experts.{leaf}"
        assert f"layers.0.ffn.experts.{leaf}.weight" not in skeleton
        assert _export_base_name(qname, profile, skeleton) == qname, (
            "precondition: the bare fallback demotes to the live spelling")
        assert _export_base_name(
            qname, profile, skeleton, assume_resolvable=True
        ) == f"layers.0.ffn.experts.{leaf}"


@needs_real_source
def test_plan_expert_stacks_bridges_checkpoint_names_to_the_recipe():
    from prismaquant.export_nvfp4_cb_streaming import _LazySkeleton

    skeleton = _LazySkeleton(REAL_SOURCE)
    groups = _plan_expert_stacks(skeleton, _dsv4_profile())
    # 43 body layers + 1 MTP block, keyed by the LIVE prefix the recipe uses.
    assert "model.layers.0.mlp.experts" in groups
    layer0 = groups["model.layers.0.mlp.experts"]
    assert sorted(layer0) == ["down_proj", "gate_proj", "up_proj"]
    assert len(layer0["gate_proj"]) == 256
    # Members stay CHECKPOINT bases so `_expert_weight` can read them.
    assert layer0["gate_proj"][0] == "layers.0.ffn.experts.0.w1"
    assert layer0["up_proj"][255] == "layers.0.ffn.experts.255.w3"
    assert layer0["down_proj"][7] == "layers.0.ffn.experts.7.w2"


@needs_real_source
def test_mxfp4_routed_expert_logical_shape_is_the_decoded_shape():
    from prismaquant.export_nvfp4_cb_streaming import _LazySkeleton

    skeleton = _LazySkeleton(REAL_SOURCE)
    key = "layers.0.ffn.experts.0.w1.weight"
    assert skeleton.get_shape(key) == (2048, 2048)      # I8 nibble pack
    assert skeleton.logical_shape(key) == (2048, 4096)  # decoded
    # The block-FP8 shared expert and attention are stored at logical width.
    for plain in ("layers.0.ffn.shared_experts.w1.weight",
                  "layers.0.attn.wq_a.weight"):
        assert skeleton.logical_shape(plain) == skeleton.get_shape(plain)


# ---------------------------------------------------------------------------
# COLLAPSE — synthetic layer configs, no checkpoint required
# ---------------------------------------------------------------------------

class _StubProfile:
    """Minimal packed-expert profile: gate_up = gate then up, down alone."""

    name = "stub"

    def packed_expert_param_names(self):
        return ("gate_up_proj", "down_proj")

    def packed_expert_projection_names(self, packed):
        return (("gate_proj", "up_proj") if packed == "gate_up_proj"
                else ("down_proj",))


def _groups(prefix="model.layers.0.mlp.experts", n=4):
    return {prefix: {proj: {e: f"ckpt.{e}.{proj}" for e in range(n)}
                     for proj in ("gate_proj", "up_proj", "down_proj")}}


def _expanded(prefix="model.layers.0.mlp.experts", n=4, fmt="NVFP4_CB_K15"):
    return {f"{prefix}.{e}.{proj}": fmt
            for e in range(n)
            for proj in ("gate_proj", "up_proj", "down_proj")}


def test_collapse_reduces_per_expert_entries_to_two_stacks():
    prefix = "model.layers.0.mlp.experts"
    assignment = _expanded(prefix, n=4)
    assignment["model.layers.0.self_attn.wq_a"] = "FP8_CB_K36"
    collapsed, members, report = _collapse_per_expert_assignment(
        assignment, _groups(prefix, 4), _StubProfile())
    assert sorted(collapsed) == [
        f"{prefix}.down_proj", f"{prefix}.gate_up_proj",
        "model.layers.0.self_attn.wq_a"]
    assert collapsed[f"{prefix}.gate_up_proj"] == "NVFP4_CB_K15"
    assert report == {"stacks": 2, "members": 12}
    # gate_up carries both projections for every expert; down carries one.
    assert len(members[f"{prefix}.gate_up_proj"]) == 8
    assert len(members[f"{prefix}.down_proj"]) == 4
    assert members[f"{prefix}.gate_up_proj"][("gate_proj", 2)] == \
        f"{prefix}.2.gate_proj"


def test_collapse_refuses_a_stack_whose_members_disagree_on_format():
    prefix = "model.layers.0.mlp.experts"
    assignment = _expanded(prefix, n=4)
    assignment[f"{prefix}.3.up_proj"] = "NVFP4_CB_K14"
    with pytest.raises(ValueError, match="uniform within a layer"):
        _collapse_per_expert_assignment(
            assignment, _groups(prefix, 4), _StubProfile())


def test_collapse_refuses_a_partially_allocated_stack():
    prefix = "model.layers.0.mlp.experts"
    assignment = _expanded(prefix, n=4)
    del assignment[f"{prefix}.2.gate_proj"]
    with pytest.raises(ValueError, match="exported whole or not at all"):
        _collapse_per_expert_assignment(
            assignment, _groups(prefix, 4), _StubProfile())


def test_collapse_refuses_a_config_carrying_both_spellings():
    prefix = "model.layers.0.mlp.experts"
    assignment = _expanded(prefix, n=4)
    assignment[f"{prefix}.gate_up_proj"] = "NVFP4_CB_K15"
    with pytest.raises(ValueError, match="BOTH the packed stack"):
        _collapse_per_expert_assignment(
            assignment, _groups(prefix, 4), _StubProfile())


def test_collapse_is_a_no_op_on_an_already_packed_assignment():
    prefix = "model.layers.0.mlp.experts"
    assignment = {f"{prefix}.gate_up_proj": "NVFP4_CB_K15",
                  f"{prefix}.down_proj": "NVFP4_CB_K15"}
    collapsed, members, report = _collapse_per_expert_assignment(
        dict(assignment), _groups(prefix, 4), _StubProfile())
    assert collapsed == assignment
    assert members == {}
    assert report == {"stacks": 0, "members": 0}


def test_packed_col_weights_pool_the_fused_projections_per_expert():
    prefix = "model.layers.0.mlp.experts"
    _collapsed, members, _r = _collapse_per_expert_assignment(
        _expanded(prefix, n=3), _groups(prefix, 3), _StubProfile())
    torch.manual_seed(0)
    cw = {f"{prefix}.{e}.{proj}": torch.rand(8) + 0.1
          for e in range(3)
          for proj in ("gate_proj", "up_proj", "down_proj")}
    out = _packed_expert_col_weights(cw, members, _StubProfile())
    gu = out[f"{prefix}.gate_up_proj"]
    dn = out[f"{prefix}.down_proj"]
    assert gu.shape == (3, 1, 8) and dn.shape == (3, 1, 8)
    for e in range(3):
        expected = torch.stack([cw[f"{prefix}.{e}.gate_proj"],
                                cw[f"{prefix}.{e}.up_proj"]]).mean(0)
        assert torch.allclose(gu[e, 0], expected)
        assert torch.allclose(dn[e, 0], cw[f"{prefix}.{e}.down_proj"])
    # The per-expert entries survive — the render identity is keyed by them.
    assert all(name in out for name in cw)


def test_packed_col_weights_refuse_a_member_with_no_imatrix():
    prefix = "model.layers.0.mlp.experts"
    _collapsed, members, _r = _collapse_per_expert_assignment(
        _expanded(prefix, n=2), _groups(prefix, 2), _StubProfile())
    cw = {f"{prefix}.{e}.{proj}": torch.rand(8) + 0.1
          for e in range(2)
          for proj in ("gate_proj", "up_proj", "down_proj")}
    del cw[f"{prefix}.1.up_proj"]
    with pytest.raises(ValueError, match="no silent RTN"):
        _packed_expert_col_weights(cw, members, _StubProfile())


# ---------------------------------------------------------------------------
# NAMESPACE — a whole synthetic export, manifest uniformity asserted
# ---------------------------------------------------------------------------

def _write_per_expert_model(mdl: Path, E=3, inter=256, hid=256, seed=3):
    mdl.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tensors = {}
    for e in range(E):
        for proj, shape in (("gate_proj", (inter, hid)),
                            ("up_proj", (inter, hid)),
                            ("down_proj", (hid, inter))):
            tensors[f"model.layers.1.mlp.experts.{e}.{proj}.weight"] = (
                torch.randn(*shape) * 0.3).to(torch.bfloat16)
    tensors["model.norm.weight"] = torch.ones(hid, dtype=torch.bfloat16)
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "mixtral", "architectures": ["MixtralForCausalLM"],
        "hidden_size": hid, "num_hidden_layers": 2,
        "torch_dtype": "bfloat16"}))
    return tensors


def _assignment_file(path: Path, mapping: dict) -> Path:
    path.write_text(json.dumps(mapping))
    return path


def test_expanded_per_expert_recipe_exports_as_packed_stacks(tmp_path):
    """End-to-end: an EXPANDED recipe produces stacked manifest names only."""
    E, inter, hid = 3, 256, 256
    mdl = tmp_path / "src"
    _write_per_expert_model(mdl, E, inter, hid)
    prefix = "model.layers.1.mlp.experts"
    cfg = _assignment_file(tmp_path / "a.json", {
        f"{prefix}.{e}.{proj}": {"data_type": "nvfp4_cb", "cb_k": 16}
        for e in range(E)
        for proj in ("gate_proj", "up_proj", "down_proj")})
    torch.manual_seed(11)
    cw = {f"{prefix}.{e}.{proj}": torch.rand(hid if proj != "down_proj"
                                             else inter) + 0.05
          for e in range(E)
          for proj in ("gate_proj", "up_proj", "down_proj")}
    _export_streaming(mdl, cfg, tmp_path / "out", cw, device="cpu",
                      allow_unstamped_research=True)
    names = set(load_file(str(tmp_path / "out" / "model.safetensors")))
    assert f"{prefix}.gate_up_proj.cb_qweight" in names
    assert f"{prefix}.down_proj.cb_qweight" in names
    # No per-expert CB tensor survives: gridbook's loader has no name for one.
    assert not [n for n in names if ".experts." in n and n.split(
        ".experts.")[1].split(".")[0].isdigit() and "cb_" in n]
    qc = json.loads((tmp_path / "out" / "quant_config.json").read_text())
    targets = sorted(t for g in qc["config_groups"].values()
                     for t in g.get("targets", []))
    assert targets == [f"{prefix}.down_proj", f"{prefix}.gate_up_proj"]


def test_manifest_namespace_is_uniform_across_cb_targets(tmp_path):
    """Every CB target and every emitted CB tensor must live in ONE namespace.

    A manifest that mixes the checkpoint spelling with the live spelling does
    not fail at load: gridbook resolves the un-mapped half to no-scheme and
    serves it unquantized (gridbook config.py's final
    `return UnquantizedLinearMethod()`), so the producer has to assert it.
    """
    E, inter, hid = 2, 256, 256
    mdl = tmp_path / "src"
    _write_per_expert_model(mdl, E, inter, hid)
    prefix = "model.layers.1.mlp.experts"
    cfg = _assignment_file(tmp_path / "a.json", {
        f"{prefix}.{e}.{proj}": {"data_type": "nvfp4_cb", "cb_k": 16}
        for e in range(E)
        for proj in ("gate_proj", "up_proj", "down_proj")})
    torch.manual_seed(12)
    cw = {f"{prefix}.{e}.{proj}": torch.rand(hid if proj != "down_proj"
                                             else inter) + 0.05
          for e in range(E)
          for proj in ("gate_proj", "up_proj", "down_proj")}
    _export_streaming(mdl, cfg, tmp_path / "out", cw, device="cpu",
                      allow_unstamped_research=True)
    qc = json.loads((tmp_path / "out" / "quant_config.json").read_text())
    names = set(load_file(str(tmp_path / "out" / "model.safetensors")))
    targets = [t for g in qc["config_groups"].values()
               for t in g.get("targets", [])]
    assert targets, "no CB targets in the manifest"
    for target in targets:
        assert any(n.startswith(target + ".") for n in names), (
            f"config target {target!r} names no emitted tensor — the manifest "
            "and the writer disagree on the namespace")


# ---------------------------------------------------------------------------
# K0.2 — per-expert stage calibration
# ---------------------------------------------------------------------------

def _write_act_entry(act_dir: Path, name: str, rows: torch.Tensor):
    import re
    act_dir.mkdir(parents=True, exist_ok=True)
    fname = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
    torch.save({"inputs": rows, "name": name,
                "row_indices": torch.arange(rows.shape[0])},
               act_dir / fname)


def _stage_members(prefix="model.layers.0.mlp.experts", n=4):
    _c, members, _r = _collapse_per_expert_assignment(
        _expanded(prefix, n), _groups(prefix, n), _StubProfile())
    return members


def test_per_expert_stage_calibration_splits_w13_from_w2(tmp_path):
    """w13 gets a value-bearing sample (the module input); w2 gets a max-abs
    over the routed intermediate. Never the other way round: the whole point
    of the stage attestation is that w2 was not calibrated on the module
    input."""
    from prismaquant.moe_imatrix import per_expert_stage_activation_calibration

    prefix = "model.layers.0.mlp.experts"
    members = _stage_members(prefix, 4)
    torch.manual_seed(5)
    act = tmp_path / "act"
    for e in range(4):
        _write_act_entry(act, f"{prefix}.{e}.gate_proj", torch.randn(6, 8))
        _write_act_entry(act, f"{prefix}.{e}.up_proj", torch.randn(6, 8))
        _write_act_entry(act, f"{prefix}.{e}.down_proj", torch.randn(6, 5) * 3)
    samples, max_abs = per_expert_stage_activation_calibration(act, members)
    assert set(samples) == {f"{prefix}.gate_up_proj"}
    assert set(max_abs) == {f"{prefix}.down_proj"}
    # One pooled row per expert, and the pooled max IS the exact global max.
    gu = samples[f"{prefix}.gate_up_proj"]
    assert gu.shape == (4, 8)
    exact = max(
        torch.load(act / f"model__layers__0__mlp__experts__{e}__{p}.pt",
                   weights_only=False)["inputs"].abs().max().item()
        for e in range(4) for p in ("gate_proj", "up_proj"))
    assert float(gu.max()) == pytest.approx(exact)


def test_per_expert_stage_calibration_skips_never_routed_experts(tmp_path):
    """An expert off the calibration distribution has NO cache entry. It
    contributes no observed activation and so nothing to a max — but a stack
    with no routed expert at all is uncalibrated and must fail closed."""
    from prismaquant.moe_imatrix import per_expert_stage_activation_calibration

    prefix = "model.layers.0.mlp.experts"
    members = _stage_members(prefix, 4)
    torch.manual_seed(6)
    act = tmp_path / "act"
    for e in (0, 2):                       # experts 1 and 3 never routed
        _write_act_entry(act, f"{prefix}.{e}.gate_proj", torch.randn(6, 8))
        _write_act_entry(act, f"{prefix}.{e}.up_proj", torch.randn(6, 8))
        _write_act_entry(act, f"{prefix}.{e}.down_proj", torch.randn(6, 5))
    samples, max_abs = per_expert_stage_activation_calibration(act, members)
    assert samples[f"{prefix}.gate_up_proj"].shape == (2, 8)
    assert max_abs[f"{prefix}.down_proj"] > 0.0

    with pytest.raises(ValueError, match="no calibrated input at all"):
        per_expert_stage_activation_calibration(tmp_path / "empty", members)


def test_per_expert_stage_calibration_refuses_the_mse_grid_policy(tmp_path):
    """The pooling is a max reduction: exact for an amax policy, meaningless
    for a distribution fit."""
    from prismaquant.moe_imatrix import per_expert_stage_activation_calibration

    with pytest.raises(ValueError, match="mse_grid_calibrated"):
        per_expert_stage_activation_calibration(
            tmp_path, _stage_members(), policy="mse_grid_calibrated.v1")
