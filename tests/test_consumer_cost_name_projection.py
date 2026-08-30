"""The cost consumer routes every name mapping through name_projection.

R5 consumer-cost migration: `_scan_source_dtype_manifest` lost its private
checkpoint->live->recipe builders (`_strip_weight_suffix`,
`_to_recipe_name`, `_packed_to_recipe_name`,
`_per_expert_packed_recipe_name`) and `decision_units` lost `_recipe_name`.
These tests pin the migrated contract:

- values are UNCHANGED versus the private spellings (same manifest keys,
  same kinds, MTP rows retained verbatim, visual keys declined);
- a profile accessor failure REFUSES (`NameProjectionError`) instead of
  silently skipping rows — the fail-closed direction requirement 3 demands;
- the packed-parent fold goes through the layer's
  `packed_parent_of_expert_param` (footprint.packed_expert_alias + the
  profile's own mapping), and does not exist without a caller-supplied
  profile.
"""
from __future__ import annotations

import torch

from prismaquant.allocator_candidates import _scan_source_dtype_manifest
from prismaquant.name_projection import NameProjectionError
from prismaquant.model_profiles import DefaultProfile


def _write_checkpoint(tmp_path, tensors: dict[str, torch.Tensor]) -> str:
    from safetensors.torch import save_file

    save_file(tensors, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def test_manifest_with_real_profile_matches_the_old_hardcoded_fallbacks(
    tmp_path,
) -> None:
    # With a real profile the old code consulted
    # checkpoint_to_live_name/live_to_recipe_name; with profile=None it
    # inlined the SAME rules as string surgery. The migrated scan builds the
    # projection over the supplied profile (or DefaultProfile when none), so
    # both call conventions must agree value-for-value.
    model_path = _write_checkpoint(tmp_path, {
        "model.layers.0.self_attn.q_proj.weight":
            torch.randn(4, 4, dtype=torch.bfloat16),
        # Umbrella-infix spelling must land on the stripped recipe key.
        "model.language_model.layers.1.self_attn.o_proj.weight":
            torch.randn(4, 4, dtype=torch.float32),
        # Visual towers are DECLARED out-of-graph: no row, not identity.
        "model.visual.encoder.patch_embed.weight":
            torch.randn(2, 2, dtype=torch.bfloat16),
    })
    expected = {
        "model.layers.0.self_attn.q_proj": "bf16",
        "model.layers.1.self_attn.o_proj": "f32",
    }
    assert _scan_source_dtype_manifest(model_path, profile=None) == expected
    assert (
        _scan_source_dtype_manifest(model_path, profile=DefaultProfile())
        == expected
    )


def test_manifest_keeps_mtp_rows_verbatim_under_a_real_profile(
    tmp_path,
) -> None:
    # MTP tensors are REAL source tensors stored under the recipe namespace
    # itself; profiles DECLINE mtp.* in checkpoint_to_live_name (DSv4 drops
    # them from the body map), but the manifest must keep them or the BF16
    # passthrough disappears again (--mtp-format=BF16 hard-failed before).
    model_path = _write_checkpoint(tmp_path, {
        "mtp.fc.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "mtp.layers.0.self_attn.q_proj.weight":
            torch.randn(4, 4, dtype=torch.bfloat16),
    })
    expected = {
        "mtp.fc": "bf16",
        "mtp.layers.0.self_attn.q_proj": "bf16",
    }
    assert _scan_source_dtype_manifest(model_path, profile=None) == expected
    assert (
        _scan_source_dtype_manifest(model_path, profile=DefaultProfile())
        == expected
    )


def test_packed_parent_fold_routes_through_the_layer(tmp_path) -> None:
    # Per-expert INDEXED source keys fold their kind onto the packed recipe
    # parent the probe/cost actually enumerate — via the layer's
    # packed_parent_of_expert_param (footprint.packed_expert_alias + the
    # profile's declared mapping), not a local structural re-parse.
    model_path = _write_checkpoint(tmp_path, {
        "model.layers.0.mlp.experts.0.gate_proj.weight":
            torch.randn(8, 4, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight":
            torch.randn(8, 4, dtype=torch.float32),
    })
    manifest = _scan_source_dtype_manifest(
        model_path, profile=DefaultProfile())
    # Own rows keep their names...
    assert manifest["model.layers.0.mlp.experts.0.gate_proj"] == "bf16"
    assert manifest["model.layers.0.mlp.experts.0.up_proj"] == "f32"
    # ...and the mixed kinds fold onto the shared packed parent as
    # heterogeneous (no byte-copy format may be admitted for it).
    assert manifest["model.layers.0.mlp.experts.gate_up_proj"] == (
        "heterogeneous"
    )
    # Without a caller-supplied profile there is NO declared mapping and
    # nothing folds (historical behavior).
    unfolded = _scan_source_dtype_manifest(model_path, profile=None)
    assert "model.layers.0.mlp.experts.gate_up_proj" not in unfolded


def test_profile_accessor_failure_refuses_instead_of_silently_skipping(
    tmp_path,
) -> None:
    # The old private mapper swallowed every accessor exception into an
    # empty recipe name and dropped the row — the wo_a shape. The layer
    # refuses; the refusal must propagate out of the scan.
    class BrokenProfile(DefaultProfile):
        def checkpoint_to_live_name(self, ckpt_key: str, *,
                                    multimodal: bool = False):
            raise RuntimeError("boom")

    model_path = _write_checkpoint(tmp_path, {
        "model.layers.0.self_attn.q_proj.weight":
            torch.randn(4, 4, dtype=torch.bfloat16),
    })
    try:
        _scan_source_dtype_manifest(model_path, profile=BrokenProfile())
    except NameProjectionError as exc:
        assert exc.code == "profile_accessor_failed"
        assert exc.source_namespace == "checkpoint"
        assert exc.target_namespace == "live"
    else:
        raise AssertionError(
            "a failing profile accessor was swallowed into a skipped row")


def test_decision_units_uses_the_shared_leaf_function() -> None:
    # decision_units._recipe_name is gone; the leaf rule lives once, in
    # name_projection. Pin the migrated call sites' values.
    from prismaquant.decision_units import (
        _decision_unit_name_from_graph_unit,
    )

    assert _decision_unit_name_from_graph_unit(
        "fused:model.layers.0.self_attn.qkv_proj.weight"
    ) == "model.layers.0.self_attn.qkv_proj"
    assert _decision_unit_name_from_graph_unit(
        "tensor:model.layers.1.mlp.down_proj.weight"
    ) == "model.layers.1.mlp.down_proj"
