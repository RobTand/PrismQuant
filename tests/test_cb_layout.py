"""Producer CB layout facts have one torch-free source of truth."""
from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from prismaquant import cb_layout


REPO = Path(__file__).resolve().parents[1]


def test_cb_layout_module_is_torch_free(tmp_path):
    probe = r'''
import builtins
import importlib.util
import pathlib
import sys

original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] == "torch":
        raise AssertionError("cb_layout imported torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("cb_layout_standalone", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.SUPERBLOCK == 256
assert module.CB_FORMAT_NAMES
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe,
         str(REPO / "prismaquant" / "cb_layout.py")],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_registry_parser_and_packer_share_the_layout_source():
    from prismaquant import format_registry
    from prismaquant import layer_config
    from prismaquant import nvfp4_cb_formats

    registry_names = {
        spec.name for spec in format_registry.list_formats()
        if spec.family in {"nvfp4_cb", "fp8_cb"}
    }
    producer_registry_names = {
        spec.name for spec in format_registry.list_producer_formats()
        if spec.family in {"nvfp4_cb", "fp8_cb"}
    }
    assert registry_names == cb_layout.ACCEPTED_CB_FORMAT_NAMES
    assert producer_registry_names == cb_layout.CB_FORMAT_NAMES
    assert (
        layer_config._NVFP4_CB_FORMAT_NAMES
        == cb_layout.ACCEPTED_CB_FORMAT_NAMES
    )
    assert nvfp4_cb_formats.VEC_DIM == cb_layout.VEC_DIM
    assert nvfp4_cb_formats.SUPERBLOCK == cb_layout.SUPERBLOCK
    assert nvfp4_cb_formats.FP4_GROUP == cb_layout.FP4_GROUP
    for family in cb_layout.FAMILIES:
        for k in family.rungs:
            for version in family.layout_versions:
                coding = (cb_layout.SCALE_CODING_TWO_TIER
                          if family.grid == "fp4" and version == 2
                          else cb_layout.SCALE_CODING_V1)
                assert nvfp4_cb_formats.nvfp4_cb_type_size(
                    k, family.grid, coding
                ) == cb_layout.type_size(k, family.grid, coding)


def test_fp8_product_and_reader_ladders_are_distinct_and_exact():
    assert cb_layout.FP8_PRODUCT_RUNGS == (40, 44, 48)
    assert cb_layout.FP8_ACCEPTED_RUNGS == tuple(range(28, 49))

    for k in cb_layout.FP8_PRODUCT_RUNGS:
        name = f"FP8_CB_K{k}"
        assert cb_layout.parse_producer_format_name(name) is not None
        assert cb_layout.is_producer_format_name(name)

    # Historical wire ids remain readable/reportable, but every rung outside
    # the evidence-backed producer trio must stay out of new artifacts.
    for k in (28, 29, 32, 36, 41, 45, 47):
        name = f"FP8_CB_K{k}"
        assert cb_layout.parse_format_name(name) is not None
        assert cb_layout.parse_producer_format_name(name) is None
        assert not cb_layout.is_producer_format_name(name)

    for k in (1, 4, 8, 12, 16, 20, 24, 25, 26, 27, 49):
        assert cb_layout.parse_format_name(f"FP8_CB_K{k}") is None


def test_nvfp4_public_reader_and_interim_producer_ladders_are_k12_k24():
    assert cb_layout.NVFP4_PRODUCT_RUNGS == tuple(range(12, 25))
    assert cb_layout.NVFP4_ACCEPTED_RUNGS == tuple(range(12, 25))
    for k in cb_layout.NVFP4_PRODUCT_RUNGS:
        name = f"NVFP4_CB_K{k}"
        assert cb_layout.parse_format_name(name) is not None
        assert cb_layout.parse_producer_format_name(name) is not None

    for k in (*range(1, 12), *range(25, 33)):
        name = f"NVFP4_CB_K{k}"
        assert cb_layout.parse_format_name(name) is None
        assert cb_layout.parse_producer_format_name(name) is None
        assert not cb_layout.is_producer_format_name(name)
    for k in (0, 33):
        assert cb_layout.parse_format_name(f"NVFP4_CB_K{k}") is None

    assert cb_layout.bit_split(1, 2) == (1, 0)
    assert cb_layout.codebook_subtable_shapes(1, "product", 2) == (
        (2, 4), (1, 4),
    )
    assert cb_layout.bit_split(32, 2) == (16, 16)
    assert cb_layout.codebook_subtable_shapes(32, "product", 2) == (
        (65536, 4), (65536, 4),
    )
    assert cb_layout.type_size(
        1, "fp4", cb_layout.SCALE_CODING_TWO_TIER
    ) == 13
    assert cb_layout.type_size(
        32, "fp4", cb_layout.SCALE_CODING_TWO_TIER
    ) == 137


def test_registry_exposes_legacy_readers_but_not_legacy_producers():
    from prismaquant import format_registry

    assert format_registry.get_format("FP8_CB_K29").name == "FP8_CB_K29"
    assert not format_registry.format_is_producer_eligible("FP8_CB_K29")
    assert not format_registry.format_is_producer_eligible("FP8_CB_K4")
    assert format_registry.format_is_producer_eligible("FP8_CB_K40")
    assert format_registry.format_is_producer_eligible("FP8_CB_K48")
    assert {
        spec.name for spec in format_registry.list_producer_formats("fp8_cb")
    } == cb_layout.FP8_CB_FORMAT_NAMES
    assert {
        spec.name for spec in format_registry.list_formats("fp8_cb")
    } == cb_layout.FP8_ACCEPTED_FORMAT_NAMES
    with pytest.raises(ValueError, match="reader-only"):
        format_registry.require_producer_formats(
            ["FP8_CB_K28", "FP8_CB_K29", "BF16"],
            where="test allocator menu",
        )

    assert format_registry.format_is_producer_eligible("NVFP4_CB_K24")
    assert not format_registry.format_is_producer_eligible("NVFP4_CB_K25")
    assert not format_registry.format_is_producer_eligible("NVFP4_CB_K26")
    assert not format_registry.format_is_producer_eligible("NVFP4_CB_K32")
    assert {
        spec.name for spec in format_registry.list_producer_formats("nvfp4_cb")
    } == cb_layout.NVFP4_CB_FORMAT_NAMES
    assert {
        spec.name for spec in format_registry.list_formats("nvfp4_cb")
    } == cb_layout.NVFP4_ACCEPTED_FORMAT_NAMES
    with pytest.raises(ValueError, match="reader-only/unsupported.*NVFP4_CB_K25"):
        format_registry.require_producer_formats(
            ["NVFP4_CB_K24", "NVFP4_CB_K25"],
            where="test allocator menu",
        )


def test_exact_accountant_uses_layout_subtable_shapes():
    from prismaquant.nvfp4_cb_footprint import codebook_subtable_shapes

    for family in cb_layout.FAMILIES:
        for k in family.rungs:
            name = family.name(k)
            assert codebook_subtable_shapes(name) == (
                cb_layout.codebook_subtable_shapes(
                    k, family.mode, family.n_sub
                )
            )

    # Reader compatibility is wider than ``family.rungs`` for FP8. Pin the
    # exact accountant's ability to report two representative off-law legacy
    # artifacts without widening the producer ladder.
    for name in ("FP8_CB_K29", "FP8_CB_K47"):
        family, k = cb_layout.parse_format_name(name)
        assert codebook_subtable_shapes(name) == (
            cb_layout.codebook_subtable_shapes(k, family.mode, family.n_sub)
        )


def test_family_and_subtable_rules_are_canonical():
    fp4_product = cb_layout.family_for("fp4", "product")
    fp8_product = cb_layout.family_for("fp8", "product")

    assert fp4_product.n_sub == 2
    assert fp8_product.n_sub == 4
    assert cb_layout.subtable_bit_widths(13, "product", 2) == (7, 6)
    assert cb_layout.subtable_bit_widths(29, "product", 4) == (8, 7, 7, 7)

    with pytest.raises(ValueError, match="unknown CB grid/mode"):
        cb_layout.family_for("fp4", "full")


def test_signed_family_is_deleted_and_refused_not_silently_reinterpreted():
    """The signed sign-magnitude family was deleted 2026-08-17.

    Gridbook's native FP4 path requires the UNSIGNED two-tier product layout
    (``n_sub=2, type_size=4*k+9``) and has no quality-preserving prefill kernel
    for ``n_sub=1``, so a signed rung could only ever ride a fallback route.

    Both halves matter. The family must be gone, AND ``subtable_bit_widths``
    must REFUSE the mode rather than fall through to the product split: a
    silent fall-through would hand a stale caller a different subtable geometry
    under the same name, which is a serialized-layout corruption rather than an
    error.
    """
    assert not any(f.mode == "signed" for f in cb_layout.FAMILIES)
    assert not any(n.startswith("NVFP4_CB_S") for n in cb_layout.CB_FORMATS)
    assert cb_layout.parse_format_name("NVFP4_CB_S16") is None

    with pytest.raises(ValueError, match="unknown CB grid/mode"):
        cb_layout.family_for("fp4", "signed")
    for n_sub in (1, 2):
        with pytest.raises(ValueError, match="signed CB mode was deleted"):
            cb_layout.subtable_bit_widths(13, "signed", n_sub)


def test_product_menu_and_lattice_generator_derive_from_layout():
    from scripts.gen_nvfp4_cb_lattices import required_lattice_specs

    suffix = ("NVFP4", "FP8_DYNAMIC", "BF16")
    expected_menu = ",".join((*cb_layout.PRODUCT_CB_FORMATS, *suffix))
    assert cb_layout.product_format_menu(*suffix) == expected_menu

    completed = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "print_cb_format_menu.py"),
         *suffix],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected_menu

    required = set(required_lattice_specs())
    for family in cb_layout.FAMILIES:
        for k in family.rungs:
            widths = cb_layout.subtable_bit_widths(
                k, family.mode, family.n_sub
            )
            shapes = cb_layout.codebook_subtable_shapes(
                k, family.mode, family.n_sub
            )
            for width, (_, dimension) in zip(widths, shapes):
                assert (
                    width,
                    family.grid,
                    dimension,
                    family.mode == "signed",
                ) in required
    assert {
        width for width, grid, dimension, positive in required
        if (grid, dimension, positive) == ("fp4", 4, False)
    } == {*range(6, 13), 14, 15, 16}

    for script in (
        "run_27b_cb_20gb.sh",
        "run_laguna_s21_prod.sh",
        "run_hy3_prod_joint.sh",
    ):
        text = (REPO / "scripts" / script).read_text(encoding="utf-8")
        assert "print_cb_format_menu.py" in text
        assert "range(12, 25)" not in text
        assert "range(28, 49)" not in text


def test_bare_k_family_resolution_uses_only_current_producer_authority():
    from scripts.build_hy3_mtp_cb_inputs import _expert_family_for_k

    with pytest.raises(ValueError, match="no producer-eligible"):
        _expert_family_for_k(1)
    assert _expert_family_for_k(40) == "fp8_cb"
    assert _expert_family_for_k(12) == "nvfp4_cb"
    assert _expert_family_for_k(12, family="nvfp4_cb") == "nvfp4_cb"
    with pytest.raises(ValueError, match="not a producer rung"):
        _expert_family_for_k(12, family="fp8_cb")


def test_production_serving_profile_cb_allowlist_matches_layout():
    from prismaquant.serving_profiles import load_serving_profile

    raw = json.loads((
        REPO / "prismaquant" / "serving_profile_specs" / "nvfp4_cb.json"
    ).read_text(encoding="utf-8"))
    raw_production = next(
        rule for rule in raw["format_rules"]
        if rule["id"] == "nvfp4_cb_container_formats"
    )
    assert raw_production["allow_formats_from"] == [
        "prismaquant.cb_layout:PRODUCT_CB_FORMAT_NAMES"
    ]

    profile = load_serving_profile("nvfp4_cb")
    production = next(
        rule for rule in profile.format_rules
        if rule.id == "nvfp4_cb_container_formats"
    )
    declared_cb = {
        name for name in production.allow_formats
        if cb_layout.parse_format_name(name) is not None
    }
    assert declared_cb == cb_layout.PRODUCT_CB_FORMAT_NAMES

    shape = next(rule for rule in profile.shape_rules
                 if rule.id == "cb_superblock_shape")
    assert set(shape.formats) == cb_layout.ACCEPTED_CB_FORMAT_NAMES

    fp8_shape = next(rule for rule in profile.shape_rules
                     if rule.id == "cb_fp8_out_features_load_gate")
    assert set(fp8_shape.formats) == cb_layout.FP8_ACCEPTED_FORMAT_NAMES

    fp4_shape = next(rule for rule in profile.shape_rules
                     if rule.id == "cb_fp4_out_features_load_gate")
    assert set(fp4_shape.formats) == cb_layout.NVFP4_ACCEPTED_FORMAT_NAMES

    fp4_lane = next(lane for lane in profile.serving_lanes
                    if lane.id == "nvfp4_cb_quality_path")
    assert set(fp4_lane.formats) == cb_layout.NVFP4_ACCEPTED_FORMAT_NAMES

    fp8_lane = next(lane for lane in profile.serving_lanes
                    if lane.id == "fp8_cb_fused_mid_m")
    assert set(fp8_lane.formats) == cb_layout.FP8_ACCEPTED_FORMAT_NAMES
