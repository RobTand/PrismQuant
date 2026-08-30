from __future__ import annotations

import dataclasses

import pytest

import prismaquant.format_registry as fr
import prismaquant.serving_profiles as serving_profiles_module
from prismaquant.gridbook_runtime_pin import load_gridbook_runtime_pin
from prismaquant.serving_profiles import (
    ExportLaneSpec,
    ServingProfile,
    check_serving_format,
    check_serving_shape,
    lane_emittable_formats,
    load_serving_profile,
    serving_profile_names,
)


VLLM_PROFILE = "vllm_packed_moe"

# Representative qnames for dense, unpacked shared-expert, and packed-expert
# targets. Shared experts are rank-2 tensors and therefore use dense scope.
DENSE_QNAME = "model.layers.0.self_attn.q_proj"
SHARED_QNAME = "model.layers.0.mlp.shared_expert.gate_proj"
EXPERT_QNAME = "model.layers.0.mlp.experts.gate_up_proj"

NVFP4_CB_SCOPE_CASES = (
    pytest.param(DENSE_QNAME, False, id="dense"),
    pytest.param(SHARED_QNAME, False, id="shared"),
    pytest.param(EXPERT_QNAME, True, id="packed"),
)

ALL_FORMAT_NAMES = tuple(sorted(set(fr.REGISTRY) | set(fr.FORMAT_ALIASES)))


def test_serving_profile_names_are_config_discovered():
    assert "research" in serving_profile_names()
    assert VLLM_PROFILE in serving_profile_names()


def test_gridbook_runtime_version_fails_closed_for_unreleased_pin(monkeypatch):
    pin = dataclasses.replace(
        load_gridbook_runtime_pin(),
        version_is_release=False,
    )
    monkeypatch.setattr(serving_profiles_module, "_RUNTIME_VERSION", None)
    monkeypatch.setattr(
        serving_profiles_module,
        "load_gridbook_runtime_pin",
        lambda: pin,
    )

    assert serving_profiles_module.gridbook_runtime_version() == ""


def test_vllm_profile_extends_runtime_shape_rules():
    profile = load_serving_profile(VLLM_PROFILE)

    assert profile.extends == ("research",)
    assert any(rule.id == "mxfp8_cutlass_shape" for rule in profile.shape_rules)
    assert any(
        rule.id == "flashinfer_mxfp8_problem_size"
        and rule.callable_path
        == "prismaquant.runtime_shape_validators:flashinfer_mxfp8_problem_size_accepts"
        for rule in profile.runtime_shape_validators
    )
    flashinfer = profile.runtime_package("flashinfer")
    assert flashinfer is not None
    assert flashinfer.version == "0.6.8.post1"
    assert flashinfer.pip_packages == ("flashinfer-python", "flashinfer-cubin")
    assert flashinfer.env_dict()["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    assert any(rule.id == "packed_moe_expert_formats" for rule in profile.format_rules)


def test_qwen_serving_profile_id_remains_compatibility_alias():
    profile = load_serving_profile("vllm_qwen3_5_packed_moe")

    assert profile.extends == ("vllm_packed_moe",)
    assert any(rule.id == "packed_moe_expert_formats" for rule in profile.format_rules)


def test_serving_profile_format_rules_are_config_backed():
    expert = "model.layers.0.mlp.experts.gate_up_proj"
    root_expert = "model.layers.0.experts.gate_up_proj"
    dense = "model.layers.0.self_attn.q_proj"

    assert check_serving_format(VLLM_PROFILE, expert, "MXFP8_E4M3").legal
    assert check_serving_format(VLLM_PROFILE, root_expert, "MXFP4").legal
    expert_fp8 = check_serving_format(VLLM_PROFILE, expert, "FP8_E4M3")
    assert expert_fp8.legal
    root_fp8 = check_serving_format(VLLM_PROFILE, root_expert, "FP8_E4M3")
    assert root_fp8.legal

    dense_mxfp4 = check_serving_format(VLLM_PROFILE, dense, "MXFP4")
    assert not dense_mxfp4.legal
    assert dense_mxfp4.rule == "dense_formats_without_vllm_fast_path"


def test_serving_profile_shape_rules_are_config_backed():
    small_n = check_serving_shape(
        "research",
        "MXFP8_E4M3",
        in_features=5120,
        out_features=48,
    )
    standard = check_serving_shape(
        VLLM_PROFILE,
        "MXFP8_E4M3",
        in_features=5120,
        out_features=10240,
    )
    nvfp4_bad_k = check_serving_shape(
        "research",
        "NVFP4",
        in_features=17,
        out_features=128,
    )

    assert not small_n.legal
    assert small_n.reason == "kernel_shape"
    assert "out_features=48" in small_n.detail
    assert standard.legal
    assert not nvfp4_bad_k.legal


def test_shape_rules_can_be_name_scoped():
    profile = ServingProfile.from_dict({
        "schema": "prismaquant.serving_profile.v1",
        "id": "unit_scoped",
        "shape_rules": [
            {
                "id": "expert_only_alignment",
                "when": {"contains": ".experts."},
                "formats": ["MXFP8_E4M3"],
                "out_features_multiple_of": 128,
            }
        ],
    })

    expert = profile.check_shape(
        "MXFP8_E4M3",
        qname="model.layers.0.mlp.experts.0.gate_proj",
        in_features=256,
        out_features=96,
    )
    dense = profile.check_shape(
        "MXFP8_E4M3",
        qname="model.layers.0.mlp.gate_proj",
        in_features=256,
        out_features=96,
    )

    assert not expert.legal
    assert expert.rule == "expert_only_alignment"
    assert dense.legal


def test_runtime_shape_validator_rules_are_config_backed(monkeypatch):
    import prismaquant.serving_profiles as serving_profiles

    def fake_loader(callable_path):
        assert callable_path == (
            "prismaquant.runtime_shape_validators:"
            "flashinfer_mxfp8_problem_size_accepts"
        )

        def fake_validator(fmt, *, in_features, out_features):
            assert fmt == "MXFP8_E4M3"
            assert (in_features, out_features) == (5120, 10240)
            return False

        return fake_validator

    monkeypatch.setattr(
        serving_profiles,
        "_load_runtime_validator",
        fake_loader,
    )

    decision = serving_profiles.check_serving_shape(
        "research",
        "MXFP8_E4M3",
        in_features=5120,
        out_features=10240,
    )

    assert not decision.legal
    assert decision.rule == "flashinfer_mxfp8_problem_size"
    assert decision.reason == "kernel_shape"


def test_runtime_shape_validator_treats_fp8_setup_failure_as_unavailable(
    monkeypatch,
):
    import sys
    import types

    from prismaquant.runtime_shape_validators import (
        flashinfer_mxfp8_problem_size_accepts,
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.uint8 = object()

    def fake_empty(*_args, **_kwargs):
        raise RuntimeError("fp8 setup unavailable")

    fake_torch.empty = fake_empty

    fake_flashinfer = types.ModuleType("flashinfer")
    fake_gemm = types.ModuleType("flashinfer.gemm")
    fake_gemm_base = types.ModuleType("flashinfer.gemm.gemm_base")
    fake_gemm_base._check_mm_mxfp8_problem_size = lambda *_args: True
    fake_gemm_base._mxfp8_swizzled_scale_len = lambda *_args: 1
    fake_gemm_base.SfLayout = types.SimpleNamespace(layout_8x4=object())

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", fake_gemm)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm.gemm_base", fake_gemm_base)

    assert (
        flashinfer_mxfp8_problem_size_accepts(
            "MXFP8_E4M3",
            in_features=5120,
            out_features=10240,
        )
        is None
    )


def test_runtime_shape_validator_legacy_id_fallback(monkeypatch):
    import prismaquant.serving_profiles as serving_profiles

    def fake_loader(callable_path):
        assert callable_path == (
            "prismaquant.runtime_shape_validators:"
            "flashinfer_mxfp8_problem_size_accepts"
        )

        def fake_validator(fmt, *, in_features, out_features):
            assert fmt == "MXFP8_E4M3"
            assert (in_features, out_features) == (5120, 10240)
            return False

        return fake_validator

    monkeypatch.setattr(
        serving_profiles,
        "_load_runtime_validator",
        fake_loader,
    )

    decision = serving_profiles._runtime_shape_validator_accepts(
        "flashinfer_mxfp8_problem_size",
        "MXFP8_E4M3",
        in_features=5120,
        out_features=10240,
    )

    assert decision is False


def test_runtime_shape_validators_can_be_name_scoped(monkeypatch):
    import prismaquant.serving_profiles as serving_profiles

    calls = []

    def fake_loader(_callable_path):
        def fake_validator(fmt, *, in_features, out_features):
            calls.append((fmt, in_features, out_features))
            return False

        return fake_validator

    monkeypatch.setattr(
        serving_profiles,
        "_load_runtime_validator",
        fake_loader,
    )

    profile = ServingProfile.from_dict({
        "schema": "prismaquant.serving_profile.v1",
        "id": "unit_runtime_scoped",
        "runtime_shape_validators": [
            {
                "id": "expert_runtime",
                "when": {"contains": ".experts."},
                "formats": ["MXFP8_E4M3"],
                "callable": "tests.fake:validator",
            }
        ],
    })

    dense = profile.check_shape(
        "MXFP8_E4M3",
        qname="model.layers.0.mlp.gate_proj",
        in_features=256,
        out_features=256,
    )
    expert = profile.check_shape(
        "MXFP8_E4M3",
        qname="model.layers.0.mlp.experts.0.gate_proj",
        in_features=256,
        out_features=256,
    )

    assert dense.legal
    assert not expert.legal
    assert expert.rule == "expert_runtime"
    assert calls == [("MXFP8_E4M3", 256, 256)]


# ---------------------------------------------------------------------------
# Export-lane bound: a serving profile must not be able to admit a format
# its lane's exporter cannot emit (issue #22 part 2).
#
# The bound is derived from each exporter's OWN declaration
# (export_native_compressed.EXPORTABLE_FORMATS for the compressed-tensors
# lane, gguf_formats.GGUF_BLOCK_BYTES for the GGUF lane), so these tests pin
# the derivation against the exporters' real accept/reject behaviour rather
# than re-listing formats.
# ---------------------------------------------------------------------------


def _shipped_profiles() -> list[ServingProfile]:
    return [load_serving_profile(name) for name in serving_profile_names()]


def test_every_shipped_profile_is_lane_bound_or_declared_emulation_only():
    """Fail-closed authoring gate. A new serving profile either names the
    exporter that bounds its menu or declares itself emulation-only; it
    cannot silently ship an unbounded production menu."""
    for profile in _shipped_profiles():
        assert profile.emulation_only or profile.export_lane is not None, (
            f"serving profile {profile.id!r} declares neither an "
            f"export_lane nor emulation_only: its format menu is not bounded "
            f"by any exporter, so the allocator could spend budget on a rung "
            f"that hard-fails (or silently BF16-coerces) at export."
        )
        # Not both: a lane-bound profile that also claimed emulation-only
        # would read as exempt while carrying an exporter.
        assert not (profile.emulation_only and profile.export_lane is not None)


def test_research_profile_is_the_declared_emulation_only_exemption():
    """`research` is deliberately unbounded: it exists so rungs with no
    served path stay measurable in the emulation harness. Nothing ships
    under it — export stages resolve a real serving profile."""
    research = load_serving_profile("research")
    assert research.emulation_only is True
    assert research.export_lane is None
    assert lane_emittable_formats("research") is None
    for fmt in ("MXFP6_E3M2", "INT4_W4A16_g128", "NVFP4A16", "MXFP8A16",
                "INT8_W8A16", "Q4_K"):
        assert check_serving_format("research", DENSE_QNAME, fmt).legal, fmt


@pytest.mark.parametrize("profile_id", ["vllm_packed_moe", "nvfp4_cb", "gguf"])
def test_production_profile_never_admits_an_unexportable_format(profile_id):
    """The invariant: effective-legal ⊆ exporter-emittable, for every
    registered format and both rule scopes."""
    emittable = lane_emittable_formats(profile_id)
    assert emittable
    scopes = (
        (DENSE_QNAME, False),
        (SHARED_QNAME, False),
        (EXPERT_QNAME, True),
    )
    for qname, packed_expert in scopes:
        for fmt in ALL_FORMAT_NAMES:
            if not check_serving_format(
                profile_id, qname, fmt, packed_expert=packed_expert
            ).legal:
                continue
            assert fr.canonical_format_name(fmt) in emittable, (
                f"{profile_id} admits {fmt} at {qname} but its exporter "
                f"cannot emit it (emittable={sorted(emittable)})"
            )


@pytest.mark.parametrize("profile_id", ["vllm_packed_moe", "gguf"])
def test_lane_bound_survives_a_widened_policy_rule(profile_id):
    """Root-cause check, not a snapshot of today's deny lists: even with
    every policy rule stripped away, the lane still refuses formats the
    exporter cannot emit. Widening an allow/deny list can therefore never
    re-admit an unexportable rung."""
    profile = load_serving_profile(profile_id)
    unpoliced = dataclasses.replace(profile, format_rules=())
    emittable = profile.export_lane.emittable_formats()
    for fmt in ALL_FORMAT_NAMES:
        decision = unpoliced.check_format(DENSE_QNAME, fmt)
        expected = fr.canonical_format_name(fmt) in emittable
        assert decision.legal is expected, fmt
        if not expected:
            assert decision.reason == "exporter_cannot_emit"
            assert decision.rule == profile.export_lane.id


def test_vllm_lane_denies_the_a16_rungs_with_a_structural_reason():
    """The concrete regression: A16 rungs were legal for dense Linears on
    the vLLM lane (the dense rule denies only MXFP4/MXFP8_E5M2/FP8_E5M2)
    while `_quantize_2d` has no branch for them — and the bit-exact
    re-encode short-circuit prices a weight-lossless A16 rung at dloss
    0.0, the unbeatable global minimum."""
    for fmt in ("NVFP4A16", "MXFP8A16", "INT8_W8A16", "INT4_W4A16_g128",
                "MXFP6_E3M2", "MXFP6_E2M3", "Q4_K", "IQ4_XS"):
        decision = check_serving_format(VLLM_PROFILE, DENSE_QNAME, fmt)
        assert not decision.legal, fmt
        assert decision.reason == "exporter_cannot_emit", fmt


def test_nvfp4_cb_all_product_rungs_remain_in_every_production_scope():
    from prismaquant.cb_layout import NVFP4_PRODUCT_RUNGS

    product_rungs = (
        *(f"NVFP4_CB_K{k}" for k in NVFP4_PRODUCT_RUNGS),
        *(f"FP8_CB_K{k}" for k in range(4, 49, 4)),
    )
    for qname, packed_expert in (
        (DENSE_QNAME, False),
        (SHARED_QNAME, False),
        (EXPERT_QNAME, True),
    ):
        for fmt in product_rungs:
            decision = check_serving_format(
                "nvfp4_cb", qname, fmt, packed_expert=packed_expert
            )
            assert decision.legal, (qname, fmt, decision)


@pytest.mark.parametrize("qname,packed_expert", NVFP4_CB_SCOPE_CASES)
def test_nvfp4_cb_w4a16_stays_denied_until_gridbook_support_lands(
    qname, packed_expert
):
    """Approved backlog status is not an exporter/runtime capability claim."""
    fmt = "INT4_W4A16_g128"
    assert fmt in fr.REGISTRY
    assert check_serving_format(
        "research", qname, fmt, packed_expert=packed_expert
    ).legal

    decision = check_serving_format(
        "nvfp4_cb", qname, fmt, packed_expert=packed_expert
    )
    assert not decision.legal
    assert decision.rule == "nvfp4_cb_container_formats"

    profile = load_serving_profile("nvfp4_cb")
    unpoliced = dataclasses.replace(profile, format_rules=())
    structural = unpoliced.check_format(
        qname, fmt, packed_expert=packed_expert
    )
    assert not structural.legal
    assert structural.reason == "exporter_cannot_emit"


def test_vllm_lane_still_admits_the_whole_production_menu():
    """Backwards compatibility: the bound must not narrow any format the
    shipped recipes actually use (run-pipeline's FORMATS default is
    NVFP4,FP8_DYNAMIC,BF16; FP8_SOURCE and MXFP8_E4M3 are in the menu)."""
    for fmt in ("NVFP4", "FP8_E4M3", "FP8_DYNAMIC", "FP8", "MXFP8_E4M3",
                "MXFP8", "BF16", "FP8_SOURCE"):
        assert check_serving_format(VLLM_PROFILE, DENSE_QNAME, fmt).legal, fmt
    for fmt in ("NVFP4", "FP8_E4M3", "MXFP8_E4M3", "MXFP4", "BF16"):
        assert check_serving_format(VLLM_PROFILE, EXPERT_QNAME, fmt).legal, fmt


def test_gguf_lane_admits_every_ggml_type_and_nothing_else():
    """The GGUF lane's legitimate formats must be untouched — the bound is
    per-lane, derived from the GGUF codec table, not from the
    compressed-tensors exporter."""
    from prismaquant.gguf_formats import GGUF_BLOCK_BYTES

    q = "model.layers.0.mlp.down_proj"
    for fmt in GGUF_BLOCK_BYTES:
        assert check_serving_format("gguf", q, fmt).legal, fmt
    assert check_serving_format("gguf", q, "BF16").legal
    assert lane_emittable_formats("gguf") == frozenset(
        set(GGUF_BLOCK_BYTES) | {"BF16"})


def test_compressed_tensors_lane_declaration_matches_exporter_behaviour():
    """Anti-drift pin. Adding a `_quantize_2d` branch without a
    FORMAT_SCHEME entry (or vice versa) breaks this, so the derived menu
    can never silently diverge from what the exporter really does."""
    import torch

    from prismaquant.allocator_candidates import (
        PASSTHROUGH_SOURCE_REQUIREMENTS,
    )
    import prismaquant.export_native_compressed as enc

    emittable = lane_emittable_formats(VLLM_PROFILE)
    # Every scheme the exporter can describe is either a codec format or a
    # declared passthrough; nothing else is in the menu. Spelled out rather
    # than compared to EXPORTABLE_FORMATS so that moving the source of truth
    # into the exporter (issue #27) is pinned as a no-op for the menu.
    assert emittable == frozenset(
        {fr.canonical_format_name(f) for f in enc.FORMAT_SCHEME} | {"BF16"})
    # ...and the lane now reads exactly that constant, so the exporter owns
    # its own bound instead of the profile spec restating it.
    assert emittable == frozenset(enc.EXPORTABLE_FORMATS)

    # Canonical names only: `layer_config.canonicalize_format` resolves the
    # FP8/FP8_DYNAMIC/MXFP8 aliases before an assignment reaches the
    # exporter, so `_quantize_2d` is only ever handed a canonical name.
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    nvfp4_cb_emittable = lane_emittable_formats("nvfp4_cb")
    for fmt in sorted(fr.REGISTRY):
        if fmt in PASSTHROUGH_SOURCE_REQUIREMENTS:
            # Source passthroughs ship through a container's passthrough
            # branch (plain bf16 tensor, verbatim fp8 + scale copy, verbatim
            # packed-MXFP4 + E8M0 copy), never through the weight codec.
            #
            # Which CONTAINER carries which passthrough is a per-lane fact,
            # not a global one: BF16 and FP8_SOURCE are compressed-tensors
            # passthroughs, while FP8_BLOCK_UE8M0_SOURCE and MXFP4_SOURCE are
            # nvfp4_cb-container passthroughs whose byte layouts the CT
            # exporter has no emit path for. The invariant that must hold
            # everywhere is the codec one — a passthrough is never quantized —
            # plus "no orphans": every declared passthrough is emittable by
            # SOME lane, so a format cannot be legal to allocate and
            # impossible to ship.
            if fmt not in emittable:
                assert fmt in nvfp4_cb_emittable, fmt
                with pytest.raises(ValueError):
                    enc._quantize_2d(w, fmt)
            continue
        if fmt in emittable:
            assert enc._quantize_2d(w, fmt), fmt
        else:
            with pytest.raises(ValueError):
                enc._quantize_2d(w, fmt)


def test_gguf_lane_declaration_matches_the_exporters_own_gate():
    """Both GGUF exporters gate emission on `fmt in GGUF_BLOCK_BYTES`; the
    lane derives its menu from that same object, and every entry has a
    field codec behind it."""
    pytest.importorskip("gguf")
    import prismaquant.export_gguf as export_gguf
    import prismaquant.export_gguf_direct as export_gguf_direct
    from prismaquant import gguf_formats

    assert (export_gguf_direct.GGUF_BLOCK_BYTES
            is gguf_formats.GGUF_BLOCK_BYTES)
    assert export_gguf.GGUF_BLOCK_BYTES is gguf_formats.GGUF_BLOCK_BYTES
    assert set(gguf_formats.GGUF_BLOCK_BYTES) == set(gguf_formats._FIELDS)


def test_every_format_named_in_a_shipped_profile_resolves_in_the_registry():
    """Typo guard: a misspelled allow entry silently narrows a menu and a
    misspelled deny entry silently widens one."""
    for profile in _shipped_profiles():
        for rule in profile.format_rules:
            for fmt in (*rule.allow_formats, *rule.deny_formats):
                fr.get_format(fmt)  # raises KeyError on an unknown name


def test_export_lane_with_a_stale_declaration_fails_loudly():
    """The declaration is a dotted path into the exporter. If the exporter
    renames its table, the profile must fail loudly rather than fall back
    to an empty (deny-everything) or unbounded menu."""
    lane = ExportLaneSpec(
        id="unit_lane",
        exporter="prismaquant.export_native_compressed",
        codec_formats_from=(
            "prismaquant.export_native_compressed:FORMAT_SCHEME_RENAMED",
        ),
    )
    with pytest.raises(RuntimeError, match="has no attribute"):
        lane.emittable_formats()

    empty = ExportLaneSpec(id="unit_empty_lane")
    with pytest.raises(RuntimeError, match="no emittable formats"):
        empty.emittable_formats()

    not_iterable = ExportLaneSpec(
        id="unit_scalar_lane",
        codec_formats_from=(
            "prismaquant.export_native_compressed:FP8_E4M3_MAX",
        ),
    )
    with pytest.raises(RuntimeError, match="not iterable"):
        not_iterable.emittable_formats()


def test_export_lane_reads_a_set_declaration_as_well_as_a_dict():
    """The compressed-tensors lane declares a `frozenset`
    (EXPORTABLE_FORMATS) where the GGUF lane declares a dict
    (GGUF_BLOCK_BYTES). Both are just iterables of format names, and both
    get canonicalized, so neither container shape is privileged."""
    as_set = ExportLaneSpec(
        id="unit_set_lane",
        codec_formats_from=("prismaquant.serving_profiles:_UNIT_SET_DECL",),
    )
    as_dict = ExportLaneSpec(
        id="unit_dict_lane",
        codec_formats_from=("prismaquant.serving_profiles:_UNIT_DICT_DECL",),
    )
    import prismaquant.serving_profiles as sp

    sp._UNIT_SET_DECL = frozenset({"NVFP4", "MXFP8"})
    sp._UNIT_DICT_DECL = {"NVFP4": object(), "MXFP8": object()}
    try:
        # `MXFP8` canonicalizes to `MXFP8_E4M3` from either container.
        expected = frozenset({"NVFP4", "MXFP8_E4M3"})
        assert as_set.emittable_formats() == expected
        assert as_dict.emittable_formats() == expected
    finally:
        del sp._UNIT_SET_DECL
        del sp._UNIT_DICT_DECL


def test_vllm_lane_needs_no_passthrough_entry_of_its_own():
    """Issue #27: the exporter's declaration already includes its container
    passthroughs, so the spec must not restate them -- one source of truth.
    BF16 staying in the menu is the check that removing the entry did not
    narrow anything."""
    import prismaquant.export_native_compressed as enc

    lane = load_serving_profile(VLLM_PROFILE).export_lane
    assert lane.passthrough_formats == ()
    assert lane.codec_formats_from == (
        "prismaquant.export_native_compressed:EXPORTABLE_FORMATS",
    )
    assert "BF16" in lane.emittable_formats()
    assert "BF16" in enc.EXPORTABLE_FORMATS
    assert check_serving_format(VLLM_PROFILE, DENSE_QNAME, "BF16").legal

    # The GGUF lane's exporter declares a bare ggml-type table, so it still
    # needs its own passthrough entry; the field is not dead.
    assert load_serving_profile("gguf").export_lane.passthrough_formats == (
        "BF16",
    )
