"""Pinned producer/runtime compatibility without a mirrored Gridbook tree."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re

import pytest


REPO = Path(__file__).resolve().parents[1]
PIN = (REPO / "prismaquant" / "gridbook_runtime" /
       "gridbook_runtime_pin.json")
REQUIRE_CONTRACT = os.environ.get(
    "PRISMAQUANT_REQUIRE_GRIDBOOK_CONTRACT") == "1"

pytestmark = pytest.mark.skipif(
    not REQUIRE_CONTRACT,
    reason="run by the pinned Gridbook compatibility CI job",
)


def _pin() -> dict:
    value = json.loads(PIN.read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", value["commit"])
    return value


def _direct_url() -> dict:
    dist = importlib.metadata.distribution("gridbook")
    matches = [file for file in (dist.files or ())
               if file.name == "direct_url.json"
               and ".dist-info" in str(file.parent)]
    assert len(matches) == 1, (
        "the compatibility job must install Gridbook from the pinned VCS URL")
    path = Path(dist.locate_file(matches[0]))
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_contract() -> dict:
    from gridbook.runtime_contract import load_runtime_contract

    return load_runtime_contract()


def test_materialized_contract_equals_the_installed_wheels_file():
    """The local copy IS the packaged contract, byte for byte (R3).

    PrismaQuant resolves serving-lane route status from a materialized copy of
    Gridbook's packaged ``runtime_contract.json`` so the producer never imports
    the runtime. That indirection is only honest while the copy is identical to
    what the pinned wheel actually ships -- a drifted copy would attest a
    release that does not exist. This job is the only place both files are
    present, so it is the only place the equality can be checked.
    """
    import hashlib
    from importlib.resources import files

    asset_dir = REPO / "prismaquant" / "gridbook_runtime"
    index = json.loads(
        (asset_dir / "gridbook_runtime_contract_index.json").read_text(
            encoding="utf-8"))
    installed_version = importlib.metadata.version("gridbook")
    entry = next(
        (e for e in index["contracts"]
         if e["version"] == installed_version),
        None,
    )
    assert entry is not None, (
        f"no materialized contract for the installed Gridbook "
        f"{installed_version}; add its packaged runtime_contract.json to "
        f"{asset_dir} and index it, or route status attests nothing")

    packaged = files("gridbook").joinpath("runtime_contract.json").read_bytes()
    local = (asset_dir / entry["path"]).read_bytes()
    assert local == packaged, (
        f"{entry['path']} differs from the installed wheel's packaged "
        "contract; re-materialize it from the pinned commit")
    assert hashlib.sha256(packaged).hexdigest() == entry["sha256"]

    # And the ABSENT claim in the index must still be true of the real wheel.
    contract = json.loads(packaged.decode("utf-8"))
    has_table = "lane_eligibility" in contract
    assert (entry["lane_eligibility"] == "present") == has_table, (
        "the index's lane_eligibility claim disagrees with the installed "
        "runtime; if Gridbook now packages the table, flip the index entry to "
        "'present' and the CB lanes become attested with no code change")


def test_installed_gridbook_is_the_exact_external_vcs_pin():
    pin = _pin()
    spec = importlib.util.find_spec("gridbook")
    assert spec is not None and spec.origin
    origin = Path(spec.origin).resolve()
    assert not origin.is_relative_to(REPO), (
        f"Gridbook imported from inside PrismaQuant: {origin}")
    assert importlib.metadata.version("gridbook") == pin["version"]

    direct = _direct_url()
    vcs = direct.get("vcs_info") or {}
    assert vcs.get("vcs") == "git"
    assert vcs.get("commit_id") == pin["commit"]
    assert vcs.get("requested_revision") == pin["commit"]


def _declared_cb_lane_profiles() -> set[str]:
    """Architectures whose model profile offers the `nvfp4_cb` export lane."""
    from prismaquant.model_profiles import registry

    declared = set()
    for cls in registry._REGISTERED:
        profile = cls()
        if "nvfp4_cb" in profile.supported_export_lanes():
            declared.add(profile.name)
    return declared


def _assert_declared_cb_lanes_are_servable() -> None:
    """The directional check, shared with its negative control below."""
    declared = _declared_cb_lane_profiles()
    supported = set(
        _runtime_contract()["producer_profiles"]["supported_ids"])
    unservable = declared - supported
    assert not unservable, (
        "these architectures declare the nvfp4_cb export lane but the pinned "
        f"Gridbook runtime does not serve them: {sorted(unservable)} "
        f"(runtime serves {sorted(supported)})")


def test_declared_cb_lanes_are_a_subset_of_gridbook_supported():
    """The producer must never declare a lane the pinned runtime cannot serve.

    The invariant is DIRECTIONAL. Only one of the two differences is a defect.

    DANGEROUS — `declared - supported`: a model profile offers the `nvfp4_cb`
    lane for an architecture the pinned Gridbook runtime does not serve. Nothing
    crashes, which is the whole problem. Per
    `ModelProfile.supported_export_lanes`, where that wiring is missing "the run
    still *completes* and the artifact serves uninitialised memory — coherent
    garbage, not a crash" (commit `9a79963`, Laguna). Catching that is why this
    test exists; the negative control below proves it still does.

    NORMAL — `supported - declared`: the runtime serves an architecture this
    producer cannot yet export. That is the ordinary consumer-first leading
    state, since a serving contract has to exist before any artifact can be
    produced against it.

    This asserted strict EQUALITY until Gridbook 0.7.0 registered
    `deepseek_v4` before PrismaQuant's streaming CB exporter, source-format
    passthrough, and packed-expert path were complete. The producer lane is now
    declared, but containment remains the right invariant: consumer-first
    additions must remain representable without weakening this safety gate.
    """
    _assert_declared_cb_lanes_are_servable()


def test_routed_moe_per_role_lut_capability_matches_the_pinned_contract():
    """The version gate and Gridbook's packaged ABI declaration must agree."""
    from prismaquant.gridbook_runtime_pin import (
        load_gridbook_runtime_pin,
        supports_routed_moe_per_role_codebook_lut,
    )

    supported = supports_routed_moe_per_role_codebook_lut(
        load_gridbook_runtime_pin()
    )
    features = _runtime_contract().get("abi_features", {})
    assert features.get("routed_moe_per_role_codebook_lut") == (
        1 if supported else None
    )


def test_source_fp8_block128_w8a16_capability_matches_the_pinned_contract():
    from prismaquant.gridbook_runtime_pin import (
        load_gridbook_runtime_pin,
        supports_source_fp8_block128_w8a16,
    )

    pin = load_gridbook_runtime_pin()
    contract = _runtime_contract()
    assert contract.get("schema") == pin.runtime_contract_schema
    assert supports_source_fp8_block128_w8a16(pin)
    assert contract.get("abi_features", {}).get(
        "source_fp8_block128_w8a16"
    ) == 1


def test_a_declared_lane_the_runtime_cannot_serve_still_fails(monkeypatch):
    """Negative control for the relaxation above.

    Containment has to stay a real gate rather than a check that passes because
    it happens to subtract in the harmless direction. Fabricate one extra
    DECLARED lane for an architecture no runtime contract lists, and the check
    must fail and name it."""
    from prismaquant.model_profiles import registry

    class _UnservableProfile:
        name = "fabricated_arch_no_runtime_serves"

        def supported_export_lanes(self):
            return ("compressed-tensors", "nvfp4_cb")

    monkeypatch.setattr(
        registry, "_REGISTERED", [*registry._REGISTERED, _UnservableProfile])

    assert _UnservableProfile.name in _declared_cb_lane_profiles()
    with pytest.raises(AssertionError, match=_UnservableProfile.name):
        _assert_declared_cb_lanes_are_servable()


def test_cb_rungs_layouts_and_quant_method_fit_the_runtime_contract():
    from prismaquant.cb_layout import (
        CODEWORDS_PER_SUPERBLOCK,
        INDEX_BIT_ORDER,
        INDEX_BYTES_PER_K,
        SCALE_CODING_TWO_TIER,
        SCALE_CODING_V1,
        SUBINDEX_SPLIT,
        SUPERBLOCK,
        VEC_DIM,
        bit_split,
        type_size,
    )
    from prismaquant.format_registry import list_formats, list_producer_formats

    contract = _runtime_contract()
    assert contract["quant_method"]["canonical"] == "gridbook"
    assert "prismaquant" in contract["quant_method"]["legacy"]
    by_family = {entry["family"]: entry for entry in contract["formats"]}

    names = {spec.name for spec in list_formats()}
    accepted_k = {
        int(name.rsplit("K", 1)[1]) for name in names
        if name.startswith("NVFP4_CB_K")
    }
    producer_s = {
        int(name.rsplit("S", 1)[1]) for name in names
        if name.startswith("NVFP4_CB_S")
    }
    accepted_fp8 = {
        int(name.rsplit("K", 1)[1]) for name in names
        if name.startswith("FP8_CB_K")
    }
    # Reader compatibility is directional while the producer and consumer are
    # released independently. The installed pin's public rows must all remain
    # understood by this producer; a validation branch may additionally stage
    # candidate rows whose exact target profile remains unreleasable until the
    # external pin advances. Candidate equality is checked separately against
    # the candidate Gridbook checkout, never asserted against this old pin.
    assert set(by_family["NVFP4_CB_K"]["rungs"]) <= accepted_k
    if "NVFP4_CB_S" in by_family:
        assert producer_s <= set(by_family["NVFP4_CB_S"]["rungs"])
    else:
        assert not producer_s
    # Registry membership is likewise the backwards-compatible FP8 reader
    # inventory. The current wheel knows only historical K28..K48; contracts
    # from v11 on explicitly distinguish their wider reader domain from
    # producer_rungs.
    assert set(by_family["FP8_CB_K"]["rungs"]) <= accepted_fp8
    # ``producer_rungs`` first arrived with a real value at the 0.9.1/v12 pin
    # (v4 published none, so this branch lay dormant and untested). It carries
    # TWO directions, and only one of them belongs here.
    #
    #   runtime.producer_rungs <= prismaquant producer menu
    #       "everything the pinned runtime attests for production, this
    #       producer can actually build."  Nothing else checks this, so it is
    #       the assertion below and it stays an assertion.
    #
    #   prismaquant producer menu <= runtime.producer_rungs
    #       "this producer offers nothing the runtime does not attest."  That
    #       is a MENU BAN, and principle 1 vetoes it in those words: the
    #       platform "never removes an honestly priced rung from the menu --
    #       an allocator that wants an unbacked route is reporting a serving
    #       gap, and that signal is the point."  Its legitimate enforcement
    #       point is principle 9's per-artifact export gate, and that gate is
    #       demonstrably live: under this same pin, exporting a CB body whose
    #       units the table does not cover raises CBRouteStatusRefusal (see
    #       tests/test_cb_route_status_gate.py, and the tiny-export fixture in
    #       tests/test_gridbook_artifact_conformance.py, which has to declare a
    #       non-native target to get past it).  Asserting equality here would
    #       amputate the shipped DSv4 FP8_CB_K28/K32 recipe and the K1..K4
    #       research band to satisfy a producer-policy question that is Rob's
    #       to answer, not a pin bump's.
    #
    # Note for whoever reads this next: at 0.9.1 the prismaquant fp8_cb
    # producer menu reaches DOWN to K4, while the runtime's *reader* domain
    # starts at K28 -- so K4..K24 are rungs no released Gridbook can decode at
    # all.  That gap predates this pin and was invisible under v4; it is
    # recorded, deliberately not closed here.
    if "producer_rungs" in by_family["FP8_CB_K"]:
        producer_fp8 = {
            int(spec.name.rsplit("K", 1)[1])
            for spec in list_producer_formats("fp8_cb")
        }
        assert set(by_family["FP8_CB_K"]["producer_rungs"]) <= producer_fp8
    if "producer_rungs" in by_family["NVFP4_CB_K"]:
        producer_nvfp4 = {
            int(spec.name.rsplit("K", 1)[1])
            for spec in list_producer_formats("nvfp4_cb")
        }
        assert set(by_family["NVFP4_CB_K"]["producer_rungs"]) <= producer_nvfp4

    packing = contract["packing"]
    assert packing == {
        "vector_dim": VEC_DIM,
        "superblock_weights": SUPERBLOCK,
        "codewords_per_superblock": CODEWORDS_PER_SUPERBLOCK,
        "index_bytes_per_k": INDEX_BYTES_PER_K,
        "index_bit_order": INDEX_BIT_ORDER,
        "subindex_split": SUBINDEX_SPLIT,
    }
    # Pin the named split rule to the producer implementation, including odd
    # rungs where ceil-first versus floor-first changes every sidecar shape.
    assert bit_split(13, 2) == (7, 6)
    assert bit_split(29, 4) == (8, 7, 7, 7)

    expected_family_fields = {
        "NVFP4_CB_K": ("NVFP4_CB_K{k}", "fp4", "product", 2),
        "NVFP4_CB_S": ("NVFP4_CB_S{k}", "fp4", "signed", 1),
        "FP8_CB_K": ("FP8_CB_K{k}", "fp8", "product", 4),
    }
    # These four fields describe a CB *product-code* family and only that: a
    # name pattern, a base grid, a signed/product mode and a sub-index count.
    # From v12 the contract also publishes trellis families, whose entries
    # carry a different shape entirely (candidate_rungs_q256,
    # native_terminal_q256, reader_rate_range_q256, residency_modes -- and no
    # grid, mode, n_sub or rungs). They are out of scope here BY SHAPE, not by
    # being skipped: the closure assert below still fails if the runtime ever
    # publishes a cb_product family this test has no expectation for.
    # ``kind`` arrived with v11; contracts before it published CB families
    # only, so its absence means cb_product and the check is unchanged there.
    cb_families = {
        family for family, entry in by_family.items()
        if entry.get("kind", "cb_product") == "cb_product"
    }
    assert cb_families <= set(expected_family_fields), (
        "the pinned runtime publishes a CB family this test does not pin: "
        f"{sorted(cb_families - set(expected_family_fields))}"
    )
    for family in cb_families:
        entry = by_family[family]
        pattern, grid, mode, n_sub = expected_family_fields[family]
        assert (entry["name_pattern"], entry["grid"], entry["mode"],
                entry["n_sub"]) == (pattern, grid, mode, n_sub)

    layout = contract["layout"]
    assert layout["supported"] == [1, 2]
    assert layout["default_when_absent"] == 1
    assert layout["field"] == "layout_version"
    assert layout["scale_coding_field"] == "scale_coding.kind"
    assert layout["scale_coding_default_when_absent"] == SCALE_CODING_V1
    rules = {
        (rule["grid"], rule["layout_version"], rule["scale_coding"]):
            rule["scale_plane_bytes"]
        for rule in layout["type_size_rules"]
    }
    assert rules == {
        ("fp4", 1, SCALE_CODING_V1): 16,
        ("fp4", 2, SCALE_CODING_TWO_TIER): 9,
        ("fp8", 1, SCALE_CODING_V1): 0,
    }
    # Same scoping as the shape check above: layout_versions/grid/rungs are
    # CB product-code fields; a trellis family publishes none of them.
    for family in cb_families:
        entry = by_family[family]
        for version in entry["layout_versions"]:
            coding = (SCALE_CODING_TWO_TIER
                      if entry["grid"] == "fp4" and version == 2
                      else SCALE_CODING_V1)
            scale_bytes = rules[(entry["grid"], version, coding)]
            for k in entry["rungs"]:
                assert type_size(k, entry["grid"], coding) == (
                    packing["index_bytes_per_k"] * k + scale_bytes
                ), (family, version, k)

    assert by_family["NVFP4_CB_K"]["layout_versions"] == [1, 2]
    assert by_family["NVFP4_CB_K"]["moe_layout_versions"] == [2]
    if "NVFP4_CB_S" in by_family:
        assert by_family["NVFP4_CB_S"]["layout_versions"] == [1, 2]
        assert by_family["NVFP4_CB_S"]["moe_layout_versions"] == [2]
    assert by_family["FP8_CB_K"]["layout_versions"] == [1]
    assert by_family["FP8_CB_K"]["moe_layout_versions"] == [1]
