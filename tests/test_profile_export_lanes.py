"""R6 (reader half) — export-lane eligibility as model configuration.

`EXPORT_CONTAINER` is an operator env var with no relationship to whether the
architecture is wired for that lane. Nothing stopped `EXPORT_CONTAINER=<lane>`
on an arch whose expert loader for that lane is a TODO: the run completes, the
artifact serves, and the FusedMoE reads uninitialised memory — coherent-looking
garbage, not a crash (commit `9a79963`, Laguna, 93% of parameters).

This file pins the spec fields, profile accessors, and the preflight helper
wired by `run-pipeline.sh`.

The lane vocabulary lost `nvfp4_cb` on 2026-09-02 when the Gridbook codebook
lane was retired (`archive/gridbook_lane_2026-09-02/`). The cases below were
re-pointed at the lanes that remain rather than deleted: the *mechanism* — an
architecture declares its lanes, and the preflight refuses an undeclared one —
is exactly what the retirement leaves standing, and it is what the Tessera
lane will be admitted through.
"""
from __future__ import annotations

import pytest

from prismaquant.model_profiles import detect_profile  # noqa: F401  (API shape)
from prismaquant.model_profiles import registry as _registry
from prismaquant.lane_spec import (
    all_lane_specs,
    lane_spec_for_container,
)
from prismaquant.model_profiles.structure import (
    DEFAULT_EXPORT_LANE,
    EXPORT_LANES,
    RETIRED_EXPORT_LANES,
    ModelStructureSpec,
    SCHEMA,
    canonical_export_lane,
    load_structure_spec,
)
from prismaquant.shipcard import ALL_SLOTS
from prismaquant.serving_profiles import (
    load_serving_profile,
    require_lane_supported,
    require_profile_export_lane,
)

# A serving runtime's supported producer IDs are never duplicated here: the
# pinned runtime publishes them in its own runtime_contract.json and the
# comparison is made against that one machine-readable table (AGENTS.md
# principle 5 / CLAUDE.md principle 14). The Gridbook half of that comparison
# retired with its lane on 2026-09-02.
#
# `GGUF_WIRED` / `TESSERA_WIRED` lived here until 2026-09-03: two module-level
# sets named after two specific lanes, asserted by name in one test. That is
# the roster-instead-of-rule shape (the eighth instance across these two
# repositories), and its cost is precise: a THIRD non-default lane would have
# been covered by nothing, because the test named two lanes and said nothing
# about any other. The rosters moved into `lane_specs/<lane>.json`'s
# `wired_architectures`, beside everything else about that lane, and the
# assertions below are properties quantified over EXPORT_LANES.
#
# `wired_architectures` moved the roster beside the lane, which is the right
# home for it, and the agreement below is two files written by two authors
# answering one question. What neither file can answer is whether the wiring
# EXISTS: there is no per-architecture lane-wiring table in this repository
# (GGUF's architecture comes out of llama.cpp's own skeleton at export time,
# and Tessera's gate constrains structure -- "dense" -- rather than
# architecture), so a lockstep edit across the two files still satisfies the
# agreement, which is the Laguna failure mode (`9a79963`) in two files instead
# of one. #150 therefore adds a THIRD conjunct, derived from a place neither
# roster maintains: `serving_profile_specs/*.json` declare an `export_lane`,
# layered through `extends`, so an architecture's `default_serving_profile`
# resolves to exactly one lane, and `lane_specs/*.json` say which serving
# profiles serve a lane. An architecture whose serving profile goes to
# compressed-tensors cannot declare the GGUF lane, whatever either roster says.
#
# It is a weaker claim than "the exporter is wired for this arch" and says so.
# The Tessera lane is the honest hole: its own spec declares NO serving
# profiles, because the plugin is chosen by the checkpoint's
# `quantization_config.quant_method`, so the third conjunct has nothing to bite
# on there and `qwen3`'s declaration rests on the two rosters alone. That
# absence is read out of the lane spec below rather than encoded as a name to
# skip.


def _resolved_serving_lane(profile) -> "str | None":
    """The export lane this architecture's own serving route reaches.

    Read through `load_serving_profile`, so `extends` inheritance answers --
    `vllm_glm5_next_packed_moe` declares no `export_lane` of its own and
    reaches compressed-tensors through `vllm_packed_moe`.
    """
    spec = load_structure_spec(profile.name)
    profile_id = getattr(spec, "default_serving_profile", None) if spec else None
    if not profile_id:
        return None
    lane = load_serving_profile(profile_id).export_lane
    return None if lane is None else canonical_export_lane(lane.id)


def _lanes_keyed_by_a_serving_profile() -> frozenset:
    """The lanes whose own spec names the serving profiles that serve them."""
    return frozenset(
        spec.export_container for spec in all_lane_specs()
        if spec.serving_profiles
    )


PROFILE_CLASSES = list(_registry._REGISTERED)
PROFILE_IDS = [c.__name__ for c in PROFILE_CLASSES]
LANE_IDS = list(EXPORT_LANES)


def _profile(cls):
    return cls()


# ------------------------------------------------------------------ vocabulary


def test_the_lane_vocabulary_and_the_lane_declarations_are_one_set():
    """PROPERTY, not a roster.

    `EXPORT_LANES` is where Rob's decision of 2026-09-02 lives -- the roster
    IS the input and there is nothing in the code to derive it from, so it is
    declared in exactly ONE place (`model_profiles/structure.py`). What this
    test used to do was re-type that place's contents into an assertion, which
    made the roster the specification: adding the sanctioned third lane broke
    a test that was supposed to COVER it. (`assert EXPORT_LANES ==
    ("compressed-tensors", "gguf")` was the state #116 reported.)

    What is checked instead is the closure: every lane in the vocabulary has a
    declaration, and every declaration names a lane in the vocabulary. Two
    files, two authors, one set -- so an orphan spec and a lane with no gates
    are both caught, and a fourth lane is covered by adding its spec rather
    than by editing this.
    """
    declared = {spec.export_container for spec in all_lane_specs()}
    assert declared == set(EXPORT_LANES), (
        "lane_specs/*.json and EXPORT_LANES disagree: "
        f"specs-only={sorted(declared - set(EXPORT_LANES))} "
        f"vocabulary-only={sorted(set(EXPORT_LANES) - declared)}")
    assert DEFAULT_EXPORT_LANE in EXPORT_LANES
    # One declared alias: the serving-profile side spells the native lane with
    # an underscore (`export_lane.id == "compressed_tensors"`).
    assert canonical_export_lane("compressed_tensors") == "compressed-tensors"


@pytest.mark.parametrize("lane", LANE_IDS)
def test_every_lane_in_the_vocabulary_round_trips(lane):
    assert canonical_export_lane(lane) == lane
    assert lane_spec_for_container(lane).export_container == lane


def test_a_lane_outside_the_vocabulary_is_refused():
    """Any non-member, generated rather than named, plus every retired lane.

    The generated name is the rule; the retired ones are the regression cases
    the rule has to keep catching. `RETIRED_EXPORT_LANES` is beside
    `EXPORT_LANES` in the same module, so retiring a lane moves a name between
    two rosters in one edit and adding a lane touches neither this test nor
    that dict.
    """
    with pytest.raises(ValueError, match="unknown export lane"):
        canonical_export_lane("no-such-lane-" + "x" * 8)
    assert not (set(RETIRED_EXPORT_LANES) & set(EXPORT_LANES)), (
        "a lane cannot be both live and retired")
    for lane, wall in RETIRED_EXPORT_LANES.items():
        with pytest.raises(ValueError, match="unknown export lane") as excinfo:
            canonical_export_lane(lane)
        # The refusal names the wall, so an operator meeting a stale driver
        # learns where the code went instead of suspecting a typo.
        assert wall in str(excinfo.value)


@pytest.mark.parametrize("lane", LANE_IDS)
def test_every_lane_declares_a_gate_set_that_can_refuse(lane):
    """A declared gate either closes a known shipcard slot or says why not.

    This is principle 9 on the gate list itself: a gate whose result is
    recorded nowhere is refused on by nothing, and until 2026-09-03 the
    Tessera lane's `route.census` -- principle 12's second leg -- was exactly
    that (`shipcard_slot: null`, no reason, nothing reading it).
    `LaneGate.from_dict` now refuses the ambiguous form; this pins that every
    live lane is on the right side of it.
    """
    spec = lane_spec_for_container(lane)
    assert spec.gates, f"{lane}: a lane with no declared gates declares no bar"
    for gate in spec.gates:
        if gate.recorded:
            assert gate.shipcard_slot in ALL_SLOTS, (
                f"{lane}/{gate.id}: closes {gate.shipcard_slot!r}, which is "
                "not a shipcard slot, so no card can record it")
        else:
            assert gate.unrecorded_reason, (
                f"{lane}/{gate.id}: unrecorded and unexplained")


@pytest.mark.parametrize("lane", LANE_IDS)
def test_every_lane_that_shells_out_declares_what_it_shells_out_to(lane):
    """External build-tool dependencies are a value, not a bash loop.

    A lane may name another repository's tool rather than vendor it -- that is
    the boundary that keeps one wire recipe in one home. What it may not do is
    depend on a file nobody can enumerate: the Tessera arm's two
    `experiments/` scripts were a hardcoded loop in `run-pipeline.sh` and a
    sentence in the spec's `notes` (RobTand/prismaquant#119).
    """
    spec = lane_spec_for_container(lane)
    for tool in spec.producer_tools:
        assert tool.stability in tool.STABILITIES
        if tool.stability != "supported":
            assert tool.tracking_issue, (
                f"{lane}: {tool.path} has no stability promise and names no "
                "tracking issue")


def test_preferred_lane_must_be_supported():
    with pytest.raises(ValueError, match="preferred_lane"):
        ModelStructureSpec.from_dict({
            "schema": SCHEMA,
            "id": "lane_test",
            "supported_lanes": ["compressed-tensors"],
            "preferred_lane": "gguf",
        })


# --------------------------------------------------------------- declarations


@pytest.mark.parametrize("cls", PROFILE_CLASSES, ids=PROFILE_IDS)
@pytest.mark.parametrize("lane", LANE_IDS)
def test_a_profile_declares_a_lane_exactly_when_the_lane_declares_it(cls, lane):
    """Two files, two authors, one answer -- for EVERY lane in the vocabulary.

    The profile side says which lanes an architecture supports; the lane side
    says which architectures are wired for it. Neither is derivable from the
    other, so both are declared and this is the agreement. The predecessor
    checked `gguf` and `tessera` by name against two sets defined in this
    file, which is why a fourth lane would have escaped it entirely.
    """
    profile = _profile(cls)
    lanes = set(profile.supported_export_lanes())
    assert DEFAULT_EXPORT_LANE in lanes, (
        f"{profile.name}: every architecture ships through the native lane")
    spec = lane_spec_for_container(lane)
    assert (lane in lanes) == spec.wires(profile.name), (
        f"{profile.name}: declares lane {lane!r}={lane in lanes} but "
        f"lane_specs/{spec.id}.json's wired_architectures says "
        f"{spec.wires(profile.name)} "
        f"(roster: {sorted(spec.wired_architectures)})")


@pytest.mark.parametrize("cls", PROFILE_CLASSES, ids=PROFILE_IDS)
def test_a_declared_lane_is_one_this_archs_serving_route_reaches(cls):
    """The third conjunct: a place neither roster maintains (#150).

    Every architecture ships through the native lane by contract. Every *other*
    declared lane has to be a lane this architecture's own
    `default_serving_profile` resolves to -- for the lanes whose spec keys on a
    serving profile at all. Two rosters edited in lockstep agree with each
    other; they do not agree with this.
    """
    profile = _profile(cls)
    lanes = set(profile.supported_export_lanes())
    arch_keyed = _lanes_keyed_by_a_serving_profile()
    # Non-vacuity: with no lane keyed on a serving profile the loop below would
    # be empty for every profile and this test would pass on nothing.
    assert len(arch_keyed) >= 2, sorted(arch_keyed)
    reached = _resolved_serving_lane(profile)
    for lane in sorted(lanes - {DEFAULT_EXPORT_LANE}):
        if lane not in arch_keyed:
            continue                    # the Tessera hole, pinned below
        assert reached == lane, (
            f"{profile.name}: declares the {lane!r} export lane, but its "
            f"serving route reaches {reached!r} -- no in-tree serving profile "
            f"takes this architecture to {lane!r}, so the declaration promises "
            f"bytes nothing loads")


def test_a_serving_route_that_reaches_a_lane_is_declared_by_that_arch():
    """Reverse half: the declaration cannot go missing either.

    An architecture whose serving profile resolves to a lane must declare it,
    or `require_lane_supported` refuses the only container that architecture is
    set up to serve -- by reading the declaration, which is the failure this
    direction catches.
    """
    arch_keyed = _lanes_keyed_by_a_serving_profile()
    checked = 0
    for cls in PROFILE_CLASSES:
        profile = _profile(cls)
        reached = _resolved_serving_lane(profile)
        if reached is None or reached not in arch_keyed:
            continue
        assert reached in profile.supported_export_lanes(), (
            f"{profile.name}: its serving route reaches the {reached!r} lane "
            f"but the profile declares "
            f"{sorted(profile.supported_export_lanes())}")
        checked += 1
    # Non-vacuity, and not a count of today's registry: two is what it takes
    # for "every" to mean anything, and the registry only grows.
    assert checked >= 2, checked


def test_every_declared_lane_resolves_to_the_consumer_it_names():
    """A declaration names a lane the tree can actually run.

    The lane's round trip and its gate set are pinned above; what is checked
    here is the one consumer every lane spec names in code -- the KL evaluator
    -- imports and holds the attribute. A lane that names a consumer nobody
    wrote fails here rather than at frontier-selection time.
    """
    import importlib

    declared = {
        lane for cls in PROFILE_CLASSES
        for lane in _profile(cls).supported_export_lanes()
    }
    assert len(declared) >= 2, sorted(declared)
    for lane in sorted(declared):
        spec = lane_spec_for_container(lane)
        module, _, attr = spec.kl_evaluator.entrypoint.partition(":")
        assert hasattr(importlib.import_module(module), attr), (
            f"{lane}: kl_evaluator {spec.kl_evaluator.entrypoint} does not "
            f"resolve")


def test_the_tessera_lane_declares_that_it_is_not_keyed_by_serving_profile():
    """The one lane the third conjunct cannot witness, read and not assumed.

    `_lanes_keyed_by_a_serving_profile` skips Tessera because Tessera's own lane
    spec declares no serving profiles: the plugin is chosen by the checkpoint's
    `quant_method`, so there is no serving-profile hop for a model profile's
    declaration to be checked against. If Tessera ever gains one, the skip in
    the forward test above stops applying on its own and `qwen3`'s declaration
    starts being checked -- which is why the absence is asserted here rather
    than encoded as a lane name to ignore.
    """
    tessera = lane_spec_for_container("tessera")
    assert tessera.serving_profiles == (), tessera.serving_profiles
    assert "tessera" not in _lanes_keyed_by_a_serving_profile()
    # And the other two are, so this is a property of Tessera rather than of
    # the accessor returning nothing for everybody.
    assert _lanes_keyed_by_a_serving_profile() == frozenset(
        {"compressed-tensors", "gguf"})


@pytest.mark.parametrize("cls", PROFILE_CLASSES, ids=PROFILE_IDS)
def test_preferred_lane_is_supported_and_defaults_to_native(cls):
    profile = _profile(cls)
    preferred = profile.preferred_export_lane()
    assert preferred in profile.supported_export_lanes()
    spec = load_structure_spec(profile.name)
    if spec is None or not spec.preferred_lane:
        assert preferred == DEFAULT_EXPORT_LANE


def test_the_script_driven_lane_is_declared():
    """The GGUF lane is tribal knowledge today; hy_v3 is the arch whose
    shipped artifact came off it.

    Laguna's `preferred_lane` was `nvfp4_cb` until 2026-09-02 and is now the
    default native lane: its shipped artifact came off the retired codebook
    lane, and it has no other declared preference.
    """
    assert load_structure_spec("hy_v3").preferred_lane == "gguf"
    assert load_structure_spec("laguna").preferred_lane == "compressed-tensors"


# ------------------------------------------------------------------- preflight


def test_require_lane_supported_accepts_declared_lanes():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
    from prismaquant.model_profiles.hy_v3 import HyV3Profile
    from prismaquant.model_profiles.laguna import LagunaProfile
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    assert require_lane_supported(HyV3Profile(), "gguf") == "gguf"
    assert require_lane_supported(
        DeepseekV4Profile(), "compressed-tensors") == "compressed-tensors"
    assert require_lane_supported(LagunaProfile(), None) == "compressed-tensors"
    assert require_lane_supported(
        Qwen3_5Profile(), "compressed_tensors") == "compressed-tensors"


def test_require_lane_supported_refuses_an_undeclared_lane():
    from prismaquant.model_profiles.gemma4 import Gemma4Profile

    # `gguf` since 2026-09-02 — `nvfp4_cb` is no longer *undeclared*, it is
    # unknown, and that is a different refusal (below).
    with pytest.raises(SystemExit) as excinfo:
        require_lane_supported(Gemma4Profile(), "gguf")
    message = str(excinfo.value)
    assert "gemma4" in message
    assert "compressed-tensors" in message      # names the declared set
    assert "garbage" in message                 # names the failure mode


def test_require_lane_supported_refuses_an_unknown_lane():
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    with pytest.raises(SystemExit, match="unknown export lane"):
        require_lane_supported(Qwen3_5Profile(), "nvfp4-cb")


def test_require_lane_supported_is_inert_without_the_accessor():
    """Duck-typed like `resolve_target_profile`: a profile object that predates
    the accessor must not break a run."""
    class _Legacy:
        name = "legacy"

    assert require_lane_supported(_Legacy(), "gguf") == "gguf"


# `test_narrow_4090_profile_inherits_cb_serializer_by_lane_identity` was
# deleted on 2026-09-02: it loaded `qwen38_rtx4090_fp8_cb`, one of the six CB
# producer profiles, all of which went to archive/gridbook_lane_2026-09-02/
# with the lane. `require_profile_export_lane` itself is still exercised by
# `test_serving_profiles.py`; what is gone is the only lane whose serving
# profile pinned a non-default `export_lane.id`.


def test_no_shipped_lane_run_becomes_illegal():
    """Non-regression: every in-tree launch script's (arch, EXPORT_CONTAINER)
    pair must still pass the preflight.

    The `nvfp4_cb` pairs — qwen3_5, qwen3_5_dense, hy_v3, laguna, deepseek_v4 —
    were removed on 2026-09-02 along with their launch scripts. Those runs ARE
    now illegal, deliberately: `canonical_export_lane` refuses the container
    before the profile is consulted (pinned above).
    """
    cases = [
        ("hy_v3", "gguf"),
        ("qwen3_5", "compressed-tensors"),
        ("gemma4", "compressed-tensors"),
        ("lfm2_moe", "compressed-tensors"),
        ("minimax_m2", "compressed-tensors"),
        ("deepseek_v4", "compressed-tensors"),
    ]
    by_name = {c().name: c() for c in PROFILE_CLASSES}
    for name, lane in cases:
        assert require_lane_supported(by_name[name], lane) == lane
