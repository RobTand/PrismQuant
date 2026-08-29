"""Conformance run of `prismaquant/model_profiles/validate.py` over every
registered ModelProfile.

`validate.py` is a manual CLI with (until this file) zero callers, which is
how the defects it exists to catch survived inside it. This test pins the
CPU-safe part of it plus the invariants that de-vacuum the checks that
short-circuit to green without vLLM.

Lanes:
  - default: pure python/CPU. No model weights, no GPU, no network. Runs
    checks 1, 6 (against synthetic index fixtures) and 8, plus four
    structural invariants (spec presence, fused-sibling source, registry
    order, name uniqueness) and the R26 declaration-conformance family
    (every declared spec key is parsed; every parsed field is read).
  - `integration`: the vLLM-registry checks (2/3/4), skipped when vLLM is
    not importable. Their answer is vLLM-version-dependent, so they are not
    part of the default lane.
  - `slow`: the safetensors-index checks (6/7) against real checkpoints
    named by $PQ_CONFORMANCE_MODELS.

Check 5 (MTP) is deliberately absent: `build_mtp_module()` materialises a
full decoder layer (multi-GB CPU allocation). Use the manual CLI for it:

    python -m prismaquant.model_profiles.validate --model /path/to/Model

Its cheap declarative half — `has_mtp()` implies an actual
`build_mtp_module()` override plus an `mtp_source_prefix()` — IS covered
here (`test_has_mtp_implies_a_buildable_mtp_module`), which is the part
that catches the D2/§8.5-L2 defect class.

Known gaps are encoded as *ratchets*, not bare xfails: each one first
asserts the gap is still real and only then xfails. Closing the gap turns
the test red with an instruction to shrink the list, so the exemption
cannot go stale silently.
"""
from __future__ import annotations

import ast
import dataclasses
import functools
import json
import os
import pathlib
import re
import struct

import pytest

import prismaquant
from prismaquant import serving_profiles as SPROF
from prismaquant.model_profiles import registry as _registry
from prismaquant.model_profiles import structure as STRUCT
from prismaquant.model_profiles import validate as V
from prismaquant.model_profiles.base import ModelProfile
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.model_profiles.laguna import LagunaProfile
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

PROFILE_CLASSES = list(_registry._REGISTERED) + [DefaultProfile]
PROFILE_IDS = [c.__name__ for c in PROFILE_CLASSES]

# Representative (model_type, architectures) per profile, so check 1 and the
# resolution check run with no checkpoint on disk. Every entry below was
# taken from a real config.json where one exists locally.
REPRESENTATIVE_CONFIGS: dict[str, tuple[tuple[str, list[str]], ...]] = {
    "Qwen3Profile": (
        ("qwen3", ["Qwen3ForCausalLM"]),
        ("qwen3_moe", ["Qwen3MoeForCausalLM"]),
    ),
    "Qwen3_5DenseProfile": (("qwen3_5", ["Qwen3_5ForConditionalGeneration"]),),
    "Qwen3_5Profile": (
        ("qwen3_5_moe", ["Qwen3_5MoeForConditionalGeneration"]),
    ),
    "Qwen4ExpProfile": (
        ("qwen4_exp", ["Qwen4ExpForConditionalGeneration"]),
    ),
    # Gemma 4 ships two config flavours; the profile claims both.
    "Gemma4Profile": (
        ("gemma4", ["Gemma4ForConditionalGeneration"]),
        ("gemma4_unified", ["Gemma4UnifiedForConditionalGeneration"]),
    ),
    "Lfm2MoeProfile": (("lfm2_moe", ["Lfm2MoeForCausalLM"]),),
    "MiniMaxM2Profile": (("minimax_m2", ["MiniMaxM2ForCausalLM"]),),
    "DeepseekV4Profile": (("deepseek_v4", ["DeepseekV4ForCausalLM"]),),
    "HyV3Profile": (("hy_v3", ["HYV3ForCausalLM"]),),
    "LagunaProfile": (("laguna", ["LagunaForCausalLM"]),),
}

CONFIG_CASES = [
    (cls_name, cfg)
    for cls_name in sorted(REPRESENTATIVE_CONFIGS)
    for cfg in REPRESENTATIVE_CONFIGS[cls_name]
]
CONFIG_IDS = [f"{n}-{cfg[0]}" for n, cfg in CONFIG_CASES]

# DefaultProfile is the terminal fallback for architectures nobody claims;
# specs are keyed by profile name, so a `specs/default.json` would describe
# no architecture. Its absence is by design, not a gap.
SPEC_EXEMPT_BY_DESIGN = {"DefaultProfile"}

# Known gaps (ratcheted — see module docstring).
# Ratchet closed 2026-07-30 (R22): `specs/minimax_m2.json` now expresses all
# eight of that profile's overrides, so every registered profile has a spec.
# `tests/test_minimax_m2_spec.py` is the equivalence gate that had to be green
# before the Python bodies may be deleted.
NO_SPEC_XFAIL: set[str] = set()
ROLE_COMPOSITE_FUSED_SOURCE_EXEMPT = {
    # Intentional, lane-aware exception. DeepSeek-V4's Gridbook consumer can
    # construct a merged Linear as independent role decoders, so its producer
    # spec must not globally force those roles to one format. A native
    # compressed-tensors constraint belongs to that lane's exporter, not this
    # architecture-wide accessor.
    "DeepseekV4Profile",
}

FUSED_PROBE_NAMES = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
)


def _instantiate(cls):
    if cls is DefaultProfile:
        return cls(architectures=["LlamaForCausalLM"])
    return cls()


@pytest.fixture(scope="module", params=PROFILE_CLASSES, ids=PROFILE_IDS)
def profile(request):
    return _instantiate(request.param)


# ---------------------------------------------------------------- CPU lane


def test_every_registered_profile_instantiates(profile):
    assert profile.name


def test_profile_names_are_unique():
    """Profile names key `specs/<name>.json` (base.py:909), so a collision
    silently hands one profile another's structure spec."""
    names = [_instantiate(c).name for c in PROFILE_CLASSES]
    assert len(names) == len(set(names)), names


def test_registry_order_is_stable():
    """registry.py:47-50 documents these precedences in comments; assert
    them. Getting profile resolution wrong is how the fused-coherence bug
    shipped unservable artifacts (DefaultProfile -> mixed-scheme fused
    groups)."""
    order = [c.__name__ for c in _registry._REGISTERED]
    assert order.index("Qwen3_5DenseProfile") < order.index("Qwen3_5Profile")
    assert order.index("Qwen3Profile") < order.index("Gemma4Profile")


@pytest.mark.parametrize("cls_name,cfg", CONFIG_CASES, ids=CONFIG_IDS)
def test_check1_matches_representative_config(cls_name, cfg):
    """validate.py check 1 — the profile claims its own representative
    config."""
    cls = next(c for c in PROFILE_CLASSES if c.__name__ == cls_name)
    model_type, archs = cfg
    result = V._check_matches(
        _instantiate(cls),
        {"model_type": model_type, "architectures": archs},
    )
    assert result.ok, result.detail


@pytest.mark.parametrize("cls_name,cfg", CONFIG_CASES, ids=CONFIG_IDS)
def test_representative_config_resolves_to_its_own_profile(cls_name, cfg):
    """No shadowing: detect-by-config lands on the intended profile."""
    model_type, archs = cfg
    resolved = _registry.profile_from_config(
        {"model_type": model_type, "architectures": archs})
    assert type(resolved).__name__ == cls_name


def test_unknown_arch_falls_back_to_default_profile():
    resolved = _registry.profile_from_config(
        {"model_type": "definitely_not_a_real_arch",
         "architectures": ["NotARealForCausalLM"]})
    assert isinstance(resolved, DefaultProfile)


def test_check8_serving_profile_loads(profile):
    """validate.py check 8 — pure python + JSON, always safe to run."""
    result = V._check_serving_profile(profile)
    assert result.ok, result.detail


def test_profile_has_structure_spec(profile):
    """De-vacuums check 2 on CPU: the "vLLM not importable" amnesty at
    validate.py:172 only applies when a declarative spec exists."""
    name = type(profile).__name__
    if name in SPEC_EXEMPT_BY_DESIGN:
        pytest.skip(f"{name} is the terminal fallback; a spec would name "
                    "no architecture")
    has_spec = profile.structure_spec() is not None
    if name in NO_SPEC_XFAIL:
        assert not has_spec, (
            f"{name} now HAS a structure spec — remove it from "
            "NO_SPEC_XFAIL")
        pytest.xfail(f"{name} has no model_profiles/specs/*.json yet")
    assert has_spec


def test_profile_has_a_fused_sibling_source(profile):
    """De-vacuums check 3 for lanes requiring uniform fused siblings.

    Gridbook role-composite architectures are explicit exceptions: each role
    may have a different storage scheme because it is decoded independently
    into the common execution type. The ratchet below keeps that exception
    named instead of silently weakening this check for every profile.
    """
    name = type(profile).__name__
    has_vllm_cls = profile.vllm_architecture_class() is not None
    spec = profile.structure_spec()
    has_spec_groups = bool(spec is not None and spec.fused_groups)
    overrides = "fused_sibling_group" in vars(type(profile))
    has_source = has_vllm_cls or has_spec_groups or overrides
    if name in ROLE_COMPOSITE_FUSED_SOURCE_EXEMPT:
        assert not has_source, (
            f"{name} now has a fused-sibling source — either remove it from "
            "ROLE_COMPOSITE_FUSED_SOURCE_EXEMPT or explain why a global "
            "producer coupling is now lane-correct"
        )
        assert "nvfp4_cb" in profile.supported_export_lanes()
        return
    assert has_source


def test_fused_group_is_self_consistent_on_cpu(profile):
    """Spec-driven variant of check 3 that works without vLLM: all q/k/v
    siblings must map to ONE canonical key (or all to None)."""
    keys = {profile.fused_sibling_group(n) for n in FUSED_PROBE_NAMES}
    assert len(keys) == 1, f"{type(profile).__name__}: {keys}"


def test_has_mtp_implies_a_buildable_mtp_module(profile):
    """Cheap surrogate for check 5 (D11's residual, closed 2026-07-30).

    Check 5 itself materialises a full decoder layer, so it stays out of
    CI — but the defect it exists to catch is declarative and free to
    test: probe, cost and export all call `profile.build_mtp_module()`
    behind `profile.has_mtp()`, so a profile that answers True with no
    override gets None and hard-fails mid-run. `deepseek_v4` sat in
    exactly that state (`has_mtp -> True`, `build_mtp_module -> None`)
    until R12; before R12 it was worse than a crash, because the three
    call sites imported the Qwen3.5 module directly and would have
    handed DSv4 a Qwen3.5 decoder layer.

    The source-prefix half is checked too: a profile that probes MTP
    must say where the tensors live in the checkpoint.

    No ratchet list: as of R12 every `has_mtp()` profile overrides, so
    the gap is closed rather than exempted."""
    if not profile.has_mtp():
        return
    name = type(profile).__name__
    assert type(profile).build_mtp_module is not ModelProfile.build_mtp_module, (
        f"{name}.has_mtp() is True but build_mtp_module() is the base "
        "no-op returning None: probe/cost/export would fail mid-run. "
        "Either override it or take the passthrough route (has_mtp() -> "
        "False + source_passthrough_prefixes(), as hy_v3 and "
        "deepseek_v4 do)."
    )
    assert profile.mtp_source_prefix(), (
        f"{name}.has_mtp() is True but mtp_source_prefix() is empty: "
        "read_mtp_source_state_dict() has no prefix to key on."
    )


# ----------------------------------- R26: every declared field has a reader
#
# The bug class (audit R26, docs/audits/architecture_re-vet_2026-07-30.md):
# a spec declares something, nothing reads it, and the declaration silently
# means nothing. Four of the five plugin-lens findings in that audit are
# this one shape. A declaration can rot at either end, so both ends are
# pinned:
#
#   json key -> parser    a key in `specs/*.json` that `structure.py`'s
#                         `from_dict` never names is dropped at load, so
#                         the file describes a model the pipeline never
#                         sees. Caught by `..._key_is_parsed`.
#   parser -> consumer    `from_dict` lands the key on a dataclass field
#                         that no `base.py` accessor and no named consumer
#                         reads, so the value never reaches production.
#                         Caught by `..._field_has_a_reader`.
#
# Both halves are *necessary*, not sufficient. The consumer half is
# name-based — an attribute access or `getattr()` on a `…spec…`-named
# receiver — so it under-reports a field bound to an unusual name, and it
# cannot tell two dataclasses apart when they share a field name. What it
# does catch is a declaration with no reader at all, which is the failure
# mode actually on record.

_PACKAGE_ROOT = pathlib.Path(prismaquant.__file__).resolve().parent
STRUCTURE_SPEC_DIR = _PACKAGE_ROOT / "model_profiles" / "specs"
STRUCTURE_PARSER = _PACKAGE_ROOT / "model_profiles" / "structure.py"
SERVING_SPEC_DIR = _PACKAGE_ROOT / "serving_profile_specs"
SERVING_PARSER = _PACKAGE_ROOT / "serving_profiles.py"

# Which nested JSON values carry *schema* keys (so the walker descends into
# them) versus *data* keys (so it stops). The line is drawn where the
# parser draws it: `from_dict` destructures each container below by name,
# whereas e.g. `packed_experts.projection_splits` is keyed by parameter
# name and `runtime_packages[].env` by env-var name — those are values, not
# vocabulary. A container NOT listed here is still covered, because its own
# top-level key must be parsed for anything inside it to be read at all.
STRUCTURE_DESCEND_DICTS = frozenset({
    "match", "naming", "moe", "probe", "staging", "shard_regexes",
    "runtime_requirements", "packed_experts",
})
STRUCTURE_DESCEND_LISTS = frozenset({
    "fused_groups", "live_to_recipe", "recipe_to_source", "recipe_to_vllm",
    "fast_kernel_packages",
})
SERVING_DESCEND_DICTS = frozenset({"export_lane", "when", "fused_mid_m"})
SERVING_DESCEND_LISTS = frozenset({
    "format_rules", "shape_rules", "runtime_shape_validators",
    "runtime_packages", "serving_lanes",
})

# Descriptive / documentation-only keys. Both spec families already share
# one convention for these — a LEADING UNDERSCORE — used by several
# independent authors across both directories, so the exemption is that
# convention rather than a hand-maintained name list that a sixth prose key
# would turn red for no reason. The convention is load-bearing in one
# direction and enforced as such: a `_`-prefixed key must never become
# parsed (`test_doc_only_convention_holds`), or prose stops being free to
# keep honest and starts silently changing behaviour.
DOC_ONLY_KEY_PREFIX = "_"

# The inventory of prose keys as of 2026-07-30, each with the reason it is
# exempt. This is documentation of intent, not the gate — the gate is the
# prefix above. `test_doc_only_allowlists_are_live` asserts no entry has
# outlived the key it excuses, so the inventory cannot rot into a place a
# real gap could hide.
STRUCTURE_DOC_ONLY_KEYS = frozenset({
    # deepseek_v4.json — a dated provenance record of the checkpoint the
    # spec was authored against (shard dtypes, E8M0 scale planes, nibble
    # packing, the byte-accounting caveat). Prose for the next reader;
    # nothing derives behaviour from it.
    "_verified_source_layout",
    # deepseek_v4.json — why both DSv4 export lanes pin the conservative
    # `vllm_packed_moe` allocation contract. Explains a sibling key that IS
    # parsed and enforced.
    "_default_serving_profile_rationale",
    # minimax_m2.json — three notes on why the declarative form of the
    # profile's ex-Python overrides looks the way it does (the
    # mlp/block_sparse_moe scheme-dispatch rename; `param_names` holding
    # unfused vLLM leafs because MiniMax ships per-expert 2D w1/w2/w3; the
    # transformers-5.x FP8Experts rewrite that staging must bypass). Each
    # explains a sibling key that IS parsed.
    "_naming_rationale",
    "_packed_experts_rationale",
    "_staging_rationale",
})
SERVING_DOC_ONLY_KEYS = frozenset({
    # research.json — prose explaining why that profile declares no
    # `export_lane`. The behaviour it explains is carried by the sibling
    # `emulation_only: true`, which IS parsed and IS enforced.
    "_emulation_only_rationale",
})
STRUCTURE_DOC_ONLY_FIELDS = frozenset({
    # Schema-version marker. Read by `from_dict` itself as a parse-time
    # guard (structure.py raises on a version mismatch) and deliberately
    # never again — nothing downstream branches on the schema version, and
    # a consumer that did would be a bug.
    "schema",
})

# Known gaps (ratcheted — see module docstring).
STRUCTURE_FIELD_NO_READER_XFAIL: frozenset[str] = frozenset()
# Empty on purpose. `match` was the one dead field when this test was
# written (2026-07-30): every spec declared a match block and profile
# resolution ran entirely through the hand-written `matches()`
# classmethods. The declarative-detection work landed `SpecMatch.claims()`
# consumers (registry.py, spec_profile.py) the same day, so the gap closed
# before the ratchet was needed. Add entries here — dated, with the reason
# — rather than leaving this file red.
SERVING_FIELD_NO_READER_XFAIL = frozenset({
    # 2026-07-30. `runtime_packages[].module`. vllm_packed_moe.json
    # declares module="flashinfer", but the only consumer
    # (validate_native_export._flashinfer_runtime_package) reads
    # version/pip_packages/env and then hardcodes `import flashinfer`.
    # Pointing the declaration at a different module name would change
    # nothing — the import-probe it exists to drive does not read it.
    # Name-keyed, so this also exempts any other serving-spec field
    # spelled `module`; there is none today.
    "module",
})


@functools.lru_cache(maxsize=None)
def _package_sources() -> tuple[pathlib.Path, ...]:
    """Every first-party module. `vendored/` is upstream transformers code
    that cannot be a spec consumer; counting it would only manufacture
    false readers for common field names."""
    return tuple(sorted(
        p for p in _PACKAGE_ROOT.rglob("*.py")
        if "vendored" not in p.parts and "__pycache__" not in p.parts
    ))


@functools.lru_cache(maxsize=None)
def _source_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


@functools.lru_cache(maxsize=None)
def _string_constants(path: pathlib.Path) -> frozenset[str]:
    """String literals in a module — how a JSON key is named by a parser
    (`payload.get("naming")`, a MATCH_KEYS frozenset, ...)."""
    tree = ast.parse(_source_text(path))
    return frozenset(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


@functools.lru_cache(maxsize=None)
def _attribute_names(path: pathlib.Path) -> frozenset[str]:
    """Attribute names read in a module. `from_dict` builds its dataclass
    with keyword arguments, which are NOT attribute nodes, so a parser does
    not count itself as a reader of the fields it fills."""
    tree = ast.parse(_source_text(path))
    return frozenset(
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    )


def _imports_module(path: pathlib.Path, target: str) -> bool:
    for node in ast.walk(ast.parse(_source_text(path))):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[-1] == target:
                return True
        elif isinstance(node, ast.Import):
            if any(a.name.split(".")[-1] == target for a in node.names):
                return True
    return False


def _spec_key_paths(payload, descend_dicts, descend_lists, prefix=""):
    """Dotted key paths declared by one spec, to the depth the parser
    destructures. `a[].b` marks b as a key of an element of list a."""
    paths = []
    for key, value in payload.items():
        path = f"{prefix}{key}"
        paths.append(path)
        if isinstance(value, dict) and key in descend_dicts:
            paths += _spec_key_paths(
                value, descend_dicts, descend_lists, f"{path}.")
        elif isinstance(value, list) and key in descend_lists:
            for entry in value:
                if isinstance(entry, dict):
                    paths += _spec_key_paths(
                        entry, descend_dicts, descend_lists, f"{path}[].")
    return paths


def _collect_spec_keys(spec_dir, descend_dicts, descend_lists):
    """{key path: [spec files that declare it]} over a whole spec dir."""
    found: dict[str, list[str]] = {}
    for path in sorted(spec_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key_path in _spec_key_paths(payload, descend_dicts, descend_lists):
            found.setdefault(key_path, []).append(path.name)
    return found


def _leaf(key_path: str) -> str:
    return key_path.rsplit(".", 1)[-1].removesuffix("[]")


STRUCTURE_SPEC_KEYS = _collect_spec_keys(
    STRUCTURE_SPEC_DIR, STRUCTURE_DESCEND_DICTS, STRUCTURE_DESCEND_LISTS)
SERVING_SPEC_KEYS = _collect_spec_keys(
    SERVING_SPEC_DIR, SERVING_DESCEND_DICTS, SERVING_DESCEND_LISTS)

# A spec-rooted attribute chain: `spec.x`, `self._structure_spec.x`,
# `profile.structure_spec().x`, `spec.packed_experts.x`.
_SPEC_ATTR_RECEIVER = r"[A-Za-z0-9_]*spec[A-Za-z0-9_]*(?:\(\))?(?:\.[A-Za-z0-9_]+)*"
# The same receiver reached dynamically: `getattr(spec, "x", None)`, which
# is how base.py reads fields a third-party spec class may not carry.
_SPEC_GETATTR_RECEIVER = r"[A-Za-z0-9_.]*spec[A-Za-z0-9_.]*"


def _structure_field_readers(field_name: str) -> list[str]:
    """Modules that read `ModelStructureSpec.<field_name>`. The definition
    site (structure.py) is excluded: a field the spec only ever reads back
    to itself has not reached base.py or any pipeline stage, which is
    exactly the gap R26 is about."""
    attr = re.compile(rf"\b{_SPEC_ATTR_RECEIVER}\.{re.escape(field_name)}\b")
    dynamic = re.compile(
        rf"getattr\(\s*{_SPEC_GETATTR_RECEIVER}\s*,\s*"
        rf"[\"']{re.escape(field_name)}[\"']")
    readers = []
    for path in _package_sources():
        if path == STRUCTURE_PARSER:
            continue
        text = _source_text(path)
        if attr.search(text) or dynamic.search(text):
            readers.append(str(path.relative_to(_PACKAGE_ROOT)))
    return readers


@functools.lru_cache(maxsize=None)
def _serving_reader_modules() -> tuple[pathlib.Path, ...]:
    """serving_profiles.py plus every module that imports it. Unlike the
    structure side, the parser module IS a legitimate reader here: it also
    implements the rule evaluation that consumes these fields."""
    return (SERVING_PARSER,) + tuple(
        p for p in _package_sources()
        if p != SERVING_PARSER and _imports_module(p, "serving_profiles")
    )


def _serving_spec_dataclasses():
    """Every spec dataclass in serving_profiles.py — identified by having a
    `from_dict`, so a new one is picked up without editing this test."""
    return [
        obj for _, obj in sorted(vars(SPROF).items())
        if isinstance(obj, type)
        and dataclasses.is_dataclass(obj)
        and getattr(obj, "__module__", "") == SPROF.__name__
        and hasattr(obj, "from_dict")
    ]


STRUCTURE_SPEC_FIELDS = [
    f.name for f in dataclasses.fields(STRUCT.ModelStructureSpec)]
SERVING_SPEC_FIELDS = [
    (cls.__name__, f.name)
    for cls in _serving_spec_dataclasses()
    for f in dataclasses.fields(cls)
]
SERVING_SPEC_FIELD_IDS = [f"{c}.{f}" for c, f in SERVING_SPEC_FIELDS]


@pytest.mark.parametrize("key_path", sorted(STRUCTURE_SPEC_KEYS))
def test_every_structure_spec_key_is_parsed(key_path):
    leaf = _leaf(key_path)
    if leaf.startswith(DOC_ONLY_KEY_PREFIX):
        pytest.skip(f"{leaf!r} is documentation-only "
                    "(see DOC_ONLY_KEY_PREFIX)")
    assert leaf in _string_constants(STRUCTURE_PARSER), (
        f"specs/{{{','.join(STRUCTURE_SPEC_KEYS[key_path])}}} declares "
        f"{key_path!r}, but {STRUCTURE_PARSER.name} never names it — the "
        "key is silently dropped at load. Parse it, delete it from the "
        "spec, or add it to STRUCTURE_DOC_ONLY_KEYS with a reason.")


@pytest.mark.parametrize("key_path", sorted(SERVING_SPEC_KEYS))
def test_every_serving_spec_key_is_parsed(key_path):
    leaf = _leaf(key_path)
    if leaf.startswith(DOC_ONLY_KEY_PREFIX):
        pytest.skip(f"{leaf!r} is documentation-only "
                    "(see DOC_ONLY_KEY_PREFIX)")
    named_by = [
        p.name for p in _serving_reader_modules()
        if leaf in _string_constants(p)
    ]
    assert named_by, (
        f"serving_profile_specs/{{{','.join(SERVING_SPEC_KEYS[key_path])}}} "
        f"declares {key_path!r}, but neither {SERVING_PARSER.name} nor any "
        "module importing it ever names it — the key is silently dropped "
        "at load. Parse it, delete it, or add it to SERVING_DOC_ONLY_KEYS "
        "with a reason.")


@pytest.mark.parametrize("field_name", STRUCTURE_SPEC_FIELDS)
def test_every_structure_spec_field_has_a_reader(field_name):
    if field_name in STRUCTURE_DOC_ONLY_FIELDS:
        pytest.skip(f"{field_name!r} is a parse-time marker "
                    "(see STRUCTURE_DOC_ONLY_FIELDS)")
    readers = _structure_field_readers(field_name)
    if field_name in STRUCTURE_FIELD_NO_READER_XFAIL:
        assert not readers, (
            f"ModelStructureSpec.{field_name} now HAS a reader "
            f"({readers}) — remove it from STRUCTURE_FIELD_NO_READER_XFAIL")
        pytest.xfail(f"ModelStructureSpec.{field_name} has no reader yet")
    assert readers, (
        f"ModelStructureSpec.{field_name} is parsed out of every spec that "
        "declares it and then read by nobody: no base.py accessor and no "
        "consumer under prismaquant/ touches it. Wire a reader, drop the "
        "field, or ratchet it in STRUCTURE_FIELD_NO_READER_XFAIL with a "
        "date and a reason.")


@pytest.mark.parametrize(
    "cls_name,field_name", SERVING_SPEC_FIELDS, ids=SERVING_SPEC_FIELD_IDS)
def test_every_serving_spec_field_has_a_reader(cls_name, field_name):
    readers = [
        p.name for p in _serving_reader_modules()
        if field_name in _attribute_names(p)
    ]
    if field_name in SERVING_FIELD_NO_READER_XFAIL:
        assert not readers, (
            f"{cls_name}.{field_name} now HAS a reader ({readers}) — "
            "remove it from SERVING_FIELD_NO_READER_XFAIL")
        pytest.xfail(f"{cls_name}.{field_name} has no reader yet")
    assert readers, (
        f"{cls_name}.{field_name} is parsed out of serving_profile_specs "
        "and then read by nobody: no method of serving_profiles.py and no "
        "module importing it touches it. Wire a reader, drop the field, or "
        "ratchet it in SERVING_FIELD_NO_READER_XFAIL with a date and a "
        "reason.")


def test_doc_only_convention_holds():
    """The prose convention only stays free if it stays prose. A parser
    that started naming a `_`-prefixed key would make an unreviewed note
    load-bearing — and would make every other note look load-bearing."""
    for parser, keys in ((STRUCTURE_PARSER, STRUCTURE_SPEC_KEYS),
                         (SERVING_PARSER, SERVING_SPEC_KEYS)):
        parsed_prose = sorted(
            leaf for leaf in {_leaf(k) for k in keys}
            if leaf.startswith(DOC_ONLY_KEY_PREFIX)
            and leaf in _string_constants(parser)
        )
        assert not parsed_prose, (
            f"{parser.name} names {parsed_prose}, but a leading underscore "
            "declares a key to be documentation. Either rename the key "
            "without the underscore (it is contract now) or stop parsing "
            "it.")
    # The inventories are the human half of the same convention.
    off_convention = sorted(
        k for k in (STRUCTURE_DOC_ONLY_KEYS | SERVING_DOC_ONLY_KEYS)
        if not k.startswith(DOC_ONLY_KEY_PREFIX)
    )
    assert not off_convention, (
        f"{off_convention} are listed as documentation-only but do not "
        "carry the leading underscore that exempts them")


def test_doc_only_allowlists_are_live():
    """An exemption must not outlive the key it excuses — otherwise the
    allow-lists rot into a place where a real gap can hide."""
    structure_leaves = {_leaf(k) for k in STRUCTURE_SPEC_KEYS}
    stale = sorted(STRUCTURE_DOC_ONLY_KEYS - structure_leaves)
    assert not stale, (
        f"no model spec declares {stale} any more — shrink "
        "STRUCTURE_DOC_ONLY_KEYS")

    serving_leaves = {_leaf(k) for k in SERVING_SPEC_KEYS}
    stale = sorted(SERVING_DOC_ONLY_KEYS - serving_leaves)
    assert not stale, (
        f"no serving spec declares {stale} any more — shrink "
        "SERVING_DOC_ONLY_KEYS")

    stale = sorted(STRUCTURE_DOC_ONLY_FIELDS - set(STRUCTURE_SPEC_FIELDS))
    assert not stale, (
        f"ModelStructureSpec no longer has {stale} — shrink "
        "STRUCTURE_DOC_ONLY_FIELDS")


def test_spec_key_enumeration_is_not_vacuous():
    """The two enumerations above are the test. If a refactor moves the
    spec dirs or empties them, every parametrised case silently vanishes
    and the suite still goes green — so pin that they found something."""
    assert STRUCTURE_SPEC_DIR.is_dir() and SERVING_SPEC_DIR.is_dir()
    assert len(list(STRUCTURE_SPEC_DIR.glob("*.json"))) >= 8
    assert len(list(SERVING_SPEC_DIR.glob("*.json"))) >= 4
    # Sanity: the walker must reach nested keys, not just the top level.
    assert any("." in k for k in STRUCTURE_SPEC_KEYS)
    assert any("[]." in k for k in STRUCTURE_SPEC_KEYS)
    assert any("[]." in k for k in SERVING_SPEC_KEYS)
    assert STRUCTURE_SPEC_FIELDS and SERVING_SPEC_FIELDS


# -------------------------------------------- check 6, synthetic fixtures
#
# Both on-disk expert layouts must validate. The pre-2026-07-30 check only
# accepted the packed one (`k.endswith(f"experts.{n}")`), so every stock HF
# MoE source — Laguna, ornith-35B, DSv4 — failed a check it should pass.


def _write_index(tmp_path, keys):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001.safetensors" for k in keys}})
    )
    return str(tmp_path)


def _write_single_file(tmp_path, shapes: dict[str, list[int]]):
    """Write a safetensors file that is header-only (no tensor payload); the
    validator reads the header and nothing else."""
    header = {
        k: {"dtype": "F32", "shape": s, "data_offsets": [0, 0]}
        for k, s in shapes.items()
    }
    blob = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob)
    return str(tmp_path)


def test_check6_accepts_packed_expert_layout(tmp_path):
    path = _write_index(tmp_path, [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    ])
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert result.ok, result.detail
    assert "packed" in result.detail


def test_check6_accepts_per_expert_layout(tmp_path):
    """The regression this file exists for: a stock HF MoE source ships
    per-expert 2D tensors, and packing happens at load/export time."""
    keys = []
    for e in range(4):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys.append(f"model.layers.3.mlp.experts.{e}.{proj}.weight")
    keys.append("model.layers.3.mlp.experts.e_score_correction_bias")
    path = _write_index(tmp_path, keys)
    result = V._check_packed_experts(LagunaProfile(), path)
    assert result.ok, result.detail
    assert "per-expert" in result.detail


def test_check6_accepts_mixed_layout(tmp_path):
    """Qwen3.5-35B-A3B really does ship both: packed body experts and
    per-expert MTP experts."""
    keys = [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.experts.7.gate_proj.weight",
        "mtp.layers.0.mlp.experts.7.down_proj.weight",
    ]
    result = V._check_packed_experts(Qwen3_5Profile(), _write_index(tmp_path, keys))
    assert result.ok, result.detail
    assert "packed" in result.detail and "per-expert" in result.detail


def test_check6_flags_undeclared_3d_expert_param(tmp_path):
    """The docstring's second clause, now real: a 3D expert tensor the
    profile does not declare would be silently skipped by the pipeline."""
    path = _write_single_file(
        tmp_path, {"model.layers.0.mlp.experts.mystery_proj": [8, 512, 256]})
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert not result.ok
    assert "mystery_proj" in result.detail


def test_check6_is_lenient_on_a_dense_family_member(tmp_path):
    """One profile covers a family; a dense member (Gemma 4 31B-IT vs
    26B-A4B) legitimately has no expert tensors at all."""
    path = _write_index(tmp_path, [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ])
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert result.ok, result.detail


def test_check6_fails_when_experts_match_no_declared_name(tmp_path):
    path = _write_index(tmp_path, [
        "model.layers.0.mlp.experts.0.wA.weight",
        "model.layers.0.mlp.experts.0.wB.weight",
    ])
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert not result.ok
    assert "expert tensors on disk" in result.detail


def test_check6_reads_a_single_file_checkpoint(tmp_path):
    """A single-shard checkpoint has no index; passing it green
    ("cannot verify") verified nothing."""
    path = _write_single_file(tmp_path, {
        "model.layers.0.mlp.experts.gate_up_proj": [8, 512, 256],
        "model.layers.0.mlp.experts.down_proj": [8, 256, 256],
    })
    result = V._check_packed_experts(Qwen3_5Profile(), path)
    assert result.ok, result.detail
    assert "single-file header" in result.detail


def test_check6_reports_cannot_verify_without_weights(tmp_path):
    result = V._check_packed_experts(Qwen3_5Profile(), str(tmp_path))
    assert result.ok
    assert "cannot verify" in result.detail


# -------------------------------------------------- integration lane (vLLM)
#
# These answers are vLLM-version-dependent, which is why they are not in the
# default lane: a check-2 red here can mean "this vLLM predates the model"
# rather than "the profile is wrong" (an April-2026 vLLM fails
# LagunaProfile, which the production image serves). Cross-check the image
# before calling a red here a profile defect.


@pytest.mark.integration
def test_check2_vllm_class_resolves(profile):
    pytest.importorskip("vllm", reason="vLLM registry checks need vLLM")
    result = V._check_vllm_class(profile)
    assert result.ok, result.detail


@pytest.mark.integration
def test_check3_fused_siblings_against_vllm(profile):
    pytest.importorskip("vllm", reason="vLLM registry checks need vLLM")
    result = V._check_fused_siblings(profile)
    assert result.ok, result.detail


@pytest.mark.integration
def test_check4_name_remap_against_vllm(profile):
    pytest.importorskip("vllm", reason="vLLM registry checks need vLLM")
    result = V._check_name_remap(profile)
    assert result.ok, result.detail


# ------------------------------------------------- slow lane (checkpoints)


def _configured_checkpoints() -> dict[str, str]:
    """PQ_CONFORMANCE_MODELS='Qwen3Profile=/path,LagunaProfile=/path'"""
    raw = os.environ.get("PQ_CONFORMANCE_MODELS", "").strip()
    out: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in raw.split(","))):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


@pytest.mark.slow
def test_check6_packed_expert_names(profile):
    path = _configured_checkpoints().get(type(profile).__name__)
    if not path:
        pytest.skip("no checkpoint configured in $PQ_CONFORMANCE_MODELS")
    result = V._check_packed_experts(profile, path)
    assert result.ok, result.detail


@pytest.mark.slow
def test_check7_source_passthrough_prefixes(profile):
    path = _configured_checkpoints().get(type(profile).__name__)
    if not path:
        pytest.skip("no checkpoint configured in $PQ_CONFORMANCE_MODELS")
    result = V._check_source_passthrough(profile, path)
    assert result.ok, result.detail
