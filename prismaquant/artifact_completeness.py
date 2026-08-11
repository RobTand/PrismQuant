"""Post-export gate: every quantized weight in an artifact has a decoder.

THE FAILURE CLASS THIS EXISTS TO MAKE IMPOSSIBLE. An exported artifact is a
promise that each tensor can be read back. The promise is carried by three
different mechanisms — a CB config group, a ``source_passthrough`` declaration,
or an ``ignore`` entry meaning "plain unquantized floats" — and nothing used to
check that every tensor is covered by exactly one of them, correctly.

A block-FP8 weight that no allocation target claimed fell through all three: it
was copied verbatim (right), its ``.scale`` sibling was skipped as "consumed
with its weight" though nothing consumed it (wrong — the scale was DROPPED),
and it was listed in ``ignore`` (wrong — it is not unquantized). A consumer
honouring that reads fp8 bytes into a bf16 parameter, passes the size check
because the element counts agree, applies no scale, and serves weights that are
each off by their own power of two. Nothing raises. On DSv4-Flash that was 43
``attn.wo_a`` + 21 ``attn.indexer.wq_b`` units — 1.44 GB, silently wrong.

So this module asks the question the exporter's own asserts could not: **for
every scale-bearing weight tensor actually present in the artifact, which
mechanism decodes it, and is that mechanism complete?** It reads only the
artifact — safetensors headers and ``quant_config.json``, never the source
checkpoint — because that is exactly what a consumer has.

Cheap by construction: safetensors headers only, no tensor data, so it runs on
a 92 GB artifact in about the time it takes to open the shards.

Run standalone::

    python -m prismaquant.artifact_completeness /path/to/artifact
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path


__all__ = [
    "ArtifactIncomplete",
    "CompletenessReport",
    "check_artifact_completeness",
    "read_artifact_header",
]

#: Element dtypes that CANNOT be read without a scale plane. A tensor stored in
#: one of these is meaningless on its own, so shipping it without a decoding
#: mechanism is the bug this module detects.
_SCALE_BEARING_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})

#: Scale-plane dtypes, recognized so an orphan scale is reported against the
#: unit it belongs to rather than as a mystery tensor.
_SCALE_PLANE_DTYPES = frozenset({"F8_E8M0"})

#: Suffixes a scale plane ships under. Three lanes, three spellings, all of
#: them live in real artifacts: ``.scale`` is the checkpoint's own (byte-
#: verbatim passthrough), ``.weight_scale`` is what the re-quant lanes and
#: compressed-tensors write, ``.weight_scale_inv`` is the legacy block-FP8
#: sibling. The gate pairs a weight with whichever one is present rather than
#: assuming a lane.
_SCALE_SUFFIXES = (".scale", ".weight_scale", ".weight_scale_inv")
_PER_EXPERT_FORMAT_GROUPS_KEY = "per_expert_format_groups"
_PER_EXPERT_FORMAT_GROUPS_VERSION = 1
_PER_EXPERT_GROUP_TOKEN = ".format_group_"


class ArtifactIncomplete(AssertionError):
    """An artifact contains a tensor no declared mechanism can decode."""


@dataclass
class CompletenessReport:
    """What each scale-bearing tensor resolved to. Empty lists == healthy."""

    #: unit -> wire id, from the artifact's own declaration
    declared_units: dict[str, str] = field(default_factory=dict)
    #: units resolved through a CB config group
    cb_units: list[str] = field(default_factory=list)
    #: declared passthrough units, weight + scale both present
    passthrough_units: list[str] = field(default_factory=list)
    #: verbatim units in a namespace no serving stack builds (e.g. DSv4 `mtp.`)
    verbatim_namespace_units: list[str] = field(default_factory=list)

    # --- the four ways an artifact can be incomplete -------------------------
    #: scale-bearing weights claimed by NO mechanism at all
    undeclared: list[str] = field(default_factory=list)
    #: scale-bearing weights declared `ignore`, i.e. claimed to be unquantized
    fp8_in_ignore: list[str] = field(default_factory=list)
    #: declared passthrough units whose scale plane is MISSING from the artifact
    missing_scale: list[str] = field(default_factory=list)
    #: scale planes present whose weight is not declared passthrough
    orphan_scale: list[str] = field(default_factory=list)

    #: PROPOSED per-expert split-stack declaration failures.  Strings carry
    #: layer/family/expert ids so a 256-way refusal is actionable.
    group_partition_errors: list[str] = field(default_factory=list)
    missing_group_tensors: list[str] = field(default_factory=list)
    undeclared_group_tensors: list[str] = field(default_factory=list)
    group_byte_mismatches: list[str] = field(default_factory=list)

    #: route-pending formats the producer explicitly acknowledged shipping
    route_pending_acknowledged: list[str] = field(default_factory=list)
    #: namespaces the producer recorded as deliberately OMITTED. An absence
    #: covered by one of these is intentional; an absence not covered by one is
    #: a dropped tensor, which is the failure this module exists to catch.
    excluded_namespaces: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.undeclared or self.fp8_in_ignore or self.missing_scale
            or self.orphan_scale or self.group_partition_errors
            or self.missing_group_tensors or self.undeclared_group_tensors
            or self.group_byte_mismatches
        )

    def failure_text(self) -> str:
        parts: list[str] = []
        if self.fp8_in_ignore:
            parts.append(
                f"{len(self.fp8_in_ignore)} scale-bearing weight(s) are listed "
                f"in `ignore`, which claims they are plain unquantized floats. "
                f"A consumer will cast them to bf16 with NO scale applied and "
                f"raise nothing: {sorted(self.fp8_in_ignore)[:5]}")
        if self.missing_scale:
            parts.append(
                f"{len(self.missing_scale)} declared passthrough unit(s) are "
                f"missing their scale plane, so the declaration promises a "
                f"decode the artifact cannot perform: "
                f"{sorted(self.missing_scale)[:5]}")
        if self.orphan_scale:
            parts.append(
                f"{len(self.orphan_scale)} scale plane(s) belong to no "
                f"declared unit — either the weight was dropped or its "
                f"declaration was: {sorted(self.orphan_scale)[:5]}")
        if self.undeclared:
            parts.append(
                f"{len(self.undeclared)} scale-bearing weight(s) are claimed "
                f"by no mechanism at all (not CB, not declared passthrough, "
                f"not a verbatim namespace): {sorted(self.undeclared)[:5]}")
        if self.group_partition_errors:
            parts.append(
                "per-expert format groups do not partition their layer/family "
                f"exactly: {self.group_partition_errors[:5]}"
            )
        if self.missing_group_tensors:
            parts.append(
                "declared per-expert format-group tensor(s) are absent: "
                f"{self.missing_group_tensors[:5]}"
            )
        if self.undeclared_group_tensors:
            parts.append(
                "per-expert format-group tensor(s) have no declaration: "
                f"{self.undeclared_group_tensors[:5]}"
            )
        if self.group_byte_mismatches:
            parts.append(
                "per-expert format-group byte accounting disagrees with "
                f"sub-group sums: {self.group_byte_mismatches[:5]}"
            )
        return "; ".join(parts)


def _read_safetensors_header(path: Path) -> dict[str, dict]:
    """Read one safetensors-compatible container header, never its data."""

    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        entries = json.loads(handle.read(length))
    return {
        name: meta for name, meta in entries.items() if name != "__metadata__"
    }


def read_artifact_header(artifact_dir: str | Path) -> dict[str, dict]:
    """``{tensor name: safetensors metadata}`` across every shard. Headers only."""

    root = Path(artifact_dir)
    index = root / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    header: dict[str, dict] = {}
    for shard in shards:
        header.update(_read_safetensors_header(root / shard))
    return header


def _checkpoint_spellings(unit: str, profile) -> set[str]:
    """Every checkpoint spelling a declaration/config-group key can denote.

    THE TWO NAMESPACES. A unit that the ALLOCATOR chose is named in the recipe
    namespace (``model.layers.0.self_attn.wq_a``), because that is where the
    DP's decisions live. A unit that only the EXPORTER saw — a floor unit, or
    a config-group target — is named in the checkpoint namespace
    (``layers.0.attn.wq_a``), because that is what is on disk. Both spellings
    denote the same tensor and both appear in real artifacts.

    This gate therefore normalizes rather than demanding the artifact be
    internally uniform: making the producer emit one spelling everywhere is a
    real improvement, but it is a different change, and a completeness gate
    that failed every mixed-vintage artifact would be useless for finding the
    bug it exists to find. ``source_tensor_name`` is the producer's own
    recipe->checkpoint map, so the normalization is the profile's answer, not
    a guess.
    """

    spellings = {unit}
    if profile is None:
        return spellings
    try:
        spellings.add(profile.source_tensor_name(unit))
        # Several rename rules are anchored on a trailing dot (they rewrite a
        # leaf's PARENT), so a bare unit name misses them.
        spellings.add(profile.source_tensor_name(unit + ".").rstrip("."))
    except Exception:                      # pragma: no cover - defensive
        pass
    return spellings


def _detect_profile_quietly(artifact_dir: Path):
    """The artifact's own profile, or None. Never fatal: the gate must still
    run on an artifact whose architecture this build does not know."""

    try:
        from prismaquant.model_profiles import detect_profile

        return detect_profile(str(artifact_dir))
    except Exception:                      # pragma: no cover - defensive
        return None


def _group_claimed_units(quant_config: dict) -> set[str]:
    """Units ANY config group claims, un-anchored from regex spellings.

    Deliberately not limited to CB groups. A config group is a decoding
    mechanism whatever its flavour — a CB ``scheme``, a ``source-passthrough``
    layout record, a ``gridbook-native`` re-quant group, or a stock
    compressed-tensors scheme — and this gate's question is only "is there
    one", not "which". Narrowing it to CB would report every stock-delegated
    and re-quantized fp8 unit as undeclared, which is noise that would get the
    gate switched off.
    """

    claimed: set[str] = set()
    for group in (quant_config.get("config_groups") or {}).values():
        if not isinstance(group, dict):
            continue
        for target in group.get("targets") or ():
            name = str(target)
            if name.startswith("re:^") and name.endswith("$"):
                name = name[len("re:^"):-1].replace("[.]", ".")
            claimed.add(name)
    return claimed


def _tensor_span(meta: dict) -> int:
    offsets = meta.get("data_offsets") or ()
    if len(offsets) != 2:
        return 0
    return int(offsets[1]) - int(offsets[0])


def _declared_expert_count(config: object) -> int | None:
    """Best-effort architecture config count; declaration checks still run
    on synthetic/minimal configs by deriving the range from the other family.
    """

    keys = ("n_routed_experts", "num_local_experts", "num_experts", "n_experts")
    if isinstance(config, dict):
        for key in keys:
            value = config.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return int(value)
        for value in config.values():
            found = _declared_expert_count(value)
            if found is not None:
                return found
    return None


def _validate_per_expert_format_groups(
    root: Path,
    quant_config: dict,
    header: dict[str, dict],
    report: CompletenessReport,
) -> None:
    """Validate PROPOSED v1 using only bytes carried by the artifact."""

    declaration = quant_config.get(_PER_EXPERT_FORMAT_GROUPS_KEY)
    physical_group_prefixes = {
        name[: -len(".cb_qweight")]
        for name in header
        if _PER_EXPERT_GROUP_TOKEN in name and name.endswith(".cb_qweight")
    }
    if declaration is None:
        report.undeclared_group_tensors.extend(sorted(
            prefix + ".cb_qweight" for prefix in physical_group_prefixes
        ))
        return
    if not isinstance(declaration, dict) or declaration.get("version") != \
            _PER_EXPERT_FORMAT_GROUPS_VERSION:
        report.group_partition_errors.append(
            "declaration version is not supported version 1"
        )
        return
    layers = declaration.get("layers")
    if not isinstance(layers, dict) or not layers:
        report.group_partition_errors.append("declaration layers is empty/malformed")
        return

    model_config = {}
    try:
        model_config = json.loads((root / "config.json").read_text())
    except Exception:
        pass
    configured_count = _declared_expert_count(model_config)
    declared_prefixes: set[str] = set()
    declaration_index: dict[str, tuple[str, str, str]] = {}

    for raw_layer, families in sorted(layers.items(), key=lambda item: str(item[0])):
        layer = str(raw_layer)
        if not isinstance(families, dict):
            report.group_partition_errors.append(
                f"layer {layer}: family map is malformed"
            )
            continue
        family_unions: dict[str, set[int]] = {}
        for family in ("w13", "w2"):
            entries = families.get(family)
            if not isinstance(entries, list) or not entries:
                report.group_partition_errors.append(
                    f"layer {layer}/{family}: no format groups"
                )
                family_unions[family] = set()
                continue
            owners: dict[int, str] = {}
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict) or set(entry) != {
                    "format_wire_id", "expert_ids", "tensor_prefix"
                }:
                    report.group_partition_errors.append(
                        f"layer {layer}/{family} group {index}: expected exact "
                        "keys format_wire_id/expert_ids/tensor_prefix"
                    )
                    continue
                wire_id = str(entry["format_wire_id"])
                tensor_prefix = str(entry["tensor_prefix"])
                raw_ids = entry["expert_ids"]
                if not isinstance(raw_ids, list) or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    or value < 0 for value in raw_ids
                ):
                    report.group_partition_errors.append(
                        f"layer {layer}/{family}/{wire_id}: expert_ids malformed"
                    )
                    continue
                expert_ids = [int(value) for value in raw_ids]
                if expert_ids != sorted(expert_ids):
                    report.group_partition_errors.append(
                        f"layer {layer}/{family}/{wire_id}: expert ids are not sorted: "
                        f"{expert_ids[:12]}"
                    )
                for expert_id in expert_ids:
                    previous = owners.setdefault(expert_id, wire_id)
                    if previous != wire_id or expert_ids.count(expert_id) > 1:
                        report.group_partition_errors.append(
                            f"layer {layer}/{family}: duplicated expert id "
                            f"{expert_id} in {previous} and {wire_id}"
                        )
                declared_prefixes.add(tensor_prefix)
                declaration_key = f"{layer}/{family}/{wire_id}"
                if declaration_key in declaration_index:
                    report.group_partition_errors.append(
                        f"layer {layer}/{family}: format {wire_id} is "
                        "declared more than once"
                    )
                declaration_index[declaration_key] = (
                    layer, family, tensor_prefix
                )

                if wire_id.startswith(("NVFP4_CB_", "FP8_CB_", "FP8_CBL_")):
                    required = [tensor_prefix + ".cb_qweight"]
                    if wire_id.startswith(("FP8_CB_", "FP8_CBL_")):
                        required.append(tensor_prefix + ".weight_scale")
                    for name in required:
                        if name not in header:
                            report.missing_group_tensors.append(
                                f"layer {layer}/{family} experts "
                                f"{expert_ids}: {name}"
                            )
                elif wire_id == "mxfp4_e2m1_ue8m0_g32":
                    leaves = (
                        ("w1", "w3", "gate_proj", "up_proj")
                        if family == "w13" else ("w2", "down_proj")
                    )
                    for expert_id in expert_ids:
                        weights = [
                            name for name in header
                            if name.startswith(f"{tensor_prefix}.{expert_id}.")
                            and name.endswith(".weight")
                            and name.rsplit(".", 2)[-2] in leaves
                        ]
                        wanted = 2 if family == "w13" else 1
                        if len(weights) != wanted:
                            report.missing_group_tensors.append(
                                f"layer {layer}/{family} expert {expert_id}: "
                                f"expected {wanted} verbatim weight slice(s) under "
                                f"{tensor_prefix}, found {len(weights)}"
                            )
                        for weight in weights:
                            base = weight[: -len(".weight")]
                            if not any(base + suffix in header for suffix in _SCALE_SUFFIXES):
                                report.missing_group_tensors.append(
                                    f"layer {layer}/{family} expert {expert_id}: "
                                    f"{base} has no verbatim scale plane"
                                )
                else:
                    report.group_partition_errors.append(
                        f"layer {layer}/{family}: unknown format wire id {wire_id}"
                    )
            family_unions[family] = set(owners)

        derived_count = max(
            (max(ids) + 1 for ids in family_unions.values() if ids),
            default=0,
        )
        expected_count = configured_count or derived_count
        expected = set(range(expected_count))
        for family, present in family_unions.items():
            missing = sorted(expected - present)
            extra = sorted(present - expected)
            if missing or extra:
                report.group_partition_errors.append(
                    f"layer {layer}/{family}: missing expert ids {missing[:16]}; "
                    f"unexpected expert ids {extra[:16]}"
                )

    for prefix in sorted(physical_group_prefixes - declared_prefixes):
        report.undeclared_group_tensors.append(prefix + ".cb_qweight")

    payload = (quant_config.get("provenance") or {}).get(
        "per_expert_format_group_payload"
    )
    if not isinstance(payload, dict):
        report.group_byte_mismatches.append(
            "per_expert_format_group_payload provenance is missing"
        )
        return
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        report.group_byte_mismatches.append("payload groups map is malformed")
        return
    missing_payload_groups = sorted(set(declaration_index) - set(groups))
    extra_payload_groups = sorted(set(groups) - set(declaration_index))
    if missing_payload_groups or extra_payload_groups:
        report.group_byte_mismatches.append(
            "payload/declaration group keys disagree: missing payload groups "
            f"{missing_payload_groups[:8]}; extra payload groups "
            f"{extra_payload_groups[:8]}"
        )
    codebook_file = quant_config.get("codebook_file")
    codebook_header: dict[str, dict] = {}
    if codebook_file is not None:
        codebook_path = root / str(codebook_file)
        if codebook_path.exists():
            codebook_header = _read_safetensors_header(codebook_path)
    tensor_sum = 0
    codebook_sum = 0
    tensor_owners: dict[str, str] = {}
    codebook_owners: dict[str, str] = {}
    for key, group in groups.items():
        if key not in declaration_index or not isinstance(group, dict):
            report.group_byte_mismatches.append(
                f"undeclared/malformed payload group {key}"
            )
            continue
        names = group.get("tensor_names") or []
        if not isinstance(names, list):
            report.group_byte_mismatches.append(
                f"{key}: tensor_names is malformed"
            )
            names = []
        for name in map(str, names):
            previous = tensor_owners.setdefault(name, key)
            if previous != key:
                report.group_byte_mismatches.append(
                    f"tensor {name} is charged to both {previous} and {key}"
                )
        actual = sum(_tensor_span(header[name]) for name in names if name in header)
        expected_bytes = int(group.get("tensor_payload_bytes", -1))
        if actual != expected_bytes:
            layer, family, _prefix = declaration_index[key]
            report.group_byte_mismatches.append(
                f"layer {layer}/{family} {key.rsplit('/', 1)[-1]}: "
                f"declared tensors sum to {actual}B, accounting says "
                f"{expected_bytes}B"
            )
        tensor_sum += expected_bytes
        codebook_names = group.get("codebook_tensor_names") or []
        if not isinstance(codebook_names, list):
            report.group_byte_mismatches.append(
                f"{key}: codebook_tensor_names is malformed"
            )
            codebook_names = []
        for name in map(str, codebook_names):
            previous = codebook_owners.setdefault(name, key)
            if previous != key:
                report.group_byte_mismatches.append(
                    f"codebook tensor {name} is charged to both {previous} "
                    f"and {key}"
                )
        missing_codebooks = sorted(
            str(name) for name in codebook_names
            if str(name) not in codebook_header
        )
        if missing_codebooks:
            layer, family, _prefix = declaration_index[key]
            report.missing_group_tensors.append(
                f"layer {layer}/{family} {key.rsplit('/', 1)[-1]}: "
                f"missing codebook sidecar tensors {missing_codebooks[:8]}"
            )
        actual_codebook_bytes = sum(
            _tensor_span(codebook_header[str(name)])
            for name in codebook_names
            if str(name) in codebook_header
        )
        expected_codebook_bytes = int(group.get("codebook_sidecar_bytes", 0))
        if actual_codebook_bytes != expected_codebook_bytes:
            layer, family, _prefix = declaration_index[key]
            report.group_byte_mismatches.append(
                f"layer {layer}/{family} {key.rsplit('/', 1)[-1]}: "
                f"codebook tensors sum to {actual_codebook_bytes}B, "
                f"accounting says {expected_codebook_bytes}B"
            )
        expected_total = int(group.get("total_bytes", -1))
        if expected_total != expected_bytes + expected_codebook_bytes:
            layer, family, _prefix = declaration_index[key]
            report.group_byte_mismatches.append(
                f"layer {layer}/{family} {key.rsplit('/', 1)[-1]}: group "
                f"total {expected_total}B != tensor/codebook sum "
                f"{expected_bytes + expected_codebook_bytes}B"
            )
        codebook_sum += expected_codebook_bytes
    if tensor_sum != int(payload.get("tensor_payload_bytes", -1)):
        report.group_byte_mismatches.append(
            f"sub-group tensor sum {tensor_sum}B != payload total "
            f"{payload.get('tensor_payload_bytes')}B"
        )
    if codebook_sum != int(payload.get("codebook_sidecar_bytes", -1)):
        report.group_byte_mismatches.append(
            f"sub-group codebook sum {codebook_sum}B != payload total "
            f"{payload.get('codebook_sidecar_bytes')}B"
        )
    if tensor_sum + codebook_sum != int(payload.get("total_bytes", -1)):
        report.group_byte_mismatches.append(
            f"sub-group total {tensor_sum + codebook_sum}B != payload total "
            f"{payload.get('total_bytes')}B"
        )


def check_artifact_completeness(
    artifact_dir: str | Path,
    *,
    verbatim_prefixes: tuple[str, ...] = ("mtp.",),
) -> CompletenessReport:
    """Classify every scale-bearing weight in the artifact, or explain why not.

    ``verbatim_prefixes`` names namespaces whose tensors ship undeclared ON
    PURPOSE because no serving stack builds those modules — DSv4-Flash's
    ``mtp.*`` DSpark blocks are the motivating case. They must still ship their
    scale planes (an incomplete block is useless later), so they are checked
    for weight/scale pairing but exempt from needing a declaration.
    """

    root = Path(artifact_dir)
    quant_config = json.loads((root / "quant_config.json").read_text())
    header = read_artifact_header(root)

    profile = _detect_profile_quietly(root)
    raw_declared = dict(
        (quant_config.get("source_passthrough") or {}).get("units") or {})
    # Resolve every declaration/ignore/CB key into the checkpoint namespace the
    # tensor names actually use, so a recipe-spelled declaration still counts.
    declared: dict[str, str] = {}
    for unit, wire in raw_declared.items():
        for spelling in _checkpoint_spellings(str(unit), profile):
            declared[spelling] = str(wire)
    ignored = {
        spelling
        for entry in (quant_config.get("ignore") or ())
        for spelling in _checkpoint_spellings(str(entry), profile)
    }
    group_claimed = {
        spelling
        for entry in _group_claimed_units(quant_config)
        for spelling in _checkpoint_spellings(entry, profile)
    }
    acknowledged = list(
        (quant_config.get("provenance") or {}).get(
            "route_pending_passthrough_acknowledged") or ())

    excluded = [
        str(prefix) for prefix in
        ((quant_config.get("provenance") or {}).get("excluded_namespaces")
         or ())
    ]
    report = CompletenessReport(
        declared_units=dict(raw_declared),
        route_pending_acknowledged=acknowledged,
        excluded_namespaces=excluded,
    )
    # A declared unit whose tensors are absent because its namespace was
    # excluded is an intended absence, not a broken promise. Drop those
    # declarations before checking, so "declared but no tensor" keeps meaning
    # the one thing that IS a bug.
    if excluded:
        declared = {
            unit: wire for unit, wire in declared.items()
            if not any(unit.startswith(prefix) for prefix in excluded)
        }

    def _scale_unit(name: str) -> str | None:
        for suffix in _SCALE_SUFFIXES:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return None

    present_scales = {
        name for name, meta in header.items()
        if _scale_unit(name) is not None
        and meta.get("dtype") in _SCALE_PLANE_DTYPES
    }
    #: unit -> the scale plane(s) actually present for it, whatever the lane
    #: spelled them.
    scales_by_unit: dict[str, set[str]] = {}
    for name in present_scales:
        scales_by_unit.setdefault(_scale_unit(name), set()).add(name)
    claimed_scales: set[str] = set()

    for name, meta in sorted(header.items()):
        if meta.get("dtype") not in _SCALE_BEARING_DTYPES:
            continue
        if not name.endswith(".weight"):
            continue
        unit = name[: -len(".weight")]
        unit_scales = scales_by_unit.get(unit, set())

        if any(unit.startswith(prefix) for prefix in verbatim_prefixes):
            report.verbatim_namespace_units.append(unit)
            if unit_scales:
                claimed_scales |= unit_scales
            else:
                # A verbatim block shipped without its scale is a future
                # re-export, so it is still a failure — just not a serving one.
                report.missing_scale.append(unit)
            continue

        if _unit_variants(unit) & declared.keys():
            report.passthrough_units.append(unit)
            if unit_scales:
                claimed_scales |= unit_scales
            else:
                report.missing_scale.append(unit)
            if _unit_variants(unit) & ignored:
                # Both statements cannot be true, and `ignore` is the one that
                # loses the scale.
                report.fp8_in_ignore.append(unit)
            continue

        if _unit_variants(unit) & ignored:
            # THE ORIGINAL BUG. `ignore` means "plain unquantized floats", and
            # this tensor is not that. Checked before the config-group test so
            # a unit that is somehow both still reports the contradiction.
            report.fp8_in_ignore.append(unit)
            continue
        if _claimed_by_self_or_ancestor(unit, group_claimed):
            report.cb_units.append(unit)
            claimed_scales |= unit_scales
            continue
        report.undeclared.append(unit)

    for scale_key in sorted(present_scales - claimed_scales):
        unit = _scale_unit(scale_key)
        if any(unit.startswith(prefix) for prefix in verbatim_prefixes):
            continue
        if _claimed_by_self_or_ancestor(unit, declared):
            continue
        if _claimed_by_self_or_ancestor(unit, group_claimed):
            # An expert stack keeps its per-expert source scale planes only
            # when the stack was NOT collapsed; either way a group claims
            # them, so they are not orphans.
            continue
        report.orphan_scale.append(scale_key)

    _validate_per_expert_format_groups(root, quant_config, header, report)

    return report


#: Module components a config-group target may COLLAPSE away. A shared-MLP
#: block lives under ``…mlp.shared_mlp.<leaf>`` / ``…mlp.shared_experts.<leaf>``
#: on disk, but its target is written against the collapsed ``…mlp.<leaf>``.
#: Gridbook bridges the same gap at load (``_alias_collapsed_shared_prefixes``),
#: so recognizing it here is matching the consumer, not inventing a rule.
_COLLAPSIBLE_COMPONENTS = ("shared_mlp", "shared_experts")


def _unit_variants(unit: str) -> set[str]:
    """Every spelling of *unit* a target may legitimately be written as."""

    variants = {unit}
    parts = unit.split(".")
    for index, part in enumerate(parts):
        if part in _COLLAPSIBLE_COMPONENTS:
            variants.add(".".join(parts[:index] + parts[index + 1:]))
    return variants


def _claimed_by_self_or_ancestor(unit: str, claimed) -> bool:
    """Whether *unit* or any dotted ancestor of it appears in *claimed*.

    Routed-expert groups are declared ONCE for the whole stack
    (``layers.1.ffn.experts``) while their tensors are per-expert,
    per-projection (``layers.1.ffn.experts.0.w1.scale``). Walking ancestors is
    what lets one declaration cover the 768 planes it is actually about,
    without letting a declaration for a NEIGHBOURING module claim them —
    ancestry is tested on dotted boundaries, so ``experts2`` never matches
    ``experts``.
    """

    for variant in _unit_variants(unit):
        if variant in claimed:
            return True
        parts = variant.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            if ".".join(parts[:cut]) in claimed:
                return True
    return False


def assert_artifact_complete(artifact_dir: str | Path, **kwargs) -> CompletenessReport:
    """:func:`check_artifact_completeness`, raising on any incompleteness."""

    report = check_artifact_completeness(artifact_dir, **kwargs)
    if not report.ok:
        raise ArtifactIncomplete(
            f"{artifact_dir}: {report.failure_text()}")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Check that every quantized weight in an artifact has a "
                    "decoder (headers only; safe on a multi-GB artifact).")
    parser.add_argument("artifact")
    parser.add_argument(
        "--verbatim-prefix", action="append", default=None,
        help="namespace shipped undeclared on purpose (default: 'mtp.')")
    args = parser.parse_args(argv)

    prefixes = tuple(args.verbatim_prefix or ("mtp.",))
    report = check_artifact_completeness(args.artifact,
                                         verbatim_prefixes=prefixes)
    print(f"declared passthrough units : {len(report.passthrough_units)}")
    print(f"verbatim-namespace units   : "
          f"{len(report.verbatim_namespace_units)}")
    if report.route_pending_acknowledged:
        print(f"route-pending acknowledged : "
              f"{sorted(report.route_pending_acknowledged)}")
    if report.ok:
        print("COMPLETE: every scale-bearing weight has a decoder")
        return 0
    print("INCOMPLETE: " + report.failure_text())
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
