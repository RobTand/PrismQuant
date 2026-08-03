"""Canonical producer-side Gridbook CB artifact configuration emitter.

The resident and streaming exporters differ in how they discover and write
weights, but they serialize the same CB scheme and ``quant_config.json``
contract.  Keep that contract here so a new layout field cannot land in one
exporter while silently being omitted from the other.

Exporter-specific namespace and provenance choices are explicit parameters:
callers provide the CB/delegated target-name mappers, identify weight-only
stock targets, and select the provenance fields that intentionally differ.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    PASSTHROUGH_WIRE_FORMAT_IDS,
    REQUANT_WIRE_FORMAT_IDS,
    SOURCE_PASSTHROUGH_CONTRACTS,
    SourcePassthroughContract,
)
from prismaquant.cb_layout import (
    FP4_GROUP,
    SCALE_CODING_TWO_TIER,
    SCALE_CODING_V1,
    SUPERBLOCK,
    VEC_DIM,
    codebook_subtable_shapes,
    family_for,
    parse_format_name,
    type_size,
)
from prismaquant.export_native_compressed import (
    FP8_E4M3_SCHEME,
    FP8_SOURCE_SCHEME,
    NVFP4_SCHEME,
    _explicit_regex,
)


CBTarget = tuple[str, str, int]
TargetName = Callable[[str], str]
_STOCK_CT_SCHEMES = {
    "NVFP4": NVFP4_SCHEME,
    "FP8_E4M3": FP8_E4M3_SCHEME,
}

# ---------------------------------------------------------------------------
# Source-passthrough wire contracts + the routed-expert routing declaration
# ---------------------------------------------------------------------------
# A unit can leave this producer on one of several EXECUTION ROUTES, and the
# difference is not a scheme detail — it decides which loader reads the bytes:
#
#   gridbook_cb              the exporter re-encoded the weight into a CB
#                            payload; gridbook's own decoder serves it.
#   delegated_native_*       the exporter copied the checkpoint's own bytes;
#                            some other loader serves them, and which one is
#                            exactly what the route id names.
#
# ``allocator_candidates.SOURCE_PASSTHROUGH_CONTRACTS`` is the single table of
# native source formats and their routes. Everything below DERIVES from it plus
# the format registry, so a newly censused native format is a table entry, not
# a new branch here: its route, its backedness, its config group and its
# routing-declaration vocabulary all follow.
#
# WHY THE DECLARATION EXISTS. Nothing else in the artifact says which route a
# routed-expert group took. The CB config_groups only name what IS CB, the
# ``ignore`` list only names unquantized passthrough, and the K0.2 activation
# record only names modules that contributed a calibrated fp4 stage. A
# delegated group is invisible in all three — so a consumer cannot tell "this
# layer is served natively, by design" from "this layer's attestation is
# missing". That indistinguishability is the whole reason this record exists;
# ``cb_activation_contract`` is the field that resolves it.

# The scale-plane dtype the compressed-tensors vocabulary reads. A passthrough
# whose serialized scale plane is ALREADY that dtype can be normalized into the
# CT namespace on write (rename + trivial cast) and delegated to vLLM's stock
# block-FP8 path; any other plane can only be read by the loader that wrote it,
# so it must ship byte-verbatim under the CHECKPOINT's own tensor names.
# Widening it instead would multiply the scale bytes and emit a tensor the
# model's own loader does not expect — the opposite of a passthrough.
_CT_SCALE_DTYPE_NAME = "fp32"
_NO_SCALE_PLANE_DTYPE_NAMES = frozenset({"none", ""})

# ---------------------------------------------------------------------------
# The `source_passthrough` declaration (top-level key in quant_config.json)
# ---------------------------------------------------------------------------
# WIRE CONTRACT. The consumer side owns this shape; the spellings below are its
# closed enum, not ours, so they are a hand-maintained table rather than
# something derived from FormatSpec fields. Deriving them would let a registry
# field rename silently change what the artifact claims — a rename is a local
# refactor, a wire-id change is a load failure at the other end.
#
# ABSENCE of the key means "legacy all-CB artifact". That is why the producer
# omits it entirely when nothing is passthrough rather than emitting an empty
# record: an empty `units` would be a positive claim that nothing is delegated,
# which is a different statement from a file written before this key existed.
SOURCE_PASSTHROUGH_DECLARATION_KEY = "source_passthrough"
SOURCE_PASSTHROUGH_DECLARATION_VERSION = 1
# The registry-name -> wire-id map, re-exported from the ONE place the
# passthrough family is declared (``allocator_candidates
# .SOURCE_PASSTHROUGH_CONTRACTS``). It is hand-pinned there rather than derived
# from FormatSpec fields, for the reason that matters here: a local rename of a
# registry format must not be able to silently change a cross-repo wire
# contract. Keeping a SECOND literal copy in this module would defeat that
# differently — two hand-pinned tables can disagree, and the artifact would
# then declare one thing while the allocator believed another. One table,
# re-exported under the name the exporter reads.
# tests/test_source_passthrough_family.py pins the literal strings.
SOURCE_PASSTHROUGH_WIRE_IDS: dict[str, str] = dict(
    PASSTHROUGH_WIRE_FORMAT_IDS
)
SOURCE_PASSTHROUGH_WIRE_FORMATS: dict[str, str] = {
    wire_id: name for name, wire_id in SOURCE_PASSTHROUGH_WIRE_IDS.items()
}

# Config-group ``format`` token for a source passthrough. Deliberately NOT a
# compressed-tensors format string: nothing in that vocabulary describes "the
# architecture's own kernel reads its own tensors". The group describes the
# TENSOR-level layout of the targets that left on that lane; the declaration
# above describes UNIT-level routing. Different granularity, and the group does
# not restate the route — one source of truth for that.
SOURCE_PASSTHROUGH_GROUP_FORMAT = "source-passthrough"


@dataclass(frozen=True)
class PassthroughWire:
    """How ONE source-passthrough format's tensors reach the artifact."""

    format_name: str
    contract: SourcePassthroughContract
    spec: Any                    # format_registry.FormatSpec
    ct_normalized: bool


def source_passthrough_wire(format_name: str) -> PassthroughWire:
    """Resolve a format name to its passthrough wire contract, or refuse.

    An unknown format is a hard error rather than a default: "copy the bytes"
    is only a safe instruction when the table says which bytes those are and
    which loader has agreed to read them.
    """

    canonical = fr.canonical_format_name(str(format_name))
    contract = SOURCE_PASSTHROUGH_CONTRACTS.get(canonical)
    if contract is None:
        raise ValueError(
            f"{format_name!r} is not a declared source-passthrough format "
            f"(allocator_candidates.SOURCE_PASSTHROUGH_CONTRACTS: "
            f"{sorted(SOURCE_PASSTHROUGH_CONTRACTS)})"
        )
    spec = fr.get_format(canonical)
    return PassthroughWire(
        format_name=canonical,
        contract=contract,
        spec=spec,
        ct_normalized=str(spec.scale_dtype_name) == _CT_SCALE_DTYPE_NAME,
    )


def source_passthrough_wire_id(format_name: str) -> str:
    """Registry format name -> the consumer's wire id. THE one mapping.

    Fails closed on a format with no wire id. A newly censused passthrough that
    reached the emitter without one must stop the export, not be dropped from
    ``units``: a silently omitted unit reads to the consumer as "this is CB",
    which is the one wrong answer that loads.
    """

    canonical = source_passthrough_wire(format_name).format_name
    try:
        return SOURCE_PASSTHROUGH_WIRE_IDS[canonical]
    except KeyError:
        raise ValueError(
            f"{canonical} is a source-passthrough format with no wire id in "
            f"SOURCE_PASSTHROUGH_WIRE_IDS {sorted(SOURCE_PASSTHROUGH_WIRE_IDS)}"
            "; the consumer's format enum is closed, so this unit cannot be "
            "declared and must not be shipped undeclared"
        ) from None


def _has_serialized_scale_plane(spec) -> bool:
    """Whether this format ships a SECOND tensor beside its element plane."""

    return (
        int(spec.scale_bits) > 0
        and str(spec.scale_dtype_name) not in _NO_SCALE_PLANE_DTYPE_NAMES
    )


# Passthrough formats the exporter emits as a WEIGHT + SCALE PAIR. BF16 is
# excluded by construction (no serialized scale plane): it is a plain verbatim
# tensor copy the generic passthrough loop already performs, and it needs no
# config group of its own because ``ignore`` already describes it.
SOURCE_PASSTHROUGH_EXPORT_FORMATS: frozenset[str] = frozenset(
    name for name in SOURCE_PASSTHROUGH_CONTRACTS
    if _has_serialized_scale_plane(fr.get_format(name))
)
# The subset that ships byte-verbatim under the CHECKPOINT's own tensor names.
# Membership is what makes a unit UNCOLLAPSIBLE in the streaming exporter (a
# packed parent it never writes must not be named), what selects the emit
# branch, and what puts the unit in the declaration — so those decisions cannot
# drift apart.
DELEGATED_NATIVE_PASSTHROUGH_FORMATS: frozenset[str] = frozenset(
    name for name in SOURCE_PASSTHROUGH_EXPORT_FORMATS
    if not source_passthrough_wire(name).ct_normalized
)

# ---------------------------------------------------------------------------
# Re-quantized (non-passthrough) wire formats the STREAMING exporter emits
# ---------------------------------------------------------------------------
# Formats this lane produces with a real encoder, under their own wire id,
# outside the compressed-tensors scheme vocabulary. Structurally these sit
# between the two existing classes: like a passthrough they get a scheme-less
# config group and a wire id (stock CT cannot describe them); unlike a
# passthrough the producer WROTE these bytes, so they are re-derivable, they
# carry a real measured cost, and they are legal on any source dtype.
#
# The literal here is the streaming exporter's emit surface, re-exported for
# the serving profile's export lane (serving_profile_specs/nvfp4_cb.json
# ``codec_formats_from``) so the allocator can never spend budget on a rung
# this exporter would hard-fail on. Declaring it here rather than in
# export_nvfp4_cb_streaming keeps the lane's format vocabulary in the module
# that already owns the wire contract, and keeps the profile's lazy import off
# a module that pulls the whole streaming stack.
STREAMING_REQUANT_EXPORT_FORMATS: frozenset[str] = frozenset(
    REQUANT_WIRE_FORMAT_IDS
)
REQUANT_WIRE_IDS: dict[str, str] = dict(REQUANT_WIRE_FORMAT_IDS)
REQUANT_WIRE_FORMATS: dict[str, str] = {
    wire_id: name for name, wire_id in REQUANT_WIRE_IDS.items()
}

# Every wire id the DELEGATED-NATIVE routing record may carry, from either
# table. This is the producer-side mirror of the consumer's own registry
# (``gridbook.source_passthrough.FORMATS``), which holds every id it can route
# regardless of what produced the bytes. Ids must be unique across the two:
# the consumer resolves a unit by id alone.
DELEGATED_NATIVE_WIRE_FORMATS: dict[str, str] = {
    **SOURCE_PASSTHROUGH_WIRE_FORMATS,
    **REQUANT_WIRE_FORMATS,
}

# Config-group ``format`` token for a re-quantized native rung. Same reasoning
# as SOURCE_PASSTHROUGH_GROUP_FORMAT: no compressed-tensors format string
# describes "the Gridbook runtime reads this element+scale pair directly", and
# borrowing one would make a stock CT loader believe it could read the group.
REQUANT_NATIVE_GROUP_FORMAT = "gridbook-native"


def requant_wire_id(format_name: str) -> str:
    """Registry format name -> the consumer's wire id, for a re-quant rung.

    Fails closed exactly like ``source_passthrough_wire_id``: the consumer's
    format enum is closed, so a rung that reached the emitter without an id
    cannot be declared, and an undeclared group reads to the consumer as CB.
    """

    canonical = fr.canonical_format_name(str(format_name))
    try:
        return REQUANT_WIRE_IDS[canonical]
    except KeyError:
        raise ValueError(
            f"{canonical} is not a declared re-quantization wire format "
            f"(allocator_candidates.REQUANT_WIRE_FORMAT_IDS: "
            f"{sorted(REQUANT_WIRE_IDS)})"
        ) from None


def requant_native_config_group(format_name: str) -> dict[str, Any]:
    """The scheme-less config group one re-quantized native rung gets.

    Built from the FormatSpec for the same reason as its passthrough twin: the
    group cannot describe a different on-disk contract than the one the
    accountant priced and the emitter wrote.
    """

    canonical = fr.canonical_format_name(str(format_name))
    wire = requant_wire_id(canonical)
    spec = fr.get_format(canonical)
    weights: dict[str, Any] = {
        "num_bits": int(spec.weight_bits),
        "type": "float",
        "element_dtype": str(spec.weight_element_dtype),
        "scale_dtype": str(spec.scale_dtype_name),
        "symmetric": True,
        "dynamic": False,
        "strategy": "group",
        "group_size": int(spec.group_size),
        # These bytes WERE produced here, unlike a passthrough group.
        "source_passthrough": False,
    }
    activations: dict[str, Any] | None = None
    if spec.act_quant_changes_input:
        # W8A8: the serving lane quantizes activations dynamically to the same
        # grid, so the group says so. Stating None here would describe a
        # weight-only contract the runtime does not implement.
        activations = {
            "num_bits": int(spec.act_bits),
            "type": "float",
            "element_dtype": str(spec.act_dtype_name),
            "scale_dtype": str(spec.scale_dtype_name),
            "symmetric": True,
            "dynamic": True,
            "strategy": "group",
            "group_size": int(spec.act_group_size),
        }
    return {
        "format": REQUANT_NATIVE_GROUP_FORMAT,
        "source_format": canonical,
        "wire_format_id": wire,
        "weights": weights,
        "input_activations": activations,
    }


def source_passthrough_config_group(format_name: str) -> dict[str, Any]:
    """The scheme-less config group one delegated-native passthrough gets.

    Built from the FormatSpec rather than hand-written per format, so the group
    cannot describe a different on-disk contract than the one the accountant
    priced and the emitter copied.  It states LAYOUT only; routing is stated
    once, in the ``source_passthrough`` declaration.
    """

    wire = source_passthrough_wire(format_name)
    spec = wire.spec
    weights: dict[str, Any] = {
        "num_bits": int(spec.weight_bits),
        "type": "float",
        "element_dtype": str(spec.weight_element_dtype),
        "scale_dtype": str(spec.scale_dtype_name),
        "symmetric": True,
        "dynamic": False,
        # Verbatim: these bytes were not produced here and are not re-derivable
        # from anything this exporter wrote.
        "source_passthrough": True,
    }
    if spec.scale_block_shape is not None:
        weights["strategy"] = "block"
        weights["block_structure"] = [int(d) for d in spec.scale_block_shape]
    else:
        weights["strategy"] = "group"
        weights["group_size"] = int(spec.group_size)
    return {
        "format": SOURCE_PASSTHROUGH_GROUP_FORMAT,
        "source_format": wire.format_name,
        "source_passthrough_id": source_passthrough_wire_id(wire.format_name),
        "weights": weights,
        # The producer applies no activation quantization of its own on a
        # passthrough route; the A side is the released checkpoint's contract.
        "input_activations": None,
    }


def build_source_passthrough_declaration(
    units: Mapping[str, str],
) -> dict[str, Any]:
    """Build + validate the declaration from ``{unit id: registry format}``.

    A UNIT is whatever the allocator decided atomically — a routed-expert group
    (``model.layers.7.mlp.experts``) or a dense Linear
    (``model.layers.7.self_attn.wq_a``). Deliberately not expert-only: the
    UE8M0 block-FP8 body ships on the same contract, and framing the record
    around expert groups would have left those units undeclarable.

    There is no exhaustiveness claim here — the record says which units ARE
    passthrough, not that it has enumerated every unit in the model — so a
    partially-allocated model needs no special case.

    WHAT THIS RECORD ACTUALLY MEANS, given it now carries re-quantized rungs
    too: it is the DELEGATED-NATIVE ROUTING record — "these units are served by
    a native Gridbook/model-owned route rather than by a CB codebook decoder",
    which is the question the consumer's dispatcher asks. It is NOT a claim
    that every listed unit's bytes came from the checkpoint unchanged; only the
    ``*_SOURCE`` members make that stronger claim, and the config group's
    ``weights.source_passthrough`` flag is where the two are distinguished per
    unit. The key keeps its historical name because it is a shipped cross-repo
    contract (``gridbook.source_passthrough.SCHEMA_KEY``) read by a released
    consumer; renaming it would be a schema break for a wording improvement.
    """

    if not units:
        raise ValueError(
            "source_passthrough needs at least one unit; an artifact with no "
            "passthrough unit must OMIT the key (its absence is what marks a "
            "legacy all-CB artifact)"
        )
    declared: dict[str, str] = {}
    for unit_id, format_name in units.items():
        unit = str(unit_id)
        if not unit or unit != unit.strip():
            raise ValueError(
                f"source_passthrough unit id {unit_id!r} is not a usable "
                "module name"
            )
        canonical = fr.canonical_format_name(str(format_name))
        declared[unit] = (
            REQUANT_WIRE_IDS[canonical]
            if canonical in REQUANT_WIRE_IDS
            else source_passthrough_wire_id(format_name)
        )
    return {
        "version": SOURCE_PASSTHROUGH_DECLARATION_VERSION,
        "units": dict(sorted(declared.items())),
    }


def _cb_config_group_targets(quant_config: Mapping[str, Any]) -> set[str]:
    """Target names the CB config groups claim, un-anchored."""

    targets: set[str] = set()
    groups = quant_config.get("config_groups")
    if not isinstance(groups, Mapping):
        return targets
    for group in groups.values():
        if not isinstance(group, Mapping) or "scheme" not in group:
            continue
        for target in group.get("targets") or ():
            name = str(target)
            if name.startswith("re:^") and name.endswith("$"):
                name = name[len("re:^"):-1].replace("[.]", ".")
            targets.add(name)
    return targets


def parse_source_passthrough_declaration(
    quant_config: Mapping[str, Any],
) -> dict[str, str] | None:
    """Read + re-validate the declaration, or ``None`` for a legacy artifact.

    THE consumer-side definition, kept beside the producer's so the two cannot
    drift, and used by the producer as its own pre-write assertion. Returns
    ``{unit id: wire format id}``.

    Refusal cases are exactly the ones a loader must reject: an unknown
    version, a malformed ``units`` map, an unknown format id, and a unit
    claimed by BOTH the CB config groups and this record. That last check is
    conservative here on purpose — it can only compare the two in the SAME
    namespace, and a checkpoint whose serialized names differ from its recipe
    names (DSv4) hides the collision from a string test. The exporter runs the
    strong version of it, where both namespaces are in hand.
    """

    record = quant_config.get(SOURCE_PASSTHROUGH_DECLARATION_KEY)
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ValueError(
            f"{SOURCE_PASSTHROUGH_DECLARATION_KEY} must be an object, got "
            f"{type(record).__name__}"
        )
    version = record.get("version")
    if version != SOURCE_PASSTHROUGH_DECLARATION_VERSION:
        raise ValueError(
            f"unsupported {SOURCE_PASSTHROUGH_DECLARATION_KEY} version "
            f"{version!r}; this reader implements "
            f"{SOURCE_PASSTHROUGH_DECLARATION_VERSION}"
        )
    units = record.get("units")
    if not isinstance(units, Mapping) or not units:
        raise ValueError(
            f"{SOURCE_PASSTHROUGH_DECLARATION_KEY} carries no units; the key "
            "is present, so it is a positive claim and cannot be empty"
        )
    parsed: dict[str, str] = {}
    for unit_id, wire_id in units.items():
        if not isinstance(unit_id, str) or not isinstance(wire_id, str):
            raise ValueError(
                f"{SOURCE_PASSTHROUGH_DECLARATION_KEY} units must map string "
                f"unit ids to string format ids, got {unit_id!r}: {wire_id!r}"
            )
        # The legal set is the UNION of the byte-verbatim and re-quantized
        # wire tables: this record is the delegated-native ROUTING map the
        # consumer dispatches on, and its ``FORMATS`` registry likewise holds
        # every id it can route, whatever produced the bytes.
        if wire_id not in DELEGATED_NATIVE_WIRE_FORMATS:
            raise ValueError(
                f"{SOURCE_PASSTHROUGH_DECLARATION_KEY} unit {unit_id!r} "
                f"declares unknown format id {wire_id!r}; legal ids are "
                f"{sorted(DELEGATED_NATIVE_WIRE_FORMATS)}"
            )
        parsed[unit_id] = wire_id
    contested = sorted(set(parsed) & _cb_config_group_targets(quant_config))
    if contested:
        raise ValueError(
            f"{contested} are claimed by BOTH a CB config group and "
            f"{SOURCE_PASSTHROUGH_DECLARATION_KEY}; a unit is decoded by "
            "gridbook's codec or handed to the model's own loader, never both"
        )
    return parsed


def _validated_codebook_sequence(
    fmt: str,
    codebook: object,
) -> tuple[torch.Tensor, ...]:
    """Return tensors only after exact canonical sidecar-shape validation."""

    parsed = parse_format_name(fmt)
    if parsed is None:
        raise ValueError(f"not a producer CB format: {fmt!r}")
    family, k = parsed
    expected = codebook_subtable_shapes(k, family.mode, family.n_sub)
    if isinstance(codebook, torch.Tensor):
        tensors: tuple[object, ...] = (codebook,)
    elif isinstance(codebook, (tuple, list)):
        tensors = tuple(codebook)
    else:
        raise TypeError(
            f"{fmt} codebook must be a tensor or tensor sequence, got "
            f"{type(codebook).__name__}"
        )
    if len(tensors) != len(expected):
        raise ValueError(
            f"{fmt} requires {len(expected)} codebook subtables, got "
            f"{len(tensors)}"
        )
    validated: list[torch.Tensor] = []
    for index, (tensor, expected_shape) in enumerate(
        zip(tensors, expected, strict=True)
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{fmt} codebook subtable {index} must be a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )
        if tensor.ndim != 2:
            raise ValueError(
                f"{fmt} codebook subtable {index} must have rank 2, got "
                f"rank {tensor.ndim}"
            )
        actual_shape = tuple(int(dim) for dim in tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{fmt} codebook subtable {index} shape {actual_shape} does "
                f"not match canonical shape {expected_shape}"
            )
        validated.append(tensor)
    return tuple(validated)


def _codebook_names_for_count(
    ref: str,
    fmt: str,
    count: int,
) -> tuple[str, ...]:
    base = f"cb_codebook.{ref}.{fmt}"
    if count > 1:
        return tuple(f"{base}.sub{i}" for i in range(count))
    return (base,)


def codebook_tensor_names(
    ref: str,
    fmt: str,
    codebook: object,
) -> tuple[str, ...]:
    """Physical sidecar tensor names for one resolved codebook."""

    tensors = _validated_codebook_sequence(fmt, codebook)
    return _codebook_names_for_count(ref, fmt, len(tensors))


def codebook_tensors(
    ref: str,
    fmt: str,
    codebook: object,
) -> dict[str, torch.Tensor]:
    """Serialize one codebook under its canonical sidecar tensor names."""

    tensors = _validated_codebook_sequence(fmt, codebook)
    names = _codebook_names_for_count(ref, fmt, len(tensors))
    return {
        name: tensor.to(torch.float16).cpu().contiguous()
        for name, tensor in zip(names, tensors, strict=True)
    }


def _two_tier_scale_coding() -> dict[str, Any]:
    # The table generator is shared with the encoder so the self-describing
    # config cannot drift from the exact E4M3 values used for packing.
    from prismaquant.nvfp4_cb_formats import (
        TWO_TIER_SUPER_BIAS,
        _two_tier_tables,
    )

    table, _, _ = _two_tier_tables("cpu")
    return {
        "kind": "two_tier",
        "sub_bits": 4,
        "super_bias": TWO_TIER_SUPER_BIAS,
        "table": [float(value) for value in table.tolist()],
    }


def build_cb_scheme(
    *,
    ref: str,
    fmt: str,
    grid: str,
    mode: str,
    k: int,
    codebook: object,
    scale_coding: str,
    activation_contract: str | None = None,
) -> dict[str, Any]:
    """Build the canonical scheme for one CB target/group.

    Layout identity comes from :mod:`prismaquant.cb_layout`; the actual
    sidecar object is checked against the family's required subtable count.
    FP8 has no serialized scale plane and therefore always carries the v1
    layout identity regardless of the exporter's FP4 scale-coding selection.
    """

    grid = str(grid).lower()
    mode = str(mode).lower()
    k = int(k)
    family = family_for(grid, mode)
    parsed = parse_format_name(fmt)
    if parsed is None or parsed[0] != family or parsed[1] != k:
        raise ValueError(
            f"CB format/fields disagree: {fmt!r} vs "
            f"grid={grid!r}, mode={mode!r}, k={k}"
        )
    tensors = _validated_codebook_sequence(fmt, codebook)
    names = _codebook_names_for_count(ref, fmt, len(tensors))
    coding = scale_coding if grid == "fp4" else SCALE_CODING_V1
    scheme: dict[str, Any] = {
        "grid": grid,
        "mode": mode,
        "k": k,
        "superblock": SUPERBLOCK,
        "group_size": FP4_GROUP if grid == "fp4" else 0,
        "vec_dim": VEC_DIM,
        "n_sub": family.n_sub,
        "type_size": type_size(k, grid, coding),
        "act_bits": 4 if grid == "fp4" else 8,
        "codebook_source": "lattice" if ref == "lattice" else "learned",
        "codebook_ref": list(names) if len(names) > 1 else names[0],
        "codebook_group": None if ref == "lattice" else ref,
    }
    if coding == SCALE_CODING_TWO_TIER:
        scheme["scale_coding"] = _two_tier_scale_coding()
    if grid == "fp4" and activation_contract is not None:
        scheme["activation_contract"] = str(activation_contract)
    return scheme


def cb_scheme_reuse_signature(scheme: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the scheme fields that make a packed CB tensor reusable."""

    scale_coding = scheme.get("scale_coding")
    coding = (
        SCALE_CODING_TWO_TIER
        if isinstance(scale_coding, Mapping)
        and scale_coding.get("kind") == "two_tier"
        else SCALE_CODING_V1
    )
    signature: dict[str, Any] = {
        "grid": scheme.get("grid"),
        "mode": scheme.get("mode"),
        "k": scheme.get("k"),
        "n_sub": scheme.get("n_sub"),
        "type_size": scheme.get("type_size"),
        "codebook_ref": scheme.get("codebook_ref"),
        "scale_coding": coding,
    }
    if scheme.get("activation_contract") is not None:
        signature["activation_contract"] = scheme["activation_contract"]
    return signature


def _identity_target(qname: str) -> str:
    return qname


def build_quant_config(
    *,
    assignment: Mapping[str, str],
    cb_targets: Mapping[str, CBTarget],
    source_targets: Iterable[str],
    native_source_targets: Mapping[str, str] | None = None,
    requant_targets: Mapping[str, str] | None = None,
    stock_targets: Mapping[str, str],
    by_group: Mapping[tuple[str, str], Sequence[str]],
    codebooks: Mapping[tuple[str, str], object],
    col_weights: Mapping[str, torch.Tensor],
    codebook_tensors_by_name: Mapping[str, torch.Tensor],
    ignore: Iterable[str],
    codebook_file: str | None,
    scale_coding: str,
    codebook_source: str,
    serialized_payload_summary: Mapping[str, Any],
    serialization_context: object,
    cb_render_identity: Mapping[str, Any] | None,
    activation_execution_contract: Mapping[str, Any] | None = None,
    git_commit: str,
    cb_target_name: TargetName = _identity_target,
    delegated_target_name: TargetName = _identity_target,
    source_target_name: TargetName = _identity_target,
    native_source_target_name: TargetName = _identity_target,
    requant_target_name: TargetName = _identity_target,
    source_passthrough_units: Mapping[str, str] | None = None,
    route_pending_passthrough_acknowledged: Iterable[str] = (),
    weight_only_stock_targets: Iterable[str] = (),
    streaming_provenance: bool | None = None,
    include_tensor_formats: bool = False,
) -> dict[str, Any]:
    """Build the complete producer-owned ``quant_config.json`` payload.

    ``source_targets`` is the CT-NORMALIZED passthrough lane (FP8_SOURCE: the
    weight copied verbatim, its FP32 scale plane renamed into the
    compressed-tensors namespace).  ``native_source_targets`` is the
    BYTE-VERBATIM lane, ``{qname: format}``, one config group per format taken
    from :func:`source_passthrough_config_group`.  The split is the wire
    contract, not a taxonomy: see ``_CT_SCALE_DTYPE_NAME``.

    ``requant_targets`` is the RE-QUANTIZED native lane, ``{qname: format}``:
    bytes this producer wrote with a real encoder, under a wire id stock
    compressed-tensors cannot express.  One config group per format, from
    :func:`requant_native_config_group`.  These units DO appear in the
    ``source_passthrough`` declaration, because that record is the
    delegated-native ROUTING map the consumer dispatches on and a unit missing
    from it reads as CB; what they do not carry is the byte-verbatim claim,
    which lives per unit in ``weights.source_passthrough``.

    ``source_passthrough_units`` is ``{unit id: registry format name}`` for the
    units this artifact delegates, and becomes the top-level
    ``source_passthrough`` key.  Empty or ``None`` omits the key entirely —
    its ABSENCE is what marks a legacy all-CB artifact, so an empty record
    would be a different (and false) claim.
    """

    source_targets = list(source_targets)
    native_source_targets = dict(native_source_targets or {})
    requant_targets = dict(requant_targets or {})
    weight_only_stock_targets = set(weight_only_stock_targets)
    assignment_sha = hashlib.sha256(
        json.dumps(
            dict(sorted(assignment.items())),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    imatrix_hasher = hashlib.sha256()
    for qname in sorted(col_weights):
        imatrix_hasher.update(qname.encode())
        imatrix_hasher.update(
            col_weights[qname].to(torch.float32).cpu().numpy().tobytes()
        )
    codebook_sha = {
        name: hashlib.sha256(
            tensor.to(torch.float16).cpu().numpy().tobytes()
        ).hexdigest()
        for name, tensor in codebook_tensors_by_name.items()
    }

    config_groups: dict[str, dict[str, Any]] = {}
    activation_contract_ref = None
    if activation_execution_contract is not None:
        from prismaquant.nvfp4_activation_contract import (
            NVFP4_ACTIVATION_CONTRACT_KEY,
        )

        activation_contract_ref = NVFP4_ACTIVATION_CONTRACT_KEY
    for group_index, ((ref, fmt), qnames) in enumerate(
        sorted(by_group.items())
    ):
        grid, mode, k = cb_targets[qnames[0]]
        scheme = build_cb_scheme(
            ref=ref,
            fmt=fmt,
            grid=grid,
            mode=mode,
            k=k,
            codebook=codebooks[(ref, fmt)],
            scale_coding=scale_coding,
            activation_contract=activation_contract_ref,
        )
        config_groups[f"group_{group_index}"] = {
            "targets": sorted(cb_target_name(qname) for qname in qnames),
            "format": fmt,
            "scheme": scheme,
        }

    stock_by_group: dict[tuple[str, bool], list[str]] = {}
    for qname, fmt in stock_targets.items():
        stock_by_group.setdefault(
            (fmt, qname in weight_only_stock_targets), []
        ).append(qname)
    for (fmt, weight_only), qnames in sorted(stock_by_group.items()):
        group = deepcopy(_STOCK_CT_SCHEMES[fmt])
        if weight_only:
            group["input_activations"] = None
        group["targets"] = sorted(
            _explicit_regex(delegated_target_name(qname))
            for qname in qnames
        )
        config_groups[f"group_{len(config_groups)}"] = group
    if source_targets:
        source_group = deepcopy(FP8_SOURCE_SCHEME)
        source_group["targets"] = sorted(
            _explicit_regex(source_target_name(qname))
            for qname in source_targets
        )
        config_groups[f"group_{len(config_groups)}"] = source_group
    native_by_format: dict[str, list[str]] = {}
    for qname, fmt in native_source_targets.items():
        native_by_format.setdefault(
            source_passthrough_wire(fmt).format_name, []
        ).append(qname)
    for fmt, qnames in sorted(native_by_format.items()):
        # Targets are the CHECKPOINT-spelled bases, anchored the same way the
        # other delegated groups are. The native loader keys off those names,
        # not off the recipe's live spelling, so naming anything else here
        # would describe tensors this artifact does not contain.
        native_group = source_passthrough_config_group(fmt)
        native_group["targets"] = sorted(
            _explicit_regex(native_source_target_name(qname))
            for qname in qnames
        )
        config_groups[f"group_{len(config_groups)}"] = native_group

    requant_by_format: dict[str, list[str]] = {}
    for qname, fmt in requant_targets.items():
        requant_by_format.setdefault(
            fr.canonical_format_name(fmt), []
        ).append(qname)
    for fmt, qnames in sorted(requant_by_format.items()):
        requant_group = requant_native_config_group(fmt)
        requant_group["targets"] = sorted(
            _explicit_regex(requant_target_name(qname))
            for qname in qnames
        )
        config_groups[f"group_{len(config_groups)}"] = requant_group

    provenance: dict[str, Any] = {
        "git_commit": git_commit,
        "assignment_sha256": assignment_sha,
        "imatrix_sha256": imatrix_hasher.hexdigest(),
        "codebook_sha256": codebook_sha,
        "codebook_source": codebook_source,
        "scale_sweep": bool(getattr(serialization_context, "scale_sweep")),
        "encode_tier": getattr(serialization_context, "encode_tier"),
        "renderer_abi": getattr(serialization_context, "renderer_abi"),
        "scale_coding": scale_coding,
        "cb_targets": len(cb_targets),
        "stock_ct_targets": len(stock_targets),
        "fp8_source_targets": len(source_targets),
        # Per-format counts rather than one key per format: a newly censused
        # passthrough shows up here without a provenance schema change.
        "source_passthrough_targets": {
            fmt: len(qnames)
            for fmt, qnames in sorted(native_by_format.items())
        },
        # Same per-format shape, different claim: these units were RE-ENCODED
        # here, so they are deliberately not folded into the count above.
        "requant_native_targets": {
            fmt: len(qnames)
            for fmt, qnames in sorted(requant_by_format.items())
        },
        "serialized_payload": dict(serialized_payload_summary),
        "render_identity_verified": cb_render_identity is not None,
    }
    if streaming_provenance is not None:
        provenance["streaming"] = bool(streaming_provenance)
    acknowledged = sorted(set(route_pending_passthrough_acknowledged))
    if acknowledged:
        # The override was a CLI flag on one machine at one moment; the
        # artifact has to carry the fact that it was used, or "was this shipped
        # knowing no serve route existed?" is unanswerable from the artifact.
        provenance["route_pending_passthrough_acknowledged"] = acknowledged
    if cb_render_identity is not None:
        provenance["cb_render_identity"] = cb_render_identity
    if include_tensor_formats:
        provenance["tensor_formats"] = {
            qname: assignment[qname]
            for qname in sorted(
                set(cb_targets) | set(stock_targets) | set(source_targets)
                | set(native_source_targets)
            )
        }

    quant_config: dict[str, Any] = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_groups": config_groups,
        "ignore": sorted(set(ignore)),
        **({"codebook_file": codebook_file} if codebook_file else {}),
        "provenance": provenance,
    }
    if source_passthrough_units:
        quant_config[SOURCE_PASSTHROUGH_DECLARATION_KEY] = (
            build_source_passthrough_declaration(source_passthrough_units)
        )
        # Read it straight back through the CONSUMER's own parser before the
        # file exists. Every refusal case that parser implements is a load
        # failure at the other end, so the producer must never be the one to
        # generate one.
        parse_source_passthrough_declaration(quant_config)
    if activation_execution_contract is not None:
        quant_config["execution_contracts"] = {
            activation_contract_ref: dict(activation_execution_contract),
        }
    if scale_coding == SCALE_CODING_TWO_TIER:
        # Missing layout_version remains the permanent v1 compatibility rule.
        quant_config["layout_version"] = 2
    return quant_config


__all__ = [
    "DELEGATED_NATIVE_PASSTHROUGH_FORMATS",
    "SOURCE_PASSTHROUGH_DECLARATION_KEY",
    "SOURCE_PASSTHROUGH_DECLARATION_VERSION",
    "SOURCE_PASSTHROUGH_EXPORT_FORMATS",
    "SOURCE_PASSTHROUGH_GROUP_FORMAT",
    "SOURCE_PASSTHROUGH_WIRE_FORMATS",
    "SOURCE_PASSTHROUGH_WIRE_IDS",
    "PassthroughWire",
    "build_cb_scheme",
    "build_quant_config",
    "build_source_passthrough_declaration",
    "cb_scheme_reuse_signature",
    "codebook_tensor_names",
    "codebook_tensors",
    "parse_source_passthrough_declaration",
    "source_passthrough_config_group",
    "source_passthrough_wire",
    "source_passthrough_wire_id",
]
