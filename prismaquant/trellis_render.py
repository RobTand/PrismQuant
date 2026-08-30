"""The production render seam for a Gridbook trellis rung.

WHAT THIS IS
------------
``ProductionWeightCache`` renders every RTN-family format to a *dequantized
tensor*: the cache stores what the weight becomes, the surrogate scores it, and
the exporter re-packs it.  That contract cannot carry a trellis rung, and the
reason is a fact about the wire rather than a preference.

``gridbook/trellis_scheme.py:11-19`` states it from the serving side: a
``TrellisWire`` serializes to ONE blob carrying the header, the per-column rate
schedule, the tight block offsets, the per-rate alphabets, the scale plane and
the padded row bodies -- and "the schedule, the alphabets and the scale plane
exist nowhere else".  A body-shaped payload cannot rebuild a wire.  So the
checkpoint carries one opaque ``wire_bytes`` blob per Linear, and the cache that
feeds it must carry the same object.

THE SHAPE, AND WHY IT IS NOT THE CB SHAPE
-----------------------------------------
The NVFP4-CB lane solves the neighbouring problem by carrying a *recipe*:
``CBSerializationContext`` is a description (knobs plus codebook digests, never
bytes), the cache holds the dequantized render for cost and KL, and the exporter
RE-ENCODES from the source checkpoint weight
(``export_nvfp4_cb.py:1652`` -> ``:1684``; ``run-pipeline.sh:544`` states the
posture outright).  That works because ``nvfp4_cb_pack`` is a deterministic pure
function of (source weight, recipe).

A trellis encode is not.  Three facts, each sufficient on its own:

1. The chunked encoder normalizes branch costs by the CHUNK's mean
   (``stage5_encoder.py:173`` inside the loop at ``:744-748``), so the fp32
   metrics the Viterbi accumulates depend on ``sb_chunk`` -- a throughput knob.
   The campaign's own provenance gate already treats ``sb_chunk`` as
   identity-bearing.
2. Encode determinism was OFF in the canonical measurement, and the guard is
   ``torch.use_deterministic_algorithms(True, warn_only=True)``; the search
   breaks ties through ``topk`` and ``argmin``.
3. The cache narrows fp32 to bf16 on write
   (``production_weight_cache._canonical_rendered_weight_tensor``), so nothing
   downstream can recover exact level indices from a cached tensor -- the
   encoder's own round-trip helper asserts ``torch.equal(lv[pick] * pes, q)``,
   which bf16 breaks.

So the blob is primary and the dequantized tensor is a derived view.
``render_production_weight`` keeps returning a tensor; the blob leaves through a
MANDATORY-for-trellis sink.  A trellis render with no sink is refused, exactly
as a CB render with no ``CBSerializationContext`` or no ``col_weights`` is
refused (``production_weight_cache.py:4324-4339``).  A silently wire-less
trellis render is not a cheaper render; it is a render whose artifact was
discarded, and this project has been burned three times by a plausible-looking
zero.

STATUS
------
PrismaQuant owns an independent wire-v1 writer and the promoted Stage-5
encoder.  Cross-repository agreement is frozen by a Gridbook golden vector;
production source does not import Gridbook.  A render is reachable only with a
value-bearing :class:`TrellisEncodePlan`: assignment digests alone cannot
recover the schedule or alphabets and are therefore still refused.

Design: ``/home/rob/dq-runs/trellis-render-design.md``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os

import torch

from .trellis_formats import parse_trellis_format_name


#: Identity of one cached (qname, trellis rung) pair.  Sibling of
#: ``production_weight_cache.CB_CACHE_PAIR_IDENTITY_SCHEMA``.
TRELLIS_PAIR_IDENTITY_SCHEMA = (
    "prismaquant.production_weight_cache.trellis_pair_identity.v1"
)
TRELLIS_PAIR_SIDECAR_SCHEMA = (
    "prismaquant.production_weight_cache.trellis_pair_sidecar.v1"
)
TRELLIS_ENCODE_PLAN_SET_SCHEMA = "prismaquant.trellis_encode_plan_set.v1"

#: The wire this repo writes.  A version change is a different artifact, not a
#: compatible one, so it is identity-bearing.  Read out of
#: ``gridbook/trellis.py:27`` and pinned here rather than imported.
TRELLIS_WIRE_SCHEMA = "gridbook.trellis.wire.v1"

#: The executed activation contract per family, read out of the lane sources
#: (``gridbook/trellis_e4m3_lane.py:106``, ``trellis_e2m1_lane.py:71,97``).
#: Both native routes are A=W and neither lane has a BF16 fallback: the
#: ``build_*_method`` factories refuse the unit rather than expand to bf16.
#:
#: This constant is what the RENDER is priced under.  It is NOT an attestation
#: that the pinned runtime serves it -- the serving pin publishes no
#: ``lane_eligibility`` table at all, so every trellis unit resolves
#: ``unattested`` today (principle 14).  Route status is resolved through
#: ``gridbook_lane_eligibility``, never from this dict.
EXECUTED_ACTIVATION_CONTRACT: Mapping[str, str] = {
    "TCQ_E2M1_R256": "e2m1_group16_ue4m3_static",
    "TCQ_E4M3_R256": "fp8_per_token_dynamic",
}


class TrellisRenderError(RuntimeError):
    """A trellis production render cannot be produced or admitted."""


class TrellisEncoderUnavailableError(TrellisRenderError):
    """Kept for old callers; current refusals name the unsupported setting."""


class TrellisWireSinkError(TrellisRenderError):
    """A wire sink was used outside its single-assignment contract."""


def _canonical_json_sha256(value: object, *, where: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TrellisRenderError(
            f"{where} is not canonically serializable: {exc}"
        ) from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def trellis_tensor_value_sha256(tensor: torch.Tensor) -> str:
    """Hash numeric values as little-endian fp32 in C order.

    This is the same value identity used by ``ProductionWeightCache`` for CB
    source weights and imatrix vectors.  Device placement and source dtype are
    deliberately not part of the value digest; dtype is bound separately.
    """
    value = torch.as_tensor(tensor).detach().reshape(-1)
    digest = hashlib.sha256()
    chunk = 4 * 1024 * 1024
    for first in range(0, int(value.numel()), chunk):
        cpu = value[first:first + chunk].to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        digest.update(cpu.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TrellisRenderRecipe:
    """Everything that changes the encoded bits, and nothing that does not.

    Every field here is identity-bearing on purpose.  ``sb_chunk`` looks like a
    throughput knob and is one, but the chunk-scoped cost normalization
    (``stage5_encoder.py:173``) makes it change the fp32 metrics the Viterbi
    accumulates, and the campaign's ``phase6_kl.py check_provenance`` already
    refuses an arm whose ``sb_chunk`` differs from the teacher cache's.  Binding
    it costs nothing; not binding it makes a cache hit unprovable.

    ``encoder_source_sha256`` hashes the encoder's ``.py`` SOURCE files only.
    A hash that covers ``__pycache__`` changes when nothing semantic did.
    """

    family: str
    body_rate_q256: int
    layout: str
    schedule_identity_sha256: str
    alphabet_identity_sha256: str
    pre_render_recipe_identity_sha256: str
    encoder_source_sha256: str
    encoder_sb_chunk: int
    encoder_determinism_mode: str
    encoder_tailbite_candidates: int
    encoder_backend: str
    encoder_point_route: str
    scale_rule: str
    wire_schema: str = TRELLIS_WIRE_SCHEMA

    def __post_init__(self) -> None:
        if self.family not in EXECUTED_ACTIVATION_CONTRACT:
            raise TrellisRenderError(
                f"unknown trellis family {self.family!r}; the executed "
                f"activation contract is only known for "
                f"{sorted(EXECUTED_ACTIVATION_CONTRACT)}"
            )
        if int(self.body_rate_q256) <= 0:
            raise TrellisRenderError("body_rate_q256 must be positive")
        if int(self.encoder_sb_chunk) < 1:
            raise TrellisRenderError("encoder_sb_chunk must be positive")
        if int(self.encoder_tailbite_candidates) < 1:
            raise TrellisRenderError(
                "encoder_tailbite_candidates must be positive"
            )
        if str(self.encoder_determinism_mode) not in {"on", "off"}:
            raise TrellisRenderError(
                "encoder_determinism_mode must be 'on' or 'off'; the two "
                "phases of the canonical campaign use opposite modes and a "
                "defaulted answer is indistinguishable from a wrong one"
            )
        if str(self.encoder_backend) not in {"eager", "triton"}:
            raise TrellisRenderError(
                "encoder_backend must be 'eager' or 'triton'"
            )
        if str(self.encoder_point_route) not in {"full", "windowed"}:
            raise TrellisRenderError(
                "encoder_point_route must be 'full' or 'windowed'"
            )
        if str(self.wire_schema) != TRELLIS_WIRE_SCHEMA:
            raise TrellisRenderError(
                f"unsupported trellis wire schema {self.wire_schema!r}; "
                f"this producer writes only {TRELLIS_WIRE_SCHEMA!r}"
            )
        expected_scale_rule = (
            "static_6"
            if self.family == "TCQ_E2M1_R256"
            else "row_fp32_amax_448"
        )
        if str(self.scale_rule) != expected_scale_rule:
            raise TrellisRenderError(
                f"{self.family} scale_rule must be {expected_scale_rule!r}; "
                "other planes have no bit-exact production writer"
            )
        for field in (
            "schedule_identity_sha256",
            "alphabet_identity_sha256",
            "pre_render_recipe_identity_sha256",
            "encoder_source_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64 or (
                value != value.lower()
            ) or any(ch not in "0123456789abcdef" for ch in value):
                raise TrellisRenderError(
                    f"{field} must be a lowercase SHA-256 hex digest"
                )

    @property
    def activation_contract(self) -> str:
        """The contract the native route EXECUTES for this family."""
        return EXECUTED_ACTIVATION_CONTRACT[self.family]

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "body_rate_q256": int(self.body_rate_q256),
            "layout": self.layout,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "alphabet_identity_sha256": self.alphabet_identity_sha256,
            "pre_render_recipe_identity_sha256": (
                self.pre_render_recipe_identity_sha256
            ),
            "encoder_source_sha256": self.encoder_source_sha256,
            "encoder_sb_chunk": int(self.encoder_sb_chunk),
            "encoder_determinism_mode": str(self.encoder_determinism_mode),
            "encoder_tailbite_candidates": int(
                self.encoder_tailbite_candidates
            ),
            "encoder_backend": str(self.encoder_backend),
            "encoder_point_route": str(self.encoder_point_route),
            "scale_rule": str(self.scale_rule),
            "wire_schema": str(self.wire_schema),
            "activation_contract": self.activation_contract,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_json_sha256(
            self.as_dict(), where="trellis render recipe"
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrellisRenderRecipe":
        """Parse a persisted recipe without accepting implicit defaults."""
        if not isinstance(value, Mapping):
            raise TrellisRenderError("trellis render recipe must be an object")
        fields = (
            "family",
            "body_rate_q256",
            "layout",
            "schedule_identity_sha256",
            "alphabet_identity_sha256",
            "pre_render_recipe_identity_sha256",
            "encoder_source_sha256",
            "encoder_sb_chunk",
            "encoder_determinism_mode",
            "encoder_tailbite_candidates",
            "encoder_backend",
            "encoder_point_route",
            "scale_rule",
            "wire_schema",
        )
        missing = [field for field in fields if field not in value]
        if missing:
            raise TrellisRenderError(
                f"trellis render recipe is missing required fields {missing}"
            )
        parsed = cls(**{field: value[field] for field in fields})  # type: ignore[arg-type]
        declared_activation = value.get("activation_contract")
        if declared_activation is None:
            raise TrellisRenderError(
                "trellis render recipe is missing required field "
                "'activation_contract'"
            )
        if str(declared_activation) != parsed.activation_contract:
            raise TrellisRenderError(
                "trellis render recipe activation_contract differs from the "
                f"native A=W route: declared={declared_activation!r}, "
                f"executed={parsed.activation_contract!r}"
            )
        return parsed


@dataclass(frozen=True)
class TrellisEncodePlan:
    """Value-bearing input needed to turn an allocated rung into a wire.

    The allocator-facing TCQ name and serialized candidate identity carry only
    schedule/alphabet digests.  Neither is reversible.  A renderer therefore
    accepts the actual values plus the exact footprint that was priced and
    recomputes every digest/byte count before encoding.  This closes a gap in
    the original seam skeleton: looking up mutable campaign data by ambient
    path would make a cache key appear valid while encoding a different rung.
    """

    shape: tuple[int, int]
    schedule: tuple[int, ...]
    alphabets: Mapping[int, tuple[int, ...]]
    priced_footprint: Mapping[str, object]
    recipe: TrellisRenderRecipe
    col_weights_sha256: str
    measured_activation_contract: str
    activation_input_global_scale: float | None

    def __post_init__(self) -> None:
        from .trellis_footprint import (
            trellis_tensor_payload_breakdown,
            validate_trellis_tensor_payload_breakdown,
        )

        shape = tuple(int(value) for value in self.shape)
        schedule = tuple(int(value) for value in self.schedule)
        alphabets = {
            int(rate): tuple(int(code) for code in codes)
            for rate, codes in self.alphabets.items()
        }
        if len(shape) != 2 or min(shape) <= 0:
            raise TrellisRenderError(
                f"trellis encode plan shape must be rank-2 positive, got {shape}"
            )
        if len(schedule) != shape[1]:
            raise TrellisRenderError(
                f"trellis schedule has {len(schedule)} columns, expected "
                f"{shape[1]}"
            )
        footprint = validate_trellis_tensor_payload_breakdown(
            self.priced_footprint
        )
        if int(footprint.get("sidecar_header_bytes", -1)) != 0:
            raise TrellisRenderError(
                "trellis wire pricing must set sidecar_header_bytes=0; the "
                "pair sidecar is cache metadata, not bytes inside the blob"
            )
        recomputed = trellis_tensor_payload_breakdown(
            shape,
            family=self.recipe.family,
            body_rate_q256=int(self.recipe.body_rate_q256),
            layout=self.recipe.layout,
            schedule=schedule,
            alphabets=alphabets,
        )
        if footprint != recomputed:
            raise TrellisRenderError(
                "value-bearing trellis plan differs from the priced footprint; "
                "refusing to encode a schedule/alphabet recipe the allocator "
                "did not price"
            )
        for field in (
            "schedule_identity_sha256",
            "alphabet_identity_sha256",
            "pre_render_recipe_identity_sha256",
        ):
            if getattr(self.recipe, field) != footprint[field]:
                raise TrellisRenderError(
                    f"trellis recipe {field} differs from the priced values"
                )
        if str(self.measured_activation_contract) != self.recipe.activation_contract:
            raise TrellisRenderError(
                "trellis anchor activation contract differs from the native "
                f"A=W render: measured={self.measured_activation_contract!r}, "
                f"executed={self.recipe.activation_contract!r}; refusing to "
                "round an unattested W*A16 price up to A=W"
            )
        activation_scale = self.activation_input_global_scale
        if self.recipe.family == "TCQ_E2M1_R256":
            if (
                activation_scale is None
                or not math.isfinite(float(activation_scale))
                or float(activation_scale) <= 0.0
            ):
                raise TrellisRenderError(
                    "E2M1 trellis plans require the exact positive finite "
                    "activation_input_global_scale used by the measured "
                    "W4A4 contract; it is an artifact value outside the wire"
                )
            activation_scale = float(activation_scale)
        elif activation_scale is not None:
            raise TrellisRenderError(
                "E4M3 trellis activation scaling is per-token dynamic; an "
                "activation_input_global_scale would describe another contract"
            )
        digest = str(self.col_weights_sha256).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise TrellisRenderError(
                "trellis encode plan col_weights_sha256 must be lowercase SHA-256"
            )
        parsed = parse_trellis_format_name(str(footprint["format"]))
        if parsed is None or (
            parsed[0].family != self.recipe.family
            or parsed[1] != self.recipe.body_rate_q256
        ):
            raise TrellisRenderError("trellis plan format and recipe disagree")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "alphabets", alphabets)
        object.__setattr__(self, "priced_footprint", copy.deepcopy(footprint))
        object.__setattr__(self, "col_weights_sha256", digest)
        object.__setattr__(
            self, "activation_input_global_scale", activation_scale
        )

    @property
    def fmt(self) -> str:
        return str(self.priced_footprint["format"])

    @property
    def expected_wire_bytes(self) -> int:
        return int(self.priced_footprint["total_bytes"])

    def as_dict(self) -> dict[str, object]:
        return {
            "format": self.fmt,
            "shape": [int(value) for value in self.shape],
            "schedule": [int(value) for value in self.schedule],
            "alphabets": {
                str(rate): [int(code) for code in codes]
                for rate, codes in sorted(self.alphabets.items())
            },
            "priced_footprint": copy.deepcopy(dict(self.priced_footprint)),
            "recipe": self.recipe.as_dict(),
            "col_weights_sha256": self.col_weights_sha256,
            "measured_activation_contract": self.measured_activation_contract,
            "activation_input_global_scale": self.activation_input_global_scale,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrellisEncodePlan":
        """Parse one exact plan from the versioned handoff representation."""
        if not isinstance(value, Mapping):
            raise TrellisRenderError("trellis encode plan must be an object")
        fields = (
            "shape",
            "schedule",
            "alphabets",
            "priced_footprint",
            "recipe",
            "col_weights_sha256",
            "measured_activation_contract",
            "activation_input_global_scale",
        )
        missing = [field for field in fields if field not in value]
        if missing:
            raise TrellisRenderError(
                f"trellis encode plan is missing required fields {missing}"
            )
        shape = value["shape"]
        schedule = value["schedule"]
        alphabets = value["alphabets"]
        footprint = value["priced_footprint"]
        recipe = value["recipe"]
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or not isinstance(schedule, Sequence)
            or isinstance(schedule, (str, bytes))
            or not isinstance(alphabets, Mapping)
            or not isinstance(footprint, Mapping)
            or not isinstance(recipe, Mapping)
        ):
            raise TrellisRenderError(
                "trellis encode plan shape/schedule/alphabets/footprint/recipe "
                "have invalid container types"
            )
        normalized_alphabets: dict[int, tuple[int, ...]] = {}
        for rate, codes in alphabets.items():
            if not isinstance(codes, Sequence) or isinstance(codes, (str, bytes)):
                raise TrellisRenderError(
                    f"trellis alphabet for rate {rate!r} must be a list"
                )
            normalized_alphabets[int(rate)] = tuple(int(code) for code in codes)
        parsed = cls(
            shape=tuple(int(item) for item in shape),
            schedule=tuple(int(item) for item in schedule),
            alphabets=normalized_alphabets,
            priced_footprint=dict(footprint),
            recipe=TrellisRenderRecipe.from_mapping(recipe),
            col_weights_sha256=str(value["col_weights_sha256"]),
            measured_activation_contract=str(
                value["measured_activation_contract"]
            ),
            activation_input_global_scale=(
                None
                if value["activation_input_global_scale"] is None
                else float(value["activation_input_global_scale"])
            ),
        )
        declared_format = value.get("format")
        if declared_format is not None and str(declared_format).upper() != parsed.fmt:
            raise TrellisRenderError(
                f"trellis plan declares format {declared_format!r}, but its "
                f"priced footprint declares {parsed.fmt!r}"
            )
        return parsed


def load_trellis_encode_plan_set(
    path: str | bytes | os.PathLike[str],
) -> dict[tuple[str, str], TrellisEncodePlan]:
    """Load the explicit qname/format -> value-bearing plan handoff.

    This loader deliberately has no manifest lookup or environment fallback.
    The allocation/measurement stage must materialize the exact values it
    priced; hashes in ``layer_config.json`` cannot be inverted into them.
    """
    from pathlib import Path

    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrellisRenderError(
            f"cannot read trellis encode plan set {resolved}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TrellisRenderError("trellis encode plan set must be an object")
    if payload.get("schema") != TRELLIS_ENCODE_PLAN_SET_SCHEMA:
        raise TrellisRenderError(
            f"trellis encode plan set schema must be "
            f"{TRELLIS_ENCODE_PLAN_SET_SCHEMA!r}"
        )
    records = payload.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TrellisRenderError("trellis encode plan set records must be a list")
    result: dict[tuple[str, str], TrellisEncodePlan] = {}
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise TrellisRenderError(
                f"trellis encode plan record {index} must be an object"
            )
        qname = str(raw.get("qname", ""))
        if not qname:
            raise TrellisRenderError(
                f"trellis encode plan record {index} has no qname"
            )
        plan = TrellisEncodePlan.from_mapping(raw)
        key = (qname, plan.fmt)
        if key in result:
            raise TrellisRenderError(
                f"duplicate trellis encode plan for {qname}@{plan.fmt}"
            )
        result[key] = plan
    return result


class TrellisWireSink:
    """Single-assignment carrier for one rendered wire blob.

    The render returns a dequantized tensor for every existing consumer; the
    blob -- the artifact -- comes out here.  Two properties matter:

    * **Single assignment.** A second :meth:`accept` raises.  A sink reused
      across two Linears would silently attribute one Linear's bytes to
      another's identity.
    * **Exact bytes, checked here.** ``expected_bytes`` is the candidate's
      ``footprint['total_bytes']``, which ``trellis_footprint`` computes
      exactly.  A blob of a different length means the encoder and the pricing
      disagree, so the DP was solved against a number the render cannot honour.
      That is a refusal, not a warning.
    """

    __slots__ = (
        "_qname", "_fmt", "_expected_bytes", "_plan", "_blob", "_recipe",
    )

    def __init__(
        self,
        *,
        qname: str,
        fmt: str,
        plan: TrellisEncodePlan,
        expected_bytes: int | None = None,
    ) -> None:
        canonical_fmt = str(fmt).strip().upper()
        if parse_trellis_format_name(canonical_fmt) is None:
            raise TrellisWireSinkError(
                f"{fmt!r} is not a trellis rung; a wire sink is meaningless "
                f"for it"
            )
        if not isinstance(plan, TrellisEncodePlan):
            raise TrellisWireSinkError(
                "a trellis wire sink requires a value-bearing encode plan; "
                "recipe digests cannot reconstruct schedules or alphabets"
            )
        if canonical_fmt != plan.fmt:
            raise TrellisWireSinkError(
                f"wire sink format {canonical_fmt} differs from plan {plan.fmt}"
            )
        priced_bytes = plan.expected_wire_bytes
        if expected_bytes is not None and int(expected_bytes) != priced_bytes:
            raise TrellisWireSinkError(
                f"wire sink expected_bytes={expected_bytes} differs from the "
                f"priced plan's {priced_bytes}"
            )
        if priced_bytes <= 0:
            raise TrellisWireSinkError("expected_bytes must be positive")
        self._qname = str(qname)
        self._fmt = canonical_fmt
        self._expected_bytes = priced_bytes
        self._plan = plan
        self._blob: bytes | None = None
        self._recipe: TrellisRenderRecipe | None = None

    @property
    def qname(self) -> str:
        return self._qname

    @property
    def fmt(self) -> str:
        return self._fmt

    @property
    def expected_bytes(self) -> int:
        return self._expected_bytes

    @property
    def plan(self) -> TrellisEncodePlan:
        return self._plan

    @property
    def filled(self) -> bool:
        return self._blob is not None

    def accept(self, blob: bytes, *, recipe: TrellisRenderRecipe) -> None:
        if self._blob is not None:
            raise TrellisWireSinkError(
                f"{self._qname}@{self._fmt}: wire sink already holds a blob; "
                f"a sink carries exactly one render"
            )
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise TrellisWireSinkError(
                f"{self._qname}@{self._fmt}: a wire is bytes, got "
                f"{type(blob).__name__}"
            )
        blob = bytes(blob)
        if len(blob) != self._expected_bytes:
            raise TrellisWireSinkError(
                f"{self._qname}@{self._fmt}: rendered wire is {len(blob)} "
                f"bytes but the candidate was priced at "
                f"{self._expected_bytes}; the encoder and "
                f"trellis_footprint disagree, so the allocation was solved "
                f"against a byte count this render cannot honour"
            )
        if not isinstance(recipe, TrellisRenderRecipe):
            raise TrellisWireSinkError(
                f"{self._qname}@{self._fmt}: a wire without its render recipe "
                f"cannot be admitted to the cache"
            )
        if recipe != self._plan.recipe:
            raise TrellisWireSinkError(
                f"{self._qname}@{self._fmt}: rendered recipe differs from "
                "the value-bearing plan supplied to the sink"
            )
        self._blob = blob
        self._recipe = recipe

    @property
    def blob(self) -> bytes:
        if self._blob is None:
            raise TrellisWireSinkError(
                f"{self._qname}@{self._fmt}: wire sink is empty; the render "
                f"did not produce a wire"
            )
        return self._blob

    @property
    def recipe(self) -> TrellisRenderRecipe:
        if self._recipe is None:
            raise TrellisWireSinkError(
                f"{self._qname}@{self._fmt}: wire sink is empty"
            )
        return self._recipe

    @property
    def rendered_wire_identity_sha256(self) -> str:
        """SHA-256 of the exact blob bytes.

        This is the through-line.  It fills the slot
        ``TrellisAllocatorCandidate.as_dict`` reserves as ``None``
        (``trellis_allocator.py:534``) and that
        ``docs/design/trellis_rate_surface.md:51`` describes as staying null
        "until a future renderer hashes actual wire bytes".  It travels
        sidecar -> selection payload -> shipcard, and export re-hashes the
        bytes it ships and refuses on disagreement.  It is deliberately NOT
        written back into the footprint:
        ``validate_trellis_tensor_payload_breakdown`` refuses a pre-render
        footprint that claims a wire identity (``trellis_footprint.py:427``),
        and that refusal is correct.
        """
        return hashlib.sha256(self.blob).hexdigest()

    def as_wire_tensor(self) -> torch.Tensor:
        """The blob as the 1-D uint8 tensor the cache shard stores."""
        return torch.frombuffer(bytearray(self.blob), dtype=torch.uint8)


def build_trellis_pair_identity(
    *,
    qname: str,
    fmt: str,
    shape: Sequence[int],
    recipe: TrellisRenderRecipe,
    source_weight_sha256: str,
    source_weight_dtype: str,
    col_weights_sha256: str,
    col_weights_shape: Sequence[int],
    activation_input_global_scale: float | None,
    calibration_hash: str,
    git_commit: str,
    producer_source_sha256: str,
) -> dict[str, object]:
    """The admission identity for one cached trellis pair.

    Modelled on ``_build_cb_cache_pair_identity``
    (``production_weight_cache.py:2300-2401``): canonical JSON, every field
    value-bearing, and admission refuses on the FIRST difference rather than
    reporting a boolean.

    The field that does not appear in the CB version and must appear here is
    the recipe digest.  The cache key is ``(qname, fmt)`` and the TCQ spelling
    is deliberately shape-free and recipe-free so fused-sibling aggregation can
    intersect member menus by format name (``trellis_menu.py:396-403``).  Two
    manifests can therefore produce the same ``TCQ_E2M1_R640`` key for one
    qname with different schedules.  The key cannot disambiguate them; this
    identity does.
    """
    dims = [int(dim) for dim in shape]
    if len(dims) != 2 or min(dims) <= 0:
        raise TrellisRenderError(
            f"{qname}@{fmt}: a trellis render needs a rank-2 shape, got {dims}"
        )
    body = {
        "schema": TRELLIS_PAIR_IDENTITY_SCHEMA,
        "qname": str(qname),
        "format": str(fmt).strip().upper(),
        "shape": dims,
        "recipe": recipe.as_dict(),
        "recipe_identity_sha256": recipe.identity_sha256,
        "source_weight_sha256": str(source_weight_sha256).lower(),
        "source_weight_dtype": str(source_weight_dtype),
        "col_weights_sha256": str(col_weights_sha256).lower(),
        "col_weights_shape": [int(dim) for dim in col_weights_shape],
        "activation_contract": recipe.activation_contract,
        "activation_input_global_scale": (
            None
            if activation_input_global_scale is None
            else float(activation_input_global_scale)
        ),
        "calibration_hash": str(calibration_hash),
        "git_commit": str(git_commit).lower(),
        "producer_source_sha256": str(producer_source_sha256).lower(),
    }
    return body


def render_trellis_production_weight(
    weight: torch.Tensor,
    fmt: str,
    *,
    qname: str,
    col_weights: torch.Tensor,
    trellis_wire_out: TrellisWireSink,
    recipe: TrellisRenderRecipe | None = None,
    weighted_vq: bool = True,
    joint_global_real: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode once, pack once, then return the blob's decoded view."""
    canonical_fmt = str(fmt).strip().upper()
    parsed = parse_trellis_format_name(canonical_fmt)
    if parsed is None:
        raise TrellisRenderError(f"{fmt!r} is not a trellis rung")
    if col_weights is None:
        raise TrellisRenderError(
            f"{qname}={fmt}: production trellis render has no col_weights; "
            f"the encoder objective is importance-weighted "
            f"(stage5_encoder.py:672, oa_common.py:219), so an unweighted "
            f"cache entry is not the artifact"
        )
    if trellis_wire_out is None:
        raise TrellisRenderError(
            f"{qname}={fmt}: production trellis render has no wire sink; "
            f"the wire IS the artifact and a render that discards it cannot "
            f"be exported, KL-validated or priced"
        )
    if not bool(weighted_vq):
        raise TrellisRenderError(
            f"{qname}={fmt}: weighted_vq cannot be disabled for a trellis "
            "render; the priced Viterbi objective is column-importance "
            "weighted"
        )
    if joint_global_real is not None:
        raise TrellisRenderError(
            f"{qname}={fmt}: fused-sibling joint_global_real is not part of "
            "the qualified trellis scale contract. E2M1 owns its exact "
            "static-6 tensor global and E4M3 owns one fp32 scale per row; "
            "refusing to transplant NVFP4 joint-scaling semantics."
        )
    if trellis_wire_out.qname != str(qname):
        raise TrellisWireSinkError(
            f"wire sink belongs to {trellis_wire_out.qname!r}, not {qname!r}"
        )
    if trellis_wire_out.fmt != canonical_fmt:
        raise TrellisWireSinkError(
            f"wire sink belongs to {trellis_wire_out.fmt}, not {canonical_fmt}"
        )
    # ``frozen=True`` prevents rebinding the plan fields, but the two mapping
    # fields are intentionally ordinary JSON-shaped containers and a caller
    # can still mutate them after sink construction.  Reparse at the trust
    # boundary so the values sent to the encoder are rechecked against the
    # priced footprint and immutable recipe digests.  Without this check a
    # post-construction alphabet edit could encode bytes other than the rung
    # the allocator priced while retaining the original cache identity.
    try:
        plan = TrellisEncodePlan.from_mapping(trellis_wire_out.plan.as_dict())
    except TrellisRenderError:
        raise
    except Exception as exc:
        raise TrellisRenderError(
            f"{qname}={fmt}: value-bearing plan failed trust-boundary "
            f"revalidation: {exc}"
        ) from exc
    if tuple(int(value) for value in weight.shape) != plan.shape:
        raise TrellisRenderError(
            f"{qname}={fmt}: source shape {tuple(weight.shape)} differs from "
            f"priced plan {plan.shape}"
        )
    if parsed[0].family != plan.recipe.family or parsed[1] != (
        plan.recipe.body_rate_q256
    ):
        raise TrellisRenderError(
            f"{qname}={fmt}: sink plan family/rate differs from format"
        )
    selected_recipe = plan.recipe if recipe is None else recipe
    if selected_recipe != plan.recipe:
        raise TrellisRenderError(
            f"{qname}={fmt}: explicit recipe differs from sink encode plan"
        )
    observed_col_weights = trellis_tensor_value_sha256(col_weights)
    if observed_col_weights != plan.col_weights_sha256:
        raise TrellisRenderError(
            f"{qname}={fmt}: col_weights value differs from the vector under "
            "which this rung was priced; refusing a currency mismatch"
        )

    from .trellis_encoder import (
        encode_trellis_planes,
        encoder_source_sha256,
    )
    from .trellis_wire import decode_values_torch, pack_planes

    current_encoder_sha256 = encoder_source_sha256()
    if selected_recipe.encoder_source_sha256 != current_encoder_sha256:
        raise TrellisRenderError(
            f"{qname}={fmt}: encoder source digest differs from the priced "
            f"recipe: planned={selected_recipe.encoder_source_sha256}, "
            f"current={current_encoder_sha256}; refusing to re-encode under "
            "different source"
        )
    encoded = encode_trellis_planes(
        weight,
        col_weights,
        family=selected_recipe.family,
        schedule=plan.schedule,
        alphabets=plan.alphabets,
        scale_rule=selected_recipe.scale_rule,
        sb_chunk=selected_recipe.encoder_sb_chunk,
        determinism_mode=selected_recipe.encoder_determinism_mode,
        tailbite_candidates=selected_recipe.encoder_tailbite_candidates,
        backend=selected_recipe.encoder_backend,
        point_route=selected_recipe.encoder_point_route,
    )
    wire = pack_planes(
        family=selected_recipe.family,
        body_rate_q256=selected_recipe.body_rate_q256,
        schedule=plan.schedule,
        layout=selected_recipe.layout,
        u_bits=encoded.u_bits,
        point_indices=encoded.point_indices,
        bypass_codes=encoded.bypass_codes,
        alphabets=plan.alphabets,
        scale_blob=encoded.scale_blob,
        global_scale_real=encoded.global_scale_real,
    )
    blob = wire.to_bytes()
    decoded = decode_values_torch(
        blob, device=weight.device, dtype=weight.dtype
    )
    if not torch.equal(decoded, encoded.reconstruction.to(weight.dtype)):
        mismatch = int(
            (decoded != encoded.reconstruction.to(weight.dtype)).sum().item()
        )
        raise TrellisRenderError(
            f"{qname}={fmt}: independent wire decode differs from encoder "
            f"reconstruction at {mismatch} values; refusing a blob the "
            "surrogate did not score"
        )
    trellis_wire_out.accept(blob, recipe=selected_recipe)
    return decoded.contiguous()


__all__ = [
    "EXECUTED_ACTIVATION_CONTRACT",
    "TRELLIS_PAIR_IDENTITY_SCHEMA",
    "TRELLIS_PAIR_SIDECAR_SCHEMA",
    "TRELLIS_WIRE_SCHEMA",
    "TrellisEncoderUnavailableError",
    "TrellisEncodePlan",
    "TrellisRenderError",
    "TrellisRenderRecipe",
    "TrellisWireSink",
    "TrellisWireSinkError",
    "build_trellis_pair_identity",
    "render_trellis_production_weight",
    "trellis_tensor_value_sha256",
]
