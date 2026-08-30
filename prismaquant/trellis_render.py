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
SKELETON.  The seam, the identity and the refusals are real.  The encoder and
the wire writer are NOT here: PrismaQuant may not import the Gridbook runtime
(``AGENTS.md:48-49``) and Gridbook ships no encoder
(``tools/make_trellis_smoke_checkpoint.py:9-15``), so the codec must be acquired
into this repo the way the GGUF writer was, with a bit-exactness gate against
the runtime's own reference.  Until then :func:`render_trellis_production_weight`
raises :class:`TrellisEncoderUnavailableError` naming the missing dependency.
It does not stub, approximate, or fall back to RTN.

Design: ``/home/rob/dq-runs/trellis-render-design.md``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json

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
    """The Viterbi encoder / wire writer is not part of this repository yet."""


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
            "wire_schema": str(self.wire_schema),
            "activation_contract": self.activation_contract,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_json_sha256(
            self.as_dict(), where="trellis render recipe"
        )


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

    __slots__ = ("_qname", "_fmt", "_expected_bytes", "_blob", "_recipe")

    def __init__(
        self,
        *,
        qname: str,
        fmt: str,
        expected_bytes: int,
    ) -> None:
        if parse_trellis_format_name(str(fmt).strip().upper()) is None:
            raise TrellisWireSinkError(
                f"{fmt!r} is not a trellis rung; a wire sink is meaningless "
                f"for it"
            )
        if int(expected_bytes) <= 0:
            raise TrellisWireSinkError("expected_bytes must be positive")
        self._qname = str(qname)
        self._fmt = str(fmt).strip().upper()
        self._expected_bytes = int(expected_bytes)
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
) -> torch.Tensor:
    """Encode ``weight`` to a trellis wire, fill the sink, return ``decode``.

    NOT IMPLEMENTED, and deliberately loud about why.

    The Viterbi encoder lives in ``/home/rob/dq-runs/trellis-stage0``
    (``stage5_encoder.mixed_tcq_encode_batched``, ``tcq_pilot.tcq_encode``) and
    the wire packer lives in the Gridbook runtime, which
    ``AGENTS.md:48-49`` forbids PrismaQuant from importing.  Gridbook ships no
    encoder of its own
    (``tools/make_trellis_smoke_checkpoint.py:9-15``: "There is no weight->wire
    encoder in this package").  Closing that gap is a dependency-acquisition
    task with a precedent -- GGUF, where PrismaQuant owns the writer and the
    cross-repository agreement is a bit-exactness gate against the other side's
    reference (``AGENTS.md:36-38``) -- not something a render seam may
    improvise.

    Raising here rather than stubbing is the point.  A stub that returned an
    RTN render and an empty wire would satisfy every type in this module and
    ship an artifact unrelated to the allocated budget.
    """
    if parse_trellis_format_name(str(fmt).strip().upper()) is None:
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
    raise TrellisEncoderUnavailableError(
        f"{qname}={fmt}: no trellis encoder is available in this repository. "
        f"The Viterbi encoder lives in /home/rob/dq-runs/trellis-stage0 "
        f"(stage5_encoder.mixed_tcq_encode_batched) and the wire writer for "
        f"{TRELLIS_WIRE_SCHEMA} lives in the Gridbook runtime, which "
        f"PrismaQuant must never import (AGENTS.md:48-49). Both must be "
        f"acquired into prismaquant/ with a bit-exactness gate against the "
        f"runtime's reference decoder before a trellis rung can render. "
        f"Refusing rather than returning an RTN render with an empty wire: a "
        f"plausible-looking substitute here would ship bytes unrelated to the "
        f"allocated budget."
    )


__all__ = [
    "EXECUTED_ACTIVATION_CONTRACT",
    "TRELLIS_PAIR_IDENTITY_SCHEMA",
    "TRELLIS_PAIR_SIDECAR_SCHEMA",
    "TRELLIS_WIRE_SCHEMA",
    "TrellisEncoderUnavailableError",
    "TrellisRenderError",
    "TrellisRenderRecipe",
    "TrellisWireSink",
    "TrellisWireSinkError",
    "build_trellis_pair_identity",
    "render_trellis_production_weight",
]
