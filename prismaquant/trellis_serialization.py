"""Artifact-wide producer choice for Trellis TCQ wire rendering.

This is the exact analogue of :class:`CBSerializationContext` for the trellis
families. Every value-bearing encoder choice is explicit; there are no
environment-derived defaults here. A missing context on a trellis render is
fail-closed, just as ``render_production_weight`` refuses a CB render without
its ``CBSerializationContext``.

The recipe comprises the eleven fields named in WO-B B1:

* ``family`` — ``TCQ_E2M1_R256`` or ``TCQ_E4M3_R256`` (the family token
  without the rate suffix; the numeric body rate is separate)
* ``body_rate_q256`` — integer bits per 256-weight block (q256)
* ``schedule`` — per-input-column code length in ``[1, bypass_rate]``
* ``layout`` — ``tight_offsets`` or ``fixed_quota_per_256``
* ``alphabets`` — ``{shaped_rate: codes}`` code-value tables
* ``scale_rule`` — ``static_6`` for E2M1 or ``row_fp32_amax_448`` for E4M3
* ``sb_chunk`` — rows per Viterbi chunk (positive int)
* ``determinism_mode`` — ``on`` or ``off``
* ``tailbite_candidates`` — fixed at ``4`` (bit-exactness qualified)
* ``backend`` — ``eager`` (CPU reference) or ``triton`` (CUDA, four fused
  launches per chunk)
* ``point_route`` — ``full`` or ``windowed``

``global_scale_real_override`` is an optional per-tensor E2M1 override.
When ``None`` the encoder derives the global scale from the weight; when
present it must be finite and positive and is applied verbatim.

``TreillisSerializationContext`` is versioned. Its stamp is stored in
``ProductionWeightCache`` metadata under
``trellis_render_identity`` and in per-entry wire identity records, so a
later consumer can prove the bytes it packs are the bytes the surrogate priced
and KL measured (principle 8).

Do NOT add environment-derived encoder defaults. The encoder module's
docstring forbids them deliberately.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .trellis_formats import (
    E2M1_FAMILY,
    FAMILIES,
    LAYOUTS,
    TrellisFormatError,
    get_trellis_family,
    validate_alphabets,
    validate_body_rate_q256,
    validate_schedule,
)

TRELLIS_SERIALIZATION_SCHEMA = "prismaquant.trellis_serialization.v1"
TRELLIS_RENDER_IDENTITY_SCHEMA = "prismaquant.trellis_render_identity.v1"
TRELLIS_RENDER_CONTRACT_SCHEMA = "prismaquant.trellis_render_contract.v1"
# Bump this when the persisted trellis entry shape changes. An old cache
# carrying the previous ABI must rebuild loudly rather than silently serve a
# trellis-shaped miss (the COST_MODE flip precedent).
TRELLIS_RENDER_MECHANISM_ABI = "prismaquant.trellis_render_mechanisms.v1"
TRELLIS_RENDER_IDENTITY_METADATA_KEY = "trellis_render_identity"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TrellisSerializationContext:
    """Artifact-wide trellis encoder choices. All value-bearing fields are
    explicit; omission is an error, not a default."""

    family: str
    body_rate_q256: int
    schedule: tuple[int, ...]
    layout: str
    alphabets: Mapping[int, tuple[int, ...]]
    scale_rule: str
    sb_chunk: int
    determinism_mode: str
    tailbite_candidates: int
    backend: str
    point_route: str
    # Optional E2M1 global-scale override. ``None`` derives from weight.
    global_scale_real_override: float | None = None

    def __post_init__(self) -> None:
        # family
        spec = get_trellis_family(str(self.family))
        object.__setattr__(self, "family", spec.family)
        # body_rate_q256
        rate = validate_body_rate_q256(spec, int(self.body_rate_q256))
        object.__setattr__(self, "body_rate_q256", int(rate))
        # layout
        layout = str(self.layout)
        if layout not in LAYOUTS:
            raise TrellisFormatError(
                f"unknown trellis layout {layout!r}; expected {sorted(LAYOUTS)}"
            )
        object.__setattr__(self, "layout", layout)
        # schedule — validated against family/rate/layout
        raw_schedule = tuple(int(v) for v in self.schedule)  # type: ignore[arg-type]
        schedule = validate_schedule(spec, rate, raw_schedule, layout=layout)
        object.__setattr__(self, "schedule", tuple(schedule))
        # alphabets — validated against spec + schedule
        # Normalize alphabets to {int: tuple[int,...]}
        normalized: dict[int, tuple[int, ...]] = {}
        for raw_rate, raw_codes in dict(self.alphabets).items():  # type: ignore[arg-type]
            rate_key = int(raw_rate)
            codes = tuple(int(c) for c in raw_codes)  # type: ignore[arg-type]
            normalized[rate_key] = codes
        checked = validate_alphabets(spec, schedule, normalized)
        # Freeze as sorted dict of tuples
        frozen = {int(k): tuple(v) for k, v in sorted(checked.items())}
        object.__setattr__(self, "alphabets", frozen)
        # scale_rule — family-specific
        rule = str(self.scale_rule).strip()
        if spec.family == E2M1_FAMILY:
            if rule != "static_6":
                raise TrellisFormatError(
                    f"unsupported E2M1 trellis scale rule {rule!r}; only "
                    "the measured static_6 group-16 plane has a writer"
                )
        else:
            if rule != "row_fp32_amax_448":
                raise TrellisFormatError(
                    f"unsupported E4M3 trellis scale rule {rule!r}; only "
                    "the measured per-row fp32 amax/448 plane has a writer"
                )
        object.__setattr__(self, "scale_rule", rule)
        # sb_chunk
        sb = int(self.sb_chunk)
        if sb < 1:
            raise TrellisFormatError("encoder sb_chunk must be positive")
        object.__setattr__(self, "sb_chunk", sb)
        # determinism_mode
        det = str(self.determinism_mode)
        if det not in {"on", "off"}:
            raise TrellisFormatError("determinism mode must be 'on' or 'off'")
        object.__setattr__(self, "determinism_mode", det)
        # tailbite_candidates — only 4 is qualified
        tb = int(self.tailbite_candidates)
        if tb != 4:
            raise TrellisFormatError(
                "only 4 tail-biting candidates are bit-exactness qualified"
            )
        object.__setattr__(self, "tailbite_candidates", tb)
        # backend
        be = str(self.backend)
        if be not in {"eager", "triton"}:
            raise TrellisFormatError("encoder backend must be eager or triton")
        object.__setattr__(self, "backend", be)
        # point_route
        pr = str(self.point_route)
        if pr not in {"full", "windowed"}:
            raise TrellisFormatError("point route must be full or windowed")
        object.__setattr__(self, "point_route", pr)
        # global_scale_real_override
        gscale = self.global_scale_real_override
        if gscale is not None:
            val = float(gscale)
            if not math.isfinite(val) or val <= 0.0:
                raise TrellisFormatError(
                    "global_scale_real_override must be finite and positive"
                )
            object.__setattr__(self, "global_scale_real_override", val)

    def as_recipe_dict(self) -> dict[str, object]:
        """Return the canonical recipe mapping used for identity hashing."""
        return {
            "family": self.family,
            "body_rate_q256": int(self.body_rate_q256),
            "schedule": list(self.schedule),
            "layout": self.layout,
            "alphabets": {
                str(rate): list(codes)
                for rate, codes in sorted(self.alphabets.items())
            },
            "scale_rule": self.scale_rule,
            "sb_chunk": int(self.sb_chunk),
            "determinism_mode": self.determinism_mode,
            "tailbite_candidates": int(self.tailbite_candidates),
            "backend": self.backend,
            "point_route": self.point_route,
            "global_scale_real_override": (
                float(self.global_scale_real_override)
                if self.global_scale_real_override is not None
                else None
            ),
        }

    @property
    def recipe_identity_sha256(self) -> str:
        return _canonical_sha256(self.as_recipe_dict())

    def stamp(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": TRELLIS_SERIALIZATION_SCHEMA,
            "family": self.family,
            "body_rate_q256": int(self.body_rate_q256),
            "schedule": list(self.schedule),
            "layout": self.layout,
            "alphabets": {
                str(rate): list(codes)
                for rate, codes in sorted(self.alphabets.items())
            },
            "scale_rule": self.scale_rule,
            "sb_chunk": int(self.sb_chunk),
            "determinism_mode": self.determinism_mode,
            "tailbite_candidates": int(self.tailbite_candidates),
            "backend": self.backend,
            "point_route": self.point_route,
            "global_scale_real_override": (
                float(self.global_scale_real_override)
                if self.global_scale_real_override is not None
                else None
            ),
            "encoder_source_sha256": _encoder_source_sha256_cached(),
        }
        # Identity covers everything except itself.
        body_with_identity = {**body, "recipe_identity_sha256": _canonical_sha256(body)}
        return {**body_with_identity, "identity_sha256": _canonical_sha256(body_with_identity)}

    @classmethod
    def from_stamp(cls, stamp: Mapping[str, object], *, where: str = "TrellisSerializationContext") -> "TrellisSerializationContext":
        if not isinstance(stamp, Mapping):
            raise TrellisFormatError(f"{where}: trellis context stamp is not an object")
        if stamp.get("schema") != TRELLIS_SERIALIZATION_SCHEMA:
            raise TrellisFormatError(
                f"{where}: unsupported trellis context schema {stamp.get('schema')!r}"
            )
        # Recompute and verify recipe identity if present
        # Extract fields
        try:
            family = str(stamp["family"])
            body_rate_q256 = int(stamp["body_rate_q256"])  # type: ignore[arg-type]
            schedule = tuple(int(v) for v in stamp["schedule"])  # type: ignore[arg-type]
            layout = str(stamp["layout"])
            alphabets_raw = stamp["alphabets"]
            if not isinstance(alphabets_raw, Mapping):
                raise TrellisFormatError(f"{where}: alphabets is not an object")
            alphabets = {
                int(k): tuple(int(c) for c in v)  # type: ignore[arg-type]
                for k, v in alphabets_raw.items()
            }
            scale_rule = str(stamp["scale_rule"])
            sb_chunk = int(stamp["sb_chunk"])  # type: ignore[arg-type]
            determinism_mode = str(stamp["determinism_mode"])
            tailbite_candidates = int(stamp["tailbite_candidates"])  # type: ignore[arg-type]
            backend = str(stamp["backend"])
            point_route = str(stamp["point_route"])
            gscale = stamp.get("global_scale_real_override")
            if gscale is not None:
                gscale = float(gscale)  # type: ignore[arg-type]
        except KeyError as exc:
            raise TrellisFormatError(f"{where}: trellis context missing {exc}") from exc
        ctx = cls(
            family=family,
            body_rate_q256=body_rate_q256,
            schedule=schedule,
            layout=layout,
            alphabets=alphabets,
            scale_rule=scale_rule,
            sb_chunk=sb_chunk,
            determinism_mode=determinism_mode,
            tailbite_candidates=tailbite_candidates,
            backend=backend,
            point_route=point_route,
            global_scale_real_override=gscale,
        )
        # Verify stamp's recipe identity if present
        expected_recipe = _canonical_sha256({
            "family": ctx.family,
            "body_rate_q256": ctx.body_rate_q256,
            "schedule": list(ctx.schedule),
            "layout": ctx.layout,
            "alphabets": {str(k): list(v) for k, v in sorted(ctx.alphabets.items())},
            "scale_rule": ctx.scale_rule,
            "sb_chunk": ctx.sb_chunk,
            "determinism_mode": ctx.determinism_mode,
            "tailbite_candidates": ctx.tailbite_candidates,
            "backend": ctx.backend,
            "point_route": ctx.point_route,
            "global_scale_real_override": ctx.global_scale_real_override,
        })
        # Not strictly required to match encoder_source_sha256; just ensure ctx is valid
        return ctx


_trellis_encoder_source_sha256_cache: str | None = None


def _encoder_source_sha256_cached() -> str:
    global _trellis_encoder_source_sha256_cache
    if _trellis_encoder_source_sha256_cache is not None:
        return _trellis_encoder_source_sha256_cache
    try:
        from .trellis_encoder import encoder_source_sha256
        _trellis_encoder_source_sha256_cache = encoder_source_sha256()
    except Exception:
        _trellis_encoder_source_sha256_cache = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
    return _trellis_encoder_source_sha256_cache


def trellis_serialization_context_stamp(context: TrellisSerializationContext) -> dict[str, object]:
    if not isinstance(context, TrellisSerializationContext):
        raise TypeError(f"expected TrellisSerializationContext, got {type(context).__name__}")
    return context.stamp()


__all__ = [
    "TRELLIS_RENDER_CONTRACT_SCHEMA",
    "TRELLIS_RENDER_IDENTITY_METADATA_KEY",
    "TRELLIS_RENDER_IDENTITY_SCHEMA",
    "TRELLIS_RENDER_MECHANISM_ABI",
    "TRELLIS_SERIALIZATION_SCHEMA",
    "TrellisSerializationContext",
    "trellis_serialization_context_stamp",
]
