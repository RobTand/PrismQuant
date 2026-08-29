"""AURA-currency dW supply for trellis rungs.  RESEARCH, OPT-IN, NOT WIRED.

WHAT PROBLEM THIS CLOSES, AND WHICH HALF OF IT
----------------------------------------------
``trellis_rate_surface`` prices its anchors in weighted SSE under a
per-input-channel activation second moment (:43-52) -- an output-MSE proxy.
The production DP ranks in the AURA KL-adjoint.  ``trellis_menu``'s
:data:`~prismaquant.trellis_menu.UNWIRED_LINKS` entry for
``trellis_rate_surface.py:43-52`` states the diagnosis exactly: because
``aura_cost``'s ``predicted_dloss`` is a pure inner product against ``dW``
(``aura_cost.py:1438`` -- ``0.5 * mean_k (<g_k, dW>)^2``), making a trellis
anchor AURA-priced is a **dW-supply** problem, not an objective change.  The
encoder's own weighted-SSE objective is a *render* choice, exactly as GPTQ's
activation-weighted local objective is; AURA prices whatever ``dW`` the render
produces.

This module supplies that ``dW`` **as data**.  It does not encode.  It is the
consumer half plus a reference producer for the store format; the producer
that actually runs the trellis encoder lives outside this repo and outside
this venv (see WHY A STORE below).  Nothing here can price a rung on its own,
and nothing here deletes an ``UNWIRED_LINKS`` entry: an entry goes when a test
exercises the behaviour it names, and no AURA-priced trellis anchor exists
until an offline encode has actually run on a target model.

WHY A STORE AND NOT AN IN-PROCESS RENDER
----------------------------------------
The trellis encoder is environment-pinned.  ``hull_sweep.STAGE6_ENV`` pins
torch ``2.13.0+cu130``, device ``NVIDIA GB10`` and one container image digest,
and ``hull_sweep.env_matches_stage6`` reports ``blocking_for_encode`` on a
torch mismatch with the rationale "A torch/triton skew flips DISCRETE encode
decisions; the reproduction gate is what detects it, and no tolerance can
absorb it."  The production venv
(``/home/rob/dq-runs/venvs/prismaquant-cu130``) is torch ``2.11.0+cu130``, so
an in-process render inside the AURA reverse pass is refused by the pin's own
reasoning, not by a policy invented here.

Only the *encode* is pinned.  A ``dW`` is portable.  So the honest shape is:
encode offline in the pinned container, persist the rendered weight together
with the identity of the source it was encoded against and the identity of the
encoder that produced it, and replay it into
``aura_cost.compute_aura_cost_streamed(anchor_renderer=...)`` -- which
duck-types its renderer and documents the injected case in so many words
("Compatibility for injected/research renderers", ``aura_cost.py:2521``).

WHAT MAKES THE REPLAY HONEST RATHER THAN A CONFOUND
---------------------------------------------------
``aura_cost`` sets ``require_source_weight_identity=anchor_renderer is not
None`` (:2246) because a ``dW`` measured against one source weight and
projected onto another model's gradients is meaningless.  This module makes
that binding load-bearing: every stored render carries the sha256 of the exact
source weight it was encoded from, computed with
``production_weight_cache._source_weight_value_identity`` -- the same function
AURA itself falls back to -- and :meth:`TrellisAnchorDeltaSource.render_layer`
**refuses** when the live module's weight does not hash to it.

It also refuses a rung that ``gridbook.trellis.wire.v1`` cannot carry.  The
wire's schedule scope is ``tensor_input_column_shared_across_rows``: one
4-bit rate code per input column, shared by every output row
(``trellis_footprint.py:474-478`` requires ``len(schedule) == columns``).  A
per-(row, superblock) mixed-rate plan -- the shape the out-of-repo one-anchor
allocator produces, ``bool [rows, n_sb]`` -- has a ROW axis the wire has no
field for and is not expressible at all.  Every stored record is therefore
priced through the repo's own ``trellis_tensor_payload_breakdown`` at
construction, so a store that cannot be a wire.v1 tensor cannot be opened.

WHAT THIS IS NOT
----------------
It declares no cost currency.  The store holds ``dW``; the currency is a
property of the AURA run that consumes it, and stamping one here would be a
second spelling of ``anchored_cost.AURA_CURRENCY``, which that module warns
against by name.  It states nothing about what any runtime executes
(principle 14): ``encoder_identity`` records what produced the bytes, never
what serves them.  It renders nothing, exports nothing, and
``export_native_compressed`` still refuses a TCQ assignment outright.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import torch

from .trellis_allocator import build_trellis_allocator_candidate
from .trellis_formats import (
    TrellisFormatError,
    get_trellis_family,
    parse_trellis_format_name,
)
from .trellis_rate_surface import uniform_column_schedule

#: Names a store directory written by an offline encode.  Unset is the
#: default; :func:`open_trellis_anchor_source` then returns ``None`` and no
#: caller acquires a trellis dW source.
TRELLIS_ANCHOR_DW_ENV = "PRISMAQUANT_TRELLIS_ANCHOR_DW"

TRELLIS_ANCHOR_DW_STORE_SCHEMA = "prismaquant.trellis_anchor_dw_store.v1"
_MANIFEST_NAME = "manifest.json"
_SHARD_DIR = "shards"

#: The encode environment must travel with the bytes.  These are exactly the
#: fields ``hull_sweep`` already computes for its own reproduction gate
#: (``snapshot_tree_sha256()``, ``STAGE6_ENV``); requiring them here means a
#: store cannot omit the one fact that decides whether two encodes are
#: comparable.  They are recorded, never interpreted: this module does not
#: decide which environment is correct, it refuses to lose the question.
REQUIRED_ENCODER_IDENTITY_FIELDS = (
    "encoder_snapshot_tree_sha256",
    "container_image",
    "torch_version",
    "device_name",
)


class TrellisAnchorStoreError(RuntimeError):
    """A trellis anchor dW store cannot be written, opened, or replayed."""


def _source_identity(weight: torch.Tensor) -> tuple[list[int], str]:
    """Shape + content sha256, via the function AURA itself uses.

    Imported lazily so this research module does not drag the 7k-line
    production cache into an import graph that has no other need for it.
    """

    from .production_weight_cache import _source_weight_value_identity

    return _source_weight_value_identity(weight)


def _assert_source_identity_coherent(
    records: Sequence[TrellisAnchorRecord],
) -> None:
    """Every rung of one qname must be encoded against the SAME weight.

    The multi-rung store is the normal case -- ``trellis_menu`` refuses a unit
    with fewer than two anchors -- so a store CAN hold two records for one
    qname that were encoded against different source weights.  That store is
    incoherent rather than merely wrong: ``source_weight_identity_for`` returns
    one identity per qname, so AURA would stamp one rung's provenance onto a
    pair whose dW came from two different weights, and the two rungs would not
    be comparable to each other at all.  Replay would catch at most one of the
    two.  Refuse the whole store instead, at write and at open.
    """

    by_qname: dict[str, TrellisAnchorRecord] = {}
    for record in records:
        first = by_qname.setdefault(record.qname, record)
        if (
            first.source_sha256 != record.source_sha256
            or tuple(first.source_shape) != tuple(record.source_shape)
        ):
            raise TrellisAnchorStoreError(
                f"{record.qname}: rung {first.format_name} was encoded "
                f"against source {tuple(first.source_shape)}/"
                f"{first.source_sha256[:12]} but rung {record.format_name} "
                f"against {tuple(record.source_shape)}/"
                f"{record.source_sha256[:12]}; the rungs of one unit must "
                "share one source weight or they cannot be compared"
            )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Content digest of the stored render, in its own storage dtype.

    Distinct from :func:`_source_identity`, which upcasts to fp32 because it
    must agree with AURA's source binding.  Here the point is that the bytes
    on disk are the bytes the encoder wrote, so the digest is taken over the
    stored dtype without conversion.

    Shape and dtype are framed into the digest rather than left implicit: a
    raw byte hash gives an ``[8, 256]`` render and its ``[256, 8]`` transpose
    the SAME value, so an unframed digest would call two different tensors
    identical.
    """

    contiguous = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes({
        "dtype": str(contiguous.dtype),
        "shape": [int(dim) for dim in contiguous.shape],
    }))
    digest.update(b"\0")
    digest.update(contiguous.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TrellisAnchorRecord:
    """One (unit, rung) render, with everything needed to refuse a mismatch."""

    qname: str
    format_name: str
    family: str
    body_rate_q256: int
    layout: str
    alphabets: Mapping[int, tuple[int, ...]]
    shape: tuple[int, int]
    rendered_dtype: str
    rendered_sha256: str
    source_shape: tuple[int, ...]
    source_sha256: str
    payload_bytes: int
    shard: str

    def to_dict(self) -> dict[str, object]:
        return {
            "qname": self.qname,
            "format": self.format_name,
            "family": self.family,
            "body_rate_q256": int(self.body_rate_q256),
            "layout": self.layout,
            "alphabets": {
                str(rate): [int(code) for code in codes]
                for rate, codes in sorted(self.alphabets.items())
            },
            "shape": [int(dim) for dim in self.shape],
            "rendered_dtype": self.rendered_dtype,
            "rendered_sha256": self.rendered_sha256,
            "source_shape": [int(dim) for dim in self.source_shape],
            "source_sha256": self.source_sha256,
            "payload_bytes": int(self.payload_bytes),
            "shard": self.shard,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrellisAnchorRecord":
        def _require(field: str) -> object:
            if field not in payload:
                raise TrellisAnchorStoreError(
                    f"trellis anchor record is missing {field!r}"
                )
            return payload[field]

        raw_shape = _require("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 2
            or any(type(dim) is not int or dim <= 0 for dim in raw_shape)
        ):
            raise TrellisAnchorStoreError(
                "trellis anchor record shape must be two positive integers"
            )
        raw_alphabets = _require("alphabets")
        if not isinstance(raw_alphabets, dict):
            raise TrellisAnchorStoreError(
                "trellis anchor record alphabets must be an object"
            )
        return cls(
            qname=str(_require("qname")),
            format_name=str(_require("format")),
            family=str(_require("family")),
            body_rate_q256=int(_require("body_rate_q256")),
            layout=str(_require("layout")),
            alphabets={
                int(rate): tuple(int(code) for code in codes)
                for rate, codes in raw_alphabets.items()
            },
            shape=(int(raw_shape[0]), int(raw_shape[1])),
            rendered_dtype=str(_require("rendered_dtype")),
            rendered_sha256=str(_require("rendered_sha256")),
            source_shape=tuple(
                int(dim) for dim in _require("source_shape")  # type: ignore[union-attr]
            ),
            source_sha256=str(_require("source_sha256")),
            payload_bytes=int(_require("payload_bytes")),
            shard=str(_require("shard")),
        )


def _validate_encoder_identity(
    encoder_identity: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(encoder_identity, Mapping) or not encoder_identity:
        raise TrellisAnchorStoreError(
            "a trellis anchor store requires a nonempty encoder identity; "
            "the encoder is environment-pinned and a dW whose encode "
            "environment is unrecorded cannot be compared with any other"
        )
    missing = [
        field for field in REQUIRED_ENCODER_IDENTITY_FIELDS
        if not str(encoder_identity.get(field, "")).strip()
    ]
    if missing:
        raise TrellisAnchorStoreError(
            "trellis anchor store encoder identity is missing required "
            f"fields {missing}; hull_sweep already computes every one of "
            "them (snapshot_tree_sha256, STAGE6_ENV)"
        )
    return {str(key): value for key, value in encoder_identity.items()}


def _price_record_on_the_wire(
    *,
    qname: str,
    format_name: str,
    family: str,
    body_rate_q256: int,
    layout: str,
    alphabets: Mapping[int, Sequence[int]],
    shape: tuple[int, int],
) -> int:
    """Refuse a rung ``gridbook.trellis.wire.v1`` cannot carry; return bytes.

    Priced through the repo's own ``build_trellis_allocator_candidate`` rather
    than a restatement of the wire law, so a store can never disagree with the
    allocator about what a rung costs.  The row-shared schedule is what is
    being enforced: :func:`uniform_column_schedule` produces exactly
    ``columns`` codes, and a plan with a per-row rate axis has no spelling
    here at all.
    """

    parsed = parse_trellis_format_name(format_name)
    if parsed is None:
        raise TrellisAnchorStoreError(
            f"{qname}: {format_name!r} is not a trellis rung name. This store "
            "supplies dW for the trellis rate surface; a scalar-menu format "
            "already has two dW sources in aura_cost and must use them"
        )
    parsed_family, parsed_rate = parsed
    spec = get_trellis_family(family)
    if parsed_family.family != spec.family or parsed_rate != int(
        body_rate_q256
    ):
        raise TrellisAnchorStoreError(
            f"{qname}: record says {spec.family}/R{body_rate_q256} but the "
            f"format name {format_name!r} spells "
            f"{parsed_family.family}/R{parsed_rate}"
        )
    rows, columns = int(shape[0]), int(shape[1])
    # Column divisibility is NOT re-stated here. ``uniform_column_schedule``
    # already owns that rule ("columns must be a multiple of 256; a short
    # final block is legal on the wire but its rate accounting is the
    # caller's to declare"), and a second spelling of one rule is how the
    # two drift. It reaches the caller through the wrap below.
    try:
        schedule = uniform_column_schedule(
            columns, int(body_rate_q256), family=spec,
        )
        candidate = build_trellis_allocator_candidate(
            qname,
            (rows, columns),
            family=spec,
            body_rate_q256=int(body_rate_q256),
            layout=layout,
            schedule=schedule,
            alphabets={
                int(rate): tuple(int(code) for code in codes)
                for rate, codes in alphabets.items()
            },
            # A store holds dW, not a cost. Zero is the neutral placeholder
            # for a field this module is deliberately not the source of; the
            # AURA run that consumes the dW supplies the real number.
            predicted_dloss=0.0,
            qname=qname,
        )
    except TrellisFormatError as exc:
        raise TrellisAnchorStoreError(
            f"{qname}: {format_name} is not expressible as a "
            f"gridbook.trellis.wire.v1 tensor of shape {(rows, columns)}: "
            f"{exc}"
        ) from exc
    return int(candidate.footprint["total_bytes"])


def write_trellis_anchor_store(
    root: str | Path,
    *,
    encoder_identity: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
) -> Path:
    """Reference producer for the store format.

    ``entries`` is a sequence of mappings with ``qname``, ``format``,
    ``family``, ``body_rate_q256``, ``layout``, ``alphabets``,
    ``rendered_weight`` and ``source_weight``.  The offline encode driver
    calls this from inside the pinned container; the signature deliberately
    takes already-rendered tensors so that this repo never has to host, pin,
    or reproduce the encoder itself.
    """

    identity = _validate_encoder_identity(encoder_identity)
    destination = Path(root)
    shard_dir = destination / _SHARD_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)

    records: list[TrellisAnchorRecord] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        qname = str(entry["qname"])
        format_name = str(entry["format"])
        key = (qname, format_name)
        if key in seen:
            raise TrellisAnchorStoreError(
                f"duplicate trellis anchor entry for {key}"
            )
        seen.add(key)
        rendered = entry["rendered_weight"]
        source = entry["source_weight"]
        if not isinstance(rendered, torch.Tensor) or not isinstance(
            source, torch.Tensor
        ):
            raise TrellisAnchorStoreError(
                f"{qname}: rendered_weight and source_weight must be tensors"
            )
        if tuple(rendered.shape) != tuple(source.shape):
            raise TrellisAnchorStoreError(
                f"{qname}: rendered shape {tuple(rendered.shape)} differs "
                f"from source shape {tuple(source.shape)}"
            )
        if rendered.ndim != 2:
            raise TrellisAnchorStoreError(
                f"{qname}: a trellis render is a 2-D Linear weight, got "
                f"{rendered.ndim} dimensions"
            )
        if not torch.isfinite(rendered.detach().float()).all():
            raise TrellisAnchorStoreError(
                f"{qname}: rendered weight is not finite"
            )
        shape = (int(rendered.shape[0]), int(rendered.shape[1]))
        payload_bytes = _price_record_on_the_wire(
            qname=qname,
            format_name=format_name,
            family=str(entry["family"]),
            body_rate_q256=int(entry["body_rate_q256"]),
            layout=str(entry["layout"]),
            alphabets=entry["alphabets"],  # type: ignore[arg-type]
            shape=shape,
        )
        source_shape, source_sha = _source_identity(source)
        rendered_cpu = rendered.detach().to(device="cpu").contiguous()
        rendered_sha = _tensor_sha256(rendered_cpu)
        shard = f"{_SHARD_DIR}/{rendered_sha}.pt"
        torch.save(rendered_cpu, destination / shard)
        records.append(TrellisAnchorRecord(
            qname=qname,
            format_name=format_name,
            family=str(entry["family"]),
            body_rate_q256=int(entry["body_rate_q256"]),
            layout=str(entry["layout"]),
            alphabets={
                int(rate): tuple(int(code) for code in codes)
                for rate, codes in entry["alphabets"].items()  # type: ignore[union-attr]
            },
            shape=shape,
            rendered_dtype=str(rendered_cpu.dtype),
            rendered_sha256=rendered_sha,
            source_shape=tuple(source_shape),
            source_sha256=source_sha,
            payload_bytes=payload_bytes,
            shard=shard,
        ))

    if not records:
        raise TrellisAnchorStoreError(
            "a trellis anchor store with no records cannot anchor anything"
        )
    _assert_source_identity_coherent(records)
    body = {
        "schema": TRELLIS_ANCHOR_DW_STORE_SCHEMA,
        "encoder_identity": identity,
        "records": [record.to_dict() for record in sorted(
            records, key=lambda item: (item.qname, item.format_name),
        )],
    }
    manifest = {**body, "identity_sha256": _sha256_json(body)}
    (destination / _MANIFEST_NAME).write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return destination / _MANIFEST_NAME


class TrellisAnchorDeltaSource:
    """Replay stored trellis renders as an ``aura_cost`` anchor renderer.

    Satisfies the duck-typed contract ``compute_aura_cost_streamed`` checks:
    ``formats_by_qname`` (compared against the AURA plan after
    ``fr.canonical_format_name``), ``identity``, ``render_layer``,
    ``render_count``, ``max_live_rendered`` and ``source_weight_identity_for``.
    It owns no residency mechanism and no cache; the caller installs a layer
    and hands over its live modules, exactly as ``StreamedProductionAnchorRenderer``
    is called.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        manifest_path = self._root / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise TrellisAnchorStoreError(
                f"no trellis anchor manifest at {manifest_path}"
            )
        payload = json.loads(manifest_path.read_text())
        if payload.get("schema") != TRELLIS_ANCHOR_DW_STORE_SCHEMA:
            raise TrellisAnchorStoreError(
                f"{manifest_path}: schema {payload.get('schema')!r} is not "
                f"{TRELLIS_ANCHOR_DW_STORE_SCHEMA}"
            )
        recorded_identity = payload.get("identity_sha256")
        body = {
            key: payload[key]
            for key in ("schema", "encoder_identity", "records")
            if key in payload
        }
        if recorded_identity != _sha256_json(body):
            raise TrellisAnchorStoreError(
                f"{manifest_path}: identity_sha256 does not match its own "
                "body; the manifest was edited after it was written"
            )
        self._encoder_identity = _validate_encoder_identity(
            payload["encoder_identity"]
        )
        self._identity_sha256 = str(recorded_identity)

        self._records: dict[tuple[str, str], TrellisAnchorRecord] = {}
        plan: dict[str, list[str]] = {}
        for raw in payload["records"]:
            record = TrellisAnchorRecord.from_dict(raw)
            # Re-price on open: a manifest whose payload_bytes disagrees with
            # what the allocator would charge is a store the DP cannot use.
            priced = _price_record_on_the_wire(
                qname=record.qname,
                format_name=record.format_name,
                family=record.family,
                body_rate_q256=record.body_rate_q256,
                layout=record.layout,
                alphabets=record.alphabets,
                shape=record.shape,
            )
            if priced != record.payload_bytes:
                raise TrellisAnchorStoreError(
                    f"{record.qname}/{record.format_name}: manifest charges "
                    f"{record.payload_bytes} bytes, the allocator prices "
                    f"{priced}"
                )
            key = (record.qname, record.format_name)
            if key in self._records:
                raise TrellisAnchorStoreError(
                    f"duplicate trellis anchor record for {key}"
                )
            self._records[key] = record
            plan.setdefault(record.qname, []).append(record.format_name)
        _assert_source_identity_coherent(list(self._records.values()))
        self._formats_by_qname = {
            qname: tuple(sorted(formats))
            for qname, formats in sorted(plan.items())
        }
        self._render_count = 0
        self._max_live_rendered = 0

    # -- the anchor-renderer duck type -------------------------------------

    @property
    def formats_by_qname(self) -> Mapping[str, tuple[str, ...]]:
        return dict(self._formats_by_qname)

    @property
    def identity(self) -> Mapping[str, object]:
        return {
            "schema": TRELLIS_ANCHOR_DW_STORE_SCHEMA,
            "store_identity_sha256": self._identity_sha256,
            "store_root": str(self._root),
            "encoder_identity": dict(self._encoder_identity),
            "records": len(self._records),
            # Stated so no reader has to infer it: this object replays bytes,
            # it does not render, and it makes no claim about what any
            # serving runtime does with them (principle 14).
            "render_source": "offline_trellis_encode_replay",
            "declares_cost_currency": False,
            "declares_runtime_contract": False,
        }

    @property
    def render_count(self) -> int:
        return self._render_count

    @property
    def max_live_rendered(self) -> int:
        return self._max_live_rendered

    def source_weight_identity_for(self, qname: str) -> Mapping[str, object]:
        for (name, _fmt), record in self._records.items():
            if name == str(qname):
                return {
                    "shape": [int(dim) for dim in record.source_shape],
                    "sha256": record.source_sha256,
                }
        raise TrellisAnchorStoreError(
            f"no trellis anchor record for {qname!r}"
        )

    def render_layer(
        self,
        *,
        layer: int,
        modules: Mapping[str, object],
        formats_by_qname: Mapping[str, Sequence[str]],
    ) -> dict[tuple[str, str], torch.Tensor]:
        """Return the stored render for every requested pair, or refuse.

        Every refusal below is a confound this replay could otherwise
        introduce silently, which is why none of them is a warning.
        """

        del layer  # residency and ordering belong to the caller
        out: dict[tuple[str, str], torch.Tensor] = {}
        for raw_name, raw_formats in formats_by_qname.items():
            name = str(raw_name)
            planned = self._formats_by_qname.get(name)
            if planned is None:
                raise TrellisAnchorStoreError(
                    f"unplanned trellis anchor render requested for {name}"
                )
            requested = tuple(str(fmt) for fmt in raw_formats)
            unplanned = sorted(set(requested) - set(planned))
            if unplanned:
                raise TrellisAnchorStoreError(
                    f"{name}: trellis anchor store holds {list(planned)}, "
                    f"the run asked for {unplanned}"
                )
            module = modules.get(name)
            if module is None:
                raise TrellisAnchorStoreError(
                    f"trellis anchor replay was not given a live module for "
                    f"{name}"
                )
            weight = getattr(module, "weight", None)
            if weight is None:
                raise TrellisAnchorStoreError(
                    f"{name}: live module exposes no weight to bind against"
                )
            live_shape, live_sha = _source_identity(weight.data)
            for fmt in requested:
                record = self._records[(name, fmt)]
                if tuple(live_shape) != tuple(record.source_shape):
                    raise TrellisAnchorStoreError(
                        f"{name}/{fmt}: live weight shape {live_shape} "
                        f"differs from the encode source {list(record.source_shape)}"
                    )
                if live_sha != record.source_sha256:
                    raise TrellisAnchorStoreError(
                        f"{name}/{fmt}: this dW was encoded against source "
                        f"weight {record.source_sha256[:16]} but the live "
                        f"model holds {live_sha[:16]}. Projecting it onto "
                        "this model's gradients would price a render that "
                        "was never of these weights"
                    )
                out[(name, fmt)] = self._load_shard(record)
        self._render_count += len(out)
        self._max_live_rendered = max(self._max_live_rendered, len(out))
        return out

    # -- internals ---------------------------------------------------------

    def _load_shard(self, record: TrellisAnchorRecord) -> torch.Tensor:
        path = self._root / record.shard
        if not path.is_file():
            raise TrellisAnchorStoreError(
                f"{record.qname}/{record.format_name}: missing shard {path}"
            )
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise TrellisAnchorStoreError(
                f"{path}: shard is not a tensor"
            )
        if tuple(tensor.shape) != record.shape:
            raise TrellisAnchorStoreError(
                f"{record.qname}/{record.format_name}: shard shape "
                f"{tuple(tensor.shape)} differs from the manifest "
                f"{record.shape}"
            )
        if str(tensor.dtype) != record.rendered_dtype:
            raise TrellisAnchorStoreError(
                f"{record.qname}/{record.format_name}: shard dtype "
                f"{tensor.dtype} differs from the manifest "
                f"{record.rendered_dtype}"
            )
        digest = _tensor_sha256(tensor)
        if digest != record.rendered_sha256:
            raise TrellisAnchorStoreError(
                f"{record.qname}/{record.format_name}: shard content "
                f"{digest[:16]} differs from the manifest "
                f"{record.rendered_sha256[:16]}"
            )
        return tensor


def open_trellis_anchor_source(
    root: str | Path | None = None,
) -> TrellisAnchorDeltaSource | None:
    """Open the store named by ``root`` or the env flag; ``None`` when unset.

    Unset is the default and is the whole of this module's default-path
    behaviour: no caller in the shipping pipeline calls it, and a caller that
    does gets ``None`` and proceeds exactly as it did before.
    """

    resolved = root if root is not None else os.environ.get(
        TRELLIS_ANCHOR_DW_ENV
    )
    if not resolved:
        return None
    return TrellisAnchorDeltaSource(resolved)


__all__ = [
    "REQUIRED_ENCODER_IDENTITY_FIELDS",
    "TRELLIS_ANCHOR_DW_ENV",
    "TRELLIS_ANCHOR_DW_STORE_SCHEMA",
    "TrellisAnchorDeltaSource",
    "TrellisAnchorRecord",
    "TrellisAnchorStoreError",
    "open_trellis_anchor_source",
    "write_trellis_anchor_store",
]
