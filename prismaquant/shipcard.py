"""The ship record (`exported/shipcard.json`) — a refusal contract.

R13 (`docs/audits/architecture_re-vet_2026-07-30.md`). The build lane and the
serve lane are separated by a physical boundary: `vllm` is not importable in the
build venv, so `run-pipeline.sh` cannot run a ship gate and never should. What it
*can* do is **open a record with required, empty slots** that only the serve lane
can close. `python -m prismaquant.shipcard_cli verify` then exits non-zero until every slot holds a
record whose `model_sha` matches the artifact on disk — which turns "we never ran
the gate" from a silent omission into an explicit refusal.

Lane-scoped slots (:data:`LANE_SCOPED_SLOTS`) are opened by the lane's own
``lane_specs/<lane>.json`` ``gates[]`` and are required of cards stamped with
that ``lane``.

Base slots (required for every artifact):

| Slot | Filled by |
|---|---|
| `native_export.eager` | `validate_native_export.py --shipcard` (eager arm) |
| `native_export.graph` | `validate_native_export.py --shipcard --no-enforce-eager` |
| `ship_gate` | `validate_quantized_model.py --shipcard` |
| `gold.kl` | `python -m prismaquant.shipcard_cli fill --slot gold.kl --record <full_kl json>` |
| `gold.ppl` | `python -m prismaquant.shipcard_cli fill --slot gold.ppl --record <ppl json>` |

One lane-scoped slot, required only where it is definable — a **rate-axis**
artifact (today: the Tessera container), whose allocation is a claim that
choosing a rung per Linear beats spending the same bytes everywhere:

| Slot | Filled by |
|---|---|
| `uniform_control` | `python -m prismaquant.shipcard_cli fill-control` |

That claim was measured false on 2026-09-02 (2.00x worse served KL than the
byte-matched uniform arm at 4.0 bpp) while every other check passed, so it is
a refusal here rather than a note. `shipcard_cli override-control` is the
deliberate, basename-confirmed, stamped escape hatch.

Until 2026-09-02 the retired Gridbook codebook lane opened three further
slots of its own (``perf.matched_budget_parity``, ``rtx4090.fp8_cb`` and the
DSv4 Gridbook gold contract).  They went into
``archive/gridbook_lane_2026-09-02/`` with the lane: a slot no live lane can
open is a gate that only teaches operators to reach for
``--force-unverified``.  A previously shipped Gridbook artifact therefore no
longer re-verifies here; its card and the verifiers that read it are in the
archive.

The two `gold.*` slots additionally require `spec_decode_detected: false` on the
record that produced the number — vLLM routes echo+logprobs through the draft
model under `--speculative-config`, so a spec-decode-on gold number is the MTP
head's NLL, not the artifact's (§7.5).

Stdlib only, no torch: the CLI must run anywhere the artifact is reachable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import re
import statistics
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "prismaquant.shipcard/1"

#: Every slot all serving lanes must close before an artifact is shippable.
#: Keep this base tuple stable: no container inherits a plugin-only gate
#: merely because one lane adds one.
REQUIRED_SLOTS: tuple[str, ...] = (
    "native_export.eager",
    "native_export.graph",
    "ship_gate",
    "gold.kl",
    "gold.ppl",
)

#: The byte-matched uniform control's verdict (RobTand/prismaquant#121).
UNIFORM_CONTROL_SLOT = "uniform_control"

#: Claims that can be attached to an already exported artifact.  Missing/null
#: claims remain non-blocking for target-only artifacts, but every non-null
#: recognized claim is verified automatically.  The only member until
#: 2026-09-02 was ``mtp.dspark``, written by the retired Gridbook lane's DSpark
#: sidecar validator; it went into the archive with that lane.
#:
#: ``uniform_control`` is optional only in the sense that a format-menu
#: artifact has no rate axis for it to be about; on a rate-axis artifact
#: :func:`required_slots` makes it mandatory and its absence is UNFILLED.
OPTIONAL_SLOTS: tuple[str, ...] = (UNIFORM_CONTROL_SLOT,)

#: Slots a LANE opens beyond the base set, because its own declaration names a
#: gate the base set has nowhere to put.  Precedent: the retired Gridbook
#: codebook lane opened three of these off the artifact's own config.
#:
#: ``route.census`` is principle 12's second leg -- the per-module route
#: histogram the serve actually emitted, against the histogram the artifact was
#: PRICED on.  The Tessera lane declared that gate from the day the lane
#: existed and gave it ``shipcard_slot: null``, so the check was named and
#: recorded nowhere; a gate nothing can record is a gate nothing can refuse on
#: (RobTand/prismaquant#119).
#:
#: WHICH lanes open which of these is read from ``lane_specs/<lane>.json``'s
#: own ``gates[]`` and from no list here, so a lane opens one of these slots by
#: declaring the gate that closes it.  The MEMBERSHIP of this tuple is still
#: enumerated, because it is also the key space :func:`verify` dispatches its
#: per-slot evidence replay on: a fourth lane declaring a slot that is not
#: here is REFUSED by :func:`lane_gate_slots` rather than silently dropped, and
#: admitting it is one edit here plus the verifier that replays it
#: (RobTand/prismaquant#162).
LANE_SCOPED_SLOTS: tuple[str, ...] = ("route.census",)

#: The vocabulary accepted by :func:`make_record`.  Whether a member is
#: required is artifact-specific and is resolved by :func:`required_slots`.
ALL_SLOTS: tuple[str, ...] = REQUIRED_SLOTS + OPTIONAL_SLOTS + LANE_SCOPED_SLOTS

#: Slots whose number is invalid if it was produced against a spec-decode serve.
GOLD_SLOTS: frozenset[str] = frozenset({"gold.kl", "gold.ppl"})

SHIPCARD_FILENAME = "shipcard.json"


SHIPCARD_RESERVED_BYTES = 256 * 1024
WEIGHT_CONTENT_MANIFEST_SCHEMA = "prismaquant.weight_content_manifest/1"
WEIGHT_STAT_ATTESTATION_SCHEMA = "prismaquant.weight_stat_attestation/1"
# Orphaned 2026-09-02, kept deliberately. `SAFETENSORS_CONTENT_RECEIPT_SCHEMA`,
# `validate_safetensors_content_receipt` and
# `safetensors_content_receipt_manifest` have no live caller since the strict
# RTX4090 FP8-CB publication gate -- their only consumer -- retired with the
# Gridbook lane (archive/gridbook_lane_2026-09-02/). They are a lane-neutral
# mechanism (span-hash a safetensors file's *content*, not just its header) and
# the schema string is stamped into receipts that exist on disk, so deleting
# them would strand those receipts unreadable rather than merely unused.
# Recorded as debt D34.
SAFETENSORS_CONTENT_RECEIPT_SCHEMA = (
    "prismaquant.safetensors_content_receipt/1"
)
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_SAFETENSORS_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "F4": 4,
    "F4_E2M1": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
}
SERVED_ARTIFACT_BINDING_SCHEMA = "prismaquant.served_artifact_binding/1"
SERVE_MANIFEST_SCHEMA = "prismaquant.serve_manifest/1"
# The name `scripts/lib/serve_manifest.sh` writes beside the artifact.
SERVE_MANIFEST_FILENAME = "serve_manifest.json"

# Card figures: rendered FROM the artifact's own attested metadata (the
# allocation map is drawn from quant_config.json), referenced only by the
# model card, decoded by no runtime. They share README.md's exclusion from
# `compute_model_sha` for the same three reasons documented there. Exact
# filenames, not a category -- anything else in the dir stays attested.
CARD_FIGURE_FILENAMES = ("allocation-map.png", "byte-budget.png")
FULL_KL_TEACHER_EVIDENCE_SCHEMA = "prismaquant.full_kl_teacher_evidence/1"
WIKITEXT_GOLD_CALIBRATION_SCHEMA = "prismaquant.wikitext_gold_calibration/1"
WIKITEXT_PPL_CALIBRATION_SCHEMA = "prismaquant.wikitext_ppl_calibration/1"
GOLD_PRODUCER_IDENTITY_SCHEMA = "prismaquant.gold_producer_identity/1"
TOPK_COVERAGE_POLICY_SCHEMA = "prismaquant.topk_tail_coverage_policy/1"

#: The block ``tessera.control.control_block()`` emits, carried verbatim.  The
#: shipcard never imports Tessera (this module is stdlib-only by contract), so
#: the schema string is the whole of what is taken on trust: every number in
#: the block is replayed here from integers.
UNIFORM_CONTROL_SCHEMA = "tessera.uniform_control.v1"
#: The override that lets a measured loss ship anyway.
UNIFORM_CONTROL_OVERRIDE_SCHEMA = "prismaquant.uniform_control_override/1"
#: The widest a control's bytes may miss the candidate's and still be a
#: control, as a :class:`~fractions.Fraction` of the candidate's own bits.
#: This is ``tessera.control.DEFAULT_MAX_RELATIVE_SLACK``, restated here
#: because a block that widened its own tolerance would otherwise certify
#: itself: the carried ``max_relative_slack`` is checked against this ceiling
#: before it is used.  Zero slack is unreachable -- the rate axis is discrete,
#: one rung quantum is ~0.0039 bpp (~0.1% at 4 bpp) -- so "exact" here means
#: exact integer arithmetic against an explicit ceiling, not zero slack.
MAX_CONTROL_RELATIVE_SLACK = Fraction(1, 1000)
#: Measurement-contract keys the two arms must agree on when the candidate's
#: own ``gold.kl`` record carries them.  Driven by the candidate rather than
#: by a fixed list so the control cannot dodge a key by omitting it: the
#: candidate side is the card's published gold number and is gated separately.
UNIFORM_CONTROL_CONTRACT_KEYS = (
    "n_samples",
    "seqlen",
    "n_positions",
    "score_positions",
    "corpus_sha256",
)
#: Which gold metric the two arms are compared on.  Both must quote the same
#: one, and it must be a KL: the gate's whole point is the serving metric.
UNIFORM_CONTROL_METRIC_KEYS = ("kl_mean", "kl_confident_mean")
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DSV4_WIKITEXT_DATASET_FINGERPRINT = "7ccd6deaa4fc56e5"
DSV4_WIKITEXT_CORPUS_SHA256 = (
    "c5b5caea5bd655cb221545a484f2f0f59d35092a17a66840d7b9513d0b99687d"
)
DSV4_WIKITEXT_TOTAL_TOKENS = 287_597
DSV4_WIKITEXT_SELECTED_TOKEN_IDS_SHA256 = (
    "6c23cefbd78c327d6edac566a5c6b419871021b6cf9890ec830713c1de704961"
)
DSV4_TOKENIZER_IDENTITY_SHA256 = (
    "9f7ee7cb93b58bf30f278965547e7584b89c848e76c3adfeb92c070a88492de0"
)
_CB_FORMAT_RE = re.compile(r"^(?:NVFP4_CB|FP8_CB)_[KS][0-9]+$")
_FP8_SOURCE_W8A16_WIRE_IDS = frozenset({
    "fp8_e4m3_ue8m0_block128",
})
_MXFP8_DENSE_WIRE_IDS = frozenset({"mxfp8_e4m3_e8m0_g32"})


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def compute_model_sha(
    model_dir: str | os.PathLike,
    *,
    legacy_native_scope: bool = False,
    legacy_readme_hashed: bool = False,
    legacy_figures_hashed: bool = False,
) -> str:
    """Cheap, stable identity for an exported checkpoint.

    CB artifacts bind canonical ``quant_config.json`` (excluding only its
    self-referential inventory), the exporter-produced exact SHA-256 manifest
    of every large safetensors container, every ``.pqcb`` sidecar content hash,
    and the content of every remaining file. Routine verification validates the
    manifest shape/sizes and uses the shipcard's stat attestation instead of
    rereading ~100 GB.

    Native artifacts once bound only ``config.json`` plus per-container SIZES,
    so the auxiliary files were unattested: swapping ``chat_template.jinja`` or
    ``tokenizer.json`` left the sha bit-identical. That is not cosmetic on a
    tool-calling model -- the chat template decides where a tool call is
    emitted, and a served artifact with the wrong one is broken in a way no
    weight check sees. Demonstrated 2026-08-15 on the published Qwen3.8-27B
    native artifact: tampering with its chat template moved nothing.

    Both lanes now hash auxiliary content. ``legacy_native_scope=True``
    reproduces the pre-fix native identity so cards written under it still
    verify -- see :func:`verify`, which tries the current scope first and
    falls back. New cards are always written with the current scope, and the
    legacy scope cannot produce one, so the fallback never weakens a new card.
    ``legacy_readme_hashed=True`` is the same tolerance for the one other
    scope change: cards stamped while ``README.md`` was still hashed.
    ``legacy_figures_hashed=True`` likewise reproduces identities stamped
    before the card figures (``CARD_FIGURE_FILENAMES``) joined the exclusion.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"model dir does not exist: {root}")
    payload: dict[str, Any] = {}
    cfg = root / "config.json"
    if cfg.is_file():
        payload["config_sha"] = hashlib.sha256(cfg.read_bytes()).hexdigest()
    quant_cfg = root / "quant_config.json"
    raw_quant_cfg: dict[str, Any] | None = None
    canonical_quant_cfg: dict[str, Any] | None = None
    if quant_cfg.is_file():
        raw_quant_cfg = json.loads(quant_cfg.read_text())
        if not isinstance(raw_quant_cfg, dict):
            raise ValueError(
                f"CB quant config must be a JSON object: {quant_cfg}"
            )
        canonical_quant_cfg = dict(raw_quant_cfg)
        provenance = raw_quant_cfg.get("provenance")
        if isinstance(provenance, dict):
            canonical_provenance = dict(provenance)
            canonical_provenance.pop("artifact_inventory", None)
            canonical_quant_cfg["provenance"] = canonical_provenance
    weights = {
        p.name: p.stat().st_size
        for p in sorted(root.glob("*.safetensors"))
    }
    if not weights:
        weights = {
            p.name: p.stat().st_size
            for p in sorted(root.glob("*.gguf"))
        }
    payload["weights"] = weights
    if raw_quant_cfg is not None and canonical_quant_cfg is not None:
        manifest = (raw_quant_cfg.get("provenance") or {}).get(
            "weight_content_manifest"
        ) if isinstance(raw_quant_cfg.get("provenance"), dict) else None
        if manifest is not None:
            _validate_weight_content_manifest(manifest, weights, where=quant_cfg)
        payload["quant_config_sha"] = hashlib.sha256(
            _canonical_json(canonical_quant_cfg).encode("utf-8")
        ).hexdigest()
    codebooks = {
        p.name: {
            "bytes": p.stat().st_size,
            "sha256": _file_content_sha256(p),
        }
        for p in sorted(root.glob("*.pqcb"))
    }
    if codebooks:
        payload["codebooks"] = codebooks
    if raw_quant_cfg is not None or not legacy_native_scope:
        # `serve_manifest.json` is not artifact content: it is the R15 serve
        # fingerprint, written INTO the model dir by `scripts/lib/
        # serve_manifest.sh` after a server comes up, and it records the
        # serving stack (image, argv, loaded .so set, hostname, boot id,
        # timestamp).  Hashing it would make the act of VALIDATING an artifact
        # invalidate its own card, and would make the identity differ between
        # two serves of byte-identical weights.  Nothing is unbound by the
        # exclusion: every slot that cites a manifest binds it by its own
        # `*serve_manifest_sha256`, which is where its integrity belongs.
        #
        # `README.md` is not artifact content either.  On the Hub it IS the
        # model card, and `tools/publish_artifact.py` has no --model-card
        # argument -- it uploads the complete local file set with no filters --
        # so the card reaches the Hub only by sitting in the model dir under
        # that name.  Hashing it made the act of DOCUMENTING an artifact
        # invalidate its own card: the same failure as the serve fingerprint,
        # one step earlier in the release.  Observed 2026-08-15 on
        # qwen38-27b-arm-b/exported, where a README dropped in at 18:33 moved
        # the identity off the 17:55 card (e7ac09f8 -> 3c4a83a1) and locked the
        # artifact out of publication.
        #
        # It also made an artifact unable to quote its own measured numbers:
        # every gate record binds `model_sha`, the gold KL/PPL numbers only
        # exist after the gates, and writing them into the card would
        # invalidate the records that produced them.  Re-running the gates does
        # not escape it -- KL drifts across docker sessions, so the second
        # round would ship records disagreeing with the printed numbers.
        #
        # Nothing behavioural goes unbound: a README is decoded by no runtime
        # and cannot change what the model computes, and upload integrity comes
        # from `publish_artifact.py` freezing the complete local file set, not
        # from `model_sha`.  Exact filenames, not a category.  The card
        # figures (2026-08-18) earn the same exclusion by the same argument:
        # they are rendered FROM the attested quant_config after the gates by
        # construction -- the allocation being drawn only exists once the
        # export finalized -- so hashing them made ILLUSTRATING an artifact
        # invalidate the records that measured it, exactly the README failure.
        excluded = {SHIPCARD_FILENAME, "quant_config.json",
                    SERVE_MANIFEST_FILENAME}
        if not legacy_readme_hashed:
            excluded.add("README.md")
        if not legacy_figures_hashed:
            excluded.update(CARD_FIGURE_FILENAMES)
        auxiliary = {
            path.relative_to(root).as_posix(): {
                "bytes": int(path.stat().st_size),
                "sha256": _file_content_sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.name not in excluded
            and path.suffix not in {".safetensors", ".pqcb"}
        }
        if auxiliary:
            payload["auxiliary_files"] = auxiliary
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def accepted_model_shas(model_dir: str | os.PathLike) -> tuple[str, ...]:
    """Current identity first, then each superseded scope as a fallback.

    A card describes its artifact faithfully under the rules it was computed
    with, so a scope change must not declare every published artifact changed.
    Each fallback is strictly narrower than the current scope (it attests MORE,
    or the same), and no legacy scope can produce a current-scope sha, so a
    card written today is never checked the weak way.
    """
    return (
        compute_model_sha(model_dir),
        # Pre-2026-08-15 native cards: auxiliary files were unattested.
        compute_model_sha(model_dir, legacy_native_scope=True),
        # Cards stamped while `README.md` was still hashed.
        compute_model_sha(model_dir, legacy_readme_hashed=True),
        # Cards stamped while the card figures were still hashed.
        compute_model_sha(model_dir, legacy_figures_hashed=True),
    )


def artifact_bytes(model_dir: str | os.PathLike) -> int:
    """Exact weight + codebook payload footprint (what the box must hold)."""
    root = Path(model_dir)
    total = 0
    for pattern in ("*.safetensors", "*.gguf", "*.pqcb"):
        for p in root.glob(pattern):
            total += p.stat().st_size
    return int(total)


def _file_content_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_weight_content_manifest(model_dir: str | os.PathLike) -> dict[str, Any]:
    """Hash each finished safetensors container once at the export boundary."""
    root = Path(model_dir)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights to attest under {root}")
    return {
        "schema": WEIGHT_CONTENT_MANIFEST_SCHEMA,
        "algorithm": "sha256",
        "files": {
            path.name: {
                "bytes": int(path.stat().st_size),
                "sha256": _file_content_sha256(path),
            }
            for path in files
        },
    }


def _validate_weight_content_manifest(
    manifest: object,
    weights: Mapping[str, int],
    *,
    where: str | os.PathLike,
) -> None:
    if not isinstance(manifest, Mapping) or manifest.get(
        "schema"
    ) != WEIGHT_CONTENT_MANIFEST_SCHEMA or manifest.get("algorithm") != "sha256":
        raise ValueError(f"invalid weight content manifest in {where}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(weights):
        raise ValueError(
            f"weight content manifest file set differs from weights in {where}"
        )
    for name, expected_bytes in weights.items():
        row = files.get(name)
        if not isinstance(row, Mapping) or row.get("bytes") != expected_bytes or not (
            isinstance(row.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256")))
        ):
            raise ValueError(
                f"invalid weight content manifest entry for {name!r} in {where}"
            )


def _content_stat(st: os.stat_result) -> dict[str, int]:
    return {
        "device": int(st.st_dev),
        "inode": int(st.st_ino),
        "bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
    }


def _closed_weight_content_manifest(
    manifest: object,
    *,
    where: str | os.PathLike,
) -> dict[str, dict[str, object]]:
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"schema", "algorithm", "files"}
        or manifest.get("schema") != WEIGHT_CONTENT_MANIFEST_SCHEMA
        or manifest.get("algorithm") != "sha256"
    ):
        raise ValueError(f"invalid closed weight content manifest in {where}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"empty weight content manifest in {where}")
    normalized: dict[str, dict[str, object]] = {}
    for name, row in files.items():
        if (
            not isinstance(name, str)
            or not name.endswith(".safetensors")
            or Path(name).name != name
            or not isinstance(row, Mapping)
            or set(row) != {"bytes", "sha256"}
            or type(row.get("bytes")) is not int
            or row.get("bytes") < 0
            or not isinstance(row.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
        ):
            raise ValueError(
                f"invalid closed weight content manifest entry {name!r} in {where}"
            )
        normalized[name] = {
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        }
    return dict(sorted(normalized.items()))


def _closed_tensor_digest_ledger(
    ledger: object,
    *,
    where: str | os.PathLike,
) -> dict[str, str]:
    if not isinstance(ledger, Mapping) or not ledger:
        raise ValueError(f"empty tensor digest ledger in {where}")
    normalized = {str(name): str(digest) for name, digest in ledger.items()}
    if (
        len(normalized) != len(ledger)
        or any(not name for name in normalized)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in normalized.values()
        )
    ):
        raise ValueError(f"invalid tensor digest ledger in {where}")
    return dict(sorted(normalized.items()))


def _strict_json_object(raw: bytes, *, where: str) -> Mapping[str, Any]:
    def _pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"{where}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def _constant(value: str) -> None:
        raise ValueError(f"{where}: nonfinite JSON constant {value!r}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where}: invalid safetensors header JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{where}: safetensors header must be an object")
    return payload


def safetensors_header_spans(
    raw_header: bytes,
    *,
    data_bytes: int,
    where: str,
) -> tuple[tuple[int, int, str], ...]:
    """Validate one header and return its exact contiguous payload geometry."""

    header = _strict_json_object(raw_header, where=where)
    spans: list[tuple[int, int, str]] = []
    for tensor_name, descriptor in header.items():
        if tensor_name == "__metadata__":
            if not isinstance(descriptor, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in descriptor.items()
            ):
                raise ValueError(
                    f"{where}: __metadata__ must be a string-to-string object"
                )
            continue
        offsets = descriptor.get("data_offsets") if isinstance(
            descriptor, Mapping
        ) else None
        dtype = descriptor.get("dtype") if isinstance(
            descriptor, Mapping
        ) else None
        shape = descriptor.get("shape") if isinstance(
            descriptor, Mapping
        ) else None
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"dtype", "shape", "data_offsets"}
            or dtype not in _SAFETENSORS_DTYPE_BITS
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise ValueError(f"{where}: invalid tensor header {tensor_name!r}")
        start, end = offsets
        if not 0 <= start <= end <= data_bytes:
            raise ValueError(f"{where}: invalid data span for {tensor_name!r}")
        expected_bytes = (
            int(math.prod(shape)) * _SAFETENSORS_DTYPE_BITS[str(dtype)] + 7
        ) // 8
        if end - start != expected_bytes:
            raise ValueError(
                f"{where}: tensor {tensor_name!r} span is {end - start}B but "
                f"{dtype}{tuple(shape)} requires {expected_bytes}B"
            )
        spans.append((start, end, tensor_name))
    spans.sort()
    if not spans:
        raise ValueError(f"{where}: safetensors container has no tensors")
    cursor = 0
    for start, end, tensor_name in spans:
        if start != cursor:
            relation = "overlaps" if start < cursor else "leaves a gap after"
            raise ValueError(
                f"{where}: tensor {tensor_name!r} {relation} the preceding "
                f"data span (expected offset {cursor}, got {start})"
            )
        cursor = end
    if cursor != data_bytes:
        raise ValueError(
            f"{where}: tensor spans end at {cursor}B but data area is "
            f"{data_bytes}B"
        )
    return tuple(spans)


def _read_exact_fd(
    fd: int,
    length: int,
    *,
    digest: Any,
    tensor_digest: Any | None = None,
    capture: bool = False,
) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    remaining = int(length)
    calls = 0
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        calls += 1
        if not chunk:
            raise ValueError("truncated safetensors content during verification")
        digest.update(chunk)
        if tensor_digest is not None:
            tensor_digest.update(chunk)
        if capture:
            chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks), calls


def _verify_open_safetensors_fd(
    fd: int,
    *,
    name: str,
    initial_stat: os.stat_result,
) -> tuple[dict[str, object], int, int]:
    """Hash one container and all tensor spans in one forward traversal."""

    full_digest = hashlib.sha256()
    raw_length, read_calls = _read_exact_fd(
        fd, 8, digest=full_digest, capture=True
    )
    header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
    if (
        header_length <= 0
        or header_length > _MAX_SAFETENSORS_HEADER_BYTES
        or header_length > int(initial_stat.st_size) - 8
    ):
        raise ValueError(f"{name}: invalid safetensors header length {header_length}")
    raw_header, calls = _read_exact_fd(
        fd, header_length, digest=full_digest, capture=True
    )
    read_calls += calls
    data_bytes = int(initial_stat.st_size) - 8 - header_length
    spans = safetensors_header_spans(
        raw_header,
        data_bytes=data_bytes,
        where=name,
    )
    tensor_digests: dict[str, str] = {}
    bytes_read = 8 + header_length
    for start, end, tensor_name in spans:
        tensor_digest = hashlib.sha256()
        _ignored, calls = _read_exact_fd(
            fd,
            end - start,
            digest=full_digest,
            tensor_digest=tensor_digest,
        )
        read_calls += calls
        bytes_read += end - start
        tensor_digests[tensor_name] = tensor_digest.hexdigest()
    final_stat = os.fstat(fd)
    if _content_stat(final_stat) != _content_stat(initial_stat):
        raise ValueError(f"{name}: file stat changed during content verification")
    return {
        "stat": _content_stat(final_stat),
        "sha256": full_digest.hexdigest(),
        "tensor_sha256": dict(sorted(tensor_digests.items())),
    }, bytes_read, read_calls


def _open_artifact_directory(root: Path) -> tuple[int, dict[str, int]]:
    before = root.lstat()
    if not stat.S_ISDIR(before.st_mode) or root.is_symlink():
        raise ValueError(f"artifact root must be a real directory: {root}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(root, flags)
    opened = os.fstat(directory_fd)
    if _content_stat(opened) != _content_stat(before):
        os.close(directory_fd)
        raise ValueError(f"artifact root changed while opening: {root}")
    return directory_fd, {
        "device": int(opened.st_dev),
        "inode": int(opened.st_ino),
    }


def _current_safetensors_names(directory_fd: int) -> list[str]:
    return sorted(
        name for name in os.listdir(directory_fd)
        if isinstance(name, str) and name.endswith(".safetensors")
    )


def _open_regular_nofollow(
    directory_fd: int,
    name: str,
) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"cannot safely open weight container {name!r}: {exc}") from exc
    opened = os.fstat(fd)
    if not stat.S_ISREG(opened.st_mode):
        os.close(fd)
        raise ValueError(f"weight container is not a regular file: {name!r}")
    return fd, opened


def verify_safetensors_content_once(
    model_dir: str | os.PathLike,
    *,
    expected_weight_manifest: Mapping[str, Any] | None,
    expected_tensor_sha256: Mapping[str, str],
    expected_tensor_to_file: Mapping[str, str] | None = None,
    expected_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Verify container and tensor hashes in one sequential pass per shard.

    The returned receipt is process-local no-clobber evidence.  It is never an
    artifact wire field: inode and timestamp bindings deliberately become
    invalid after a copy, at which point an independent validator performs one
    fresh pass over the copied containers.
    """

    root = Path(model_dir)
    manifest_files = (
        _closed_weight_content_manifest(expected_weight_manifest, where=root)
        if expected_weight_manifest is not None
        else None
    )
    expected_names = (
        sorted(str(name) for name in expected_files)
        if expected_files is not None
        else (sorted(manifest_files) if manifest_files is not None else None)
    )
    if expected_names is None or not expected_names or (
        len(set(expected_names)) != len(expected_names)
        or any(Path(name).name != name for name in expected_names)
    ):
        raise ValueError(
            f"one-pass verification requires exact container names in {root}"
        )
    if manifest_files is not None and expected_names != sorted(manifest_files):
        raise ValueError(
            f"expected container names differ from weight manifest in {root}"
        )
    expected_tensors = _closed_tensor_digest_ledger(
        expected_tensor_sha256, where=root
    )
    directory_fd, root_identity = _open_artifact_directory(root)
    files: dict[str, dict[str, object]] = {}
    content_bytes_read = 0
    read_calls = 0
    try:
        names = _current_safetensors_names(directory_fd)
        if names != expected_names:
            raise ValueError(
                f"safetensors set differs from weight manifest in {root}"
            )
        for name in names:
            fd, opened = _open_regular_nofollow(directory_fd, name)
            try:
                record, bytes_read, calls = _verify_open_safetensors_fd(
                    fd, name=name, initial_stat=opened
                )
            finally:
                os.close(fd)
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode) or (
                _content_stat(current) != record["stat"]
            ):
                raise ValueError(
                    f"{name}: path changed during content verification"
                )
            files[name] = record
            content_bytes_read += bytes_read
            read_calls += calls
        if _current_safetensors_names(directory_fd) != names:
            raise ValueError(
                f"safetensors namespace changed during verification: {root}"
            )
        current_root = root.lstat()
        if (
            not stat.S_ISDIR(current_root.st_mode)
            or int(current_root.st_dev) != root_identity["device"]
            or int(current_root.st_ino) != root_identity["inode"]
        ):
            raise ValueError(f"artifact root changed during verification: {root}")
    finally:
        os.close(directory_fd)

    observed_manifest = {
        "schema": WEIGHT_CONTENT_MANIFEST_SCHEMA,
        "algorithm": "sha256",
        "files": {
            name: {
                "bytes": row["stat"]["bytes"],
                "sha256": row["sha256"],
            }
            for name, row in sorted(files.items())
        },
    }
    if expected_weight_manifest is not None and (
        observed_manifest != dict(expected_weight_manifest)
    ):
        raise ValueError(
            f"weight content manifest differs from finalized bytes in {root}"
        )
    observed_tensors = {
        tensor_name: digest
        for row in files.values()
        for tensor_name, digest in row["tensor_sha256"].items()
    }
    if dict(sorted(observed_tensors.items())) != expected_tensors:
        raise ValueError(
            f"tensor digest ledger differs from finalized bytes in {root}"
        )
    if expected_tensor_to_file is not None:
        normalized_map = {
            str(name): str(filename)
            for name, filename in expected_tensor_to_file.items()
        }
        observed_map = {
            tensor_name: filename
            for filename, row in files.items()
            for tensor_name in row["tensor_sha256"]
        }
        if normalized_map != dict(sorted(observed_map.items())):
            raise ValueError(
                f"tensor-to-shard map differs from finalized headers in {root}"
            )
    return {
        "schema": SAFETENSORS_CONTENT_RECEIPT_SCHEMA,
        "source": "verified_read",
        "content_read_passes": 1,
        "content_bytes_read": int(content_bytes_read),
        "read_calls": int(read_calls),
        "root": root_identity,
        "files": files,
    }


def capture_attested_safetensors_write_receipt(
    model_dir: str | os.PathLike,
    *,
    weight_manifest: Mapping[str, Any],
    tensor_sha256: Mapping[str, str],
    tensor_to_file: Mapping[str, str],
) -> dict[str, Any]:
    """Capture no-clobber stats for bytes hashed by an atomic writer.

    This function does not infer or hash content.  Its digest arguments must
    come from the writer that fed the same bytes to the just-published files.
    The receipt's inode/stat binding is rechecked before strict finalization,
    so any intervening path swap or mutation invalidates the zero-reread path.
    """

    root = Path(model_dir)
    manifest_files = _closed_weight_content_manifest(weight_manifest, where=root)
    tensor_ledger = _closed_tensor_digest_ledger(tensor_sha256, where=root)
    normalized_map = {
        str(name): str(filename) for name, filename in tensor_to_file.items()
    }
    if (
        len(normalized_map) != len(tensor_to_file)
        or set(normalized_map) != set(tensor_ledger)
        or set(normalized_map.values()) != set(manifest_files)
        or any(
            Path(filename).name != filename
            for filename in normalized_map.values()
        )
    ):
        raise ValueError(
            f"attested writer tensor-to-file map is incomplete in {root}"
        )
    directory_fd, root_identity = _open_artifact_directory(root)
    files: dict[str, dict[str, object]] = {}
    try:
        names = _current_safetensors_names(directory_fd)
        if names != sorted(manifest_files):
            raise ValueError(
                f"safetensors set differs from attested writer output in {root}"
            )
        for name in names:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode):
                raise ValueError(
                    f"attested weight container is not regular: {name!r}"
                )
            expected_file = manifest_files[name]
            if int(current.st_size) != expected_file["bytes"]:
                raise ValueError(
                    f"attested weight size changed before receipt: {name!r}"
                )
            files[name] = {
                "stat": _content_stat(current),
                "sha256": expected_file["sha256"],
                "tensor_sha256": {
                    tensor_name: tensor_ledger[tensor_name]
                    for tensor_name, filename in sorted(normalized_map.items())
                    if filename == name
                },
            }
        if _current_safetensors_names(directory_fd) != names:
            raise ValueError(
                f"safetensors namespace changed while capturing receipt: {root}"
            )
        current_root = root.lstat()
        if (
            not stat.S_ISDIR(current_root.st_mode)
            or int(current_root.st_dev) != root_identity["device"]
            or int(current_root.st_ino) != root_identity["inode"]
        ):
            raise ValueError(f"artifact root changed while capturing receipt: {root}")
    finally:
        os.close(directory_fd)
    return {
        "schema": SAFETENSORS_CONTENT_RECEIPT_SCHEMA,
        "source": "attested_write",
        "content_read_passes": 0,
        "content_bytes_read": 0,
        "read_calls": 0,
        "root": root_identity,
        "files": files,
    }


def validate_safetensors_content_receipt(
    model_dir: str | os.PathLike,
    receipt: Mapping[str, Any],
    *,
    expected_weight_manifest: Mapping[str, Any],
    expected_tensor_sha256: Mapping[str, str],
    expected_tensor_to_file: Mapping[str, str],
) -> None:
    """Reuse a content receipt only while every bound path stat is unchanged."""

    root = Path(model_dir)
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != {
            "schema",
            "source",
            "content_read_passes",
            "content_bytes_read",
            "read_calls",
            "root",
            "files",
        }
        or receipt.get("schema") != SAFETENSORS_CONTENT_RECEIPT_SCHEMA
        or receipt.get("source") not in {"attested_write", "verified_read"}
        or type(receipt.get("content_read_passes")) is not int
        or receipt.get("content_read_passes") not in {0, 1}
        or type(receipt.get("content_bytes_read")) is not int
        or receipt.get("content_bytes_read") < 0
        or type(receipt.get("read_calls")) is not int
        or receipt.get("read_calls") < 0
    ):
        raise ValueError(f"invalid safetensors content receipt in {root}")
    if receipt.get("source") == "attested_write" and (
        receipt.get("content_read_passes") != 0
        or receipt.get("content_bytes_read") != 0
        or receipt.get("read_calls") != 0
    ):
        raise ValueError(f"invalid attested-write pass accounting in {root}")
    if receipt.get("source") == "verified_read" and (
        receipt.get("content_read_passes") != 1
        or receipt.get("content_bytes_read") <= 0
        or receipt.get("read_calls") <= 0
    ):
        raise ValueError(f"invalid verified-read pass accounting in {root}")
    root_record = receipt.get("root")
    files = receipt.get("files")
    if (
        not isinstance(root_record, Mapping)
        or set(root_record) != {"device", "inode"}
        or any(type(root_record.get(key)) is not int for key in root_record)
        or not isinstance(files, Mapping)
        or not files
    ):
        raise ValueError(f"invalid safetensors content receipt scope in {root}")
    manifest_files = _closed_weight_content_manifest(
        expected_weight_manifest, where=root
    )
    tensor_ledger = _closed_tensor_digest_ledger(
        expected_tensor_sha256, where=root
    )
    normalized_map = {
        str(name): str(filename)
        for name, filename in expected_tensor_to_file.items()
    }
    if set(normalized_map) != set(tensor_ledger):
        raise ValueError(f"expected tensor-to-file map is incomplete in {root}")
    directory_fd, current_root_record = _open_artifact_directory(root)
    try:
        if current_root_record != dict(root_record):
            raise ValueError(f"artifact root differs from content receipt: {root}")
        names = _current_safetensors_names(directory_fd)
        if names != sorted(files) or names != sorted(manifest_files):
            raise ValueError(
                f"safetensors set differs from content receipt in {root}"
            )
        observed_tensors: dict[str, str] = {}
        observed_map: dict[str, str] = {}
        for name in names:
            row = files.get(name)
            row_stat = row.get("stat") if isinstance(row, Mapping) else None
            row_tensors = row.get("tensor_sha256") if isinstance(
                row, Mapping
            ) else None
            if (
                not isinstance(row, Mapping)
                or set(row) != {"stat", "sha256", "tensor_sha256"}
                or not isinstance(row_stat, Mapping)
                or set(row_stat) != {
                    "device", "inode", "bytes", "mtime_ns", "ctime_ns"
                }
                or any(type(row_stat.get(key)) is not int for key in row_stat)
                or not isinstance(row.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                or not isinstance(row_tensors, Mapping)
            ):
                raise ValueError(
                    f"invalid safetensors content receipt row {name!r} in {root}"
                )
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode) or (
                _content_stat(current) != dict(row_stat)
            ):
                raise ValueError(
                    f"weight container changed after content receipt: {name!r}"
                )
            expected_file = manifest_files[name]
            if (
                row["sha256"] != expected_file["sha256"]
                or row_stat["bytes"] != expected_file["bytes"]
            ):
                raise ValueError(
                    f"weight manifest differs from content receipt: {name!r}"
                )
            normalized_row = _closed_tensor_digest_ledger(
                row_tensors, where=f"{root}/{name}"
            )
            for tensor_name, digest in normalized_row.items():
                if tensor_name in observed_tensors:
                    raise ValueError(
                        f"tensor appears in multiple receipt shards: {tensor_name!r}"
                    )
                observed_tensors[tensor_name] = digest
                observed_map[tensor_name] = name
        if _current_safetensors_names(directory_fd) != names:
            raise ValueError(
                f"safetensors namespace changed while checking receipt: {root}"
            )
        if dict(sorted(observed_tensors.items())) != tensor_ledger:
            raise ValueError(f"tensor ledger differs from content receipt in {root}")
        if dict(sorted(observed_map.items())) != dict(sorted(normalized_map.items())):
            raise ValueError(
                f"tensor-to-file map differs from content receipt in {root}"
            )
    finally:
        os.close(directory_fd)


def safetensors_content_receipt_manifest(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the portable content manifest from a validated-style receipt."""

    files = receipt.get("files") if isinstance(receipt, Mapping) else None
    if not isinstance(files, Mapping) or not files:
        raise ValueError("safetensors content receipt has no file rows")
    return {
        "schema": WEIGHT_CONTENT_MANIFEST_SCHEMA,
        "algorithm": "sha256",
        "files": {
            str(name): {
                "bytes": int(row["stat"]["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for name, row in sorted(files.items())
        },
    }


def weight_stat_attestation(model_dir: str | os.PathLike) -> dict[str, Any]:
    """Cheap post-hash mutation detector for the large weight containers."""
    root = Path(model_dir)
    files = sorted(root.glob("*.safetensors"))
    return {
        "schema": WEIGHT_STAT_ATTESTATION_SCHEMA,
        "files": {
            path.name: {
                "bytes": int((info := path.stat()).st_size),
                "mtime_ns": int(info.st_mtime_ns),
                "ctime_ns": int(info.st_ctime_ns),
            }
            for path in files
        },
    }


def assert_weight_stat_attestation(
    card: Mapping[str, Any],
    model_dir: str | os.PathLike,
) -> None:
    """Fail if a weight changed after the card bound its exact content claim."""
    expected = card.get("weight_stat_attestation")
    if expected is None:
        return  # Backward-compatible verification of historical cards.
    if not isinstance(expected, Mapping) or expected.get(
        "schema"
    ) != WEIGHT_STAT_ATTESTATION_SCHEMA:
        raise ValueError("shipcard carries an invalid weight stat attestation")
    observed = weight_stat_attestation(model_dir)
    if observed != expected:
        raise ValueError(
            "weight file stats changed after export; refusing cached content "
            "identity (run shipcard_cli reattest after a legitimate "
            "cross-filesystem copy)"
        )


def reattest_weight_stats(
    shipcard_path: str | os.PathLike,
    model_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Full-hash a copied CB artifact, then refresh only its cheap stat cache."""
    path = Path(shipcard_path)
    root = Path(model_dir) if model_dir is not None else path.resolve().parent
    card = load_shipcard(path)
    quant_path = root / "quant_config.json"
    if not quant_path.is_file():
        raise ValueError("weight re-attestation requires a CB quant_config.json")
    quant_config = json.loads(quant_path.read_text(encoding="utf-8"))
    provenance = quant_config.get("provenance")
    expected = provenance.get("weight_content_manifest") if isinstance(
        provenance, Mapping
    ) else None
    if expected is None:
        raise ValueError("CB artifact has no immutable weight content manifest")
    observed = build_weight_content_manifest(root)
    if observed != expected:
        raise ValueError(
            "weight content differs from the immutable export manifest; "
            "refusing to refresh the stat attestation"
        )
    expected_model_sha = card.get("model_sha")
    if expected_model_sha not in set(accepted_model_shas(root)):
        raise ValueError("copied artifact model_sha differs from the shipcard")
    card["weight_stat_attestation"] = weight_stat_attestation(root)
    card["updated"] = _now()
    write_shipcard(path, card)
    return card


def file_sha256(path: str | os.PathLike) -> str | None:
    try:
        return _file_content_sha256(path)
    except Exception:
        return None


def git_provenance(repo: str | os.PathLike | None = None) -> dict[str, Any]:
    """``{commit, dirty}`` for the tree that produced this record.

    Read-only Docker exports may have the source mounted without usable
    worktree metadata.  In that case the launch boundary can pass the same
    exact commit override used by producer identities plus an independently
    preflighted dirty bit.  When git is available, both overrides are checked
    against it rather than silently replacing contradictory observations.
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]

    def _run(cmd: list[str]) -> str | None:
        try:
            return subprocess.run(
                cmd, cwd=root, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            ).stdout.strip()
        except Exception:
            return None

    commit_override = str(os.environ.get(
        "PRISMAQUANT_IDENTITY_GIT_COMMIT", ""
    )).strip().lower()
    if commit_override and re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit_override
    ) is None:
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_COMMIT must be a full 40- or "
            "64-character hexadecimal commit id"
        )
    dirty_override_raw = str(os.environ.get(
        "PRISMAQUANT_IDENTITY_GIT_DIRTY", ""
    )).strip().lower()
    dirty_values = {
        "0": False, "false": False, "no": False,
        "1": True, "true": True, "yes": True,
    }
    if dirty_override_raw and dirty_override_raw not in dirty_values:
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_DIRTY must be one of "
            "0/1/false/true/no/yes"
        )

    observed_commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--short"])
    observed_dirty = None if status is None else bool(status)
    if (
        commit_override
        and observed_commit is not None
        and observed_commit.lower() != commit_override
    ):
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_COMMIT contradicts the mounted "
            f"worktree HEAD {observed_commit}"
        )
    dirty_override = (
        dirty_values[dirty_override_raw] if dirty_override_raw else None
    )
    if (
        dirty_override is not None
        and observed_dirty is not None
        and observed_dirty != dirty_override
    ):
        raise ValueError(
            "PRISMAQUANT_IDENTITY_GIT_DIRTY contradicts the mounted "
            f"worktree dirty={observed_dirty}"
        )
    return {
        "commit": commit_override or observed_commit,
        "dirty": (
            dirty_override if dirty_override is not None else observed_dirty
        ),
    }


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


# ---------------------------------------------------------------------------
# Build lane
# ---------------------------------------------------------------------------
def lane_gate_slots(lane: str | None) -> tuple[str, ...]:
    """The shipcard slots ``lane``'s own declaration says its gates close.

    Read from ``lane_specs/<lane>.json``, so a lane that declares a gate opens
    that gate's slot without an edit here.  An unknown or absent lane resolves
    to ``()`` -- historical cards carry no ``lane`` and must keep verifying
    against exactly the base set they were opened with.  It is not a widening
    to answer ``()``: :func:`required_slots` UNIONS this with
    :data:`REQUIRED_SLOTS` and never replaces it, so no lane can subtract a
    base requirement by under-declaring (the GGUF lane declares no
    ``native_export.graph`` gate and is still required to close that slot).

    A declared slot outside :data:`ALL_SLOTS` RAISES.  Filtering it away would
    mean the lane declared a gate, the card never opened it and ``verify``
    passed an artifact that closed nothing -- the same silence one link
    earlier that RobTand/prismaquant#119 reported.  Admitting a new one is an
    edit here plus the verifier ``verify`` replays for it
    (RobTand/prismaquant#162).
    """
    if not lane:
        return ()
    from prismaquant.lane_spec import lane_spec_for_container, load_lane_spec

    # The card names the EXPORT_CONTAINER (`compressed-tensors`, hyphen); the
    # spec FILE is named for the lane id (`compressed_tensors`, underscore).
    # Resolve by container first so the card can carry the operator-facing
    # spelling, and fall back to the id so a caller holding a spec id is not
    # silently answered with "no slots".
    try:
        spec = lane_spec_for_container(str(lane))
    except Exception:
        try:
            spec = load_lane_spec(str(lane))
        except Exception:
            return ()
    declared = spec.shipcard_slots()
    # FAIL CLOSED, not filter. A lane that declares a slot this module has no
    # name for is a repository defect, and the two ways of meeting it are not
    # equivalent: dropping the slot means the lane declared a gate, the card
    # never opened it, and `verify` passed an artifact that closed nothing --
    # silently, which is the exact shape #119 reported one link earlier.
    # Raising says which lane and which slot, so the fix is one edit here.
    unknown = [slot for slot in declared if slot not in ALL_SLOTS]
    if unknown:
        raise KeyError(
            f"lane {lane!r} declares shipcard slot(s) {unknown} that "
            f"shipcard.py does not know; known: {list(ALL_SLOTS)}. Add the "
            "slot to LANE_SCOPED_SLOTS together with the verifier `verify` "
            "must replay for it -- a slot with no verifier is a slot any "
            "record closes (RobTand/prismaquant#162)"
        )
    return declared


def build_shipcard(
    model_dir: str | os.PathLike,
    *,
    build: Mapping[str, Any] | None = None,
    lane: str | None = None,
) -> dict[str, Any]:
    """Open a fresh record: build-lane facts filled, every slot empty.

    ``lane`` is the ``EXPORT_CONTAINER`` this artifact was built for.  It is
    stamped on the card and it decides which lane-scoped slots the card opens,
    so the gates a lane DECLARES and the slots its artifact must CLOSE are one
    object.  Omitting it reproduces the historical card exactly (base slots,
    no ``lane`` key) -- which is what every pre-2026-09-03 native card is.
    """
    root = Path(model_dir)
    from prismaquant.export_output_safety import directory_publication_target

    build_payload = dict(build or {})
    slots = list(REQUIRED_SLOTS) + [
        slot for slot in lane_gate_slots(lane) if slot not in REQUIRED_SLOTS
    ]
    # A rate-axis artifact owes the byte-matched uniform control's verdict, so
    # open the slot at export time.  `required_slots` re-derives the
    # obligation from the artifact itself either way; opening the key here is
    # what keeps the refusal a FORCEABLE evidence failure rather than the
    # publisher's non-forceable "omits required slot key(s)" structural one.
    if _is_rate_axis_artifact({"build": build_payload}, model_dir=root):
        slots.append(UNIFORM_CONTROL_SLOT)
    card = {
        "schema": SCHEMA,
        "created": _now(),
        "model_dir": str(directory_publication_target(root)),
        "model_sha": compute_model_sha(root),
        "artifact_bytes": artifact_bytes(root),
        "reserved_file_bytes": SHIPCARD_RESERVED_BYTES,
        "build": build_payload,
        "slots": {slot: None for slot in slots},
    }
    if lane:
        card["lane"] = str(lane)
    quant_path = root / "quant_config.json"
    if quant_path.is_file():
        try:
            quant_config = json.loads(quant_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            quant_config = None
        provenance = quant_config.get("provenance") if isinstance(
            quant_config, Mapping
        ) else None
        if isinstance(provenance, Mapping) and provenance.get(
            "weight_content_manifest"
        ) is not None:
            card["weight_stat_attestation"] = weight_stat_attestation(root)
        # Principle 9 requires the artifact's route disposition -- a declared
        # non-native target or an explicit override -- to be stamped on the
        # card, and principle 12 requires the route census to travel with the
        # bpp claim rather than only inside quant_config.  The exporters that
        # stamped ``provenance.cb_route_status`` were the retired Gridbook
        # lane's, and both the field and its two summarisers went into
        # ``archive/gridbook_lane_2026-09-02/`` with them.  A live lane that
        # needs the same stamp adds its own summariser here rather than
        # inheriting a retired schema.
    return card


def write_shipcard(path: str | os.PathLike, card: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(card, indent=2, default=str) + "\n").encode("utf-8")
    reserved = card.get("reserved_file_bytes")
    if reserved is not None:
        if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved <= 0:
            raise ValueError("shipcard reserved_file_bytes must be a positive integer")
        if len(encoded) > reserved:
            # A full card outgrows pretty-printing long before it outgrows its
            # reservation: indent=2 inflates this schema ~1.5x, and a six-slot
            # DSv4 card crossed the line on the LAST slot fill -- dropping a
            # record every gate had already passed.  The reservation is a byte
            # contract (the file's size is part of the frozen artifact
            # inventory), not a formatting contract; fall back to compact JSON
            # before refusing.
            encoded = (
                json.dumps(card, separators=(",", ":"), default=str) + "\n"
            ).encode("utf-8")
        if len(encoded) > reserved:
            raise ValueError(
                f"shipcard needs {len(encoded)} bytes but its fixed reservation "
                f"is {reserved} bytes; refusing to invalidate the artifact inventory"
            )
        encoded += b" " * (reserved - len(encoded))
    out.write_bytes(encoded)
    return out


def load_shipcard(path: str | os.PathLike) -> dict[str, Any]:
    card = json.loads(Path(path).read_text())
    if not isinstance(card, dict) or "slots" not in card:
        raise ValueError(f"not a shipcard: {path}")
    return card


# ---------------------------------------------------------------------------
# Serve lane
# ---------------------------------------------------------------------------
def make_record(
    *,
    slot: str,
    tool: str,
    passed: bool,
    model_sha: str | None,
    metrics: Mapping[str, Any] | None = None,
    detail: str = "",
    spec_decode_detected: bool | None = None,
    serve_fingerprint: str | None = None,
    git_commit: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One serve-lane verdict block."""
    if slot not in ALL_SLOTS:
        raise KeyError(
            f"unknown shipcard slot {slot!r}; known: {list(ALL_SLOTS)}")
    record: dict[str, Any] = {
        "slot": slot,
        "tool": tool,
        "filled_at": _now(),
        "passed": bool(passed),
        "model_sha": model_sha,
        "spec_decode_detected": spec_decode_detected,
        "serve_fingerprint": serve_fingerprint,
        "git_commit": git_commit,
        "detail": detail,
        "metrics": dict(metrics or {}),
    }
    if extra:
        record.update(dict(extra))
    return record


def fill_slot(
    path: str | os.PathLike,
    slot: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a verdict block to `slot` in the shipcard at `path` (in place)."""
    card = load_shipcard(path)
    card_model_dir = Path(path).resolve().parent
    if card.get("weight_stat_attestation") is not None:
        assert_weight_stat_attestation(card, card_model_dir)
    if slot not in card["slots"]:
        raise KeyError(
            f"shipcard {path} has no slot {slot!r}; known: "
            f"{sorted(card['slots'])}")
    card["slots"][slot] = dict(record)
    card["updated"] = _now()
    write_shipcard(path, card)
    return card


def ensure_optional_slot(
    path: str | os.PathLike,
    slot: str,
) -> dict[str, Any]:
    """Add one recognized optional claim slot to an existing shipcard."""

    if slot not in OPTIONAL_SLOTS:
        raise KeyError(
            f"{slot!r} is not an optional shipcard slot; "
            f"known={list(OPTIONAL_SLOTS)}"
        )
    card = load_shipcard(path)
    if slot not in card["slots"]:
        card["slots"][slot] = None
        card["updated"] = _now()
        write_shipcard(path, card)
    return card


def fill_if_requested(
    path: str | os.PathLike | None,
    slot: str,
    record: Mapping[str, Any],
) -> None:
    """`fill_slot` when a `--shipcard` path was supplied; loud no-op otherwise.

    Serve-lane tools must never fail because of the record — the measurement is
    the point and the refusal lives in `verify`. Failures print and are ignored.
    """
    if not path:
        return
    try:
        fill_slot(path, slot, record)
        print(f"[shipcard] filled {slot} in {path}", flush=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[shipcard] WARN could not fill {slot} in {path}: {exc!r}",
              flush=True)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
    required: Iterable[str] | None = None,
) -> list[str]:
    """Return the list of reasons this artifact is not shippable (empty = OK)."""
    problems: list[str] = []
    expected_sha = card.get("model_sha")

    if model_dir is not None:
        try:
            assert_weight_stat_attestation(card, model_dir)
            accepted = accepted_model_shas(model_dir)
            on_disk = accepted[0]
            if expected_sha and on_disk != expected_sha:
                # A card written under a superseded scope still describes its
                # artifact faithfully under the identity it was computed with.
                # Accept that identity rather than declaring every published
                # artifact changed -- but only as a fallback, so a
                # current-scope card is never checked the weak way (no legacy
                # scope can produce a current-scope sha).
                if expected_sha in accepted[1:]:
                    on_disk = expected_sha
        except Exception as exc:
            problems.append(
                "artifact changed since the shipcard was opened or its "
                f"model_sha is not computable: {exc!r}"
            )
            on_disk = None
        if on_disk is not None and expected_sha and on_disk != expected_sha:
            problems.append(
                f"artifact changed since the shipcard was opened: "
                f"on-disk model_sha {on_disk[:12]} != shipcard "
                f"{str(expected_sha)[:12]} — re-run the serve lane")
            expected_sha = on_disk

    slots = card.get("slots") or {}
    # The published bpp is a load-bearing public claim: every matched-bpp
    # comparison against this artifact inherits it. A card whose claim
    # contradicts the bytes its own recipe declares has shipped twice (see
    # allocator_achieved_bpp), so refuse here rather than at card-writing time
    # — an export that already cost GPU hours should fail at PUBLICATION, the
    # blocking point, not lose its artifact to a gate. A card written before
    # this check existed carries no verdict and is left alone.
    achieved = ((card.get("build") or {}).get("achieved_bpp") or {})
    cross = achieved.get("cross_check") if isinstance(achieved, Mapping) else None
    if isinstance(cross, Mapping) and cross.get("verdict") == "DISAGREE":
        problems.append(
            "build.achieved_bpp contradicts the recipe's own serialized "
            f"bytes: {cross.get('detail')}"
        )
    # The exporter stamps ten build-lane keys; the cross-check above read one.
    # The rest of the forensic block is replayed here (#158).
    problems.extend(_verify_build_block(card))

    if required is None:
        required = required_slots(card, model_dir=model_dir)
    for slot in required:
        record = slots.get(slot)
        if not record:
            problems.append(f"{slot}: UNFILLED")
            continue
        if not isinstance(record, dict):
            problems.append(f"{slot}: malformed record ({type(record).__name__})")
            continue
        if record.get("slot") != slot:
            problems.append(
                f"{slot}: record declares slot {record.get('slot')!r}"
            )
        got = record.get("model_sha")
        if not got:
            problems.append(f"{slot}: record carries no model_sha")
        elif expected_sha and got != expected_sha:
            problems.append(
                f"{slot}: record model_sha {str(got)[:12]} != artifact "
                f"{str(expected_sha)[:12]} (record belongs to another build)")
        if slot == UNIFORM_CONTROL_SLOT:
            # Routed PAST the generic `passed` check, not around it: the
            # verdict lives in the two KLs, and this slot's verifier is the
            # one that understands the (deliberate, stamped, bound) override.
            # It re-derives `passed` from those KLs itself, so nothing is lost
            # by not testing the flag here.
            problems.extend(_verify_uniform_control_record(
                slot, record, card=card, model_dir=model_dir))
            continue
        if record.get("passed") is not True:
            problems.append(
                f"{slot}: FAILED — {record.get('detail') or 'no detail'}")
        if slot == "ship_gate":
            # Every lane. This replay ran only behind the retired Gridbook
            # codebook lane's `is_gridbook_cb` flag until 2026-09-02, which
            # meant a NATIVE card -- the default lane, and the one shipping
            # artifacts today -- closed its one universal slot on a bare
            # `passed` flag while the producer filed a threshold contract, a
            # check ledger, token evidence and an endpoint binding nobody
            # read (#156). None of that evidence was ever CB-specific: the
            # catastrophic bounds, the four checks and the validator's tool
            # identity are the same program on every lane, so generalising
            # the replay is a REMOVAL of the flag, not a new refusal.
            problems.extend(_verify_ship_gate_record(slot, record))
        if slot in GOLD_SLOTS:
            # Every lane. This ran under a Gridbook-CB flag until 2026-08-15,
            # which meant a NATIVE card -- the default lane, and the one
            # shipping artifacts today -- had its gold slots checked for
            # nothing but spec-decode: no finite metric, no serve fingerprint,
            # no producer commit, no position count. The generic evidence
            # requirements were never CB-specific and should never have been
            # behind it.
            problems.extend(_verify_gold_record(
                slot,
                record,
                model_dir=model_dir,
                require_current_artifact_path=False,
            ))
        if slot in GOLD_SLOTS:
            spec = record.get("spec_decode_detected")
            if spec is None:
                problems.append(
                    f"{slot}: spec_decode_detected is unknown — a gold number "
                    "measured against a spec-decode serve is the draft model's "
                    "NLL, so 'unknown' is not acceptable (§7.5)")
            elif spec:
                problems.append(
                    f"{slot}: spec_decode_detected is TRUE — this is draft-model "
                    "NLL, not the artifact's; re-measure on a no-spec serve")
        if slot.startswith("native_export."):
            # The slot-name check above compares `record["slot"]` to the slot
            # key; nothing compared `metrics.arm` to the slot suffix, so a
            # fabricated or mislabeled arm record passed. Replay it here.
            problems.extend(_verify_native_export_record(slot, record))
    return problems


def _gold_extension_requirements(
    quant_config: Mapping[str, Any],
) -> frozenset[str]:
    """Extension families implied by the finalized live assignment."""
    cb_formats: set[str] = set()
    dense_mxfp8 = False
    source_fp8_w8a16 = False

    config_groups = quant_config.get("config_groups")
    if config_groups is not None:
        if not isinstance(config_groups, Mapping):
            raise ValueError("quant_config config_groups is not an object")
        for key, group in config_groups.items():
            if not isinstance(key, str) or not isinstance(group, Mapping):
                raise ValueError("quant_config config_groups is malformed")
            fmt = group.get("format")
            if isinstance(fmt, str) and _CB_FORMAT_RE.fullmatch(fmt):
                cb_formats.add(fmt)

    provenance = quant_config.get("provenance")
    tensor_formats = provenance.get("tensor_formats") if isinstance(
        provenance, Mapping
    ) else None
    if tensor_formats is not None:
        if not isinstance(tensor_formats, Mapping):
            raise ValueError("quant_config tensor_formats is not an object")
        for qname, fmt in tensor_formats.items():
            if not isinstance(qname, str) or not isinstance(fmt, str):
                raise ValueError("quant_config tensor_formats is malformed")
            if _CB_FORMAT_RE.fullmatch(fmt):
                cb_formats.add(fmt)
            if fmt in {"MXFP8_UE8M0_G32", *_MXFP8_DENSE_WIRE_IDS}:
                dense_mxfp8 = True
            if fmt in {
                "FP8_BLOCK_UE8M0_SOURCE", *_FP8_SOURCE_W8A16_WIRE_IDS,
            }:
                source_fp8_w8a16 = True

    delegated = quant_config.get("source_passthrough")
    if delegated is not None:
        if (
            not isinstance(delegated, Mapping)
            or set(delegated) != {"version", "units"}
            or delegated.get("version") != 1
            or not isinstance(delegated.get("units"), Mapping)
        ):
            raise ValueError("source_passthrough is not the closed v1 declaration")
        for qname, wire in delegated["units"].items():
            if not isinstance(qname, str) or not isinstance(wire, str):
                raise ValueError("source_passthrough route is malformed")
            if wire in _MXFP8_DENSE_WIRE_IDS:
                dense_mxfp8 = True
            if wire in _FP8_SOURCE_W8A16_WIRE_IDS:
                source_fp8_w8a16 = True

    per_expert = quant_config.get("per_expert_format_groups")
    if per_expert is not None:
        if (
            not isinstance(per_expert, Mapping)
            or per_expert.get("version") != 1
            or not isinstance(per_expert.get("layers"), Mapping)
        ):
            raise ValueError("per_expert_format_groups is malformed")
        for families in per_expert["layers"].values():
            if not isinstance(families, Mapping):
                raise ValueError("per-expert family declaration is malformed")
            for entries in families.values():
                if not isinstance(entries, list):
                    raise ValueError("per-expert format groups are malformed")
                for entry in entries:
                    wire = entry.get("format_wire_id") if isinstance(
                        entry, Mapping
                    ) else None
                    if not isinstance(wire, str):
                        raise ValueError("per-expert format route is malformed")
                    if _CB_FORMAT_RE.fullmatch(wire):
                        cb_formats.add(wire)
                    if wire in _MXFP8_DENSE_WIRE_IDS:
                        dense_mxfp8 = True
                    if wire in _FP8_SOURCE_W8A16_WIRE_IDS:
                        source_fp8_w8a16 = True

    required: set[str] = set()
    if cb_formats:
        # Every CB execution family uses the main module for activation QDQ,
        # expansion/GEMV, and/or routed combine. Layout-v2 FP4 additionally
        # requires its v2 quality expander at load.
        required.add("cb_main")
        if quant_config.get("layout_version", 1) == 2 and any(
            name.startswith("NVFP4_CB_") for name in cb_formats
        ):
            required.add("cb_v2")
    if dense_mxfp8:
        required.add("mxfp8_dense")
    if source_fp8_w8a16:
        required.update({"fp8_source_w8a16", "cb_bf16_grouped"})
    return frozenset(required)


def _verify_gold_producer_identity(
    slot: str,
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    canonical_sha,
) -> list[str]:
    problems: list[str] = []
    producer = manifest.get("producer_identity")
    tool = manifest.get("measurement_tool")
    try:
        from tools.serve_fingerprint import (
            _GOLD_PRODUCER_COMMON_FILES,
            _GOLD_PRODUCER_TOOL_FILES,
        )
        expected_files = sorted(set(
            _GOLD_PRODUCER_COMMON_FILES + _GOLD_PRODUCER_TOOL_FILES[str(tool)]
        ))
    except Exception:
        expected_files = []
    if (
        not isinstance(producer, Mapping)
        or set(producer) != {
            "schema", "measurement_tool", "git_commit", "git_tree",
            "git_dirty", "source_files", "source_files_sha256",
        }
        or producer.get("schema") != GOLD_PRODUCER_IDENTITY_SCHEMA
        or producer.get("measurement_tool") != tool
        or producer.get("git_commit") != record.get("git_commit")
        or re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
            str(producer.get("git_commit", "")),
        ) is None
        or producer.get("git_dirty") is not False
        or (
            producer.get("git_tree") is not None
            and re.fullmatch(
                r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
                str(producer.get("git_tree")),
            ) is None
        )
    ):
        return [f"{slot}: gold producer is not a clean exact commit identity"]
    files = producer.get("source_files")
    if (
        not expected_files
        or not isinstance(files, Mapping)
        or sorted(files) != expected_files
        or producer.get("source_files_sha256") != canonical_sha(files)
    ):
        return [f"{slot}: gold producer source-file closure/digest differs"]
    for name, identity in files.items():
        if (
            not isinstance(name, str)
            or name.startswith("/")
            or ".." in Path(name).parts
            or not isinstance(identity, Mapping)
            or set(identity) != {"bytes", "sha256"}
            or isinstance(identity.get("bytes"), bool)
            or not isinstance(identity.get("bytes"), int)
            or identity.get("bytes", -1) <= 0
            or re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256", "")))
            is None
        ):
            problems.append(f"{slot}: gold producer source identity is malformed")
            break
    return problems


def _verify_gold_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
    require_current_artifact_path: bool = True,
) -> list[str]:
    """Require a slot-specific finite gold metric plus measurement identity."""
    problems: list[str] = []
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return [f"{slot}: missing structured gold metrics"]
    allowed = (
        ("kl_confident_mean", "kl_mean")
        if slot == "gold.kl"
        else ("ppl", "mean_nll")
    )
    finite = [
        key for key in allowed
        if key in metrics
        and not isinstance(metrics[key], bool)
        and isinstance(metrics[key], (int, float))
        and math.isfinite(float(metrics[key]))
        and float(metrics[key]) >= 0
    ]
    if not finite:
        problems.append(
            f"{slot}: carries no finite non-negative slot-specific metric "
            f"from {allowed}"
        )
    fingerprint = record.get("serve_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ) is None:
        problems.append(f"{slot}: missing exact serve fingerprint")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        problems.append(f"{slot}: missing full producer git commit")
    tool = record.get("tool")
    if not isinstance(tool, str) or not tool:
        problems.append(f"{slot}: missing measurement tool identity")
    if slot == "gold.kl":
        if not any(
            isinstance(metrics.get(key), int)
            and not isinstance(metrics.get(key), bool)
            and metrics.get(key, 0) > 0
            for key in ("n_positions", "n_samples")
        ):
            problems.append(f"{slot}: missing positive KL sample/position count")
        # A positive count was never enough. `--score-positions final` scores
        # ONE position per sequence -- the window-final context -- which is the
        # cheap last-token "hook KL" screen that is triage-only and must never
        # be a promotion number (§5). It reports n_samples=8 and sails through
        # the count check above, so the card could not tell an 8-position screen
        # from a 4088-position gold measurement. Caught 2026-08-14 on the
        # Qwen3.8-27B lane, where the driver simply omitted the flag and the
        # tool's default is `final`.
        if metrics.get("score_positions") != "all":
            problems.append(
                f"{slot}: score_positions={metrics.get('score_positions')!r} — "
                "the gold KL contract is every prompt position "
                "(--score-positions all, n_positions = n_samples*(seqlen-1)). "
                "'final' is the last-token hook screen: triage only, never a "
                "promotion metric."
            )
    elif not isinstance(metrics.get("n_tokens_scored"), int) or isinstance(
        metrics.get("n_tokens_scored"), bool
    ) or metrics.get("n_tokens_scored", 0) <= 0:
        problems.append(f"{slot}: missing positive scored-token count")
    return problems


def _verify_native_export_record(
    slot: str,
    record: Mapping[str, Any],
) -> list[str]:
    """Replay the smoke arm's own stamped metrics against the slot it closes.

    `validate_native_export._record_arm` stamps the arm it actually ran
    (`metrics.arm`), the residency it ran under (`enforce_eager`), and how
    much it generated (`generated_chars`) — but `verify` compared only the
    slot key against `record["slot"]`, so an eager receipt closed the graph
    slot and a pass that generated nothing verified clean. The only site
    that knows the truth already writes it; this reads it.
    """
    problems: list[str] = []
    arm = slot.split(".", 1)[1] if "." in slot else slot
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return [f"{slot}: missing structured arm metrics"]
    if metrics.get("arm") != arm:
        problems.append(
            f"{slot}: metrics.arm={metrics.get('arm')!r} does not match the "
            f"slot arm {arm!r} — a mislabeled receipt, not a result"
        )
    # `validate_native_export._run_arm`: `arm = "eager" if enforce_eager else
    # "graph"`. The two residencies are different numeric objects on the
    # Tessera lane, exercised by separate gates.
    expected_eager = {"eager": True, "graph": False}.get(arm)
    if (
        expected_eager is not None
        and metrics.get("enforce_eager") is not expected_eager
    ):
        problems.append(
            f"{slot}: enforce_eager={metrics.get('enforce_eager')!r} "
            f"disagrees with the {arm} arm — this receipt ran under the "
            "other residency"
        )
    produced = metrics.get("generated_chars")
    if record.get("passed") is True:
        if (
            isinstance(produced, bool)
            or not isinstance(produced, int)
            or produced <= 0
        ):
            problems.append(
                f"{slot}: passed but generated_chars={produced!r} — a smoke "
                "pass that generated nothing is not evidence"
            )
    elif (
        not isinstance(produced, bool)
        and isinstance(produced, int)
        and produced > 0
    ):
        problems.append(
            f"{slot}: failed yet claims generated_chars={produced!r}"
        )
    max_new_tokens = metrics.get("max_new_tokens")
    if max_new_tokens is not None and (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        problems.append(
            f"{slot}: max_new_tokens={max_new_tokens!r} is not a positive "
            "integer"
        )
    return problems


# ---------------------------------------------------------------------------
# The byte-matched uniform control (#121, closing prismaquant#117 / tessera#1)
# ---------------------------------------------------------------------------
# An allocation over a rate *axis* is a claim: that choosing a rung per Linear
# beats spending the same bytes everywhere.  On 2026-09-02 that claim was
# measured and it was false -- a PrismaQuant-allocated Tessera checkpoint
# served 2.00x worse KL than a byte-matched uniform arm at 4.0 bpp (0.3485 vs
# 0.1746; 2.33x at 3.0, 2.88x at 5.0) while every other check the pipeline
# owns passed: 196/196 units matched the plan's bytes, a layer-0 re-encode was
# byte-identical, census read 112/112.  The pipeline computed one artifact
# correctly.  It was the wrong artifact.
#
# On 2026-09-03 an ORACLE handed the measured per-unit KL table reached only
# 0.941x the uniform control at 4.0 bpp, P(worse) = 0.075 -- not significant.
# So the closure is a REFUSAL, not a better cost model: there is no prize at
# that rung to fund one with.  Below the knee the axis is real (3.0 bpp:
# oracle 0.748x, AURA 0.780x, both P = 0.000), which is why this is a gate on
# the claim and not a verdict on allocation.
#
# Principle 3 applied to allocation itself: an allocation that cannot beat its
# byte-matched uniform control has not earned its allocation, and does not
# ship.  The lesson this is written against is that provenance nothing
# consumes is a confession log -- 73.7% of a 92 GB body once rode an
# `arch::Sm80` fallback on Blackwell, recorded in its own selection.json,
# refused by nothing.  So this REFUSES, in `verify`, at publication.
#
# How the record gets here:
#   1. Tessera builds and prices the control:
#        python experiments/uniform_control.py plan --plan-json <candidate plan>
#      then serves it, and `verify` re-asserts the byte match on the two
#      exported manifests.
#   2. The SAME KL tool that filled `gold.kl` is run on the control checkpoint
#      -- the control arm's record is gold-record shaped and is replayed
#      through `_verify_gold_record` below, so a last-token or weight-space
#      number cannot close this slot.
#   3. `python -m prismaquant.shipcard_cli fill-control` writes both, plus the
#      `tessera.control.control_block()` JSON, into the slot.
def _is_rate_axis_artifact(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> bool:
    """Does this artifact's menu have a rate axis a uniform control is about?

    Read from the artifact's own ``config.json`` OR from the card's build
    block, OR-ed rather than either alone: `required_slots` runs both with and
    without the artifact (the CLI's ``verify`` takes ``--model-dir`` and
    defaults to none), and an obligation that a single erasure removes is not
    an obligation.  The archived Gridbook lane pinned exactly this shape
    (``test_required_slots_rederives_strict_obligation_after_card_erasure``).

    Today the only rate-axis container is Tessera, whose checkpoints declare
    ``quantization_config.quant_method: "tessera"``.  A future container with
    a continuous rung axis adds itself here, in the commit that declares its
    lane.
    """
    build = card.get("build")
    if isinstance(build, Mapping):
        for key in ("quant_method", "export_container"):
            if str(build.get(key) or "").strip().lower() == "tessera":
                return True
    if model_dir is None:
        return False
    try:
        config = json.loads(
            (Path(model_dir) / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    quant = config.get("quantization_config") if isinstance(
        config, Mapping) else None
    if isinstance(quant, Mapping):
        return str(quant.get("quant_method") or "").strip().lower() == "tessera"
    return False


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _close(a: float, b: float, *, rel: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)


def _replay_byte_match(
    slot: str,
    match: Any,
) -> tuple[list[str], bool]:
    """Recompute the byte match from the integers the block carries.

    "Byte-matched" is checked here, never read: a boolean the producer wrote
    is exactly the kind of claim principle 14 refuses to consume.  Everything
    below is exact :class:`~fractions.Fraction` arithmetic on integer bit
    totals, and the carried ``byte_matched`` flag has to agree with it.
    """
    problems: list[str] = []
    if not isinstance(match, Mapping):
        return [f"{slot}: the control block carries no byte match"], False
    candidate = _positive_int(match.get("candidate_bits"))
    control = _positive_int(match.get("control_bits"))
    params = _positive_int(match.get("varying_params"))
    if candidate is None or control is None or params is None:
        return [
            f"{slot}: byte match needs positive integer candidate_bits, "
            f"control_bits and varying_params; got "
            f"{match.get('candidate_bits')!r}, {match.get('control_bits')!r}, "
            f"{match.get('varying_params')!r}"
        ], False

    tolerance = match.get("max_relative_slack")
    if (
        not isinstance(tolerance, (list, tuple))
        or len(tolerance) != 2
        or any(_positive_int(part) is None for part in tolerance)
    ):
        return [
            f"{slot}: byte match declares no [numerator, denominator] "
            f"max_relative_slack; got {tolerance!r}"
        ], False
    tolerance = Fraction(int(tolerance[0]), int(tolerance[1]))
    if tolerance > MAX_CONTROL_RELATIVE_SLACK:
        problems.append(
            f"{slot}: the control widened its own tolerance to "
            f"{float(tolerance) * 100:.4f}%, over the "
            f"{float(MAX_CONTROL_RELATIVE_SLACK) * 100:.4f}% a control may be. "
            "Two arms that differ by more than that are two products, not an "
            "arm and its control."
        )
        tolerance = MAX_CONTROL_RELATIVE_SLACK

    slack_bits = control - candidate
    relative = Fraction(abs(slack_bits), candidate)
    matched = relative <= tolerance
    carried = match.get("byte_matched")
    if carried is not None and bool(carried) is not matched:
        problems.append(
            f"{slot}: the block claims byte_matched={carried!r} but its own "
            f"integers replay to {matched} "
            f"({int(slack_bits)} bits apart, {float(relative) * 1e6:.1f} ppm)"
        )
    carried_slack = match.get("slack_bits")
    if carried_slack is not None and (
        isinstance(carried_slack, bool)
        or not isinstance(carried_slack, int)
        or carried_slack != slack_bits
    ):
        problems.append(
            f"{slot}: slack_bits={carried_slack!r} but control_bits - "
            f"candidate_bits = {int(slack_bits)}"
        )
    for key, bits in (("candidate_bpp", candidate), ("control_bpp", control)):
        reported = _finite_float(match.get(key))
        if reported is not None and not _close(reported, bits / params):
            problems.append(
                f"{slot}: {key}={reported!r} but {bits} bits over {params} "
                f"parameters is {bits / params!r}"
            )
    if not matched:
        problems.append(
            f"{slot}: the arms are NOT byte-matched — the control weighs "
            f"{control} bits against the candidate's {candidate}, "
            f"{float(relative) * 100:.4f}% apart, over the "
            f"{float(tolerance) * 100:.4f}% a control may be. The "
            f"{'control' if slack_bits > 0 else 'candidate'} arm is the fatter "
            "one, so the comparison would price those bytes as quality."
        )
    return problems, matched


def _replay_control_arms(
    slot: str,
    record: Mapping[str, Any],
    verdict: Mapping[str, Any],
    *,
    card: Mapping[str, Any],
) -> list[str]:
    """Bind both arms to the serving metric, structurally.

    The candidate arm is not accepted as a number at all: it must BE the
    card's own ``gold.kl``, which is already gated to exact full-vocab
    KL-vs-BF16 with ``score_positions=all`` on a no-spec-decode serve.  So a
    last-token hook screen or a weight-space error cannot reach this slot
    through the candidate leg, and a block measured on some other allocation
    cannot be pasted onto this artifact -- its candidate KL would not be this
    card's gold number.

    The control arm carries its own gold-shaped record and is replayed through
    the same :func:`_verify_gold_record`, then held to the candidate's
    measurement contract key by key.
    """
    problems: list[str] = []
    metrics = record.get("metrics")
    key = (metrics or {}).get("gold_metric_key") if isinstance(
        metrics, Mapping) else None
    if key not in UNIFORM_CONTROL_METRIC_KEYS:
        return [
            f"{slot}: metrics.gold_metric_key={key!r} — the record must name "
            f"which of {list(UNIFORM_CONTROL_METRIC_KEYS)} both arms are "
            "compared on, so the comparison is not a choice made after the "
            "fact"
        ]

    gold = ((card.get("slots") or {}).get("gold.kl"))
    if not isinstance(gold, Mapping):
        return [
            f"{slot}: the card carries no gold.kl record, so the candidate "
            "arm has nothing to be. The candidate leg of this comparison is "
            "the artifact's own gold KL, not a number handed to the card."
        ]
    gold_metrics = gold.get("metrics")
    if not isinstance(gold_metrics, Mapping):
        return [f"{slot}: the card's gold.kl record carries no metrics"]
    gold_value = _finite_float(gold_metrics.get(key))
    candidate = _finite_float(verdict.get("candidate"))
    if gold_value is None:
        problems.append(
            f"{slot}: the card's gold.kl carries no finite {key}, so the "
            "candidate arm cannot be bound to it"
        )
    elif candidate is None or not _close(candidate, gold_value):
        problems.append(
            f"{slot}: the verdict's candidate {verdict.get('candidate')!r} is "
            f"not the card's own gold.kl {key}={gold_value!r}. The candidate "
            "arm must be the artifact's served gold KL — a comparison against "
            "some other measurement of some other artifact is not this "
            "artifact's control."
        )

    arm = record.get("control_arm")
    if not isinstance(arm, Mapping):
        return problems + [
            f"{slot}: carries no control_arm record. The control's KL needs "
            "the same evidence the candidate's does: it is a second served "
            "checkpoint, not a number."
        ]
    problems.extend(
        f"{slot}: control arm: {item.split(': ', 1)[-1]}"
        for item in _verify_gold_record(
            "gold.kl", arm, model_dir=None, require_current_artifact_path=False,
        )
    )
    spec = arm.get("spec_decode_detected")
    if spec is None:
        problems.append(
            f"{slot}: control arm: spec_decode_detected is unknown — a gold "
            "number measured against a spec-decode serve is the draft model's "
            "NLL (§7.5)"
        )
    elif spec:
        problems.append(
            f"{slot}: control arm: spec_decode_detected is TRUE — that is "
            "draft-model NLL, not the control's"
        )

    arm_metrics = arm.get("metrics")
    arm_metrics = arm_metrics if isinstance(arm_metrics, Mapping) else {}
    arm_value = _finite_float(arm_metrics.get(key))
    control_kl = _finite_float(verdict.get("control"))
    if arm_value is None:
        problems.append(
            f"{slot}: control arm: carries no finite {key}, the metric this "
            "verdict is stated on"
        )
    elif control_kl is None or not _close(arm_value, control_kl):
        problems.append(
            f"{slot}: the verdict's control {verdict.get('control')!r} is not "
            f"the control arm's own {key}={arm_value!r}"
        )

    arm_sha = arm.get("model_sha")
    if not isinstance(arm_sha, str) or re.fullmatch(
        r"[0-9a-f]{64}", arm_sha
    ) is None:
        problems.append(
            f"{slot}: control arm: model_sha is not one lowercase SHA-256 — "
            "the control is a checkpoint that was built and served, and it is "
            "identified like one"
        )
    elif arm_sha == card.get("model_sha"):
        problems.append(
            f"{slot}: control arm: model_sha equals the candidate's — an arm "
            "compared against itself is not a control"
        )

    gold_tool = gold.get("tool")
    if arm.get("tool") != gold_tool:
        problems.append(
            f"{slot}: control arm: measured by {arm.get('tool')!r} but the "
            f"candidate by {gold_tool!r} — two evaluators are two metrics"
        )
    for contract_key in UNIFORM_CONTROL_CONTRACT_KEYS:
        if contract_key not in gold_metrics:
            continue
        want = gold_metrics.get(contract_key)
        got = arm_metrics.get(contract_key)
        if got != want:
            problems.append(
                f"{slot}: control arm: {contract_key}={got!r} but the "
                f"candidate's gold.kl says {want!r} — the arms did not run "
                "the same measurement contract"
            )
    return problems


def _verify_uniform_control_override(
    slot: str,
    override: Any,
    *,
    card: Mapping[str, Any],
    model_dir: str | os.PathLike | None,
    ratio: float | None,
) -> tuple[bool, list[str]]:
    """Is this override a deliberate act, bound to this artifact and verdict?

    Same bar as ``publish_artifact.py --force-unverified``: the artifact
    directory's basename has to have been re-typed, and the act is stamped
    into the bytes that get uploaded.  Two bindings on top of it, because this
    override outlives the command that made it: it names the artifact's
    ``model_sha`` and the exact ratio it forgives, so a re-export or a
    re-measurement voids it rather than inheriting it.
    """
    if not isinstance(override, Mapping):
        return False, [
            f"{slot}: override is not an object ({type(override).__name__})"
        ]
    problems: list[str] = []
    if override.get("schema") != UNIFORM_CONTROL_OVERRIDE_SCHEMA:
        return False, [
            f"{slot}: override schema {override.get('schema')!r} != "
            f"{UNIFORM_CONTROL_OVERRIDE_SCHEMA!r}"
        ]
    for key in ("reason", "authorized_by", "stamped_at"):
        value = override.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{slot}: override {key} is empty")
    sha = override.get("model_sha")
    if sha != card.get("model_sha"):
        problems.append(
            f"{slot}: override names model_sha {str(sha)[:12]} but this card "
            f"is {str(card.get('model_sha'))[:12]} — an override does not "
            "survive a re-export"
        )
    typed = override.get("confirmed_artifact_name")
    if not isinstance(typed, str) or not typed.strip():
        problems.append(
            f"{slot}: override records no re-typed artifact directory name"
        )
    # The re-typed name is checked against the directory ONCE, when it is
    # typed (``shipcard_cli override-control``), exactly as ``--force-unverified``
    # checks its re-type at publish time.  It is not re-checked here: ``verify``
    # runs on the publisher's snapshot copy under a randomised basename and on
    # downloaded artifacts wherever they land, and a stamp that stops verifying
    # when the directory moves is a stamp bound to a path, not to the card.
    # What binds the override to THIS card is below: model_sha and the ratio.
    forgiven = _finite_float(override.get("candidate_over_control"))
    if forgiven is None or ratio is None or not _close(forgiven, ratio):
        problems.append(
            f"{slot}: override forgives a "
            f"{override.get('candidate_over_control')!r}x loss but the record "
            f"states {ratio!r}x — an override does not survive a "
            "re-measurement"
        )
    return (not problems), problems


def _verify_uniform_control_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    card: Mapping[str, Any],
    model_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Refuse an allocation that lost to spending the same bytes uniformly."""
    problems: list[str] = []
    block = record.get("uniform_control")
    if not isinstance(block, Mapping):
        return [
            f"{slot}: carries no uniform_control block "
            f"(tessera.control.control_block(), schema "
            f"{UNIFORM_CONTROL_SCHEMA!r}), so there is nothing to replay"
        ]
    if block.get("schema") != UNIFORM_CONTROL_SCHEMA:
        return [
            f"{slot}: block schema {block.get('schema')!r} != "
            f"{UNIFORM_CONTROL_SCHEMA!r}"
        ]

    control = block.get("control")
    control = control if isinstance(control, Mapping) else {}
    byte_problems, byte_matched = _replay_byte_match(slot, control.get("match"))
    problems.extend(byte_problems)
    dominated = control.get("dominated_by")
    if dominated is not None:
        problems.append(
            f"{slot}: the control sits on R{control.get('q256')} while "
            f"R{dominated} weighs no more on this plan's shapes — a "
            "handicapped uniform arm. Beating a rung the allocator should not "
            "have been offered either is not beating the control "
            "(tessera#43)."
        )

    verdict = block.get("verdict")
    if not isinstance(verdict, Mapping):
        return problems + [f"{slot}: the block carries no verdict"]
    if verdict.get("metric") != "kl_vs_bf16":
        problems.append(
            f"{slot}: verdict metric {verdict.get('metric')!r} — this gate is "
            "on the serving metric, exact full-vocab vLLM KL-vs-BF16 "
            "('kl_vs_bf16'), and on nothing else"
        )
    if verdict.get("measured") is not True:
        # Deliberately NOT overridable, and deliberately not silence: this is
        # the `{"measured": false}` variant `control_block()` emits for a
        # control that was built and priced but never served.  A built control
        # is not a passed gate.
        return problems + [
            f"{slot}: the uniform control was built and priced but never "
            f"SERVED ({verdict.get('detail') or 'no detail'}). Missing is not "
            "passing: serve the control checkpoint on the same corpus that "
            "produced this card's gold.kl and fill the verdict, or the "
            "allocation's claim to be worth allocating is untested."
        ]

    candidate = _finite_float(verdict.get("candidate"))
    control_kl = _finite_float(verdict.get("control"))
    if candidate is None or candidate < 0 or control_kl is None or (
        control_kl <= 0
    ):
        return problems + [
            f"{slot}: verdict needs a finite non-negative candidate KL and a "
            f"finite positive control KL; got {verdict.get('candidate')!r} and "
            f"{verdict.get('control')!r}"
        ]
    ratio = candidate / control_kl
    beat = candidate < control_kl
    reported_ratio = _finite_float(verdict.get("candidate_over_control"))
    if reported_ratio is None or not _close(reported_ratio, ratio):
        problems.append(
            f"{slot}: candidate_over_control="
            f"{verdict.get('candidate_over_control')!r} but "
            f"{candidate!r} / {control_kl!r} = {ratio!r}"
        )
    if verdict.get("beat_control") is not None and bool(
        verdict.get("beat_control")
    ) is not beat:
        problems.append(
            f"{slot}: the block claims beat_control="
            f"{verdict.get('beat_control')!r} but its own two KLs replay to "
            f"{beat}"
        )
    if record.get("passed") is not beat:
        # The flag is not the gate -- the two KLs are -- but a record whose
        # flag disagrees with its own numbers is lying, and saying so is
        # cheaper than letting a hand-edited `passed: true` look ordinary.
        problems.append(
            f"{slot}: record says passed={record.get('passed')!r} while its "
            f"own KLs say the candidate {'beat' if beat else 'did NOT beat'} "
            "the control"
        )

    problems.extend(_replay_control_arms(slot, record, verdict, card=card))

    if beat:
        return problems

    losing = (
        f"{slot}: the allocation LOST to its byte-matched uniform control — "
        f"{candidate:.6g} against {control_kl:.6g} KL-vs-BF16 at matched "
        f"bytes ({ratio:.4g}x worse) on {control.get('grid')} "
        f"R{control.get('q256')}. An allocation that cannot beat spending the "
        "same bytes at one rung has not earned its allocation and does not "
        "ship (prismaquant#117, tessera#1)."
    )
    override = record.get("override")
    if override is None:
        return problems + [losing]
    if problems or not byte_matched:
        # An override forgives a MEASURED loss on a REAL control.  It does not
        # forgive a control that was not byte-matched or a record that does
        # not replay -- those are not results to accept, they are results that
        # do not exist yet.  `--force-unverified` is the (stamped) instrument
        # for those, and it is meant to feel heavier than this.
        return problems + [
            losing + " The override on this record does not apply: an "
            "override forgives a measured loss on a byte-matched control, and "
            "this record has unresolved problems above."
        ]
    ok, override_problems = _verify_uniform_control_override(
        slot, override, card=card, model_dir=model_dir, ratio=ratio,
    )
    if not ok:
        return problems + override_problems + [losing]
    return problems


def uniform_control_summary(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """The control verdict as data, to print beside a published bpp claim.

    Principle 12: every published size or quality claim carries the honesty
    that qualifies it.  A bpp on a rate-axis artifact is a claim that those
    bytes were spent well, so the verdict of the arm that tests it travels
    with the number.  ``applicable: False`` is a stated scope, not silence:
    a format-menu artifact has no rung axis, `tessera.control.uniform_control`
    refuses to build a control for it, and requiring one would be a gate no
    correct artifact can pass.
    """
    record = (card.get("slots") or {}).get(UNIFORM_CONTROL_SLOT)
    rate_axis = _is_rate_axis_artifact(card, model_dir=model_dir)
    summary: dict[str, Any] = {
        "applicable": bool(rate_axis or isinstance(record, Mapping)),
        "rate_axis_artifact": bool(rate_axis),
        "filled": isinstance(record, Mapping),
        "measured": None,
        "beat_control": None,
        "candidate_over_control": None,
        "control": None,
        "overridden": False,
        "detail": "",
    }
    if not summary["applicable"]:
        summary["detail"] = (
            "not applicable: this artifact's menu has no rate axis, so there "
            "is no single uniform rung to spend the same bytes at"
        )
        return summary
    if not isinstance(record, Mapping):
        summary["detail"] = (
            "NOT MEASURED: this is a rate-axis artifact and no byte-matched "
            "uniform control has been served against it"
        )
        return summary
    block = record.get("uniform_control")
    block = block if isinstance(block, Mapping) else {}
    control = block.get("control")
    control = control if isinstance(control, Mapping) else {}
    match = control.get("match")
    match = match if isinstance(match, Mapping) else {}
    verdict = block.get("verdict")
    verdict = verdict if isinstance(verdict, Mapping) else {}
    summary["measured"] = bool(verdict.get("measured"))
    summary["overridden"] = isinstance(record.get("override"), Mapping)
    summary["control"] = {
        "grid": control.get("grid"),
        "q256": control.get("q256"),
        "candidate_bpp": match.get("candidate_bpp"),
        "control_bpp": match.get("control_bpp"),
        "relative_slack_ppm": match.get("relative_slack_ppm"),
        "byte_matched": match.get("byte_matched"),
    }
    if not summary["measured"]:
        summary["detail"] = (
            "NOT MEASURED: the control was built and priced; neither arm was "
            "served"
        )
        return summary
    summary["beat_control"] = verdict.get("beat_control")
    summary["candidate_over_control"] = verdict.get("candidate_over_control")
    summary["detail"] = str(verdict.get("detail") or "")
    return summary


def make_uniform_control_record(
    *,
    tool: str,
    model_sha: str | None,
    control_block: Mapping[str, Any],
    control_arm: Mapping[str, Any],
    gold_metric_key: str,
    git_commit: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """One byte-matched-uniform-control verdict block.

    ``control_block`` is stored verbatim -- it is Tessera's accountant's
    output and this module does not import Tessera -- and every number in it
    is replayed by :func:`verify`.  ``control_arm`` is the gold-shaped record
    of the CONTROL checkpoint's own served KL, produced by the same tool that
    filled ``gold.kl`` on this artifact.
    """
    if gold_metric_key not in UNIFORM_CONTROL_METRIC_KEYS:
        raise ValueError(
            f"gold_metric_key must be one of {list(UNIFORM_CONTROL_METRIC_KEYS)}"
        )
    block = json.loads(json.dumps(dict(control_block)))
    if block.get("schema") != UNIFORM_CONTROL_SCHEMA:
        raise ValueError(
            f"control block schema {block.get('schema')!r} != "
            f"{UNIFORM_CONTROL_SCHEMA!r}"
        )
    verdict = block.get("verdict") or {}
    beat = bool(verdict.get("measured")) and bool(verdict.get("beat_control"))
    if detail is None:
        detail = str(
            verdict.get("detail")
            or "the control was built and priced; neither arm was served"
        )
    return make_record(
        slot=UNIFORM_CONTROL_SLOT,
        tool=tool,
        passed=beat,
        model_sha=model_sha,
        metrics={"gold_metric_key": gold_metric_key},
        detail=detail,
        git_commit=git_commit,
        extra={
            "uniform_control": block,
            "control_arm": json.loads(json.dumps(dict(control_arm))),
        },
    )


def record_uniform_control_override(
    card: dict[str, Any],
    *,
    reason: str,
    authorized_by: str,
    confirmed_artifact_name: str,
) -> dict[str, Any]:
    """Stamp a deliberate, bound override onto an already-filled record.

    The confirmation itself (re-typing the artifact directory's basename) is
    the CLI's job, exactly as it is ``publish_artifact.py``'s; what is typed
    is recorded here so ``verify`` can check it against the directory.
    """
    record = (card.get("slots") or {}).get(UNIFORM_CONTROL_SLOT)
    if not isinstance(record, Mapping):
        raise KeyError(
            "this card carries no uniform_control record to override; fill "
            "the slot first"
        )
    for name, value in (
        ("reason", reason),
        ("authorized_by", authorized_by),
        ("confirmed_artifact_name", confirmed_artifact_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"override {name} must be a non-empty string")
    verdict = ((record.get("uniform_control") or {}).get("verdict") or {})
    override = {
        "schema": UNIFORM_CONTROL_OVERRIDE_SCHEMA,
        "reason": reason,
        "authorized_by": authorized_by,
        "confirmed_artifact_name": confirmed_artifact_name,
        "model_sha": card.get("model_sha"),
        "candidate_over_control": verdict.get("candidate_over_control"),
        "stamped_at": _now(),
    }
    record = dict(record)
    record["override"] = override
    card["slots"][UNIFORM_CONTROL_SLOT] = record
    card["uniform_control_override"] = True
    history = list(card.get("uniform_control_override_history") or [])
    history.append(dict(override))
    card["uniform_control_override_history"] = history
    card["updated"] = _now()
    return card


def _verify_ship_gate_record(
    slot: str,
    record: Mapping[str, Any],
) -> list[str]:
    """Replay the fixed catastrophic-quality thresholds, the check ledger
    with its token evidence, and the endpoint binding.

    The binding half is presence-and-shape, stated honestly: an offline
    `verify` has no live session to compare against, so it refuses a record
    that names no server, no served model and no artifact path -- but a
    well-formed binding to the WRONG server still passes here. Catching that
    needs the nonce-bound live-session check (the retired CB eager driver
    did exactly that before teardown); see the `ship_gate` paragraph in
    §7.2 of docs/ARCHITECTURE.md.
    """
    problems: list[str] = []
    if record.get("tool") != "validate_quantized_model.py":
        problems.append(f"{slot}: not filled by validate_quantized_model.py")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        problems.append(f"{slot}: missing full producer git commit")
    endpoint = record.get("base_url")
    if (
        not isinstance(endpoint, str)
        or not endpoint.startswith(("http://", "https://"))
    ):
        problems.append(f"{slot}: missing endpoint binding (base_url)")
    served = record.get("served_model_name")
    if not isinstance(served, str) or not served.strip():
        problems.append(
            f"{slot}: missing served-model binding (served_model_name)")
    source = record.get("model_sha_source")
    if not isinstance(source, str) or not source.strip():
        problems.append(
            f"{slot}: missing artifact-path binding (model_sha_source)")
    metrics = record.get("metrics")
    thresholds = record.get("thresholds")
    expected_checks = {
        "serve_ready", "generation_sanity", "perplexity", "mtp_acceptance"
    }
    if not isinstance(metrics, Mapping) or set(metrics) != expected_checks:
        return problems + [f"{slot}: validation check ledger is incomplete"]
    for name, row in metrics.items():
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            problems.append(f"{slot}: check {name} is not a structured pass")
    expected_thresholds = {
        "max_ppl": 25.0,
        "max_mean_nll": 3.0,
        "max_p99_nll": 6.0,
        "min_gen_len": 30,
        "min_mtp_accept_p0": 0.60,
    }
    if not isinstance(thresholds, Mapping):
        problems.append(f"{slot}: missing threshold contract")
    else:
        for key, expected in expected_thresholds.items():
            value = thresholds.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != float(expected)
            ):
                problems.append(
                    f"{slot}: threshold {key}={value!r}, expected {expected!r}"
                )
    perplexity = metrics.get("perplexity")
    if isinstance(perplexity, Mapping):
        numeric = {
            "perplexity": ("max_ppl", 25.0),
            "mean_nll_per_tok": ("max_mean_nll", 3.0),
            "max_nll_per_tok": ("max_p99_nll", 6.0),
        }
        for key, (_threshold_key, limit) in numeric.items():
            value = perplexity.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                or float(value) > limit
            ):
                problems.append(
                    f"{slot}: perplexity metric {key} does not clear {limit}"
                )
        tokens = perplexity.get("n_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            problems.append(f"{slot}: perplexity check scored no tokens")
        if perplexity.get("spec_decode_detected") is True:
            problems.append(f"{slot}: perplexity ran under speculative decode")
        elif perplexity.get("spec_decode_detected") is not False:
            problems.append(
                f"{slot}: perplexity speculative-decode state is unknown"
            )
    else:
        problems.append(f"{slot}: missing structured perplexity evidence")
    return problems


def unfilled_slots(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> list[str]:
    slots = card.get("slots") or {}
    return [
        slot
        for slot in required_slots(card, model_dir=model_dir)
        if not slots.get(slot)
    ]


def required_slots(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> tuple[str, ...]:
    """Return every slot default verification must replay.

    Container-required slots, then whatever the card's own ``lane`` declares,
    then recognized optional claims that are present and non-null.

    Until 2026-09-02 the retired Gridbook codebook lane added three lane-scoped
    slots here, keyed off the artifact's own ``config.json``/``quant_config.json``;
    they are in ``archive/gridbook_lane_2026-09-02/`` with the lane.  The lane
    hook came back on 2026-09-03 in a form that reads the lane's OWN
    declaration instead of branching per lane here: the Tessera lane declares
    ``route.census`` and that slot is now required of a Tessera card.

    One live obligation is derived from the artifact rather than from a lane: a
    **rate-axis** artifact must carry the byte-matched uniform control's
    verdict.  It is re-derived from the artifact's own ``config.json`` as well
    as from the card, so nulling the claim or emptying the build block cannot
    erase the obligation -- and "no control was ever served" therefore reads as
    ``UNFILLED`` rather than as silence (#121).
    """
    required: list[str] = list(REQUIRED_SLOTS)
    # UNION, never replacement. A lane adds what its own gates[] declares and
    # can subtract nothing: the GGUF lane declares no `native_export.graph`
    # gate and is still required to close that slot, so lane-derivation cannot
    # be used to shrink a bar.
    required.extend(lane_gate_slots(card.get("lane")))
    if _is_rate_axis_artifact(card, model_dir=model_dir):
        required.append(UNIFORM_CONTROL_SLOT)
    slots = card.get("slots")
    if isinstance(slots, Mapping):
        required.extend(
            slot for slot in OPTIONAL_SLOTS if slots.get(slot) is not None
        )
    return tuple(dict.fromkeys(required))


# ---------------------------------------------------------------------------
# Build-lane fact collection (used by export_native_compressed)
# ---------------------------------------------------------------------------
def kv_shared_fisher_echo(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Echo the KV-cotangent / shared-Fisher flag state (D24 caveat).

    An allocation probed with `PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1` (or with the
    cotangent path switched off) rode an under-counted `k_proj`/`v_proj`
    `h_trace`. That is currently visible only in a probe log; putting it on the
    ship record makes it visible on the artifact.
    """
    env = os.environ if env is None else env
    allow = env.get("PRISMAQUANT_ALLOW_KV_SHARED_FISHER", "0")
    cotangent = env.get("PRISMAQUANT_KV_COTANGENT", "1")
    allow_on = allow not in ("", "0", "false", "False")
    cotangent_on = cotangent not in ("", "0", "false", "False")
    return {
        "PRISMAQUANT_ALLOW_KV_SHARED_FISHER": allow,
        "PRISMAQUANT_KV_COTANGENT": cotangent,
        "kv_cotangent_path_enabled": cotangent_on,
        "unvalidated_kv_fisher_correction": bool(allow_on or not cotangent_on),
        "caveat": (
            "D24: the KV-cotangent path has never been run on a real "
            "num_kv_shared_layers>0 checkpoint; ALLOW_KV_SHARED_FISHER=1 or "
            "KV_COTANGENT=0 means this allocation rode an under-counted "
            "k_proj/v_proj h_trace"
        ),
    }


def _verify_build_block(card: Mapping[str, Any]) -> list[str]:
    """Replay the build-lane forensic block the exporter stamps on the card.

    `export_native_compressed._write_shipcard` stamps ten keys and `verify`
    read one (`achieved_bpp`, refused above on DISAGREE) — the rest was
    provenance theater inside the 256 KiB byte reservation (#158). So a card
    could name any `source_model`, point at a missing recipe, mismatch its
    own histogram, or carry an unvalidated Fisher correction, and still
    verify.

    What is replayed here, and why each half is shaped this way:

    * The `kv_shared_fisher` policy bit is a gate: the echo's own caveat says
      the allocation rode an under-counted `k_proj`/`v_proj` `h_trace`, and
      the probe lane already refuses to PRODUCE such an allocation by default
      (`incremental_probe.kv_shared_fisher_block_reason`) — reaching export
      takes an explicit override, and publication past this gate takes the
      stamped `--force-unverified` hatch. Default-path cards echo `False`
      and are silent here.
    * The forensic hashes and counters are shape-replayed against what the
      producer can stamp (`file_sha256` emits 64 hex chars or None; the
      assignment digests are 16 hex chars or None; the histogram is a
      `Counter` rendering with positive counts), so a fabricated value is
      refused. Their preimages are build-machine-local (the recipe file is
      not shipped), so no offline cross-check exists for them — and none is
      pretended.
    * `source_model` / `layer_config` path strings are audit trail, not gate
      input: they name build-machine paths no verifier can resolve, and the
      same strings already travel in `mixed_native_manifest.json`. They are
      kept, labeled here as unread-by-design, rather than dropped.

    Every check fires only on a key the producer stamps, so a card written
    before a key existed is left alone — the same tolerance the
    `achieved_bpp` cross-check practices.
    """
    problems: list[str] = []
    build = card.get("build")
    if not isinstance(build, Mapping):
        return problems
    fisher = build.get("kv_shared_fisher")
    if isinstance(fisher, Mapping):
        flag = fisher.get("unvalidated_kv_fisher_correction")
        if flag is True:
            problems.append(
                "build.kv_shared_fisher reports "
                "unvalidated_kv_fisher_correction=true: this allocation rode "
                "an under-counted k_proj/v_proj h_trace "
                f"({fisher.get('caveat') or 'D24'}) — re-probe with the "
                "KV-cotangent path enabled and without "
                "PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1, or ship it stamped "
                "--force-unverified"
            )
        elif flag is not False:
            problems.append(
                "build.kv_shared_fisher carries no boolean "
                f"unvalidated_kv_fisher_correction flag: {flag!r}"
            )
    for key, width in (
        ("layer_config_sha", 64),
        ("assignment_hash", 16),
        ("config_assignment_hash", 16),
    ):
        value = build.get(key)
        if value is not None and (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{%d}" % width, value) is None
        ):
            problems.append(
                f"build.{key}={value!r} is not a {width}-char hex digest — "
                "the producer stamps a file hash or None, nothing else"
            )
    entries = build.get("n_assignment_entries")
    if entries is not None and (
        isinstance(entries, bool)
        or not isinstance(entries, int)
        or entries < 0
    ):
        problems.append(
            f"build.n_assignment_entries={entries!r} is not a non-negative "
            "integer"
        )
    histogram = build.get("format_histogram")
    if histogram is not None:
        if not isinstance(histogram, Mapping):
            problems.append(
                "build.format_histogram is not an object "
                f"({type(histogram).__name__})"
            )
        else:
            for name, count in histogram.items():
                if (
                    not isinstance(name, str)
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    problems.append(
                        "build.format_histogram carries a non-positive "
                        f"render count: {name!r}={count!r}"
                    )
                    break
    traffic = build.get("read_gb_per_token")
    if traffic is not None:
        value = traffic.get("value") if isinstance(traffic, Mapping) else None
        if not isinstance(traffic, Mapping) or (
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            )
        ):
            problems.append(
                "build.read_gb_per_token carries no finite non-negative "
                f"value: {traffic!r}"
            )
    for key in ("render_levers", "git"):
        value = build.get(key)
        if value is not None and not isinstance(value, Mapping):
            problems.append(
                f"build.{key} is not an object ({type(value).__name__})"
            )
    return problems


# A claimed bpp may legitimately sit a few percent above the priced floor
# below (the floor omits whatever units carry no per-unit price, and those
# only ever ADD bytes), and label scope differs by a pin or two. 10% is far
# wider than any such drift and far narrower than the 57% error this exists
# to catch, so a trip is a defect and never a convention argument.
RECIPE_BPP_CROSS_CHECK_TOLERANCE = 0.10

# The floor is only a useful bound when it covers most of the recipe. Only CB
# units declare a per-unit serialized price, so a recipe that mixes CB with
# plain NVFP4 / FP8 / passthrough prices only part of itself and its true rate
# can sit far from the covered blend in EITHER direction. Measured on the real
# DSv4 recipes: allocation-112p69-ldlq prices 99.4% of units and lands 1.5%
# from its claim, while allocation-112p69-raw prices 70% and lands 30.6% away
# while still being truthful. Below this coverage a mismatch is reported as
# inconclusive rather than as a defect: refusing on a bound you know is loose
# is a worse failure than not refusing, because it teaches the operator to
# reach for --force-unverified.
RECIPE_BPP_CROSS_CHECK_MIN_COVERAGE = 0.95


def recipe_priced_bpp(
    layer_config_path: str | os.PathLike | None,
) -> dict[str, Any]:
    """Price the recipe from the per-unit bytes the recipe itself declares.

    This is deliberately NOT a second opinion about accounting convention. It
    sums ``cb_serialized_identity.tensor_payload_bytes`` and ``.params`` over
    the units that carry them, so numerator and denominator come from the same
    entries and the result is scope-matched *by construction* — no probe, no
    source manifest, no sidecar or header estimate.

    The result is a **lower bound** on the recipe's true rate: units without a
    per-unit price (FP8_SOURCE passthrough Linears, for instance) are excluded
    from both sums, and they can only add bytes. ``coverage_units`` reports how
    much of the recipe was priced so a caller can judge the bound's tightness.

    Returns ``value=None`` with a ``reason`` when nothing is priceable, which
    is the normal case for a non-CB recipe. That is a "not applicable", not a
    failure.
    """
    out: dict[str, Any] = {"value": None, "reason": None}
    if not layer_config_path:
        out["reason"] = "no recipe path"
        return out
    try:
        from prismaquant.layer_config import is_layer_config_meta_key
        payload = json.loads(Path(layer_config_path).read_text())
    except Exception as exc:
        out["reason"] = f"recipe unreadable: {exc!r}"
        return out
    if not isinstance(payload, Mapping):
        out["reason"] = "recipe is not a mapping"
        return out

    # Two recipe layouts carry these prices, and both are production. The body
    # allocator writes per-unit mappings with `cb_serialized_identity` inside
    # each record; the DSpark CB sidecar builder writes a flat
    # `name -> "FORMAT"` map with the identities collected under
    # `__prismaquant__.cb_serialized_identities`. Reading only the first shape
    # silently downgraded the whole sidecar lane to "not applicable", which
    # turns the gate off on an artifact that is fully priceable.
    meta_identities: Mapping[str, Any] = {}
    try:
        from prismaquant.layer_config import read_layer_config_metadata
        raw = (read_layer_config_metadata(layer_config_path) or {}).get(
            "cb_serialized_identities"
        )
        if isinstance(raw, Mapping):
            meta_identities = raw
    except Exception:
        meta_identities = {}

    total = priced = 0
    total_bytes = total_params = 0
    for name, record in payload.items():
        if is_layer_config_meta_key(name):
            continue
        total += 1
        identity = None
        if isinstance(record, Mapping):
            identity = record.get("cb_serialized_identity")
        if not identity:
            identity = meta_identities.get(name)
        if not identity:
            continue
        try:
            if isinstance(identity, str):
                identity = json.loads(identity)
            nbytes = int(identity["tensor_payload_bytes"])
            nparams = int(identity["params"])
        except Exception:
            continue
        if nparams <= 0:
            continue
        priced += 1
        total_bytes += nbytes
        total_params += nparams

    if not priced or total_params <= 0:
        out["reason"] = (
            "no unit carries a per-unit serialized price "
            f"(0 of {total} recipe entries)"
        )
        return out
    out.update(
        value=8.0 * total_bytes / total_params,
        reason=None,
        priced_units=priced,
        total_units=total,
        coverage_units=priced / total if total else None,
        tensor_payload_bytes=total_bytes,
        params=total_params,
        note=(
            "lower bound: summed over the units that declare a per-unit "
            "serialized price; unpriced units only add bytes"
        ),
    )
    return out


def _allocator_achieved_bpp_claim(
    layer_config_path: str | os.PathLike | None,
) -> dict[str, Any]:
    """Best-effort achieved bpp, with its provenance named.

    The exporter is handed a recipe, not a bpp, so this reports someone else's
    number and *says whose* rather than recomputing an
    accounting-convention-sensitive one (CLAUDE.md principle 12: bpp labels are
    not comparable across eras).

    Prefer the recipe's OWN metadata over `pareto.knees.json`. The knee file is
    a different file describing the *surrogate* frontier, and under
    `SELECTION_MODE=validated-surrogate` it does not describe the recipe beside
    it: `select_validated_frontier` overwrites `layer_config.json` with the
    measured pick and stamps `selected_achieved_bits` into it. On Qwen3.8-27B
    arm B those disagreed by 1.25 bpp — the card claimed the surrogate knee's
    5.9994 for bytes that were the validated 4.7496 — which is a false public
    claim and would silently break any matched-bpp comparison built on it.

    Note that the same metadata's plain `achieved_bits` is *stale* on a
    validated recipe: the frontier carries the allocator's block forward
    verbatim (it owns the serving profile the exporter needs), so that field
    still describes the allocator's pre-selection run. Hence `selected_by`
    wins, and a recipe that announces a validated selection without a number
    reports nothing rather than falling through to a knee file that is
    describing some other point.
    """
    if not layer_config_path:
        return {"value": None, "source": None}
    try:
        from prismaquant.layer_config import read_layer_config_metadata
        meta = dict(read_layer_config_metadata(layer_config_path) or {})
    except Exception:
        meta = {}
    selected_by = meta.get("selected_by")
    if selected_by:
        value = meta.get("selected_achieved_bits")
        if value is None:
            return {
                "value": None,
                "source": f"layer_config.json:{selected_by} (no bpp stamped)",
            }
        return {
            "value": float(value),
            "source": f"layer_config.json:{selected_by}",
            "selected_label": meta.get("selected_label"),
            "scope": meta.get("achieved_bits_scope"),
            "note": (
                "the bpp of the assignment this recipe actually holds, stamped "
                "by the stage that selected it; it describes the recipe, not "
                "the exported bytes"
            ),
        }
    if meta.get("achieved_bits") is not None:
        return {
            "value": float(meta["achieved_bits"]),
            "source": "layer_config.json:achieved_bits",
            "target_bits": meta.get("target_bits"),
            "scope": meta.get("achieved_bits_scope"),
            "note": (
                "the allocator's achieved bpp for this recipe; it describes "
                "the recipe, not the exported bytes"
            ),
        }
    knees = Path(layer_config_path).parent / "pareto.knees.json"
    if not knees.is_file():
        return {"value": None, "source": None}
    try:
        payload = json.loads(knees.read_text())
        mode = payload.get("primary") or "log_error"
        entry = payload.get(mode) or {}
        value = entry.get("achieved_bits")
        if value is None:
            return {"value": None, "source": None}
        return {
            "value": float(value),
            "source": f"pareto.knees.json:{mode}",
            "target_bits": entry.get("target_bits"),
            "note": (
                "the allocator's achieved bpp for the knee it selected; it "
                "describes the recipe, not the exported bytes"
            ),
        }
    except Exception:
        return {"value": None, "source": None}


def allocator_achieved_bpp(
    layer_config_path: str | os.PathLike | None,
) -> dict[str, Any]:
    """The claimed achieved bpp, cross-checked against the recipe's own bytes.

    The claim is whatever upstream stamped (see
    :func:`_allocator_achieved_bpp_claim`), and it is reported unchanged. What
    is added here is a ``cross_check`` block, because this exact class of
    defect has now shipped twice:

    * **Qwen3.8-27B arm B** — the card claimed the surrogate knee's 5.9994 for
      bytes that were the validated 4.7496 (1.25 bpp wide), which is what the
      ``selected_by`` precedence above was written to fix.
    * **DSv4-Flash `artifact-aura-cb-112p69`** — the card claimed 4.3065, read
      from a Pareto point (``allocator_target_4p5000_achieved_4p3065``) that
      was internally consistent but described an assignment that was **never
      exported**. The recipe it shipped prices to 2.7385. That is 57% wide and
      no precedence rule catches it, because the stale number is not the wrong
      *field* — it is the right field describing the wrong *point*.

    Both are the same failure: the label describes a different assignment than
    the recipe holds. A false public bpp silently breaks every matched-bpp
    comparison built on it, so it is worth a gate rather than a convention.

    The check is a **floor test**, not an equality test. The priced value is a
    lower bound (see :func:`recipe_priced_bpp`), so a claim below it is
    impossible and a claim far above it is describing something else. It is
    advisory here — :func:`verify` is where it refuses — so that a gate bug can
    never strand a finished export at card-writing time.

    The floor only bounds the claim when it covers nearly the whole recipe, so
    a mismatch under ``RECIPE_BPP_CROSS_CHECK_MIN_COVERAGE`` is reported as
    ``inconclusive_low_coverage`` and does **not** refuse. This is not
    timidity: the real ``allocation-112p69-raw`` recipe prices 70% of its units
    at 2.109 bpp and truthfully claims 2.755, because its unpriced units are
    plain NVFP4 at 4.25 bpp. A bare 10% test would have refused that correct
    artifact at publication — and a gate that false-refuses teaches the
    operator to reach for ``--force-unverified``, which is the worse outcome.

    Both directions are gated, including undershoot. See the comment in the
    body: "below the priced blend is impossible" holds only when every unpriced
    unit is higher-rate than the blend, the sound bound needs *parameter*
    coverage, and a non-CB unit records no shape in the recipe to compute it
    from.
    """
    claim = _allocator_achieved_bpp_claim(layer_config_path)
    priced = recipe_priced_bpp(layer_config_path)
    claimed = claim.get("value")
    floor = priced.get("value")

    if floor is None:
        cross: dict[str, Any] = {
            "verdict": "not_applicable",
            "detail": priced.get("reason"),
        }
    elif claimed is None:
        cross = {
            "verdict": "no_claim",
            "recipe_priced_bpp": floor,
            "detail": "nothing to cross-check; the recipe declares no bpp",
        }
    else:
        rel = abs(float(claimed) - floor) / floor
        agree = rel <= RECIPE_BPP_CROSS_CHECK_TOLERANCE
        coverage = priced.get("coverage_units")
        # Both directions are gated on coverage, and the undershoot side is the
        # non-obvious half. It is tempting to call a claim below the priced
        # blend arithmetically impossible -- that holds only when every unpriced
        # unit is HIGHER-rate than the blend. On the real DSv4 recipes the
        # unpriced units are plain NVFP4 (4.25 bpp), so it does hold there; but
        # a recipe whose priced subset is FP8_CB-heavy (~8 bpp) with NVFP4
        # unpriced would have a true rate legitimately BELOW its own priced
        # blend. The sound bound would be blend x parameter-coverage, and
        # parameter coverage is not computable here: a non-CB unit records no
        # shape in the recipe, so only UNIT coverage is observable. Rather than
        # assert a bound the data cannot support, the floor is trusted in
        # either direction only when it is nearly complete.
        below = float(claimed) < floor
        loose = (
            coverage is None
            or coverage < RECIPE_BPP_CROSS_CHECK_MIN_COVERAGE
        )
        if agree:
            verdict = "agree"
        elif loose:
            verdict = "inconclusive_low_coverage"
        else:
            verdict = "DISAGREE"
        cross = {
            "verdict": verdict,
            "claimed_bpp": float(claimed),
            "claimed_source": claim.get("source"),
            "recipe_priced_bpp": floor,
            "relative_difference": rel,
            "claim_is_below_floor": below,
            "tolerance": RECIPE_BPP_CROSS_CHECK_TOLERANCE,
            "min_coverage_units": RECIPE_BPP_CROSS_CHECK_MIN_COVERAGE,
            "priced_units": priced.get("priced_units"),
            "total_units": priced.get("total_units"),
            "coverage_units": coverage,
            "tensor_payload_bytes": priced.get("tensor_payload_bytes"),
            "params": priced.get("params"),
        }
        if verdict == "inconclusive_low_coverage":
            cross["detail"] = (
                f"claimed {float(claimed):.4f} bpp from "
                f"{claim.get('source')!r} sits {rel:.1%} "
                f"{'below' if below else 'above'} the {floor:.4f} bpp priced "
                f"blend, but only {priced.get('priced_units')} of "
                f"{priced.get('total_units')} units carry a per-unit price "
                f"(coverage {coverage:.1%} < "
                f"{RECIPE_BPP_CROSS_CHECK_MIN_COVERAGE:.0%}), so the blend is "
                "too loose to indict the claim: the unpriced units are a large "
                "enough share to explain the gap on their own. Not refused."
            )
        elif not agree:
            cross["detail"] = (
                ("claim is BELOW an arithmetic lower bound: " if below else "")
                + (
                f"claimed {float(claimed):.4f} bpp from "
                f"{claim.get('source')!r}, but the recipe's own per-unit "
                f"serialized bytes price {priced.get('priced_units')} of "
                f"{priced.get('total_units')} units at "
                f"{priced.get('tensor_payload_bytes')} bytes over "
                f"{priced.get('params')} params = {floor:.4f} bpp "
                f"({rel:.1%} apart, tolerance "
                f"{RECIPE_BPP_CROSS_CHECK_TOLERANCE:.0%}). The priced value is "
                "a LOWER bound, so the claim is describing a different "
                "assignment than the recipe holds — most likely a stale "
                "Pareto point or a superseded selection."
                )
            )
    return {**claim, "cross_check": cross}
