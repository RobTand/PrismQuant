"""The ship record (`exported/shipcard.json`) — a refusal contract.

R13 (`docs/audits/architecture_re-vet_2026-07-30.md`). The build lane and the
serve lane are separated by a physical boundary: `vllm` is not importable in the
build venv, so `run-pipeline.sh` cannot run a ship gate and never should. What it
*can* do is **open a record with required, empty slots** that only the serve lane
can close. `python -m prismaquant.shipcard_cli verify` then exits non-zero until every slot holds a
record whose `model_sha` matches the artifact on disk — which turns "we never ran
the gate" from a silent omission into an explicit refusal.

Base slots (required for every artifact):

| Slot | Filled by |
|---|---|
| `native_export.eager` | `validate_native_export.py --shipcard` (eager arm) |
| `native_export.graph` | `validate_native_export.py --shipcard --no-enforce-eager` |
| `ship_gate` | `validate_quantized_model.py --shipcard` |
| `gold.kl` | `python -m prismaquant.shipcard_cli fill --slot gold.kl --record <full_kl json>` |
| `gold.ppl` | `python -m prismaquant.shipcard_cli fill --slot gold.ppl --record <ppl json>` |

Gridbook CB artifacts on the **DSv4 lane** open one additional blocking slot,
``perf.matched_budget_parity``.  It can only be filled by the paired DSv4
performance validator after the candidate clears the predeclared served matrix
against the exact eligible container this release displaces under the same
byte budget.  The generic record importer cannot close this slot.  A CB
artifact that displaces no container -- a net-new size class -- cannot satisfy
the verifier's five ``displaced_container_*`` digests at all, so the slot is
blocking only on the lane whose release argument it encodes (`cb_serving_lane`);
elsewhere it is honoured when present and not demanded when absent.

The two `gold.*` slots additionally require `spec_decode_detected: false` on the
record that produced the number — vLLM routes echo+logprobs through the draft
model under `--speculative-config`, so a spec-decode-on gold number is the MTP
head's NLL, not the artifact's (§7.5).

The optional ``mtp.dspark`` slot is written in both the target and draft cards
by ``validate_dspark_target_draft.py``.  A target-only artifact may omit it or
leave it null.  Any non-null recognized optional claim is verified by default,
and an on-disk ``provenance.dspark_cb_sidecar`` makes ``mtp.dspark`` mandatory;
its specialized verifier replays the paired artifact, runtime, graph, and
acceptance receipt.

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
from pathlib import Path
from typing import Any, Iterable, Mapping

from .gridbook_environment import (
    CANONICAL_GOLD_ENVIRONMENT,
    CANONICAL_GOLD_SET_ENVIRONMENT,
)
from .gridbook_serving_runtime_pin import (
    GridbookServingRuntimePin,
    load_gridbook_serving_runtime_pin,
    require_exact_gridbook_serving_runtime_release,
)

SCHEMA = "prismaquant.shipcard/1"

#: Every slot all serving lanes must close before an artifact is shippable.
#: Keep this base tuple stable: non-CB containers do not inherit plugin-only
#: gates merely because a Gridbook lane adds one.
REQUIRED_SLOTS: tuple[str, ...] = (
    "native_export.eager",
    "native_export.graph",
    "ship_gate",
    "gold.kl",
    "gold.ppl",
)

#: Additional slots required only for Gridbook CB artifacts.
CB_REQUIRED_SLOTS: tuple[str, ...] = (
    "perf.matched_budget_parity",
)

#: Physical-Ada qualification for the strict context-first FP8-CB artifact.
#: This is independent of the generic Gridbook native eager/graph pair: the
#: latter proves load/generation, while this slot binds the exact RTX 4090,
#: 32K/4-GiB serving shape and full-graph compiler receipt.  It is re-derived
#: from ``quant_config.json`` by :func:`required_slots`, so deleting it from a
#: mutable card cannot waive the hardware obligation.
RTX4090_REQUIRED_SLOTS: tuple[str, ...] = ("rtx4090.fp8_cb",)

#: Claims that can be attached to an already exported artifact.  Missing/null
#: claims remain non-blocking for target-only artifacts, but every non-null
#: recognized claim is verified automatically.  A DSpark sidecar artifact
#: additionally requires its claim even when the card slot is missing/null.
OPTIONAL_SLOTS: tuple[str, ...] = ("mtp.dspark",)

#: The vocabulary accepted by :func:`make_record`.  Whether a member is
#: required is artifact-specific and is resolved by :func:`required_slots`.
ALL_SLOTS: tuple[str, ...] = (
    REQUIRED_SLOTS
    + CB_REQUIRED_SLOTS
    + RTX4090_REQUIRED_SLOTS
    + OPTIONAL_SLOTS
)

#: Slots whose number is invalid if it was produced against a spec-decode serve.
GOLD_SLOTS: frozenset[str] = frozenset({"gold.kl", "gold.ppl"})

SHIPCARD_FILENAME = "shipcard.json"


def _released_gridbook_runtime_pin() -> GridbookServingRuntimePin:
    """Return the sole tracked pin only when it is an immutable release."""

    pin = load_gridbook_serving_runtime_pin()
    require_exact_gridbook_serving_runtime_release(pin)
    return pin


# The refusal record is intentionally mutated after export as independent
# serve/gold gates close its slots.  A fixed-size JSON file keeps the exporter
# inventory and hard whole-artifact budget exact across those mutations.
# Trailing JSON whitespace is semantically inert and accepted by every reader.
SHIPCARD_RESERVED_BYTES = 256 * 1024
WEIGHT_CONTENT_MANIFEST_SCHEMA = "prismaquant.weight_content_manifest/1"
WEIGHT_STAT_ATTESTATION_SCHEMA = "prismaquant.weight_stat_attestation/1"
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
CB_PERFORMANCE_RESULT_SCHEMA = "prismaquant.cb_performance_parity/1"
CB_PERFORMANCE_EVIDENCE_SCHEMA = "prismaquant.cb_performance_evidence/1"
DISPLACED_CONTAINER_ELIGIBILITY_SCHEMA = (
    "prismaquant.displaced_container_eligibility/1"
)
CB_PERFORMANCE_TOOL = "validate_cb_performance.py"
DSV4_GRIDBOOK_CONTRACT_SCHEMA = "prismaquant.dsv4_gridbook_llm_contract/1"
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
GRIDBOOK_DISTRIBUTION_SCHEMA = (
    "prismaquant.installed_gridbook_distribution/2"
)
TOPK_COVERAGE_POLICY_SCHEMA = "prismaquant.topk_tail_coverage_policy/1"
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
_TOKENIZER_IDENTITY_FILENAMES = (
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
_CB_FORMAT_RE = re.compile(r"^(?:NVFP4_CB|FP8_CB)_[KS][0-9]+$")
_FP8_SOURCE_W8A16_WIRE_IDS = frozenset({
    "fp8_e4m3_ue8m0_block128",
})
_MXFP8_DENSE_WIRE_IDS = frozenset({"mxfp8_e4m3_e8m0_g32"})
_CB_MAIN_EXTENSION_RE = re.compile(r"^prismaquant_cb_ext(?:[.][^/]*)?[.]so$")
_CB_V2_EXTENSION_RE = re.compile(
    r"^prismaquant_cb_v2_ext(?:[.][^/]*)?[.]so$"
)
_MXFP8_DENSE_EXTENSION_RE = re.compile(
    r"^pq_mxfp8_dense_[A-Za-z0-9_-]+(?:[.][^/]*)?[.]so$"
)
_FP8_SOURCE_W8A16_EXTENSION_RE = re.compile(
    r"^pq_fp8_source_w8a16_[A-Za-z0-9_-]+(?:[.][^/]*)?[.]so$"
)
_CB_BF16_GROUPED_EXTENSION_RE = re.compile(
    r"^pq_cb_bf16_grouped_[A-Za-z0-9_-]+(?:[.][^/]*)?[.]so$"
)
CB_PERFORMANCE_TELEMETRY_KINDS = frozenset({
    "routing_per_layer_per_step",
    "expert_occupancy",
    "active_experts",
    "grouped_moe_whole_operator",
})
CB_PERFORMANCE_PHASE_METRICS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "prefill": (("p95_ttft_ms", "baseline/candidate"),),
    "decode": (
        ("p95_tpot_ms", "baseline/candidate"),
        ("p95_itl_ms", "baseline/candidate"),
        ("output_throughput", "candidate/baseline"),
    ),
    "mixed": (
        ("p95_ttft_ms", "baseline/candidate"),
        ("p95_tpot_ms", "baseline/candidate"),
        ("p95_itl_ms", "baseline/candidate"),
        ("p95_e2el_ms", "baseline/candidate"),
        ("request_throughput", "candidate/baseline"),
        ("output_throughput", "candidate/baseline"),
    ),
}


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
def build_shipcard(
    model_dir: str | os.PathLike,
    *,
    build: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Open a fresh record: build-lane facts filled, every slot empty."""
    root = Path(model_dir)
    from prismaquant.export_output_safety import directory_publication_target

    build_payload = dict(build or {})
    slots = list(
        REQUIRED_SLOTS + CB_REQUIRED_SLOTS
        if build_payload.get("quant_method") == "gridbook"
        and cb_serving_lane(root) == CB_LANE_DSV4_FLASH
        else REQUIRED_SLOTS
    )
    if _is_rtx4090_fp8_cb_artifact({}, model_dir=root):
        slots.extend(RTX4090_REQUIRED_SLOTS)
    if _is_dspark_cb_sidecar_artifact(root):
        slots.append("mtp.dspark")
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
        # Principle 9 / 12: trellis route provenance (D1) travels with the
        # artifact's bpp/quality claim. Both trellis and CB exporters stamp
        # their verdicts into quant_config provenance; lifting here makes the
        # card the refusal contract.
        if isinstance(provenance, Mapping) and isinstance(
            provenance.get("trellis_route_status"), Mapping
        ):
            # Direct trellis verdict (structured per-unit records)
            card["trellis_route_status"] = dict(provenance["trellis_route_status"])
        elif isinstance(provenance, Mapping) and isinstance(
            provenance.get("selection_serving_lane_provenance"), Mapping
        ):
            # Allocator-written selection provenance containing trellis units
            sel = provenance["selection_serving_lane_provenance"]
            if isinstance(sel.get("trellis_units"), list) and sel["trellis_units"]:
                card["trellis_route_status"] = {
                    "schema": sel.get("schema", "prismaquant.serving_lane_route.v1"),
                    "trellis_units": sel["trellis_units"],
                    "trellis_route_histogram": sel.get("trellis_route_histogram", {}),
                    "trellis_route_status_counts": sel.get("trellis_route_status_counts", {}),
                    "trellis_requires_serve_flags": sel.get("trellis_requires_serve_flags", []),
                    "trellis_unbacked_units": sel.get("trellis_unbacked_units", []),
                    "trellis_unbacked_units_detail": sel.get("trellis_unbacked_units_detail", []),
                    "target_profile": sel.get("target_profile"),
                    "gridbook_runtime_version": sel.get("gridbook_runtime_version"),
                }
        # Principle 9 requires the artifact's route disposition -- a declared
        # non-native target or an explicit override -- to be stamped on the
        # card, and principle 12 requires the route census to travel with the
        # bpp claim rather than only inside quant_config. Both CB exporters
        # stamp the gate's verdict into quant_config provenance before opening
        # this card, so lifting it here covers the streaming and non-streaming
        # paths with one seam. The summary keeps the gate's own shape rule: an
        # unattested lane publishes no backed/unbacked counters to misread.
        if isinstance(provenance, Mapping) and isinstance(
            provenance.get("cb_route_status"), Mapping
        ):
            route_status = provenance["cb_route_status"]
            if route_status.get("schema") == (
                "prismaquant.rtx4090_fp8_cb_route_status.v1"
            ):
                # Strict Ada evidence comes from the supplied candidate-v11
                # contract, not the historical generic serving-pin gate.  Its
                # compact row must not be projected through the v1 generic
                # summary, which would emit misleading None fields.
                from .rtx4090_qwen38_policy import (
                    rtx4090_route_status_summary,
                )

                policy = provenance.get("producer_policy")
                assignment = provenance.get("tensor_formats")
                if not isinstance(policy, Mapping) or not isinstance(
                    assignment, Mapping
                ):
                    raise ValueError(
                        "strict RTX4090 route status has no policy/assignment binding"
                    )
                card["cb_route_status"] = rtx4090_route_status_summary(
                    route_status,
                    producer_policy=policy,
                    assignment=assignment,
                    where="shipcard strict RTX4090 route status",
                )
            else:
                # local: the generic gate reaches the serving-lane resolution
                # stack, and shipcard is imported very early.
                from .cb_route_status_gate import shipcard_route_summary

                card["cb_route_status"] = shipcard_route_summary(route_status)
    return card


def open_cb_export_shipcard(
    model_dir: str | os.PathLike,
    quant_config: Mapping[str, Any],
    *,
    source_model: str | os.PathLike,
    layer_config_path: str | os.PathLike,
    exporter: str,
    weight_content_manifest: Mapping[str, Any] | None = None,
    build_extra: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write the preliminary CB config and open its refusal record.

    ``build_extra`` carries exporter-measured build-lane facts that belong on
    the card itself — a per-run override an operator passed, for instance — and
    may not overwrite a field this helper computes.

    CB inventory finalization must run *after* this helper.  The preliminary
    config lets :func:`compute_model_sha` bind all value-bearing CB metadata;
    the subsequently written shipcard is then part of the final recursive
    artifact inventory and of any hard whole-artifact budget check.  Inventory
    finalization changes only the field excluded from CB identity, so the
    freshly opened card remains valid after the fixed-point write.
    """
    root = Path(model_dir)
    provenance = quant_config.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise TypeError("CB quant config provenance must be an object")
    if "weight_content_manifest" in provenance:
        raise ValueError(
            "CB quant config already carries a weight content manifest before export finalization"
        )
    if weight_content_manifest is None:
        print(
            f"[shipcard] hashing exact final weight content under {root}",
            flush=True,
        )
        manifest = build_weight_content_manifest(root)
    else:
        weights = {
            path.name: int(path.stat().st_size)
            for path in sorted(root.glob("*.safetensors"))
        }
        if not weights:
            raise FileNotFoundError(
                f"no safetensors weights to attest under {root}"
            )
        _validate_weight_content_manifest(
            weight_content_manifest, weights, where=root
        )
        # Detach the value placed into the mutable quant-config payload from
        # any mapping the caller retains.
        manifest = json.loads(json.dumps(weight_content_manifest))
        print(
            f"[shipcard] binding {len(weights)} in-stream weight SHA-256 "
            f"digest(s) under {root}",
            flush=True,
        )
    provenance["weight_content_manifest"] = manifest
    from . import read_traffic  # local: shipcard is imported very early
    config_path = root / "quant_config.json"
    config_path.write_text(json.dumps(
        dict(quant_config), indent=2, sort_keys=True
    ))
    build = {
        "git": git_provenance(),
        "exporter": str(exporter),
        "quant_method": "gridbook",
        "source_model": str(source_model),
        "layer_config": str(layer_config_path),
        "layer_config_sha": file_sha256(layer_config_path),
        "achieved_bpp": allocator_achieved_bpp(layer_config_path),
        # See `read_traffic`: bpp measures what the artifact costs on disk,
        # this measures what it costs the memory bus per generated token --
        # the quantity that actually sets decode throughput, and the one a
        # byte budget is blind to on a sparse MoE. Measured from the shards
        # written above, so it cannot describe a different assignment.
        "read_gb_per_token": read_traffic.read_traffic_claim(root),
    }
    policy_stamp = provenance.get("producer_policy")
    if isinstance(policy_stamp, Mapping):
        from .rtx4090_qwen38_policy import (
            is_rtx4090_validation_only_policy,
        )

        if is_rtx4090_validation_only_policy(policy_stamp):
            # The on-disk stamp remains authoritative. Carry the disposition
            # into the card as well so card-only verification cannot pass.
            build.update({
                "producer_policy": str(policy_stamp["id"]),
                "serving_profile": str(policy_stamp["serving_profile"]),
                "artifact_disposition": str(
                    policy_stamp["artifact_disposition"]
                ),
            })
    collisions = sorted(set(build_extra or {}) & set(build))
    if collisions:
        raise ValueError(
            f"build_extra may not overwrite computed build fields: {collisions}"
        )
    build.update(json.loads(json.dumps(dict(build_extra or {}))))
    card = build_shipcard(root, build=build)
    path = write_shipcard(root / SHIPCARD_FILENAME, card)
    return path, card


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
    is_gridbook_cb = _is_gridbook_card(card, model_dir=model_dir)
    if _is_rtx4090_validation_only_artifact(card, model_dir):
        problems.append(
            "RTX4090 FP8-CB artifact is stamped "
            "UNRELEASABLE_VALIDATION_ONLY: compile_only SM89 evidence can "
            "never satisfy a shipcard, publication, or physical RTX4090 gate"
        )
    if is_gridbook_cb and card.get("reserved_file_bytes") != SHIPCARD_RESERVED_BYTES:
        problems.append(
            "shipcard reserved_file_bytes is not the fixed "
            f"{SHIPCARD_RESERVED_BYTES}-byte release reservation"
        )
    if is_gridbook_cb and not isinstance(
        card.get("weight_stat_attestation"), Mapping
    ):
        problems.append(
            "Gridbook artifact lacks the required weight-stat attestation"
        )

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
        if record.get("passed") is not True:
            problems.append(
                f"{slot}: FAILED — {record.get('detail') or 'no detail'}")
        if is_gridbook_cb and slot in {
            "native_export.eager", "native_export.graph"
        }:
            if _is_rtx4090_fp8_cb_artifact(card, model_dir=model_dir):
                from .validate_rtx4090_fp8_cb import (
                    verify_rtx4090_shipcard_record,
                )

                problems.extend(verify_rtx4090_shipcard_record(
                    slot, record, model_dir=model_dir
                ))
            else:
                problems.extend(_verify_gridbook_native_record(
                    slot,
                    record,
                    model_dir=model_dir,
                ))
        if slot in RTX4090_REQUIRED_SLOTS:
            from .validate_rtx4090_fp8_cb import (
                verify_rtx4090_shipcard_record,
            )

            problems.extend(verify_rtx4090_shipcard_record(
                slot, record, model_dir=model_dir
            ))
        if is_gridbook_cb and slot == "perf.matched_budget_parity":
            problems.extend(_verify_gridbook_performance_record(
                slot,
                record,
                model_dir=model_dir,
            ))
        if is_gridbook_cb and slot == "ship_gate":
            problems.extend(_verify_ship_gate_record(slot, record))
        if slot in GOLD_SLOTS:
            # Every lane, not just Gridbook CB. This ran under `is_gridbook_cb`
            # only, which meant a NATIVE card -- the default lane, and the one
            # shipping artifacts today -- had its gold slots checked for nothing
            # but spec-decode: no finite metric, no serve fingerprint, no
            # producer commit, no position count. The DSv4/Gridbook contract
            # stays lane-gated via the flag; the generic evidence requirements
            # were never CB-specific and should never have been behind it.
            #
            # The flag was `is_gridbook_cb` until 2026-08-15. That was DSv4's
            # own release contract standing in for "CB", which no CB artifact
            # off that lane can satisfy -- see
            # `_is_dsv4_gridbook_artifact`. It is now the lane itself, read
            # fail-closed off the identity-bound `config.json`.
            problems.extend(_verify_gold_record(
                slot,
                record,
                model_dir=model_dir,
                require_dsv4_gridbook_contract=(
                    is_gridbook_cb and _is_dsv4_gridbook_artifact(model_dir)
                ),
                require_current_artifact_path=False,
            ))
        if slot == "mtp.dspark":
            from .validate_dspark_target_draft import (
                validate_dspark_shipcard_record,
            )

            problems.extend(validate_dspark_shipcard_record(
                slot,
                record,
                model_dir=model_dir,
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
    # --- Trellis serving-lane provenance gates (WO-D D2) ---
    # These are principle 9 / 12 / 14 gates for the trellis lane, parallel to
    # the CB route gate but structured around the trellis v3 cells.
    trellis_prov = None
    if isinstance(card.get("trellis_route_status"), Mapping):
        trellis_prov = card["trellis_route_status"]
    elif isinstance(card.get("selection_serving_lane_provenance"), Mapping):
        trellis_prov = card["selection_serving_lane_provenance"]
    elif isinstance((card.get("build") or {}).get("selection_serving_lane_provenance"), Mapping):
        trellis_prov = (card["build"] or {})["selection_serving_lane_provenance"]
    # Also support direct trellis_units at top level (test seam)
    if trellis_prov is None and isinstance(card.get("trellis_units"), list):
        trellis_prov = card
    # If still none, try to synthesize from on-disk quant_config (best-effort)
    # so a card lacking a lifted provenance but whose artifact manifest contains
    # trellis formats still fails closed rather than silently passing.
    if trellis_prov is None and model_dir is not None:
        try:
            qc_path = Path(model_dir) / "quant_config.json"
            if qc_path.is_file():
                qc = json.loads(qc_path.read_text())
                prov = (qc.get("provenance") or {}) if isinstance(qc, Mapping) else {}
                if isinstance(prov.get("trellis_route_status"), Mapping):
                    trellis_prov = prov["trellis_route_status"]
                elif isinstance(prov.get("selection_serving_lane_provenance"), Mapping):
                    sel = prov["selection_serving_lane_provenance"]
                    if isinstance(sel.get("trellis_units"), list) and sel["trellis_units"]:
                        trellis_prov = sel
        except Exception:
            pass
    # Determine if this artifact claims to contain trellis units.
    # We treat any provenance with trellis_units non-empty as a trellis artifact,
    # or any card that explicitly declares trellis_route_status.
    is_trellis_artifact = False
    trellis_units_list: list = []
    if isinstance(trellis_prov, Mapping):
        if isinstance(trellis_prov.get("trellis_units"), list):
            trellis_units_list = trellis_prov["trellis_units"]
            if trellis_units_list:
                is_trellis_artifact = True
        # Also consider the case where provenance is the top-level card itself
        # with trellis_units (handled above).
        if not is_trellis_artifact and isinstance(card.get("trellis_units"), list) and card["trellis_units"]:
            trellis_units_list = card["trellis_units"]
            is_trellis_artifact = True
        # If the card declares a trellis provenance key at all, consider it trellis-related
        if not is_trellis_artifact and "trellis_route_status" in card:
            is_trellis_artifact = True
    if is_trellis_artifact:
        # Gate 1: unbacked trellis units require explicit non-native target or override.
        unbacked = [u for u in trellis_units_list if isinstance(u, Mapping) and u.get("route_status") in ("unbacked", "unattested")]
        # Also treat "unbacked" exposed via histogram: check Histogram keys containing :unbacked
        # But primary signal is per-unit.
        if unbacked:
            declared = None
            override = None
            # Check stamped declarations on card (top-level and trellis_prov)
            for src in (card, trellis_prov or {}):
                if isinstance(src, Mapping):
                    if src.get("declared_non_native_target"):
                        declared = src["declared_non_native_target"]
                    if src.get("declared_trellis_non_native_target"):
                        declared = src["declared_trellis_non_native_target"]
                    if src.get("override"):
                        override = src["override"]
                    if src.get("trellis_override"):
                        override = src["trellis_override"]
                    # Also check nested "declared_non_native_target" under provenance
                    if isinstance(src.get("trellis_route_status"), Mapping):
                        inner = src["trellis_route_status"]
                        if isinstance(inner, Mapping) and inner.get("declared_non_native_target"):
                            declared = inner["declared_non_native_target"]
            # Also check explicit fields that shipcard lifts from CB provenance
            if not declared:
                declared = card.get("declared_non_native_target") or (trellis_prov or {}).get("declared_non_native_target")
            if not override:
                override = card.get("override") or (trellis_prov or {}).get("override") or card.get("trellis_override")
            # Allow explicit per-card fields that tests may use
            if not declared and not override:
                # Check environment-style override stamped as string field
                declared = card.get("trellis_non_native_target") or card.get("non_native_target")
                override = card.get("trellis_per_run_override") or override
            if not declared and not override:
                sample = ", ".join(u.get("qname", "?") for u in unbacked[:3])
                reasons = "; ".join(u.get("unattested_reason", u.get("reason", ""))[:120] for u in unbacked[:1])
                problems.append(
                    f"trellis route: {len(unbacked)} unit(s) are unbacked (no attested trellis cell covers them) "
                    f"[{sample}]; {reasons} — export fails closed unless the artifact explicitly declares "
                    "a non-native target platform or carries an explicit per-run override stamped on the card "
                    "(principle 9)"
                )
        # Gate 2: backed_with_serve_flag must record exact requires_serve_flags.
        flagged_units = [u for u in trellis_units_list if isinstance(u, Mapping) and u.get("route_status") == "backed_with_serve_flag"]
        if flagged_units:
            # The artifact must record the flags in a structured field a gate can read.
            recorded_flags = None
            for src in (card, trellis_prov or {}):
                if isinstance(src, Mapping):
                    if isinstance(src.get("trellis_requires_serve_flags"), list):
                        recorded_flags = src["trellis_requires_serve_flags"]
                        break
                    if isinstance(src.get("requires_serve_flags"), list):
                        recorded_flags = src["requires_serve_flags"]
                        break
            # Also check histogram provenance's aggregated flags
            if recorded_flags is None:
                recorded_flags = (trellis_prov or {}).get("trellis_requires_serve_flags")
            if not recorded_flags:
                sample = ", ".join(u.get("qname", "?") for u in flagged_units[:3])
                problems.append(
                    f"trellis route: {len(flagged_units)} unit(s) are backed_with_serve_flag but the artifact "
                    f"records no requires_serve_flags [{sample}]; the flags are artifact metadata, not operator "
                    "folklore, and must be stamped as structured data (principle 9/14)"
                )
            else:
                # Verify that recorded flags cover every per-unit requirement (exact set)
                expected = set()
                for u in flagged_units:
                    for f in u.get("requires_serve_flags", []) or []:
                        expected.add(str(f))
                recorded_set = set(str(f) for f in recorded_flags)
                if expected - recorded_set:
                    problems.append(
                        f"trellis route: backed_with_serve_flag units require flags {sorted(expected)} "
                        f"but the artifact only records {sorted(recorded_set)}"
                    )
        # Gate 3: bpp or quality claim without route histogram.
        has_bpp_claim = False
        has_quality_claim = False
        build = card.get("build") if isinstance(card.get("build"), Mapping) else {}
        if isinstance(build.get("achieved_bpp"), Mapping) and build["achieved_bpp"].get("value") is not None:
            has_bpp_claim = True
        elif isinstance(build.get("achieved_bpp"), (int, float)):
            has_bpp_claim = True
        # Alternative bpp location: card["bpp"], card["achieved_bpp"], or any slot that is gold
        if not has_bpp_claim and isinstance(card.get("bpp"), (int, float, Mapping)):
            has_bpp_claim = True
        if not has_bpp_claim:
            for slot in GOLD_SLOTS:
                rec = (card.get("slots") or {}).get(slot)
                if isinstance(rec, Mapping) and rec.get("passed") is True:
                    has_quality_claim = True
                    break
        if not has_quality_claim:
            # Also check if any trellis unit's bpp claim is implied via build.achieved_bpp cross_check
            if any(k in card for k in ("gold.kl", "gold.ppl")):
                has_quality_claim = True
        if (has_bpp_claim or has_quality_claim) and is_trellis_artifact:
            hist = None
            if isinstance(trellis_prov, Mapping):
                hist = trellis_prov.get("trellis_route_histogram") or trellis_prov.get("route_histogram")
            if not hist:
                hist = card.get("trellis_route_histogram") or card.get("route_histogram")
            if not hist:
                problems.append(
                    "trellis route: artifact publishes a bpp or quality claim (achieved_bpp / gold.kl/ppl) "
                    "without the accompanying route histogram by (family, activation_contract, route_status) "
                    "(principle 12) — a win on a non-native kernel is not a win on the named hardware"
                )
        # Gate 4 (WO-C correction): trellis serving is TP=1 only.
        # Any trellis artifact that declares TP>1 is unservable (sharded trellis needs per-rank wires).
        if is_trellis_artifact:
            tp_declared = None
            # Check structured provenance fields first
            for src in (card, trellis_prov or {}):
                if not isinstance(src, Mapping):
                    continue
                for key in ("trellis_tensor_parallel_world_size", "tensor_parallel_world_size", "vllm_tensor_parallel_world_size", "tp"):
                    if key in src:
                        try:
                            tp_declared = int(src[key])
                        except Exception:
                            tp_declared = None
                        if tp_declared is not None:
                            break
                if tp_declared is not None:
                    break
            # Also check build dict
            if tp_declared is None and isinstance(card.get("build"), Mapping):
                for key in ("tensor_parallel_world_size", "tp"):
                    if key in card["build"]:
                        try:
                            tp_declared = int(card["build"][key])
                        except Exception:
                            pass
            # If provenance explicitly says it requires TP=1 but card declares >1, refuse.
            # If no TP declared, treat as TP=1 (the default) — do not refuse.
            if tp_declared is not None and tp_declared != 1:
                problems.append(
                    f"trellis route: artifact requires TP=1 but declares tensor_parallel_world_size={tp_declared}; "
                    "a sharded trellis needs per-rank wires, not a byte range into a shared schedule (WO-C correction, max_world_size 1 per trellis_format_family)"
                )
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


def _manifest_gridbook_runtime_pin(
    manifest: Mapping[str, Any],
    runtime_pin: Mapping[str, Any],
) -> dict[str, str] | None:
    """Return a live VCS/wheel install pin only when it matches tracked code."""

    observed = manifest.get("gridbook_runtime_pin")
    expected_wheel = runtime_pin.get("wheel_sha256")
    expected_members = (
        {"commit", "version", "wheel_sha256"}
        if expected_wheel is not None
        else None
    )
    if not isinstance(observed, Mapping) or (
        set(observed) != expected_members
        if expected_members is not None
        else set(observed) not in (
            {"commit", "version"},
            {"commit", "version", "wheel_sha256"},
        )
    ):
        return None
    if (
        observed.get("commit") != runtime_pin.get("commit")
        or observed.get("version") != runtime_pin.get("version")
        or (
            expected_wheel is not None
            and observed.get("wheel_sha256") != expected_wheel
        )
        or (
            "wheel_sha256" in observed
            and re.fullmatch(
                r"[0-9a-f]{64}", str(observed.get("wheel_sha256", ""))
            ) is None
        )
    ):
        return None
    return {key: str(observed[key]) for key in sorted(observed)}


def _verify_gridbook_distribution_identity(
    slot: str,
    manifest: Mapping[str, Any],
    runtime_pin: Mapping[str, Any],
    *,
    canonical_sha,
) -> list[str]:
    distribution = manifest.get("gridbook_distribution")
    manifest_pin = _manifest_gridbook_runtime_pin(manifest, runtime_pin)
    if manifest_pin is None:
        return [f"{slot}: live Gridbook runtime pin differs from the tracked pin"]
    if not isinstance(distribution, Mapping) or set(distribution) != {
        "schema", "name", "repository", "version", "direct_url",
        "direct_url_path", "direct_url_identity", "metadata_path",
        "metadata_identity", "record_path", "record_identity",
        "source_files", "source_files_sha256", "import_origin",
    }:
        return [f"{slot}: installed Gridbook distribution evidence is not closed"]
    direct_url = distribution.get("direct_url")
    try:
        from tools.serve_fingerprint import (
            validate_gridbook_pep610_direct_url,
        )

        validate_gridbook_pep610_direct_url(direct_url, {
            "repository": str(runtime_pin.get("repository", "")),
            **manifest_pin,
        })
    except Exception:
        return [
            f"{slot}: installed Gridbook PEP 610 identity differs from the pin"
        ]
    if (
        distribution.get("schema") != GRIDBOOK_DISTRIBUTION_SCHEMA
        or distribution.get("name") != "gridbook"
        or distribution.get("repository") != runtime_pin.get("repository")
        or distribution.get("version") != runtime_pin.get("version")
    ):
        return [
            f"{slot}: installed Gridbook PEP 610 identity differs from the pin"
        ]
    try:
        from tools.serve_fingerprint import (
            validate_gridbook_import_origin_identity,
        )

        validate_gridbook_import_origin_identity(
            distribution.get("import_origin"),
            expected_version=str(runtime_pin.get("version", "")),
        )
    except Exception:
        return [
            f"{slot}: imported Gridbook origin is outside the selected distribution"
        ]

    def descriptor(value: object) -> bool:
        return (
            isinstance(value, Mapping)
            and set(value) == {"bytes", "sha256"}
            and isinstance(value.get("bytes"), int)
            and not isinstance(value.get("bytes"), bool)
            and value.get("bytes", -1) > 0
            and re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))
            is not None
        )

    metadata_paths = [
        distribution.get("direct_url_path"),
        distribution.get("metadata_path"),
        distribution.get("record_path"),
    ]
    if (
        any(
            not isinstance(value, str)
            or value.startswith("/")
            or ".dist-info/" not in value
            for value in metadata_paths
        )
        or not descriptor(distribution.get("direct_url_identity"))
        or not descriptor(distribution.get("metadata_identity"))
        or not descriptor(distribution.get("record_identity"))
    ):
        return [f"{slot}: installed Gridbook metadata/RECORD identity is malformed"]

    source_files = distribution.get("source_files")
    required = {
        "gridbook/__init__.py",
        "gridbook/cuda_ext.py",
        "gridbook/plugin.py",
        "gridbook/runtime_contract.json",
        "gridbook/source_passthrough.py",
        "gridbook/fp8_source_w8a16.py",
        "gridbook/csrc/cb_gemv.cu",
        "gridbook/csrc/fp8_source_w8a16.cu",
        "gridbook/csrc/mxfp8_dense_gemm.cu",
    }
    if (
        not isinstance(source_files, Mapping)
        or not required <= set(source_files)
        or distribution.get("source_files_sha256") != canonical_sha(source_files)
        or any(
            not isinstance(name, str)
            or not name.startswith("gridbook/")
            or ".." in Path(name).parts
            or not descriptor(identity)
            for name, identity in source_files.items()
        )
    ):
        return [f"{slot}: installed Gridbook source/RECORD closure differs"]
    return []


def _verify_dsv4_gridbook_gold_contract(
    slot: str,
    record: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None,
    require_current_artifact_path: bool = True,
) -> list[str]:
    """Replay the closed one-Spark in-process Gridbook gold contract.

    Absolute paths in a receipt identify the measurement mount.  They are
    re-resolved and required to be ``samefile`` at fill time, but a shipped
    artifact remains the same object after a copy/move: later verification is
    therefore keyed by model SHA plus its recursive inventory digest/bytes.
    """
    problems: list[str] = []

    def problem(detail: str) -> None:
        problems.append(f"{slot}: {detail}")

    def sha(value: object) -> bool:
        return isinstance(value, str) and re.fullmatch(
            r"[0-9a-f]{64}", value
        ) is not None

    def canonical_sha(value: object) -> str | None:
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(encoded).hexdigest()

    def positive_int(value: object) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        )

    def finite_nonnegative(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
        )

    contract = metrics.get("dsv4_gridbook_contract")
    if not isinstance(contract, Mapping):
        return [f"{slot}: missing DSv4 Gridbook gold contract"]
    expected_contract_keys = {
        "schema",
        "artifact_dir",
        "requires_moe_backend_marlin",
        "llm_kwargs",
        "environment",
        "tensor_parallel_size",
        "dtype",
        "trust_remote_code",
        "gpu_memory_utilization",
        "disable_log_stats",
        "speculative_decoding",
    }
    if set(contract) != expected_contract_keys:
        problem("DSv4 Gridbook contract fields are not closed and exact")
    if contract.get("schema") != DSV4_GRIDBOOK_CONTRACT_SCHEMA:
        problem("unsupported DSv4 Gridbook contract schema")
    artifact_dir = contract.get("artifact_dir")
    if (
        not isinstance(artifact_dir, str)
        or not artifact_dir
        or not Path(artifact_dir).is_absolute()
    ):
        problem("DSv4 Gridbook contract has no absolute artifact directory")
    requires_marlin = contract.get("requires_moe_backend_marlin")
    if not isinstance(requires_marlin, bool):
        problem("DSv4 Gridbook Marlin requirement is not boolean")
    expected_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.84,
        "max_logprobs": 248_320,
        "quantization": "gridbook",
        "kv_cache_dtype": "fp8",
        "tokenizer_mode": "deepseek_v4",
        "generation_config": "vllm",
        "enable_prefix_caching": False,
        "max_model_len": 8192,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 1_073_741_824,
        "seed": 0,
        "enforce_eager": True,
        "disable_log_stats": True,
    }
    if requires_marlin is True:
        expected_kwargs["moe_backend"] = "marlin"
    if contract.get("llm_kwargs") != expected_kwargs:
        problem("DSv4 Gridbook LLM kwargs differ from the exact release contract")
    expected_environment = dict(CANONICAL_GOLD_ENVIRONMENT)
    if contract.get("environment") != expected_environment:
        problem("DSv4 Gridbook environment is not closed and exact")
    expected_top_level = {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.84,
        "disable_log_stats": True,
        "speculative_decoding": False,
    }
    for key, expected in expected_top_level.items():
        if contract.get(key) != expected:
            problem(f"DSv4 Gridbook contract {key} differs from {expected!r}")

    if slot == "gold.kl":
        workload = {
            "mode": "student",
            "score_positions": "all",
            "prompt_top_k": 8192,
            "n_samples": 8,
            "seqlen": 512,
            "n_positions": 4088,
            "quantization": "gridbook",
        }
        for key, expected in workload.items():
            if metrics.get(key) != expected:
                problem(f"DSv4 KL workload {key} differs from {expected!r}")
        if not positive_int(metrics.get("vocab_size")) or int(
            metrics.get("vocab_size", 0)
        ) <= 1024:
            problem("DSv4 KL workload has an invalid vocabulary size")
        if not positive_int(metrics.get("n_confident")) or not finite_nonnegative(
            metrics.get("kl_confident_mean")
        ):
            problem("DSv4 KL workload lacks confident-position evidence")

        teacher = metrics.get("teacher_evidence")
        expected_teacher_keys = {
            "schema", "payload_sha256", "payload_bytes",
            "payload_semantic_sha256", "meta_sha256", "source_model",
            "source_model_identity_sha256", "calibration_contract",
            "calibration_contract_sha256", "topk_coverage_mean",
            "topk_coverage_min", "topk_coverage_policy",
        }
        if not isinstance(teacher, Mapping) or set(teacher) != expected_teacher_keys:
            problem("DSv4 KL teacher evidence is missing or not closed")
        else:
            for key in (
                "payload_sha256", "payload_semantic_sha256", "meta_sha256",
                "source_model_identity_sha256", "calibration_contract_sha256",
            ):
                if not sha(teacher.get(key)):
                    problem(f"DSv4 KL teacher {key} is not digest-bound")
            if (
                teacher.get("schema") != FULL_KL_TEACHER_EVIDENCE_SCHEMA
                or not positive_int(teacher.get("payload_bytes"))
            ):
                problem("DSv4 KL teacher evidence schema/byte count is invalid")
            coverage_mean = teacher.get("topk_coverage_mean")
            coverage_min = teacher.get("topk_coverage_min")
            coverage_policy = teacher.get("topk_coverage_policy")
            expected_coverage_policy = {
                "schema": TOPK_COVERAGE_POLICY_SCHEMA,
                "top_k": 8192,
                "minimum_probability_mass_per_position": 0.90,
                "maximum_probability_mass": 1.0,
                "probability_mass_absolute_tolerance": 1e-6,
                "maximum_declared_tail_mass_per_position": 1.0 - 0.90,
                "tail_bucket": True,
            }
            if (
                not finite_nonnegative(coverage_mean)
                or not finite_nonnegative(coverage_min)
                or float(coverage_min) < 0.90
                or float(coverage_mean) < float(coverage_min)
                or float(coverage_mean) > 1.0 + 1e-6
                or coverage_policy != expected_coverage_policy
            ):
                problem("DSv4 KL teacher top-K/tail coverage policy differs")
            source = teacher.get("source_model")
            if (
                not isinstance(source, Mapping)
                or set(source) != {
                    "schema", "content_sha256", "resolved_commit",
                    "checkpoint_shards", "checkpoint_tensors",
                }
                or source.get("schema")
                != "prismaquant.streamed_model.identity.v1"
                or not sha(source.get("content_sha256"))
                or (
                    source.get("resolved_commit") is not None
                    and (
                        not isinstance(source.get("resolved_commit"), str)
                        or not source.get("resolved_commit")
                    )
                )
                or not positive_int(source.get("checkpoint_shards"))
                or not positive_int(source.get("checkpoint_tensors"))
            ):
                problem("DSv4 KL teacher source-model identity is invalid")
            calibration = teacher.get("calibration_contract")
            expected_calibration_keys = {
                "schema", "dataset", "corpus_construction", "tokenizer",
                "window_seed", "sampler", "n_samples", "seqlen", "starts",
                "total_tokens", "calib_ids_sha256", "scoring",
            }
            if (
                not isinstance(calibration, Mapping)
                or set(calibration) != expected_calibration_keys
                or calibration.get("schema") != WIKITEXT_GOLD_CALIBRATION_SCHEMA
                or teacher.get("calibration_contract_sha256")
                != canonical_sha(calibration)
            ):
                problem("DSv4 KL calibration contract/digest is invalid")
            else:
                dataset = calibration.get("dataset")
                if (
                    not isinstance(dataset, Mapping)
                    or set(dataset) != {
                        "name", "config", "split", "revision", "fingerprint",
                        "corpus_sha256",
                    }
                    or dataset.get("name") != "wikitext"
                    or dataset.get("config") != "wikitext-2-raw-v1"
                    or dataset.get("split") != "train"
                    or dataset.get("revision")
                    != "b08601e04326c79dfdd32d625aee71d232d685c3"
                    or not isinstance(dataset.get("fingerprint"), str)
                    or not dataset.get("fingerprint")
                    or not sha(dataset.get("corpus_sha256"))
                ):
                    problem("DSv4 KL calibration dataset identity differs")
                if calibration.get("corpus_construction") != {
                    "row_filter": (
                        "include iff bool(text.strip()); preserve text verbatim"
                    ),
                    "join_separator": "\n\n",
                    "normalization": "none",
                }:
                    problem("DSv4 KL calibration corpus construction differs")
                tokenizer = calibration.get("tokenizer")
                if (
                    not isinstance(tokenizer, Mapping)
                    or tokenizer != {
                        "identity_sha256": tokenizer.get("identity_sha256"),
                        "trust_remote_code": True,
                        "add_special_tokens": False,
                    }
                    or not sha(tokenizer.get("identity_sha256"))
                ):
                    problem("DSv4 KL calibration tokenizer identity differs")
                starts = calibration.get("starts")
                if (
                    calibration.get("window_seed") != 42
                    or calibration.get("sampler") != (
                        "python.random.Random(seed).sample(range(max_start), "
                        "n_samples)/v1"
                    )
                    or calibration.get("n_samples") != 8
                    or calibration.get("seqlen") != 512
                    or not isinstance(starts, list)
                    or len(starts) != 8
                    or len(set(starts)) != 8
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in starts
                    )
                    or not positive_int(calibration.get("total_tokens"))
                    or int(calibration.get("total_tokens", 0)) < 513
                    or not sha(calibration.get("calib_ids_sha256"))
                    or calibration.get("scoring") != {
                        "positions": "all",
                        "prompt_top_k": 8192,
                        "logprob_dtype": "float32",
                        "tail_bucket": True,
                    }
                ):
                    problem("DSv4 KL calibration window/scoring contract differs")
    else:
        workload = {
            "split": "test",
            "n_tokens_requested": 8192,
            "n_tokens_scored": 8176,
            "seqlen": 512,
            "quantization": "gridbook",
        }
        for key, expected in workload.items():
            if metrics.get(key) != expected:
                problem(f"DSv4 PPL workload {key} differs from {expected!r}")
        mean_nll = metrics.get("mean_nll")
        perplexity = metrics.get("ppl")
        per_chunk = metrics.get("per_chunk_mean_nll")
        max_chunk = metrics.get("max_chunk_mean_nll")
        if not finite_nonnegative(mean_nll):
            problem("DSv4 PPL mean_nll is missing, non-finite, or negative")
        if (
            not finite_nonnegative(perplexity)
            or float(perplexity) < 1.0
        ):
            problem("DSv4 PPL ppl is missing, non-finite, or below one")
        if (
            not isinstance(per_chunk, list)
            or len(per_chunk) != 16
            or any(not finite_nonnegative(value) for value in per_chunk)
        ):
            problem(
                "DSv4 PPL requires exactly 16 finite non-negative "
                "per-chunk mean NLL values"
            )
        elif not finite_nonnegative(max_chunk) or not math.isclose(
            float(max_chunk),
            max(float(value) for value in per_chunk),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            problem("DSv4 PPL max_chunk_mean_nll arithmetic differs")
        if finite_nonnegative(mean_nll) and isinstance(per_chunk, list) and (
            len(per_chunk) == 16
            and all(finite_nonnegative(value) for value in per_chunk)
        ) and not math.isclose(
            float(mean_nll),
            math.fsum(float(value) for value in per_chunk) / 16.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            problem("DSv4 PPL mean_nll differs from the per-chunk mean")
        if finite_nonnegative(mean_nll) and (
            finite_nonnegative(perplexity) and float(perplexity) >= 1.0
        ):
            try:
                expected_perplexity = math.exp(float(mean_nll))
            except OverflowError:
                expected_perplexity = math.inf
            if not math.isclose(
                float(perplexity),
                expected_perplexity,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                problem("DSv4 PPL ppl differs from exp(mean_nll)")
        calibration = metrics.get("calibration_contract")
        expected_calibration_keys = {
            "schema", "dataset", "corpus_construction", "tokenizer",
            "token_selection", "scoring",
        }
        if (
            not isinstance(calibration, Mapping)
            or set(calibration) != expected_calibration_keys
            or calibration.get("schema") != WIKITEXT_PPL_CALIBRATION_SCHEMA
            or metrics.get("calibration_contract_sha256")
            != canonical_sha(calibration)
        ):
            problem("DSv4 PPL calibration contract/digest is invalid")
        else:
            dataset = calibration.get("dataset")
            if (
                not isinstance(dataset, Mapping)
                or set(dataset) != {
                    "name", "config", "split", "revision", "fingerprint",
                    "corpus_sha256",
                }
                or dataset.get("name") != "wikitext"
                or dataset.get("config") != "wikitext-2-raw-v1"
                or dataset.get("split") != "test"
                or dataset.get("revision") != WIKITEXT_REVISION
                or dataset.get("fingerprint")
                != DSV4_WIKITEXT_DATASET_FINGERPRINT
                or dataset.get("corpus_sha256")
                != DSV4_WIKITEXT_CORPUS_SHA256
            ):
                problem("DSv4 PPL calibration dataset identity differs")
            if calibration.get("corpus_construction") != {
                "row_filter": (
                    "include iff bool(text.strip()); preserve text verbatim"
                ),
                "join_separator": "\n\n",
                "normalization": "none",
            }:
                problem("DSv4 PPL corpus construction differs")
            tokenizer = calibration.get("tokenizer")
            if (
                not isinstance(tokenizer, Mapping)
                or set(tokenizer) != {
                    "identity_sha256", "trust_remote_code",
                    "add_special_tokens",
                }
                or tokenizer.get("trust_remote_code") is not True
                or tokenizer.get("add_special_tokens") is not False
                or tokenizer.get("identity_sha256")
                != DSV4_TOKENIZER_IDENTITY_SHA256
            ):
                problem("DSv4 PPL tokenizer identity is invalid")
            elif model_dir is not None:
                tokenizer_files: dict[str, dict[str, object]] = {}
                try:
                    tokenizer_root = Path(model_dir).resolve(strict=True)
                    for name in _TOKENIZER_IDENTITY_FILENAMES:
                        path = tokenizer_root / name
                        if path.is_file() and not path.is_symlink():
                            tokenizer_files[name] = {
                                "bytes": int(path.stat().st_size),
                                "sha256": _file_content_sha256(path),
                            }
                    observed_tokenizer_sha = canonical_sha({
                        "files": tokenizer_files
                    })
                except OSError:
                    observed_tokenizer_sha = None
                if (
                    not tokenizer_files
                    or tokenizer.get("identity_sha256")
                    != observed_tokenizer_sha
                ):
                    problem(
                        "DSv4 PPL tokenizer identity differs from artifact"
                    )
            selection = calibration.get("token_selection")
            if (
                not isinstance(selection, Mapping)
                or set(selection) != {
                    "strategy", "n_tokens_requested", "n_tokens_available",
                    "selected_token_count", "token_ids_sha256",
                    "digest_encoding",
                }
                or selection.get("strategy")
                != "contiguous_prefix_after_full_corpus_tokenization/v1"
                or selection.get("n_tokens_requested") != 8192
                or selection.get("selected_token_count") != 8192
                or selection.get("n_tokens_available")
                != DSV4_WIKITEXT_TOTAL_TOKENS
                or selection.get("token_ids_sha256")
                != DSV4_WIKITEXT_SELECTED_TOKEN_IDS_SHA256
                or selection.get("digest_encoding")
                != "canonical_json_integer_array/v1"
            ):
                problem("DSv4 PPL token-prefix identity differs")
            scoring = calibration.get("scoring")
            expected_starts = list(range(0, 8192, 512))
            if (
                not isinstance(scoring, Mapping)
                or set(scoring) != {
                    "chunking", "seqlen", "chunk_starts",
                    "chunk_token_counts", "positions", "n_tokens_scored",
                    "prompt_logprobs", "temperature", "max_tokens",
                    "detokenize",
                }
                or scoring.get("chunking")
                != "nonoverlapping_contiguous/v1"
                or scoring.get("seqlen") != 512
                or scoring.get("chunk_starts") != expected_starts
                or scoring.get("chunk_token_counts") != [512] * 16
                or scoring.get("positions")
                != "within_each_chunk_positions_1_through_N_minus_1"
                or scoring.get("n_tokens_scored") != 8176
                or scoring.get("prompt_logprobs") != 1
                or scoring.get("temperature") != 0.0
                or scoring.get("max_tokens") != 1
                or scoring.get("detokenize") is not False
            ):
                problem("DSv4 PPL scoring-window contract differs")

    manifest = metrics.get("serve_manifest")
    if not isinstance(manifest, Mapping):
        return problems + [f"{slot}: missing in-process serve-manifest binding"]
    if manifest.get("schema") != SERVE_MANIFEST_SCHEMA:
        problem("unsupported gold serve-manifest schema")
    contract_sha = canonical_sha(contract)
    if (
        contract_sha is None
        or metrics.get("dsv4_gridbook_contract_sha256") != contract_sha
        or manifest.get("dsv4_gridbook_contract_sha256") != contract_sha
        or manifest.get("effective_llm_kwargs") != expected_kwargs
    ):
        problem("gold manifest does not bind the exact effective LLM contract")
    try:
        from tools.serve_fingerprint import (
            SERVER_ENV_ALLOWLIST,
            _serve_model,
            argv_identifies_vllm_engine,
            fingerprint,
            performance_stack_fingerprint,
            process_identity_sha256,
            serve_session_fingerprint,
        )

        recomputed_fingerprint = fingerprint(manifest)
    except Exception as exc:
        problem(f"gold serve manifest is not canonical: {exc}")
        recomputed_fingerprint = None
    manifest_fingerprint = manifest.get("serve_fingerprint")
    if (
        not sha(manifest_fingerprint)
        or manifest_fingerprint != recomputed_fingerprint
        or manifest_fingerprint != record.get("serve_fingerprint")
    ):
        problem("gold serve fingerprint is missing, stale, or differs from its record")
    expected_tool = (
        "measure_vllm_full_kl"
        if slot == "gold.kl"
        else "measure_vllm_wikitext_ppl"
    )
    if (
        manifest.get("source") != "in_process"
        or manifest.get("measurement_tool") != expected_tool
        or manifest.get("attestation_phase") != "snapshot"
    ):
        problem("gold serve manifest has the wrong in-process measurement identity")
    if manifest.get("speculative_config") is not None:
        problem("gold serve manifest carries speculative decoding")
    from .validate_cb_endpoint import (
        DSV4_SPARK_GPU_NAME,
        DSV4_SPARK_VLLM_IMAGE,
        DSV4_SPARK_VLLM_VERSION,
        _gridbook_runtime_pin,
    )

    try:
        runtime_pin = _gridbook_runtime_pin()
    except Exception as exc:
        problem(f"tracked Gridbook release pin unavailable: {exc}")
        return problems
    manifest_runtime_pin = _manifest_gridbook_runtime_pin(manifest, runtime_pin)
    packages = manifest.get("package_versions")
    extensions = manifest.get("resident_extensions")
    gpu_uuid = manifest.get("gpu_uuid")
    if (
        manifest.get("image") != DSV4_SPARK_VLLM_IMAGE
        or manifest.get("gpu_name") != DSV4_SPARK_GPU_NAME
        or manifest.get("gpu_count") != 1
        or not isinstance(gpu_uuid, str)
        or not gpu_uuid
        or not isinstance(manifest.get("driver_version"), str)
        or not manifest.get("driver_version")
        or not isinstance(packages, Mapping)
        or packages.get("vllm") != DSV4_SPARK_VLLM_VERSION
        or packages.get("gridbook") != runtime_pin["version"]
        or manifest_runtime_pin is None
        or manifest.get("residency_readable") is not True
        or not isinstance(extensions, list)
        or any(not isinstance(name, str) for name in extensions)
        or extensions != sorted(set(extensions))
    ):
        problem("gold serve manifest is not the exact one-Spark Gridbook runtime")
    problems.extend(_verify_gold_producer_identity(
        slot, record, manifest, canonical_sha=canonical_sha
    ))
    problems.extend(_verify_gridbook_distribution_identity(
        slot, manifest, runtime_pin, canonical_sha=canonical_sha
    ))

    host = manifest.get("host_identity")
    processes = manifest.get("processes")
    process_pids: list[int] = []
    process_hashes: list[str] = []
    process_valid = (
        isinstance(host, Mapping)
        and isinstance(host.get("boot_id"), str)
        and bool(host.get("boot_id"))
        and sha(host.get("machine_id_sha256"))
        and isinstance(processes, list)
        and len(processes) >= 2
    )
    if process_valid:
        for row in processes:
            if not isinstance(row, Mapping):
                process_valid = False
                break
            pid = row.get("pid")
            argv = row.get("argv")
            identity = row.get("identity_sha256")
            if (
                set(row) != {
                    "pid", "argv", "cmdline", "start_time_ticks",
                    "pid_namespace", "executable", "identity_sha256",
                }
                or
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 0
                or not isinstance(argv, list)
                or not argv
                or any(not isinstance(value, str) for value in argv)
                or not sha(identity)
                or identity != process_identity_sha256(
                    row, boot_id=host.get("boot_id")
                )
            ):
                process_valid = False
                break
            process_pids.append(pid)
            process_hashes.append(str(identity))
    parent_pid = manifest.get("measurement_parent_pid")
    engine_pids = manifest.get("engine_descendant_pids")
    observed_engine_pids = [
        row.get("pid")
        for row in (processes if isinstance(processes, list) else [])
        if isinstance(row, Mapping)
        and argv_identifies_vllm_engine(row.get("argv", []))
    ]
    if (
        not process_valid
        or process_pids != sorted(set(process_pids))
        or len(process_hashes) != len(set(process_hashes))
        or not positive_int(parent_pid)
        or parent_pid not in process_pids
        or not isinstance(engine_pids, list)
        or not engine_pids
        or engine_pids != sorted(set(engine_pids))
        or engine_pids != observed_engine_pids
        or parent_pid in engine_pids
        or any(pid not in process_pids for pid in engine_pids)
        or not sha(manifest.get("serve_session_id"))
        or manifest.get("serve_session_id") != serve_session_fingerprint(manifest)
        or not sha(manifest.get("performance_stack_fingerprint"))
        or manifest.get("performance_stack_fingerprint")
        != performance_stack_fingerprint(manifest)
    ):
        problem("gold process/session identity is incomplete or inconsistent")

    manifest_environment = manifest.get("server_process_environment")
    gold_process_environment = {
        **dict(CANONICAL_GOLD_SET_ENVIRONMENT),
        "PYTHONSAFEPATH": "1",
    }
    environment_rows = manifest_environment.get("processes") if isinstance(
        manifest_environment, Mapping
    ) else None
    environment_valid = (
        isinstance(manifest_environment, Mapping)
        and set(manifest_environment) == {
            "schema", "allowlist", "readable_pids", "unreadable_pids",
            "consistent", "values", "processes",
        }
        and manifest_environment.get("schema")
        == "prismaquant.server_process_environment/1"
        and manifest_environment.get("allowlist")
        == sorted(SERVER_ENV_ALLOWLIST)
        and manifest_environment.get("readable_pids") == process_pids
        and manifest_environment.get("unreadable_pids") == []
        and manifest_environment.get("consistent") is True
        and manifest_environment.get("values")
        == gold_process_environment
        and manifest.get("pq_env") == gold_process_environment
        and isinstance(environment_rows, list)
        and len(environment_rows) == len(process_pids)
    )
    if environment_valid:
        for index, row in enumerate(environment_rows):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"pid", "values", "sha256"}
                or row.get("pid") != process_pids[index]
                or row.get("values") != gold_process_environment
                or row.get("sha256")
                != canonical_sha(gold_process_environment)
            ):
                environment_valid = False
                break
    if not environment_valid:
        problem("gold process environment differs from the exact contract")

    binding = manifest.get("artifact_binding")
    if not isinstance(binding, Mapping):
        return problems + [f"{slot}: gold serve manifest has no artifact binding"]
    if set(binding) != {
        "schema", "resolved_path", "launch_model", "model_sha",
        "artifact_inventory_sha256", "artifact_bytes",
    } or binding.get("schema") != SERVED_ARTIFACT_BINDING_SCHEMA:
        problem("gold serve manifest has the wrong artifact-binding schema")
    if binding.get("model_sha") != record.get("model_sha"):
        problem("gold serve manifest binds a different model_sha")
    if (
        isinstance(artifact_dir, str)
        and binding.get("resolved_path") != artifact_dir
    ):
        problem("gold contract and serve manifest name different artifact paths")
    launch_model = binding.get("launch_model")
    if not isinstance(launch_model, str) or not Path(launch_model).is_absolute():
        problem("gold serve manifest does not bind an absolute launch model")

    launch_argv = manifest.get("launch_argv")
    observed_launch_model = (
        _serve_model(launch_argv)
        if isinstance(launch_argv, list)
        and all(isinstance(value, str) for value in launch_argv)
        else None
    )
    if (
        not isinstance(manifest.get("model"), str)
        or manifest.get("model") != launch_model
        or observed_launch_model != launch_model
    ):
        problem("gold launch argv/model do not bind the measured artifact")

    if model_dir is not None:
        try:
            root = Path(model_dir).resolve(strict=True)
            if not root.is_dir():
                raise ValueError("verified artifact path is not a directory")
            if require_current_artifact_path:
                canonical_root = str(root)
                for label, value in (
                    ("contract artifact_dir", artifact_dir),
                    ("binding resolved_path", binding.get("resolved_path")),
                    ("binding launch_model", launch_model),
                    ("manifest model", manifest.get("model")),
                    ("argv model", observed_launch_model),
                ):
                    if not isinstance(value, str) or not Path(value).is_absolute():
                        raise ValueError(f"{label} is not an absolute path")
                    resolved = Path(value).resolve(strict=True)
                    if not resolved.is_dir() or not os.path.samefile(root, resolved):
                        raise ValueError(f"{label} differs from the verified artifact")
                if artifact_dir != canonical_root or binding.get(
                    "resolved_path"
                ) != canonical_root:
                    raise ValueError("artifact receipt paths are not canonical")
            quant_config = json.loads((root / "quant_config.json").read_text(
                encoding="utf-8"
            ))
            provenance = quant_config.get("provenance")
            inventory = provenance.get("artifact_inventory") if isinstance(
                provenance, Mapping
            ) else None
            if not isinstance(inventory, Mapping):
                raise ValueError("artifact has no finalized inventory")
            canonical = json.dumps(
                inventory,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            inventory_sha = hashlib.sha256(canonical).hexdigest()
            if binding.get("artifact_inventory_sha256") != inventory_sha:
                raise ValueError("inventory digest differs")
            if binding.get("artifact_bytes") != inventory.get(
                "export_directory_bytes"
            ):
                raise ValueError("artifact byte count differs")
            if slot == "gold.kl":
                teacher = metrics.get("teacher_evidence")
                teacher_source = teacher.get("source_model") if isinstance(
                    teacher, Mapping
                ) else None
                candidate_source = provenance.get(
                    "source_model_identity"
                ) if isinstance(provenance, Mapping) else None
                if teacher_source != candidate_source:
                    raise ValueError(
                        "teacher source identity differs from candidate provenance"
                    )
            declared_files = inventory.get("file_bytes")
            if not isinstance(declared_files, Mapping) or not declared_files:
                raise ValueError("artifact inventory file ledger is empty")
            observed_files: dict[str, int] = {}
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    # `tools/publish_artifact.py` replays this contract on its
                    # frozen view, where large files are links to the
                    # publisher's own held descriptors (`/proc/self/fd/N`).
                    # The freeze opens every source entry O_NOFOLLOW and so
                    # can never ingest a real symlink, which means a link of
                    # this exact form inside a replayed tree is the freeze's
                    # representation of a regular file, not artifact content.
                    # Any other symlink -- including one an artifact tries to
                    # ship literally -- still refuses, and the exporter's
                    # inventory can never contain one.
                    target = os.readlink(path)
                    if not target.startswith("/proc/self/fd/"):
                        raise ValueError(
                            "artifact inventory contains a symlink"
                        )
                    followed = path.stat()
                    if not stat.S_ISREG(followed.st_mode):
                        raise ValueError(
                            "artifact inventory contains a symlink"
                        )
                    observed_files[path.relative_to(root).as_posix()] = int(
                        followed.st_size
                    )
                    continue
                if path.is_file():
                    rel = path.relative_to(root).as_posix()
                    if rel == "README.md" or rel in CARD_FIGURE_FILENAMES:
                        # `compute_model_sha` deliberately excludes the model
                        # card and its figures -- exact filenames -- so that
                        # DOCUMENTING an artifact cannot invalidate the records
                        # that measured it (an artifact must be able to quote
                        # its own gold numbers and draw its own allocation; see
                        # the exclusion doctrine there).  The exporter's
                        # finalized inventory predates any card by
                        # construction, so this replay honors the same
                        # exclusion or no documented artifact can ever pass.
                        continue
                    observed_files[rel] = int(path.stat().st_size)
            if dict(declared_files) != observed_files or sum(
                observed_files.values()
            ) != binding.get("artifact_bytes"):
                raise ValueError("artifact recursive files differ from inventory")
            from tools.dsv4_gridbook_contract import requires_moe_backend_marlin

            if requires_moe_backend_marlin(root) is not requires_marlin:
                raise ValueError("Marlin requirement differs from quant_config")
            required_extensions = _gold_extension_requirements(quant_config)
            observed_extensions = (
                extensions if isinstance(extensions, list) else []
            )
            extension_checks = {
                "cb_main": _CB_MAIN_EXTENSION_RE,
                "cb_v2": _CB_V2_EXTENSION_RE,
                "mxfp8_dense": _MXFP8_DENSE_EXTENSION_RE,
                "fp8_source_w8a16": _FP8_SOURCE_W8A16_EXTENSION_RE,
                "cb_bf16_grouped": _CB_BF16_GROUPED_EXTENSION_RE,
            }
            missing_extension_families = sorted(
                family
                for family in required_extensions
                if not any(
                    extension_checks[family].fullmatch(name) is not None
                    for name in observed_extensions
                )
            )
            if missing_extension_families:
                raise ValueError(
                    "resident extension set does not cover finalized routes: "
                    + ", ".join(missing_extension_families)
                )
        except Exception as exc:
            problem(f"gold artifact-binding replay failed — {exc}")
    else:
        problem("gold artifact-binding replay requires the verified artifact path")
    return problems


def _verify_gold_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
    require_dsv4_gridbook_contract: bool = False,
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
    if require_dsv4_gridbook_contract:
        problems.extend(_verify_dsv4_gridbook_gold_contract(
            slot,
            record,
            metrics,
            model_dir=model_dir,
            require_current_artifact_path=require_current_artifact_path,
        ))
    return problems


def _verify_ship_gate_record(
    slot: str,
    record: Mapping[str, Any],
) -> list[str]:
    """Replay the fixed catastrophic-quality thresholds and check ledger."""
    problems: list[str] = []
    if record.get("tool") != "validate_quantized_model.py":
        problems.append(f"{slot}: not filled by validate_quantized_model.py")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        problems.append(f"{slot}: missing full producer git commit")
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


def _verify_gridbook_native_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Structural proof that a CB native slot came from the exact validator.

    `model_dir` selects the serving lane whose stack the contract must name.
    It is re-derived from disk here rather than read out of the receipt, so a
    receipt cannot nominate the lane whose contract it happens to satisfy.
    """
    problems: list[str] = []
    arm = slot.rsplit(".", 1)[-1]
    if record.get("tool") != "validate_cb_endpoint.py":
        problems.append(f"{slot}: not filled by validate_cb_endpoint.py")
    fingerprint = record.get("serve_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ):
        problems.append(f"{slot}: missing exact serve fingerprint")
    if record.get("spec_decode_detected") is not False:
        problems.append(f"{slot}: speculative-decode state is not false")

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return problems + [f"{slot}: missing structured CB endpoint metrics"]
    expected = {
        "arm": arm,
        "enforce_eager": arm == "eager",
        "quantization": "gridbook",
        "kv_cache_dtype": "fp8",
        "tensor_parallel_size": 1,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            problems.append(
                f"{slot}: endpoint metric {key}={metrics.get(key)!r}, expected {value!r}"
            )
    try:
        pin = _released_gridbook_runtime_pin()
    except Exception as exc:
        problems.append(f"{slot}: tracked Gridbook release pin unavailable: {exc}")
    else:
        if metrics.get("gridbook_runtime_commit") != pin.commit:
            problems.append(f"{slot}: Gridbook runtime commit is not the tracked pin")
        if metrics.get("gridbook_runtime_version") != pin.version:
            problems.append(f"{slot}: Gridbook runtime version is not the tracked pin")

    contract = metrics.get("endpoint_contract")
    if not isinstance(contract, Mapping):
        problems.append(f"{slot}: missing canonical endpoint contract")
    else:
        try:
            # Runtime import avoids a module cycle: the endpoint writer imports
            # shipcard to construct records, while verify only needs this
            # stdlib structural replay after both modules are initialized.
            from .validate_cb_endpoint import validate_endpoint_contract_record

            validate_endpoint_contract_record(
                contract,
                arm=arm,
                model_sha=record.get("model_sha"),
                serve_fingerprint=fingerprint,
                lane=cb_serving_lane(model_dir),
            )
        except Exception as exc:
            problems.append(f"{slot}: invalid endpoint contract — {exc}")
        manifest_binding = contract.get("serve_manifest")
        if not isinstance(manifest_binding, Mapping) or metrics.get(
            "serve_manifest_sha256"
        ) != manifest_binding.get("sha256"):
            problems.append(
                f"{slot}: serve-manifest digest differs from endpoint contract"
            )
        stack = contract.get("stack")
        if isinstance(stack, Mapping):
            for metric_key, stack_key in (
                ("gridbook_runtime_commit", "gridbook_runtime_commit"),
                ("gridbook_runtime_version", "gridbook_runtime_version"),
                ("vllm_version", "vllm_version"),
                ("vllm_commit", "vllm_commit"),
            ):
                if metrics.get(metric_key) != stack.get(stack_key):
                    problems.append(
                        f"{slot}: {metric_key} differs from endpoint contract"
                    )
        smoke = contract.get("endpoint_smoke")
        if isinstance(smoke, Mapping):
            for key, value in smoke.items():
                if metrics.get(key) != value:
                    problems.append(
                        f"{slot}: endpoint metric {key} differs from endpoint contract"
                    )
    graph = metrics.get("cuda_graph")
    if arm == "graph":
        if not isinstance(graph, Mapping) or not isinstance(
            graph.get("serve_log_sha256"), str
        ) or not str(graph.get("capture_marker", "")).startswith(
            "Graph capturing finished"
        ):
            problems.append(f"{slot}: missing positive CUDA-graph capture evidence")
        elif isinstance(contract, Mapping) and (
            contract.get("cuda_graph")
            != {
                "capture_marker": graph.get("capture_marker"),
                "serve_log_sha256": graph.get("serve_log_sha256"),
            }
        ):
            problems.append(
                f"{slot}: CUDA-graph evidence differs from endpoint contract"
            )
    elif graph is not None:
        problems.append(f"{slot}: eager receipt unexpectedly carries graph evidence")
    return problems


def _verify_gridbook_performance_record(
    slot: str,
    record: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> list[str]:
    """Replay the self-contained structure of the blocking CB parity proof.

    The performance validator reads the large paired report/telemetry corpus
    once.  The fixed-size shipcard persists their unique SHA-256 identities,
    compact paired raw metric projections, the exact matrix coverage, the
    independently eligible displaced container, and the candidate inventory.
    Publication recomputes every ratio and verdict here instead of trusting a
    generic ``passed: true`` row or a producer-computed statistic.
    """
    problems: list[str] = []

    def problem(detail: str) -> None:
        problems.append(f"{slot}: {detail}")

    def sha(value: object) -> bool:
        return isinstance(value, str) and re.fullmatch(
            r"[0-9a-f]{64}", value
        ) is not None

    def positive_int(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value > 0
        )

    def finite(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            parsed = float(value)
        except (OverflowError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    if record.get("slot") != slot:
        problem("record slot does not identify the parity gate")
    if record.get("tool") != CB_PERFORMANCE_TOOL:
        problem(f"not filled by {CB_PERFORMANCE_TOOL}")
    if record.get("spec_decode_detected") is not None:
        problem("performance receipt unexpectedly carries speculative-decode state")
    if record.get("serve_fingerprint") is not None:
        problem("performance receipt unexpectedly carries one-arm serve fingerprint")
    commit = record.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        problem("missing full producer git commit")

    metrics = record.get("metrics")
    evidence = record.get("evidence")
    if not isinstance(metrics, Mapping):
        return problems + [f"{slot}: missing structured parity metrics"]
    if not isinstance(evidence, Mapping):
        return problems + [f"{slot}: missing digest-bound parity evidence"]
    if metrics.get("schema") != CB_PERFORMANCE_RESULT_SCHEMA:
        problem("unsupported parity metrics schema")
    if evidence.get("schema") != CB_PERFORMANCE_EVIDENCE_SCHEMA:
        problem("unsupported parity evidence schema")
    if metrics.get("prismaquant_runtime_commit") != commit:
        problem("PrismaQuant validator commit differs from record provenance")

    try:
        pin = _released_gridbook_runtime_pin()
    except Exception as exc:
        problem(f"tracked Gridbook release pin unavailable: {exc}")
        pin = None
    if isinstance(pin, GridbookServingRuntimePin):
        if metrics.get("gridbook_runtime_commit") != pin.commit:
            problem("Gridbook runtime commit is not the tracked pin")
        if metrics.get("gridbook_runtime_version") != pin.version:
            problem("Gridbook runtime version is not the tracked pin")

    environment_contract = metrics.get("server_environment_contract")
    expected_performance_environment = {
        **dict(CANONICAL_GOLD_ENVIRONMENT),
        "PRISMAQUANT_PRELOAD_FUSED": "1",
    }
    if environment_contract != {
        "schema": "prismaquant.gridbook_environment/1",
        "profile": "matched_budget_performance",
        "base_profile": "canonical_gold",
        "overrides": {"PRISMAQUANT_PRELOAD_FUSED": "1"},
        "environment": expected_performance_environment,
    }:
        problem(
            "performance environment does not carry the exact canonical-plus-preload profile"
        )

    digest_fields = (
        "candidate_inventory_sha256",
        "matrix_digest",
        "displaced_container_eligibility_sha256",
        "displaced_container_model_sha",
        "displaced_container_inventory_sha256",
        "displaced_container_assignment_sha256",
        "native_baseline_feasibility_sha256",
    )
    for key in digest_fields:
        if not sha(metrics.get(key)):
            problem(f"metric {key} is not a lowercase SHA-256")
    if not sha(evidence.get("comparison_manifest_sha256")):
        problem("comparison manifest is not digest-bound")

    budget = metrics.get("byte_budget")
    candidate_bytes = metrics.get("candidate_artifact_bytes")
    displaced_bytes = metrics.get("displaced_container_artifact_bytes")
    if not positive_int(budget):
        problem("byte budget is not a positive integer")
    if not positive_int(candidate_bytes) or (
        positive_int(budget) and candidate_bytes > budget
    ):
        problem("candidate artifact is not within the exact byte budget")
    if not positive_int(displaced_bytes) or (
        positive_int(budget) and displaced_bytes > budget
    ):
        problem("displaced artifact is not within the exact byte budget")

    parity_floor = finite(metrics.get("parity_floor"))
    minimum = finite(metrics.get("min_conservative_ratio"))
    tolerance = finite(metrics.get("predeclared_tolerance"))
    predeclared_at = metrics.get("predeclared_at")
    if (
        tolerance is None
        or not 0 <= tolerance <= 0.05
        or parity_floor is None
        or not math.isclose(parity_floor, 1.0 - tolerance, abs_tol=1e-12)
    ):
        problem("parity floor does not equal one minus predeclared tolerance")
    if tolerance and (
        not isinstance(metrics.get("tolerance_rationale"), str)
        or not str(metrics.get("tolerance_rationale")).strip()
    ):
        problem("nonzero performance tolerance has no rationale")
    if not isinstance(predeclared_at, str) or not predeclared_at:
        problem("performance comparison has no predeclaration timestamp")
    if parity_floor is None or not 0.95 <= parity_floor <= 1.0:
        problem("parity floor is outside the predeclared 0-5% tolerance range")
    if minimum is None or minimum <= 0 or (
        parity_floor is not None and minimum < parity_floor
    ):
        problem("conservative parity ratio does not clear the declared floor")

    coverage = metrics.get("coverage")
    cell_count = metrics.get("cell_count")
    cell_ids: list[str] = []
    expected_verdict_tuples: set[tuple[object, ...]] | None = None
    if not isinstance(coverage, Mapping):
        problem("missing exact comparison-matrix coverage")
    else:
        concurrencies = coverage.get("concurrencies")
        shipped_max = coverage.get("shipped_max_concurrency")
        valid_concurrencies = (
            isinstance(concurrencies, list)
            and bool(concurrencies)
            and all(positive_int(value) for value in concurrencies)
            and concurrencies == sorted(set(concurrencies))
            and positive_int(shipped_max)
            and shipped_max >= 8
            and shipped_max == max(concurrencies)
            and set(concurrencies) == {1, 2, 4, 8, shipped_max}
        )
        expected_count = 6 * len(concurrencies) + 2 if valid_concurrencies else None
        expected_coverage = (
            coverage.get("phases") == ["prefill", "decode", "mixed"]
            and coverage.get("chunked_prefill") == [False, True]
            and coverage.get("decode_modes") == ["plain", "shipped"]
            and coverage.get("nonzero_input_distribution") is True
            and positive_int(shipped_max)
            and valid_concurrencies
            and coverage.get("configuration_tuple_count") == expected_count
        )
        if not expected_coverage:
            problem("comparison matrix does not equal the release Cartesian product")
        if valid_concurrencies:
            expected_verdict_tuples = set()
            for concurrency in concurrencies:
                for chunked in (False, True):
                    expected_verdict_tuples.add(
                        ("prefill", concurrency, chunked, None)
                    )
                    expected_verdict_tuples.add(
                        ("mixed", concurrency, chunked, None)
                    )
                    expected_verdict_tuples.add(
                        ("decode", concurrency, chunked, "shipped")
                    )
            for chunked in (False, True):
                expected_verdict_tuples.add(("decode", 1, chunked, "plain"))
        if (
            not positive_int(cell_count)
            or cell_count != coverage.get("configuration_tuple_count")
        ):
            problem("cell count differs from exact matrix coverage")

    verdicts = metrics.get("cell_verdicts")
    verdict_minima: list[float] = []
    verdict_tuples: set[tuple[object, ...]] = set()
    verdict_ids: list[str] = []
    if not isinstance(verdicts, list) or not positive_int(cell_count) or len(
        verdicts if isinstance(verdicts, list) else []
    ) != cell_count:
        problem("cell verdict ledger does not exactly cover the matrix")
    else:
        for index, verdict in enumerate(verdicts):
            if not isinstance(verdict, Mapping):
                problem(f"cell verdict {index} is malformed")
                continue
            phase = verdict.get("phase")
            concurrency = verdict.get("concurrency")
            chunked = verdict.get("chunked_prefill")
            decode_mode = verdict.get("decode_mode")
            verdict_id = verdict.get("id")
            rows = verdict.get("metrics")
            row_minima: list[float] = []
            if not isinstance(verdict_id, str) or not verdict_id:
                problem(f"cell verdict {index} has no id")
            else:
                verdict_ids.append(verdict_id)
            configuration_valid = (
                isinstance(phase, str)
                and phase in {"prefill", "decode", "mixed"}
                and positive_int(concurrency)
                and isinstance(chunked, bool)
                and (
                    (
                        phase == "decode"
                        and isinstance(decode_mode, str)
                        and decode_mode in {"plain", "shipped"}
                    )
                    or (phase != "decode" and decode_mode is None)
                )
            )
            if not configuration_valid:
                problem(f"cell verdict {index} has an invalid configuration")
            else:
                verdict_tuples.add((phase, concurrency, chunked, decode_mode))
            if not isinstance(rows, list) or not rows:
                problem(f"cell verdict {index} has no metric ratios")
                continue
            expected_metrics = (
                CB_PERFORMANCE_PHASE_METRICS.get(phase, ())
                if isinstance(phase, str)
                else ()
            )
            observed_metrics = [
                (row.get("metric"), row.get("direction"))
                if isinstance(row, Mapping)
                else (None, None)
                for row in rows
            ]
            if observed_metrics != list(expected_metrics):
                problem(
                    f"cell verdict {index} metric names or directions differ "
                    "from the phase contract"
            )
            for row_index, row in enumerate(rows):
                ratios = row.get("paired_ratios") if isinstance(row, Mapping) else None
                paired_values = row.get("paired_values") if isinstance(
                    row, Mapping
                ) else None
                if (
                    not isinstance(ratios, list)
                    or not isinstance(paired_values, list)
                    or len(ratios) < 3
                    or len(paired_values) != len(ratios)
                    or any(
                        not isinstance(pair, list) or len(pair) != 2
                        for pair in paired_values
                    )
                ):
                    problem(
                        f"cell verdict {index} metric {row_index} lacks exact "
                        "paired raw values and ratios"
                    )
                    continue
                parsed_pairs = [
                    (finite(pair[0]), finite(pair[1]))
                    for pair in paired_values
                ]
                parsed_ratios = [finite(value) for value in ratios]
                if (
                    any(
                        candidate is None
                        or candidate <= 0
                        or baseline is None
                        or baseline <= 0
                        for candidate, baseline in parsed_pairs
                    )
                    or any(value is None or value <= 0 for value in parsed_ratios)
                ):
                    problem(
                        f"cell verdict {index} metric {row_index} has invalid "
                        "raw values or ratios"
                    )
                    continue
                numeric_pairs = [
                    (float(candidate), float(baseline))
                    for candidate, baseline in parsed_pairs
                    if candidate is not None and baseline is not None
                ]
                numeric = [
                    (
                        candidate / baseline
                        if row.get("direction") == "candidate/baseline"
                        else baseline / candidate
                    )
                    for candidate, baseline in numeric_pairs
                ]
                if any(
                    declared is None
                    or not math.isclose(
                        computed,
                        declared,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for computed, declared in zip(
                        numeric, parsed_ratios, strict=True
                    )
                ):
                    problem(
                        f"cell verdict {index} metric {row_index} ratios differ "
                        "from paired raw values"
                    )
                ordered = sorted(numeric)
                position = (len(ordered) - 1) * 0.05
                lower = math.floor(position)
                upper = math.ceil(position)
                p05 = ordered[lower] if lower == upper else (
                    ordered[lower] * (1 - (position - lower))
                    + ordered[upper] * (position - lower)
                )
                declared_p05 = finite(row.get("conservative_p05_ratio"))
                declared_median = finite(row.get("median_ratio"))
                if (
                    declared_p05 is None
                    or declared_median is None
                    or not math.isclose(
                        p05, declared_p05, rel_tol=1e-12, abs_tol=1e-12
                    )
                    or not math.isclose(
                        statistics.median(numeric),
                        declared_median,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    problem(f"cell verdict {index} metric {row_index} statistics differ")
                row_minima.append(p05)
            if row_minima:
                calculated = min(row_minima)
                declared_minimum = finite(verdict.get("min_conservative_ratio"))
                if (
                    declared_minimum is None
                    or not math.isclose(
                        calculated,
                        declared_minimum,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or verdict.get("passed") is not (
                        parity_floor is not None and calculated >= parity_floor
                    )
                ):
                    problem(f"cell verdict {index} decision is inconsistent")
                verdict_minima.append(calculated)
        if len(verdict_ids) != len(set(verdict_ids)):
            problem("cell verdict ids are missing or not unique")
        if len(verdict_tuples) != len(verdicts):
            problem("cell verdict configurations are not unique")
        if (
            expected_verdict_tuples is None
            or verdict_tuples != expected_verdict_tuples
        ):
            problem("cell verdict configurations do not equal the release matrix")
        try:
            matrix_digest = hashlib.sha256(json.dumps(
                verdicts, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            matrix_digest = None
        if matrix_digest != metrics.get("matrix_digest"):
            problem("matrix digest differs from cell verdict ledger")
        if not verdict_minima or minimum is None or not math.isclose(
            min(verdict_minima), minimum, rel_tol=1e-12, abs_tol=1e-12
        ):
            problem("global conservative ratio differs from cell verdicts")

    paired = evidence.get("paired_reports")
    report_digests: list[str] = []
    if not isinstance(paired, list) or not positive_int(cell_count) or len(
        paired if isinstance(paired, list) else []
    ) != cell_count:
        problem("paired report evidence does not exactly cover every matrix cell")
    elif isinstance(paired, list):
        for index, row in enumerate(paired):
            if not isinstance(row, Mapping):
                problem(f"paired report {index} is malformed")
                continue
            expected_report_keys = {
                "cell_id", "candidate_sha256", "baseline_sha256",
                "candidate_pre_serve_manifest_sha256",
                "candidate_post_serve_manifest_sha256",
                "baseline_pre_serve_manifest_sha256",
                "baseline_post_serve_manifest_sha256",
                "candidate_serve_session_id", "baseline_serve_session_id",
                "performance_stack_fingerprint",
            }
            if set(row) != expected_report_keys:
                problem(f"paired report {index} fields are not closed and exact")
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id:
                problem(f"paired report {index} has no cell id")
            else:
                cell_ids.append(cell_id)
            for arm in ("candidate_sha256", "baseline_sha256"):
                value = row.get(arm)
                if not sha(value):
                    problem(f"paired report {index} {arm} is not digest-bound")
                else:
                    report_digests.append(value)
            manifest_digests = [
                row.get(key)
                for key in (
                    "candidate_pre_serve_manifest_sha256",
                    "candidate_post_serve_manifest_sha256",
                    "baseline_pre_serve_manifest_sha256",
                    "baseline_post_serve_manifest_sha256",
                )
            ]
            if any(not sha(value) for value in manifest_digests) or len(
                set(manifest_digests)
            ) != 4:
                problem(
                    f"paired report {index} does not bind four distinct "
                    "pre/post arm manifests"
                )
            candidate_session = row.get("candidate_serve_session_id")
            baseline_session = row.get("baseline_serve_session_id")
            if (
                not sha(candidate_session)
                or not sha(baseline_session)
                or candidate_session == baseline_session
                or not sha(row.get("performance_stack_fingerprint"))
            ):
                problem(
                    f"paired report {index} has invalid serve-session or "
                    "matched-stack identities"
                )
        if len(cell_ids) != len(set(cell_ids)):
            problem("paired report cell ids are not unique")
        if cell_ids != verdict_ids:
            problem("paired report ids/order differ from the cell verdict ledger")
        if len(report_digests) != len(set(report_digests)):
            problem("a benchmark report is reused across arms or matrix cells")

    telemetry = evidence.get("telemetry_sha256")
    telemetry_kinds: set[str] = set()
    telemetry_digests: list[str] = []
    if not isinstance(telemetry, list) or len(telemetry) != len(
        CB_PERFORMANCE_TELEMETRY_KINDS
    ):
        problem("telemetry evidence does not cover all four required classes")
    else:
        for index, row in enumerate(telemetry):
            if not isinstance(row, Mapping):
                problem(f"telemetry evidence {index} is malformed")
                continue
            kind = row.get("kind")
            if not isinstance(kind, str):
                problem(f"telemetry evidence {index} has no kind")
            else:
                telemetry_kinds.add(kind)
            value = row.get("sha256")
            if not sha(value):
                problem(f"telemetry evidence {index} is not digest-bound")
            else:
                telemetry_digests.append(value)
            if row.get("cell_ids") != cell_ids:
                problem(f"telemetry evidence {index} does not cover ordered cells")
        if telemetry_kinds != set(CB_PERFORMANCE_TELEMETRY_KINDS):
            problem("telemetry evidence names the wrong required classes")
        if len(telemetry_digests) != len(set(telemetry_digests)):
            problem("a telemetry payload is reused across required classes")

    displaced = evidence.get("displaced_container")
    displaced_digest: str | None = None
    if not isinstance(displaced, Mapping):
        problem("missing displaced-container eligibility proof")
    else:
        try:
            displaced_digest = hashlib.sha256(json.dumps(
                displaced,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            problem("displaced-container proof is not canonical JSON")
        if displaced.get("schema") != DISPLACED_CONTAINER_ELIGIBILITY_SCHEMA:
            problem("unsupported displaced-container eligibility schema")
        if displaced.get("status") != "eligible":
            problem("displaced container is not eligible")
        for key in ("reason", "mechanism", "model_id", "cost_currency"):
            if not isinstance(displaced.get(key), str) or not str(
                displaced.get(key)
            ).strip():
                problem(f"displaced container {key} is not explicit")
        for key in (
            "model_sha",
            "artifact_inventory_sha256",
            "assignment_sha256",
            "layer_config_sha256",
            "weight_content_manifest_sha256",
            "shipcard_sha256",
            "assignment_receipt_sha256",
        ):
            if not sha(displaced.get(key)):
                problem(f"displaced container {key} is not digest-bound")
        if displaced.get("byte_budget") != budget:
            problem("displaced container binds a different byte budget")
        if displaced.get("artifact_bytes") != displaced_bytes:
            problem("displaced-container byte count differs from metrics")
        runtime = displaced.get("gridbook_runtime")
        if not isinstance(runtime, Mapping) or runtime != {
            "commit": metrics.get("gridbook_runtime_commit"),
            "version": metrics.get("gridbook_runtime_version"),
        }:
            problem("displaced container binds a different Gridbook runtime")
        endpoints = displaced.get("endpoint_record_sha256")
        if not isinstance(endpoints, Mapping) or set(endpoints) != {
            "native_export.eager", "native_export.graph"
        } or any(not sha(value) for value in (
            endpoints.values() if isinstance(endpoints, Mapping) else ()
        )):
            problem("displaced container lacks both digest-bound endpoint records")
        source = displaced.get("source_model_identity")
        if (
            not isinstance(source, Mapping)
            or source.get("schema") != "prismaquant.streamed_model.identity.v1"
            or (
                source.get("resolved_commit") is not None
                and (
                    not isinstance(source.get("resolved_commit"), str)
                    or not source.get("resolved_commit")
                )
            )
            or not sha(source.get("content_sha256"))
            or not positive_int(source.get("checkpoint_shards"))
            or not positive_int(source.get("checkpoint_tensors"))
        ):
            problem("displaced container lacks full source-model identity")

    expected_displaced = metrics.get("displaced_container_eligibility_sha256")
    if displaced_digest != expected_displaced or evidence.get(
        "displaced_container_eligibility_sha256"
    ) != expected_displaced:
        problem("displaced-container proof digest is inconsistent")
    duplicated = {
        "model_sha": "displaced_container_model_sha",
        "artifact_inventory_sha256": "displaced_container_inventory_sha256",
        "artifact_bytes": "displaced_container_artifact_bytes",
        "assignment_sha256": "displaced_container_assignment_sha256",
        "reason": "displaced_container_reason",
    }
    if isinstance(displaced, Mapping):
        if displaced.get("model_sha") == record.get("model_sha"):
            problem("displaced container is identical to the candidate model")
        for proof_key, metric_key in duplicated.items():
            if displaced.get(proof_key) != metrics.get(metric_key):
                problem(f"displaced container {proof_key} differs from metrics")
    if evidence.get("native_baseline_feasibility_sha256") != metrics.get(
        "native_baseline_feasibility_sha256"
    ):
        problem("native infeasibility proof digest is inconsistent")

    if model_dir is not None:
        try:
            root = Path(model_dir)
            quant_config = json.loads((root / "quant_config.json").read_text(
                encoding="utf-8"
            ))
            provenance = quant_config.get("provenance")
            inventory = provenance.get("artifact_inventory") if isinstance(
                provenance, Mapping
            ) else None
            if not isinstance(inventory, Mapping):
                raise ValueError("quant_config has no finalized artifact inventory")
            inventory_digest = hashlib.sha256(json.dumps(
                inventory,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest()
            if inventory_digest != metrics.get("candidate_inventory_sha256"):
                raise ValueError("candidate inventory digest differs from metrics")
            if inventory.get("export_directory_bytes") != candidate_bytes:
                raise ValueError("candidate inventory byte count differs from metrics")
            if inventory.get("whole_artifact_budget_bytes") != budget:
                raise ValueError("candidate inventory binds a different budget")
            source_identity = provenance.get("source_model_identity") if isinstance(
                provenance, Mapping
            ) else None
            if not isinstance(displaced, Mapping) or displaced.get(
                "source_model_identity"
            ) != source_identity:
                raise ValueError(
                    "candidate and displaced source-model identities differ"
                )
            if inventory.get("schema") != (
                "prismaquant.cb_export_artifact_inventory.v1"
            ) or inventory.get("scope") != "all_regular_files_recursive":
                raise ValueError("candidate inventory schema or scope is invalid")
            declared_files = inventory.get("file_bytes")
            if not isinstance(declared_files, Mapping) or not declared_files:
                raise ValueError("candidate inventory file ledger is empty")
            declared: dict[str, int] = {}
            for name, size in declared_files.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or name.startswith("/")
                    or ".." in Path(name).parts
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                ):
                    raise ValueError("candidate inventory file ledger is malformed")
                declared[name] = size
            observed: dict[str, int] = {}
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError(
                        f"candidate artifact contains symlink {path.relative_to(root)}"
                    )
                if path.is_file():
                    observed[path.relative_to(root).as_posix()] = int(
                        path.stat().st_size
                    )
            if observed != declared:
                raise ValueError(
                    "candidate recursive files differ from finalized inventory"
                )
            if sum(observed.values()) != candidate_bytes or sum(
                observed.values()
            ) > budget:
                raise ValueError(
                    "candidate recursive byte sum differs from metrics or exceeds budget"
                )
        except Exception as exc:
            problem(f"candidate artifact inventory replay failed — {exc}")
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


def _is_dspark_cb_sidecar_artifact(
    model_dir: str | os.PathLike | None,
) -> bool:
    """Whether on-disk provenance identifies a physical DSpark sidecar.

    The target artifact's ``dspark_source_overlay`` is deliberately not a
    sidecar identity: target-only publication remains valid when no paired
    DSpark claim was made.  Presence of a non-null sidecar value is enough to
    retain the obligation, including when the value itself is malformed.
    """
    if model_dir is None:
        return False
    quant_path = Path(model_dir) / "quant_config.json"
    if not quant_path.is_file():
        return False
    try:
        payload = json.loads(quant_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance") if isinstance(
        payload, Mapping
    ) else None
    return (
        isinstance(provenance, Mapping)
        and "dspark_cb_sidecar" in provenance
        and provenance.get("dspark_cb_sidecar") is not None
    )


def _cb_body_excludes_source_namespaces(
    model_dir: str | os.PathLike | None,
) -> bool:
    """Whether this CB artifact was exported with source namespaces excluded.

    A namespace-excluded export is a *body-only* artifact: the excluded
    construction units (DSv4: the ``mtp.`` draft stack) live in a separate
    sidecar artifact by construction.  Read off ``quant_config.json``
    ``provenance.excluded_namespaces``, which ``compute_model_sha`` binds with
    the artifact bytes, so this scoping cannot be flipped without breaking
    artifact identity.  Fail-closed: absent or unreadable provenance is a
    full-source artifact, so a release can never shed an obligation by hiding
    or corrupting its quant config.
    """
    if model_dir is None:
        return False
    quant_path = Path(model_dir) / "quant_config.json"
    if not quant_path.is_file():
        return False
    try:
        payload = json.loads(quant_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance") if isinstance(
        payload, Mapping
    ) else None
    if not isinstance(provenance, Mapping):
        return False
    excluded = provenance.get("excluded_namespaces")
    return isinstance(excluded, (list, tuple)) and any(
        isinstance(ns, str) and ns for ns in excluded
    )


#: `model_type` values whose gold lane IS the DSv4-Flash release contract.
_DSV4_MODEL_TYPES = frozenset({"deepseek_v4"})
_DSV4_ARCHITECTURE_RE = re.compile(r"^DeepseekV4[A-Za-z0-9_]*$")

#: Lane identifiers.  A CB *lane* is the (architecture, serving stack) pair a
#: gate replays against.  `validate_cb_endpoint` owns the per-lane serving
#: constants; this module owns only which lane an artifact is on, because the
#: answer is read off `config.json` -- already bound by `compute_model_sha` as
#: `config_sha`, so the lane cannot be flipped without breaking identity.
CB_LANE_DSV4_FLASH = "dsv4_flash"
CB_LANE_GRIDBOOK_GENERIC = "gridbook_generic"


def cb_serving_lane(model_dir: str | os.PathLike | None) -> str:
    """Which CB serving lane's contract this artifact is held to.

    Fail-closed: an unreadable architecture is the DSv4 lane, so a DSv4
    release can never shed its contract by hiding or corrupting its config.
    """
    return (
        CB_LANE_DSV4_FLASH
        if _is_dsv4_gridbook_artifact(model_dir)
        else CB_LANE_GRIDBOOK_GENERIC
    )


def _is_dsv4_gridbook_artifact(
    model_dir: str | os.PathLike | None,
) -> bool:
    """Whether the DSv4-Flash gold contract is this artifact's contract.

    `_verify_dsv4_gridbook_gold_contract` replays ONE lane's release contract:
    it pins `tokenizer_mode="deepseek_v4"`, `max_logprobs=248_320` (the DSv4
    vocabulary), `moe_backend="marlin"`, an in-process `LLM(**kwargs)` shape,
    and an exact 8x512/4088-position workload. It was introduced by `80a1c25`
    ("ship: harden DSv4Flash AURA Gridbook release") and gated on
    `is_gridbook_cb`, because DSv4 was then the only CB lane with gold slots --
    a proxy that was true when written and is not a property of CB.

    Qwen3.8-27B CB is the first CB artifact off that lane, and the proxy makes
    its `gold.kl`/`gold.ppl` unfillable by construction: no Qwen artifact can
    ever report `tokenizer_mode="deepseek_v4"`. A gate no correct artifact can
    pass is a measurement gap, not a missing measurement (principle 1) -- the
    same class as the `TOKENIZER_IDENTITY_SHA256` module constant that already
    refused this lane's gold-inputs tool. The generic gold requirements (finite
    slot metric, exact serve fingerprint, producer commit, position counts,
    `score_positions=all`) were never CB-specific and still apply to every lane.

    Read off `config.json`, which `compute_model_sha` already binds as
    `config_sha`: the lane cannot be flipped without breaking artifact
    identity. Fail-closed -- an artifact whose architecture cannot be read is
    treated as on-lane, so a DSv4 release can never shed the contract by
    hiding or corrupting its config.
    """
    if model_dir is None:
        return True
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        return True
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True
    if not isinstance(payload, Mapping):
        return True
    model_type = payload.get("model_type")
    architectures = payload.get("architectures")
    if isinstance(model_type, str) and model_type in _DSV4_MODEL_TYPES:
        return True
    if isinstance(architectures, (list, tuple)) and any(
        isinstance(name, str) and _DSV4_ARCHITECTURE_RE.fullmatch(name)
        for name in architectures
    ):
        return True
    # Positively identified as some other architecture.
    if isinstance(model_type, str) and model_type:
        return False
    if isinstance(architectures, (list, tuple)) and any(
        isinstance(name, str) and name for name in architectures
    ):
        return False
    return True


def _is_gridbook_card(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> bool:
    """Resolve CB identity without trusting only the mutable receipt.

    A shipcard is intentionally mutated as gates close, so publication with an
    on-disk artifact also reads ``quant_config.json``.  Removing the CB slot and
    changing ``build.quant_method`` in the receipt therefore cannot erase the
    performance obligation.
    """
    if (card.get("build") or {}).get("quant_method") == "gridbook":
        return True
    slots = card.get("slots")
    if isinstance(slots, Mapping) and any(slot in slots for slot in CB_REQUIRED_SLOTS):
        return True
    if model_dir is None:
        return False
    quant_path = Path(model_dir) / "quant_config.json"
    if not quant_path.is_file():
        return False
    try:
        payload = json.loads(quant_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and payload.get("quant_method") == "gridbook"


def _is_rtx4090_fp8_cb_artifact(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> bool:
    """Resolve the strict Ada obligation from both card and artifact bytes.

    ``shipcard.json`` is mutable by design, so its build block or slot map can
    never be the sole authority.  The FP8-only exporter writes the distinct
    top-level ``format: fp8_cb`` token into identity-bound
    ``quant_config.json``; that token is sufficient to make the hardware slot
    mandatory even if the policy stamp is malformed or missing.
    """

    slots = card.get("slots")
    if isinstance(slots, Mapping) and any(
        slot in slots for slot in RTX4090_REQUIRED_SLOTS
    ):
        return True
    build = card.get("build")
    if isinstance(build, Mapping) and (
        build.get("producer_policy") == "qwen38_27b_rtx4090_fp8_cb"
        or build.get("serving_profile") == "qwen38_rtx4090_fp8_cb"
    ):
        return True
    if model_dir is None:
        return False
    quant_path = Path(model_dir) / "quant_config.json"
    if not quant_path.is_file():
        return False
    try:
        payload = json.loads(quant_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    if payload.get("format") == "fp8_cb":
        return True
    provenance = payload.get("provenance")
    policy = provenance.get("producer_policy") if isinstance(
        provenance, Mapping
    ) else None
    return isinstance(policy, Mapping) and policy.get(
        "id"
    ) == "qwen38_27b_rtx4090_fp8_cb"


def _is_rtx4090_validation_only_artifact(
    card: Mapping[str, Any],
    model_dir: str | os.PathLike | None,
) -> bool:
    """Read the immutable producer stamp that categorically forbids release."""

    build = card.get("build")
    if isinstance(build, Mapping) and build.get(
        "artifact_disposition"
    ) == "UNRELEASABLE_VALIDATION_ONLY":
        return True
    if model_dir is None:
        return False
    quant_path = Path(model_dir) / "quant_config.json"
    try:
        payload = json.loads(quant_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance") if isinstance(
        payload, Mapping
    ) else None
    policy = provenance.get("producer_policy") if isinstance(
        provenance, Mapping
    ) else None
    from .rtx4090_qwen38_policy import is_rtx4090_validation_only_policy

    return is_rtx4090_validation_only_policy(policy)


def required_slots(
    card: Mapping[str, Any],
    *,
    model_dir: str | os.PathLike | None = None,
) -> tuple[str, ...]:
    """Return every slot default verification must replay.

    Container-required slots are followed by recognized optional claims that
    are present and non-null.  The physical DSpark sidecar additionally makes
    ``mtp.dspark`` mandatory from on-disk provenance, so deleting or nulling
    the mutable shipcard claim cannot erase the draft artifact's obligation.
    """
    required: list[str] = list(REQUIRED_SLOTS)
    if _is_gridbook_card(card, model_dir=model_dir):
        # `perf.matched_budget_parity` is a DSv4 *release argument*: it proves
        # the candidate displaces one exact eligible container at the same byte
        # budget.  Its verifier structurally requires that container
        # (`displaced_container_model_sha` and four sibling digests), so an
        # artifact that displaces nothing -- a net-new size class, which is
        # what the Qwen3.8 5080 CB artifact is -- cannot fill it by
        # construction.  Requiring it there is the same gate-no-correct-
        # artifact-can-pass defect as the DSv4 gold contract, so it is scoped
        # to the same lane.  A card that nonetheless CARRIES the slot is still
        # held to it below, via OPTIONAL_SLOTS-style presence.
        #
        # Body-only scoping (2026-08-18): the parity verifier's census walks
        # every construction unit of the displaced container, and a
        # namespace-excluded export moves the excluded units into a separate
        # sidecar artifact by construction (DSv4: `mtp.` lives in the
        # 4.597 GB DSpark draft).  No body-only artifact can produce that
        # census -- the same defect class, scoped by the same principle, and
        # keyed on `quant_config.json` provenance that `compute_model_sha`
        # binds.  An opened-but-null parity slot on such a card is an honest
        # record that the demand was scoped out, not a waiver channel; a
        # NON-null parity claim is still replayed via the presence branch.
        if (
            cb_serving_lane(model_dir) == CB_LANE_DSV4_FLASH
            and not _cb_body_excludes_source_namespaces(model_dir)
        ):
            required.extend(CB_REQUIRED_SLOTS)
        else:
            slots_present = card.get("slots")
            if isinstance(slots_present, Mapping):
                required.extend(
                    slot
                    for slot in CB_REQUIRED_SLOTS
                    if slots_present.get(slot) is not None
                )
    if _is_rtx4090_fp8_cb_artifact(card, model_dir=model_dir):
        required.extend(RTX4090_REQUIRED_SLOTS)
    slots = card.get("slots")
    if isinstance(slots, Mapping):
        required.extend(
            slot for slot in OPTIONAL_SLOTS if slots.get(slot) is not None
        )
    if _is_dspark_cb_sidecar_artifact(model_dir):
        required.append("mtp.dspark")
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
