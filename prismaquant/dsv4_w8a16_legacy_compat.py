"""Receipt-derived producer authority for one sealed DSv4 W8A16 export.

This module does not widen a public format family.  It replays the canonical
handoff against every live input and returns authority only for the 6,147
historical cells whose formats are outside today's producer ladder: the exact
6,144 routed K28 rows and the exact three dense K36 rows.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import torch

from prismaquant.cluster_transport import (
    ClusterTransportError,
    canonical_json_bytes,
    read_regular_file_nofollow,
)
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.dsv4_w8a16_export_handoff import (
    DSV4_W8A16_EXPORT_HANDOFF_SCHEMA,
    DSV4_W8A16_LEGACY_COMPATIBILITY_SCHEMA,
    W8A16ExportHandoffError,
    _sealed_memory_snapshot,
    legacy_w8a16_assignment_compatibility,
    verify_dsv4_w8a16_export_handoff,
)
from prismaquant.layer_config import canonicalize_assignment
from prismaquant.nvfp4_cb_footprint import assignment_serialization_sha256
from prismaquant.schemas import validate_layer_config_payload


_ROUTED_STACK = re.compile(
    r"model[.]layers[.](?P<layer>18|19|22|33|34|35|36|39)"
    r"[.]mlp[.]experts[.](?P<projection>gate_up_proj|down_proj)\Z"
)


class W8A16LegacyCompatibilityError(RuntimeError):
    """The caller did not prove the exact sealed compatibility authority."""


def col_weights_content_sha256(
    col_weights: Mapping[str, object],
) -> str:
    digest = hashlib.sha256()
    for raw_name in sorted(col_weights, key=str):
        name = str(raw_name)
        if raw_name != name or not name:
            raise W8A16LegacyCompatibilityError(
                "column weights have a noncanonical key"
            )
        try:
            tensor = torch.as_tensor(col_weights[raw_name]).to(
                torch.float32
            ).cpu().contiguous()
        except Exception as exc:
            raise W8A16LegacyCompatibilityError(
                f"column weights are invalid at {name!r}"
            ) from exc
        if tensor.ndim != 1:
            raise W8A16LegacyCompatibilityError(
                f"column weights must be one-dimensional at {name!r}"
            )
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_receipt(
    value: Mapping[str, object] | str | Path,
) -> dict[str, object]:
    if isinstance(value, Mapping):
        try:
            receipt = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise W8A16LegacyCompatibilityError(
                "W8A16 handoff receipt is not canonical JSON data"
            ) from exc
    else:
        path = Path(value)
        try:
            payload = read_regular_file_nofollow(
                path, where="W8A16 handoff receipt",
            )
        except ClusterTransportError as exc:
            raise W8A16LegacyCompatibilityError(
                f"W8A16 handoff receipt is unreadable: {path}"
            ) from exc
        try:
            receipt = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise W8A16LegacyCompatibilityError(
                "W8A16 handoff receipt is invalid JSON"
            ) from exc
        if canonical_json_bytes(receipt) + b"\n" != payload:
            raise W8A16LegacyCompatibilityError(
                "W8A16 handoff receipt must use canonical JSON file encoding"
            )
    identity = receipt.get("identity_sha256")
    if (
        receipt.get("schema") != DSV4_W8A16_EXPORT_HANDOFF_SCHEMA
        or not isinstance(identity, str)
        or canonical_json_sha256(
            {key: receipt[key] for key in receipt if key != "identity_sha256"},
            where="W8A16 handoff receipt identity",
        ) != identity
    ):
        raise W8A16LegacyCompatibilityError(
            "W8A16 handoff receipt identity is invalid"
        )
    return receipt


@dataclass(frozen=True)
class DSV4W8A16LegacyCompatibility:
    receipt_identity_sha256: str
    publication_identity_sha256: str
    assignment_sha256: str
    output_path: str
    layer_config_path: str
    layer_config_file_sha256: str
    source_identity_file_sha256: str
    source_content_sha256: str
    source_model_identity: Mapping[str, object]
    codebook_bundle_file_sha256: str
    codebook_bundle_content_sha256: str
    runtime_pin_sha256: str
    runtime_closure_identity_sha256: str
    col_weights_content_sha256: str
    ledger: Mapping[str, object]
    _routed_k28_qnames: frozenset[str]
    _dense_k36_qnames: frozenset[str]
    _codebook_bundle_payload: bytes = field(repr=False, compare=False)

    def allows(self, qname: str, fmt: str) -> bool:
        return (
            fmt == "FP8_CB_K28" and qname in self._routed_k28_qnames
        ) or (
            fmt == "FP8_CB_K36" and qname in self._dense_k36_qnames
        )

    def allows_group(
        self,
        qname: str,
        fmt: str,
        members: object = (),
    ) -> bool:
        if self.allows(qname, fmt):
            return True
        if fmt != "FP8_CB_K28" or not isinstance(members, Mapping):
            return False
        match = _ROUTED_STACK.fullmatch(str(qname))
        if match is None:
            return False
        layer = int(match.group("layer"))
        projections = (
            ("gate_proj", "up_proj")
            if match.group("projection") == "gate_up_proj"
            else ("down_proj",)
        )
        expected = frozenset(
            f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            for expert in range(256)
            for projection in projections
        )
        observed = tuple(str(member) for member in members.values())
        return (
            len(observed) == len(expected)
            and len(set(observed)) == len(observed)
            and frozenset(observed) == expected
            and expected <= self._routed_k28_qnames
        )

    def read_bound_layer_config(
        self,
        path: str | Path,
    ) -> tuple[dict[str, object], dict[str, str]]:
        candidate = Path(os.path.abspath(os.fspath(path)))
        if str(candidate) != self.layer_config_path:
            raise W8A16LegacyCompatibilityError(
                "exporter layer config crosses the sealed W8A16 assignment"
            )
        try:
            raw = read_regular_file_nofollow(
                candidate, where="sealed W8A16 layer config",
            )
        except ClusterTransportError as exc:
            raise W8A16LegacyCompatibilityError(
                "sealed W8A16 layer config is unreadable"
            ) from exc
        if hashlib.sha256(raw).hexdigest() != self.layer_config_file_sha256:
            raise W8A16LegacyCompatibilityError(
                "sealed W8A16 layer config bytes changed after handoff replay"
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
            validate_layer_config_payload(payload, str(candidate))
            assignment = canonicalize_assignment(payload)
        except Exception as exc:
            raise W8A16LegacyCompatibilityError(
                "sealed W8A16 layer config is invalid"
            ) from exc
        if assignment_serialization_sha256(assignment) != self.assignment_sha256:
            raise W8A16LegacyCompatibilityError(
                "sealed W8A16 assignment identity changed after handoff replay"
            )
        return dict(payload), assignment

    def open_bound_codebook_bundle(self):
        return _sealed_memory_snapshot(
            self._codebook_bundle_payload,
            where="sealed W8A16 codebook bundle",
        )

    def stamp(self) -> dict[str, object]:
        stamp: dict[str, object] = {
            "schema": DSV4_W8A16_LEGACY_COMPATIBILITY_SCHEMA,
            "handoff_receipt_identity_sha256": self.receipt_identity_sha256,
            "publication_identity_sha256": self.publication_identity_sha256,
            "assignment_sha256": self.assignment_sha256,
            "layer_config_file_sha256": self.layer_config_file_sha256,
            "output_path": self.output_path,
            "source_identity_file_sha256": self.source_identity_file_sha256,
            "source_content_sha256": self.source_content_sha256,
            "source_model_identity": dict(self.source_model_identity),
            "codebook_bundle_file_sha256": self.codebook_bundle_file_sha256,
            "codebook_bundle_content_sha256": (
                self.codebook_bundle_content_sha256
            ),
            "runtime_pin_sha256": self.runtime_pin_sha256,
            "runtime_closure_identity_sha256": (
                self.runtime_closure_identity_sha256
            ),
            "col_weights_content_sha256": self.col_weights_content_sha256,
            "exception_map": dict(self.ledger["exception_map"]),
        }
        stamp["identity_sha256"] = canonical_json_sha256(
            stamp, where="W8A16 artifact compatibility stamp",
        )
        return stamp


def derive_dsv4_w8a16_legacy_compatibility(
    handoff_receipt: Mapping[str, object] | str | Path,
    *,
    model_dir: str | Path,
    layer_config_path: str | Path,
    out_dir: str | Path,
    col_weights: Mapping[str, object],
    col_weights_path: str | Path,
    codebook_bundle_path: str | Path,
    repo_root: str | Path | None = None,
    subset_prefixes: object = None,
    reuse_prior: object = None,
    per_expert_config_path: object = None,
    dspark_cb_sidecar: bool = False,
    exclude_namespaces: object = None,
) -> DSV4W8A16LegacyCompatibility:
    """Replay the handoff; never derive authority from receipt fields alone."""

    if (
        subset_prefixes is not None
        or reuse_prior is not None
        or per_expert_config_path is not None
        or dspark_cb_sidecar
        or exclude_namespaces not in (None, (), [])
    ):
        raise W8A16LegacyCompatibilityError(
            "W8A16 legacy compatibility forbids subset, reuse, per-expert, "
            "DSpark, and namespace-exclusion modes"
        )
    receipt = _load_receipt(handoff_receipt)
    source = receipt.get("source_checkpoint")
    bundle = receipt.get("codebook_bundle")
    published_col_weights = receipt.get("col_weights")
    if not all(isinstance(item, Mapping) for item in (
        source, bundle, published_col_weights,
    )):
        raise W8A16LegacyCompatibilityError(
            "W8A16 handoff receipt lacks bound source/bundle/column weights"
        )
    assert isinstance(source, Mapping)
    assert isinstance(bundle, Mapping)
    assert isinstance(published_col_weights, Mapping)
    requested_paths = {
        "model": Path(os.path.abspath(os.fspath(model_dir))),
        "layer": Path(os.path.abspath(os.fspath(layer_config_path))),
        "out": Path(os.path.abspath(os.fspath(out_dir))),
        "col_weights": Path(os.path.abspath(os.fspath(col_weights_path))),
        "bundle": Path(os.path.abspath(os.fspath(codebook_bundle_path))),
    }
    try:
        expected_paths = {
            "model": Path(str(source["model_path"])),
            "layer": Path(str(receipt["publication"])) / "layer_config.json",
            "out": Path(str(receipt["output_path"])),
            "col_weights": Path(str(published_col_weights["path"])),
            "bundle": Path(str(bundle["path"])),
        }
    except KeyError as exc:
        raise W8A16LegacyCompatibilityError(
            "W8A16 handoff receipt lacks a bound export path"
        ) from exc
    if any(not path.is_absolute() for path in expected_paths.values()):
        raise W8A16LegacyCompatibilityError(
            "W8A16 handoff receipt contains a non-absolute export path"
        )
    if requested_paths != expected_paths:
        raise W8A16LegacyCompatibilityError(
            "W8A16 export inputs cross or differ from the handoff assignment"
        )
    try:
        replayed = verify_dsv4_w8a16_export_handoff(
            publication_dir=receipt["publication"],
            approved_raw_publication_dir=receipt["approved_raw_publication"],
            source_model_dir=requested_paths["model"],
            source_identity_path=source["identity_path"],
            codebook_bundle_path=requested_paths["bundle"],
            output_path=requested_paths["out"],
            repo_root=repo_root,
        )
    except (KeyError, OSError, W8A16ExportHandoffError) as exc:
        raise W8A16LegacyCompatibilityError(
            f"W8A16 handoff replay failed: {exc}"
        ) from exc
    if replayed != receipt:
        raise W8A16LegacyCompatibilityError(
            "W8A16 handoff receipt differs from independently replayed facts"
        )
    try:
        layer_bytes = read_regular_file_nofollow(
            requested_paths["layer"], where="sealed W8A16 layer config",
        )
        published_sha256 = receipt["published_sha256"]
        if (
            not isinstance(published_sha256, Mapping)
            or hashlib.sha256(layer_bytes).hexdigest()
            != published_sha256.get("layer_config.json")
        ):
            raise W8A16LegacyCompatibilityError(
                "sealed W8A16 layer config differs after handoff replay"
            )
        layer_payload = json.loads(layer_bytes.decode("utf-8"))
        validate_layer_config_payload(
            layer_payload, str(requested_paths["layer"]),
        )
        assignment = canonicalize_assignment(layer_payload)
    except W8A16LegacyCompatibilityError:
        raise
    except Exception as exc:
        raise W8A16LegacyCompatibilityError(
            "sealed W8A16 layer config is unreadable"
        ) from exc
    ledger = legacy_w8a16_assignment_compatibility(assignment)
    if ledger != receipt.get("legacy_compatibility"):
        raise W8A16LegacyCompatibilityError(
            "W8A16 exception ledger differs from the handoff receipt"
        )
    observed_col_weights_sha256 = col_weights_content_sha256(col_weights)
    if observed_col_weights_sha256 != published_col_weights.get("content_sha256"):
        raise W8A16LegacyCompatibilityError(
            "exporter column weights differ from the exact published payload"
        )
    try:
        bundle_payload = read_regular_file_nofollow(
            requested_paths["bundle"], where="sealed W8A16 codebook bundle",
        )
    except ClusterTransportError as exc:
        raise W8A16LegacyCompatibilityError(
            "sealed W8A16 codebook bundle is unreadable"
        ) from exc
    bundle_file_sha256 = hashlib.sha256(bundle_payload).hexdigest()
    if bundle_file_sha256 != bundle.get("file_sha256"):
        raise W8A16LegacyCompatibilityError(
            "sealed W8A16 codebook bundle changed after handoff replay"
        )
    runtime = receipt.get("gridbook_runtime_pin")
    closure = receipt.get("frozen_export_source_closure")
    compact_source = source.get("compact_identity")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(closure, Mapping)
        or not isinstance(compact_source, Mapping)
        or compact_source.get("content_sha256") != source.get("content_sha256")
    ):
        raise W8A16LegacyCompatibilityError(
            "W8A16 handoff receipt lacks source/runtime/closure identity"
        )
    return DSV4W8A16LegacyCompatibility(
        receipt_identity_sha256=str(receipt["identity_sha256"]),
        publication_identity_sha256=str(
            receipt["publication_identity_sha256"]
        ),
        assignment_sha256=str(receipt["assignment_sha256"]),
        output_path=str(receipt["output_path"]),
        layer_config_path=str(requested_paths["layer"]),
        layer_config_file_sha256=hashlib.sha256(layer_bytes).hexdigest(),
        source_identity_file_sha256=str(source["identity_file_sha256"]),
        source_content_sha256=str(source["content_sha256"]),
        source_model_identity=dict(compact_source),
        codebook_bundle_file_sha256=bundle_file_sha256,
        codebook_bundle_content_sha256=str(bundle["bundle_content_sha256"]),
        runtime_pin_sha256=canonical_json_sha256(
            runtime, where="sealed W8A16 Gridbook runtime pin",
        ),
        runtime_closure_identity_sha256=str(closure["identity_sha256"]),
        col_weights_content_sha256=observed_col_weights_sha256,
        ledger=ledger,
        _routed_k28_qnames=frozenset(
            qname for qname, fmt in assignment.items()
            if fmt == "FP8_CB_K28"
        ),
        _dense_k36_qnames=frozenset(
            qname for qname, fmt in assignment.items()
            if fmt == "FP8_CB_K36"
        ),
        _codebook_bundle_payload=bundle_payload,
    )


__all__ = [
    "DSV4W8A16LegacyCompatibility",
    "W8A16LegacyCompatibilityError",
    "col_weights_content_sha256",
    "derive_dsv4_w8a16_legacy_compatibility",
]
