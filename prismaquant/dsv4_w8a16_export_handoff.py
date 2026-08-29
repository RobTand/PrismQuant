"""Fail-closed pre-export gate for the fixed DSv4-Flash W8A16 release.

This module does not launch an exporter and never writes an artifact.  It
turns the reviewed readmission publication and every immutable exporter input
into one machine-readable handoff receipt immediately before the GPU job is
started.  The release driver may consume stdout only after this function
returns successfully.
"""
from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import stat
from typing import Any

from prismaquant.allocator_candidates import (
    ROUTE_GRIDBOOK_FP8_SOURCE_W8A16,
    ROUTE_PENDING_PASSTHROUGH_FORMATS,
    SOURCE_PASSTHROUGH_CONTRACTS,
)
from prismaquant.anchored_cost import AURA_CURRENCY
from prismaquant.cb_anchored_cost import (
    CB_ANCHORED_COST_SCHEMA,
    CB_ARTIFACT_PUBLISH_SCHEMA,
)
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.dsv4_aura_cb_reprice import (
    DSV4_W8A16_APPROVED_BUDGET_BYTES,
    DSV4_TOTAL_UNITS,
    DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256,
    DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256,
    DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256,
    DSV4_W8A16_APPROVED_SELECTION,
    DSV4_W8A16_APPROVED_SELECTION_SHA256,
    DSV4_W8A16_READMISSION_SCHEMA,
)
from prismaquant.format_registry import get_format
from prismaquant.cluster_transport import (
    ClusterTransportError,
    canonical_json_bytes,
    read_regular_file_nofollow,
)
from prismaquant.layer_config import canonicalize_assignment
from prismaquant.nvfp4_cb_footprint import (
    assignment_serialization_sha256,
    cb_serialization_metadata_from_assignment_payload,
)


DSV4_W8A16_EXPORT_HANDOFF_SCHEMA = (
    "prismaquant.dsv4_w8a16.export_handoff.v3"
)
DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA = (
    "prismaquant.dsv4_w8a16.export_source_closure.v2"
)
DSV4_W8A16_LEGACY_COMPATIBILITY_SCHEMA = (
    "prismaquant.dsv4_w8a16.legacy_compatibility.v1"
)
DSV4_W8A16_EXPORT_SOURCE_CLOSURE_PIN_SCHEMA = (
    "prismaquant.dsv4_w8a16.export_source_closure_pin.v1"
)
_SOURCE_CLOSURE_PIN_NAME = "dsv4_w8a16_legacy_closure_pin.json"
_LEGACY_GRIDBOOK_RUNTIME = {
    "schema": "prismaquant.gridbook_runtime_pin.v3",
    "repository": "https://github.com/RobTand/gridbook.git",
    "commit": "e992e5980c96333a48149f96392d6cff56ae9e3f",
    "version": "0.8.5",
    "version_is_release": True,
    "runtime_contract_schema": "gridbook.runtime-contract.v3",
    "required_abi_features": {
        "routed_moe_per_role_codebook_lut": 1,
        "source_fp8_block128_w8a16": 1,
    },
    "serving_route": ROUTE_GRIDBOOK_FP8_SOURCE_W8A16,
}
_LEGACY_W8A16_FORMAT_COUNTS = {
    "FP8_BLOCK_UE8M0_SOURCE": 120,
    "FP8_CB_K28": 6144,
    "FP8_CB_K36": 3,
    "FP8_CB_K44": 36,
    "FP8_CB_K48": 142,
    "NVFP4_CB_K16": 1536,
    "NVFP4_CB_K18": 25344,
}
_LEGACY_W8A16_ROUTED_K28_QNAMES = frozenset(
    f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
    for layer in (18, 19, 22, 33, 34, 35, 36, 39)
    for expert in range(256)
    for projection in ("down_proj", "gate_proj", "up_proj")
)
_LEGACY_W8A16_DENSE_K36_QNAMES = frozenset({
    "model.layers.0.self_attn.wq_b",
    "model.layers.1.self_attn.wq_b",
    "model.layers.24.self_attn.wq_b",
})
_LEGACY_W8A16_PUBLICATION_IDENTITY_SHA256 = (
    "75574db23d9171a7808efc94a7bbb6d25ba59d48e274a11fe28e7147b07fd829"
)
_LEGACY_W8A16_PUBLISHED_SHA256 = {
    "cb_col_weights.pkl": (
        "df045bde786f7d092e501bfa856984243106a13f05594f4a11fe30270fb09379"
    ),
    "layer_config.json": (
        "39070ffd7bb0f22353bc1857ab129a4aceb581265612a6ae74836a438a890618"
    ),
    "pareto.knees.json": (
        "906d68c2b9d073bf304ba9088a3f3b586c9a0cc76996e917c723fa771059f2c0"
    ),
    "selection.json": (
        "c116981675551d8cc88d138437ddca3b72b629e1734a4c10c1b1cb4895644813"
    ),
}
_LEGACY_W8A16_SOURCE_IDENTITY_FILE_SHA256 = (
    "a1f27124c6356a33cad17b7d64155f509eeb6ee03d6044d22ac14a1cec996e76"
)
_LEGACY_W8A16_SOURCE_CONTENT_SHA256 = (
    "50d0e40217a3feece2afc5e80a32e0a1a119ddcf8d40cffecc05e108128da642"
)
_LEGACY_W8A16_SOURCE_SHARD_COUNT = 48
_LEGACY_W8A16_BUNDLE_FILE_SHA256 = (
    "dfeee6d592402dc2ff63ad43118537a8ebdae155e05109f651a1b7922ff643a5"
)
_LEGACY_W8A16_BUNDLE_CONTENT_SHA256 = (
    "4b0d551aa041876c1976736202960f137f492942311633a7f623a506a8abb17f"
)
_PUBLISH_MANIFEST = ".anchored_publish.json"
_PUBLISHED_FILES = frozenset({
    "layer_config.json",
    "selection.json",
    "pareto.knees.json",
    "cb_col_weights.pkl",
})
# The active source authority is the complete descriptor-stable package
# inventory in ``dsv4_w8a16_legacy_closure_pin.json`` below.  No hand-picked
# subset of exporter files is accepted.

class W8A16ExportHandoffError(RuntimeError):
    """The exact reviewed DSv4 W8A16 export handoff is not intact."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_bytes(path, where="SHA-256 input")).hexdigest()


def _read_bytes(path: Path, *, where: str) -> bytes:
    try:
        return read_regular_file_nofollow(path, where=where)
    except ClusterTransportError as exc:
        raise W8A16ExportHandoffError(f"{where} is unreadable: {path}") from exc


@contextmanager
def _sealed_memory_snapshot(payload: bytes, *, where: str):
    """Expose exact held bytes to a path-only parser without a path reopen."""

    required = (
        "F_ADD_SEALS",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_GROW",
        "F_SEAL_WRITE",
    )
    if not hasattr(os, "memfd_create") or any(
        not hasattr(fcntl, name) for name in required
    ):
        raise W8A16ExportHandoffError(
            f"{where} cannot create a sealed in-memory snapshot"
        )
    descriptor = os.memfd_create(
        "prismaquant-w8a16-snapshot",
        getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:  # pragma: no cover - defensive syscall guard
                raise W8A16ExportHandoffError(
                    f"{where} in-memory snapshot write made no progress"
                )
            offset += written
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        yield Path(f"/proc/self/fd/{descriptor}")
    except W8A16ExportHandoffError:
        raise
    except OSError as exc:
        raise W8A16ExportHandoffError(
            f"{where} in-memory snapshot failed"
        ) from exc
    finally:
        os.close(descriptor)


def _json_object_from_bytes(payload: bytes, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise W8A16ExportHandoffError(f"{where} is unreadable") from exc
    if not isinstance(value, dict):
        raise W8A16ExportHandoffError(f"{where} is not a JSON object")
    return value


def _verify_publication(
    root: Path,
) -> tuple[Path, dict[str, Any], dict[str, str], dict[str, bytes]]:
    directory_fd, absolute_root = _open_real_directory(
        root, where="readmission publication",
    )
    try:
        observed_names = set(os.listdir(directory_fd))
        expected_names = _PUBLISHED_FILES | {_PUBLISH_MANIFEST}
        if observed_names != expected_names:
            raise W8A16ExportHandoffError(
                "readmission publication file set differs: "
                f"missing={sorted(expected_names - observed_names)}, "
                f"extra={sorted(observed_names - expected_names)}"
            )
        payloads = {
            name: _read_regular_at(
                directory_fd,
                name,
                where=f"published {name}",
            )[0]
            for name in sorted(expected_names)
        }
        if set(os.listdir(directory_fd)) != observed_names:
            raise W8A16ExportHandoffError(
                "readmission publication file set changed while reading"
            )
    finally:
        os.close(directory_fd)
    manifest = _json_object_from_bytes(
        payloads[_PUBLISH_MANIFEST],
        where="readmission publication manifest",
    )
    identity = manifest.get("identity")
    outputs = manifest.get("outputs")
    if not isinstance(identity, Mapping) or not isinstance(outputs, Mapping):
        raise W8A16ExportHandoffError(
            "readmission publication lacks identity/output mappings"
        )
    try:
        identity_sha256 = canonical_json_sha256(
            identity, where="DSv4 W8A16 export publication identity"
        )
    except (TypeError, ValueError) as exc:
        raise W8A16ExportHandoffError(
            "readmission publication identity is non-canonical"
        ) from exc
    if (
        manifest.get("schema") != CB_ARTIFACT_PUBLISH_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("identity_sha256") != identity_sha256
        or identity.get("schema") != CB_ARTIFACT_PUBLISH_SCHEMA
        or outputs != identity.get("outputs")
        or set(map(str, outputs)) != _PUBLISHED_FILES
    ):
        raise W8A16ExportHandoffError(
            "readmission publication is incomplete, unbound, or has the "
            "wrong output set"
        )
    observed: dict[str, str] = {}
    for name in sorted(_PUBLISHED_FILES):
        descriptor = outputs.get(name)
        if not isinstance(descriptor, Mapping):
            raise W8A16ExportHandoffError(
                f"published {name} has no checksum descriptor"
            )
        payload = payloads[name]
        digest = hashlib.sha256(payload).hexdigest()
        actual = {"size_bytes": len(payload), "sha256": digest}
        if descriptor != actual:
            raise W8A16ExportHandoffError(
                f"published {name} differs from its atomic manifest"
            )
        observed[name] = digest
    return absolute_root, manifest, observed, payloads


def _selection_contract(selection: Mapping[str, object]) -> dict[str, object]:
    whole = selection.get("whole_artifact_budget")
    if not isinstance(whole, Mapping):
        raise W8A16ExportHandoffError(
            "readmitted selection lacks whole-artifact accounting"
        )
    observed = {
        "budget_bytes": selection.get("budget_bytes"),
        "chosen_achieved_bits": selection.get("chosen_achieved_bits"),
        "predicted_dloss": selection.get("predicted_dloss"),
        "selection_tensor_payload_bytes": whole.get(
            "selection_tensor_payload_bytes"
        ),
        "selection_whole_artifact_upper_bound_bytes": whole.get(
            "selection_whole_artifact_upper_bound_bytes"
        ),
    }
    if observed != DSV4_W8A16_APPROVED_SELECTION:
        raise W8A16ExportHandoffError(
            f"readmitted selection metrics differ from approval: {observed}"
        )
    return observed


def _open_real_directory(path: Path, *, where: str) -> tuple[int, Path]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise W8A16ExportHandoffError(
            f"{where} cannot be traversed without following links"
        )
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            before = os.stat(
                component, dir_fd=descriptor, follow_symlinks=False,
            )
            if not stat.S_ISDIR(before.st_mode):
                raise W8A16ExportHandoffError(
                    f"{where} ancestry is not a real directory: {absolute}"
                )
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                os.close(child)
                raise W8A16ExportHandoffError(
                    f"{where} ancestry changed while opening: {absolute}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_at(
    parent_fd: int,
    name: str,
    *,
    where: str,
) -> tuple[bytes, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise W8A16ExportHandoffError(f"{where} is not a regular file")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise W8A16ExportHandoffError(
                f"{where} changed while being opened"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(opened, key) != getattr(after, key) for key in stable)
            or any(getattr(opened, key) != getattr(current, key) for key in stable)
            or len(payload) != after.st_size
        ):
            raise W8A16ExportHandoffError(
                f"{where} changed while being read"
            )
        return payload, after
    finally:
        os.close(descriptor)


def _runtime_inventory(
    package_fd: int,
    *,
    pin_payload_sha256: str,
) -> dict[str, str]:
    observed: dict[str, str] = {}
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")

    def walk(directory_fd: int, relative_parts: tuple[str, ...]) -> None:
        opened_directory = os.fstat(directory_fd)
        names = sorted(os.listdir(directory_fd))
        advertised: dict[str, os.stat_result] = {}
        for name in names:
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\x00" in name
            ):
                raise W8A16ExportHandoffError(
                    "frozen runtime closure contains an unsafe entry name"
                )
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            advertised[name] = before
            relative_path = Path("prismaquant", *relative_parts, name).as_posix()
            if stat.S_ISLNK(before.st_mode):
                raise W8A16ExportHandoffError(
                    f"frozen runtime closure contains a symlink: {relative_path}"
                )
            if stat.S_ISDIR(before.st_mode):
                if name == "__pycache__":
                    continue
                if name == ".git" or name.startswith(".tmp") or name.endswith(".tmp"):
                    raise W8A16ExportHandoffError(
                        "frozen runtime closure contains a temporary/control "
                        f"directory: {relative_path}"
                    )
                child = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory_fd,
                )
                try:
                    child_opened = os.fstat(child)
                    if (
                        not stat.S_ISDIR(child_opened.st_mode)
                        or (child_opened.st_dev, child_opened.st_ino)
                        != (before.st_dev, before.st_ino)
                    ):
                        raise W8A16ExportHandoffError(
                            "frozen runtime directory changed while opening: "
                            f"{relative_path}"
                        )
                    walk(child, (*relative_parts, name))
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(before.st_mode):
                raise W8A16ExportHandoffError(
                    f"frozen runtime closure contains a special file: {relative_path}"
                )
            if name.endswith((".pyc", ".pyo")):
                raise W8A16ExportHandoffError(
                    f"frozen runtime closure contains bytecode: {relative_path}"
                )
            payload, _after = _read_regular_at(
                directory_fd, name, where=f"frozen runtime closure {relative_path}",
            )
            digest = hashlib.sha256(payload).hexdigest()
            if relative_path == f"prismaquant/{_SOURCE_CLOSURE_PIN_NAME}":
                if digest != pin_payload_sha256:
                    raise W8A16ExportHandoffError(
                        "source-closure pin changed during runtime inventory"
                    )
                continue
            observed[relative_path] = digest

        if sorted(os.listdir(directory_fd)) != names:
            raise W8A16ExportHandoffError(
                "frozen runtime directory entries changed during inventory"
            )
        for name, before in advertised.items():
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if any(getattr(before, key) != getattr(current, key) for key in stable):
                raise W8A16ExportHandoffError(
                    "frozen runtime entry changed during inventory: "
                    + Path("prismaquant", *relative_parts, name).as_posix()
                )
        after_directory = os.fstat(directory_fd)
        if any(
            getattr(opened_directory, key) != getattr(after_directory, key)
            for key in ("st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns")
        ):
            raise W8A16ExportHandoffError(
                "frozen runtime directory changed during inventory"
            )

    walk(package_fd, ())
    return observed


def _verify_frozen_export_source_closure_from_package_fd(
    package_fd: int,
) -> dict[str, object]:
    pin_payload, _pin_stat = _read_regular_at(
        package_fd,
        _SOURCE_CLOSURE_PIN_NAME,
        where="DSv4 W8A16 source-closure pin manifest",
    )
    try:
        pin = json.loads(pin_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W8A16ExportHandoffError(
            "DSv4 W8A16 source-closure pin manifest is invalid JSON"
        ) from exc
    if (
        not isinstance(pin, dict)
        or canonical_json_bytes(pin) + b"\n" != pin_payload
        or set(pin) != {"schema", "files_sha256", "identity_sha256"}
        or pin.get("schema")
        != DSV4_W8A16_EXPORT_SOURCE_CLOSURE_PIN_SCHEMA
        or not isinstance(pin.get("files_sha256"), dict)
    ):
        raise W8A16ExportHandoffError(
            "DSv4 W8A16 source-closure pin manifest is noncanonical"
        )
    pin_identity = pin.get("identity_sha256")
    expected_pin_identity = canonical_json_sha256(
        {key: pin[key] for key in ("schema", "files_sha256")},
        where="DSv4 W8A16 source-closure pin identity",
    )
    if pin_identity != expected_pin_identity:
        raise W8A16ExportHandoffError(
            "DSv4 W8A16 source-closure pin identity differs"
        )
    observed = _runtime_inventory(
        package_fd,
        pin_payload_sha256=hashlib.sha256(pin_payload).hexdigest(),
    )

    expected_files = pin["files_sha256"]
    assert isinstance(expected_files, dict)
    if any(
        not isinstance(relative, str)
        or not relative.startswith("prismaquant/")
        or relative == f"prismaquant/{_SOURCE_CLOSURE_PIN_NAME}"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for relative, digest in expected_files.items()
    ):
        raise W8A16ExportHandoffError(
            "DSv4 W8A16 source-closure pin ledger is invalid"
        )
    missing = sorted(set(expected_files) - set(observed))
    extra = sorted(set(observed) - set(expected_files))
    drift = sorted(
        relative for relative in set(observed) & set(expected_files)
        if observed[relative] != expected_files[relative]
    )
    if missing or extra or drift:
        raise W8A16ExportHandoffError(
            "frozen exporter/source runtime closure changed: "
            f"missing={missing[:16]}, extra={extra[:16]}, drift={drift[:16]}"
        )
    closure: dict[str, object] = {
        "schema": DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA,
        "file_count": len(observed),
        "tree_sha256": canonical_json_sha256(
            observed, where="DSv4 W8A16 runtime-tree ledger",
        ),
        "pin_manifest_sha256": hashlib.sha256(pin_payload).hexdigest(),
        "pin_identity_sha256": pin_identity,
    }
    closure["identity_sha256"] = canonical_json_sha256(
        closure,
        where="DSv4 W8A16 exporter/source closure",
    )
    return closure


def _verify_frozen_export_source_closure(
    repo_root: Path,
) -> dict[str, object]:
    package_fd, _package_root = _open_real_directory(
        Path(repo_root) / "prismaquant", where="PrismaQuant runtime package",
    )
    try:
        return _verify_frozen_export_source_closure_from_package_fd(
            package_fd,
        )
    finally:
        os.close(package_fd)


def _verify_runtime_contract() -> dict[str, object]:
    contract = SOURCE_PASSTHROUGH_CONTRACTS["FP8_BLOCK_UE8M0_SOURCE"]
    if (
        contract.serving_route != ROUTE_GRIDBOOK_FP8_SOURCE_W8A16
        or not contract.route_backed
        or "FP8_BLOCK_UE8M0_SOURCE" in ROUTE_PENDING_PASSTHROUGH_FORMATS
    ):
        raise W8A16ExportHandoffError(
            "FP8 block W8A16 is not backed by the exact released Gridbook "
            "runtime contract"
        )
    block = get_format("FP8_BLOCK_UE8M0_SOURCE")
    direct = get_format("MXFP8_UE8M0_G32")
    if (
        block.act_quant_changes_input
        or not direct.act_quant_changes_input
        or direct.act_bits != 8
    ):
        raise W8A16ExportHandoffError(
            "source W8A16 and direct group-32 W8A8 contracts have collapsed"
        )
    return dict(_LEGACY_GRIDBOOK_RUNTIME)


def _verify_bundle(
    bundle_path: Path, layer_payload: Mapping[str, object]
) -> dict[str, object]:
    bundle_payload = _read_bytes(
        bundle_path, where="immutable codebook bundle",
    )
    context_stamp, _tensor_stamps = (
        cb_serialization_metadata_from_assignment_payload(layer_payload)
    )
    if not isinstance(context_stamp, Mapping):
        raise W8A16ExportHandoffError(
            "readmitted assignment lacks a CB serialization stamp"
        )
    try:
        from prismaquant.cb_learned_bundle import load_bundle

        with _sealed_memory_snapshot(
            bundle_payload, where="immutable codebook bundle",
        ) as snapshot_path:
            bundle = load_bundle(snapshot_path)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            f"immutable codebook bundle is invalid: {bundle_path}"
        ) from exc
    if (
        context_stamp.get("codebook_content_sha256")
        != bundle.codebook_content_digests
        or context_stamp.get("codebook_source_by_format")
        != bundle.codebook_source_by_format
    ):
        raise W8A16ExportHandoffError(
            "codebook bundle bytes/source map differ from the assignment stamp"
        )
    return {
        "path": str(Path(os.path.abspath(os.fspath(bundle_path)))),
        "file_sha256": hashlib.sha256(bundle_payload).hexdigest(),
        "bundle_content_sha256": bundle.bundle_content_sha256,
        "codebook_count": len(bundle.codebook_content_digests),
    }


def _col_weights_content_sha256(payload: bytes) -> str:
    """Bind the exact approved pickle to the tensor mapping the exporter sees."""

    try:
        import torch

        value = pickle.loads(payload)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "approved W8A16 column-weight payload is unreadable"
        ) from exc
    if not isinstance(value, Mapping):
        raise W8A16ExportHandoffError(
            "approved W8A16 column weights are not a mapping"
        )
    digest = hashlib.sha256()
    for raw_name in sorted(value, key=str):
        name = str(raw_name)
        if raw_name != name or not name:
            raise W8A16ExportHandoffError(
                "approved W8A16 column weights have a noncanonical key"
            )
        try:
            tensor = torch.as_tensor(value[raw_name]).to(torch.float32).cpu()
            if tensor.ndim != 1 or not tensor.is_contiguous():
                tensor = tensor.contiguous()
            tensor_bytes = tensor.numpy().tobytes()
        except Exception as exc:
            raise W8A16ExportHandoffError(
                f"approved W8A16 column weights are invalid at {name!r}"
            ) from exc
        digest.update(name.encode("utf-8"))
        digest.update(tensor_bytes)
    return digest.hexdigest()


def legacy_w8a16_assignment_compatibility(
    assignment: Mapping[str, str],
) -> dict[str, object]:
    """Validate and describe only the sealed W8A16 producer exceptions."""

    observed_counts = dict(sorted(Counter(assignment.values()).items()))
    if observed_counts != _LEGACY_W8A16_FORMAT_COUNTS:
        raise W8A16ExportHandoffError(
            "W8A16 assignment format counts differ from the sealed ledger: "
            f"observed={observed_counts}"
        )
    routed_k28 = frozenset(
        qname for qname, fmt in assignment.items() if fmt == "FP8_CB_K28"
    )
    dense_k36 = frozenset(
        qname for qname, fmt in assignment.items() if fmt == "FP8_CB_K36"
    )
    if routed_k28 != _LEGACY_W8A16_ROUTED_K28_QNAMES:
        raise W8A16ExportHandoffError(
            "W8A16 routed FP8_CB_K28 cells differ from the sealed 6,144-row ledger"
        )
    if dense_k36 != _LEGACY_W8A16_DENSE_K36_QNAMES:
        raise W8A16ExportHandoffError(
            "W8A16 dense FP8_CB_K36 cells differ from the sealed three-row ledger"
        )
    exception_map = {
        "routed_fp8_cb_k28": {
            "format": "FP8_CB_K28",
            "count": len(routed_k28),
            "qnames_sha256": canonical_json_sha256(
                sorted(routed_k28), where="W8A16 routed K28 qnames",
            ),
        },
        "dense_fp8_cb_k36": {
            "format": "FP8_CB_K36",
            "count": len(dense_k36),
            "qnames_sha256": canonical_json_sha256(
                sorted(dense_k36), where="W8A16 dense K36 qnames",
            ),
        },
    }
    result: dict[str, object] = {
        "schema": DSV4_W8A16_LEGACY_COMPATIBILITY_SCHEMA,
        "assignment_sha256": assignment_serialization_sha256(assignment),
        "format_counts": observed_counts,
        "exception_map": exception_map,
    }
    result["identity_sha256"] = canonical_json_sha256(
        result, where="DSv4 W8A16 legacy compatibility ledger",
    )
    return result


def verify_dsv4_w8a16_export_handoff(
    *,
    publication_dir: str | Path,
    approved_raw_publication_dir: str | Path,
    source_model_dir: str | Path,
    source_identity_path: str | Path,
    codebook_bundle_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify the fixed release handoff without mutating any input or output."""

    publication = Path(publication_dir)
    approved_raw = Path(approved_raw_publication_dir)
    output = Path(os.path.abspath(os.fspath(output_path)))
    if not output.name:
        raise W8A16ExportHandoffError("export output must name a child path")
    output_parent_fd, output_parent = _open_real_directory(
        output.parent, where="export output parent",
    )
    try:
        try:
            os.stat(output.name, dir_fd=output_parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise W8A16ExportHandoffError(
                f"export output already exists; refusing clobber: {output}"
            )
    finally:
        os.close(output_parent_fd)
    output = output_parent / output.name

    (
        publication_root,
        manifest,
        published_sha256,
        published_payloads,
    ) = _verify_publication(publication)
    (
        approved_raw_root,
        _raw_manifest,
        raw_sha256,
        raw_payloads,
    ) = _verify_publication(approved_raw)
    if (
        manifest.get("identity_sha256")
        != _LEGACY_W8A16_PUBLICATION_IDENTITY_SHA256
        or published_sha256 != _LEGACY_W8A16_PUBLISHED_SHA256
    ):
        raise W8A16ExportHandoffError(
            "readmitted publication differs from the canonical packed-alias-v2 "
            "W8A16 handoff"
        )
    expected_raw = {
        "layer_config.json": DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256,
        "selection.json": DSV4_W8A16_APPROVED_SELECTION_SHA256,
        "cb_col_weights.pkl": DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256,
    }
    for name, expected in expected_raw.items():
        if raw_sha256[name] != expected:
            raise W8A16ExportHandoffError(
                f"approved raw publication changed at {name}"
            )

    layer_path = publication_root / "layer_config.json"
    selection_path = publication_root / "selection.json"
    layer_payload = _json_object_from_bytes(
        published_payloads["layer_config.json"],
        where="readmitted layer config",
    )
    selection = _json_object_from_bytes(
        published_payloads["selection.json"],
        where="readmitted selection",
    )
    try:
        raw_layer_payload = _json_object_from_bytes(
            raw_payloads["layer_config.json"],
            where="approved raw layer config",
        )
        raw_assignment = canonicalize_assignment(raw_layer_payload)
        assignment = canonicalize_assignment(layer_payload)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "approved/readmitted assignments are unreadable"
        ) from exc
    assignment_sha256 = assignment_serialization_sha256(assignment)
    if (
        assignment != raw_assignment
        or len(assignment) != DSV4_TOTAL_UNITS
        or assignment_sha256 != DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256
    ):
        raise W8A16ExportHandoffError(
            "readmitted full qname/format map differs from the approved "
            "33,325-unit assignment"
        )
    compatibility = legacy_w8a16_assignment_compatibility(assignment)
    metrics = _selection_contract(selection)
    whole = selection["whole_artifact_budget"]
    if whole.get("selection_assignment_sha256") != assignment_sha256:
        raise W8A16ExportHandoffError(
            "readmitted whole-artifact accounting binds another assignment"
        )

    metadata = layer_payload.get("__prismaquant__")
    stamp = (
        metadata.get("aura_cb_reprice")
        if isinstance(metadata, Mapping) else None
    )
    readmission = stamp.get("cpu_replay") if isinstance(stamp, Mapping) else None
    attestation = (
        stamp.get("approved_raw_assignment_attestation")
        if isinstance(stamp, Mapping) else None
    )
    if (
        not isinstance(stamp, Mapping)
        or stamp.get("schema") != CB_ANCHORED_COST_SCHEMA
        or stamp.get("cost_currency") != AURA_CURRENCY
        or stamp.get("budget_bytes") != DSV4_W8A16_APPROVED_BUDGET_BYTES
        or selection.get("aura_cb_reprice") != stamp
        or selection.get("cost_currency") != AURA_CURRENCY
        or selection.get("feasible") is not True
        or not isinstance(readmission, Mapping)
        or readmission.get("schema") != DSV4_W8A16_READMISSION_SCHEMA
        or readmission.get("measurement_invoked") is not False
        or readmission.get("no_gpu_measurement_or_render") is not True
        or not isinstance(attestation, Mapping)
        or attestation.get("full_qname_format_map_equal") is not True
        or attestation.get("approved_assignment_sha256") != assignment_sha256
        or attestation.get("readmitted_assignment_sha256") != assignment_sha256
        or attestation.get("selection") != metrics
    ):
        raise W8A16ExportHandoffError(
            "publication lacks one matching CPU-only W8A16 readmission proof"
        )
    raw_stamp = readmission.get("approved_raw_publication")
    if (
        not isinstance(raw_stamp, Mapping)
        or Path(str(raw_stamp.get("publication", ""))) != approved_raw_root
        or raw_stamp.get("assignment_sha256") != assignment_sha256
        or raw_stamp.get("selection") != metrics
        or raw_stamp.get("layer_config_sha256")
        != DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256
        or raw_stamp.get("selection_sha256")
        != DSV4_W8A16_APPROVED_SELECTION_SHA256
        or raw_stamp.get("cb_col_weights_sha256")
        != DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256
    ):
        raise W8A16ExportHandoffError(
            "readmission provenance does not bind the exact approved raw "
            "publication"
        )

    runtime = _verify_runtime_contract()
    stamped_runtime = readmission.get("gridbook_runtime_pin")
    if stamped_runtime != {key: runtime[key] for key in (
        "schema", "repository", "commit", "version", "version_is_release",
        "runtime_contract_schema", "required_abi_features",
    )}:
        raise W8A16ExportHandoffError(
            "readmission was produced under a different Gridbook runtime pin"
        )

    from prismaquant.cost_streaming import (
        compact_streamed_model_identity,
        validate_cached_streamed_model_identity,
    )
    source_identity_absolute = Path(
        os.path.abspath(os.fspath(source_identity_path))
    )
    source_identity_payload = _read_bytes(
        source_identity_absolute, where="sealed source identity",
    )
    source_directory_fd, source_model_root = _open_real_directory(
        Path(source_model_dir), where="sealed source checkpoint",
    )
    try:
        with _sealed_memory_snapshot(
            source_identity_payload, where="sealed source identity",
        ) as source_identity_snapshot:
            source_identity = validate_cached_streamed_model_identity(
                Path(f"/proc/self/fd/{source_directory_fd}"),
                source_identity_snapshot,
                require_complete_checkpoint=True,
                cached_source_model=source_model_root,
            )
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "source checkpoint no longer matches its complete content identity"
        ) from exc
    finally:
        os.close(source_directory_fd)
    source_identity_file_sha256 = hashlib.sha256(
        source_identity_payload
    ).hexdigest()
    try:
        compact_source_identity = compact_streamed_model_identity(
            source_identity,
            where="sealed DSv4 W8A16 source identity",
        )
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "sealed source checkpoint identity cannot be compacted"
        ) from exc
    if (
        source_identity_file_sha256
        != _LEGACY_W8A16_SOURCE_IDENTITY_FILE_SHA256
        or source_identity.get("content_sha256")
        != _LEGACY_W8A16_SOURCE_CONTENT_SHA256
        or len(source_identity.get("shards", ()))
        != _LEGACY_W8A16_SOURCE_SHARD_COUNT
    ):
        raise W8A16ExportHandoffError(
            "source checkpoint differs from the sealed DSv4 W8A16 identity"
        )

    from prismaquant.dspark_source_metadata import (
        discover_dspark_source_overlay_from_artifact,
    )
    try:
        overlay = discover_dspark_source_overlay_from_artifact(source_model_dir)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "DSpark source-header overlay is invalid"
        ) from exc
    routed_formats = set(assignment.values())
    if overlay is not None:
        routed_formats.update(overlay.construction_units.values())
    pending = sorted(routed_formats & ROUTE_PENDING_PASSTHROUGH_FORMATS)
    if pending:
        raise W8A16ExportHandoffError(
            f"release assignment still uses route-pending formats: {pending}"
        )

    bundle = _verify_bundle(Path(codebook_bundle_path), layer_payload)
    if (
        bundle.get("file_sha256") != _LEGACY_W8A16_BUNDLE_FILE_SHA256
        or bundle.get("bundle_content_sha256")
        != _LEGACY_W8A16_BUNDLE_CONTENT_SHA256
    ):
        raise W8A16ExportHandoffError(
            "codebook bundle differs from the sealed DSv4 W8A16 identity"
        )
    col_weights_path = publication_root / "cb_col_weights.pkl"
    col_weights_payload = published_payloads["cb_col_weights.pkl"]
    col_weights = {
        "path": str(col_weights_path),
        "file_sha256": hashlib.sha256(col_weights_payload).hexdigest(),
        "content_sha256": _col_weights_content_sha256(col_weights_payload),
    }
    root = (
        Path(repo_root) if repo_root is not None
        else Path(__file__).resolve(strict=True).parent.parent
    )
    frozen = _verify_frozen_export_source_closure(root)
    receipt: dict[str, object] = {
        "schema": DSV4_W8A16_EXPORT_HANDOFF_SCHEMA,
        "publication": str(publication_root),
        "publication_identity_sha256": manifest["identity_sha256"],
        "published_sha256": published_sha256,
        "approved_raw_publication": str(approved_raw_root),
        "assignment_sha256": assignment_sha256,
        "unit_count": len(assignment),
        "fp8_block_w8a16_count": _LEGACY_W8A16_FORMAT_COUNTS[
            "FP8_BLOCK_UE8M0_SOURCE"
        ],
        "legacy_compatibility": compatibility,
        "selection": metrics,
        "source_checkpoint": {
            "model_path": str(source_model_root),
            "identity_path": str(source_identity_absolute),
            "identity_file_sha256": source_identity_file_sha256,
            "content_sha256": source_identity["content_sha256"],
            "shard_count": len(source_identity["shards"]),
            "compact_identity": compact_source_identity,
        },
        "col_weights": col_weights,
        "codebook_bundle": bundle,
        "gridbook_runtime_pin": runtime,
        "frozen_export_source_closure": frozen,
        "output_path": str(output),
        "output_absent": True,
    }
    receipt["identity_sha256"] = canonical_json_sha256(
        receipt, where="DSv4 W8A16 export handoff receipt",
    )
    return receipt


__all__ = [
    "DSV4_W8A16_EXPORT_HANDOFF_SCHEMA",
    "DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA",
    "DSV4_W8A16_LEGACY_COMPATIBILITY_SCHEMA",
    "W8A16ExportHandoffError",
    "legacy_w8a16_assignment_compatibility",
    "verify_dsv4_w8a16_export_handoff",
]
