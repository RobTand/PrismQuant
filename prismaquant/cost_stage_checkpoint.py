"""Identity-bound, durable per-unit journals for streamed cost stages.

The journal deliberately follows the production weight-cache pair-shard
contract: the manifest binds the complete run identity, shard names are a
SHA-256 of the semantic unit qname (never a list position), and every shard
is an atomically published, checksummed envelope.  Existing but unverifiable
state is an error; it is never silently overwritten or recomputed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
import pickle
from pathlib import Path


MANIFEST_SCHEMA = "prismaquant.cost_stage_checkpoint.manifest.v1"
UNIT_SCHEMA = "prismaquant.cost_stage_checkpoint.unit.v1"


def canonical_json(value: object, *, where: str) -> object:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not canonical JSON data") from exc
    return json.loads(encoded)


def canonical_json_sha256(value: object, *, where: str) -> str:
    canonical = canonical_json(value, where=where)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes durably; a crash leaves either the old or new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def unit_path(root: Path, qname: str) -> Path:
    digest = hashlib.sha256(str(qname).encode("utf-8")).hexdigest()
    return root / "units" / f"{digest}.pkl"


def _mismatch(
    stage: str,
    *,
    field: str,
    stored: object,
    expected: object,
) -> None:
    from prismaquant.production_weight_cache import identity_value_for_error

    raise RuntimeError(
        f"{stage} checkpoint identity mismatch at {field}: "
        f"stored={identity_value_for_error(stored)} "
        f"current={identity_value_for_error(expected)}; refusing reuse or "
        "recompute"
    )


def write_unit(
    root: Path,
    *,
    stage: str,
    qname: str,
    identity_sha256: str,
    state: Mapping[str, object],
) -> None:
    state_bytes = pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "schema": UNIT_SCHEMA,
        "stage": str(stage),
        "qname": str(qname),
        "identity_sha256": str(identity_sha256),
        "payload_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "payload": state_bytes,
    }
    atomic_write_bytes(
        unit_path(root, qname),
        pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL),
    )


def _load_unit(
    path: Path,
    *,
    stage: str,
    qname: str,
    identity_sha256: str,
) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"{stage} unit checkpoint {path} is corrupt for {qname}; "
            "refusing reuse or recompute"
        ) from exc
    if not isinstance(envelope, Mapping):
        raise RuntimeError(
            f"{stage} unit checkpoint {path} is not an envelope for {qname}; "
            "refusing reuse or recompute"
        )
    for field, expected in (
        ("schema", UNIT_SCHEMA),
        ("stage", str(stage)),
        ("qname", str(qname)),
        ("identity_sha256", str(identity_sha256)),
    ):
        if envelope.get(field) != expected:
            _mismatch(
                stage,
                field=f"unit[{qname}].{field}",
                stored=envelope.get(field),
                expected=expected,
            )
    payload = envelope.get("payload")
    if not isinstance(payload, bytes):
        raise RuntimeError(
            f"{stage} unit checkpoint {path} has no byte payload for {qname}; "
            "refusing reuse or recompute"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if envelope.get("payload_sha256") != digest:
        raise RuntimeError(
            f"{stage} unit checkpoint {path} payload_sha256 differs for "
            f"{qname}; refusing reuse or recompute"
        )
    try:
        state = pickle.loads(payload)
    except Exception as exc:
        raise RuntimeError(
            f"{stage} unit checkpoint {path} state is corrupt for {qname}; "
            "refusing reuse or recompute"
        ) from exc
    if not isinstance(state, Mapping):
        raise RuntimeError(
            f"{stage} unit checkpoint {path} state is not an object for "
            f"{qname}; refusing reuse or recompute"
        )
    return dict(state)


def prepare_journal(
    checkpoint_dir: str | Path,
    *,
    stage: str,
    resume: bool,
    identity: Mapping[str, object],
    qnames: Sequence[str],
    manifest_path: str | Path | None = None,
) -> tuple[Path, str, dict[str, dict[str, object]]]:
    """Create/validate a journal and return all exact completed unit states.

    File-oriented callers can retain their explicit manifest pathname while
    placing unit shards in ``checkpoint_dir``. The same manifest/unit schemas
    and refusal rules apply; directory-oriented callers keep ``manifest.json``.
    """
    root = Path(checkpoint_dir)
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"{stage} checkpoint path is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    canonical_identity = canonical_json(identity, where=f"{stage} identity")
    identity_sha256 = canonical_json_sha256(
        canonical_identity, where=f"{stage} identity"
    )
    manifest_path = Path(manifest_path) if manifest_path is not None else root / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise RuntimeError(
                f"{stage} checkpoint manifest already exists at "
                f"{manifest_path}; pass --resume to validate and reuse it"
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            _mismatch(
                stage,
                field="manifest_json",
                stored="<invalid>",
                expected="<valid canonical JSON>",
            )
            raise AssertionError("unreachable") from exc
        if not isinstance(manifest, Mapping):
            _mismatch(stage, field="manifest", stored=manifest, expected="<object>")
        if manifest.get("schema") != MANIFEST_SCHEMA:
            _mismatch(
                stage,
                field="manifest.schema",
                stored=manifest.get("schema"),
                expected=MANIFEST_SCHEMA,
            )
        if manifest.get("stage") != str(stage):
            _mismatch(
                stage,
                field="manifest.stage",
                stored=manifest.get("stage"),
                expected=str(stage),
            )
        from prismaquant.production_weight_cache import first_identity_difference

        difference = first_identity_difference(
            manifest.get("identity"), canonical_identity
        )
        if difference is not None:
            field, stored, expected = difference
            _mismatch(stage, field=field, stored=stored, expected=expected)
        if manifest.get("identity_sha256") != identity_sha256:
            _mismatch(
                stage,
                field="manifest.identity_sha256",
                stored=manifest.get("identity_sha256"),
                expected=identity_sha256,
            )
    else:
        existing = sorted((root / "units").glob("*.pkl"))
        if existing:
            raise RuntimeError(
                f"{stage} checkpoint units exist without a manifest; "
                f"refusing name-gated reuse or recompute. sample={existing[:8]}"
            )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "stage": str(stage),
            "identity_sha256": identity_sha256,
            "identity": canonical_identity,
            "units": [
                {
                    "qname": str(qname),
                    "file": str(unit_path(root, qname).relative_to(root)),
                }
                for qname in qnames
            ],
        }
        atomic_write_bytes(
            manifest_path,
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )

    expected_paths = {unit_path(root, qname): str(qname) for qname in qnames}
    unexpected = sorted(
        path for path in (root / "units").glob("*.pkl")
        if path not in expected_paths
    )
    if unexpected:
        _mismatch(
            stage,
            field="units.unexpected",
            stored=[path.name for path in unexpected[:8]],
            expected=[],
        )
    completed: dict[str, dict[str, object]] = {}
    for path, qname in expected_paths.items():
        if path.is_file():
            completed[qname] = _load_unit(
                path,
                stage=stage,
                qname=qname,
                identity_sha256=identity_sha256,
            )
    return root, identity_sha256, completed
