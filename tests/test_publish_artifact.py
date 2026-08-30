"""Publication is the blocking point (R16, ruled 2026-07-30).

The tests exercise both refusal policy and the two publication races that a
plain folder upload cannot close: local path/content mutation after verification
and a concurrent remote-head update after stale-file enumeration.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import types

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip(
        "requires a repo checkout (tools/ scripts)",
        allow_module_level=True,
    )

from prismaquant.shipcard import (
    CB_REQUIRED_SLOTS,
    GOLD_SLOTS,
    REQUIRED_SLOTS,
    RTX4090_REQUIRED_SLOTS,
    WEIGHT_CONTENT_MANIFEST_SCHEMA,
    build_shipcard,
    compute_model_sha,
    fill_slot,
    load_shipcard,
    make_record,
    write_shipcard,
)
from prismaquant.shard_layout import tensor_payload_identity
import tools.publish_artifact as publisher
from tools.publish_artifact import main as publish_cli


def _artifact(tmp_path, name="exported"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}')
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    card = build_shipcard(model_dir, build={"achieved_bpp": {"value": 4.75}})
    write_shipcard(model_dir / "shipcard.json", card)
    return model_dir


def _safetensors_bytes(tensors):
    offset = 0
    header = {}
    payloads = []
    for tensor_name, payload in tensors.items():
        payload = bytes(payload)
        header[tensor_name] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    raw_header = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw_header += b" " * (-len(raw_header) % 8)
    return (
        len(raw_header).to_bytes(8, "little")
        + raw_header
        + b"".join(payloads)
    )


def _strict_rtx4090_artifact(tmp_path, name="strict-ada"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"qwen3_5_text"}', encoding="utf-8"
    )
    tensor_name = "model.layers.0.self_attn.q_proj.weight"
    tensor_payload = b"weights"
    weight_bytes = _safetensors_bytes({tensor_name: tensor_payload})
    weight_name = "model.safetensors"
    (model_dir / weight_name).write_bytes(weight_bytes)
    tensor_sha256 = {tensor_name: hashlib.sha256(tensor_payload).hexdigest()}
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "fp8_cb",
        "provenance": {
            "producer_policy": {
                "id": publisher.RTX4090_FP8_CB_POLICY_ID,
            },
            "weight_content_manifest": {
                "schema": WEIGHT_CONTENT_MANIFEST_SCHEMA,
                "algorithm": "sha256",
                "files": {
                    weight_name: {
                        "bytes": len(weight_bytes),
                        "sha256": hashlib.sha256(weight_bytes).hexdigest(),
                    },
                },
            },
            "tensor_payload_identity": tensor_payload_identity(
                tensor_sha256,
                include_tensor_sha256=True,
            ),
        },
    }), encoding="utf-8")
    card = build_shipcard(model_dir, build={
        "producer_policy": publisher.RTX4090_FP8_CB_POLICY_ID,
        "serving_profile": publisher.RTX4090_FP8_CB_SERVING_PROFILE,
    })
    write_shipcard(model_dir / "shipcard.json", card)
    return model_dir


_FAKE_FINGERPRINT = "f" * 64
_FAKE_COMMIT = "a" * 40

#: A gold record's generic evidence (finite metric, serve fingerprint, producer
#: commit, position count, score_positions=all) is now required on EVERY lane,
#: not just Gridbook CB, so the publish fixtures must close the gold slots with
#: records that look like real measurements.
_GOLD_METRICS = {
    "gold.kl": {
        "kl_mean": 0.0151,
        "kl_confident_mean": 0.0143,
        "n_positions": 4088,
        "n_samples": 8,
        "seqlen": 512,
        "score_positions": "all",
    },
    "gold.ppl": {"ppl": 8.33, "mean_nll": 2.12, "n_tokens_scored": 8192},
}


def _close_all_slots(model_dir, *, passed=True, spec=False):
    path = model_dir / "shipcard.json"
    sha = compute_model_sha(model_dir)
    for slot in REQUIRED_SLOTS:
        is_gold = slot in GOLD_SLOTS
        fill_slot(path, slot, make_record(
            slot=slot,
            tool="test",
            passed=passed,
            model_sha=sha,
            spec_decode_detected=(spec if is_gold else None),
            metrics=(_GOLD_METRICS.get(slot) if is_gold else None),
            serve_fingerprint=(_FAKE_FINGERPRINT if is_gold else None),
            git_commit=(_FAKE_COMMIT if is_gold else None),
        ))
    return path


def _argv(model_dir, *extra):
    return [
        str(model_dir),
        "--repo-id",
        "rdtand/test-artifact",
        "--dry-run",
        *extra,
    ]


@dataclasses.dataclass
class _FakeUploadInfo:
    sha256: bytes
    size: int
    sample: bytes


class _FakeAdd:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj
        if isinstance(path_or_fileobj, bytes):
            data = path_or_fileobj
        else:  # pragma: no cover - production creates file-backed ops via stub
            with self.as_file() as handle:
                data = handle.read()
        self.upload_info = _FakeUploadInfo(
            sha256=hashlib.sha256(data).digest(),
            size=len(data),
            sample=data[:512],
        )
        self._upload_mode = None
        self._should_ignore = None
        self._remote_oid = None
        self._is_uploaded = False

    @contextlib.contextmanager
    def as_file(self):
        if isinstance(self.path_or_fileobj, bytes):
            yield io.BytesIO(self.path_or_fileobj)
            return
        if isinstance(self.path_or_fileobj, (str, pathlib.Path)):
            # Production hands large files over as the frozen view's link
            # path (the Xet transport cannot read a Python file object).
            with open(self.path_or_fileobj, "rb") as handle:
                yield handle
            return
        previous = self.path_or_fileobj.tell()
        self.path_or_fileobj.seek(0)
        try:
            yield self.path_or_fileobj
        finally:
            self.path_or_fileobj.seek(previous)


@dataclasses.dataclass
class _FakeDelete:
    path_in_repo: str
    is_folder: bool = False


class _ConflictError(RuntimeError):
    def __init__(self):
        super().__init__("parent commit conflict: remote head changed")
        self.response = types.SimpleNamespace(status_code=409)


@dataclasses.dataclass
class _FakeHubState:
    remote: dict[str, bytes]
    mutate_after_freeze: object = None
    mutate_before_commit: object = None
    conflict: bool = False
    parent: str = "a" * 40
    commit: str = "b" * 40
    repo_info_calls: int = 0
    parent_list_calls: int = 0
    preupload_calls: list[dict] = dataclasses.field(default_factory=list)
    create_calls: list[dict] = dataclasses.field(default_factory=list)
    uploaded_lfs: dict[str, bytes] = dataclasses.field(default_factory=dict)


def _read_all(handle):
    chunks = []
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _install_fake_hub(monkeypatch, state: _FakeHubState):
    class FakeApi:
        def repo_info(self, **kwargs):
            state.repo_info_calls += 1
            if state.mutate_after_freeze is not None:
                callback = state.mutate_after_freeze
                state.mutate_after_freeze = None
                callback()
            return types.SimpleNamespace(sha=state.parent, private=False)

        def list_repo_files(self, **kwargs):
            revision = kwargs["revision"]
            if revision == state.parent:
                state.parent_list_calls += 1
            return sorted(state.remote)

        def preupload_lfs_files(self, **kwargs):
            state.preupload_calls.append(dict(kwargs))
            assert kwargs["revision"] == state.parent
            assert kwargs["gitignore_content"] == ""
            for operation in kwargs["additions"]:
                operation._should_ignore = False
                operation._remote_oid = "already-present"
                # File-backed operations model LFS; byte-backed operations
                # model ordinary Git blobs. Both are captured exactly.
                if isinstance(operation.path_or_fileobj, bytes):
                    operation._upload_mode = "regular"
                    continue
                operation._upload_mode = "lfs"
                # Bytes are captured AT UPLOAD TIME, before create_commit --
                # exactly like the real LFS/Xet transports. The fake hub does
                # not verify declared digests here: detecting divergence is
                # the publisher's post-commit replay's job, which these tests
                # exercise.
                with operation.as_file() as handle:
                    data = _read_all(handle)
                state.uploaded_lfs[operation.path_in_repo] = data
                operation._is_uploaded = True

        def create_commit(self, **kwargs):
            state.create_calls.append(dict(kwargs))
            assert kwargs["revision"] == "main"
            assert kwargs["parent_commit"] == state.parent
            if state.mutate_before_commit is not None:
                callback = state.mutate_before_commit
                state.mutate_before_commit = None
                callback()
            if state.conflict:
                raise _ConflictError()
            for operation in kwargs["operations"]:
                if isinstance(operation, _FakeDelete):
                    state.remote.pop(operation.path_in_repo, None)
                    continue
                if operation._upload_mode == "lfs":
                    data = state.uploaded_lfs[operation.path_in_repo]
                else:
                    with operation.as_file() as handle:
                        data = _read_all(handle)
                state.remote[operation.path_in_repo] = data
            return types.SimpleNamespace(
                oid=state.commit,
                commit_url=f"https://hub.test/commit/{state.commit}",
            )

    monkeypatch.setattr(
        publisher,
        "_load_hub_bindings",
        lambda: publisher._HubBindings(
            api_type=FakeApi,
            add_type=_FakeAdd,
            delete_type=_FakeDelete,
        ),
    )


def test_verified_card_dry_run_freezes_and_prints_no_raw_command(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)

    assert publish_cli(_argv(model_dir)) == 0
    captured = capsys.readouterr()
    assert "frozen snapshot VERIFIED" in captured.out
    assert "--dry-run complete" in captured.out
    assert "hf upload" not in captured.out + captured.err
    assert "forced_unverified" not in load_shipcard(model_dir / "shipcard.json")


def test_unfilled_card_refuses_and_prints_no_upload_command(tmp_path, capsys):
    model_dir = _artifact(tmp_path)

    assert publish_cli(_argv(model_dir)) == 1
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    for slot in REQUIRED_SLOTS:
        assert f"{slot}: UNFILLED" in captured.err
    assert "hf upload" not in captured.out + captured.err


def test_gridbook_card_cannot_publish_without_matched_budget_parity(
    tmp_path, capsys,
):
    model_dir = tmp_path / "gridbook-exported"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"deepseek_v4"}')
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "quant_config.json").write_text(json.dumps({
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
    }))
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})
    write_shipcard(model_dir / "shipcard.json", card)
    sha = compute_model_sha(model_dir)
    for slot in REQUIRED_SLOTS:
        is_gold = slot in GOLD_SLOTS
        fill_slot(model_dir / "shipcard.json", slot, make_record(
            slot=slot,
            tool="test",
            passed=True,
            model_sha=sha,
            spec_decode_detected=False if is_gold else None,
            metrics=(_GOLD_METRICS.get(slot) if is_gold else None),
            serve_fingerprint=(_FAKE_FINGERPRINT if is_gold else None),
            git_commit=(_FAKE_COMMIT if is_gold else None),
        ))

    assert publish_cli(_argv(model_dir)) == 1
    captured = capsys.readouterr()
    assert f"{CB_REQUIRED_SLOTS[0]}: UNFILLED" in captured.err
    assert "hf upload" not in captured.out + captured.err


def test_failed_slot_refuses_naming_the_slot(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir, passed=False)

    assert publish_cli(_argv(model_dir)) == 1
    assert "ship_gate: FAILED" in capsys.readouterr().err


def test_normal_publish_replays_claimed_or_sidecar_required_dspark_slot(
    tmp_path, capsys,
):
    claimed = _artifact(tmp_path, name="claimed-target")
    path = _close_all_slots(claimed)
    card = load_shipcard(path)
    card["slots"]["mtp.dspark"] = make_record(
        slot="mtp.dspark",
        tool="test",
        passed=False,
        model_sha=card["model_sha"],
        detail="claim did not replay",
    )
    write_shipcard(path, card)
    assert publish_cli(_argv(claimed)) == 1
    assert "mtp.dspark: FAILED" in capsys.readouterr().err

    sidecar = _artifact(tmp_path, name="physical-draft")
    (sidecar / "quant_config.json").write_text(json.dumps({
        "quant_method": "test",
        "provenance": {"dspark_cb_sidecar": {"schema": "test"}},
    }))
    write_shipcard(sidecar / "shipcard.json", build_shipcard(sidecar))
    _close_all_slots(sidecar)
    assert publish_cli(_argv(sidecar)) == 1
    assert "mtp.dspark: UNFILLED" in capsys.readouterr().err


def test_spec_decode_tainted_gold_refuses(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir, spec=True)

    assert publish_cli(_argv(model_dir)) == 1
    assert "spec_decode_detected is TRUE" in capsys.readouterr().err


def test_missing_shipcard_refuses(tmp_path, capsys):
    model_dir = _artifact(tmp_path)
    (model_dir / "shipcard.json").unlink()

    assert publish_cli(_argv(model_dir)) == 2
    err = capsys.readouterr().err
    assert "canonical shipcard" in err
    assert "hf upload" not in err


@pytest.mark.parametrize(
    "malformed",
    ["{not-json", "[]", "{}", '{"slots": {}}'],
)
def test_malformed_canonical_shipcard_is_not_force_overrideable(
    tmp_path, capsys, malformed,
):
    model_dir = _artifact(tmp_path)
    (model_dir / "shipcard.json").write_text(malformed)

    assert publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        model_dir.name,
    )) == 2
    captured = capsys.readouterr()
    assert "malformed" in captured.err
    assert "never overrideable" in captured.err
    assert "frozen snapshot" not in captured.out + captured.err


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda card: card.__setitem__("schema", "prismaquant.shipcard/evil"),
        lambda card: card.__setitem__("model_sha", None),
        lambda card: card["slots"].pop("ship_gate"),
        lambda card: card["slots"].__setitem__("ship_gate", []),
    ],
)
def test_structurally_corrupt_shipcard_is_not_force_overrideable(
    tmp_path, capsys, corrupt,
):
    model_dir = _artifact(tmp_path)
    path = model_dir / "shipcard.json"
    card = load_shipcard(path)
    corrupt(card)
    write_shipcard(path, card)

    assert publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        model_dir.name,
    )) == 2
    captured = capsys.readouterr()
    assert "malformed" in captured.err
    assert "frozen snapshot" not in captured.out + captured.err


def test_force_unverified_requires_the_basename_retyped(tmp_path, capsys):
    model_dir = _artifact(tmp_path, name="ornith-35b-exported")

    assert publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        "exported",
    )) == 2
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert "hf upload" not in captured.out + captured.err
    assert "forced_unverified" not in load_shipcard(model_dir / "shipcard.json")


def test_force_unverified_stamps_the_frozen_card_and_proceeds(tmp_path, capsys):
    model_dir = _artifact(tmp_path, name="ornith-35b-exported")

    assert publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        "ornith-35b-exported",
    )) == 0
    captured = capsys.readouterr()
    assert "recorded override" in captured.err
    assert "--dry-run complete" in captured.out
    assert "hf upload" not in captured.out + captured.err

    card = load_shipcard(model_dir / "shipcard.json")
    assert card["forced_unverified"] is True
    history = card["forced_unverified_history"]
    assert len(history) == 1
    assert history[0]["repo_id"] == "rdtand/test-artifact"
    assert history[0]["unfilled_slots"] == list(REQUIRED_SLOTS)
    assert any("UNFILLED" in problem for problem in history[0]["problems"])
    assert history[0]["model_sha"] == compute_model_sha(model_dir)


@pytest.mark.parametrize("confirm_name", ("strict-ada", "wrong-name"))
@pytest.mark.parametrize("strict_slot_state", ("unfilled", "invalid"))
def test_strict_rtx4090_evidence_is_never_force_overrideable_or_stamped(
    tmp_path, capsys, confirm_name, strict_slot_state,
):
    model_dir = _strict_rtx4090_artifact(tmp_path)
    path = model_dir / "shipcard.json"
    if strict_slot_state == "invalid":
        card = load_shipcard(path)
        slot = RTX4090_REQUIRED_SLOTS[0]
        card["slots"][slot] = make_record(
            slot=slot,
            tool="not-the-specialized-validator",
            passed=False,
            model_sha=card["model_sha"],
        )
        write_shipcard(path, card)

    assert publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        confirm_name,
    )) == 1
    captured = capsys.readouterr()
    assert "strict RTX4090 FP8-CB evidence failures are non-forceable" in (
        captured.err
    )
    assert "frozen snapshot" not in captured.out + captured.err
    card = load_shipcard(path)
    assert "forced_unverified" not in card
    assert "forced_unverified_history" not in card


def test_validation_only_rtx4090_artifact_is_never_publishable(
    tmp_path, capsys,
):
    model_dir = _strict_rtx4090_artifact(tmp_path, name="validation-only")
    quant_path = model_dir / "quant_config.json"
    quant = json.loads(quant_path.read_text(encoding="utf-8"))
    quant["provenance"]["producer_policy"] = {
        "schema": "prismaquant.rtx4090_qwen38_fp8_validation_only_policy.v1",
        "id": "qwen38_27b_rtx4090_fp8_cb_validation_only",
        "artifact_disposition": "UNRELEASABLE_VALIDATION_ONLY",
    }
    quant_path.write_text(json.dumps(quant), encoding="utf-8")

    assert publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        "validation-only",
    )) == 1
    captured = capsys.readouterr()
    assert "UNRELEASABLE_VALIDATION_ONLY" in captured.err
    assert "can never be uploaded" in captured.err


@pytest.mark.parametrize("confirm_name", ("strict-ada", "wrong-name"))
def test_frozen_strict_rtx4090_replay_failure_is_non_forceable(
    tmp_path, capsys, monkeypatch, confirm_name,
):
    model_dir = _strict_rtx4090_artifact(tmp_path)
    path = model_dir / "shipcard.json"
    real_check = publisher.check_shipcard
    calls = 0

    def staged_check(artifact_dir, shipcard_path):
        nonlocal calls
        calls += 1
        card, problems = real_check(artifact_dir, shipcard_path)
        if calls == 1:
            # Exercise the second, authoritative frozen replay independently
            # of mutable preflight.  The strict classifier still reads the
            # real card and quantization bytes at both boundaries.
            return card, []
        assert card is not None
        return card, [f"{RTX4090_REQUIRED_SLOTS[0]}: replay FAILED"]

    monkeypatch.setattr(publisher, "check_shipcard", staged_check)

    assert publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        confirm_name,
    )) == 1
    captured = capsys.readouterr()
    assert "frozen strict RTX4090 FP8-CB snapshot" in captured.err
    assert "non-forceable" in captured.err
    assert "replay FAILED" in captured.err
    card = load_shipcard(path)
    assert "forced_unverified" not in card
    assert "forced_unverified_history" not in card


def test_external_or_symlinked_shipcard_is_never_publication_authority(
    tmp_path, capsys,
):
    model_dir = _artifact(tmp_path)
    canonical = _close_all_slots(model_dir)
    external = tmp_path / "external-shipcard.json"
    external.write_bytes(canonical.read_bytes())

    assert publish_cli(_argv(
        model_dir,
        "--shipcard",
        str(external),
        "--force-unverified",
        "--confirm-name",
        model_dir.name,
    )) == 2
    assert "canonical in-tree" in capsys.readouterr().err

    canonical.unlink()
    canonical.symlink_to(external)
    assert publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        model_dir.name,
    )) == 2
    assert "must not be a symlink" in capsys.readouterr().err


def test_nonregular_canonical_shipcard_is_never_publication_authority(
    tmp_path, capsys,
):
    model_dir = _artifact(tmp_path)
    canonical = model_dir / "shipcard.json"
    canonical.unlink()
    canonical.mkdir()

    assert publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        model_dir.name,
    )) == 2
    assert "not a regular file" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--allow-patterns", "*.safetensors"), "complete artifact"),
        (("--allow-patterns",), "complete artifact"),
        (("--ignore-patterns", "shipcard.json"), "complete artifact"),
        (("--ignore-patterns",), "complete artifact"),
        (("--path-in-repo", "nested/model"), "repository root"),
    ],
)
def test_model_publish_rejects_subsets_and_non_root_paths(
    tmp_path, capsys, extra, message,
):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)

    assert publish_cli(_argv(model_dir, *extra)) == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "frozen snapshot" not in captured.out + captured.err


def test_missing_hub_refuses_without_raw_cli_fallback(tmp_path, capsys, monkeypatch):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)

    def unavailable():
        raise ImportError("not installed")

    monkeypatch.setattr(publisher, "_load_hub_bindings", unavailable)
    assert publish_cli([
        str(model_dir),
        "--repo-id",
        "rdtand/test-artifact",
    ]) == 2
    captured = capsys.readouterr()
    assert "rerun this publisher" in captured.err
    assert "hf upload" not in captured.out + captured.err


def test_frozen_snapshot_survives_original_path_and_symlink_swaps(
    tmp_path, capsys, monkeypatch,
):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)
    original_config = (model_dir / "config.json").read_bytes()
    original_weights = (
        model_dir / "model-00001-of-00001.safetensors"
    ).read_bytes()
    original_shipcard = (model_dir / "shipcard.json").read_bytes()
    evil_config = tmp_path / "evil-config.json"
    evil_config.write_bytes(b'{"model_type":"evil"}')
    evil_weights = tmp_path / "evil.safetensors"
    evil_weights.write_bytes(b"EVIL!!!")
    evil_shipcard = tmp_path / "evil-shipcard.json"
    evil_shipcard.write_bytes(b'{"slots":{}}')

    def swap_original_paths():
        config = model_dir / "config.json"
        weights = model_dir / "model-00001-of-00001.safetensors"
        shipcard = model_dir / "shipcard.json"
        config.unlink()
        config.symlink_to(evil_config)
        weights.unlink()
        weights.symlink_to(evil_weights)
        shipcard.unlink()
        shipcard.symlink_to(evil_shipcard)

    # Force even the tiny fixture through the held-fd/block-verified path.
    monkeypatch.setattr(publisher, "SNAPSHOT_INLINE_BYTES", 4)
    state = _FakeHubState(
        remote={
            ".gitattributes": b"hub-managed",
            "config.json": b"remote-stale-config",
            "stale.bin": b"stale",
        },
        mutate_after_freeze=swap_original_paths,
    )
    _install_fake_hub(monkeypatch, state)

    assert publish_cli([
        str(model_dir),
        "--repo-id",
        "rdtand/test-artifact",
    ]) == 0
    assert state.remote["config.json"] == original_config
    assert state.remote["model-00001-of-00001.safetensors"] == original_weights
    assert state.remote["shipcard.json"] == original_shipcard
    assert state.remote[".gitattributes"] == b"hub-managed"
    assert "stale.bin" not in state.remote
    assert set(state.remote) == {
        ".gitattributes",
        "config.json",
        "model-00001-of-00001.safetensors",
        "shipcard.json",
    }
    assert state.create_calls[0]["parent_commit"] == state.parent
    assert state.create_calls[0]["revision"] == "main"
    assert "stale_deleted=1" in capsys.readouterr().out


def test_in_place_mutation_after_freeze_is_detected_after_commit(
    tmp_path, capsys, monkeypatch,
):
    """Same-inode mutation across the upload window fails the publish loudly.

    The Xet transport consumes paths, not the streaming verified reader, so a
    same-inode mutation after freeze IS uploaded; the publisher's post-commit
    replay of the declared digests across its held descriptors detects it and
    refuses to report success (the commit exists and the operator is told to
    inspect before announcing)."""
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)
    weight = model_dir / "model-00001-of-00001.safetensors"

    def mutate_same_inode():
        weight.write_bytes(b"EVIL!!!")

    monkeypatch.setattr(publisher, "SNAPSHOT_INLINE_BYTES", 4)
    state = _FakeHubState(
        remote={".gitattributes": b"hub-managed"},
        mutate_after_freeze=mutate_same_inode,
    )
    _install_fake_hub(monkeypatch, state)

    assert publish_cli([
        str(model_dir),
        "--repo-id",
        "rdtand/test-artifact",
    ]) == 1
    assert len(state.create_calls) == 1
    err = capsys.readouterr().err
    assert "post-commit re-verification" in err
    assert "inspect the repository" in err


def test_in_place_mutation_after_preupload_cannot_change_committed_bytes(
    tmp_path, monkeypatch,
):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)
    weight = model_dir / "model-00001-of-00001.safetensors"
    original_weight = weight.read_bytes()

    monkeypatch.setattr(publisher, "SNAPSHOT_INLINE_BYTES", 4)
    state = _FakeHubState(
        remote={".gitattributes": b"hub-managed"},
        mutate_before_commit=lambda: weight.write_bytes(b"EVIL!!!"),
    )
    _install_fake_hub(monkeypatch, state)

    # The upload captured the frozen bytes before the mutation, so the
    # committed object is unchanged -- and the post-commit digest replay
    # still fails the run loudly because the local inode diverged during
    # the publish window (fail-safe: never report clean success over a
    # mid-publish mutation, even a harmless-to-the-commit one).
    assert publish_cli([
        str(model_dir),
        "--repo-id",
        "rdtand/test-artifact",
    ]) == 1
    assert state.remote[weight.name] == original_weight


def test_remote_head_conflict_refuses_without_retrying_or_leaving_stale_success(
    tmp_path, capsys, monkeypatch,
):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)
    state = _FakeHubState(
        remote={".gitattributes": b"hub-managed", "stale.bin": b"stale"},
        conflict=True,
    )
    _install_fake_hub(monkeypatch, state)

    assert publish_cli([
        str(model_dir),
        "--repo-id",
        "rdtand/test-artifact",
    ]) == 1
    captured = capsys.readouterr()
    assert "parent_commit CAS" in captured.err
    assert "Rerun this publisher from the beginning" in captured.err
    assert state.repo_info_calls == 1
    assert state.parent_list_calls == 1
    assert len(state.preupload_calls) == 1
    assert len(state.create_calls) == 1
    # The fake conflict occurs before applying any operation.
    assert "stale.bin" in state.remote


def test_identical_additions_still_force_one_parent_cas_commit(
    tmp_path, monkeypatch,
):
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)
    state = _FakeHubState(remote={
        ".gitattributes": b"hub-managed",
        "config.json": (model_dir / "config.json").read_bytes(),
        "model-00001-of-00001.safetensors": (
            model_dir / "model-00001-of-00001.safetensors"
        ).read_bytes(),
        "shipcard.json": (model_dir / "shipcard.json").read_bytes(),
    })
    _install_fake_hub(monkeypatch, state)

    assert publish_cli([
        str(model_dir),
        "--repo-id",
        "rdtand/test-artifact",
    ]) == 0
    assert len(state.create_calls) == 1
    assert state.create_calls[0]["parent_commit"] == state.parent
    add_operations = [
        operation
        for operation in state.create_calls[0]["operations"]
        if isinstance(operation, _FakeAdd)
    ]
    assert add_operations
    assert all(operation._remote_oid is None for operation in add_operations)


def test_installed_hub_operation_reads_the_frozen_descriptor_exactly(
    tmp_path, monkeypatch,
):
    pytest.importorskip("huggingface_hub")
    model_dir = _artifact(tmp_path)
    _close_all_slots(model_dir)
    monkeypatch.setattr(publisher, "SNAPSHOT_INLINE_BYTES", 4)
    shipcard_sha = hashlib.sha256(
        (model_dir / "shipcard.json").read_bytes()
    ).hexdigest()

    with publisher._freeze_artifact(
        model_dir,
        expected_shipcard_sha256=shipcard_sha,
    ) as snapshot:
        bindings = publisher._load_hub_bindings()
        additions = publisher._make_additions(
            snapshot,
            prefix="",
            bindings=bindings,
        )
        assert len(additions) == len(snapshot.entries)
        for entry, operation in zip(snapshot.entries, additions, strict=True):
            assert operation.upload_info.size == entry.size
            assert operation.upload_info.sha256.hex() == entry.sha256
            with operation.as_file() as handle:
                observed = _read_all(handle)
            assert hashlib.sha256(observed).hexdigest() == entry.sha256


def test_verification_is_a_library_call_and_upload_folder_is_absent():
    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "tools"
        / "publish_artifact.py"
    ).read_text()
    assert "from prismaquant.shipcard import" in src
    assert "subprocess" not in src
    assert "from huggingface_hub import upload_folder" not in src
    assert "parent_commit=parent_commit" in src
    assert "gitignore_content=\"\"" in src
    main_src = src[src.index("def main("):]
    assert main_src.index("check_shipcard(artifact_dir") < main_src.index(
        "_load_hub_bindings()"
    )


def test_json_round_trip_of_a_forced_card(tmp_path):
    model_dir = _artifact(tmp_path, name="forced")
    publish_cli(_argv(
        model_dir,
        "--force-unverified",
        "--confirm-name",
        "forced",
    ))
    raw = json.loads((model_dir / "shipcard.json").read_text())
    assert raw["forced_unverified"] is True


def test_rtx4090_whole_publication_ceiling_includes_post_export_files(
    tmp_path, monkeypatch,
):
    model_dir = _artifact(tmp_path, name="strict-ada")
    quant = {
        "quant_method": "gridbook",
        "format": "fp8_cb",
        "provenance": {
            "producer_policy": {
                "id": "qwen38_27b_rtx4090_fp8_cb",
            },
        },
    }
    (model_dir / "quant_config.json").write_text(json.dumps(quant))
    card = load_shipcard(model_dir / "shipcard.json")
    monkeypatch.setattr(
        publisher, "RTX4090_FP8_CB_ARTIFACT_CEILING_BYTES", 32
    )

    problem = publisher._rtx4090_publication_size_problem(
        model_dir,
        card,
        total_bytes=33,
    )
    assert problem is not None
    assert "including documentation/evidence" in problem
    assert "non-forceable" in problem
    assert publisher._rtx4090_publication_size_problem(
        model_dir,
        card,
        total_bytes=32,
    ) is None


def test_strict_frozen_replay_consumes_one_pass_container_and_tensor_hashes(
    tmp_path, monkeypatch,
):
    model_dir = _strict_rtx4090_artifact(tmp_path)
    weight = model_dir / "model.safetensors"
    weight_stat = weight.stat()
    real_pread = publisher._pread_exact
    weight_reads = []

    def counted_pread(fd, size, offset):
        opened = os.fstat(fd)
        if (
            opened.st_dev == weight_stat.st_dev
            and opened.st_ino == weight_stat.st_ino
        ):
            weight_reads.append((size, offset))
        return real_pread(fd, size, offset)

    monkeypatch.setattr(publisher, "_pread_exact", counted_pread)
    real_check = publisher.check_shipcard

    def evidence_closed_for_fixture(artifact_dir, shipcard_path):
        card, _problems = real_check(artifact_dir, shipcard_path)
        return card, []

    monkeypatch.setattr(
        publisher,
        "check_shipcard",
        evidence_closed_for_fixture,
    )
    replay_calls = 0
    real_replay = publisher._verify_strict_frozen_safetensors_content

    def counted_replay(snapshot):
        nonlocal replay_calls
        replay_calls += 1
        return real_replay(snapshot)

    monkeypatch.setattr(
        publisher,
        "_verify_strict_frozen_safetensors_content",
        counted_replay,
    )

    assert publish_cli(_argv(model_dir)) == 0
    assert replay_calls == 1
    assert weight_reads == [(weight_stat.st_size, 0)]


def test_strict_frozen_tensor_digest_mismatch_refuses_without_a_second_scan(
    tmp_path, monkeypatch, capsys,
):
    model_dir = _strict_rtx4090_artifact(tmp_path)
    quant_path = model_dir / "quant_config.json"
    quant = json.loads(quant_path.read_text(encoding="utf-8"))
    tensor_name = next(iter(
        quant["provenance"]["tensor_payload_identity"]["tensor_sha256"]
    ))
    forged_ledger = {tensor_name: "0" * 64}
    quant["provenance"]["tensor_payload_identity"] = tensor_payload_identity(
        forged_ledger,
        include_tensor_sha256=True,
    )
    quant_path.write_text(json.dumps(quant), encoding="utf-8")

    weight = model_dir / "model.safetensors"
    weight_stat = weight.stat()
    real_pread = publisher._pread_exact
    weight_bytes_read = 0

    def counted_pread(fd, size, offset):
        nonlocal weight_bytes_read
        opened = os.fstat(fd)
        if (
            opened.st_dev == weight_stat.st_dev
            and opened.st_ino == weight_stat.st_ino
        ):
            weight_bytes_read += size
        return real_pread(fd, size, offset)

    monkeypatch.setattr(publisher, "_pread_exact", counted_pread)
    real_check = publisher.check_shipcard

    def evidence_closed_for_fixture(artifact_dir, shipcard_path):
        card, _problems = real_check(artifact_dir, shipcard_path)
        return card, []

    monkeypatch.setattr(
        publisher,
        "check_shipcard",
        evidence_closed_for_fixture,
    )

    assert publish_cli(_argv(model_dir)) == 1
    assert weight_bytes_read == weight_stat.st_size
    err = capsys.readouterr().err
    assert "frozen strict RTX4090 FP8-CB snapshot" in err
    assert "tensor digest ledger differs from payload bytes" in err
    assert "non-forceable" in err


def test_strict_freeze_rejects_noncontiguous_safetensors_geometry(
    tmp_path, monkeypatch, capsys,
):
    model_dir = _strict_rtx4090_artifact(tmp_path)
    tensor_name = "model.layers.0.self_attn.q_proj.weight"
    header = json.dumps({
        tensor_name: {
            "dtype": "U8",
            "shape": [7],
            "data_offsets": [1, 8],
        },
    }, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    (model_dir / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"xweights"
    )
    real_check = publisher.check_shipcard

    def evidence_closed_for_fixture(artifact_dir, shipcard_path):
        card, _problems = real_check(artifact_dir, shipcard_path)
        return card, []

    monkeypatch.setattr(
        publisher,
        "check_shipcard",
        evidence_closed_for_fixture,
    )

    assert publish_cli(_argv(model_dir)) == 1
    err = capsys.readouterr().err
    assert "artifact freeze failed" in err
    assert "leaves a gap" in err


def test_strict_frozen_receipt_binds_exact_shard_index_map(tmp_path):
    model_dir = _strict_rtx4090_artifact(tmp_path)
    (model_dir / "model.safetensors").unlink()
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": b"query",
        "model.layers.0.self_attn.k_proj.weight": b"key",
    }
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    shard_payloads = {
        shard_names[0]: _safetensors_bytes({next(iter(tensors)): b"query"}),
        shard_names[1]: _safetensors_bytes({list(tensors)[1]: b"key"}),
    }
    for name, payload in shard_payloads.items():
        (model_dir / name).write_bytes(payload)
    weight_map = dict(zip(tensors, shard_names, strict=True))
    index_path = model_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps({
        "metadata": {"total_size": sum(map(len, tensors.values()))},
        "weight_map": weight_map,
    }), encoding="utf-8")
    quant_path = model_dir / "quant_config.json"
    quant = json.loads(quant_path.read_text(encoding="utf-8"))
    tensor_ledger = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in tensors.items()
    }
    quant["provenance"]["tensor_payload_identity"] = tensor_payload_identity(
        tensor_ledger,
        include_tensor_sha256=True,
    )
    quant["provenance"]["weight_content_manifest"] = {
        "schema": WEIGHT_CONTENT_MANIFEST_SCHEMA,
        "algorithm": "sha256",
        "files": {
            name: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in shard_payloads.items()
        },
    }
    quant_path.write_text(json.dumps(quant), encoding="utf-8")
    shipcard_sha = hashlib.sha256(
        (model_dir / "shipcard.json").read_bytes()
    ).hexdigest()

    with publisher._freeze_artifact(
        model_dir,
        expected_shipcard_sha256=shipcard_sha,
        capture_safetensors_content=True,
    ) as snapshot:
        publisher._verify_strict_frozen_safetensors_content(snapshot)
        assert set(snapshot.safetensors_content_receipt["files"]) == set(
            shard_names
        )

    index_path.write_text(json.dumps({
        "metadata": {"total_size": sum(map(len, tensors.values()))},
        "weight_map": {
            tensor_name: shard_names[1 - index]
            for index, tensor_name in enumerate(tensors)
        },
    }), encoding="utf-8")
    with publisher._freeze_artifact(
        model_dir,
        expected_shipcard_sha256=shipcard_sha,
        capture_safetensors_content=True,
    ) as snapshot:
        with pytest.raises(
            publisher.FrozenSnapshotError,
            match="index differs from payload headers",
        ):
            publisher._verify_strict_frozen_safetensors_content(snapshot)


def test_strict_one_pass_scanner_handles_header_and_tensor_block_boundaries(
    tmp_path, monkeypatch,
):
    model_dir = _strict_rtx4090_artifact(tmp_path)
    weight = model_dir / "model.safetensors"
    weight_stat = weight.stat()
    monkeypatch.setattr(publisher, "SNAPSHOT_BLOCK_BYTES", 32)
    real_pread = publisher._pread_exact
    weight_reads = []

    def counted_pread(fd, size, offset):
        opened = os.fstat(fd)
        if (
            opened.st_dev == weight_stat.st_dev
            and opened.st_ino == weight_stat.st_ino
        ):
            weight_reads.append((offset, size))
        return real_pread(fd, size, offset)

    monkeypatch.setattr(publisher, "_pread_exact", counted_pread)
    shipcard_sha = hashlib.sha256(
        (model_dir / "shipcard.json").read_bytes()
    ).hexdigest()
    with publisher._freeze_artifact(
        model_dir,
        expected_shipcard_sha256=shipcard_sha,
        capture_safetensors_content=True,
    ) as snapshot:
        publisher._verify_strict_frozen_safetensors_content(snapshot)

    assert weight_reads
    assert [offset for offset, _size in weight_reads] == list(
        range(0, weight_stat.st_size, 32)
    )
    assert sum(size for _offset, size in weight_reads) == weight_stat.st_size
