from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from prismaquant import prismabuild as pb


def _body(
    checkout: Path,
    *,
    task_class: str = "generation",
    determinism: str = "deterministic",
    artifact_kind: str = "generic",
    portability: str = "portable",
    platform_key: str | None = None,
    host_class: str | None = None,
    argv: list[str] | None = None,
    result_path: str = "result.bin",
) -> dict[str, object]:
    code = checkout / "task_code.py"
    if not code.exists():
        code.write_text("# closure member\n", encoding="utf-8")
    return {
        "schema": pb.ACTION_SCHEMA_V1,
        "task": {
            "definition_id": "tests/build-result",
            "definition_version": "v1",
            "task_class": task_class,
            "determinism": determinism,
            "artifact_kind": artifact_kind,
            "argv": argv
            or [
                sys.executable,
                "-c",
                (
                    "import pathlib; "
                    f"pathlib.Path({result_path!r}).write_bytes(b'result')"
                ),
            ],
            "working_directory": ".",
            "result_path": result_path,
        },
        "inputs": [
            {"id": "model/config", "sha256": "2" * 64, "bytes": 20},
            {"id": "model/weights", "sha256": "3" * 64, "bytes": 30},
        ],
        "code_closure": pb.build_code_closure(checkout, ["task_code.py"]),
        "params": {"alpha": 1, "nested": {"enabled": True}},
        "environment": {
            "variables": {"DECLARED": "yes"},
            "toolchain": {"python": "3.12", "torch": "2.11+cu130"},
        },
        "execution_scope": {
            "portability": portability,
            "platform_key": platform_key,
            "host_class": host_class,
        },
    }


def _action(checkout: Path, **kwargs: object) -> dict[str, object]:
    return pb.seal_action(_body(checkout, **kwargs))


def test_action_key_is_canonical_and_inputs_are_sorted(tmp_path: Path):
    body = _body(tmp_path)
    body["inputs"] = list(reversed(body["inputs"]))  # type: ignore[index]
    first = pb.seal_action(body)
    second_body = _body(tmp_path)
    second_body["params"] = {"nested": {"enabled": True}, "alpha": 1}
    second = pb.seal_action(second_body)

    assert first == second
    assert [row["id"] for row in first["inputs"]] == [  # type: ignore[index]
        "model/config",
        "model/weights",
    ]
    assert pb.validate_action(first) == first


def test_unsealed_or_non_normalized_action_is_rejected(tmp_path: Path):
    action = _action(tmp_path)
    action["inputs"] = list(reversed(action["inputs"]))  # type: ignore[index]
    with pytest.raises(pb.ActionContractError, match="normalized contract form"):
        pb.validate_action(action)

    body = _body(tmp_path)
    body["campaign_id"] = "must-not-enter-action-key"
    with pytest.raises(pb.ActionContractError, match="extra=.*campaign_id"):
        pb.seal_action(body)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("task", "definition_version"), "v2"),
        (("task", "argv"), ["/bin/true", "different"]),
        (("inputs", 0, "sha256"), "4" * 64),
        (("code_closure", "closure_sha256"), "5" * 64),
        (("params", "alpha"), 2),
        (("environment", "variables", "DECLARED"), "different"),
        (("environment", "toolchain", "torch"), "different"),
        (("execution_scope", "host_class"), "different"),
    ],
)
def test_action_key_binds_every_semantic_input(
    tmp_path: Path, path: tuple[object, ...], replacement: object
):
    original_body = _body(
        tmp_path, portability="host_class_keyed", host_class="gb10"
    )
    original = pb.seal_action(original_body)
    changed = copy.deepcopy(original_body)
    cursor: object = changed
    for component in path[:-1]:
        cursor = cursor[component]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    if path == ("code_closure", "closure_sha256"):
        with pytest.raises(pb.ActionContractError, match="does not match"):
            pb.seal_action(changed)
    else:
        assert pb.seal_action(changed)["action_key"] != original["action_key"]


def test_portability_contracts_and_d29_refusal(tmp_path: Path):
    with pytest.raises(pb.ActionContractError, match="measurement actions"):
        _action(tmp_path, task_class="measurement")
    with pytest.raises(pb.ActionContractError, match="D29"):
        _action(tmp_path, artifact_kind="fp8_cb")
    with pytest.raises(pb.ActionContractError, match="D29"):
        _action(tmp_path, artifact_kind="fp8-cb")
    for spelling in (
        "fp8cb",
        "cb_fp8",
        "qwen3-fp8-cb",
        "fp8_cb_v2",
        "fp8_codebook",
        "nvfp4-cb-rerender",
    ):
        with pytest.raises(pb.ActionContractError, match="D29"):
            _action(tmp_path, artifact_kind=spelling)

    platform = _action(
        tmp_path,
        artifact_kind="fp8_cb",
        portability="platform_keyed",
        platform_key="linux-aarch64-sm121",
    )
    pb.validate_worker_scope(
        platform, platform_key="linux-aarch64-sm121", host_class="gb10"
    )
    with pytest.raises(pb.ActionContractError, match="platform_key"):
        pb.validate_worker_scope(
            platform, platform_key="linux-x86_64-sm89", host_class="rtx4090"
        )

    host = _action(
        tmp_path,
        task_class="measurement",
        portability="host_class_keyed",
        host_class="gb10",
    )
    pb.validate_worker_scope(host, platform_key=None, host_class="gb10")
    with pytest.raises(pb.ActionContractError, match="host_class"):
        pb.validate_worker_scope(host, platform_key=None, host_class="cpu-x86-large")


def test_code_closure_is_root_independent_and_live_verified(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for root in (left, right):
        (root / "a.py").write_text("A\n", encoding="utf-8")
        (root / "b.py").write_text("B\n", encoding="utf-8")
    first = pb.build_code_closure(left, ["b.py", "a.py"])
    second = pb.build_code_closure(right, ["a.py", "b.py"])
    assert first == second
    assert [row["path"] for row in first["files"]] == ["a.py", "b.py"]
    assert pb.verify_code_closure(first, right) == first

    (right / "b.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(pb.ActionContractError, match="live code closure differs"):
        pb.verify_code_closure(first, right)


def test_code_closure_refuses_traversal_duplicates_and_symlinks(tmp_path: Path):
    (tmp_path / "a.py").write_text("A", encoding="utf-8")
    with pytest.raises(pb.ActionContractError, match="relative POSIX"):
        pb.build_code_closure(tmp_path, ["../a.py"])
    with pytest.raises(pb.ActionContractError, match="unique"):
        pb.build_code_closure(tmp_path, ["a.py", "a.py"])
    (tmp_path / "link.py").symlink_to("a.py")
    with pytest.raises(pb.ActionContractError, match="regular file"):
        pb.build_code_closure(tmp_path, ["link.py"])


def test_paths_and_nested_params_reject_ambiguous_json(tmp_path: Path):
    for path in (r"sub\result.bin", "C:result.bin"):
        with pytest.raises(pb.ActionContractError, match="relative POSIX"):
            _action(tmp_path, result_path=path)

    for params in (
        {"nested": {1: "coerced"}},
        {"nested": {"nul\x00key": 1}},
        {"nested": ["bad\nvalue"]},
    ):
        body = _body(tmp_path)
        body["params"] = params
        with pytest.raises(pb.ActionContractError):
            pb.seal_action(body)


def test_local_worker_uses_exact_argv_and_closed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setenv("UNDECLARED_SECRET", "must-not-leak")
    code = (
        "import os,pathlib; "
        "pathlib.Path('result.bin').write_text("
        "os.environ.get('DECLARED','missing')+'|' + "
        "str('UNDECLARED_SECRET' in os.environ))"
    )
    action = _action(checkout, argv=[sys.executable, "-c", code])
    result = pb.run_local_action(
        action,
        cas_root=tmp_path / "cas",
        checkout_root=checkout,
        worker_id="worker-1",
        platform_key="linux-test",
        host_class="test",
    )

    assert result["status"] == "published"
    payload = Path(result["payload_path"]).read_text(encoding="utf-8")
    assert payload == "yes|False"
    assert pb.PrismaBuildCAS(tmp_path / "cas").lookup(action) == result["receipt"]


def test_cache_hit_skips_execution_and_fully_revalidates(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    code = (
        "import pathlib; p=pathlib.Path('marker'); "
        "p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
        "pathlib.Path('result.bin').write_bytes(b'ok')"
    )
    action = _action(checkout, argv=[sys.executable, "-c", code])
    arguments = dict(
        action=action,
        cas_root=tmp_path / "cas",
        checkout_root=checkout,
        worker_id="worker-1",
        platform_key=None,
        host_class=None,
    )
    first = pb.run_local_action(**arguments)
    second = pb.run_local_action(**arguments)
    assert first["status"] == "published"
    assert second["status"] == "cache_hit"
    assert (checkout / "marker").read_text(encoding="utf-8") == "x"

    (checkout / "task_code.py").write_text("# closure changed\n", encoding="utf-8")
    with pytest.raises(pb.ActionContractError, match="live code closure differs"):
        pb.run_local_action(**arguments)


def test_local_worker_fails_closed_on_dirty_output_or_missing_result(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    dirty = _action(checkout)
    (checkout / "result.bin").write_bytes(b"stale")
    with pytest.raises(pb.LocalActionError, match="must be absent"):
        pb.run_local_action(
            dirty,
            cas_root=tmp_path / "cas-dirty",
            checkout_root=checkout,
            worker_id="worker",
            platform_key=None,
            host_class=None,
        )
    (checkout / "result.bin").unlink()
    missing = _action(checkout, argv=[sys.executable, "-c", "pass"])
    with pytest.raises(pb.LocalActionError, match="without its declared result"):
        pb.run_local_action(
            missing,
            cas_root=tmp_path / "cas-missing",
            checkout_root=checkout,
            worker_id="worker",
            platform_key=None,
            host_class=None,
        )


def test_local_worker_refuses_symlinked_result_parent(tmp_path: Path):
    checkout = tmp_path / "checkout"
    outside = tmp_path / "outside"
    checkout.mkdir()
    outside.mkdir()
    (checkout / "sub").symlink_to(outside, target_is_directory=True)
    action = _action(checkout, result_path="sub/result.bin")
    with pytest.raises(pb.LocalActionError, match="traverses a symlink"):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
            worker_id="worker",
            platform_key=None,
            host_class=None,
        )
    assert not (outside / "result.bin").exists()


def test_local_worker_serializes_shared_checkout_output(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    actions = [
        _action(
            checkout,
            argv=[
                sys.executable,
                "-c",
                (
                    "import pathlib,time; time.sleep(0.15); "
                    f"pathlib.Path('shared.bin').write_bytes({payload!r})"
                ),
            ],
            result_path="shared.bin",
        )
        for payload in (b"AAAA", b"BBBB")
    ]
    errors: list[BaseException] = []
    results: list[dict[str, object]] = []
    barrier = threading.Barrier(2)

    def run(action: dict[str, object]) -> None:
        try:
            barrier.wait()
            results.append(
                pb.run_local_action(
                    action,
                    cas_root=tmp_path / "cas",
                    checkout_root=checkout,
                    worker_id="worker",
                    platform_key=None,
                    host_class=None,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(action,)) for action in actions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], pb.LocalActionError)
    published = Path(str(results[0]["payload_path"])).read_bytes()
    assert published == (checkout / "shared.bin").read_bytes()


def test_timeout_terminates_child_process_group(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    marker = tmp_path / "grandchild-marker"
    child_code = f"import time,pathlib; time.sleep(0.8); pathlib.Path({str(marker)!r}).write_text('alive')"
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); time.sleep(5)"
    )
    action = _action(checkout, argv=[sys.executable, "-c", parent_code])
    with pytest.raises(pb.LocalActionError, match="timed out"):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
            worker_id="worker",
            platform_key=None,
            host_class=None,
            timeout_seconds=0.1,
        )
    time.sleep(1.0)
    assert not marker.exists()


def test_deterministic_recompute_must_match(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"canonical")
    second.write_bytes(b"different")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    receipt, won = cas.publish_result(
        action,
        first,
        worker_id="worker-a",
        platform_key=None,
        host_class=None,
    )
    assert won
    with pytest.raises(pb.CASConflictError, match="deterministic recomputation"):
        cas.publish_result(
            action,
            second,
            worker_id="worker-b",
            platform_key=None,
            host_class=None,
        )
    assert cas.lookup(action) == receipt


def test_stochastic_action_is_atomic_first_result_wins(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout, determinism="stochastic")
    outputs = []
    for index in range(2):
        path = tmp_path / f"candidate-{index}"
        path.write_bytes(f"candidate-{index}".encode())
        outputs.append(path)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    barrier = threading.Barrier(2)
    results: list[tuple[dict[str, object], bool]] = []
    errors: list[BaseException] = []

    def publish(index: int) -> None:
        try:
            barrier.wait()
            results.append(
                cas.publish_result(
                    action,
                    outputs[index],
                    worker_id=f"worker-{index}",
                    platform_key=None,
                    host_class=None,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=publish, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 2
    assert sum(won for _, won in results) == 1
    assert results[0][0] == results[1][0] == cas.lookup(action)
    receipt = results[0][0]
    winner_payload = cas.result_path(receipt, action).read_bytes()
    assert winner_payload in {b"candidate-0", b"candidate-1"}


def test_cache_detects_payload_and_receipt_tampering(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    output = tmp_path / "output"
    output.write_bytes(b"trusted")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    receipt, _ = cas.publish_result(
        action,
        output,
        worker_id="worker",
        platform_key=None,
        host_class=None,
    )
    payload = cas.result_path(receipt, action)
    payload.chmod(0o644)
    payload.write_bytes(b"tampered")
    with pytest.raises(pb.CASTamperError, match="payload content differs"):
        cas.lookup(action)

    # Restore the content and then alter the immutable receipt's bytes.
    payload.write_bytes(b"trusted")
    receipt_path = (
        tmp_path
        / "cas"
        / "actions"
        / str(action["action_key"])[:2]
        / f"{action['action_key']}.json"
    )
    receipt_path.chmod(0o644)
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(pb.CASTamperError, match="fields differ"):
        cas.lookup(action)

    receipt_path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(pb.CASTamperError, match="strict UTF-8 JSON"):
        cas.lookup(action)


def test_cache_structural_error_is_not_a_clean_miss(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    shard = tmp_path / "cas" / "actions" / str(action["action_key"])[:2]
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"not-a-directory")
    with pytest.raises(pb.CASTamperError):
        pb.PrismaBuildCAS(tmp_path / "cas").lookup(action)


def test_cli_seals_keys_runs_and_verifies(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    body = _body(checkout)
    body_path = tmp_path / "body.json"
    action_path = tmp_path / "action.json"
    body_path.write_text(json.dumps(body), encoding="utf-8")

    assert pb.main(["seal-action", "--body", str(body_path), "--output", str(action_path)]) == 0
    action = json.loads(action_path.read_text(encoding="utf-8"))
    assert pb.main(["key", "--action", str(action_path)]) == 0
    assert str(action["action_key"]) in capsys.readouterr().out
    assert pb.main(
        [
            "run-local",
            "--action",
            str(action_path),
            "--cas-root",
            str(tmp_path / "cas"),
            "--checkout-root",
            str(checkout),
            "--worker-id",
            "cli-worker",
        ]
    ) == 0
    assert '"status": "published"' in capsys.readouterr().out
    assert pb.main(
        [
            "verify",
            "--action",
            str(action_path),
            "--cas-root",
            str(tmp_path / "cas"),
        ]
    ) == 0
    assert str(action["action_key"]) in capsys.readouterr().out


def test_action_key_has_expected_plain_sha256_shape(tmp_path: Path):
    action = _action(tmp_path)
    assert len(action["action_key"]) == 64
    int(str(action["action_key"]), 16)
    body = {key: action[key] for key in action if key != "action_key"}
    expected = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert action["action_key"] == expected
