from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from prismaquant import prismabuild as pb


def _body(
    checkout: Path,
    *,
    task_class: str = "generation",
    determinism: str = "deterministic",
    artifact_family: str = "generic",
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
    resolved_argv = argv or [
        sys.executable,
        "-c",
        (
            "import pathlib; "
            f"pathlib.Path({result_path!r}).write_bytes(b'result')"
        ),
    ]
    # These generic actions do not consume Torch. Keep their contract limited
    # to facts that matter to execution so the suite is portable across the
    # supported producer images.
    toolchain: dict[str, str] = {}
    if portability != "portable":
        toolchain.update(pb.executable_toolchain_contract(resolved_argv[0]))
        evidence = pb._collect_worker_evidence()
        toolchain.update(
            {
                "system": str(evidence["system"]),
                "machine": str(evidence["machine"]),
                "libc": str(evidence["libc"]),
            }
        )
    return {
        "schema": pb.ACTION_SCHEMA_V2,
        "task": {
            "definition_id": "tests/build-result",
            "definition_version": "v1",
            "task_class": task_class,
            "determinism": determinism,
            "artifact_family": artifact_family,
            "artifact_kind": artifact_kind,
            "argv": resolved_argv,
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
            "toolchain": toolchain,
        },
        "execution_scope": {
            "portability": portability,
            "platform_key": platform_key,
            "host_class": host_class,
        },
    }


def _action(checkout: Path, **kwargs: object) -> dict[str, object]:
    return pb.seal_action(_body(checkout, **kwargs))


def _attestation(
    checkout: Path, action: dict[str, object], cas_root: Path
) -> dict[str, object]:
    return pb.preflight_action(
        action, cas_root=cas_root, checkout_root=checkout
    )


def _rendezvous_manifest(
    path: Path,
    *,
    action: dict[str, object],
    namespace: Path,
    cas_root: Path | None = None,
    participants: list[str] | None = None,
    timeout_seconds: float = 1.0,
    run_nonce: str = "1" * 32,
) -> tuple[Path, dict[str, object]]:
    body: dict[str, object] = {
        "schema": pb.INITIAL_MISS_RENDEZVOUS_MANIFEST_SCHEMA_V1,
        "rendezvous_namespace": str(namespace),
        "cas_root": str(cas_root or (path.parent / "cas")),
        "run_nonce": run_nonce,
        "action_key": action["action_key"],
        "participants": participants or ["host-a", "host-b"],
        "timeout_seconds": timeout_seconds,
    }
    manifest = {**body, "manifest_sha256": pb.canonical_sha256(body)}
    path.write_bytes(pb._canonical_file_bytes(manifest))
    path.chmod(0o444)
    return path, manifest


def _write_rendezvous_record(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pb._canonical_file_bytes(value))
    path.chmod(0o444)


def _reseal(value: dict[str, object], digest_key: str) -> None:
    body = {key: item for key, item in value.items() if key != digest_key}
    value[digest_key] = pb.canonical_sha256(body)


def _live_nonportable_body(
    checkout: Path,
    *,
    inputs: list[dict[str, object]] | None = None,
    argv: list[str] | None = None,
) -> dict[str, object]:
    evidence = pb._collect_worker_evidence()
    platform_key = pb._platform_key_from_evidence(evidence)
    body = _body(
        checkout,
        portability="platform_keyed",
        platform_key=platform_key,
        argv=argv,
    )
    body["inputs"] = [] if inputs is None else inputs
    toolchain = {
        **pb.executable_toolchain_contract(body["task"]["argv"][0]),  # type: ignore[index]
        "system": str(evidence["system"]),
        "machine": str(evidence["machine"]),
        "libc": str(evidence["libc"]),
    }
    accelerators = evidence["accelerators"]
    if accelerators:
        toolchain.update(
            {
                "cuda_compute_capability": accelerators[0]["compute_capability"],
                "nvidia_driver": accelerators[0]["driver_version"],
            }
        )
    body["environment"]["toolchain"] = toolchain  # type: ignore[index]
    return body


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
        (("task", "artifact_family"), "codebook"),
        (("task", "argv"), ["/bin/true", "different"]),
        (("inputs", 0, "sha256"), "4" * 64),
        (("code_closure", "closure_sha256"), "5" * 64),
        (("params", "alpha"), 2),
        (("environment", "variables", "DECLARED"), "different"),
        (("environment", "toolchain", "system"), "different"),
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


def test_portability_contracts_and_explicit_codebook_family_refusal(tmp_path: Path):
    with pytest.raises(pb.ActionContractError, match="measurement actions"):
        _action(tmp_path, task_class="measurement")
    # These labels were false negatives under the retired substring heuristic.
    # The closed family, not a spelling convention, now carries the policy.
    for artifact_kind in (
        "fp8-book",
        "fp8-lut",
        "fp8-palette",
        "nvfp4-quantizer",
        "scale-book",
        "cb-training",
        "codebook-train",
        "fp8-dictionary",
    ):
        with pytest.raises(pb.ActionContractError, match="D29"):
            _action(
                tmp_path,
                artifact_family="codebook",
                artifact_kind=artifact_kind,
            )

    # These labels were false positives under substring matching. An ordinary
    # family remains portable even when its descriptive kind happens to contain
    # the letters "cb" near a quantization-family token.
    for artifact_kind in ("nvfp4-recbuild", "fp8-cbor-dump"):
        action = _action(
            tmp_path,
            artifact_family="generic",
            artifact_kind=artifact_kind,
        )
        assert (
            action["execution_scope"]["portability"]  # type: ignore[index]
            == "portable"
        )

    platform = _action(
        tmp_path,
        artifact_family="codebook",
        artifact_kind="fp8-lut",
        portability="platform_keyed",
        platform_key="linux-aarch64-sm121",
    )
    assert (
        platform["execution_scope"]["platform_key"]  # type: ignore[index]
        == "linux-aarch64-sm121"
    )

    host = _action(
        tmp_path,
        task_class="measurement",
        portability="host_class_keyed",
        host_class="gb10",
    )
    assert host["execution_scope"]["host_class"] == "gb10"  # type: ignore[index]


def test_action_v2_requires_closed_artifact_family_and_refuses_v1(tmp_path: Path):
    body = _body(tmp_path)
    body["task"]["artifact_family"] = "codebook-ish"  # type: ignore[index]
    with pytest.raises(pb.ActionContractError, match="artifact_family must be one of"):
        pb.seal_action(body)

    legacy = _body(tmp_path)
    legacy["schema"] = pb.ACTION_SCHEMA_V1
    del legacy["task"]["artifact_family"]  # type: ignore[index]
    with pytest.raises(
        pb.ActionContractError,
        match="v1 actions must be redeclared and resealed",
    ):
        pb.seal_action(legacy)


def test_nonportable_action_refuses_unattestable_toolchain_or_unbound_argv0(
    tmp_path: Path,
):
    body = _body(
        tmp_path,
        portability="platform_keyed",
        platform_key="linux-aarch64-sm121",
    )
    body["environment"]["toolchain"] = {"container": "claimed-not-probed"}  # type: ignore[index]
    with pytest.raises(pb.ActionContractError, match="no worker preflight"):
        pb.seal_action(body)

    body["environment"]["toolchain"] = {"python": "3.12"}  # type: ignore[index]
    with pytest.raises(pb.ActionContractError, match=r"bind argv\[0\]"):
        pb.seal_action(body)


def test_platform_scope_is_derived_from_live_facts_not_a_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    body = _body(
        tmp_path,
        portability="platform_keyed",
        platform_key="linux-aarch64-sm121",
    )
    body["inputs"] = []
    body["environment"]["toolchain"].update(  # type: ignore[index]
        {
            "cuda_compute_capability": "8.9",
            "nvidia_driver": "550.54",
            "system": "linux",
            "machine": "x86_64",
            "libc": "glibc-2.39",
        }
    )
    action = pb.seal_action(body)
    monkeypatch.setattr(
        pb,
        "_collect_worker_evidence",
        lambda: {
            "source": "local",
            "hostname": "foreign-worker",
            "system": "linux",
            "machine": "x86_64",
            "libc": "glibc-2.39",
            "accelerators": [
                {
                    "kind": "nvidia",
                    "compute_capability": "8.9",
                    "driver_version": "550.54",
                }
            ],
            "slurm": None,
        },
    )
    with pytest.raises(pb.ActionContractError, match="platform_key"):
        pb.preflight_action(
            action, cas_root=tmp_path / "cas", checkout_root=tmp_path
        )


def test_host_class_requires_kernel_attested_slurm_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    body = _body(
        tmp_path,
        portability="host_class_keyed",
        host_class="gb10",
    )
    body["inputs"] = []
    action = pb.seal_action(body)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURMD_NODENAME", "claimed-node")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gb10")
    with pytest.raises(pb.ActionContractError, match="cgroup membership"):
        pb.preflight_action(
            action, cas_root=tmp_path / "cas", checkout_root=tmp_path
        )


def test_slurm_cgroup_attestation_accepts_duplicate_v1_controller_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    cgroup = b"2:cpu:/slurm/job_4242/step_batch\n3:memory:/slurm/job_4242/step_batch\n"
    monkeypatch.setattr(pb, "_read_regular_file", lambda *args, **kwargs: cgroup)
    assert (
        pb._verify_slurm_process_membership("4242")
        == "/slurm/job_4242/step_batch"
    )


def test_host_class_preflight_records_slurm_and_machine_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    body = _live_nonportable_body(tmp_path)
    body["execution_scope"] = {
        "portability": "host_class_keyed",
        "platform_key": None,
        "host_class": "gb10",
    }
    action = pb.seal_action(body)
    live = pb._collect_worker_evidence()
    live["source"] = "slurm"
    live["slurm"] = {
        "job_id": "77",
        "node_name": "sparky",
        "partition": "gb10",
        "constraints": ["gb10"],
        "cgroup": "/slurm/job_77/step_batch",
    }
    monkeypatch.setattr(pb, "_collect_worker_evidence", lambda: live)
    result = pb.run_local_action(
        action, cas_root=tmp_path / "cas", checkout_root=tmp_path
    )
    producer = result["receipt"]["producer"]  # type: ignore[index]
    assert producer["worker_id"] == "sparky"
    assert producer["host_class"] == "gb10"
    assert producer["evidence"]["slurm"]["cgroup"] == "/slurm/job_77/step_batch"
    assert pb.validate_worker_attestation(producer, action=action) == producer


def test_worker_runtime_attestation_binds_core_and_optional_launcher(
    tmp_path: Path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    launcher = tmp_path / "worker-launcher.py"
    launcher.write_text("# exact launcher\n", encoding="utf-8")
    launcher_identity = pb._identify_runtime_source(
        launcher, where="test worker launcher"
    )

    direct = pb.preflight_action(
        action, cas_root=tmp_path / "cas", checkout_root=checkout
    )
    assert direct["schema"] == pb.WORKER_ATTESTATION_SCHEMA_V2
    assert direct["runtime"]["schema"] == pb.WORKER_RUNTIME_SCHEMA_V1
    assert direct["runtime"]["launch_kind"] == "in_process"
    assert direct["runtime"]["launcher"] is None
    assert direct["runtime"]["core"]["sha256"] == hashlib.sha256(
        Path(pb.__file__).read_bytes()
    ).hexdigest()

    launched = pb.preflight_action(
        action,
        cas_root=tmp_path / "cas",
        checkout_root=checkout,
        worker_launcher_identity=launcher_identity,
    )
    assert launched["runtime"]["launch_kind"] == "script"
    assert launched["runtime"]["launcher"]["sha256"] == hashlib.sha256(
        launcher.read_bytes()
    ).hexdigest()
    assert pb.validate_worker_attestation(launched, action=action) == launched

    malformed = copy.deepcopy(launched)
    malformed["runtime"]["unrecognized"] = True
    runtime_body = {
        key: value
        for key, value in malformed["runtime"].items()
        if key != "runtime_sha256"
    }
    malformed["runtime"]["runtime_sha256"] = pb.canonical_sha256(runtime_body)
    attestation_body = {
        key: value for key, value in malformed.items() if key != "attestation_sha256"
    }
    malformed["attestation_sha256"] = pb.canonical_sha256(attestation_body)
    with pytest.raises(pb.ActionContractError, match="runtime fields differ"):
        pb.validate_worker_attestation(malformed, action=action)

    inconsistent = copy.deepcopy(direct)
    inconsistent["runtime"]["launch_kind"] = "script"
    runtime_body = {
        key: value
        for key, value in inconsistent["runtime"].items()
        if key != "runtime_sha256"
    }
    inconsistent["runtime"]["runtime_sha256"] = pb.canonical_sha256(runtime_body)
    attestation_body = {
        key: value
        for key, value in inconsistent.items()
        if key != "attestation_sha256"
    }
    inconsistent["attestation_sha256"] = pb.canonical_sha256(attestation_body)
    with pytest.raises(
        pb.ActionContractError, match="launch_kind and launcher disagree"
    ):
        pb.validate_worker_attestation(inconsistent, action=action)


def test_worker_core_has_no_unattested_repository_imports():
    source = Path(pb.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    repository_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.level > 0 or (node.module or "").startswith("prismaquant")
        ):
            repository_imports.append(ast.unparse(node))
        if isinstance(node, ast.Import):
            repository_imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "prismaquant"
                or alias.name.startswith("prismaquant.")
            )
    assert repository_imports == []


def test_slurm_worker_entrypoint_attests_its_early_launcher_snapshot(
    tmp_path: Path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    action_path = tmp_path / "action.json"
    action_path.write_text(json.dumps(action), encoding="utf-8")
    repository_root = Path(pb.__file__).resolve().parents[1]
    launcher = repository_root / "tools" / "prismabuild_worker.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "preflight",
            "--action",
            str(action_path),
            "--cas-root",
            str(tmp_path / "cas"),
            "--checkout-root",
            str(checkout),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    attestation = json.loads(completed.stdout)
    runtime = attestation["runtime"]
    assert runtime["launch_kind"] == "script"
    assert runtime["launcher"]["sha256"] == hashlib.sha256(
        launcher.read_bytes()
    ).hexdigest()
    assert runtime["core"]["sha256"] == hashlib.sha256(
        Path(pb.__file__).read_bytes()
    ).hexdigest()


def test_preflight_refuses_core_changed_after_module_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    fake_core = tmp_path / "prismabuild.py"
    fake_core.write_text("# loaded worker core\n", encoding="utf-8")
    captured = pb._identify_runtime_source(fake_core, where="test worker core")
    monkeypatch.setattr(pb, "_LOADED_WORKER_CORE_IDENTITY", captured)
    # The old implementation took its first snapshot from __file__ during
    # preflight and therefore accepted these changed bytes as the loaded core.
    monkeypatch.setattr(pb, "__file__", str(fake_core))
    fake_core.write_text("# changed before first preflight\n", encoding="utf-8")

    with pytest.raises(pb.LocalActionError, match="core changed after module import"):
        pb.preflight_action(
            _action(checkout),
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
        )


def test_preflight_refuses_launcher_changed_after_entrypoint_capture(
    tmp_path: Path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    launcher = tmp_path / "worker-launcher.py"
    launcher.write_text("# launched bytes\n", encoding="utf-8")
    captured = pb._identify_runtime_source(
        launcher, where="test worker launcher"
    )
    launcher.write_text("# changed before preflight\n", encoding="utf-8")

    with pytest.raises(
        pb.LocalActionError, match="launcher changed after entry-point capture"
    ):
        pb.preflight_action(
            _action(checkout),
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
            worker_launcher_identity=captured,
        )


def test_nonportable_preflight_refuses_unknown_libc_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    live = pb._collect_worker_evidence()
    body = _live_nonportable_body(tmp_path)
    body["environment"]["toolchain"]["libc"] = "unknown"  # type: ignore[index]
    action = pb.seal_action(body)
    live["libc"] = "unknown"
    monkeypatch.setattr(pb, "_collect_worker_evidence", lambda: live)
    with pytest.raises(pb.ActionContractError, match="unknown libc"):
        pb.preflight_action(
            action, cas_root=tmp_path / "cas", checkout_root=tmp_path
        )


def test_preflight_refuses_toolchain_mismatch_before_execution(tmp_path: Path):
    body = _body(tmp_path)
    body["inputs"] = []
    body["environment"]["toolchain"] = {"python": "0.0"}  # type: ignore[index]
    action = pb.seal_action(body)
    with pytest.raises(pb.ActionContractError, match="toolchain field 'python'"):
        pb.run_local_action(
            action, cas_root=tmp_path / "cas", checkout_root=tmp_path
        )
    assert not (tmp_path / "result.bin").exists()


def test_input_ingestion_publishes_canonical_readback_verified_contract(
    tmp_path: Path,
):
    source = tmp_path / "source.bin"
    payload = b"immutable input payload"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    cas = pb.PrismaBuildCAS(tmp_path / "cas")

    entry, won = cas.ingest_input(
        source,
        input_id="model/weights",
        expected_sha256=digest,
        expected_bytes=len(payload),
    )

    assert won
    assert entry == {
        "id": "model/weights",
        "sha256": digest,
        "bytes": len(payload),
    }
    path = cas.input_path(entry)
    assert path == cas._blob_path(digest)
    assert path.read_bytes() == payload
    assert path.stat().st_mode & 0o222 == 0
    source.write_bytes(b"source may change after the stable snapshot")
    assert cas.input_path(entry).read_bytes() == payload


def test_input_ingestion_rejects_expectation_mismatch_before_publication(
    tmp_path: Path,
):
    source = tmp_path / "source.bin"
    source.write_bytes(b"actual")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")

    with pytest.raises(pb.ActionContractError, match="sha256 differs"):
        cas.ingest_input(
            source,
            input_id="dataset",
            expected_sha256="0" * 64,
        )
    with pytest.raises(pb.ActionContractError, match="byte count differs"):
        cas.ingest_input(
            source,
            input_id="dataset",
            expected_bytes=999,
        )

    assert not list((tmp_path / "cas" / "blobs").rglob("*"))
    assert not list((tmp_path / "cas" / ".staging").glob("*.tmp"))


def test_input_ingestion_rejects_symlink_source_and_malformed_contract(
    tmp_path: Path,
):
    source = tmp_path / "source.bin"
    source.write_bytes(b"actual")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(source)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")

    with pytest.raises(pb.LocalActionError, match="readable regular file"):
        cas.ingest_input(alias, input_id="dataset")
    with pytest.raises(pb.ActionContractError, match="fields differ"):
        cas.input_path(
            {
                "id": "dataset",
                "sha256": hashlib.sha256(b"actual").hexdigest(),
                "bytes": len(b"actual"),
                "path": str(source),
            }
        )


def test_input_ingestion_race_reuses_one_verified_content_address(tmp_path: Path):
    payload = b"same immutable bytes"
    sources = [tmp_path / f"source-{index}.bin" for index in range(2)]
    for source in sources:
        source.write_bytes(payload)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    barrier = threading.Barrier(2)
    results: list[tuple[dict[str, object], bool]] = []
    errors: list[BaseException] = []

    def ingest(index: int) -> None:
        try:
            barrier.wait()
            results.append(
                cas.ingest_input(sources[index], input_id=f"source/{index}")
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=ingest, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 2
    assert sum(won for _, won in results) == 1
    assert {entry["sha256"] for entry, _ in results} == {
        hashlib.sha256(payload).hexdigest()
    }
    assert all(cas.input_path(entry).read_bytes() == payload for entry, _ in results)


@pytest.mark.parametrize("transition", ["two-to-one", "one-to-one"])
def test_stable_file_identity_replays_transient_link_metadata_from_fresh_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
):
    payload_bytes = b"stable content-addressed payload"
    payload = tmp_path / "payload"
    payload.write_bytes(payload_bytes)
    payload.chmod(0o444)
    extra_link = tmp_path / "publication-link"
    if transition == "two-to-one":
        os.link(payload, extra_link)
        assert payload.stat().st_nlink == 2

    original_open = pb._open_regular_nofollow
    original_read = pb.os.read
    open_count = 0
    mutated = False

    def tracked_open(path: Path, *, where: str) -> tuple[int, int]:
        nonlocal open_count
        open_count += 1
        return original_open(path, where=where)

    def read_then_change_link_metadata(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            if transition == "two-to-one":
                extra_link.unlink()
            else:
                # Model a link/unlink ctime disturbance whose before/after
                # link count remains one.  The first attempt must be thrown
                # away rather than accepted under a ctime exception.
                time.sleep(0.001)
                os.link(payload, extra_link)
                extra_link.unlink()
        return chunk

    monkeypatch.setattr(pb, "_open_regular_nofollow", tracked_open)
    monkeypatch.setattr(pb.os, "read", read_then_change_link_metadata)
    digest = hashlib.sha256(payload_bytes).hexdigest()

    assert pb._file_identity_nofollow(
        payload,
        where="test payload",
        expected_sha256=digest,
        expected_bytes=len(payload_bytes),
    ) == (digest, len(payload_bytes), payload.stat().st_mode)
    assert open_count == 2
    assert payload.stat().st_nlink == 1


def test_stable_receipt_read_replays_publication_link_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    receipt_bytes = b'{"receipt":"stable"}\n'
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(receipt_bytes)
    receipt.chmod(0o444)
    publication_link = tmp_path / "receipt-publication-link"
    os.link(receipt, publication_link)

    original_open = pb._open_regular_nofollow
    original_read = pb.os.read
    open_count = 0
    mutated = False

    def tracked_open(path: Path, *, where: str) -> tuple[int, int]:
        nonlocal open_count
        open_count += 1
        return original_open(path, where=where)

    def read_then_unlink(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            publication_link.unlink()
        return chunk

    monkeypatch.setattr(pb, "_open_regular_nofollow", tracked_open)
    monkeypatch.setattr(pb.os, "read", read_then_unlink)

    assert pb._read_regular_file_nofollow(
        receipt, where="test receipt", require_readonly=True
    ) == receipt_bytes
    assert open_count == 2


def test_stable_file_identity_refuses_perpetual_one_link_ctime_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload_bytes = b"same bytes, hostile metadata churn"
    payload = tmp_path / "payload"
    payload.write_bytes(payload_bytes)
    payload.chmod(0o444)
    transient_link = tmp_path / "transient-link"

    original_open = pb._open_regular_nofollow
    original_read = pb.os.read
    open_count = 0
    mutation_count = 0

    def tracked_open(path: Path, *, where: str) -> tuple[int, int]:
        nonlocal open_count
        open_count += 1
        return original_open(path, where=where)

    def read_then_churn_link(descriptor: int, count: int) -> bytes:
        nonlocal mutation_count
        chunk = original_read(descriptor, count)
        if chunk:
            time.sleep(0.001)
            os.link(payload, transient_link)
            transient_link.unlink()
            mutation_count += 1
        return chunk

    monkeypatch.setattr(pb, "_open_regular_nofollow", tracked_open)
    monkeypatch.setattr(pb.os, "read", read_then_churn_link)
    digest = hashlib.sha256(payload_bytes).hexdigest()

    with pytest.raises(
        pb.CASTamperError,
        match=rf"did not stabilize after {pb._STABLE_FILE_READ_ATTEMPTS} fresh reads",
    ):
        pb._file_identity_nofollow(
            payload,
            where="test payload",
            expected_sha256=digest,
            expected_bytes=len(payload_bytes),
        )
    assert open_count == pb._STABLE_FILE_READ_ATTEMPTS
    assert mutation_count == pb._STABLE_FILE_READ_ATTEMPTS
    assert payload.stat().st_nlink == 1


@pytest.mark.parametrize("mutation", ["mode", "mtime"])
def test_stable_file_identity_refuses_substantive_mutation_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    payload_bytes = b"immutable payload"
    payload = tmp_path / "payload"
    payload.write_bytes(payload_bytes)
    payload.chmod(0o444)
    original = payload.stat()

    original_open = pb._open_regular_nofollow
    original_read = pb.os.read
    open_count = 0
    mutated = False

    def tracked_open(path: Path, *, where: str) -> tuple[int, int]:
        nonlocal open_count
        open_count += 1
        return original_open(path, where=where)

    def read_then_mutate(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            if mutation == "mode":
                payload.chmod(0o400)
            else:
                os.utime(
                    payload,
                    ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000),
                )
        return chunk

    monkeypatch.setattr(pb, "_open_regular_nofollow", tracked_open)
    monkeypatch.setattr(pb.os, "read", read_then_mutate)
    digest = hashlib.sha256(payload_bytes).hexdigest()

    with pytest.raises(pb.CASTamperError, match="changed substantively"):
        pb._file_identity_nofollow(
            payload,
            where="test payload",
            expected_sha256=digest,
            expected_bytes=len(payload_bytes),
        )
    assert open_count == 1


def test_stable_file_identity_refuses_owner_mutation_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload_bytes = b"immutable payload"
    payload = tmp_path / "payload"
    payload.write_bytes(payload_bytes)
    payload.chmod(0o444)
    digest = hashlib.sha256(payload_bytes).hexdigest()

    original_open = pb._open_regular_nofollow
    original_fstat = pb.os.fstat
    payload_descriptor: int | None = None
    payload_fstat_count = 0
    open_count = 0

    def tracked_open(path: Path, *, where: str) -> tuple[int, int]:
        nonlocal open_count, payload_descriptor
        open_count += 1
        payload_descriptor, parent_fd = original_open(path, where=where)
        return payload_descriptor, parent_fd

    def fstat_with_changed_owner(descriptor: int) -> os.stat_result:
        nonlocal payload_fstat_count
        observed = original_fstat(descriptor)
        if descriptor != payload_descriptor:
            return observed
        payload_fstat_count += 1
        if payload_fstat_count != 2:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mode=observed.st_mode,
            st_uid=observed.st_uid + 1,
            st_gid=observed.st_gid,
            st_mtime_ns=observed.st_mtime_ns,
            st_nlink=observed.st_nlink,
            st_ctime_ns=observed.st_ctime_ns,
        )  # type: ignore[return-value]

    monkeypatch.setattr(pb, "_open_regular_nofollow", tracked_open)
    monkeypatch.setattr(pb.os, "fstat", fstat_with_changed_owner)

    with pytest.raises(pb.CASTamperError, match="changed substantively"):
        pb._file_identity_nofollow(
            payload,
            where="test payload",
            expected_sha256=digest,
            expected_bytes=len(payload_bytes),
        )
    assert open_count == 1


def test_stable_file_identity_refuses_wrong_address_before_metadata_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = tmp_path / "payload"
    payload.write_bytes(b"wrong address")
    payload.chmod(0o444)
    transient_link = tmp_path / "transient-link"
    original_open = pb._open_regular_nofollow
    original_read = pb.os.read
    open_count = 0

    def tracked_open(path: Path, *, where: str) -> tuple[int, int]:
        nonlocal open_count
        open_count += 1
        return original_open(path, where=where)

    def read_then_churn_link(descriptor: int, count: int) -> bytes:
        chunk = original_read(descriptor, count)
        if chunk and not transient_link.exists():
            os.link(payload, transient_link)
            transient_link.unlink()
        return chunk

    monkeypatch.setattr(pb, "_open_regular_nofollow", tracked_open)
    monkeypatch.setattr(pb.os, "read", read_then_churn_link)

    with pytest.raises(pb.CASTamperError, match="expected address"):
        pb._file_identity_nofollow(
            payload,
            where="test payload",
            expected_sha256="0" * 64,
            expected_bytes=len(b"wrong address"),
        )
    assert open_count == 1


def test_input_ingestion_and_lookup_detect_conflict_and_tampering(tmp_path: Path):
    payload = b"trusted input"
    digest = hashlib.sha256(payload).hexdigest()
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    blob = cas._blob_path(digest)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"wrong content")

    with pytest.raises(pb.CASTamperError, match="payload content differs"):
        cas.ingest_input(source, input_id="dataset")

    blob.write_bytes(payload)
    with pytest.raises(pb.CASTamperError, match="input payload is writable"):
        cas.ingest_input(source, input_id="dataset")
    blob.chmod(0o444)
    entry, won = cas.ingest_input(source, input_id="dataset")
    assert not won
    blob.chmod(0o644)
    with pytest.raises(pb.CASTamperError, match="input payload is writable"):
        cas.input_path(entry)


def test_input_ingestion_revalidates_canonical_inode_after_winning_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = b"trusted input"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    original_link = pb.os.link

    def link_then_corrupt(
        source_path: Path, destination_path: str, **kwargs: object
    ) -> None:
        original_link(source_path, destination_path, **kwargs)
        directory_fd = kwargs["dst_dir_fd"]
        destination = Path(f"/proc/self/fd/{directory_fd}") / destination_path
        destination.chmod(0o644)
        destination.write_bytes(b"changed input")

    monkeypatch.setattr(pb.os, "link", link_then_corrupt)
    with pytest.raises(pb.CASTamperError, match="verified staging inode"):
        cas.ingest_input(source, input_id="dataset")


def test_blob_publication_skips_only_winning_inode_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = b"trusted input"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    original_identity = pb._file_identity_nofollow
    consumed: list[Path] = []

    def track_identity(path: Path, **kwargs: object):
        consumed.append(path)
        return original_identity(path, **kwargs)

    monkeypatch.setattr(pb, "_file_identity_nofollow", track_identity)
    entry, won = cas.ingest_input(source, input_id="dataset")
    assert won is True
    assert consumed == []

    assert cas.input_path(entry) == cas._blob_path(str(entry["sha256"]))
    assert consumed == [cas._blob_path(str(entry["sha256"]))]

    consumed.clear()
    repeated, won = cas.ingest_input(source, input_id="dataset")
    assert repeated == entry
    assert won is False
    assert consumed == [cas._blob_path(str(entry["sha256"]))]


def test_input_ingestion_refuses_symlinked_cas_shard(tmp_path: Path):
    payload = b"trusted input"
    digest = hashlib.sha256(payload).hexdigest()
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    outside = tmp_path / "outside"
    outside.mkdir()
    blobs = tmp_path / "cas" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / digest[:2]).symlink_to(outside, target_is_directory=True)

    with pytest.raises(pb.CASTamperError, match="not a real directory"):
        pb.PrismaBuildCAS(tmp_path / "cas").ingest_input(
            source, input_id="dataset"
        )
    assert list(outside.iterdir()) == []


def test_input_ingestion_refuses_source_changed_during_stable_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.bin"
    source.write_bytes(b"a" * (5 * 1024 * 1024))
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    original_read = pb.os.read
    changed = False

    def read_then_change(descriptor: int, count: int) -> bytes:
        nonlocal changed
        chunk = original_read(descriptor, count)
        if chunk and not changed:
            changed = True
            with source.open("ab") as handle:
                handle.write(b"changed")
        return chunk

    monkeypatch.setattr(pb.os, "read", read_then_change)
    with pytest.raises(pb.LocalActionError, match="changed while it was copied"):
        cas.ingest_input(source, input_id="dataset")
    assert not list((tmp_path / "cas" / "blobs").rglob("*"))


def test_nonportable_preflight_requires_and_verifies_cas_inputs(tmp_path: Path):
    payload = b"immutable upstream"
    digest = hashlib.sha256(payload).hexdigest()
    entry = {"id": "upstream/result", "sha256": digest, "bytes": len(payload)}
    action = pb.seal_action(_live_nonportable_body(tmp_path, inputs=[entry]))
    with pytest.raises(pb.ActionContractError, match="input is absent"):
        pb.preflight_action(
            action, cas_root=tmp_path / "cas", checkout_root=tmp_path
        )

    source = tmp_path / "input.bin"
    source.write_bytes(payload)
    observed, won = pb.PrismaBuildCAS(tmp_path / "cas").ingest_input(
        source,
        input_id="upstream/result",
        expected_sha256=digest,
        expected_bytes=len(payload),
    )
    assert won
    assert observed == entry
    attestation = pb.preflight_action(
        action, cas_root=tmp_path / "cas", checkout_root=tmp_path
    )
    assert attestation["inputs"] == [entry]


def test_nonportable_preflight_binds_executable_bytes(tmp_path: Path):
    executable = tmp_path / "task-executable"
    executable.write_text(
        "#!/usr/bin/python3.12\nimport pathlib\npathlib.Path('result.bin').write_bytes(b'ok')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    body = _live_nonportable_body(tmp_path, argv=[str(executable)])
    body["environment"]["toolchain"] = {  # type: ignore[index]
        **pb.executable_toolchain_contract(executable),
        **{
            key: value
            for key, value in body["environment"]["toolchain"].items()  # type: ignore[index]
            if key
            in {
                "cuda_compute_capability",
                "nvidia_driver",
                "system",
                "machine",
                "libc",
            }
        },
    }
    action = pb.seal_action(body)
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    with pytest.raises(pb.ActionContractError, match=r"argv0\.(sha256|bytes)"):
        pb.run_local_action(
            action, cas_root=tmp_path / "cas", checkout_root=tmp_path
        )
    assert not (tmp_path / "result.bin").exists()


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
    )
    first = pb.run_local_action(**arguments)
    second = pb.run_local_action(**arguments)
    assert first["status"] == "published"
    assert second["status"] == "cache_hit"
    assert (checkout / "marker").read_text(encoding="utf-8") == "x"

    (checkout / "task_code.py").write_text("# closure changed\n", encoding="utf-8")
    with pytest.raises(pb.ActionContractError, match="live code closure differs"):
        pb.run_local_action(**arguments)


def test_initial_miss_rendezvous_two_hosts_publish_once_then_hit_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    code = (
        "import pathlib,time; "
        "p=pathlib.Path('task-starts'); "
        "p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
        "time.sleep(0.05); pathlib.Path('result.bin').write_bytes(b'ok')"
    )
    action = _action(checkout, argv=[sys.executable, "-c", code])
    manifest_path, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=tmp_path / "rendezvous-state",
    )
    monkeypatch.setattr(
        pb, "_initial_miss_hostname", lambda: threading.current_thread().name
    )
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def run() -> None:
        try:
            start.wait()
            results.append(
                pb.run_local_action(
                    action,
                    cas_root=tmp_path / "cas",
                    checkout_root=checkout,
                    initial_miss_rendezvous=manifest_path,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, name=participant)
        for participant in manifest["participants"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(result["status"] for result in results) == [
        "cache_hit",
        "published",
    ]
    assert (checkout / "task-starts").read_text(encoding="utf-8") == "x"
    proofs = [result["initial_miss_rendezvous"] for result in results]
    assert {proof["participant"] for proof in proofs} == {"host-a", "host-b"}
    assert {proof["arrival_set_sha256"] for proof in proofs} == {
        proofs[0]["arrival_set_sha256"]
    }
    assert {proof["ready_set_sha256"] for proof in proofs} == {
        proofs[0]["ready_set_sha256"]
    }
    for proof in proofs:
        body = {
            key: proof[key]
            for key in proof
            if key != "receipt_sha256"
        }
        assert proof["schema"] == pb.INITIAL_MISS_RENDEZVOUS_RECEIPT_SCHEMA_V1
        assert proof["receipt_sha256"] == pb.canonical_sha256(body)


def test_initial_miss_rendezvous_refuses_two_cas_roots_before_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkouts = {
        "host-a": tmp_path / "checkout-a",
        "host-b": tmp_path / "checkout-b",
    }
    for checkout in checkouts.values():
        checkout.mkdir()
        (checkout / "task_code.py").write_text(
            "# closure member\n", encoding="utf-8"
        )
    action = _action(checkouts["host-a"])
    cas_roots = {
        "host-a": tmp_path / "cas-a",
        "host-b": tmp_path / "cas-b",
    }
    manifest_path, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=tmp_path / "rendezvous-state",
        cas_root=cas_roots["host-a"],
        timeout_seconds=0.1,
    )
    monkeypatch.setattr(
        pb, "_initial_miss_hostname", lambda: threading.current_thread().name
    )
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def run(host: str) -> None:
        try:
            start.wait()
            results.append(
                pb.run_local_action(
                    action,
                    cas_root=cas_roots[host],
                    checkout_root=checkouts[host],
                    initial_miss_rendezvous=manifest_path,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(host,), name=host)
        for host in manifest["participants"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)
    assert not any(thread.is_alive() for thread in threads)
    assert results == []
    assert len(errors) == 2
    assert any("CAS root differs" in str(error) for error in errors)
    assert any("timed out waiting for exact arrivals" in str(error) for error in errors)
    assert all(not (checkout / "result.bin").exists() for checkout in checkouts.values())
    assert all(not (root / "actions").exists() for root in cas_roots.values())


def test_initial_cache_hit_never_enters_or_parses_rendezvous(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    arguments = {
        "action": action,
        "cas_root": tmp_path / "cas",
        "checkout_root": checkout,
    }
    first = pb.run_local_action(**arguments)
    nonexistent_manifest = tmp_path / "must-not-be-read.json"
    second = pb.run_local_action(
        **arguments,
        initial_miss_rendezvous=nonexistent_manifest,
    )
    assert first["status"] == "published"
    assert second["status"] == "cache_hit"
    assert "initial_miss_rendezvous" not in second
    assert not nonexistent_manifest.exists()


def test_initial_miss_rendezvous_rejects_recompute_through_api_and_cli(
    tmp_path: Path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    action_path = tmp_path / "action.json"
    action_path.write_text(json.dumps(action), encoding="utf-8")
    cas_root = tmp_path / "cas"
    nonexistent_manifest = tmp_path / "must-not-be-read.json"

    with pytest.raises(
        pb.ActionContractError,
        match="initial-miss rendezvous is incompatible with recompute",
    ):
        pb.run_local_action(
            action,
            cas_root=cas_root,
            checkout_root=checkout,
            recompute=True,
            initial_miss_rendezvous=nonexistent_manifest,
        )

    with pytest.raises(
        pb.ActionContractError,
        match="initial-miss rendezvous is incompatible with recompute",
    ):
        pb.main(
            [
                "run-local",
                "--action",
                str(action_path),
                "--cas-root",
                str(cas_root),
                "--checkout-root",
                str(checkout),
                "--recompute",
                "--initial-miss-rendezvous",
                str(nonexistent_manifest),
            ]
        )

    assert not nonexistent_manifest.exists()
    assert not (checkout / "result.bin").exists()


def test_initial_miss_rendezvous_one_arrival_times_out_without_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    manifest_path, _ = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=tmp_path / "rendezvous-state",
        timeout_seconds=0.05,
    )
    monkeypatch.setattr(pb, "_initial_miss_hostname", lambda: "host-a")
    with pytest.raises(
        pb.InitialMissRendezvousError, match="timed out waiting for exact arrivals"
    ):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
            initial_miss_rendezvous=manifest_path,
        )
    assert not (checkout / "result.bin").exists()
    assert {path.name for path in (tmp_path / "rendezvous-state/arrivals").iterdir()} == {
        "host-a.json"
    }
    assert list((tmp_path / "rendezvous-state/ready").iterdir()) == []


def test_initial_miss_rendezvous_defeats_post_wrapper_sigstop_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A wrapper marker cannot release a peer before the stalled worker enters."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(
        checkout,
        argv=[
            sys.executable,
            "-c",
            (
                "import pathlib; "
                "pathlib.Path('task-entered').write_text('entered'); "
                "pathlib.Path('result.bin').write_bytes(b'ok')"
            ),
        ],
    )
    manifest_path, _ = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=tmp_path / "rendezvous-state",
    )
    monkeypatch.setattr(
        pb, "_initial_miss_hostname", lambda: threading.current_thread().name
    )
    wrapper_marker = threading.Event()
    release_stalled_wrapper = threading.Event()
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def worker_a() -> None:
        try:
            results.append(
                pb.run_local_action(
                    action,
                    cas_root=tmp_path / "cas",
                    checkout_root=checkout,
                    initial_miss_rendezvous=manifest_path,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def stalled_worker_b() -> None:
        wrapper_marker.set()  # V3 treated this pre-exec event as worker start.
        release_stalled_wrapper.wait(timeout=2.0)  # deterministic SIGSTOP seam
        worker_a()

    first = threading.Thread(target=worker_a, name="host-a")
    stalled = threading.Thread(target=stalled_worker_b, name="host-b")
    first.start()
    stalled.start()
    assert wrapper_marker.wait(timeout=1.0)
    time.sleep(0.08)
    assert first.is_alive()
    assert not (checkout / "task-entered").exists()
    release_stalled_wrapper.set()
    first.join(timeout=5.0)
    stalled.join(timeout=5.0)
    assert not first.is_alive() and not stalled.is_alive()
    assert errors == []
    assert sorted(result["status"] for result in results) == [
        "cache_hit",
        "published",
    ]
    assert (checkout / "task-entered").read_text(encoding="utf-8") == "entered"


def test_unset_initial_miss_rendezvous_preserves_exact_result_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)

    def must_not_run(**_: object) -> dict[str, object]:
        raise AssertionError("unset rendezvous path was called")

    monkeypatch.setattr(pb, "_run_initial_miss_rendezvous", must_not_run)
    arguments = {
        "action": action,
        "cas_root": tmp_path / "cas",
        "checkout_root": checkout,
    }
    first = pb.run_local_action(**arguments)
    second = pb.run_local_action(**arguments)
    assert set(first) == {
        "status",
        "receipt",
        "payload_path",
        "recovered_declared_result",
        "reaped_staging_files",
        "local_result_claim_sha256",
    }
    assert set(second) == {"status", "receipt", "payload_path"}


@pytest.mark.parametrize(
    "mutation",
    [
        "cross_manifest",
        "stale_nonce",
        "wrong_action",
        "wrong_host",
        "wrong_runtime",
        "forged_digest",
    ],
)
def test_initial_miss_rendezvous_rejects_mismatched_or_corrupt_peer_arrival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    namespace = tmp_path / "rendezvous-state"
    manifest_path, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=namespace,
        timeout_seconds=0.2,
    )
    runtime = pb._worker_runtime_identity(None)
    peer_process = pb._initial_miss_process_identity(
        hostname="host-b", worker_launcher_identity=None
    )
    peer_arrival = pb._initial_miss_arrival(
        manifest=manifest,
        participant="host-b",
        process=peer_process,
    )
    if mutation == "cross_manifest":
        peer_arrival["manifest_sha256"] = "6" * 64
        _reseal(peer_arrival, "arrival_sha256")
    elif mutation == "stale_nonce":
        peer_arrival["run_nonce"] = "2" * 32
        _reseal(peer_arrival, "arrival_sha256")
    elif mutation == "wrong_action":
        peer_arrival["action_key"] = "3" * 64
        _reseal(peer_arrival, "arrival_sha256")
    elif mutation == "wrong_host":
        peer_arrival["participant"] = "host-a"
        _reseal(peer_arrival, "arrival_sha256")
    elif mutation == "wrong_runtime":
        hostile_runtime = copy.deepcopy(runtime)
        hostile_runtime["core"]["sha256"] = "4" * 64  # type: ignore[index]
        _reseal(hostile_runtime, "runtime_sha256")
        peer_process["runtime"] = hostile_runtime
        _reseal(peer_process, "process_identity_sha256")
        peer_arrival["process"] = peer_process
        _reseal(peer_arrival, "arrival_sha256")
    else:
        peer_arrival["arrival_sha256"] = "5" * 64
    _write_rendezvous_record(
        namespace / "arrivals" / "host-b.json", peer_arrival
    )
    (namespace / "ready").mkdir()
    monkeypatch.setattr(pb, "_initial_miss_hostname", lambda: "host-a")

    with pytest.raises(
        pb.PrismaBuildError,
        match="manifest_sha256|run_nonce|action_key|participant|runtime|digest",
    ):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
            initial_miss_rendezvous=manifest_path,
        )
    assert not (checkout / "result.bin").exists()


def test_initial_miss_rendezvous_post_release_stop_allows_peer_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A stop after exact ready release cannot invalidate the causal proof."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    manifest_path, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=tmp_path / "rendezvous-state",
    )
    monkeypatch.setattr(
        pb, "_initial_miss_hostname", lambda: threading.current_thread().name
    )
    original_rendezvous = pb._run_initial_miss_rendezvous
    stopped_after_release = threading.Event()
    resume_after_peer_publish = threading.Event()

    def stop_host_b_after_release(**kwargs: object) -> dict[str, object]:
        proof = original_rendezvous(**kwargs)
        if threading.current_thread().name == "host-b":
            stopped_after_release.set()
            resume_after_peer_publish.wait(timeout=3.0)
        return proof

    monkeypatch.setattr(
        pb, "_run_initial_miss_rendezvous", stop_host_b_after_release
    )
    results: dict[str, dict[str, object]] = {}
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results[threading.current_thread().name] = pb.run_local_action(
                action,
                cas_root=tmp_path / "cas",
                checkout_root=checkout,
                initial_miss_rendezvous=manifest_path,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, name=participant)
        for participant in manifest["participants"]
    ]
    for thread in threads:
        thread.start()
    assert stopped_after_release.wait(timeout=2.0)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    deadline = time.monotonic() + 3.0
    while cas.lookup(action) is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert cas.lookup(action) is not None
    assert threads[1].is_alive()
    resume_after_peer_publish.set()
    for thread in threads:
        thread.join(timeout=3.0)
    assert errors == []
    assert results["host-a"]["status"] == "published"
    assert results["host-b"]["status"] == "cache_hit"
    proof = results["host-b"]["initial_miss_rendezvous"]
    assert proof["participant"] == "host-b"


def test_ready_wait_polls_through_multiple_stale_scans_after_post_release_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    _, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=tmp_path / "rendezvous-state",
    )
    arrivals: dict[str, dict[str, object]] = {}
    ready: dict[str, dict[str, object]] = {}
    for participant in manifest["participants"]:
        process = pb._initial_miss_process_identity(
            hostname=participant, worker_launcher_identity=None
        )
        arrivals[participant] = pb._initial_miss_arrival(
            manifest=manifest, participant=participant, process=process
        )
    arrival_set_sha256 = pb.canonical_sha256(
        [arrivals[name] for name in manifest["participants"]]
    )
    for participant in manifest["participants"]:
        ready[participant] = pb._initial_miss_ready(
            manifest=manifest,
            participant=participant,
            process=arrivals[participant]["process"],
            arrival=arrivals[participant],
            arrival_set_sha256=arrival_set_sha256,
        )
    scans = iter([None, None, ready])
    monkeypatch.setattr(
        pb, "_scan_initial_miss_phase", lambda *args, **kwargs: next(scans)
    )
    fake_cas = SimpleNamespace(lookup=lambda _: {"receipt": "post-release"})
    observed = pb._wait_initial_miss_ready(
        directory=tmp_path / "not-read",
        manifest=manifest,
        arrivals=arrivals,
        arrival_set_sha256=arrival_set_sha256,
        cas=fake_cas,
        action=action,
        deadline=time.monotonic() + 1.0,
    )
    assert observed == ready


@pytest.mark.parametrize(
    "mutation",
    ["wrong_arrival", "wrong_arrival_set", "wrong_process", "forged_digest"],
)
def test_initial_miss_rendezvous_rejects_ready_not_bound_to_exact_arrivals(
    tmp_path: Path, mutation: str
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    _, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=tmp_path / "rendezvous-state",
    )
    runtime = pb._worker_runtime_identity(None)
    arrivals: dict[str, dict[str, object]] = {}
    for participant in manifest["participants"]:
        process = pb._initial_miss_process_identity(
            hostname=participant, worker_launcher_identity=None
        )
        arrivals[participant] = pb._initial_miss_arrival(
            manifest=manifest,
            participant=participant,
            process=process,
        )
    arrival_set_sha256 = pb.canonical_sha256(
        [arrivals[name] for name in manifest["participants"]]
    )
    peer = pb._initial_miss_ready(
        manifest=manifest,
        participant="host-b",
        process=arrivals["host-b"]["process"],
        arrival=arrivals["host-b"],
        arrival_set_sha256=arrival_set_sha256,
    )
    if mutation == "wrong_arrival":
        peer["arrival_sha256"] = "7" * 64
        _reseal(peer, "ready_sha256")
    elif mutation == "wrong_arrival_set":
        peer["arrival_set_sha256"] = "8" * 64
        _reseal(peer, "ready_sha256")
    elif mutation == "wrong_process":
        peer["process_identity_sha256"] = "9" * 64
        _reseal(peer, "ready_sha256")
    else:
        peer["ready_sha256"] = "a" * 64
    with pytest.raises(pb.PrismaBuildError, match="arrival|process|digest"):
        pb._validate_initial_miss_ready(
            peer,
            manifest=manifest,
            participant="host-b",
            arrivals=arrivals,
            arrival_set_sha256=arrival_set_sha256,
        )
    # The runtime used to construct both exact arrivals remains the live one.
    assert arrivals["host-a"]["process"]["runtime"] == runtime


def test_initial_miss_rendezvous_rejects_duplicate_replayed_participant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    namespace = tmp_path / "rendezvous-state"
    manifest_path, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=namespace,
    )
    stale_process = pb._initial_miss_process_identity(
        hostname="host-a", worker_launcher_identity=None
    )
    replay = pb._initial_miss_arrival(
        manifest=manifest,
        participant="host-a",
        process=stale_process,
    )
    _write_rendezvous_record(namespace / "arrivals/host-a.json", replay)
    (namespace / "ready").mkdir()
    monkeypatch.setattr(pb, "_initial_miss_hostname", lambda: "host-a")
    with pytest.raises(
        pb.InitialMissRendezvousError, match="duplicate or replayed.*arrival"
    ):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
            initial_miss_rendezvous=manifest_path,
        )


def test_initial_miss_rendezvous_rejects_receipt_before_ready_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    namespace = tmp_path / "rendezvous-state"
    manifest_path, _ = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=namespace,
        timeout_seconds=1.0,
    )
    monkeypatch.setattr(pb, "_initial_miss_hostname", lambda: "host-a")
    errors: list[BaseException] = []

    def waiting_worker() -> None:
        try:
            pb.run_local_action(
                action,
                cas_root=tmp_path / "cas",
                checkout_root=checkout,
                initial_miss_rendezvous=manifest_path,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=waiting_worker)
    thread.start()
    arrival = namespace / "arrivals/host-a.json"
    deadline = time.monotonic() + 2.0
    while not arrival.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert arrival.exists()
    external = tmp_path / "external-result"
    external.write_bytes(b"external")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    cas.publish_result(
        action,
        external,
        attestation=_attestation(checkout, action, tmp_path / "cas"),
    )
    thread.join(timeout=3.0)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], pb.InitialMissRendezvousError)
    assert "CAS receipt appeared" in str(errors[0])
    assert not (checkout / "result.bin").exists()


def test_initial_miss_rendezvous_rejects_receipt_after_arrivals_before_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    namespace = tmp_path / "rendezvous-state"
    manifest_path, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=namespace,
        timeout_seconds=1.0,
    )
    monkeypatch.setattr(
        pb, "_initial_miss_hostname", lambda: threading.current_thread().name
    )
    original_publish = pb._publish_initial_miss_record
    ready_callers = 0
    ready_callers_lock = threading.Lock()
    both_ready_attempts = threading.Event()
    release_ready_attempts = threading.Event()

    def pause_ready(path: Path, record: object, **kwargs: object) -> None:
        nonlocal ready_callers
        if kwargs["phase"] == "ready":
            with ready_callers_lock:
                ready_callers += 1
                if ready_callers == 2:
                    both_ready_attempts.set()
            release_ready_attempts.wait(timeout=2.0)
        original_publish(path, record, **kwargs)

    monkeypatch.setattr(pb, "_publish_initial_miss_record", pause_ready)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            pb.run_local_action(
                action,
                cas_root=tmp_path / "cas",
                checkout_root=checkout,
                initial_miss_rendezvous=manifest_path,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, name=participant)
        for participant in manifest["participants"]
    ]
    for thread in threads:
        thread.start()
    assert both_ready_attempts.wait(timeout=2.0)
    external = tmp_path / "external-result"
    external.write_bytes(b"external")
    pb.PrismaBuildCAS(tmp_path / "cas").publish_result(
        action,
        external,
        attestation=_attestation(checkout, action, tmp_path / "cas"),
    )
    release_ready_attempts.set()
    for thread in threads:
        thread.join(timeout=3.0)
    assert not any(thread.is_alive() for thread in threads)
    assert len(errors) == 2
    assert all(
        isinstance(error, pb.InitialMissRendezvousError)
        and "CAS receipt appeared before initial-miss rendezvous release" in str(error)
        for error in errors
    )
    assert list((namespace / "ready").iterdir()) == []
    assert not (checkout / "result.bin").exists()


@pytest.mark.parametrize("phase", ["arrival", "ready"])
@pytest.mark.parametrize(
    "hostility", ["missing", "extra", "hardlink", "symlink", "special", "mutable"]
)
def test_initial_miss_phase_rejects_nonexact_or_unsafe_records(
    tmp_path: Path,
    phase: str,
    hostility: str,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    _, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=tmp_path / "unused-namespace",
    )
    runtime = pb._worker_runtime_identity(None)
    arrivals: dict[str, dict[str, object]] = {}
    for participant in manifest["participants"]:
        process = pb._initial_miss_process_identity(
            hostname=participant, worker_launcher_identity=None
        )
        arrivals[participant] = pb._initial_miss_arrival(
            manifest=manifest,
            participant=participant,
            process=process,
        )
    arrival_set_sha256 = pb.canonical_sha256(
        [arrivals[name] for name in manifest["participants"]]
    )
    if phase == "arrival":
        records = arrivals

        def validate(value: object, participant: str) -> dict[str, object]:
            return pb._validate_initial_miss_arrival(
                value,
                manifest=manifest,
                participant=participant,
                runtime=runtime,
            )

    else:
        records = {
            participant: pb._initial_miss_ready(
                manifest=manifest,
                participant=participant,
                process=arrivals[participant]["process"],
                arrival=arrivals[participant],
                arrival_set_sha256=arrival_set_sha256,
            )
            for participant in manifest["participants"]
        }

        def validate(value: object, participant: str) -> dict[str, object]:
            return pb._validate_initial_miss_ready(
                value,
                manifest=manifest,
                participant=participant,
                arrivals=arrivals,
                arrival_set_sha256=arrival_set_sha256,
            )

    directory = tmp_path / phase
    for participant, record in records.items():
        _write_rendezvous_record(directory / f"{participant}.json", record)
    target = directory / "host-b.json"
    if hostility == "missing":
        target.unlink()
    elif hostility == "extra":
        _write_rendezvous_record(directory / "intruder.json", {"bad": True})
    elif hostility == "hardlink":
        raw = target.read_bytes()
        target.unlink()
        other = tmp_path / f"{phase}-hardlink-source"
        other.write_bytes(raw)
        other.chmod(0o444)
        os.link(other, target)
    elif hostility == "symlink":
        target.unlink()
        target.symlink_to(directory / "host-a.json")
    elif hostility == "special":
        target.unlink()
        os.mkfifo(target)
    else:
        target.chmod(0o644)

    if hostility in {"missing", "hardlink"}:
        assert (
            pb._scan_initial_miss_phase(
                directory,
                phase=phase,
                participants=manifest["participants"],
                validate=validate,
            )
            is None
        )
    else:
        with pytest.raises(pb.PrismaBuildError):
            pb._scan_initial_miss_phase(
                directory,
                phase=phase,
                participants=manifest["participants"],
                validate=validate,
            )


def test_regular_identity_fifo_swap_never_performs_blocking_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "record.json"
    path.write_bytes(b"{}\n")
    descriptor, parent_fd = pb._open_regular_nofollow(
        path, where="FIFO-swap fixture"
    )
    path.unlink()
    os.mkfifo(path)
    real_open = pb.os.open

    def guarded_open(candidate: object, *args: object, **kwargs: object) -> int:
        if candidate == path.name and kwargs.get("dir_fd") == parent_fd:
            raise AssertionError("blocking canonical FIFO reopen attempted")
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(pb.os, "open", guarded_open)
    try:
        with pytest.raises(pb.CASTamperError, match="changed during operation"):
            pb._assert_regular_identity(
                descriptor, parent_fd, path, where="FIFO-swap fixture"
            )
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def test_atomic_publish_fifo_swap_never_performs_blocking_readback_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "published.json"
    real_link = pb.os.link
    real_open = pb.os.open
    swapped = False

    def swapping_link(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        real_link(*args, **kwargs)
        destination = args[1]
        destination_directory = kwargs["dst_dir_fd"]
        os.unlink(destination, dir_fd=destination_directory)
        os.mkfifo(destination, dir_fd=destination_directory)
        swapped = True

    def guarded_open(candidate: object, *args: object, **kwargs: object) -> int:
        if swapped and candidate == path.name:
            raise AssertionError("blocking published FIFO readback open attempted")
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(pb.os, "link", swapping_link)
    monkeypatch.setattr(pb.os, "open", guarded_open)
    with pytest.raises(pb.CASTamperError, match="changed before readback"):
        pb._atomic_publish(path, b"immutable\n")


@pytest.mark.parametrize(
    "hostility", ["namespace_symlink", "arrival_symlink", "extra", "entry_bound"]
)
def test_initial_miss_rendezvous_rejects_hostile_namespace_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hostility: str
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    namespace = tmp_path / "rendezvous-state"
    outside = tmp_path / "outside"
    outside.mkdir()
    if hostility == "namespace_symlink":
        namespace.symlink_to(outside, target_is_directory=True)
    else:
        namespace.mkdir()
        if hostility == "arrival_symlink":
            (namespace / "arrivals").symlink_to(outside, target_is_directory=True)
            (namespace / "ready").mkdir()
        else:
            (namespace / "arrivals").mkdir()
            (namespace / "ready").mkdir()
            if hostility == "extra":
                (namespace / "intruder").mkdir()
            else:
                for index in range(
                    pb._MAX_INITIAL_MISS_RENDEZVOUS_DIRECTORY_ENTRIES + 1
                ):
                    (namespace / "arrivals" / f"extra-{index}").write_bytes(b"")
    manifest_path, _ = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=namespace,
        timeout_seconds=0.05,
    )
    monkeypatch.setattr(pb, "_initial_miss_hostname", lambda: "host-a")
    with pytest.raises(pb.PrismaBuildError):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
            initial_miss_rendezvous=manifest_path,
        )
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("hostility", ["mutable", "hardlink", "symlink", "special"])
def test_initial_miss_rendezvous_manifest_must_be_immutable_real_single_link(
    tmp_path: Path, hostility: str
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    path, _ = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=tmp_path / "rendezvous-state",
    )
    if hostility == "mutable":
        path.chmod(0o644)
    elif hostility == "hardlink":
        os.link(path, tmp_path / "second-manifest-link")
    elif hostility == "symlink":
        target = tmp_path / "manifest-target"
        path.rename(target)
        path.symlink_to(target)
    else:
        path.unlink()
        os.mkfifo(path)
    with pytest.raises(pb.PrismaBuildError):
        pb._load_initial_miss_rendezvous_manifest(
            path, action=action, cas=pb.PrismaBuildCAS(tmp_path / "cas")
        )


def test_initial_miss_rendezvous_does_not_mutate_manifest_source_or_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    namespace = tmp_path / "rendezvous-state"
    manifest_path, manifest = _rendezvous_manifest(
        tmp_path / "rendezvous.json",
        action=action,
        namespace=namespace,
    )
    source = Path(pb.__file__)
    source_before = (source.read_bytes(), source.stat())
    manifest_before = (manifest_path.read_bytes(), manifest_path.stat())
    monkeypatch.setattr(
        pb, "_initial_miss_hostname", lambda: threading.current_thread().name
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            pb.run_local_action(
                action,
                cas_root=tmp_path / "cas",
                checkout_root=checkout,
                initial_miss_rendezvous=manifest_path,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, name=participant)
        for participant in manifest["participants"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert errors == []
    assert source.read_bytes() == source_before[0]
    assert (source.stat().st_dev, source.stat().st_ino, source.stat().st_mode) == (
        source_before[1].st_dev,
        source_before[1].st_ino,
        source_before[1].st_mode,
    )
    assert manifest_path.read_bytes() == manifest_before[0]
    assert (
        manifest_path.stat().st_dev,
        manifest_path.stat().st_ino,
        manifest_path.stat().st_mode,
    ) == (
        manifest_before[1].st_dev,
        manifest_before[1].st_ino,
        manifest_before[1].st_mode,
    )
    assert not list(namespace.rglob("__pycache__"))


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
        )
    (checkout / "result.bin").unlink()
    missing = _action(checkout, argv=[sys.executable, "-c", "pass"])
    with pytest.raises(pb.LocalActionError, match="without its declared result"):
        pb.run_local_action(
            missing,
            cas_root=tmp_path / "cas-missing",
            checkout_root=checkout,
        )


def test_sigkill_after_result_staging_is_reaped_and_retry_recomputes(
    tmp_path: Path,
):
    """A killed worker cannot permanently wedge its declared result path."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    action_path = tmp_path / "action.json"
    action_path.write_text(json.dumps(action), encoding="utf-8")
    cas_root = tmp_path / "cas"
    repository_root = Path(__file__).resolve().parents[1]
    crash_worker = """
import json
import os
from pathlib import Path
import signal
import sys
from prismaquant import prismabuild as pb

action = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
real_atomic_publish = pb._atomic_publish

def kill_after_receipt_staging(path, raw, *, prelink_verify=None):
    if "/actions/v3/" in str(path):
        def kill_at_prelink():
            if prelink_verify is not None:
                prelink_verify()
            os.kill(os.getpid(), signal.SIGKILL)
        return real_atomic_publish(path, raw, prelink_verify=kill_at_prelink)
    return real_atomic_publish(path, raw, prelink_verify=prelink_verify)

pb._atomic_publish = kill_after_receipt_staging
pb.run_local_action(
    action,
    cas_root=Path(sys.argv[2]),
    checkout_root=Path(sys.argv[3]),
)
"""
    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            crash_worker,
            str(action_path),
            str(cas_root),
            str(checkout),
        ],
        cwd=repository_root,
        check=False,
    )
    assert crashed.returncode == -signal.SIGKILL
    assert (checkout / "result.bin").read_bytes() == b"result"
    assert pb.PrismaBuildCAS(cas_root).lookup(action) is None
    claims = list((cas_root / "local-results" / "v1").glob("*/*.json"))
    assert len(claims) == 1
    staging = list(
        (cas_root / ".staging" / "local-results").glob("*/*.tmp")
    )
    assert len(staging) == 1

    recovered = pb.run_local_action(
        action,
        cas_root=cas_root,
        checkout_root=checkout,
    )
    assert recovered["status"] == "published"
    assert recovered["recovered_declared_result"] is True
    assert recovered["reaped_staging_files"] == 1
    assert not list(
        (cas_root / ".staging" / "local-results").glob("*/*.tmp")
    )
    assert Path(recovered["payload_path"]).read_bytes() == b"result"


def test_sigkill_during_action_keeps_output_locked_until_orphan_exits(
    tmp_path: Path,
):
    """A retry cannot overlap an action child that survived worker death."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    task = (
        "import os,pathlib,time; "
        "f=open('starts.log','ab',buffering=0); "
        "f.write((str(os.getpid())+'\\n').encode()); os.fsync(f.fileno()); "
        "time.sleep(1.25); pathlib.Path('result.bin').write_bytes(b'result')"
    )
    action = _action(checkout, argv=[sys.executable, "-c", task])
    action_path = tmp_path / "action.json"
    action_path.write_text(json.dumps(action), encoding="utf-8")
    cas_root = tmp_path / "cas"
    repository_root = Path(__file__).resolve().parents[1]
    worker = repository_root / "tools" / "prismabuild_worker.py"
    argv = [
        sys.executable,
        str(worker),
        "run-local",
        "--action",
        str(action_path),
        "--cas-root",
        str(cas_root),
        "--checkout-root",
        str(checkout),
    ]
    first = subprocess.Popen(argv)
    retry: subprocess.Popen[bytes] | None = None
    action_process_groups: set[int] = set()
    starts = checkout / "starts.log"
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if starts.exists():
                lines = starts.read_text(encoding="utf-8").splitlines()
                if len(lines) == 1:
                    action_process_groups.add(int(lines[0]))
                    break
            time.sleep(0.02)
        else:
            pytest.fail("first action process did not start")

        first.send_signal(signal.SIGKILL)
        assert first.wait(timeout=2.0) == -signal.SIGKILL
        retry = subprocess.Popen(argv)
        time.sleep(0.25)

        # The inherited flock must keep the retry outside the action body.
        assert retry.poll() is None
        assert starts.read_text(encoding="utf-8").splitlines() == lines

        assert retry.wait(timeout=8.0) == 0
        all_starts = starts.read_text(encoding="utf-8").splitlines()
        assert len(all_starts) == 2
        action_process_groups.update(int(value) for value in all_starts)
        receipt = pb.PrismaBuildCAS(cas_root).lookup(action)
        assert receipt is not None
        assert (checkout / "result.bin").read_bytes() == b"result"
    finally:
        for process in (first, retry):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=2.0)
        for process_group in action_process_groups:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_closing_worker_lock_copy_does_not_unlock_inheriting_action(
    tmp_path: Path,
):
    """No explicit unlock may defeat a surviving task's shared lock lease."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    output = checkout / "result.bin"
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    holder: subprocess.Popen[bytes] | None = None
    try:
        with pb._local_output_lock(cas, checkout, output) as descriptor:
            holder = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(1)"],
                pass_fds=(descriptor,),
            )

        lock_paths = list((cas.root / ".worker-locks").glob("*.lock"))
        assert len(lock_paths) == 1
        with lock_paths[0].open("r+b") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert holder.wait(timeout=3.0) == 0
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
    finally:
        if holder is not None and holder.poll() is None:
            holder.kill()
            holder.wait(timeout=2.0)


def test_sigint_worker_reaps_action_group_before_releasing_output_lock(
    tmp_path: Path,
):
    """Handled worker interruption cannot leave an unlocked action writer."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    task = (
        "import os,pathlib,time; "
        "first=pathlib.Path('first-attempt'); "
        "f=open('starts.log','ab',buffering=0); "
        "f.write((str(os.getpid())+'\\n').encode()); os.fsync(f.fileno()); "
        "is_retry=first.exists(); first.touch(); "
        "time.sleep(0 if is_retry else 10); "
        "pathlib.Path('result.bin').write_bytes(b'result')"
    )
    action = _action(checkout, argv=[sys.executable, "-c", task])
    action_path = tmp_path / "action.json"
    action_path.write_text(json.dumps(action), encoding="utf-8")
    cas_root = tmp_path / "cas"
    repository_root = Path(__file__).resolve().parents[1]
    worker = repository_root / "tools" / "prismabuild_worker.py"
    argv = [
        sys.executable,
        str(worker),
        "run-local",
        "--action",
        str(action_path),
        "--cas-root",
        str(cas_root),
        "--checkout-root",
        str(checkout),
    ]
    interrupted = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    retry: subprocess.Popen[bytes] | None = None
    starts = checkout / "starts.log"
    action_pid: int | None = None
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if starts.exists():
                lines = starts.read_text(encoding="utf-8").splitlines()
                if len(lines) == 1:
                    action_pid = int(lines[0])
                    break
            time.sleep(0.02)
        else:
            pytest.fail("interrupted action process did not start")

        interrupted.send_signal(signal.SIGINT)
        assert interrupted.wait(timeout=4.0) != 0
        assert action_pid is not None
        with pytest.raises(ProcessLookupError):
            os.kill(action_pid, 0)

        retry = subprocess.Popen(argv)
        assert retry.wait(timeout=5.0) == 0
        assert len(starts.read_text(encoding="utf-8").splitlines()) == 2
        assert pb.PrismaBuildCAS(cas_root).lookup(action) is not None
    finally:
        for process in (interrupted, retry):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=2.0)
        if action_pid is not None:
            try:
                os.killpg(action_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_local_result_repair_requires_exact_claim_and_refuses_symlink(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    cas_root = tmp_path / "cas"
    cas = pb.PrismaBuildCAS(cas_root)
    output = checkout / "result.bin"
    output.write_bytes(b"unclaimed")
    with pytest.raises(pb.LocalActionError, match="no matching immutable"):
        pb.repair_local_result(
            action, cas_root=cas_root, checkout_root=checkout
        )
    assert output.read_bytes() == b"unclaimed"

    output.unlink()
    claim = pb._ensure_local_result_claim(cas, action, checkout)
    target = checkout / "target.bin"
    target.write_bytes(b"must remain")
    output.symlink_to(target.name)
    with pytest.raises(pb.LocalActionError, match="escapes|symlink"):
        pb.repair_local_result(
            action, cas_root=cas_root, checkout_root=checkout
        )
    assert output.is_symlink()
    assert target.read_bytes() == b"must remain"

    output.unlink()
    output.write_bytes(b"claimed partial")
    repaired = pb.repair_local_result(
        action, cas_root=cas_root, checkout_root=checkout
    )
    assert repaired == {
        "status": "removed",
        "action_key": action["action_key"],
        "claim_sha256": claim["claim_sha256"],
        "result_path": str(output),
        "reaped_staging_files": 0,
    }
    assert not output.exists()

    output.write_bytes(b"claimed partial again")
    action_path = tmp_path / "repair-action.json"
    action_path.write_text(json.dumps(action), encoding="utf-8")
    assert pb.main(
        [
            "repair-local-result",
            "--action",
            str(action_path),
            "--cas-root",
            str(cas_root),
            "--checkout-root",
            str(checkout),
        ]
    ) == 0
    cli_repair = json.loads(capsys.readouterr().out)
    assert cli_repair["status"] == "removed"
    assert cli_repair["claim_sha256"] == claim["claim_sha256"]

    published = pb.run_local_action(
        action, cas_root=cas_root, checkout_root=checkout
    )
    assert published["status"] == "published"
    with pytest.raises(pb.LocalActionError, match="has a CAS receipt"):
        pb.repair_local_result(
            action, cas_root=cas_root, checkout_root=checkout
        )


def test_local_result_repair_rechecks_receipt_after_lock_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Repair cannot erase output certified while it waits for exclusion."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    cas_root = tmp_path / "cas"
    cas = pb.PrismaBuildCAS(cas_root)
    output = checkout / "result.bin"
    output.write_bytes(b"result")
    pb._ensure_local_result_claim(cas, action, checkout)
    attestation = pb.preflight_action(
        action, cas_root=cas_root, checkout_root=checkout
    )

    @contextmanager
    def publish_while_repair_waits(*_args: object, **_kwargs: object):
        receipt, won = cas.publish_result(
            action, output, attestation=attestation
        )
        assert won is True
        assert receipt["result"]["sha256"] == hashlib.sha256(b"result").hexdigest()
        yield

    monkeypatch.setattr(pb, "_local_output_lock", publish_while_repair_waits)
    with pytest.raises(pb.LocalActionError, match="has a CAS receipt"):
        pb.repair_local_result(
            action, cas_root=cas_root, checkout_root=checkout
        )
    assert output.read_bytes() == b"result"
    assert cas.lookup(action) is not None


def test_local_worker_refuses_closure_changed_by_action_before_publish(
    tmp_path: Path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    code = (
        "import pathlib; "
        "pathlib.Path('task_code.py').write_text('# changed by action\\n'); "
        "pathlib.Path('result.bin').write_bytes(b'untrusted')"
    )
    action = _action(checkout, argv=[sys.executable, "-c", code])
    cas = pb.PrismaBuildCAS(tmp_path / "cas")

    with pytest.raises(pb.ActionContractError, match="live code closure differs"):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
        )

    assert (checkout / "result.bin").read_bytes() == b"untrusted"
    assert cas.lookup(action) is None


def test_local_worker_rechecks_closure_after_result_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    original_copy = pb._copy_to_staging

    def copy_then_mutate(source: Path, staging_dir: Path):
        staged = original_copy(source, staging_dir)
        (checkout / "task_code.py").write_text(
            "# changed during CAS staging\n", encoding="utf-8"
        )
        return staged

    monkeypatch.setattr(pb, "_copy_to_staging", copy_then_mutate)
    with pytest.raises(pb.ActionContractError, match="live code closure differs"):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
        )

    assert cas.lookup(action) is None


def test_local_worker_rechecks_closure_at_receipt_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    original_publish = pb._atomic_publish

    def publish_after_receipt_staging(
        path: Path,
        raw: bytes,
        *,
        prelink_verify=None,
    ):
        if prelink_verify is None:
            return original_publish(path, raw)

        def mutate_then_verify():
            (checkout / "task_code.py").write_text(
                "# changed while receipt was staged\n", encoding="utf-8"
            )
            prelink_verify()

        return original_publish(
            path,
            raw,
            prelink_verify=mutate_then_verify,
        )

    monkeypatch.setattr(pb, "_atomic_publish", publish_after_receipt_staging)
    with pytest.raises(pb.ActionContractError, match="live code closure differs"):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
        )

    assert cas.lookup(action) is None


def test_local_worker_refuses_launcher_changed_by_action_before_publish(
    tmp_path: Path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    launcher = tmp_path / "worker-launcher.py"
    launcher.write_text("# original launcher\n", encoding="utf-8")
    launcher_identity = pb._identify_runtime_source(
        launcher, where="test worker launcher"
    )
    code = (
        "import pathlib; "
        f"pathlib.Path({str(launcher)!r}).write_text('# changed launcher\\n'); "
        "pathlib.Path('result.bin').write_bytes(b'untrusted')"
    )
    action = _action(checkout, argv=[sys.executable, "-c", code])
    cas = pb.PrismaBuildCAS(tmp_path / "cas")

    with pytest.raises(pb.LocalActionError, match="worker launcher changed"):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
            worker_launcher_identity=launcher_identity,
        )

    assert (checkout / "result.bin").read_bytes() == b"untrusted"
    assert cas.lookup(action) is None


def test_local_worker_rechecks_launcher_after_result_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    launcher = tmp_path / "worker-launcher.py"
    launcher.write_text("# original launcher\n", encoding="utf-8")
    launcher_identity = pb._identify_runtime_source(
        launcher, where="test worker launcher"
    )
    action = _action(checkout)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    original_copy = pb._copy_to_staging

    def copy_then_mutate(source: Path, staging_dir: Path):
        staged = original_copy(source, staging_dir)
        launcher.write_text("# changed during CAS staging\n", encoding="utf-8")
        return staged

    monkeypatch.setattr(pb, "_copy_to_staging", copy_then_mutate)
    with pytest.raises(pb.LocalActionError, match="worker launcher changed"):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
            worker_launcher_identity=launcher_identity,
        )

    assert cas.lookup(action) is None


def test_local_worker_rechecks_core_after_result_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    fake_core = tmp_path / "prismabuild.py"
    fake_core.write_text("# original worker core\n", encoding="utf-8")
    monkeypatch.setattr(
        pb,
        "_LOADED_WORKER_CORE_IDENTITY",
        pb._identify_runtime_source(fake_core, where="test worker core"),
    )
    action = _action(checkout)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    original_copy = pb._copy_to_staging

    def copy_then_mutate(source: Path, staging_dir: Path):
        staged = original_copy(source, staging_dir)
        fake_core.write_text("# changed during CAS staging\n", encoding="utf-8")
        return staged

    monkeypatch.setattr(pb, "_copy_to_staging", copy_then_mutate)
    with pytest.raises(pb.LocalActionError, match="worker core changed"):
        pb.run_local_action(
            action,
            cas_root=tmp_path / "cas",
            checkout_root=checkout,
        )

    assert cas.lookup(action) is None


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
    attestation = _attestation(checkout, action, tmp_path / "cas")
    receipt, won = cas.publish_result(
        action,
        first,
        attestation=attestation,
    )
    assert won
    with pytest.raises(pb.CASConflictError, match="deterministic recomputation"):
        cas.publish_result(
            action,
            second,
            attestation=attestation,
        )
    assert cas.lookup(action) == receipt


def test_fresh_result_publish_reuses_verified_winning_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    output = tmp_path / "output"
    output.write_bytes(b"canonical")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    attestation = _attestation(checkout, action, tmp_path / "cas")
    original_identity = pb._file_identity_nofollow
    consumed: list[Path] = []

    def track_identity(path: Path, **kwargs: object):
        consumed.append(path)
        return original_identity(path, **kwargs)

    monkeypatch.setattr(pb, "_file_identity_nofollow", track_identity)
    receipt, won = cas.publish_result(
        action, output, attestation=attestation
    )
    assert won is True
    assert consumed == []
    assert cas._verified_receipt_result_path(receipt).is_file()
    assert consumed == []

    assert cas.lookup(action) == receipt
    assert consumed == [cas._blob_path(str(receipt["result"]["sha256"]))]


def test_stochastic_receipt_loser_hashes_different_canonical_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout, determinism="stochastic")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"first canonical result")
    second.write_bytes(b"different losing result")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    attestation = _attestation(checkout, action, tmp_path / "cas")
    canonical, won = cas.publish_result(
        action, first, attestation=attestation
    )
    assert won is True
    original_identity = pb._file_identity_nofollow
    consumed: list[Path] = []

    def track_identity(path: Path, **kwargs: object):
        consumed.append(path)
        return original_identity(path, **kwargs)

    monkeypatch.setattr(pb, "_file_identity_nofollow", track_identity)
    observed, won = cas.publish_result(
        action, second, attestation=attestation
    )
    assert won is False
    assert observed == canonical
    assert consumed == [
        cas._blob_path(str(canonical["result"]["sha256"]))
    ]


def test_cas_publish_refuses_worker_core_changed_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    fake_core = tmp_path / "prismabuild.py"
    fake_core.write_text("# original worker core\n", encoding="utf-8")
    monkeypatch.setattr(
        pb,
        "_LOADED_WORKER_CORE_IDENTITY",
        pb._identify_runtime_source(fake_core, where="test worker core"),
    )
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    attestation = pb.preflight_action(
        action, cas_root=tmp_path / "cas", checkout_root=checkout
    )
    result = tmp_path / "result"
    result.write_bytes(b"candidate")
    fake_core.write_text("# changed worker core\n", encoding="utf-8")

    with pytest.raises(pb.LocalActionError, match="worker core changed"):
        cas.publish_result(action, result, attestation=attestation)

    assert cas.lookup(action) is None


def test_cas_refuses_attestation_from_a_different_loaded_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    first_core = tmp_path / "first-prismabuild.py"
    second_core = tmp_path / "second-prismabuild.py"
    first_core.write_text("# first loaded core\n", encoding="utf-8")
    second_core.write_text("# second loaded core\n", encoding="utf-8")
    monkeypatch.setattr(
        pb,
        "_LOADED_WORKER_CORE_IDENTITY",
        pb._identify_runtime_source(first_core, where="first test worker core"),
    )
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    attestation = pb.preflight_action(
        action, cas_root=tmp_path / "cas", checkout_root=checkout
    )
    monkeypatch.setattr(
        pb,
        "_LOADED_WORKER_CORE_IDENTITY",
        pb._identify_runtime_source(second_core, where="second test worker core"),
    )
    result = tmp_path / "result"
    result.write_bytes(b"candidate")

    with pytest.raises(
        pb.LocalActionError, match="core differs from module import identity"
    ):
        cas.publish_result(action, result, attestation=attestation)

    assert cas.lookup(action) is None


def test_cas_runs_runtime_recheck_after_action_precommit_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    fake_core = tmp_path / "prismabuild.py"
    fake_core.write_text("# original worker core\n", encoding="utf-8")
    monkeypatch.setattr(
        pb,
        "_LOADED_WORKER_CORE_IDENTITY",
        pb._identify_runtime_source(fake_core, where="test worker core"),
    )
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    attestation = pb.preflight_action(
        action, cas_root=tmp_path / "cas", checkout_root=checkout
    )
    result = tmp_path / "result"
    result.write_bytes(b"candidate")

    def mutate_core_during_action_check() -> None:
        fake_core.write_text(
            "# changed by action-specific callback\n", encoding="utf-8"
        )

    with pytest.raises(pb.LocalActionError, match="core changed after module import"):
        cas.publish_result(
            action,
            result,
            attestation=attestation,
            precommit_verify=mutate_core_during_action_check,
        )

    assert cas.lookup(action) is None


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
    attestation = _attestation(checkout, action, tmp_path / "cas")
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
                    attestation=attestation,
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
    attestation = _attestation(checkout, action, tmp_path / "cas")
    receipt, _ = cas.publish_result(
        action,
        output,
        attestation=attestation,
    )
    payload = cas.result_path(receipt, action)
    payload.chmod(0o644)
    payload.write_bytes(b"tampered")
    with pytest.raises(pb.CASTamperError, match="payload content differs"):
        cas.lookup(action)

    # Restore the content and then alter the immutable receipt's bytes.
    payload.write_bytes(b"trusted")
    receipt_path = cas._receipt_path(str(action["action_key"]))
    receipt_path.chmod(0o644)
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(pb.CASTamperError, match="writable|fields differ"):
        cas.lookup(action)

    receipt_path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(pb.CASTamperError, match="writable|strict UTF-8 JSON"):
        cas.lookup(action)


def test_cas_lookup_rejects_receipt_parent_swapped_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    output = tmp_path / "output"
    output.write_bytes(b"trusted")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    receipt, _ = cas.publish_result(
        action,
        output,
        attestation=_attestation(checkout, action, tmp_path / "cas"),
    )
    actions = cas.root / "actions"
    parked = tmp_path / "parked-actions"
    outside = tmp_path / "outside-actions"
    hostile = (
        outside
        / "v3"
        / str(action["action_key"])[:2]
        / f"{action['action_key']}.json"
    )
    hostile.parent.mkdir(parents=True)
    hostile.write_text("{}\n", encoding="utf-8")
    hostile.chmod(0o444)
    original_open = pb.os.open
    swapped = False

    def open_then_swap(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]
        if path == "actions" and dir_fd is not None and not swapped:
            swapped = True
            actions.rename(parked)
            actions.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(pb.os, "open", open_then_swap)
    with pytest.raises(pb.CASTamperError, match="not a real directory|changed"):
        cas.lookup(action)
    assert hostile.read_bytes() == b"{}\n"
    assert receipt["action_key"] == action["action_key"]


def test_cas_publish_result_parent_swap_never_writes_through_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    output = tmp_path / "output"
    output.write_bytes(b"trusted")
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    attestation = _attestation(checkout, action, tmp_path / "cas")
    actions = cas.root / "actions"
    parked = tmp_path / "parked-publish-actions"
    outside = tmp_path / "outside-publish-actions"
    outside.mkdir()
    original_open = pb.os.open
    swapped = False

    def open_then_swap(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]
        if path == "actions" and dir_fd is not None and not swapped:
            swapped = True
            actions.rename(parked)
            actions.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(pb.os, "open", open_then_swap)
    with pytest.raises(pb.CASTamperError, match="not a real directory|changed"):
        cas.publish_result(action, output, attestation=attestation)
    assert list(outside.rglob("*")) == []
    assert not list((cas.root / ".staging").glob("*.tmp"))


def test_cache_structural_error_is_not_a_clean_miss(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    shard = cas._receipt_path(str(action["action_key"])).parent
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"not-a-directory")
    with pytest.raises(pb.CASTamperError):
        cas.lookup(action)


def test_v3_receipt_namespace_preserves_and_ignores_legacy_v2(
    tmp_path: Path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    action = _action(checkout)
    cas = pb.PrismaBuildCAS(tmp_path / "cas")
    action_key = str(action["action_key"])
    legacy_path = cas._legacy_v2_receipt_path(action_key)
    legacy_path.parent.mkdir(parents=True)
    current_attestation = _attestation(
        checkout, action, tmp_path / "cas"
    )
    legacy_producer_body = {
        key: value
        for key, value in current_attestation.items()
        if key not in {"runtime", "attestation_sha256"}
    }
    legacy_producer_body["schema"] = (
        "prismaquant.prismabuild.worker_attestation.v1"
    )
    legacy_producer = {
        **legacy_producer_body,
        "attestation_sha256": pb.canonical_sha256(legacy_producer_body),
    }
    legacy_payload = b"legacy v2 result"
    legacy_digest = hashlib.sha256(legacy_payload).hexdigest()
    legacy_blob = cas._blob_path(legacy_digest)
    legacy_blob.parent.mkdir(parents=True)
    legacy_blob.write_bytes(legacy_payload)
    legacy_body = {
        "schema": "prismaquant.prismabuild.cas_receipt.v2",
        "action_key": action_key,
        "action_manifest_sha256": pb.canonical_sha256(action),
        "result": {"sha256": legacy_digest, "bytes": len(legacy_payload)},
        "producer": legacy_producer,
    }
    legacy_receipt = {
        **legacy_body,
        "receipt_sha256": pb.canonical_sha256(legacy_body),
    }
    legacy_bytes = (
        json.dumps(
            legacy_receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    legacy_path.write_bytes(legacy_bytes)
    legacy_path.chmod(0o444)

    # A legacy entry is not v3 tamper and cannot satisfy a v3 lookup.
    assert cas.lookup(action) is None
    output = tmp_path / "output"
    output.write_bytes(b"v3 result")
    receipt, won = cas.publish_result(
        action,
        output,
        attestation=current_attestation,
    )

    assert won
    assert receipt["schema"] == pb.CAS_RECEIPT_SCHEMA_V3
    assert cas.lookup(action) == receipt
    assert cas._receipt_path(action_key) != legacy_path
    assert legacy_path.read_bytes() == legacy_bytes
    assert legacy_blob.read_bytes() == legacy_payload


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


def test_cli_ingests_and_verifies_input_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    source = tmp_path / "model.bin"
    payload = b"model bytes"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    cas_root = tmp_path / "cas"

    command = [
        "ingest-input",
        "--source",
        str(source),
        "--cas-root",
        str(cas_root),
        "--input-id",
        "model/weights",
        "--expected-sha256",
        digest,
        "--expected-bytes",
        str(len(payload)),
    ]
    assert pb.main(command) == 0
    published = json.loads(capsys.readouterr().out)
    assert published["status"] == "published"
    assert published["input"] == {
        "id": "model/weights",
        "sha256": digest,
        "bytes": len(payload),
    }

    assert pb.main(command) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "already_present"

    contract_path = tmp_path / "input.json"
    contract_path.write_text(json.dumps(published["input"]), encoding="utf-8")
    assert pb.main(
        [
            "verify-input",
            "--input-contract",
            str(contract_path),
            "--cas-root",
            str(cas_root),
        ]
    ) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["input"] == published["input"]
    assert Path(verified["payload_path"]).read_bytes() == payload


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
