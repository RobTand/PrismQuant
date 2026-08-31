from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
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
        "schema": pb.ACTION_SCHEMA_V1,
        "task": {
            "definition_id": "tests/build-result",
            "definition_version": "v1",
            "task_class": task_class,
            "determinism": determinism,
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


def test_input_ingestion_rehashes_canonical_name_after_winning_link(
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
    with pytest.raises(pb.CASTamperError, match="payload content differs"):
        cas.ingest_input(source, input_id="dataset")


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
    with pytest.raises(pb.CASTamperError, match="fields differ"):
        cas.lookup(action)

    receipt_path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(pb.CASTamperError, match="strict UTF-8 JSON"):
        cas.lookup(action)


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
