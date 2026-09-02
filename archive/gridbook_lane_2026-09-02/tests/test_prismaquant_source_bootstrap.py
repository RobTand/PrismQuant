"""Verified source bootstrap without a server-side PYTHONPATH."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prismaquant.gridbook_runtime_pin import (
    GRIDBOOK_RUNTIME_RELEASE_VERSION,
)


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "prismaquant_source_bootstrap.py"
SNAPSHOT_TOOL = REPO / "tools" / "prismaquant_runtime_snapshot.py"


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PQ_RUNTIME_PRISMAQUANT_ROOT"] = str(REPO)
    return environment


def _run(*args: str, environment: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, "-P", str(TOOL), *args],
        cwd="/",
        env=environment or _environment(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_strict_bootstrap_resolves_the_transport_attested_source_root():
    result = _run("check", "--source-root", str(REPO))
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == REPO.resolve()


def test_strict_bootstrap_rejects_even_an_empty_pythonpath():
    environment = _environment()
    environment["PYTHONPATH"] = ""
    result = _run(
        "check", "--source-root", str(REPO), environment=environment
    )
    assert result.returncode == 2
    assert "PYTHONPATH must be absent" in result.stderr


def test_strict_bootstrap_rejects_a_different_transport_root(tmp_path):
    result = _run("check", "--source-root", str(tmp_path))
    assert result.returncode == 2
    assert "differs from the bootstrap snapshot" in result.stderr


def test_strict_bootstrap_rejects_env_strings_without_active_interpreter_flags():
    environment = _environment()
    environment.pop("PYTHONSAFEPATH")
    command = (
        "import os,runpy,sys; "
        "os.environ['PYTHONSAFEPATH']='1'; "
        f"sys.argv=['{TOOL}','check','--source-root','{REPO}']; "
        f"runpy.run_path('{TOOL}',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-s", "-c", command],
        cwd="/",
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "active Python safe-path mode" in result.stderr


def test_strict_bootstrap_rejects_env_no_bytecode_without_active_flag():
    environment = _environment()
    environment.pop("PYTHONDONTWRITEBYTECODE")
    command = (
        "import os,runpy,sys; "
        "os.environ['PYTHONDONTWRITEBYTECODE']='1'; "
        f"sys.argv=['{TOOL}','check','--source-root','{REPO}']; "
        f"runpy.run_path('{TOOL}',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-P", "-c", command],
        cwd="/",
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "active no-bytecode mode" in result.stderr


def test_strict_bootstrap_rejects_env_no_user_site_without_active_flag():
    environment = _environment()
    environment.pop("PYTHONNOUSERSITE")
    command = (
        "import os,runpy,sys; "
        "os.environ['PYTHONNOUSERSITE']='1'; "
        f"sys.argv=['{TOOL}','check','--source-root','{REPO}']; "
        f"runpy.run_path('{TOOL}',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-P", "-B", "-c", command],
        cwd="/",
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "disabled user-site mode" in result.stderr


def test_run_module_executes_only_after_the_source_proof():
    result = _run(
        "run-module", "--source-root", str(REPO),
        "prismaquant.gridbook_runtime_pin",
    )
    assert result.returncode == 0, result.stderr


def test_strict_bootstrap_namespace_loads_packaged_profile_data():
    program = f"""
import importlib.util
from pathlib import Path
import sys
import types

tool = Path({str(TOOL)!r})
spec = importlib.util.spec_from_file_location("pq_source_bootstrap", tool)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = module.activate_prismaquant_source({str(REPO)!r})
module._install_exact_package_namespace(root)

# This test exercises package-data lookup under the bootstrap's synthetic
# namespace, not PyTorch.  Keep it independent of optional user-site packages
# by supplying the two tiny interfaces needed while these modules are loaded.
format_registry = types.ModuleType("prismaquant.format_registry")
format_registry.canonical_format_name = lambda name: str(name)
format_registry.aliases_for = lambda name: (str(name),)
sys.modules[format_registry.__name__] = format_registry
torch = types.ModuleType("torch")
torch_nn = types.ModuleType("torch.nn")
torch_nn.Module = type("Module", (), {{}})
torch_nn.Linear = type("Linear", (torch_nn.Module,), {{}})
torch_nn.Embedding = type("Embedding", (torch_nn.Module,), {{}})
torch.nn = torch_nn
sys.modules[torch.__name__] = torch
sys.modules[torch_nn.__name__] = torch_nn

from prismaquant.serving_profiles import (
    gridbook_runtime_version,
    load_serving_profile,
)
from prismaquant.model_profiles.structure import load_structure_spec
assert load_serving_profile("nvfp4_cb").id == "nvfp4_cb"
assert gridbook_runtime_version() == {GRIDBOOK_RUNTIME_RELEASE_VERSION!r}
assert load_structure_spec("deepseek_v4").id == "deepseek_v4"
"""
    result = subprocess.run(
        [sys.executable, "-P", "-c", program],
        cwd="/",
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_run_tool_executes_real_write_from_snapshot_and_preserves_closure(
    tmp_path,
):
    source = tmp_path / "source"
    required = (
        "prismaquant/__init__.py",
        "prismaquant/shipcard.py",
        "prismaquant/gridbook_environment.py",
        "prismaquant/gridbook_runtime_pin.py",
        "prismaquant/gridbook_serving_runtime_pin.py",
        "prismaquant/gridbook_runtime/gridbook_runtime_pin.json",
        "prismaquant/gridbook_runtime/gridbook_serving_runtime_pin.json",
        "tools/container_runtime_identity.py",
        "tools/prismaquant_runtime_snapshot.py",
        "tools/prismaquant_source_bootstrap.py",
        "tools/serve_fingerprint.py",
    )
    for relative in required:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "PQ test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "pq@test.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-qm", "fixture"], check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD^{commit}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    materialized = subprocess.run(
        [
            sys.executable, str(SNAPSHOT_TOOL), "materialize",
            "--source-root", str(source), "--cache-root", str(tmp_path / "cache"),
            "--commit", commit,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    identity = json.loads(materialized.stdout)
    snapshot = Path(identity["snapshot"])
    package_root = snapshot / "prismaquant"
    assert not list(package_root.rglob("__pycache__"))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update({
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PQ_RUNTIME_PRISMAQUANT_ROOT": str(snapshot),
    })
    environment.pop("PQ_GRIDBOOK_RUNTIME_COMMIT", None)
    environment.pop("PQ_GRIDBOOK_RUNTIME_VERSION", None)
    environment.pop("PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256", None)
    output = tmp_path / "serve-manifest.json"
    fake_vllm = tmp_path / "vllm-fixture"
    fake_vllm.write_text("#!/usr/bin/env bash\nsleep 30\n", encoding="utf-8")
    fake_vllm.chmod(0o755)
    server = subprocess.Popen(
        [str(fake_vllm), "serve", "/model"],
        cwd="/",
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Real server-side write path from neutral cwd. The fake process only
        # carries a vLLM-shaped argv and sleeps; it performs no model, network,
        # or GPU work. Artifact-free mode still forces the exact shipcard
        # import-origin proof before process inspection.
        result = subprocess.run(
            [
                sys.executable, "-P",
                str(snapshot / "tools" / "prismaquant_source_bootstrap.py"),
                "run-tool", "--source-root", str(snapshot),
                "serve-fingerprint", "write", "--out", str(output),
            ],
            cwd="/",
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        server.terminate()
        server.wait(timeout=5)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "prismaquant.serve_manifest/1"
    assert payload["source"] == "server"
    assert not list(package_root.rglob("__pycache__"))
    verified = subprocess.run(
        [
            sys.executable, str(snapshot / "tools" / "prismaquant_runtime_snapshot.py"),
            "verify", "--snapshot", str(snapshot),
            "--expected-commit", identity["commit"],
            "--expected-tree", identity["tree"],
            "--expected-closure-sha256", identity["closure_sha256"],
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert verified.returncode == 0, verified.stderr


def test_all_gold_producers_activate_the_shared_source_bootstrap():
    for name in (
        "build_streamed_full_kl_teacher.py",
        "measure_vllm_full_kl.py",
        "measure_vllm_wikitext_ppl.py",
    ):
        source = (REPO / "tools" / name).read_text(encoding="utf-8")
        assert "from prismaquant_source_bootstrap import" in source, name
        assert "activate_prismaquant_source()" in source, name
