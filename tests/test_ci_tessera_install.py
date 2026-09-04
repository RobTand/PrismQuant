"""CI must install the exact Tessera checkout PrismaQuant was reviewed against."""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PIN_SOURCE = ROOT / "prismaquant" / "tessera_runtime_contract.py"
PIN_RESOLVER = ROOT / "tools" / "resolve_tessera_dev_pin.py"


def _dev_pin_literal() -> str:
    tree = ast.parse(PIN_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name)
            and target.id == "TESSERA_DEV_PIN_COMMIT"
            for target in targets
        ):
            continue
        value = ast.literal_eval(node.value)
        assert isinstance(value, str)
        return value
    raise AssertionError("TESSERA_DEV_PIN_COMMIT is not a literal assignment")


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)",
        workflow,
    )
    assert match is not None, f"CI workflow has no {name!r} job"
    return match.group(0)


@pytest.mark.parametrize("job_name", ["tests", "imports"])
@pytest.mark.parametrize("valid_pin", [False, True])
def test_ci_pin_step_propagates_resolver_status(tmp_path, job_name, valid_pin):
    job = _job(WORKFLOW.read_text(encoding="utf-8"), job_name)
    step = re.search(
        r"(?ms)^      - name: Resolve Tessera development pin\n"
        r"(.*?)(?=^      -|\Z)",
        job,
    )
    assert step is not None
    run = re.search(r"(?m)^        run: (.*)$", step.group(1))
    assert run is not None
    command = run.group(1)
    if command == "|":
        command = textwrap.dedent(step.group(1)[run.end():])

    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / PIN_RESOLVER.name).write_text(
        PIN_RESOLVER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "prismaquant").mkdir()
    pin = _dev_pin_literal() if valid_pin else "not-a-commit"
    (tmp_path / "prismaquant" / PIN_SOURCE.name).write_text(
        f"TESSERA_DEV_PIN_COMMIT = {pin!r}\n", encoding="utf-8"
    )
    output = tmp_path / "github-output"
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", command],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if valid_pin:
        assert completed.returncode == 0, completed.stderr
        assert output.read_text(encoding="utf-8") == f"commit={pin}\n"
    else:
        assert completed.returncode != 0, "CI masked a failed pin resolver"
        assert "must be a full lowercase Git SHA" in completed.stderr
        assert not output.exists() or not output.read_text(encoding="utf-8")


def test_stdlib_resolver_reads_the_authoritative_pin_without_package_imports():
    completed = subprocess.run(
        [sys.executable, str(PIN_RESOLVER)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == _dev_pin_literal()
    resolver = PIN_RESOLVER.read_text(encoding="utf-8")
    assert "import prismaquant" not in resolver
    assert _dev_pin_literal() not in resolver


def test_ci_installs_the_authoritative_tessera_pin_without_private_credentials():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    # A copied SHA becomes a second pin and can drift from the admission rule.
    # The workflow must run the stdlib-only source resolver from its checkout.
    assert _dev_pin_literal() not in workflow

    for name in ("tests", "imports"):
        job = _job(workflow, name)
        resolve = "python tools/resolve_tessera_dev_pin.py"
        assert resolve in job, name
        assert "id: tessera-pin" in job, name
        assert "TESSERA_REPO_TOKEN" not in job, name
        assert "repository: RobTand/tessera" in job, name
        assert "ref: ${{ steps.tessera-pin.outputs.commit }}" in job, name
        assert "persist-credentials: false" in job, name
        assert "python -m pip install --no-deps .ci/tessera" in job, name

        # Resolve the pin before checkout; install those bytes before
        # importing/installing PrismaQuant itself. Ordinary checkout access
        # becomes sufficient when Tessera is published, including fork PRs.
        assert job.index(resolve) < job.index(
            "repository: RobTand/tessera"
        )
        assert job.index("repository: RobTand/tessera") < job.index(
            "python -m pip install --no-deps .ci/tessera"
        )
        assert job.index("python -m pip install --no-deps .ci/tessera") < job.index(
            "name: Install prismaquant"
        )
