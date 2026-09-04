"""CI must install the exact Tessera checkout PrismaQuant was reviewed against."""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


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


def test_ci_installs_the_authoritative_private_tessera_pin_in_every_job():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    # A copied SHA becomes a second pin and can drift from the admission rule.
    # The workflow must run the stdlib-only source resolver from its checkout.
    assert _dev_pin_literal() not in workflow

    for name in ("tests", "imports"):
        job = _job(workflow, name)
        resolve = "python tools/resolve_tessera_dev_pin.py"
        assert resolve in job, name
        assert "id: tessera-pin" in job, name
        assert "name: Require Tessera repository token" in job, name
        assert "TESSERA_REPO_TOKEN is required" in job, name
        assert 'if [ -z "${TESSERA_REPO_TOKEN:-}" ]' in job, name
        assert "repository: RobTand/tessera" in job, name
        assert "ref: ${{ steps.tessera-pin.outputs.commit }}" in job, name
        assert "token: ${{ secrets.TESSERA_REPO_TOKEN }}" in job, name
        assert "persist-credentials: false" in job, name
        assert "python -m pip install --no-deps .ci/tessera" in job, name

        # Resolve and validate credentials before touching the private remote;
        # install those bytes before importing/installing PrismaQuant itself.
        assert job.index(resolve) < job.index("Require Tessera repository token")
        assert job.index("Require Tessera repository token") < job.index(
            "repository: RobTand/tessera"
        )
        assert job.index("repository: RobTand/tessera") < job.index(
            "python -m pip install --no-deps .ci/tessera"
        )
        assert job.index("python -m pip install --no-deps .ci/tessera") < job.index(
            "name: Install prismaquant"
        )
