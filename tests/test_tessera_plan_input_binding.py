"""Execute the real Tessera driver arm without encoding or serving work."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from prismaquant import pipeline


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "example/runtime@sha256:" + "a" * 64


def _run(tmp_path, *, changed=False, manifest="bound", mode="compiled"):
    work = tmp_path / "work with spaces"
    for sub in ("artifacts", "logs", "exported"):
        (work / sub).mkdir(parents=True)
    assignment = work / "artifacts/layer_config.json"
    assignment.write_text('{"layer": "BF16"}')
    plan = work / "artifacts/tessera_plan.json"
    settings = {
        "MODEL_PATH": str(tmp_path / "model"), "TESSERA_PLAN_COVER": "as-allocated",
        "TESSERA_PLATFORM": "sm_121", "TESSERA_RUNTIME_IMAGE": IMAGE,
        "TESSERA_EXECUTION_MODE": mode, "TESSERA_RESIDENCY": "resident",
    }
    document = pipeline.stage_settings_document(settings)
    stage_path = work / "artifacts/stage_settings.json"
    stage_path.write_text(json.dumps(document))
    if manifest == "bound":
        code, messages = pipeline.check_stage_settings(
            plan, "tessera-plan", document,
            overrides={"ASSIGNMENT_DIGEST": hashlib.sha256(assignment.read_bytes()).hexdigest()},
        )
        assert code == 0, messages
    elif manifest == "other-stage":
        Path(str(plan) + ".settings.json").write_text(json.dumps({
            "schema": pipeline.STAGE_MANIFEST_SCHEMA, "stages": {"other": {}},
        }))
    plan.write_text("old translated plan")
    (work / "exported/shipcard.json").write_text("{}")
    if changed:
        assignment.write_text('{"layer": "TESSERA_E4M3_K1_R1024"}')
    script = (ROOT / "prismaquant/run-pipeline.sh").read_text()
    helper = script[script.index("require_stage_settings() {"):].split("\n}\n", 1)[0] + "\n}\n"
    start = script.index('  # Tessera lane: one blob per vLLM module')
    start = script.rfind('if [[ "$EXPORT_CONTAINER" == "tessera" ]]; then', 0, start)
    end = script.index('\nif [[ "$EXPORT_CONTAINER" == "gguf" ]]; then', start)
    block = script[start:end]
    preamble = r'''
set -euo pipefail
TESSERA_SCOPE_ARGS=()
python3() {
  if [[ "$1" == "-m" && "$2" == "prismaquant.pipeline" ]]; then
    "$PYTEST_PYTHON" "$@"
  elif [[ "$1" == "-m" && "$2" == "prismaquant.tessera_export_lane" ]]; then
    return 0
  elif [[ "$1" == */plan_from_layer_config.py ]]; then
    echo "TEST_TRANSLATOR_REACHED" >&2
    return 81
  elif [[ "$1" == */export_tessera_serving.py ]]; then
    echo "TEST_EXPORT_REACHED"
    return 0
  else
    echo "unexpected worker: $*" >&2
    return 99
  fi
}
'''
    env = dict(os.environ, WORK_DIR=str(work), EXPORT_CONTAINER="tessera",
               TESSERA_PLAN_COVER="as-allocated", MODEL_PATH=settings["MODEL_PATH"],
               TESSERA_RUNTIME_IMAGE=IMAGE, TESSERA_EXECUTION_MODE=mode,
               TESSERA_PLATFORM="sm_121", TESSERA_RESIDENCY="resident",
               TESSERA_SERVE_MODE="resident", TESSERA_REPO=str(tmp_path / "producer tree"),
               TARGET_PROFILE_RESOLVED="vllm_tessera", EXPORT_DEVICE="cuda",
               PIPELINE_SCRIPT_DIR=str(ROOT / "prismaquant"),
               STAGE_SETTINGS_PATH=str(stage_path), PYTEST_PYTHON=sys.executable)
    return subprocess.run(["bash", "-c", preamble + helper + block], env=env,
                          cwd=ROOT, capture_output=True, text=True, timeout=30)


def test_actual_driver_refuses_old_plan_after_allocation_bytes_change(tmp_path):
    result = _run(tmp_path, changed=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "ASSIGNMENT_DIGEST" in result.stdout + result.stderr
    assert "TEST_EXPORT_REACHED" not in result.stdout


@pytest.mark.parametrize("manifest", ["missing", "other-stage"])
def test_actual_driver_refuses_plan_without_independent_allocation_binding(tmp_path, manifest):
    result = _run(tmp_path, manifest=manifest)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "allocation" in (result.stdout + result.stderr).lower()
    assert "TEST_EXPORT_REACHED" not in result.stdout


def test_actual_driver_reuses_plan_for_identical_allocation(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TEST_EXPORT_REACHED" in result.stdout
    assert "TEST_TRANSLATOR_REACHED" not in result.stderr


@pytest.mark.parametrize("mode,eager", [("eager", "1"), ("compiled", "0")])
def test_printed_serve_recipe_retains_exact_runtime_scope(tmp_path, mode, eager):
    result = _run(tmp_path, mode=mode)
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(line.split("Serve:", 1)[1].strip() for line in result.stdout.splitlines()
                   if "Serve:" in line)
    tokens = shlex.split(command)
    assert f"IMAGE={IMAGE}" in tokens
    assert f"TESSERA_LANE_EAGER={eager}" in tokens
    assert f"TS={tmp_path / 'producer tree'}" in tokens
    assert str(tmp_path / "work with spaces/exported") in tokens
