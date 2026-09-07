"""Exercise real sampling and preserve failures of the observed child."""
import json
import os
import sys

import pytest

from experiments.profile_command import run_profile


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.skipif("PQ_TEST_PY_SPY" not in os.environ,
                    reason="PB scoped py-spy integration tooling is not configured")
def test_sampler_preserves_observed_child_result_and_stack_bytes(tmp_path, exit_code):
    profiler = os.environ["PQ_TEST_PY_SPY"]
    target = tmp_path / "target.py"
    target.write_text("import time, os, json\nfrom pathlib import Path\n"
                      "session=json.loads(Path(os.environ['PRISMAQUANT_SAMPLER_SESSION']).read_text())\n"
                      "assert session['wrapper_pid'] == os.getppid()\n"
                      "end=time.monotonic()+1.0\n"
                      "while time.monotonic()<end:\n    sum(range(1000))\n"
                      f"raise SystemExit({exit_code})\n")
    output = tmp_path / "capture"
    command = [sys.executable, str(target)]
    assert run_profile(output, command, profiler=profiler, rate=50) == exit_code
    result = json.loads((output / "profile-result.json").read_text())
    assert result["passed"] is (exit_code == 0)
    assert result["child"]["returncode"] == exit_code
    assert result["child"]["command"] == command
    assert result["artifacts"]["stacks.raw"]["bytes"] > 0
    assert "target.py" in (output / "stacks.raw").read_text()


def test_profiler_success_without_child_receipt_is_not_success(tmp_path):
    output = tmp_path / "capture"
    # /bin/true exits zero without running or recording its trailing command.
    assert run_profile(output, [sys.executable, "-c", "raise SystemExit(7)"],
                       profiler="/bin/true", rate=50) == 1
    result = json.loads((output / "profile-result.json").read_text())
    assert result["profiler_returncode"] == 0
    assert result["passed"] is False
    assert result["child"] is None
