"""Execute the shell gates in ``prismaquant/run-pipeline.sh``.

A gate that is only asserted as a *string* would still pass with its
predicate inverted, its variable misspelled, or the whole block sitting
after an early ``exit 0``.  Every test here therefore extracts the gate's
own shell block out of the shipped script and RUNS it in a subshell under
``set -euo pipefail`` with a controlled environment, asserting the exact
exit status (2 = the gate fired, 0 = it did not).

The pattern generalizes ``test_run_pipeline_defaults.py::
test_cb_unlicensed_guard_actually_fires`` from a one-line predicate to a
whole ``if``/``case`` block.

``PQ_RUN_PIPELINE_PATH_FOR_TESTS`` overrides which script is read; it
exists so a mutation harness can point these tests at a deliberately
broken copy and confirm they go red.  A normal pytest run never sets it
and exercises the real, shipped script.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPT = REPO / "prismaquant" / "run-pipeline.sh"
# The gate blocks shell out to a bare `python3`, so the interpreter they find
# has to be the one running pytest.  Keep the lexical venv path: resolving its
# `python` symlink reaches `/usr/bin` and discards the adjacent `pyvenv.cfg`,
# making the heredoc depend on whichever packages happen to be in user-site.
# An import failure is especially dangerous here because the outer `if ! ...`
# turns it into the gate's own `exit 2`, making every FIRED control pass for the
# wrong reason while the PASSED controls fail.
VENV_BIN = str(Path(sys.executable).parent)
SCRATCH = REPO / "scratch" / "run_pipeline_gate_tests"  # gitignored; never /tmp


def _script_path() -> Path:
    return Path(os.environ.get("PQ_RUN_PIPELINE_PATH_FOR_TESTS", str(DEFAULT_SCRIPT)))


def _is_opener(stripped: str) -> bool:
    return (
        stripped == "if"
        or stripped.startswith("if ")
        or stripped == "case"
        or stripped.startswith("case ")
    )


def _is_closer(stripped: str) -> bool:
    return stripped in ("fi", "esac", "fi;", "esac;")


def extract_block(anchor: str, must_contain: str) -> str:
    """Return the whole ``if``/``case`` construct that starts at ``anchor``.

    Heredoc bodies are skipped so embedded Python (``if not ...:``) cannot
    unbalance the shell keyword scan.  The extraction is hardened three
    ways: the anchor must be unique, the block must contain the gate's own
    error text, and the block must parse under ``bash -n``.
    """
    lines = _script_path().read_text().splitlines()
    hits = [i for i, line in enumerate(lines) if anchor in line]
    assert len(hits) == 1, (
        f"anchor {anchor!r} matched {len(hits)} lines in {_script_path()}; "
        "the script drifted and this test must be re-pointed, not relaxed"
    )
    start = hits[0]
    assert _is_opener(lines[start].strip()), lines[start]

    depth = 0
    heredoc: str | None = None
    out: list[str] = []
    for line in lines[start:]:
        out.append(line)
        stripped = line.strip()
        if heredoc is not None:
            if stripped == heredoc:
                heredoc = None
            continue
        if _is_opener(stripped):
            depth += 1
        elif _is_closer(stripped):
            depth -= 1
            if depth == 0:
                break
        here = re.search(r"<<-?'?([A-Za-z_][A-Za-z_0-9]*)'?\s*$", line)
        if here is not None:
            heredoc = here.group(1)
    else:  # pragma: no cover - only on a malformed script
        raise AssertionError(f"unterminated block for anchor {anchor!r}")

    block = "\n".join(out)
    assert must_contain in block, (
        f"extracted block for {anchor!r} does not contain {must_contain!r}"
    )
    check = subprocess.run(
        ["bash", "-n"], input=block, text=True, capture_output=True, check=False
    )
    assert check.returncode == 0, f"extracted block does not parse: {check.stderr}"
    return block


def run_block(block: str, env: dict, preamble: str = "") -> int:
    """Run one extracted gate block and return its exit status."""
    body = "set -euo pipefail\n" + (preamble + "\n" if preamble else "") + block
    full_env = {
        "PATH": f"{VENV_BIN}:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", str(REPO)),
        "PYTHONPATH": str(REPO),
        "CUDA_VISIBLE_DEVICES": "",
        "LC_ALL": "C",
    }
    full_env.update(env)
    proc = subprocess.run(
        ["bash", "-c", body],
        env=full_env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert proc.returncode in (0, 2), (
        f"block exited {proc.returncode}\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
    )
    return proc.returncode


FIRED = 2
PASSED = 0


# RETIRED 2026-09-02 with the Gridbook codebook lane
# (archive/gridbook_lane_2026-09-02/). Every gate these tests executed --
# the CB learned-bundle trainer-version enum and its four v2 preconditions,
# CB_ACTIVATION_SCOPE, the three `EXPORT_CONTAINER=nvfp4_cb` preconditions,
# and the three CB producer-policy resolutions -- lived behind that container,
# which now `exit 2`s before any of them is reached. They are deleted, not
# skipped: a gate test for a gate that cannot be reached asserts nothing.


def test_gate_python_path_preserves_pytest_virtual_environment():
    """The controlled PATH must not resolve the venv launcher to `/usr/bin`."""
    selected = shutil.which("python3", path=f"{VENV_BIN}:/usr/bin:/bin")

    assert selected is not None
    assert Path(selected).parent == Path(sys.executable).parent


@pytest.fixture(scope="module")
def scratch() -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    yield SCRATCH
    shutil.rmtree(SCRATCH, ignore_errors=True)


# --------------------------------------------------------------------------
# PrismaSnap lane admission
# --------------------------------------------------------------------------


def test_prismasnap_source_lane_admission_fires(scratch):
    """PrismaSnap-prepared sources are admitted only to compressed-tensors."""
    block = extract_block(
        'if [[ ( -e "${MODEL_PATH}/prismasnap_provenance.json" \\',
        "PrismaSnap-prepared sources are admitted only",
    )
    prepared = scratch / "prismasnap_model"
    prepared.mkdir(exist_ok=True)
    (prepared / "prismasnap_provenance.json").write_text("{}")
    plain = scratch / "plain_model"
    plain.mkdir(exist_ok=True)

    def status(model_path: Path, container: str) -> int:
        return run_block(
            block,
            {"MODEL_PATH": str(model_path), "EXPORT_CONTAINER": container},
        )

    assert status(prepared, "nvfp4_cb") == FIRED
    assert status(prepared, "gguf") == FIRED
    # A PrismaSnap source on the measured native lane is admitted...
    assert status(prepared, "compressed-tensors") == PASSED
    # ...and a plain source is admitted on any lane.
    assert status(plain, "nvfp4_cb") == PASSED


# --------------------------------------------------------------------------
# CB learned-codebook trainer / receipt / scope
# --------------------------------------------------------------------------


def _v2_scope_block() -> str:
    return extract_block(
        'if [[ "$CB_LEARNED_TRAINER_VERSION" == "v2" \\',
        "learned-v2 requires CB_IMATRIX_SOURCE=probe",
    )


def _v2_env(**overrides) -> dict:
    env = {
        "CB_LEARNED_TRAINER_VERSION": "v2",
        "CB_CODEBOOK_SOURCE_SCOPE": "fp8",
        "CB_IMATRIX_SOURCE": "probe",
        "CB_LEARNED_PROMOTION_RECEIPT": "",
        "CB_LEARNED_SOURCE_MODEL_IDENTITY_CACHE": "",
    }
    env.update(overrides)
    return env


# --------------------------------------------------------------------------
# CB activation scope
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# nvfp4_cb export lane / cache policy
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# lm_head policy resolver
# --------------------------------------------------------------------------


def test_lm_head_policy_resolver_failure_fires(scratch):
    """The resolver's own failure arm, exercised through the real heredoc."""
    block = extract_block(
        "if ! LM_HEAD_POLICY_TEXT=",
        "failed to resolve fixed/unpinned lm_head policy.",
    )

    # `detect_profile` refuses a path that does not exist, so the negative
    # control needs a real directory. Its config matches no registered
    # profile, which is the DefaultProfile fallback this gate expects.
    model_dir = scratch / "lm-head-policy-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        '{"model_type": "not_a_registered_arch", "architectures": []}'
    )

    def status(lm_head_format: str, model_path: str | None = None) -> int:
        return run_block(
            block,
            {
                "MODEL_PATH": model_path or str(model_dir),
                "ALLOW_PINNED": "",
                "LM_HEAD_FORMAT": lm_head_format,
                "FORMATS": "NVFP4,FP8_DYNAMIC,BF16",
            },
        )

    assert status("NOT_A_REAL_FORMAT") == FIRED
    assert status("") == FIRED
    assert status("BF16") == PASSED
    assert status("NVFP4") == PASSED

    # A typo'd MODEL_PATH must now reach this gate rather than sailing through
    # profile detection on default structure assumptions.
    assert status("BF16", model_path="/nonexistent/model/path") == FIRED


def test_lm_head_policy_incomplete_result_fires():
    """The ``< 5`` arity check on the resolver's parsed output.

    The array is fabricated in the preamble, so this covers the predicate,
    not the ``mapfile`` that feeds it.
    """
    block = extract_block(
        "if (( ${#LM_HEAD_POLICY_LINES[@]} < 5 )); then",
        "lm_head policy resolver returned an incomplete result.",
    )

    def status(count: int) -> int:
        items = " ".join(f"line{i}" for i in range(count))
        return run_block(block, {}, preamble=f"LM_HEAD_POLICY_LINES=({items})")

    assert status(0) == FIRED
    assert status(4) == FIRED
    assert status(5) == PASSED
    assert status(6) == PASSED


# --------------------------------------------------------------------------
# Validated-frontier materialization
# --------------------------------------------------------------------------


def test_validated_frontier_materialization_enum_fires():
    block = extract_block(
        'case "$VALIDATED_FRONTIER_MATERIALIZATION" in',
        "VALIDATED_FRONTIER_MATERIALIZATION must be hooks or inplace",
    )

    def status(value: str) -> int:
        return run_block(block, {"VALIDATED_FRONTIER_MATERIALIZATION": value})

    assert status("in-place") == FIRED
    assert status("") == FIRED
    assert status("hooks") == PASSED
    assert status("inplace") == PASSED


# --------------------------------------------------------------------------
# AURA streaming / checkpoint dir
# --------------------------------------------------------------------------


def _aura_block() -> str:
    return extract_block(
        'case "$AURA_COST_STREAMING" in',
        "AURA_COST_STREAMING=1 requires an absolute AURA_COST_CHECKPOINT_DIR",
    )


def _aura_status(streaming: str, checkpoint_dir: str) -> int:
    return run_block(
        _aura_block(),
        {
            "AURA_COST_STREAMING": streaming,
            "AURA_COST_CHECKPOINT_DIR": checkpoint_dir,
        },
    )


def test_aura_streaming_requires_absolute_checkpoint_dir_fires():
    assert _aura_status("1", "") == FIRED
    assert _aura_status("1", "relative/ckpt") == FIRED
    assert _aura_status("true", "relative/ckpt") == FIRED
    assert _aura_status("1", "/home/rob/dq-runs/ckpt") == PASSED


def test_aura_checkpoint_dir_requires_streaming_fires():
    assert _aura_status("0", "/home/rob/dq-runs/ckpt") == FIRED
    assert _aura_status("off", "/home/rob/dq-runs/ckpt") == FIRED
    assert _aura_status("0", "") == PASSED


def test_aura_streaming_enum_fires():
    assert _aura_status("2", "") == FIRED
    assert _aura_status("", "") == FIRED
    assert _aura_status("auto", "") == FIRED


# --------------------------------------------------------------------------
# CB producer policy / target platform / Gridbook contract pin
# --------------------------------------------------------------------------


def _producer_policy_block() -> str:
    return extract_block(
        'if [[ -n "$CB_PRODUCER_POLICY" ]]; then',
        "declares producer_policy=",
    )


def _producer_status(policy: str, platform: str, contract: str) -> int:
    return run_block(
        _producer_policy_block(),
        {
            "CB_PRODUCER_POLICY": policy,
            "CB_TARGET_PLATFORM": platform,
            "TARGET_PROFILE_RESOLVED": "qwen38_rtx4090_fp8_cb",
            "GRIDBOOK_PRODUCER_RUNTIME_CONTRACT": contract,
        },
        preamble="CB_EXPORT_ARGS=(placeholder)",
    )


