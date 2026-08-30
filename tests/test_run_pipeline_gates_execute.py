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
# has to be one that can `import prismaquant`. Deriving it from sys.executable
# means that is ALWAYS the interpreter running these tests -- Rob's cu130 venv
# locally, setup-python's 3.11/3.12 in CI. The previous hardcoded venv path
# does not exist on a CI runner, so `python3` fell back to /usr/bin/python3,
# which has no prismaquant; the heredoc died on the import and the outer
# `if ! ...` turned every failure into the gate's own `exit 2`. That made the
# FIRED assertions pass for the wrong reason and the PASSED assertions fail,
# which is exactly the signature these three tests showed in CI while passing
# locally. A gate test must exercise the gate, not the absence of an import.
VENV_BIN = str(Path(sys.executable).resolve().parent)
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


def test_cb_learned_trainer_version_enum_fires():
    block = extract_block(
        'case "$CB_LEARNED_TRAINER_VERSION" in',
        "CB_LEARNED_TRAINER_VERSION must be v1 or v2",
    )

    def status(value: str) -> int:
        return run_block(block, {"CB_LEARNED_TRAINER_VERSION": value})

    assert status("v3") == FIRED
    assert status("") == FIRED
    assert status("V2") == FIRED  # the enum is case-sensitive
    assert status("v1") == PASSED
    assert status("v2") == PASSED


def test_cb_learned_receipt_requires_v2_trainer_fires():
    block = extract_block(
        'if [[ "$CB_LEARNED_TRAINER_VERSION" == "v1" \\',
        "CB_LEARNED_PROMOTION_RECEIPT requires CB_LEARNED_TRAINER_VERSION=v2",
    )

    def status(version: str, receipt: str) -> int:
        return run_block(
            block,
            {
                "CB_LEARNED_TRAINER_VERSION": version,
                "CB_LEARNED_PROMOTION_RECEIPT": receipt,
            },
        )

    assert status("v1", "/some/receipt.json") == FIRED
    assert status("v1", "") == PASSED
    assert status("v2", "/some/receipt.json") == PASSED


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


def test_cb_learned_v2_requires_probe_imatrix_fires(scratch):
    block = _v2_scope_block()
    receipt = scratch / "receipt.json"
    receipt.write_text("{}")
    identity = scratch / "identity.json"
    identity.write_text("{}")
    good = _v2_env(
        CB_LEARNED_PROMOTION_RECEIPT=str(receipt),
        CB_LEARNED_SOURCE_MODEL_IDENTITY_CACHE=str(identity),
    )

    assert run_block(block, {**good, "CB_IMATRIX_SOURCE": "activation-cache"}) == FIRED
    # The default (unset) is activation-cache, which must also fire.
    unset = dict(good)
    unset.pop("CB_IMATRIX_SOURCE")
    assert run_block(block, unset) == FIRED
    assert run_block(block, good) == PASSED
    # The whole scope gate is inert for v1 and for the lattice scope.
    assert run_block(
        block, _v2_env(CB_LEARNED_TRAINER_VERSION="v1", CB_IMATRIX_SOURCE="x")
    ) == PASSED
    assert run_block(
        block, _v2_env(CB_CODEBOOK_SOURCE_SCOPE="none", CB_IMATRIX_SOURCE="x")
    ) == PASSED


def test_cb_learned_v2_requires_existing_absolute_receipt_fires(scratch):
    block = _v2_scope_block()
    receipt = scratch / "receipt.json"
    receipt.write_text("{}")
    identity = scratch / "identity.json"
    identity.write_text("{}")

    def status(value: str) -> int:
        return run_block(
            block,
            _v2_env(
                CB_LEARNED_PROMOTION_RECEIPT=value,
                CB_LEARNED_SOURCE_MODEL_IDENTITY_CACHE=str(identity),
            ),
        )

    assert status("") == FIRED
    assert status("receipt.json") == FIRED  # relative
    assert status(str(scratch / "absent.json")) == FIRED  # absolute but missing
    assert status(str(receipt)) == PASSED


def test_cb_learned_v2_requires_source_model_identity_cache_fires(scratch):
    block = _v2_scope_block()
    receipt = scratch / "receipt.json"
    receipt.write_text("{}")
    identity = scratch / "identity.json"
    identity.write_text("{}")

    def status(value: str) -> int:
        return run_block(
            block,
            _v2_env(
                CB_LEARNED_PROMOTION_RECEIPT=str(receipt),
                CB_LEARNED_SOURCE_MODEL_IDENTITY_CACHE=value,
            ),
        )

    assert status("") == FIRED
    assert status("identity.json") == FIRED
    assert status(str(scratch / "absent.json")) == FIRED
    assert status(str(identity)) == PASSED


# --------------------------------------------------------------------------
# CB activation scope
# --------------------------------------------------------------------------


def test_cb_activation_scope_enum_fires():
    block = extract_block(
        'case "$CB_ACTIVATION_SCOPE" in',
        "CB_ACTIVATION_SCOPE must be none or nvfp4",
    )

    def status(value: str) -> int:
        return run_block(block, {"CB_ACTIVATION_SCOPE": value})

    assert status("fp8") == FIRED
    assert status("") == FIRED
    assert status("NVFP4") == FIRED
    assert status("none") == PASSED
    assert status("nvfp4") == PASSED


# --------------------------------------------------------------------------
# nvfp4_cb export lane / cache policy
# --------------------------------------------------------------------------


def test_nvfp4_cb_requires_profile_with_inherited_cb_export_lane_fires():
    block = extract_block(
        'if ! PQ_TARGET_PROFILE_RESOLVED="$TARGET_PROFILE_RESOLVED" python3 - ',
        "requires a serving profile whose inherited export_lane.id is nvfp4_cb",
    )

    def status(profile: str) -> int:
        return run_block(block, {"TARGET_PROFILE_RESOLVED": profile})

    assert status("vllm_packed_moe") == FIRED  # exports through compressed-tensors
    assert status("no_such_profile") == FIRED
    assert status("nvfp4_cb") == PASSED
    assert status("qwen38_rtx4090_fp8_cb") == PASSED  # inherits the lane


def test_nvfp4_cb_requires_production_recache_off_fires():
    block = extract_block(
        'if [[ "${PRODUCTION_RECACHE:-1}" != "0" ]]; then',
        "requires PRODUCTION_RECACHE=0",
    )

    assert run_block(block, {"PRODUCTION_RECACHE": "1"}) == FIRED
    assert run_block(block, {}) == FIRED  # the pipeline default is 1
    assert run_block(block, {"PRODUCTION_RECACHE": "0"}) == PASSED


def test_nvfp4_cb_production_cache_only_under_validated_surrogate_fires():
    block = extract_block(
        'if [[ "${PRODUCTION_CACHE:-1}" != "0" \\',
        "permits PRODUCTION_CACHE=1 only for SELECTION_MODE=validated-surrogate",
    )

    def status(cache: str | None, mode: str) -> int:
        env = {"SELECTION_MODE": mode}
        if cache is not None:
            env["PRODUCTION_CACHE"] = cache
        return run_block(block, env)

    assert status("1", "surrogate") == FIRED
    assert status(None, "surrogate") == FIRED  # default cache is 1
    assert status("1", "validated-surrogate") == PASSED
    assert status("0", "surrogate") == PASSED


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


def test_cb_producer_policy_resolution_failure_fires():
    block = extract_block(
        'if ! CB_PRODUCER_META="$(',
        "failed to resolve the CB profile's producer policy.",
    )

    def status(profile: str) -> int:
        return run_block(block, {"TARGET_PROFILE_RESOLVED": profile})

    assert status("no_such_profile") == FIRED
    assert status("nvfp4_cb") == PASSED
    assert status("qwen38_rtx4090_fp8_cb") == PASSED


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


def test_cb_producer_policy_requires_exact_target_platform_fires(scratch):
    contract = scratch / "runtime_contract.json"
    contract.write_text("{}")

    assert _producer_status("some_policy", "", str(contract)) == FIRED
    assert _producer_status("some_policy", "sm_89", str(contract)) == PASSED
    # No producer policy at all: the whole block is inert.
    assert _producer_status("", "", "") == PASSED


def test_cb_producer_policy_requires_existing_gridbook_contract_fires(scratch):
    contract = scratch / "runtime_contract.json"
    contract.write_text("{}")

    assert _producer_status("some_policy", "sm_89", "") == FIRED
    assert _producer_status(
        "some_policy", "sm_89", str(scratch / "absent.json")
    ) == FIRED
    assert _producer_status("some_policy", "sm_89", str(contract)) == PASSED
