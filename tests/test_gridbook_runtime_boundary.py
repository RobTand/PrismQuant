"""Gridbook is an immutable external runtime, never vendored into PrismaQuant."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path
import re
import subprocess

from prismaquant.gridbook_runtime_pin import (
    GRIDBOOK_REQUIRED_ABI_FEATURES,
    GRIDBOOK_RUNTIME_CONTRACT_SCHEMA,
    GRIDBOOK_RUNTIME_PIN_SCHEMA,
    GRIDBOOK_RUNTIME_RELEASE_COMMIT,
    GRIDBOOK_RUNTIME_RELEASE_VERSION,
)
from prismaquant.gridbook_serving_runtime_pin import (
    GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT,
    GRIDBOOK_SERVING_RUNTIME_RELEASE_WHEEL_SHA256,
    GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
    GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING,
    GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING,
)
from prismaquant.gridbook_validation_only_policy import (
    SM120_CANDIDATE_GRIDBOOK_COMMIT,
    SM120_CANDIDATE_GRIDBOOK_VERSION,
    SM120_VALIDATION_CANDIDATE_PIN_PATH,
    VALIDATION_ONLY_DISPOSITION,
    load_sm120_validation_candidate_pin,
)


REPO = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO / "prismaquant" / "gridbook_runtime"
HELPER = ASSET_DIR / "gridbook_runtime.sh"
PIN = ASSET_DIR / "gridbook_runtime_pin.json"
SERVING_HELPER = ASSET_DIR / "gridbook_serving_runtime.sh"
SERVING_PIN = ASSET_DIR / "gridbook_serving_runtime_pin.json"
VALIDATION_CANDIDATE_PIN = SM120_VALIDATION_CANDIDATE_PIN_PATH
LIVE_SCRIPTS = (
    "canary_ladder.sh",
    "serve_dsv4_cb_validate.sh",
    "serve_hy3_smoke.sh",
    "serve_hy3_teb.sh",
    "serve_laguna_smoke.sh",
    "serve_qwen27b_smoke.sh",
    "serve_qwen38_cb_a_smoke.sh",
    "smoke_nvfp4_cb_delegation.sh",
)
# Launchers that cross the *serving* pin (contract v4) rather than the build
# pin (contract v3).  DSv4 was the only one until 2026-08-15: a CB artifact
# whose recipe assigns model.embed_tokens to NVFP4 cannot load under the build
# pin's Gridbook 0.8.5, because the quantized embedding mechanism ships in
# 0.8.7.  Membership is a property of the artifact, not of the model family,
# so this is a set rather than a name comparison.
SERVING_PIN_SCRIPTS = frozenset({
    "serve_dsv4_cb_validate.sh",
    "serve_qwen38_cb_a_smoke.sh",
})


def _bash(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "bash", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_gridbook_pins_separate_release_authority_from_validation_candidate():
    pins = [
        path
        for root in (REPO / "prismaquant", REPO / "scripts")
        for path in root.rglob("*gridbook*pin*.json")
    ]
    assert set(pins) == {PIN, SERVING_PIN, VALIDATION_CANDIDATE_PIN}
    payload = json.loads(PIN.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema", "repository", "commit", "version", "version_is_release",
        "runtime_contract_schema", "required_abi_features",
    }
    # Bound to the module constants for the same reason the serving block
    # below is (see its comment): the invariant is that the packaged JSON and
    # the module agree, not that the pin may never move.  Re-typed literals
    # here are what made the 0.8.5 -> 0.8.11 advance a multi-file edit.
    assert payload["schema"] == GRIDBOOK_RUNTIME_PIN_SCHEMA
    assert payload["repository"] == "https://github.com/RobTand/gridbook.git"
    assert payload["commit"] == GRIDBOOK_RUNTIME_RELEASE_COMMIT
    assert re.fullmatch(r"[0-9]+(?:[.][0-9]+)+(?:[A-Za-z0-9.+-]*)?",
                        payload["version"])
    assert payload["version"] == GRIDBOOK_RUNTIME_RELEASE_VERSION
    assert isinstance(payload["version_is_release"], bool)
    assert payload["version_is_release"] is True
    assert payload["runtime_contract_schema"] == GRIDBOOK_RUNTIME_CONTRACT_SCHEMA
    assert payload["required_abi_features"] == dict(
        GRIDBOOK_REQUIRED_ABI_FEATURES
    )
    serving = json.loads(SERVING_PIN.read_text(encoding="utf-8"))
    assert serving == {
        "schema": "prismaquant.gridbook_serving_runtime_pin.v1",
        "repository": "https://github.com/RobTand/gridbook.git",
        # Bound to the module constants rather than re-typed as literals.
        # The invariant worth defending is that the packaged JSON and the
        # module agree -- two files that must move together on a release.
        # Literals additionally assert "the pin may never change", which is
        # false, and which is exactly how the 0.8.6 -> 0.8.7 bump found this.
        "commit": GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT,
        "version": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
        "version_is_release": True,
        "wheel_sha256": GRIDBOOK_SERVING_RUNTIME_RELEASE_WHEEL_SHA256,
        "runtime_contract_schema": "gridbook.runtime-contract.v4",
        "required_abi_features": {
            "routed_moe_per_role_codebook_lut": 1,
            "source_fp8_block128_w8a16": 1,
            "dspark_construction_physical_bridge": 1,
        },
    }
    # The pending sentinels are a BUILD-TIME state, not a shippable one: a
    # serving pin still carrying them resolves to no verifiable runtime at all.
    # Assert their absence separately from the equality above, so that a future
    # edit which reintroduces a placeholder fails on the reason rather than on
    # an opaque dict diff.
    assert serving["commit"] != GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING
    assert serving["wheel_sha256"] != GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING
    assert serving["version_is_release"] is True

    # The third pin is intentionally not part of the producer/serving release
    # pair below.  It authorizes only deterministic SM120 validation exports,
    # and its schema carries a categorical, compile-only disposition instead
    # of repository-install or wheel-release authority.
    candidate = load_sm120_validation_candidate_pin()
    assert candidate["gridbook"]["commit"] == (
        SM120_CANDIDATE_GRIDBOOK_COMMIT
    )
    assert candidate["gridbook"]["version"] == (
        SM120_CANDIDATE_GRIDBOOK_VERSION
    )
    assert candidate["gridbook"]["version_is_release"] is False
    assert candidate["gridbook"]["release_tag"] is None
    assert candidate["artifact_disposition"] == VALIDATION_ONLY_DISPOSITION
    assert candidate["runtime_qualification_ceiling"] == "compile_only"


def test_producer_and_serving_pins_name_the_same_gridbook_release():
    """The two pins move in lockstep; drift between them is a defect.

    2026-08-21.  PrismaQuant carries two pins for two different jobs -- the
    producer pin authorizes builds/exports and the gold measurement
    environment, the serving pin authorizes route status and the serve wheel
    -- and nothing required them to agree.  They drifted three releases: the
    producer pin sat at 0.8.5 while every gate, certificate and shipped
    artifact resolved through the serving pin's 0.8.11.  CI went red the way
    that kind of drift always surfaces sideways: the "pinned Gridbook
    contract" job installs the PRODUCER pin's commit, and the materialized
    contract test could only find an indexed contract for a version nothing
    still pinned.

    Two pins remain the right shape -- the serving pin additionally binds a
    wheel digest, and a future release may legitimately land on one side
    first.  But divergence must be a deliberate, visible act.  This test is
    what makes it visible: advancing one pin without the other fails here,
    naming both, rather than surfacing three files away as a missing
    contract.
    """
    producer = json.loads(PIN.read_text(encoding="utf-8"))
    serving = json.loads(SERVING_PIN.read_text(encoding="utf-8"))
    assert producer["repository"] == serving["repository"]
    assert producer["commit"] == serving["commit"], (
        "producer/serving Gridbook pins name different commits: "
        f"{producer['commit']} vs {serving['commit']}"
    )
    assert producer["version"] == serving["version"], (
        "producer/serving Gridbook pins name different versions: "
        f"{producer['version']} vs {serving['version']}"
    )
    # Same release => same runtime contract and the same ABI closure.  The
    # pin-file schemas differ on purpose (only the serving pin carries a wheel
    # digest); the RUNTIME contract they describe cannot.
    assert producer["runtime_contract_schema"] == (
        serving["runtime_contract_schema"]
    )
    assert producer["required_abi_features"] == (
        serving["required_abi_features"]
    )
    # And the module constants track the files they read.
    assert GRIDBOOK_RUNTIME_RELEASE_COMMIT == (
        GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT
    )
    assert GRIDBOOK_RUNTIME_RELEASE_VERSION == (
        GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION
    )


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_rung_tables_may_only_credit_a_runtime_that_was_actually_released():
    """The version-drift ratchet.

    ``rungs_by_runtime_version`` is keyed by runtime VERSION, but the pin's
    identity is its COMMIT -- and a feature merge does not bump
    ``gridbook.__version__``. So a pin can advance to a post-release master
    commit while keeping the same version string, and a rung table keyed on
    that string silently starts describing a runtime nobody released. Three
    different commits self-reported 0.7.0 during the source-passthrough work,
    which is exactly how this gap was found.

    Two properties close it without inventing a release-history table:

      * no key may name a runtime NEWER than the one actually pinned -- a
        table cannot promise rungs from a version this producer has never
        resolved; and
      * the pinned version may appear as a key only when the pinned commit IS
        that version's release (``version_is_release``). An unreleased pin
        backs nothing, which is the fail-closed direction the spec's own
        "when the pin advances, ADD the version key" rule already asks for.
    """
    payload = json.loads(PIN.read_text(encoding="utf-8"))
    serving_payload = json.loads(SERVING_PIN.read_text(encoding="utf-8"))
    # The ceiling is the SERVING pin, not the producer pin. `fused_mid_m` is a
    # statement about which rungs the runtime that SERVES the artifact routes
    # through the fused lane, so the runtime that must have resolved a version
    # before the table may credit it is the serving runtime. The producer pin
    # is deliberately frozen at the version that BUILT the artifact and does
    # not advance with the serve path -- reading it here would cap the table at
    # the build-time runtime forever and make every serving-runtime bump look
    # like an unresolved promise. Both pins are still required to be resolved
    # releases before their version may appear as a key, which is the property
    # this test exists for; only which pin supplies the ceiling changed.
    pin_version = serving_payload["version"]
    spec = json.loads(
        (REPO / "prismaquant" / "serving_profile_specs" / "nvfp4_cb.json")
        .read_text(encoding="utf-8"))

    keyed: set[str] = set()
    for lane in spec["serving_lanes"]:
        fused = lane.get("fused_mid_m")
        if fused is not None:
            keyed |= set(fused["rungs_by_runtime_version"])

    for version in sorted(keyed):
        assert _version_tuple(version) <= _version_tuple(pin_version), (
            f"serving profile credits runtime {version}, which is newer than "
            f"the pinned serving runtime {pin_version}: a rung table cannot "
            f"promise a version that has never been resolved")

    if pin_version in keyed:
        assert serving_payload["version_is_release"] is True, (
            f"serving profile keys runtime {pin_version}, but the serving pin "
            f"does not resolve it to a release: an unreleased pin backs "
            f"nothing")

    if not payload["version_is_release"]:
        # The prospective table may be reviewed before the tag exists, but
        # every release/readmission/install boundary must reject the unresolved
        # commit. This lets the exact commit be the only late-bound code field.
        assert payload["commit"].startswith("PENDING_GRIDBOOK_")


def test_no_gridbook_runtime_or_tests_are_vendored():
    assert not (REPO / "plugins" / "gridbook").exists()
    assert not (REPO / "scripts" / "sync_gridbook.py").exists()
    assert not (REPO / "tests" / "test_gridbook_sync.py").exists()


def test_the_materialized_runtime_contract_is_data_not_runtime_code():
    """The CONTRACT crosses the boundary; the RUNTIME never does (R3).

    AGENTS.md:38 sanctions exactly one crossing -- "the immutable pin and
    contract" -- and principle 14 says the attestation travels in the contract
    file. So a byte-verbatim copy of Gridbook's packaged ``runtime_contract.json``
    belongs here, and a ``.py``/``.cu``/``.so`` from that repository never does.
    This test is the line between the two.
    """
    stray = sorted(
        path.relative_to(REPO).as_posix()
        for path in ASSET_DIR.iterdir()
        if path.suffix not in {".json", ".sh"}
    )
    assert not stray, (
        f"only pin/contract JSON and the resolver helper belong in {ASSET_DIR}: "
        f"{stray}")

    index = json.loads(
        (ASSET_DIR / "gridbook_runtime_contract_index.json").read_text(
            encoding="utf-8"))
    assert index["packaged_path"] == "gridbook/runtime_contract.json"
    for entry in index["contracts"]:
        contract = json.loads(
            (ASSET_DIR / entry["path"]).read_text(encoding="utf-8"))
        assert contract["schema"] == entry["runtime_contract_schema"]
        assert hashlib.sha256(
            (ASSET_DIR / entry["path"]).read_bytes()).hexdigest() == (
                entry["sha256"]), (
            f"{entry['path']} has drifted from the release it claims to be; "
            "re-materialize it from the pinned commit rather than editing it")
        # `lane_eligibility` is either absent (and the index says so, loudly)
        # or present -- never quietly invented here.
        declared = entry["lane_eligibility"]
        assert declared in {"absent", "present"}
        assert (declared == "present") == ("lane_eligibility" in contract), (
            f"{entry['path']}: the index and the contract disagree about "
            "whether a lane-eligibility table is published")


def test_tests_do_not_import_gridbook_outside_the_pinned_compat_job():
    """AGENTS.md:38 -- the attestation travels in the contract file."""
    # The sanctioned importers are the pinned-compatibility CI job's modules,
    # which run only under PRISMAQUANT_REQUIRE_GRIDBOOK_CONTRACT=1 with the
    # pinned wheel installed. Membership is derived from that GUARD rather
    # than from a filename list, so a new compat test inherits the rule and a
    # test that drops its guard loses the exemption in the same commit.
    guard = "PRISMAQUANT_REQUIRE_GRIDBOOK_CONTRACT"
    violations: list[str] = []
    for path in (REPO / "tests").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if guard in text and "pytestmark" in text:
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "gridbook" or name.startswith("gridbook.")
                   for name in names):
                violations.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not violations, (
        "a test may not import the Gridbook runtime; read the materialized "
        f"contract in {ASSET_DIR.relative_to(REPO)} instead: {violations}")


def test_producer_does_not_import_external_gridbook_runtime():
    violations: list[str] = []
    for path in (REPO / "prismaquant").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "gridbook" or name.startswith("gridbook.")
                   for name in names):
                violations.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not violations, (
        "PrismaQuant is the producer and must not import Gridbook runtime code: "
        f"{violations}")


def test_every_live_script_uses_the_one_external_runtime_helper():
    discovered = {
        path.name
        for path in (REPO / "scripts").glob("*.sh")
        if any(
            marker in path.read_text(encoding="utf-8")
            for marker in (
                "gridbook_runtime_prepare",
                "install-container",
                "gridbook_runtime_install_container",
            )
        )
    }
    assert discovered == set(LIVE_SCRIPTS)
    for name in LIVE_SCRIPTS:
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        if name in SERVING_PIN_SCRIPTS:
            assert "gridbook_serving_runtime.sh" in text, name
            assert (
                "prismaquant/gridbook_runtime/gridbook_serving_runtime.sh"
                in text
            ), name
            assert "gridbook_serving_runtime_prepare" in text, name
            assert "GRIDBOOK_SERVING_RUNTIME_DOCKER_ARGS" in text, name
        else:
            assert "gridbook_runtime.sh" in text, name
            assert "prismaquant/gridbook_runtime/gridbook_runtime.sh" in text, name
            assert "gridbook_runtime_prepare" in text, name
            assert "GRIDBOOK_RUNTIME_DOCKER_ARGS" in text, name
        assert "PQ_GRIDBOOK_RUNTIME_HELPER" in text, name
        assert (
            "install-container" in text
            or "gridbook_runtime_install_container" in text
        ), name
        assert "set -euo pipefail" in text, name
        assert "plugins/gridbook" not in text, name
        assert "/repo/scripts/lib/gridbook_runtime.sh" not in text, name
        assert "--quantization prismaquant" not in text, name
        # Import safety is intentionally owned once by the shared Docker
        # argument vector.  A launcher-local override could appear after that
        # vector and silently win Docker's last-value environment resolution.
        if name == "serve_dsv4_cb_validate.sh":
            # This launcher has a host-side, artifact-build-commit snapshot
            # bootstrap before Gridbook preparation. Its strict source
            # bootstrap requires PYTHONSAFEPATH in the host environment; the
            # container value remains owned by GRIDBOOK_RUNTIME_DOCKER_ARGS.
            assert "-e PYTHONSAFEPATH" not in text, name
            assert "--env PYTHONSAFEPATH" not in text, name
        else:
            assert "PYTHONSAFEPATH" not in text, name
    delegation = (REPO / "scripts" /
                  "smoke_nvfp4_cb_delegation.sh").read_text(encoding="utf-8")
    assert "--quantization gridbook" in delegation


def test_helper_and_live_scripts_are_valid_bash():
    paths = [
        HELPER,
        SERVING_HELPER,
        *(REPO / "scripts" / name for name in LIVE_SCRIPTS),
    ]
    for path in paths:
        proc = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 0, f"{path}: {proc.stderr}"


def _make_gridbook_checkout(root: Path) -> str:
    (root / "gridbook").mkdir(parents=True)
    (root / "gridbook" / "__init__.py").write_text(
        f'__version__ = "{GRIDBOOK_RUNTIME_RELEASE_VERSION}"\n',
        encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "gridbook"\n'
        f'version = "{GRIDBOOK_RUNTIME_RELEASE_VERSION}"\n',
        encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Gridbook Test"],
                   cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"],
                   cwd=root, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def test_checkout_override_requires_exact_clean_commit(tmp_path):
    checkout = tmp_path / "gridbook"
    checkout.mkdir()
    commit = _make_gridbook_checkout(checkout)
    command = (
        f'. "{HELPER}"; '
        'gridbook_runtime_verify_checkout "$1" "$2" '
        f'{GRIDBOOK_RUNTIME_RELEASE_VERSION}')
    clean = _bash(command, str(checkout), commit)
    assert clean.returncode == 0, clean.stderr
    assert clean.stdout.strip() == str(checkout)

    # Exercise the Docker-root compatibility contract: every Git read in the
    # verifier must mark only the exact resolved checkout as safe. A wrapper
    # fails the verification if any rev-parse/status call omits that option.
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    real_git = shutil.which("git")
    assert real_git is not None
    wrapper = wrapper_dir / "git"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'expected=${GRIDBOOK_TEST_SAFE_DIRECTORY:?}\n'
        'if [[ "$1" != "-c" || "$2" != "safe.directory=$expected" '
        '|| "$3" != "-C" || "$4" != "$expected" ]]; then\n'
        '  printf "unsafe git invocation: %q " "$@" >&2\n'
        '  printf "\\n" >&2\n'
        "  exit 97\n"
        "fi\n"
        f'exec "{real_git}" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    safe_command = (
        f'. "{HELPER}"; '
        'PATH="$1:$PATH" GRIDBOOK_TEST_SAFE_DIRECTORY="$2" '
        'gridbook_runtime_verify_checkout "$2" "$3" '
        f'{GRIDBOOK_RUNTIME_RELEASE_VERSION}')
    safe = _bash(safe_command, str(wrapper_dir), str(checkout), commit)
    assert safe.returncode == 0, safe.stderr
    assert safe.stdout.strip() == str(checkout)

    (checkout / "untracked").write_text("dirty", encoding="utf-8")
    dirty = _bash(command, str(checkout), commit)
    assert dirty.returncode == 2
    assert "is dirty" in dirty.stderr

    wrong = _bash(command, str(checkout), "0" * 40)
    assert wrong.returncode == 2
    assert "does not equal pinned" in wrong.stderr


def test_standalone_checkout_rejects_linked_git_metadata(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    commit = _make_gridbook_checkout(repository)
    linked = tmp_path / "linked"
    subprocess.run(
        ["git", "worktree", "add", "--quiet", "--detach", str(linked), commit],
        cwd=repository,
        check=True,
    )
    command = (
        f'. "{HELPER}"; '
        'gridbook_runtime_verify_standalone_checkout "$1" "$2" '
        f'{GRIDBOOK_RUNTIME_RELEASE_VERSION}'
    )
    refused = _bash(command, str(linked), commit)
    assert refused.returncode == 2
    assert "not self-contained" in refused.stderr

    accepted = _bash(command, str(repository), commit)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == str(repository)


def test_prepare_mounts_runtime_source_and_contract_read_only(tmp_path):
    checkout = tmp_path / "gridbook"
    checkout.mkdir()
    commit = _make_gridbook_checkout(checkout)

    assets = tmp_path / "contract"
    assets.mkdir()
    copied_helper = assets / HELPER.name
    shutil.copy2(HELPER, copied_helper)
    (assets / PIN.name).write_text(json.dumps({
        "schema": GRIDBOOK_RUNTIME_PIN_SCHEMA,
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": commit,
        "version": GRIDBOOK_RUNTIME_RELEASE_VERSION,
        "version_is_release": True,
        "runtime_contract_schema": GRIDBOOK_RUNTIME_CONTRACT_SCHEMA,
        "required_abi_features": dict(GRIDBOOK_REQUIRED_ABI_FEATURES),
    }), encoding="utf-8")

    prepared = _bash(
        'GRIDBOOK_RUNTIME_CHECKOUT="$1"; '
        'GRIDBOOK_RUNTIME_CACHE_DIR="$3"; '
        '. "$2"; '
        'gridbook_runtime_prepare; '
        'printf "%s\\n" "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}"',
        str(checkout), str(copied_helper), str(tmp_path / "cache"),
    )
    assert prepared.returncode == 0, prepared.stderr
    args = prepared.stdout.splitlines()
    materialized = tmp_path / "cache" / commit
    assert f"{materialized}:/opt/prismaquant-gridbook-source:ro" in args
    assert (materialized / ".git").is_dir()
    assert (
        f"{assets}:/opt/prismaquant-gridbook-runtime-contract:ro" in args
    )
    assert (
        "PQ_GRIDBOOK_RUNTIME_HELPER="
        "/opt/prismaquant-gridbook-runtime-contract/gridbook_runtime.sh"
    ) in args
    assert "--workdir" in args
    assert "/" in args
    assert "--env" in args
    assert "PYTHONSAFEPATH=1" in args
    assert "SPT_NOENV=1" in args
    assert all("/repo/" not in arg for arg in args)


def test_both_runtime_vectors_keep_proc_environ_readable():
    """Serve attestations read /proc/<pid>/environ, so it must survive a rename.

    vLLM's EngineCore renames itself via setproctitle
    (vllm/v1/engine/core.py -> vllm.utils.system_utils.set_process_title). On
    Linux setproctitle overwrites the contiguous argv+envp block, which
    destroys /proc/<pid>/environ while leaving the process's real os.environ
    untouched. Measured inside the pinned serve image, same process, across
    that one call: /proc/self/environ went from all six probed variables to
    zero while os.environ kept all six.

    That made the environment census structurally unsatisfiable -- it refused a
    correct server because the single process that runs the CB kernels reported
    a destroyed remnant. SPT_NOENV confines the title to the argv area.

    This is deliberately not a relaxation. The census still compares every
    allowlisted name's value exactly; SPT_NOENV only makes those values
    legible, so a genuinely mismatched EngineCore environment still fails. The
    guard below pins that: SPT_NOENV must never enter a compared allowlist,
    because a variable that exists to make the measurement possible must not
    become part of the thing being measured.
    """
    for helper in (HELPER, SERVING_HELPER):
        assert 'SPT_NOENV=1' in helper.read_text(encoding="utf-8"), helper.name

    from prismaquant.dspark_serving_profile import DSPARK_SERVER_ENV_ALLOWLIST
    from tools.serve_fingerprint import SERVER_ENV_ALLOWLIST

    for allowlist in (SERVER_ENV_ALLOWLIST, DSPARK_SERVER_ENV_ALLOWLIST):
        assert "SPT_NOENV" not in allowlist


def test_prepare_materializes_linked_worktree_as_standalone_checkout(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    commit = _make_gridbook_checkout(repository)
    checkout = tmp_path / "linked-worktree"
    subprocess.run(
        ["git", "worktree", "add", "--quiet", "--detach", str(checkout), commit],
        cwd=repository,
        check=True,
    )
    assert (checkout / ".git").is_file()

    assets = tmp_path / "contract"
    assets.mkdir()
    copied_helper = assets / HELPER.name
    shutil.copy2(HELPER, copied_helper)
    (assets / PIN.name).write_text(json.dumps({
        "schema": GRIDBOOK_RUNTIME_PIN_SCHEMA,
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": commit,
        "version": GRIDBOOK_RUNTIME_RELEASE_VERSION,
        "version_is_release": True,
        "runtime_contract_schema": GRIDBOOK_RUNTIME_CONTRACT_SCHEMA,
        "required_abi_features": dict(GRIDBOOK_REQUIRED_ABI_FEATURES),
    }), encoding="utf-8")
    cache = tmp_path / "cache"
    prepared = _bash(
        'GRIDBOOK_RUNTIME_CHECKOUT="$1"; '
        'GRIDBOOK_RUNTIME_CACHE_DIR="$3"; '
        '. "$2"; '
        'gridbook_runtime_prepare; '
        'printf "%s\\n" "$GRIDBOOK_RUNTIME_SOURCE"',
        str(checkout), str(copied_helper), str(cache),
    )
    assert prepared.returncode == 0, prepared.stderr
    materialized = cache / commit
    assert prepared.stdout.strip() == str(materialized)
    assert (materialized / ".git").is_dir()
    assert not (materialized / ".git" / "objects" / "info" / "alternates").exists()
    verified = subprocess.run(
        ["git", "-C", str(materialized), "fsck", "--no-dangling"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert verified.returncode == 0, verified.stderr


def test_container_install_reloads_and_enforces_the_tracked_pin():
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    resolved = _bash(f'bash "{HELPER}" print-pin')
    assert resolved.returncode == 0, resolved.stderr
    assert resolved.stdout.strip().split() == [
        pin["repository"], pin["commit"], pin["version"]
    ]
    wrong_commit = "f" * 40
    commit = _bash(
        f'PQ_GRIDBOOK_RUNTIME_SOURCE=/not-used '
        f'PQ_GRIDBOOK_RUNTIME_COMMIT={wrong_commit} '
        f'PQ_GRIDBOOK_RUNTIME_VERSION={pin["version"]} '
        f'bash "{HELPER}" install-container')
    assert commit.returncode == 2
    assert "does not equal tracked pin" in commit.stderr

    version = _bash(
        f'PQ_GRIDBOOK_RUNTIME_SOURCE=/not-used '
        f'PQ_GRIDBOOK_RUNTIME_COMMIT={pin["commit"]} '
        f'PQ_GRIDBOOK_RUNTIME_VERSION=999.0.0 '
        f'bash "{HELPER}" install-container')
    assert version.returncode == 2
    assert "does not equal tracked pin" in version.stderr


def test_runtime_helper_rejects_resolved_but_unreleased_pin(tmp_path):
    assets = tmp_path / "contract"
    assets.mkdir()
    copied_helper = assets / HELPER.name
    shutil.copy2(HELPER, copied_helper)
    (assets / PIN.name).write_text(json.dumps({
        "schema": GRIDBOOK_RUNTIME_PIN_SCHEMA,
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "a" * 40,
        "version": GRIDBOOK_RUNTIME_RELEASE_VERSION,
        "version_is_release": False,
        "runtime_contract_schema": GRIDBOOK_RUNTIME_CONTRACT_SCHEMA,
        "required_abi_features": dict(GRIDBOOK_REQUIRED_ABI_FEATURES),
    }), encoding="utf-8")
    refused = _bash(f'bash "{copied_helper}" print-pin')
    assert refused.returncode == 2
    assert "not an exact released commit" in refused.stderr


def test_runtime_helper_has_no_wheel_or_runtime_kind_branch():
    text = HELPER.read_text(encoding="utf-8")
    assert "GRIDBOOK_RUNTIME_WHEEL" not in text
    assert "PQ_GRIDBOOK_RUNTIME_KIND" not in text
    assert "gridbook_runtime_verify_wheel" not in text


def test_container_install_preserves_exact_vcs_provenance():
    text = HELPER.read_text(encoding="utf-8")
    assert (
        'git+file://${install_source}@${GRIDBOOK_RUNTIME_COMMIT}' in text
    )
    assert "--force-reinstall" in text
    assert "cp -a --no-preserve=ownership" in text
    assert "--no-build-isolation" in text
    assert '"commit_id": expected_commit' in text
    assert '"requested_revision": expected_commit' in text
    assert '"vcs": "git"' in text
    assert "importlib.import_module(\"gridbook\")" in text
    assert "module_file != installed_init" in text
    assert "entry.relative_to(package_root)" in text
    assert "imported_version != expected" in text
    assert 'importlib.import_module("gridbook.runtime_contract")' in text
    assert 'source_fp8_block128_w8a16' in text


def test_runtime_helper_owns_safe_import_path_and_neutral_workdir():
    text = HELPER.read_text(encoding="utf-8")
    assert '--workdir /' in text
    assert '--env "PYTHONSAFEPATH=1"' in text
    assert "export PYTHONSAFEPATH=1" in text


def test_container_install_path_is_owned_only_by_runtime_helper():
    marker = "/tmp/gridbook-runtime-"
    assert marker in HELPER.read_text(encoding="utf-8")
    for path in (REPO / "scripts").rglob("*.sh"):
        if path == HELPER:
            continue
        assert marker not in path.read_text(encoding="utf-8"), path
    canary = (REPO / "scripts" / "canary_ladder.sh").read_text(
        encoding="utf-8"
    )
    assert canary.count("gridbook_runtime_container_install_target") == 2
