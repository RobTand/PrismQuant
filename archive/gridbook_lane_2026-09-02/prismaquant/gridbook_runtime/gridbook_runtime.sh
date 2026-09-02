#!/usr/bin/env bash
# Resolve Gridbook from one immutable external-source pin.
#
# Host use (source this file):
#   . prismaquant/gridbook_runtime/gridbook_runtime.sh
#   gridbook_runtime_prepare
#   docker run "${GRIDBOOK_RUNTIME_DOCKER_ARGS[@]}" ... \
#     bash -c 'bash "${PQ_GRIDBOOK_RUNTIME_HELPER:?}" install-container; ...'
#
# The only override is deliberately narrow:
#   GRIDBOOK_RUNTIME_CHECKOUT=/clean/checkout
#       The checkout root must be clean and HEAD must equal the JSON pin.
#
# With neither override, the exact commit is fetched into a content-addressed
# user cache. Branches, tags, dirty trees, abbreviated hashes, and unhashed
# artifacts are rejected. Gridbook remains a separate project and is never
# copied back into the PrismaQuant source tree.

_GRIDBOOK_RUNTIME_ASSET_DIR="$({
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
})"
_GRIDBOOK_RUNTIME_PIN_FILE="${_GRIDBOOK_RUNTIME_ASSET_DIR}/gridbook_runtime_pin.json"

_gridbook_runtime_error() {
    printf 'gridbook-runtime: ERROR: %s\n' "$*" >&2
    return 2
}

_gridbook_runtime_is_commit() {
    [[ "$1" =~ ^[0-9a-f]{40}$ ]]
}

gridbook_runtime_load_pin() {
    local values
    if [[ ! -f "$_GRIDBOOK_RUNTIME_PIN_FILE" ]]; then
        _gridbook_runtime_error \
            "missing pin file $_GRIDBOOK_RUNTIME_PIN_FILE"
        return
    fi
    if ! values="$(python3 - "$_GRIDBOOK_RUNTIME_PIN_FILE" <<'PY'
import json
import re
import sys

path = sys.argv[1]
def strict_object(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON member {name!r}")
        value[name] = item
    return value

with open(path, encoding="utf-8") as f:
    pin = json.load(f, object_pairs_hook=strict_object)
required = {
    "schema", "repository", "commit", "version", "version_is_release",
    "runtime_contract_schema", "required_abi_features",
}
if set(pin) != required:
    raise SystemExit(
        f"{path}: expected exactly {sorted(required)}, got {sorted(pin)}")
if pin["schema"] != "prismaquant.gridbook_runtime_pin.v3":
    raise SystemExit(f"{path}: unsupported schema {pin['schema']!r}")
repo = pin["repository"]
if repo != "https://github.com/RobTand/gridbook.git":
    raise SystemExit(f"{path}: repository is not the reviewed Gridbook origin")
commit = pin["commit"]
pending = "PENDING_GRIDBOOK_V0_8_11_RELEASE_COMMIT"
if not isinstance(commit, str) or (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None and commit != pending):
    raise SystemExit(f"{path}: invalid exact release commit field")
version = pin["version"]
if not isinstance(version, str) or re.fullmatch(
        r"[0-9]+(?:[.][0-9]+)*(?:[A-Za-z0-9.+-]*)?", version) is None:
    raise SystemExit(f"{path}: invalid package version {version!r}")
if version != "0.9.1":
    raise SystemExit(
        f"{path}: runtime-contract v12 release must be version 0.9.1")
# True only when `commit` IS the release tag commit for `version`. A pin that
# advances to a post-release master commit keeps the same `version` string (the
# runtime self-reports it until the next bump), so the version alone cannot say
# whether this runtime was ever released -- that is what this flag records, and
# what stops a rung table from crediting an unreleased runtime.
if not isinstance(pin["version_is_release"], bool):
    raise SystemExit(f"{path}: version_is_release must be a JSON boolean")
if commit == pending and pin["version_is_release"]:
    raise SystemExit(f"{path}: unresolved commit cannot be marked released")
contract_schema = pin["runtime_contract_schema"]
if contract_schema != "gridbook.runtime-contract.v12":
    raise SystemExit(f"{path}: unsupported runtime contract schema")
features = pin["required_abi_features"]
required_features = {
    "routed_moe_per_role_codebook_lut": 1,
    "source_fp8_block128_w8a16": 1,
    "dspark_construction_physical_bridge": 1,
}
if features != required_features or any(
        type(value) is not int for value in features.values()):
    raise SystemExit(f"{path}: required ABI feature set differs")
print(
    pin["schema"], repo, commit, version, int(pin["version_is_release"]),
    contract_schema,
    features["routed_moe_per_role_codebook_lut"],
    features["source_fp8_block128_w8a16"],
    features["dspark_construction_physical_bridge"], sep="\t",
)
PY
)"; then
        _gridbook_runtime_error "invalid pin file $_GRIDBOOK_RUNTIME_PIN_FILE"
        return
    fi

    IFS=$'\t' read -r GRIDBOOK_RUNTIME_PIN_SCHEMA \
        GRIDBOOK_RUNTIME_REPOSITORY GRIDBOOK_RUNTIME_COMMIT \
        GRIDBOOK_RUNTIME_VERSION GRIDBOOK_RUNTIME_VERSION_IS_RELEASE \
        GRIDBOOK_RUNTIME_CONTRACT_SCHEMA \
        GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT \
        GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16 \
        GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE \
        <<<"$values"
    if ! _gridbook_runtime_is_commit "$GRIDBOOK_RUNTIME_COMMIT"; then
        _gridbook_runtime_error \
            "Gridbook v0.9.1 exact release commit is unresolved; replace the tracked placeholder"
        return
    fi
    if [[ "$GRIDBOOK_RUNTIME_VERSION_IS_RELEASE" != 1 ]]; then
        _gridbook_runtime_error \
            "Gridbook v0.9.1 pin is not an exact released commit"
        return
    fi
    export GRIDBOOK_RUNTIME_PIN_SCHEMA GRIDBOOK_RUNTIME_REPOSITORY \
        GRIDBOOK_RUNTIME_COMMIT GRIDBOOK_RUNTIME_VERSION \
        GRIDBOOK_RUNTIME_VERSION_IS_RELEASE \
        GRIDBOOK_RUNTIME_CONTRACT_SCHEMA \
        GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT \
        GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16 \
        GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE
}

_gridbook_runtime_source_version() {
    python3 - "$1" <<'PY'
import pathlib
import re
import sys
import tomllib

root = pathlib.Path(sys.argv[1])
with (root / "pyproject.toml").open("rb") as f:
    project = tomllib.load(f).get("project", {})
if project.get("name") != "gridbook":
    raise SystemExit(f"{root}: pyproject project.name is not 'gridbook'")
text = (root / "gridbook" / "__init__.py").read_text(encoding="utf-8")
matches = re.findall(r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
                     text, flags=re.MULTILINE)
if len(matches) != 1:
    raise SystemExit(f"{root}: expected exactly one gridbook __version__")
print(matches[0])
PY
}

gridbook_runtime_verify_checkout() {
    local checkout=${1:-}
    local expected_commit=${2:-}
    local expected_version=${3:-}
    local resolved root head dirty source_version
    local -a safe_git

    if [[ -z "$checkout" || ! -d "$checkout" ]]; then
        _gridbook_runtime_error "checkout is not a directory: ${checkout:-<empty>}"
        return
    fi
    if ! _gridbook_runtime_is_commit "$expected_commit"; then
        _gridbook_runtime_error \
            "expected checkout commit must be full 40-hex, got $expected_commit"
        return
    fi
    if ! command -v git >/dev/null 2>&1; then
        _gridbook_runtime_error "git is required to attest a Gridbook checkout"
        return
    fi
    resolved="$(cd -- "$checkout" && pwd -P)" || return 2
    # Docker commonly reads a host-owned bind mount as root. Attest this exact
    # resolved checkout without weakening Git's ownership checks globally.
    safe_git=(git -c "safe.directory=$resolved" -C "$resolved")
    if ! root="$("${safe_git[@]}" rev-parse --show-toplevel 2>/dev/null)"; then
        _gridbook_runtime_error "$resolved is not a Git checkout"
        return
    fi
    root="$(cd -- "$root" && pwd -P)" || return 2
    if [[ "$root" != "$resolved" ]]; then
        _gridbook_runtime_error \
            "checkout override must name the repository root ($root), got $resolved"
        return
    fi
    if ! head="$("${safe_git[@]}" rev-parse 'HEAD^{commit}' 2>/dev/null)"; then
        _gridbook_runtime_error "cannot resolve Gridbook checkout HEAD"
        return
    fi
    if [[ "$head" != "$expected_commit" ]]; then
        _gridbook_runtime_error \
            "checkout HEAD $head does not equal pinned $expected_commit"
        return
    fi
    dirty="$("${safe_git[@]}" status --porcelain --untracked-files=all)" \
        || return 2
    if [[ -n "$dirty" ]]; then
        _gridbook_runtime_error \
            "checkout $resolved is dirty; immutable runtime inputs must be clean"
        return
    fi
    if ! source_version="$(_gridbook_runtime_source_version "$resolved")"; then
        _gridbook_runtime_error "$resolved is not a valid Gridbook source tree"
        return
    fi
    if [[ "$source_version" != "$expected_version" ]]; then
        _gridbook_runtime_error \
            "checkout version $source_version does not equal pinned $expected_version"
        return
    fi
    printf '%s\n' "$resolved"
}

gridbook_runtime_verify_standalone_checkout() {
    local checkout=${1:-}
    local expected_commit=${2:-}
    local expected_version=${3:-}
    local resolved git_dir common_dir expected_git_dir
    local -a safe_git

    resolved="$(gridbook_runtime_verify_checkout "$checkout" \
        "$expected_commit" "$expected_version")" || return
    if [[ ! -d "$resolved/.git" || -L "$resolved/.git" ]]; then
        _gridbook_runtime_error \
            "cached checkout is not self-contained: $resolved/.git is not a real directory"
        return
    fi
    safe_git=(git -c "safe.directory=$resolved" -C "$resolved")
    git_dir="$("${safe_git[@]}" rev-parse --git-dir 2>/dev/null)" || return 2
    common_dir="$("${safe_git[@]}" rev-parse --git-common-dir 2>/dev/null)" \
        || return 2
    git_dir="$(cd -- "$resolved" && cd -- "$git_dir" && pwd -P)" || return 2
    common_dir="$(cd -- "$resolved" && cd -- "$common_dir" && pwd -P)" \
        || return 2
    expected_git_dir="$(cd -- "$resolved/.git" && pwd -P)" || return 2
    if [[ "$git_dir" != "$expected_git_dir" \
          || "$common_dir" != "$expected_git_dir" ]]; then
        _gridbook_runtime_error \
            "cached checkout Git metadata escapes its standalone .git directory"
        return
    fi
    if [[ -e "$expected_git_dir/objects/info/alternates" ]]; then
        _gridbook_runtime_error \
            "cached checkout uses an external Git object alternate"
        return
    fi
    printf '%s\n' "$resolved"
}

_gridbook_runtime_cache_root() {
    local root
    if [[ -n "${GRIDBOOK_RUNTIME_CACHE_DIR:-}" ]]; then
        root=$GRIDBOOK_RUNTIME_CACHE_DIR
    elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
        root="${XDG_CACHE_HOME}/prismaquant/gridbook-runtime"
    elif [[ -n "${HOME:-}" ]]; then
        root="${HOME}/.cache/prismaquant/gridbook-runtime"
    else
        _gridbook_runtime_error \
            "set GRIDBOOK_RUNTIME_CACHE_DIR (HOME and XDG_CACHE_HOME are unset)"
        return
    fi
    if [[ -z "$root" || "$root" == "/" ]]; then
        _gridbook_runtime_error "unsafe Gridbook cache root: ${root:-<empty>}"
        return
    fi
    printf '%s\n' "$root"
}

_gridbook_runtime_fetch_pinned_checkout() (
    set -euo pipefail
    local cache_root destination tmp=""
    cache_root="$(_gridbook_runtime_cache_root)"
    mkdir -p -- "$cache_root"
    cache_root="$(cd -- "$cache_root" && pwd -P)"
    destination="${cache_root}/${GRIDBOOK_RUNTIME_COMMIT}"

    if [[ -e "$destination" ]]; then
        gridbook_runtime_verify_standalone_checkout "$destination" \
            "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION"
        exit
    fi

    tmp="$(mktemp -d \
        "${cache_root}/.fetch-${GRIDBOOK_RUNTIME_COMMIT:0:12}.XXXXXX")"
    cleanup() {
        if [[ -n "$tmp" && -d "$tmp" ]]; then
            chmod -R u+w -- "$tmp" 2>/dev/null || true
            rm -rf -- "$tmp"
        fi
    }
    trap cleanup EXIT

    git -C "$tmp" init --quiet
    git -C "$tmp" remote add origin "$GRIDBOOK_RUNTIME_REPOSITORY"
    git -C "$tmp" fetch --quiet --depth=1 --no-tags origin \
        "$GRIDBOOK_RUNTIME_COMMIT"
    local fetched
    fetched="$(git -C "$tmp" rev-parse 'FETCH_HEAD^{commit}')"
    if [[ "$fetched" != "$GRIDBOOK_RUNTIME_COMMIT" ]]; then
        _gridbook_runtime_error \
            "remote returned $fetched for pinned $GRIDBOOK_RUNTIME_COMMIT"
        exit 2
    fi
    git -C "$tmp" checkout --quiet --detach "$GRIDBOOK_RUNTIME_COMMIT"
    gridbook_runtime_verify_standalone_checkout "$tmp" \
        "$GRIDBOOK_RUNTIME_COMMIT" \
        "$GRIDBOOK_RUNTIME_VERSION" >/dev/null

    if mv -T -- "$tmp" "$destination" 2>/dev/null; then
        tmp=""
    elif [[ ! -e "$destination" ]]; then
        _gridbook_runtime_error \
            "could not publish fetched checkout at $destination"
        exit 2
    fi
    gridbook_runtime_verify_standalone_checkout "$destination" \
        "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION"
)

_gridbook_runtime_materialize_checkout() (
    set -euo pipefail
    local supplied=${1:-}
    local source cache_root destination tmp="" fetched
    source="$(gridbook_runtime_verify_checkout "$supplied" \
        "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION")"
    cache_root="$(_gridbook_runtime_cache_root)"
    mkdir -p -- "$cache_root"
    cache_root="$(cd -- "$cache_root" && pwd -P)"
    destination="${cache_root}/${GRIDBOOK_RUNTIME_COMMIT}"

    # A linked worktree's .git file points outside its own directory.  It is a
    # valid host input but ceases to be a repository when Docker mounts only
    # that directory.  Import every override into the same commit-addressed
    # standalone cache used by the remote-fetch path, so the mounted checkout
    # is self-contained and independent of host worktree layout.
    if [[ -e "$destination" ]]; then
        gridbook_runtime_verify_standalone_checkout "$destination" \
            "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION"
        exit
    fi

    tmp="$(mktemp -d \
        "${cache_root}/.materialize-${GRIDBOOK_RUNTIME_COMMIT:0:12}.XXXXXX")"
    cleanup() {
        if [[ -n "$tmp" && -d "$tmp" ]]; then
            chmod -R u+w -- "$tmp" 2>/dev/null || true
            rm -rf -- "$tmp"
        fi
    }
    trap cleanup EXIT

    git -C "$tmp" init --quiet
    git -c "safe.directory=$source" -C "$tmp" fetch --quiet --depth=1 \
        --no-tags "$source" "$GRIDBOOK_RUNTIME_COMMIT"
    fetched="$(git -C "$tmp" rev-parse 'FETCH_HEAD^{commit}')"
    if [[ "$fetched" != "$GRIDBOOK_RUNTIME_COMMIT" ]]; then
        _gridbook_runtime_error \
            "checkout override returned $fetched for pinned $GRIDBOOK_RUNTIME_COMMIT"
        exit 2
    fi
    git -C "$tmp" checkout --quiet --detach "$GRIDBOOK_RUNTIME_COMMIT"
    gridbook_runtime_verify_standalone_checkout "$tmp" \
        "$GRIDBOOK_RUNTIME_COMMIT" \
        "$GRIDBOOK_RUNTIME_VERSION" >/dev/null

    if mv -T -- "$tmp" "$destination" 2>/dev/null; then
        tmp=""
    elif [[ ! -e "$destination" ]]; then
        _gridbook_runtime_error \
            "could not publish materialized checkout at $destination"
        exit 2
    fi
    gridbook_runtime_verify_standalone_checkout "$destination" \
        "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION"
)

gridbook_runtime_prepare() {
    local checkout_override=${GRIDBOOK_RUNTIME_CHECKOUT:-}
    local source container_source contract_source container_contract

    gridbook_runtime_load_pin || return
    if [[ -n "$checkout_override" ]]; then
        source="$(_gridbook_runtime_materialize_checkout \
            "$checkout_override")" || return
    else
        source="$(_gridbook_runtime_fetch_pinned_checkout)" || return
    fi
    container_source=/opt/prismaquant-gridbook-source
    contract_source=$_GRIDBOOK_RUNTIME_ASSET_DIR
    container_contract=/opt/prismaquant-gridbook-runtime-contract

    local path
    for path in "$source" "$contract_source"; do
        if [[ "$path" == *:* || "$path" == *$'\n'* ]]; then
            _gridbook_runtime_error \
                "bind source path contains a character Docker -v cannot encode: $path"
            return
        fi
    done

    GRIDBOOK_RUNTIME_SOURCE=$source
    GRIDBOOK_RUNTIME_CONTAINER_SOURCE=$container_source
    GRIDBOOK_RUNTIME_CONTRACT_SOURCE=$contract_source
    GRIDBOOK_RUNTIME_CONTAINER_CONTRACT=$container_contract
    GRIDBOOK_RUNTIME_CONTAINER_HELPER="${container_contract}/gridbook_runtime.sh"
    export GRIDBOOK_RUNTIME_SOURCE GRIDBOOK_RUNTIME_CONTAINER_SOURCE \
        GRIDBOOK_RUNTIME_CONTRACT_SOURCE GRIDBOOK_RUNTIME_CONTAINER_CONTRACT \
        GRIDBOOK_RUNTIME_CONTAINER_HELPER

    GRIDBOOK_RUNTIME_DOCKER_ARGS=(
        --workdir /
        --volume "${source}:${container_source}:ro"
        --volume "${contract_source}:${container_contract}:ro"
        --env "PYTHONSAFEPATH=1"
        # Keep /proc/<pid>/environ readable on processes that rename
        # themselves.  vLLM's EngineCore calls setproctitle
        # (vllm/v1/engine/core.py -> vllm.utils.system_utils.set_process_title),
        # and on Linux setproctitle overwrites the contiguous argv+envp block,
        # destroying /proc/<pid>/environ while leaving the process's real
        # os.environ intact.  Every serve attestation reads /proc, so without
        # this the census sees a destroyed remnant for the one process that
        # actually runs the kernels and refuses a correct server.  SPT_NOENV
        # tells setproctitle to use only the argv area (the title is truncated
        # to argv length, which is cosmetic).  This makes the attestation
        # readable, never laxer: values are still compared exactly, so a
        # genuinely wrong EngineCore environment still fails.
        --env "SPT_NOENV=1"
        --env "PQ_GRIDBOOK_RUNTIME_SOURCE=${container_source}"
        --env "PQ_GRIDBOOK_RUNTIME_COMMIT=${GRIDBOOK_RUNTIME_COMMIT}"
        --env "PQ_GRIDBOOK_RUNTIME_VERSION=${GRIDBOOK_RUNTIME_VERSION}"
        --env "PQ_GRIDBOOK_RUNTIME_CONTRACT_SCHEMA=${GRIDBOOK_RUNTIME_CONTRACT_SCHEMA}"
        --env "PQ_GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT=${GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT}"
        --env "PQ_GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16=${GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16}"
        --env "PQ_GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE=${GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE}"
        --env "PQ_GRIDBOOK_RUNTIME_HELPER=${GRIDBOOK_RUNTIME_CONTAINER_HELPER}"
    )
    printf 'gridbook-runtime: checkout %s (%s); contract %s\n' \
        "$source" "$GRIDBOOK_RUNTIME_COMMIT" "$contract_source" >&2
}

gridbook_runtime_container_install_target() {
    local commit=${GRIDBOOK_RUNTIME_COMMIT:-${PQ_GRIDBOOK_RUNTIME_COMMIT:-}}
    if ! _gridbook_runtime_is_commit "$commit"; then
        _gridbook_runtime_error \
            "cannot resolve container install target without a full commit SHA"
        return
    fi
    printf '/tmp/gridbook-runtime-%s\n' "${commit:0:12}"
}

gridbook_runtime_install_container() {
    local source=${PQ_GRIDBOOK_RUNTIME_SOURCE:-}
    local supplied_commit=${PQ_GRIDBOOK_RUNTIME_COMMIT:-}
    local supplied_version=${PQ_GRIDBOOK_RUNTIME_VERSION:-}
    local supplied_contract_schema=${PQ_GRIDBOOK_RUNTIME_CONTRACT_SCHEMA:-}
    local install_source install_spec

    # Apply safe-path mode to pip and to the post-install import proof even
    # when this function is called directly rather than through the prepared
    # Docker argument vector.  The central Docker environment carries the same
    # value into the later vLLM process; this local export alone would not cross
    # the common `bash helper install-container; exec vllm ...` process boundary.
    export PYTHONSAFEPATH=1

    if ! _gridbook_runtime_is_commit "$supplied_commit"; then
        _gridbook_runtime_error \
            "container commit attestation is not a full 40-hex SHA: $supplied_commit"
        return
    fi
    if [[ -z "$supplied_version" ]]; then
        _gridbook_runtime_error "container version attestation is empty"
        return
    fi

    # Re-read the pin from the independently mounted PrismaQuant contract.
    # Docker environment values are transport only: they cannot authorize a
    # different runtime commit or same-version payload.
    gridbook_runtime_load_pin || return
    if [[ "$supplied_commit" != "$GRIDBOOK_RUNTIME_COMMIT" ]]; then
        _gridbook_runtime_error \
            "container commit $supplied_commit does not equal tracked pin $GRIDBOOK_RUNTIME_COMMIT"
        return
    fi
    if [[ "$supplied_version" != "$GRIDBOOK_RUNTIME_VERSION" ]]; then
        _gridbook_runtime_error \
            "container version $supplied_version does not equal tracked pin $GRIDBOOK_RUNTIME_VERSION"
        return
    fi
    if [[ "$supplied_contract_schema" != "$GRIDBOOK_RUNTIME_CONTRACT_SCHEMA" ]]; then
        _gridbook_runtime_error \
            "container runtime-contract schema differs from tracked pin"
        return
    fi

    gridbook_runtime_verify_standalone_checkout "$source" \
        "$GRIDBOOK_RUNTIME_COMMIT" \
        "$GRIDBOOK_RUNTIME_VERSION" >/dev/null || return
    install_source=$(gridbook_runtime_container_install_target) || return
    if [[ -e "$install_source" ]]; then
        _gridbook_runtime_error \
            "container install target already exists: $install_source"
        return
    fi
    mkdir -p -- "$install_source"
    # The host checkout may be owned by a uid that does not exist in the
    # serving image.  Preserving it makes Git's safe-directory check reject
    # the copied repository before pip can install the pinned commit.
    cp -a --no-preserve=ownership -- "$source/." "$install_source/"
    chmod -R u+w -- "$install_source"

    # Install through the copied checkout's VCS URL, not as a bare local
    # directory.  Both inputs contain the same verified bytes, but only the
    # VCS form produces truthful PEP 610 ``vcs_info``.  Gold evidence can then
    # bind the installed distribution to the exact release commit instead of
    # trusting a same-version package or the transport environment alone.
    install_spec="git+file://${install_source}@${GRIDBOOK_RUNTIME_COMMIT}"
    python3 -m pip install --no-deps --no-build-isolation --no-cache-dir --quiet \
        --force-reinstall "$install_spec"
    python3 - "$GRIDBOOK_RUNTIME_VERSION" "$GRIDBOOK_RUNTIME_COMMIT" \
        "$GRIDBOOK_RUNTIME_CONTRACT_SCHEMA" \
        "$GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT" \
        "$GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16" \
        "$GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE" <<'PY'
import importlib
from importlib.metadata import distribution
from importlib.metadata import version
import json
from pathlib import Path
import sys

expected = sys.argv[1]
expected_commit = sys.argv[2]
expected_contract_schema = sys.argv[3]
expected_features = {
    "routed_moe_per_role_codebook_lut": int(sys.argv[4]),
    "source_fp8_block128_w8a16": int(sys.argv[5]),
    "dspark_construction_physical_bridge": int(sys.argv[6]),
}
actual = version("gridbook")
if actual != expected:
    raise SystemExit(f"installed gridbook {actual!r}, expected {expected!r}")
dist = distribution("gridbook")
init_files = [
    item for item in (dist.files or ())
    if str(item) == "gridbook/__init__.py"
]
if len(init_files) != 1:
    raise SystemExit(
        "installed gridbook distribution must contain exactly one "
        "gridbook/__init__.py"
    )
installed_init_path = Path(dist.locate_file(init_files[0]))
if not installed_init_path.is_file() or installed_init_path.is_symlink():
    raise SystemExit(
        "installed gridbook distribution __init__.py is missing or is a symlink"
    )
installed_init = installed_init_path.resolve(strict=True)
package_root = installed_init.parent.resolve(strict=True)
direct_url_files = [
    file for file in (dist.files or ())
    if file.name == "direct_url.json" and ".dist-info" in str(file.parent)
]
if len(direct_url_files) != 1:
    raise SystemExit(
        "installed gridbook must contain exactly one PEP 610 direct_url.json"
    )
direct_url_path = Path(dist.locate_file(direct_url_files[0]))
direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
vcs = direct_url.get("vcs_info") or {}
expected_vcs = {
    "vcs": "git",
    "commit_id": expected_commit,
    "requested_revision": expected_commit,
}
if vcs != expected_vcs:
    raise SystemExit(
        f"installed gridbook PEP 610 vcs_info {vcs!r} does not equal "
        f"the pinned identity {expected_vcs!r}"
    )

# Distribution metadata and Python imports resolve independently.  Prove that
# the module this interpreter would actually hand to vLLM is the selected VCS
# distribution, not a stale same-name directory in CWD or on PYTHONPATH.
module = importlib.import_module("gridbook")
imported_version = getattr(module, "__version__", None)
if imported_version != expected:
    raise SystemExit(
        f"imported gridbook version {imported_version!r}, expected {expected!r}"
    )
module_file_value = getattr(module, "__file__", None)
module_path_value = getattr(module, "__path__", None)
if not isinstance(module_file_value, str) or module_path_value is None:
    raise SystemExit("imported gridbook has no concrete __file__/__path__")
module_file = Path(module_file_value).resolve(strict=True)
module_paths = sorted({
    Path(value).resolve(strict=True) for value in module_path_value
}, key=str)
if module_file != installed_init:
    raise SystemExit(
        "imported gridbook __file__ is not the selected installed "
        "distribution (CWD/PYTHONPATH shadow suspected): "
        f"imported={module_file}, selected={installed_init}"
    )
if not module_paths:
    raise SystemExit("imported gridbook has an empty __path__")
try:
    module_file.relative_to(package_root)
    for entry in module_paths:
        entry.relative_to(package_root)
except ValueError as exc:
    raise SystemExit(
        "imported gridbook path escapes the selected installed distribution "
        f"root {package_root}: {module_paths}"
    ) from exc
spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
if not isinstance(spec_origin, str) or Path(spec_origin).resolve(
    strict=True
) != module_file:
    raise SystemExit("imported gridbook __spec__.origin differs from __file__")

# Version/commit identity is insufficient for a cross-repository wire route.
# Read Gridbook's packaged, standard-library-only consumer contract and require
# the exact feature values the producer pin names.
runtime_contract_module = importlib.import_module("gridbook.runtime_contract")
load_contract = getattr(runtime_contract_module, "load_runtime_contract", None)
if not callable(load_contract):
    raise SystemExit("installed gridbook has no runtime-contract loader")
runtime_contract = load_contract()
if not isinstance(runtime_contract, dict):
    raise SystemExit("installed gridbook runtime contract is not a mapping")
if runtime_contract.get("schema") != expected_contract_schema:
    raise SystemExit(
        "installed gridbook runtime-contract schema differs: "
        f"{runtime_contract.get('schema')!r}"
    )
abi_features = runtime_contract.get("abi_features")
if not isinstance(abi_features, dict):
    raise SystemExit("installed gridbook runtime contract lacks abi_features")
for name, expected_value in expected_features.items():
    if abi_features.get(name) != expected_value:
        raise SystemExit(
            f"installed gridbook ABI feature {name!r} is "
            f"{abi_features.get(name)!r}, expected {expected_value}"
        )
print(
    f"gridbook-runtime: installed gridbook {actual} "
    f"from git commit {expected_commit}; import root {package_root}"
)
PY
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    case "${1:-}" in
        install-container)
            gridbook_runtime_install_container
            ;;
        print-pin)
            gridbook_runtime_load_pin
            printf '%s %s %s\n' "$GRIDBOOK_RUNTIME_REPOSITORY" \
                "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION"
            ;;
        *)
            printf 'usage: %s {install-container|print-pin}\n' "$0" >&2
            exit 2
            ;;
    esac
fi
