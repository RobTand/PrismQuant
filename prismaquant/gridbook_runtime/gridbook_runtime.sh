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
with open(path, encoding="utf-8") as f:
    pin = json.load(f)
required = {"schema", "repository", "commit", "version", "version_is_release"}
if set(pin) != required:
    raise SystemExit(
        f"{path}: expected exactly {sorted(required)}, got {sorted(pin)}")
if pin["schema"] != "prismaquant.gridbook_runtime_pin.v2":
    raise SystemExit(f"{path}: unsupported schema {pin['schema']!r}")
repo = pin["repository"]
if not isinstance(repo, str) or not repo.startswith("https://") \
        or not repo.endswith(".git"):
    raise SystemExit(f"{path}: repository must be an https .git URL")
commit = pin["commit"]
if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise SystemExit(f"{path}: commit must be a lowercase full 40-hex SHA")
version = pin["version"]
if not isinstance(version, str) or re.fullmatch(
        r"[0-9]+(?:[.][0-9]+)*(?:[A-Za-z0-9.+-]*)?", version) is None:
    raise SystemExit(f"{path}: invalid package version {version!r}")
# True only when `commit` IS the release tag commit for `version`. A pin that
# advances to a post-release master commit keeps the same `version` string (the
# runtime self-reports it until the next bump), so the version alone cannot say
# whether this runtime was ever released -- that is what this flag records, and
# what stops a rung table from crediting an unreleased runtime.
if not isinstance(pin["version_is_release"], bool):
    raise SystemExit(f"{path}: version_is_release must be a JSON boolean")
print(pin["schema"], repo, commit, version, sep="\t")
PY
)"; then
        _gridbook_runtime_error "invalid pin file $_GRIDBOOK_RUNTIME_PIN_FILE"
        return
    fi

    IFS=$'\t' read -r GRIDBOOK_RUNTIME_PIN_SCHEMA \
        GRIDBOOK_RUNTIME_REPOSITORY GRIDBOOK_RUNTIME_COMMIT \
        GRIDBOOK_RUNTIME_VERSION <<<"$values"
    if ! _gridbook_runtime_is_commit "$GRIDBOOK_RUNTIME_COMMIT"; then
        _gridbook_runtime_error "pin parser did not return a full commit"
        return
    fi
    export GRIDBOOK_RUNTIME_PIN_SCHEMA GRIDBOOK_RUNTIME_REPOSITORY \
        GRIDBOOK_RUNTIME_COMMIT GRIDBOOK_RUNTIME_VERSION
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
        gridbook_runtime_verify_checkout "$destination" \
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
    gridbook_runtime_verify_checkout "$tmp" "$GRIDBOOK_RUNTIME_COMMIT" \
        "$GRIDBOOK_RUNTIME_VERSION" >/dev/null

    if mv -T -- "$tmp" "$destination" 2>/dev/null; then
        tmp=""
    elif [[ ! -e "$destination" ]]; then
        _gridbook_runtime_error \
            "could not publish fetched checkout at $destination"
        exit 2
    fi
    gridbook_runtime_verify_checkout "$destination" \
        "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION"
)

gridbook_runtime_prepare() {
    local checkout_override=${GRIDBOOK_RUNTIME_CHECKOUT:-}
    local source container_source contract_source container_contract

    gridbook_runtime_load_pin || return
    if [[ -n "$checkout_override" ]]; then
        source="$(gridbook_runtime_verify_checkout "$checkout_override" \
            "$GRIDBOOK_RUNTIME_COMMIT" "$GRIDBOOK_RUNTIME_VERSION")" || return
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
        --volume "${source}:${container_source}:ro"
        --volume "${contract_source}:${container_contract}:ro"
        --env "PQ_GRIDBOOK_RUNTIME_SOURCE=${container_source}"
        --env "PQ_GRIDBOOK_RUNTIME_COMMIT=${GRIDBOOK_RUNTIME_COMMIT}"
        --env "PQ_GRIDBOOK_RUNTIME_VERSION=${GRIDBOOK_RUNTIME_VERSION}"
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
    local install_source

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

    gridbook_runtime_verify_checkout "$source" "$GRIDBOOK_RUNTIME_COMMIT" \
        "$GRIDBOOK_RUNTIME_VERSION" >/dev/null || return
    install_source=$(gridbook_runtime_container_install_target) || return
    if [[ -e "$install_source" ]]; then
        _gridbook_runtime_error \
            "container install target already exists: $install_source"
        return
    fi
    mkdir -p -- "$install_source"
    cp -a -- "$source/." "$install_source/"
    chmod -R u+w -- "$install_source"

    python3 -m pip install --no-deps --no-cache-dir --quiet "$install_source"
    python3 - "$GRIDBOOK_RUNTIME_VERSION" <<'PY'
from importlib.metadata import version
import sys

expected = sys.argv[1]
actual = version("gridbook")
if actual != expected:
    raise SystemExit(f"installed gridbook {actual!r}, expected {expected!r}")
print(f"gridbook-runtime: installed gridbook {actual}")
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
