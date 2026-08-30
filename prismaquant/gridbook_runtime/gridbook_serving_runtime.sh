#!/usr/bin/env bash
# Resolve and install the exact published Gridbook serving wheel.
#
# This is deliberately separate from gridbook_runtime.sh.  The latter is the
# producer/handoff environment pin (0.9.1/v12 since 2026-08-30, held in
# lockstep with this one by tests/test_gridbook_runtime_boundary.py); serving
# additionally binds the independently reviewed release wheel and its
# published SHA-256, which the producer pin does not carry.

_GRIDBOOK_SERVING_ASSET_DIR="$({
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
})"
_GRIDBOOK_SERVING_PIN_FILE="${_GRIDBOOK_SERVING_ASSET_DIR}/gridbook_serving_runtime_pin.json"

_gridbook_serving_error() {
    printf 'gridbook-serving-runtime: ERROR: %s\n' "$*" >&2
    return 2
}

gridbook_serving_runtime_load_pin() {
    local values
    if ! values="$(python3 - "$_GRIDBOOK_SERVING_PIN_FILE" <<'PY'
import json
import re
import sys

path = sys.argv[1]
def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate member {key!r}")
        result[key] = value
    return result

with open(path, encoding="utf-8") as handle:
    pin = json.load(handle, object_pairs_hook=strict_object)
required = {
    "schema", "repository", "commit", "version", "version_is_release",
    "wheel_sha256", "runtime_contract_schema", "required_abi_features",
}
if set(pin) != required:
    raise SystemExit(f"{path}: pin members differ")
if pin["schema"] != "prismaquant.gridbook_serving_runtime_pin.v1":
    raise SystemExit(f"{path}: unsupported serving-pin schema")
if pin["repository"] != "https://github.com/RobTand/gridbook.git":
    raise SystemExit(f"{path}: unreviewed repository")
if pin["version"] != "0.9.1" or pin["version_is_release"] is not True:
    raise SystemExit(f"{path}: Gridbook 0.9.1 is not marked released")
if re.fullmatch(r"[0-9a-f]{40}", str(pin["commit"])) is None:
    raise SystemExit(f"{path}: release commit remains unresolved")
if re.fullmatch(r"[0-9a-f]{64}", str(pin["wheel_sha256"])) is None:
    raise SystemExit(f"{path}: published wheel digest remains unresolved")
if pin["runtime_contract_schema"] != "gridbook.runtime-contract.v12":
    raise SystemExit(f"{path}: runtime contract is not v12")
features = pin["required_abi_features"]
expected = {
    "routed_moe_per_role_codebook_lut": 1,
    "source_fp8_block128_w8a16": 1,
    "dspark_construction_physical_bridge": 1,
}
if features != expected or any(type(value) is not int for value in features.values()):
    raise SystemExit(f"{path}: ABI feature closure differs")
print(
    pin["schema"], pin["repository"], pin["commit"], pin["version"],
    pin["wheel_sha256"], pin["runtime_contract_schema"],
    features["routed_moe_per_role_codebook_lut"],
    features["source_fp8_block128_w8a16"],
    features["dspark_construction_physical_bridge"],
    sep="\t",
)
PY
)"; then
        _gridbook_serving_error "invalid or pending serving pin $_GRIDBOOK_SERVING_PIN_FILE"
        return
    fi
    IFS=$'\t' read -r GRIDBOOK_RUNTIME_PIN_SCHEMA \
        GRIDBOOK_RUNTIME_REPOSITORY GRIDBOOK_RUNTIME_COMMIT \
        GRIDBOOK_RUNTIME_VERSION GRIDBOOK_RUNTIME_WHEEL_SHA256 \
        GRIDBOOK_RUNTIME_CONTRACT_SCHEMA \
        GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT \
        GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16 \
        GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE \
        <<<"$values"
    export GRIDBOOK_RUNTIME_PIN_SCHEMA GRIDBOOK_RUNTIME_REPOSITORY \
        GRIDBOOK_RUNTIME_COMMIT GRIDBOOK_RUNTIME_VERSION \
        GRIDBOOK_RUNTIME_WHEEL_SHA256 GRIDBOOK_RUNTIME_CONTRACT_SCHEMA \
        GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT \
        GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16 \
        GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE
}

_gridbook_serving_cache_root() {
    local root
    if [[ -n "${GRIDBOOK_SERVING_RUNTIME_CACHE_DIR:-}" ]]; then
        root=$GRIDBOOK_SERVING_RUNTIME_CACHE_DIR
    elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
        root="${XDG_CACHE_HOME}/prismaquant/gridbook-serving-runtime"
    elif [[ -n "${HOME:-}" ]]; then
        root="${HOME}/.cache/prismaquant/gridbook-serving-runtime"
    else
        _gridbook_serving_error "set GRIDBOOK_SERVING_RUNTIME_CACHE_DIR"
        return
    fi
    if [[ -z "$root" || "$root" == / ]]; then
        _gridbook_serving_error "unsafe serving-wheel cache root"
        return
    fi
    printf '%s\n' "$root"
}

gridbook_serving_runtime_verify_wheel() {
    local wheel=${1:-}
    python3 - "$wheel" "$GRIDBOOK_RUNTIME_VERSION" \
        "$GRIDBOOK_RUNTIME_WHEEL_SHA256" <<'PY'
from email.parser import BytesParser
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

path = Path(sys.argv[1])
version = sys.argv[2]
expected_sha = sys.argv[3]
if not path.is_file() or path.is_symlink():
    raise SystemExit(f"wheel is missing, non-regular, or a symlink: {path}")
if re.fullmatch(rf"gridbook-{re.escape(version)}-[A-Za-z0-9_.+-]+[.]whl", path.name) is None:
    raise SystemExit(f"wheel filename does not identify gridbook {version}: {path.name}")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != expected_sha:
    raise SystemExit(f"wheel SHA-256 {digest} differs from pin {expected_sha}")
with zipfile.ZipFile(path) as archive:
    infos = archive.infolist()
    names = [item.filename for item in infos]
    if len(names) != len(set(names)):
        raise SystemExit("wheel contains duplicate archive names")
    for info in infos:
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise SystemExit(f"wheel contains unsafe path {info.filename!r}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise SystemExit(f"wheel contains symlink {info.filename!r}")
    metadata_names = [
        name for name in names
        if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_names) != 1:
        raise SystemExit("wheel does not contain exactly one METADATA")
    metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    if metadata.get("Name", "").lower() != "gridbook" or metadata.get("Version") != version:
        raise SystemExit("wheel METADATA name/version differs")
print(path.resolve(strict=True))
PY
}

_gridbook_serving_materialize_wheel() (
    set -euo pipefail
    local cache_root destination supplied tmp="" wheel
    local -a existing=() published=()
    cache_root="$(_gridbook_serving_cache_root)"
    mkdir -p -- "$cache_root"
    cache_root="$(cd -- "$cache_root" && pwd -P)"
    destination="${cache_root}/${GRIDBOOK_RUNTIME_WHEEL_SHA256}"
    if [[ -d "$destination" ]]; then
        mapfile -t existing < <(find "$destination" -maxdepth 1 -type f -name '*.whl' -print)
        if [[ ${#existing[@]} -ne 1 ]]; then
            _gridbook_serving_error "cached digest directory does not contain one wheel"
            exit 2
        fi
        if ! gridbook_serving_runtime_verify_wheel "${existing[0]}"; then
            # Name the cache explicitly.  A digest-named directory holding the
            # wrong wheel is unrecoverable through this path -- the branch
            # above short-circuits before any download or supplied wheel is
            # considered -- so an operator who is only told "SHA-256 X differs
            # from pin Y" has no way to know the CACHE is what must be
            # removed, and will keep re-supplying a correct wheel that is
            # never read.
            _gridbook_serving_error \
                "cached wheel does not match the pin; remove $destination and retry"
            exit 2
        fi
        exit
    fi
    tmp="$(mktemp -d "${cache_root}/.wheel-${GRIDBOOK_RUNTIME_WHEEL_SHA256:0:12}.XXXXXX")"
    cleanup() {
        if [[ -n "$tmp" && -d "$tmp" ]]; then
            rm -rf -- "$tmp"
        fi
    }
    trap cleanup EXIT
    supplied=${GRIDBOOK_SERVING_RUNTIME_WHEEL:-}
    if [[ -n "$supplied" ]]; then
        # Test the status explicitly rather than leaning on errexit; see the
        # pre-publish guard below for why `set -e` is inert on this call path.
        # Without this, a rejected wheel leaves "$wheel" empty and the failure
        # surfaces as a confusing `cp` error against an empty operand.
        if ! wheel="$(gridbook_serving_runtime_verify_wheel "$supplied")"; then
            _gridbook_serving_error \
                "supplied wheel does not match the pin: $supplied"
            exit 2
        fi
        cp -- "$wheel" "$tmp/$(basename -- "$wheel")"
    else
        python3 -m pip download --quiet --disable-pip-version-check --no-deps \
            --only-binary=:all: --dest "$tmp" \
            "gridbook==${GRIDBOOK_RUNTIME_VERSION}"
    fi
    mapfile -t published < <(find "$tmp" -maxdepth 1 -type f -name '*.whl' -print)
    if [[ ${#published[@]} -ne 1 ]]; then
        _gridbook_serving_error "download did not yield exactly one wheel"
        exit 2
    fi
    # Never publish an unverified wheel into the digest-named cache.
    #
    # This MUST test the status explicitly.  The only caller reaches this
    # function as `wheel="$(_gridbook_serving_materialize_wheel)" || return`,
    # and Bash disables errexit for a command substitution whose enclosing
    # command is part of a `||` list -- re-arming `set -e` inside the subshell
    # does not restore it.  The `set -euo pipefail` above is therefore inert
    # here, so a bare call would print its rejection and then fall through to
    # the `mv`, permanently caching a wheel the pin rejects: every later
    # invocation takes the fast path above, finds the bad wheel and refuses,
    # and no supplied wheel or download is ever consulted again.  One attempt
    # without a matching wheel would brick the serving lane on that machine.
    # Observed 2026-08-14: the PyPI 0.8.6 wheel (content-identical to the
    # pinned artifact but a different archive) was cached under the pinned
    # digest and refused the lane until the directory was removed by hand.
    # The 0.8.8 pin cannot reproduce that specific collision -- its image
    # was built FROM the published archive, so the PyPI wheel and the pin
    # agree -- but the fall-through defect this guards is version-agnostic.
    if ! gridbook_serving_runtime_verify_wheel "${published[0]}" >/dev/null; then
        _gridbook_serving_error \
            "materialized wheel does not match the pin; refusing to cache it"
        exit 2
    fi
    if mv -T -- "$tmp" "$destination" 2>/dev/null; then
        tmp=""
    elif [[ ! -d "$destination" ]]; then
        _gridbook_serving_error "could not publish serving wheel cache"
        exit 2
    fi
    mapfile -t published < <(find "$destination" -maxdepth 1 -type f -name '*.whl' -print)
    [[ ${#published[@]} -eq 1 ]] || exit 2
    gridbook_serving_runtime_verify_wheel "${published[0]}"
)

gridbook_serving_runtime_prepare() {
    local wheel contract_source container_contract container_wheel
    gridbook_serving_runtime_load_pin || return
    wheel="$(_gridbook_serving_materialize_wheel)" || return
    contract_source=$_GRIDBOOK_SERVING_ASSET_DIR
    container_contract=/opt/prismaquant-gridbook-serving-contract
    container_wheel="/opt/prismaquant-gridbook-serving-wheel/$(basename -- "$wheel")"
    GRIDBOOK_RUNTIME_CONTAINER_HELPER="${container_contract}/gridbook_serving_runtime.sh"
    export GRIDBOOK_RUNTIME_CONTAINER_HELPER
    GRIDBOOK_SERVING_RUNTIME_DOCKER_ARGS=(
        --workdir /
        --volume "${contract_source}:${container_contract}:ro"
        --volume "${wheel}:${container_wheel}:ro"
        --env "PYTHONSAFEPATH=1"
        # See gridbook_runtime.sh for the full rationale: vLLM's EngineCore
        # renames itself with setproctitle, which on Linux overwrites the
        # argv+envp block and destroys /proc/<pid>/environ -- the source every
        # serve attestation reads.  SPT_NOENV confines the title to the argv
        # area so the census can actually read the environment it compares.
        --env "SPT_NOENV=1"
        --env "PQ_GRIDBOOK_RUNTIME_COMMIT=${GRIDBOOK_RUNTIME_COMMIT}"
        --env "PQ_GRIDBOOK_RUNTIME_VERSION=${GRIDBOOK_RUNTIME_VERSION}"
        --env "PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256=${GRIDBOOK_RUNTIME_WHEEL_SHA256}"
        --env "PQ_GRIDBOOK_RUNTIME_CONTRACT_SCHEMA=${GRIDBOOK_RUNTIME_CONTRACT_SCHEMA}"
        --env "PQ_GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT=${GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT}"
        --env "PQ_GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16=${GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16}"
        --env "PQ_GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE=${GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE}"
        --env "PQ_GRIDBOOK_RUNTIME_WHEEL=${container_wheel}"
        --env "PQ_GRIDBOOK_RUNTIME_HELPER=${GRIDBOOK_RUNTIME_CONTAINER_HELPER}"
    )
    printf 'gridbook-serving-runtime: wheel %s (%s); release %s\n' \
        "$wheel" "$GRIDBOOK_RUNTIME_WHEEL_SHA256" "$GRIDBOOK_RUNTIME_COMMIT" >&2
}

gridbook_serving_runtime_install_container() {
    local supplied_commit=${PQ_GRIDBOOK_RUNTIME_COMMIT:-}
    local supplied_version=${PQ_GRIDBOOK_RUNTIME_VERSION:-}
    local supplied_sha=${PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256:-}
    local wheel=${PQ_GRIDBOOK_RUNTIME_WHEEL:-}
    export PYTHONSAFEPATH=1
    gridbook_serving_runtime_load_pin || return
    if [[ "$supplied_commit" != "$GRIDBOOK_RUNTIME_COMMIT" \
          || "$supplied_version" != "$GRIDBOOK_RUNTIME_VERSION" \
          || "$supplied_sha" != "$GRIDBOOK_RUNTIME_WHEEL_SHA256" ]]; then
        _gridbook_serving_error "container transport pin differs from tracked release"
        return
    fi
    gridbook_serving_runtime_verify_wheel "$wheel" >/dev/null || return
    python3 -m pip install --disable-pip-version-check --no-deps --no-index \
        --no-cache-dir --force-reinstall "$wheel"
    python3 - "$GRIDBOOK_RUNTIME_VERSION" \
        "$GRIDBOOK_RUNTIME_WHEEL_SHA256" \
        "$GRIDBOOK_RUNTIME_CONTRACT_SCHEMA" \
        "$GRIDBOOK_RUNTIME_FEATURE_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT" \
        "$GRIDBOOK_RUNTIME_FEATURE_SOURCE_FP8_BLOCK128_W8A16" \
        "$GRIDBOOK_RUNTIME_FEATURE_DSPARK_CONSTRUCTION_PHYSICAL_BRIDGE" <<'PY'
from importlib.metadata import distribution, version
import json
from pathlib import Path
import sys

expected_version, expected_sha, expected_schema = sys.argv[1:4]
expected_features = {
    "routed_moe_per_role_codebook_lut": int(sys.argv[4]),
    "source_fp8_block128_w8a16": int(sys.argv[5]),
    "dspark_construction_physical_bridge": int(sys.argv[6]),
}
if version("gridbook") != expected_version:
    raise SystemExit("installed Gridbook version differs")
dist = distribution("gridbook")
direct = [item for item in (dist.files or ()) if item.name == "direct_url.json"]
if len(direct) != 1:
    raise SystemExit("installed Gridbook has no unique PEP 610 identity")
payload = json.loads(Path(dist.locate_file(direct[0])).read_text(encoding="utf-8"))
archive = payload.get("archive_info") or {}
if archive.get("hashes") != {"sha256": expected_sha}:
    raise SystemExit("installed Gridbook PEP 610 wheel digest differs")
from gridbook.runtime_contract import load_runtime_contract
contract = load_runtime_contract()
if contract.get("schema") != expected_schema:
    raise SystemExit("installed Gridbook runtime contract differs")
features = contract.get("abi_features") or {}
if any(features.get(name) != value for name, value in expected_features.items()):
    raise SystemExit("installed Gridbook ABI feature closure differs")
PY
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        install-container) gridbook_serving_runtime_install_container ;;
        *) _gridbook_serving_error "usage: $0 install-container" ; exit 2 ;;
    esac
fi
