"""Strict, torch-free reader for PrismaQuant's Tessera serving-runtime pin.

Tessera serves ITSELF.  Since 2026-09-02 the Tessera repository ships its own
vLLM plugin -- package ``tessera.serving``, entry point
``tessera = "tessera.serving:register"`` under
``[project.entry-points."vllm.general_plugins"]``, registering
``quant_method = "tessera"`` -- so the runtime that reads Tessera bytes is
Tessera's, not Gridbook's.  Gridbook's Tessera lane is withdrawn; its contract
v14, which carried the two Tessera rows, was never released, so nothing that
shipped is broken by the move.  This module is the producer side of the new
boundary and it is deliberately the same shape as
``gridbook_serving_runtime_pin.py``: PrismaQuant never vendors or imports the
serving runtime's *serving* half, and compatibility crosses the repository
boundary through exactly two objects -- this immutable pin, and the contract
the runtime packages (``tessera/serving/runtime_contract.json``, read through
``importlib.resources``).

**ADMISSION IS FAIL-CLOSED TODAY, BY THIS PIN, NOT BY AN EDIT.**  There is no
Tessera release tag: cutting one is Rob's decision, not an agent's.  So the
tracked pin's ``version`` and ``commit`` are conspicuous PENDING sentinels,
``version_is_release`` is ``false``, and
:func:`require_exact_tessera_runtime_release` REFUSES them.
``tessera_render.tessera_lane_attested`` ANDs that refusal into its answer, so
every Tessera rung is producer-ineligible even though the packaged contract
publishes ``device_qualified`` cells for both families and the ``tessera``
package is importable in this environment.  That is the point: without the pin
conjunct, admission would flip to True the moment somebody put the Tessera
source tree on ``PYTHONPATH``, which is the opposite of a reviewed release
boundary.

**Cutting the release is ONE reviewed change, not three.**  Following the
Gridbook discipline (a pin whose schema, version and commit cannot move by
halves), resolving the tag means editing, in a single commit: the JSON pin's
``commit``/``version``/``version_is_release``, AND the two module constants
:data:`TESSERA_SERVING_RUNTIME_RELEASE_VERSION` /
:data:`TESSERA_SERVING_RUNTIME_RELEASE_COMMIT` below.  The reader requires the
pin to equal the constants, so a JSON edit alone cannot admit anything and a
constant edit alone cannot either.

**No wheel digest, and why.**  Gridbook's serving pin additionally binds an
exact reviewed wheel SHA-256, because Gridbook is installed into a serving
container from a published archive.  Tessera's plugin is installed from a
source checkout (``pip install --no-deps --no-build-isolation -e <tessera>``)
and publishes no wheel, so asserting a digest here would be a claim about an
artifact that does not exist.  The commit is the binding fact and it is the one
this pin carries; when Tessera starts publishing wheels, a ``wheel_sha256``
member is added here and to the JSON in the same reviewed commit.

**The repository is local-only today.**  ``repository`` names the origin the
release will be cut from; the tree lives at ``/home/rob/tessera`` and has not
been pushed.  The field is the reviewed identity of the runtime, not a
reachability claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any


TESSERA_SERVING_RUNTIME_PIN_SCHEMA = (
    "prismaquant.tessera_serving_runtime_pin.v1"
)
TESSERA_SERVING_RUNTIME_REPOSITORY = (
    "https://github.com/RobTand/tessera.git"
)

#: The contract schema this pin binds.  ``parse_tessera_serving_runtime_pin``
#: refuses a pin naming any other, so a runtime that moves its contract schema
#: cannot be pinned by halves -- the same "three are one change" rule the
#: Gridbook pins carry.
TESSERA_SERVING_RUNTIME_CONTRACT_SCHEMA = "tessera.runtime-contract.v1"

#: The conspicuous sentinels a pin carries while no release tag exists.  They
#: are structurally ACCEPTED by the parser -- so the pin file is reviewable,
#: and so this module has something honest to say -- and REFUSED by every live
#: admission gate through :func:`require_exact_tessera_runtime_release`.
TESSERA_SERVING_RUNTIME_COMMIT_PENDING = "PENDING_TESSERA_RELEASE_COMMIT"
TESSERA_SERVING_RUNTIME_VERSION_PENDING = "PENDING_TESSERA_RELEASE_VERSION"

#: The exact reviewed release.  Both are the PENDING sentinels because there is
#: no Tessera release tag and creating one is Rob's call.  While they hold
#: these values ``require_exact_tessera_runtime_release`` can never pass: the
#: sentinel is not a resolved commit, and no resolved commit equals it.  The
#: gate is therefore closed twice over, which is deliberate.
TESSERA_SERVING_RUNTIME_RELEASE_VERSION = TESSERA_SERVING_RUNTIME_VERSION_PENDING
TESSERA_SERVING_RUNTIME_RELEASE_COMMIT = TESSERA_SERVING_RUNTIME_COMMIT_PENDING

#: The vLLM plugin entry-point name the released runtime registers.  It is the
#: value every packaged eligibility cell publishes in ``requires_plugin``, and
#: ``tessera_render.tessera_lane_attested`` refuses a cell that claims a route
#: without naming it.
TESSERA_SERVING_PLUGIN_NAME = "tessera"

#: The ``quantization_config.quant_method`` value that selects the plugin.
#: There is no enable flag; the checkpoint selects the runtime.
TESSERA_SERVING_QUANT_METHOD = "tessera"

#: The one operator knob the plugin declares.  Named here because a serve
#: command that omits it is serving a different residency than the pin's
#: receipts covered.
TESSERA_SERVING_RESIDENCY_ENV = "TESSERA_SERVE_MODE"

_REQUIRED_MEMBERS = {
    "schema",
    "repository",
    "commit",
    "version",
    "version_is_release",
    "runtime_contract_schema",
    "plugin_entry_point",
    "serving_extension_basenames",
}
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_VERSION_RE = re.compile(r"[0-9]+(?:[.][0-9]+)*(?:[A-Za-z0-9.+-]*)?")
#: A JIT extension basename is a Python module name, and the loaded `.so` is
#: that name plus a build identity suffix, so the pin declares the PREFIX.
_EXTENSION_BASENAME_RE = re.compile(r"[a-z][a-z0-9_]*")


class TesseraServingRuntimePinError(ValueError):
    """The Tessera serving pin is missing, pending, or malformed."""


@dataclass(frozen=True)
class TesseraServingRuntimePin:
    schema: str
    repository: str
    commit: str
    version: str
    version_is_release: bool
    runtime_contract_schema: str
    plugin_entry_point: str
    #: Basename PREFIXES of the CUDA extensions the released plugin loads into
    #: a serving process, e.g. ``("tessera_nvfp4",)`` for the span-2 NVFP4
    #: decoder that ``tessera.serving.ext`` JIT-builds as
    #: ``tessera_nvfp4_<build identity>.so``.  This is the reproducibility
    #: contract's half of the pin, not the admission half: §7.4 says an A/B's
    #: arms must have identical extension residency, and a lane whose `.so` no
    #: fingerprint pattern matches reports "nothing resident" -- a serve
    #: running Tessera's own native decode looking exactly like a stock serve.
    #: ``tools/serve_fingerprint.py`` is stdlib-only and cannot read this file
    #: from inside a serving container, so it carries the same tuple and
    #: ``tests/test_tessera_serve_fingerprint.py`` refuses any disagreement.
    serving_extension_basenames: tuple[str, ...]

    @property
    def commit_is_resolved(self) -> bool:
        return _FULL_COMMIT_RE.fullmatch(self.commit) is not None

    @property
    def version_is_resolved(self) -> bool:
        return (self.version != TESSERA_SERVING_RUNTIME_VERSION_PENDING
                and _VERSION_RE.fullmatch(self.version) is not None)


def parse_tessera_serving_runtime_pin(
    payload: Mapping[str, Any],
    *,
    where: str = "tessera_serving_runtime_pin.json",
) -> TesseraServingRuntimePin:
    """Structural read of a pin payload.  Accepts the PENDING sentinels.

    Accepting a pending pin *structurally* is what lets the file be reviewed
    before a tag exists.  It admits nothing: only
    :func:`require_exact_tessera_runtime_release` is a gate, and it refuses
    every sentinel.
    """
    if not isinstance(payload, Mapping) or set(payload) != _REQUIRED_MEMBERS:
        observed = sorted(payload) if isinstance(payload, Mapping) else []
        raise TesseraServingRuntimePinError(
            f"{where}: expected exactly {sorted(_REQUIRED_MEMBERS)}, "
            f"got {observed}"
        )
    if payload["schema"] != TESSERA_SERVING_RUNTIME_PIN_SCHEMA:
        raise TesseraServingRuntimePinError(
            f"{where}: unsupported schema {payload['schema']!r}"
        )
    if payload["repository"] != TESSERA_SERVING_RUNTIME_REPOSITORY:
        raise TesseraServingRuntimePinError(
            f"{where}: repository differs from the reviewed Tessera origin"
        )
    commit = payload["commit"]
    if not isinstance(commit, str) or (
        _FULL_COMMIT_RE.fullmatch(commit) is None
        and commit != TESSERA_SERVING_RUNTIME_COMMIT_PENDING
    ):
        raise TesseraServingRuntimePinError(
            f"{where}: commit must be a full lowercase SHA or the exact "
            f"pending sentinel {TESSERA_SERVING_RUNTIME_COMMIT_PENDING!r}"
        )
    version = payload["version"]
    if not isinstance(version, str) or (
        _VERSION_RE.fullmatch(version) is None
        and version != TESSERA_SERVING_RUNTIME_VERSION_PENDING
    ):
        raise TesseraServingRuntimePinError(
            f"{where}: version must be a release version or the exact "
            f"pending sentinel {TESSERA_SERVING_RUNTIME_VERSION_PENDING!r}"
        )
    released = payload["version_is_release"]
    if not isinstance(released, bool):
        raise TesseraServingRuntimePinError(
            f"{where}: version_is_release must be a JSON boolean"
        )
    if released and (commit == TESSERA_SERVING_RUNTIME_COMMIT_PENDING
                     or version == TESSERA_SERVING_RUNTIME_VERSION_PENDING):
        raise TesseraServingRuntimePinError(
            f"{where}: a pending commit/version cannot be marked released"
        )
    if payload["runtime_contract_schema"] != (
        TESSERA_SERVING_RUNTIME_CONTRACT_SCHEMA
    ):
        raise TesseraServingRuntimePinError(
            f"{where}: serving runtime contract must be "
            f"{TESSERA_SERVING_RUNTIME_CONTRACT_SCHEMA}"
        )
    entry_point = payload["plugin_entry_point"]
    if not isinstance(entry_point, str) or not entry_point.startswith(
        f"{TESSERA_SERVING_PLUGIN_NAME} = "
    ):
        raise TesseraServingRuntimePinError(
            f"{where}: plugin_entry_point must name the "
            f"{TESSERA_SERVING_PLUGIN_NAME!r} vllm.general_plugins entry "
            "point the released runtime registers"
        )
    basenames = payload["serving_extension_basenames"]
    if (not isinstance(basenames, (list, tuple)) or not basenames
            or not all(isinstance(name, str)
                       and _EXTENSION_BASENAME_RE.fullmatch(name)
                       for name in basenames)):
        raise TesseraServingRuntimePinError(
            f"{where}: serving_extension_basenames must be a non-empty list "
            "of lowercase module-name prefixes naming the CUDA extensions the "
            "released plugin loads into a serving process"
        )
    return TesseraServingRuntimePin(
        schema=str(payload["schema"]),
        repository=str(payload["repository"]),
        commit=commit,
        version=version,
        version_is_release=released,
        runtime_contract_schema=str(payload["runtime_contract_schema"]),
        plugin_entry_point=entry_point,
        serving_extension_basenames=tuple(str(n) for n in basenames),
    )


def _reject_duplicate_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TesseraServingRuntimePinError(
                f"tessera_serving_runtime_pin.json: duplicate member {key!r}"
            )
        result[key] = value
    return result


def tessera_serving_runtime_pin_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "tessera_runtime"
        / "tessera_serving_runtime_pin.json"
    )


def _read_pin(location: Path) -> TesseraServingRuntimePin:
    try:
        payload = json.loads(
            location.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_members,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TesseraServingRuntimePinError(
            f"{location}: cannot read serving pin: {exc}"
        ) from exc
    return parse_tessera_serving_runtime_pin(payload, where=str(location))


@lru_cache(maxsize=1)
def _load_tracked_pin() -> TesseraServingRuntimePin:
    return _read_pin(tessera_serving_runtime_pin_path())


def load_tessera_serving_runtime_pin(
    path: Path | str | None = None,
) -> TesseraServingRuntimePin:
    """The tracked pin, or an explicit one.

    ``path`` exists for tests and for reviewing a candidate pin file; the
    tracked pin is the only one any gate reads, and it is cached because it is
    immutable within a process.
    """
    if path is None:
        return _load_tracked_pin()
    return _read_pin(Path(path))


def require_exact_tessera_runtime_release(
    pin: TesseraServingRuntimePin,
) -> None:
    """Refuse anything that is not the exact reviewed Tessera release.

    Today it refuses everything, because both reviewed-release constants are
    the PENDING sentinels: there is no Tessera release tag.  That refusal IS
    the current admission answer -- ``tessera_lane_attested`` is False by this
    function, not by a hand-typed constant somewhere -- and it lifts when Rob
    cuts a tag and one reviewed commit resolves the JSON and the constants
    together.
    """
    if not pin.commit_is_resolved or not pin.version_is_resolved:
        raise TesseraServingRuntimePinError(
            "Tessera serving runtime commit/version is still PENDING: no "
            "Tessera release tag exists, so no Tessera route may be admitted. "
            f"pin.version={pin.version!r} pin.commit={pin.commit!r}"
        )
    if (
        pin.commit != TESSERA_SERVING_RUNTIME_RELEASE_COMMIT
        or pin.version != TESSERA_SERVING_RUNTIME_RELEASE_VERSION
        or pin.version_is_release is not True
    ):
        raise TesseraServingRuntimePinError(
            "Tessera serving runtime differs from the exact reviewed release "
            f"({TESSERA_SERVING_RUNTIME_RELEASE_VERSION!r} / "
            f"{TESSERA_SERVING_RUNTIME_RELEASE_COMMIT!r}); resolve the pin "
            "file and the module constants in one reviewed commit"
        )


__all__ = [
    "TESSERA_SERVING_PLUGIN_NAME",
    "TESSERA_SERVING_QUANT_METHOD",
    "TESSERA_SERVING_RESIDENCY_ENV",
    "TESSERA_SERVING_RUNTIME_COMMIT_PENDING",
    "TESSERA_SERVING_RUNTIME_CONTRACT_SCHEMA",
    "TESSERA_SERVING_RUNTIME_PIN_SCHEMA",
    "TESSERA_SERVING_RUNTIME_RELEASE_COMMIT",
    "TESSERA_SERVING_RUNTIME_RELEASE_VERSION",
    "TESSERA_SERVING_RUNTIME_REPOSITORY",
    "TESSERA_SERVING_RUNTIME_VERSION_PENDING",
    "TesseraServingRuntimePin",
    "TesseraServingRuntimePinError",
    "load_tessera_serving_runtime_pin",
    "parse_tessera_serving_runtime_pin",
    "require_exact_tessera_runtime_release",
    "tessera_serving_runtime_pin_path",
]
