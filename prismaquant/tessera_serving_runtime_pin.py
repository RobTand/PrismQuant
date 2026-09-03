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

**THE EXTENSION TABLE IS TRANSCRIBED, AND REFUSED AGAINST ITS SOURCE.**
``serving_native_extensions`` is not this repository's opinion about which
CUDA extensions the Tessera plugin loads: since Tessera contract v7 the
runtime publishes that itself, in ``native_extensions``, as a
``module_name_prefix``/``filename_glob``/``match`` triple.  The pin transcribes
it because ``tools/serve_fingerprint.py`` runs inside a serving container from
a five-file bootstrap and can read neither the contract nor this file, and
``tessera_runtime_contract.require_pin_native_extensions_match_contract``
refuses a transcription that is not the pinned contract's table.  So the chain
is contract -> pin -> fingerprint with a refusal at each link, instead of a
refusal at the last link only.  It used to be the last link only, and the
hand-written value was already wrong by one character: the pin said
``"tessera_nvfp4"`` where the load path's constant is ``"tessera_nvfp4_"``.

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
import fnmatch
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

#: The rule the runtime names for turning a ``filename_glob`` into a
#: decision: fnmatch the glob against the BASENAME of a mapped ``.so``.  It is
#: a VALUE in the contract (``native_extensions[].match``) rather than prose,
#: because a consumer cannot otherwise tell a stem from a prefix from a
#: pattern -- and a substring search over the whole mapped path, which is what
#: this repository used to do, is a different predicate that matches
#: ``/x/tessera_nvfp4/unrelated.so`` too.
MATCH_BASENAME_FNMATCH = "basename_fnmatch"

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
    "serving_native_extensions",
}
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_VERSION_RE = re.compile(r"[0-9]+(?:[.][0-9]+)*(?:[A-Za-z0-9.+-]*)?")

#: The exact keys one ``serving_native_extensions`` row carries, spelled the
#: way the runtime's ``native_extensions`` table spells them.  Verbatim, so the
#: contract-vs-pin refusal in ``tessera_runtime_contract`` is a dict
#: comparison over the same field names rather than a re-mapping that could
#: itself be the drift.
_NATIVE_EXTENSION_MEMBERS = {
    "module_name_prefix",
    "filename_glob",
    "match",
}

#: A JIT extension module name is a Python identifier and the loaded `.so` is
#: that name plus a build-identity suffix, so what the pin declares is the
#: PREFIX -- including its trailing separator, exactly as the runtime
#: publishes it.
_EXTENSION_PREFIX_RE = re.compile(r"[a-z][a-z0-9_]*")


class TesseraServingRuntimePinError(ValueError):
    """The Tessera serving pin is missing, pending, or malformed."""


@dataclass(frozen=True)
class TesseraServingNativeExtension:
    """One row of the runtime's ``native_extensions`` table, transcribed.

    Three fields and no fourth, because these three are what a residency
    decision is made of: WHICH module the load path builds
    (``module_name_prefix``), WHICH filename that produces
    (``filename_glob``), and WHICH RULE turns the glob into a decision
    (``match``).  ``source``/``loaded_by``/``routes``/``when_unavailable`` are
    published too and are read from the contract where they are needed; they
    are not transcribed here, because a pin field nothing reads is a field
    nothing keeps honest.
    """

    #: The constant Tessera's JIT load path itself passes to
    #: ``cpp_extension.load``, e.g. ``"tessera_nvfp4_"``.  It carries its
    #: trailing separator: the module name is the prefix plus a build-identity
    #: hash, so the prefix is not a stem and trimming it would be a
    #: transcription that changed the value.
    module_name_prefix: str
    #: The library name that produces, e.g. ``"tessera_nvfp4_*.so"``.  There is
    #: no exact basename to pin -- the build identity is in the name.
    filename_glob: str
    #: The rule a consumer applies, named by the runtime rather than guessed by
    #: the consumer: ``"basename_fnmatch"`` means fnmatch the glob against the
    #: BASENAME of a mapped ``.so``.  A substring search over the whole mapped
    #: path is a DIFFERENT predicate and only one of them is the runtime's.
    match: str

    def as_contract_row(self) -> dict:
        """The row as the contract spells it, for a field-level comparison."""
        return {
            "module_name_prefix": self.module_name_prefix,
            "filename_glob": self.filename_glob,
            "match": self.match,
        }


@dataclass(frozen=True)
class TesseraServingRuntimePin:
    schema: str
    repository: str
    commit: str
    version: str
    version_is_release: bool
    runtime_contract_schema: str
    plugin_entry_point: str
    #: The CUDA extensions the released plugin loads into a serving process,
    #: transcribed from the runtime contract's own ``native_extensions``
    #: table.  This is the reproducibility contract's half of the pin, not the
    #: admission half: §7.4 says an A/B's arms must have identical extension
    #: residency, and a lane whose `.so` no fingerprint pattern matches
    #: reports "nothing resident" -- a serve running Tessera's own native
    #: decode looking exactly like a stock serve.
    #:
    #: The chain is contract -> pin -> tool, with a refusal at each link.
    #: ``tessera_runtime_contract.require_pin_native_extensions_match_contract``
    #: refuses a pin that is not the pinned contract's table;
    #: ``tools/serve_fingerprint.py`` is stdlib-only and cannot read this file
    #: from inside a serving container, so it carries the same rows and
    #: ``tests/test_tessera_serve_fingerprint.py`` refuses any disagreement.
    serving_native_extensions: tuple[TesseraServingNativeExtension, ...]

    def native_extension_rows(self) -> list[dict]:
        """Every row as the contract spells it, in the pin's order."""
        return [ext.as_contract_row() for ext in self.serving_native_extensions]

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
    rows = payload["serving_native_extensions"]
    if not isinstance(rows, (list, tuple)) or not rows:
        raise TesseraServingRuntimePinError(
            f"{where}: serving_native_extensions must be a non-empty list of "
            "rows transcribing the runtime contract's native_extensions "
            "table; an empty table is a fingerprint that reports every serve "
            "identical"
        )
    extensions: list[TesseraServingNativeExtension] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        at = f"{where}.serving_native_extensions[{i}]"
        if not isinstance(row, Mapping) or set(row) != _NATIVE_EXTENSION_MEMBERS:
            observed = sorted(row) if isinstance(row, Mapping) else []
            raise TesseraServingRuntimePinError(
                f"{at}: expected exactly {sorted(_NATIVE_EXTENSION_MEMBERS)}, "
                f"got {observed}. The keys are the runtime contract's own "
                "spelling so the two tables compare field for field."
            )
        prefix = row["module_name_prefix"]
        if (not isinstance(prefix, str)
                or _EXTENSION_PREFIX_RE.fullmatch(prefix) is None):
            raise TesseraServingRuntimePinError(
                f"{at}.module_name_prefix must be a lowercase module-name "
                f"prefix, got {prefix!r}"
            )
        if prefix in seen:
            raise TesseraServingRuntimePinError(
                f"{at}.module_name_prefix {prefix!r} is declared twice"
            )
        seen.add(prefix)
        glob = row["filename_glob"]
        if not isinstance(glob, str) or not glob:
            raise TesseraServingRuntimePinError(
                f"{at}.filename_glob must be a non-empty string"
            )
        # Checked by MEANING and not by spelling, the way the runtime's own
        # validator checks it: a library name the load path can produce must
        # match, or the pin transcribes a glob that matches nothing a serve
        # maps -- which reads as "no lane extension resident" on every arm.
        if not fnmatch.fnmatch(f"{prefix}0123456789abcdef.so", glob):
            raise TesseraServingRuntimePinError(
                f"{at}.filename_glob {glob!r} matches no library name the "
                f"load path can produce ({prefix}<build identity>.so)"
            )
        rule = row["match"]
        if rule != MATCH_BASENAME_FNMATCH:
            raise TesseraServingRuntimePinError(
                f"{at}.match is {rule!r}; this reader knows only "
                f"{MATCH_BASENAME_FNMATCH!r} and will not guess at another "
                "rule's predicate. The rule is a value because a consumer "
                "cannot otherwise tell a stem from a prefix from a pattern."
            )
        extensions.append(TesseraServingNativeExtension(
            module_name_prefix=prefix,
            filename_glob=glob,
            match=rule,
        ))
    return TesseraServingRuntimePin(
        schema=str(payload["schema"]),
        repository=str(payload["repository"]),
        commit=commit,
        version=version,
        version_is_release=released,
        runtime_contract_schema=str(payload["runtime_contract_schema"]),
        plugin_entry_point=entry_point,
        serving_native_extensions=tuple(extensions),
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
    "MATCH_BASENAME_FNMATCH",
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
    "TesseraServingNativeExtension",
    "TesseraServingRuntimePin",
    "TesseraServingRuntimePinError",
    "load_tessera_serving_runtime_pin",
    "parse_tessera_serving_runtime_pin",
    "require_exact_tessera_runtime_release",
    "tessera_serving_runtime_pin_path",
]
