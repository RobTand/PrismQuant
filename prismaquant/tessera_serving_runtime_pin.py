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

**THE PIN IS AN EXACT COMMIT PLUS THE CONTRACT'S DIGEST -- NOT A TAG.**
Until 2026-09-04 this pin required a Tessera RELEASE, and admission was
fail-closed because no tag existed.  Rob's instruction retired that:
*"can we just pin prismaquant to latest version of tessera? then we won't have
to keep cutting releases."*  "Latest" is read here as **an exact commit plus
the packaged contract's raw SHA-256**, never as a floating ref.  A floating
ref (``main``, "an installed source tree", "whatever imports") is precisely
the failure principle 14 exists to prevent, and this module's own history
names it: admission would flip to True the moment somebody put the Tessera
source tree on ``PYTHONPATH``.

Immutability now comes from those two values rather than from a tag, and the
DIGEST is the enforced half.  PrismaQuant cannot verify a sibling checkout's
git history from inside its own process -- a commit string is a claim about
another repository -- but it CAN hash the contract bytes it is about to read,
which is the only thing about the runtime a gate here actually consumes.  So:

* :data:`TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256` is checked against
  the installed ``tessera/serving/runtime_contract.json``, byte for byte, and
  a mismatch REFUSES.  A stray Tessera checkout on ``PYTHONPATH`` whose
  contract is not the pinned one is refused exactly as a PENDING pin was.
* :data:`TESSERA_SERVING_RUNTIME_PINNED_COMMIT` is recorded identity: it says
  which reviewed Tessera commit those bytes came from, so a shipcard and a
  human can find the tree, and so ``git`` can settle any question the digest
  raises.  The two are bound at review time by one command
  (``git -C <tessera> show HEAD:src/tessera/serving/runtime_contract.json |
  sha256sum``), which is what the pin's provenance records.

**``version_is_release`` is ADVISORY.**  It is still required, still parsed,
still recorded, and it still cannot be ``true`` over a PENDING commit -- so it
keeps saying something true for an actual release.  What it no longer does is
GATE: requiring it would re-impose the tag Rob just removed, and the
immutability it was standing in for is now carried directly by the digest.

**Moving the pin is ONE reviewed change, not three.**  Following the Gridbook
discipline (a pin whose schema, version, commit and digest cannot move by
halves), moving it means editing, in a single commit: the JSON pin's
``commit``/``version``/``contract_sha256``, AND the module constants
:data:`TESSERA_SERVING_RUNTIME_PINNED_VERSION` /
:data:`TESSERA_SERVING_RUNTIME_PINNED_COMMIT` /
:data:`TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256` below.  The reader
requires the pin to equal the constants, so a JSON edit alone cannot admit
anything and a constant edit alone cannot either.

**THE EXTENSION TABLE IS TRANSCRIBED, AND REFUSED AGAINST ITS SOURCE.**
``serving_native_extensions`` is not this repository's opinion about which
CUDA extensions the Tessera plugin loads: since Tessera contract v7 the
runtime publishes that itself, in ``native_extensions``, as a
``module_name_prefix``/``filename_glob``/``match`` triple plus the
``when_unavailable`` block saying what a serve does when the library is
absent.  The pin transcribes all four because
``tools/serve_fingerprint.py`` runs inside a serving container from
a bootstrap with no installed package and can read neither the contract nor
this reader module -- but the pin is JSON, so the tool reads the transported
pin file beside itself (a member of its gold-producer source closure, hence
digest-covered) and refuses a missing or malformed one instead of falling
back to a constant -- and
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

**A consequence worth stating.**  Under commit pinning, a developer checkout
of Tessera that has moved past the pin makes this repository's Tessera tests
go RED, by design: the installed contract is not the pinned contract, and
fail-closed is the whole point.  The fix is environmental -- install Tessera
at the pinned commit -- never a check that reads whatever is installed.

**``repository`` names the origin the pin was reviewed from.**  It is the
reviewed identity of the runtime, not a reachability claim, and no gate here
fetches from it.
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


#: v2 (2026-09-04) added ``contract_sha256`` and demoted ``version_is_release``
#: from gate to record.  The schema id moves with the member set, exactly as
#: the Gridbook pin's did when ``version_is_release`` was added, so a v1 pin is
#: refused rather than read as a v2 with a field missing.
TESSERA_SERVING_RUNTIME_PIN_SCHEMA = (
    "prismaquant.tessera_serving_runtime_pin.v2"
)
TESSERA_SERVING_RUNTIME_REPOSITORY = (
    "https://github.com/RobTand/tessera.git"
)

#: The contract schema this pin binds.  ``parse_tessera_serving_runtime_pin``
#: refuses a pin naming any other, so a runtime that moves its contract schema
#: cannot be pinned by halves -- the same "three are one change" rule the
#: Gridbook pins carry.
TESSERA_SERVING_RUNTIME_CONTRACT_SCHEMA = "tessera.runtime-contract.v1"

#: The conspicuous sentinels a pin carries while no Tessera commit has been
#: reviewed (they outlived the release-tag requirement Rob retired).  They
#: are structurally ACCEPTED by the parser -- so the pin file is reviewable,
#: and so this module has something honest to say -- and REFUSED by every live
#: admission gate through :func:`require_exact_tessera_runtime_pin`.
TESSERA_SERVING_RUNTIME_COMMIT_PENDING = "PENDING_TESSERA_RELEASE_COMMIT"
TESSERA_SERVING_RUNTIME_VERSION_PENDING = "PENDING_TESSERA_RELEASE_VERSION"
TESSERA_SERVING_RUNTIME_CONTRACT_SHA256_PENDING = "PENDING_TESSERA_CONTRACT_SHA256"

#: The exact reviewed Tessera runtime: a commit, the distribution version at
#: that commit, and the SHA-256 of the ``runtime_contract.json`` it packages.
#:
#: Bound at review time by one command against the CANONICAL remote, so the
#: commit and the digest cannot be two independent assertions about one
#: runtime -- and so the binding does not depend on whose working tree ran it::
#:
#:     TS=$(mktemp -d) && git -C "$TS" init -q
#:     git -C "$TS" fetch -q https://github.com/RobTand/tessera master
#:     SHA=$(git -C "$TS" rev-parse FETCH_HEAD)
#:     git -C "$TS" cat-file -p "$SHA:src/tessera/serving/runtime_contract.json" | sha256sum
#:
#: First recorded 2026-09-04T21:29-04:00 against Tessera master at 5acc2a6f
#: (contract v17, lane schema ``tessera.lane-eligibility.v6``, its PR #176).
#: Re-pinned 2026-09-05 to Tessera master at 8ed1d9a -- the merge of its
#: PR #332, and master's tip when this was bound (contract v22, lane schema
#: ``tessera.lane-eligibility.v9``).  The release checkout Rob named,
#: e78959ed, carried contract v20; two published moves lie past it.  #313
#: (b8b1cb38) re-measured the routed-MoE smoke through the checkpoint's own
#: chat template and recorded it clean (v21, receipt
#: ``docs/measurements/moe-smoke-recorded-2026-09-05.md``; prismaquant #198
#: option C).  #332 then answered Tessera's own #327 (P1): that ``recorded``
#: rested on a repetition rule which lived only in the measurements file and
#: was checked by nothing, so v22 moves the rule and its rows INTO the
#: contract as ``smoke.record`` and derives both the status and the
#: attribution from them.  Relative to the first pin (v17) the lane table
#: also gained a smoke control and attribution (v7), an encoder artifact
#: (v8) and the per-extension lane predicate (v20), and
#: ``tessera_runtime_contract.TESSERA_DEV_PIN_ANSWER`` moved with all of it in
#: the same commit.  Re-check it against the COMMIT rather than a past HEAD,
#: which nobody can re-run::
#:
#:     git -C "$TS" cat-file -p ba582d476a3b6db9057ebd1385dc52926f171451:src/tessera/serving/runtime_contract.json | sha256sum
#:
#: Re-pinned 2026-09-05 to ba582d4 (Tessera #356) for the priced-input
#: exporter snapshot API required by PrismaQuant #231. The v22 contract bytes
#: and reviewed admission answer are unchanged. Producer API dependencies can
#: require a newer pin even when the published runtime contract is identical.
#: No git tag names this commit, so ``version_is_release`` stays false in the
#: JSON beside this module: ``0.1.0`` is what the checkout's ``pyproject``
#: says, not a cut release.
TESSERA_SERVING_RUNTIME_PINNED_COMMIT = (
    "ba582d476a3b6db9057ebd1385dc52926f171451"
)
TESSERA_SERVING_RUNTIME_PINNED_VERSION = "0.1.0"
TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256 = (
    "719daa02da1564b56a141ca2702ae29d4fda553460978efbb6510ddcd1824927"
)

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
#: receipts covered -- and transcribed into the pin JSON's
#: ``serving_residency_env`` member because ``tools/serve_fingerprint.py``
#: records it in the serving stack's environment projection: the tool is
#: stdlib-only inside a serving container, so the name it projects reaches it
#: through the JSON, not through this constant.  The parser requires the two
#: to agree, so renaming the knob is a reviewed change to both in one commit.
TESSERA_SERVING_RESIDENCY_ENV = "TESSERA_SERVE_MODE"

_REQUIRED_MEMBERS = {
    "schema",
    "repository",
    "commit",
    "version",
    "version_is_release",
    "contract_sha256",
    "runtime_contract_schema",
    "plugin_entry_point",
    "serving_residency_env",
    "serving_native_extensions",
}
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VERSION_RE = re.compile(r"[0-9]+(?:[.][0-9]+)*(?:[A-Za-z0-9.+-]*)?")

#: The exact keys one ``serving_native_extensions`` row carries, spelled the
#: way the runtime's ``native_extensions`` table spells them.  Verbatim, so the
#: contract-vs-pin refusal in ``tessera_runtime_contract`` is a dict
#: comparison over the same field names rather than a re-mapping that could
#: itself be the drift.  ``when_unavailable`` joined this set for PrismaQuant
#: #142: a manifest that records only the basenames it *found* cannot say what
#: an absent library *means*, so the pin transcribes the block that says it --
#: per residency mode, the substitute decoder a serve keeps running on, or
#: that there is no serve at all.
_NATIVE_EXTENSION_MEMBERS = {
    "module_name_prefix",
    "filename_glob",
    "match",
    "when_unavailable",
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

    Four fields and no fifth, because these four are what a residency
    decision -- and the reading of an ABSENT library -- is made of: WHICH
    module the load path builds (``module_name_prefix``), WHICH filename that
    produces (``filename_glob``), WHICH RULE turns the glob into a decision
    (``match``), and WHAT RUNS INSTEAD when the library is absent
    (``when_unavailable``: per residency mode, the substitute decoder a serve
    keeps running on, or that there is no serve at all).  ``source``/
    ``loaded_by``/``routes`` name files, modules and route ids in the
    runtime's own tree and move nothing on this side, so they are identity and
    stay out -- exactly like ``plugin_version`` stays out of the dev-pin
    answer -- while ``when_unavailable`` is read by ``tools/kl_ab.py``'s §7.4
    refusal (PrismaQuant #142), which is why it is transcribed here rather
    than merely read from the contract where it is needed.
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
    #: Per residency mode, what a serve does when the library is absent, e.g.
    #: ``{"resident": {"status": "substituted",
    #: "decoder": "torch_materialize_stock"}, "streamed": {"status":
    #: "refused", "decoder": None}}``.  Stored as a plain
    #: ``{mode: {"status": str, "decoder": str | None}}`` mapping, modes in
    #: sorted order, so the transcription compares field for field.
    when_unavailable: Mapping[str, Mapping[str, str | None]]

    def as_contract_row(self) -> dict:
        """The row as the contract spells it, for a field-level comparison."""
        return {
            "module_name_prefix": self.module_name_prefix,
            "filename_glob": self.filename_glob,
            "match": self.match,
            "when_unavailable": {
                mode: {"status": behaviour["status"],
                       "decoder": behaviour["decoder"]}
                for mode, behaviour in sorted(self.when_unavailable.items())
            },
        }


@dataclass(frozen=True)
class TesseraServingRuntimePin:
    schema: str
    repository: str
    commit: str
    version: str
    #: ADVISORY since pin schema v2.  Recorded and structurally constrained (a
    #: PENDING commit/version cannot be marked released), read by no gate: see
    #: the module docstring for why requiring it would re-impose the tag.
    version_is_release: bool
    #: SHA-256 of the ``runtime_contract.json`` the pinned commit packages.
    #: The ENFORCED half of the pin -- the one fact about the serving runtime
    #: this producer can verify from inside its own process.
    contract_sha256: str
    runtime_contract_schema: str
    plugin_entry_point: str
    #: The serving runtime's one operator knob, transcribed so the serve
    #: fingerprint can project it: ``tools/serve_fingerprint.py`` records this
    #: name out of each server process's environment, and its value rides the
    #: performance-stack fingerprint.  A serve that omits it serves a different
    #: residency than the pin's receipts covered.
    serving_residency_env: str
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
    #: ``tools/serve_fingerprint.py`` is stdlib-only and cannot read this reader
    #: module from inside a serving container, so it reads the transported pin
    #: JSON beside itself -- and refuses a missing or malformed one -- while
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

    @property
    def contract_sha256_is_resolved(self) -> bool:
        return _SHA256_RE.fullmatch(self.contract_sha256) is not None


def parse_tessera_serving_runtime_pin(
    payload: Mapping[str, Any],
    *,
    where: str = "tessera_serving_runtime_pin.json",
) -> TesseraServingRuntimePin:
    """Structural read of a pin payload.  Accepts the PENDING sentinels.

    Accepting a pending pin *structurally* is what lets the file be reviewed
    before a tag exists.  It admits nothing: only
    :func:`require_exact_tessera_runtime_pin` is a gate, and it refuses
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
    contract_sha256 = payload["contract_sha256"]
    if not isinstance(contract_sha256, str) or (
        _SHA256_RE.fullmatch(contract_sha256) is None
        and contract_sha256 != TESSERA_SERVING_RUNTIME_CONTRACT_SHA256_PENDING
    ):
        raise TesseraServingRuntimePinError(
            f"{where}: contract_sha256 must be 64 lowercase hex digits (the "
            "SHA-256 of the runtime_contract.json the pinned commit packages) "
            f"or the exact pending sentinel "
            f"{TESSERA_SERVING_RUNTIME_CONTRACT_SHA256_PENDING!r}"
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
    residency_env = payload["serving_residency_env"]
    if (not isinstance(residency_env, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", residency_env) is None):
        raise TesseraServingRuntimePinError(
            f"{where}: serving_residency_env must be an environment-variable "
            f"name, got {residency_env!r}"
        )
    if residency_env != TESSERA_SERVING_RESIDENCY_ENV:
        raise TesseraServingRuntimePinError(
            f"{where}: serving_residency_env is {residency_env!r}; it must name "
            f"the {TESSERA_SERVING_RESIDENCY_ENV!r} operator knob the released "
            "runtime declares. A serving-lane env belongs to another runtime, "
            "so the fingerprint projects this name rather than a typed one -- "
            "and renaming it is a reviewed change to the JSON and the reader's "
            "constant in one commit."
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
        when = row["when_unavailable"]
        if not isinstance(when, Mapping) or not when:
            raise TesseraServingRuntimePinError(
                f"{at}.when_unavailable must be a non-empty object keyed by "
                "residency mode, transcribing the runtime contract's "
                "native_extensions table: it says what a serve does when "
                "this library is absent, which is what makes an absent .so "
                "readable"
            )
        behaviours: dict[str, dict[str, str | None]] = {}
        for mode, behaviour in when.items():
            bat = f"{at}.when_unavailable[{mode!r}]"
            if not isinstance(mode, str) or not mode:
                raise TesseraServingRuntimePinError(
                    f"{at}.when_unavailable keys must be non-empty residency "
                    f"mode names, got {mode!r}")
            if not isinstance(behaviour, Mapping):
                raise TesseraServingRuntimePinError(
                    f"{bat} must be an object with 'status' and 'decoder'")
            for member in ("status", "decoder"):
                if member not in behaviour:
                    raise TesseraServingRuntimePinError(
                        f"{bat} publishes no {member!r}")
            status = behaviour["status"]
            decoder = behaviour["decoder"]
            if not isinstance(status, str) or not status:
                raise TesseraServingRuntimePinError(
                    f"{bat}.status must be a non-empty string, "
                    f"got {status!r}")
            if decoder is not None and (
                    not isinstance(decoder, str) or not decoder):
                raise TesseraServingRuntimePinError(
                    f"{bat}.decoder must be a decoder name or null, "
                    f"got {decoder!r}")
            behaviours[str(mode)] = {"status": str(status),
                                     "decoder": (None if decoder is None
                                                 else str(decoder))}
        extensions.append(TesseraServingNativeExtension(
            module_name_prefix=prefix,
            filename_glob=glob,
            match=rule,
            when_unavailable=behaviours,
        ))
    return TesseraServingRuntimePin(
        schema=str(payload["schema"]),
        repository=str(payload["repository"]),
        commit=commit,
        version=version,
        version_is_release=released,
        contract_sha256=contract_sha256,
        runtime_contract_schema=str(payload["runtime_contract_schema"]),
        plugin_entry_point=entry_point,
        serving_residency_env=residency_env,
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


def installed_tessera_contract_sha256() -> str:
    """SHA-256 of the ``runtime_contract.json`` the INSTALLED Tessera packages.

    Resolved through ``tessera_runtime_contract.contract_path()`` -- the one
    resolver both producer readers share -- so the bytes hashed here are the
    bytes every other gate on this side reads.  Hashing a different copy would
    make the pin attest a file nothing consumes.

    Raises :class:`TesseraServingRuntimePinError` when there is no importable
    Tessera at all, because "no runtime" and "the wrong runtime" must both
    refuse; neither may read as "fine".
    """
    from importlib.resources import as_file
    import hashlib

    try:
        from .tessera_runtime_contract import contract_path
        with as_file(contract_path()) as path:
            return hashlib.sha256(path.read_bytes()).hexdigest()
    except TesseraServingRuntimePinError:
        raise
    except Exception as exc:  # ImportError, OSError, ModuleNotFoundError...
        raise TesseraServingRuntimePinError(
            "cannot read the installed Tessera runtime contract, so the pin "
            f"cannot be verified: {exc}"
        ) from exc


def require_exact_tessera_runtime_pin(
    pin: TesseraServingRuntimePin,
    *,
    installed_contract_sha256: str,
) -> None:
    """Refuse anything that is not the exact pinned Tessera runtime.

    A pure predicate over two inputs -- the tracked pin and the digest of the
    contract actually installed -- so the rule can be tested without an
    importable Tessera and cannot be satisfied by reading whatever happens to
    be on ``PYTHONPATH``.  :func:`require_pinned_tessera_runtime` is the thin
    resolver every live caller uses.

    Three conjuncts, in the order a reader needs them:

    1. The pin is RESOLVED -- no PENDING sentinel in the commit, the version
       or the digest.
    2. The pin EQUALS the module constants.  This is the "one reviewed change"
       rule: a JSON edit alone admits nothing and a constant edit alone admits
       nothing.
    3. The INSTALLED contract hashes to the pinned digest.  This is the
       enforced binding, and the reason a stray Tessera checkout on
       ``PYTHONPATH`` is refused: the commit is a claim about another
       repository that this process cannot check, but the bytes it is about to
       read are right here.

    ``version_is_release`` is deliberately absent from all three.  Rob retired
    the tag requirement; a gate that still demanded the flag would re-impose
    it.
    """
    if not (pin.commit_is_resolved and pin.version_is_resolved
            and pin.contract_sha256_is_resolved):
        raise TesseraServingRuntimePinError(
            "the Tessera serving runtime pin is still PENDING, so no Tessera "
            "route may be admitted. "
            f"pin.version={pin.version!r} pin.commit={pin.commit!r} "
            f"pin.contract_sha256={pin.contract_sha256!r}"
        )
    if (
        pin.commit != TESSERA_SERVING_RUNTIME_PINNED_COMMIT
        or pin.version != TESSERA_SERVING_RUNTIME_PINNED_VERSION
        or pin.contract_sha256 != TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256
    ):
        raise TesseraServingRuntimePinError(
            "the tracked Tessera pin file and this module's constants "
            "disagree; they are ONE reviewed change and neither half admits "
            "anything alone.\n"
            f"  constants: {TESSERA_SERVING_RUNTIME_PINNED_VERSION!r} / "
            f"{TESSERA_SERVING_RUNTIME_PINNED_COMMIT} / "
            f"{TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256}\n"
            f"  pin file:  {pin.version!r} / {pin.commit} / "
            f"{pin.contract_sha256}"
        )
    if installed_contract_sha256 != pin.contract_sha256:
        raise TesseraServingRuntimePinError(
            "the installed Tessera is not the pinned Tessera. Its "
            "runtime_contract.json hashes to\n"
            f"  {installed_contract_sha256}\n"
            "and the pin binds\n"
            f"  {pin.contract_sha256}\n"
            f"  (Tessera commit {TESSERA_SERVING_RUNTIME_PINNED_COMMIT}, "
            f"version {TESSERA_SERVING_RUNTIME_PINNED_VERSION}).\n"
            "Install Tessera at the pinned commit, or move the pin -- which "
            "is a reviewed change to "
            "prismaquant/tessera_runtime/tessera_serving_runtime_pin.json AND "
            "the constants in prismaquant/tessera_serving_runtime_pin.py, "
            "together. Reading whatever is installed is the floating ref this "
            "pin exists to refuse."
        )


def require_pinned_tessera_runtime(
    pin: TesseraServingRuntimePin | None = None,
) -> None:
    """The live gate: the tracked pin against the installed contract's bytes."""
    if pin is None:
        pin = load_tessera_serving_runtime_pin()
    require_exact_tessera_runtime_pin(
        pin, installed_contract_sha256=installed_tessera_contract_sha256())


__all__ = [
    "MATCH_BASENAME_FNMATCH",
    "TESSERA_SERVING_PLUGIN_NAME",
    "TESSERA_SERVING_QUANT_METHOD",
    "TESSERA_SERVING_RESIDENCY_ENV",
    "TESSERA_SERVING_RUNTIME_COMMIT_PENDING",
    "TESSERA_SERVING_RUNTIME_CONTRACT_SCHEMA",
    "TESSERA_SERVING_RUNTIME_CONTRACT_SHA256_PENDING",
    "TESSERA_SERVING_RUNTIME_PINNED_COMMIT",
    "TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256",
    "TESSERA_SERVING_RUNTIME_PINNED_VERSION",
    "TESSERA_SERVING_RUNTIME_PIN_SCHEMA",
    "TESSERA_SERVING_RUNTIME_REPOSITORY",
    "TESSERA_SERVING_RUNTIME_VERSION_PENDING",
    "TesseraServingNativeExtension",
    "TesseraServingRuntimePin",
    "TesseraServingRuntimePinError",
    "load_tessera_serving_runtime_pin",
    "parse_tessera_serving_runtime_pin",
    "installed_tessera_contract_sha256",
    "require_exact_tessera_runtime_pin",
    "require_pinned_tessera_runtime",
    "tessera_serving_runtime_pin_path",
]
