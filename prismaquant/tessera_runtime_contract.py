"""Read the Tessera serving plugin's own packaged runtime contract.

Principle 14 in one file: every statement PrismaQuant makes about what a
Tessera artifact *serves* -- which family, at which rate, on which route,
behind which serve flags, at which tensor-parallel world size -- is read from
``tessera/serving/runtime_contract.json``, the table the plugin packages and
publishes, or it is refused.  Nothing here decides anything; it parses.

**Why this is not** ``lane_eligibility``.  That module is the generic
engine: it reads a serving release's contract through that lane's pin and
admits units against it.  It was written for the Gridbook lane's schemas
(``gridbook.lane-eligibility.v3``, ``cb_product``/``tcq_trellis`` format
kinds) and kept its shape when that lane was retired on 2026-09-02.  Tessera publishes its own
contract with its own schema ids and its own ``tessera_wire`` format kind, and
the honest reading of a second runtime's table is a second reader -- not a
widened set of accepted schema strings on the first, which would let either
runtime's table answer a question asked about the other.  The two share only
the route-status vocabulary, imported rather than restated.

**The dev pin.**  PrismaQuant's Tessera admission is fail-closed until a
Tessera RELEASE tag exists, and none has been cut.  A development override is
therefore explicit, named, and loud:

* ``PRISMAQUANT_TESSERA_DEV_PIN=<anything non-empty>`` opts in.  The value is
  recorded verbatim in provenance as what the operator asked for; it is not
  the gate.
* the installed contract's **answer** -- every value a gate reads, in the
  vocabulary of :func:`contract_answer` -- must equal
  :data:`TESSERA_DEV_PIN_ANSWER` or the read raises, with a field-level diff
  naming what moved.
* unset is production: no Tessera contract is read at all, and every rung
  stays ``unattested`` exactly as before.

There is no third state.  A mismatch never degrades to "unattested" -- that
would turn a stale pin into a silently empty menu, which is the failure mode
this whole file exists to prevent.

**Why the answer and not the bytes** (issue #38).  This pin used to compare
the environment's value against an exact commit and the file's sha256 against
a recorded one.  Both legs fired on *identity*, and the thing they name is an
editable checkout on the same box, so every Tessera commit that touched the
contract -- a prose ``detail``, a changelog paragraph, a ``contract_version``
bump -- turned PrismaQuant's attested path off and a fistful of tests red in a
repo nobody was editing, while the two rungs' meaning had not moved at all.
That is principle 14 read backwards: prose fields explain, they are never a
value a gate reads, so a prose edit is not a thing to re-review.  The gate now
reads the answer.  A commit that moves no answer passes silently; a commit
that moves one refuses and says which field, which is a review prompt rather
than a corruption warning.  :data:`TESSERA_DEV_PIN_COMMIT` and
:data:`TESSERA_DEV_PIN_CONTRACT_SHA256` survive as the *record of the review*
-- the build and the bytes a human read when the answer was accepted -- and
travel into provenance alongside the bytes this run actually read, so
prose-only drift is visible without being fatal.

This is deliberately weaker than a release pin and says so: it admits any
Tessera whose table answers identically, which is exactly the claim
PrismaQuant makes about it.  A Tessera RELEASE tag (issue #17) is still what
retires the override.

The contract's own identity (commit, sha, path, schema, contract_version)
travels into every allocation's provenance as ``tessera_dev_pin`` so a
shipcard records which table admitted its units.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lane_eligibility import (
    QUALIFICATION_DEVICE_QUALIFIED,
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
)

__all__ = [
    "MATCH_BASENAME_FNMATCH",
    "TESSERA_CONTRACT_SCHEMA",
    "TESSERA_DEV_PIN_COMMIT",
    "TESSERA_DEV_PIN_ANSWER",
    "TESSERA_DEV_PIN_CONTRACT_SHA256",
    "TESSERA_DEV_PIN_ENV",
    "TESSERA_LANE_SCHEMA",
    "TesseraContract",
    "TesseraContractError",
    "TesseraNativeExtension",
    "TesseraRouteCell",
    "contract_answer",
    "dev_pin_requested",
    "load_tessera_contract",
    "native_extension_pin_payload",
    "read_tessera_contract",
    "require_serving_pin_matches_contract",
]


class TesseraContractError(RuntimeError):
    """The Tessera runtime contract is absent, malformed, or off its pin."""


#: The schema ids this reader accepts.  Both are checked before any key is
#: read: an older table is not a subset of this one, and "missing field" is the
#: wrong error to hand someone whose contract predates the field.
TESSERA_CONTRACT_SCHEMA = "tessera.runtime-contract.v1"
TESSERA_LANE_SCHEMA = "tessera.lane-eligibility.v3"

#: The one ``native_extensions[].match`` rule this repository implements:
#: ``fnmatch`` an entry's ``filename_glob`` against the BASENAME of a mapped
#: ``.so``.  Tessera publishes the rule as a VALUE (``tessera.serving.ext``'s
#: ``MATCH_BASENAME_FNMATCH``) precisely so a consumer never has to guess
#: whether the published string is a stem, a prefix or a pattern, and a rule
#: this reader does not implement is REFUSED rather than approximated.
#: Applying a substring search to a table that says ``basename_fnmatch`` is a
#: predicate of our own wearing the runtime's name, which is the shape
#: principle 14 exists to refuse.
MATCH_BASENAME_FNMATCH = "basename_fnmatch"
_IMPLEMENTED_MATCH_RULES = frozenset({MATCH_BASENAME_FNMATCH})

#: A stand-in build identity, used only to check that an entry's
#: ``module_name_prefix`` and its ``filename_glob`` describe one library: the
#: JIT module name is ``<prefix><identity>`` and the file is that plus ``.so``,
#: so a glob that cannot match this cannot match a real one.  The literal
#: mirrors the probe Tessera's own ``_validate_native_extensions`` uses, so the
#: producer and the runtime run the same legibility check.
_IDENTITY_PROBE = "0123456789abcdef"

#: The development override.  See the module docstring; there is no default.
TESSERA_DEV_PIN_ENV = "PRISMAQUANT_TESSERA_DEV_PIN"

#: The Tessera commit this pin's answer was reviewed against.  Declared and
#: recorded; NOT compared to anything.  A moving ``master`` is not a review
#: event -- :data:`TESSERA_DEV_PIN_ANSWER` is what refuses.
#: Moved 2026-09-03 (issue #133) to the Tessera build whose contract first
#: publishes ``native_extensions`` -- the HEAD of the reviewed checkout whose
#: tree holds the bytes hashed below, not merely the commit that last touched
#: the file, because the record is of a build a human read.
TESSERA_DEV_PIN_COMMIT = "379a14d93f503db13c47deaf46b16bfb0ca2b7f7"

#: sha256 of ``tessera/serving/runtime_contract.json`` at that commit -- the
#: bytes a human read when the answer below was accepted.  Recorded, and
#: compared into provenance against the bytes this run read, so prose-only
#: drift is visible; it is not the refusal.
TESSERA_DEV_PIN_CONTRACT_SHA256 = (
    "bedb74655ae21a9b6e8f7547271954843ae81388f540dd1146bef5233b462920"
)

#: The ANSWER this pin was reviewed against -- every value a gate reads, in
#: the vocabulary of :func:`contract_answer`.  This literal, not the file's
#: bytes, is what refuses: a Tessera commit that rewrites prose, reorders keys
#: or bumps ``contract_version`` publishes the same answer and does not
#: re-stale the pin, while any move in a family, a rate range, an attested
#: rung, a world-size ceiling, a cell or the canonical ``quant_method`` does --
#: with a field-level diff naming it.  The git diff of this literal is the
#: review.
TESSERA_DEV_PIN_ANSWER = {'schema': 'tessera.runtime-contract.v1',
     'lane_schema': 'tessera.lane-eligibility.v3',
     'quant_method': 'tessera',
     'native_extensions': [['tessera_nvfp4_',
                            'tessera_nvfp4_*.so',
                            'basename_fnmatch']],
     'families': {'TESSERA_BF16_K1': {'reader_rate_range_q256': [256, 4096],
                                      'attested_rungs_q256': [1792],
                                      'max_world_size': 1},
                  'TESSERA_E2M1_K2': {'reader_rate_range_q256': [896, 896],
                                      'attested_rungs_q256': [896],
                                      'max_world_size': 1},
                  'TESSERA_E4M3_K1': {'reader_rate_range_q256': [256, 2048],
                                      'attested_rungs_q256': [1024],
                                      'max_world_size': 1}},
     'cells': [['tessera_bf16_k1_dense_sm121_batch_mm_w16a16',
                'sm_121',
                'TESSERA_BF16_K1',
                'dense',
                'batch',
                [1792],
                'bf16_unquantized',
                'backed_with_serve_flag',
                'device_qualified',
                'tessera',
                ['TESSERA_SERVE_MODE=resident|streamed']],
               ['tessera_bf16_k1_dense_sm121_decode_mm_w16a16',
                'sm_121',
                'TESSERA_BF16_K1',
                'dense',
                'decode',
                [1792],
                'bf16_unquantized',
                'backed_with_serve_flag',
                'device_qualified',
                'tessera',
                ['TESSERA_SERVE_MODE=resident|streamed']],
               ['tessera_e2m1_k2_dense_sm121_batch_scaled_mm_w4a4',
                'sm_121',
                'TESSERA_E2M1_K2',
                'dense',
                'batch',
                [896],
                'e2m1_group16_ue4m3_static',
                'backed_with_serve_flag',
                'device_qualified',
                'tessera',
                ['TESSERA_SERVE_MODE=resident|streamed']],
               ['tessera_e2m1_k2_dense_sm121_decode_scaled_mm_w4a4',
                'sm_121',
                'TESSERA_E2M1_K2',
                'dense',
                'decode',
                [896],
                'e2m1_group16_ue4m3_static',
                'backed_with_serve_flag',
                'device_qualified',
                'tessera',
                ['TESSERA_SERVE_MODE=resident|streamed']],
               ['tessera_e4m3_k1_dense_sm121_batch_scaled_mm_w8a8',
                'sm_121',
                'TESSERA_E4M3_K1',
                'dense',
                'batch',
                [1024],
                'fp8_per_token_dynamic',
                'backed_with_serve_flag',
                'device_qualified',
                'tessera',
                ['TESSERA_SERVE_MODE=resident|streamed']],
               ['tessera_e4m3_k1_dense_sm121_decode_scaled_mm_w8a8',
                'sm_121',
                'TESSERA_E4M3_K1',
                'dense',
                'decode',
                [1024],
                'fp8_per_token_dynamic',
                'backed_with_serve_flag',
                'device_qualified',
                'tessera',
                ['TESSERA_SERVE_MODE=resident|streamed']]]}

#: Route statuses under which a cell says a native route EXECUTES.
_NATIVE_ROUTE_STATUSES = frozenset(
    {ROUTE_STATUS_BACKED, ROUTE_STATUS_BACKED_WITH_SERVE_FLAG}
)


@dataclass(frozen=True, slots=True)
class TesseraRouteCell:
    """One ``lane_eligibility.cells[]`` row, verbatim in the fields we read."""

    cell_id: str
    platform: str
    family: str
    structure: str
    regime: str
    rungs_q256: frozenset[int]
    activation_contract: str
    route_status: str
    qualification: str
    requires_plugin: str
    requires_serve_flags: tuple[str, ...]

    @property
    def native(self) -> bool:
        """Does this cell attest a route the runtime executes natively?"""
        return (
            self.qualification == QUALIFICATION_DEVICE_QUALIFIED
            and self.route_status in _NATIVE_ROUTE_STATUSES
        )


@dataclass(frozen=True, slots=True)
class TesseraNativeExtension:
    """One ``native_extensions[]`` row: what the plugin LOADS, verbatim.

    ``formats``/``lane_eligibility`` say what a serve EXECUTES; this says which
    compiled library can be resident in the serving process while it does.
    They are different claims and the second one has its own consumer:
    ``tools/serve_fingerprint.py`` keys KL comparability (ARCHITECTURE.md
    section 7.4) on whether a lane's ``.so`` was mapped, and until Tessera
    contract v7 published this table PrismaQuant mirrored the name into its own
    pin -- a claim about Tessera's runtime, maintained here.

    ``source``, ``loaded_by``, ``routes`` and ``when_unavailable`` are parsed
    and carried but are not read by any gate in this repository today; see
    :func:`contract_answer` for why that keeps them out of the reviewed answer.
    """

    module_name_prefix: str
    filename_glob: str
    match: str
    source: str
    loaded_by: str
    routes: tuple[str, ...]
    #: ``residency mode -> {"status": ..., "decoder": ...}``, verbatim.  The
    #: vocabulary is Tessera's and is NOT re-validated here: nothing in this
    #: repository reads it yet, and restating another runtime's enum is how a
    #: legitimate new value becomes a false refusal.
    when_unavailable: Mapping[str, Mapping[str, Any]]

    def matches(self, path: str) -> bool:
        """Does ``path`` name this extension, under the rule the TABLE names?"""
        if self.match not in _IMPLEMENTED_MATCH_RULES:
            raise TesseraContractError(
                f"native extension {self.module_name_prefix!r} publishes "
                f"match rule {self.match!r}, which this reader does not "
                "implement"
            )
        return fnmatch.fnmatch(os.path.basename(path), self.filename_glob)


@dataclass(frozen=True, slots=True)
class TesseraContract:
    """The plugin's packaged contract, parsed.  Every field is read, not typed."""

    #: ``family -> (lo, hi)`` inclusive q256 rate range the reader accepts.
    reader_rate_range: Mapping[str, tuple[int, int]]
    #: ``family -> the rungs a ``lane_eligibility`` cell attests``.
    attested_rungs: Mapping[str, frozenset[int]]
    cells: tuple[TesseraRouteCell, ...]
    #: Every compiled library the plugin can map into a serving process.
    native_extensions: tuple[TesseraNativeExtension, ...]
    #: ``family -> max tensor-parallel world size``, closed world.
    max_world_size: Mapping[str, int]
    quant_method: str
    contract_version: int
    plugin_version: str
    attested_on: Mapping[str, str]
    #: Identity -- what travels into provenance.
    commit: str
    sha256: str
    path: str

    def governs(self, family: str) -> bool:
        """Does the contract publish this payload family at all?"""
        return str(family) in self.reader_rate_range

    def native_cells(self, family: str, rate_q256: int
                     ) -> tuple[TesseraRouteCell, ...]:
        """Every native cell covering ``(family, rate)``, on any platform."""
        return tuple(
            cell for cell in self.cells
            if cell.family == str(family)
            and int(rate_q256) in cell.rungs_q256
            and cell.native
        )

    def identity(self) -> dict:
        """The ``tessera_dev_pin`` provenance block.

        Records the review *and* the read: which Tessera build and bytes a
        human accepted this answer against, which bytes this run actually
        consumed, and whether the two are the same file.  They can differ
        legitimately -- the gate is the answer, not the bytes -- and a
        shipcard that could not tell the two apart would be asserting a
        review it did not get.
        """
        return {
            "requested": dev_pin_requested(),
            "commit": self.commit,
            "contract_sha256": self.sha256,
            "reviewed_contract_sha256": TESSERA_DEV_PIN_CONTRACT_SHA256,
            "bytes_are_the_reviewed_bytes":
                self.sha256 == TESSERA_DEV_PIN_CONTRACT_SHA256,
            "contract_path": self.path,
            "schema": TESSERA_CONTRACT_SCHEMA,
            "contract_version": self.contract_version,
            "plugin_version": self.plugin_version,
            "quant_method": self.quant_method,
            "attested_on": dict(self.attested_on),
            "note": (
                "development override: no Tessera RELEASE tag exists, so this "
                "allocation's Tessera routes were admitted by the packaged "
                "contract, whose ANSWER (every value a gate reads) equals the "
                "one reviewed at the named commit. The bytes need not be the "
                "reviewed bytes; bytes_are_the_reviewed_bytes says which"
            ),
        }


def contract_answer(contract: "TesseraContract") -> dict:
    """Exactly the values a gate reads, canonicalised, and nothing else.

    Principle 14's line, made mechanical.  ``detail``, ``rationale``, the
    changelog and every other prose field explains; none of them is a value a
    gate reads, so none of them appears here.  Neither do ``contract_version``,
    ``plugin_version`` or ``attested_on``: those are the table's *identity*,
    which travels into provenance, and a version bump that moved no answer is
    not a thing to re-review.

    What is here is what an admission decision is made of -- which families
    exist, the rate range the decoder accepts for each, the rungs a cell
    attests, the tensor-parallel ceiling, the canonical ``quant_method`` this
    producer writes into the checkpoint, and every cell field the route gate
    reads.  Two contracts with the same answer admit the same units.

    ``native_extensions`` entered the answer on 2026-09-03 (issue #133) because
    a gate started reading it: the serve fingerprint's Tessera predicate is
    derived from ``module_name_prefix``/``filename_glob``/``match``, so a
    Tessera commit that renames the extension moves an answer here and must be
    re-reviewed rather than silently turning a Tessera serve back into "no lane
    extension resident".  Only those three fields are here.  ``source``,
    ``loaded_by``, ``routes`` and ``when_unavailable`` are parsed onto
    :class:`TesseraNativeExtension` and travel nowhere else: nothing in this
    repository reads them, and putting a field in the answer before a gate
    reads it manufactures review events out of edits that change no decision.
    The day something reads ``when_unavailable.decoder`` -- telling a native
    serve from a substituted one is exactly such a gate -- is the day it joins.
    """
    return {
        "schema": TESSERA_CONTRACT_SCHEMA,
        "lane_schema": TESSERA_LANE_SCHEMA,
        "quant_method": contract.quant_method,
        "families": {
            family: {
                "reader_rate_range_q256": [int(rng[0]), int(rng[1])],
                "attested_rungs_q256": sorted(
                    int(r) for r in contract.attested_rungs.get(family, ())),
                "max_world_size": int(contract.max_world_size.get(family, 0)),
            }
            for family, rng in sorted(contract.reader_rate_range.items())
        },
        "native_extensions": sorted(
            [ext.module_name_prefix, ext.filename_glob, ext.match]
            for ext in contract.native_extensions
        ),
        "cells": sorted(
            [
                cell.cell_id,
                cell.platform,
                cell.family,
                cell.structure,
                cell.regime,
                sorted(int(r) for r in cell.rungs_q256),
                cell.activation_contract,
                cell.route_status,
                cell.qualification,
                cell.requires_plugin,
                sorted(cell.requires_serve_flags),
            ]
            for cell in contract.cells
        ),
    }


def _answer_drift(reviewed: Mapping[str, Any], installed: Mapping[str, Any]
                  ) -> list[str]:
    """Field-level lines naming what moved, so the refusal is reviewable."""
    lines: list[str] = []
    for key in ("schema", "lane_schema", "quant_method"):
        if reviewed.get(key) != installed.get(key):
            lines.append(
                f"  {key}: reviewed {reviewed.get(key)!r}, installed "
                f"{installed.get(key)!r}")
    r_fam, i_fam = reviewed.get("families", {}), installed.get("families", {})
    for family in sorted(set(r_fam) | set(i_fam)):
        if family not in r_fam:
            lines.append(f"  families[{family}]: NEW, not in the reviewed answer")
        elif family not in i_fam:
            lines.append(f"  families[{family}]: GONE from the installed contract")
        elif r_fam[family] != i_fam[family]:
            for k in sorted(set(r_fam[family]) | set(i_fam[family])):
                if r_fam[family].get(k) != i_fam[family].get(k):
                    lines.append(
                        f"  families[{family}].{k}: reviewed "
                        f"{r_fam[family].get(k)!r}, installed "
                        f"{i_fam[family].get(k)!r}")
    _NATIVE_FIELDS = ("module_name_prefix", "filename_glob", "match")
    r_ext = {row[0]: row for row in reviewed.get("native_extensions", ())}
    i_ext = {row[0]: row for row in installed.get("native_extensions", ())}
    for prefix in sorted(set(r_ext) | set(i_ext)):
        if prefix not in r_ext:
            lines.append(
                f"  native_extensions[{prefix}]: NEW, not in the reviewed answer")
        elif prefix not in i_ext:
            lines.append(
                f"  native_extensions[{prefix}]: GONE from the installed "
                "contract -- a serve loading it would fingerprint as no lane "
                "extension resident")
        elif list(r_ext[prefix]) != list(i_ext[prefix]):
            for field, was, now in zip(_NATIVE_FIELDS, r_ext[prefix],
                                       i_ext[prefix]):
                if was != now:
                    lines.append(
                        f"  native_extensions[{prefix}].{field}: reviewed "
                        f"{was!r}, installed {now!r}")
    r_cells = {tuple(c[:1])[0]: c for c in reviewed.get("cells", ())}
    i_cells = {tuple(c[:1])[0]: c for c in installed.get("cells", ())}
    for cell_id in sorted(set(r_cells) | set(i_cells)):
        if cell_id not in r_cells:
            lines.append(f"  cells[{cell_id}]: NEW, not in the reviewed answer")
        elif cell_id not in i_cells:
            lines.append(f"  cells[{cell_id}]: GONE from the installed contract")
        elif list(r_cells[cell_id]) != list(i_cells[cell_id]):
            lines.append(
                f"  cells[{cell_id}]: reviewed {r_cells[cell_id]!r}, installed "
                f"{i_cells[cell_id]!r}")
    return lines


def dev_pin_requested() -> str:
    """The commit the environment asks for, or ``""``."""
    return str(os.environ.get(TESSERA_DEV_PIN_ENV, "")).strip()


def contract_path() -> Path:
    """Where the packaged contract lives, by path arithmetic on ``tessera``.

    Deliberately **not** ``importlib.resources.files("tessera.serving")``:
    importing that package registers the vLLM plugin, and a producer-side
    contract read must not have that side effect.  ``tessera.__init__`` is a
    version string and a schema id; its directory is the anchor.
    """
    import tessera

    return Path(tessera.__file__).resolve().parent / "serving" / "runtime_contract.json"


def _require(block: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise TesseraContractError(f"{where} publishes no {key!r}")
    return block[key]


def _parse(payload: Mapping[str, Any], *, commit: str, sha: str, path: str
           ) -> TesseraContract:
    schema = payload.get("schema")
    if schema != TESSERA_CONTRACT_SCHEMA:
        raise TesseraContractError(
            f"{path}: schema must be {TESSERA_CONTRACT_SCHEMA!r}, got "
            f"{schema!r}. An older contract is not a subset of this one, so it "
            "is refused rather than partially read."
        )
    if "native_extensions" not in payload:
        raise TesseraContractError(
            f"{path} publishes no 'native_extensions'. That table is what "
            "names the compiled libraries the plugin loads into a serving "
            "process, and PrismaQuant's serve fingerprint keys KL "
            "comparability on their residency (ARCHITECTURE.md section 7.4). "
            "It arrived at Tessera contract_version 7 (RobTand/tessera#28); "
            "an older contract does not publish it and is refused rather than "
            "read as 'this runtime loads nothing', which is the exact false "
            "negative the table exists to remove."
        )
    natives = payload["native_extensions"]
    if (not isinstance(natives, Sequence) or isinstance(natives, (str, bytes))
            or not natives):
        raise TesseraContractError(
            f"{path}.native_extensions must be a non-empty JSON array"
        )
    extensions: list[TesseraNativeExtension] = []
    for i, entry in enumerate(natives):
        where = f"{path}.native_extensions[{i}]"
        if not isinstance(entry, Mapping):
            raise TesseraContractError(f"{where} must be a JSON object")
        prefix = str(_require(entry, "module_name_prefix", where))
        glob = str(_require(entry, "filename_glob", where))
        rule = str(_require(entry, "match", where))
        if rule not in _IMPLEMENTED_MATCH_RULES:
            raise TesseraContractError(
                f"{where}.match is {rule!r}; this reader implements only "
                f"{sorted(_IMPLEMENTED_MATCH_RULES)} and will not substitute a "
                "predicate of its own for a rule the runtime published"
            )
        if not fnmatch.fnmatch(f"{prefix}{_IDENTITY_PROBE}.so", glob):
            raise TesseraContractError(
                f"{where}: filename_glob {glob!r} does not match the library "
                f"module_name_prefix {prefix!r} builds "
                f"({prefix}{_IDENTITY_PROBE}.so); the two describe different "
                "files, so a fingerprint derived from them would record the "
                "wrong residency"
            )
        when = _require(entry, "when_unavailable", where)
        if not isinstance(when, Mapping) or not when:
            raise TesseraContractError(
                f"{where}.when_unavailable must be a non-empty object keyed by "
                "residency mode"
            )
        extensions.append(TesseraNativeExtension(
            module_name_prefix=prefix,
            filename_glob=glob,
            match=rule,
            source=str(_require(entry, "source", where)),
            loaded_by=str(_require(entry, "loaded_by", where)),
            routes=tuple(str(r) for r in _require(entry, "routes", where)),
            when_unavailable={
                str(mode): dict(behaviour) if isinstance(behaviour, Mapping)
                else behaviour
                for mode, behaviour in when.items()
            },
        ))

    reader_range: dict[str, tuple[int, int]] = {}
    attested: dict[str, frozenset[int]] = {}
    formats = _require(payload, "formats", path)
    if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
        raise TesseraContractError(f"{path}.formats must be a JSON array")
    for i, entry in enumerate(formats):
        if not isinstance(entry, Mapping):
            raise TesseraContractError(f"{path}.formats[{i}] must be an object")
        where = f"{path}.formats[{i}]"
        kind = str(_require(entry, "kind", where))
        if kind != "tessera_wire":
            raise TesseraContractError(
                f"{where}.kind is {kind!r}; this reader knows only "
                "'tessera_wire' and will not guess at another kind's rung "
                "vocabulary"
            )
        family = str(_require(entry, "family", where))
        lo, hi = (int(v) for v in _require(entry, "reader_rate_range_q256", where))
        reader_range[family] = (lo, hi)
        # ``attested_rungs_q256`` is the field's name since Tessera contract
        # v2; ``candidate_rungs_q256`` is the deprecated alias it kept so the
        # rename stayed additive, and Tessera's own reader refuses the two if
        # they disagree.  Read the current name first so this reader survives
        # the alias being dropped, and accept the alias alone so it can still
        # read a v1 table.  Reading the alias *preferentially* is how the gap
        # the rename closed would reopen: the alias never was the decodable
        # set.
        rungs = entry.get("attested_rungs_q256", entry.get("candidate_rungs_q256"))
        if rungs is None:
            raise TesseraContractError(
                f"{where} publishes no 'attested_rungs_q256' (nor its "
                "deprecated alias 'candidate_rungs_q256')"
            )
        attested[family] = frozenset(int(r) for r in rungs)

    lane = _require(payload, "lane_eligibility", path)
    if not isinstance(lane, Mapping):
        raise TesseraContractError(f"{path}.lane_eligibility must be an object")
    lane_schema = lane.get("schema")
    if lane_schema != TESSERA_LANE_SCHEMA:
        raise TesseraContractError(
            f"{path}.lane_eligibility.schema must be {TESSERA_LANE_SCHEMA!r}, "
            f"got {lane_schema!r}"
        )
    cells: list[TesseraRouteCell] = []
    for i, cell in enumerate(_require(lane, "cells", f"{path}.lane_eligibility")):
        where = f"{path}.lane_eligibility.cells[{i}]"
        if not isinstance(cell, Mapping):
            raise TesseraContractError(f"{where} must be a JSON object")
        family = str(_require(cell, "family", where))
        if family not in reader_range:
            raise TesseraContractError(
                f"{where} names family {family!r}, which the contract's "
                "formats table does not publish"
            )
        cells.append(TesseraRouteCell(
            cell_id=str(_require(cell, "id", where)),
            platform=str(_require(cell, "platform", where)),
            family=family,
            structure=str(_require(cell, "structure", where)),
            regime=str(_require(cell, "regime", where)),
            rungs_q256=frozenset(
                int(r) for r in _require(cell, "rungs_q256", where)),
            activation_contract=str(_require(cell, "activation_contract", where)),
            route_status=str(_require(cell, "route_status", where)),
            qualification=str(_require(cell, "qualification", where)),
            requires_plugin=str(cell.get("requires_plugin", "")),
            requires_serve_flags=tuple(
                str(f) for f in cell.get("requires_serve_flags", ())),
        ))

    tp = _require(payload, "tensor_parallel", path)
    if str(tp.get("semantics")) != "closed_world":
        raise TesseraContractError(
            f"{path}.tensor_parallel.semantics is "
            f"{tp.get('semantics')!r}; this reader treats the block as a closed "
            "world (a family absent from it is not attested at any degree) and "
            "will not read an open-world table under that assumption"
        )
    world: dict[str, int] = {}
    for i, unit in enumerate(tp.get("units", ())):
        where = f"{path}.tensor_parallel.units[{i}]"
        world[str(_require(unit, "unit", where))] = int(
            _require(unit, "max_world_size", where))

    versions = payload.get("versions", {})
    method = payload.get("quant_method", {})
    return TesseraContract(
        reader_rate_range=reader_range,
        attested_rungs=attested,
        cells=tuple(cells),
        native_extensions=tuple(extensions),
        max_world_size=world,
        quant_method=str(method.get("canonical", "")),
        contract_version=int(payload.get("contract_version", 0)),
        plugin_version=str(versions.get("tessera", "")),
        attested_on={str(k): str(v)
                     for k, v in dict(versions.get("attested_on", {})).items()},
        commit=commit,
        sha256=sha,
        path=path,
    )


def native_extension_pin_payload(
    contract: "TesseraContract",
) -> list[dict[str, str]]:
    """The pin's ``serving_native_extensions`` member, DERIVED from the table.

    One function, so the pin file is refreshed by the same code that refuses a
    stale one.  It projects exactly the fields the fingerprint's predicate is
    built from; everything else in the row explains the runtime and is not the
    producer's to restate.
    """
    return [
        {
            "module_name_prefix": ext.module_name_prefix,
            "filename_glob": ext.filename_glob,
            "match": ext.match,
        }
        for ext in sorted(contract.native_extensions,
                          key=lambda e: e.module_name_prefix)
    ]


def require_serving_pin_matches_contract(
    contract: "TesseraContract",
    pin: Any = None,
) -> None:
    """Refuse a serving pin whose extension table is not the contract's.

    This is the middle link of principle 14's chain for the serve fingerprint:
    contract -> pin -> ``tools/serve_fingerprint.py``.  The last link is
    refused by ``tests/test_tessera_serve_fingerprint.py`` (the tool is
    stdlib-only and reads no package data inside a serving container, so it
    carries a copy); this link is refused live, wherever the contract is read,
    because a pin that has drifted from the runtime it pins makes every
    section-7.4 residency claim built under it wrong.
    """
    from .tessera_serving_runtime_pin import (
        TesseraServingRuntimePinError,
        load_tessera_serving_runtime_pin,
    )

    if pin is None:
        pin = load_tessera_serving_runtime_pin()
    expected = native_extension_pin_payload(contract)
    observed = [
        {
            "module_name_prefix": ext.module_name_prefix,
            "filename_glob": ext.filename_glob,
            "match": ext.match,
        }
        for ext in pin.serving_native_extensions
    ]
    if observed == expected:
        return
    by_prefix_expected = {row["module_name_prefix"]: row for row in expected}
    by_prefix_observed = {row["module_name_prefix"]: row for row in observed}
    lines: list[str] = []
    for prefix in sorted(set(by_prefix_expected) | set(by_prefix_observed)):
        if prefix not in by_prefix_observed:
            lines.append(
                f"  {prefix}: the contract publishes it, the pin does not -- a "
                "serve loading it would fingerprint as no lane extension "
                "resident")
        elif prefix not in by_prefix_expected:
            lines.append(
                f"  {prefix}: the pin declares it, the contract does not "
                "publish it")
        else:
            for field in ("filename_glob", "match"):
                if by_prefix_observed[prefix][field] != (
                        by_prefix_expected[prefix][field]):
                    lines.append(
                        f"  {prefix}.{field}: pin "
                        f"{by_prefix_observed[prefix][field]!r}, contract "
                        f"{by_prefix_expected[prefix][field]!r}")
    raise TesseraServingRuntimePinError(
        "the Tessera serving pin's native-extension table is not the one the "
        f"runtime publishes ({contract.path}):\n" + "\n".join(lines) + "\n"
        "Regenerate the pin's serving_native_extensions from "
        "tessera_runtime_contract.native_extension_pin_payload() and mirror it "
        "into tools/serve_fingerprint.py in the same commit."
    )


def read_tessera_contract(path: Path | str) -> TesseraContract:
    """Parse one contract file, with no dev-pin gate and no answer check.

    The gates live in :func:`load_tessera_contract`.  This is the plain read a
    test or a pin refresh needs: the table as the installed runtime publishes
    it, whether or not this environment is opted into the development override.
    """
    location = Path(path)
    raw = location.read_bytes()
    return _load_at(str(location), hashlib.sha256(raw).hexdigest(),
                    TESSERA_DEV_PIN_COMMIT)


@lru_cache(maxsize=8)
def _load_at(path: str, sha: str, commit: str) -> TesseraContract:
    """Parse one contract file.  Keyed on the SHA, so a changed file re-reads.

    ``route_admission``'s docstring explains why a cache over a runtime
    contract is a defect when its key cannot state the contract's identity.
    This key is exactly that identity, so the cache is safe: edit the file and
    the sha changes, which is a different key.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _parse(payload, commit=commit, sha=sha, path=path)


def load_tessera_contract() -> "TesseraContract | None":
    """The packaged Tessera contract under the dev pin, or ``None``.

    ``None`` means the pin is not requested, which is production: no Tessera
    route is attested and the attested menu is empty.  Every other failure --
    a contract whose *answer* is not the reviewed one, a missing or malformed
    file -- raises.  A mismatch never degrades to "unattested"; that would turn
    a stale pin into a silently empty menu.
    """
    requested = dev_pin_requested()
    if not requested:
        return None
    path = contract_path()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TesseraContractError(
            f"cannot read the packaged Tessera contract at {path}: {exc}"
        ) from exc
    sha = hashlib.sha256(raw).hexdigest()
    contract = _load_at(str(path), sha, TESSERA_DEV_PIN_COMMIT)
    drift = _answer_drift(TESSERA_DEV_PIN_ANSWER, contract_answer(contract))
    if drift:
        raise TesseraContractError(
            "Tessera moved and its answer moved with it -- re-review the pin.\n"
            f"The pin in {__name__} was reviewed against Tessera "
            f"{TESSERA_DEV_PIN_COMMIT} (contract sha256 "
            f"{TESSERA_DEV_PIN_CONTRACT_SHA256}); {path} hashes to {sha} and "
            "publishes a different answer:\n" + "\n".join(drift) + "\n"
            "This is not a corruption warning. Read what moved, decide whether "
            "PrismaQuant should admit it, and update TESSERA_DEV_PIN_ANSWER "
            "(with the commit and sha) in the same commit -- that diff is the "
            "review."
        )
    require_serving_pin_matches_contract(contract)
    return contract
