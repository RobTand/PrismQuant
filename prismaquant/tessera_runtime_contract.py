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

* ``PRISMAQUANT_TESSERA_DEV_PIN=<commit>`` must equal
  :data:`TESSERA_DEV_PIN_COMMIT` -- the exact Tessera commit this pin was
  written against -- or the read raises.
* the installed contract's sha256 must equal
  :data:`TESSERA_DEV_PIN_CONTRACT_SHA256` or the read raises.  The commit is
  *declared*; the sha is what actually travels, because a worktree rsync'd to
  a second box is not a git checkout and cannot be asked its HEAD.
* unset is production: no Tessera contract is read at all, and every rung
  stays ``unattested`` exactly as before.

There is no third state.  A mismatch never degrades to "unattested" -- that
would turn a stale pin into a silently empty menu, which is the failure mode
this whole file exists to prevent.

The contract's own identity (commit, sha, path, schema, contract_version)
travels into every allocation's provenance as ``tessera_dev_pin`` so a
shipcard records which table admitted its units.
"""
from __future__ import annotations

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
    "TESSERA_CONTRACT_SCHEMA",
    "TESSERA_DEV_PIN_COMMIT",
    "TESSERA_DEV_PIN_CONTRACT_SHA256",
    "TESSERA_DEV_PIN_ENV",
    "TESSERA_LANE_SCHEMA",
    "TesseraContract",
    "TesseraContractError",
    "TesseraRouteCell",
    "dev_pin_requested",
    "load_tessera_contract",
]


class TesseraContractError(RuntimeError):
    """The Tessera runtime contract is absent, malformed, or off its pin."""


#: The schema ids this reader accepts.  Both are checked before any key is
#: read: an older table is not a subset of this one, and "missing field" is the
#: wrong error to hand someone whose contract predates the field.
TESSERA_CONTRACT_SCHEMA = "tessera.runtime-contract.v1"
TESSERA_LANE_SCHEMA = "tessera.lane-eligibility.v3"

#: The development override.  See the module docstring; there is no default.
TESSERA_DEV_PIN_ENV = "PRISMAQUANT_TESSERA_DEV_PIN"

#: The exact Tessera commit this pin was written against.  Declared, and
#: compared against the environment's value so a stale export raises.
TESSERA_DEV_PIN_COMMIT = "f3e7d0ae78e64fcc1a13d5b9553a95fe4006bef4"

#: sha256 of ``tessera/serving/runtime_contract.json`` at that commit.  This is
#: the leg that actually attests: an rsync'd source tree has no git history to
#: interrogate, and the bytes are what the reader consumed.
TESSERA_DEV_PIN_CONTRACT_SHA256 = (
    "dff4fef7e6db72b97d7cba306cd280ae3d989d9bb310a64b9cf9f4a94a858976"
)

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
class TesseraContract:
    """The plugin's packaged contract, parsed.  Every field is read, not typed."""

    #: ``family -> (lo, hi)`` inclusive q256 rate range the reader accepts.
    reader_rate_range: Mapping[str, tuple[int, int]]
    #: ``family -> the rungs the contract publishes as candidates``.
    candidate_rungs: Mapping[str, frozenset[int]]
    cells: tuple[TesseraRouteCell, ...]
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
        """The ``tessera_dev_pin`` provenance block."""
        return {
            "commit": self.commit,
            "contract_sha256": self.sha256,
            "contract_path": self.path,
            "schema": TESSERA_CONTRACT_SCHEMA,
            "contract_version": self.contract_version,
            "plugin_version": self.plugin_version,
            "quant_method": self.quant_method,
            "attested_on": dict(self.attested_on),
            "note": (
                "development override: no Tessera RELEASE tag exists, so this "
                "allocation's Tessera routes were admitted by the packaged "
                "contract at the named commit rather than by a pinned release"
            ),
        }


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
    reader_range: dict[str, tuple[int, int]] = {}
    candidates: dict[str, frozenset[int]] = {}
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
        candidates[family] = frozenset(
            int(r) for r in _require(entry, "candidate_rungs_q256", where)
        )

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
        candidate_rungs=candidates,
        cells=tuple(cells),
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
    a commit that is not this pin's, a contract whose bytes are not this pin's,
    a missing or malformed file -- raises.
    """
    requested = dev_pin_requested()
    if not requested:
        return None
    if requested != TESSERA_DEV_PIN_COMMIT:
        raise TesseraContractError(
            f"{TESSERA_DEV_PIN_ENV}={requested!r} is not the commit this pin "
            f"was written against ({TESSERA_DEV_PIN_COMMIT}). The pin names one "
            "exact Tessera build; pointing it at another would attest routes "
            "against a table nobody read."
        )
    path = contract_path()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TesseraContractError(
            f"cannot read the packaged Tessera contract at {path}: {exc}"
        ) from exc
    sha = hashlib.sha256(raw).hexdigest()
    if sha != TESSERA_DEV_PIN_CONTRACT_SHA256:
        raise TesseraContractError(
            f"{path} hashes to {sha}, not the {TESSERA_DEV_PIN_CONTRACT_SHA256} "
            f"this pin recorded for {TESSERA_DEV_PIN_COMMIT}. The installed "
            "Tessera is not the one the pin names."
        )
    return _load_at(str(path), sha, TESSERA_DEV_PIN_COMMIT)
