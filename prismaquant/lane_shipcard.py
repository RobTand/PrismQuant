"""Open a ship record for ANY export lane, with the slots that lane declares.

**The gap this closes.**  ``shipcard.py``'s contract is that the build lane
OPENS a record and the serve lane CLOSES it, and ``tools/publish_artifact.py``
refuses an artifact whose slots are not closed.  Until 2026-09-03 the only
thing in the tree that opened one was ``export_native_compressed.py`` --- the
*native lane's* exporter.  The GGUF arm exits before the driver's shipcard
block and writes no card; the Tessera arm calls Tessera's own exporter, which
has no concept of a PrismaQuant shipcard, and then exits at
``run-pipeline.sh:2432`` --- 130 lines above the block that would have printed
one.

So on two of the three sanctioned lanes the chain

    gate declared in ``lane_specs/<lane>.json``
      -> slot opened by the build lane
      -> filled by the serve lane
      -> refused on by ``publish_artifact``

was broken at its second link, and every gate those lanes declare was enforced
by nothing.  ``publish_artifact`` still refused such an artifact --- but for
*absence of a card*, which an operator closes by writing one by hand with the
base slots, and a hand-written base card never carries the lane's own gates.
A refusal you can dissolve by writing an emptier file is not the refusal the
declaration described (RobTand/prismaquant#119, principle 9: registry support
alone is not enough).

**What this module does not do.**  It does not RUN a gate.  Every Tessera gate
needs a fresh vLLM container with the pinned plugin editable-installed, and
the build lane must not be spawning those inside a pipeline run; building that
runner is R16's open half and stays with #119.  What this does is make the
declaration and the enforcement the same object, so a gate that is not run is
an *unfilled slot on a real card* --- visible, refused on at publication, and
countable --- rather than a sentence in a JSON file nothing reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lane_spec import LaneSpec, lane_spec_for_container
from .model_profiles.structure import canonical_export_lane
from .shipcard import (
    SHIPCARD_FILENAME,
    build_shipcard,
    lane_gate_slots,
    load_shipcard,
    write_shipcard,
)


class LaneShipcardError(RuntimeError):
    """Refusal to open a ship record.  Always actionable."""


def lane_spec_for_lane(lane: str) -> LaneSpec:
    """The declaration for ``lane``, refusing a lane with none.

    ``canonical_export_lane`` first, so an unknown container is refused by the
    vocabulary rather than by a missing file: the two answers are different
    facts and an operator acts on them differently.
    """
    canonical = canonical_export_lane(lane)
    try:
        return lane_spec_for_container(canonical)
    except KeyError as exc:
        raise LaneShipcardError(
            f"lane {canonical!r} is in the EXPORT_CONTAINER vocabulary but no "
            f"lane_specs/*.json declares it: {exc}. A lane with no declaration "
            "has no gates, so an artifact built through it could be published "
            "having closed nothing"
        ) from None


def open_lane_shipcard(
    artifact_dir: str | Path,
    lane: str,
    *,
    build: Mapping[str, Any] | None = None,
    shipcard_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Open the ship record for ``artifact_dir`` under ``lane``'s gate set.

    Refuses to clobber an existing card unless ``overwrite=True``: re-opening
    a card drops every slot the serve lane has already filled, and a build
    re-run over a served artifact is exactly when that would happen silently.
    """
    root = Path(artifact_dir)
    if not root.is_dir():
        raise LaneShipcardError(f"artifact directory does not exist: {root}")
    spec = lane_spec_for_lane(lane)
    path = Path(shipcard_path) if shipcard_path else root / SHIPCARD_FILENAME
    if path.exists() and not overwrite:
        raise LaneShipcardError(
            f"a ship record already exists at {path}; re-opening it would "
            "discard every slot the serve lane has filled. Pass --overwrite "
            "only when the artifact's bytes changed"
        )
    # The EXPORT_CONTAINER spelling, not the spec-file id: the card records
    # what the operator set and what `canonical_export_lane` speaks.
    #
    # `export_container` is also stamped into the BUILD block, because a
    # second obligation is derived from it: `shipcard._is_rate_axis_artifact`
    # ORs the card's build block with the artifact's own `config.json`, and
    # ORing is the whole design -- an obligation a single erasure removes is
    # not an obligation (#121). Without the stamp the Tessera lane's
    # `uniform_control` slot would rest on `config.json` alone, so a card
    # opened beside a checkpoint whose config went missing would owe the
    # control nothing. A caller that already declared the key keeps it.
    build_payload = dict(build or {})
    build_payload.setdefault("export_container", spec.export_container)
    card = build_shipcard(
        root, build=build_payload, lane=spec.export_container)
    write_shipcard(path, card)
    return path


def open_gate_report(spec: LaneSpec, card: Mapping[str, Any]) -> list[str]:
    """Human lines naming what this card has NOT closed, and what would."""
    from .lane_spec import lane_gate_report

    lines: list[str] = []
    for row in lane_gate_report(spec, card):
        if not row["recorded"]:
            lines.append(
                f"  [advisory by declaration] {row['gate']}: "
                f"{row['unrecorded_reason']}")
            continue
        state = "filled" if row["filled"] else "OPEN"
        lines.append(
            f"  [{state}] {row['gate']} -> slots['{row['shipcard_slot']}']\n"
            f"      run: {row['runner']}")
    for tool in spec.producer_tools:
        if tool.stability != "supported":
            lines.append(
                f"  [producer-tool debt] ${{{tool.repo_env}}}/{tool.path} is "
                f"{tool.stability} ({tool.tracking_issue})")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Open a lane-aware ship record for an exported artifact")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_open = sub.add_parser(
        "open", help="open a record whose slots are this lane's gates")
    p_open.add_argument("--lane", required=True, help="EXPORT_CONTAINER value")
    p_open.add_argument("--artifact", required=True,
                        help="exported artifact directory")
    p_open.add_argument("--build-json", default=None,
                        help="JSON file of build-lane facts to stamp")
    p_open.add_argument("--shipcard", default=None,
                        help="write here instead of <artifact>/shipcard.json")
    p_open.add_argument("--overwrite", action="store_true")

    p_slots = sub.add_parser(
        "slots", help="print the slots a lane's card must close")
    p_slots.add_argument("--lane", required=True)

    args = p.parse_args(argv)

    try:
        if args.cmd == "slots":
            spec = lane_spec_for_lane(args.lane)
            for slot in lane_gate_slots(spec.export_container):
                print(slot)
            return 0

        build: dict[str, Any] = {}
        if args.build_json:
            build = json.loads(Path(args.build_json).read_text(
                encoding="utf-8"))
        path = open_lane_shipcard(
            args.artifact, args.lane, build=build,
            shipcard_path=args.shipcard, overwrite=args.overwrite)
    except (LaneShipcardError, ValueError) as exc:
        print(f"[lane-shipcard] ERROR: {exc}", file=sys.stderr)
        return 2

    spec = lane_spec_for_lane(args.lane)
    card = load_shipcard(path)
    print(f"[lane-shipcard] opened {path} for lane "
          f"{spec.export_container!r}")
    print(f"[lane-shipcard] the serve lane must close "
          f"{sum(1 for v in card['slots'].values() if v is None)} slot(s):")
    for line in open_gate_report(spec, card):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
